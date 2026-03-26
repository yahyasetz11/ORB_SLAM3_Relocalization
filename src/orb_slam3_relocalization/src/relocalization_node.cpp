#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include "relocalization.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <cstdlib>  // getenv

// Rodrigues rvec → quaternion (Shepperd's method)
static void rvecToQuaternion(const cv::Mat &rvec,
                             double &qx, double &qy, double &qz, double &qw)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    double r00 = R.at<double>(0, 0), r01 = R.at<double>(0, 1), r02 = R.at<double>(0, 2);
    double r10 = R.at<double>(1, 0), r11 = R.at<double>(1, 1), r12 = R.at<double>(1, 2);
    double r20 = R.at<double>(2, 0), r21 = R.at<double>(2, 1), r22 = R.at<double>(2, 2);

    double trace = r00 + r11 + r22;
    if (trace > 0.0)
    {
        double s = 0.5 / std::sqrt(trace + 1.0);
        qw = 0.25 / s;
        qx = (r21 - r12) * s;
        qy = (r02 - r20) * s;
        qz = (r10 - r01) * s;
    }
    else if (r00 > r11 && r00 > r22)
    {
        double s = 2.0 * std::sqrt(1.0 + r00 - r11 - r22);
        qw = (r21 - r12) / s;
        qx = 0.25 * s;
        qy = (r01 + r10) / s;
        qz = (r02 + r20) / s;
    }
    else if (r11 > r22)
    {
        double s = 2.0 * std::sqrt(1.0 + r11 - r00 - r22);
        qw = (r02 - r20) / s;
        qx = (r01 + r10) / s;
        qy = 0.25 * s;
        qz = (r12 + r21) / s;
    }
    else
    {
        double s = 2.0 * std::sqrt(1.0 + r22 - r00 - r11);
        qw = (r10 - r01) / s;
        qx = (r02 + r20) / s;
        qy = (r12 + r21) / s;
        qz = 0.25 * s;
    }
}

class RelocalizationNode : public rclcpp::Node
{
public:
    RelocalizationNode() : Node("relocalization_node"), ready_(false)
    {
        declare_parameter("vocab_path",  "");
        declare_parameter("config_path", "");
        declare_parameter("visualize",   true);

        vocab_path_  = get_parameter("vocab_path").as_string();
        config_path_ = get_parameter("config_path").as_string();
        visualize_   = get_parameter("visualize").as_bool();

        if (vocab_path_.empty() || config_path_.empty())
        {
            RCLCPP_ERROR(get_logger(), "Required parameters missing. Set vocab_path and config_path.");
            return;
        }

        if (visualize_ && !std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
        {
            RCLCPP_WARN(get_logger(), "No display found — visualization disabled.");
            visualize_ = false;
        }

        RCLCPP_INFO(get_logger(), "Vocabulary  : %s", vocab_path_.c_str());
        RCLCPP_INFO(get_logger(), "Config      : %s", config_path_.c_str());
        RCLCPP_INFO(get_logger(), "Visualize   : %s", visualize_ ? "yes" : "no");

        pose_pub_ = create_publisher<geometry_msgs::msg::Pose>("/relocalization/pose", 10);

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&RelocalizationNode::imageCallback, this, std::placeholders::_1));
        RCLCPP_INFO(get_logger(), "Subscribed to /camera/image_raw");

        yolo_sub_ = create_subscription<std_msgs::msg::String>(
            "yolo/results", 10,
            std::bind(&RelocalizationNode::yoloCallback, this, std::placeholders::_1));
        RCLCPP_INFO(get_logger(), "Subscribed to yolo/results");

        reloc_ = std::make_unique<Relocalization::RelocalizationModule>(vocab_path_, config_path_);

        RCLCPP_INFO(get_logger(), "Loading map...");
        if (!reloc_->loadMap())
        {
            RCLCPP_ERROR(get_logger(), "Failed to load map. Check System.LoadAtlasFromFile in config.");
            return;
        }
        RCLCPP_INFO(get_logger(), "Map loaded.");
        ready_ = true;
    }

    // Called from main() on the main thread — keeps imshow on the correct thread.
    void run()
    {
        if (!ready_)
        {
            rclcpp::shutdown();
            return;
        }

        const cv::Size displaySize = reloc_->getDisplaySize();
        const cv::Size processSize = reloc_->getProcessSize();
        const float    kpScale     = (float)displaySize.width / processSize.width;

        int frame_count = 0;

        RCLCPP_INFO(get_logger(), "Waiting for frames on /camera/image_raw ...");

        while (rclcpp::ok())
        {
            // Process incoming callbacks (fills latest_frame_ via imageCallback)
            rclcpp::spin_some(shared_from_this());

            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                if (!has_new_frame_)
                    continue;
                frame = latest_frame_.clone();
                has_new_frame_ = false;
            }

            frame_count++;
            auto result = reloc_->processFrame(frame);

            // ── Landmark keypoint segmentation ───────────────────────────────
            // Scale YOLO bbox coords (640x480) → process resolution
            {
                const float sx = (float)processSize.width  / 640.0f;
                const float sy = (float)processSize.height / 480.0f;

                std::vector<LandmarkDetection> bboxes;
                {
                    std::lock_guard<std::mutex> lock(bbox_mutex_);
                    bboxes = latest_bboxes_;
                }

                for (const auto &det : bboxes)
                {
                    Relocalization::LandmarkRegion region;
                    region.cls_id = det.cls_id;
                    region.bbox   = cv::Rect(
                        (int)(det.bbox.x      * sx), (int)(det.bbox.y      * sy),
                        (int)(det.bbox.width  * sx), (int)(det.bbox.height * sy));

                    for (const auto &kp : result.queryKeypoints)
                    {
                        if (region.bbox.contains(cv::Point((int)kp.pt.x, (int)kp.pt.y)))
                            region.keypoints.push_back(kp);
                    }

                    RCLCPP_INFO(get_logger(),
                        "Landmark cls=%d: %zu keypoints in bbox [%d,%d,%dx%d]",
                        region.cls_id, region.keypoints.size(),
                        region.bbox.x, region.bbox.y,
                        region.bbox.width, region.bbox.height);

                    result.landmarkRegions.push_back(std::move(region));
                }
            }

            if (result.success)
            {
                RCLCPP_INFO(get_logger(),
                    "Localized  frame=%d  inliers=%d/%d  conf=%.1f%%",
                    frame_count, result.numInliers, result.totalMatches, result.confidence);
                publishPose(result);
            }
            else
            {
                RCLCPP_DEBUG(get_logger(), "Frame %d: relocalization failed", frame_count);
            }

            if (!visualize_)
                continue;

            try
            {
                // ── Left panel: camera frame ──────────────────────────────────
                cv::Mat displayFrame;
                cv::resize(frame, displayFrame, displaySize);

                // All ORB keypoints (gray)
                for (const auto &kp : result.queryKeypoints)
                {
                    cv::Point2f scaledPt(kp.pt.x * kpScale, kp.pt.y * kpScale);
                    cv::circle(displayFrame, scaledPt, 2, cv::Scalar(150, 150, 150), -1);
                }

                if (result.success)
                {
                    // Inlier matches (green)
                    for (int idx : result.inlierIndices)
                    {
                        cv::Point2f scaledPt(result.matched2DPoints[idx].x * kpScale,
                                             result.matched2DPoints[idx].y * kpScale);
                        cv::circle(displayFrame, scaledPt, 5, cv::Scalar(0, 255, 0), 2);
                    }
                }

                // ── Right panel: top-down map ─────────────────────────────────
                cv::Mat mapViz = reloc_->createMapVisualization(result, displaySize);

                // ── Combined side-by-side ─────────────────────────────────────
                cv::Mat combined(displaySize.height, displaySize.width * 2, CV_8UC3);
                displayFrame.copyTo(combined(cv::Rect(0, 0, displaySize.width, displaySize.height)));
                mapViz.copyTo(combined(cv::Rect(displaySize.width, 0, displaySize.width, displaySize.height)));

                if (result.success)
                {
                    // Lines connecting inlier 2D points to their 3D map projections
                    for (int idx : result.inlierIndices)
                    {
                        cv::Point2f pt2D(result.matched2DPoints[idx].x * kpScale,
                                         result.matched2DPoints[idx].y * kpScale);
                        cv::Point2f pt3DProj = reloc_->project3DTo2D(
                            result.matched3DPoints[idx], displaySize.height);
                        pt3DProj.x += displaySize.width;
                        cv::line(combined, pt2D, pt3DProj, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                    }

                    cv::Scalar statusColor;
                    std::string statusStr;
                    if (result.confidence >= 70)      { statusColor = cv::Scalar(0, 255, 0);   statusStr = "EXCELLENT"; }
                    else if (result.confidence >= 50) { statusColor = cv::Scalar(0, 200, 255); statusStr = "GOOD"; }
                    else                              { statusColor = cv::Scalar(0, 165, 255); statusStr = "WEAK"; }

                    cv::putText(combined, "LOCALIZED - " + statusStr,
                                cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, statusColor, 2);

                    std::ostringstream info;
                    info << "Inliers: " << result.numInliers << "/" << result.totalMatches
                         << " | Conf: " << std::fixed << std::setprecision(1)
                         << result.confidence << "%";
                    cv::putText(combined, info.str(),
                                cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                                cv::Scalar(255, 255, 255), 1);
                }
                else
                {
                    cv::putText(combined, "SEARCHING...",
                                cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(0, 100, 255), 2);
                }

                cv::imshow("Relocalization Node: Camera + Map", combined);
                int key = cv::waitKey(1);
                if (key == 27)
                {
                    RCLCPP_INFO(get_logger(), "ESC pressed — stopping.");
                    break;
                }
            }
            catch (const cv::Exception &e)
            {
                RCLCPP_WARN(get_logger(), "Display error: %s — disabling visualization.", e.what());
                visualize_ = false;
            }
        }

        cv::destroyAllWindows();
        rclcpp::shutdown();
    }

private:
    struct LandmarkDetection
    {
        int cls_id;
        cv::Rect bbox;   // in YOLO resolution (640x480)
    };

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            auto cv_img = cv_bridge::toCvCopy(msg, "bgr8");
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_  = cv_img->image;
            has_new_frame_ = true;
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_WARN(get_logger(), "cv_bridge error: %s", e.what());
        }
    }

    void yoloCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        try
        {
            auto data = nlohmann::json::parse(msg->data);
            std::lock_guard<std::mutex> lock(bbox_mutex_);
            latest_bboxes_.clear();
            for (const auto &det : data["detections"])
            {
                int x1 = det["bbox_coords"]["x1"].get<int>();
                int y1 = det["bbox_coords"]["y1"].get<int>();
                int x2 = det["bbox_coords"]["x2"].get<int>();
                int y2 = det["bbox_coords"]["y2"].get<int>();
                latest_bboxes_.push_back({det["cls_id"].get<int>(),
                                          cv::Rect(x1, y1, x2 - x1, y2 - y1)});
            }
        }
        catch (const nlohmann::json::exception &e)
        {
            RCLCPP_WARN(get_logger(), "Failed to parse yolo/results: %s", e.what());
        }
    }

    void publishPose(const Relocalization::LocationResult &result)
    {
        geometry_msgs::msg::Pose msg;

        msg.position.x = result.position.x;
        msg.position.y = result.position.y;
        msg.position.z = result.position.z;

        if (!result.rvec.empty())
        {
            cv::Mat rvec_d;
            result.rvec.convertTo(rvec_d, CV_64F);
            double qx, qy, qz, qw;
            rvecToQuaternion(rvec_d, qx, qy, qz, qw);
            msg.orientation.x = qx;
            msg.orientation.y = qy;
            msg.orientation.z = qz;
            msg.orientation.w = qw;
        }
        else
        {
            msg.orientation.w = 1.0;
        }

        pose_pub_->publish(msg);
    }

    std::string vocab_path_;
    std::string config_path_;
    bool        visualize_;
    bool        ready_;

    std::unique_ptr<Relocalization::RelocalizationModule> reloc_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr           pose_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr         image_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr           yolo_sub_;

    // Camera frame (populated by imageCallback)
    cv::Mat    latest_frame_;
    bool       has_new_frame_{false};
    std::mutex frame_mutex_;

    // YOLO bboxes (populated by yoloCallback)
    std::vector<LandmarkDetection> latest_bboxes_;
    std::mutex                     bbox_mutex_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RelocalizationNode>();
    node->run();   // blocks on main thread — OpenCV imshow works correctly here
    rclcpp::shutdown();
    return 0;
}
