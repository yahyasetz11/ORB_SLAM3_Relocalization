#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include "relocalization.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <iomanip>
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
        declare_parameter("video_path",  "");   // empty → use webcam
        declare_parameter("camera_id",   0);
        declare_parameter("visualize",   true);

        vocab_path_  = get_parameter("vocab_path").as_string();
        config_path_ = get_parameter("config_path").as_string();
        video_path_  = get_parameter("video_path").as_string();
        camera_id_   = get_parameter("camera_id").as_int();
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
        RCLCPP_INFO(get_logger(), "Input       : %s",
            video_path_.empty()
                ? (std::string("webcam device ") + std::to_string(camera_id_)).c_str()
                : video_path_.c_str());
        RCLCPP_INFO(get_logger(), "Visualize   : %s", visualize_ ? "yes" : "no");

        pose_pub_ = create_publisher<geometry_msgs::msg::Pose>("/relocalization/pose", 10);

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

        cv::VideoCapture cap;
        if (video_path_.empty())
            cap.open(camera_id_);
        else
            cap.open(video_path_);

        if (!cap.isOpened())
        {
            RCLCPP_ERROR(get_logger(), "Cannot open input source.");
            rclcpp::shutdown();
            return;
        }

        const cv::Size displaySize = reloc_->getDisplaySize();
        const cv::Size processSize = reloc_->getProcessSize();
        const float    kpScale     = (float)displaySize.width / processSize.width;

        cv::Mat frame;
        int frame_count = 0;

        RCLCPP_INFO(get_logger(), "Starting frame processing...");

        while (rclcpp::ok())
        {
            // Let ROS2 process any pending callbacks (param updates, etc.)
            rclcpp::spin_some(shared_from_this());

            if (!cap.read(frame) || frame.empty())
            {
                if (video_path_.empty())
                {
                    RCLCPP_WARN(get_logger(), "Failed to grab webcam frame, retrying...");
                    continue;
                }
                RCLCPP_INFO(get_logger(), "End of video.");
                break;
            }

            frame_count++;
            auto result = reloc_->processFrame(frame);

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
    std::string video_path_;
    int         camera_id_;
    bool        visualize_;
    bool        ready_;

    std::unique_ptr<Relocalization::RelocalizationModule> reloc_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RelocalizationNode>();
    node->run();   // blocks on main thread — OpenCV imshow works correctly here
    rclcpp::shutdown();
    return 0;
}
