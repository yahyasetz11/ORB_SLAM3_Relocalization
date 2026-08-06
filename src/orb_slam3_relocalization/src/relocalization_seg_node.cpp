#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include "relocalization.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <array>
#include <cstdlib>

static std::string expandPath(const std::string &path)
{
    if (path.empty() || path[0] != '~')
        return path;
    const char *home = std::getenv("HOME");
    if (!home)
        return path;
    return std::string(home) + path.substr(1);
}

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

// Matches CLASSES in yolo_seg_video.py: {0:door, 1:fire_hydrant, 2:metal_door, 3:pillar, 4:window}
static const std::array<std::string, 9> SEG_CLASS_NAMES = {
    "door", "fire_hydrant", "metal_door", "pillar", "window", "keyboard", "monitor", "teddy_bear", "globe"};

class RelocalizationSegNode : public rclcpp::Node
{
public:
    RelocalizationSegNode() : Node("relocalization_seg_node"), ready_(false)
    {
        declare_parameter("vocab_path", "");
        declare_parameter("config_path", "");
        declare_parameter("visualize", true);
        declare_parameter("use_weighted_pnp", false);
        declare_parameter("landmark_weight", 1.0f);
        declare_parameter("background_weight", 0.3f);
        declare_parameter("dynamic_bg_weight", false);

        vocab_path_ = expandPath(get_parameter("vocab_path").as_string());
        config_path_ = expandPath(get_parameter("config_path").as_string());
        visualize_ = get_parameter("visualize").as_bool();
        use_weighted_pnp_ = get_parameter("use_weighted_pnp").as_bool();
        landmark_weight_ = get_parameter("landmark_weight").as_double();
        background_weight_ = get_parameter("background_weight").as_double();
        dynamic_bg_weight_ = get_parameter("dynamic_bg_weight").as_bool();

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

        RCLCPP_INFO(get_logger(), "Vocabulary        : %s", vocab_path_.c_str());
        RCLCPP_INFO(get_logger(), "Config            : %s", config_path_.c_str());
        RCLCPP_INFO(get_logger(), "Visualize         : %s", visualize_ ? "yes" : "no");
        RCLCPP_INFO(get_logger(), "PnP mode          : %s", use_weighted_pnp_ ? "weighted" : "standard");
        RCLCPP_INFO(get_logger(), "Landmark weight   : %.2f", landmark_weight_);
        RCLCPP_INFO(get_logger(), "Background weight : %.2f", background_weight_);
        RCLCPP_INFO(get_logger(), "Dynamic bg weight : %s", dynamic_bg_weight_ ? "yes" : "no");

        pose_pub_ = create_publisher<geometry_msgs::msg::Pose>("/relocalization/pose", 10);

        reloc_ = std::make_unique<Relocalization::RelocalizationModule>(vocab_path_, config_path_);

        RCLCPP_INFO(get_logger(), "Loading map...");
        if (!reloc_->loadMap())
        {
            RCLCPP_ERROR(get_logger(), "Failed to load map. Check System.LoadAtlasFromFile in config.");
            return;
        }
        RCLCPP_INFO(get_logger(), "Map loaded.");

        csv_log_.open("comparison_log_seg.csv");
        if (csv_log_.is_open())
        {
            csv_log_ << "frame,timestamp,"
                     << "std_x,std_y,std_z,std_inliers,std_total,std_reproj_px,"
                     << "wpnp_x,wpnp_y,wpnp_z,wpnp_inliers,wpnp_total,wpnp_reproj_px,"
                     << "wpnp_weighted_cost,pose_delta_m,wpnp_iterations,"
                     << "is_localized,result_tx,result_ty,result_tz,"
                     << "result_qx,result_qy,result_qz,result_qw,"
                     << "std_reproj_inliers_only,bg_weight_used\n";
            RCLCPP_INFO(get_logger(), "CSV log: comparison_log_seg.csv");
        }

        ready_ = true;

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&RelocalizationSegNode::imageCallback, this, std::placeholders::_1));
        RCLCPP_INFO(get_logger(), "Subscribed to /camera/image_raw");

        for (int cls_id = 0; cls_id < 9; cls_id++)
        {
            std::string topic = "/seg/" + SEG_CLASS_NAMES[cls_id];
            auto sub = create_subscription<sensor_msgs::msg::Image>(
                topic, 10,
                [this, cls_id](const sensor_msgs::msg::Image::SharedPtr msg)
                {
                    this->maskCallback(cls_id, msg);
                });
            seg_subs_.push_back(sub);
            RCLCPP_INFO(get_logger(), "Subscribed to %s", topic.c_str());
        }
    }

    void run()
    {
        if (!ready_)
        {
            rclcpp::shutdown();
            return;
        }

        const cv::Size displaySize = reloc_->getDisplaySize();
        const cv::Size processSize = reloc_->getProcessSize();
        const float kpScale = (float)displaySize.width / processSize.width;

        int frame_count = 0;
        double start_timestamp = 0.0;

        RCLCPP_INFO(get_logger(), "Waiting for frames on /camera/image_raw ...");

        while (rclcpp::ok())
        {
            rclcpp::spin_some(shared_from_this());

            cv::Mat frame;
            double frame_timestamp = 0.0;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                if (!has_new_frame_)
                    continue;
                frame = latest_frame_.clone();
                frame_timestamp = latest_frame_stamp_;
                has_new_frame_ = false;
            }

            frame_count++;

            // ── Segmentation mask snapshot ────────────────────────────────────
            cv::Mat mask_snapshot = getCombinedMask();

            cv::Mat proc_mask;
            if (!mask_snapshot.empty())
                cv::resize(mask_snapshot, proc_mask, processSize, 0, 0, cv::INTER_NEAREST);
            const bool mask_has_detections = !proc_mask.empty() &&
                                             (cv::countNonZero(proc_mask) > 0);

            auto result = reloc_->processFrame(frame, frame_timestamp);

            // ── Capture std PnP pose before WPnP may overwrite ────────────────
            cv::Point3f std_position = result.position;
            cv::Mat std_rvec = result.rvec.clone();
            cv::Mat std_tvec = result.tvec.clone();

            float effective_bg_weight = static_cast<float>(background_weight_);
            Relocalization::WeightedPnPResult wpnp;
            wpnp.success = false;
            wpnp.position = {0.0f, 0.0f, 0.0f};
            wpnp.numInliers = 0;
            wpnp.totalCorrespondences = 0;
            wpnp.meanReprojectionError = 0.0f;
            wpnp.weightedReprojectionError = 0.0f;
            wpnp.iterations = 0;

            // ── Weighted PnP — only when seg mask has detections ──────────────
            if (!result.inlierIndices.empty() && mask_has_detections)
            {
                std::vector<cv::Point2f> inlier2D;
                std::vector<cv::Point3f> inlier3D;
                inlier2D.reserve(result.inlierIndices.size());
                inlier3D.reserve(result.inlierIndices.size());
                for (int idx : result.inlierIndices)
                {
                    inlier2D.push_back(result.matched2DPoints[idx]);
                    inlier3D.push_back(result.matched3DPoints[idx]);
                }

                // Pixel-lookup weights: landmark if mask pixel > 0, background otherwise.
                // Dynamic mode: bg_weight = min(bg_inliers/landmark_inliers, 1.0),
                //   falling back to 1.0 if no landmark inliers found.
                const float lw = static_cast<float>(landmark_weight_);
                if (dynamic_bg_weight_)
                {
                    int lm_count = 0, bg_count = 0;
                    for (const auto &pt : inlier2D)
                    {
                        int px = std::clamp((int)std::round(pt.x), 0, proc_mask.cols - 1);
                        int py = std::clamp((int)std::round(pt.y), 0, proc_mask.rows - 1);
                        if (proc_mask.at<uint8_t>(py, px) > 0)
                            ++lm_count;
                        else
                            ++bg_count;
                    }
                    effective_bg_weight = (lm_count > 0)
                                              ? std::max(std::min((float)bg_count / (4 * lm_count), 1.0f), 0.3f)
                                              : 1.0f;
                }
                std::vector<float> weights;
                weights.reserve(inlier2D.size());
                for (const auto &pt : inlier2D)
                {
                    int px = std::clamp((int)std::round(pt.x), 0, proc_mask.cols - 1);
                    int py = std::clamp((int)std::round(pt.y), 0, proc_mask.rows - 1);
                    weights.push_back(proc_mask.at<uint8_t>(py, px) > 0 ? lw : effective_bg_weight);
                }

                wpnp = reloc_->solvePnPWeighted(
                    inlier3D, inlier2D, weights,
                    result.rvec, result.tvec,
                    50, 1e-6, 8.0f);

                if (wpnp.success)
                {
                    float dx = wpnp.position.x - result.position.x;
                    float dy = wpnp.position.y - result.position.y;
                    float dz = wpnp.position.z - result.position.z;
                    float pose_delta = std::sqrt(dx * dx + dy * dy + dz * dz);

                    RCLCPP_INFO(get_logger(),
                                "[WeightedPnP/seg] inliers=%d/%d  reproj=%.3fpx  "
                                "[StandardPnP] inliers=%d/%d  "
                                "pose_delta=%.4fm  wpnp_iters=%d  bg_w=%.3f",
                                wpnp.numInliers, wpnp.totalCorrespondences,
                                wpnp.meanReprojectionError,
                                result.numInliers, result.totalMatches,
                                pose_delta, wpnp.iterations, effective_bg_weight);

                    if (use_weighted_pnp_)
                    {
                        result.position = wpnp.position;
                        result.rvec = wpnp.rvec.clone();
                        result.tvec = wpnp.tvec.clone();
                    }
                }
                else
                {
                    RCLCPP_WARN(get_logger(),
                                "[WeightedPnP/seg] failed — standard PnP result kept");
                }
            }

            // ── CSV: one row per successfully localized frame ─────────────────
            if (result.success && csv_log_.is_open())
            {
                float std_reproj = reloc_->computeMeanReprojError(
                    result.matched3DPoints, result.matched2DPoints,
                    std_rvec, std_tvec);

                float std_reproj_inliers_only = -1.0f;
                if (!result.inlierIndices.empty())
                {
                    std::vector<cv::Point3f> inlier3D;
                    std::vector<cv::Point2f> inlier2D;
                    inlier3D.reserve(result.inlierIndices.size());
                    inlier2D.reserve(result.inlierIndices.size());
                    for (int idx : result.inlierIndices)
                    {
                        inlier3D.push_back(result.matched3DPoints[idx]);
                        inlier2D.push_back(result.matched2DPoints[idx]);
                    }
                    std_reproj_inliers_only = reloc_->computeMeanReprojError(
                        inlier3D, inlier2D, std_rvec, std_tvec);
                }

                float dx = wpnp.position.x - std_position.x;
                float dy = wpnp.position.y - std_position.y;
                float dz = wpnp.position.z - std_position.z;
                float pose_delta = wpnp.success
                                       ? std::sqrt(dx * dx + dy * dy + dz * dz)
                                       : 0.0f;

                int is_localized = result.success ? 1 : 0;
                float result_tx = 0.0f, result_ty = 0.0f, result_tz = 0.0f;
                float result_qx = 0.0f, result_qy = 0.0f, result_qz = 0.0f, result_qw = 1.0f;

                if (result.success)
                {
                    double qx, qy, qz, qw;
                    rvecToQuaternion(result.rvec, qx, qy, qz, qw);
                    result_tx = static_cast<float>(result.tvec.at<double>(0));
                    result_ty = static_cast<float>(result.tvec.at<double>(1));
                    result_tz = static_cast<float>(result.tvec.at<double>(2));
                    result_qx = static_cast<float>(qx);
                    result_qy = static_cast<float>(qy);
                    result_qz = static_cast<float>(qz);
                    result_qw = static_cast<float>(qw);
                }

                csv_log_ << frame_count << ","
                         << std::fixed << std::setprecision(6) << frame_timestamp << ","
                         << result.position.x << "," << result.position.y << "," << result.position.z << ","
                         << result.numInliers << "," << result.totalMatches << ","
                         << std::setprecision(4) << std_reproj << ","
                         << std::setprecision(6)
                         << wpnp.position.x << "," << wpnp.position.y << "," << wpnp.position.z << ","
                         << wpnp.numInliers << "," << wpnp.totalCorrespondences << ","
                         << std::setprecision(4) << wpnp.meanReprojectionError << ","
                         << wpnp.weightedReprojectionError << ","
                         << pose_delta << "," << wpnp.iterations
                         << "," << is_localized
                         << "," << result_tx << "," << result_ty << "," << result_tz
                         << "," << result_qx << "," << result_qy << "," << result_qz << "," << result_qw
                         << "," << std_reproj_inliers_only
                         << "," << std::setprecision(4) << effective_bg_weight
                         << "\n";
                csv_log_.flush();
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

                // ── Segmentation mask at display resolution ───────────────────
                cv::Mat display_mask;
                if (!mask_snapshot.empty())
                    cv::resize(mask_snapshot, display_mask, displaySize, 0, 0, cv::INTER_NEAREST);

                // Overlay segmented area as 20% opacity red fill
                if (!display_mask.empty())
                {
                    cv::Mat red_layer = cv::Mat::zeros(displayFrame.size(), CV_8UC3);
                    red_layer.setTo(cv::Scalar(0, 0, 255), display_mask);
                    cv::addWeighted(displayFrame, 1.0, red_layer, 0.50, 0.0, displayFrame);
                }

                // Pixel-lookup: is a display-resolution point inside any segmented area?
                auto isInLandmark = [&](const cv::Point2f &displayPt) -> bool
                {
                    if (display_mask.empty())
                        return false;
                    int px = std::clamp((int)std::round(displayPt.x), 0, display_mask.cols - 1);
                    int py = std::clamp((int)std::round(displayPt.y), 0, display_mask.rows - 1);
                    return display_mask.at<uint8_t>(py, px) > 0;
                };

                // All ORB keypoints (gray)
                for (const auto &kp : result.queryKeypoints)
                {
                    cv::Point2f scaledPt(kp.pt.x * kpScale, kp.pt.y * kpScale);
                    cv::circle(displayFrame, scaledPt, 2, cv::Scalar(150, 150, 150), -1);
                }

                // Landmark keypoints (red, drawn on top of gray)
                for (const auto &kp : result.queryKeypoints)
                {
                    cv::Point2f scaledPt(kp.pt.x * kpScale, kp.pt.y * kpScale);
                    if (isInLandmark(scaledPt))
                        cv::circle(displayFrame, scaledPt, 4, cv::Scalar(0, 0, 255), -1);
                }

                if (result.success)
                {
                    // Inlier matches: red if in seg mask, green if background
                    for (int idx : result.inlierIndices)
                    {
                        cv::Point2f scaledPt(result.matched2DPoints[idx].x * kpScale,
                                             result.matched2DPoints[idx].y * kpScale);
                        cv::Scalar color = isInLandmark(scaledPt)
                                               ? cv::Scalar(0, 0, 255)
                                               : cv::Scalar(0, 255, 0);
                        cv::circle(displayFrame, scaledPt, 5, color, 2);
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
                        cv::Scalar lineColor = isInLandmark(pt2D)
                                                   ? cv::Scalar(0, 0, 255)
                                                   : cv::Scalar(0, 255, 0);
                        cv::line(combined, pt2D, pt3DProj, lineColor, 1, cv::LINE_AA);
                    }

                    cv::Scalar statusColor;
                    std::string statusStr;
                    if (result.confidence >= 70)
                    {
                        statusColor = cv::Scalar(0, 255, 0);
                        statusStr = "EXCELLENT";
                    }
                    else if (result.confidence >= 50)
                    {
                        statusColor = cv::Scalar(0, 200, 255);
                        statusStr = "GOOD";
                    }
                    else
                    {
                        statusColor = cv::Scalar(0, 165, 255);
                        statusStr = "WEAK";
                    }

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

                // ── Seg mask status line ──────────────────────────────────────
                {
                    std::string segStr;
                    cv::Scalar segColor;
                    if (display_mask.empty() || cv::countNonZero(display_mask) == 0)
                    {
                        segStr = "Seg Mask: NONE";
                        segColor = cv::Scalar(150, 150, 150);
                    }
                    else
                    {
                        float coverage = 100.0f * cv::countNonZero(display_mask) / (display_mask.rows * display_mask.cols);
                        std::ostringstream lm;
                        lm << "Seg Coverage: " << std::fixed << std::setprecision(1)
                           << coverage << "%";
                        segStr = lm.str();
                        segColor = cv::Scalar(0, 0, 255);
                    }
                    cv::putText(combined, segStr,
                                cv::Point(30, 55), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                                segColor, 1);
                }

                if (frame_timestamp > 0.0)
                {
                    if (start_timestamp == 0.0)
                        start_timestamp = frame_timestamp;
                    std::ostringstream ts;
                    ts << "t=" << std::fixed << std::setprecision(3)
                       << (frame_timestamp - start_timestamp) << "s";
                    cv::putText(combined, ts.str(),
                                cv::Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                                cv::Scalar(200, 200, 200), 1);
                }

                cv::imshow("Relocalization Seg Node: Camera + Map", combined);
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
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            auto cv_img = cv_bridge::toCvCopy(msg, "bgr8");
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = cv_img->image;
            latest_frame_stamp_ = rclcpp::Time(msg->header.stamp).seconds();
            has_new_frame_ = true;
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_WARN(get_logger(), "cv_bridge error: %s", e.what());
        }
    }

    void maskCallback(int cls_id, const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            auto cv_img = cv_bridge::toCvCopy(msg, "mono8");
            std::lock_guard<std::mutex> lock(mask_mutex_);
            masks_[cls_id] = cv_img->image.clone();
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_WARN(get_logger(), "cv_bridge mask error (cls=%d): %s", cls_id, e.what());
        }
    }

    cv::Mat getCombinedMask()
    {
        std::lock_guard<std::mutex> lock(mask_mutex_);
        cv::Mat combined;
        for (int i = 0; i < 5; i++)
        {
            if (masks_[i].empty())
                continue;
            if (combined.empty())
                combined = masks_[i].clone();
            else
                cv::bitwise_or(combined, masks_[i], combined);
        }
        return combined;
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
    bool visualize_;
    bool use_weighted_pnp_;
    bool dynamic_bg_weight_;
    bool ready_;
    float landmark_weight_;
    float background_weight_;

    std::unique_ptr<Relocalization::RelocalizationModule> reloc_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> seg_subs_;

    cv::Mat latest_frame_;
    double latest_frame_stamp_{0.0};
    bool has_new_frame_{false};
    std::mutex frame_mutex_;

    // One mask per seg class (0=door..4=window), OR'd on demand
    std::array<cv::Mat, 5> masks_;
    std::mutex mask_mutex_;

    std::ofstream csv_log_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RelocalizationSegNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
