#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cmath>

#include "System.h"

static std::string expandPath(const std::string &path)
{
    if (path.empty() || path[0] != '~')
        return path;
    const char *home = std::getenv("HOME");
    if (!home)
        return path;
    return std::string(home) + path.substr(1);
}

class LocalizationTestNode : public rclcpp::Node
{
public:
    LocalizationTestNode() : Node("localization_test_node"), slam_(nullptr), running_(false)
    {
        declare_parameter("vocab_path", "");
        declare_parameter("config_path", "");
        declare_parameter("video_path", "");
        declare_parameter("output_csv", "localization_test_log.csv");
        declare_parameter("mode", "video");

        vocab_path_ = expandPath(get_parameter("vocab_path").as_string());
        config_path_ = expandPath(get_parameter("config_path").as_string());
        video_path_ = expandPath(get_parameter("video_path").as_string());
        output_csv_ = expandPath(get_parameter("output_csv").as_string());
        mode_ = get_parameter("mode").as_string();

        if (vocab_path_.empty() || config_path_.empty())
        {
            RCLCPP_ERROR(get_logger(),
                         "Required parameters missing. Set vocab_path and config_path.");
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(get_logger(), "Vocabulary : %s", vocab_path_.c_str());
        RCLCPP_INFO(get_logger(), "Config     : %s", config_path_.c_str());
        RCLCPP_INFO(get_logger(), "Output CSV : %s", output_csv_.c_str());
        RCLCPP_INFO(get_logger(), "Mode       : %s", mode_.c_str());

        slam_ = new ORB_SLAM3::System(
            vocab_path_, config_path_, ORB_SLAM3::System::MONOCULAR, /*viewer=*/true);
        // slam_->ActivateLocalizationMode();

        csv_.open(output_csv_);
        if (!csv_.is_open())
        {
            RCLCPP_ERROR(get_logger(), "Cannot open output CSV: %s", output_csv_.c_str());
            rclcpp::shutdown();
            return;
        }
        csv_ << "timestamp,tracking_state,is_localized,"
             << "tx,ty,tz,qx,qy,qz,qw,num_tracked_points\n";

        running_ = true;
        worker_thread_ = std::thread(&LocalizationTestNode::runSlam, this);
    }

    ~LocalizationTestNode()
    {
        running_ = false;
        if (worker_thread_.joinable())
            worker_thread_.join();
        if (csv_.is_open())
            csv_.close();
        if (slam_)
        {
            slam_->Shutdown();
            delete slam_;
            slam_ = nullptr;
        }
    }

private:
    void runSlam()
    {
        cv::VideoCapture cap;
        if (mode_ == "stream")
            cap.open(0);
        else
            cap.open(video_path_);

        if (!cap.isOpened())
        {
            RCLCPP_ERROR(get_logger(), "Cannot open input source.");
            rclcpp::shutdown();
            return;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0 || std::isnan(fps))
            fps = 30.0;

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        RCLCPP_INFO(get_logger(), "Video FPS: %.1f  |  Total frames: %d", fps, total_frames);

        cv::Mat frame;
        int frame_count = 0;
        double timestamp = 0.0;

        while (running_ && rclcpp::ok())
        {
            if (!cap.read(frame) || frame.empty())
            {
                if (mode_ == "stream")
                {
                    RCLCPP_WARN(get_logger(), "Failed to grab webcam frame, retrying...");
                    continue;
                }
                RCLCPP_INFO(get_logger(), "End of video reached.");
                break;
            }

            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(640, 480));

            Sophus::SE3f Tcw = slam_->TrackMonocular(resized, timestamp);
            int state = slam_->GetTrackingState();
            // state: -1=SYSTEM_NOT_READY, 0=NO_IMAGES_YET, 1=NOT_INITIALIZED, 2=OK, 3=RECENTLY_LOST, 4=LOST
            bool localized = (state == 2);
            int n_pts = static_cast<int>(slam_->GetTrackedMapPoints().size());

            csv_ << std::fixed << std::setprecision(6)
                 << timestamp << "," << state << "," << (localized ? 1 : 0) << ",";

            if (localized)
            {
                // Invert Tcw to get camera position in world frame (Twc)
                Sophus::SE3f Twc = Tcw.inverse();
                Eigen::Vector3f t = Twc.translation();
                Eigen::Quaternionf q = Twc.unit_quaternion();
                csv_ << t.x() << "," << t.y() << "," << t.z() << ","
                     << q.x() << "," << q.y() << "," << q.z() << "," << q.w();
            }
            else
            {
                csv_ << "nan,nan,nan,nan,nan,nan,nan";
            }
            csv_ << "," << n_pts << "\n";

            if (frame_count % 30 == 0)
            {
                RCLCPP_INFO(get_logger(), "Frame %d/%d  t=%.2fs  state=%d  pts=%d",
                            frame_count, total_frames, timestamp, state, n_pts);
            }

            timestamp += 1.0 / fps;
            frame_count++;
        }

        RCLCPP_INFO(get_logger(), "Localization test complete. CSV written to: %s",
                    output_csv_.c_str());
        rclcpp::shutdown();
    }

    std::string vocab_path_;
    std::string config_path_;
    std::string video_path_;
    std::string output_csv_;
    std::string mode_;

    ORB_SLAM3::System *slam_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    std::ofstream csv_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LocalizationTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
