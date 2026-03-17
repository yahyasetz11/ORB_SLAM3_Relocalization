#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "System.h"

#include <thread>
#include <atomic>
#include <chrono>

class MapCreatorNode : public rclcpp::Node
{
public:
    MapCreatorNode() : Node("map_creator_node"), slam_(nullptr), running_(false)
    {
        declare_parameter("vocab_path",  "");
        declare_parameter("config_path", "");
        declare_parameter("video_path",  "");   // empty → use webcam
        declare_parameter("camera_id",   0);

        vocab_path_  = get_parameter("vocab_path").as_string();
        config_path_ = get_parameter("config_path").as_string();
        video_path_  = get_parameter("video_path").as_string();
        camera_id_   = get_parameter("camera_id").as_int();

        if (vocab_path_.empty() || config_path_.empty())
        {
            RCLCPP_ERROR(get_logger(),
                "Required parameters missing. Set vocab_path and config_path.");
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(get_logger(), "Vocabulary : %s", vocab_path_.c_str());
        RCLCPP_INFO(get_logger(), "Config     : %s", config_path_.c_str());
        RCLCPP_INFO(get_logger(), "Input      : %s",
            video_path_.empty()
                ? (std::string("webcam device ") + std::to_string(camera_id_)).c_str()
                : video_path_.c_str());

        running_ = true;
        worker_thread_ = std::thread(&MapCreatorNode::runSlam, this);
    }

    ~MapCreatorNode()
    {
        running_ = false;
        if (worker_thread_.joinable())
            worker_thread_.join();
    }

private:
    void runSlam()
    {
        const bool is_stream = video_path_.empty();

        cv::VideoCapture cap;
        if (is_stream)
            cap.open(camera_id_);
        else
            cap.open(video_path_);

        if (!cap.isOpened())
        {
            RCLCPP_ERROR(get_logger(), "Cannot open input source.");
            rclcpp::shutdown();
            return;
        }

        if (is_stream)
        {
            RCLCPP_INFO(get_logger(), "Webcam opened. Press Ctrl+C to stop and save map.");
        }
        else
        {
            double fps       = cap.get(cv::CAP_PROP_FPS);
            int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
            if (fps <= 0) fps = 30.0;
            RCLCPP_INFO(get_logger(), "Video FPS: %.1f  |  Total frames: %d", fps, total_frames);
        }

        slam_ = new ORB_SLAM3::System(
            vocab_path_, config_path_, ORB_SLAM3::System::MONOCULAR, true);

        RCLCPP_INFO(get_logger(), "SLAM system initialized. Processing frames...");

        cv::Mat frame;
        int frame_count = 0;

        // For video: derive timestamp from frame index / fps
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        // For stream: use wall-clock elapsed time
        auto t_start = std::chrono::steady_clock::now();

        const char* state_str[] = {
            "SYSTEM_NOT_READY", "NO_IMAGES_YET", "OK", "NOT_INITIALIZED", "LOST"
        };

        while (running_ && rclcpp::ok())
        {
            if (!cap.read(frame) || frame.empty())
            {
                if (is_stream)
                {
                    RCLCPP_WARN(get_logger(), "Failed to grab webcam frame, retrying...");
                    continue;
                }
                RCLCPP_INFO(get_logger(), "End of video reached. Saving map...");
                break;
            }

            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(640, 480));

            double timestamp;
            if (is_stream)
            {
                auto now = std::chrono::steady_clock::now();
                timestamp = std::chrono::duration<double>(now - t_start).count();
            }
            else
            {
                timestamp = frame_count / fps;
            }

            slam_->TrackMonocular(resized, timestamp);

            int state = slam_->GetTrackingState();
            if (state >= 0 && state <= 4)
            {
                if (is_stream)
                    RCLCPP_INFO(get_logger(), "Frame %d  t=%.2fs  [%s]",
                        frame_count, timestamp, state_str[state]);
                else
                    RCLCPP_INFO(get_logger(), "Frame %d/%d  [%s]",
                        frame_count, total_frames, state_str[state]);
            }

            frame_count++;
        }

        RCLCPP_INFO(get_logger(), "Shutting down SLAM and saving map...");
        slam_->Shutdown();
        delete slam_;
        slam_ = nullptr;

        RCLCPP_INFO(get_logger(), "Map saved. Node shutting down.");
        rclcpp::shutdown();
    }

    std::string vocab_path_;
    std::string config_path_;
    std::string video_path_;
    int         camera_id_;

    ORB_SLAM3::System*  slam_;
    std::atomic<bool>   running_;
    std::thread         worker_thread_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapCreatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
