#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "System.h"

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Usage: ./slam_realsense path_to_vocabulary path_to_settings" << endl;
        cerr << "Example: ./slam_realsense ../ORB_SLAM3/Vocabulary/ORBvoc.txt config/RealSenseD435i.yaml" << endl;
        return 1;
    }

    // Configure RealSense pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // Start pipeline
    rs2::pipeline_profile profile = pipe.start(cfg);

    // Get camera intrinsics for configuration
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = color_stream.get_intrinsics();

    cout << "Camera Intrinsics:" << endl;
    cout << "  Width: " << intrinsics.width << endl;
    cout << "  Height: " << intrinsics.height << endl;
    cout << "  fx: " << intrinsics.fx << endl;
    cout << "  fy: " << intrinsics.fy << endl;
    cout << "  cx: " << intrinsics.ppx << endl;
    cout << "  cy: " << intrinsics.ppy << endl;

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);
    cout << "SLAM system initialized. Press ESC to exit." << endl;

    while (true)
    {
        // Wait for frames
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame)
            continue;

        // Convert to OpenCV Mat
        cv::Mat rgb(cv::Size(640, 480), CV_8UC3, (void *)color_frame.get_data());
        cv::Mat depth(cv::Size(640, 480), CV_16UC1, (void *)depth_frame.get_data());

        // Convert RGB to BGR for OpenCV
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

        // Get timestamp
        double timestamp = color_frame.get_timestamp() / 1000.0; // Convert to seconds

        // Track - returns Sophus::SE3f in newer versions
        Sophus::SE3f pose = SLAM.TrackRGBD(bgr, depth, timestamp);

        // Get the transformation matrix
        Eigen::Matrix4f transformation = pose.matrix();

        // Extract position (translation)
        float x = transformation(0, 3);
        float y = transformation(1, 3);
        float z = transformation(2, 3);

        // Check if tracking is valid (not lost)
        if (SLAM.GetTrackingState() == ORB_SLAM3::Tracking::OK)
        {
            cout << "\rPosition: [" << x << ", " << y << ", " << z << "]" << flush;
        }
        else if (SLAM.GetTrackingState() == ORB_SLAM3::Tracking::LOST)
        {
            cout << "\rTracking LOST!" << flush;
        }
        else if (SLAM.GetTrackingState() == ORB_SLAM3::Tracking::NOT_INITIALIZED)
        {
            cout << "\rInitializing..." << flush;
        }

        // Show frames (optional)
        cv::imshow("RGB", bgr);
        cv::imshow("Depth", depth * 15); // Scale for visualization

        // Exit on ESC
        if (cv::waitKey(1) == 27)
            break;
    }

    // Save map and shutdown
    cout << "\nShutting down and saving map..." << endl;
    SLAM.Shutdown();

    pipe.stop();
    cv::destroyAllWindows();

    return 0;
}