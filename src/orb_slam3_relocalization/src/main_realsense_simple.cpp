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
        cerr << "Usage: ./slam_realsense vocabulary settings" << endl;
        return 1;
    }

    // Configure RealSense
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // Start pipeline
    rs2::pipeline_profile profile = pipe.start(cfg);

    // Print intrinsics (useful for config file)
    auto intrinsics = profile.get_stream(RS2_STREAM_COLOR)
                          .as<rs2::video_stream_profile>()
                          .get_intrinsics();
    cout << "Update your config with these values:" << endl;
    cout << "Camera.fx: " << intrinsics.fx << endl;
    cout << "Camera.fy: " << intrinsics.fy << endl;
    cout << "Camera.cx: " << intrinsics.ppx << endl;
    cout << "Camera.cy: " << intrinsics.ppy << endl;

    // Create SLAM system - RGBD mode for RealSense
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);
    cout << "SLAM initialized. Press ESC to exit." << endl;

    int frame_count = 0;
    while (true)
    {
        // Get frames
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color = frames.get_color_frame();
        rs2::depth_frame depth = frames.get_depth_frame();

        if (!color || !depth)
            continue;

        // Convert to OpenCV
        cv::Mat rgb(cv::Size(640, 480), CV_8UC3, (void *)color.get_data());
        cv::Mat depth_mat(cv::Size(640, 480), CV_16UC1, (void *)depth.get_data());

        // RealSense gives BGR directly with RS2_FORMAT_BGR8, so no conversion needed

        double timestamp = color.get_timestamp() / 1000.0;

        // Track
        Sophus::SE3f pose = SLAM.TrackRGBD(rgb, depth_mat, timestamp);

        // Simple status display
        cout << "\rFrame " << frame_count++ << " ";
        int state = SLAM.GetTrackingState();
        if (state == 2)
            cout << "TRACKING OK" << flush;
        else
            cout << "State: " << state << flush;

        // Show frames
        cv::imshow("RGB", rgb);
        cv::imshow("Depth", depth_mat * 15); // Scale for visualization

        if (cv::waitKey(1) == 27)
            break; // ESC
    }

    cout << "\nShutting down..." << endl;
    SLAM.Shutdown();
    pipe.stop();

    return 0;
}