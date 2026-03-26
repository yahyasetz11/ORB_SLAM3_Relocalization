#include <iostream>
#include <chrono>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include "System.h"

using namespace std;
int frame_count = 0;
volatile sig_atomic_t shutdown_flag = 0;

ORB_SLAM3::System *SLAM = nullptr;

void signalHandler(int signal)
{
    if (signal == SIGINT)
    {
        std::cout << "\nCtrl+C detected! Starting cleanup..." << std::endl;
        std::cout << "Saving map..." << std::endl;
        SLAM->Shutdown();
        std::cout << "Shutting down threads..." << std::endl;
        shutdown_flag = 1;
        std::cout << "Cleanup completed. Exiting..." << std::endl;
        exit(0);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);

    if (argc != 4) // Changed from 3 to 4
    {
        cerr << "Usage: ./slam_webcam vocabulary settings video_file.mp4" << endl;
        return 1;
    }

    // Open video file instead of webcam
    cv::VideoCapture cap(argv[3]);
    if (!cap.isOpened())
    {
        cerr << "Cannot open video file: " << argv[3] << endl;
        return 1;
    }

    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cout << "Video FPS: " << fps << endl;
    cout << "Total frames: " << total_frames << endl;

    // Create SLAM system
    SLAM = new ORB_SLAM3::System(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    cout << "SLAM initialized. Press ESC to exit." << endl;

    cv::Mat frame;
    double timestamp = 0.0;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "End of video reached." << endl;
            std::cout << "Saving map..." << std::endl;
            SLAM->Shutdown();
            std::cout << "Cleanup completed. Exiting..." << std::endl;
            exit(0);
            break;
        }

        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(640, 480));

        // Use frame-based timestamp (more accurate for video files)
        timestamp = frame_count / fps;

        cout << "Frame " << frame_count++ << " / " << total_frames
             << " - Timestamp: " << timestamp << "s" << endl;

        Sophus::SE3f pose = SLAM->TrackMonocular(resized_frame, timestamp);

        // Check tracking state
        int state = SLAM->GetTrackingState();
        if (state == 0)
            cout << "SYSTEM_NOT_READY" << endl;
        else if (state == 1)
            cout << "NO_IMAGES_YET" << endl;
        else if (state == 2)
            cout << "OK" << endl;
        else if (state == 3)
            cout << "NOT_INITIALIZED" << endl;
        else if (state == 4)
            cout << "LOST" << endl;

        cv::imshow("Video Playback", resized_frame);

        // Wait time based on FPS (or press ESC to exit)
        int delay = max(1, (int)(1000.0 / fps));
        if (cv::waitKey(delay) == 27)
            break;
    }

    SLAM->Shutdown();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}