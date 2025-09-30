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

        // Your cleanup instructions here
        std::cout << "Saving map..." << std::endl;
        // SLAM.SaveTrajectoryTUM("final_trajectory.txt");
        // SLAM->SaveAtlas("final_map.osa");
        SLAM->Shutdown();

        std::cout << "Shutting down threads..." << std::endl;
        // Stop threads, close files, etc.

        shutdown_flag = 1;
        std::cout << "Cleanup completed. Exiting..." << std::endl;
        exit(0);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);

    if (argc != 3)
    {
        cerr << "Usage: ./slam_webcam vocabulary settings" << endl;
        return 1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Cannot open webcam" << endl;
        return 1;
    }

    // Create SLAM system - NO map loading
    SLAM = new ORB_SLAM3::System(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    cout << "SLAM initialized. Press ESC to exit." << endl;

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(640, 480));
        // Add debug output
        cout << "Frame " << frame_count++ << " size: " << resized_frame.cols << "x" << resized_frame.rows << endl;

        auto now = chrono::steady_clock::now();
        double timestamp = chrono::duration_cast<chrono::duration<double>>(now.time_since_epoch()).count();

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

        cv::imshow("Webcam", resized_frame);
        if (cv::waitKey(1) == 27)
            break;
    }

    SLAM->Shutdown();
    cap.release();
    return 0;
}