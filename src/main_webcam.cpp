#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "System.h"

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Usage: ./slam_webcam path_to_vocabulary path_to_settings" << endl;
        cerr << "Example: ./slam_webcam ../ORB_SLAM3/Vocabulary/ORBvoc.txt config/webcam.yaml" << endl;
        return 1;
    }

    // Open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Cannot open webcam" << endl;
        return 1;
    }

    // Set camera resolution (optional)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    cout << "SLAM system initialized. Press ESC to exit." << endl;

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            cerr << "Failed to capture frame" << endl;
            break;
        }

        // Get timestamp
        auto now = chrono::steady_clock::now();
        double timestamp = chrono::duration_cast<chrono::duration<double>>(now.time_since_epoch()).count();

        // Track - returns Sophus::SE3f
        Sophus::SE3f pose = SLAM.TrackMonocular(frame, timestamp);

        // Check tracking state
        if (SLAM.GetTrackingState() == ORB_SLAM3::Tracking::OK)
        {
            // Get transformation matrix
            Eigen::Matrix4f transformation = pose.matrix();
            float x = transformation(0, 3);
            float y = transformation(1, 3);
            float z = transformation(2, 3);
            cout << "\rPosition: [" << x << ", " << y << ", " << z << "]" << flush;
        }
        else if (SLAM.GetTrackingState() == ORB_SLAM3::Tracking::LOST)
        {
            cout << "\rTracking LOST!" << flush;
        }
        else
        {
            cout << "\rInitializing..." << flush;
        }

        // Show frame (optional)
        cv::imshow("Webcam", frame);

        // Exit on ESC
        if (cv::waitKey(1) == 27)
            break;
    }

    // Save map and shutdown
    cout << "\nShutting down and saving map..." << endl;
    SLAM.Shutdown();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}