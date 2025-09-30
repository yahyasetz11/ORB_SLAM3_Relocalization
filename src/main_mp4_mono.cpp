/**
 * File: main_mp4_mono.cpp
 * Description: ORB_SLAM3 implementation for reading MP4 video files
 * Based on main_webcam_simple but reads from video file instead of live camera
 */

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <System.h>
#include <Converter.h>

using namespace std;

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./main_mp4_mono path_to_vocabulary path_to_settings path_to_video.mp4" << endl;
        return 1;
    }

    string vocabularyPath = argv[1];
    string settingsPath = argv[2];
    string videoPath = argv[3];

    // Open video file
    cv::VideoCapture cap(videoPath);
    if(!cap.isOpened())
    {
        cerr << "Error: Cannot open video file: " << videoPath << endl;
        return -1;
    }

    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cout << "Video Properties:" << endl;
    cout << "  - Path: " << videoPath << endl;
    cout << "  - FPS: " << fps << endl;
    cout << "  - Total Frames: " << totalFrames << endl;
    cout << "  - Resolution: " << frameWidth << "x" << frameHeight << endl;

    // Calculate frame period in seconds
    double framePeriod = 1.0 / fps;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::MONOCULAR, true);

    cout << endl << "-------" << endl;
    cout << "Start processing video sequence ..." << endl;
    cout << "Press ESC to stop" << endl;

    cv::Mat frame;
    int frameNumber = 0;
    
    // Vector to store trajectory for later saving
    vector<ORB_SLAM3::KeyFrame*> vpKFs;
    
    // Main loop
    auto start_time = chrono::steady_clock::now();
    
    while(cap.read(frame))
    {
        if(frame.empty())
        {
            cerr << "Warning: Empty frame at frame number " << frameNumber << endl;
            continue;
        }

        // Calculate timestamp based on frame number and FPS
        double timestamp = frameNumber * framePeriod;
        
        // Convert to grayscale if needed
        cv::Mat grayFrame;
        if(frame.channels() == 3)
        {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        }
        else
        {
            grayFrame = frame;
        }

        // Pass the image to the SLAM system
        auto start_process = chrono::steady_clock::now();
        cv::Mat Tcw = SLAM.TrackMonocular(grayFrame, timestamp);
        auto end_process = chrono::steady_clock::now();
        
        double process_time = chrono::duration_cast<chrono::duration<double>>(end_process - start_process).count();

        // Display frame info
        cout << "\rFrame: " << frameNumber << "/" << totalFrames 
             << " (" << fixed << setprecision(1) << (100.0 * frameNumber / totalFrames) << "%)"
             << " | Process time: " << fixed << setprecision(3) << process_time << "s"
             << " | Timestamp: " << fixed << setprecision(3) << timestamp << "s" << flush;

        // Optional: Display the frame with tracked features
        cv::Mat frameToShow;
        frame.copyTo(frameToShow);
        
        // Get current tracked map points for visualization (optional)
        vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();
        vector<ORB_SLAM3::MapPoint*> vMPs = SLAM.GetTrackedMapPoints();
        
        // Draw tracked keypoints
        for(size_t i = 0; i < vKeys.size(); i++)
        {
            if(vMPs[i])
            {
                cv::circle(frameToShow, vKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
            }
        }
        
        // Add frame info on image
        cv::putText(frameToShow, "Frame: " + to_string(frameNumber) + "/" + to_string(totalFrames), 
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frameToShow, "FPS: " + to_string(int(fps)), 
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("ORB-SLAM3: MP4 Input", frameToShow);
        
        // Check for ESC key
        char key = cv::waitKey(1);
        if(key == 27) // ESC key
        {
            cout << endl << "Processing stopped by user" << endl;
            break;
        }
        
        frameNumber++;
        
        // Optional: Add delay to match real-time playback
        // This is useful if you want to see the video at normal speed
        // Comment out for faster processing
        /*
        auto current_time = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::duration<double>>(current_time - start_time).count();
        double expected_time = frameNumber * framePeriod;
        if(expected_time > elapsed)
        {
            int wait_ms = (expected_time - elapsed) * 1000;
            cv::waitKey(wait_ms);
        }
        */
    }

    cout << endl << endl << "Video processing completed!" << endl;
    cout << "Total frames processed: " << frameNumber << endl;

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    string trajectoryFile = "CameraTrajectory_" + to_string(chrono::system_clock::now().time_since_epoch().count()) + ".txt";
    cout << "Saving camera trajectory to " << trajectoryFile << " ..." << endl;
    SLAM.SaveTrajectoryTUM(trajectoryFile);

    // Save keyframe trajectory  
    string keyframeFile = "KeyFrameTrajectory_" + to_string(chrono::system_clock::now().time_since_epoch().count()) + ".txt";
    cout << "Saving keyframe trajectory to " << keyframeFile << " ..." << endl;
    SLAM.SaveKeyFrameTrajectoryTUM(keyframeFile);

    // Optional: Save the map
    cout << "Saving map..." << endl;
    SLAM.SaveMap("map_from_video.osa");
    
    cout << "Map and trajectories saved successfully!" << endl;

    cap.release();
    cv::destroyAllWindows();

    return 0;
}