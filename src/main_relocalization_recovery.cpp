#include "relocalization.h"
#include "System.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "=========================================" << std::endl;
        std::cout << " Robust Relocalization with Recovery" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "\nUsage: " << argv[0]
                  << " <ORBvoc.txt> <config.yaml> <video>" << std::endl;
        std::cout << "\nFeatures:" << std::endl;
        std::cout << "  ✓ Initial global relocalization" << std::endl;
        std::cout << "  ✓ Continuous SLAM tracking" << std::endl;
        std::cout << "  ✓ Automatic recovery on tracking loss" << std::endl;
        std::cout << "  ✓ Pangolin 3D visualization" << std::endl;
        return 1;
    }

    std::string vocabPath = argv[1];
    std::string configPath = argv[2];
    std::string videoPath = argv[3];

    // ============================================================
    // INITIALIZATION
    // ============================================================
    std::cout << "\n=========================================" << std::endl;
    std::cout << " ROBUST RELOCALIZATION SYSTEM" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Config: " << configPath << std::endl;
    std::cout << "Video: " << videoPath << std::endl;

    // Initialize relocalization module (for recovery)
    Relocalization::RelocalizationModule reloc(vocabPath, configPath);

    std::cout << "\nLoading map..." << std::endl;
    if (!reloc.loadMap())
    {
        std::cerr << "✗ Failed to load map!" << std::endl;
        return 1;
    }
    std::cout << "✓ Map loaded" << std::endl;

    // ============================================================
    // FIND INITIAL POSE
    // ============================================================
    std::cout << "\n┌─────────────────────────────────────────┐" << std::endl;
    std::cout << "│  Finding Initial Position                │" << std::endl;
    std::cout << "└─────────────────────────────────────────┘\n"
              << std::endl;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "✗ Cannot open video!" << std::endl;
        return 1;
    }

    cv::Mat frame;
    int frameCount = 0;
    bool foundInitialPose = false;
    int initialFrameIndex = 0;

    while (cap.read(frame) && !foundInitialPose)
    {
        frameCount++;
        if (frameCount % 5 != 0)
            continue;

        std::cout << "\rScanning frame " << frameCount << "..." << std::flush;

        auto result = reloc.processFrame(frame);
        if (result.success)
        {
            foundInitialPose = true;
            initialFrameIndex = frameCount;

            std::cout << "\n\n✓ INITIAL POSE FOUND!" << std::endl;
            std::cout << "  Frame: " << frameCount << std::endl;
            std::cout << "  Position: [" << result.position.x << ", "
                      << result.position.y << ", " << result.position.z << "]" << std::endl;
            std::cout << "  Inliers: " << result.numInliers << std::endl;
        }
    }

    if (!foundInitialPose)
    {
        std::cerr << "\n✗ Could not find initial position!" << std::endl;
        return 1;
    }

    // ============================================================
    // START SLAM TRACKING
    // ============================================================
    std::cout << "\n┌─────────────────────────────────────────┐" << std::endl;
    std::cout << "│  Starting SLAM Tracking                  │" << std::endl;
    std::cout << "└─────────────────────────────────────────┘\n"
              << std::endl;

    std::cout << "Initializing ORB-SLAM3..." << std::endl;
    ORB_SLAM3::System SLAM(vocabPath, configPath,
                           ORB_SLAM3::System::MONOCULAR, true);

    std::cout << "✓ SLAM ready with Pangolin viewer" << std::endl;

    // Reset video to initial frame
    cap.release();
    cap.open(videoPath);
    for (int i = 0; i < initialFrameIndex - 1; i++)
        cap.read(frame);

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0)
        fps = 30.0;

    // Tracking statistics
    int currentFrame = initialFrameIndex;
    int successfulFrames = 0;
    int lostFrames = 0;
    int recoveryAttempts = 0;
    int successfulRecoveries = 0;
    int consecutiveLost = 0;

    std::cout << "\n=== TRACKING STARTED ===" << std::endl;
    std::cout << "Starting from frame " << initialFrameIndex << std::endl;
    std::cout << "Press ESC to stop\n"
              << std::endl;

    while (cap.read(frame))
    {
        if (frame.empty())
            break;

        cv::Mat resizedFrame;
        cv::resize(frame, resizedFrame, cv::Size(640, 480));

        double timestamp = currentFrame / fps;

        // Track with SLAM
        Sophus::SE3f pose = SLAM.TrackMonocular(resizedFrame, timestamp);
        int state = SLAM.GetTrackingState();

        // Update statistics
        if (state == 2) // OK
        {
            successfulFrames++;
            consecutiveLost = 0;
        }
        else if (state == 4) // LOST
        {
            lostFrames++;
            consecutiveLost++;
        }

        // Display status
        std::cout << "\rFrame " << currentFrame
                  << " | State: ";

        switch (state)
        {
        case 2:
            std::cout << "✓ OK       ";
            break;
        case 4:
            std::cout << "✗ LOST     ";
            break;
        default:
            std::cout << "? INIT     ";
            break;
        }

        std::cout << "| OK:" << successfulFrames
                  << " Lost:" << lostFrames
                  << " Recovered:" << successfulRecoveries;

        if (consecutiveLost > 0)
            std::cout << " | ConsecLost:" << consecutiveLost;

        std::cout << std::flush;

        // ========================================================
        // AUTOMATIC RECOVERY ON TRACKING LOSS
        // ========================================================
        if (state == 4 && consecutiveLost >= 3) // Lost for 3 frames
        {
            std::cout << "\n\n⚠️  TRACKING LOST - Attempting recovery..." << std::endl;
            recoveryAttempts++;

            // Try to relocalize
            auto recoveryResult = reloc.processFrame(resizedFrame);

            if (recoveryResult.success)
            {
                successfulRecoveries++;
                consecutiveLost = 0;

                std::cout << "✓ RECOVERY SUCCESSFUL!" << std::endl;
                std::cout << "  New position: [" << recoveryResult.position.x
                          << ", " << recoveryResult.position.y
                          << ", " << recoveryResult.position.z << "]" << std::endl;
                std::cout << "  Inliers: " << recoveryResult.numInliers << std::endl;
                std::cout << "\nResuming tracking...\n"
                          << std::endl;
            }
            else
            {
                std::cout << "✗ Recovery failed, continuing..." << std::endl;
            }
        }

        // Abort if lost for too long
        if (consecutiveLost > 30)
        {
            std::cout << "\n\n✗ Tracking lost for too long (>30 frames)" << std::endl;
            std::cout << "Unable to recover. Stopping..." << std::endl;
            break;
        }

        // Visualization
        cv::Mat display = resizedFrame.clone();

        std::string statusText;
        cv::Scalar color;

        if (state == 2)
        {
            statusText = "TRACKING";
            color = cv::Scalar(0, 255, 0);
        }
        else if (state == 4)
        {
            statusText = "LOST";
            color = cv::Scalar(0, 0, 255);
        }
        else
        {
            statusText = "INIT";
            color = cv::Scalar(0, 255, 255);
        }

        cv::putText(display, statusText, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 3);

        // Show stats
        std::string statsText = "OK:" + std::to_string(successfulFrames) +
                                " Lost:" + std::to_string(lostFrames) +
                                " Rec:" + std::to_string(successfulRecoveries);
        cv::putText(display, statsText, cv::Point(20, 80),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Tracking", display);

        if (cv::waitKey(1) == 27) // ESC
        {
            std::cout << "\n\nESC pressed - stopping" << std::endl;
            break;
        }

        currentFrame++;
    }

    // ============================================================
    // FINAL REPORT
    // ============================================================
    std::cout << "\n\n=========================================" << std::endl;
    std::cout << " FINAL TRACKING REPORT" << std::endl;
    std::cout << "=========================================" << std::endl;

    int totalFrames = currentFrame - initialFrameIndex;

    std::cout << "\nFrames:" << std::endl;
    std::cout << "  Total processed: " << totalFrames << std::endl;
    std::cout << "  Successfully tracked: " << successfulFrames << std::endl;
    std::cout << "  Lost: " << lostFrames << std::endl;

    if (totalFrames > 0)
    {
        float successRate = 100.0f * successfulFrames / totalFrames;
        std::cout << "  Success rate: " << successRate << "%" << std::endl;
    }

    std::cout << "\nRecovery:" << std::endl;
    std::cout << "  Recovery attempts: " << recoveryAttempts << std::endl;
    std::cout << "  Successful recoveries: " << successfulRecoveries << std::endl;

    if (recoveryAttempts > 0)
    {
        float recoveryRate = 100.0f * successfulRecoveries / recoveryAttempts;
        std::cout << "  Recovery success rate: " << recoveryRate << "%" << std::endl;
    }

    // Overall assessment
    std::cout << "\nOverall Performance: ";
    if (totalFrames > 0)
    {
        float successRate = 100.0f * successfulFrames / totalFrames;
        if (successRate > 90)
        {
            std::cout << "✓ EXCELLENT" << std::endl;
        }
        else if (successRate > 70)
        {
            std::cout << "✓ GOOD" << std::endl;
        }
        else if (successRate > 50)
        {
            std::cout << "⚠ MODERATE" << std::endl;
        }
        else
        {
            std::cout << "✗ POOR" << std::endl;
        }
    }

    std::cout << "\nShutting down..." << std::endl;
    SLAM.Shutdown();
    cv::destroyAllWindows();

    std::cout << "✓ Complete!" << std::endl;
    return 0;
}