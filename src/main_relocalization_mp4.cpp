/**
 * Simple Feature-Based Relocalization
 * This is the simplest version that just matches features between videos
 * to find your location in the map.
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <map>

using namespace std;
using namespace cv;

class SimpleRelocalization
{
private:
    // Store keyframes from mapping video
    struct MapFrame
    {
        int frameNumber;
        Mat image;
        vector<KeyPoint> keypoints;
        Mat descriptors;
        Point2f mapPosition; // Position in 2D map view
    };

    vector<MapFrame> mapFrames;
    Ptr<ORB> orb;
    Ptr<DescriptorMatcher> matcher;

    // Map visualization
    Mat mapImage;
    Point2f currentPosition;
    vector<Point2f> trajectory;

    // Parameters
    const int KEYFRAME_INTERVAL = 30; // Sample every 30 frames (1 sec at 30fps)
    const int MIN_MATCHES = 20;
    const float MATCH_RATIO = 0.7f;

public:
    SimpleRelocalization()
    {
        // Create ORB detector
        orb = ORB::create(1000);

        // Create matcher
        matcher = DescriptorMatcher::create("BruteForce-Hamming");

        // Initialize map visualization
        mapImage = Mat::zeros(600, 800, CV_8UC3);
        mapImage.setTo(Scalar(30, 30, 30));
    }

    // Step 1: Build map from mapping video
    void BuildMapFromVideo(const string &mappingVideo)
    {
        cout << "\n=== BUILDING MAP FROM VIDEO ===" << endl;
        cout << "Video: " << mappingVideo << endl;

        VideoCapture cap(mappingVideo);
        if (!cap.isOpened())
        {
            cerr << "Cannot open mapping video!" << endl;
            return;
        }

        double fps = cap.get(CAP_PROP_FPS);
        int totalFrames = cap.get(CAP_PROP_FRAME_COUNT);
        cout << "FPS: " << fps << ", Total frames: " << totalFrames << endl;

        Mat frame;
        int frameCount = 0;

        while (cap.read(frame))
        {
            // Sample keyframes at intervals
            if (frameCount % KEYFRAME_INTERVAL == 0)
            {
                MapFrame mf;
                mf.frameNumber = frameCount;

                // Resize and convert to gray
                resize(frame, frame, Size(640, 480));
                cvtColor(frame, mf.image, COLOR_BGR2GRAY);

                // Extract ORB features
                orb->detectAndCompute(mf.image, noArray(), mf.keypoints, mf.descriptors);

                // Assign a position in the map (simple linear path for now)
                // You can modify this based on your actual camera path
                float progress = float(frameCount) / totalFrames;
                mf.mapPosition.x = 100 + progress * 600;                  // Move left to right
                mf.mapPosition.y = 300 + sin(progress * 2 * CV_PI) * 200; // Some variation

                mapFrames.push_back(mf);

                cout << "Keyframe " << mapFrames.size()
                     << " (frame " << frameCount << "): "
                     << mf.keypoints.size() << " features" << endl;
            }

            frameCount++;
        }

        cap.release();
        cout << "Map built with " << mapFrames.size() << " keyframes" << endl;

        // Draw the map path
        DrawMapPath();
    }

    // Step 2: Relocalize using validation video
    void Relocalize(const string &validationVideo)
    {
        cout << "\n=== RELOCALIZING FROM VIDEO ===" << endl;
        cout << "Video: " << validationVideo << endl;

        if (mapFrames.empty())
        {
            cerr << "No map loaded! Build map first." << endl;
            return;
        }

        VideoCapture cap(validationVideo);
        if (!cap.isOpened())
        {
            cerr << "Cannot open validation video!" << endl;
            return;
        }

        namedWindow("Current View", WINDOW_AUTOSIZE);
        namedWindow("Best Match", WINDOW_AUTOSIZE);
        namedWindow("Your Location on Map", WINDOW_AUTOSIZE);

        Mat frame;
        int frameCount = 0;

        cout << "\nPress ESC to exit\n"
             << endl;

        while (cap.read(frame))
        {
            // Process frame
            resize(frame, frame, Size(640, 480));
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            // Extract features
            vector<KeyPoint> keypoints;
            Mat descriptors;
            orb->detectAndCompute(gray, noArray(), keypoints, descriptors);

            // Find best matching keyframe
            int bestMatch = -1;
            int maxMatches = 0;
            vector<DMatch> bestDMatches;

            for (size_t i = 0; i < mapFrames.size(); i++)
            {
                if (descriptors.empty() || mapFrames[i].descriptors.empty())
                    continue;

                // Match features
                vector<vector<DMatch>> matches;
                matcher->knnMatch(descriptors, mapFrames[i].descriptors, matches, 2);

                // Filter good matches using Lowe's ratio test
                vector<DMatch> goodMatches;
                for (const auto &match : matches)
                {
                    if (match.size() == 2 &&
                        match[0].distance < MATCH_RATIO * match[1].distance)
                    {
                        goodMatches.push_back(match[0]);
                    }
                }

                // Check if this is the best match so far
                if (goodMatches.size() > maxMatches && goodMatches.size() >= MIN_MATCHES)
                {
                    maxMatches = goodMatches.size();
                    bestMatch = i;
                    bestDMatches = goodMatches;
                }
            }

            // Update position if match found
            Mat currentView = frame.clone();
            Mat matchView = Mat::zeros(480, 640, CV_8UC3);

            if (bestMatch >= 0)
            {
                currentPosition = mapFrames[bestMatch].mapPosition;
                trajectory.push_back(currentPosition);

                // Draw current view with status
                putText(currentView, "LOCALIZED", Point(10, 30),
                        FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                putText(currentView, "Matches: " + to_string(maxMatches), Point(10, 60),
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

                // Show matches
                drawMatches(gray, keypoints,
                            mapFrames[bestMatch].image, mapFrames[bestMatch].keypoints,
                            bestDMatches, matchView);

                cout << "Frame " << frameCount
                     << " -> Matched to map frame " << mapFrames[bestMatch].frameNumber
                     << " (" << maxMatches << " matches)" << endl;
            }
            else
            {
                putText(currentView, "LOST", Point(10, 30),
                        FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                putText(matchView, "No match found", Point(200, 240),
                        FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

                cout << "Frame " << frameCount << " -> No match found" << endl;
            }

            // Draw map with current position
            Mat mapView = DrawMap();

            // Show all windows
            imshow("Current View", currentView);
            imshow("Best Match", matchView);
            imshow("Your Location on Map", mapView);

            frameCount++;

            // Check for exit
            if (waitKey(30) == 27)
                break; // ESC key
        }

        cap.release();
        destroyAllWindows();

        // Print summary
        cout << "\n=== SUMMARY ===" << endl;
        cout << "Processed " << frameCount << " frames" << endl;
        cout << "Successfully localized " << trajectory.size() << " times" << endl;
        float successRate = 100.0f * trajectory.size() / frameCount;
        cout << "Success rate: " << successRate << "%" << endl;
    }

private:
    void DrawMapPath()
    {
        // Draw the path of keyframes on the map
        for (size_t i = 1; i < mapFrames.size(); i++)
        {
            line(mapImage, mapFrames[i - 1].mapPosition, mapFrames[i].mapPosition,
                 Scalar(100, 100, 100), 2);
        }

        // Draw keyframe positions
        for (const auto &mf : mapFrames)
        {
            circle(mapImage, mf.mapPosition, 5, Scalar(0, 150, 0), -1);
        }
    }

    Mat DrawMap()
    {
        Mat view = mapImage.clone();

        // Draw trajectory
        for (size_t i = 1; i < trajectory.size(); i++)
        {
            line(view, trajectory[i - 1], trajectory[i], Scalar(0, 255, 255), 2);
        }

        // Draw current position as a large red dot
        circle(view, currentPosition, 10, Scalar(0, 0, 255), -1);
        circle(view, currentPosition, 15, Scalar(0, 0, 255), 2);

        // Draw arrow pointing in movement direction
        if (trajectory.size() > 1)
        {
            Point2f dir = currentPosition - trajectory[trajectory.size() - 2];
            float len = norm(dir);
            if (len > 0)
            {
                dir = dir * (20.0f / len);
                Point2f arrowEnd = currentPosition + dir;
                arrowedLine(view, currentPosition, arrowEnd, Scalar(255, 0, 0), 3);
            }
        }

        // Add legend
        putText(view, "MAP VIEW", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        putText(view, "Gray: Mapping path", Point(10, 55),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 100, 100), 1);
        putText(view, "Green: Keyframe positions", Point(10, 75),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 150, 0), 1);
        putText(view, "Yellow: Your trajectory", Point(10, 95),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        putText(view, "Red: Current position", Point(10, 115),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

        // Position coordinates
        stringstream ss;
        ss << "Position: [" << (int)currentPosition.x << ", " << (int)currentPosition.y << "]";
        putText(view, ss.str(), Point(10, 580),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        return view;
    }
};

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "\nUsage: ./simple_relocalization mapping_video.mp4 validation_video.mp4\n"
             << endl;
        cout << "This program:" << endl;
        cout << "1. Builds a map from the mapping video" << endl;
        cout << "2. Finds your location by matching features from validation video" << endl;
        cout << "3. Shows your position on the map in real-time\n"
             << endl;
        return 1;
    }

    SimpleRelocalization reloc;

    // Step 1: Build map from mapping video
    reloc.BuildMapFromVideo(argv[1]);

    // Step 2: Relocalize using validation video
    reloc.Relocalize(argv[2]);

    return 0;
}