/**
 * Simple test to load map using YAML config
 * The System constructor automatically loads the atlas from the YAML
 */

#include <iostream>
#include <thread>
#include <chrono>
#include "System.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <vocab.txt> <config.yaml>" << std::endl;
        std::cout << "\nThe config.yaml should contain:" << std::endl;
        std::cout << "  System.LoadAtlasFromFile: \"map\"" << std::endl;
        std::cout << "\nAnd the map file should be: ./map.osa" << std::endl;
        return 1;
    }

    std::string vocabPath = argv[1];
    std::string configPath = argv[2];

    std::cout << "==========================================" << std::endl;
    std::cout << " Map Loading Test (YAML Config)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Vocabulary: " << vocabPath << std::endl;
    std::cout << "Config: " << configPath << std::endl;
    std::cout << "\nNote: Map file path is read from the YAML config" << std::endl;
    std::cout << "      System will look for ./<name>.osa" << std::endl;

    std::cout << "\nInitializing SLAM system..." << std::endl;
    std::cout << "(LoadAtlas() is called automatically by System constructor)\n"
              << std::endl;

    try
    {
        // Create SLAM system
        // The constructor reads System.LoadAtlasFromFile from YAML
        // and automatically calls LoadAtlas()
        ORB_SLAM3::System SLAM(vocabPath, configPath,
                               ORB_SLAM3::System::MONOCULAR,
                               false); // false = no viewer

        std::cout << "✓ System created!" << std::endl;

        // Give it a moment to finish initialization
        std::cout << "\nWaiting for initialization..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Access the atlas (now public)
        std::cout << "\nAccessing atlas..." << std::endl;

        if (SLAM.mpAtlas == nullptr)
        {
            std::cerr << "✗ mpAtlas is NULL!" << std::endl;
            std::cerr << "\nPossible reasons:" << std::endl;
            std::cerr << "  1. Map file not found (check path in YAML)" << std::endl;
            std::cerr << "  2. Vocabulary mismatch" << std::endl;
            std::cerr << "  3. Corrupted map file" << std::endl;
            return 1;
        }

        std::cout << "✓ Atlas accessed!" << std::endl;

        // Get current map
        std::cout << "\nGetting current map..." << std::endl;
        ORB_SLAM3::Map *pMap = SLAM.mpAtlas->GetCurrentMap();

        if (!pMap)
        {
            std::cerr << "✗ No current map found in atlas!" << std::endl;
            return 1;
        }

        std::cout << "✓ Map obtained!" << std::endl;

        // Extract data
        std::cout << "\nExtracting map data..." << std::endl;
        auto keyframes = pMap->GetAllKeyFrames();
        auto mappoints = pMap->GetAllMapPoints();

        std::cout << "\n==========================================" << std::endl;
        std::cout << " Map Statistics" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Total KeyFrames: " << keyframes.size() << std::endl;
        std::cout << "Total MapPoints: " << mappoints.size() << std::endl;

        // Count valid points
        int validPoints = 0;
        for (auto pMP : mappoints)
        {
            if (pMP && !pMP->isBad())
            {
                validPoints++;
            }
        }
        std::cout << "Valid MapPoints: " << validPoints << std::endl;

        // Sample map points
        std::cout << "\n==========================================" << std::endl;
        std::cout << " Sample Map Points (first 5)" << std::endl;
        std::cout << "==========================================" << std::endl;

        int count = 0;
        for (auto pMP : mappoints)
        {
            if (pMP && !pMP->isBad())
            {
                Eigen::Vector3f pos = pMP->GetWorldPos();
                std::cout << "Point " << count << ": ["
                          << pos(0) << ", " << pos(1) << ", " << pos(2) << "]" << std::endl;
                count++;
                if (count >= 5)
                    break;
            }
        }

        // Sample keyframes
        std::cout << "\n==========================================" << std::endl;
        std::cout << " Sample KeyFrames (first 3)" << std::endl;
        std::cout << "==========================================" << std::endl;

        for (size_t i = 0; i < std::min((size_t)3, keyframes.size()); i++)
        {
            auto pKF = keyframes[i];
            if (pKF && !pKF->isBad())
            {
                std::cout << "KeyFrame " << pKF->mnId << ":" << std::endl;
                std::cout << "  - Keypoints: " << pKF->mvKeysUn.size() << std::endl;
                std::cout << "  - Descriptors: " << pKF->mDescriptors.rows
                          << " x " << pKF->mDescriptors.cols << std::endl;
                std::cout << "  - BoW vector size: " << pKF->mBowVec.size() << std::endl;
            }
        }

        std::cout << "\n==========================================" << std::endl;
        std::cout << "✓✓✓ ALL TESTS PASSED! ✓✓✓" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "\n🎉 Your map loaded successfully!" << std::endl;
        std::cout << "You can now proceed with relocalization.\n"
                  << std::endl;

        // Shutdown
        std::cout << "Shutting down..." << std::endl;
        SLAM.Shutdown();
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n✗ Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}