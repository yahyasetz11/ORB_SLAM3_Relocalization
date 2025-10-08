#include <iostream>
#include <thread>
#include <chrono>
#include "System.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <vocab> <config>" << std::endl;
        return 1;
    }

    try
    {
        ORB_SLAM3::System SLAM(argv[1], argv[2],
                               ORB_SLAM3::System::MONOCULAR, false);
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Get ALL maps
        auto vpMaps = SLAM.mpAtlas->GetAllMaps();

        std::cout << "\n========================================" << std::endl;
        std::cout << "Atlas has " << vpMaps.size() << " map(s)" << std::endl;
        std::cout << "========================================\n"
                  << std::endl;

        ORB_SLAM3::Map *pMapWithData = nullptr;

        for (size_t i = 0; i < vpMaps.size(); i++)
        {
            auto pMap = vpMaps[i];
            auto kfs = pMap->GetAllKeyFrames();
            auto mps = pMap->GetAllMapPoints();

            std::cout << "Map " << i << " (ID: " << pMap->GetId() << "):" << std::endl;
            std::cout << "  KeyFrames: " << kfs.size() << std::endl;
            std::cout << "  MapPoints: " << mps.size() << std::endl;

            if (!kfs.empty() && !pMapWithData)
            {
                pMapWithData = pMap;
                std::cout << "  ✓ This is the map with data!" << std::endl;
            }
            std::cout << std::endl;
        }

        if (pMapWithData)
        {
            std::cout << "========================================" << std::endl;
            std::cout << "✓ Found map with data!" << std::endl;
            std::cout << "========================================" << std::endl;

            auto kfs = pMapWithData->GetAllKeyFrames();
            auto mps = pMapWithData->GetAllMapPoints();

            std::cout << "Total KeyFrames: " << kfs.size() << std::endl;
            std::cout << "Total MapPoints: " << mps.size() << std::endl;

            // Sample
            int count = 0;
            std::cout << "\nSample map points:" << std::endl;
            for (auto mp : mps)
            {
                if (mp && !mp->isBad())
                {
                    Eigen::Vector3f pos = mp->GetWorldPos();
                    std::cout << "  [" << pos(0) << ", " << pos(1) << ", " << pos(2) << "]" << std::endl;
                    if (++count >= 5)
                        break;
                }
            }

            std::cout << "\n✓✓✓ SUCCESS! Map loaded correctly! ✓✓✓" << std::endl;
        }
        else
        {
            std::cout << "✗ No map with data found!" << std::endl;
        }

        SLAM.Shutdown();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}