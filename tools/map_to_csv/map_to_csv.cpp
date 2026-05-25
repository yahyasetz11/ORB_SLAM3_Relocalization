/**
 * Standalone tool — no ROS 2 required.
 *
 * Loads an ORB-SLAM3 atlas (.osa) referenced by the YAML config and writes:
 *   map_points.csv  — id, x, y, z   (all valid MapPoints)
 *   keyframes.csv   — id, tx, ty, tz, r00…r22  (camera-to-world poses)
 *
 * Usage:
 *   map_to_csv <vocab.txt> <config.yaml> [output_dir]
 *
 * The YAML must contain  System.LoadAtlasFromFile: "<map_stem>"  so that
 * ORB_SLAM3::System automatically loads <map_stem>.osa on construction.
 */

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "System.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <vocab.txt> <config.yaml> [output_dir]\n";
        return 1;
    }

    const std::string vocabPath  = argv[1];
    const std::string configPath = argv[2];
    const std::string outputDir  = (argc >= 4) ? argv[3] : ".";

    std::filesystem::create_directories(outputDir);

    // ── Load atlas (same pattern as test_map.cpp) ─────────────────────────────
    std::cout << "Initializing SLAM system (map path from YAML)...\n";
    ORB_SLAM3::System SLAM(vocabPath, configPath,
                           ORB_SLAM3::System::MONOCULAR, false /*no viewer*/);

    std::cout << "Waiting 2 s for atlas to finish loading...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));

    if (!SLAM.mpAtlas) {
        std::cerr << "ERROR: mpAtlas is null — check map path in YAML.\n";
        return 1;
    }

    // GetCurrentMap() returns a fresh empty map created post-load; the saved
    // data lives in the non-current maps, so pick the largest one.
    std::vector<ORB_SLAM3::Map *> vpMaps = SLAM.mpAtlas->GetAllMaps();
    std::cout << "Atlas contains " << vpMaps.size() << " map(s):\n";
    ORB_SLAM3::Map *pMap = nullptr;
    size_t bestKFCount = 0;
    for (auto *pCand : vpMaps) {
        size_t n = pCand->GetAllKeyFrames().size();
        std::cout << "  map id=" << pCand->GetId() << "  keyframes=" << n << "\n";
        if (n > bestKFCount) { bestKFCount = n; pMap = pCand; }
    }
    if (!pMap || bestKFCount == 0) {
        std::cerr << "ERROR: no map with keyframe data found in atlas.\n";
        return 1;
    }
    std::cout << "Using map id=" << pMap->GetId() << "\n\n";

    auto mappoints = pMap->GetAllMapPoints();
    auto keyframes = pMap->GetAllKeyFrames();
    std::cout << "Atlas has " << mappoints.size() << " total map points, "
              << keyframes.size() << " total keyframes.\n\n";

    // ── map_points.csv ────────────────────────────────────────────────────────
    const std::string mpPath = outputDir + "/map_points.csv";
    {
        std::ofstream f(mpPath);
        if (!f) { std::cerr << "ERROR: cannot write " << mpPath << "\n"; return 1; }
        f << std::fixed << std::setprecision(8);
        f << "id,x,y,z\n";
        int n = 0;
        for (auto pMP : mappoints) {
            if (!pMP || pMP->isBad()) continue;
            Eigen::Vector3f pos = pMP->GetWorldPos();
            f << pMP->mnId << ","
              << pos(0) << "," << pos(1) << "," << pos(2) << "\n";
            ++n;
        }
        std::cout << "Wrote " << n << " valid map points  ->  " << mpPath << "\n";
    }

    // ── keyframes.csv ─────────────────────────────────────────────────────────
    const std::string kfPath = outputDir + "/keyframes.csv";
    {
        std::ofstream f(kfPath);
        if (!f) { std::cerr << "ERROR: cannot write " << kfPath << "\n"; return 1; }
        f << std::fixed << std::setprecision(8);
        f << "id,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";
        int n = 0;
        for (auto pKF : keyframes) {
            if (!pKF || pKF->isBad()) continue;
            Sophus::SE3f    pose = pKF->GetPoseInverse(); // camera-to-world
            Eigen::Vector3f t    = pose.translation();
            Eigen::Matrix3f R    = pose.rotationMatrix();
            f << pKF->mnId << ","
              << t(0) << "," << t(1) << "," << t(2) << ","
              << R(0,0) << "," << R(0,1) << "," << R(0,2) << ","
              << R(1,0) << "," << R(1,1) << "," << R(1,2) << ","
              << R(2,0) << "," << R(2,1) << "," << R(2,2) << "\n";
            ++n;
        }
        std::cout << "Wrote " << n << " valid keyframes    ->  " << kfPath << "\n";
    }

    SLAM.Shutdown();

    // ── pandas snippet ────────────────────────────────────────────────────────
    std::cout << "\n"
              << "# Load the CSVs in Python:\n"
              << "#\n"
              << "#   import pandas as pd\n"
              << "#\n"
              << "#   mp = pd.read_csv('" << mpPath << "')\n"
              << "#   print(mp.head())\n"
              << "#   #    id         x         y         z\n"
              << "#   # 0   0  0.12345  -0.04567   0.98765\n"
              << "#\n"
              << "#   kf = pd.read_csv('" << kfPath << "')\n"
              << "#   print(kf[['id','tx','ty','tz']].head())\n"
              << "#   # Rotation as 3x3 matrix for row i:\n"
              << "#   # R = kf.iloc[i][['r00','r01','r02',\n"
              << "#   #                  'r10','r11','r12',\n"
              << "#   #                  'r20','r21','r22']].values.reshape(3,3)\n";

    return 0;
}
