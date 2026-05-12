// One-time offline conversion tool — NOT part of the running ROS navigation
// stack. Run this once before launching the navigation system to convert an
// ORB-SLAM3 binary atlas (.osa) into a human-readable CSV of 3-D map points.
//
// Usage:
//   ros2 run navigation osa_to_csv --osa map1.osa --output map.csv

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <unistd.h>

#include "pangolin_stub.h"
#include "System.h"
#include "Atlas.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include <Eigen/Core>

// ---------------------------------------------------------------------------
// load_map
//   Opens the .osa binary atlas file via ORB_SLAM3::System (which reads
//   System.LoadAtlasFromFile from the settings YAML) and extracts all
//   map points from the current map.
//
//   Returns: vector of ORB_SLAM3::MapPoint* (non-null pointers).
// ---------------------------------------------------------------------------
std::vector<ORB_SLAM3::MapPoint*> load_map(
    const std::string& osa_path,
    const std::string& vocab_path,
    const std::string& config_path)
{
    std::cout << "[load_map] Config path passed to ORB_SLAM3::System: " << config_path << "\n";
    std::cout << "[load_map] OSA path (for reference): " << osa_path << "\n";

    ORB_SLAM3::System slam(vocab_path, config_path,
                           ORB_SLAM3::System::MONOCULAR, false /*no viewer*/);

    ORB_SLAM3::Atlas* atlas = slam.mpAtlas;

    // GetCurrentMap() returns a newly created empty map that ORB-SLAM3 creates
    // for continued operation after loading the atlas.  The saved data lives in
    // the other maps returned by GetAllMaps(), so we collect from all of them.
    std::vector<ORB_SLAM3::Map*> all_maps = atlas->GetAllMaps();
    std::cout << "[load_map] Total maps in atlas: " << all_maps.size() << "\n";
    std::cout << "[load_map] Current map id: " << atlas->GetCurrentMap()->GetId() << " (empty — new map created post-load)\n";

    size_t total_raw = 0;
    int total_bad = 0;
    std::vector<ORB_SLAM3::MapPoint*> points;
    for (ORB_SLAM3::Map* m : all_maps) {
        std::vector<ORB_SLAM3::KeyFrame*> kfs = m->GetAllKeyFrames();
        std::vector<ORB_SLAM3::MapPoint*> pts = m->GetAllMapPoints();
        std::cout << "[load_map]   map id=" << m->GetId()
                  << "  keyframes=" << kfs.size()
                  << "  raw_points=" << pts.size() << "\n";
        total_raw += pts.size();
        for (ORB_SLAM3::MapPoint* pMP : pts) {
            if (!pMP) continue;
            if (pMP->isBad()) { total_bad++; continue; }
            points.push_back(pMP);
        }
    }
    std::cout << "[load_map] Total map points across all maps BEFORE isBad() filter: " << total_raw << "\n";
    std::cout << "[load_map] Map points marked isBad(): " << total_bad << "\n";
    std::cout << "[load_map] Map points AFTER isBad() filter: " << points.size() << "\n";

    return points;
}

// ---------------------------------------------------------------------------
// write_csv
//   Writes map points to a comma-separated file with the header:
//     x,y,z,label
//   Each row is one map point in world coordinates (metres), labelled "wall".
// ---------------------------------------------------------------------------
void write_csv(const std::vector<ORB_SLAM3::MapPoint*>& points,
               const std::string& output_path)
{
    std::cout << "[write_csv] Writing " << points.size()
              << " points to: " << output_path << "\n";

    std::ofstream out(output_path);
    // Caller already verified this opens; no need to re-check.
    out << "x,y,z,label\n";
    for (ORB_SLAM3::MapPoint* pMP : points) {
        Eigen::Vector3f pos = pMP->GetWorldPos();
        out << pos(0) << ","
            << pos(1) << ","
            << pos(2) << ","
            << "wall\n";
    }
    std::cout << "[write_csv] Done.\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string osa_path;
    std::string output_path;
    std::string vocab_path;
    std::string config_path;

    // Simple flag parsing — no external dependency required.
    for (int i = 1; i < argc - 1; ++i) {
        std::string flag(argv[i]);
        if (flag == "--osa")    osa_path    = argv[i + 1];
        if (flag == "--output") output_path = argv[i + 1];
        if (flag == "--vocab")  vocab_path  = argv[i + 1];
        if (flag == "--config") config_path = argv[i + 1];
    }

    if (osa_path.empty() || output_path.empty() ||
        vocab_path.empty() || config_path.empty()) {
        std::cerr << "Usage: osa_to_csv --osa <map.osa> --output <map.csv>"
                     " --vocab <vocab.txt> --config <settings.yaml>\n";
        return 1;
    }

    // Validate input exists before attempting to load ORB-SLAM3.
    {
        std::ifstream probe(osa_path);
        if (!probe.good()) {
            std::cerr << "Error: cannot open input file: " << osa_path << "\n";
            return 1;
        }
    }

    // Validate output is writable before doing expensive map loading.
    {
        std::ofstream probe(output_path);
        if (!probe.good()) {
            std::cerr << "Error: cannot open output file: " << output_path << "\n";
            return 1;
        }
    }

    // Strip .osa extension if present — ORB-SLAM3 appends it automatically.
    std::string atlas_stem = osa_path;
    if (atlas_stem.size() >= 4 &&
        atlas_stem.substr(atlas_stem.size() - 4) == ".osa")
        atlas_stem.erase(atlas_stem.size() - 4);

    // ORB-SLAM3's LoadAtlas() prepends "./" unconditionally, so an absolute
    // path like "/home/.../maps/map" becomes ".//home/.../maps/map" which
    // resolves as a relative path and fails.  Fix: chdir to the atlas directory
    // and pass only the basename — then "./" + "map" = "./map" works correctly.
    {
        auto slash = atlas_stem.rfind('/');
        if (slash != std::string::npos) {
            std::string atlas_dir  = atlas_stem.substr(0, slash);
            std::string atlas_base = atlas_stem.substr(slash + 1);
            if (chdir(atlas_dir.c_str()) != 0) {
                std::cerr << "Error: cannot chdir to atlas directory: " << atlas_dir << "\n";
                return 1;
            }
            atlas_stem = atlas_base;
        }
    }

    // Patch System.LoadAtlasFromFile in a temp copy of the YAML so that
    // ORB-SLAM3 reads the atlas we actually want, not whatever the file says.
    const std::string tmp_config = "/tmp/osa_to_csv_config.yaml";
    {
        std::ifstream cfg_in(config_path);
        if (!cfg_in.good()) {
            std::cerr << "Error: cannot open config file: " << config_path << "\n";
            return 1;
        }
        std::ofstream cfg_out(tmp_config);
        if (!cfg_out.good()) {
            std::cerr << "Error: cannot write temp config: " << tmp_config << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(cfg_in, line)) {
            if (line.find("System.LoadAtlasFromFile") != std::string::npos) {
                auto colon = line.find(':');
                if (colon != std::string::npos)
                    line = line.substr(0, colon + 1) + " \"" + atlas_stem + "\"";
            }
            cfg_out << line << "\n";
        }
    }

    std::cout << "[osa_to_csv] Step 1/2 — loading map...\n";
    auto points = load_map(osa_path, vocab_path, tmp_config);

    std::cout << "[osa_to_csv] Step 2/2 — writing CSV...\n";
    write_csv(points, output_path);

    std::remove(tmp_config.c_str());

    std::cout << "[osa_to_csv] Conversion complete.\n";
    return 0;
}
