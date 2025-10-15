#include "relocalization.h"
#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "=========================================" << std::endl;
        std::cout << " ORB-SLAM3 Relocalization Module" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "\nUsage: " << argv[0]
                  << " <ORBvoc.txt> <config.yaml> <validation_video>" << std::endl;
        std::cout << "\nArguments:" << std::endl;
        std::cout << "  ORBvoc.txt        - Path to ORB vocabulary file" << std::endl;
        std::cout << "  config.yaml       - Path to configuration file" << std::endl;
        std::cout << "  validation_video  - Path to validation video file" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " \\" << std::endl;
        std::cout << "    ~/ORB_SLAM3/Vocabulary/ORBvoc.txt \\" << std::endl;
        std::cout << "    config/webcam_complete.yaml \\" << std::endl;
        std::cout << "    data/validation.mp4" << std::endl;
        std::cout << std::endl;
        return 1;
    }

    // Parse arguments
    std::string vocabPath = argv[1];
    std::string configPath = argv[2];
    std::string videoPath = argv[3];

    std::cout << "==================================" << std::endl;
    std::cout << " ORB-SLAM3 Relocalization Module " << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Vocabulary: " << vocabPath << std::endl;
    std::cout << "  Config: " << configPath << std::endl;
    std::cout << "  Video: " << videoPath << std::endl;
    std::cout << std::endl;

    // Create relocalization module (loads config internally)
    Relocalization::RelocalizationModule reloc(vocabPath, configPath);

    // Load the map
    std::cout << "Step 1: Loading map..." << std::endl;
    if (!reloc.loadMap())
    {
        std::cerr << "Failed to load map!" << std::endl;
        return 1;
    }

    reloc.debugStatus();

    // Process video
    std::cout << "\nStep 2: Processing validation video..." << std::endl;
    reloc.processVideo(videoPath);

    std::cout << "\n✓ Done!" << std::endl;
    std::cout << "\nPress any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}