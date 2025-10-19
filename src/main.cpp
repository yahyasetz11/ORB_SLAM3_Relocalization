#include "relocalization.h"
#include <iostream>
#include <cstdlib> // for getenv

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "=========================================" << std::endl;
        std::cout << " ORB-SLAM3 Relocalization Module" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "\nUsage: " << argv[0]
                  << " <ORBvoc.txt> <config.yaml> <validation_video> [--no-viz]" << std::endl;
        std::cout << "\nArguments:" << std::endl;
        std::cout << "  ORBvoc.txt        - Path to ORB vocabulary file" << std::endl;
        std::cout << "  config.yaml       - Path to configuration file" << std::endl;
        std::cout << "  validation_video  - Path to validation video file" << std::endl;
        std::cout << "  --no-viz         - (Optional) Disable visualization windows" << std::endl;
        return 1;
    }

    // Parse arguments
    std::string vocabPath = argv[1];
    std::string configPath = argv[2];
    std::string videoPath = argv[3];

    // Check if user wants to disable visualization
    bool enableVisualization = true;
    if (argc > 4 && std::string(argv[4]) == "--no-viz")
    {
        enableVisualization = false;
        std::cout << "[INFO] Visualization disabled via command line" << std::endl;
    }

    // Auto-detect if display is available
    const char *display = std::getenv("DISPLAY");
    const char *waylandDisplay = std::getenv("WAYLAND_DISPLAY");

    if (!display && !waylandDisplay)
    {
        std::cout << "[WARNING] No DISPLAY or WAYLAND_DISPLAY found - disabling visualization" << std::endl;
        enableVisualization = false;
    }

    std::cout << "==================================" << std::endl;
    std::cout << " ORB-SLAM3 Relocalization Module " << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Vocabulary: " << vocabPath << std::endl;
    std::cout << "  Config: " << configPath << std::endl;
    std::cout << "  Video: " << videoPath << std::endl;
    std::cout << "  Visualization: " << (enableVisualization ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << std::endl;

    // Create relocalization module
    Relocalization::RelocalizationModule reloc(vocabPath, configPath);

    // Override visualization setting
    reloc.setVisualizationEnabled(enableVisualization);

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

    // Only wait for keypress if visualization is enabled
    if (enableVisualization)
    {
        std::cout << "\nPress any key to exit..." << std::endl;
        try
        {
            cv::waitKey(0);
        }
        catch (const cv::Exception &e)
        {
            std::cout << "[WARNING] Display error during waitKey: " << e.what() << std::endl;
        }
        cv::destroyAllWindows();
    }

    return 0;
}