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
        std::cout << "\nUsage:" << std::endl;
        std::cout << "  Video:  " << argv[0] << " <ORBvoc.txt> <config.yaml> <video_path> [--no-viz]" << std::endl;
        std::cout << "  Webcam: " << argv[0] << " <ORBvoc.txt> <config.yaml> --webcam [device_id] [--no-viz]" << std::endl;
        std::cout << "\nArguments:" << std::endl;
        std::cout << "  ORBvoc.txt        - Path to ORB vocabulary file" << std::endl;
        std::cout << "  config.yaml       - Path to configuration file" << std::endl;
        std::cout << "  video_path        - Path to video file" << std::endl;
        std::cout << "  --webcam [id]     - Use webcam input (default device id: 0)" << std::endl;
        std::cout << "  --no-viz          - (Optional) Disable visualization windows" << std::endl;
        return 1;
    }

    // Parse arguments
    std::string vocabPath = argv[1];
    std::string configPath = argv[2];
    std::string videoPath;
    bool useWebcam = false;
    int webcamId = 0;
    bool enableVisualization = true;

    // argv[3] is either a video path or --webcam
    if (std::string(argv[3]) == "--webcam")
    {
        useWebcam = true;
        int nextArg = 4;
        // Check if next arg is a device id (numeric)
        if (argc > nextArg && std::string(argv[nextArg]).find("--") == std::string::npos)
        {
            webcamId = std::stoi(argv[nextArg]);
            nextArg++;
        }
        if (argc > nextArg && std::string(argv[nextArg]) == "--no-viz")
        {
            enableVisualization = false;
            std::cout << "[INFO] Visualization disabled via command line" << std::endl;
        }
    }
    else
    {
        videoPath = argv[3];
        if (argc > 4 && std::string(argv[4]) == "--no-viz")
        {
            enableVisualization = false;
            std::cout << "[INFO] Visualization disabled via command line" << std::endl;
        }
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
    if (useWebcam)
        std::cout << "  Input: Webcam (device " << webcamId << ")" << std::endl;
    else
        std::cout << "  Input: Video - " << videoPath << std::endl;
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

    // Process input
    if (useWebcam)
    {
        std::cout << "\nStep 2: Processing webcam input..." << std::endl;
        reloc.processWebcam(webcamId);
    }
    else
    {
        std::cout << "\nStep 2: Processing validation video..." << std::endl;
        reloc.processVideo(videoPath);
    }

    std::cout << "\n✓ Done!" << std::endl;

    // Only wait for keypress if visualization is enabled (video mode only;
    // webcam already blocks until ESC)
    if (enableVisualization && !useWebcam)
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