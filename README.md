# ORB-SLAM3 Relocalization

Monocular SLAM with relocalization capabilities using ORB-SLAM3.

## Prerequisites

- ORB_SLAM3 installed at `../ORB_SLAM3`
- OpenCV 4
- Eigen3
- Pangolin
- RealSense SDK (optional, for RealSense camera)

## Camera Calibration

### 1. Change Camera Parameters

Edit the configuration file in `config/`:
- `webcam.yaml` - for webcam
- `ip12_cam.yaml` - for iPhone camera
- `realsense.yaml` - for RealSense camera

Key parameters to adjust:
```yaml
Camera.fx: 500.0    # Focal length X
Camera.fy: 500.0    # Focal length Y
Camera.cx: 320.0    # Principal point X
Camera.cy: 240.0    # Principal point Y
Camera.k1: 0.0      # Distortion coefficients
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
```

### 2. Calibrate Your Camera (Optional)

```bash
# Generate checkerboard pattern
python3 calibration/generate_checkerboard.py --cols 9 --rows 6

# Record calibration video (move checkerboard around)
# Then run calibration:
python3 calibration/calibrate_camera.py calibration_video.mp4 9 6

# Use generated camera_calibration.yaml as your config
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Usage

### 1. Create Map with Webcam

```bash
./build/slam_webcam_simple \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam.yaml
```

Map will be saved to location specified in `config/webcam.yaml`:
```yaml
System.SaveAtlasToFile: "maps/indoor_map.osa"
```

### 2. Create Map with Pre-recorded MP4

```bash
./build/main_mp4_mono \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam.yaml \
    video.mp4
```

### 3. Run Relocalization

First, ensure your config has the map path:
```yaml
System.LoadAtlasFromFile: "maps/indoor_map.osa"
```

Then run:
```bash
./build/relocalization \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam_complete.yaml \
    validation_video.mp4
```

**Headless mode (no visualization):**
```bash
./build/relocalization \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam_complete.yaml \
    validation_video.mp4 \
    --no-viz
```

## Configuration Tips

- **Creating map**: Use `System.SaveAtlasToFile` in config
- **Relocalization**: Use `System.LoadAtlasFromFile` in config
- **Better tracking**: Increase `ORBextractor.nFeatures` (e.g., 2000)
- **Low-light scenes**: Lower `ORBextractor.iniThFAST` and `minThFAST`

## Troubleshooting

**Map not loading?**
- Check file path in YAML is correct
- Verify map file exists (`.osa` extension)
- Use `./build/test_map` to verify map loads

**Tracking lost?**
- Ensure proper camera calibration
- Check lighting conditions
- Increase feature count in config

**Relocalization failing?**
- Adjust `Relocalization.BowSimilarityThreshold` (lower = more permissive)
- Adjust `Relocalization.MinInliers` (lower = easier matching)