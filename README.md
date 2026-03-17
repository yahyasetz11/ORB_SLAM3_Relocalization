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

The relocalization module supports two input modes: **video file** or **live webcam**.

#### Video Mode

```bash
./build/relocalization \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam_complete.yaml \
    validation_video.mp4
```

#### Webcam Mode

```bash
# Default webcam (device 0)
./build/relocalization \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam_complete.yaml \
    --webcam

# Specific webcam device
./build/relocalization \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam_complete.yaml \
    --webcam 1
```

Press `ESC` to stop webcam mode.

#### Headless Mode (no visualization)

```bash
# Video
./build/relocalization \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam_complete.yaml \
    validation_video.mp4 \
    --no-viz

# Webcam
./build/relocalization \
    ../ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config/webcam_complete.yaml \
    --webcam 0 \
    --no-viz
```

## ROS2 Humble Nodes

### Prerequisites

- ROS2 Humble installed and sourced
- A colcon workspace

### Build

```bash
cd ~/your_ws
colcon build --symlink-install --packages-select orb_slam3_relocalization
source install/setup.bash
```

### 1. Create Map Node

Runs ORB-SLAM3 monocular SLAM and saves the map to a `.osa` file.

#### From video file (default)

```bash
ros2 launch orb_slam3_relocalization map_creator.launch.py
```

#### From webcam (stream mode)

```bash
ros2 launch orb_slam3_relocalization map_creator.launch.py mode:=stream
```

Press `Ctrl+C` to stop and save the map.

#### Change input paths or camera

Edit `config/map_creator_params.yaml`:

```yaml
map_creator_node:
  ros__parameters:
    vocab_path:  "./../ORB_SLAM3/Vocabulary/ORBvoc.txt"
    config_path: "./config/webcam_complete.yaml"
    video_path:  "./data/your_video.mp4"   # path to MP4 (used in video mode)
    camera_id:   0                          # webcam device index (used in stream mode)
```

### 2. Relocalization Node

Loads a pre-built map and publishes estimated pose on `/relocalization/pose`.

First, ensure your config has the map path:
```yaml
System.LoadAtlasFromFile: "maps/indoor_map.osa"
```

#### From video file (default)

```bash
ros2 launch orb_slam3_relocalization relocalization.launch.py
```

#### From webcam (stream mode)

```bash
ros2 launch orb_slam3_relocalization relocalization.launch.py mode:=stream
```

Press `ESC` in the visualization window or `Ctrl+C` to stop.

#### Change input paths or camera

Edit `config/relocalization_params.yaml`:

```yaml
relocalization_node:
  ros__parameters:
    vocab_path:  "./../ORB_SLAM3/Vocabulary/ORBvoc.txt"
    config_path: "./config/webcam_complete.yaml"
    video_path:  "./data/your_video.mp4"   # path to MP4 (used in video mode)
    camera_id:   0                          # webcam device index (used in stream mode)
    visualize:   true                       # set false to disable OpenCV window
```

#### Monitor published pose

```bash
ros2 topic echo /relocalization/pose
```

---

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