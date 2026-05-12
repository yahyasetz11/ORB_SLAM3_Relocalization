# Navigation Package — Usage Guide

## Overview

The navigation package does **A\* path planning** on a 2-D occupancy grid built from ORB-SLAM3 map points. It has three main components:

| Component | Executable | Role |
|---|---|---|
| `map_publisher` | `ros2 run navigation map_publisher` | Loads a map (CSV → PNG occupancy grid) and publishes it at 1 Hz |
| `navigation` | `ros2 run navigation navigation` | Subscribes to the grid, runs A\*, publishes planned path |
| `navigation_ui` | `ros2 run navigation navigation_ui` | PyQt5 window — shows the map, lets you click a goal |

---

## Do you need to convert the map to CSV first?

**Yes — once.** The pipeline is:

```
ORB-SLAM3 map (.osa)
        │
        ▼  [osa_to_csv — run once offline]
  map points (.csv)
        │
        ▼  [map_publisher — automatic, at runtime]
  occupancy grid (.png, cached)
        │
        ▼  [navigation_node]
    A* path
```

After the first run `map_publisher` saves a `.png` beside the `.csv`. On every subsequent run it loads the PNG directly — **the CSV conversion only happens once**.

---

## Step-by-step: using a real ORB-SLAM3 map

### 1. Build and source

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select navigation
source install/setup.bash
```

### 2. Convert the `.osa` map to CSV

```bash
ros2 run navigation osa_to_csv \
  --osa    ~/Documents/ORB_SLAM3_Relocalization/maps/<map_name>.osa \
  --output ~/Documents/ORB_SLAM3_Relocalization/src/navigation/navigation/maps/<map_name>.csv \
  --vocab  ~/Documents/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --config ~/Documents/ORB_SLAM3_Relocalization/src/orb_slam3_relocalization/config/tum_fr3.yaml
```

- The tool strips `.osa` automatically, so passing the full path (with or without extension) is fine.
- The CSV is written to `src/navigation/navigation/maps/` — that is the directory `map_publisher` looks in.
- Run this **once per map**. If the `.png` already exists for a map name, `map_publisher` skips CSV loading entirely.

### 3. Launch the navigation stack

```bash
ros2 launch navigation nav.launch.py
```

This starts `map_publisher`, `navigation_node`, `navigation_ui`, and `position_publisher`. The UI window opens automatically.

### 4. Load your map from the UI

In the UI window there is a **text input field** at the top. Type the map name (filename without extension, e.g. `map`) and press **Enter**. The UI publishes the name to the `/map_name` topic, `map_publisher` loads the file, and the map appears in the window.

- First run: `map_publisher` reads the CSV, builds the occupancy grid, and saves a PNG cache alongside it.
- Every subsequent run: it loads the PNG directly — the CSV conversion only ever runs once.

You can switch maps at runtime by typing a different name and pressing Enter again — no restart needed.

### 5. Set a goal

**Click anywhere on the map** in the UI window. The UI converts the click position to pixel coordinates and publishes it to `goal_position`. The navigation node replans immediately.

### 6. Send the robot's current position

`navigation_node` listens on `current_position` (`PointStamped`, pixel coordinates).  
In the full pipeline this is wired to the relocalization node output. The `position_publisher` node included in `nav.launch.py` is a placeholder stub for testing.

---

## Testing without a real map or robot

Use the fake-data launch. It publishes a synthetic 200×200 grid with obstacles and moves a fake robot in an oval:

```bash
ros2 launch navigation test_navigation.launch.py
```

No map file needed — the fake publisher generates the occupancy grid in memory.

---

## ROS topics reference

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/map_name` | `std_msgs/String` | UI → map_publisher | Map filename (no extension) — sent by the UI text box |
| `/map_image` | `sensor_msgs/Image` | map_publisher → | Occupancy grid as a BGR image |
| `current_position` | `geometry_msgs/PointStamped` | → navigation_node | Robot pixel position in the grid |
| `goal_position` | `geometry_msgs/PointStamped` | → navigation_node | Goal pixel position in the grid |
| `smooth_path` | `nav_msgs/Path` | navigation_node → | Planned and smoothed path |
| `static_image` | `sensor_msgs/Image` | navigation_node → | Grid image without path |
| `path_image` | `sensor_msgs/Image` | navigation_node → | Grid image with path drawn |
| `trails_image` | `sensor_msgs/Image` | navigation_node → | Grid image with path + robot trail |

---

## Map directory

Maps live in:
```
~/Documents/ORB_SLAM3_Relocalization/src/navigation/navigation/maps/
```

Expected files:

| File | Created by | Required |
|---|---|---|
| `<name>.csv` | `osa_to_csv` | Yes (first run only) |
| `<name>.png` | `map_publisher` (auto-saved) | Cached — skips CSV on next run |

After adding a new CSV, run `colcon build --symlink-install --packages-select navigation` once so the install tree picks up the new file symlink.

---

## How `map_publisher` builds the occupancy grid from CSV

1. **Load** `x,y,z` columns from the CSV.
2. **Height filter** — keeps only points within ±1 std dev of the mean Y (removes floor/ceiling noise).
3. **Density filter** — removes points with fewer than 3 neighbours within 0.3 m (removes stray points).
4. **Grid projection** — projects X and Z (ignores Y/height) onto a 2-D grid at 0.02 m/cell resolution.
5. **Smoothing** — dilates with a 7×7 kernel then Gaussian blur, then re-thresholds.
6. **Save PNG** alongside the CSV for fast reloading.

---

## Replanning

`navigation_node` replans automatically when the robot deviates more than **10 pixels** from the current planned path. No manual action needed.

---

## Named locations (UI)

The UI supports a named-location dropdown loaded from a JSON file. Format:

```json
{
  "Lab A": { "x": 120.0, "y": 80.0 },
  "Exit":  { "x": 15.0,  "y": 160.0 }
}
```

Coordinates are pixel positions in the occupancy grid.
