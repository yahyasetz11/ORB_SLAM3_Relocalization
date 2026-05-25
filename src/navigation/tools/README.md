# tools/osa_to_csv

Offline utility that reads an ORB-SLAM3 binary atlas (`.osa`) and writes every
active map point to a CSV file.  Run this **once before launching the
navigation stack** — not while it is running.

---

## What it does

`osa_to_csv` opens an `.osa` file produced by ORB-SLAM3's atlas serialisation,
extracts all non-bad 3-D map points from the active map, and writes them as
rows in a plain-text CSV.  The resulting file can be imported into any map
construction or visualisation tool.

---

## Running it

```bash
# Build first (see section below)
ros2 run navigation osa_to_csv --osa /path/to/map1.osa --output /path/to/map.csv
```

Both flags are required.  The tool will print progress at each step:

```
[osa_to_csv] Step 1/2 — loading map...
[load_map]   Loading atlas from: map1.osa
[load_map]   Loaded 12453 map points.
[osa_to_csv] Step 2/2 — writing CSV...
[write_csv]  Writing 12453 points to: map.csv
[write_csv]  Done.
[osa_to_csv] Conversion complete.
```

---

## Output CSV columns

| Column         | Type    | Description                                     |
|----------------|---------|-------------------------------------------------|
| `id`           | integer | Zero-based row index (not the ORB-SLAM3 MP id)  |
| `x`            | float   | World-frame X position (metres)                 |
| `y`            | float   | World-frame Y position (metres)                 |
| `z`            | float   | World-frame Z position (metres)                 |
| `observations` | integer | Number of keyframes in which this point was seen |

Bad map points (flagged by the ORB-SLAM3 tracker) are excluded.

---

## Building

This tool must be built before running the navigation stack.

```bash
# From the workspace root
colcon build --packages-select navigation
source install/setup.bash
```

> **Note:** The `navigation` package was converted from `ament_python` to
> `ament_cmake` + `ament_cmake_python` to support this C++ tool.  The Python
> nodes (`navigation`, `navigation_ui`, `fake_data_publisher`) continue to
> work as before.

---

## Troubleshooting: ORB-SLAM3 headers not found

The `#include <ORB_SLAM3/System.h>` lines in `osa_to_csv.cpp` are commented
out until the ORB-SLAM3 headers are available at build time.

**Option 1 — system install:**
```bash
sudo make install   # inside your ORB-SLAM3 build directory
```
Then uncomment `find_package(ORB_SLAM3 REQUIRED)` in `CMakeLists.txt`.

**Option 2 — manual path:**
Pass the include and library paths at configure time:
```bash
colcon build --packages-select navigation \
  --cmake-args \
    -DORB_SLAM3_INCLUDE_DIRS=/path/to/ORB-SLAM3/include \
    -DORB_SLAM3_LIBS=/path/to/ORB-SLAM3/lib/libORB_SLAM3.so
```
Then update `target_include_directories` and `target_link_libraries` in
`CMakeLists.txt` to use those variables.

**Verify Eigen3 is installed:**
```bash
dpkg -l libeigen3-dev   # Ubuntu/Debian
```
