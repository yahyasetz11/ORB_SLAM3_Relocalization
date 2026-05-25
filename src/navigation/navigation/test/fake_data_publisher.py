#!/usr/bin/env python3
# TEST ONLY — remove before deployment

# navigation/test/fake_data_publisher.py
#
# Standalone ROS 2 test node that publishes fake data to test the
# navigation UI without needing real hardware or SLAM.
#
# Map layout  (200 x 200 cells, 0.05 m/cell = 10 m x 10 m world)
# ──────────────────────────────────────────────────────────────────
#
#   (0,0)──────────────────────── x →
#     │  [OBS-1]        [OBS-2]
#     │
#     │        (oval robot path)
#     │
#     │  [OBS-5]              [OBS-3]
#     │
#     │        [OBS-4]
#     y ↓
#
# Obstacle pixel coordinates  (x1, y1, x2, y2)
# ──────────────────────────────────────────────
#   OBS-1  (  5,  5,  40,  50)  top-left corner
#   OBS-2  (130,  5, 185,  45)  top-right corner
#   OBS-3  (165, 80, 195, 140)  right edge
#   OBS-4  ( 40,155, 130, 195)  bottom strip
#   OBS-5  (  5, 90,  35, 155)  left edge
#
# Robot oval path  (world coords, metres)
# ───────────────────────────────────────
#   centre  = (5.0, 5.0)
#   radius x = 3.0 m  →  x ∈ [2.0, 8.0]
#   radius y = 2.0 m  →  y ∈ [3.0, 7.0]
#   one full loop ≈ 60 s   (angle_step = 2π / (60 × 2Hz) per tick)

# ─── Standard library ────────────────────────────────────────────────────────
import math

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np

# ─── ROS core ────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node

# ─── ROS messages ────────────────────────────────────────────────────────────
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid


# ─────────────────────────────────────────────────────────────────────────────
# Map parameters  —  edit here to change the map
# ─────────────────────────────────────────────────────────────────────────────

MAP_W      = 200          # cells
MAP_H      = 200          # cells
RESOLUTION = 0.05         # metres per cell

# Rectangular obstacles: (x1, y1, x2, y2) in cell/pixel coordinates
OBSTACLES = [
    (  5,  5,  40,  50),   # OBS-1  top-left corner
    (130,  5, 185,  45),   # OBS-2  top-right corner
    (165, 80, 195, 140),   # OBS-3  right edge
    ( 40,155, 130, 195),   # OBS-4  bottom strip
    (  5, 90,  35, 155),   # OBS-5  left edge
]


# ─────────────────────────────────────────────────────────────────────────────
# Robot oval path parameters  —  edit here to change the robot's route
# ─────────────────────────────────────────────────────────────────────────────

OVAL_CX   = 5.0   # centre x  (metres)
OVAL_CY   = 5.0   # centre y  (metres)
OVAL_RX   = 3.0   # x radius  (metres)
OVAL_RY   = 2.0   # y radius  (metres)

# One full loop takes LOOP_SECONDS seconds.  The position timer fires at ~0.67 Hz,
# so angle advances by  2π / (LOOP_SECONDS × 0.67)  each tick.
LOOP_SECONDS = 120.0


# ─────────────────────────────────────────────────────────────────────────────
# Build occupancy grid
# ─────────────────────────────────────────────────────────────────────────────

def _build_occupancy_data() -> list:
    grid = np.zeros((MAP_H, MAP_W), dtype=np.int8)
    for x1, y1, x2, y2 in OBSTACLES:
        grid[y1:y2, x1:x2] = 100          # 100 = occupied
    return grid.flatten().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Publisher node
# ─────────────────────────────────────────────────────────────────────────────

class FakeDataPublisher(Node):

    def __init__(self):
        super().__init__('fake_data_publisher')

        self._occupancy_data = _build_occupancy_data()
        self._angle          = 0.0
        self._angle_step     = (2.0 * math.pi) / (LOOP_SECONDS * 2.0)  # per 2 Hz tick
        self._map_published  = False

        self._map_pub  = self.create_publisher(OccupancyGrid, 'world_map',         10)
        self._pos_pub  = self.create_publisher(PointStamped,  'current_position',  10)

        # One-shot map publish: fires at 1 Hz until the flag is set
        self.create_timer(1.0, self._publish_map_once)

        # Position at 2 Hz
        self.create_timer(1.5, self._publish_position)

        # Log position at 0.5 Hz
        self.create_timer(2.0, self._log_position)

        self.get_logger().info(
            f'FakeDataPublisher ready  |  map {MAP_W}x{MAP_H} @ {RESOLUTION} m/cell'
        )

    # ── Map ───────────────────────────────────────────────────────────────────

    def _publish_map_once(self):
        if self._map_published:
            return

        msg = OccupancyGrid()
        msg.header.stamp              = self.get_clock().now().to_msg()
        msg.header.frame_id           = 'map'
        msg.info.resolution           = RESOLUTION
        msg.info.width                = MAP_W
        msg.info.height               = MAP_H
        msg.info.origin.position.x    = 0.0
        msg.info.origin.position.y    = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data                      = self._occupancy_data

        self._map_pub.publish(msg)
        self._map_published = True
        self.get_logger().info('world_map published (one-shot)')

    # ── Robot position ────────────────────────────────────────────────────────

    def _current_xy(self) -> tuple[float, float]:
        x = OVAL_CX + OVAL_RX * math.cos(self._angle)
        y = OVAL_CY + OVAL_RY * math.sin(self._angle)
        return x, y

    def _publish_position(self):
        x, y = self._current_xy()

        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x         = x
        msg.point.y         = y
        msg.point.z         = 0.0
        self._pos_pub.publish(msg)

        self._angle += self._angle_step

    # ── Logger ────────────────────────────────────────────────────────────────

    def _log_position(self):
        x, y = self._current_xy()
        self.get_logger().info(f'robot position  x={x:.2f} m  y={y:.2f} m')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = FakeDataPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
