#!/usr/bin/env python3
# navigation/navigation/main.py

import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
from sensor_msgs.msg import Image
from std_msgs.msg import String
from collections import deque
from typing import List, Optional

from navigation.planner.a_star import AStarImplementation
from navigation.planner.primitives import PixelCoords, PathNode
from navigation.planner.utils import smooth_path_generation, draw_path, draw_target, pixel_to_world, euclidean_distance, inflated_obstacles


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Subscriptions
        self.map_sub = self.create_subscription(
            Image, '/map_image', self.map_callback, 10)
        self.meta_sub = self.create_subscription(
            String, '/map_meta', self.meta_callback, 10)
        self.goal_sub = self.create_subscription(
            PointStamped, 'goal_position', self.goal_callback, 10)
        self.declare_parameter('pose_source', 'dummy')
        self.declare_parameter('pose_scale', 1.0)
        self._pose_source = self.get_parameter('pose_source').get_parameter_value().string_value
        self._pose_scale  = self.get_parameter('pose_scale').get_parameter_value().double_value

        if self._pose_source == 'relocalization':
            self.position_sub = self.create_subscription(
                Pose, '/relocalization/pose', self.relocalization_pose_callback, 10)
            self.get_logger().info('Pose source: /relocalization/pose')
        else:
            self.position_sub = self.create_subscription(
                PointStamped, 'current_position', self.position_callback, 10)
            self.get_logger().info('Pose source: current_position (dummy)')

        # Publishers
        self.static_image_pub = self.create_publisher(Image, 'static_image', 10)
        self.path_image_pub = self.create_publisher(Image, 'path_image', 10)
        self.trails_image_pub = self.create_publisher(Image, 'trails_image', 10)
        
        self.path_pub = self.create_publisher(Path, 'smooth_path', 10)

        # State
        self.current_position = None
        self.world_map = None
        self.start_coords = None
        self.goal_coords = None
        self.smooth_path = None
        self.bridge = CvBridge()

        # Map metadata (from /map_meta) — used by relocalization pose conversion
        self._map_x_min = 0.0
        self._map_z_min = 0.0
        self._map_resolution = 0.02   # metres per pixel (map_publisher default)
        
        # Visualization
        self.static_base = None       # grid image, built once
        self.path_base = None       # static_base + current path
        self.trail = deque(maxlen=500)
        self.inflated_map_raw: Optional[np.ndarray] = None
        self._last_heading = -np.pi / 2  # default: facing up on screen

        # Goal tracking
        self.has_active_goal = False
        self.show_goal_reached = False
        self._goal_reached_frames = 0

    # ── Subscribers ────────────────────────────────────────────────────────────

    def map_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.flip(img, 0)   # flip vertically so forward (Z↑) = up on screen

        # convert to binary grid for A*
        # white (255) = wall = 1, black (0) = free = 0
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grey, 127, 1, cv2.THRESH_BINARY_INV)
        self.world_map = binary
        self.inflated_map_raw = inflated_obstacles(self.world_map, 8)

        # store visual base layers
        self.static_base = img.copy()
        self.path_base = self.static_base.copy()

        # publish static base
        ros_base_img = self.bridge.cv2_to_imgmsg(self.static_base, encoding='bgr8')
        ros_base_img.header.stamp = self.get_clock().now().to_msg()
        ros_base_img.header.frame_id = 'map'
        self.static_image_pub.publish(ros_base_img)

        # replan first so path_base is redrawn before pushing to UI,
        # otherwise the UI flashes a blank map on every 1Hz map update
        self.try_plan()

        self.publish_trails()

    def meta_callback(self, msg: String):
        try:
            meta = json.loads(msg.data)
            self._map_x_min     = meta['x_min']
            self._map_z_min     = meta['z_min']
            self._map_resolution = meta['resolution']
        except Exception as e:
            self.get_logger().warn(f'Bad /map_meta message: {e}')

    def goal_callback(self, msg: PointStamped):
        if self.world_map is None:
            self.get_logger().warn('Map not received yet, ignoring goal.')
            return
        if self.current_position is None:
            self.get_logger().warn('No position yet, ignoring goal.')
            return
        # msg.point.x = user X (upward), msg.point.y = user Y (rightward), origin bottom-left
        map_h = self.world_map.shape[0]
        coords = PixelCoords(int(msg.point.y), map_h - 1 - int(msg.point.x))
        snapped = self.snap_to_free(coords)
        if snapped is None:
            self.get_logger().warn('No free cell near goal — ignoring.')
            return
        self.goal_coords = snapped
        self.start_coords = self.current_position
        self.has_active_goal = True
        self.show_goal_reached = False
        self.try_plan()

    def position_callback(self, msg: PointStamped):
        if self.world_map is None:
            return
        # msg.point.x = user X (upward), msg.point.y = user Y (rightward), origin bottom-left
        map_h = self.world_map.shape[0]
        self.current_position = PixelCoords(int(msg.point.y), map_h - 1 - int(msg.point.x))

        # update trail and publish new frame
        self.publish_trails(self.current_position)

        if self.has_active_goal:
            if self.check_goal_reached(self.current_position):
                self.get_logger().info('Goal reached!')
                self.has_active_goal = False
                self.show_goal_reached = True
                self._goal_reached_frames = 20
                self.smooth_path = None
                self.trail = deque(maxlen=500)
                if self.static_base is not None:
                    self.path_base = self.static_base.copy()
                    ros_path_img = self.bridge.cv2_to_imgmsg(self.path_base, encoding='bgr8')
                    ros_path_img.header.stamp = self.get_clock().now().to_msg()
                    ros_path_img.header.frame_id = 'map'
                    self.path_image_pub.publish(ros_path_img)
                return

            # check if robot has deviated from planned path
            if self.check_deviation(self.current_position):
                self.get_logger().info('Deviation detected — replanning from current position...')
                self.start_coords = self.current_position
                self.try_plan()

    def relocalization_pose_callback(self, msg: Pose):
        if self.world_map is None:
            return
        # Convert ORB-SLAM3 world coords to map pixel coords using map metadata.
        # map_publisher builds the grid as: col=(world_x - x_min)/res, row=(world_z - z_min)/res
        map_h, map_w = self.world_map.shape
        col = int((msg.position.x - self._map_x_min) / self._map_resolution)
        # map is flipped vertically: large Z (forward) = small row = top of screen
        row = map_h - 1 - int((msg.position.z - self._map_z_min) / self._map_resolution)
        col = max(0, min(map_w - 1, col))
        row = max(0, min(map_h - 1, row))
        current = PixelCoords(col, row)
        self.current_position = current

        self.publish_trails(current)

        if self.has_active_goal:
            if self.check_goal_reached(current):
                self.get_logger().info('Goal reached!')
                self.has_active_goal = False
                self.show_goal_reached = True
                self._goal_reached_frames = 20
                self.smooth_path = None
                self.trail = deque(maxlen=500)
                if self.static_base is not None:
                    self.path_base = self.static_base.copy()
                    ros_path_img = self.bridge.cv2_to_imgmsg(self.path_base, encoding='bgr8')
                    ros_path_img.header.stamp = self.get_clock().now().to_msg()
                    ros_path_img.header.frame_id = 'map'
                    self.path_image_pub.publish(ros_path_img)
                return

            if self.check_deviation(current):
                self.get_logger().info('Deviation detected — replanning from current position...')
                self.start_coords = current
                self.try_plan()

    # ── Planning ───────────────────────────────────────────────────────────────

    def try_plan(self):
        if self.world_map is None or self.start_coords is None or self.goal_coords is None:
            return
        if self.inflated_map_raw is None:
            return

        # copy and clear start/goal cells so they're never treated as obstacles
        inflated = self.inflated_map_raw.copy()
        inflated[self.start_coords.y_coords, self.start_coords.x_coords] = 0
        inflated[self.goal_coords.y_coords, self.goal_coords.x_coords] = 0

        planner = AStarImplementation(
            world_map=self.world_map,
            start_coords=self.start_coords,
            goal_coords=self.goal_coords,
            iter_limit=10000,
            inflated_map=inflated,
        )

        try:
            path, _ = planner.plan()
        except ValueError as e:
            self.get_logger().warn(f'Planning failed: {e}')
            return

        self.smooth_path = smooth_path_generation(path, occupancy_map=planner.inflated_obstacle_map)
        if not self.smooth_path:
            self.get_logger().warn('Path smoothing returned empty path.')
            return

        self.draw_planned_path()

    # ── visualization (publisher) ─────────────────────────────────────────────────────────────

    def draw_planned_path(self):
        # called from try_plan() when A* produces a path
        # clones self.static_base (start fresh, no old path)
        # draws smooth_path on the clone
        # saves result to self.path_base
        if self.static_base is None:
            return
        
        base_img = self.static_base.copy()
        self.path_base = draw_path(base_img, self.smooth_path, arrow=False)
        
        if self.start_coords is not None:
            self.path_base = draw_target(self.path_base, self.start_coords, color=(0, 255, 0), radius=4)
        if self.goal_coords is not None:
            self.path_base = draw_target(self.path_base, self.goal_coords, color=(0, 0, 255), radius=4)
            
        ros_path_img = self.bridge.cv2_to_imgmsg(self.path_base, encoding='bgr8')
        ros_path_img.header.stamp = self.get_clock().now().to_msg()
        ros_path_img.header.frame_id = 'map'
        self.path_image_pub.publish(ros_path_img)
    
    def publish_trails(self, current_position=None):
        base = self.path_base if self.path_base is not None else self.static_base
        if base is None:
            return

        if current_position is not None:
            self.trail.append((current_position.x_coords, current_position.y_coords))

        path_base_img = base.copy()
        
        if len(self.trail) >= 2:
            points = np.array(list(self.trail), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(path_base_img, [points], isClosed=False, color=(200, 200, 200), thickness=2)

        if current_position is not None:
            cx, cy = current_position.x_coords, current_position.y_coords
            # determine heading: prefer trail direction, fall back to next path node
            angle = self._last_heading
            if len(self.trail) >= 2:
                # look back up to 10 steps for a stable direction baseline
                lookback = min(10, len(self.trail) - 1)
                px, py = self.trail[-(lookback + 1)]
                dx, dy = cx - px, cy - py
                if dx * dx + dy * dy >= 4:  # require at least 2 px net movement
                    angle = np.arctan2(dy, dx)
                    self._last_heading = angle
            elif self.smooth_path and len(self.smooth_path) > 1:
                nx = self.smooth_path[1].coords.x_coords
                ny = self.smooth_path[1].coords.y_coords
                dx, dy = nx - cx, ny - cy
                if dx != 0 or dy != 0:
                    angle = np.arctan2(dy, dx)
                    self._last_heading = angle
            # pointy triangle: tip forward, wide base behind
            tip_len, half_base = 9, 4
            local = np.array([[tip_len, 0], [-tip_len // 2, -half_base], [-tip_len // 2, half_base]], dtype=np.float32)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            pts = (local @ rot.T + np.array([cx, cy])).astype(np.int32)
            cv2.polylines(path_base_img, [pts], isClosed=True, color=(30, 30, 30), thickness=2, lineType=cv2.LINE_AA)
            cv2.fillPoly(path_base_img, [pts], color=(0, 255, 255), lineType=cv2.LINE_AA)

        if self.show_goal_reached:
            label = 'Goal Reached'
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            tx, ty = 6, 6 + th
            cv2.rectangle(path_base_img, (tx - 3, ty - th - 3), (tx + tw + 3, ty + baseline + 3), (0, 0, 0), -1)
            cv2.putText(path_base_img, label, (tx, ty), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
            self._goal_reached_frames -= 1
            if self._goal_reached_frames <= 0:
                self.show_goal_reached = False

        ros_trail_img = self.bridge.cv2_to_imgmsg(path_base_img, encoding='bgr8')
        ros_trail_img.header.stamp = self.get_clock().now().to_msg()
        ros_trail_img.header.frame_id = 'map'
        self.trails_image_pub.publish(ros_trail_img)

    def publish_path(self):
        now = self.get_clock().now().to_msg()

        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = 'map'

        for node in self.smooth_path:
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = 'map'
            wx, wy = pixel_to_world(node.coords.x_coords, node.coords.y_coords, self.map_info)
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        
    def snap_to_free(self, coords: PixelCoords, search_radius: int = 20) -> Optional[PixelCoords]:
        for r in range(1, search_radius):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx = coords.x_coords + dx
                    ny = coords.y_coords + dy
                    if 0 <= nx < self.world_map.shape[1] and \
                       0 <= ny < self.world_map.shape[0] and \
                       self.world_map[ny, nx] == 0:
                        return PixelCoords(nx, ny)
        return None

    def check_goal_reached(self, current_position, threshold=15) -> bool:
        if self.goal_coords is None:
            return False
        dist = np.sqrt(
            (self.goal_coords.x_coords - current_position.x_coords) ** 2 +
            (self.goal_coords.y_coords - current_position.y_coords) ** 2
        )
        return dist < threshold

    def check_deviation(self, current_position, threshold=5) -> bool:
        if self.smooth_path is None:
            return False
        min_distance = float('inf')
        for node in self.smooth_path:
            dist = np.sqrt(
                (node.coords.x_coords - current_position.x_coords) ** 2 +
                (node.coords.y_coords - current_position.y_coords) ** 2
            )
            if dist < min_distance:
                min_distance = dist
        
        return min_distance > threshold
    

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
