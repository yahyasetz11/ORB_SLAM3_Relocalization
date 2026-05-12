#!/usr/bin/env python3
# navigation/navigation/main.py

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Image
from collections import deque
from typing import List, Optional

from navigation.planner.a_star import AStarImplementation
from navigation.planner.primitives import PixelCoords, PathNode
from navigation.planner.utils import smooth_path_generation, draw_path, draw_target, pixel_to_world, euclidean_distance


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Subscriptions
        self.map_sub = self.create_subscription(
            Image, '/map_image', self.map_callback, 10)
        self.goal_sub = self.create_subscription(
            PointStamped, 'goal_position', self.goal_callback, 10)
        self.position_sub = self.create_subscription(
            PointStamped, 'current_position', self.position_callback, 10)

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
        
        # Visualization
        self.static_base = None       # grid image, built once
        self.path_base = None       # static_base + current path
        self.trail = deque(maxlen=500) 

    # ── Subscribers ────────────────────────────────────────────────────────────

    def map_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # convert to binary grid for A*
        # white (255) = wall = 1, black (0) = free = 0
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grey, 127, 1, cv2.THRESH_BINARY_INV)
        self.world_map = binary

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

    def goal_callback(self, msg: PointStamped):
        if self.world_map is None:
            self.get_logger().warn('Map not received yet, ignoring goal.')
            return
        if self.current_position is None:
            self.get_logger().warn('No position yet, ignoring goal.')
            return
        coords = PixelCoords(int(msg.point.x), int(msg.point.y))
        snapped = self.snap_to_free(coords)
        if snapped is None:
            self.get_logger().warn('No free cell near goal — ignoring.')
            return
        self.goal_coords = snapped
        self.start_coords = self.current_position
        self.try_plan()

    def position_callback(self, msg: PointStamped):
        if self.world_map is None:
            return
        self.current_position = PixelCoords(int(msg.point.x), int(msg.point.y))

        # update trail and publish new frame
        self.publish_trails(self.current_position)

        # check if robot has deviated from planned path
        if self.check_deviation(self.current_position):
            self.get_logger().info('Deviation detected — replanning from current position...')
            self.start_coords = self.current_position
            self.try_plan()

    # ── Planning ───────────────────────────────────────────────────────────────

    def try_plan(self):
        if self.world_map is None or self.start_coords is None or self.goal_coords is None:
            return

        planner = AStarImplementation(
            world_map=self.world_map,
            start_coords=self.start_coords,
            goal_coords=self.goal_coords,
            iter_limit=10000,
        )

        try:
            path, _ = planner.plan()
        except ValueError as e:
            self.get_logger().warn(f'Planning failed: {e}')
            return

        self.smooth_path = smooth_path_generation(path)
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
            angle = 0.0
            if len(self.trail) >= 2:
                px, py = self.trail[-2]
                dx, dy = cx - px, cy - py
                if dx != 0 or dy != 0:
                    angle = np.arctan2(dy, dx)
            elif self.smooth_path and len(self.smooth_path) > 1:
                nx = self.smooth_path[1].coords.x_coords
                ny = self.smooth_path[1].coords.y_coords
                dx, dy = nx - cx, ny - cy
                if dx != 0 or dy != 0:
                    angle = np.arctan2(dy, dx)
            # pointy triangle: tip forward, wide base behind
            tip_len, half_base = 9, 4
            local = np.array([[tip_len, 0], [-tip_len // 2, -half_base], [-tip_len // 2, half_base]], dtype=np.float32)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            pts = (local @ rot.T + np.array([cx, cy])).astype(np.int32)
            cv2.polylines(path_base_img, [pts], isClosed=True, color=(30, 30, 30), thickness=2, lineType=cv2.LINE_AA)
            cv2.fillPoly(path_base_img, [pts], color=(0, 255, 255), lineType=cv2.LINE_AA)
        
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
