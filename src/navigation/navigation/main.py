# navigation/navigation/main.py

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Image
from collections import deque
from typing import List

from navigation.planner.a_star import AStarImplementation
from navigation.planner.primitives import PixelCoords, PathNode
from navigation.planner.utils import smooth_path_generation, draw_path, draw_target, world_to_pixel, pixel_to_world, euclidean_distance


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Subscriptions
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'world_map', self.map_callback, 10)
        self.start_sub = self.create_subscription(
            PointStamped, 'start_position', self.start_callback, 10)
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
        self.map_info = None
        self.start_coords = None
        self.goal_coords = None
        self.smooth_path = None
        self.bridge = CvBridge()
        
        # Visualization
        self.static_base = None       # grid image, built once
        self.path_base = None       # static_base + current path
        self.trail = deque(maxlen=500) 

    # ── Subscribers ────────────────────────────────────────────────────────────

    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info
        self.raw_map_data = msg   # store the whole message
        self.build_base()      # let build_base do the processing
        self.try_plan()

    def start_callback(self, msg: PointStamped):
        if self.map_info is None:
            self.get_logger().warn('Map not received yet, ignoring start position.')
            return
        self.start_coords = world_to_pixel(msg.point.x, msg.point.y, self.map_info)
        self.try_plan()

    def goal_callback(self, msg: PointStamped):
        if self.map_info is None:
            self.get_logger().warn('Map not received yet, ignoring goal position.')
            return
        self.goal_coords = world_to_pixel(msg.point.x, msg.point.y, self.map_info)
        self.try_plan()
    
    def position_callback(self, msg: PointStamped):
        if self.map_info is None:
            return

        # convert world coordinates to pixel coordinates
        self.current_position = world_to_pixel(
            msg.point.x,
            msg.point.y,
            self.map_info
        )

        # update trail and publish new frame
        self.publish_trails(self.current_position)

        # check if robot has deviated from planned path
        if self.check_deviation(self.current_position):
            self.get_logger().info('Deviation detected — replanning...')
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

        self.publish_path()
        self.draw_planned_path()

    # ── visualization (publisher) ─────────────────────────────────────────────────────────────

    def build_base(self):
        # 1. process raw occupancy grid
        w, h = self.map_info.width, self.map_info.height
        data = np.array(self.raw_map_data.data, dtype=np.int8).reshape((h, w))
        self.world_map = np.where(data > 50, 1, 0).astype(np.uint8)

        # 2. render grey BGR image
        grey = np.where(self.world_map == 0, 220, 40).astype(np.uint8)
        self.static_base = np.stack([grey] * 3, axis=-1)

        # 3. initialize path_base so it's never None
        self.path_base = self.static_base.copy()

        # 4. publish
        ros_base_img = self.bridge.cv2_to_imgmsg(self.static_base, encoding='bgr8')
        ros_base_img.header.stamp = self.get_clock().now().to_msg()
        ros_base_img.header.frame_id = 'map'
        self.static_image_pub.publish(ros_base_img)
        
    
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
            self.path_base = draw_target(self.path_base, self.start_coords, color=(0, 255, 0))
        if self.goal_coords is not None:
            self.path_base = draw_target(self.path_base, self.goal_coords, color=(0, 0, 255))
            
        ros_path_img = self.bridge.cv2_to_imgmsg(self.path_base, encoding='bgr8')
        ros_path_img.header.stamp = self.get_clock().now().to_msg()
        ros_path_img.header.frame_id = 'map'
        self.path_image_pub.publish(ros_path_img)
    
    def publish_trails(self, current_position=None):
        if self.path_base is None:
            return
        
        if current_position is not None:
            self.trail.append((current_position.x_coords, current_position.y_coords))
            
        path_base_img = self.path_base.copy()
        
        if len(self.trail) >= 2:
            points = np.array(list(self.trail), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(path_base_img, [points], isClosed=False, color=(200, 200, 200), thickness=2)
        
        if current_position is not None:
            cv2.circle(path_base_img, (current_position.x_coords, current_position.y_coords), radius=6, color=(0, 255, 255), thickness=-1)
        
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
        
    def check_deviation(self, current_position, threshold=25) -> bool:
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
