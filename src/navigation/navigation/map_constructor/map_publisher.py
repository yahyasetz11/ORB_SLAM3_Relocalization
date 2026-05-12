#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import cv2
import os

class MapPublisherNode(Node):
    def __init__(self):
        super().__init__("map_publisher")
        self.declare_parameter("map_dir", "")
        map_dir_param = self.get_parameter("map_dir").get_parameter_value().string_value
        if map_dir_param:
            self.map_dir = map_dir_param
        else:
            share = get_package_share_directory("navigation")
            self.map_dir = os.path.join(share, "maps")
        os.makedirs(self.map_dir, exist_ok=True)
        self.get_logger().info(f"Map directory: {self.map_dir}")

        self.declare_parameter("resolution", 0.02)

        self.bridge = CvBridge()
        self.image_msg = None

        self.publisher = self.create_publisher(Image, "/map_image", 10)
        self.create_subscription(String, "/map_name", self.map_name_callback, 10)
        self.create_timer(1.0, self.publish_map)

    def map_name_callback(self, msg):
        map_name = msg.data
        map_path = os.path.join(self.map_dir, f"{map_name}.png")

        if os.path.exists(map_path):
            self.image_msg = self.load(map_name)
        else:
            csv_path = os.path.join(self.map_dir, f"{map_name}.csv")
            resolution = self.get_parameter("resolution").get_parameter_value().double_value

            points = self.load_points(csv_path)
            points = self.filter_by_height(points)
            points = self.filter_outliers(points)
            grid = self.points_to_grid(points, resolution)
            smooth_grid = self.smooth_map(grid)
            self.image_msg = self.grid_to_image_msg(smooth_grid)
            self.save(map_name)

    def load(self, map_name):
        map_path = os.path.join(self.map_dir, f"{map_name}.png")
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        return self.bridge.cv2_to_imgmsg(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), encoding="rgb8")

    def save(self, map_name):
        map_path = os.path.join(self.map_dir, f"{map_name}.png")
        img = self.bridge.imgmsg_to_cv2(self.image_msg, desired_encoding="rgb8")
        cv2.imwrite(map_path, img)

    def load_points(self, csv_path):
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("CSV has no content.")
        return df[["x", "y", "z"]].values

    def filter_by_height(self, points):
        y = points[:, 1]
        y_mid = np.mean(y)
        y_std = np.std(y)
        mask = (y > y_mid - y_std) & (y < y_mid + y_std)
        return points[mask]

    def filter_outliers(self, points):
        xz = points[:, [0, 2]]
        tree = BallTree(xz)
        counts = tree.query_radius(xz, r=0.3, count_only=True)
        mask = counts >= 3
        return points[mask]

    def points_to_grid(self, points, resolution):
        x_points = points[:, 0]
        z_points = points[:, 2]

        x_points = x_points - np.min(x_points)
        z_points = z_points - np.min(z_points)

        xi = (x_points / resolution).astype(int)
        zi = (z_points / resolution).astype(int)

        grid = np.zeros((np.max(zi) + 1, np.max(xi) + 1), dtype=np.uint8)
        grid[zi, xi] = 255
        return grid

    def smooth_map(self, grid):
        kernel = np.ones((7, 7), np.uint8)
        smooth_grid = cv2.dilate(grid, kernel, iterations=2)
        smooth_grid = cv2.GaussianBlur(smooth_grid, (9, 9), sigmaX=2)
        _, smooth_grid = cv2.threshold(smooth_grid, 127, 255, cv2.THRESH_BINARY)
        return smooth_grid

    def grid_to_image_msg(self, grid):
        # path (255, point cloud hits) → white 255, walls/empty (0) → grey 120
        display = np.where(grid == 255, 255, 120).astype(np.uint8)
        img = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
        return self.bridge.cv2_to_imgmsg(img, encoding="rgb8")

    def publish_map(self):
        if self.image_msg is None:
            return
        self.publisher.publish(self.image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()