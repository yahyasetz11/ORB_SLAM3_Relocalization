#!/usr/bin/env python3
# navigation/navigation/yolo_seg_publisher.py

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ultralytics import YOLO


CLASSES = {
    0: 'door',
    1: 'fire_hydrant',
    2: 'metal_door',
    3: 'pillar',
    4: 'window',
}


class YoloSegPublisher(Node):

    def __init__(self):
        super().__init__('yolo_seg_publisher')

        self.declare_parameter('model_path', 'segv11_conf40.pt')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('confidence', 0.5)

        model_path   = self.get_parameter('model_path').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self._conf   = self.get_parameter('confidence').get_parameter_value().double_value

        self._model  = YOLO(model_path)
        self._bridge = CvBridge()

        # one binary-mask publisher per class
        self._pubs = {
            cls_id: self.create_publisher(Image, f'/seg/{cls_name}', 10)
            for cls_id, cls_name in CLASSES.items()
        }
        self._vis_pub = self.create_publisher(Image, '/seg/visualization', 10)

        self.create_subscription(Image, camera_topic, self._image_callback, 10)
        self.get_logger().info(
            f'YoloSegPublisher ready — model: {model_path}  camera: {camera_topic}'
        )

    def _image_callback(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w  = frame.shape[:2]

        results = self._model(frame, conf=self._conf, verbose=False)
        result  = results[0]

        # one blank mask per class — all instances of the same class are OR'd together
        masks_by_class = {cls_id: np.zeros((h, w), dtype=np.uint8) for cls_id in CLASSES}

        if result.masks is not None:
            for i, seg_mask in enumerate(result.masks.data):
                cls_id = int(result.boxes.cls[i])
                if cls_id not in CLASSES:
                    continue
                mask_np = seg_mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                masks_by_class[cls_id] = cv2.bitwise_or(
                    masks_by_class[cls_id],
                    (mask_resized > 0.5).astype(np.uint8) * 255,
                )

        stamp    = msg.header.stamp
        frame_id = msg.header.frame_id

        for cls_id, mask in masks_by_class.items():
            out = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
            out.header.stamp    = stamp
            out.header.frame_id = frame_id
            self._pubs[cls_id].publish(out)

        # colour overlay of all detections for debugging
        vis = result.plot() if result.masks is not None else frame.copy()
        vis_msg = self._bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        vis_msg.header.stamp    = stamp
        vis_msg.header.frame_id = frame_id
        self._vis_pub.publish(vis_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
