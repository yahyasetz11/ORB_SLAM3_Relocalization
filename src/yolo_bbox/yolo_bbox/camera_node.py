import cv2
import os
import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('video_path', '')        # empty = webcam
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('timestamp_start', 0.0)  # 0.0 = use ROS wall clock

        video_path = os.path.expanduser(
            self.get_parameter('video_path').get_parameter_value().string_value)
        camera_id        = self.get_parameter('camera_id').get_parameter_value().integer_value
        timestamp_start  = self.get_parameter('timestamp_start').get_parameter_value().double_value

        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.get_logger().info(f'Camera source: video  → {video_path}')
        else:
            self.cap = cv2.VideoCapture(camera_id)
            self.get_logger().info(f'Camera source: webcam → device {camera_id}')

        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera source')
            raise RuntimeError('Unable to open camera source')

        # Publish at source FPS; fall back to 30 if unavailable
        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = src_fps if src_fps > 0 else 30.0
        self.get_logger().info(f'Publishing at {self._fps:.1f} fps on /camera/image_raw')

        self._timestamp_start = timestamp_start
        self._frame_index = 0
        if timestamp_start > 0.0:
            self.get_logger().info(
                f'Timestamp mode: TUM-aligned  start={timestamp_start:.6f}')
        else:
            self.get_logger().info('Timestamp mode: ROS wall clock')

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)

        # Wait until at least one subscriber is ready before publishing.
        # The relocalization node takes several seconds to load the vocab and
        # map, so frames published before it subscribes would be silently lost.
        self.get_logger().info(
            'Waiting for subscribers on /camera/image_raw ...')
        self._pub_timer = None
        self._wait_timer = self.create_timer(0.5, self._wait_for_subscribers)

    def _wait_for_subscribers(self):
        if self.pub.get_subscription_count() >= 2:
            self._wait_timer.cancel()
            self.get_logger().info(
                f'{self.pub.get_subscription_count()} subscriber(s) connected'
                ' — starting frame publishing')
            self._pub_timer = self.create_timer(1.0 / self._fps, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().info('End of video or failed to read frame — shutting down.')
            rclpy.shutdown()
            return

        frame = cv2.resize(frame, (640, 480))
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')

        if self._timestamp_start > 0.0:
            t = self._timestamp_start + self._frame_index / self._fps
            msg.header.stamp.sec     = int(t)
            msg.header.stamp.nanosec = int((t % 1.0) * 1e9)
        else:
            msg.header.stamp = self.get_clock().now().to_msg()

        self._frame_index += 1
        msg.header.frame_id = 'camera'
        self.pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
