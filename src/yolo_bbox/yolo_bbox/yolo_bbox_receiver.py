import rclpy
import json

from rclpy.node import Node
from std_msgs.msg import String


class BBOX_Coords_Receiver(Node):
    def __init__(self):
        super().__init__('bbox_coords_receiver')

        self.subscription = self.create_subscription(
            String,
            'yolo/results',
            self.results_callback,
            10
        )

        self.get_logger().info("YOLO receiver started")

    def results_callback(self, msg):
        try:
            data = json.loads(msg.data)

            frame_id = data["frame_id"]
            detections = data["detections"]

            self.get_logger().info(
                f"Frame {frame_id} | Number of door detections: {len(detections)}"
            )

            for i, det in enumerate(detections):
                cls_id = det["cls_id"]
                conf = det["conf_score"]

                bbox = det["bbox_coords"]
                x1 = bbox["x1"]
                y1 = bbox["y1"]
                x2 = bbox["x2"]
                y2 = bbox["y2"]

                corners = det["corner_coords"]
                top_left = corners["top_left"]
                top_right = corners["top_right"]
                bot_left = corners["bot_left"]
                bot_right = corners["bot_right"]

                self.get_logger().info(f"[{i}] cls_id={cls_id}, conf={conf:.2f}, "f"bbox=({x1}, {y1}, {x2}, {y2})")

                print("Top Left:   ", top_left)
                print("Top Right:  ", top_right)
                print("Bottom Left:", bot_left)
                print("Bottom Right:", bot_right)

        except Exception as e:
            self.get_logger().error(f"Failed to parse message: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = BBOX_Coords_Receiver()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()