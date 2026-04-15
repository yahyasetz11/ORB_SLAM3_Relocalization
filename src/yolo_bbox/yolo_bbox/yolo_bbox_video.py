import cv2
import os
import signal
import rclpy
import time
import json
import queue
import threading

import torch
from ultralytics import YOLO
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image


def _enqueue_latest(frame_queue: queue.Queue, frame) -> None:
    """Drop stale frame and put the newest one. Called from image_callback."""
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put_nowait(frame)


def _select_device(requested: str) -> str:
    """Return 'cpu' if CUDA requested but unavailable, else return requested."""
    if requested == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return requested


class BBOX_Coords(Node):
    def __init__(self):
        super().__init__('yolo_bbox_node')

        # Parameters (loaded from yolo_bbox_params.yaml via --params-file)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('display', True)
        self.declare_parameter('conf', 0.2)
        self.declare_parameter('save_json', False)
        self.declare_parameter('json_path', 'validation_back.jsonl')

        device_param = self.get_parameter('device').get_parameter_value().string_value
        self.display = self.get_parameter('display').get_parameter_value().bool_value
        self.conf_value = self.get_parameter('conf').get_parameter_value().double_value
        self.save_json = self.get_parameter('save_json').get_parameter_value().bool_value
        self.json_path = self.get_parameter('json_path').get_parameter_value().string_value

        # GPU fallback
        self.device = _select_device(device_param)
        if self.device != device_param:
            self.get_logger().warn(
                'CUDA requested but not available — falling back to CPU')
        self.get_logger().info(f'Running YOLO on device: {self.device}')

        # Model
        pkg_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(pkg_dir, 'model', 'model1.pt')
        self.model = YOLO(self.model_path)

        # Publishers
        self.results_pub = self.create_publisher(String, 'yolo/results', 10)

        # Camera subscription — pushes frames into a single-slot queue
        self.bridge = CvBridge()
        self.frame_queue = queue.Queue(maxsize=1)
        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Start background inference thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            'YOLO bbox node started — subscribed to /camera/image_raw')

    # ------------------------------------------------------------------
    # ROS2 callback — runs on the spin thread, must never block
    # ------------------------------------------------------------------

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        _enqueue_latest(self.frame_queue, frame)

    # ------------------------------------------------------------------
    # Inference thread — runs independently of the spin thread
    # ------------------------------------------------------------------

    def _inference_loop(self):
        frame_id = 0
        prev_time = time.time()

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue  # check stop_event and retry

            detected = self.model.predict(
                frame, conf=self.conf_value, device=self.device, verbose=False)
            detections = []

            for obj in detected:
                for box in obj.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    detections.append({
                        'cls_id': cls_id,
                        'conf_score': conf,
                        'bbox_coords': {
                            'x1': int(x1), 'y1': int(y1),
                            'x2': int(x2), 'y2': int(y2),
                        },
                        'corner_coords': {
                            'top_left':  [int(x1), int(y1)],
                            'top_right': [int(x2), int(y1)],
                            'bot_left':  [int(x1), int(y2)],
                            'bot_right': [int(x2), int(y2)],
                        },
                    })

            # Publish results
            msg_out = String()
            msg_out.data = json.dumps({
                'frame_id': frame_id,
                'detections': detections,
            })
            self.results_pub.publish(msg_out)

            if self.save_json:
                with open(self.json_path, 'a') as f:
                    f.write(msg_out.data + '\n')

            # Optional display window
            if self.display:
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                display_frame = frame.copy()
                for obj in detected:
                    for box in obj.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = f'id:{cls_id} conf:{conf:.2f}'
                        cv2.rectangle(
                            display_frame,
                            (int(x1), int(y1)), (int(x2), int(y2)),
                            (0, 255, 0), 2)
                        cv2.putText(
                            display_frame, label,
                            (int(x1), max(20, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(display_frame, f'Frame: {frame_id}',
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(display_frame, f'Doors: {len(detections)}',
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(display_frame, f'FPS: {fps:.1f}',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, f'Device: {self.device}',
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow('YOLO Video', display_frame)
                if cv2.waitKey(1) == 27:
                    self.get_logger().info('ESC pressed — shutting down')
                    self._stop_event.set()
                    os.kill(os.getpid(), signal.SIGINT)
                    break

            frame_id += 1

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def destroy_node(self):
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BBOX_Coords()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
