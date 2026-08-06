import cv2
import os
import signal
import rclpy
import time
import json
import queue
import threading

import torch
import numpy as np
from ultralytics import YOLO
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image


CLASSES = {
    0: 'door',
    1: 'fire_hydrant',
    2: 'metal_door',
    3: 'pillar',
    4: 'window',
    5: 'keyboard',
    6: 'monitor',
    7: 'teddy_bear',
    8: 'globe',
}


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


class SegCoords(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')

        # Parameters (loaded from yolo_bbox_params.yaml via --params-file)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('display', True)
        self.declare_parameter('conf', 0.2)
        self.declare_parameter('save_json', False)
        self.declare_parameter('json_path', 'seg_results.jsonl')

        device_param   = self.get_parameter('device').get_parameter_value().string_value
        self.display   = self.get_parameter('display').get_parameter_value().bool_value
        self.conf_value = self.get_parameter('conf').get_parameter_value().double_value
        self.save_json  = self.get_parameter('save_json').get_parameter_value().bool_value
        self.json_path  = self.get_parameter('json_path').get_parameter_value().string_value

        # GPU fallback
        self.device = _select_device(device_param)
        if self.device != device_param:
            self.get_logger().warn(
                'CUDA requested but not available — falling back to CPU')
        self.get_logger().info(f'Running YOLO seg on device: {self.device}')

        # Model
        pkg_dir = os.path.dirname(os.path.realpath(__file__))
        # self.model_path = os.path.join(pkg_dir, 'model', 'segv11_conf40.pt')
        self.model_path = os.path.join(pkg_dir, 'model', 'best.pt')
        self.model = YOLO(self.model_path)

        self.bridge = CvBridge()

        # One binary-mask publisher per class + JSON metadata + visualization
        self._mask_pubs = {
            cls_id: self.create_publisher(Image, f'/seg/{cls_name}', 10)
            for cls_id, cls_name in CLASSES.items()
        }
        self._vis_pub     = self.create_publisher(Image,  '/seg/visualization', 10)
        self._results_pub = self.create_publisher(String, 'yolo/seg_results',   10)

        # Camera subscription — pushes frames into a single-slot queue
        self.frame_queue = queue.Queue(maxsize=1)
        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Background inference thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            'YOLO seg node started — subscribed to /camera/image_raw')

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
        frame_id  = 0
        prev_time = time.time()

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]

            results = self.model.predict(
                frame, conf=self.conf_value, device=self.device, verbose=False)
            result = results[0]

            # Build one binary mask per class — OR all instances of the same class
            masks_by_class = {cls_id: np.zeros((h, w), dtype=np.uint8)
                              for cls_id in CLASSES}

            detections = []

            if result.masks is not None:
                for i, seg_mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    if cls_id not in CLASSES:
                        continue
                    conf = float(result.boxes.conf[i])
                    x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()

                    mask_np = seg_mask.cpu().numpy()
                    mask_resized = cv2.resize(
                        mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks_by_class[cls_id] = cv2.bitwise_or(
                        masks_by_class[cls_id],
                        (mask_resized > 0.5).astype(np.uint8) * 255,
                    )

                    detections.append({
                        'cls_id':   cls_id,
                        'cls_name': CLASSES[cls_id],
                        'conf':     round(conf, 3),
                        'bbox':     {'x1': int(x1), 'y1': int(y1),
                                     'x2': int(x2), 'y2': int(y2)},
                    })

            # Publish per-class masks
            for cls_id, mask in masks_by_class.items():
                mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                self._mask_pubs[cls_id].publish(mask_msg)

            # Publish JSON metadata
            msg_out = String()
            msg_out.data = json.dumps({'frame_id': frame_id, 'detections': detections})
            self._results_pub.publish(msg_out)

            if self.save_json:
                with open(self.json_path, 'a') as f:
                    f.write(msg_out.data + '\n')

            # Visualization
            vis = result.plot() if result.masks is not None else frame.copy()

            if self.display:
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                cv2.putText(vis, f'Frame: {frame_id}',
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(vis, f'Detections: {len(detections)}',
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(vis, f'FPS: {fps:.1f}',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis, f'Device: {self.device}',
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow('YOLO Seg', vis)
                if cv2.waitKey(1) == 27:
                    self.get_logger().info('ESC pressed — shutting down')
                    self._stop_event.set()
                    os.kill(os.getpid(), signal.SIGINT)
                    break

            # Publish visualization regardless of display flag
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            self._vis_pub.publish(vis_msg)

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
    node = SegCoords()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
