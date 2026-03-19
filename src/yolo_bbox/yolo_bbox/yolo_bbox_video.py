import cv2
import os
import rclpy
import time
import json

from ultralytics import YOLO
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String, Int32MultiArray, Int32
from sensor_msgs.msg import Image

class BBOX_Coords(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        # Model init — realpath resolves symlink from install/ back to source
        pkg_dir = os.path.dirname(os.path.realpath(__file__))
        ws_root = os.path.normpath(os.path.join(pkg_dir, '..', '..', '..'))
        self.model_path = os.path.join(pkg_dir, "model", "yolov8n-oiv7")
        self.model = YOLO(self.model_path)
        self.conf_value = 0.2

        # Video settings — data lives at workspace root /data/
        self.cap = cv2.VideoCapture(os.path.join(ws_root, "data", "validation_back.mp4"))
        if not self.cap.isOpened():
            self.get_logger().info("Cannot open video")
            raise RuntimeError("Unable to open video")
        
        self.results = self.create_publisher(String, 'yolo/results', 10)

        self.frame_id = 0 
        
        # self.img = self.create_subscription(Image, )

        self.bbox_coords = self.create_publisher(Int32MultiArray, 'bbox_coords', 10)

        self.timer = self.create_timer(0.3, self.detection_callback)
        
        self.DOOR_ID = 164
        
        self.frame_skip = 3
        
        self.json_path = "door_detections.jsonl"
        self.save_json = False

        self.get_logger().info("Start YOLO Video Detection")

    def detection_callback(self):
        if self.frame_id % self.frame_skip != 0:
            self.frame_id += 1
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().info("End of video or failed to read frame")
            rclpy.shutdown()
            return
        
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
    
        detected = self.model.predict(frame, conf=self.conf_value, verbose=False)
        detections = []

        for obj in detected: 
            for box in obj.boxes:
                # Extracting the coords
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Extract conf score and cls label
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if cls_id != self.DOOR_ID:
                    continue

                det = {
                    "cls_id": cls_id,
                    "conf_score": conf,
                    "bbox_coords": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "corner_coords": {
                        "top_left": [int(x1), int(y1)],
                        "top_right": [int(x2), int(y1)],
                        "bot_left": [int(x1), int(y2)],
                        "bot_right": [int(x2), int(y2)]
                    }
                }
                
                detections.append(det)
                
                # Debugging
                label = f"id:{cls_id} conf:{conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # # Coordinates
                # top_l = (int(x1), int(y1))
                # top_r = (int(x2), int(y1))
                # bot_l = (int(x1), int(y2))
                # bot_r = (int(x2), int(y2))

                # print("Top Left: ", top_l)
                # print("Top Right: ", top_r)
                # print("Bottom Left: ", bot_l)
                # print("Bottom Right: ", bot_r)

        msg_out = String()
        msg_out.data = json.dumps({
            "frame_id": self.frame_id,
            "detections": detections
        })
        self.results.publish(msg_out)
        
        if self.save_json:
            with open(self.json_path, "a") as file:
                file.write(msg_out.data + "\n")
        
        self.frame_id += 1
        
        cv2.putText(frame, f"Frame: {self.frame_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Doors: {len(detections)}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # Debugging Show
        cv2.imshow("YOLO Video", frame)
        if cv2.waitKey(1) == 27:
            self.get_logger().info("Shutting down")
            rclpy.shutdown()
                
    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    bbox = BBOX_Coords()
    rclpy.spin(bbox)

    bbox.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()