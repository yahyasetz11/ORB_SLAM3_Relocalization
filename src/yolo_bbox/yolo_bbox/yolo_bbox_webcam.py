import cv2
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
        # Model init
        self.model_path = "/home/orange/VSLAMxYOLO/src/yolo_bbox/yolo_bbox/model/yolov8n-oiv7"
        self.model = YOLO(self.model_path)
        self.conf_value = 0.5
        
        # Camera settings
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().info("Cannot open webcam")
            raise RuntimeError("Unable to open webcam")
        
        self.results = self.create_publisher(String, 'yolo/results', 10)

        self.frame_id = 0 
        
        # self.img = self.create_subscription(Image, )

        self.bbox_coords = self.create_publisher(Int32MultiArray, 'bbox_coords', 10)

        self.timer = self.create_timer(0.2, self.detection_callback)
        
        self.DOOR_ID = 164

        self.get_logger().info("Start YOLO Webcam Detection")
    def detection_callback(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warning("Fail to read frame from webcam")
            return
        
        # Resize the frame
        frame = cv2.resize(frame, (640, 480))
    
        detected = self.model.predict(frame, conf=self.conf_value, verbose=False)
        # results = self.model(frame)[0]
        detections = []
    
        for obj in detected: 
            for box in obj.boxes:
                # Extracting the coords (Str->int)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Extrac conf score and cls label
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id != self.DOOR_ID:
                    continue

                det = (
                    {
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
                )
                
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
                # time.sleep(1)
                
                # # Coordinates
                # top_l = (x1, y1)
                # top_r = (x2, y1)
                # bot_l = (x1, y2)
                # bot_r = (x2, y2)
                # # Print out the coords
                # print("Top Left: ", top_l)
                # print("Top Right: ", top_r)
                # print("Bottom Left: ", bot_l)
                # print("Bottom Right: ", bot_r)

        msg_out = String()
        msg_out.data = json.dumps({
            "frame_id": self.frame_id,
            "detections":detections
            })
        self.results.publish(msg_out)
        
        self.frame_id += 1
        
        # Debugging Show
        cv2.imshow("YOLO Webcam", frame)
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
    # bbox.detection_callback()

    rclpy.spin(bbox)

    bbox.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main() 