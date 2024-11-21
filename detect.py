import cv2
import torch
import numpy as np
import mss
import time
import sys
import os

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device


# Add YOLOv7 directory to path
YOLOV7_PATH = 'C:/src/python UiPath/yolov7'  # Adjust this to your YOLOv7 directory
sys.path.append(YOLOV7_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device


class LicensePlateDetector:
    def __init__(self, weights_path='license_plate_detection/weights/best.pt', device='0'):
        self.device = select_device(device)
        self.model = attempt_load(weights_path, device=self.device)
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        
        # Initialize screen capture
        self.sct = mss.mss()
        
    def preprocess_image(self, img):
        # Resize and normalize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img.unsqueeze(0)  # Add batch dimension
        return img
        
    def detect(self, frame):
        # Preprocess image
        img = self.preprocess_image(frame)
        
        # Inference
        pred = self.model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        detections = []
        if pred[0] is not None:
            # Rescale boxes to original image
            for *xyxy, conf, cls in pred[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append((x1, y1, x2, y2, float(conf)))
                
        return detections
    
    def run(self, monitor_number=1):
        # Get monitor
        monitor = self.sct.monitors[monitor_number]
        
        while True:
            # Capture screen
            screen = np.array(self.sct.grab(monitor))
            
            # BGR conversion for OpenCV
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
            
            # Get detections
            detections = self.detect(frame)
            
            # Draw detections
            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Plate {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Show result
            cv2.imshow('YOLOv7 License Plate Detection', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize detector
    detector = LicensePlateDetector(
        weights_path='c:/b/best.pt',
        device='0'  # Use '0' for GPU, 'cpu' for CPU
    )
    
    # Start detection
    detector.run(monitor_number=1)  # Adjust monitor_number as needed