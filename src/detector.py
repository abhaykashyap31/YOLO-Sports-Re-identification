import cv2
from ultralytics import YOLO
import numpy as np

# Fix for NumPy compatibility issue
import numpy
if not hasattr(numpy, 'float'):
    numpy.float = numpy.float64
if not hasattr(numpy, 'int'):
    numpy.int = numpy.int_
if not hasattr(numpy, 'bool'):
    numpy.bool = numpy.bool_

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def detect_players(self, frame):
        results = self.model(frame)
        players = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0]
                if not (hasattr(xyxy, '__iter__') and len(xyxy) == 4):
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                players.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'class': cls})
        return players

class YOLOv11Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
                
            for box in r.boxes:
                try:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0]
                    
                    # Ensure xyxy is properly converted to CPU and numpy
                    if hasattr(xyxy, 'cpu'):
                        xyxy = xyxy.cpu().numpy()
                    elif hasattr(xyxy, 'numpy'):
                        xyxy = xyxy.numpy()
                    
                    # Validate xyxy format
                    if not (hasattr(xyxy, '__len__') and len(xyxy) == 4):
                        print(f"Warning: Malformed bbox from YOLO: {xyxy}")
                        continue
                    
                    # Convert to float first, then int to avoid tensor issues
                    x1, y1, x2, y2 = [float(coord) for coord in xyxy]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Validate dimensions
                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        print(f"Warning: Invalid bbox dimensions: w={w}, h={h}")
                        continue
                    
                    det = {
                        'bbox': [x1, y1, x2, y2],  # xyxy format for drawing
                        'conf': float(conf),  # Ensure it's a Python float
                        'class': cls
                    }
                    
                    # Strict validation
                    if (
                        isinstance(det['bbox'], list) and len(det['bbox']) == 4 and
                        all(isinstance(v, (int, float)) and not np.isnan(v) for v in det['bbox']) and
                        isinstance(det['conf'], float) and
                        det['conf'] > 0 and not np.isnan(det['conf'])
                    ):
                        detections.append(det)
                    else:
                        print(f"Warning: Malformed detection dict: {det}")
                        
                except Exception as e:
                    print(f"Warning: Exception in detection parsing: {e}")
                    continue
        
        return detections
