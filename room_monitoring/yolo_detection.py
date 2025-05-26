import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
from datetime import datetime
import csv
import json

# Model path
YOLO_MODEL_PATH = 'models/yolo11s.pt'

class YOLORoomMonitor:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        print("Loading YOLO model...")
        self.model = YOLO(YOLO_MODEL_PATH)
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
        # Room monitoring parameters
        self.entry_line = 0.3  # Virtual line at 30% of frame height
        self.total_count = 0
        self.last_positions = {}
        self.tracked_people = {}
        
        # Initialize logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging files"""
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.logs_dir / f'yolo_room_log_{timestamp}.csv'
        self.json_path = self.logs_dir / f'yolo_room_log_{timestamp}.json'
        
        # Initialize CSV file
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'frame_number',
                'total_count',
                'standing_count',
                'sitting_count',
                'phone_usage_count'
            ])
        
        self.json_log = []
        
    def log_frame(self, behaviors):
        """Log frame statistics"""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.frame_count,
                behaviors['total'],
                behaviors['standing'],
                behaviors['sitting'],
                behaviors['using_phone']
            ])
        
        # Log to JSON
        self.json_log.append({
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'behaviors': behaviors
        })
        
    def analyze_pose(self, bbox):
        """Analyze if person is sitting or standing"""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        ratio = height / width
        return 'standing' if ratio > 0.7 else 'sitting'
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with YOLO"""
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        height = frame.shape[0]
        entry_y = int(height * self.entry_line)
        
        # Get detections
        results = self.model(frame, conf=0.5, device=self.device)
        
        # Initialize behavior counts
        behaviors = {
            'total': self.total_count,
            'standing': 0,
            'sitting': 0,
            'using_phone': 0
        }
        
        # Track phones
        phone_boxes = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                if class_name == 'person':
                    # Calculate center point for entry/exit detection
                    center_y = (y1 + y2) / 2
                    person_id = f"{x1}_{y1}_{x2}_{y2}"  # Simple ID based on position
                    
                    # Check for entry/exit
                    if person_id not in self.last_positions:
                        self.last_positions[person_id] = center_y
                    else:
                        last_y = self.last_positions[person_id]
                        if last_y < entry_y and center_y >= entry_y:
                            self.total_count += 1
                        elif last_y > entry_y and center_y <= entry_y:
                            self.total_count -= 1
                        self.last_positions[person_id] = center_y
                    
                    # Analyze pose
                    pose = self.analyze_pose((x1, y1, x2, y2))
                    if pose == 'standing':
                        behaviors['standing'] += 1
                    else:
                        behaviors['sitting'] += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with pose
                    label = f"Person ({pose}): {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                elif class_name == 'cell phone':
                    phone_boxes.append((x1, y1, x2, y2))
                    behaviors['using_phone'] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Phone: {conf:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw entry line
        cv2.line(frame, (0, entry_y), (frame.shape[1], entry_y), (0, 0, 255), 2)
        
        # Add statistics overlay
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Total: {behaviors['total']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Standing: {behaviors['standing']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Sitting: {behaviors['sitting']}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Using Phone: {behaviors['using_phone']}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Log frame
        self.log_frame(behaviors)
        self.frame_count += 1
        
        return frame
    
    def close(self):
        """Save logs and cleanup"""
        with open(self.json_path, 'w') as f:
            json.dump(self.json_log, f, indent=2)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize monitor
    monitor = YOLORoomMonitor(device=device)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            processed_frame = monitor.process_frame(frame)
            cv2.imshow("YOLO Room Monitor", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.close()

if __name__ == "__main__":
    main() 