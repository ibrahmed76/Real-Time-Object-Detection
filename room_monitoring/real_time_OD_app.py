import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import sys
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from src.utils.tracker import PersonTracker, BehaviorAnalyzer
from src.utils.logger import BehaviorLogger

# Model paths
YOLO_MODEL_PATH = 'trained_models/trained_model6.pt'  # Updated to model 6
DFINE_MODEL_PATH = '/home/ibrahim/Documents/Study/Computer Vision/Project/models/DFINE/models--ustc-community--dfine-small-obj2coco/snapshots/2b548d12a7623d674bb2aa2bf24022ce090b7736'

class SmartRoomMonitor:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Initialize both models
        print("Loading YOLO model...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO model loaded successfully from {YOLO_MODEL_PATH}")
        print(f"YOLO model classes: {self.yolo_model.names}")
        
        print("Loading DFine model...")
        self.dfine_processor = AutoImageProcessor.from_pretrained(DFINE_MODEL_PATH)
        self.dfine_model = DFineForObjectDetection.from_pretrained(DFINE_MODEL_PATH)
        self.dfine_model = self.dfine_model.to(device)
        print("DFine model loaded successfully")
            
        # Initialize tracker and analyzer
        self.tracker = PersonTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.logger = BehaviorLogger()
        
        # Initialize statistics
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        self.current_people = set()  # Track current people in frame
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """Process a single frame using both models"""
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        # Clear current people set for this frame
        self.current_people.clear()
        
        # Get YOLO detections with lower confidence threshold
        yolo_results = self.yolo_model(frame, conf=0.25, device=self.device)
        yolo_detections = []
        
        # Initialize behaviors
        behaviors = {
            'total': 0,
            'standing': 0,
            'sitting': 0,
            'using_phone': 0
        }
        
        # Process YOLO detections
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.yolo_model.names[cls]
                
                # Only track people (sitting or standing)
                if class_name in ['sitting', 'standing']:
                    person_id = f"yolo_{x1}_{y1}_{x2}_{y2}"
                    self.current_people.add(person_id)
                    yolo_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': class_name,
                        'confidence': conf,
                        'source': 'yolo',
                        'id': person_id
                    })
                    
                    # Update behavior counts
                    if class_name == 'standing':
                        behaviors['standing'] += 1
                    elif class_name == 'sitting':
                        behaviors['sitting'] += 1
                
                # Handle phone detection
                elif class_name == 'phone':
                    behaviors['using_phone'] += 1
                
                # Draw bounding box with different colors for different classes
                if class_name == 'standing':
                    color = (0, 255, 0)  # Green for standing
                elif class_name == 'sitting':
                    color = (255, 0, 0)  # Red for sitting
                else:  # phone
                    color = (0, 0, 255)  # Blue for phone
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with class and confidence
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update total count based on current detections
        behaviors['total'] = len(self.current_people)
        
        # Add statistics overlay
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Total People: {behaviors['total']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Standing: {behaviors['standing']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Sitting: {behaviors['sitting']}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Using Phone: {behaviors['using_phone']}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Log frame statistics
        self.logger.log_frame(self.frame_count, behaviors)
        self.frame_count += 1
        
        return frame, behaviors
    
    def _combine_detections(self, yolo_dets: List[Dict], dfine_dets: List[Dict]) -> List[Dict]:
        """Combine detections from both models using NMS"""
        all_detections = yolo_dets + dfine_dets
        if not all_detections:
            return []
            
        # Sort by confidence
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply NMS
        final_detections = []
        used_boxes = set()
        
        for det in all_detections:
            x1, y1, x2, y2 = det['bbox']
            current_box = (x1, y1, x2, y2)
            
            # Check if this box overlaps with any used box
            overlap = False
            for used_box in used_boxes:
                if self._calculate_iou(current_box, used_box) > 0.5:
                    overlap = True
                    break
            
            if not overlap:
                final_detections.append(det)
                used_boxes.add(current_box)
        
        return final_detections
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def close(self):
        """Clean up resources"""
        self.logger.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize monitor
    monitor = SmartRoomMonitor(device=device)
    
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
                
            processed_frame, behaviors = monitor.process_frame(frame)
            cv2.imshow("Smart Room Monitor", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.close()

if __name__ == "__main__":
    main() 