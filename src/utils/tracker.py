import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)
        self.tracked_people: Dict[int, Dict] = {}
        self.entry_line = 0.3  # Virtual line at 30% of frame height
        self.total_count = 0
        self.last_positions: Dict[int, float] = {}
        
    def update(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, Dict[str, int]]:
        """Update tracking and count entries/exits"""
        height = frame.shape[0]
        entry_y = int(height * self.entry_line)
        
        # Convert detections to tracker format
        tracks = []
        for det in detections:
            if det['class'] == 'person':
                x1, y1, x2, y2 = det['bbox']
                tracks.append(([x1, y1, x2, y2], det['confidence'], 'person'))
        
        # Update tracker
        tracks = self.tracker.update_tracks(tracks, frame=frame)
        
        # Initialize behavior counts
        behaviors = {
            'total': self.total_count,
            'standing': 0,
            'sitting': 0,
            'using_phone': 0
        }
        
        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate center point
            center_y = (y1 + y2) / 2
            
            # Check for entry/exit
            if track_id not in self.last_positions:
                self.last_positions[track_id] = center_y
            else:
                last_y = self.last_positions[track_id]
                if last_y < entry_y and center_y >= entry_y:
                    self.total_count += 1
                elif last_y > entry_y and center_y <= entry_y:
                    self.total_count -= 1
                self.last_positions[track_id] = center_y
            
            # Update tracked person data
            self.tracked_people[track_id] = {
                'bbox': (x1, y1, x2, y2),
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
            }
            
            # Draw ID and count
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw entry line
        cv2.line(frame, (0, entry_y), (frame.shape[1], entry_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Total: {self.total_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        behaviors['total'] = self.total_count
        return frame, behaviors

class BehaviorAnalyzer:
    def __init__(self):
        self.standing_threshold = 0.7  # Height/width ratio threshold
        
    def analyze_pose(self, bbox: Tuple[int, int, int, int]) -> str:
        """Analyze if person is sitting or standing based on bbox ratio"""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        ratio = height / width
        
        return 'standing' if ratio > self.standing_threshold else 'sitting'
    
    def check_phone_usage(self, person_bbox: Tuple[int, int, int, int],
                         phone_boxes: List[Tuple[int, int, int, int]]) -> bool:
        """Check if person is using phone based on proximity"""
        if not phone_boxes:
            return False
            
        px1, py1, px2, py2 = person_bbox
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        
        for phone_box in phone_boxes:
            tx1, ty1, tx2, ty2 = phone_box
            phone_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
            
            # Calculate distance between centers
            distance = np.sqrt((person_center[0] - phone_center[0])**2 +
                             (person_center[1] - phone_center[1])**2)
            
            # If phone is close to person's upper body
            if distance < (px2 - px1) * 0.5 and ty1 > py1 and ty2 < py2:
                return True
                
        return False 