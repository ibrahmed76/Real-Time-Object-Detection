import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Dict, List, Tuple
import cv2

class PersonTracker:
    def __init__(self, entry_line_y: int = 300):
        self.tracker = DeepSort(max_age=30)
        self.entry_line_y = entry_line_y
        self.people_count = 0
        self.tracked_people: Dict[int, Dict] = {}  # id -> {status, last_position}
        self.next_id = 0
        
    def update(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Update tracking and analyze behaviors
        detections: List of dicts with keys: 'bbox', 'class', 'confidence'
        """
        # Convert detections to tracker format
        tracks = []
        for det in detections:
            if det['class'] == 'person':
                x1, y1, x2, y2 = det['bbox']
                tracks.append(([x1, y1, x2, y2], det['confidence'], 'person'))
        
        # Update tracker
        tracks = self.tracker.update_tracks(tracks, frame=frame)
        
        # Process tracks and analyze behaviors
        current_people = set()
        behaviors = {
            'total': 0,
            'standing': 0,
            'sitting': 0,
            'using_phone': 0,
            'sleeping': 0
        }
        
        # Draw entry line
        cv2.line(frame, (0, self.entry_line_y), 
                (frame.shape[1], self.entry_line_y), (0, 255, 0), 2)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Update person status
            if track_id not in self.tracked_people:
                self.tracked_people[track_id] = {
                    'status': 'unknown',
                    'last_position': (y1 + y2) / 2
                }
            
            # Check crossing entry line
            last_y = self.tracked_people[track_id]['last_position']
            current_y = (y1 + y2) / 2
            
            if last_y < self.entry_line_y and current_y >= self.entry_line_y:
                self.people_count += 1
            elif last_y > self.entry_line_y and current_y <= self.entry_line_y:
                self.people_count -= 1
            
            self.tracked_people[track_id]['last_position'] = current_y
            current_people.add(track_id)
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Clean up old tracks
        self.tracked_people = {k: v for k, v in self.tracked_people.items() 
                             if k in current_people}
        
        # Update behaviors
        behaviors['total'] = self.people_count
        
        return frame, behaviors

class BehaviorAnalyzer:
    def __init__(self):
        self.standing_threshold = 0.7  # Ratio of height to width for standing
        
    def analyze_pose(self, bbox: Tuple[int, int, int, int]) -> str:
        """Analyze if person is sitting or standing based on bbox ratio"""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        ratio = height / width
        
        return 'standing' if ratio > self.standing_threshold else 'sitting'
    
    def check_phone_usage(self, person_bbox: Tuple[int, int, int, int], 
                         phone_bboxes: List[Tuple[int, int, int, int]]) -> bool:
        """Check if person is using phone based on bbox proximity"""
        if not phone_bboxes:
            return False
            
        px1, py1, px2, py2 = person_bbox
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        
        for phone_bbox in phone_bboxes:
            phx1, phy1, phx2, phy2 = phone_bbox
            phone_center = ((phx1 + phx2) / 2, (phy1 + phy2) / 2)
            
            # Calculate distance between centers
            distance = np.sqrt((person_center[0] - phone_center[0])**2 + 
                             (person_center[1] - phone_center[1])**2)
            
            # If phone is close to person's upper body
            if distance < (px2 - px1) * 0.5 and phy1 > py1 and phy2 < py2:
                return True
                
        return False 