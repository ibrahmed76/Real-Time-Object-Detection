import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class BehaviorLogger:
    def __init__(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize CSV logger
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.logs_dir / f'behavior_log_{timestamp}.csv'
        self.json_path = self.logs_dir / f'behavior_log_{timestamp}.json'
        
        # Initialize CSV file with headers
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
        
        # Initialize JSON log
        self.json_log = []
        
    def log_frame(self, frame_number: int, behaviors: Dict[str, int]):
        """Log frame statistics"""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                frame_number,
                behaviors['total'],
                behaviors['standing'],
                behaviors['sitting'],
                behaviors['using_phone']
            ])
        
        # Log to JSON
        self.json_log.append({
            'timestamp': timestamp,
            'frame_number': frame_number,
            'behaviors': behaviors
        })
        
    def log_event(self, event_type: str, track_id: int, behavior: str, behaviors: Dict[str, int]):
        """Log specific events (e.g., behavior changes)"""
        timestamp = datetime.now().isoformat()
        
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'track_id': track_id,
            'behavior': behavior,
            'current_behaviors': behaviors
        }
        
        self.json_log.append(event)
        
    def close(self):
        """Save JSON log and close files"""
        with open(self.json_path, 'w') as f:
            json.dump(self.json_log, f, indent=2) 