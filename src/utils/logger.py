import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json

class BehaviorLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize CSV log file
        self.csv_path = self.log_dir / f"behavior_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'event_type', 'person_id', 'behavior',
            'total_count', 'standing_count', 'sitting_count',
            'phone_users', 'sleeping_count'
        ])
        
        # Initialize JSON log file for detailed events
        self.json_path = self.log_dir / f"detailed_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.json_file = open(self.json_path, 'w')
        self.events = []
        
    def log_event(self, event_type: str, person_id: int, behavior: str, 
                 counts: Dict[str, int]):
        """Log a single event"""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        self.csv_writer.writerow([
            timestamp, event_type, person_id, behavior,
            counts['total'], counts['standing'], counts['sitting'],
            counts['using_phone'], counts['sleeping']
        ])
        
        # Log to JSON
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'person_id': person_id,
            'behavior': behavior,
            'counts': counts
        }
        self.events.append(event)
        
    def log_frame(self, frame_number: int, behaviors: Dict[str, int]):
        """Log frame-level statistics"""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        self.csv_writer.writerow([
            timestamp, 'frame', -1, 'frame_stats',
            behaviors['total'], behaviors['standing'], behaviors['sitting'],
            behaviors['using_phone'], behaviors['sleeping']
        ])
        
        # Log to JSON
        event = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'behaviors': behaviors
        }
        self.events.append(event)
        
    def close(self):
        """Close log files and save final data"""
        # Save JSON events
        json.dump(self.events, self.json_file, indent=2)
        
        # Close files
        self.csv_file.close()
        self.json_file.close()
        
    def __del__(self):
        """Ensure files are closed on object destruction"""
        try:
            self.close()
        except:
            pass 