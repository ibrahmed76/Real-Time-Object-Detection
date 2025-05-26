import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import time
from pathlib import Path
from datetime import datetime
import csv
import json

# Model path
DFINE_MODEL_PATH = '/home/ibrahim/Documents/Study/Computer Vision/Project/models/DFINE/models--ustc-community--dfine-small-obj2coco/snapshots/2b548d12a7623d674bb2aa2bf24022ce090b7736'

class DFineRoomMonitor:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        print("Loading DFine model...")
        self.processor = AutoImageProcessor.from_pretrained(DFINE_MODEL_PATH)
        self.model = DFineForObjectDetection.from_pretrained(DFINE_MODEL_PATH)
        self.model = self.model.to(device)
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
        # Room monitoring parameters
        self.entry_line = 0.3  # Virtual line at 30% of frame height
        self.total_count = 0
        self.last_positions = {}
        self.tracked_people = {}
        self.current_people = set()  # Track current people in frame
        
        # Initialize logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging files"""
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.logs_dir / f'dfine_room_log_{timestamp}.csv'
        self.json_path = self.logs_dir / f'dfine_room_log_{timestamp}.json'
        
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
        """Process a single frame with DFine"""
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        height = frame.shape[0]
        entry_y = int(height * self.entry_line)
        
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Process results
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([pil_image.size[::-1]]),
            threshold=0.3
        )
        
        # Convert back to numpy array for OpenCV
        frame_np = np.array(pil_image)
        
        # Initialize behavior counts
        behaviors = {
            'total': 0,  # Will be updated based on current detections
            'standing': 0,
            'sitting': 0,
            'using_phone': 0
        }
        
        # Clear current people set for this frame
        self.current_people.clear()
        
        # Process detections
        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                score = score.item()
                label = self.model.config.id2label[label_id.item()]
                box = [int(i) for i in box.tolist()]
                
                if label == 'person':
                    # Calculate center point for entry/exit detection
                    center_y = (box[1] + box[3]) / 2
                    person_id = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
                    self.current_people.add(person_id)
                    
                    # Check for entry/exit
                    if person_id not in self.last_positions:
                        self.last_positions[person_id] = center_y
                        self.total_count += 1  # New person detected
                    else:
                        last_y = self.last_positions[person_id]
                        if last_y < entry_y and center_y >= entry_y:
                            self.total_count += 1
                        elif last_y > entry_y and center_y <= entry_y:
                            self.total_count -= 1
                        self.last_positions[person_id] = center_y
                    
                    # Analyze pose
                    pose = self.analyze_pose(tuple(box))
                    if pose == 'standing':
                        behaviors['standing'] += 1
                    else:
                        behaviors['sitting'] += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    # Draw label with pose
                    label_text = f"Person ({pose}): {score:.2f}"
                    cv2.putText(frame_np, label_text, (box[0], box[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                elif label == 'cell phone':
                    behaviors['using_phone'] += 1
                    cv2.rectangle(frame_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    cv2.putText(frame_np, f"Phone: {score:.2f}", (box[0], box[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Update total count based on current detections
        behaviors['total'] = len(self.current_people)
        
        # Clean up old positions
        self.last_positions = {k: v for k, v in self.last_positions.items() if k in self.current_people}
        
        # Draw entry line
        cv2.line(frame_np, (0, entry_y), (frame_np.shape[1], entry_y), (0, 0, 255), 2)
        
        # Add statistics overlay
        cv2.putText(frame_np, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_np, f"Total: {behaviors['total']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_np, f"Standing: {behaviors['standing']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_np, f"Sitting: {behaviors['sitting']}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_np, f"Using Phone: {behaviors['using_phone']}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Log frame
        self.log_frame(behaviors)
        self.frame_count += 1
        
        # Convert back to BGR for display
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        return frame_bgr
    
    def close(self):
        """Save logs and cleanup"""
        with open(self.json_path, 'w') as f:
            json.dump(self.json_log, f, indent=2)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize monitor
    monitor = DFineRoomMonitor(device=device)
    
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
            cv2.imshow("DFine Room Monitor", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.close()

if __name__ == "__main__":
    main() 