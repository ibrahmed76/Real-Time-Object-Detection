import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import torch
import argparse
from pathlib import Path

def process_image(model, image_path, output_path):
    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Perform object detection
    results = model(frame, conf=0.5, device="cpu")  # Using CPU with OpenVINO acceleration

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result
    cv2.imwrite(str(output_path), frame)
    print(f"Processed {image_path.name} -> {output_path.name}")

def process_folder():
    # Initialize YOLO model
    model_path = os.path.join('models', 'yolo11x.pt')
    model = YOLO(model_path)

    # Create images directory and its subdirectories if they don't exist
    images_dir = Path('images')
    input_dir = images_dir / 'input'
    output_dir = images_dir / 'results' / 'yolo_results'
    
    images_dir.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpeg'))
    
    if not image_files:
        print("No input images found in images/input directory.")
        return

    print(f"Found {len(image_files)} images to process...")
    
    # Process each image
    for image_path in image_files:
        output_path = output_dir / f"detected_{image_path.name}"
        process_image(model, image_path, output_path)

    print("Processing complete!")

def process_webcam():
    # Initialize YOLO model
    model_path = os.path.join('models', 'yolo11x.pt')
    model = YOLO(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting real-time object detection...")
    print("Press 'q' to quit")

    # Initialize FPS calculation variables
    prev_time = time.time()
    fps = 0

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Perform object detection
        results = model(frame, conf=0.5, device="cpu")

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Real-time Object Detection", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--mode', type=str, choices=['webcam', 'folder'], default='webcam',
                      help='Detection mode: webcam for real-time detection, folder for processing images')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        process_webcam()
    else:
        process_folder()

if __name__ == "__main__":
    main() 