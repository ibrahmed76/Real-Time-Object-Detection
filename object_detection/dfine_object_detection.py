import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import time
import os
import argparse
from pathlib import Path

def process_image(model, image_processor, image_path, output_path):
    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Process image
    inputs = image_processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform object detection
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = image_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([pil_image.size[::-1]]), 
        threshold=0.3
    )

    # Process results
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = model.config.id2label[label_id.item()]
            box = [int(i) for i in box.tolist()]

            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Add label
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result
    cv2.imwrite(str(output_path), frame)
    print(f"Processed {image_path.name} -> {output_path.name}")

def process_folder():
    # Initialize DFine model and processor
    model_path = "/home/ibrahim/Documents/Study/Computer Vision/Project/models/DFINE/models--ustc-community--dfine-xlarge-obj2coco/snapshots/15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d"
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    model = DFineForObjectDetection.from_pretrained(model_path)
    model = model.to(device)

    # Create images directory and its subdirectories if they don't exist
    images_dir = Path('images')
    input_dir = images_dir / 'input'
    output_dir = images_dir / 'results'/ 'dfine_results'
    
    images_dir.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Get all image files
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpeg'))
    
    if not image_files:
        print("No input images found in images/input directory.")
        return

    print(f"Found {len(image_files)} images to process...")
    
    # Process each image
    for image_path in image_files:
        output_path = output_dir / f"detected_{image_path.name}"
        process_image(model, image_processor, image_path, output_path)

    print("Processing complete!")

def process_webcam():
    # Initialize DFine model and processor
    model_path = "/home/ibrahim/Documents/Study/Computer Vision/Project/models/DFINE/models--ustc-community--dfine-xlarge-obj2coco/snapshots/15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d"
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    model = DFineForObjectDetection.from_pretrained(model_path)
    model = model.to(device)
    
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

        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Process image
        inputs = image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform object detection
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = image_processor.post_process_object_detection(
            outputs, 
            target_sizes=torch.tensor([pil_image.size[::-1]]), 
            threshold=0.3
        )

        # Process results
        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                score = score.item()
                label = model.config.id2label[label_id.item()]
                box = [int(i) for i in box.tolist()]

                # Draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Add label
                label_text = f"{label}: {score:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Real-time Object Detection (DFine)", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Check for AMD GPU
    global device
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("No GPU available, using CPU")
        device = "cpu"

    parser = argparse.ArgumentParser(description='DFine Object Detection')
    parser.add_argument('--mode', type=str, choices=['webcam', 'folder'], default='webcam',
                      help='Detection mode: webcam for real-time detection, folder for processing images')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        process_webcam()
    else:
        process_folder()

if __name__ == "__main__":
    main() 