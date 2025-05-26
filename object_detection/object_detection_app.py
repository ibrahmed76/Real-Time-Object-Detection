import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import threading
import queue
import os

# Global variables for model instances
yolo_model = None
dfine_model = None
dfine_processor = None
device = None
webcam_active = False
current_cap = None

def initialize_models():
    global yolo_model, dfine_model, dfine_processor, device
    
    # Initialize device
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("No GPU available, using CPU")
        device = "cpu"
    
    # Initialize YOLO model
    yolo_model = YOLO('trained_models/trained_model4.pt')
    
    # Initialize DFine model
    model_path = "/home/ibrahim/Documents/Study/Computer Vision/Project/models/DFINE/models--ustc-community--dfine-xlarge-obj2coco/snapshots/15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d"
    dfine_processor = AutoImageProcessor.from_pretrained(model_path)
    dfine_model = DFineForObjectDetection.from_pretrained(model_path)
    dfine_model = dfine_model.to(device)

def process_image_yolo(image):
    if image is None:
        return None
    
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Perform detection
    results = yolo_model(image, conf=0.5, device="cpu")
    
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
            class_name = yolo_model.names[cls]

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def process_image_dfine(image):
    if image is None:
        return None
    
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Process image
    inputs = dfine_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform detection
    with torch.no_grad():
        outputs = dfine_model(**inputs)

    # Post-process results
    results = dfine_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([image.size[::-1]]), 
        threshold=0.3
    )

    # Convert back to numpy for drawing
    image_np = np.array(image)
    
    # Process results
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = dfine_model.config.id2label[label_id.item()]
            box = [int(i) for i in box.tolist()]

            # Draw bounding box
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Add label
            label_text = f"{label}: {score:.2f}"
            cv2.putText(image_np, label_text, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_np

def process_video_yolo(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = str(video_path).replace('.mp4', '_detected.mp4')
    out = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_image_yolo(frame)
        if out is None:
            h, w = processed.shape[:2]
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
        out.write(processed)
    cap.release()
    if out:
        out.release()
    return out_path

def process_video_dfine(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = str(video_path).replace('.mp4', '_detected.mp4')
    out = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_image_dfine(frame)
        if out is None:
            h, w = processed.shape[:2]
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
        out.write(processed)
    cap.release()
    if out:
        out.release()
    return out_path

def process_folder(model_type):
    input_dir = Path('images/input')
    if not input_dir.exists() or not input_dir.is_dir():
        return "Input directory 'images/input' does not exist."
    results_dir = input_dir.parent / 'results'
    results_subdir = results_dir / f'{model_type.lower()}_results'
    results_subdir.mkdir(parents=True, exist_ok=True)
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpeg'))
    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
    if not image_files and not video_files:
        return "No image or video files found in the input directory."
    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        if model_type == 'YOLO':
            processed = process_image_yolo(image)
        else:
            processed = process_image_dfine(image)
        if processed is not None:
            output_path = results_subdir / f"detected_{image_path.name}"
            cv2.imwrite(str(output_path), processed)
    for video_path in video_files:
        if model_type == 'YOLO':
            out_path = process_video_yolo(video_path)
        else:
            out_path = process_video_dfine(video_path)
        if out_path:
            final_path = results_subdir / Path(out_path).name
            os.rename(out_path, final_path)
    return f"Processing complete. Results saved in:\n{results_subdir}"

def process_single(input_file, model_type):
    if input_file is None:
        return None
    ext = str(input_file).lower()
    if ext.endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(input_file)
        if model_type == 'YOLO':
            processed = process_image_yolo(image)
        else:
            processed = process_image_dfine(image)
        return processed
    elif ext.endswith(('.mp4', '.avi')):
        if model_type == 'YOLO':
            out_path = process_video_yolo(input_file)
        else:
            out_path = process_video_dfine(input_file)
        return out_path
    else:
        return None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Object Detection App") as app:
        gr.Markdown("# Object Detection App")
        with gr.Tabs():
            with gr.TabItem("Folder Processing"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Instructions")
                        gr.Markdown("1. Place images/videos in the 'images/input' folder.")
                        gr.Markdown("2. Select your preferred model.")
                        gr.Markdown("3. Click 'Process Folder' to start processing.")
                        gr.Markdown("4. Results will be saved in a 'results' folder next to your input folder.")
                        folder_model_type = gr.Radio([
                            "YOLO", "DFine"],
                            label="Select Model",
                            value="YOLO"
                        )
                        process_folder_btn = gr.Button("Process Folder")
                    with gr.Column():
                        folder_output = gr.Textbox(label="Processing Status")
                process_folder_btn.click(
                    fn=process_folder,
                    inputs=[folder_model_type],
                    outputs=[folder_output]
                )
            with gr.TabItem("Single Image/Video"):
                with gr.Row():
                    with gr.Column():
                        upload_model_type = gr.Radio(["YOLO", "DFine"], label="Select Model", value="YOLO")
                        file_input = gr.File(label="Upload Image or Video", type="filepath")
                        process_file_btn = gr.Button("Process File")
                    with gr.Column():
                        file_output = gr.Image(label="Processed Image/Video Output")
                def process_and_display(input_file, model_type):
                    result = process_single(input_file, model_type)
                    if isinstance(result, str):
                        return gr.update(value=None), result  # For video, return path as text
                    return result, None
                process_file_btn.click(
                    fn=process_single,
                    inputs=[file_input, upload_model_type],
                    outputs=[file_output]
                )
    
    return app

if __name__ == "__main__":
    # Initialize models
    initialize_models()
    
    # Create and launch the interface
    app = create_interface()
    app.launch(share=True) 