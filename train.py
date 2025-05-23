import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path

def train_yolov11x():
    # Load dataset configuration
    yaml_path = Path('sitting and standing.v3i.yolov11/data.yaml')
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # Initialize YOLOv11x model
    model = YOLO('models/yolo11x.pt')  # Load pretrained YOLOv11x weights

    # Training configuration
    training_args = {
        'data': str(yaml_path),  # Path to data.yaml
        'epochs': 2,           # Number of training epochs
        'imgsz': 640,           # Image size
        'batch': 16,            # Batch size
        'device': '0' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        'workers': 8,           # Number of worker threads
        'patience': 50,         # Early stopping patience
        'save': True,           # Save checkpoints
        'save_period': 10,      # Save checkpoint every 10 epochs
        'cache': False,         # Cache images in memory
        'exist_ok': False,      # Overwrite existing experiment
        'pretrained': True,     # Use pretrained weights
        'optimizer': 'auto',    # Optimizer (SGD, Adam, etc.)
        'verbose': True,        # Print verbose output
        'seed': 42,            # Random seed for reproducibility
        'deterministic': True,  # Deterministic training
    }

    # Start training
    results = model.train(**training_args)

    # Save the trained model
    model.save('models/yolov11x_trained.pt')

if __name__ == '__main__':
    train_yolov11x() 