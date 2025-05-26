import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path

def train_yolov11s():
    # Load dataset configuration
    # yaml_path = Path('/content/dataset/data.yaml')
    # with open(yaml_path, 'r') as f:
    #     data_config = yaml.safe_load(f)

    # Initialize YOLOv11x model
    model = YOLO('models/yolo11s.pt')  # Load pretrained YOLOv11x weights

    # Training configuration
    training_args = {
        'data': 'combined_dataset/data.yaml',  # Path to your dataset configuration
        'model': 'yolov11s.pt',               # Use a smaller model for limited data
        'epochs': 10,                        # Sufficient epochs for convergence
        'imgsz': 640,                         # Standard image size
        'batch': 192,                          # Adjust based on your GPU capacity
        'device': '0',                        # Use GPU if available
        'optimizer': 'AdamW',                 # Recommended optimizer for stability
        'lr0': 0.0001,                         # Initial learning rate
        'lrf': 0.000001,                          # Final learning rate fraction
        'momentum': 0.937,                    # Momentum for SGD (if used)
        'weight_decay': 0.0005,               # Regularization to prevent overfitting
        'patience': 50,                       # Early stopping patience
        'augment': True,                      # Enable data augmentation
        'mosaic': 1.0,                        # Mosaic augmentation probability
        'mixup': 0.0,                         # MixUp augmentation probability
        'hsv_h': 0.015,                       # HSV-Hue augmentation
        'hsv_s': 0.7,                         # HSV-Saturation augmentation
        'hsv_v': 0.4,                         # HSV-Value augmentation
        'degrees': 0.0,                       # Rotation augmentation
        'translate': 0.1,                     # Translation augmentation
        'scale': 0.5,                         # Scaling augmentation
        'shear': 0.0,                         # Shear augmentation
        'perspective': 0.0,                   # Perspective augmentation
        'flipud': 0.0,                        # Vertical flip augmentation
        'fliplr': 0.5,                        # Horizontal flip augmentation
        'cache': False,                       # Cache images for faster training
        'workers': 8,                         # Number of data loading workers
        'project': 'runs/train',              # Project directory
        'name': 'yolov11s_sit_stand',         # Experiment name
        'exist_ok': False,                    # Overwrite existing experiment
        'save_period': 10,                    # Save model every 10 epochs
        'seed': 42,                           # Reproducibility
        'deterministic': True,                # Deterministic training
        'verbose': True,                      # Verbose output
}


    # Start training
    results = model.train(**training_args)

    # Save the trained model
    model.save('trained_models/trained_model7.pt')

if __name__ == '__main__':
    train_yolov11s()