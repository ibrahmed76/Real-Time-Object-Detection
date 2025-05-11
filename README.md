# Real-time Object Detection Project

This project implements real-time object detection using two different models: YOLOv11 and DFine.

## Requirements

- Python 3.8 or higher
- Webcam
- CUDA-capable GPU (recommended for better performance)

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create models directory:
```bash
mkdir models
```

4. Download YOLOv11x model:
   - Visit [YOLOv11 Documentation](https://docs.ultralytics.com/tasks/detect/)
   - Download the YOLOv11x model
   - Place the downloaded `yolo11x.pt` file in the `models` directory

5. Download DFine model:
```bash
python download_dfine.py
```

## Running the Code

You can run either of the two object detection scripts:

1. YOLO Object Detection:
```bash
python yolo_object_detection.py
```

2. DFine Object Detection:
```bash
python dfine_object_detection.py
```

## Controls

- Press 'q' to quit the application
- FPS is displayed in the top-left corner of the window

## Notes

- The YOLO model provides faster inference but may be less accurate
- The DFine model provides more accurate detections but may be slower
- Both models will automatically use GPU if available through ROCm, otherwise will fall back to CPU 