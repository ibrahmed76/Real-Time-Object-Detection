# Smart Room Monitoring and Object Detection System

A comprehensive computer vision system that combines YOLO and DFine models for both room monitoring and general object detection tasks.

## Project Structure

```
smart-room-monitor/
├── room_monitoring/
│   ├── yolo_detection.py          # YOLO-based room monitoring
│   ├── dfine_detection.py         # DFine-based room monitoring
│   └── real_time_OD_app.py        # Combined model room monitoring
├── object_detection/
│   ├── yolo_object_detection.py   # YOLO-based general object detection
│   ├── dfine_object_detection.py  # DFine-based general object detection
│   └── object_detection_app.py    # Unified object detection application
├── models/
│   └── DFINE/                     # DFine model directory
├── utils/
│   └── download_dfine.py          # DFine model downloader
├── requirements.txt
└── README.md
```

## Room Monitoring System

### Individual Models

1. **YOLO-based Monitor** (`yolo_detection.py`)
   - Real-time people detection and tracking
   - Entry/exit counting
   - Pose analysis (standing/sitting)
   - Phone usage detection
   - High-speed processing

2. **DFine-based Monitor** (`dfine_detection.py`)
   - Detailed pose estimation
   - Fine-grained activity recognition
   - Better accuracy for complex poses
   - More detailed behavior analysis

### Combined Model (`real_time_OD_app.py`)

The combined system leverages the strengths of both models:
- Uses YOLO for fast initial detection and tracking
- Employs DFine for detailed pose analysis
- Combines results for more accurate monitoring
- Provides comprehensive room analytics

## Object Detection System

### Individual Models

1. **YOLO Object Detection** (`yolo_object_detection.py`)
   - Real-time object detection
   - High-speed processing
   - Good for general objects
   - Efficient resource usage

2. **DFine Object Detection** (`dfine_object_detection.py`)
   - Detailed object analysis
   - Better for fine-grained detection
   - Higher accuracy for specific objects
   - More detailed feature extraction

### Unified Application (`object_detection_app.py`)

Features:
- Supports both image and video input
- Batch processing for folders
- Multiple model selection
- Real-time and offline processing
- Comprehensive detection results

## Model Installation

### YOLO Models
1. Visit [YOLO Documentation](https://docs.ultralytics.com/tasks/detect/)

2. Choose your appropriate model size from documentation

3. Download the model by clicking on the appropriate link above

4. Place the downloaded model in the models directory:
```bash
# Create models directory if it doesn't exist
mkdir -p models

# Move the downloaded model to the models directory
mv yolov11*.pt models/
```

### DFine Models
1. Create the DFINE directory in models:
```bash
mkdir -p models/DFINE
```

2. Run the DFine downloader script:
```bash
python utils/download_dfine.py
```

The script will download and place all required model files in the models/DFINE directory.

## Model Strengths and Advantages

### YOLO Strengths
- Fast processing speed
- Good for real-time applications
- Efficient resource usage
- Strong general object detection
- Well-suited for tracking

### DFine Strengths
- Detailed pose estimation
- Better accuracy for specific objects
- Fine-grained feature detection
- Superior in complex scenarios
- Better for detailed analysis

### Combined System Advantages
1. **Complementary Detection**
   - YOLO handles fast, general detection
   - DFine provides detailed analysis
   - Better overall accuracy

2. **Enhanced Features**
   - More accurate pose estimation
   - Better activity recognition
   - Improved tracking capabilities
   - Comprehensive monitoring

3. **Flexible Processing**
   - Can use either model independently
   - Combined processing for better results
   - Adaptable to different scenarios

## Requirements

The project requires several Python packages which are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

### Key Dependencies

1. **Core Dependencies**
   - `torch` (>=2.0.0) - Deep learning framework
   - `opencv-python` (>=4.5.0) - Computer vision operations
   - `numpy` (>=1.19.0) - Numerical computations
   - `Pillow` (>=8.0.0) - Image processing

2. **YOLO Specific**
   - `ultralytics` (>=8.0.0) - YOLO implementation

3. **DFine Specific**
   - `transformers` (>=4.0.0) - DFine model support

4. **Utilities**
   - `tqdm` (>=4.65.0) - Progress bars
   - `PyYAML` (>=6.0) - Configuration files

5. **Web Interface**
   - `gradio` (>=3.0.0) - Web UI for object detection app
## Usage

### Room Monitoring
```bash
# YOLO-based monitoring
python room_monitoring/yolo_detection.py

# DFine-based monitoring
python room_monitoring/dfine_detection.py

# Combined monitoring
python room_monitoring/real_time_OD_app.py
```

### Object Detection
```bash
# General object detection
python object_detection/object_detection_app.py --input path/to/input --model [yolo|dfine|combined]
```

## Acknowledgments

- YOLO model: [Ultralytics](https://github.com/ultralytics/yolov5)
- DFine model: [USTC Community](https://github.com/ustc-community/dfine)
- OpenCV for computer vision capabilities
