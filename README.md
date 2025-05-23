# Smart Room Monitoring System

A real-time room monitoring system that uses computer vision to track people, their poses, and activities in a room. The system provides two different implementations using YOLO and DFine models for object detection.

## Features

### Core Features
- Real-time people detection and tracking
- Entry/exit counting
- Pose analysis (standing/sitting detection)
- Phone usage detection
- FPS monitoring
- Comprehensive logging system

### Monitoring Capabilities
- Total people count in the room
- Standing vs. sitting count
- Phone usage tracking
- Real-time visual feedback
- Entry/exit line visualization

### Logging System
- Automatic log creation with timestamps
- Dual format logging (CSV and JSON)
- Detailed frame-by-frame statistics
- Easy data analysis and visualization

## Requirements

### Hardware Requirements
- Webcam or IP camera
- CUDA-capable GPU (recommended) or CPU
- Minimum 4GB RAM

### Software Requirements
```bash
# Core dependencies
torch>=2.0.0
opencv-python>=4.5.0
numpy>=1.19.0
Pillow>=8.0.0

# YOLO specific
ultralytics>=8.0.0

# DFine specific
transformers>=4.0.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd smart-room-monitor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models:
- YOLO model will be downloaded automatically on first run
- DFine model should be placed in the specified path

## Usage

### Running YOLO Monitor
```bash
python yolo_detection.py
```

### Running DFine Monitor
```bash
python dfine_detection.py
```

### Controls
- Press 'q' to quit the application
- The system will automatically save logs before closing

## Log Files

### Location
- Logs are stored in the `logs` directory
- Each session creates timestamped log files

### File Formats
1. CSV Logs:
   - Timestamp
   - Frame number
   - Total count
   - Standing count
   - Sitting count
   - Phone usage count

2. JSON Logs:
   - Detailed frame-by-frame data
   - Timestamp information
   - Behavior statistics

## Model Comparison

### YOLO Monitor
- Faster processing
- Better real-time performance
- More robust detection
- Suitable for general monitoring

### DFine Monitor
- More detailed detection
- Better accuracy for specific objects
- More precise pose estimation
- Suitable for detailed analysis

## Directory Structure
```
smart-room-monitor/
├── yolo_detection.py
├── dfine_detection.py
├── requirements.txt
├── README.md
├── models/
│   ├── yolo11s.pt
│   └── DFINE/
└── logs/
    ├── yolo_room_log_*.csv
    ├── yolo_room_log_*.json
    ├── dfine_room_log_*.csv
    └── dfine_room_log_*.json
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO model: [Ultralytics](https://github.com/ultralytics/yolov5)
- DFine model: [USTC Community](https://github.com/ustc-community/dfine)
- OpenCV for computer vision capabilities
