# Object Detection App

This project provides two main ways to perform object detection using YOLO or DFine models:

## 1. Gradio Web App

- **Folder Processing:**
  - Place your images and videos in the `images/input` directory.
  - Select the model (YOLO or DFine) and click "Process Folder".
  - Results will be saved in a `results` folder next to your input folder, under a subfolder for the selected model.

- **Single Image/Video Processing:**
  - Upload an image or video file.
  - Select the model (YOLO or DFine) and click "Process File".
  - The processed image or video will be displayed or the output file path will be shown.

- **Webcam:**
  - **Webcam detection is NOT included in the Gradio app.**
  - See below for the standalone webcam script.

### To Run the Gradio App
```bash
python object_detection_app.py
```
- The app will open in your browser at `http://127.0.0.1:7860`.

## 2. Real-Time Webcam Detection (Standalone Script)

- Run the script:
```bash
python real_time_OD_app.py
```
- You will be prompted to select a model (YOLO or DFine).
- The webcam window will open and show real-time detections.
- Press `q` to quit.

## Requirements
- Python 3.8+
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

1. Create the models directory:
```bash
mkdir models
```

2. Download YOLOv11x model:
   - Visit [YOLOv11 Documentation](https://docs.ultralytics.com/tasks/detect/)
   - Download the YOLOv11x model
   - Place the downloaded `yolo11x.pt` file in the `models` directory

3. Download DFine model:
   - Download the DFine model files as required (see HuggingFace or project instructions)
   - Place the DFine model files in the path specified in the scripts (see `DFINE_MODEL_PATH` in both scripts)

## Folder Structure
```
project_root/
├── models/
│   ├── yolo11x.pt
│   └── DFINE/... (DFine model files)
├── images/
│   └── input/   # Place your images and videos here
├── results/
│   └── ...      # Processed outputs will be saved here
├── object_detection_app.py
├── real_time_OD_app.py
├── requirements.txt
└── README.md
```

## Notes
- For best results, ensure your input images/videos are clear and well-lit.
- The DFine model may require more memory and time to process than YOLO.
- If you encounter issues with model paths, update the `YOLO_MODEL_PATH` and `DFINE_MODEL_PATH` variables in the scripts.

