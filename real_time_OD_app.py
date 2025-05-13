import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import sys

# Model paths
YOLO_MODEL_PATH = 'models/yolo11x.pt'
DFINE_MODEL_PATH = '/home/ibrahim/Documents/Study/Computer Vision/Project/models/DFINE/models--ustc-community--dfine-xlarge-obj2coco/snapshots/15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d'

def process_image_yolo(image, yolo_model):
    results = yolo_model(image, conf=0.5, device="cpu")
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names[cls]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def process_image_dfine(image, dfine_model, dfine_processor, device):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = dfine_processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dfine_model(**inputs)
    results = dfine_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([pil_image.size[::-1]]),
        threshold=0.3
    )
    image_np = np.array(pil_image)
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = dfine_model.config.id2label[label_id.item()]
            box = [int(i) for i in box.tolist()]
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(image_np, label_text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def main():
    print("Select model for real-time object detection:")
    print("1. YOLO")
    print("2. DFine")
    choice = input("Enter 1 or 2: ").strip()
    if choice == '1':
        model_type = 'YOLO'
    elif choice == '2':
        model_type = 'DFine'
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if model_type == 'YOLO':
        print("Loading YOLO model...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        dfine_model = None
        dfine_processor = None
    else:
        print("Loading DFine model...")
        dfine_processor = AutoImageProcessor.from_pretrained(DFINE_MODEL_PATH)
        dfine_model = DFineForObjectDetection.from_pretrained(DFINE_MODEL_PATH)
        dfine_model = dfine_model.to(device)
        yolo_model = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        if model_type == 'YOLO':
            processed = process_image_yolo(frame, yolo_model)
        else:
            processed = process_image_dfine(frame, dfine_model, dfine_processor, device)
        cv2.imshow("Real-Time Object Detection", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 