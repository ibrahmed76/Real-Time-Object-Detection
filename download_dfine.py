import torch
import requests

from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco",cache_dir="/home/ibrahim/Documents/Study/Computer Vision/Project/models/DFINE")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco",cache_dir="/home/ibrahim/Documents/Study/Computer Vision/Project/models/DFINE")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
