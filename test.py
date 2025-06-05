import torch
from ultralytics import YOLO
# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Perform inference
results = model('./frames/frame_0.jpg')  # Replace with the path to one frame

# Print results
print(results.pandas().xyxy[0])  # Bounding boxes and confidence scores
