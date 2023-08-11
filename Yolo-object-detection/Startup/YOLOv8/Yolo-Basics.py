from ultralytics import YOLO
import cv2

# Train model on YOLOv8 weights (.pt) - automatic downloading of standard weights
model = YOLO('../resources/weights/yolov8n.pt') # nameslist: model.model.names 

# Test trained model on 
results = model("../resources/images/bus.jpg", show=True)
cv2.waitKey(0)