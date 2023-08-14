from ultralytics import YOLO
import cv2
import cvzone
import math
import os

# Train model on YOLOv8 weights (.pt) - automatic downloading of standard weights
model = YOLO('../resources/weights/yolov8n.pt')  

classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
              'teddy bear', 'hair drier', 'toothbrush']

# Set video capture
RTSP_URL = 'rtsp://admin:L260D0E2@192.168.5.91/cam/realmonitor?channel=1&subtype=0'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG) # IP camera stream

if not cap.isOpened():
    print('Cannot open webcam or video file')
    exit(-1)

# Classify with trained model
while True:
    _, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0] 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1 
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, bbox, l=9)
            # confidence calculation
            conf = math.ceil(box.conf[0]*100)/100
            # class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), 
                               scale=0.6, thickness=1, offset=3)

    cv2.imshow('VideoCapture', img)

    if cv2.waitKey(1) == 27: # Press ESC to exit/close each window.
        break

cap.release()
cv2.destroyAllWindows()