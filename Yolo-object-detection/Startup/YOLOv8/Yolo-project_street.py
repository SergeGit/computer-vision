from ultralytics import YOLO
import cv2
import cvzone
import math
import os
from sort import *

# Train model on YOLOv8 weights (.pt) - automatic downloading of standard weights
model = YOLO('../resources/weights/yolov8n.pt')  
classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
              'teddy bear', 'hair drier', 'toothbrush']
mask = cv2.imread('../resources/masks/mask_cars.png')

# Tracking of objects
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
# Counting of vehicles
vehicleCount = []
# Parking boundaries
limit_parking1 = [550,500,1280,800]
limit_parking2 = [270,500,470,720]
limit_parking3 = [100,435,800,385]
line_counter = [270, 600, 1280, 600]

# Set video capture
RTSP_URL = 'rtsp://admin:L260D0E2@192.168.5.91/cam/realmonitor?channel=1&subtype=0'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG) # IP camera stream
cap = cv2.VideoCapture('../resources/video/IMOU-20230810-200110.mp4') # For video

if not cap.isOpened():
    print('Cannot open webcam or video file')
    exit(-1)

# Classify with trained model
while True:
    _, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(img, stream=True) # can add mask for results
    detections = np.empty((0, 5))


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0] 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1 
            bbox = int(x1), int(y1), int(w), int(h)
            
            # confidence calculation
            conf = math.ceil(box.conf[0]*100)/100
            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            # pedestrian detection
            if currentClass == "person" or currentClass == "cat" or currentClass == "dog" and conf > 0.3: 
                cvzone.cornerRect(img, bbox, l=9)
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), 
                               scale=0.6, thickness=1, offset=3)
                
            # vehicle detection
            if currentClass == "car" or currentClass == "truck" or currentClass == "motorcycle" or currentClass == "bicycle" and conf > 0.3: 
                # cvzone.cornerRect(img, bbox, l=9, rt=5)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), 
                #                scale=0.6, thickness=1, offset=3)

                cx, cy = x1+w//2,  y1+h//2
                cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)
                
                # parked vehicles not taken into account
                if (cy < ((limit_parking1[3]-limit_parking1[1])/(limit_parking1[2]-limit_parking1[0])*cx +274.5)) or (cy > ((limit_parking2[3]-limit_parking2[1])/(limit_parking2[2]-limit_parking2[0])*cx +203)):
                    pass
                    # cvzone.cornerRect(img, bbox, l=9, rt=5)
                    # cvzone.putTextRect(img, 'parked', (max(0, x1), max(35, y1)), 
                    #                    scale=0.6, thickness=1, offset=3)   
                elif cy < limit_parking3[1]:
                    pass        
                else:
                    # not parking - in bounds
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections, currentArray))

                    if line_counter[1] - 20 < cy < line_counter[1] + 20:
                        if vehicleCount.count(id)==0:
                            vehicleCount.append(id)
                    
                
    resultsTracker = tracker.update(detections)
    
    cvzone.putTextRect(img, f'Vehicle count: {len(vehicleCount)}', (50,50))
    cv2.line(img,(limit_parking1[0], limit_parking1[1]), (limit_parking1[2], limit_parking1[3]), (0,0,255), 2)
    cv2.line(img,(limit_parking2[0], limit_parking2[1]), (limit_parking2[2], limit_parking2[3]), (0,0,255), 2)
    cv2.line(img,(limit_parking3[0], limit_parking3[1]), (limit_parking3[2], limit_parking3[3]), (0,0,255), 2)
    cv2.line(img,(line_counter[0], line_counter[1]), (line_counter[2], line_counter[3]), (0,255,255), 2)

    # Create box for each tracked car
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), 
                               scale=2, thickness=3, offset=10)

    cv2.imshow('VideoCapture', img)

    if cv2.waitKey(1) == 27: # Press ESC to exit/close each window.
        break

cap.release()
cv2.destroyAllWindows()