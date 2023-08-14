from ultralytics import YOLO
import cv2

from sort import *
from functions.util import get_car, read_license_plate

# load models
coco_model = YOLO('../resources/weights/yolov8n.pt')
classNames_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
              'teddy bear', 'hair drier', 'toothbrush']
vehicles = [2, 3, 5, 7,]

license_plate_detector = YOLO('../resources/weights/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('../resources/video/cars.mp4')

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        pass
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = [] # bounding boxes detected vehicles
        for detection in detections.boxes.data.toList():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles: 
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles (using sorting function)
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.toList():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # read license plate number
            util.read_license_plate()

            # https://youtu.be/fyJB1t0o0ms?list=PLFX5gC6vx_Ht9_ooCPSIqprgBufKuUs4y&t=1961
            # https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8
            # https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide
            # https://broutonlab.com/blog/opencv-object-tracking


            # write results