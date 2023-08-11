import cv2
import os

RTSP_URL = 'rtsp://admin:L260D0E2@192.168.5.91/cam/realmonitor?channel=1&subtype=0'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    _, frame = cap.read()
    cv2.imshow('RTSP stream', frame)

    if cv2.waitKey(1) == 27: # Press ESC to exit/close each window.
        break

cap.release()
cv2.destroyAllWindows()