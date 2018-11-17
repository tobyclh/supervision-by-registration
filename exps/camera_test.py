import time
import numpy as np
from skimage import transform, img_as_float
from YFYF.Alignment import Detector
from YFYF.Tracking import WebcamVideoStream
import torch
import cv2


detector = Detector(backend='dface')
stream = WebcamVideoStream()
stream.start()
while True:
    frame, dirty = stream.read()
    if not dirty:
        continue
    print(detector.detect([frame]))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


