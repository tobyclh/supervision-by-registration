import time
import numpy as np
from skimage import transform, img_as_float
from dface.core.detect import MtcnnDetector, create_mtcnn_net
import os
from YFYF.Tracking import WebcamVideoStream
import torch
import cv2
data_path = os.getenv('YFYF_data')
join = lambda x : os.path.join(data_path, x)
pnet, rnet, onet = create_mtcnn_net(p_model_path=join("dface/model_store/pnet_epoch.pt"), r_model_path=join("dface/model_store/rnet_epoch.pt"), o_model_path=join("dface/model_store/onet_epoch.pt"), use_cuda=True)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

stream = WebcamVideoStream()
stream.start()
while True:
    frame, dirty = stream.read()
    if not dirty:
        continue
    rects, _ = mtcnn_detector.detect_face(frame)
    print(rects)
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


