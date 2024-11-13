import cv2
import pandas as pd
import numpy as np

from frame_manager import *

frame_number = 1
sequence_number = 1

frame_1 = get_frame(frame_number, sequence_number, 2)
frame_2 = get_frame(frame_number, sequence_number, 3)
bboxes = get_detection_results(frame_number, sequence_number)

frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create()
kp1, des = sift.detectAndCompute(frame_1_gray, None)
kp2, des2 = sift.detectAndCompute(frame_2_gray, None)


def filter_features(frame, bboxes, features, descriptors):
    
    bboxes_mask = np.zeros(frame.shape)

    for bbox in bboxes:
        # 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom'
        bbox_left, bbox_top = int(row["bbox_left"]), int(row["bbox_top"])
        bbox_right, bbox_bottom = int(row["bbox_right"]), int(row["bbox_bottom"])
        obj_type = row["type"]

filter_features(frame_1, bboxes, kp1, des)



