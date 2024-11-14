import cv2
import pandas as pd
import numpy as np

from frame_manager import *

frame_number = 50
sequence_number = 1

frame_1 = get_frame(frame_number, sequence_number, 2)
frame_2 = get_frame(frame_number, sequence_number, 3)
detection_output = get_detection_results(frame_number, sequence_number)

frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create()
kp1, des = sift.detectAndCompute(frame_1_gray, None)
kp2, des2 = sift.detectAndCompute(frame_2_gray, None)


object_container = []

'''
object

type
pedestrian
car
cyclist

keypoints

2d position

'''
class TrackedObject:
    def __init__(self, type, position, bbox, features):
        self.type = type
        self.position = position
        self.bbox = bbox
        self.features = features

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)
        




def filter_features(frame, detection_output, features, descriptors):
    bboxes = detection_output[0]
    
    bboxes_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    features_sorted = sorted(features, key=lambda k: k.response, reverse=True)

    for bbox in bboxes:

        # maybe sort detections by score and then remove features?

        [bbox_left, bbox_top, bbox_right, bbox_bottom] = bbox
        bbox_left, bbox_top = int(bbox_left), int(bbox_top)
        bbox_right, bbox_bottom = int(bbox_right), int(bbox_bottom)

        object_features = []

        for feature in features_sorted:
            x, y = feature.pt[0], feature.pt[1]

            if x > bbox_top and x < bbox_bottom and y > bbox_top and y < bbox_right:
                object_features.append(feature)
                features_sorted = features_sorted[1:]

        position = calculate_position(bbox_left, bbox_top, bbox_right, bbox_bottom)

        object_container.append(TrackedObject(detection_output[1], position, bbox, object_features))
    
    return object_container

def calculate_position(bbox_left, bbox_top, bbox_right, bbox_bottom):
    x = (bbox_left + bbox_right) / 2
    y = (bbox_top + bbox_bottom) / 2
    position = (x, y)
    return position
    



    

tracked_objects = filter_features(frame_1, detection_output, kp1, des)

print(f"tracked objects: {len(tracked_objects)}")

visualize_frame(frame_number, sequence_number, 2)

# cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
# cv2.imshow("Video", bboxes_mask * 255)  # Multiply by 255 to visualize as white (255) on black (0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



