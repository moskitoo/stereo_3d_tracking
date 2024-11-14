import cv2
import numpy as np
import random
from frame_manager import *

class TrackedObject:
    def __init__(self, type, position, bbox, features, color):
        self.type = type
        self.position = position
        self.bbox = bbox
        self.features = features
        self.color = color

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)


def filter_features(frame, detection_output, features, descriptors):
    object_container = []
    bboxes = detection_output[0]

    features_sorted = sorted(features, key=lambda k: k.response, reverse=True)

    for bbox in bboxes:
        [bbox_left, bbox_top, bbox_right, bbox_bottom] = bbox
        bbox_left, bbox_top = int(bbox_left), int(bbox_top)
        bbox_right, bbox_bottom = int(bbox_right), int(bbox_bottom)

        object_features = []

        for feature in features_sorted:
            x, y = feature.pt[0], feature.pt[1]

            if x > bbox_left and x < bbox_right and y > bbox_top and y < bbox_bottom:
                object_features.append(feature)
                features_sorted = features_sorted[1:]

                # features have to be filtered to be sure that they represent only objects not background!
                # its quite often that bounding box is bigger than the object and a lot of background features are passed

        position = calculate_position(bbox_left, bbox_top, bbox_right, bbox_bottom)
        
        color = tuple(random.randint(0, 255) for _ in range(3))

        object_container.append(TrackedObject(detection_output[1], position, bbox, object_features, color))
    
    return object_container

def calculate_position(bbox_left, bbox_top, bbox_right, bbox_bottom):
    x = (bbox_left + bbox_right) / 2
    y = (bbox_top + bbox_bottom) / 2
    position = (x, y)
    return position


def visualize_objects(frame, tracked_objects):
    frame_copy = frame.copy()

    # Display features for each object in its unique color
    for obj in tracked_objects:
        for feature in obj.features:
            x, y = feature.pt
            cv2.circle(frame_copy, (int(x), int(y)), 5, obj.color, -1)  # Draw the feature points

    # Display all features in a separate window
    frame_features = frame.copy()
    for feature in kp1:
        x, y = feature.pt
        cv2.circle(frame_features, (int(x), int(y)), 5, (0, 255, 0), -1)  # All features in green

    return frame_copy, frame_features


# frame_number = 10
# sequence_number = 1

# frame_1 = get_frame(frame_number, sequence_number, 2)
# frame_2 = get_frame(frame_number, sequence_number, 3)
# detection_output = get_detection_results(frame_number, sequence_number)

# frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
# frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

# sift = cv2.SIFT_create()
# kp1, des = sift.detectAndCompute(frame_1_gray, None)
# kp2, des2 = sift.detectAndCompute(frame_2_gray, None)

# tracked_objects = filter_features(frame_1, detection_output, kp1, des)

# print(f"tracked objects: {len(tracked_objects)}")

# frame_with_objects, all_features_frame = visualize_objects(frame_1, tracked_objects)

# # Show frames
# cv2.namedWindow("Frame with Tracked Objects", cv2.WINDOW_NORMAL)
# cv2.imshow("Frame with Tracked Objects", frame_with_objects)

# cv2.namedWindow("All Features", cv2.WINDOW_NORMAL)
# cv2.imshow("All Features", all_features_frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


frame_start = 1  # Start frame number
frame_end = 140  # End frame number
sequence_number = 1  # Sequence number

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Loop through all frames in the sequence
for frame_number in range(frame_start, frame_end + 1):
    frame_1 = get_frame(frame_number, sequence_number, 2)
    frame_2 = get_frame(frame_number, sequence_number, 3)
    detection_output = get_detection_results(frame_number, sequence_number)

    frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Detect features in both frames
    kp1, des = sift.detectAndCompute(frame_1_gray, None)
    kp2, des2 = sift.detectAndCompute(frame_2_gray, None)

    # Filter features based on detection output
    tracked_objects = filter_features(frame_1, detection_output, kp1, des)

    print(f"Tracked objects in frame {frame_number}: {len(tracked_objects)}")

    # Visualize tracked objects and all features
    frame_with_objects, all_features_frame = visualize_objects(frame_1, tracked_objects)

    # Show frames
    cv2.namedWindow("Frame with Tracked Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame with Tracked Objects", frame_with_objects)

    cv2.namedWindow("All Features", cv2.WINDOW_NORMAL)
    cv2.imshow("All Features", all_features_frame)

    # Wait for a key press for a short period to create a video effect
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Close all windows
cv2.destroyAllWindows()


