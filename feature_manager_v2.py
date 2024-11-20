import cv2
import numpy as np
import random
from frame_manager import *
import time


class TrackedObject:
    def __init__(self, type, position, bbox, features, color):
        self.type = type
        self.position = [position]
        self.bbox = bbox
        self.features = features
        self.color = color


    def __getattribute__(self, name):
        return object.__getattribute__(self, name)


    def update_state(self, features, descriptors, position, bbox):
        self.prv_features = self.features
        self.features = features
        self.descriptors = descriptors  # not used in this approach
        self.position.append(position)
        self.bbox = bbox

class BoundingBox:
    def __init__(self, bbox_left, bbox_top, bbox_right, bbox_bottom):
        self.left = bbox_left
        self.top = bbox_top
        self.right = bbox_right
        self.bottom = bbox_bottom

        self.width = self.right - self.left
        self.height = self.bottom - self.top

        self.get_bbox_centre()
    
    def get_bbox_centre(self):
        x = (self.left + self.right) / 2
        y = (self.top + self.bottom) / 2
        self.position = np.array((x, y))

    def get_bbox_img(self, frame):
        return frame[self.top:self.bottom, self.left:self.right]

def visualize_objects(frame, tracked_objects):
    frame_copy = frame.copy()

    # Display features for each object in its unique color
    for obj in tracked_objects:
        for feature in obj.features:
            try:
                x, y = feature
            except:
                x, y = feature.pt
            cv2.circle(
                frame_copy, (int(x), int(y)), 5, obj.color, -1
            )  # Draw the feature points

    # Display all features in a separate window
    frame_features = frame.copy()
    for feature in kp1:
        x, y = feature.pt
        cv2.circle(
            frame_features, (int(x), int(y)), 5, (0, 255, 0), -1
        )  # All features in green

    return frame_copy, frame_features

def get_rand_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def get_detection_results(frame_number, seq_num):
    seq_labels = labels[seq_num]
    frame_labels = seq_labels[seq_labels["frame"] == frame_number]

    # Extract columns 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom' as a NumPy array
    bboxes = frame_labels[['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']].to_numpy().astype(int)
    
    # Extract 'type' column as a NumPy array
    obj_types = frame_labels['type'].to_numpy()
    
    return bboxes, obj_types

def detect_objects(frame, detection_output):
    detected_objects = []
    for bbox, obj_type in zip(detection_output[0], detection_output[1]):
        bbox_obj = BoundingBox(*bbox)

        cropped_frame = bbox_obj.get_bbox_img(frame)
        frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(frame_gray, None)

        detected_objects.append(TrackedObject(obj_type, bbox_obj.position, bbox_obj, kp, get_rand_color()))

    return detected_objects



frame_start = 1  # Start frame number
frame_end = 140  # End frame number
sequence_number = 1  # Sequence number

# Initialize SIFT detector
sift = cv2.SIFT_create()

object_container = []

frame_1_prev = None

# Loop through all frames in the sequence
# while True:
frame_counter = 0
for frame_number in range(frame_start, frame_end + 1):
    if frame_1_prev is not None:
        start = time.time()


        frame_1 = get_frame(frame_number, sequence_number, 2)
        # frame_2 = get_frame(frame_number, sequence_number, 3)
        detection_output = get_detection_results(frame_number, sequence_number)

        detected_objects = detect_objects(frame_1, detection_output)




        frame_1_prev_gray = cv2.cvtColor(frame_1_prev, cv2.COLOR_BGR2GRAY)
        # frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        masked_frame_1 = get_masked_image(frame_1, detection_output)
        frame_1_gray = cv2.cvtColor(masked_frame_1, cv2.COLOR_BGR2GRAY)

        # Detect features in both frames
        kp1, des = sift.detectAndCompute(frame_1_gray, None)

        tracked_objects = filter_features_optical_flow(frame_1_gray, frame_1_prev_gray, detection_output, kp1, des, object_container)

        # # Visualize tracked objects and all features
        frame_with_objects, all_features_frame = visualize_objects(
            frame_1, tracked_objects
        )

        time_diff = time.time() - start
        # print(f"time: {time_diff}")

        frame_counter += 1
        

        # Show frames
        cv2.namedWindow("Frame with Masked Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame with Masked Image", masked_frame_1)

        cv2.namedWindow("Frame with Tracked Objects", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame with Tracked Objects", frame_with_objects)

        cv2.namedWindow("All Features", cv2.WINDOW_NORMAL)
        cv2.imshow("All Features", all_features_frame)

        # Wait for a key press for a short period to create a video effect
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break
    frame_1_prev = get_frame(frame_number, sequence_number, 2)

# Close all windows
cv2.destroyAllWindows()

print(object_container)
