import cv2
import numpy as np
import random
from frame_manager import *
import time


K = 2

class TrackedObject:
    def __init__(self, type, position, bbox, features, descriptors, color):
        self.type = type
        self.position = position
        self.bbox = bbox
        self.features = features # list of features in basic version len of 2 to store 
        self.descriptors = descriptors
        # self.k_features = k_features
        self.color = color
        # previous features

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

def filter_features(frame, detection_output, features, descriptors, object_container):
    minimal_object_features = 15
    matching_quality = 0.75 # default = 0.75
    bbox_feature_roi = [1, 1] # what part of the width and height aroung the center of the bounding box has to be considered

    bboxes = detection_output[0]

    # Sort feature-descriptor pairs by response in descending order
    feature_descriptor_pairs = sorted(zip(features, descriptors), key=lambda x: x[0].response, reverse=True)

    for bbox in bboxes:
        bbox_left, bbox_top, bbox_right, bbox_bottom = map(int, bbox)
        bbox_width = bbox_right - bbox_left
        bbox_height = bbox_bottom - bbox_top
        bbox_features = []
        bbox_descriptors = []

        left_roi = -bbox_width * bbox_feature_roi[0] / 2 + bbox_left + 0.5 * bbox_width
        right_roi = bbox_width * bbox_feature_roi[0] / 2 + bbox_right - 0.5 * bbox_width
        top_roi = -bbox_height * bbox_feature_roi[1] / 2 + bbox_top + 0.5 * bbox_height
        bottom_roi = bbox_height * bbox_feature_roi[1] / 2 + bbox_bottom - 0.5 * bbox_height

        # Separate features inside and outside of the bounding box
        for feature, descriptor in feature_descriptor_pairs:
            x, y = feature.pt
            if left_roi < x < right_roi and top_roi < y < bottom_roi:
                bbox_features.append(feature)
                bbox_descriptors.append(descriptor)

        # Skip processing if there are no descriptors in the bounding box
        if not bbox_descriptors:
            continue

        # Convert bbox_descriptors to a numpy array
        bbox_descriptors = np.array(bbox_descriptors)

        # State 1: If no objects exist, create the first one
        if not object_container:
            position = calculate_position(bbox_left, bbox_top, bbox_right, bbox_bottom)
            color = tuple(random.randint(0, 255) for _ in range(3))
            object_container.append(TrackedObject(detection_output[1], position, bbox, bbox_features, bbox_descriptors, color))
            continue

        # State 2: Try to match with existing objects
        matched_any = False
        bf = cv2.BFMatcher()
        for obj in object_container:
            object_descriptors = obj.descriptors

            # Ensure descriptors are numpy arrays and not empty
            if isinstance(object_descriptors, list):
                object_descriptors = np.array(object_descriptors)

            if object_descriptors is not None and len(object_descriptors) > 0:
                # Perform knnMatch
                matches = bf.knnMatch(object_descriptors, bbox_descriptors, k=2)

                # Apply Lowe's ratio test to find good matches
                matched_features = []
                matched_descriptors = []
                for m, n in matches:
                    if m.distance < matching_quality * n.distance:
                        matched_features.append(bbox_features[m.trainIdx])
                        matched_descriptors.append(bbox_descriptors[m.trainIdx])

                # If matches were found, update the object and mark as matched
                if len(matched_features) > minimal_object_features:
                    obj.features = matched_features
                    obj.descriptors = matched_descriptors
                    matched_any = True
                    break

        # State 3: No matches found with any existing objects, so create a new one
        if not matched_any and len(bbox_descriptors) > minimal_object_features:
            position = calculate_position(bbox_left, bbox_top, bbox_right, bbox_bottom)
            color = tuple(random.randint(0, 255) for _ in range(3))
            object_container.append(TrackedObject(detection_output[1], position, bbox, bbox_features, bbox_descriptors, color))

    return object_container


def get_masked_image(frame, detection_output):
    bboxes = detection_output[0]

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for bbox in bboxes:
        [bbox_left, bbox_top, bbox_right, bbox_bottom] = bbox
        bbox_left, bbox_top = int(bbox_left), int(bbox_top)
        bbox_right, bbox_bottom = int(bbox_right), int(bbox_bottom)

        mask[bbox_top:bbox_bottom, bbox_left:bbox_right] = 255
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
    return masked_frame

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

object_container = []

# Loop through all frames in the sequence
for frame_number in range(frame_start, frame_end + 1):
    start = time.time()
    frame_1 = get_frame(frame_number, sequence_number, 2)
    # frame_2 = get_frame(frame_number, sequence_number, 3)
    detection_output = get_detection_results(frame_number, sequence_number)
    
    masked_frame_1 = get_masked_image(frame_1, detection_output)

    frame_1_gray = cv2.cvtColor(masked_frame_1, cv2.COLOR_BGR2GRAY)
    # frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Detect features in both frames
    kp1, des = sift.detectAndCompute(frame_1_gray, None)
    # kp2, des2 = sift.detectAndCompute(frame_2_gray, None)

    # Filter features based on detection output
    tracked_objects = filter_features(frame_1, detection_output, kp1, des, object_container)

    # print(f"Tracked objects in frame {frame_number}: {len(tracked_objects)}")

    # # Visualize tracked objects and all features
    frame_with_objects, all_features_frame = visualize_objects(frame_1, tracked_objects)

    time_diff = time.time() - start
    print(f"time: {time_diff}")

    # Show frames
    cv2.namedWindow("Frame with Masked Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame with Masked Image", masked_frame_1)

    cv2.namedWindow("Frame with Tracked Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame with Tracked Objects", frame_with_objects)

    cv2.namedWindow("All Features", cv2.WINDOW_NORMAL)
    cv2.imshow("All Features", all_features_frame)

    # Wait for a key press for a short period to create a video effect
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Close all windows
cv2.destroyAllWindows()


