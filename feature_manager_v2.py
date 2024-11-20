import cv2
import numpy as np
import random
from frame_manager import *
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


class TrackedObject:
    def __init__(self, type, position, bbox, features, color):
        self.type = type
        self.position = [position]
        self.bbox = bbox
        self.features = features
        self.color = color


    def __getattribute__(self, name):
        return object.__getattribute__(self, name)


    def update_state(self, detected_object):
        # self.prv_features = self.features
        self.position.append(detected_object.position[0])
        self.bbox = detected_object.bbox
        self.features = detected_object.features
        # self.descriptors = descriptors  # not used in this approach

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
        self.position = np.array((x, y)).astype(int)

    def get_bbox_img(self, frame):
        return frame[self.top:self.bottom, self.left:self.right]
    
    def get_bbox_area(self):
        return self.width * self.height
    
    def get_bbox_aspect_ratio(self):
        return self.width / self.height

def visualize_objects(frame, tracked_objects):
    frame_copy = frame.copy()

    # Display features for each object in its unique color
    for obj in tracked_objects:
        x = obj.position[-1][0]
        y = obj.position[-1][1]
        cv2.circle(
            frame_copy, (x, y), 10, obj.color, -1
        )

        # print(f"points: x:{x}, y: {y}")
        # for feature in obj.features:
        #     try:
        #         x, y = feature
        #     except:
        #         x, y = feature.pt
        #     cv2.circle(
        #         frame_copy, (int(x), int(y)), 10, obj.color, -1
        #     )  # Draw the feature points

    # Display all features in a separate window
    # frame_features = frame.copy()
    # for feature in kp1:
    #     x, y = feature.pt
    #     cv2.circle(
    #         frame_features, (int(x), int(y)), 5, (0, 255, 0), -1
    #     )  # All features in green

    return frame_copy#, frame_features

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

        if des is None:
            features = np.zeros(128)
        else:
            features = np.mean(des, axis=0)

        detected_objects.append(TrackedObject(obj_type, bbox_obj.position, bbox_obj, features, get_rand_color()))

    return detected_objects

def get_cost_matrix(detected_objects, object_container, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):

    num_tracked = len(object_container)
    num_detections = len(detected_objects)
    cost_matrix = np.zeros((num_tracked, num_detections))

    for i, tracked_object in enumerate(object_container):
        for j, detected_object in enumerate(detected_objects):
            # Position cost
            pos_cost = np.linalg.norm(tracked_object.position[0] - detected_object.position[0])
            
            # Bbox area cost
            det_obj_area = detected_object.bbox.get_bbox_area()
            tracked_obj_area = tracked_object.bbox.get_bbox_area()
            bbox_area_cost = abs(tracked_obj_area - det_obj_area) / max(det_obj_area, tracked_obj_area)

            # Bbox shape cost
            shape_cost = abs(tracked_object.bbox.get_bbox_aspect_ratio() - detected_object.bbox.get_bbox_aspect_ratio())

            # Feature cost
            feat_cost = 1 - cosine_similarity([tracked_object.features], [detected_object.features])[0, 0]

            # Class cost
            class_cost = 0 if tracked_object.type == detected_object.type else 100

            # Total cost
            cost_matrix[i, j] = alpha * pos_cost + beta * bbox_area_cost + gamma * shape_cost + delta * feat_cost + class_cost
    
    return cost_matrix

def match_objects(detected_objects, object_container, alpha=0.4, beta=0.3, gamma=0.3, delta=0.1, cost_threshold=1000.0):

    # State 1: If no objects exist, create the first one
    if not object_container:
        object_container = detected_objects
    else:
        cost_matrix = get_cost_matrix(detected_objects, object_container, alpha, beta, gamma, delta)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches, unmatched_tracked, unmatched_detected = filter_false_matches(detected_objects, object_container, cost_threshold, cost_matrix, row_indices, col_indices)
        
        print(f"detected objects: {len(detected_objects)}")
        print(f"tracked objects: {len(object_container)}")
        print(f"cost_matrix: {cost_matrix.shape}")
        print(f"Matches: {matches}")
        print(f"unmatched_tracked: {unmatched_tracked}")
        print(f"unmatched_detected: {unmatched_detected}")

        print("\n\n")


        # matches = [row, col] -> row: tracked, col: detected

        # State 2: Match with existing objects
        for tracked_object_id, detect_object_id in matches:
            tracked_object = object_container[tracked_object_id]
            detected_object = detected_objects[detect_object_id]
            tracked_object.update_state(detected_object)
        
        # State 3: Remove non matched trakced objects
        for unmatched_tracked_id in unmatched_tracked:
            del object_container[unmatched_tracked_id]

        # State 4: Add non matched detected objects to tracking
        for unmatched_detected_id in unmatched_detected:
            object_container.append(detected_objects[unmatched_detected_id])

    return object_container

def filter_false_matches(detected_objects, object_container, cost_threshold, cost_matrix, row_indices, col_indices):
    matches = []
    unmatched_tracked = set(range(len(object_container)))
    unmatched_detected = set(range(len(detected_objects)))
        
    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] < cost_threshold:
            matches.append((row, col))
            unmatched_tracked.discard(row)
            unmatched_detected.discard(col)
        
    unmatched_tracked = list(unmatched_tracked)
    unmatched_detected = list(unmatched_detected)
    return matches, unmatched_tracked, unmatched_detected

frame_start = 1  # Start frame number
frame_end = 140  # End frame number
sequence_number = 1  # Sequence number

# Initialize SIFT detector
sift = cv2.SIFT_create()

object_container = None

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

        object_container = match_objects(detected_objects, object_container)



        # # Visualize tracked objects and all features
        # frame_with_objects, all_features_frame = visualize_objects(
        #     frame_1, object_container
        # )
        frame_with_objects = visualize_objects(
            frame_1, object_container
        )

        time_diff = time.time() - start
        # print(f"time: {time_diff}")

        frame_counter += 1
        
        # Show frames
        # cv2.namedWindow("Frame with Masked Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Frame with Masked Image", masked_frame_1)

        cv2.namedWindow("Frame with Tracked Objects", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame with Tracked Objects", frame_with_objects)

        # cv2.namedWindow("All Features", cv2.WINDOW_NORMAL)
        # cv2.imshow("All Features", all_features_frame)

        # Wait for a key press for a short period to create a video effect
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break
    frame_1_prev = get_frame(frame_number, sequence_number, 2)

# Close all windows
cv2.destroyAllWindows()

print(object_container)
