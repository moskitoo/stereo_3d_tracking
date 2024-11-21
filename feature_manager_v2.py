import cv2
import numpy as np
import random
from frame_manager import *
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


id_counter = 0

class TrackedObject:
    def __init__(self, type, position, bbox, features, color, id):
        self.type = type
        self.position = [position]
        self.bbox = bbox
        self.features = features
        self.color = color
        self.id = id


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
        cv2.putText(frame_copy, str(obj.id), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return frame_copy

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

        detected_objects.append(TrackedObject(obj_type, bbox_obj.position, bbox_obj, features, get_rand_color(), -1))

    return detected_objects

def get_cost_matrix(detected_objects, object_container, pos_w=0.4, bbox_area_w=0.3, bbox_shape_w=0.2, feat_w=0.1):

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
            cost_matrix[i, j] = pos_w * pos_cost + bbox_area_w * bbox_area_cost + bbox_shape_w * shape_cost + feat_w * feat_cost + class_cost
    
    return cost_matrix

def match_objects(detected_objects, object_container, pos_w=0.4, bbox_area_w=0.3, bbox_shape_w=0.3, feat_w=0.1, cost_threshold=100.0):
    global id_counter

    # State 1: If no objects exist, create the first one
    if not object_container:
        for detected_object in detected_objects:
            detected_object.id = id_counter
            id_counter += 1
        object_container = detected_objects
    else:
        cost_matrix = get_cost_matrix(detected_objects, object_container, pos_w, bbox_area_w, bbox_shape_w, feat_w)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches, unmatched_tracked, unmatched_detected = filter_false_matches(detected_objects, object_container, cost_threshold, cost_matrix, row_indices, col_indices)
        
        # print(f"detected objects: {len(detected_objects)}")
        # print(f"tracked objects: {len(object_container)}")
        # print(f"cost_matrix: {cost_matrix.shape}")
        # print(f"Matches: {matches}")
        # print(f"unmatched_tracked: {unmatched_tracked}")
        # print(f"unmatched_detected: {unmatched_detected}")

        # print("\n\n")


        # matches = [row, col] -> row: tracked, col: detected

        # State 2: Match with existing objects
        for tracked_object_id, detect_object_id in matches:
            tracked_object = object_container[tracked_object_id]
            detected_object = detected_objects[detect_object_id]
            tracked_object.update_state(detected_object)
        
        # State 3: Remove non matched trakced objects
        # Convert unmatched_tracked to a set for faster lookup
        unmatched_tracked_set = set(unmatched_tracked)
        object_container = [tracked_object for id, tracked_object in enumerate(object_container) if id not in unmatched_tracked_set]

        # State 4: Add non matched detected objects to tracking
        for unmatched_detected_id in unmatched_detected:
            non_matched_object = detected_objects[unmatched_detected_id]
            non_matched_object.id = id_counter
            object_container.append(non_matched_object)
            id_counter += 1

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

def draw_bounding_boxes(frame, detection_output):
    bboxes = detection_output[0]

    for bbox in bboxes:
        # Extract bounding box coordinates
        [bbox_left, bbox_top, bbox_right, bbox_bottom] = bbox
        bbox_left, bbox_top = int(bbox_left), int(bbox_top)
        bbox_right, bbox_bottom = int(bbox_right), int(bbox_bottom)

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bottom), color=(0, 255, 0), thickness=2)

    return frame

def combine_frames(frames):

    # Split the list of frames into two columns
    mid_index = (len(frames) + 1) // 2  # Handle odd number of frames
    column1 = frames[:mid_index]
    column2 = frames[mid_index:]
    
    # Add blank frames to the shorter column if necessary
    frame_height, frame_width = frames[0].shape[:2]
    blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    if len(column1) > len(column2):
        column2.append(blank_frame)
    
    # Stack frames vertically in each column
    column1_combined = cv2.vconcat(column1)
    column2_combined = cv2.vconcat(column2)
    
    # Combine the two columns horizontally
    combined_frame = cv2.hconcat([column1_combined, column2_combined])
    return combined_frame

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

        detection_output = get_detection_results(frame_number, sequence_number)

        detected_objects = detect_objects(frame_1, detection_output)

        object_container = match_objects(detected_objects, object_container)

        frame_with_objects = visualize_objects(
            frame_1, object_container
        )

        time_diff = time.time() - start
        # print(f"time: {time_diff} s")

        frame_counter += 1

        masked_frame_1 = get_masked_image(frame_1, detection_output)
        bbox_frame = draw_bounding_boxes(frame_1, detection_output)
        
        # Show frames
        # cv2.namedWindow("Frame with Tracked Objects", cv2.WINDOW_NORMAL)
        # cv2.imshow("Frame with Tracked Objects", frame_with_objects)

        # cv2.namedWindow("Frame with Masked Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Frame with Masked Image", masked_frame_1)

        # cv2.namedWindow("Frame with Bboxes", cv2.WINDOW_NORMAL)
        # cv2.imshow("Frame with Bboxes", bbox_frame)

        combined_frames = combine_frames([frame_with_objects, masked_frame_1, bbox_frame])
        cv2.namedWindow("Frame with combined_frames", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame with combined_frames", combined_frames)

        # Wait for a key press for a short period to create a video effect
        if cv2.waitKey(0) & 0xFF == ord("q"):  # Press 'q' to quit
            break
    frame_1_prev = get_frame(frame_number, sequence_number, 2)

# Close all windows
cv2.destroyAllWindows()

print(object_container)
