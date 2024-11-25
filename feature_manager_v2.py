import cv2
import numpy as np
import random
from frame_manager import *
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import pandas as pd
from datetime import datetime
import torch


id_counter = 0
current_frame_number = 0

cost_matrix_storage = []
SAVE_DIR = "cost_matrices"

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
        if type(bbox_left) == torch.Tensor:
            self.left = int(bbox_left.item())
            self.top = int(bbox_top.item())
            self.right = int(bbox_right.item())
            self.bottom = int(bbox_bottom.item())
        else:
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

        print(frame.shape)
        print(self.top)
        print(self.bottom)
        print(self.left)
        print(self.right)

        return frame[self.top:self.bottom, self.left:self.right]
    
    def get_bbox_area(self):
        return self.width * self.height
    
    def get_bbox_aspect_ratio(self):
        return self.width / self.height

def visualize_objects(frame, tracked_objects):
    frame_copy = frame.copy()

    counter = 0
    # Display features for each object in its unique color
    for obj in tracked_objects:
        x = obj.position[-1][0]
        y = obj.position[-1][1]
        cv2.circle(
            frame_copy, (x, y), 10, obj.color, -1
        )
        cv2.putText(frame_copy, str(obj.id), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        counter += 1
    print(counter)
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
    sift = cv2.SIFT_create()
    detected_objects = []
    for id, [bbox, obj_type] in enumerate(zip(detection_output[0], detection_output[1])):
        bbox_obj = BoundingBox(*bbox)

        cropped_frame = bbox_obj.get_bbox_img(frame)
        frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(frame_gray, None)

        if des is None:
            features = np.zeros(128)
        else:
            features = np.mean(des, axis=0)

        detected_objects.append(TrackedObject(obj_type, bbox_obj.position, bbox_obj, features, get_rand_color(), id))

    return detected_objects

def detect_objects_yolo(frame, detection_output):
    sift = cv2.SIFT_create()
    detected_objects = []
    print(type(detection_output))
    for id, [bbox, obj_type] in enumerate(zip(detection_output.boxes.xyxy, detection_output.boxes.cls)):
        bbox_obj = BoundingBox(*bbox)

        cropped_frame = bbox_obj.get_bbox_img(frame)

        frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(frame_gray, None)

        if des is None:
            features = np.zeros(128)
        else:
            features = np.mean(des, axis=0)

        detected_objects.append(TrackedObject(obj_type, bbox_obj.position, bbox_obj, features, get_rand_color(), id))

    return detected_objects

def get_cost_matrix(detected_objects, object_container, pos_w=0.4, bbox_area_w=0.3, bbox_shape_w=0.2, feat_w=0.1):
    num_tracked = len(object_container)
    num_detections = len(detected_objects)
    cost_matrix = np.zeros((num_tracked, num_detections))
    
    # Initialize detailed cost matrix with shape (num_tracked, num_detections, 6)
    # Added extra dimension for total cost
    cost_matrix_detailed = np.zeros((num_tracked, num_detections, 6))
    
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
            
            # Calculate total cost
            total_cost = pos_w * pos_cost + bbox_area_w * bbox_area_cost + bbox_shape_w * shape_cost + feat_w * feat_cost + class_cost
            cost_matrix[i, j] = total_cost
            
            # Store detailed costs including total cost
            detailed_cost = [pos_cost, bbox_area_cost, shape_cost, feat_cost, class_cost, total_cost]
            cost_matrix_detailed[i, j] = [round(x,2) for x in detailed_cost]
    
    row_ids = [obj.id for obj in object_container]
    column_ids = [obj.id for obj in detected_objects]
    pd.options.display.float_format = '{:,.2f}'.format
    
    # Print detailed costs with total cost included
    # print(pd.DataFrame(cost_matrix_detailed.reshape(num_tracked, num_detections * 6), 
    #                   index=row_ids, 
    #                   columns=column_ids * 6))

    cost_matrix_storage.append({
        'frame': current_frame_number,  # You'll need to make current_frame_number accessible
        'matrix': cost_matrix_detailed,
        'tracked_ids': row_ids,
        'detected_ids': column_ids
    })


    print(pd.DataFrame(cost_matrix, index=row_ids, columns=column_ids))
    
    return cost_matrix

def match_objects(detected_objects, object_container, pos_w=0.6, bbox_area_w=0.3, bbox_shape_w=0.3, feat_w=0.1, cost_threshold=100.0):
    global id_counter

    # State 1: If no objects exist, create the first one
    if not object_container:
        for detected_object in detected_objects:
            detected_object.id = id_counter
            id_counter += 1
        object_container = detected_objects

        detected_objects_ids = [x.id for x in detected_objects]
        print(f"added objects: {detected_objects_ids}")

        return object_container, []
    else:
        cost_matrix = get_cost_matrix(detected_objects, object_container, pos_w, bbox_area_w, bbox_shape_w, feat_w)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches, unmatched_tracked, unmatched_detected = filter_false_matches(detected_objects, object_container, cost_threshold, cost_matrix, row_indices, col_indices)
        
        # print(f"detected objects: {len(detected_objects)}")
        # print(f"tracked objects: {len(object_container)}")
        # print(f"cost_matrix: {cost_matrix.shape}")
        print(f"Matches:         {matches}")
        # print(f"unmatched_tracked: {unmatched_tracked}")
        # print(f"unmatched_detected: {unmatched_detected}")
        matches_decoded = [(object_container[tr_id].id, detected_objects[det_id].id) for (tr_id, det_id) in matches]
        # print("\n\n")
        print(f"Matches decoded: {matches_decoded}")

        # State 2: Match with existing objects
        for tracked_object_id, detect_object_id in matches:
            tracked_object = object_container[tracked_object_id]
            detected_object = detected_objects[detect_object_id]
            tracked_object.update_state(detected_object)
        
        # State 3: Remove non matched trakced objects
        # Convert unmatched_tracked to a set for faster lookup
        unmatched_tracked_set = set(unmatched_tracked)
        print(f"unmatched tracked objects: {unmatched_tracked}")
        object_container = [tracked_object for id, tracked_object in enumerate(object_container) if id not in unmatched_tracked_set]

        # State 4: Add non matched detected objects to tracking
        unmatched_detected_ids = []
        for unmatched_detected_id in unmatched_detected:
            non_matched_object = detected_objects[unmatched_detected_id]
            # non_matched_object.id = id_counter
            new_tracked_object = TrackedObject(
                non_matched_object.type,
                non_matched_object.position[0],
                non_matched_object.bbox,
                non_matched_object.features,
                non_matched_object.color,
                id_counter  # New ID for tracked object
            )
            object_container.append(new_tracked_object)
            unmatched_detected_ids.append(id_counter)
            id_counter += 1

        print(f"unmatched detected objects: {unmatched_detected_ids}")
        
        return object_container, matches

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

def visualize_matched_objects(frame, tracked_objects, detected_objects, matches):
    # Create a frame twice the height of the original
    split_frame = np.zeros((frame.shape[0] * 2, frame.shape[1], 3), dtype=np.uint8)
    
    # Copy the original frame to the top half
    split_frame[:frame.shape[0], :, :] = frame.copy()
    split_frame[frame.shape[0]:, :, :] = frame.copy()
    
    # Top frame - tracked objects
    for obj in tracked_objects: 
        x = obj.position[-1][0]
        y = obj.position[-1][1]
        cv2.circle(
            split_frame, (x, y), 10, obj.color, -1
        )
        cv2.putText(split_frame, str(obj.id), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Bottom frame - detected objects
    for obj in detected_objects:
        x = obj.position[0][0]
        y = obj.position[0][1] + frame.shape[0]  # Offset y to place in bottom half
        cv2.circle(
            split_frame, (x, y), 10, obj.color, -1
        )
        cv2.putText(split_frame, str(obj.id), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw lines connecting matched objects
    for tracked_idx, detected_idx in matches:
        tracked_obj = tracked_objects[tracked_idx]
        detected_obj = detected_objects[detected_idx]
        
        start_point = tracked_obj.position[-1]
        adjusted_start_point = (start_point[0], start_point[1] + 10)
        end_point = (detected_obj.position[0][0], detected_obj.position[0][1] + frame.shape[0] - 10)
        
        cv2.line(split_frame, adjusted_start_point, end_point, (0, 255, 0), 2)
    
    
    # print_objects(detected_objects)

    return split_frame

def save_cost_matrices():
    """Save all accumulated cost matrices to disk"""
    if not cost_matrix_storage:
        return
    
    # Create directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert the list of matrices into a single DataFrame
    all_data = []
    for frame_data in cost_matrix_storage:
        frame_num = frame_data['frame']
        matrix = frame_data['matrix']
        
        # Flatten the matrix and create rows
        for tracked_idx in range(matrix.shape[0]):
            for detected_idx in range(matrix.shape[1]):
                costs = matrix[tracked_idx, detected_idx]
                row = {
                    'frame': frame_num,
                    'tracked_object_id': frame_data['tracked_ids'][tracked_idx],
                    'detected_object_id': frame_data['detected_ids'][detected_idx],
                    'position_cost': costs[0],
                    'bbox_area_cost': costs[1],
                    'shape_cost': costs[2],
                    'feature_cost': costs[3],
                    'class_cost': costs[4],
                    'total_cost': costs[5]
                }
                all_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    filename = f"cost_matrices_{timestamp}.csv"
    filepath = os.path.join(SAVE_DIR, filename)
    df.to_csv(filepath, index=False)
    
    print(f"Saved cost matrices to {filepath}")
    
    # Clear storage after saving
    cost_matrix_storage.clear()

def print_objects(print_objects):
    for obj in print_objects:
        print(f"Object ID: {obj.id}, Position: {obj.position}")

def main():
    global object_container
    global current_frame_number
    
    frame_start = 1  # Start frame number

    frame_end = 140  # End frame number
    sequence_number = 1  # Sequence number

    # frame_end = 205  # End frame number
    # sequence_number = 2  # Sequence number

    object_container = []
    current_frame_number = frame_start
    frame_1_prev = None

    while True:
        print(f"frame number: {current_frame_number}")

        frame_1 = get_frame(current_frame_number, sequence_number, 2)

        detection_output = get_detection_results(current_frame_number, sequence_number)

        detected_objects = detect_objects(frame_1, detection_output)

        # frame_with_tracked_objects = visualize_objects(frame_1, object_container)

        object_container, matches = match_objects(detected_objects, object_container)

        # frame_with_detected_objects = visualize_objects(frame_1, detected_objects)

        frame_with_matched_objects = visualize_matched_objects(frame_1, object_container, detected_objects, matches)

        # masked_frame_1 = get_masked_image(frame_1, detection_output)
        # bbox_frame = draw_bounding_boxes(frame_1, detection_output)
        
        # combined_frames = combine_frames([frame_with_tracked_objects, frame_with_detected_objects, masked_frame_1, bbox_frame])
        
        print("\n\n")

        # Display current frame number
        # cv2.putText(combined_frames, f"Frame: {current_frame_number}", (10, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # cv2.namedWindow("Frame with combined_frames", cv2.WINDOW_NORMAL)
        # cv2.imshow("Frame with combined_frames", combined_frames)

        cv2.namedWindow("Frame with frame_with_matched_objects", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame with frame_with_matched_objects", frame_with_matched_objects)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == 81:  # Left arrow key
            current_frame_number = max(frame_start, current_frame_number - 1)
        elif key == 83:  # Right arrow key
            current_frame_number = min(frame_end, current_frame_number + 1)
        elif key == ord('a'):  # Go back multiple frames
            current_frame_number = max(frame_start, current_frame_number - 3)
        elif key == ord('d'):  # Go forward multiple frames
            current_frame_number = min(frame_end, current_frame_number + 3)

    save_cost_matrices()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()