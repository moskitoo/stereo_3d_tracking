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
import copy

from kalman_filter import KalmanTracker


id_counter = 0
current_frame_number = 0

cost_matrix_storage = []
SAVE_DIR = "cost_matrices"

match_correct_frame_no = 5


class TrackedObject:
    def __init__(self, type, position, bbox, features, color, id):
        self.type = type
        self.position = [position]
        self.bbox = bbox
        self.features = features
        self.color = color
        self.id = id
        self.unmatched_counter = 0
        self.initialize_kalman(position)

    def initialize_kalman(self, position, velocity=(0,0)):
        self.kalman_tracker = KalmanTracker()
        self.kalman_tracker.X[0] = position[0]
        self.kalman_tracker.X[3] = position[1]
        self.kalman_position = [position]
        self.kalman_velocity = [velocity]
        self.kalman_pred_position = [position]

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)
    
    def clone(self):
        return copy.deepcopy(self)

    def update_state(self, detected_object):
        self.position.append(detected_object.position[-1])

        self.bbox = detected_object.bbox
        self.features = detected_object.features
        self.unmatched_counter = 0
        x = detected_object.position[-1][0]
        y = detected_object.position[-1][1]

        kalman_position = np.array([[x], [y]])
        update = self.kalman_tracker.update(kalman_position)
        self.kalman_position.append((int(update[0, 0]), int(update[3, 0])))
        self.kalman_velocity.append((int(update[1, 0]), int(update[4, 0])))
        self.kalman_pred_position.append(
            [
                self.kalman_position[-1][0] + self.kalman_velocity[-1][0],
                self.kalman_position[-1][1] + self.kalman_velocity[-1][1],
            ]
        )

    def update_state_rematch(self, detected_object):

        if len(detected_object.position) < match_correct_frame_no:
            new_position = detected_object.position
        else:
            new_position = detected_object.position[-match_correct_frame_no:]

        print(f"obj {self.id} new pos: {new_position}")

        print(f"obj {self.id}  pre pos: {self.position}")
        
        if len(self.position) < match_correct_frame_no - 1:
            self.position = new_position
        else:
            self.position = self.position[:-match_correct_frame_no]
            self.position += new_position

        print(f"obj {self.id} post pos: {self.position}")

        # self.position.append(detected_object.position[-1])

        self.bbox = detected_object.bbox
        self.features = detected_object.features
        self.unmatched_counter = 0

        print(f"pred pos kalman: {self.kalman_position}")
        print(f"pred velocity kalman: {self.kalman_velocity}")

        self.initialize_kalman(detected_object.position[-1], velocity=detected_object.kalman_velocity[-1])
        
        print(f"post pos kalman: {self.kalman_position}")
        print(f"post velocity kalman: {self.kalman_velocity}")
    
    def predict_position_from_prev_state(self, prev_frame_no):
        velocity_array = np.array(self.kalman_velocity)

        print(f"ID: {self.id}")
        
        print(f"velocities: {velocity_array}")

        # Check the length of the velocity array
        if len(velocity_array) < prev_frame_no:
            # Not enough frames for a meaningful calculation
            avg_kalman_vector = np.mean(velocity_array, axis=0)
            print(f"velocities: {velocity_array}")
        elif len(velocity_array) < 2 * prev_frame_no:
            # Use the available frames before match_correct_frame_no
            avg_kalman_vector = np.mean(velocity_array[:prev_frame_no], axis=0)
            print(f"velocities: {velocity_array[:prev_frame_no]}")
        else:
            # Use the specified range for averaging
            avg_kalman_vector = np.mean(velocity_array[-2 * prev_frame_no:-prev_frame_no], axis=0)
            print(f"velocities: {velocity_array[-2 * prev_frame_no:-prev_frame_no]}")
            print(f"Using range: -2 * match_correct_frame_no to -match_correct_frame_no.")

        # print(f"avg kalman vector: {avg_kalman_vector}")
        
        # Estimate position using average velocity
        if len(self.position) < prev_frame_no:
            estimated_position = self.position[0] + avg_kalman_vector * prev_frame_no
        else:
            estimated_position = self.position[-prev_frame_no] + avg_kalman_vector * prev_frame_no

        drift = np.linalg.norm(estimated_position - self.position[-1])
        print(f"Drift: {round(drift, 2)}, Estimated position: {estimated_position}, Actual position: {self.position[-1]}")

        return estimated_position


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
        return frame[self.top: self.bottom, self.left: self.right]

    def get_bbox_area(self):
        return self.width * self.height

    def get_bbox_aspect_ratio(self):
        return self.width / self.height


def visualize_objects(frame, tracked_objects, match_correct_frame_no):
    frame_copy = frame.copy()

    counter = 0
    # Display features for each object in its unique color
    for obj in tracked_objects.values():
        x = obj.position[-1][0]
        y = obj.position[-1][1]

        # if obj.id < 5:
        #     print(f"position (t): {obj.position}")
        #     print(f"kalman position: {obj.kalman_position}")
        #     print(f"kalman velocity: {obj.kalman_velocity}")
        #     print(f"kalman position predicted: {obj.kalman_pred_position}")

        # Calculate a longer arrow by extending the line
        dx = obj.kalman_pred_position[-1][0] - obj.kalman_position[-1][0]
        dy = obj.kalman_pred_position[-1][1] - obj.kalman_position[-1][1]

        # Multiply the difference by a scaling factor (e.g., 3)
        extended_end = (
            obj.kalman_position[-1][0] + dx * 1,
            obj.kalman_position[-1][1] + dy * 1,
        )

        if len(obj.kalman_position) >= 5:
            cv2.arrowedLine(frame_copy, obj.kalman_position[-5],
                        extended_end, obj.color, 2)
            # take the oldest one when we dont have enough records
        else: 
            cv2.arrowedLine(frame_copy, obj.kalman_position[0],
                        extended_end, obj.color, 2)             

        cv2.circle(frame_copy, obj.kalman_pred_position[-1], 10, obj.color, -1)
        cv2.putText(
            frame_copy,
            str(obj.id),
            (obj.kalman_pred_position[-1][0] - 5, obj.kalman_pred_position[-1][1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.circle(frame_copy, (x, y), 10, obj.color, -1)
        cv2.putText(
            frame_copy,
            str(obj.id),
            (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        counter += 1
    print(counter)
    return frame_copy


def get_rand_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def get_detection_results(frame_number, seq_num):
    seq_labels = labels[seq_num]
    frame_labels = seq_labels[seq_labels["frame"] == frame_number]

    # Extract columns 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom' as a NumPy array
    bboxes = (
        frame_labels[["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]]
        .to_numpy()
        .astype(int)
    )

    # Extract 'type' column as a NumPy array
    obj_types = frame_labels["type"].to_numpy()

    return bboxes, obj_types


def detect_objects(frame, detection_output):
    sift = cv2.SIFT_create()
    detected_objects = []
    for id, [bbox, obj_type] in enumerate(
        zip(detection_output[0], detection_output[1])
    ):
        bbox_obj = BoundingBox(*bbox)

        cropped_frame = bbox_obj.get_bbox_img(frame)
        frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(frame_gray, None)

        if des is None:
            features = np.zeros(128)
        else:
            features = np.mean(des, axis=0)

        detected_objects.append(
            TrackedObject(
                obj_type, bbox_obj.position, bbox_obj, features, get_rand_color(), id
            )
        )

    return detected_objects


def detect_objects_yolo(frame, detection_output):
    sift = cv2.SIFT_create()
    detected_objects = {}
    print(type(detection_output))
    for id, [bbox, obj_type] in enumerate(
        zip(detection_output.boxes.xyxy, detection_output.boxes.cls)
    ):
        bbox_obj = BoundingBox(*bbox)

        cropped_frame = bbox_obj.get_bbox_img(frame)

        frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(frame_gray, None)

        if des is None:
            features = np.zeros(128)
        else:
            features = np.mean(des, axis=0)

        detected_objects[id] = TrackedObject(
            obj_type, bbox_obj.position, bbox_obj, features, get_rand_color(), id
        )

    return detected_objects

def apply_nms(detected_objects, nms_iou_threshold):
    # Convert to list for easier sorting and processing
    objects_list = list(detected_objects.items())
    
    objects_to_keep = []
    
    while objects_list:
        # Take the object with highest confidence
        current_idx, current_obj = objects_list.pop(0)
        objects_to_keep.append((current_idx, current_obj))
        
        # Remove objects with high IOU
        objects_list = [
            (idx, obj) for (idx, obj) in objects_list 
            if IOU(current_obj.bbox, obj.bbox) <= nms_iou_threshold
        ]
    
    # Convert back to dictionary
    return dict(objects_to_keep)

def IOU(box1, box2):
    x1 = max(box1.left, box2.left)
    y1 = max(box1.top, box2.top)
    x2 = min(box1.right, box2.right)
    y2 = min(box1.bottom, box2.bottom)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    union = box1.get_bbox_area() + box2.get_bbox_area() - intersection

    if union == 0:
        return 0

    iou = intersection / union

    return iou


def get_cost_matrix(
    detected_objects,
    object_container,
    pos_w=0.006,
    bbox_area_w=0.4,
    bbox_shape_w=0.2,
    iou_w=1.0,
    feat_w=0.1,
    kalman_vector_w=1.0,
    iou_threshold=0.95,
    past_pos_id = 5
):
    num_tracked = len(object_container)
    num_detections = len(detected_objects)
    cost_matrix = np.zeros((num_tracked, num_detections))

    # Initialize detailed cost matrix with shape (num_tracked, num_detections, 6)
    # Added extra dimension for total cost
    cost_matrix_detailed = np.zeros((num_tracked, num_detections, 5))
    cost_matrix_detailed_not_scaled = np.zeros(
        (num_tracked, num_detections, 5))
    cost_matrix_basic = np.zeros((num_tracked, num_detections, 3))

    for i, (tracked_id, tracked_object) in enumerate(object_container.items()):
        # Precompute normalization factors for the current tracked object
        max_pos_cost = 0
        max_bbox_area_cost = 0
        max_kalman_euc_cost = 0

        for j, (detected_id, detected_object) in enumerate(detected_objects.items()):
            # Position cost
            pos_cost = np.linalg.norm(
                tracked_object.position[0] - detected_object.position[0]
            )
            max_pos_cost = max(max_pos_cost, pos_cost)

            # Bbox area cost
            det_obj_area = detected_object.bbox.get_bbox_area()
            tracked_obj_area = tracked_object.bbox.get_bbox_area()
            bbox_area_cost = abs(tracked_obj_area - det_obj_area) / max(
                det_obj_area, tracked_obj_area
            )
            max_bbox_area_cost = max(max_bbox_area_cost, bbox_area_cost)

            # Kalman orientation cost (Euclidean)
            if len(tracked_object.position) >= past_pos_id:
                kalman_vector = (
                    tracked_object.kalman_pred_position -
                    tracked_object.position[-past_pos_id]
                )
                # take the oldest one when we dont have enough records
            else: 
                kalman_vector = (
                    tracked_object.kalman_pred_position -
                    tracked_object.position[0]
                )
            detection_vector = detected_object.position[0] - \
                tracked_object.position[-1]
            kalman_euc_cost = np.linalg.norm(kalman_vector - detection_vector)
            max_kalman_euc_cost = max(max_kalman_euc_cost, kalman_euc_cost)

        # Second pass: Normalize and compute total cost for the current tracked object
        for j, (detected_id, detected_object) in enumerate(detected_objects.items()):
            # Position cost (normalized)
            pos_cost = np.linalg.norm(
                tracked_object.position[0] - detected_object.position[0]
            )
            pos_cost_normalized = pos_cost / max_pos_cost if max_pos_cost > 0 else 0

            # Bbox area cost (normalized)
            det_obj_area = detected_object.bbox.get_bbox_area()
            tracked_obj_area = tracked_object.bbox.get_bbox_area()
            bbox_area_cost = abs(tracked_obj_area - det_obj_area) / max(
                det_obj_area, tracked_obj_area
            )
            bbox_area_cost_normalized = (
                bbox_area_cost / max_bbox_area_cost if max_bbox_area_cost > 0 else 0
            )

            # Bbox shape cost
            shape_cost = abs(
                tracked_object.bbox.get_bbox_aspect_ratio()
                - detected_object.bbox.get_bbox_aspect_ratio()
            )

            # IoU cost
            iou_cost = 1 - IOU(tracked_object.bbox, detected_object.bbox)
            if iou_cost >= iou_threshold:
                iou_cost = 2 * iou_cost

            # Feature cost
            feat_cost = (
                1
                - cosine_similarity(
                    [tracked_object.features], [detected_object.features]
                )[0, 0]
            )

            # Kalman orientation cost (normalized)
            kalman_vector = (
                tracked_object.kalman_pred_position -
                tracked_object.position[-1]
            )
            detection_vector = detected_object.position[0] - \
                tracked_object.position[-1]
            kalman_euc_cost = np.linalg.norm(kalman_vector - detection_vector)
            kalman_euc_cost_normalized = (
                kalman_euc_cost / max_kalman_euc_cost if max_kalman_euc_cost > 0 else 0
            )

            # Class cost
            class_cost = 0 if tracked_object.type == detected_object.type else 100

            # Calculate total cost
            total_cost = (
                iou_w * iou_cost
                + kalman_vector_w * kalman_euc_cost_normalized
                + class_cost
            )

            cost_matrix[i, j] = total_cost

            # Store detailed costs including total cost
            # detailed_cost = [iou_w * iou_cost, feat_w * feat_cost, class_cost, total_cost]
            # detailed_cost_not_scaled = [iou_cost, feat_cost, class_cost, total_cost]
            detailed_cost_basic = [
                iou_w * iou_cost,
                feat_w * feat_cost,
                kalman_vector_w * kalman_euc_cost_normalized,
            ]
            detailed_cost = [
                iou_w * iou_cost,
                feat_w * feat_cost,
                kalman_vector_w * kalman_euc_cost_normalized,
                class_cost,
                total_cost,
            ]
            detailed_cost_not_scaled = [
                iou_cost,
                feat_cost,
                kalman_euc_cost_normalized,
                class_cost,
                total_cost,
            ]
            cost_matrix_basic[i, j] = [round(x, 2)
                                       for x in detailed_cost_basic]
            cost_matrix_detailed[i, j] = [round(x, 2) for x in detailed_cost]
            cost_matrix_detailed_not_scaled[i, j] = [
                round(x, 2) for x in detailed_cost_not_scaled
            ]

    row_ids = object_container.keys()
    column_ids = detected_objects.keys()
    pd.options.display.float_format = "{:,.2f}".format

    # Print detailed costs with total cost included
    print(
        pd.DataFrame(
            cost_matrix_basic.reshape(num_tracked, num_detections * 3),
            index=row_ids,
            columns=list(column_ids) * 3,
        )
    )

    print(f"Current frame number: {current_frame_number}")
    # print(pd.DataFrame(cost_matrix_detailed[0]))
    # print(pd.DataFrame(cost_matrix_detailed[1]))
    # print(pd.DataFrame(cost_matrix_detailed_not_scaled[0]))
    # print(pd.DataFrame(cost_matrix_detailed_not_scaled[1]))
    print(pd.DataFrame(cost_matrix_basic[0]))

    cost_matrix_storage.append(
        {
            "frame": current_frame_number,  # You'll need to make current_frame_number accessible
            "matrix": cost_matrix_detailed,
            "tracked_ids": row_ids,
            "detected_ids": column_ids,
        }
    )

    print(pd.DataFrame(cost_matrix, index=row_ids, columns=column_ids))

    return cost_matrix, list(row_ids), list(column_ids)


def match_objects(
    detected_objects, object_container, cost_threshold=1.5, unmatched_threshold=5
):
    global id_counter
    global current_frame_number

    # State 1: If no objects exist, create the first one
    if not object_container:
        for container_id, detected_object in detected_objects.items():
            container_id = id_counter
            detected_object.id = id_counter
            id_counter += 1
        object_container = detected_objects

        detected_objects_ids = [x.id for x in detected_objects.values()]
        print(f"added objects: {detected_objects_ids}")

        detected_objects_ids = [x for x in detected_objects]
        print(f"added objects container ids: {detected_objects_ids}")

        return object_container, [], []
    else:
        cost_matrix, row_ids, column_ids = get_cost_matrix(
            detected_objects, object_container
        )
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        print(f"rows: {row_indices}")
        print(f"cols: {col_indices}")

        matches, matches_nonfiltered, unmatched_tracked, unmatched_detected = (
            filter_false_matches(
                detected_objects,
                object_container,
                cost_threshold,
                cost_matrix,
                row_indices,
                col_indices,
            )
        )

        matches_decoded = []
        for match in matches:
            matches_decoded.append((row_ids[match[0]], column_ids[match[1]]))

        matches_cost = []
        for row, col in zip(row_indices, col_indices):
            matches_cost.append(round(float(cost_matrix[row, col]), 2))

        print(f"matches_cost:        {matches_cost}")
        print(f"matches_nonfiltered: {matches_nonfiltered}")
        print(f"matches:             {matches}")
        print(f"matches decoded:     {matches_decoded}")

        # State 2: Match with existing objects
        for tracked_object_id, detect_object_id in matches_decoded:
            tracked_object = object_container[tracked_object_id]
            detected_object = detected_objects[detect_object_id]
            tracked_object.update_state(detected_object)

        # State 3: Remove non matched trakced objects
        # Convert unmatched_tracked to a set for faster lookup
        unmatched_tracked = [row_ids[id] for id in unmatched_tracked]
        unmatched_tracked_set = set(unmatched_tracked)
        print(f"unmatched tracked objects: {unmatched_tracked}")

        # Remove unmatched objects after several unsuccessful matches
        for unmatched_id in unmatched_tracked_set:
            object_container[unmatched_id].unmatched_counter += 1

        object_container = {
            id: tracked_object
            for id, tracked_object in object_container.items()
            if tracked_object.unmatched_counter < unmatched_threshold
        }

        # State 4: Add non matched detected objects to tracking
        unmatched_detected_ids = []
        for unmatched_detected_id in unmatched_detected:
            non_matched_object = detected_objects[unmatched_detected_id]
            # non_matched_object.id = id_counter
            new_tracked_object = TrackedObject(
                non_matched_object.type,
                non_matched_object.position[-1],
                non_matched_object.bbox,
                non_matched_object.features,
                non_matched_object.color,
                id_counter,  # New ID for tracked object
            )
            object_container[id_counter] = new_tracked_object
            unmatched_detected_ids.append(id_counter)
            id_counter += 1

        print(f"unmatched detected objects: {unmatched_detected_ids}")

        current_frame_number += 1

        return object_container, matches, matches_decoded

def correct_matches(object_container, match_correct_frame_no, drift_threshold, cost_threshold):

    mismatched_instances = []
    for i, (tracked_id, tracked_object) in enumerate(object_container.items()):
        # Convert kalman_velocity to a numpy array for vectorized operations
        
        estimated_position = tracked_object.predict_position_from_prev_state(match_correct_frame_no)

        drift = np.linalg.norm(estimated_position - tracked_object.position[-1])

        if drift > drift_threshold:
            mismatched_instances.append(tracked_object)

        print(f"drift {drift}")

    if len(mismatched_instances) == 0:
        print("Didnt find any mismatches")
    
    cost_matrix = np.zeros((len(mismatched_instances), len(mismatched_instances)))

    for i, row_obj in enumerate(mismatched_instances):
        for j, col_obj in enumerate(mismatched_instances):
            if i == j:
                cost_matrix[i, j] = np.inf
            elif row_obj.type != col_obj.type:
                cost_matrix[i, j] = cost_threshold
            else:
                cost_matrix[i, j] = np.linalg.norm(row_obj.position[-1] - col_obj.position[-1])

    print("mismatches cost matrix")
    print(cost_matrix)
    
    if len(mismatched_instances) <= 1:
        return
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    print(f"rows: {row_indices}")
    print(f"cols: {col_indices}")

    matches = []
    matches_decoded = []
    costs = []

    for row, col in zip(row_indices, col_indices):
        matches.append((row, col))
        matches_decoded.append((mismatched_instances[row].id, mismatched_instances[col].id))
        costs.append(cost_matrix[row, col])

    print(f"matches: {matches}")
    print(f"matches_decoded: {matches_decoded}")

    remached_instances = set()  # Use a set for faster lookup
    for cost, (match_id_1, match_id_2) in zip(costs, matches_decoded):
        print(f"rematched: {remached_instances}")
        
        # Skip if either match ID is already processed or the cost is too high
        if match_id_1 in remached_instances or match_id_2 in remached_instances or cost >= cost_threshold:
            print(f"rematched discarded: {match_id_1},{match_id_2}")
            continue
        
        # Ensure both match IDs are valid
        if match_id_1 is not None and match_id_2 is not None:
            try:
                object_buffer = object_container[match_id_1].clone()
                # target_object = object_container[match_id_2]

                print(f"obj1 {match_id_1} og positions: {object_container[match_id_1].position}")
                print(f"obj2 {match_id_2} og positions: {object_container[match_id_2].position}")
                
                # Update states
                object_container[match_id_1].update_state_rematch(object_container[match_id_2])
                object_container[match_id_2].update_state_rematch(object_buffer)
                # object_container[match_id_1].update_state(object_container[match_id_2])
                # object_container[match_id_2].update_state(object_buffer)

                print(f"obj1 {match_id_1} NEW positions: {object_container[match_id_1].position}")
                print(f"obj2 {match_id_2} NEW positions: {object_container[match_id_2].position}")
                
                # Mark both IDs as processed
                remached_instances.add(match_id_1)
                remached_instances.add(match_id_2)
                
                # Optional: Debug prints
                print(f"Rematched objects: {match_id_1} and {match_id_2}")
                print(f"Cost: {cost}")
            
            except KeyError:
                print(f"Warning: Invalid object IDs - {match_id_1}, {match_id_2}")
            except Exception as e:
                print(f"Error during rematching: {e}")


def filter_false_matches(
    detected_objects,
    object_container,
    cost_threshold,
    cost_matrix,
    row_indices,
    col_indices,
):
    matches = []
    matches_unfiltered = []
    unmatched_tracked = set(range(len(object_container)))
    unmatched_detected = set(range(len(detected_objects)))

    for row, col in zip(row_indices, col_indices):
        row = int(row)
        col = int(col)
        if cost_matrix[row, col] < cost_threshold:
            matches.append((row, col))
            unmatched_tracked.discard(row)
            unmatched_detected.discard(col)
        matches_unfiltered.append((row, col))

    unmatched_tracked = list(unmatched_tracked)
    unmatched_detected = list(unmatched_detected)
    return matches, matches_unfiltered, unmatched_tracked, unmatched_detected


def get_masked_image(frame, detection_output):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # VERSION MADE FOR YOLO OUTPUT WORKING PREVIOUS VERSION IN THE COMMITS
    for id, [bbox, obj_type] in enumerate(
        zip(detection_output.boxes.xyxy, detection_output.boxes.cls)
    ):
        bbox_obj = BoundingBox(*bbox)

        mask[bbox_obj.top: bbox_obj.bottom, bbox_obj.left: bbox_obj.right] = 255

    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return masked_frame


def draw_bounding_boxes(frame, detection_output):

    for id, [bbox, obj_type] in enumerate(
        zip(detection_output.boxes.xyxy, detection_output.boxes.cls)
    ):
        bbox_obj = BoundingBox(*bbox)

        left = bbox_obj.left
        top = bbox_obj.top
        right = bbox_obj.right
        bottom = bbox_obj.bottom

        # Draw the bounding box on the frame
        cv2.rectangle(
            frame, (left, top), (right, bottom), color=(0, 255, 0), thickness=2
        )

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


def visualize_matched_objects(
    prev_frame, frame, tracked_objects, detected_objects, matches
):
    # Create a frame twice the height of the original
    split_frame = np.zeros(
        (frame.shape[0] * 2, frame.shape[1], 3), dtype=np.uint8)

    # Copy the original frame to the top half
    split_frame[: frame.shape[0], :, :] = prev_frame.copy()
    split_frame[frame.shape[0]:, :, :] = frame.copy()

    # Top frame - tracked objects
    for obj in tracked_objects.values():
        if len(obj.position) > 1:
            x = obj.position[-2][0]
            y = obj.position[-2][1]
        else:
            x = obj.position[-1][0]
            y = obj.position[-1][1]
        cv2.circle(split_frame, (x, y), 10, obj.color, -1)
        cv2.putText(
            split_frame,
            str(obj.id),
            (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Bottom frame - detected objects
    for obj in detected_objects.values():
        x = obj.position[0][0]
        # Offset y to place in bottom half
        y = obj.position[0][1] + frame.shape[0]
        cv2.circle(split_frame, (x, y), 10, obj.color, -1)
        cv2.putText(
            split_frame,
            str(obj.id),
            (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw lines connecting matched objects
    for tracked_idx, detected_idx in matches:
        tracked_obj = tracked_objects[tracked_idx]
        detected_obj = detected_objects[detected_idx]

        if len(tracked_obj.position) > 1:
            start_point = tracked_obj.position[-2]
        else:
            start_point = tracked_obj.position[-1]
        adjusted_start_point = (start_point[0], start_point[1] + 10)
        end_point = (
            detected_obj.position[0][0],
            detected_obj.position[0][1] + frame.shape[0] - 10,
        )

        cv2.line(split_frame, adjusted_start_point, end_point, (0, 255, 0), 2)

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
        frame_num = frame_data["frame"]
        matrix = frame_data["matrix"]

        # Flatten the matrix and create rows
        for tracked_idx in range(matrix.shape[0]):
            for detected_idx in range(matrix.shape[1]):
                costs = matrix[tracked_idx, detected_idx]
                row = {
                    "frame": frame_num,
                    "tracked_object_id": frame_data["tracked_ids"][tracked_idx],
                    "detected_object_id": frame_data["detected_ids"][detected_idx],
                    "position_cost": costs[0],
                    "bbox_area_cost": costs[1],
                    "shape_cost": costs[2],
                    "feature_cost": costs[3],
                    "class_cost": costs[4],
                    "total_cost": costs[5],
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

        detection_output = get_detection_results(
            current_frame_number, sequence_number)

        detected_objects = detect_objects(frame_1, detection_output)

        # frame_with_tracked_objects = visualize_objects(frame_1, object_container)

        object_container, matches = match_objects(
            detected_objects, object_container)

        # frame_with_detected_objects = visualize_objects(frame_1, detected_objects)

        frame_with_matched_objects = visualize_matched_objects(
            frame_1, object_container, detected_objects, matches
        )

        # masked_frame_1 = get_masked_image(frame_1, detection_output)
        # bbox_frame = draw_bounding_boxes(frame_1, detection_output)

        # combined_frames = combine_frames([frame_with_tracked_objects, frame_with_detected_objects, masked_frame_1, bbox_frame])

        print("\n\n")

        # Display current frame number
        # cv2.putText(combined_frames, f"Frame: {current_frame_number}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # cv2.namedWindow("Frame with combined_frames", cv2.WINDOW_NORMAL)
        # cv2.imshow("Frame with combined_frames", combined_frames)

        cv2.namedWindow("Frame with frame_with_matched_objects",
                        cv2.WINDOW_NORMAL)
        cv2.imshow("Frame with frame_with_matched_objects",
                   frame_with_matched_objects)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):  # Quit
            break
        elif key == 81:  # Left arrow key
            current_frame_number = max(frame_start, current_frame_number - 1)
        elif key == 83:  # Right arrow key
            current_frame_number = min(frame_end, current_frame_number + 1)
        elif key == ord("a"):  # Go back multiple frames
            current_frame_number = max(frame_start, current_frame_number - 3)
        elif key == ord("d"):  # Go forward multiple frames
            current_frame_number = min(frame_end, current_frame_number + 3)

    save_cost_matrices()

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
