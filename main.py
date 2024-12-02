import numpy as np
import cv2
from typing import List, Tuple

from IPython.core.display_functions import display
from scipy.optimize import fsolve

from sympy.stats.sampling.sample_scipy import scipy
from tqdm import tqdm
import pickle

from feature_manager_v2 import *

from object_detection import *
from depth_image_manager import *
from kalman_filter_3d import *


class ObjectTracker:
    """Manages object tracking across video frames."""
    def __init__(self, 
                 sequence_number: int = 3, 
                 save_detections: bool = False, 
                 load_detections: bool = False, 
                 save_tracking: bool = False,
                 enable_tracking: bool = True,
                 detections_dir='results/detections', 
                 tracking_dir='results/tracking'):
        self.sequence_number = sequence_number
        self.object_detector = ObjectDetector(sequence_number)
        self.depth_manager = DepthManager()
        self.object_container = {}
        self.previous_frame = None
        self.frame_number = 0
        self.rematching_freq = 3
        self.rematching_frame_no = 5
        self.drift_threshold = 100
        self.rematch_cost_threshold = 1000
        self.image_width = 1223
        self.image_height = 370

        self.save_detections = save_detections
        self.load_detections = load_detections
        self.detections_dir = detections_dir
        self.save_tracking = save_tracking
        self.tracking_dir = tracking_dir
        self.enable_tracking = enable_tracking

    def process_frame(self, image, raw_image, path) -> None:
        """Process a single frame for object detection and tracking."""
        image = image.to(self.object_detector.device)
        image.requires_grad = True

        combined_frames = None
        frame_with_matched_objects = None

        if self.load_detections:
            try:
                detection_path = self.detections_dir + f'/seq{self.sequence_number}/frame_{self.frame_number}_detection.pkl'
                if not os.path.exists(detection_path):
                    raise FileNotFoundError(f"Pickle file not found: {detection_path}")
                
                with open(detection_path, 'rb') as f:
                    detection_outputs = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Error loading pickle file: {e}")
                return None
        else:
            detection_outputs = self.object_detector.detect_objects(path)

        if self.enable_tracking:

            for detection_output in detection_outputs:

                detected_objects = detect_objects_yolo(raw_image, detection_output, self.depth_manager)
                detected_objects = apply_nms(detected_objects, 0.5)

                # for detected_object in detected_objects.values():
                #     print(f"ID: {detected_object.id}, world pos.: {detected_object.position}, image pos.: {detected_object.frame_2d_position}")

                # Track objects
                self.object_container, matches, matches_decoded = match_objects(detected_objects, self.object_container)
                frame_with_tracked_objects = visualize_objects(raw_image, self.object_container, self.depth_manager)

                if self.frame_number % self.rematching_freq == 0:
                    correct_matches(self.object_container, self.rematching_frame_no, self.drift_threshold, self.rematch_cost_threshold)

                # Visualization
                # frame_with_detected_objects = visualize_objects(raw_image, detected_objects)
                frame_with_matched_objects = visualize_matched_objects(self.previous_frame.copy(), raw_image, self.object_container, detected_objects, matches_decoded, self.depth_manager)
                masked_frame = get_masked_image(raw_image, detection_output)
                bbox_frame = draw_bounding_boxes(raw_image, detection_output)

                self.object_container = {
                    id: obj for id, obj in self.object_container.items() 
                    if (0 < self.depth_manager.get_object_position_in_img_frame(obj.kalman_pred_position[-1])[0] < self.image_width and 
                        0 < self.depth_manager.get_object_position_in_img_frame(obj.kalman_pred_position[-1])[1] < self.image_height)
                }

                combined_frames = combine_frames([
                    frame_with_tracked_objects, 
                    # frame_with_detected_objqects, 
                    masked_frame,
                    bbox_frame
                ])

        return combined_frames, frame_with_matched_objects, detection_outputs

    def process_raw_image(self, raw_image):
        raw_image = raw_image.squeeze(0)
        raw_image = raw_image.permute(1, 2, 0).numpy()
        raw_image = (raw_image * 255).astype(np.uint8)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        return raw_image
    
            # if not hasattr(self.object_container[object_id],'kalman_tracker3d'):
            #     self.object_container[object_id].initialize_3d_kalman(average_position)
            #     self.object_container[object_id].world_3d_position = average_position
            # elif self.object_container[object_id].unmatched_counter == 0:
            #    self.object_container[object_id].kalman_tracker3d.update(average_position)
            #    self.object_container[object_id].world_3d_position = self.object_container[object_id].kalman_tracker3d.get_position()
            # else:
            #     self.object_container[object_id].kalman_tracker3d.update()
            #     self.object_container[object_id].world_3d_position = self.object_container[
            #         object_id].kalman_tracker3d.get_position()

    def run(self):
        """Run object tracking on all frames."""
        for left_image, right_image, left_raw_img, right_raw_img, left_img_path, right_img_path in tqdm(self.object_detector.dataloader):

            left_raw_img = self.process_raw_image(left_raw_img)
            right_raw_img = self.process_raw_image(right_raw_img)

            if self.previous_frame is None:
                self.previous_frame = left_raw_img.copy()

            self.depth_manager.update_disparity_map(left_raw_img, right_raw_img)

            combined_frames, frame_with_matched_objects, detection_outputs = self.process_frame(left_image, left_raw_img.copy(), left_img_path)

            if self.save_detections:
                pickle_dir = os.path.join(self.detections_dir, f'seq{self.sequence_number}')
                os.makedirs(pickle_dir, exist_ok=True)
                
                pickle_path = os.path.join(pickle_dir, f'frame_{self.frame_number}_detection.pkl')
                
                with open(pickle_path, 'wb') as f:
                    pickle.dump(detection_outputs, f)

            if self.enable_tracking:
                cv2.namedWindow("Frame with combined_frames", cv2.WINDOW_NORMAL)
                cv2.imshow("Frame with combined_frames", combined_frames)

                cv2.namedWindow("Frame with matched objects", cv2.WINDOW_NORMAL)
                cv2.imshow("Frame with matched objects", frame_with_matched_objects)

            cv2.namedWindow('Disparity Map', cv2.WINDOW_NORMAL)
            cv2.imshow('Disparity Map', cv2.normalize(self.depth_manager.disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

            self.previous_frame = left_raw_img.copy()

            self.frame_number += 1

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Quit
                break

def main():
    sequence_number = 2
    tracker = ObjectTracker(sequence_number=sequence_number, load_detections=True, enable_tracking=True)
    tracker.run()

if __name__ == '__main__':
    main()