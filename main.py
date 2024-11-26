import numpy as np
import cv2
from typing import List, Tuple
from tqdm import tqdm

from feature_manager_v2 import (
    detect_objects_yolo, 
    visualize_objects, 
    match_objects, 
    visualize_matched_objects, 
    get_masked_image, 
    combine_frames
)

from object_detection import *
from depth_image_manager import *


class ObjectTracker:
    """Manages object tracking across video frames."""
    def __init__(self, sequence_number: int = 3):
        self.object_detector = ObjectDetector(sequence_number)
        self.depth_manager = DepthManager()
        self.object_container = {}

    def process_frame(self, image, raw_image, path, disparity) -> None:
        """Process a single frame for object detection and tracking."""
        image = image.to(self.object_detector.device)
        image.requires_grad = True

        self.disparity = disparity

        # Detect objects
        detection_output = self.object_detector.detect_objects(path)
        detected_objects = detect_objects_yolo(raw_image, detection_output[0])

        # Track objects
        frame_with_tracked_objects = visualize_objects(raw_image, self.object_container)
        self.object_container, matches, matches_decoded = match_objects(detected_objects, self.object_container)

        # Visualization
        frame_with_detected_objects = visualize_objects(raw_image, detected_objects)
        frame_with_matched_objects = visualize_matched_objects(raw_image, self.object_container, detected_objects, matches_decoded)
        masked_frame = get_masked_image(raw_image, detection_output[0])
        
        self.extend_tracked_position_to_3d(disparity)

        combined_frames = combine_frames([
            frame_with_tracked_objects, 
            frame_with_detected_objects, 
            masked_frame
        ])

        return combined_frames, frame_with_matched_objects

    def process_raw_image(self, raw_image):
        raw_image = raw_image.squeeze(0)
        raw_image = raw_image.permute(1, 2, 0).numpy()
        raw_image = (raw_image * 255).astype(np.uint8)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        return raw_image

    def extend_tracked_position_to_3d(self, disparity):
        pass


    def run(self):
        """Run object tracking on all frames."""
        for left_image, right_image, left_raw_img, right_raw_img, left_img_path, right_img_path in tqdm(self.object_detector.dataloader):

            left_raw_img = self.process_raw_image(left_raw_img)
            right_raw_img = self.process_raw_image(right_raw_img)

            disparity = self.depth_manager.get_disparity_map(left_raw_img, right_raw_img)

            combined_frames, frame_with_matched_objects = self.process_frame(left_image, left_raw_img, left_img_path, disparity)

            cv2.namedWindow("Frame with combined_frames", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame with combined_frames", combined_frames)

            cv2.namedWindow("Frame with matched objects", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame with matched objects", frame_with_matched_objects)

            cv2.imshow('Disparity Map', disparity)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Quit
                break

def main():
    sequence_number = 1
    tracker = ObjectTracker(sequence_number=sequence_number)
    tracker.run()

if __name__ == '__main__':
    main()