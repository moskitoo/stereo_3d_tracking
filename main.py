import numpy as np
import cv2
from typing import List, Tuple

from IPython.core.display_functions import display
from scipy.optimize import fsolve

from sympy.stats.sampling.sample_scipy import scipy
from tqdm import tqdm

from feature_manager_v2 import *

from object_detection import *
from depth_image_manager import *

camera_projection_matrix_left = np.array([[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01],
                                          [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]])

camera_calibration_matrix_left = np.array([[9.569475e+02, 0.000000e+00, 6.939767e+02],
                                           [0.000000e+00, 9.522352e+02, 2.386081e+02],
                                           [0.000000e+00, 0.000000e+00, 1.000000e+00]])

camera_calibration_matrix_right = np.array([[9.011007e+02, 0.000000e+00, 6.982947e+02],
                                            [0.000000e+00, 8.970639e+02, 2.377447e+02],
                                            [0.000000e+00, 0.000000e+00, 1.000000e+00]])

camera_distortion_coeff_left = np.array([-3.750956e-01, 2.076838e-01, 4.348525e-04, 1.603162e-03, -7.469243e-02])
camera_distortion_coeff_right = np.array([-3.686011e-01, 1.908666e-01, -5.689518e-04, 3.332341e-04, -6.302873e-02])

R_left = np.array([[9.999838e-01, -5.012736e-03, -2.710741e-03],
                   [5.002007e-03, 9.999797e-01, -3.950381e-03],
                   [2.730489e-03, 3.936758e-03, 9.999885e-01]])
R_right = np.array([[9.995054e-01, 1.665288e-02, -2.667675e-02],
                    [-1.671777e-02, 9.998578e-01, -2.211228e-03],
                    [2.663614e-02, 2.656110e-03, 9.996417e-01]])

T_left = np.array([5.989688e-02, -1.367835e-03, 4.637624e-03])
T_right = np.array([-4.756270e-01, 5.296617e-03, -5.437198e-03])

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
        detection_outputs = self.object_detector.detect_objects(path)

        for detection_output in detection_outputs:

            # detection_output.show()

            detected_objects = detect_objects_yolo(raw_image, detection_output)

            # Track objects
            frame_with_tracked_objects = visualize_objects(raw_image, self.object_container)
            self.object_container, matches, matches_decoded = match_objects(detected_objects, self.object_container)
            self.get_object_3d_location()

            # Visualization
            frame_with_detected_objects = visualize_objects(raw_image, detected_objects)
            frame_with_matched_objects = visualize_matched_objects(raw_image, self.object_container, detected_objects, matches_decoded)
            masked_frame = get_masked_image(raw_image, detection_output)
            bbox_frame = draw_bounding_boxes(raw_image, detection_output)
            

            combined_frames = combine_frames([
                frame_with_tracked_objects, 
                frame_with_detected_objects, 
                masked_frame,
                bbox_frame
            ])

        return combined_frames, frame_with_matched_objects

    def process_raw_image(self, raw_image):
        raw_image = raw_image.squeeze(0)
        raw_image = raw_image.permute(1, 2, 0).numpy()
        raw_image = (raw_image * 255).astype(np.uint8)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        return raw_image

    def get_object_3d_location(self):
        T_left_to_right = T_right - T_left

        Depths = -camera_calibration_matrix_left[0, 0] * T_left_to_right[0] * 2 / self.disparity

        for object in self.object_container.items():
            object_id = object[0]
            bbox = object[1].bbox
            # average_depth = np.mean(Depths[bbox.position[1]-round(bbox.height/9):bbox.position[1]+round(bbox.height/9),
            #                         bbox.position[0]-round(bbox.width/9):bbox.position[0]+round(bbox.width/9)])
            average_depth = np.mean(
                Depths[bbox.position[1] -2:bbox.position[1] + 2,
                bbox.position[0] - 2 : bbox.position[0] + 2])

            R_left_to_right = np.linalg.inv(R_left)@R_right



            imageSize = np.array([1.392000e+03, 5.120000e+02], dtype=int)

            # R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(cameraMatrix1=camera_calibration_matrix_left, distCoeffs1=camera_distortion_coeff_left,
            #                                         cameraMatrix2=camera_calibration_matrix_right,distCoeffs2=camera_distortion_coeff_right,
            #                                         imageSize=imageSize,R=R_left_to_right,T=T_left_to_right,newImageSize=[1224,370])
            # points = cv2.reprojectImageTo3D(disparity=self.disparity,Q=Q)
            # def min_max_normalize_positive(data):
            #     data = np.array(data)
            #     # Mask positive values
            #     positive_mask = data > 0
            #     positive_values = data[positive_mask]
            #
            #     if len(positive_values) > 0:
            #         # Perform Min-Max normalization on positive values
            #         min_val = positive_values.min()
            #         max_val = positive_values.max()
            #
            #         # Avoid division by zero
            #         if max_val - min_val > 0:
            #             normalized_values = (positive_values - min_val) / (max_val - min_val)
            #         else:
            #             normalized_values = positive_values  # All values are the same
            #
            #         # Replace normalized positive values back into the original array
            #         data[positive_mask] = normalized_values
            #         data[data < 0] = 1
            #
            #     return data

            # dis = np.where(np.abs(self.disparity) <= 1, 1, self.disparity)
                       # def equations(vars):
            self.object_container[object_id].depth = average_depth*5.9
            print(average_depth)


    def run(self):
        """Run object tracking on all frames."""
        for left_image, right_image, left_raw_img, right_raw_img, left_img_path, right_img_path in tqdm(self.object_detector.dataloader):

            left_raw_img = self.process_raw_image(left_raw_img)
            right_raw_img = self.process_raw_image(right_raw_img)

            disparity = self.depth_manager.get_disparity_map(left_raw_img, right_raw_img)





            combined_frames, frame_with_matched_objects = self.process_frame(left_image, left_raw_img, left_img_path, disparity)

            # cv2.namedWindow("Frame with combined_frames", cv2.WINDOW_NORMAL)
            # cv2.imshow("Frame with combined_frames", combined_frames)

            cv2.namedWindow("Frame with matched objects", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame with matched objects", frame_with_matched_objects)

            cv2.namedWindow('Disparity Map', cv2.WINDOW_NORMAL)
            cv2.imshow('Disparity Map', cv2.normalize(self.disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Quit
                break

def main():
    sequence_number = 1
    tracker = ObjectTracker(sequence_number=sequence_number)
    tracker.run()

if __name__ == '__main__':
    main()