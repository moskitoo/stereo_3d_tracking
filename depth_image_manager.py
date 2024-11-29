import cv2
import numpy as np
from numpy.ma.extras import average


class DepthManager():
    def __init__(self):
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 6,  # Must be divisible by 16
            blockSize=7,
            P1=8 * 1 * 7**2,  # 8 * number_of_channels * blockSize^2
            P2=32 * 1 * 7**2,  # 32 * number_of_channels * blockSize^2
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def get_disparity_map(self, left_image, right_image):

        left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)

        # Check if images are loaded
        if left_image is None or right_image is None:
            raise FileNotFoundError("Stereo images not found!")        

        # Compute the disparity map
        disparity = self.stereo.compute(left_image, right_image)

        # Normalize the disparity map for visualization
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return disparity_normalized



