import cv2
import numpy as np

camera_projection_matrix_left = np.array(
    [
        [7.070493e02, 0.000000e00, 6.040814e02, 4.575831e01],
        [0.000000e00, 7.070493e02, 1.805066e02, -3.454157e-01],
        [0.000000e00, 0.000000e00, 1.000000e00, 4.981016e-03],
    ]
)

camera_calibration_matrix_left = np.array(
    [
        [9.569475e02, 0.000000e00, 6.939767e02],
        [0.000000e00, 9.522352e02, 2.386081e02],
        [0.000000e00, 0.000000e00, 1.000000e00],
    ]
)

camera_calibration_matrix_right = np.array(
    [
        [9.011007e02, 0.000000e00, 6.982947e02],
        [0.000000e00, 8.970639e02, 2.377447e02],
        [0.000000e00, 0.000000e00, 1.000000e00],
    ]
)

camera_distortion_coeff_left = np.array(
    [-3.750956e-01, 2.076838e-01, 4.348525e-04, 1.603162e-03, -7.469243e-02]
)
camera_distortion_coeff_right = np.array(
    [-3.686011e-01, 1.908666e-01, -5.689518e-04, 3.332341e-04, -6.302873e-02]
)

R_left = np.array(
    [
        [9.999838e-01, -5.012736e-03, -2.710741e-03],
        [5.002007e-03, 9.999797e-01, -3.950381e-03],
        [2.730489e-03, 3.936758e-03, 9.999885e-01],
    ]
)
R_right = np.array(
    [
        [9.995054e-01, 1.665288e-02, -2.667675e-02],
        [-1.671777e-02, 9.998578e-01, -2.211228e-03],
        [2.663614e-02, 2.656110e-03, 9.996417e-01],
    ]
)

T_left = np.array([5.989688e-02, -1.367835e-03, 4.637624e-03])
T_right = np.array([-4.756270e-01, 5.296617e-03, -5.437198e-03])

R_left_to_right = np.linalg.inv(R_left) @ R_right

imageSize = np.array([1.392000e03, 5.120000e02], dtype=int)
T_left_to_right = T_right - T_left
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1=camera_calibration_matrix_left,
    distCoeffs1=camera_distortion_coeff_left,
    cameraMatrix2=camera_calibration_matrix_right,
    distCoeffs2=camera_distortion_coeff_right,
    imageSize=imageSize,
    R=R_left_to_right,
    T=T_left_to_right,
    newImageSize=[1224, 370],
)


class DepthManager:
    def __init__(self):
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=2,
            numDisparities=16 * 7,  # Must be divisible by 16
            blockSize=6,
            P1=8 * 1 * 6**2,  # 8 * number_of_channels * blockSize^2
            P2=64 * 1 * 6**2,  # 32 * number_of_channels * blockSize^2
            disp12MaxDiff=2000,
            uniquenessRatio=5,
            speckleWindowSize=32,
            speckleRange=128,
        )

    def update_disparity_map(self, left_image, right_image):
        gamma = 0.5
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        left_image = cv2.cvtColor(cv2.LUT(left_image, lookUpTable), cv2.COLOR_RGB2GRAY)
        right_image = cv2.cvtColor(
            cv2.LUT(right_image, lookUpTable), cv2.COLOR_RGB2GRAY
        )

        # Check if images are loaded
        if left_image is None or right_image is None:
            raise FileNotFoundError("Stereo images not found!")

        # Compute the disparity map
        self.disparity = self.stereo.compute(left_image, right_image)

        self.points = cv2.reprojectImageTo3D(
            disparity=self.disparity.astype("f") / 16, Q=Q
        )

        # Normalize the disparity map for visualization
        self.disparity_normalized = cv2.normalize(
            self.disparity,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

    def img_2d_to_world_3d(self, bbox):
        average_position = np.mean(
            self.points[
                bbox.position[1] - 2 : bbox.position[1] + 2,
                bbox.position[0] - 2 : bbox.position[0] + 2,
            ],
            axis=(0, 1),
        )

        return average_position

    def position_img_2d_to_world_3d(self, position):
        average_position = np.mean(
            self.points[
                position[1] - 2 : position[1] + 2, position[0] - 2 : position[0] + 2
            ],
            axis=(0, 1),
        )

        return average_position

    def world_3d_to_img_2D(self, world_point):
        world_point_homogeneous = np.append(world_point, 1)

        image_point = P1 @ world_point_homogeneous

        image_point /= image_point[2]

        return image_point[:2]
