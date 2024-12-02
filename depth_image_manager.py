import cv2
import numpy as np

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

R_left_to_right = np.linalg.inv(R_left) @ R_right

imageSize = np.array([1.392000e+03, 5.120000e+02], dtype=int)


class DepthManager():
    def __init__(self):
        # self.stereo = cv2.StereoSGBM_create(
        #     minDisparity=1,
        #     numDisparities=16 * 6,  # Must be divisible by 16
        #     blockSize=7,
        #     P1=8 * 1 * 7**2,  # 8 * number_of_channels * blockSize^2
        #     P2=32 * 1 * 7**2,  # 32 * number_of_channels * blockSize^2
        #     disp12MaxDiff=1,
        #     uniquenessRatio=10,
        #     speckleWindowSize=100,
        #     speckleRange=32
        # )
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=15,
            numDisparities=16 * 10,  # Must be divisible by 16
            blockSize=7,
            P1=8 * 1 * 7 ** 2,  # 8 * number_of_channels * blockSize^2
            P2=32 * 1 * 7 ** 2,  # 32 * number_of_channels * blockSize^2
            disp12MaxDiff=2000,
            uniquenessRatio=12,
            speckleWindowSize=100,
            speckleRange=64)

    def get_disparity_map(self, left_image, right_image):

        left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)

        # Check if images are loaded
        if left_image is None or right_image is None:
            raise FileNotFoundError("Stereo images not found!")        

        # Compute the disparity map
        disparity = self.stereo.compute(left_image, right_image)

        # Normalize the disparity map for visualization
        # disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return disparity

def get_object_3d_location(self):
    T_left_to_right = T_right - T_left

    Depths = -camera_calibration_matrix_left[0, 0] * T_left_to_right[0]/ (self.disparity)

    R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(cameraMatrix1=camera_calibration_matrix_left, distCoeffs1=camera_distortion_coeff_left,
                                            cameraMatrix2=camera_calibration_matrix_right,distCoeffs2=camera_distortion_coeff_right,
                                            imageSize=imageSize,R=R_left_to_right,T=T_left_to_right,newImageSize=[1224,370])
    points = cv2.reprojectImageTo3D(disparity=self.disparity.astype("f")/16,Q=Q)

    for object in self.object_container.items():
        object_id = object[0]
        bbox = object[1].bbox
        # average_depth = np.mean(Depths[bbox.position[1]-round(bbox.height/9):bbox.position[1]+round(bbox.height/9),
        #                         bbox.position[0]-round(bbox.width/9):bbox.position[0]+round(bbox.width/9)])
        # average_depth = np.mean(
        #     Depths[bbox.position[1] -2:bbox.position[1] + 2,
        #     bbox.position[0] - 2 : bbox.position[0] + 2])

        average_position = np.mean(
            points[bbox.position[1] -2:bbox.position[1] + 2,
            bbox.position[0] - 2 : bbox.position[0] + 2],axis = (0,1))
        # self.object_container[object_id].depth = average_depth*5.9
        self.object_container[object_id].world_3d_position = average_position
        # print(average_depth)
        # print(average_position)

    # def update_kalman_filter(self, detected_object=None):





