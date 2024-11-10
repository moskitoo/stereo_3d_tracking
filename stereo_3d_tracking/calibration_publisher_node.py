import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import numpy as np
import os

class CalibrationPublisher(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # Declare parameter for calibration file
        self.declare_parameter('calib_file', '')
        calib_file = self.get_parameter('calib_file').get_parameter_value().string_value
        
        if not calib_file:
            self.get_logger().error("No calibration file provided.")
            rclpy.shutdown()
            return
            
        self.get_logger().info(f'calib file: {calib_file}')
        # Read calibration data
        self.calib_data = self.read_calibration_file(calib_file)
        
        # Create publisher
        topic_name = f"{self.get_name()}/camera_info"
        self.publisher = self.create_publisher(CameraInfo, topic_name, 10)
        
        # Create timer for publishing camera info
        self.timer = self.create_timer(0.1, self.publish_camera_info)
        
    def read_calibration_file(self, file_path):
        data = {}
        current_matrix = None
        matrix_data = []
        
        with open(file_path, 'r') as f:
            for line in f:
                # self.get_logger().info(f'line: {line}')
                line = line.strip()
                # self.get_logger().info(f'line: {line}')
                if not line or line.startswith('#'):
                    continue
                    
                # Check if line starts new matrix
                # if ':' in line:
                #     if current_matrix and matrix_data:
                #         data[current_matrix] = np.array(matrix_data)
                #         self.get_logger().info(f'current_matrix: {current_matrix}')
                #         self.get_logger().info(f'matrix_data: {matrix_data}')
                #         matrix_data = []
                #     current_matrix = line.split(':')[0].strip()
                # else:
                #     # Parse space-separated numbers
                #     values = [float(x) for x in line.split()]
                #     matrix_data.append(values)
                [current_matrix, matrix_data] = line.split(':')
                matrix_data = matrix_data.split(' ')[1:]
                # self.get_logger().info(f'current_matrix: >{current_matrix}<')
                # self.get_logger().info(f'matrix_data: >{matrix_data}<')
                # self.get_logger().info('\n\n')
                # data[current_matrix] = np.array(matrix_data)
                data[current_matrix] = np.array(matrix_data, dtype=float)

                # self.get_logger().info(f'matrix_data: >{data}<')
                # self.get_logger().info('\n\n')
                
            
            # Don't forget to add the last matrix
            # if current_matrix and matrix_data:
            #     data[current_matrix] = np.array(matrix_data)

        # self.get_logger().info(f'data: {data}')
        # print(data)
                
        return data
        
    def create_camera_info_msg(self, is_left):
        msg = CameraInfo()
        
        # Set the timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set frame ID based on camera
        msg.header.frame_id = 'left_camera' if is_left else 'right_camera'
        
        # self.get_logger().info(f'type: >{type(self.calib_data)}<')
        # self.get_logger().info(f'type: >{type(self.calib_data.keys())}<')
        # for key in self.calib_data:
            # print(f"Key: {key}, Type: {type(key)}")
        # self.get_logger().info(f'calib_data: >{self.calib_data.}<')
        # self.get_logger().info(f'calib_data: >{self.calib_data}<')
        # Set image size
        if is_left:
            msg.height = int(self.calib_data["S_rect_02"][1])
            msg.width = int(self.calib_data["S_rect_02"][0])
        else:
            msg.height = int(self.calib_data["S_rect_03"][1])
            msg.width = int(self.calib_data["S_rect_03"][0])
        
        # Set rectification rotation
        rect_mat = self.calib_data["R_rect_02" if is_left else "R_rect_03"]
        msg.r = rect_mat.flatten().tolist()
        
        # Set projection matrix
        proj_mat = self.calib_data["P_rect_02" if is_left else "P_rect_03"]
        msg.p = proj_mat.flatten().tolist()
        
        # Set camera matrix (K)
        cam_mat = self.calib_data["K_02" if is_left else "K_03"]
        msg.k = cam_mat.flatten().tolist()
        
        # Set distortion parameters
        dist_coeffs = self.calib_data["D_02" if is_left else "D_03"]
        msg.d = dist_coeffs.flatten().tolist()
        
        return msg
        
    def publish_camera_info(self):
        # Create and publish camera info message
        is_left = self.get_name() == 'left'
        camera_info_msg = self.create_camera_info_msg(is_left)
        self.publisher.publish(camera_info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationPublisher('camera_info_publisher')
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()