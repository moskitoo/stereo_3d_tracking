from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the full package path
    package_dir = get_package_share_directory('stereo_3d_tracking')
    
    workspace_root = os.path.abspath(os.path.join(package_dir, os.pardir, os.pardir, os.pardir, os.pardir))
    
    # Define image directories
    left_image_dir = os.path.join(workspace_root, 'data/34759_final_project_rect/seq_01/image_02/data')
    right_image_dir = os.path.join(workspace_root, 'data/34759_final_project_rect/seq_01/image_03/data')
    
    # Define calibration file path
    calib_file = os.path.join(workspace_root, 'data/34759_final_project_rect/calib_cam_to_cam.txt')
    
    return LaunchDescription([
        # Left image publisher
        Node(
            package='stereo_3d_tracking',
            executable='image_publisher_node',
            name='left',
            parameters=[
                {'image_dir': left_image_dir}
            ]
        ),
        
        # Right image publisher
        Node(
            package='stereo_3d_tracking',
            executable='image_publisher_node',
            name='right',
            parameters=[
                {'image_dir': right_image_dir}
            ]
        ),
        
        # Left camera calibration publisher
        Node(
            package='stereo_3d_tracking',
            executable='calibration_publisher_node',
            name='left_calib',  # Changed name
            parameters=[
                {'calib_file': calib_file}
            ],
            remappings=[
                ('/left_calib/camera_info', '/left/camera_info')  # Remap to match expected topic
            ],
            output='screen'
        ),
        
        # Right camera calibration publisher
        Node(
            package='stereo_3d_tracking',
            executable='calibration_publisher_node',
            name='right_calib',  # Changed name
            parameters=[
                {'calib_file': calib_file}
            ],
            remappings=[
                ('/right_calib/camera_info', '/right/camera_info')  # Remap to match expected topic
            ]
        ),
        
        # Disparity node    
        Node(
            package='stereo_image_proc',
            executable='disparity_node',
            name='disparity_node',
            # remappings=[
            #     ('image_left', '/left/image_rect'),
            #     ('image_right', '/right/image_rect'),
            #     ('left_camera_info', '/left/camera_info'),
            #     ('right_camera_info', '/right/camera_info')
            # ],
            output='screen'
        ),

        Node(
            package='stereo_image_proc',
            executable='point_cloud_node',
            name='point_cloud_node',
            parameters=[
                {'approximate_sync': True},
                {'queue_size': 5}
            ],
            remappings=[
                ('left/image_rect', '/left/image_rect'),
                ('right/image_rect', '/right/image_rect'),
                ('left/camera_info', '/left/camera_info'),
                ('right/camera_info', '/right/camera_info'),
                ('disparity', '/disparity')
            ]
        )
    ])