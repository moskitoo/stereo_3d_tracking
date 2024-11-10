from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the full package path
    package_dir = get_package_share_directory('stereo_3d_tracking')

    print(package_dir)
    
    workspace_root = os.path.abspath(os.path.join(package_dir, os.pardir, os.pardir, os.pardir, os.pardir))

    print(workspace_root)

    # Define the full data path by appending the relative path
    left_image_dir = os.path.join(workspace_root, 'data/34759_final_project_rect/seq_01/image_02/data')
    right_image_dir = os.path.join(workspace_root, 'data/34759_final_project_rect/seq_01/image_03/data')

    return LaunchDescription([
        Node(
            package='stereo_3d_tracking',
            executable='image_publisher_node',
            name='left_image_publisher',
            parameters=[
                {'image_dir': left_image_dir}  # Use dynamically calculated path
            ]
        ),
        Node(
            package='stereo_3d_tracking',
            executable='image_publisher_node',
            name='right_image_publisher',
            parameters=[
                {'image_dir': right_image_dir}  # Use dynamically calculated path
            ]
        )
    ])
