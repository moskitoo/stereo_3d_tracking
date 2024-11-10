import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from rclpy.parameter import Parameter
from time import sleep


class ImagePublisherNode(Node):
    def __init__(self, name, image_dir):
        super().__init__(name)
        self.image_dir = image_dir
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, 'camera/image', 10)
        
        # Get all image files from the directory
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.image_files.sort()  # Sort the images so they are published in order
        
        # Create a timer to publish images every 0.1s
        self.timer = self.create_timer(0.1, self.publish_image)

    def publish_image(self):
        for image_file in self.image_files:
            image_path = os.path.join(self.image_dir, image_file)
            cv_image = cv2.imread(image_path)
            if cv_image is not None:
                # Convert to ROS Image message
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.publisher.publish(ros_image)
                self.get_logger().info(f'Publishing image: {image_file}')
                sleep(0.1)  # Control the rate of publishing (10 FPS)


def main(args=None):
    rclpy.init(args=args)

    # Define image directories for multiple nodes
    image_dirs = [
        '/home/moskit/dtu/perception_final_project_ws/data/34759_final_project_rect/seq_01/image_02/data',
        # Add more paths if needed
    ]

    nodes = []
    for i, image_dir in enumerate(image_dirs):
        node = ImagePublisherNode(f'image_publisher_{i}', image_dir)
        nodes.append(node)

    try:
        # Spin all nodes
        rclpy.spin(nodes[0])  # This will spin the first node and block execution
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up and shut down nodes
        for node in nodes:
            node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
