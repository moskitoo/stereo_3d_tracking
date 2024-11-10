import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from time import sleep


class ImagePublisherNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # Declare and get the parameter for image directory
        self.declare_parameter('image_dir', '')
        self.image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
        
        if not self.image_dir:
            self.get_logger().error("No image directory provided.")
            rclpy.shutdown()
            return

        self.bridge = CvBridge()

        # topic_name = f"{self.get_name()}/image"
        topic_name = f"{self.get_name()}/image_rect"

        self.get_logger().info(f'topic name: {topic_name}')

        self.publisher = self.create_publisher(Image, topic_name, 10)
        
        # Get all image files from the directory
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.image_files.sort()  # Sort the images so they are published in order
        
        # Create a timer to publish images every 0.1s
        self.timer = self.create_timer(0.1, self.publish_image)
        self.image_index = 0  # Track the current image index for sequential publishing

    def publish_image(self):
        while True:
            image_file = self.image_files[self.image_index]
            image_path = os.path.join(self.image_dir, image_file)
            cv_image = cv2.imread(image_path)
            if cv_image is not None:
                # Convert to ROS Image message
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.publisher.publish(ros_image)
                self.get_logger().info(f'Publishing image: {image_file}')
                sleep(0.1)  # Control the rate of publishing (10 FPS)
            self.image_index += 1
            if self.image_index == len(self.image_files):
                self.image_index = 0


def main(args=None):
    rclpy.init(args=args)

    node = ImagePublisherNode('image_publisher_node')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
