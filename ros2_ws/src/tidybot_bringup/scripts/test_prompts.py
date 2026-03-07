#!/usr/bin/env python3

import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import String




class RandomWordPublisher(Node):

    def __init__(self):
        super().__init__('random_word_publisher')

        # List of words to choose from
        self.words = [
            "bottle",
            "keyboard",
            "person"
        ]

        # Create publisher
        self.publisher_ = self.create_publisher(String, '/prompts', 10)

        # Create timer (10 seconds)
        timer_period = 5.0  # seconds
        self.timer = self.create_timer(timer_period, self.publish_random_word)

        self.get_logger().info("Random word publisher node started.")

    def publish_random_word(self):
        word = random.choice(self.words)

        msg = String()
        msg.data = word

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{word}"')


def main(args=None):
    rclpy.init(args=args)

    node = RandomWordPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()