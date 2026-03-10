#!/usr/bin/env python3
"""
Text-based task input — replaces voice pipeline for testing.

Usage:
    ros2 run tidybot_bringup text_task_input.py

    # Then type commands like:
    #   Task2 banana
    #   Task3
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tidybot_msgs.msg import TaskRequest
import threading


class TextTaskInput(Node):
    def __init__(self):
        super().__init__('text_task_input')
        self.pub = self.create_publisher(TaskRequest, '/task_request', 10)
        self.get_logger().info('Text task input ready. Format: <TaskType> <object>')
        self.get_logger().info('  e.g.  Task2 banana')
        self.get_logger().info('  e.g.  Task3')

    def publish_task(self, task_type: str, obj: str):
        msg = TaskRequest()
        msg.task_type = String(data=task_type)
        msg.object = String(data=obj)
        self.pub.publish(msg)
        self.get_logger().info(f'Published: task_type={task_type}, object={obj}')


def main():
    rclpy.init()
    node = TextTaskInput()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while True:
            line = input('> ').strip()
            if not line:
                continue
            if line.lower() in ('quit', 'exit', 'q'):
                break

            parts = line.split(None, 1)
            task_type = parts[0]
            obj = parts[1] if len(parts) > 1 else ''
            node.publish_task(task_type, obj)
    except (EOFError, KeyboardInterrupt):
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
