#!/usr/bin/env python3
"""
approach_node.py

Subscribe to an object PoseStamped in the *world* frame (the same frame the
position controller expects). Compute an approach pose at `approach_distance`
from the object (so the robot will face the object), convert to the Pose2D
format accepted by phoenix6_base_node (/base/target_pose), and publish it.

Also subscribes to /base/goal_reached (Bool) and logs when goal is reached.

Run:
  python3 tidybot_bringup/scripts/approach_node.py
or
  ros2 run tidybot_bringup approach_node   # after adding entrypoint and installing package

Parameters (ROS2 parameters / can be set on CLI):
  object_topic (str)           topic for object PoseStamped (world frame). default: /detected_object
  approach_distance (float)    meters to stop away from object. default: 0.30
  publish_target_topic (str)   where to publish Pose2D for base. default: /base/target_pose
  sample_mode (bool)           if True and no object seen, publish a single sample target. default: True
  sample_pose_x, sample_pose_y (floats) sample pose in world frame used in sample_mode.
"""

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def quat_to_yaw(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ApproachNode(Node):
    def __init__(self):
        super().__init__('approach_node')

        # params
        self.declare_parameter('object_topic', '/detected_object')
        self.declare_parameter('approach_distance', 0.30)
        self.declare_parameter('publish_target_topic', '/base/target_pose')
        self.declare_parameter('sample_mode', False)
        self.declare_parameter('sample_pose_x', 1.0)
        self.declare_parameter('sample_pose_y', 1.0)
        self.declare_parameter('odom_topic', '/odom')  # used to know robot pose (optional)
        self.declare_parameter('sample_once', True)    # if True, publish sample just once

        self.object_topic = self.get_parameter('object_topic').value
        self.approach_distance = float(self.get_parameter('approach_distance').value)
        self.publish_target_topic = self.get_parameter('publish_target_topic').value
        self.sample_mode = bool(self.get_parameter('sample_mode').value)
        self.sample_pose_x = float(self.get_parameter('sample_pose_x').value)
        self.sample_pose_y = float(self.get_parameter('sample_pose_y').value)
        self.odom_topic = self.get_parameter('odom_topic').value
        self.sample_once = bool(self.get_parameter('sample_once').value)

        # state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_th = 0.0
        self.odom_seen = False

        self.last_object_stamp = None
        self.last_object_pose: Optional[PoseStamped] = None
        self.sent_sample = False
        self.wait_for_goal_reached = False

        # subs / pubs
        self.create_subscription(PoseStamped, self.object_topic, self.object_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.goal_sub = self.create_subscription(Bool, '/base/goal_reached', self.goal_cb, 10)

        self.target_pub = self.create_publisher(Pose2D, self.publish_target_topic, 10)

        # periodic
        self.create_timer(0.5, self.timer_cb)

        self.get_logger().info(f'ApproachNode listening for objects on: {self.object_topic}')
        self.get_logger().info(f'Publishing Pose2D targets to: {self.publish_target_topic}')
        if self.sample_mode:
            self.get_logger().info(f'Sample mode enabled: sample pose ({self.sample_pose_x}, {self.sample_pose_y})')

    # callbacks
    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.current_x = float(p.x)
        self.current_y = float(p.y)
        self.current_th = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.odom_seen = True

    def object_cb(self, msg: PoseStamped):
        # expecting object pose in world frame (the same world frame the controller expects)
        self.last_object_stamp = self.get_clock().now()
        self.last_object_pose = msg
        self.get_logger().info(f'Received object at world ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}), frame={msg.header.frame_id}')
        self.compute_and_send(msg)

    def goal_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().info('Controller reported: goal_reached == True')
            # optionally do something else (publish event, chain next action)
            self.wait_for_goal_reached = False

    # periodic timer
    def timer_cb(self):
        # If we have received an object recently we already sent target in object_cb
        if self.last_object_pose is not None:
            # optionally check staleness; keep as-is
            return

        # else if sample mode, and haven't sent sample, publish a single sample
        if self.sample_mode and not self.sent_sample and self.odom_seen:
            self.get_logger().info('No object detected yet — publishing sample approach target.')
            sample = PoseStamped()
            sample.header.stamp = self.get_clock().now().to_msg()
            sample.header.frame_id = 'world'   # user said "world" frame
            sample.pose.position.x = self.sample_pose_x
            sample.pose.position.y = self.sample_pose_y
            sample.pose.position.z = 0.0
            sample.pose.orientation.x = 0.0
            sample.pose.orientation.y = 0.0
            sample.pose.orientation.z = 0.0
            sample.pose.orientation.w = 1.0
            self.compute_and_send(sample)
            self.sent_sample = self.sample_once

    # main logic
    def compute_and_send(self, pose_stamped: PoseStamped):
        # sanity: require odom known (to compute approach vector if desired)
        if not self.odom_seen:
            self.get_logger().warn('No odom available yet; cannot compute approach. Waiting...')
            return

        # check frame — we expect 'world' frame (user requested). If header not 'world', we still accept but warn.
        if pose_stamped.header.frame_id != 'world':
            self.get_logger().warn(f"Object pose frame is '{pose_stamped.header.frame_id}', expected 'world'. "
                                   "Make sure pose is provided in world frame (controller expects world coords).")

        obj_x = float(pose_stamped.pose.position.x)
        obj_y = float(pose_stamped.pose.position.y)

        # vector robot -> object in world frame
        dx = obj_x - self.current_x
        dy = obj_y - self.current_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            yaw_to_object = 0.0
        else:
            yaw_to_object = math.atan2(dy, dx)

        # approach point (world coords): back off by approach_distance along line from object -> robot
        approach_x = obj_x - self.approach_distance * math.cos(yaw_to_object)
        approach_y = obj_y - self.approach_distance * math.sin(yaw_to_object)
        approach_theta = yaw_to_object  # face the object when arriving

        self.get_logger().info(f'Approach point (world): x={approach_x:.3f}, y={approach_y:.3f}, theta={approach_theta:.3f}')

        target = Pose2D()
        target.x = approach_x
        target.y = approach_y
        target.theta = approach_theta

        self.target_pub.publish(target)
        self.get_logger().info(f'Published Pose2D target: x(front)={target.x:.3f}, y(left)={target.y:.3f}, theta={target.theta:.3f}')
        # mark that we are waiting for goal_reached notification
        self.wait_for_goal_reached = True

    # helper: short sleep wrapper
    def sleep(self, sec: float):
        t0 = time.time()
        while time.time() - t0 < sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)


def main(args=None):
    rclpy.init(args=args)
    node = ApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()