#!/usr/bin/env python3
"""
TidyBot2 Motion Planner Test for Real Hardware

Now: listens for /right_arm_target_xyz (Float64MultiArray: [x,y,z])
and moves the RIGHT arm there with use_orientation=False.

Usage:
    ros2 launch tidybot_bringup real.launch.py use_planner:=true
    ros2 run tidybot_bringup test_planner_real.py

Send a target:
    ros2 topic pub --once /right_arm_target_xyz std_msgs/msg/Float64MultiArray "{data: [0.1, -0.5, 0.3]}"
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from tidybot_msgs.srv import PlanToTarget
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from std_msgs.msg import Float64MultiArray
import numpy as np
import time


class TestPlannerReal(Node):
    """Node that moves right arm to XYZ targets from a topic."""

    SLEEP_POSE = [0.0, -1.80, 1.55, 0.0, 0.8, 0.0]

    def __init__(self):
        super().__init__('test_planner_real')

        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        self.arm_cmd_pubs = {
            'right': self.create_publisher(JointGroupCommand, '/right_arm/commands/joint_group', 10),
            'left': self.create_publisher(JointGroupCommand, '/left_arm/commands/joint_group', 10),
        }

        self.right_gripper_pub = self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10)

        self.joint_states_received = False
        self.current_joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self._js_callback, 10)

        # NEW: subscribe to xyz commands
        self.create_subscription(  # ADDED
            Float64MultiArray, '/right_arm_target_xyz', self._xyz_callback, 10
        )

        self.get_logger().info('=' * 50)
        self.get_logger().info('TidyBot2 IK Planner (Real Hardware) - XYZ topic control')
        self.get_logger().info('Publish [x,y,z] to /right_arm_target_xyz to move (use_orientation=False)')
        self.get_logger().info('=' * 50)

        self.get_logger().info('Waiting for /plan_to_target service...')
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service not available! Make sure motion_planner_real_node is running.')
            self.get_logger().error('Launch with: ros2 launch tidybot_bringup real.launch.py use_planner:=true')
            raise RuntimeError('Planning service not available')
        self.get_logger().info('Service connected!')

        self.get_logger().info('Waiting for joint states...')
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_states_received:
                break
        if not self.joint_states_received:
            self.get_logger().warn('No joint states received - proceeding anyway')

        self.get_logger().info('Ready. Example:')
        self.get_logger().info('  ros2 topic pub --once /right_arm_target_xyz std_msgs/msg/Float64MultiArray "{data: [0.1, -0.5, 0.3]}"')

    def _js_callback(self, msg):
        self.joint_states_received = True
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def _xyz_callback(self, msg: Float64MultiArray):  # ADDED
        if len(msg.data) < 3:
            self.get_logger().warn('Received /right_arm_target_xyz with <3 values; expected [x,y,z]')
            return
        x, y, z = float(msg.data[0]), float(msg.data[1]), float(msg.data[2])
        self.get_logger().info(f'Received target XYZ: ({x:.3f}, {y:.3f}, {z:.3f})')
        pose = self.create_pose(x, y, z)
        self.plan_and_execute('right', pose, use_orientation=False, duration=3.0)

    def create_pose(self, x: float, y: float, z: float,
                    qw: float = 1.0, qx: float = 0.0,
                    qy: float = 0.0, qz: float = 0.0) -> Pose:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = qw
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        return pose

    def call_service_sync(self, request, timeout_sec=15.0):
        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if not future.done():
            self.get_logger().error('Service call timed out!')
            return None
        if future.exception() is not None:
            self.get_logger().error(f'Service call exception: {future.exception()}')
            return None
        return future.result()

    def plan_and_execute(self, arm_name: str, pose: Pose,
                         use_orientation: bool = True,
                         duration: float = 3.0) -> bool:
        request = PlanToTarget.Request()
        request.arm_name = arm_name
        request.target_pose = pose
        request.use_orientation = use_orientation
        request.execute = True
        request.duration = duration
        request.max_condition_number = 100.0

        self.get_logger().info(
            f'Planning and executing {arm_name} arm to: '
            f'({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}) '
            f'use_orientation={use_orientation}'
        )

        result = self.call_service_sync(request)
        if result is None:
            return False

        if result.success:
            self.get_logger().info(f'  SUCCESS: {result.message}')
            if result.executed:
                self.get_logger().info(f'  Executing over {duration}s...')
            return True
        else:
            self.get_logger().warn(f'  FAILED: {result.message}')
            return False

    def get_arm_positions(self, arm_name: str) -> np.ndarray:
        joint_names = [f'{arm_name}_waist', f'{arm_name}_shoulder', f'{arm_name}_elbow',
                       f'{arm_name}_forearm_roll', f'{arm_name}_wrist_angle', f'{arm_name}_wrist_rotate']
        positions = np.array([self.current_joint_positions.get(jname, 0.0) for jname in joint_names])
        return positions

    def go_to_sleep_pose(self, arm_name: str, max_joint_speed: float = 0.5):
        rclpy.spin_once(self, timeout_sec=0.1)
        current = self.get_arm_positions(arm_name)
        target = np.array(self.SLEEP_POSE)

        max_diff = np.max(np.abs(target - current))
        duration = max(max_diff / max_joint_speed, 1.0)

        self.get_logger().info(f'Moving {arm_name} arm to sleep pose over {duration:.1f}s')

        rate_hz = 50.0
        dt = 1.0 / rate_hz
        num_steps = max(int(duration * rate_hz), 1)

        for i in range(num_steps + 1):
            t = i / num_steps
            alpha = 0.5 * (1 - np.cos(np.pi * t))
            q = current + alpha * (target - current)

            cmd = JointGroupCommand()
            cmd.name = f'{arm_name}_arm'
            cmd.cmd = q.tolist()
            self.arm_cmd_pubs[arm_name].publish(cmd)

            if i < num_steps:
                time.sleep(dt)


def main(args=None):
    rclpy.init(args=args)
    node = TestPlannerReal()
    try:
        rclpy.spin(node)  # CHANGED: spin forever, respond to topic commands
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
