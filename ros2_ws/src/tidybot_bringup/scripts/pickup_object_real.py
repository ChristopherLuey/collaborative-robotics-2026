#!/usr/bin/env python3
"""
TidyBot2 Object Pickup Script for Real Hardware

Executes a pick-up sequence using the right arm and /plan_to_target IK service.
Takes a 4D target (x, y, z, yaw) as CLI arguments.

Usage:
    # Terminal 1: Start real hardware with motion planner
    ros2 launch tidybot_bringup real.launch.py use_planner:=true

    # Terminal 2: Run pickup
    ros2 run tidybot_bringup pickup_object_real.py -- 0.1 -0.4 0.15 0.0
"""

import sys
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from interbotix_xs_msgs.msg import JointGroupCommand
from tidybot_msgs.srv import PlanToTarget


class PickupObjectReal(Node):
    """Pick up an object at a given (x, y, z, yaw) using the right arm."""

    SLEEP_POSE = [0.0, -1.80, 1.55, 0.0, 0.8, 0.0]

    def __init__(self):
        super().__init__('pickup_object_real')

        # Service client for IK planning
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        # Arm command publisher (for sleep pose recovery)
        self.arm_cmd_pub = self.create_publisher(
            JointGroupCommand, '/right_arm/commands/joint_group', 10
        )

        # Gripper publisher
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, '/right_gripper/cmd', 10
        )

        # Joint state tracking
        self.joint_states_received = False
        self.current_joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self._js_callback, 10)

        self.get_logger().info('=' * 50)
        self.get_logger().info('TidyBot2 Object Pickup (Real Hardware)')
        self.get_logger().info('=' * 50)

        # Wait for service
        self.get_logger().info('Waiting for /plan_to_target service...')
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service not available! Launch with: ros2 launch tidybot_bringup real.launch.py use_planner:=true')
            raise RuntimeError('Planning service not available')
        self.get_logger().info('Service connected!')

        # Wait for joint states
        self.get_logger().info('Waiting for joint states...')
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_states_received:
                break
        if not self.joint_states_received:
            self.get_logger().warn('No joint states received - proceeding anyway')

    def _js_callback(self, msg):
        self.joint_states_received = True
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def call_service_sync(self, request, timeout_sec=15.0):
        """Call service synchronously."""
        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if not future.done():
            self.get_logger().error('Service call timed out!')
            return None
        if future.exception() is not None:
            self.get_logger().error(f'Service call exception: {future.exception()}')
            return None
        return future.result()

    def create_pose(self, x, y, z, yaw):
        """Create a Pose message with yaw converted to quaternion."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = math.cos(yaw / 2.0)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = math.sin(yaw / 2.0)
        return pose

    def plan_and_execute(self, pose, duration=3.0):
        """Plan and execute a move to pose using the right arm."""
        request = PlanToTarget.Request()
        request.arm_name = 'right'
        request.target_pose = pose
        request.use_orientation = False
        request.execute = True
        request.duration = duration
        request.max_condition_number = 100.0

        self.get_logger().info(
            f'Moving right arm to: '
            f'({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})'
        )

        result = self.call_service_sync(request)
        if result is None:
            return False

        if result.success:
            self.get_logger().info(f'  SUCCESS: {result.message}')
            return True
        else:
            self.get_logger().warn(f'  FAILED: {result.message}')
            return False

    def set_gripper(self, position, duration=2.0):
        """Set right gripper position (0.0=open, 1.0=closed)."""
        msg = Float64MultiArray()
        msg.data = [float(position)]

        start = time.time()
        while (time.time() - start) < duration:
            self.gripper_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.1)  # 10 Hz

        # Send stop command
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.5]
        self.gripper_pub.publish(stop_msg)
        rclpy.spin_once(self, timeout_sec=0.05)

    def get_arm_positions(self):
        """Get current right arm joint positions."""
        joint_names = [
            'right_waist', 'right_shoulder', 'right_elbow',
            'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate'
        ]
        return np.array([self.current_joint_positions.get(n, 0.0) for n in joint_names])

    def go_to_sleep_pose(self, max_joint_speed=0.5):
        """Send right arm to sleep pose with cosine interpolation."""
        rclpy.spin_once(self, timeout_sec=0.1)

        current = self.get_arm_positions()
        target = np.array(self.SLEEP_POSE)

        max_diff = np.max(np.abs(target - current))
        duration = max(max_diff / max_joint_speed, 1.0)

        self.get_logger().info(f'Returning to sleep pose over {duration:.1f}s')

        rate_hz = 50.0
        dt = 1.0 / rate_hz
        num_steps = max(int(duration * rate_hz), 1)

        for i in range(num_steps + 1):
            t = i / num_steps
            alpha = 0.5 * (1 - np.cos(np.pi * t))
            q = current + alpha * (target - current)

            cmd = JointGroupCommand()
            cmd.name = 'right_arm'
            cmd.cmd = q.tolist()
            self.arm_cmd_pub.publish(cmd)

            if i < num_steps:
                time.sleep(dt)

    def run_pickup(self, x, y, z, yaw):
        """Execute the pickup sequence."""
        self.get_logger().info(f'Target: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw:.3f}')
        self.get_logger().info('')

        # Step 1: Open gripper
        self.get_logger().info('[Step 1/6] Opening gripper...')
        self.set_gripper(0.0, duration=2.0)
        time.sleep(1.0)

        # Step 2: Move to hover position (5cm above target)
        self.get_logger().info('[Step 2/6] Moving to hover position...')
        hover_pose = self.create_pose(x, y, z + 0.05, yaw)
        if not self.plan_and_execute(hover_pose):
            self.get_logger().error('Failed to reach hover position. Aborting.')
            self.go_to_sleep_pose()
            return False
        time.sleep(2.0)

        # Step 3: Descend to grasp position
        self.get_logger().info('[Step 3/6] Descending to grasp position...')
        grasp_pose = self.create_pose(x, y, z, yaw)
        if not self.plan_and_execute(grasp_pose):
            self.get_logger().error('Failed to reach grasp position. Aborting.')
            self.go_to_sleep_pose()
            return False
        time.sleep(2.0)

        # Step 4: Close gripper
        self.get_logger().info('[Step 4/6] Closing gripper...')
        self.set_gripper(1.0, duration=2.0)
        time.sleep(2.0)

        # Step 5: Lift object (10cm above target)
        self.get_logger().info('[Step 5/6] Lifting object...')
        lift_pose = self.create_pose(x, y, z + 0.10, yaw)
        if not self.plan_and_execute(lift_pose):
            self.get_logger().error('Failed to lift. Returning to sleep.')
            self.go_to_sleep_pose()
            return False
        time.sleep(2.0)

        # Step 6: Return to sleep pose
        self.get_logger().info('[Step 6/6] Returning to sleep pose...')
        self.go_to_sleep_pose()

        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Pickup complete!')
        self.get_logger().info('=' * 50)
        return True


def main(args=None):
    rclpy.init(args=args)

    # Parse CLI arguments (after ROS strips its own args)
    cli_args = rclpy.utilities.remove_ros_args(sys.argv)

    if len(cli_args) != 5:
        print(f'Usage: {cli_args[0]} <x> <y> <z> <yaw>')
        print(f'Example: ros2 run tidybot_bringup pickup_object_real.py -- 0.1 -0.4 0.15 0.0')
        rclpy.shutdown()
        sys.exit(1)

    try:
        x = float(cli_args[1])
        y = float(cli_args[2])
        z = float(cli_args[3])
        yaw = float(cli_args[4])
    except ValueError as e:
        print(f'Error: all arguments must be floats: {e}')
        rclpy.shutdown()
        sys.exit(1)

    node = PickupObjectReal()
    try:
        node.run_pickup(x, y, z, yaw)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted - returning to sleep pose...')
        node.go_to_sleep_pose()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
