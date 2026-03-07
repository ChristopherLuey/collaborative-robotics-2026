#!/usr/bin/env python3
"""
Simple manipulation bridge for the high-level planner.

Provides one service:
  /request_arm_motion   (tidybot_msgs/srv/RequestArmMotion)

Supported motion types:
  - "move"    : move arm to target pose
  - "grab"    : move arm to target pose, then close gripper
  - "release" : move arm to target pose, then open gripper

Uses:
  - /plan_to_target
  - /right_gripper/cmd
  - /left_gripper/cmd

Usage:
  1. Launch the robot with the planner:
       ros2 launch tidybot_bringup real.launch.py use_planner:=true

  2. Run this node:
       ros2 run tidybot_bringup pickup_object_real.py

  3. Call /request_arm_motion with:
       - arm_name: "right" or "left"
       - motion_type: "move", "grab", or "release"
       - target_pose: geometry_msgs/Pose in base_link

This is intentionally simple and meant as a basic arm/gripper interface
for the high-level planner.
"""

import time
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from tidybot_msgs.srv import PlanToTarget, RequestArmMotion


GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0
GRIPPER_STOP = 0.5


class SimpleManipulationNode(Node):
    def __init__(self):
        super().__init__('simple_manipulation_node')

        # Planner service client
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        # Gripper publishers
        self.gripper_cmd_pubs = {
            'right': self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10),
            'left': self.create_publisher(Float64MultiArray, '/left_gripper/cmd', 10),
        }

        # Public service for higher-level code
        self.motion_service = self.create_service(
            RequestArmMotion,
            '/request_arm_motion',
            self._handle_request_arm_motion,
        )

        self.get_logger().info('=' * 60)
        self.get_logger().info('Simple Manipulation Node')
        self.get_logger().info('Service: /request_arm_motion')
        self.get_logger().info('Motion types: move, grab, release')
        self.get_logger().info('=' * 60)

        self.get_logger().info('Waiting for /plan_to_target...')
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/plan_to_target not available')
            raise RuntimeError('/plan_to_target not available')

        self.get_logger().info('Connected to /plan_to_target')
        self.get_logger().info('Ready.')

    # --------------------------------------------------
    # Public ROS service
    # --------------------------------------------------

    def _handle_request_arm_motion(
        self,
        request: RequestArmMotion.Request,
        response: RequestArmMotion.Response,
    ) -> RequestArmMotion.Response:
        arm_name = request.arm_name.strip().lower()
        motion_type = request.motion_type.strip().lower()
        target_pose = request.target_pose

        response.success = False
        response.executed = False
        response.message = ''

        if arm_name not in ['right', 'left']:
            response.message = f'Invalid arm_name: {arm_name}'
            return response

        if motion_type not in ['move', 'grab', 'release']:
            response.message = f'Invalid motion_type: {motion_type}'
            return response

        self.get_logger().info(
            f'RequestArmMotion: arm={arm_name}, motion={motion_type}, '
            f'pos=({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f})'
        )

        # Route request to the matching helper
        if motion_type == 'move':
            result = self.move_arm(arm_name, target_pose, use_orientation=True)
        elif motion_type == 'grab':
            result = self.pick_up(arm_name, target_pose)
        else:  # release
            result = self.place_at(arm_name, target_pose)

        response.success = bool(result.get('success', False))
        response.executed = bool(result.get('executed', False))
        response.message = str(result.get('message', ''))

        return response

    # --------------------------------------------------
    # High-level helper functions that planning team wants
    # --------------------------------------------------

    def move_arm(
        self,
        arm_name: str,
        target_pose: Pose,
        use_orientation: bool = True,
        duration: float = 3.0,
        timeout_sec: float = 20.0,
    ) -> dict:
        """
        Move one arm to a target Cartesian pose.
        This is the low-level escape hatch used by the high-level planner.
        """
        plan_result = self._plan_and_execute(
            arm_name=arm_name,
            pose=target_pose,
            use_orientation=use_orientation,
            duration=duration,
            timeout_sec=timeout_sec,
        )

        if plan_result is None:
            return {
                'success': False,
                'executed': False,
                'message': 'Planner call failed or timed out.',
            }

        return {
            'success': bool(plan_result.success),
            'executed': bool(getattr(plan_result, 'executed', True)),
            'message': str(plan_result.message),
        }

    def pick_up(
        self,
        arm_name: str,
        target_pose: Pose,
        duration: float = 3.0,
    ) -> dict:
        """
        Simple placeholder pick-up:
          1. move arm to target pose
          2. close gripper

        Later this can become:
          scan/find -> navigate close -> align -> grasp
        """
        move_result = self.move_arm(
            arm_name=arm_name,
            target_pose=target_pose,
            use_orientation=True,
            duration=duration,
        )

        if not move_result['success']:
            return {
                'success': False,
                'executed': move_result['executed'],
                'message': f'pick_up failed during move: {move_result["message"]}',
            }

        gripper_ok = self._command_gripper(arm_name, GRIPPER_CLOSED, duration=1.0)
        if not gripper_ok:
            return {
                'success': False,
                'executed': True,
                'message': 'pick_up reached target pose but failed to close gripper.',
            }

        return {
            'success': True,
            'executed': True,
            'message': f'pick_up succeeded for {arm_name} arm.',
        }

    def place_at(
        self,
        arm_name: str,
        target_pose: Pose,
        duration: float = 3.0,
    ) -> dict:
        """
        Simple placeholder place-at:
          1. move arm to target pose
          2. open gripper

        Later this can become:
          find destination -> navigate close -> align -> release
        """
        move_result = self.move_arm(
            arm_name=arm_name,
            target_pose=target_pose,
            use_orientation=True,
            duration=duration,
        )

        if not move_result['success']:
            return {
                'success': False,
                'executed': move_result['executed'],
                'message': f'place_at failed during move: {move_result["message"]}',
            }

        gripper_ok = self._command_gripper(arm_name, GRIPPER_OPEN, duration=1.0)
        if not gripper_ok:
            return {
                'success': False,
                'executed': True,
                'message': 'place_at reached target pose but failed to open gripper.',
            }

        return {
            'success': True,
            'executed': True,
            'message': f'place_at succeeded for {arm_name} arm.',
        }

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _plan_and_execute(
        self,
        arm_name: str,
        pose: Pose,
        use_orientation: bool,
        duration: float,
        timeout_sec: float,
    ) -> Optional[PlanToTarget.Response]:
        req = PlanToTarget.Request()
        req.arm_name = arm_name
        req.target_pose = pose
        req.use_orientation = use_orientation
        req.execute = True
        req.duration = float(duration)
        req.max_condition_number = 100.0

        self.get_logger().info(
            f'Calling /plan_to_target: arm={arm_name}, '
            f'pos=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}), '
            f'use_orientation={use_orientation}'
        )

        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if not future.done():
            self.get_logger().error('/plan_to_target timed out')
            return None

        if future.exception() is not None:
            self.get_logger().error(f'/plan_to_target exception: {future.exception()}')
            return None

        result = future.result()
        if result is None:
            self.get_logger().error('/plan_to_target returned no result')
            return None

        if result.success:
            self.get_logger().info(f'Planner success: {result.message}')
        else:
            self.get_logger().warn(f'Planner failed: {result.message}')

        return result

    def _command_gripper(self, arm_name: str, position: float, duration: float = 1.0) -> bool:
        """
        Repeatedly publish gripper command, then send STOP.
        """
        if arm_name not in self.gripper_cmd_pubs:
            self.get_logger().error(f'Invalid gripper arm_name: {arm_name}')
            return False

        pub = self.gripper_cmd_pubs[arm_name]

        cmd_msg = Float64MultiArray()
        cmd_msg.data = [float(position)]

        stop_msg = Float64MultiArray()
        stop_msg.data = [GRIPPER_STOP]

        state_desc = 'OPEN' if position < 0.5 else 'CLOSED'
        self.get_logger().info(
            f'{arm_name.capitalize()} gripper -> {state_desc} ({position:.1f}) for {duration:.1f}s'
        )

        try:
            start_time = time.time()
            while (time.time() - start_time) < duration:
                pub.publish(cmd_msg)
                time.sleep(0.1)

            pub.publish(stop_msg)
            return True

        except Exception as e:
            self.get_logger().error(f'Gripper command failed: {e}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = SimpleManipulationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()