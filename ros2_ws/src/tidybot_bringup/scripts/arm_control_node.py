#!/usr/bin/env python3
"""
TidyBot2 Motion Planner (Real Hardware) - Robust Pose + Grasp/Release + Reset

IMPORTANT: This version avoids calling spin_until_future_complete() inside
subscription callbacks (which can cause service timeouts). Instead it runs a
tiny job state machine in a timer.

Topics:
  /gripper_pose_cmd      (geometry_msgs/PoseStamped)
      - Queue move of RIGHT arm to the commanded pose (pos + ori)
      - Stores it as "last target pose"

  /gripper_action_cmd    (std_msgs/String) values: "grasp" or "release"
      - Uses the last stored pose:
          grasp   = move -> close
          release = move -> open

  /arm_planner/reset     (std_msgs/Empty)
      - Opens BOTH grippers and sends BOTH arms to sleep pose (joint-space)

Usage:
  Terminal 1:
    ros2 launch tidybot_bringup real.launch.py use_planner:=true

  Terminal 2:
    ros2 run tidybot_bringup pickup_object_real.py

Send a pose:
  ros2 topic pub --once /gripper_pose_cmd geometry_msgs/msg/PoseStamped "{
    header: {frame_id: 'base_link'},
    pose: {
      position: {x: -0.10, y: -0.30, z: 0.55},
      orientation: {w: 0.5, x: 0.5, y: 0.5, z: -0.5}
    }
  }"

Grasp / Release:
  ros2 topic pub --once /gripper_action_cmd std_msgs/msg/String "{data: grasp}"
  ros2 topic pub --once /gripper_action_cmd std_msgs/msg/String "{data: release}"

Reset:
  ros2 topic pub --once /arm_planner/reset std_msgs/msg/Empty "{}"
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Pose
from tidybot_msgs.srv import PlanToTarget, RequestArmMotion
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from std_msgs.msg import Float64MultiArray, String, Empty


GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0

class ArmPlanner(Node):

    def __init__(self):
        super().__init__('arm_planner')

        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        self.gripper_cmd_pubs = {
            'right': self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10),
            'left': self.create_publisher(Float64MultiArray, '/left_gripper/cmd', 10),
        }

        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service not available! Launch with use_planner:=true')
            raise RuntimeError('Planning service not available')
        
        self.get_logger().info('Service connected!')

        self.get_logger().info('Ready.')

        self.hosted_service = self.create_service(
            RequestArmMotion,             # Service type
            'request_arm_motion',         # Service name
            self._request_arm_motion  # Callback
        )

    # ---------------- Subscribers ----------------

    def _pose_callback(self, msg: PoseStamped):
        # Store pose and queue a MOVE job (RIGHT arm)
        self.last_target_pose = self.copy_pose(msg.pose)
        p = msg.pose.position
        self.get_logger().info(f'Received pose_cmd (queued MOVE): ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
        self._enqueue_job({'type': 'MOVE', 'arm': 'right', 'pose': self.last_target_pose})

    # ---------------- Planner service (async, non-blocking) ----------------

    def _start_plan_request(self, arm_name: str, pose: Pose, duration: float = 3.0, use_orientation: bool = True):
        req = PlanToTarget.Request()
        req.arm_name = arm_name
        req.target_pose = pose
        req.use_orientation = use_orientation
        req.execute = True
        req.duration = float(duration)
        req.max_condition_number = 100.0

        p = pose.position
        self.get_logger().info(
            f'Calling /plan_to_target async: arm={arm_name} '
            f'pos=({p.x:.3f},{p.y:.3f},{p.z:.3f}) use_ori={use_orientation}'
        )
    
        # Send request async and wait for result
        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)  # blocks until the service responds

        if future.result() is not None:
            self.get_logger().info(f'Plan result received: {future.result()}')
            return future.result()
        else:
            self.get_logger().error('Service call failed')
            return None

    # ---------------- Gripper timed publish (non-blocking) ----------------
    
    def _set_gripper(self, arm_name: str, position: float, duration: float = 2.0):
        """
        Set gripper position using wrapper node.

        Args:
            arm_name: 'right' or 'left'
            position: 0.0 (open) to 1.0 (closed)
            duration: Time to hold the command (seconds)
        """
        msg = Float64MultiArray()
        msg.data = [float(position)]

        pub = self.gripper_cmd_pubs[arm_name]
        state_desc = 'OPEN' if position < 0.5 else 'CLOSED'
        self.get_logger().info(f'{arm_name.capitalize()} gripper -> {state_desc} ({position:.1f})')

        # Publish for duration (reduced rate to avoid bus overload)
        start = time.time()
        while (time.time() - start) < duration:
            pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)  # 10Hz instead of 20Hz

        # Send stop command (0.5 maps to PWM=0 in wrapper)
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.5]
        pub.publish(stop_msg)
        rclpy.spin_once(self, timeout_sec=0.05)


    def _request_arm_motion(self, request: RequestArmMotion.Request, response: RequestArmMotion.Response):
        self.get_logger().info(f'Received RequestArmMotion: arm={request.arm_name} motion={request.motion_type}')
        arm_name = request.arm_name
        if arm_name not in ['right', 'left']:
            response.success = False
            response.message = f'Invalid arm_name: {arm_name}'
            return response
        
        motion_type = request.motion_type
        if motion_type not in ['grab', 'release', 'move']:
            response.success = False
            response.message = f'Invalid motion_type: {motion_type}'
            return response
        
        target_pose = request.target_pose
        if type(target_pose) != Pose:
            response.success = False
            response.message = 'Invalid target_pose type'
            return response
        
        res = self._start_plan_request(arm_name, target_pose, use_orientation=True)
        if res is None:
            self.get_logger().info(f'Motion planning failed...')
            response.success = False
            response.message = 'Planning service call failed'
            return response
        
        # close or open the gripper if requested
        if motion_type == "grab":
            self.get_logger().info(f'Closing Gripper...')
            self._set_gripper(arm_name, GRIPPER_CLOSED)

        elif motion_type == "release":
            self.get_logger().info(f'Opening Gripper...')
            self._set_gripper(arm_name, GRIPPER_OPEN)
        
        response.success = True
        response.message = f'Executed {motion_type} motion for {arm_name} arm'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ArmPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Guard shutdown to avoid "rcl_shutdown already called"
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()