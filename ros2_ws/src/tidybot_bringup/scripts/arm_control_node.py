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
from std_msgs.msg import Float64MultiArray, String, Empty, Bool


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

        self.joint_cmd_sub = {
            'right': self.create_subscription(Float64MultiArray, '/right_arm/joint_cmd', self._right_joint_state_cb, 10),
            'left': self.create_subscription(Float64MultiArray, '/left_arm/joint_cmd', self._left_joint_state_cb, 10),
        }

        self.last_cmd_publish = {
            'right': self.get_clock().now(),
            'left': self.get_clock().now(),
        }

        self.queue_full_pub = self.create_publisher(Bool, '/arm_queue_full', 10)

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
        
        self.job_ticker = self.create_timer(0.1, self._check_job)  # 10Hz timer for job processing
        self.active_future = None
        self.action_queue = []
        self.current_state = "idle"
        self.gripper_action_start = self.get_clock().now()


    # ---------------- subscribers
    # we want these literally just to check when arm is done moving
    def _right_joint_state_cb(self, msg: Float64MultiArray):
        self.last_cmd_publish['right'] = self.get_clock().now()

    def _left_joint_state_cb(self, msg: Float64MultiArray):
        self.last_cmd_publish['left'] = self.get_clock().now()

    # ---------------- Planner service (async, non-blocking) ----------------

    def _check_job(self):

        self.get_logger().info("checking for job")
        self.get_logger().info("current state: " + self.current_state)
        self.get_logger().info("current action queue: " + str(self.action_queue))
        self.get_logger().info("active future: " + str(self.active_future.done()) if self.active_future else "None")

        # check if arm is currently moving, if it is we just continue:
        now = self.get_clock().now()
        for arm in ['right', 'left']:
            if (now - self.last_cmd_publish[arm]).nanoseconds < 500_000_000:  # 0.5 second threshold for "still moving"
                self.get_logger().info(f'{arm.capitalize()} arm is still moving...')

                bool_msg = Bool()
                bool_msg.data = True
                self.queue_full_pub.publish(bool_msg)
                return

        if self.active_future is not None and self.active_future.done():
            result = self.active_future.result()
            if result is not None:
                self.get_logger().info(f'Plan result: {result}')
            else:
                self.get_logger().error('Planner service returned nothing')
            self.active_future = None
            self.current_state = "idle"

        if self.current_state == "idle" and self.action_queue is not None and len(self.action_queue) > 0:
            next_action = self.action_queue.pop(0)
            self.get_logger().info(f"Next action: {next_action}")

            if next_action == "reach":
                self.active_future = self._start_plan_request(arm_name=self.current_request.arm_name, pose=self.current_request.target_pose)
                self.current_state = "working"
            if next_action == "close":
                self._set_gripper(self.current_request.arm_name, GRIPPER_CLOSED)
            if next_action == "open":
                self._set_gripper(self.current_request.arm_name, GRIPPER_OPEN)

        bool_msg = Bool()
        if self.current_state is not "idle":
            bool_msg.data = True
        else:
            bool_msg.data = False

        self.queue_full_pub.publish(bool_msg)
            

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
    
        return self.plan_client.call_async(req)
        #self.get_logger().info('Plan request sent, waiting for response...')

    # ---------------- Gripper timed publish (non-blocking) ----------------
    
    def _set_gripper(self  , arm_name: str, position: float, duration: float = 2.0):
        msg = Float64MultiArray()
        msg.data = [float(position)]
        pub = self.gripper_cmd_pubs[arm_name]

        state_desc = 'OPEN' if position < 0.5 else 'CLOSED'
        if position == 0.5:
            state_desc = 'STOP'

        self.get_logger().info(f'{arm_name.capitalize()} gripper -> {state_desc} ({position:.1f})')

        pub.publish(msg)

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
        
        self.current_request = request

        # now queue up motion.
        if motion_type == "grab":
            self.action_queue.extend(["reach", "close"]) 
        
        elif motion_type == "release":
            self.action_queue.extend(["reach", "open"])

        elif motion_type == "move":
            self.action_queue.extend(["reach"])

        response.success = True
        response.message = f'Sent execution {motion_type} motion for {arm_name} arm'
        return response

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