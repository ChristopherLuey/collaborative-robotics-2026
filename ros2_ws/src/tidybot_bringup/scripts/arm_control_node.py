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

from geometry_msgs.msg import PoseStamped, Pose, PoseStamped
from tidybot_msgs.srv import PlanToTarget, RequestArmMotion
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String, Empty, Bool
from tf2_ros import TransformListener, Buffer

from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from scipy.spatial.transform import Rotation as rot_obj


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
        self._prev_state = None
        self._prev_queue_len = None
        self._prev_arms_moving = set()
        self.gripper_action_start = self.get_clock().now()

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    # ---------------- subscribers
    # we want these literally just to check when arm is done moving
    def _right_joint_state_cb(self, msg: Float64MultiArray):
        self.last_cmd_publish['right'] = self.get_clock().now()

    def _left_joint_state_cb(self, msg: Float64MultiArray):
        self.last_cmd_publish['left'] = self.get_clock().now()

    # ---------------- Planner service (async, non-blocking) ----------------

    def _check_job(self):
        # check if arm is currently moving, if it is we just continue:
        now = self.get_clock().now()
        arms_moving = set()
        for arm in ['right', 'left']:
            if (now - self.last_cmd_publish[arm]).nanoseconds < 500_000_000:  # 0.5 second threshold for "still moving"
                arms_moving.add(arm)

        if arms_moving:
            if arms_moving != self._prev_arms_moving:
                for arm in arms_moving:
                    self.get_logger().info(f'{arm.capitalize()} arm is still moving...')
            self._prev_arms_moving = arms_moving
            bool_msg = Bool()
            bool_msg.data = True
            self.queue_full_pub.publish(bool_msg)
            return

        if self._prev_arms_moving:
            self._prev_arms_moving = set()

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

        # Log only when state or queue length changes
        queue_len = len(self.action_queue)
        if self.current_state != self._prev_state or queue_len != self._prev_queue_len:
            self.get_logger().info(f'State: {self.current_state}, queue: {self.action_queue}')
            self._prev_state = self.current_state
            self._prev_queue_len = queue_len

        bool_msg = Bool()
        bool_msg.data = self.current_state != "idle"
        self.queue_full_pub.publish(bool_msg)
            

    def _start_plan_request(self, arm_name: str, pose: Pose, duration: float = 3.0, use_orientation: bool = True):
        req = PlanToTarget.Request()
        req.arm_name = arm_name
        
        tf_msg = self.tf_buffer.lookup_transform(
            "base_link",
            "odom",
            rclpy.time.Time()
        )

        req.target_pose = transform_pose(pose, tf_msg)

        self.get_logger().info(
            f'IK request ({arm_name}): '
            f'odom=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}) '
            f'-> base_link=({req.target_pose.position.x:.3f}, {req.target_pose.position.y:.3f}, {req.target_pose.position.z:.3f})'
        )
        req.use_orientation = True
        req.execute = True
        req.duration = float(duration)
        req.max_condition_number = 100.0
    
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
            self.action_queue.extend(["open", "reach", "close"]) 
        
        elif motion_type == "release":
            self.action_queue.extend(["close", "reach", "open"])

        elif motion_type == "move":
            self.action_queue.extend(["reach"])

        response.success = True
        response.message = f'Sent execution {motion_type} motion for {arm_name} arm'
        return response
    
def transform_pose(pose: Pose, tf: TransformStamped) -> Pose:
    # Build transformation matrix from TransformStamped
    t = tf.transform.translation
    q = tf.transform.rotation
    rot = rot_obj.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [t.x, t.y, t.z]

    # Build homogeneous coordinate for the input pose
    p = np.array([pose.position.x, pose.position.y, pose.position.z, 1.0])

    # Apply transformation
    p_transformed = T @ p

    # Rotate the orientation
    q_pose = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    r_pose = rot_obj.from_quat(q_pose)
    r_transformed = rot_obj.from_matrix(rot) * r_pose  # Apply rotation
    q_transformed = r_transformed.as_quat()  # [x, y, z, w]

    # Create new Pose
    transformed_pose = Pose()
    transformed_pose.position.x = p_transformed[0]
    transformed_pose.position.y = p_transformed[1]
    transformed_pose.position.z = p_transformed[2]
    transformed_pose.orientation.x = q_transformed[0]
    transformed_pose.orientation.y = q_transformed[1]
    transformed_pose.orientation.z = q_transformed[2]
    transformed_pose.orientation.w = q_transformed[3]

    return transformed_pose

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
