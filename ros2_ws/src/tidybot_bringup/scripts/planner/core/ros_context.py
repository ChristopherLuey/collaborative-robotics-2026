"""
Shared ROS2 context — single node with all publishers, subscribers, and service clients.

All tools receive a reference to this context rather than creating their own nodes.
This avoids multiple-node complexity and ensures a single consistent view of robot state.
"""

import time
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, Pose, Pose2D
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Bool, Float64MultiArray
from tidybot_msgs.msg import ArmCommand, PanTilt
from tidybot_msgs.srv import PlanToTarget, GetObjectPose, RequestArmMotion, ApproachPose

from planner.utils import log_info, log_error, log_service
from planner import config

try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


class RosContext(Node):
    """
    Centralized ROS2 node providing hardware access to all tool modules.

    Owns:
    - Publishers for base, arms, grippers, camera pan/tilt
    - Service clients for IK planning
    - Subscribers for camera, joint states
    - Shared robot state (joint positions, holding status, pose estimate)
    """

    def __init__(self):
        super().__init__('gemini_planner')

        # ── Publishers ──────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pan_tilt_pub = self.create_publisher(PanTilt, '/camera/pan_tilt', 10)

        self.arm_cmd_pubs = {
            'right': self.create_publisher(ArmCommand, '/right_arm/cmd', 10),
            'left': self.create_publisher(ArmCommand, '/left_arm/cmd', 10),
        }
        self.gripper_pubs = {
            'right': self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10),
            'left': self.create_publisher(Float64MultiArray, '/left_gripper/cmd', 10),
        }

        # ── Service clients ─────────────────────────────────────────
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')
        self.object_pose_client = self.create_client(GetObjectPose, '/sam3/get_object_pose')
        self.arm_motion_client = self.create_client(RequestArmMotion, '/request_arm_motion')
        self.approach_client = self.create_client(ApproachPose, '/approach_pose')

        # ── Subscribers ─────────────────────────────────────────────
        qos_be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.cv_bridge = CvBridge() if CV_AVAILABLE else None

        if CV_AVAILABLE:
            self.create_subscription(Image, '/camera/color/image_raw', self._rgb_cb, qos_be)
            self.create_subscription(Image, '/camera/depth/image_raw', self._depth_cb, qos_be)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self._cam_info_cb, 10)

        self.current_joint_positions = {}
        self.joint_lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        # ── Goal reached (for approach_pose wait) ──────────────────
        self.goal_reached_event = threading.Event()
        self.create_subscription(Bool, '/base/goal_reached', self._goal_reached_cb, 10)

        # ── Robot state ─────────────────────────────────────────────
        self.holding_object = False
        self.current_pose = (0.0, 0.0, 0.0)  # Dead-reckoned (x, y, theta)

        # ── Service readiness ───────────────────────────────────────
        log_info("Waiting for /plan_to_target service (5s)...")
        self.planner_available = self.plan_client.wait_for_service(timeout_sec=5.0)
        if self.planner_available:
            log_info("Motion planner service connected.")
        else:
            log_info("Motion planner not available — arm tools will be limited.")

        log_info("Waiting for /request_arm_motion service (3s)...")
        self.arm_motion_available = self.arm_motion_client.wait_for_service(timeout_sec=3.0)
        if self.arm_motion_available:
            log_info("Arm motion service connected.")
        else:
            log_info("Arm motion service not available — arm tools will fail.")

        log_info("Waiting for /approach_pose service (3s)...")
        self.approach_available = self.approach_client.wait_for_service(timeout_sec=3.0)
        if self.approach_available:
            log_info("Approach service connected.")
        else:
            log_info("Approach service not available — navigation tools will fail.")

        log_info("Waiting for /sam3/get_object_pose service (5s)...")
        self.object_pose_available = self.object_pose_client.wait_for_service(timeout_sec=5.0)
        if self.object_pose_available:
            log_info("SAM3 object pose service connected.")
        else:
            log_info("SAM3 object pose not available — perception tools will fail.")

    # ── Callbacks ───────────────────────────────────────────────────

    def _rgb_cb(self, msg):
        if self.cv_bridge:
            try:
                self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
            except Exception:
                pass

    def _depth_cb(self, msg):
        if self.cv_bridge:
            try:
                self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
            except Exception:
                pass

    def _cam_info_cb(self, msg):
        self.camera_info = msg

    def _goal_reached_cb(self, msg):
        if msg.data:
            self.goal_reached_event.set()

    def _joint_state_cb(self, msg):
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.current_joint_positions[name] = msg.position[i]

    # ── Convenience methods (used by multiple tools) ────────────────

    def _wait_for_future(self, future, timeout_sec: float):
        """Wait for an async service future. Works when node is spun in a background thread."""
        deadline = time.time() + timeout_sec
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.done()

    def plan_and_execute(self, arm_name: str, x: float, y: float, z: float,
                         duration: float = None, use_orientation: bool = False) -> bool:
        """Call PlanToTarget for position-only IK. Returns True on success."""
        log_service("/plan_to_target", f"arm={arm_name} pos=({x:.3f}, {y:.3f}, {z:.3f})")
        duration = duration or config.DEFAULT_MOTION_DURATION

        if not self.planner_available:
            log_error("plan_to_target", "Service not available")
            return False

        request = PlanToTarget.Request()
        request.arm_name = arm_name
        request.target_pose = Pose()
        request.target_pose.position.x = x
        request.target_pose.position.y = y
        request.target_pose.position.z = z
        request.target_pose.orientation.w = 1.0
        request.use_orientation = use_orientation
        request.execute = True
        request.duration = duration
        request.max_condition_number = config.IK_MAX_CONDITION_NUMBER

        future = self.plan_client.call_async(request)
        if not self._wait_for_future(future, config.IK_TIMEOUT_SEC):
            log_error("plan_to_target", "Timed out")
            return False
        if future.exception():
            log_error("plan_to_target", f"Exception: {future.exception()}")
            return False

        result = future.result()
        if result.success:
            log_info(f"  IK: {result.message}")
            return True
        else:
            log_error("plan_to_target", result.message)
            return False

    def set_gripper(self, arm: str, closed: bool):
        """Open or close a gripper. closed=True → grip, False → release."""
        log_service(f"/{arm}_gripper/cmd", f"{'close' if closed else 'open'}")
        msg = Float64MultiArray()
        msg.data = [1.0 if closed else 0.0]
        self.gripper_pubs[arm].publish(msg)

    def publish_twist_for(self, twist: Twist, duration: float, rate_hz: float = 50.0):
        """Publish a velocity command for a fixed duration, then stop."""
        log_service("/cmd_vel", f"linear=({twist.linear.x:.2f}, {twist.linear.y:.2f}) angular={twist.angular.z:.2f} duration={duration:.2f}s")
        dt = 1.0 / rate_hz
        for _ in range(int(duration * rate_hz)):
            self.cmd_vel_pub.publish(twist)
            time.sleep(dt)
        self.cmd_vel_pub.publish(Twist())

    def capture_image_bytes(self, quality: int = 85) -> Optional[bytes]:
        """Grab latest RGB frame as JPEG bytes for Gemini Vision. Returns None if unavailable."""
        if not CV_AVAILABLE or self.latest_rgb is None:
            return None
        bgr = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def get_arm_positions(self, arm_name: str) -> np.ndarray:
        """Current joint positions for an arm (6 values)."""
        joints = [f'{arm_name}_waist', f'{arm_name}_shoulder', f'{arm_name}_elbow',
                  f'{arm_name}_forearm_roll', f'{arm_name}_wrist_angle', f'{arm_name}_wrist_rotate']
        with self.joint_lock:
            return np.array([self.current_joint_positions.get(j, 0.0) for j in joints])

    def request_arm_motion(self, arm_name: str, motion_type: str,
                           x: float, y: float, z: float) -> bool:
        """Call RequestArmMotion service (grab/release/move). Returns True on success."""
        log_service("/request_arm_motion", f"arm={arm_name} type={motion_type} pos=({x:.3f}, {y:.3f}, {z:.3f})")
        if not self.arm_motion_available:
            log_error("request_arm_motion", "Service /request_arm_motion not available")
            return False

        request = RequestArmMotion.Request()
        request.arm_name = arm_name
        request.motion_type = motion_type
        request.target_pose = Pose()
        request.target_pose.position.x = x
        request.target_pose.position.y = y
        request.target_pose.position.z = z
        request.target_pose.orientation.w = 1.0

        future = self.arm_motion_client.call_async(request)
        if not self._wait_for_future(future, config.IK_TIMEOUT_SEC):
            log_error("request_arm_motion", "Timed out")
            return False
        if future.exception():
            log_error("request_arm_motion", f"Exception: {future.exception()}")
            return False

        result = future.result()
        if result.success:
            log_info(f"  Arm motion: {result.message}")
            return True
        else:
            log_error("request_arm_motion", result.message)
            return False

    def approach_pose(self, x: float, y: float, theta: float, relative: bool = False) -> bool:
        """Call ApproachPose service for base navigation. Waits for goal_reached. Returns True on success."""
        log_service("/approach_pose", f"target=({x:.3f}, {y:.3f}, {theta:.3f}) relative={relative}")
        if not self.approach_available:
            log_error("approach_pose", "Service /approach_pose not available")
            return False

        # Clear the event before sending a new target
        self.goal_reached_event.clear()

        request = ApproachPose.Request()
        request.pose = Pose2D()
        request.pose.x = x
        request.pose.y = y
        request.pose.theta = theta
        request.relative = relative

        future = self.approach_client.call_async(request)
        if not self._wait_for_future(future, 30.0):
            log_error("approach_pose", "Service call timed out")
            return False
        if future.exception():
            log_error("approach_pose", f"Exception: {future.exception()}")
            return False

        log_info(f"  Approach: target ({x:.2f}, {y:.2f}, {theta:.2f}), waiting for goal_reached...")

        # Wait for the base to actually arrive (phoenix6_base_node publishes /base/goal_reached)
        if not self.goal_reached_event.wait(timeout=30.0):
            log_error("approach_pose", "Timed out waiting for /base/goal_reached")
            return False

        log_info("  Approach: goal reached.")
        return True

    def call_object_isolator(self, query: str):
        """
        Call the SAM3 GetObjectPose service with a text prompt.

        Returns a simple namespace with .x, .y, .z attributes on success, or None on failure.
        """
        log_service("/sam3/get_object_pose", f"query='{query}'")
        if not self.object_pose_available:
            log_error("object_pose", "SAM3 service not available")
            return None

        request = GetObjectPose.Request()
        request.prompt = query

        future = self.object_pose_client.call_async(request)
        if not self._wait_for_future(future, 10.0):
            log_error("object_pose", "Timed out waiting for SAM3 response")
            return None
        if future.exception():
            log_error("object_pose", f"Exception: {future.exception()}")
            return None

        result = future.result()
        if not result.success:
            log_error("object_pose", result.message)
            return None

        pos = result.pose.pose.position
        log_info(f"Object '{query}' found at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

        class ObjectPose:
            pass
        obj = ObjectPose()
        obj.x = pos.x
        obj.y = pos.y
        obj.z = pos.z
        return obj
