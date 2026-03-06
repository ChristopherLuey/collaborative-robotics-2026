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
from tidybot_msgs.srv import PlanToTarget
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from std_msgs.msg import Float64MultiArray, String, Empty


GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0


class ArmPlanner(Node):
    # Joint-space sleep pose: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    SLEEP_POSE = [0.0, -1.80, 1.55, 0.0, 0.8, 0.0]

    def __init__(self):
        super().__init__('arm_planner')

        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        self.arm_cmd_pubs = {
            'right': self.create_publisher(JointGroupCommand, '/right_arm/commands/joint_group', 10),
            'left': self.create_publisher(JointGroupCommand, '/left_arm/commands/joint_group', 10),
        }
        self.gripper_cmd_pubs = {
            'right': self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10),
            'left': self.create_publisher(Float64MultiArray, '/left_gripper/cmd', 10),
        }

        # State from joint_states
        self.joint_states_received = False
        self.current_joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self._js_callback, 10)

        # Command topics
        self.create_subscription(PoseStamped, '/gripper_pose_cmd', self._pose_callback, 10)
        self.create_subscription(String, '/gripper_action_cmd', self._action_callback, 10)
        self.create_subscription(Empty, '/arm_planner/reset', self._reset_callback, 10)

        # Command memory
        self.last_target_pose = None

        # ---- Job / state machine ----
        self.job_queue = []          # list of dict jobs
        self.active_job = None       # current job dict
        self.active_future = None    # service future
        self.phase = 'IDLE'          # state machine phase

        # gripper timed publish
        self._gripper_pub = None
        self._gripper_msg = None
        self._gripper_stop_msg = Float64MultiArray()
        self._gripper_stop_msg.data = [0.5]
        self._gripper_end_time = 0.0

        self.get_logger().info('=' * 50)
        self.get_logger().info('TidyBot2 IK Planner (Real Hardware) - Non-blocking control')
        self.get_logger().info('  /gripper_pose_cmd      PoseStamped: queue MOVE + store pose')
        self.get_logger().info('  /gripper_action_cmd    String: "grasp" or "release" (uses stored pose)')
        self.get_logger().info('  /arm_planner/reset     Empty: open both grippers + both arms sleep')
        self.get_logger().info('=' * 50)

        self.get_logger().info('Waiting for /plan_to_target service...')
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service not available! Launch with use_planner:=true')
            raise RuntimeError('Planning service not available')
        self.get_logger().info('Service connected!')

        self.get_logger().info('Waiting for joint states...')
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_states_received:
                break
        if not self.joint_states_received:
            self.get_logger().warn('No joint states received - proceeding anyway')

        # Timer runs the state machine
        self.timer = self.create_timer(0.05, self._tick)  # 20 Hz

        self.get_logger().info('Ready.')

    # ---------------- Subscribers ----------------

    def _js_callback(self, msg: JointState):
        self.joint_states_received = True
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def _pose_callback(self, msg: PoseStamped):
        # Store pose and queue a MOVE job (RIGHT arm)
        self.last_target_pose = self.copy_pose(msg.pose)
        p = msg.pose.position
        self.get_logger().info(f'Received pose_cmd (queued MOVE): ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
        self._enqueue_job({'type': 'MOVE', 'arm': 'right', 'pose': self.last_target_pose})

    def _action_callback(self, msg: String):
        action = msg.data.strip().lower()
        if self.last_target_pose is None:
            self.get_logger().warn('No stored pose yet. Publish /gripper_pose_cmd first.')
            return

        if action == 'grasp':
            self.get_logger().info('Queued GRASP: move -> close')
            self._enqueue_job({'type': 'GRASP', 'arm': 'right', 'pose': self.last_target_pose})
        elif action == 'release':
            self.get_logger().info('Queued RELEASE: move -> open')
            self._enqueue_job({'type': 'RELEASE', 'arm': 'right', 'pose': self.last_target_pose})
        else:
            self.get_logger().warn(f'Unknown action "{msg.data}". Use "grasp" or "release".')

    def _reset_callback(self, _: Empty):
        self.get_logger().info('Queued RESET: open both grippers + both arms sleep')
        self._enqueue_job({'type': 'RESET'})

    # ---------------- Job queue helpers ----------------

    def _enqueue_job(self, job: dict):
        # Simple policy: keep queue from growing unbounded; you can tune this
        if len(self.job_queue) > 20:
            self.get_logger().warn('Job queue too long; dropping oldest job.')
            self.job_queue.pop(0)
        self.job_queue.append(job)

    def _start_next_job_if_idle(self):
        if self.active_job is not None:
            return
        if not self.job_queue:
            return
        self.active_job = self.job_queue.pop(0)
        self.phase = 'START'

    # ---------------- State machine tick ----------------

    def _tick(self):
        # 1) keep gripper publishing if in progress
        self._tick_gripper_publish()

        # 2) start job if idle
        self._start_next_job_if_idle()
        if self.active_job is None:
            return

        job_type = self.active_job['type']

        # RESET is implemented as a blocking operation but triggered from timer (not a callback)
        if job_type == 'RESET':
            self._do_reset_job()
            self._finish_job()
            return

        # MOVE / GRASP / RELEASE use planner service
        if self.phase == 'START':
            if job_type == 'MOVE':
                self._start_plan_request(self.active_job['arm'], self.active_job['pose'])
                self.phase = 'WAIT_PLAN'
                return

            if job_type == 'GRASP':
                self._start_plan_request(self.active_job['arm'], self.active_job['pose'])
                self.phase = 'WAIT_PLAN_THEN_CLOSE'
                return

            if job_type == 'RELEASE':
                self._start_plan_request(self.active_job['arm'], self.active_job['pose'])
                self.phase = 'WAIT_PLAN_THEN_OPEN'
                return

        # Wait for service response
        if self.phase.startswith('WAIT_PLAN'):
            if self.active_future is None:
                self.get_logger().error('Internal error: no active future.')
                self._finish_job()
                return

            if not self.active_future.done():
                return  # keep waiting; no blocking

            result = None
            try:
                result = self.active_future.result()
            except Exception as e:
                self.get_logger().error(f'Planner future exception: {e}')
                self._finish_job()
                return

            self.active_future = None

            if (result is None) or (not result.success):
                msg = getattr(result, 'message', '(no message)')
                self.get_logger().error(f'Plan failed: {msg}')
                self._finish_job()
                return

            # success
            self.get_logger().info(f'Plan success: {result.message}')

            if self.phase == 'WAIT_PLAN':
                self._finish_job()
                return

            if self.phase == 'WAIT_PLAN_THEN_CLOSE':
                self._start_gripper_timed_publish(self.active_job['arm'], GRIPPER_CLOSED, duration=1.0)
                self.phase = 'WAIT_GRIPPER_DONE'
                self.active_job['_next'] = 'FINISH'
                return

            if self.phase == 'WAIT_PLAN_THEN_OPEN':
                self._start_gripper_timed_publish(self.active_job['arm'], GRIPPER_OPEN, duration=1.0)
                self.phase = 'WAIT_GRIPPER_DONE'
                self.active_job['_next'] = 'FINISH'
                return

        if self.phase == 'WAIT_GRIPPER_DONE':
            if time.time() < self._gripper_end_time:
                return
            # ensure we sent stop at end
            if self._gripper_pub is not None:
                self._gripper_pub.publish(self._gripper_stop_msg)
            self._finish_job()
            return

    def _finish_job(self):
        self.active_job = None
        self.active_future = None
        self.phase = 'IDLE'

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
        self.active_future = self.plan_client.call_async(req)

    # ---------------- Gripper timed publish (non-blocking) ----------------

    def _start_gripper_timed_publish(self, arm_name: str, position: float, duration: float = 1.0):
        msg = Float64MultiArray()
        msg.data = [float(position)]
        self._gripper_pub = self.gripper_cmd_pubs[arm_name]
        self._gripper_msg = msg
        self._gripper_end_time = time.time() + float(duration)

        state_desc = 'OPEN' if position < 0.5 else 'CLOSED'
        self.get_logger().info(f'{arm_name.capitalize()} gripper -> {state_desc} ({position:.1f})')

        # publish immediately once
        self._gripper_pub.publish(self._gripper_msg)

    def _tick_gripper_publish(self):
        if self._gripper_pub is None or self._gripper_msg is None:
            return
        if time.time() < self._gripper_end_time:
            # publish at timer rate (20 Hz); should be fine
            self._gripper_pub.publish(self._gripper_msg)
        else:
            # stop publishing after end
            self._gripper_pub.publish(self._gripper_stop_msg)
            self._gripper_pub = None
            self._gripper_msg = None

    # ---------------- Reset (open both + sleep both) ----------------

    def _do_reset_job(self):
        # Open both grippers (timed publish, but we’ll do quick blocking here safely since not in callback)
        self._blocking_gripper('right', GRIPPER_OPEN, 1.0)
        self._blocking_gripper('left', GRIPPER_OPEN, 1.0)

        # Sleep both arms (blocking interpolation)
        self.go_to_sleep_pose('right')
        self.go_to_sleep_pose('left')

        self.get_logger().info('RESET complete.')

    def _blocking_gripper(self, arm_name: str, position: float, duration: float):
        pub = self.gripper_cmd_pubs[arm_name]
        msg = Float64MultiArray()
        msg.data = [float(position)]
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.5]

        state_desc = 'OPEN' if position < 0.5 else 'CLOSED'
        self.get_logger().info(f'{arm_name.capitalize()} gripper -> {state_desc} ({position:.1f})')

        start = time.time()
        while time.time() - start < duration:
            pub.publish(msg)
            time.sleep(0.1)
        pub.publish(stop_msg)

    # ---------------- Utilities ----------------

    def copy_pose(self, pose: Pose) -> Pose:
        p = Pose()
        p.position.x = pose.position.x
        p.position.y = pose.position.y
        p.position.z = pose.position.z
        p.orientation.w = pose.orientation.w
        p.orientation.x = pose.orientation.x
        p.orientation.y = pose.orientation.y
        p.orientation.z = pose.orientation.z
        return p

    def get_arm_positions(self, arm_name: str) -> np.ndarray:
        joint_names = [
            f'{arm_name}_waist', f'{arm_name}_shoulder', f'{arm_name}_elbow',
            f'{arm_name}_forearm_roll', f'{arm_name}_wrist_angle', f'{arm_name}_wrist_rotate'
        ]
        return np.array([self.current_joint_positions.get(j, 0.0) for j in joint_names], dtype=float)

    def go_to_sleep_pose(self, arm_name: str, max_joint_speed: float = 0.5):
        # Get latest joint states once
        rclpy.spin_once(self, timeout_sec=0.05)

        current = self.get_arm_positions(arm_name)
        target = np.array(self.SLEEP_POSE, dtype=float)

        max_diff = float(np.max(np.abs(target - current)))
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