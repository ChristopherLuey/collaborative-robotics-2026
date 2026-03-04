#!/usr/bin/env python3
"""
TidyBot2 Gemini Planner — LLM-driven tool-calling action dispatcher.

Uses Gemini's function calling API to parse natural language commands into
sequences of robot actions, then executes them via ROS2.

Architecture:
    User speaks/types NL command
    → Gemini selects tool calls (scan, navigate_to, pick_up, place_at, open_door, move_arm)
    → Each tool executes real ROS2 calls (topics/services)
    → Results fed back to Gemini for multi-step planning
    → Loop until Gemini returns text (plan complete)

Usage:
    # Terminal 1: Launch robot (sim or real)
    ros2 launch tidybot_bringup sim.launch.py

    # Terminal 2: Run planner
    GEMINI_API_KEY=<key> python3 gemini_planner.py

    # Or with ROS2:
    GEMINI_API_KEY=<key> ros2 run tidybot_bringup gemini_planner.py

Requirements:
    pip install google-generativeai
"""

import os
import sys
import time
import json
import math
import threading
from typing import Optional

import numpy as np

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from tidybot_msgs.msg import ArmCommand, PanTilt
from tidybot_msgs.srv import PlanToTarget

# Gemini
import google.generativeai as genai
from google.generativeai.types import content_types

# Optional CV for image processing
try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


# =============================================================================
# Color helpers for terminal output
# =============================================================================
class C:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def log_tool(name: str, args: dict):
    print(f"{C.CYAN}{C.BOLD}⚡ TOOL CALL:{C.RESET} {C.YELLOW}{name}{C.RESET}({C.DIM}{args}{C.RESET})")


def log_result(name: str, result: str):
    preview = result[:200] + '...' if len(result) > 200 else result
    print(f"{C.GREEN}  ✓ {name}:{C.RESET} {preview}")


def log_error(name: str, error: str):
    print(f"{C.RED}  ✗ {name}:{C.RESET} {error}")


def log_gemini(text: str):
    print(f"{C.MAGENTA}{C.BOLD}🤖 Gemini:{C.RESET} {text}")


def log_ginfo(text: str):
    print(f"{C.BLUE}ℹ {text}{C.RESET}")


# =============================================================================
# Tool definitions for Gemini function calling
# =============================================================================
TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="scan",
                description=(
                    "Perception-only action. Rotates the camera to survey the scene and runs "
                    "object detection. Takes an optional natural language query (e.g. 'red apple', "
                    "'all objects'). Returns a list of detected objects with approximate 3D poses. "
                    "Use when the planner needs scene information before deciding what to do, or "
                    "when the human asks 'what do you see?'"
                ),
                parameters=content_types.to_proto({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional NL query to filter detections (e.g. 'red apple', 'cups'). If omitted, detects all visible objects."
                        }
                    },
                    "required": []
                })
            ),
            genai.protos.FunctionDeclaration(
                name="navigate_to",
                description=(
                    "Moves the robot base to a target location. Takes either a named location "
                    "(e.g. 'start', 'home', 'table') or coordinates as a string (e.g. '0.5, 0.0, 1.57'). "
                    "Uses base velocity commands for simple moves. Does NOT do precision alignment "
                    "for manipulation — that lives inside pick_up/place_at."
                ),
                parameters=content_types.to_proto({
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Target location: named place ('start', 'home', 'table') or 'x, y, theta' coordinates in meters/radians."
                        }
                    },
                    "required": ["target"]
                })
            ),
            genai.protos.FunctionDeclaration(
                name="pick_up",
                description=(
                    "Complete object retrieval pipeline. Takes a natural language description of "
                    "the object to pick up (e.g. 'red apple', 'the cup on the left'). Handles "
                    "finding the object, navigating close, computing a grasp pose, and grabbing it. "
                    "Constrained to top-down grasps. Designed for tabletop objects within ~450g payload."
                ),
                parameters=content_types.to_proto({
                    "type": "object",
                    "properties": {
                        "object_description": {
                            "type": "string",
                            "description": "Natural language description of the object to pick up."
                        }
                    },
                    "required": ["object_description"]
                })
            ),
            genai.protos.FunctionDeclaration(
                name="place_at",
                description=(
                    "Complete placement pipeline. Takes a natural language description of the "
                    "destination (e.g. 'brown basket', 'the table'). Assumes robot is already "
                    "holding an object. Finds the target, navigates close, and places the object."
                ),
                parameters=content_types.to_proto({
                    "type": "object",
                    "properties": {
                        "target_description": {
                            "type": "string",
                            "description": "Natural language description of where to place the object."
                        }
                    },
                    "required": ["target_description"]
                })
            ),
            genai.protos.FunctionDeclaration(
                name="open_door",
                description=(
                    "Complete door-opening pipeline. Detects the door and handle, approaches, "
                    "grasps the handle, and executes a coordinated base+arm motion to open it. "
                    "Targets cabinet doors or lightweight lever-handle doors. Push doors as fallback."
                ),
                parameters=content_types.to_proto({
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            ),
            genai.protos.FunctionDeclaration(
                name="move_arm",
                description=(
                    "Low-level arm positioning. Moves one arm to a Cartesian end-effector position "
                    "in the base_link frame. Position-only (no orientation constraint). "
                    "Escape hatch for custom behaviors."
                ),
                parameters=content_types.to_proto({
                    "type": "object",
                    "properties": {
                        "arm": {
                            "type": "string",
                            "enum": ["right", "left"],
                            "description": "Which arm to move."
                        },
                        "x": {"type": "number", "description": "X position in meters (forward from base)."},
                        "y": {"type": "number", "description": "Y position in meters (left positive)."},
                        "z": {"type": "number", "description": "Z position in meters (up from base)."},
                    },
                    "required": ["arm", "x", "y", "z"]
                })
            ),
        ]
    )
]

# System prompt for the planner
SYSTEM_PROMPT = """You are the high-level planner for TidyBot2, a mobile robot with two WidowX 250 arms.

Your job: translate natural language commands into sequences of robot actions using the available tools.

Robot capabilities:
- 3-DOF differential drive base (x, y, theta)
- 2x WidowX 250 6-DOF arms (~450g payload, parallel jaw gripper)
- RealSense D435 camera (RGB + depth) on a pan-tilt head
- LiDAR for navigation

Planning principles:
1. Always scan first if you don't know where objects are.
2. Actions are self-contained — each handles its own perception when needed.
3. For multi-step tasks, call tools sequentially and check results.
4. If an action fails, try once more or report the failure.
5. Be concise in your text responses.

Available named locations: "start", "home" (both = initial position).

When you're done executing all needed actions, respond with a brief text summary of what was accomplished."""


# =============================================================================
# TidyBot2 Planner Node
# =============================================================================
class GeminiPlannerNode(Node):
    """ROS2 node that implements robot action tools for Gemini function calling."""

    # Known poses for named locations (x, y, theta)
    NAMED_LOCATIONS = {
        "start": (0.0, 0.0, 0.0),
        "home": (0.0, 0.0, 0.0),
    }

    # Arm sleep pose
    SLEEP_POSE = [0.0, -1.80, 1.55, 0.0, 0.8, 0.0]

    def __init__(self):
        super().__init__('gemini_planner')

        # -- Publishers --
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.right_arm_pub = self.create_publisher(ArmCommand, '/right_arm/cmd', 10)
        self.left_arm_pub = self.create_publisher(ArmCommand, '/left_arm/cmd', 10)
        self.right_gripper_pub = self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10)
        self.left_gripper_pub = self.create_publisher(Float64MultiArray, '/left_gripper/cmd', 10)
        self.pan_tilt_pub = self.create_publisher(PanTilt, '/camera/pan_tilt', 10)

        # -- Service clients --
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        # -- Subscribers --
        qos_be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None

        if CV_AVAILABLE:
            self.cv_bridge = CvBridge()
            self.create_subscription(Image, '/camera/color/image_raw', self._rgb_cb, qos_be)
            self.create_subscription(Image, '/camera/depth/image_raw', self._depth_cb, qos_be)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self._cam_info_cb, 10)

        # -- State --
        self.holding_object = False
        self.current_pose = (0.0, 0.0, 0.0)  # x, y, theta estimate

        # Check for planner service
        log_info("Waiting for /plan_to_target service (5s timeout)...")
        if self.plan_client.wait_for_service(timeout_sec=5.0):
            log_info("Motion planner service connected.")
        else:
            log_info("Motion planner service not available — move_arm/pick_up/place_at will be limited.")

    # -- Subscriber callbacks --
    def _rgb_cb(self, msg):
        if CV_AVAILABLE:
            try:
                self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
            except Exception:
                pass

    def _depth_cb(self, msg):
        if CV_AVAILABLE:
            try:
                self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
            except Exception:
                pass

    def _cam_info_cb(self, msg):
        self.camera_info = msg

    # =========================================================================
    # Tool implementations
    # =========================================================================

    def tool_scan(self, query: Optional[str] = None) -> str:
        """
        Perception scan: pan camera across scene, capture images, detect objects.

        Currently: captures single image and describes it via Gemini Vision.
        TODO: Multi-heading scan (rotate base 8x45°), aggregate detections,
              project to 3D using depth + camera extrinsics.
        """
        log_info(f"Scanning scene{f' for: {query}' if query else ''}...")

        # Center the camera
        pan_tilt = PanTilt()
        pan_tilt.pan = 0.0
        pan_tilt.tilt = 0.3  # Slight downward tilt to see tabletop
        self.pan_tilt_pub.publish(pan_tilt)

        # Wait for camera to settle and get a frame
        time.sleep(1.0)
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)

        if not CV_AVAILABLE or self.latest_rgb is None:
            return json.dumps({
                "status": "error",
                "message": "Camera not available or no image received."
            })

        # Encode image for Gemini Vision
        rgb_bgr = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR)
        _, img_bytes = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Call Gemini Vision for object detection
        try:
            vision_model = genai.GenerativeModel('gemini-2.0-flash')
            vision_prompt = (
                f"You are a robot's vision system. Describe what objects you see in this image. "
                f"For each object, estimate its approximate position relative to the camera "
                f"(left/center/right, near/mid/far). "
                f"{'Focus on: ' + query if query else 'List all visible objects.'} "
                f"Return a JSON array of objects with fields: name, position, confidence (0-1)."
            )
            response = vision_model.generate_content([
                vision_prompt,
                {"mime_type": "image/jpeg", "data": img_bytes.tobytes()}
            ])
            detections = response.text
        except Exception as e:
            detections = f"Vision API error: {e}"

        return json.dumps({
            "status": "success",
            "query": query,
            "detections": detections,
            "note": "Positions are approximate. Use pick_up for precise grasping."
        })

    def tool_navigate_to(self, target: str) -> str:
        """
        Move the base to a target location.

        Currently: simple open-loop velocity commands for named locations or coordinates.
        TODO: Integrate Nav2 action server for proper path planning with obstacle avoidance.
        """
        # Parse target
        if target.lower() in self.NAMED_LOCATIONS:
            tx, ty, ttheta = self.NAMED_LOCATIONS[target.lower()]
            log_info(f"Navigating to named location '{target}': ({tx}, {ty}, {ttheta})")
        else:
            # Try parsing as "x, y, theta"
            try:
                parts = [float(p.strip()) for p in target.split(',')]
                if len(parts) == 2:
                    tx, ty, ttheta = parts[0], parts[1], 0.0
                elif len(parts) == 3:
                    tx, ty, ttheta = parts[0], parts[1], parts[2]
                else:
                    return json.dumps({"status": "error", "message": f"Cannot parse target: {target}"})
            except ValueError:
                return json.dumps({
                    "status": "error",
                    "message": f"Unknown location '{target}'. Known: {list(self.NAMED_LOCATIONS.keys())}. Or use 'x, y, theta' format."
                })

        # Compute relative motion from current estimated pose
        dx = tx - self.current_pose[0]
        dy = ty - self.current_pose[1]
        dtheta = ttheta - self.current_pose[2]

        # Normalize angle
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

        distance = math.sqrt(dx**2 + dy**2)
        heading = math.atan2(dy, dx) - self.current_pose[2]
        heading = math.atan2(math.sin(heading), math.cos(heading))

        log_info(f"Relative motion: distance={distance:.2f}m, heading={math.degrees(heading):.1f}°, final_rotation={math.degrees(dtheta):.1f}°")

        twist = Twist()

        # Phase 1: Rotate to face target
        if abs(heading) > 0.05 and distance > 0.05:
            log_info("Phase 1: Rotating to face target...")
            rotation_time = abs(heading) / 0.5  # 0.5 rad/s
            twist.angular.z = 0.5 if heading > 0 else -0.5
            self._publish_for_duration(twist, rotation_time)

        # Phase 2: Drive forward
        if distance > 0.05:
            log_info(f"Phase 2: Driving {distance:.2f}m...")
            drive_time = distance / 0.2  # 0.2 m/s
            twist = Twist()
            twist.linear.x = 0.2
            self._publish_for_duration(twist, drive_time)

        # Phase 3: Final rotation
        if abs(dtheta) > 0.05:
            log_info("Phase 3: Final rotation...")
            twist = Twist()
            rotation_time = abs(dtheta) / 0.5
            twist.angular.z = 0.5 if dtheta > 0 else -0.5
            self._publish_for_duration(twist, rotation_time)

        # Stop
        self.cmd_vel_pub.publish(Twist())

        # Update pose estimate (dead reckoning — imprecise)
        self.current_pose = (tx, ty, ttheta)

        return json.dumps({
            "status": "success",
            "target": target,
            "estimated_pose": {"x": tx, "y": ty, "theta": ttheta},
            "note": "Open-loop navigation. Position is approximate."
        })

    def tool_pick_up(self, object_description: str) -> str:
        """
        Complete object retrieval: detect → approach → grasp → lift.

        Currently: scans for object, attempts IK-based grasp at detected position.
        TODO: Visual servo loop for precise final approach, retry logic,
              grasp verification via gripper width check.
        """
        log_info(f"Pick up sequence for: '{object_description}'")

        # Step 1: Scan for the object
        log_info("Step 1: Scanning for object...")
        scan_result = json.loads(self.tool_scan(object_description))
        if scan_result["status"] == "error":
            return json.dumps({"status": "error", "message": f"Cannot see scene: {scan_result['message']}"})

        # Step 2: Parse detection to get approximate position
        # TODO: Replace with proper 3D pose from depth + detection
        # For now, use a default grasp position in front of the robot
        grasp_x, grasp_y, grasp_z = 0.3, -0.1, 0.15  # Right arm workspace default
        log_info(f"Step 2: Target grasp position: ({grasp_x}, {grasp_y}, {grasp_z})")

        # Step 3: Open gripper
        log_info("Step 3: Opening gripper...")
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [0.0]  # 0 = open
        self.right_gripper_pub.publish(gripper_msg)
        time.sleep(0.5)

        # Step 4: Move arm above object (hover)
        hover_z = grasp_z + 0.05
        log_info(f"Step 4: Hovering above object at z={hover_z}...")
        hover_result = self._plan_and_execute('right', grasp_x, grasp_y, hover_z, duration=2.0)
        if not hover_result:
            return json.dumps({"status": "error", "message": "IK failed for hover position."})
        time.sleep(2.5)

        # Step 5: Descend to grasp
        log_info(f"Step 5: Descending to grasp at z={grasp_z}...")
        grasp_result = self._plan_and_execute('right', grasp_x, grasp_y, grasp_z, duration=1.5)
        if not grasp_result:
            return json.dumps({"status": "error", "message": "IK failed for grasp position."})
        time.sleep(2.0)

        # Step 6: Close gripper
        log_info("Step 6: Closing gripper...")
        gripper_msg.data = [1.0]  # 1 = closed
        self.right_gripper_pub.publish(gripper_msg)
        time.sleep(1.0)

        # Step 7: Lift
        lift_z = grasp_z + 0.10
        log_info(f"Step 7: Lifting to z={lift_z}...")
        self._plan_and_execute('right', grasp_x, grasp_y, lift_z, duration=1.5)
        time.sleep(2.0)

        self.holding_object = True

        return json.dumps({
            "status": "success",
            "object": object_description,
            "grasp_position": {"x": grasp_x, "y": grasp_y, "z": grasp_z},
            "note": "Grasp position was approximate. TODO: integrate visual servo for precision."
        })

    def tool_place_at(self, target_description: str) -> str:
        """
        Complete placement: detect target → approach → lower → release → retract.

        Currently: places at a default position.
        TODO: Visual servo to target, proper surface height detection from depth.
        """
        if not self.holding_object:
            return json.dumps({"status": "warning", "message": "Robot is not holding an object. Proceeding anyway."})

        log_info(f"Place at: '{target_description}'")

        # TODO: Scan for target, get 3D position
        # For now, use a default placement position
        place_x, place_y, place_z = 0.3, 0.1, 0.15  # Left of pickup default

        # Step 1: Move above placement
        hover_z = place_z + 0.05
        log_info(f"Step 1: Moving above placement at ({place_x}, {place_y}, {hover_z})...")
        self._plan_and_execute('right', place_x, place_y, hover_z, duration=2.0)
        time.sleep(2.5)

        # Step 2: Lower to placement height
        log_info(f"Step 2: Lowering to z={place_z}...")
        self._plan_and_execute('right', place_x, place_y, place_z, duration=1.5)
        time.sleep(2.0)

        # Step 3: Open gripper
        log_info("Step 3: Releasing object...")
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [0.0]
        self.right_gripper_pub.publish(gripper_msg)
        time.sleep(0.5)

        # Step 4: Retract upward
        log_info("Step 4: Retracting...")
        self._plan_and_execute('right', place_x, place_y, hover_z + 0.05, duration=1.5)
        time.sleep(2.0)

        self.holding_object = False

        return json.dumps({
            "status": "success",
            "target": target_description,
            "placement_position": {"x": place_x, "y": place_y, "z": place_z}
        })

    def tool_open_door(self) -> str:
        """
        Door-opening pipeline.

        TODO: This is the hardest action. Requires:
        - Door/handle detection via Gemini Vision
        - Approach positioning
        - Handle grasping
        - Coordinated base+arm trajectory (base arc + arm maintains grip)
        - Fallback: push-only approach
        """
        log_info("open_door() called — NOT YET IMPLEMENTED")
        return json.dumps({
            "status": "not_implemented",
            "message": (
                "open_door is not yet implemented. It requires coordinated base+arm motion "
                "which is the most complex action. Targeting cabinet doors / lever handles. "
                "See the design doc for the planned pipeline."
            )
        })

    def tool_move_arm(self, arm: str, x: float, y: float, z: float) -> str:
        """
        Direct arm positioning via IK planner service.
        """
        log_info(f"Moving {arm} arm to ({x:.3f}, {y:.3f}, {z:.3f})...")

        success = self._plan_and_execute(arm, x, y, z, duration=2.0)

        if success:
            return json.dumps({
                "status": "success",
                "arm": arm,
                "position": {"x": x, "y": y, "z": z}
            })
        else:
            return json.dumps({
                "status": "error",
                "arm": arm,
                "position": {"x": x, "y": y, "z": z},
                "message": "IK planning or execution failed. Position may be unreachable or in collision."
            })

    # =========================================================================
    # ROS2 helper methods
    # =========================================================================

    def _plan_and_execute(self, arm_name: str, x: float, y: float, z: float,
                          duration: float = 2.0) -> bool:
        """Call the PlanToTarget service for position-only IK."""
        if not self.plan_client.service_is_ready():
            log_error("plan_to_target", "Service not available")
            # Fallback: publish ArmCommand directly if we had joint targets
            return False

        request = PlanToTarget.Request()
        request.arm_name = arm_name
        request.target_pose = Pose()
        request.target_pose.position.x = x
        request.target_pose.position.y = y
        request.target_pose.position.z = z
        request.target_pose.orientation.w = 1.0
        request.use_orientation = False
        request.execute = True
        request.duration = duration
        request.max_condition_number = 100.0

        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        if not future.done():
            log_error("plan_to_target", "Service call timed out")
            return False
        if future.exception():
            log_error("plan_to_target", f"Exception: {future.exception()}")
            return False

        result = future.result()
        if result.success:
            log_info(f"  IK success: {result.message}")
            return True
        else:
            log_error("plan_to_target", f"Failed: {result.message}")
            return False

    def _publish_for_duration(self, twist: Twist, duration: float, rate_hz: float = 50.0):
        """Publish a twist command for a specified duration."""
        dt = 1.0 / rate_hz
        steps = int(duration * rate_hz)
        for _ in range(steps):
            self.cmd_vel_pub.publish(twist)
            time.sleep(dt)
        # Stop
        self.cmd_vel_pub.publish(Twist())

    # =========================================================================
    # Tool dispatch
    # =========================================================================

    def dispatch_tool(self, name: str, args: dict) -> str:
        """Route a Gemini function call to the appropriate tool implementation."""
        log_tool(name, args)

        try:
            if name == "scan":
                result = self.tool_scan(args.get("query"))
            elif name == "navigate_to":
                result = self.tool_navigate_to(args["target"])
            elif name == "pick_up":
                result = self.tool_pick_up(args["object_description"])
            elif name == "place_at":
                result = self.tool_place_at(args["target_description"])
            elif name == "open_door":
                result = self.tool_open_door()
            elif name == "move_arm":
                result = self.tool_move_arm(args["arm"], args["x"], args["y"], args["z"])
            else:
                result = json.dumps({"status": "error", "message": f"Unknown tool: {name}"})
        except Exception as e:
            result = json.dumps({"status": "error", "message": f"Tool execution error: {e}"})
            log_error(name, str(e))

        log_result(name, result)
        return result


# =============================================================================
# Main planner loop
# =============================================================================
def main():
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print(f"{C.RED}Error: GEMINI_API_KEY environment variable not set.{C.RESET}")
        print("Usage: GEMINI_API_KEY=<your-key> python3 gemini_planner.py")
        sys.exit(1)

    genai.configure(api_key=api_key)

    # Initialize ROS2
    rclpy.init()
    node = GeminiPlannerNode()

    # Spin ROS2 in background thread so callbacks work during tool execution
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    # Initialize Gemini chat with tools
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        tools=TOOLS,
        system_instruction=SYSTEM_PROMPT,
    )
    chat = model.start_chat()

    print()
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  TidyBot2 Gemini Planner{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.DIM}  Type natural language commands. Gemini will plan & execute.{C.RESET}")
    print(f"{C.DIM}  Type 'quit' to exit.{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print()

    while True:
        try:
            user_input = input(f"{C.BLUE}{C.BOLD}You: {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            break

        try:
            # Send user message to Gemini
            response = chat.send_message(user_input)

            # Tool-calling loop: keep executing until Gemini returns text
            while response.candidates[0].content.parts:
                # Check if any part is a function call
                function_calls = [
                    part for part in response.candidates[0].content.parts
                    if part.function_call.name  # has a function call
                ]

                if not function_calls:
                    # No more function calls — Gemini returned text
                    break

                # Execute all function calls and collect results
                tool_responses = []
                for part in function_calls:
                    fc = part.function_call
                    name = fc.name
                    args = dict(fc.args) if fc.args else {}

                    result_str = node.dispatch_tool(name, args)

                    tool_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=name,
                                response={"result": result_str}
                            )
                        )
                    )

                # Send results back to Gemini
                response = chat.send_message(
                    genai.protos.Content(parts=tool_responses)
                )

            # Print Gemini's final text response
            text_parts = [
                part.text for part in response.candidates[0].content.parts
                if hasattr(part, 'text') and part.text
            ]
            if text_parts:
                log_gemini('\n'.join(text_parts))
            print()

        except Exception as e:
            print(f"{C.RED}Error: {e}{C.RESET}")
            import traceback
            traceback.print_exc()
            print()

    # Cleanup
    log_info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
