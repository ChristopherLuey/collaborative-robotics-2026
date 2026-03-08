"""
scan(query) — Perception-only action.

Continuously rotates the base searching for a target object. At each step,
Gemini Vision checks if the object is visible and where it is in the frame.
The robot adjusts its heading to center the object, then stops.

Assigned to: Max
"""

import json
import time
import math
import os
from datetime import datetime

import rclpy
from geometry_msgs.msg import Twist

from google import genai as genai_client
from google.genai import types as genai_types

from planner.tools.base_tool import BaseTool
from planner.core.ros_context import RosContext, CV_AVAILABLE
from planner.utils import log_info
from planner import config

if CV_AVAILABLE:
    import cv2


# Small rotation step when adjusting to center an object
_ADJUST_STEP_RAD = 0.15
_MAX_STEPS = 24  # safety limit (~2 full rotations at 90° steps, plus adjustments)


class ScanTool(BaseTool):

    @property
    def name(self) -> str:
        return "scan"

    def run(self, query: str = "") -> str:
        """Continuously rotate and search for an object. Stops when the object is centered in the frame.

        Args:
            query: Object to search for (e.g. 'red apple', 'banana'). Required.
        """
        query = query or None
        if not query:
            return json.dumps({"status": "error", "message": "scan() requires a query argument describing what to look for."})

        log_info(f"Scanning for: '{query}'...")

        # Create log directory
        scan_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     '..', 'logs', scan_timestamp, 'images')
        os.makedirs(self._log_dir, exist_ok=True)

        angle_step = 2 * math.pi / config.SCAN_HEADINGS
        step = 0

        # Start video recording
        video_path = os.path.join(os.path.dirname(self._log_dir), 'scan.mp4')
        video_writer = None
        if CV_AVAILABLE:
            video_writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
            log_info(f"  Recording video to {video_path}")

        try:
            while step < _MAX_STEPS:
                step += 1
                log_info(f"  Step {step}/{_MAX_STEPS}...")

                # Reinforce camera tilt to ensure it stays pointed down
                self.ctx.set_pan_tilt(config.CAMERA_PAN, config.CAMERA_TILT)

                # Discard any stale frame from the rotation, then wait for
                # the camera to settle and deliver a fresh post-settle image.
                self.ctx.rgb_updated = False
                time.sleep(config.CAMERA_SETTLE_TIME)
                self._spin_until_fresh_frame(timeout=1.0)

                # Write frame to video
                self._write_video_frame(video_writer)

                img_bytes = self.ctx.capture_image_bytes()
                if img_bytes is None:
                    log_info("  No image available, rotating...")
                    self._rotate(angle_step, video_writer)
                    continue

                # Save image
                try:
                    img_path = os.path.join(self._log_dir, f'scan_step_{step:03d}.jpg')
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)
                except Exception:
                    pass

                # Ask Gemini where the object is
                position = self._detect_position(img_bytes, query)
                log_info(f"  Detection result: {position}")

                if position == "center":
                    log_info(f"  '{query}' centered in frame. Scan complete.")
                    return json.dumps({
                        "status": "success",
                        "query": query,
                        "message": f"Found '{query}' centered in camera frame.",
                        "steps_taken": step,
                    })

                if position == "left":
                    log_info(f"  '{query}' on the left, adjusting...")
                    self._rotate(_ADJUST_STEP_RAD, video_writer)
                elif position == "right":
                    log_info(f"  '{query}' on the right, adjusting...")
                    self._rotate(-_ADJUST_STEP_RAD, video_writer)
                else:
                    # Not visible, do a large rotation step to scan next heading
                    log_info(f"  '{query}' not visible, rotating to next heading...")
                    self._rotate(angle_step, video_writer)

            return json.dumps({
                "status": "error",
                "query": query,
                "message": f"Could not find '{query}' after {_MAX_STEPS} steps.",
                "steps_taken": _MAX_STEPS,
            })
        finally:
            if video_writer is not None:
                video_writer.release()
                log_info(f"  Video saved to {video_path}")

    def _spin_until_fresh_frame(self, timeout: float = 1.0):
        """Spin until a new RGB frame arrives, or timeout."""
        deadline = time.time() + timeout
        while not self.ctx.rgb_updated and time.time() < deadline:
            rclpy.spin_once(self.ctx, timeout_sec=0.05)

    def _write_video_frame(self, video_writer):
        """Write the current camera frame to the video, only if a new frame arrived."""
        if video_writer is None or not CV_AVAILABLE or self.ctx.latest_rgb is None:
            return
        if not self.ctx.rgb_updated:
            return
        self.ctx.rgb_updated = False
        bgr = cv2.cvtColor(self.ctx.latest_rgb, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr)

    def _rotate(self, angle_rad: float, video_writer=None):
        """Rotate the base by a given angle (positive = CCW), recording frames to video."""
        twist = Twist()
        twist.angular.z = math.copysign(config.BASE_ANGULAR_SPEED, angle_rad)
        duration = abs(angle_rad) / config.BASE_ANGULAR_SPEED
        rate_hz = 50.0
        dt = 1.0 / rate_hz
        frame_interval = max(1, int(rate_hz / 15))  # capture at ~15 fps
        for i in range(int(duration * rate_hz)):
            self.ctx.cmd_vel_pub.publish(twist)
            time.sleep(dt)
            rclpy.spin_once(self.ctx, timeout_sec=0.01)
            if video_writer is not None and i % frame_interval == 0:
                self._write_video_frame(video_writer)
        self.ctx.cmd_vel_pub.publish(Twist())

    def _detect_position(self, img_bytes: bytes, query: str) -> str:
        """Ask Gemini Vision where the object is in the frame.

        Returns one of: "left", "center", "right", "not_visible".
        """
        try:
            if config.USE_VERTEX_AI:
                client = genai_client.Client(
                    vertexai=True,
                    project=config.VERTEX_PROJECT,
                    location=config.VERTEX_LOCATION,
                )
            else:
                client = genai_client.Client(api_key=config.GOOGLE_API_KEY)

            prompt = (
                f"You are a robot's vision system. Look at this image and determine "
                f"if you can see: '{query}'.\n\n"
                f"If the object is visible, determine its horizontal position in the frame:\n"
                f"- 'left' if the object is in the left third of the image\n"
                f"- 'center' if the object is in the middle third of the image\n"
                f"- 'right' if the object is in the right third of the image\n"
                f"- 'not_visible' if the object is not in the image at all\n\n"
                f"Respond with ONLY one word: left, center, right, or not_visible"
            )
            image_part = genai_types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/jpeg",
            )
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[prompt, image_part],
            )
            result = response.text.strip().lower()
            if result in ("left", "center", "right", "not_visible"):
                return result
            # If Gemini returned something unexpected, treat as not visible
            return "not_visible"
        except Exception as e:
            log_info(f"  Gemini error: {e}")
            return "not_visible"
