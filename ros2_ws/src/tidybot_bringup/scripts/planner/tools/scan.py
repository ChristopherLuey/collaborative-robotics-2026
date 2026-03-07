"""
scan(query?) — Perception-only action.

Rotates the base to survey the scene and runs object detection via
Gemini Vision + RealSense depth. Returns detected objects with poses.

Assigned to: Max
"""

import json
import time
import math

import rclpy
from geometry_msgs.msg import Twist

from google import genai as genai_client

from planner.tools.base_tool import BaseTool
from planner.core.ros_context import RosContext, CV_AVAILABLE
from planner.utils import log_info
from planner import config

if CV_AVAILABLE:
    import cv2


class ScanTool(BaseTool):

    @property
    def name(self) -> str:
        return "scan"

    def run(self, query: str = "") -> str:
        """Rotate base to survey the scene and detect objects via Gemini Vision + RealSense depth.

        Args:
            query: Optional NL query to filter detections (e.g. 'red apple'). Empty string to detect all objects.
        """
        query = query or None
        log_info(f"Scanning scene{f' for: {query}' if query else ''}...")

        all_detections = []
        num_headings = config.SCAN_HEADINGS
        angle_step = 2 * math.pi / num_headings

        for i in range(num_headings):
            heading_deg = i * (360 / num_headings)
            log_info(f"  Heading {i+1}/{num_headings} ({heading_deg:.0f}°)...")

            if i > 0:
                twist = Twist()
                twist.angular.z = config.BASE_ANGULAR_SPEED
                duration = angle_step / config.BASE_ANGULAR_SPEED
                self.ctx.publish_twist_for(twist, duration)

            time.sleep(config.CAMERA_SETTLE_TIME)
            for _ in range(10):
                rclpy.spin_once(self.ctx, timeout_sec=0.05)

            img_bytes = self.ctx.capture_image_bytes()
            if img_bytes is None:
                continue

            try:
                client = genai_client.Client(api_key=config.GOOGLE_API_KEY)
                prompt = (
                    f"You are a robot's vision system. Describe what objects you see. "
                    f"For each, estimate position relative to camera (left/center/right, near/mid/far). "
                    f"{'Focus on: ' + query if query else 'List all visible objects.'} "
                    f"Return a JSON array: [{{\"name\": ..., \"position\": ..., \"confidence\": 0-1}}]"
                )
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=[
                        prompt,
                        {"mime_type": "image/jpeg", "data": img_bytes}
                    ]
                )
                all_detections.append({
                    "heading_deg": heading_deg,
                    "detections": response.text
                })
            except Exception as e:
                all_detections.append({
                    "heading_deg": heading_deg,
                    "error": str(e)
                })

        return json.dumps({
            "status": "success",
            "query": query,
            "headings_scanned": num_headings,
            "detections_by_heading": all_detections,
            "note": "Positions are approximate camera-relative. 3D projection TODO."
        })
