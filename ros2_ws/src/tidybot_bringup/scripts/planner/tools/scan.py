"""
scan(query?) — Perception-only action.

Rotates the base to survey the scene and runs object detection via
Gemini Vision + RealSense depth. Returns detected objects with poses.

Pipeline:
  1. Rotate base incrementally (8 x 45°)
  2. At each heading, capture RGB+depth from RealSense
  3. Send RGB to Gemini Vision with query
  4. For detections, project to 3D using depth + camera extrinsics
  5. Aggregate detections, deduplicate, return object list with poses

Assigned to: Max
"""

import json
import time
import math

import rclpy
from geometry_msgs.msg import Twist

import google.generativeai as genai

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

    @property
    def description(self) -> str:
        return (
            "Perception-only action. Rotates the base to survey the scene and runs "
            "object detection via Gemini Vision + RealSense depth. Takes an optional "
            "natural language query (e.g. 'red apple', 'all objects'). Returns a list "
            "of detected objects with approximate 3D poses. Use when the planner needs "
            "scene information, or when the human asks 'what do you see?'"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional NL query to filter detections (e.g. 'red apple'). Omit to detect all objects."
                }
            },
            "required": []
        }

    def run(self, query: str = None) -> str:
        log_info(f"Scanning scene{f' for: {query}' if query else ''}...")

        all_detections = []

        # Multi-heading scan: rotate base and capture at each heading
        num_headings = config.SCAN_HEADINGS
        angle_step = 2 * math.pi / num_headings

        for i in range(num_headings):
            heading_deg = i * (360 / num_headings)
            log_info(f"  Heading {i+1}/{num_headings} ({heading_deg:.0f}°)...")

            # Rotate base to next heading
            if i > 0:
                twist = Twist()
                twist.angular.z = config.BASE_ANGULAR_SPEED
                duration = angle_step / config.BASE_ANGULAR_SPEED
                self.ctx.publish_twist_for(twist, duration)

            # Wait for camera to settle
            time.sleep(config.CAMERA_SETTLE_TIME)
            for _ in range(10):
                rclpy.spin_once(self.ctx, timeout_sec=0.05)

            # Capture and detect
            img_bytes = self.ctx.capture_image_bytes()
            if img_bytes is None:
                continue

            try:
                vision_model = genai.GenerativeModel(config.GEMINI_MODEL)
                prompt = (
                    f"You are a robot's vision system. Describe what objects you see. "
                    f"For each, estimate position relative to camera (left/center/right, near/mid/far). "
                    f"{'Focus on: ' + query if query else 'List all visible objects.'} "
                    f"Return a JSON array: [{{\"name\": ..., \"position\": ..., \"confidence\": 0-1}}]"
                )
                response = vision_model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_bytes}
                ])
                all_detections.append({
                    "heading_deg": heading_deg,
                    "detections": response.text
                })
            except Exception as e:
                all_detections.append({
                    "heading_deg": heading_deg,
                    "error": str(e)
                })

        # TODO: Aggregate detections across headings, deduplicate,
        #       project to 3D using depth + camera extrinsics + TF

        return json.dumps({
            "status": "success",
            "query": query,
            "headings_scanned": num_headings,
            "detections_by_heading": all_detections,
            "note": "Positions are approximate camera-relative. 3D projection TODO."
        })
