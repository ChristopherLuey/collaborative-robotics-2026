"""
NOTE: Thif file would be different than what Chris made
We couldn't figure out that code and felt it would be easier if we start from scratch.

High level planner. Uses Gemini's function calling API.

Archtecture:
1. Human inputs NL command (eg: 'locate the apple and bring it back'). 
The API needs to call the different tools it has and plan in a high level what to do.
2. API calls the ROS actions it wants to.
3. takes feedback from each action it performs
4. using this feedback tries to plan for the next action.

Downstream actions so far:
    scan(query?)         Perception-only. Gemini Vision + RealSense depth. Returns objects + 3D poses.
    navigate_to(target)  Nav2-based movement. Coordinates or named location. No precision alignment.
    pick_up(desc)        Full pipeline: find → navigate close → visual servo → top-down grasp.
    place_at(desc)       Full pipeline: find target → navigate close → visual servo → release.
    open_door()          Full pipeline: detect door/handle → approach → coordinated base+arm open.
    move_arm(pose)       Low-level escape hatch. Joint-space or Cartesian. No perception.
"""

import os
import sys
import time
import json
import math
import threading
from typing import Optional, Any, Dict

import numpy as np

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose, Point, Quarternion

# Gemini
import google.generativeai as genai
from google.generativeai import protos as genai_protos
from google.generativeai.types import content_types

#TODO: Check with Max about the tools we have. Currently took the definition from the Excel sheet.
TOOLS = [
    genai_protos.Tool(
        function_declarations=[
            genai_protos.FunctionDeclaration(
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
                            "description": (
                                "Optional NL query to filter detections (e.g. 'red apple', 'cups'). "
                                "If omitted, detects all visible objects."
                            ),
                        }
                    },
                    "required": [],
                }),
            ),
            genai_protos.FunctionDeclaration(
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
                            "description": (
                                "Target location: named place ('start', 'home', 'table') "
                                "or 'x, y, theta' coordinates in meters/radians."
                            ),
                        }
                    },
                    "required": ["target"],
                }),
            ),
            genai_protos.FunctionDeclaration(
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
                            "description": "Natural language description of the object to pick up.",
                        }
                    },
                    "required": ["object_description"],
                }),
            ),
            genai_protos.FunctionDeclaration(
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
                            "description": "Natural language description of where to place the object.",
                        }
                    },
                    "required": ["target_description"],
                }),
            ),
            genai_protos.FunctionDeclaration(
                name="open_door",
                description=(
                    "Complete door-opening pipeline. Detects the door and handle, approaches, "
                    "grasps the handle, and executes a coordinated base+arm motion to open it. "
                    "Targets cabinet doors or lightweight lever-handle doors. Push doors as fallback."
                ),
                parameters=content_types.to_proto({
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
            ),
            genai_protos.FunctionDeclaration(
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
                            "description": "Which arm to move.",
                        },
                        "x": {"type": "number", "description": "X position in meters (forward from base)."},
                        "y": {"type": "number", "description": "Y position in meters (left positive)."},
                        "z": {"type": "number", "description": "Z position in meters (up from base)."},
                    },
                    "required": ["arm", "x", "y", "z"],
                }),
            ),
        ]
    )
]


#TODO: Currently only having NL as input but adding voice commands shouldn't be that hard.  
class ActionDispatcher(Node):
    """
    Uses GEMINI API to call the downstream tools.
    """

    def __init__(self):
        super().__init__("gemini planner")
        self.get_logger().info("Action dispacther started")
    
    def excecute(self, tool_name: str, tool_arg: Dict[str, Any]) -> dict:
        self.get_logger().info(f"[dispatch] {tool_name} and args: {tool_arg}")

        try:
            self._dispatch(self, tool_name, tool_arg)
        except Exception as e:
            self.get_logger().error(f"tool {tool_name} failed with arg: {arg}")