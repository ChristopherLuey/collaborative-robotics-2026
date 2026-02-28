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
            result = self._dispatch(self, tool_name, tool_arg)
        except Exception as e:
            self.get_logger().error(f"tool {tool_name} failed with arg: {tool_arg} with error {e}")
            result = {
                "success": False, 
                "error": e 
            }
        return result
    
    def _dispatch(self, name: String, args: Dict[str, Any]) -> dict:
        """
        Actually calls the different fucntions we should have access to
        """
        if name == "scan":
            return self._scan(args.get("query"))
        elif name == "navigate_to":
            return self._navigate_to(args.get("target"))
        elif name == "pick_up":
            return self._pick_up(args["object_description"])
        elif name == "place_at":
            return self._place_at(args["target_description"])
        elif name == "open_door":
            return self._open_door()
        elif name == "move_arm":
            return self._move_arm(
                arm=args["arm"],
                x=float(args["x"]),
                y=float(args["y"]),
                z=float(args["z"]),
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    def _scan(self, query: str | None = None) -> dict:
        """
        Would need to define the scan here.
        """
        return None

    def _navigate_to(self, target: str) -> dict:
        return None
    
    def _pick_up(self, object_description: str) -> dict:
        return None
    
    def _place_at(self, target_description: str) -> dict:
        return None
    
    def _open_door(self) -> dict:
        """
        NOTE: opening the door doesn't need any query?
        """
        return None
    
    def _move_arm(self, arm: str, x: float, y: float, z: float) -> dict:
        return None
    
SYSTEM_PROMPT = """\
        You are the high-level planner for a mobile-manipulator robot with two arms.
        Given a natural-language command from a human operator, decompose it into a
        sequence of tool calls. After each tool result you may issue more calls or
        return a final text summary when the task is complete.

        Available actions (in order of autonomy):
        scan(query?)                : Look around. Returns detected objects + 3D poses.
        navigate_to(target)         : Drive to a named location or coordinates. No precision alignment.
        pick_up(object_description) : Full grasp pipeline (includes its own perception + close-range nav).
        place_at(target_description): Full placement pipeline (includes its own perception + close-range nav).
        open_door()                 : Full door-opening pipeline. Push doors as fallback.
        move_arm(arm, x, y, z)     : Raw Cartesian arm move in base_link frame. Escape hatch only.

        Key planning rules:
        1. pick_up, place_at, and open_door handle their own perception and close-range
        navigation internally. Do NOT call scan() or navigate_to() before them
        unless you need long-distance travel first or need scene info for a decision.
        2. Call scan() when you need information to decide what to do — e.g. the human
        asks "what's on the table?" or you must choose among objects.
        3. Use navigate_to() for travel between named waypoints or distant areas.
        4. If an action fails, try one reasonable recovery (re-scan, reposition), then
        report the failure.
        5. After a retrieval task, navigate back to 'start' unless told otherwise.

        Examples:
        "Locate the apple and bring it back"
            → scan("apple") → pick_up("apple") → navigate_to("start")

        "Put the cup in the brown basket"
            → pick_up("cup") → place_at("brown basket")

        "What do you see on the table?"
            → scan("objects on the table") → [return description as text]

        Always reason step-by-step before choosing actions.
"""

MAX_PLANNING_TURNS = 4

class GeminiPlanner:
    """
    The high level gemini planner which makes the high level plan and uses ActionDispatcher to fulfill tasks. 
    """
    def __init__(self, dispatcher: ActionDispatcher, model_name: str = "gemini-3.0-flash"):
        self.dispatcher = dispatcher
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS
        )
    
    def run(self, user_command: str) -> str:
        """
        Accepts the NL user command, excetues all the plans and returns a final text summary.
        """
        logger = self.dispatcher.get_logger()
        logger.info(f"[planner] new command : {user_command}")

        chat = self.model.start_chat()
        response = chat.send_message(user_command)

        for planning_turn in range(MAX_PLANNING_TURNS):
            function_calls = [
                part for part in response.candidates[0].content.parts
                if part.function_call.name
            ]

            if not function_calls:
                break

            logger.info(
                f"[planner] Turn : {planning_turn + 1} :"
                f"{[fc.function_call.name for fc in function_calls]}"
            )

            tool_responses: list[genai_protos.Part] = []
            for fc in function_calls:
                name = fc.function_call.name
                args = dict(fc.function_call.args) if fc.function_call.args else {}

                result = self.dispatcher.execute(name, args)

                tool_responses.append(
                    genai_protos.Part(
                        function_response=genai_protos.FunctionResponse(
                            name=name,
                            response={"result": json.dumps(result, default=str)},
                        )
                    )
                )

            response = chat.send_message(tool_responses)

        final_text = "".join(
            part.text
            for part in response.candidates[0].content.parts
            if hasattr(part, "text") and part.text
        )

        logger.info(f"[planner] Complete: {final_text}")
        return final_text
    
def main():
    genai.configure(api_key="AIzaSyBOD4MdtZ5XWbC8mXSmpDI7eqlCa2tKGCc")
    rclpy.init()

    dispatcher = ActionDispatcher()
    planner = GeminiPlanner(dispatcher)

    print("Robot planner ready. Type a command (Ctrl-C to quit).\n")
    try:
        while True:
            cmd = input(">>> ").strip()
            if not cmd:
                continue
            print(f"\n[Planning] {cmd}")
            summary = planner.run(cmd)
            print(f"\n[Done] {summary}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nShutting down.")
    finally:
        dispatcher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()