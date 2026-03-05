"""
High level planner. Uses Gemini's function calling API.

Architecture:
1. Human inputs NL command (eg: 'locate the apple and bring it back').
2. Gemini selects tool calls based on the command.
3. Each tool executes and returns feedback.
4. Gemini uses feedback to plan the next action.
5. Loop until Gemini returns text (plan complete).
"""

import os
import json

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Tool declarations (plain dicts — works with all SDK versions)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "function_declarations": [
            {
                "name": "scan",
                "description": (
                    "Perception-only action. Rotates the camera to survey the scene and runs "
                    "object detection. Takes an optional natural language query (e.g. 'red apple', "
                    "'all objects'). Returns a list of detected objects with approximate 3D poses. "
                    "Use when the planner needs scene information before deciding what to do, or "
                    "when the human asks 'what do you see?'"
                ),
                "parameters": {
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
                },
            },
            {
                "name": "navigate_to",
                "description": (
                    "Moves the robot base to a target location. Takes either a named location "
                    "(e.g. 'start', 'home', 'table') or coordinates as a string (e.g. '0.5, 0.0, 1.57'). "
                    "Uses base velocity commands for simple moves. Does NOT do precision alignment "
                    "for manipulation — that lives inside pick_up/place_at."
                ),
                "parameters": {
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
                },
            },
            {
                "name": "pick_up",
                "description": (
                    "Complete object retrieval pipeline. Takes a natural language description of "
                    "the object to pick up (e.g. 'red apple', 'the cup on the left'). Handles "
                    "finding the object, navigating close, computing a grasp pose, and grabbing it. "
                    "Constrained to top-down grasps. Designed for tabletop objects within ~450g payload."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_description": {
                            "type": "string",
                            "description": "Natural language description of the object to pick up.",
                        }
                    },
                    "required": ["object_description"],
                },
            },
            {
                "name": "place_at",
                "description": (
                    "Complete placement pipeline. Takes a natural language description of the "
                    "destination (e.g. 'brown basket', 'the table'). Assumes robot is already "
                    "holding an object. Finds the target, navigates close, and places the object."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_description": {
                            "type": "string",
                            "description": "Natural language description of where to place the object.",
                        }
                    },
                    "required": ["target_description"],
                },
            },
            {
                "name": "open_door",
                "description": (
                    "Complete door-opening pipeline. Detects the door and handle, approaches, "
                    "grasps the handle, and executes a coordinated base+arm motion to open it. "
                    "Targets cabinet doors or lightweight lever-handle doors. Push doors as fallback."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "move_arm",
                "description": (
                    "Low-level arm positioning. Moves one arm to a Cartesian end-effector position "
                    "in the base_link frame. Position-only (no orientation constraint). "
                    "Escape hatch for custom behaviors."
                ),
                "parameters": {
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
                },
            },
        ]
    }
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the high-level planner for a mobile-manipulator robot with two arms.

Given a natural-language command from a human operator, decompose it into a
sequence of tool calls. After each tool result you may issue more calls or
return a final text summary when the task is complete.

Available actions (in order of autonomy):
  scan(query?)                – Look around. Returns detected objects + 3D poses.
  navigate_to(target)         – Drive to a named location or coordinates. No precision alignment.
  pick_up(object_description) – Full grasp pipeline (includes its own perception + close-range nav).
  place_at(target_description)– Full placement pipeline (includes its own perception + close-range nav).
  open_door()                 – Full door-opening pipeline. Push doors as fallback.
  move_arm(arm, x, y, z)     – Raw Cartesian arm move in base_link frame. Escape hatch only.

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

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class GeminiPlanner:
    """High-level Gemini planner. For local testing, returns stub results."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,
        )

    def _stub_execute(self, name: str, args: dict) -> dict:
        """Fake execution for local testing — just returns success."""
        print(f"  [STUB] {name}({args})")
        stubs = {
            "scan": {"success": True, "objects": [
                {"label": "red apple", "pose": {"x": 1.2, "y": 0.3, "z": 0.75}},
                {"label": "blue cup", "pose": {"x": 1.5, "y": -0.2, "z": 0.74}},
            ]},
            "navigate_to": {"success": True},
            "pick_up": {"success": True, "holding": args.get("object_description", "unknown")},
            "place_at": {"success": True},
            "open_door": {"success": True},
            "move_arm": {"success": True},
        }
        return stubs.get(name, {"success": False, "error": f"Unknown tool: {name}"})

    def run(self, user_command: str) -> str:
        chat = self.model.start_chat()
        response = chat.send_message(user_command)

        for turn in range(MAX_PLANNING_TURNS):
            function_calls = [
                part for part in response.candidates[0].content.parts
                if part.function_call.name
            ]

            if not function_calls:
                break

            print(f"  [Turn {turn + 1}] Gemini wants: "
                  f"{[fc.function_call.name for fc in function_calls]}")

            tool_responses = []
            for fc in function_calls:
                name = fc.function_call.name
                args = dict(fc.function_call.args) if fc.function_call.args else {}

                result = self._stub_execute(name, args)

                tool_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
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
        return final_text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    planner = GeminiPlanner()

    print("Robot planner ready (LOCAL TEST MODE). Type a command (Ctrl-C to quit).\n")
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


if __name__ == "__main__":
    main()