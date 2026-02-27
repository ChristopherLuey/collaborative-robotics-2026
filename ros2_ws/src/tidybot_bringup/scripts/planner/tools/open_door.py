"""
open_door() — Door-opening pipeline.

SCOPING: Full-size doors with round knobs likely infeasible (WidowX torque).
Targeting cabinet doors or lightweight lever-handle doors.

Pipeline:
  1. Scan for door/handle (Gemini Vision)
  2. Navigate to handle approach pose (~0.5m)
  3. Visual servo: align gripper with handle using depth
  4. Grasp handle (lever: from above; cabinet: grip edge)
  5. For lever: press down while holding grip
  6. Coordinated trajectory: base arcs, arm maintains grip
  7. Release handle
  FALLBACK: push-only approach (no handle grasp)

TODO: This is the most complex action — requires coordinated base+arm motion.
"""

import json

from planner.tools.base_tool import BaseTool
from planner.utils import log_info


class OpenDoorTool(BaseTool):

    @property
    def name(self) -> str:
        return "open_door"

    @property
    def description(self) -> str:
        return (
            "Complete door-opening pipeline. Detects the door and handle, approaches, "
            "grasps the handle, and executes coordinated base+arm motion to open it. "
            "Targets cabinet doors or lever-handle doors. Push doors as fallback."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def run(self) -> str:
        log_info("open_door() — NOT YET IMPLEMENTED")

        return json.dumps({
            "status": "not_implemented",
            "message": (
                "open_door requires coordinated base+arm motion which is the most "
                "complex action. Implementation plan: "
                "1) Handle detection via Gemini Vision, "
                "2) Approach positioning, "
                "3) Handle grasping, "
                "4) Coordinated base arc + arm trajectory, "
                "5) Fallback: push-only. "
                "See design doc for details."
            )
        })
