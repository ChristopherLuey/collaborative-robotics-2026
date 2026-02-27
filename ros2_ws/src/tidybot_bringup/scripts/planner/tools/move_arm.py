"""
move_arm(arm, x, y, z) — Low-level arm positioning.

Moves one arm to a Cartesian end-effector position via IK planner.
Position-only (no orientation constraint). Escape hatch for custom behaviors.
"""

import json

from planner.tools.base_tool import BaseTool
from planner.utils import log_info
from planner import config


class MoveArmTool(BaseTool):

    @property
    def name(self) -> str:
        return "move_arm"

    @property
    def description(self) -> str:
        return (
            "Low-level arm positioning. Moves one arm to a Cartesian end-effector "
            "position in the base_link frame. Position-only (no orientation). "
            "Escape hatch for custom behaviors."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "arm": {
                    "type": "string",
                    "enum": ["right", "left"],
                    "description": "Which arm to move."
                },
                "x": {"type": "number", "description": "X in meters (forward)."},
                "y": {"type": "number", "description": "Y in meters (left positive)."},
                "z": {"type": "number", "description": "Z in meters (up from base)."},
            },
            "required": ["arm", "x", "y", "z"]
        }

    def run(self, arm: str, x: float, y: float, z: float) -> str:
        log_info(f"Moving {arm} arm to ({x:.3f}, {y:.3f}, {z:.3f})...")

        success = self.ctx.plan_and_execute(arm, float(x), float(y), float(z))

        if success:
            return json.dumps({"status": "success", "arm": arm, "position": {"x": x, "y": y, "z": z}})
        else:
            return json.dumps({
                "status": "error", "arm": arm,
                "position": {"x": x, "y": y, "z": z},
                "message": "IK planning or execution failed."
            })
