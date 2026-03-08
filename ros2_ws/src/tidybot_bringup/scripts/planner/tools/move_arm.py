"""
move_arm(arm, x, y, z) — Low-level arm positioning via IK planner.
"""

import json

from planner.tools.base_tool import BaseTool
from planner.utils import log_info


class MoveArmTool(BaseTool):

    @property
    def name(self) -> str:
        return "move_arm"

    def run(self, arm: str, x: float, y: float, z: float) -> str:
        """Move one arm to a Cartesian position in the base_link frame. Position-only, no orientation.

        Args:
            arm: Which arm to move ('right' or 'left').
            x: X position in meters (forward from base).
            y: Y position in meters (left positive).
            z: Z position in meters (up from base).
        """
        log_info(f"Moving {arm} arm to ({x:.3f}, {y:.3f}, {z:.3f})...")

        success = self.ctx.request_arm_motion(arm, 'move', float(x), float(y), float(z))

        if success:
            return json.dumps({"status": "success", "arm": arm, "position": {"x": x, "y": y, "z": z}})
        else:
            return json.dumps({
                "status": "error", "arm": arm,
                "position": {"x": x, "y": y, "z": z},
                "message": "IK planning or execution failed."
            })
