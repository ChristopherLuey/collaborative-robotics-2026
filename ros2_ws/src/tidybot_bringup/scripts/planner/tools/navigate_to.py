"""
navigate_to(target) — Move base to a target location via approach_pose service.
"""

import json

from planner.tools.base_tool import BaseTool
from planner.utils import log_info
from planner import config


class NavigateToTool(BaseTool):

    @property
    def name(self) -> str:
        return "navigate_to"

    def run(self, target: str) -> str:
        """Move the robot base to a target location.

        Args:
            target: Named location ('start', 'home') or 'x, y, theta' coordinates in meters/radians.
        """
        if target.lower() in config.NAMED_LOCATIONS:
            tx, ty, ttheta = config.NAMED_LOCATIONS[target.lower()]
            log_info(f"Navigating to '{target}': ({tx}, {ty}, {ttheta})")
        else:
            try:
                parts = [float(p.strip()) for p in target.split(',')]
                tx = parts[0]
                ty = parts[1] if len(parts) > 1 else 0.0
                ttheta = parts[2] if len(parts) > 2 else 0.0
            except ValueError:
                return json.dumps({
                    "status": "error",
                    "message": f"Unknown location '{target}'. Known: {list(config.NAMED_LOCATIONS.keys())}."
                })

        # Use approach_pose service (closed-loop control via odom)
        if not self.ctx.approach_pose(tx, ty, ttheta):
            return json.dumps({
                "status": "error",
                "message": "approach_pose service failed or unavailable.",
            })

        return json.dumps({
            "status": "success",
            "target": target,
            "estimated_pose": {"x": tx, "y": ty, "theta": ttheta},
        })
