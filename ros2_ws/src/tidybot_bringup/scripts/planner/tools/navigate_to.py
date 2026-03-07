"""
navigate_to(target) — Move base to a target location.

Currently uses open-loop velocity commands.
TODO: Integrate Nav2 action server for path planning + obstacle avoidance.
"""

import json
import math

from geometry_msgs.msg import Twist

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

        cx, cy, ctheta = self.ctx.current_pose
        dx, dy = tx - cx, ty - cy
        dtheta = math.atan2(math.sin(ttheta - ctheta), math.cos(ttheta - ctheta))
        distance = math.sqrt(dx**2 + dy**2)
        heading = math.atan2(math.sin(math.atan2(dy, dx) - ctheta),
                             math.cos(math.atan2(dy, dx) - ctheta))

        if abs(heading) > 0.05 and distance > 0.05:
            log_info(f"Rotating {math.degrees(heading):.1f}°...")
            twist = Twist()
            twist.angular.z = config.BASE_ANGULAR_SPEED * (1 if heading > 0 else -1)
            self.ctx.publish_twist_for(twist, abs(heading) / config.BASE_ANGULAR_SPEED)

        if distance > 0.05:
            log_info(f"Driving {distance:.2f}m...")
            twist = Twist()
            twist.linear.x = config.BASE_LINEAR_SPEED
            self.ctx.publish_twist_for(twist, distance / config.BASE_LINEAR_SPEED)

        if abs(dtheta) > 0.05:
            log_info(f"Final rotation {math.degrees(dtheta):.1f}°...")
            twist = Twist()
            twist.angular.z = config.BASE_ANGULAR_SPEED * (1 if dtheta > 0 else -1)
            self.ctx.publish_twist_for(twist, abs(dtheta) / config.BASE_ANGULAR_SPEED)

        self.ctx.current_pose = (tx, ty, ttheta)

        return json.dumps({
            "status": "success",
            "target": target,
            "estimated_pose": {"x": tx, "y": ty, "theta": ttheta},
            "note": "Open-loop. TODO: Nav2 integration."
        })
