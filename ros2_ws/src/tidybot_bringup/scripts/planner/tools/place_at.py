"""
place_at(target_description) — Complete placement pipeline.

Uses SAM3 perception to locate the target, then places the held object there.
Falls back to a default position if vision is unavailable.
"""

import json
import time

from planner.tools.base_tool import BaseTool
from planner.utils import log_info
from planner import config


class PlaceAtTool(BaseTool):

    @property
    def name(self) -> str:
        return "place_at"

    def run(self, target_description: str) -> str:
        """Place the held object at a target location.

        Args:
            target_description: Natural language description of where to place the object (e.g. 'brown basket', 'the table').
        """
        if not self.ctx.holding_object:
            log_info("Warning: not holding an object. Proceeding anyway.")

        log_info(f"Place at: '{target_description}'")

        # Try to locate the target via vision
        log_info(f"Locating '{target_description}' via Object Isolator...")
        detection = self.ctx.call_object_isolator(target_description)
        if detection is not None:
            place_x = detection.x
            place_y = detection.y
            # Place slightly above detected surface
            place_z = detection.z + 0.03
            log_info(f"Target found at ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})")
        else:
            place_x, place_y, place_z = 0.3, 0.1, config.DEFAULT_GRASP_HEIGHT
            log_info(f"Vision fallback — using default position ({place_x}, {place_y}, {place_z})")

        # Pick arm based on target Y position
        arm = 'left' if place_y > 0 else 'right'
        log_info(f"Using {arm} arm")

        hover_z = place_z + config.DEFAULT_HOVER_CLEARANCE

        log_info(f"Moving above target ({place_x:.3f}, {place_y:.3f}, {hover_z:.3f})...")
        if not self.ctx.plan_and_execute(arm, place_x, place_y, hover_z):
            return json.dumps({"status": "error", "message": "IK failed for hover position."})
        time.sleep(2.5)

        log_info(f"Lowering to z={place_z:.3f}...")
        if not self.ctx.plan_and_execute(arm, place_x, place_y, place_z, duration=1.5):
            return json.dumps({"status": "error", "message": "IK failed for place position."})
        time.sleep(2.0)

        log_info("Releasing object...")
        self.ctx.set_gripper(arm, closed=False)
        time.sleep(0.5)

        log_info("Retracting...")
        self.ctx.plan_and_execute(arm, place_x, place_y, hover_z + 0.05, duration=1.5)
        time.sleep(2.0)

        self.ctx.holding_object = False

        return json.dumps({
            "status": "success",
            "target": target_description,
            "arm_used": arm,
            "placement_position": {"x": place_x, "y": place_y, "z": place_z}
        })
