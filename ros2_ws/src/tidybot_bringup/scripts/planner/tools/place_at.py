"""
place_at(target_description) — Complete placement pipeline.

TODO: Visual servo, surface height detection from depth.
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

        place_x, place_y, place_z = 0.3, 0.1, config.DEFAULT_GRASP_HEIGHT
        hover_z = place_z + config.DEFAULT_HOVER_CLEARANCE

        log_info(f"Moving above target ({place_x}, {place_y}, {hover_z})...")
        self.ctx.plan_and_execute('right', place_x, place_y, hover_z)
        time.sleep(2.5)

        log_info(f"Lowering to z={place_z:.3f}...")
        self.ctx.plan_and_execute('right', place_x, place_y, place_z, duration=1.5)
        time.sleep(2.0)

        log_info("Releasing object...")
        self.ctx.set_gripper('right', closed=False)
        time.sleep(0.5)

        log_info("Retracting...")
        self.ctx.plan_and_execute('right', place_x, place_y, hover_z + 0.05, duration=1.5)
        time.sleep(2.0)

        self.ctx.holding_object = False

        return json.dumps({
            "status": "success",
            "target": target_description,
            "placement_position": {"x": place_x, "y": place_y, "z": place_z}
        })
