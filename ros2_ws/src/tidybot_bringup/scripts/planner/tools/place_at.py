"""
place_at(target_description) — Complete placement pipeline.

Assumes robot is holding an object. Finds target, navigates close, places precisely.

Pipeline:
  1. If target pose unknown: internal scan
  2. Navigate near target
  3. Visual servo: align arm over target using depth
  4. Lower to placement height (from target surface depth)
  5. Open gripper
  6. Retract arm
  7. Verify placement (optional camera check)

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

    @property
    def description(self) -> str:
        return (
            "Complete placement pipeline. Takes a NL description of the destination "
            "(e.g. 'brown basket', 'the table'). Assumes robot is holding an object."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target_description": {
                    "type": "string",
                    "description": "Natural language description of where to place the object."
                }
            },
            "required": ["target_description"]
        }

    def run(self, target_description: str) -> str:
        if not self.ctx.holding_object:
            log_info("Warning: not holding an object. Proceeding anyway.")

        log_info(f"Place at: '{target_description}'")

        # TODO: Scan for target, get 3D position from Object Isolator
        place_x, place_y, place_z = 0.3, 0.1, config.DEFAULT_GRASP_HEIGHT
        hover_z = place_z + config.DEFAULT_HOVER_CLEARANCE

        # Step 1: Move above placement
        log_info(f"Moving above target ({place_x}, {place_y}, {hover_z})...")
        self.ctx.plan_and_execute('right', place_x, place_y, hover_z)
        time.sleep(2.5)

        # Step 2: Lower
        log_info(f"Lowering to z={place_z:.3f}...")
        self.ctx.plan_and_execute('right', place_x, place_y, place_z, duration=1.5)
        time.sleep(2.0)

        # Step 3: Release
        log_info("Releasing object...")
        self.ctx.set_gripper('right', closed=False)
        time.sleep(0.5)

        # Step 4: Retract
        log_info("Retracting...")
        self.ctx.plan_and_execute('right', place_x, place_y, hover_z + 0.05, duration=1.5)
        time.sleep(2.0)

        self.ctx.holding_object = False

        return json.dumps({
            "status": "success",
            "target": target_description,
            "placement_position": {"x": place_x, "y": place_y, "z": place_z}
        })
