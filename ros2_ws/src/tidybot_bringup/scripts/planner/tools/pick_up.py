"""
pick_up(object_description) — Complete object retrieval pipeline.

TODO: Visual servo loop, proper 3D pose from Object Isolator Node.
"""

import json
import time

from planner.tools.base_tool import BaseTool
from planner.utils import log_info
from planner import config


class PickUpTool(BaseTool):

    @property
    def name(self) -> str:
        return "pick_up"

    def run(self, object_description: str) -> str:
        """Pick up an object. Handles finding it, navigating close, grasping, and lifting.

        Args:
            object_description: Natural language description of the object (e.g. 'red apple', 'the cup on the left').
        """
        log_info(f"Pick up: '{object_description}'")

        grasp_x, grasp_y, grasp_z = 0.3, -0.1, config.DEFAULT_GRASP_HEIGHT
        log_info(f"Target grasp: ({grasp_x}, {grasp_y}, {grasp_z}) [DEFAULT — TODO: perception]")

        log_info("Opening gripper...")
        self.ctx.set_gripper('right', closed=False)
        time.sleep(0.5)

        hover_z = grasp_z + config.DEFAULT_HOVER_CLEARANCE
        log_info(f"Hovering at z={hover_z:.3f}...")
        if not self.ctx.plan_and_execute('right', grasp_x, grasp_y, hover_z):
            return json.dumps({"status": "error", "message": "IK failed for hover position."})
        time.sleep(2.5)

        log_info(f"Descending to z={grasp_z:.3f}...")
        if not self.ctx.plan_and_execute('right', grasp_x, grasp_y, grasp_z, duration=1.5):
            return json.dumps({"status": "error", "message": "IK failed for grasp position."})
        time.sleep(2.0)

        log_info("Closing gripper...")
        self.ctx.set_gripper('right', closed=True)
        time.sleep(1.0)

        lift_z = grasp_z + config.DEFAULT_LIFT_HEIGHT
        log_info(f"Lifting to z={lift_z:.3f}...")
        self.ctx.plan_and_execute('right', grasp_x, grasp_y, lift_z, duration=1.5)
        time.sleep(2.0)

        self.ctx.holding_object = True

        return json.dumps({
            "status": "success",
            "object": object_description,
            "grasp_position": {"x": grasp_x, "y": grasp_y, "z": grasp_z},
            "note": "Used default position. TODO: perception-driven grasp pose."
        })
