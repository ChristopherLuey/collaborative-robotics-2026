"""
pick_up(object_description) — Complete object retrieval pipeline.

Takes a NL description of the object. Handles everything from finding
the object to holding it in the gripper.

Pipeline:
  1. If object pose unknown: internal scan (rotate + Gemini detect)
  2. Navigate near object via Nav2 (~1m away)
  3. Visual servo: depth camera tracks object continuously
  4. Drive base + position arm for reachable grasp pose
  5. Compute top-down grasp from depth point cloud
  6. Execute grasp trajectory
  7. Close gripper
  8. Lift to carry height
  9. Verify grasp (gripper width check)
  10. Retry from step 4 if failed (max 2 retries)

TODO: Visual servo loop (classical CV, <100ms latency — NOT through Gemini).
TODO: Proper 3D pose from depth + Object Isolator Node.
TODO: Grasp verification via gripper width.
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

    @property
    def description(self) -> str:
        return (
            "Complete object retrieval pipeline. Takes a natural language description "
            "(e.g. 'red apple', 'the cup on the left'). Handles finding the object, "
            "navigating close, computing a top-down grasp, and grabbing it. "
            "Designed for tabletop objects within ~450g payload."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "Natural language description of the object to pick up."
                }
            },
            "required": ["object_description"]
        }

    def run(self, object_description: str) -> str:
        log_info(f"Pick up: '{object_description}'")

        # TODO: Step 1 — call Object Isolator Node (Max's service) to get 3D pose
        # For now, use default position in right arm workspace
        grasp_x, grasp_y, grasp_z = 0.3, -0.1, config.DEFAULT_GRASP_HEIGHT
        log_info(f"Target grasp: ({grasp_x}, {grasp_y}, {grasp_z}) [DEFAULT — TODO: perception]")

        # Step 2: Open gripper
        log_info("Opening gripper...")
        self.ctx.set_gripper('right', closed=False)
        time.sleep(0.5)

        # Step 3: Hover above object
        hover_z = grasp_z + config.DEFAULT_HOVER_CLEARANCE
        log_info(f"Hovering at z={hover_z:.3f}...")
        if not self.ctx.plan_and_execute('right', grasp_x, grasp_y, hover_z):
            return json.dumps({"status": "error", "message": "IK failed for hover position."})
        time.sleep(2.5)

        # Step 4: Descend to grasp
        log_info(f"Descending to z={grasp_z:.3f}...")
        if not self.ctx.plan_and_execute('right', grasp_x, grasp_y, grasp_z, duration=1.5):
            return json.dumps({"status": "error", "message": "IK failed for grasp position."})
        time.sleep(2.0)

        # Step 5: Close gripper
        log_info("Closing gripper...")
        self.ctx.set_gripper('right', closed=True)
        time.sleep(1.0)

        # Step 6: Lift
        lift_z = grasp_z + config.DEFAULT_LIFT_HEIGHT
        log_info(f"Lifting to z={lift_z:.3f}...")
        self.ctx.plan_and_execute('right', grasp_x, grasp_y, lift_z, duration=1.5)
        time.sleep(2.0)

        # TODO: Step 7 — verify grasp (check gripper width > threshold)

        self.ctx.holding_object = True

        return json.dumps({
            "status": "success",
            "object": object_description,
            "grasp_position": {"x": grasp_x, "y": grasp_y, "z": grasp_z},
            "note": "Used default position. TODO: perception-driven grasp pose."
        })
