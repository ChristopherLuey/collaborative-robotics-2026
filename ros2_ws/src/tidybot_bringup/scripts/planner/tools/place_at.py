"""
place_at(target_description) — Complete placement pipeline.

Uses SAM3 perception to locate the target, then places the held object there.
"""

import json
import time
import math

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

        # Locate the target via vision
        log_info(f"Locating '{target_description}' via Object Isolator...")
        detection = self.ctx.call_object_isolator(target_description)
        if detection is None:
            return json.dumps({
                "status": "error",
                "message": f"Target '{target_description}' not found. Try calling scan() first."
            })
        place_x = detection.x
        place_y = detection.y
        # Place slightly above detected surface
        place_z = detection.z + 0.03
        log_info(f"Target found at ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})")

        # Pick arm based on target Y position
        arm = 'left' if place_y > 0 else 'right'
        log_info(f"Using {arm} arm")

        # Approach if target is beyond arm reach
        distance = math.sqrt(place_x**2 + place_y**2)
        if distance > 0.6:
            log_info(f"Target too far ({distance:.2f}m), approaching...")
            heading = math.atan2(place_y, place_x)
            if not self.ctx.approach_pose(place_x - 0.5 * math.cos(heading),
                                          place_y - 0.5 * math.sin(heading),
                                          heading,
                                          relative=True):
                return json.dumps({"status": "error", "message": "approach_pose failed — cannot reach target."})

            # Re-detect target after repositioning (old coords are stale)
            time.sleep(1.0)
            log_info(f"Re-locating '{target_description}' after approach...")
            detection = self.ctx.call_object_isolator(target_description)
            if detection is None:
                return json.dumps({
                    "status": "error",
                    "message": f"Lost sight of '{target_description}' after approaching."
                })
            place_x = detection.x
            place_y = detection.y
            place_z = detection.z + 0.03
            log_info(f"Updated placement: ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})")

        hover_z = place_z + config.DEFAULT_HOVER_CLEARANCE

        log_info(f"Moving above target ({place_x:.3f}, {place_y:.3f}, {hover_z:.3f})...")
        if not self.ctx.request_arm_motion(arm, 'move', place_x, place_y, hover_z):
            return json.dumps({"status": "error", "message": "IK failed for hover position."})
        time.sleep(2.5)

        log_info(f"Lowering and releasing at z={place_z:.3f}...")
        if not self.ctx.request_arm_motion(arm, 'release', place_x, place_y, place_z):
            return json.dumps({"status": "error", "message": "IK failed for place position."})
        time.sleep(2.0)

        log_info("Retracting...")
        self.ctx.request_arm_motion(arm, 'move', place_x, place_y, hover_z + 0.05)
        time.sleep(2.0)

        self.ctx.holding_object = False

        return json.dumps({
            "status": "success",
            "target": target_description,
            "arm_used": arm,
            "placement_position": {"x": place_x, "y": place_y, "z": place_z}
        })
