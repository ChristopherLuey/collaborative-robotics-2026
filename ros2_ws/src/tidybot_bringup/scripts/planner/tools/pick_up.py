"""
pick_up(object_description)  Complete object retrieval pipeline.

TODO: Visual servo loop, proper 3D pose from Object Isolator Node.
"""

import json
import time
import math

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

        #  CHANGE 1 (was line 24) 
        # BEFORE: grasp_x, grasp_y, grasp_z = 0.3, -0.1, config.DEFAULT_GRASP_HEIGHT
        # WHY:    hardcoded point — arm always goes to same place regardless of object
        # NOW:    ask Object Isolator Node for real XYZ from depth camera
        log_info(f"Locating '{object_description}' via Object Isolator...")
        detection = self.ctx.call_object_isolator(object_description)
        if detection is None:
            return json.dumps({
                "status": "error",
                "message": f"Object '{object_description}' not found. Try calling scan() first."
            })
        grasp_x = detection.x
        grasp_y = detection.y
        grasp_z = detection.z
        log_info(f"Target grasp: ({grasp_x:.3f}, {grasp_y:.3f}, {grasp_z:.3f})")

        #  CHANGE 2 (was line 30  'right' hardcoded everywhere) 
        # BEFORE: self.ctx.set_gripper('right', ...)  everywhere
        # WHY:    always right arm even if object is on the left
        # NOW:    pick arm based on Y position of object
        arm = 'left' if grasp_y > 0 else 'right'
        log_info(f"Using {arm} arm")

        #  CHANGE 3 (new  was completely missing) 
        # BEFORE: nothing  arm reached directly even if object was 2m away
        # WHY:    WidowX reach is ~0.65m  anything further = IK fail
        # NOW:    navigate to 0.5m from object before moving arm
        distance = math.sqrt(grasp_x**2 + grasp_y**2)
        if distance > 0.6:
            log_info(f"Object too far ({distance:.2f}m), approaching...")
            heading = math.atan2(grasp_y, grasp_x)
            if not self.ctx.approach_pose(grasp_x - 0.5 * math.cos(heading),
                                          grasp_y - 0.5 * math.sin(heading),
                                          heading):
                return json.dumps({"status": "error", "message": "approach_pose failed — cannot reach object."})

        log_info("Opening gripper...")
        self.ctx.set_gripper(arm, closed=False)
        time.sleep(0.5)

        hover_z = grasp_z + config.DEFAULT_HOVER_CLEARANCE
        log_info(f"Hovering at z={hover_z:.3f}...")
        if not self.ctx.request_arm_motion(arm, 'move', grasp_x, grasp_y, hover_z):
            return json.dumps({"status": "error", "message": "IK failed for hover position."})
        time.sleep(2.5)

        log_info(f"Descending and grasping at z={grasp_z:.3f}...")
        if not self.ctx.request_arm_motion(arm, 'grab', grasp_x, grasp_y, grasp_z):
            return json.dumps({"status": "error", "message": "IK failed for grasp position."})
        time.sleep(2.0)

        #  CHANGE 4 (was line 44 — right after set_gripper closed) 
        # BEFORE: nothing — assumed grasp worked, went straight to lift
        # WHY:    gripper can close on air and code still returns success
        # NOW:    read gripper joint width, retry x2 if empty
        grasped = False
        for attempt in range(2):
            gripper_width = self.ctx.get_arm_positions(arm)[-1]
            if gripper_width > 0.01:
                grasped = True
                log_info(f"Grasp confirmed (width={gripper_width:.3f})")
                break
            log_info(f"Grasp failed (width={gripper_width:.3f}), retry {attempt+1}/2...")
            self.ctx.set_gripper(arm, closed=False)
            time.sleep(0.5)
            self.ctx.request_arm_motion(arm, 'grab', grasp_x, grasp_y, grasp_z)
            time.sleep(2.0)
        if not grasped:
            self.ctx.set_gripper(arm, closed=False)
            return json.dumps({"status": "error", "message": "Grasp failed after 2 attempts."})
        

        lift_z = grasp_z + config.DEFAULT_LIFT_HEIGHT
        log_info(f"Lifting to z={lift_z:.3f}...")

        #  CHANGE 5 (was line 48)
        # BEFORE: self.ctx.plan_and_execute(...)  with no check
        # WHY:    if lift IK fails, code still sets holding_object=True and returns success
        # NOW:    check return value like hover and descend already do
        if not self.ctx.request_arm_motion(arm, 'move', grasp_x, grasp_y, lift_z):
            return json.dumps({"status": "error", "message": "IK failed for lift."})
        # 

        time.sleep(2.0)

        self.ctx.holding_object = True

        return json.dumps({
            "status": "success",
            "object": object_description,
            "arm_used": arm,
            "grasp_position": {"x": grasp_x, "y": grasp_y, "z": grasp_z},
        })