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

        arm = 'left' if grasp_y > 0 else 'right'
        log_info(f"Using {arm} arm")

        distance = math.sqrt(grasp_x**2 + grasp_y**2)
        if distance > 0.6:
            log_info(f"Object too far ({distance:.2f}m), approaching...")
            heading = math.atan2(grasp_y, grasp_x)
            if not self.ctx.approach_pose(grasp_x - 0.5 * math.cos(heading),
                                          grasp_y - 0.5 * math.sin(heading),
                                          heading,
                                          relative=True):
                return json.dumps({"status": "error", "message": "approach_pose failed — cannot reach object."})

            # Re-detect object after repositioning (old coords are stale)
            time.sleep(1.0)
            log_info(f"Re-locating '{object_description}' after approach...")
            detection = self.ctx.call_object_isolator(object_description)
            if detection is None:
                return json.dumps({
                    "status": "error",
                    "message": f"Lost sight of '{object_description}' after approaching."
                })
            grasp_x = detection.x
            grasp_y = detection.y
            grasp_z = detection.z
            log_info(f"Updated grasp: ({grasp_x:.3f}, {grasp_y:.3f}, {grasp_z:.3f})")

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

        # Verify grasp by checking gripper finger position
        # Prismatic finger joint > 0 means fingers didn't fully close → object present
        gripper_joint = f'{arm}_left_finger'
        with self.ctx.joint_lock:
            finger_pos = self.ctx.current_joint_positions.get(gripper_joint, 0.0)
        if finger_pos > 0.005:
            log_info(f"Grasp confirmed ({gripper_joint}={finger_pos:.4f})")
        else:
            log_info(f"Grasp may have missed ({gripper_joint}={finger_pos:.4f}), retrying...")
            self.ctx.set_gripper(arm, closed=False)
            time.sleep(0.5)
            if not self.ctx.request_arm_motion(arm, 'grab', grasp_x, grasp_y, grasp_z):
                return json.dumps({"status": "error", "message": "IK failed on grasp retry."})
            time.sleep(2.0)
            with self.ctx.joint_lock:
                finger_pos = self.ctx.current_joint_positions.get(gripper_joint, 0.0)
            if finger_pos <= 0.005:
                self.ctx.set_gripper(arm, closed=False)
                return json.dumps({"status": "error", "message": "Grasp failed after retry."})
            log_info(f"Grasp confirmed on retry ({gripper_joint}={finger_pos:.4f})")


        lift_z = grasp_z + config.DEFAULT_LIFT_HEIGHT
        log_info(f"Lifting to z={lift_z:.3f}...")

        if not self.ctx.request_arm_motion(arm, 'move', grasp_x, grasp_y, lift_z):
            return json.dumps({"status": "error", "message": "IK failed for lift."})

        time.sleep(2.0)

        self.ctx.holding_object = True

        return json.dumps({
            "status": "success",
            "object": object_description,
            "arm_used": arm,
            "grasp_position": {"x": grasp_x, "y": grasp_y, "z": grasp_z},
        })