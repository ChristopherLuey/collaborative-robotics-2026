"""
gripper(arm, action) — Open or close a gripper independently.
"""

import json
import time

from planner.tools.base_tool import BaseTool
from planner.utils import log_info


class GripperTool(BaseTool):

    @property
    def name(self) -> str:
        return "gripper"

    def run(self, arm: str, action: str) -> str:
        """Open or close a gripper on one arm.

        Args:
            arm: Which arm's gripper to control ('right' or 'left').
            action: 'open' to release or 'close' to grip.
        """
        if arm not in ('right', 'left'):
            return json.dumps({"status": "error", "message": "arm must be 'right' or 'left'"})

        closed = action.lower().strip() == 'close'
        log_info(f"{'Closing' if closed else 'Opening'} {arm} gripper...")
        self.ctx.set_gripper(arm, closed=closed)
        time.sleep(0.5)

        if closed:
            self.ctx.holding_object = True
        else:
            self.ctx.holding_object = False

        return json.dumps({
            "status": "success",
            "arm": arm,
            "action": "close" if closed else "open"
        })
