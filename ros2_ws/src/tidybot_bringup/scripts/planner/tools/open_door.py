"""
open_door() — Door-opening pipeline. NOT YET IMPLEMENTED.

TODO: Coordinated base+arm motion for lever/cabinet handles.
"""

import json

from planner.tools.base_tool import BaseTool
from planner.utils import log_info


class OpenDoorTool(BaseTool):

    @property
    def name(self) -> str:
        return "open_door"

    def run(self) -> str:
        """Open a door. Detects the handle, approaches, grasps, and executes coordinated base+arm motion. Targets cabinet doors or lever-handle doors."""
        log_info("open_door() — NOT YET IMPLEMENTED")
        return json.dumps({
            "status": "not_implemented",
            "message": "open_door requires coordinated base+arm motion. See design doc."
        })
