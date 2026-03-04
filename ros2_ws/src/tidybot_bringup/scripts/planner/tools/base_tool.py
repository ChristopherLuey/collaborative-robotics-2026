"""
Abstract base class for all planner tools.

To add a new action:
  1. Create a new file in tools/
  2. Subclass BaseTool
  3. Implement name, description, and run() with typed parameters + docstring
  4. It's auto-discovered by the registry — no other changes needed.

The new google-genai SDK auto-generates FunctionDeclarations from Python
functions using their name, docstring, and type annotations. Each tool
exposes a `callable` property that returns a plain function for this purpose.
"""

from abc import ABC, abstractmethod
from typing import Any

from planner.core.ros_context import RosContext


class BaseTool(ABC):
    """Base class for all robot action tools."""

    def __init__(self, ctx: RosContext):
        self.ctx = ctx

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (e.g. 'scan')."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> str:
        """
        Execute the tool. Returns a JSON string with results.

        All implementations should return JSON with at minimum:
          {"status": "success"|"error"|"not_implemented", ...}
        """
        ...

    @property
    def callable(self):
        """
        Return a plain function suitable for the google-genai SDK's
        automatic function calling. The SDK reads __name__, __doc__,
        and type annotations to build FunctionDeclarations automatically.
        """
        return self.run
