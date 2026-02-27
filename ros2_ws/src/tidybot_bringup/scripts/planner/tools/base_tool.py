"""
Abstract base class for all planner tools.

To add a new action:
  1. Create a new file in tools/
  2. Subclass BaseTool
  3. Implement name, description, parameters, and run()
  4. It's auto-discovered by the registry — no other changes needed.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import google.generativeai as genai
from google.generativeai.types import content_types

from planner.core.ros_context import RosContext


class BaseTool(ABC):
    """Base class for all robot action tools."""

    def __init__(self, ctx: RosContext):
        self.ctx = ctx

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name matching the Gemini function name (e.g. 'scan')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for Gemini (explains when/how to use it)."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema dict for tool parameters."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> str:
        """
        Execute the tool. Returns a JSON string with results.

        All implementations should return JSON with at minimum:
          {"status": "success"|"error"|"not_implemented", ...}
        """
        ...

    def declaration(self) -> genai.protos.FunctionDeclaration:
        """Build the Gemini FunctionDeclaration proto for this tool."""
        return genai.protos.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=content_types.to_proto(self.parameters),
        )
