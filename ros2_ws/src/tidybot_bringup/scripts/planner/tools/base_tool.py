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

import inspect
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

    def declaration(self):
        """Build a FunctionDeclaration for the Gemini Live API from run() signature."""
        from google.genai import types as genai_types

        sig = inspect.signature(self.run)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'kwargs'):
                continue
            ptype = param.annotation
            if ptype == float or ptype == int:
                json_type = "number"
            elif ptype == bool:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param_name] = {"type": json_type, "description": param_name}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        doc = inspect.getdoc(self.run) or ""
        # Use text before Args: section as the description
        description = doc.split("\n\nArgs:")[0].strip()
        # Parse Args: section for better parameter descriptions
        if "\nArgs:\n" in doc:
            args_section = doc.split("\nArgs:\n")[1]
            for line in args_section.strip().splitlines():
                line = line.strip()
                if ":" in line:
                    pname, pdesc = line.split(":", 1)
                    pname = pname.strip()
                    if pname in properties:
                        properties[pname]["description"] = pdesc.strip()

        return genai_types.FunctionDeclaration(
            name=self.name,
            description=description,
            parameters={"type": "object", "properties": properties, "required": required},
        )
