"""
Tool registry — auto-discovers tool modules from the tools/ package.

Adding a new action:
  1. Create tools/my_action.py
  2. Define a class inheriting BaseTool
  3. It's automatically registered and available to Gemini

No changes needed in the planner, registry, or anywhere else.
"""

import importlib
import pkgutil
from typing import Dict

import google.generativeai as genai

from planner.tools.base_tool import BaseTool
from planner.core.ros_context import RosContext
from planner.utils import log_info


def discover_tools(ctx: RosContext) -> Dict[str, BaseTool]:
    """
    Import all modules in planner.tools, instantiate any BaseTool subclass found,
    and return a name→instance mapping.
    """
    import planner.tools as tools_pkg

    registry: Dict[str, BaseTool] = {}

    for importer, modname, ispkg in pkgutil.iter_modules(tools_pkg.__path__):
        if modname == 'base_tool':
            continue
        module = importlib.import_module(f'planner.tools.{modname}')

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type)
                    and issubclass(attr, BaseTool)
                    and attr is not BaseTool):
                instance = attr(ctx)
                registry[instance.name] = instance
                log_info(f"  Registered tool: {instance.name}")

    return registry


def build_gemini_tools(registry: Dict[str, BaseTool]) -> list:
    """Build the Gemini Tool proto from all registered tools."""
    declarations = []
    for tool in registry.values():
        declarations.append(tool.declaration())
    return [genai.protos.Tool(function_declarations=declarations)]
