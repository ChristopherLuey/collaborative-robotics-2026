"""
Tool registry — auto-discovers tool modules and builds callable functions.

The new google-genai SDK auto-generates FunctionDeclarations from plain Python
functions (using name, docstring, type annotations). We just need to hand it
a list of functions.
"""

import importlib
import pkgutil
from typing import Dict, List, Callable

from planner.tools.base_tool import BaseTool
from planner.core.ros_context import RosContext
from planner.utils import log_info


def discover_tools(ctx: RosContext) -> Dict[str, BaseTool]:
    """
    Import all modules in planner.tools, instantiate BaseTool subclasses,
    return name→instance mapping.
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


def build_tool_functions(registry: Dict[str, BaseTool]) -> List[Callable]:
    """
    Build a list of plain Python callables for the google-genai SDK.
    The SDK reads __name__, __doc__, and type annotations to auto-generate
    FunctionDeclarations — no manual proto building needed.
    """
    functions = []
    for tool in registry.values():
        fn = tool.run
        # Ensure the function has the right __name__ for the SDK
        fn.__name__ = tool.name
        functions.append(fn)
    return functions
