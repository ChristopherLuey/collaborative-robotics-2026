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
from planner.utils import log_info, log_tool, log_result, log_error


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
    import functools
    functions = []
    for tool in registry.values():

        def _make_fn(t):
            @functools.wraps(t.run)
            def fn(**kw):
                log_tool(t.name, kw)
                try:
                    result = t.run(**kw)
                    log_result(t.name, result)
                    return result
                except Exception as e:
                    log_error(t.name, str(e))
                    raise
            fn.__name__ = t.name
            return fn

        functions.append(_make_fn(tool))
    return functions
