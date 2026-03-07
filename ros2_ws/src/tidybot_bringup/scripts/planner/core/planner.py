"""
Gemini planner core — manages the chat session with automatic tool calling.

Uses the new google-genai SDK which handles:
  - Auto-generating FunctionDeclarations from Python functions
  - Automatic function calling (executes tools, feeds results back)
  - Multi-step planning loops

The planner just sends a message and gets back the final text response.
"""

from typing import Dict, Optional, Callable

from google import genai
from google.genai import types

from planner.tools.base_tool import BaseTool
from planner.core.tool_registry import discover_tools, build_tool_functions
from planner.core.ros_context import RosContext
from planner import config
from planner.utils import log_info, log_gemini


class Planner:
    """
    Stateful Gemini planner with automatic function calling.

    Usage:
        planner = Planner(ros_context)
        response_text = planner.execute("pick up the red apple")
    """

    def __init__(self, ctx: RosContext):
        self.ctx = ctx

        # Discover and register tools
        log_info("Registering tools...")
        self.tools: Dict[str, BaseTool] = discover_tools(ctx)
        self.tool_functions = build_tool_functions(self.tools)

        # Initialize google-genai client
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)

        # Create chat session with automatic function calling
        self.chat = self.client.chats.create(
            model=config.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                tools=self.tool_functions,
                system_instruction=config.SYSTEM_PROMPT,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=False  # Enabled — SDK calls our functions automatically
                ),
            ),
        )

        log_info(f"Planner ready — {len(self.tools)} tools, model={config.GEMINI_MODEL}")

    def reset_chat(self):
        """Start a fresh conversation (clears multi-turn context)."""
        self.chat = self.client.chats.create(
            model=config.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                tools=self.tool_functions,
                system_instruction=config.SYSTEM_PROMPT,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=False
                ),
            ),
        )

    def execute(self, user_input: str, on_status: Optional[Callable[[str], None]] = None) -> str:
        """
        Process a natural language command.

        With automatic_function_calling enabled, the SDK handles the full loop:
          user message → Gemini returns function call → SDK executes our function
          → SDK sends result back → Gemini returns next call or final text

        Args:
            user_input: Natural language command from the user.
            on_status: Optional callback for status updates (voice mode).

        Returns:
            Gemini's final text response.
        """
        response = self.chat.send_message(user_input)

        final_text = response.text if response.text else "(no response)"
        log_gemini(final_text)
        return final_text
