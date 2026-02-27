"""
Gemini planner core — manages the chat loop with tool dispatch.

Handles both text and voice input modalities. The planner:
1. Receives NL input (text or transcribed speech)
2. Sends to Gemini with tool definitions
3. Executes any function calls Gemini returns
4. Feeds results back to Gemini
5. Loops until Gemini returns a text response (plan complete)
"""

from typing import Dict, Optional, Callable

import google.generativeai as genai

from planner.tools.base_tool import BaseTool
from planner.core.tool_registry import discover_tools, build_gemini_tools
from planner.core.ros_context import RosContext
from planner import config
from planner.utils import log_tool, log_result, log_error, log_gemini, log_info


class Planner:
    """
    Stateful Gemini planner that dispatches tool calls to registered actions.

    Usage:
        planner = Planner(ros_context)
        response_text = planner.execute("pick up the red apple")
    """

    def __init__(self, ctx: RosContext):
        self.ctx = ctx

        # Discover and register tools
        log_info("Registering tools...")
        self.tools: Dict[str, BaseTool] = discover_tools(ctx)
        self.gemini_tools = build_gemini_tools(self.tools)

        # Initialize Gemini model + chat
        self.model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            tools=self.gemini_tools,
            system_instruction=config.SYSTEM_PROMPT,
        )
        self.chat = self.model.start_chat()
        log_info(f"Planner ready — {len(self.tools)} tools, model={config.GEMINI_MODEL}")

    def reset_chat(self):
        """Start a fresh conversation (clears multi-turn context)."""
        self.chat = self.model.start_chat()

    def execute(self, user_input: str, on_status: Optional[Callable[[str], None]] = None) -> str:
        """
        Process a natural language command through the full plan-execute loop.

        Args:
            user_input: Natural language command from the user.
            on_status: Optional callback for intermediate status updates
                       (used by voice interface to speak progress).

        Returns:
            Gemini's final text response summarizing what was done.
        """
        response = self.chat.send_message(user_input)

        while response.candidates[0].content.parts:
            # Extract function calls
            function_calls = [
                part for part in response.candidates[0].content.parts
                if part.function_call.name
            ]

            if not function_calls:
                break  # No more tool calls — Gemini returned text

            # Execute each function call
            tool_responses = []
            for part in function_calls:
                fc = part.function_call
                name = fc.name
                args = dict(fc.args) if fc.args else {}

                log_tool(name, args)

                # Status callback for voice
                if on_status:
                    on_status(f"Executing {name}...")

                # Dispatch to registered tool
                if name in self.tools:
                    try:
                        result_str = self.tools[name].run(**args)
                    except Exception as e:
                        result_str = f'{{"status":"error","message":"Tool error: {e}"}}'
                        log_error(name, str(e))
                else:
                    result_str = f'{{"status":"error","message":"Unknown tool: {name}"}}'
                    log_error(name, f"Not registered")

                log_result(name, result_str)

                tool_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=name,
                            response={"result": result_str}
                        )
                    )
                )

            # Feed results back to Gemini
            response = self.chat.send_message(
                genai.protos.Content(parts=tool_responses)
            )

        # Extract final text
        text_parts = [
            part.text for part in response.candidates[0].content.parts
            if hasattr(part, 'text') and part.text
        ]
        final_text = '\n'.join(text_parts) if text_parts else "(no response)"

        log_gemini(final_text)
        return final_text
