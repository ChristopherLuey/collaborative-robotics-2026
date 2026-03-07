"""
Gemini Live API — always-on voice interface for TidyBot2.

Per the spec: Voice is always-on infrastructure, not an action.
It listens, parses NL commands into planner calls, and speaks feedback.

Architecture:
  - Microphone → Gemini Live API (bidirectional audio stream)
  - Gemini transcribes + understands intent
  - Commands are routed to the Planner for tool execution
  - Status updates and results are spoken back via Gemini Live TTS

Requirements:
  pip install google-genai pyaudio

Usage:
  The VoiceInterface is started by main.py in --voice mode.
  It runs as a background asyncio loop alongside the ROS2 spin thread.
"""

import asyncio
import json
import traceback
from typing import Optional

from planner.utils import log_info, log_voice, log_error
from planner import config

try:
    from google import genai as genai_live
    from google.genai import types as genai_types
    GENAI_LIVE_AVAILABLE = True
except ImportError:
    GENAI_LIVE_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


# Audio constants
AUDIO_FORMAT = 'pcm'    # 16-bit PCM
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024


class VoiceInterface:
    """
    Always-on bidirectional voice interface using Gemini Live API.

    Streams microphone audio to Gemini, receives transcribed commands,
    routes them through the Planner, and speaks results back.
    """

    def __init__(self, planner):
        """
        Args:
            planner: Planner instance for dispatching tool calls.
        """
        self.planner = planner
        self._running = False

        if not GENAI_LIVE_AVAILABLE:
            raise ImportError(
                "google-genai package required for voice. "
                "Install: pip install google-genai"
            )
        if not PYAUDIO_AVAILABLE:
            raise ImportError(
                "pyaudio required for microphone access. "
                "Install: pip install pyaudio (may need: apt install portaudio19-dev)"
            )

        self.client = genai_live.Client(api_key=config.GOOGLE_API_KEY)
        self.audio = pyaudio.PyAudio()

    async def run(self):
        """
        Main voice loop. Connects to Gemini Live API and streams audio
        bidirectionally until stopped.
        """
        self._running = True
        log_info("Starting voice interface (Gemini Live)...")

        # Build tool declarations from planner's registered tools
        live_tools = []
        for tool in self.planner.tools.values():
            live_tools.append(tool.declaration())

        model_config = genai_types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name="Kore"  # Clear, neutral voice
                    )
                )
            ),
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=(
                    config.SYSTEM_PROMPT + "\n\n"
                    "You are speaking with a human operator. Be concise and natural. "
                    "Announce what you're doing before executing actions. "
                    "After completing actions, briefly confirm what happened."
                ))]
            ),
            tools=[genai_types.Tool(function_declarations=live_tools)],
        )

        try:
            async with self.client.aio.live.connect(
                model=config.GEMINI_LIVE_MODEL,
                config=model_config,
            ) as session:
                log_info("Connected to Gemini Live. Listening...")

                # Start concurrent tasks for sending and receiving audio
                send_task = asyncio.create_task(self._send_audio(session))
                recv_task = asyncio.create_task(self._receive_and_handle(session))

                await asyncio.gather(send_task, recv_task)

        except Exception as e:
            log_error("voice", f"Live API error: {e}")
            traceback.print_exc()
        finally:
            self._running = False
            log_info("Voice interface stopped.")

    async def _send_audio(self, session):
        """Capture microphone audio and stream to Gemini Live."""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SEND_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        log_info("Microphone active. Speak naturally.")

        try:
            while self._running:
                data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                await session.send(
                    input=genai_types.LiveClientRealtimeInput(
                        media_chunks=[
                            genai_types.Blob(data=data, mime_type=f"audio/{AUDIO_FORMAT}")
                        ]
                    )
                )
        finally:
            stream.stop_stream()
            stream.close()

    async def _receive_and_handle(self, session):
        """
        Receive responses from Gemini Live — audio, text, and tool calls.
        Route tool calls through the planner, speak results.
        """
        playback_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RECV_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        try:
            while self._running:
                turn = session.receive()
                async for response in turn:
                    # Handle audio output (Gemini speaking)
                    if hasattr(response, 'data') and response.data:
                        await asyncio.to_thread(playback_stream.write, response.data)

                    # Handle text (transcription or response)
                    if hasattr(response, 'text') and response.text:
                        log_voice("out", response.text)

                    # Handle tool calls
                    if hasattr(response, 'tool_call') and response.tool_call:
                        for fc in response.tool_call.function_calls:
                            name = fc.name
                            args = dict(fc.args) if fc.args else {}

                            log_voice("in", f"[Tool call: {name}({args})]")

                            # Execute tool via planner's registered tools
                            if name in self.planner.tools:
                                try:
                                    result_str = self.planner.tools[name].run(**args)
                                except Exception as e:
                                    result_str = json.dumps({"status": "error", "message": str(e)})
                            else:
                                result_str = json.dumps({"status": "error", "message": f"Unknown tool: {name}"})

                            log_voice("out", f"[Result: {result_str[:100]}...]")

                            # Send result back to Gemini Live
                            await session.send(
                                input=genai_types.LiveClientToolResponse(
                                    function_responses=[
                                        genai_types.FunctionResponse(
                                            name=name,
                                            response={"result": result_str},
                                        )
                                    ]
                                )
                            )
        finally:
            playback_stream.stop_stream()
            playback_stream.close()

    def stop(self):
        """Stop the voice interface."""
        self._running = False

    def cleanup(self):
        """Release audio resources."""
        self.audio.terminate()
