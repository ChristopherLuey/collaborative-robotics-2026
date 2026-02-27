#!/usr/bin/env python3
"""
TidyBot2 Gemini Planner — entry point.

Two modes:
  --text   Interactive CLI (default). Type commands, Gemini plans & executes.
  --voice  Always-on voice via Gemini Live API. Speak commands, hear responses.

Usage:
  GOOGLE_API_KEY=<key> python3 -m planner               # text mode
  GOOGLE_API_KEY=<key> python3 -m planner --voice        # voice mode
"""

import sys
import argparse
import threading

import rclpy

from planner import config
from planner.utils import C, log_info
from planner.core.ros_context import RosContext
from planner.core.planner import Planner


def run_text_mode(planner: Planner):
    """Interactive text CLI loop."""
    print()
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  TidyBot2 Gemini Planner (text mode){C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.DIM}  Type natural language commands. 'quit' to exit.{C.RESET}")
    print(f"{C.DIM}  Tools: {', '.join(planner.tools.keys())}{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print()

    while True:
        try:
            user_input = input(f"{C.BLUE}{C.BOLD}You: {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            break

        try:
            planner.execute(user_input)
        except Exception as e:
            print(f"{C.RED}Error: {e}{C.RESET}")
            import traceback
            traceback.print_exc()
        print()


def run_voice_mode(planner: Planner):
    """Always-on voice interface via Gemini Live API."""
    import asyncio
    from planner.voice.gemini_live import VoiceInterface

    voice = VoiceInterface(planner)

    print()
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  TidyBot2 Gemini Planner (voice mode){C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.DIM}  Speak naturally. Press Ctrl+C to stop.{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print()

    try:
        asyncio.run(voice.run())
    except KeyboardInterrupt:
        voice.stop()
    finally:
        voice.cleanup()


def main():
    parser = argparse.ArgumentParser(description='TidyBot2 Gemini Planner')
    parser.add_argument('--voice', action='store_true', help='Use Gemini Live voice interface')
    parser.add_argument('--text', action='store_true', help='Use text CLI (default)')
    args = parser.parse_args()

    if not config.GOOGLE_API_KEY:
        print(f"{C.RED}Error: GOOGLE_API_KEY environment variable not set.{C.RESET}")
        sys.exit(1)

    # Initialize ROS2
    rclpy.init()
    ctx = RosContext()

    # Spin ROS2 in background
    spin_thread = threading.Thread(target=lambda: rclpy.spin(ctx), daemon=True)
    spin_thread.start()

    # Build planner (auto-discovers tools)
    planner = Planner(ctx)

    try:
        if args.voice:
            run_voice_mode(planner)
        else:
            run_text_mode(planner)
    finally:
        log_info("Shutting down...")
        ctx.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
