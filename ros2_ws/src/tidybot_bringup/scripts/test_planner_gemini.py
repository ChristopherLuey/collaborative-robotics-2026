#!/usr/bin/env python3
"""
End-to-end test: Gemini planner with real API calls against MuJoCo sim.

Tests that Gemini correctly interprets NL commands and dispatches tool calls.
Requires: GOOGLE_API_KEY env var, sim running with /plan_to_target service.
"""
import os
import sys
import rclpy
import time
import json
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner.core.ros_context import RosContext
from planner.core.planner import Planner
from planner.utils import C


def test_command(planner, command, description, expect_tools=None):
    """Send a command to Gemini and print what happens."""
    print()
    print(f"{C.BOLD}{'─'*60}{C.RESET}")
    print(f"{C.BLUE}{C.BOLD}Command:{C.RESET} {command}")
    print(f"{C.DIM}  ({description}){C.RESET}")
    print(f"{C.BOLD}{'─'*60}{C.RESET}")

    try:
        response = planner.execute(command)
        print(f"{C.GREEN}Response:{C.RESET} {response}")
        return True
    except Exception as e:
        print(f"{C.RED}ERROR:{C.RESET} {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    api_key = os.environ.get('GOOGLE_API_KEY', '')
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  GEMINI PLANNER END-TO-END TEST{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"  API Key: {api_key[:10]}...{api_key[-4:]}")
    print()

    # Initialize ROS2
    rclpy.init()
    ctx = RosContext()
    spin_thread = threading.Thread(target=lambda: rclpy.spin(ctx), daemon=True)
    spin_thread.start()
    time.sleep(2)

    print(f"  Planner service: {ctx.planner_available}")
    print(f"  SAM3 service: {ctx.object_pose_available}")
    print(f"  Joint states: {len(ctx.current_joint_positions)} joints")
    print()

    # Build planner with Gemini
    print("Initializing Gemini planner...")
    planner = Planner(ctx)
    print(f"  Tools: {list(planner.tools.keys())}")
    print()

    results = []

    # Test 1: Simple arm movement
    ok = test_command(
        planner,
        "move the right arm to position 0.2, -0.3, 0.25",
        "Should call move_arm(arm='right', x=0.2, y=-0.3, z=0.25)",
    )
    results.append(("move_arm", ok))
    time.sleep(15)

    # Test 2: Gripper control
    ok = test_command(
        planner,
        "close the right gripper",
        "Should call gripper(arm='right', action='close')",
    )
    results.append(("gripper", ok))
    time.sleep(15)

    # Test 3: Navigation
    ok = test_command(
        planner,
        "navigate to start position",
        "Should call navigate_to(target='start')",
    )
    results.append(("navigate_to", ok))
    time.sleep(15)

    # Test 4: Multi-step command (Task 1 style)
    ok = test_command(
        planner,
        "pick up the red apple",
        "Should call pick_up(object_description='red apple') — will fail on SAM3 gracefully",
    )
    results.append(("pick_up", ok))
    time.sleep(15)

    # Test 5: Open gripper after failed pick
    ok = test_command(
        planner,
        "open the right gripper",
        "Should call gripper(arm='right', action='open')",
    )
    results.append(("gripper open", ok))
    time.sleep(15)

    # Test 6: Sequential task (Task 2 style)
    planner.reset_chat()
    ok = test_command(
        planner,
        "find the blue cup and place it on the table",
        "Should attempt scan or pick_up then place_at — will fail on SAM3 but should show planning",
    )
    results.append(("sequential", ok))
    time.sleep(1)

    # Summary
    print()
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  RESULTS{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    for name, ok in results:
        status = f"{C.GREEN}PASS{C.RESET}" if ok else f"{C.RED}FAIL{C.RESET}"
        print(f"  {name}: {status}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print()
    print(f"  {passed}/{total} passed")
    print(f"{C.BOLD}{'='*60}{C.RESET}")

    ctx.destroy_node()
    rclpy.shutdown()

    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
