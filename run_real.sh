#!/bin/bash
# TidyBot2 Gemini Planner — real robot version.
#
# Launches real hardware drivers + IK planner + Gemini planner.
# Run this on the robot mini PC (locobot).
#
# Usage:
#   ./run_real.sh              # text mode (default)
#   ./run_real.sh --voice      # voice mode (needs mic + pyaudio)
#   ./run_real.sh --no-rviz    # skip RViz
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS="$SCRIPT_DIR/ros2_ws"
MODE="--text"
USE_RVIZ="true"

for arg in "$@"; do
    case $arg in
        --voice) MODE="--voice" ;;
        --text)  MODE="--text" ;;
        --no-rviz) USE_RVIZ="false" ;;
    esac
done

# ── Environment ────────────────────────────────────────────
# setup_env.bash handles: ROS2, PYTHONPATH, TIDYBOT2_PATH, DDS, domain ID
cd "$WS"
source setup_env.bash

# Interbotix SDK (needed for arm drivers on the real robot)
if [ -f ~/interbotix_humble_ws/install/setup.bash ]; then
    source ~/interbotix_humble_ws/install/setup.bash
    echo "Sourced Interbotix workspace"
fi

# Extra pip packages (google-genai, etc.)
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"

# ── Launch real hardware ───────────────────────────────────
echo ""
echo "Launching real hardware (rviz=$USE_RVIZ)..."
ros2 launch tidybot_bringup real.launch.py \
    use_rviz:=$USE_RVIZ \
    use_planner:=true &
HW_PID=$!
echo "Hardware PID: $HW_PID"
echo "Waiting 10s for hardware to initialize..."
sleep 10

# ── Run planner ────────────────────────────────────────────
echo ""
echo "Starting planner ($MODE)..."
cd src/tidybot_bringup/scripts

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $HW_PID 2>/dev/null
    wait $HW_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

python3 -m planner $MODE
