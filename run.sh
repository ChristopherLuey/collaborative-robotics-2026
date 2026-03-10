#!/bin/bash
# TidyBot2 Gemini Planner — launches sim + planner in one shot.
#
# Usage:
#   ./run.sh              # text mode (default)
#   ./run.sh --voice      # voice mode (needs mic + pyaudio)
#   ./run.sh --no-viewer  # headless sim (no MuJoCo window)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS="$SCRIPT_DIR/ros2_ws"
MODE="${1:---text}"
SHOW_VIEWER="true"

if [[ "$*" == *"--no-viewer"* ]]; then
    SHOW_VIEWER="false"
    # Remove --no-viewer from args so it doesn't get passed to planner
    MODE=$(echo "$@" | sed 's/--no-viewer//' | xargs)
    MODE="${MODE:---text}"
fi

# ── Environment ────────────────────────────────────────────
# Deactivate conda if active (ROS2 Humble needs system Python 3.10)
if [ -n "$CONDA_PREFIX" ]; then
    echo "Deactivating conda..."
    conda deactivate 2>/dev/null || true
fi

# Use the real desktop display, not virtual framebuffer
if [ "$DISPLAY" = ":99" ] || [ -z "$DISPLAY" ]; then
    # Find the actual X display from logged-in sessions
    REAL_DISPLAY=$(who 2>/dev/null | grep '(:[0-9]' | head -1 | sed 's/.*(\(:[0-9]*\)).*/\1/')
    if [ -n "$REAL_DISPLAY" ]; then
        export DISPLAY="$REAL_DISPLAY"
        echo "Set DISPLAY=$DISPLAY (real desktop)"
    fi
fi

cd "$WS"
source setup_env.bash
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"

# ── Launch sim ─────────────────────────────────────────────
echo "Launching simulation (viewer=$SHOW_VIEWER)..."
ros2 launch tidybot_bringup sim.launch.py use_rviz:=false show_mujoco_viewer:=$SHOW_VIEWER &
SIM_PID=$!
echo "Sim PID: $SIM_PID"
echo "Waiting 18s for sim to start..."
sleep 18

# ── Run planner ────────────────────────────────────────────
echo ""
echo "Starting planner ($MODE)..."
cd src/tidybot_bringup/scripts

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SIM_PID 2>/dev/null
    wait $SIM_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

python3 -m planner $MODE
