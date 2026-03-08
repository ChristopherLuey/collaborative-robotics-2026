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

# ── Launch service nodes needed by planner ─────────────────
SCRIPTS_DIR="$WS/src/tidybot_bringup/scripts"

echo "Starting approach_node.py (/approach_pose service)..."
python3 "$SCRIPTS_DIR/approach_node.py" &
APPROACH_PID=$!

echo "Starting pickup_object_real.py (/request_arm_motion service)..."
python3 "$SCRIPTS_DIR/pickup_object_real.py" &
ARM_PID=$!

echo "Starting sam3_pointcloud_node.py (/sam3/get_object_pose service)..."
python3 "$SCRIPTS_DIR/sam3_pointcloud_node.py" &
SAM3_PID=$!

echo "Waiting 5s for service nodes..."
sleep 5

# ── Run planner ────────────────────────────────────────────
echo ""
echo "Starting planner ($MODE)..."
cd "$SCRIPTS_DIR"

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SAM3_PID $ARM_PID $APPROACH_PID $HW_PID 2>/dev/null
    wait $SAM3_PID $ARM_PID $APPROACH_PID $HW_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

python3 -m planner $MODE
