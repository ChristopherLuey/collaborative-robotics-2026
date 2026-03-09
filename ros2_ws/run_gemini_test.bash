#!/bin/bash
set -e

export PATH="/usr/bin:$PATH"
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONPATH="/home/luey/Code/collaborative-robotics-2026-fork/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=42
export TIDYBOT_REPO_ROOT="/home/luey/Code/collaborative-robotics-2026-fork"

# Load API key from .env
if [ -f "$TIDYBOT_REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$TIDYBOT_REPO_ROOT/.env" | xargs)
fi

echo "Launching sim..."
ros2 launch tidybot_bringup sim.launch.py use_rviz:=false show_mujoco_viewer:=true &
SIM_PID=$!

echo "Sim PID: $SIM_PID"
echo "Waiting 18s for sim to start..."
sleep 18

echo ""
echo "Running Gemini planner end-to-end test..."
cd src/tidybot_bringup/scripts
python3 test_planner_gemini.py
TEST_EXIT=$?

echo "Killing sim..."
kill $SIM_PID 2>/dev/null
wait $SIM_PID 2>/dev/null || true

exit $TEST_EXIT
