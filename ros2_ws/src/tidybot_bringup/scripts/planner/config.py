"""Configuration constants for TidyBot2 planner."""

import os

# Gemini
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', os.environ.get('GEMINI_API_KEY', ''))
GEMINI_MODEL = 'gemini-2.0-flash'
GEMINI_LIVE_MODEL = 'gemini-2.0-flash-live-001'

# Vertex AI (preferred over API key when available)
USE_VERTEX_AI = os.environ.get('USE_VERTEX_AI', 'true').lower() in ('1', 'true', 'yes')
VERTEX_PROJECT = os.environ.get('VERTEX_PROJECT', 'gen-lang-client-0801728030')
VERTEX_LOCATION = os.environ.get('VERTEX_LOCATION', 'us-central1')

# Named locations (x, y, theta) in world frame
NAMED_LOCATIONS = {
    "start": (0.0, 0.0, 0.0),
    "home": (0.0, 0.0, 0.0),
}

# Arm configuration
SLEEP_POSE = [0.0, -1.80, 1.55, 0.0, 0.8, 0.0]
ARM_NAMES = ['right', 'left']

# Navigation
BASE_LINEAR_SPEED = 0.2   # m/s
BASE_ANGULAR_SPEED = 0.5  # rad/s

# Grasping defaults (TODO: replace with perception-derived poses)
DEFAULT_GRASP_HEIGHT = 0.15       # meters above base
DEFAULT_HOVER_CLEARANCE = 0.05    # meters above grasp
DEFAULT_LIFT_HEIGHT = 0.10        # meters to lift after grasp

# IK planning
IK_TIMEOUT_SEC = 15.0
IK_MAX_CONDITION_NUMBER = 100.0
DEFAULT_MOTION_DURATION = 2.0

# Camera
CAMERA_SETTLE_TIME = 1.0  # seconds to wait after pan/tilt command
SCAN_HEADINGS = 8          # number of base rotations for full scan (8 x 45°)

# System prompt for the Gemini planner
SYSTEM_PROMPT = """You are the high-level planner for TidyBot2, a mobile robot with two WidowX 250 arms.

Your job: translate natural language commands into sequences of robot actions using the available tools.

Robot capabilities:
- 3-DOF differential drive base (x, y, theta)
- 2x WidowX 250 6-DOF arms (~450g payload, parallel jaw gripper)
- RealSense D435 camera (RGB + depth) on a pan-tilt head
- LiDAR for navigation

Planning principles:
1. Always scan first if you don't know where objects are.
2. Actions are self-contained — each handles its own perception when needed.
3. For multi-step tasks, call tools sequentially and check results.
4. If an action fails, try once more or report the failure.
5. Be concise in your text responses.

Available named locations: "start", "home" (both = initial position).

When you're done executing all needed actions, respond with a brief text summary of what was accomplished."""
