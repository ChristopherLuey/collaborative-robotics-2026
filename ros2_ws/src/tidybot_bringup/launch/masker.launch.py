from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tidybot_bringup',   # <-- Replace with your package name
            executable='camera_to_robot_coordinate',  # entry point name (no .py)
            name='camera_to_robot_coordinate_node',
            output='screen'
        ),

        #Node(
        #    package='tidybot_bringup',   # <-- Replace with your package name
        #    executable='get_grasp_pose',  # entry point name (no .py)
        #    name='target_pose_node',
        #    output='screen'
        #)
    ])