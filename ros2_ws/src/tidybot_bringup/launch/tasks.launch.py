from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # Example node
    example_node = Node(
        package='your_package_name',
        executable='your_executable_name',
        name='your_node_name',
        output='screen',
        parameters=[
            {'param_name': 'param_value'}
        ],
        remappings=[
            ('/input_topic', '/remapped_input')
        ]
    )

    return LaunchDescription([
        example_node
    ])