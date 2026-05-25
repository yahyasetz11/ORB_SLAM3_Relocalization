# TEST ONLY — remove before deployment

# launch/test_navigation.launch.py
#
# Launches the full fake navigation test stack:
#   fake_data_publisher  →  world_map + current_position
#   navigation_node      →  path planning, publishes navigation_image
#   map_ui_node          →  UI display + goal selection

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='navigation',
            executable='fake_data_publisher',
            name='fake_data_publisher',
            output='screen',
        ),

        Node(
            package='navigation',
            executable='navigation',
            name='navigation_node',
            output='screen',
        ),

        Node(
            package='navigation',
            executable='navigation_ui',
            name='map_ui_node',
            output='screen',
        ),

    ])
