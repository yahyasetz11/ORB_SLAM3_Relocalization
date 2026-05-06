# Main navigation launch file — starts the full navigation stack
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='navigation',
            executable='map_publisher',
            name='map_publisher',
        ),
        Node(
            package='navigation',
            executable='navigation_ui',
            name='navigation_ui',
        ),
        Node(
            package='navigation',
            executable='navigation',
            name='navigation_node',
        ),
        Node(
            package='navigation',
            executable='position_publisher',
            name='position_publisher',
        ),
    ])
