import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nav_params = os.path.join(
        get_package_share_directory('navigation'),
        'config',
        'navigation_params.yaml',
    )

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
            parameters=[nav_params],
        ),
        Node(
            package='navigation',
            executable='position_publisher',
            name='position_publisher',
        ),
    ])
