from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('orb_slam3_relocalization')
    params_file = os.path.join(pkg_share, 'config', 'relocalization_params.yaml')

    mode = LaunchConfiguration('mode')
    is_video = PythonExpression(["'", mode, "' == 'video'"])

    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='video',
            description='Input mode: video (default) or stream (webcam)'
        ),

        # ── Video mode ────────────────────────────────────────────────────────
        Node(
            package='orb_slam3_relocalization',
            executable='relocalization_node',
            name='relocalization_node',
            output='screen',
            condition=IfCondition(is_video),
            parameters=[params_file],
        ),

        # ── Stream (webcam) mode — override video_path to empty ───────────────
        Node(
            package='orb_slam3_relocalization',
            executable='relocalization_node',
            name='relocalization_node',
            output='screen',
            condition=UnlessCondition(is_video),
            parameters=[params_file, {'video_path': ''}],
        ),
    ])
