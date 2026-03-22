from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share    = get_package_share_directory('orb_slam3_relocalization')
    reloc_params  = os.path.join(pkg_share, 'config', 'relocalization_params.yaml')
    camera_params = os.path.join(pkg_share, 'config', 'camera_params.yaml')

    mode     = LaunchConfiguration('mode')
    is_video = PythonExpression(["'", mode, "' == 'video'"])

    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='video',
            description='Camera source: video (default) or stream (webcam)'
        ),

        # ── camera_node: video mode ───────────────────────────────────────────
        Node(
            package='yolo_bbox',
            executable='camera_node',
            name='camera_node',
            output='screen',
            condition=IfCondition(is_video),
            parameters=[camera_params],
        ),

        # ── camera_node: stream (webcam) mode ────────────────────────────────
        Node(
            package='yolo_bbox',
            executable='camera_node',
            name='camera_node',
            output='screen',
            condition=UnlessCondition(is_video),
            parameters=[camera_params, {'video_path': ''}],
        ),

        # ── relocalization_node ───────────────────────────────────────────────
        Node(
            package='orb_slam3_relocalization',
            executable='relocalization_node',
            name='relocalization_node',
            output='screen',
            parameters=[reloc_params],
        ),
    ])
