from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('orb_slam3_relocalization')
    params_file = os.path.join(
        pkg_share, '..', '..', '..', '..', 'src',
        'orb_slam3_relocalization', 'config',
        'localization_test_params.yaml')

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='video',
        description='Input mode: video (default) or stream (webcam)'
    )

    node = Node(
        package='orb_slam3_relocalization',
        executable='localization_test_node',
        name='localization_test_node',
        output='screen',
        parameters=[params_file, {'mode': LaunchConfiguration('mode')}],
    )

    return LaunchDescription([mode_arg, node])
