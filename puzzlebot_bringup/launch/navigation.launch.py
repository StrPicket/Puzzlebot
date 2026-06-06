from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='puzzlebot_navigation',
            executable='semantic_planner',
            name='semantic_planner',
            output='screen',
        ),

        # Node(
        #     package='puzzlebot_navigation',
        #     executable='waypoint_controller',
        #     name='waypoint_controller',
        #     output='screen',
        # ),

        Node(
            package='puzzlebot_navigation',
            executable='bug2_monitor',
            name='bug2_monitor',
            output='screen',
        ),

        Node(
            package='puzzlebot_mapping',
            executable='map_publisher',
            name='map_publisher',
            output='screen',
        ),

        # Node(
        #     package='puzzlebot_navigation',
        #     executable='mission_manager',
        #     name='mission_manager',
        #     output='screen',
        # ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf',
            arguments=[
                '0', '0', '0',
                '3.14159', '0', '0',
                'base_link', 'laser'
            ],
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=[
                '0', '0', '0',
                '0', '0', '0',
                'map', 'odom'
            ],
        ),
    ])