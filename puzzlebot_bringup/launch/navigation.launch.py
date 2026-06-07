from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node( # -------Navegacion Semantica A*---------
            package='puzzlebot_navigation',
            executable='semantic_planner',
            name='semantic_planner',
            output='screen',
        ),

        Node( # -------------Waypoints-----------------
            package='puzzlebot_navigation',
            executable='waypoint_controller',
            name='waypoint_controller',
            output='screen',
        ),

        Node( # ----------Evasor Obstaculos-------------
            package='puzzlebot_navigation',
            executable='bug2_monitor',
            name='bug2_monitor',
            output='screen',
        ),

        Node( # ---------------Mapa---------------------
            package='puzzlebot_mapping',
            executable='map_publisher',
            name='map_publisher',
            output='screen',
        ),

        Node( # ----------Maquina Estados---------------
            package='puzzlebot_navigation',
            executable='mission_manager',
            name='mission_manager',
            output='screen',
        ),

        """Node( # ------------Centrado QR------------------
            package='puzzlebot_navigation',
            executable='centerQR',
            name='centerQR',
            output='screen',
        )""",

        """Node( # -----------Centrado Yolo----------------
            package='yolov8_detection',
            executable= 'centeryolo',
            name= 'centeryolo',
            output='screen',
        )""",

        Node( # ------------Deteccion Yolo-----------------
            package='yolov8_detection',
            executable='Trademarks_test',
            name='Trademarks_test',
            output='screen',
            parameters=[
                {'image_topic': '/video_source/compressed'},
                {'use_compressed': True}
            ],
        ),
        
        """Node( # -----------Forklift---------------------
            package='puzzlebot_navigation',
            executable='forklift_routine',
            name='forklift_routine',
            output='screen',
        )""",

        Node( # -------------TF Lidar-----------------
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf',
            arguments=[
                '0', '0', '0',
                '3.14159', '0', '0',
                'base_link', 'laser'
            ],
        ),

        Node( # -------------TF Odom-----------------
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