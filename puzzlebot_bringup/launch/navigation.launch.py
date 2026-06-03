"""
navigation.launch.py
Lanza los 3 nodos de navegación semántica:
  1. semantic_planner    — A* sobre mapa semántico → /plan
  2. waypoint_controller — seguimiento de waypoints → /cmd_vel
  3. bug2_monitor        — evasión Bug2 con LiDAR   → /cmd_vel + /bug2/active

TFs estáticos:
  - base_link → laser  (LiDAR montado 180° girado)
  - map → odom         (identidad inicial; el SLAM lo corregirá)
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── 1. Planificador semántico A* ──────────────────────────────
        Node(
            package='puzzlebot_navigation',
            executable='semantic_planner',
            name='semantic_planner',
            output='screen',
        ),

        # ── 2. Controlador de waypoints ───────────────────────────────
        Node(
            package='puzzlebot_navigation',
            executable='waypoint_controller',
            name='waypoint_controller',
            output='screen',
        ),

        # ── 3. Monitor Bug2 (evasión de obstáculos) ───────────────────
        Node(
            package='puzzlebot_navigation',
            executable='bug2_monitor',
            name='bug2_monitor',
            output='screen',
        ),

        # ── 4. Map Publisher (publica el mapa) ───────────────────
        Node(
            package='puzzlebot_mapping',
            executable='map_publisher',
            name='map_publisher',
            output='screen',
        ),


        # ── TF estático: base_link → laser ────────────────────────────
        # El LiDAR está montado apuntando hacia atrás (yaw = π)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf',
            arguments=['0', '0', '0', '3.14159', '0', '0', 'base_link', 'laser'],
        ),

        # ── TF estático: map → odom ───────────────────────────────────
        # Transformación identidad inicial; slam_node la sobreescribirá
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        ),
    ])