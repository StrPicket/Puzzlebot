#!/usr/bin/env python3
"""
map_publisher.py
═══════════════════════════════════════════════════════════════════════
Publica el mapa semántico como nav_msgs/OccupancyGrid para visualizarlo
en RViz como fondo, con el origen correctamente alineado al frame odom
(origen en el centro del mapa).

Tópicos publicados:
  /map  — nav_msgs/OccupancyGrid  (frame_id = "map")
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

SEMANTIC_MAP_PATH       = "/home/strpicket/semantic_map.png"
MAP_PATH                = "/home/strpicket/route_map.png" 
MAP_RESOLUTION = 0.05   # m/pixel — mismo valor que semantic_planner


# ═══════════════════════════════════════════════════════════════════════
#  NODO
# ═══════════════════════════════════════════════════════════════════════

class MapPublisherNode(Node):
    def __init__(self):
        super().__init__('map_publisher')

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.grid_msg = self._build_occupancy_grid()

        if self.grid_msg is None:
            self.get_logger().error(f'No se pudo cargar el mapa: {SEMANTIC_MAP_PATH}')
            return

        # Publicar a 1 Hz (latched-like)
        self.timer = self.create_timer(1.0, self._publish)
        self.get_logger().info(
            f'map_publisher listo — publicando /map a 1 Hz\n'
            f'  Tamaño : {self.grid_msg.info.width}x{self.grid_msg.info.height} celdas\n'
            f'  Origen : ({self.grid_msg.info.origin.position.x:.2f}, '
            f'{self.grid_msg.info.origin.position.y:.2f}) m\n'
            f'  Frame  : {self.grid_msg.header.frame_id}')

    def _build_occupancy_grid(self):
        semantic = cv2.imread(SEMANTIC_MAP_PATH)
        base_map = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
        if semantic is None or base_map is None:
            return None

        H, W, _ = semantic.shape

        # Zonas navegables del semántico
        green_mask = (
            (semantic[:, :, 1] > 200) &
            (semantic[:, :, 0] < 80)  &
            (semantic[:, :, 2] < 80)
        )
        blue_mask = (
            (semantic[:, :, 0] > 200) &
            (semantic[:, :, 1] < 80)  &
            (semantic[:, :, 2] < 80)
        )
        red_mask = (
            (semantic[:, :, 2] > 200) &
            (semantic[:, :, 0] < 80)  &
            (semantic[:, :, 1] < 80)
        )
        navigable = green_mask | blue_mask | red_mask

        # Partir del mapa real como base (0-255 → 0-100 para OccupancyGrid)
        # En el PGM: blanco=libre(255), negro=ocupado(0)
        # En OccupancyGrid: 0=libre, 100=ocupado
        grid = (100 - (base_map.astype(np.int16) * 100 // 255)).astype(np.int8)

        # Sobreescribir zonas navegables semánticas como completamente libres
        grid[navigable] = 30

        origin_x = -(W / 2) * MAP_RESOLUTION
        origin_y = -(H / 2) * MAP_RESOLUTION

        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = MAP_RESOLUTION
        msg.info.width      = W
        msg.info.height     = H
        msg.info.origin.position.x    = origin_x
        msg.info.origin.position.y    = origin_y
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        #grid_flipped = np.flipud(grid)
        grid_flipped = np.flipud(np.fliplr(grid))
        msg.data = grid_flipped.flatten().tolist()

        return msg

    def _publish(self):
        if self.grid_msg is None:
            return
        self.grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.grid_msg)


# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()