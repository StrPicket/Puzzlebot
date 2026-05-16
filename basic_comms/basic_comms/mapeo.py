#!/usr/bin/env python3
"""
occupancy_grid_node.py  --  corre en la LAPTOP
Suscribe : /scan  (sensor_msgs/LaserScan)
           /odom  (nav_msgs/Odometry)
Publica  : /map   (nav_msgs/OccupancyGrid)  compatible con RViz

Algoritmo: Occupancy Grid Mapping con log-odds por celda.
           Trazado de rayos con el algoritmo de Bresenham.

Parámetros ajustables al inicio de __init__:
  self.resolution   -- metros por celda  (0.05 = 5 cm/celda)
  self.width_m      -- ancho del mapa en metros
  self.height_m     -- alto  del mapa en metros
  self.l_occ        -- incremento log-odds si celda es ocupada
  self.l_free       -- decremento log-odds si celda es libre
  self.l_min        -- clamping mínimo log-odds
  self.l_max        -- clamping máximo log-odds
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy import qos

from sensor_msgs.msg import LaserScan
from nav_msgs.msg    import OccupancyGrid, Odometry
from std_msgs.msg    import Header
from builtin_interfaces.msg import Time


# ---------------------------------------------------------------------------
# Algoritmo de Bresenham  (traza una línea entre dos celdas en la grid)
# ---------------------------------------------------------------------------

def bresenham(x0: int, y0: int, x1: int, y1: int):
    """
    Genera todas las celdas (col, row) entre (x0,y0) y (x1,y1).
    Devuelve lista de tuplas; la celda final (x1,y1) NO se incluye
    (se trata aparte como celda ocupada).
    """
    cells = []
    dx =  abs(x1 - x0);  sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0);  sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if x0 == x1 and y0 == y1:
            break
        cells.append((x0, y0))
        e2 = 2 * err
        if e2 >= dy:
            if x0 == x1:
                break
            err += dy;  x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            err += dx;  y0 += sy

    return cells   # excluye la celda de impacto


# ---------------------------------------------------------------------------
# Nodo principal
# ---------------------------------------------------------------------------

class OccupancyGridNode(Node):

    def __init__(self):
        super().__init__('occupancy_grid_node')

        # ------------------------------------------------------------------ #
        # Parámetros del mapa  (ajusta según el tamaño de tu entorno)
        # ------------------------------------------------------------------ #
        self.resolution = 0.05      # metros por celda  (5 cm)
        self.width_m    = 5.5      # ancho total del mapa en metros
        self.height_m   = 3.7      # alto  total del mapa en metros

        # Log-odds: cuánto sube/baja la probabilidad por cada rayo
        # Valores típicos: l_occ entre 0.4 y 0.9 / l_free entre -0.4 y -0.2
        self.l_occ  =  0.7
        self.l_free = -0.4
        self.l_min  = -5.0          # evita que celdas libres queden "eternas"
        self.l_max  =  5.0          # evita que obstáculos queden "eternos"

        # Rango mínimo del LiDAR a ignorar (lecturas muy cercanas = ruido)
        self.range_min_threshold = 0.05   # metros
        # ------------------------------------------------------------------ #

        # Dimensiones en celdas
        self.cols = int(self.width_m  / self.resolution)
        self.rows = int(self.height_m / self.resolution)

        # Origen del mapa: ponemos al robot al centro para tener espacio en todas direcciones
        self.origin_x = 0.0   # esquina inferior izquierda en metros
        self.origin_y = 0.0

        # Grid de log-odds: empieza en 0.0 (probabilidad 0.5 = desconocido)
        self.logodds = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Pose actual del robot (se actualiza con /odom)
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.odom_ok   = False

        # Publisher del mapa
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', 10)

        # Subscriptions
        self.sub_scan = self.create_subscription(
            LaserScan, 'scan', self.scan_cb,
            qos.qos_profile_sensor_data)

        self.sub_odom = self.create_subscription(
            Odometry, 'odom', self.odom_cb,
            qos.qos_profile_sensor_data)

        # Publicar el mapa a 1 Hz (RViz lo recibe sin saturar la red)
        self.create_timer(1.0, self.publish_map)

        self.get_logger().info(
            f'occupancy_grid_node listo  '
            f'({self.cols}x{self.rows} celdas, {self.resolution*100:.0f} cm/celda, '
            f'{self.width_m}x{self.height_m} m)')

    # ---------------------------------------------------------------------- #
    # Callback odometría  — solo guarda la pose
    # ---------------------------------------------------------------------- #

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Extraer yaw del quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.odom_ok = True

    # ---------------------------------------------------------------------- #
    # Callback scan  — aquí ocurre el mapeo
    # ---------------------------------------------------------------------- #

    def scan_cb(self, msg: LaserScan):
        if not self.odom_ok:
            # No actualizamos hasta tener una pose confiable
            return

        rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw

        # Celda del robot en la grid
        robot_col, robot_row = self._world_to_cell(rx, ry)

        # Recorrer cada rayo del LiDAR
        angle = msg.angle_min
        for dist in msg.ranges:
            # Ignorar lecturas inválidas
            if math.isnan(dist) or math.isinf(dist):
                angle += msg.angle_increment
                continue
            if dist < self.range_min_threshold:
                angle += msg.angle_increment
                continue

            # ¿El rayo llegó a un obstáculo o fue un miss (max range)?
            hit = dist < (msg.range_max - 0.05)

            # Si es miss, usar range_max para marcar celdas libres hasta el límite
            effective_dist = dist if hit else msg.range_max

            # Punto de impacto (o fin del rayo) en coordenadas mundo
            world_angle = ryaw + angle
            hit_x = rx + effective_dist * math.cos(world_angle)
            hit_y = ry + effective_dist * math.sin(world_angle)

            hit_col, hit_row = self._world_to_cell(hit_x, hit_y)

            # Trazar rayo con Bresenham y marcar celdas libres
            free_cells = bresenham(robot_col, robot_row, hit_col, hit_row)
            for col, row in free_cells:
                if self._in_bounds(col, row):
                    self.logodds[row, col] = np.clip(
                        self.logodds[row, col] + self.l_free,
                        self.l_min, self.l_max)

            # Marcar celda de impacto como ocupada (solo si fue hit real)
            if hit and self._in_bounds(hit_col, hit_row):
                self.logodds[hit_row, hit_col] = np.clip(
                    self.logodds[hit_row, hit_col] + self.l_occ,
                    self.l_min, self.l_max)

            angle += msg.angle_increment

    # ---------------------------------------------------------------------- #
    # Publicar OccupancyGrid
    # ---------------------------------------------------------------------- #

    def publish_map(self):
        msg = OccupancyGrid()

        msg.header = Header()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = self.resolution
        msg.info.width      = self.cols
        msg.info.height     = self.rows
        msg.info.origin.position.x    = self.origin_x
        msg.info.origin.position.y    = self.origin_y
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        # Convertir log-odds a probabilidad, luego al rango [0,100] de OccupancyGrid
        # Celdas desconocidas (logodds == 0) se publican como -1
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.logodds))   # sigmoid

        # Mapa de int8: -1 = desconocido, 0 = libre, 100 = ocupado
        grid_int = np.full((self.rows, self.cols), -1, dtype=np.int8)

        known = self.logodds != 0.0
        grid_int[known] = (prob[known] * 100).astype(np.int8)

        # OccupancyGrid se publica row-major, fila 0 = y mínima
        msg.data = grid_int.flatten().tolist()

        self.map_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Helpers de coordenadas
    # ---------------------------------------------------------------------- #

    def _world_to_cell(self, x: float, y: float):
        """Convierte coordenadas mundo (m) a índices (col, row) de la grid."""
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return col, row

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()
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