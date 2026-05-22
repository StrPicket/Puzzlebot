#!/usr/bin/env python3
"""
mapeo.py  --  corre en la LAPTOP
Suscribe : /scan  (sensor_msgs/LaserScan)
           /odom  (nav_msgs/Odometry)
Publica  : /map   (nav_msgs/OccupancyGrid)  compatible con RViz2

FIXES:
  1. origin_x / origin_y centrados para que el robot arranque en el centro del mapa
  2. range_max ignorado correctamente (no marca celdas libres hasta el infinito)
  3. Logs de debug para verificar que llegan datos
"""

import math
import random
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy import qos

from sensor_msgs.msg import LaserScan
from nav_msgs.msg    import OccupancyGrid, Odometry
from std_msgs.msg    import Header
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, TransformStamped


# ---------------------------------------------------------------------------
# Bresenham
# ---------------------------------------------------------------------------

def bresenham(x0: int, y0: int, x1: int, y1: int):
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

    return cells

class Particle:

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = 1.0

# ---------------------------------------------------------------------------
# Nodo principal
# ---------------------------------------------------------------------------

class OccupancyGridNode(Node):

    def __init__(self):
        super().__init__('occupancy_grid_node')

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self) # base_link → laser (estático)
        self.publish_static_laser_tf()
        self.robot_yaw = 0.0
        self.prev_map_x = 0.0
        self.prev_map_y = 0.0
        self.prev_map_theta = 0.0

        # ------------------------------------------------------------------ #
        # Particle Filter
        # ------------------------------------------------------------------ #

        self.num_particles = 500

        self.particles = [
            Particle(
                random.gauss(0.0, 0.05),
                random.gauss(0.0, 0.05),
                random.gauss(0.0, 0.03)
            )
            for _ in range(self.num_particles)
        ]

        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_odom_yaw = 0.0

        # ------------------------------------------------------------------ #
        # Parámetros del mapa
        # ------------------------------------------------------------------ #
        self.resolution = 0.05    # metros por celda  (5 cm)
        self.width_m    = 5.591       # ancho total del mapa en metros  ← más grande para no salirse
        self.height_m   = 3.788       # alto  total del mapa en metros

        self.l_occ  =  0.7
        self.l_free = -0.4
        self.l_min  = -5.0
        self.l_max  =  5.0

        self.range_min_threshold = 0.10   # metros — ignora lecturas muy cercanas
        # ------------------------------------------------------------------ #

        self.cols = int(self.width_m  / self.resolution)
        self.rows = int(self.height_m / self.resolution)

        # FIX 1: origen centrado — el robot arranca en (0,0) y el mapa
        # tiene espacio en todas las direcciones
        self.origin_x = -(self.width_m  / 2.0)   # esquina inferior izquierda
        self.origin_y = -(self.height_m / 2.0)

        self.logodds = np.zeros((self.rows, self.cols), dtype=np.float32)

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.odom_ok   = False

        # Contadores para debug
        self._scan_count = 0
        self._odom_count = 0

        # Publisher
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', 10)

        # Subscriptions
        self.sub_scan = self.create_subscription(
            LaserScan, 'scan', self.scan_cb,
            qos.qos_profile_sensor_data)

        self.sub_odom = self.create_subscription(
            Odometry, 'odom', self.odom_cb,
            qos.qos_profile_sensor_data)

        # Mapa a 1 Hz
        self.create_timer(1.0, self.publish_map)

        # Timers TF
        self.create_timer(0.1, self.publish_tf)

        # Log de estado cada 5 s
        self.create_timer(5.0, self._status_log)

        self.get_logger().info(
            f'occupancy_grid_node listo  '
            f'({self.cols}x{self.rows} celdas, {self.resolution*100:.0f} cm/celda)\n'
            f'Mapa: {self.width_m}x{self.height_m} m  '
            f'origen: ({self.origin_x:.2f}, {self.origin_y:.2f}) m\n'
            f'El robot debe arrancar en (0,0) en odom para quedar centrado.'
        )

    # ---------------------------------------------------------------------- #
    # Odometría
    # ---------------------------------------------------------------------- #

    def odom_cb(self, msg: Odometry):

        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        new_yaw = math.atan2(siny_cosp, cosy_cosp)

        # ---------------------------------------------------------------
        # Delta odometría
        # ---------------------------------------------------------------

        dx = new_x - self.last_odom_x
        dy = new_y - self.last_odom_y

        dtheta = math.atan2(
            math.sin(new_yaw - self.last_odom_yaw),
            math.cos(new_yaw - self.last_odom_yaw)
        )

        self.last_odom_x = new_x
        self.last_odom_y = new_y
        self.last_odom_yaw = new_yaw

        # ---------------------------------------------------------------
        # Ignorar micro movimiento
        # ---------------------------------------------------------------

        motion = abs(dx) + abs(dy) + abs(dtheta)

        if motion < 0.0005:
            noise_enabled = False
        else:
            noise_enabled = True

        # ---------------------------------------------------------------
        # Motion model MCL
        # ---------------------------------------------------------------
        dist = math.sqrt(dx**2 + dy**2)

        for p in self.particles:

            if noise_enabled:
                noise_x = random.gauss(0.0, 0.0005)
                noise_y = random.gauss(0.0, 0.0005)
                noise_theta = random.gauss(0.0, 0.003)
            else:
                noise_x = 0.0
                noise_y = 0.0
                noise_theta = 0.0

            # movimiento relativo
            p.x     += dist * math.cos(p.theta) + noise_x
            p.y     += dist * math.sin(p.theta) + noise_y
            p.theta += dtheta + random.gauss(0, 0.003)

            # normalizar theta
            p.theta = math.atan2(
                math.sin(p.theta),
                math.cos(p.theta)
            )

        # ---------------------------------------------------------------
        # Mejor partícula temporal
        # ---------------------------------------------------------------

        self.robot_x = sum(p.x * p.weight for p in self.particles)
        self.robot_y = sum(p.y * p.weight for p in self.particles)

        sin_sum = sum(math.sin(p.theta) * p.weight for p in self.particles)
        cos_sum = sum(math.cos(p.theta) * p.weight for p in self.particles)

        self.robot_yaw = math.atan2(sin_sum, cos_sum)

        self.odom_ok = True
        self._odom_count += 1

    # ---------------------------------------------------------------------- #
    # Scan
    # ---------------------------------------------------------------------- #

    def scan_cb(self, msg: LaserScan):

        if not self.odom_ok:
            return

        # ===============================================================
        # MCL UPDATE
        # ===============================================================

        for p in self.particles:
            p.weight = self.compute_particle_weight(p, msg)

        self.normalize_weights()
        self.resample_particles()

        best = max(self.particles, key=lambda p: p.weight)

        self.robot_x = best.x
        self.robot_y = best.y
        self.robot_yaw = best.theta

        # ===============================================================
        # MOVEMENT GATING
        # ===============================================================

        dx = self.robot_x - self.prev_map_x
        dy = self.robot_y - self.prev_map_y

        dist_moved = math.sqrt(dx * dx + dy * dy)

        angle_moved = abs(
            math.atan2(
                math.sin(self.robot_yaw - self.prev_map_theta),
                math.cos(self.robot_yaw - self.prev_map_theta)
            )
        )

        # ---------------------------------------------------------------
        # NO mapear durante giros grandes
        # ---------------------------------------------------------------

        if angle_moved > 0.12:
            return

        # ---------------------------------------------------------------
        # Detectar movimiento estable
        # ---------------------------------------------------------------

        moving_straight = (
            dist_moved > 0.02 and
            angle_moved < 0.05
        )

        robot_still = (
            dist_moved < 0.01 and
            angle_moved < 0.02
        )

        if not (moving_straight or robot_still):
            return

        # ===============================================================
        # MAP UPDATE
        # ===============================================================

        rx = self.robot_x
        ry = self.robot_y
        ryaw = self.robot_yaw

        robot_col, robot_row = self._world_to_cell(rx, ry)

        if not self._in_bounds(robot_col, robot_row):
            self.get_logger().warn(
                f'Robot fuera del grid: ({rx:.2f}, {ry:.2f})',
                throttle_duration_sec=5.0
            )
            return

        angle = msg.angle_min

        # ---------------------------------------------------------------
        # Scan subsampling
        # ---------------------------------------------------------------

        for i in range(0, len(msg.ranges), 4):

            dist = msg.ranges[i]

            if math.isnan(dist) or math.isinf(dist):
                angle += msg.angle_increment * 2
                continue

            if dist < self.range_min_threshold:
                angle += msg.angle_increment * 2
                continue

            # limitar rango útil
            dist = min(dist, 3.0)

            hit = dist < (msg.range_max - 0.05)

            world_angle = ryaw + angle

            hit_x = rx + dist * math.cos(world_angle)
            hit_y = ry + dist * math.sin(world_angle)

            hit_col, hit_row = self._world_to_cell(hit_x, hit_y)

            # -----------------------------------------------------------
            # FREE CELLS
            # -----------------------------------------------------------

            for col, row in bresenham(
                robot_col,
                robot_row,
                hit_col,
                hit_row
            ):

                if self._in_bounds(col, row):

                    self.logodds[row, col] = np.clip(
                        self.logodds[row, col] + self.l_free,
                        self.l_min,
                        self.l_max
                    )

            # -----------------------------------------------------------
            # OCCUPIED CELL
            # -----------------------------------------------------------

            if hit and self._in_bounds(hit_col, hit_row):

                self.logodds[hit_row, hit_col] = np.clip(
                    self.logodds[hit_row, hit_col] + self.l_occ,
                    self.l_min,
                    self.l_max
                )

            angle += msg.angle_increment * 2

        # ===============================================================
        # STORE PREVIOUS MAP POSE
        # ===============================================================

        self.prev_map_x = self.robot_x
        self.prev_map_y = self.robot_y
        self.prev_map_theta = self.robot_yaw

        self._scan_count += 1

    # ---------------------------------------------------------------------- #
    # Publicar mapa
    # ---------------------------------------------------------------------- #

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = self.resolution
        msg.info.width      = self.cols
        msg.info.height     = self.rows
        # CRÍTICO: origin debe coincidir con self.origin_x/y
        msg.info.origin.position.x    = self.origin_x
        msg.info.origin.position.y    = self.origin_y
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        prob     = 1.0 - 1.0 / (1.0 + np.exp(self.logodds))
        grid_int = np.full((self.rows, self.cols), -1, dtype=np.int8)
        known    = self.logodds != 0.0
        grid_int[known] = (prob[known] * 100).astype(np.int8)

        msg.data = grid_int.flatten().tolist()
        self.map_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Log de estado
    # ---------------------------------------------------------------------- #

    def _status_log(self):
        self.get_logger().info(
            f'odom msgs={self._odom_count}  scan msgs={self._scan_count}  '
            f'robot=({self.robot_x:.2f}, {self.robot_y:.2f})  '
            f'odom_ok={self.odom_ok}'
        )

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _world_to_cell(self, x: float, y: float):
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return col, row

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows
    
    def publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        # Corrección: map_pose - odom_pose
        t.transform.translation.x = self.robot_x - self.last_odom_x
        t.transform.translation.y = self.robot_y - self.last_odom_y
        t.transform.translation.z = 0.0

        yaw_correction = self.robot_yaw - self.last_odom_yaw
        t.transform.rotation.z = math.sin(yaw_correction / 2.0)
        t.transform.rotation.w = math.cos(yaw_correction / 2.0)

        self.tf_broadcaster.sendTransform(t)
    
    def publish_static_laser_tf(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'laser'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.10

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 1.0
        t.transform.rotation.w = 0.0

        self.static_tf_broadcaster.sendTransform(t)

    def compute_particle_weight(self, particle, scan_msg):

        known_cells = np.count_nonzero(self.logodds)
        if known_cells < 50:
            return 1.0

        score = 0.0
        angle = scan_msg.angle_min

        # Pre-calcular mapa de distancias (solo cuando el mapa cambia significativamente)
        # Por ahora: buscar celda ocupada más cercana en ventana pequeña
        SIGMA = 0.15   # metros — tolerancia de matching
        sigma_cells = max(1, int(SIGMA / self.resolution))

        for i in range(0, len(scan_msg.ranges), 4):

            dist = scan_msg.ranges[i]

            if math.isnan(dist) or math.isinf(dist):
                angle += scan_msg.angle_increment * 4
                continue
            if dist < self.range_min_threshold:
                angle += scan_msg.angle_increment * 4
                continue

            dist = min(dist, 2.5)

            world_angle = particle.theta + angle
            hit_x = particle.x + dist * math.cos(world_angle)
            hit_y = particle.y + dist * math.sin(world_angle)

            col, row = self._world_to_cell(hit_x, hit_y)

            # ── Likelihood field: buscar celda ocupada más cercana ──────────
            if self._in_bounds(col, row):

                # ventana de búsqueda alrededor del hit
                best_dist_sq = float('inf')

                r0 = max(0,            row - sigma_cells)
                r1 = min(self.rows-1,  row + sigma_cells)
                c0 = max(0,            col - sigma_cells)
                c1 = min(self.cols-1,  col + sigma_cells)

                window = self.logodds[r0:r1+1, c0:c1+1]
                occ_positions = np.argwhere(window > 0.3)

                if len(occ_positions) > 0:
                    # distancia en celdas al ocupado más cercano
                    dr = occ_positions[:, 0] - (row - r0)
                    dc = occ_positions[:, 1] - (col - c0)
                    dist_sq = dr**2 + dc**2
                    best_dist_sq = float(dist_sq.min())

                    # gaussiana sobre distancia métrica
                    dist_m = math.sqrt(best_dist_sq) * self.resolution
                    score += math.exp(-(dist_m**2) / (2 * SIGMA**2))
                else:
                    score -= 0.1   # penalización suave si no hay ocupado cercano

            angle += scan_msg.angle_increment * 4

        return max(score, 0.0001)
        
    def normalize_weights(self):
        total = sum(p.weight for p in self.particles)

        if total <= 0.0:
            return

        for p in self.particles:
            p.weight /= total

    def resample_particles(self):

        weights = [p.weight for p in self.particles]
        n_eff = 1.0 / sum(w**2 for w in weights)

        if n_eff > self.num_particles * 0.5:
            return  # partículas todavía diversas, no resamplear
        
        indices = np.random.choice(
            len(self.particles),
            size=len(self.particles),
            p=weights
        )

        new_particles = []

        for idx in indices:

            p = self.particles[idx]

            new_particles.append(
                Particle(
                    p.x + random.gauss(0.0, 0.002),
                    p.y + random.gauss(0.0, 0.002),
                    p.theta + random.gauss(0.0, 0.01)
                )
            )

        self.particles = new_particles


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