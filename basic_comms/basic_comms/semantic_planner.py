#!/usr/bin/env python3
"""
semantic_planner.py
═══════════════════════════════════════════════════════════════════════
Nodo 1 de 3 — Planificación A* semántica

CORRECCIÓN DE SISTEMAS DE COORDENADAS
──────────────────────────────────────
El mapa semántico (imagen PNG) tiene su origen en la esquina
superior-izquierda (px=0, py=0).

La odometría del robot tiene su origen en el CENTRO del mapa:
  odom_origin_world = (W/2 * res, H/2 * res)

Para convertir entre ambos:

  mundo_imagen  = odom + odom_origin_world
  odom          = mundo_imagen - odom_origin_world

Todas las coordenadas que se publican en /plan están en el
frame odométrico del robot (lo que espera waypoint_controller).

FUNCIONALIDAD
─────────────
• Se suscribe a /odom para conocer la posición inicial del robot.
• Acepta destino por terminal en dos formatos:
    - Nombre de zona:   "carga" | "descarga" | "racks"
    - Coordenadas:      "x y"  (en metros, frame odométrico)
• Corre A* desde la posición actual hasta el destino.
• Publica el path en /plan como nav_msgs/Path (frame odom).
• Ofrece servicio /replan para recalcular bajo demanda.

Tópicos suscritos:
  /odom  — nav_msgs/Odometry

Tópicos publicados:
  /plan  — nav_msgs/Path  (frame_id = "odom")

Servicios:
  /replan — std_srvs/Trigger
"""

import rclpy
from rclpy.node import Node
from rclpy import qos
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from scipy.interpolate import splprep, splev
import cv2
import heapq
import numpy as np
import math

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

MAP_PATH       = "/home/juanjo/semantic_map.png"
MAP_RESOLUTION = 0.05   # m/pixel

# Submuestreo: publicar 1 de cada N puntos del path crudo
PATH_SUBSAMPLE = 5

# Nombres de zona reconocidos (terminal)
ZONE_NAMES = ("carga", "descarga", "racks")


# ═══════════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE COORDENADAS
# ═══════════════════════════════════════════════════════════════════════

class CoordConverter:
    """
    Centraliza todas las conversiones entre los tres sistemas:

      PIXEL  (px, py)  — origen esquina sup-izq, y crece hacia abajo
      WORLD  (wx, wy)  — origen esquina inf-izq de la imagen, y crece hacia arriba
                         wx = px * res
                         wy = (H-1-py) * res
      ODOM   (ox, oy)  — origen en el CENTRO del mapa
                         ox = wx - W/2*res
                         oy = wy - H/2*res
    """

    def __init__(self, W: int, H: int, resolution: float):
        self.W   = W
        self.H   = H
        self.res = resolution
        # Offset del origen odométrico en coordenadas mundo-imagen
        self.ox_offset = (W / 2) * resolution
        self.oy_offset = (H / 2) * resolution

    # ── Pixel → World (imagen) ─────────────────────────────────────────
    def pixel_to_world(self, px: int, py: int):
        wx = px * self.res
        wy = (self.H - 1 - py) * self.res
        return wx, wy

    # ── World → Pixel ──────────────────────────────────────────────────
    def world_to_pixel(self, wx: float, wy: float):
        px = int(wx / self.res)
        py = int((self.H - 1) - wy / self.res)
        px = int(np.clip(px, 0, self.W - 1))
        py = int(np.clip(py, 0, self.H - 1))
        return px, py

    # ── Odom → World ───────────────────────────────────────────────────
    def odom_to_world(self, ox: float, oy: float):
        wx = ox + self.ox_offset
        wy = oy + self.oy_offset
        return wx, wy

    # ── World → Odom ───────────────────────────────────────────────────
    def world_to_odom(self, wx: float, wy: float):
        ox = wx - self.ox_offset
        oy = wy - self.oy_offset
        return ox, oy

    # ── Odom → Pixel (combinado, el más usado) ─────────────────────────
    def odom_to_pixel(self, ox: float, oy: float):
        wx, wy = self.odom_to_world(ox, oy)
        return self.world_to_pixel(wx, wy)

    # ── Pixel → Odom (combinado) ───────────────────────────────────────
    def pixel_to_odom(self, px: int, py: int):
        wx, wy = self.pixel_to_world(px, py)
        return self.world_to_odom(wx, wy)


# ═══════════════════════════════════════════════════════════════════════
#  MAPA SEMÁNTICO
# ═══════════════════════════════════════════════════════════════════════

def load_map(path: str):
    semantic = cv2.imread(path)
    if semantic is None:
        raise FileNotFoundError(f"No se pudo cargar el mapa: {path}")
    return semantic


def build_navigable_mask(semantic: np.ndarray):
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
    nav = np.zeros(semantic.shape[:2], dtype=np.uint8)
    nav[green_mask] = 1
    nav[blue_mask]  = 1
    nav[red_mask]   = 1
    return nav, {"carga": green_mask, "racks": blue_mask, "descarga": red_mask}


def get_zone_center_px(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))


# ═══════════════════════════════════════════════════════════════════════
#  A*
# ═══════════════════════════════════════════════════════════════════════

def heuristic(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def astar(grid, distance_map, start, goal):
    H, W    = grid.shape
    MOVES   = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    close_set = set()
    came_from = {}
    gscore    = {start: 0}
    fscore    = {start: heuristic(start, goal)}
    heap      = [(fscore[start], start)]

    while heap:
        current = heapq.heappop(heap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)
        for dx, dy in MOVES:
            nb = (current[0] + dx, current[1] + dy)
            nx, ny = nb
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 0:
                continue
            if dx != 0 and dy != 0:
                if grid[current[1], current[0] + dx] == 0:
                    continue
                if grid[current[1] + dy, current[0]] == 0:
                    continue
            base_cost = heuristic(current, nb)

            # Penalización por cercanía a obstáculos
            d = distance_map[ny, nx]

            # evitar división por cero
            obstacle_penalty = 8.0 / (d + 1.0)

            tg = gscore[current] + base_cost + obstacle_penalty
            if nb in close_set and tg >= gscore.get(nb, 0):
                continue
            if tg < gscore.get(nb, float('inf')):
                came_from[nb] = current
                gscore[nb]    = tg
                fscore[nb]    = tg + heuristic(nb, goal)
                heapq.heappush(heap, (fscore[nb], nb))
    return None


# ═══════════════════════════════════════════════════════════════════════
#  NODO
# ═══════════════════════════════════════════════════════════════════════

class SemanticPlannerNode(Node):

    def __init__(self):
        super().__init__('semantic_planner')

        # ── Cargar mapa ───────────────────────────────────────────────
        self.semantic  = load_map(MAP_PATH)
        H, W, _        = self.semantic.shape
        self.nav, self.zones = build_navigable_mask(self.semantic)
        # Distancia a obstáculos
        self.distance_map = cv2.distanceTransform(
            self.nav.astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        self.distance_map = self.distance_map / np.max(self.distance_map)
        self.conv      = CoordConverter(W, H, MAP_RESOLUTION)

        self.get_logger().info(
            f'Mapa cargado: {W}×{H} px  |  '
            f'Offset odométrico: ({self.conv.ox_offset:.2f}, {self.conv.oy_offset:.2f}) m')

        # ── Estado ────────────────────────────────────────────────────
        self.robot_ox  = 0.0   # posición actual en frame odom
        self.robot_oy  = 0.0
        self.odom_ok   = False

        self.path_msg  = None   # último plan calculado
        self.goal_desc = None   # descripción del destino (para logs)

        # ── ROS ───────────────────────────────────────────────────────
        self.plan_pub = self.create_publisher(Path, '/plan', 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback,
            qos.qos_profile_sensor_data)

        self.replan_srv = self.create_service(
            Trigger, '/replan', self._replan_callback)

        # Publica el plan periódicamente (1 Hz, tipo latched)
        self.timer_pub = self.create_timer(1.0, self._publish_plan)

        # Suscriptor para recibir destino desde otra terminal
        self.goal_sub = self.create_subscription(
            String, '/goal_input', self._goal_input_callback, 10)

        self.get_logger().info(
            'semantic_planner listo.\n'
            '  Envía el destino con:\n'
            '  ros2 topic pub --once /goal_input std_msgs/String "data: \'descarga\'"\n'
            '  ros2 topic pub --once /goal_input std_msgs/String "data: \'1.5 -0.8\'"')

    # ─────────────────────────────────────────────────────────────────
    #  CALLBACK ODOMETRÍA
    # ─────────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        self.robot_ox = msg.pose.pose.position.x
        self.robot_oy = msg.pose.pose.position.y
        if not self.odom_ok:
            self.odom_ok = True
            wx, wy = self.conv.odom_to_world(self.robot_ox, self.robot_oy)
            self.get_logger().info(
                f'Odometría recibida — '
                f'odom=({self.robot_ox:.2f}, {self.robot_oy:.2f}) m  |  '
                f'world=({wx:.2f}, {wy:.2f}) m')

    # ─────────────────────────────────────────────────────────────────
    #  CALLBACK /goal_input
    # ─────────────────────────────────────────────────────────────────

    def _goal_input_callback(self, msg: String):
        raw = msg.data.strip().lower()
        self.get_logger().info(f'Destino recibido: "{raw}"')

        goal_px = self._parse_goal(raw)
        if goal_px is None:
            self.get_logger().warn(
                f'Destino no reconocido: "{raw}" | '
                f'Zonas válidas: {", ".join(ZONE_NAMES)} | '
                f'O coordenadas: "x y" en metros (ej: "1.5 -0.8")')
            return

        self._run_astar(goal_px, description=raw)

    def _parse_goal(self, text: str):
        """
        Devuelve (px, py) del destino o None si no se reconoce.

        Acepta:
          - Nombre de zona: "carga" | "descarga" | "racks"
          - Coordenadas odom: "x y"  (dos números en metros)
        """
        # ── Zona semántica ────────────────────────────────────────────
        words = text.strip().split()
        for name in ZONE_NAMES:
            if name in words:
                center = get_zone_center_px(self.zones[name])
                if center is None:
                    print(f"  ✗ Zona '{name}' no encontrada en el mapa")
                    return None
                return center   # ya es (px, py)

        # ── Coordenadas numéricas ─────────────────────────────────────
        parts = text.split()
        if len(parts) == 2:
            try:
                ox, oy = float(parts[0]), float(parts[1])
                px, py = self.conv.odom_to_pixel(ox, oy)
                print(f"  → Destino odom=({ox:.2f}, {oy:.2f}) m  "
                      f"pixel=({px}, {py})")
                return (px, py)
            except ValueError:
                pass

        return None

    # ─────────────────────────────────────────────────────────────────
    #  PLANIFICACIÓN
    # ─────────────────────────────────────────────────────────────────

    def _smooth_path(self, path):
        """
        Elimina puntos innecesarios manteniendo el path
        completamente dentro de zonas navegables.
        """

        if len(path) < 3:
            return path

        simplified = [path[0]]

        prev_dx = path[1][0] - path[0][0]
        prev_dy = path[1][1] - path[0][1]

        for i in range(1, len(path) - 1):

            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]

            # mantener solo cuando cambia dirección
            if (dx, dy) != (prev_dx, prev_dy):
                simplified.append(path[i])

            prev_dx, prev_dy = dx, dy

        simplified.append(path[-1])

        return simplified

    def _run_astar(self, goal_px, description=""):
        if not self.odom_ok:
            print("  ✗ Aún no se recibe odometría — espera a que /odom publique")
            return

        # Posición actual en píxeles (desde odometría)
        start_px = self.conv.odom_to_pixel(self.robot_ox, self.robot_oy)

        print(f"  Inicio odom =({self.robot_ox:.2f}, {self.robot_oy:.2f}) m  "
              f"→ pixel {start_px}")
        print(f"  Meta        = pixel {goal_px}")
        print("  Ejecutando A*...")

        # Verificar navegabilidad de inicio y meta
        sx, sy = start_px
        gx, gy = goal_px
        if self.nav[sy, sx] == 0:
            print(f"  ⚠ Inicio px=({sx},{sy}) no es navegable — "
                  "buscando celda navegable más cercana...")
            start_px = self._nearest_navigable(start_px)
            if start_px is None:
                print("  ✗ No se encontró celda navegable cerca del inicio")
                return
            print(f"    Ajustado a {start_px}")

        if self.nav[gy, gx] == 0:
            print(f"  ⚠ Meta px=({gx},{gy}) no es navegable — "
                  "buscando celda navegable más cercana...")
            goal_px = self._nearest_navigable(goal_px)
            if goal_px is None:
                print("  ✗ No se encontró celda navegable cerca de la meta")
                return
            print(f"    Ajustado a {goal_px}")

        raw_path = astar(self.nav, self.distance_map,start_px, goal_px)

        if raw_path is None:
            print("  ✗ A* no encontró trayectoria")
            return

        # Submuestrear
        sampled = raw_path[::PATH_SUBSAMPLE]
        if sampled[-1] != raw_path[-1]:
            sampled.append(raw_path[-1])
        sampled = self._smooth_path(sampled)  # suavizar

        print(f"  ✓ Path: {len(raw_path)} px crudos → {len(sampled)} waypoints")

        # Construir nav_msgs/Path en frame odom
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp    = self.get_clock().now().to_msg()

        for px, py in sampled:
            ox, oy = self.conv.pixel_to_odom(px, py)
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp    = msg.header.stamp
            pose.pose.position.x = ox
            pose.pose.position.y = oy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_msg  = msg
        self.goal_desc = description

        # Publicar inmediatamente
        self.plan_pub.publish(self.path_msg)
        print(f"  ✓ /plan publicado ({len(sampled)} poses, frame=odom)")

    def _nearest_navigable(self, px_py, search_radius=20):
        """Busca la celda navegable más cercana a px_py dentro de search_radius."""
        px, py = px_py
        best_dist = math.inf
        best = None
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = px + dx, py + dy
                if not (0 <= nx < self.conv.W and 0 <= ny < self.conv.H):
                    continue
                if self.nav[ny, nx] == 1:
                    d = math.hypot(dx, dy)
                    if d < best_dist:
                        best_dist = d
                        best = (nx, ny)
        return best

    # ─────────────────────────────────────────────────────────────────
    #  PUBLICAR PLAN (1 Hz latched)
    # ─────────────────────────────────────────────────────────────────

    def _publish_plan(self):
        if self.path_msg is None:
            return
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.plan_pub.publish(self.path_msg)

    # ─────────────────────────────────────────────────────────────────
    #  SERVICIO /replan
    # ─────────────────────────────────────────────────────────────────

    def _replan_callback(self, request, response):
        """Recalcula el último destino conocido desde la posición actual."""
        if self.goal_desc is None:
            response.success = False
            response.message = 'Sin destino previo — escribe uno en la terminal'
            return response

        goal_px = self._parse_goal(self.goal_desc)
        if goal_px is None:
            response.success = False
            response.message = f'No se pudo resolver el destino "{self.goal_desc}"'
            return response

        self._run_astar(goal_px, description=self.goal_desc)

        if self.path_msg is not None:
            response.success = True
            response.message = f'{len(self.path_msg.poses)} waypoints generados'
        else:
            response.success = False
            response.message = 'A* falló'
        return response


# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = SemanticPlannerNode()
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