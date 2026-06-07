#!/usr/bin/env python3
"""
mission_manager.py
═══════════════════════════════════════════════════════════════════════
Nodo maestro — Orquestador de misiones del Puzzlebot

ESQUEMA DE COLORES EN waypoint_map
────────────────────────────────────
  Verde   (0,255,0   BGR) → waypoint solo búsqueda QR
  Azul    (255,0,0   BGR) → waypoint solo tránsito/navegación (reutilizable)
  Rojo    (0,0,255   BGR) → waypoint búsqueda + tránsito (reutilizable, escanea cada vez)
  Morado  (~128,0,128 BGR)→ ruta navegable que conecta los waypoints

BÚSQUEDA DE QR — ESQUEMA DE VISITADOS Y RUTAS
───────────────────────────────────────────────
  • Waypoints VERDES  → se marcan como visitados; no se repiten hasta
                        completar todos los verdes del ciclo.
  • Waypoints ROJOS   → NUNCA se marcan como visitados; son tránsito
                        reutilizable Y además escanean QR cada vez que
                        el planificador los elige como destino.
  • Waypoints AZULES  → NUNCA se marcan como visitados; son tránsito
                        puro reutilizable (sin escaneo).

  BFS para trazar el path entre dos waypoints:
    La máscara navegable = morado | verde | rojo | azul
    (todos los píxeles de color del waypoint_map son transitables)
    Esto evita callejones sin salida al permitir pasar por encima de
    cualquier waypoint intermedio de cualquier color.

  Secuencia al llegar a un waypoint destino:
    - Si es VERDE o ROJO → ejecuta rutina de escaneo (SCAN_ROTATE + SCAN_WAIT)
    - Si es AZUL         → marca como intermedio de tránsito, elige siguiente

  Orden de visita de destinos:
    1. Todos los ROJOS de la zona (reutilizables, se visitan siempre).
    2. Todos los VERDES no visitados en este ciclo.
    El siguiente destino se elige nearest-neighbor desde la posición actual.
    Al agotar los verdes → reiniciar ciclo de verdes (los rojos siguen disponibles).

Máquina de estados principal:
  IDLE → NAVIGATE_TO_ZONE → SEARCH_QR → CENTER_QR
       → LIFT_PALLET → NAVIGATE_TO_TRUCK → DROP_PALLET → DONE

Sub-FSM dentro de SEARCH_QR:
  MOVE_TO_WP → SCAN_ROTATE_A → SCAN_WAIT_A
             → [SCAN_ROTATE_B → SCAN_WAIT_B]  (misión 2)
             → NEXT_WP

Tópicos suscritos:
  /mission              — std_msgs/String
  /nav/status           — std_msgs/String
  /qr/detected          — std_msgs/Bool
  /qr/centered          — std_msgs/Bool
  /aruco/pose_centered  — geometry_msgs/PoseWithCovarianceStamped
  /forklift/status      — std_msgs/String
  /aruco/image_detected/compressed
  /qr/image_detected/compressed

Tópicos publicados:
  /goal_input           — std_msgs/String   (navegación inter-zona)
  /plan                 — nav_msgs/Path     (path sobre ruta navegable)
  /center_qr/enable     — std_msgs/Bool
  /nav/cancel           — std_msgs/Bool
  /mission/status       — std_msgs/String
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np
import math
from collections import deque

# ═══════════════════════════════════════════════════════════════════════
#  RUTAS DE MAPAS Y RESOLUCIÓN
# ═══════════════════════════════════════════════════════════════════════

SEMANTIC_MAP_PATH = "/home/strpicket/semantic_map.png"
ROUTE_MAP_PATH    = "/home/strpicket/waypoint_map.png"
MAP_RESOLUTION    = 0.05   # m/pixel

# ═══════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════

SEARCH_WP_SUBSAMPLE  = 1
SEARCH_WP_MIN_DIST   = 0.10   # m — distancia mínima entre waypoints extraídos
SCAN_ANGLE_TOL       = math.radians(3)
SCAN_WAIT_S          = 3.0
SCAN_OMEGA           = 0.20   # rad/s máx durante giro de escaneo
SCAN_KP_W            = 0.80
QR_LOST_COUNT_THRESHOLD = 30
MISSION1_SCAN_THETA  = 0.0
MISSION2_SCAN_DELTA_A = math.pi / 2.0
MISSION2_SCAN_DELTA_B = -math.pi

# Tolerancia RDP para simplificación de paths (píxeles).
# 1.5 = muy fiel; 3.0 = muy reducido.  2.0 es un buen balance.
RDP_EPSILON = 2.0


# ═══════════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE COORDENADAS
# ═══════════════════════════════════════════════════════════════════════

class CoordConverter:
    def __init__(self, W, H, resolution):
        self.W  = W
        self.H  = H
        self.res = resolution
        self.ox_offset = (W / 2) * resolution
        self.oy_offset = (H / 2) * resolution

    def pixel_to_world(self, px, py):
        return px * self.res, (self.H - 1 - py) * self.res

    def world_to_pixel(self, wx, wy):
        px = int(np.clip(int(wx / self.res),                 0, self.W - 1))
        py = int(np.clip(int((self.H - 1) - wy / self.res), 0, self.H - 1))
        return px, py

    def odom_to_world(self, ox, oy):
        return ox + self.ox_offset, oy + self.oy_offset

    def world_to_odom(self, wx, wy):
        return wx - self.ox_offset, wy - self.oy_offset

    def odom_to_pixel(self, ox, oy):
        return self.world_to_pixel(*self.odom_to_world(ox, oy))

    def pixel_to_odom(self, px, py):
        return self.world_to_odom(*self.pixel_to_world(px, py))


# ═══════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE MÁSCARAS
# ═══════════════════════════════════════════════════════════════════════

def build_zone_masks(semantic: np.ndarray):
    green = (semantic[:,:,1] > 200) & (semantic[:,:,0] < 80) & (semantic[:,:,2] < 80)
    blue  = (semantic[:,:,0] > 200) & (semantic[:,:,1] < 80) & (semantic[:,:,2] < 80)
    red   = (semantic[:,:,2] > 200) & (semantic[:,:,0] < 80) & (semantic[:,:,1] < 80)
    return {"carga": green, "racks": blue, "descarga": red}


def build_route_masks(route_img: np.ndarray):
    """
    Extrae todas las máscaras del waypoint_map.

    Colores esperados (BGR):
      Verde  (  0,255,  0) → waypoint solo búsqueda
      Azul   (255,  0,  0) → waypoint solo tránsito
      Rojo   (  0,  0,255) → waypoint búsqueda + tránsito
      Morado (~128,  0,128)→ ruta navegable

    Máscara 'navigable' = morado | verde | rojo | azul
    Es la que se usa para el BFS — permite al robot transitar por
    cualquier color sin quedar bloqueado en callejones.
    """
    b = route_img[:,:,0].astype(np.int16)
    g = route_img[:,:,1].astype(np.int16)
    r = route_img[:,:,2].astype(np.int16)

    mask_green  = (g > 180) & (r <  80) & (b <  80)
    mask_blue   = (b > 180) & (g <  80) & (r <  80)
    mask_red    = (r > 180) & (g <  80) & (b <  80)
    mask_purple = (r > 80) & (b > 80) & (g < 80) & (np.abs(r - b) < 80)

    # Todos los píxeles de color son transitables para el BFS
    mask_navigable = mask_purple | mask_green | mask_blue | mask_red

    return {
        'search':     mask_green,                   # solo verdes
        'nav':        mask_blue,                    # solo azules
        'both':       mask_red,                     # rojos (búsqueda+nav)
        'search_all': mask_green | mask_red,        # todos los de búsqueda
        'nav_all':    mask_blue  | mask_red,        # todos los de tránsito
        'purple':     mask_purple,
        'navigable':  mask_navigable,               # BFS usa esta
    }


# ═══════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE WAYPOINTS
# ═══════════════════════════════════════════════════════════════════════

def extract_search_waypoints(zone_mask, route_search_mask, conv,
                              search_green_mask=None,
                              subsample=SEARCH_WP_SUBSAMPLE,
                              min_dist_m=SEARCH_WP_MIN_DIST):
    """
    Devuelve lista de (ox, oy, is_green).
      is_green=True  → verde (se marca como visitado al escanearlo)
      is_green=False → rojo  (nunca se marca visitado, escanea siempre)
    """
    zone_route = route_search_mask & zone_mask
    ys, xs = np.where(zone_route)
    if len(xs) == 0:
        return []

    pixels = list(zip(xs[::subsample], ys[::subsample]))
    if not pixels:
        return []

    candidates = []
    for px, py in pixels:
        ox, oy = conv.pixel_to_odom(px, py)
        is_green = bool(search_green_mask[py, px]) if search_green_mask is not None else True
        candidates.append((ox, oy, is_green))

    waypoints = [candidates[0]]
    for ox, oy, ig in candidates[1:]:
        if math.hypot(ox - waypoints[-1][0], oy - waypoints[-1][1]) >= min_dist_m:
            waypoints.append((ox, oy, ig))
    return waypoints


def extract_nav_waypoints(zone_mask, route_nav_mask, conv,
                           subsample=1, min_dist_m=0.10):
    zone_route = route_nav_mask & zone_mask
    ys, xs = np.where(zone_route)
    if len(xs) == 0:
        return []
    pixels = list(zip(xs[::subsample], ys[::subsample]))
    ordered = [pixels[0]]
    remaining = pixels[1:]
    while remaining:
        last = ordered[-1]
        dists = [math.hypot(p[0]-last[0], p[1]-last[1]) for p in remaining]
        ordered.append(remaining.pop(int(np.argmin(dists))))
    waypoints, last_ox, last_oy = [], None, None
    for px, py in ordered:
        ox, oy = conv.pixel_to_odom(px, py)
        if last_ox is None or math.hypot(ox-last_ox, oy-last_oy) >= min_dist_m:
            waypoints.append((ox, oy))
            last_ox, last_oy = ox, oy
    return waypoints


# ═══════════════════════════════════════════════════════════════════════
#  BFS SOBRE MÁSCARA NAVEGABLE
# ═══════════════════════════════════════════════════════════════════════

def bfs_path_pixels(navigable_mask: np.ndarray,
                    start_px, start_py,
                    goal_px,  goal_py) -> list:
    """
    BFS 8-conectado sobre navigable_mask (morado | verde | rojo | azul).

    start y goal se anclan al píxel navegable más cercano si no caen
    exactamente sobre la máscara.

    Devuelve lista de píxeles [(px, py), ...] o [] si no hay camino.
    """
    H, W = navigable_mask.shape

    def nearest_navigable(px, py):
        ys, xs = np.where(navigable_mask)
        if len(xs) == 0:
            return px, py
        dists = (xs - px)**2 + (ys - py)**2
        idx   = int(np.argmin(dists))
        return int(xs[idx]), int(ys[idx])

    spx, spy = nearest_navigable(start_px, start_py)
    gpx, gpy = nearest_navigable(goal_px,  goal_py)

    if (spx, spy) == (gpx, gpy):
        return [(spx, spy)]

    visited = np.zeros((H, W), dtype=bool)
    parent  = {}
    queue   = deque()
    visited[spy, spx] = True
    queue.append((spx, spy))

    found = False
    while queue:
        cx, cy = queue.popleft()
        if cx == gpx and cy == gpy:
            found = True
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if navigable_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        parent[(nx, ny)] = (cx, cy)
                        queue.append((nx, ny))

    if not found:
        return []

    path, node = [], (gpx, gpy)
    while node != (spx, spy):
        path.append(node)
        node = parent[node]
    path.append((spx, spy))
    path.reverse()
    return path


def rdp_simplify(points: list, epsilon: float) -> list:
    """
    Ramer-Douglas-Peucker: reduce una polilínea de píxeles manteniendo
    solo los puntos que se desvían más de `epsilon` píxeles de la línea
    entre extremos.  Siempre conserva inicio y fin.
    """
    if len(points) < 3:
        return list(points)

    def perp_dist(pt, start, end):
        dx, dy = end[0]-start[0], end[1]-start[1]
        if dx == 0 and dy == 0:
            return math.hypot(pt[0]-start[0], pt[1]-start[1])
        t = ((pt[0]-start[0])*dx + (pt[1]-start[1])*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(pt[0]-(start[0]+t*dx), pt[1]-(start[1]+t*dy))

    # Índice del punto con mayor distancia a la línea start→end
    dmax, imax = 0.0, 0
    for i in range(1, len(points)-1):
        d = perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            dmax, imax = d, i

    if dmax > epsilon:
        left  = rdp_simplify(points[:imax+1], epsilon)
        right = rdp_simplify(points[imax:],   epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


# Tolerancia RDP en píxeles.  Valores entre 1.5–3.0 dan rutas suaves
# con muy pocas poses sin perder esquinas importantes.
RDP_EPSILON = 2.0


def pixels_to_path_msg(pixel_path: list, conv: CoordConverter,
                        frame_id='map') -> Path:
    """
    Convierte lista de píxeles en nav_msgs/Path aplicando RDP para
    reducir el número de poses al mínimo necesario.
    """
    simplified = rdp_simplify(pixel_path, RDP_EPSILON)
    msg = Path()
    msg.header.frame_id = frame_id
    for px, py in simplified:
        ox, oy = conv.pixel_to_odom(px, py)
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = ox
        pose.pose.position.y = oy
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg


# ═══════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ═══════════════════════════════════════════════════════════════════════
#  NODO
# ═══════════════════════════════════════════════════════════════════════

class MissionManagerNode(Node):

    MISSION_MAP = {
        'mission_1': 'carga',
        'mission_2': 'racks',
    }

    def __init__(self):
        super().__init__('mission_manager')

        # ── Cargar mapas ──────────────────────────────────────────────
        semantic_img = cv2.imread(SEMANTIC_MAP_PATH)
        route_img    = cv2.imread(ROUTE_MAP_PATH)
        if semantic_img is None or route_img is None:
            self.get_logger().error('No se pudieron cargar los mapas')
            raise RuntimeError('Mapas no encontrados')

        H, W, _ = semantic_img.shape
        self.conv        = CoordConverter(W, H, MAP_RESOLUTION)
        self.zone_masks  = build_zone_masks(semantic_img)
        self.route_masks = build_route_masks(route_img)

        # Pre-calcular waypoints por zona
        self.search_waypoints = {}
        for zone in ('carga', 'racks'):
            wps = extract_search_waypoints(
                self.zone_masks[zone],
                self.route_masks['search_all'],
                self.conv,
                search_green_mask=self.route_masks['search'],
            )
            self.search_waypoints[zone] = wps
            n_g = sum(1 for *_, ig in wps if ig)
            n_r = len(wps) - n_g
            self.get_logger().info(
                f'Zona {zone}: {len(wps)} waypoints de búsqueda '
                f'({n_g} verdes[visitables], {n_r} rojos[siempre activos])')

        self.nav_waypoints = {}
        for zone in ('carga', 'racks', 'descarga'):
            wps = extract_nav_waypoints(
                self.zone_masks[zone], self.route_masks['nav_all'], self.conv)
            self.nav_waypoints[zone] = wps
            self.get_logger().info(
                f'Zona {zone}: {len(wps)} waypoints de navegación')

        # ── Estado principal ──────────────────────────────────────────
        self.state           = 'IDLE'
        self.current_mission = None
        self.current_zone    = None

        self.robot_x  = 0.0
        self.robot_y  = 0.0
        self.robot_th = 0.0
        self.pose_ok  = False

        self.nav_state      = 'WAIT_PLAN'
        self.forklift_state = 'IDLE'

        self._qr_scan_enabled = False
        self.qr_detected      = False
        self.qr_centered      = False
        self.qr_lost_count    = 0

        self.latest_aruco_msg = None
        self.latest_qr_msg    = None
        self.state_start_time = self.get_clock().now()

        # ── Sub-FSM búsqueda de QR ────────────────────────────────────
        self._search_sub_state   = 'MOVE_TO_WP'
        self._search_wp_idx      = 0
        self._search_wps_ordered = []   # [(ox, oy, is_green), ...]
        # Solo se registran los VERDES visitados en el ciclo actual.
        # Los rojos NUNCA entran aquí.
        self._green_visited      = set()   # índices de verdes ya escaneados
        self._scan_target_theta  = 0.0
        self._scan_sub_timer     = 0.0
        self._scan_sub_dt        = 0.1
        self._move_wp_published  = False

        # ── Publishers ────────────────────────────────────────────────
        self.goal_pub    = self.create_publisher(String, '/goal_input',       10)
        self.plan_pub    = self.create_publisher(Path,   '/plan',             10)
        self.cmd_vel_pub = self.create_publisher(Twist,  '/cmd_vel',          10)
        self.enable_pub  = self.create_publisher(Bool,   '/center_qr/enable', 10)
        self.cancel_pub  = self.create_publisher(Bool,   '/nav/cancel',       10)
        self.status_pub  = self.create_publisher(String, '/mission/status',   10)
        self.img_pub     = self.create_publisher(
            CompressedImage, '/mission_manager_image/compressed', 10)

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(String, '/mission',      self._mission_cb,        10)
        self.create_subscription(String, '/nav/status',   self._nav_status_cb,     10)
        self.create_subscription(String, '/forklift/status', self._forklift_status_cb, 10)
        self.create_subscription(Bool,   '/qr/detected',  self._qr_detected_cb,    10)
        self.create_subscription(Bool,   '/qr/centered',  self._qr_centered_cb,    10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/aruco/pose_centered', self._pose_cb, 10)
        self.create_subscription(
            CompressedImage, '/aruco/image_detected/compressed', self._aruco_cam_cb, 10)
        self.create_subscription(
            CompressedImage, '/qr/image_detected/compressed', self._qr_cam_cb, 10)

        self.timer = self.create_timer(0.1, self._fsm_step)

        self.get_logger().info(
            'mission_manager listo\n'
            '  ros2 topic pub --once /mission std_msgs/String "data: \'mission_1\'"')

    # ─────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────

    def _aruco_cam_cb(self, msg): self.latest_aruco_msg = msg
    def _qr_cam_cb(self, msg):    self.latest_qr_msg    = msg

    def _mission_cb(self, msg: String):
        mission = msg.data.strip().lower()
        if mission not in self.MISSION_MAP:
            self.get_logger().warn(f'Misión desconocida: "{mission}"')
            return
        if self.state not in ('IDLE', 'DONE'):
            self.get_logger().warn(f'Misión recibida en {self.state} — ignorada')
            return
        self.current_mission = mission
        self.current_zone    = self.MISSION_MAP[mission]
        self.get_logger().info(f'Misión: {mission} → zona: {self.current_zone}')
        self._transition('NAVIGATE_TO_ZONE')

    def _nav_status_cb(self, msg: String):
        for part in msg.data.split('|'):
            part = part.strip()
            if part.startswith('state='):
                self.nav_state = part.split('=')[1].strip()
                break

    def _forklift_status_cb(self, msg: String):
        for part in msg.data.split('|'):
            part = part.strip()
            if part.startswith('state='):
                self.forklift_state = part.split('=')[1].strip()
                break

    def _qr_detected_cb(self, msg: Bool):
        raw = msg.data
        self.qr_detected = raw if self._qr_scan_enabled else False
        if self.state == 'CENTER_QR' and not raw:
            self.qr_lost_count += 1
        else:
            self.qr_lost_count = 0

    def _qr_centered_cb(self, msg: Bool): self.qr_centered = msg.data

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        from tf_transformations import euler_from_quaternion
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_th = yaw
        self.pose_ok  = True

    # ─────────────────────────────────────────────────────────────────
    #  FSM PRINCIPAL
    # ─────────────────────────────────────────────────────────────────

    def _fsm_step(self):
        self._publish_status()
        self._publish_current_image()

        if   self.state == 'IDLE':               pass
        elif self.state == 'NAVIGATE_TO_ZONE':   self._step_navigate_to_zone()
        elif self.state == 'SEARCH_QR':          self._step_search_qr()
        elif self.state == 'CENTER_QR':          self._step_center_qr()
        elif self.state == 'LIFT_PALLET':        self._step_lift_pallet()
        elif self.state == 'NAVIGATE_TO_TRUCK':  self._step_navigate_to_truck()
        elif self.state == 'DROP_PALLET':        self._step_drop_pallet()
        elif self.state == 'DONE':               pass

    # ─────────────────────────────────────────────────────────────────
    #  NAVIGATE_TO_ZONE
    # ─────────────────────────────────────────────────────────────────

    def _step_navigate_to_zone(self):
        if self._elapsed() < 0.15:
            if self.pose_ok and self._robot_in_zone(self.current_zone):
                self.get_logger().info(
                    f'Robot ya en {self.current_zone} → SEARCH_QR directo')
                self._transition('SEARCH_QR')
                return
            wps = self.search_waypoints.get(self.current_zone, [])
            if not wps:
                goal_msg = String(); goal_msg.data = self.current_zone
                self.goal_pub.publish(goal_msg)
            else:
                if self.pose_ok:
                    wps = sorted(wps, key=lambda w: math.hypot(
                        w[0]-self.robot_x, w[1]-self.robot_y))
                goal_msg = String()
                goal_msg.data = f'{wps[0][0]:.3f} {wps[0][1]:.3f}'
                self.goal_pub.publish(goal_msg)
                self.get_logger().info(
                    f'Navegando a primer wp: ({wps[0][0]:.2f},{wps[0][1]:.2f})')
            return

        if self.nav_state == 'DONE':
            self.get_logger().info(f'Llegó a {self.current_zone} → SEARCH_QR')
            self._transition('SEARCH_QR')

    # ─────────────────────────────────────────────────────────────────
    #  SEARCH_QR
    # ─────────────────────────────────────────────────────────────────

    def _step_search_qr(self):
        ss = self._search_sub_state

        # QR detectado durante espera → CENTER_QR
        if self.qr_detected and ss in ('SCAN_WAIT_A', 'SCAN_WAIT_B'):
            self.get_logger().info('QR detectado → CENTER_QR')
            self._qr_scan_enabled = False
            self._stop_robot()
            self._cancel_navigation()
            self._transition('CENTER_QR')
            return

        # ── MOVE_TO_WP ───────────────────────────────────────────────
        if ss == 'MOVE_TO_WP':
            if not self._move_wp_published:
                self._publish_path_to_wp(self._search_wp_idx)
                self._move_wp_published = True
                self._qr_scan_enabled   = False
                wp = self._search_wps_ordered[self._search_wp_idx]
                self.get_logger().info(
                    f'MOVE_TO_WP idx={self._search_wp_idx} '
                    f'{"verde" if wp[2] else "rojo"} '
                    f'({wp[0]:.2f},{wp[1]:.2f}) | '
                    f'verdes visitados={len(self._green_visited)}/'
                    f'{sum(1 for *_,ig in self._search_wps_ordered if ig)}')
                return

            if self.nav_state == 'DONE':
                self._move_wp_published = False
                wp = self._search_wps_ordered[self._search_wp_idx]
                if wp[2]:
                    # Verde → escanear y luego marcar visitado en NEXT_WP
                    self._enter_scan_rotate_a()
                else:
                    # Rojo → siempre escanear (nunca se marca visitado)
                    self._enter_scan_rotate_a()
                # Nota: ambos casos entran a escaneo; la diferencia es en NEXT_WP

        # ── SCAN_ROTATE_A ────────────────────────────────────────────
        elif ss == 'SCAN_ROTATE_A':
            if self._do_scan_rotate(self._scan_target_theta):
                self._stop_robot()
                self._search_sub_state = 'SCAN_WAIT_A'
                self._scan_sub_timer   = 0.0
                self._qr_scan_enabled  = True
                self.get_logger().info('SCAN_WAIT_A — esperando QR lado A')

        # ── SCAN_WAIT_A ──────────────────────────────────────────────
        elif ss == 'SCAN_WAIT_A':
            self._scan_sub_timer += self._scan_sub_dt
            if self._scan_sub_timer >= SCAN_WAIT_S:
                self._qr_scan_enabled = False
                if self.current_mission == 'mission_2':
                    self._enter_scan_rotate_b()
                else:
                    self._search_sub_state = 'NEXT_WP'

        # ── SCAN_ROTATE_B (misión 2) ─────────────────────────────────
        elif ss == 'SCAN_ROTATE_B':
            if self._do_scan_rotate(self._scan_target_theta):
                self._stop_robot()
                self._search_sub_state = 'SCAN_WAIT_B'
                self._scan_sub_timer   = 0.0
                self._qr_scan_enabled  = True
                self.get_logger().info('SCAN_WAIT_B — esperando QR lado B')

        # ── SCAN_WAIT_B (misión 2) ───────────────────────────────────
        elif ss == 'SCAN_WAIT_B':
            self._scan_sub_timer += self._scan_sub_dt
            if self._scan_sub_timer >= SCAN_WAIT_S:
                self._qr_scan_enabled  = False
                self._search_sub_state = 'NEXT_WP'

        # ── NEXT_WP ──────────────────────────────────────────────────
        elif ss == 'NEXT_WP':
            wp = self._search_wps_ordered[self._search_wp_idx]
            is_green = wp[2]

            # Solo los verdes se marcan como visitados
            if is_green:
                self._green_visited.add(self._search_wp_idx)

            # Excluir el wp actual para que un rojo no se reelija a sí mismo
            current_idx = self._search_wp_idx
            self._move_wp_published = False
            self._search_wp_idx     = self._pick_next_wp(exclude_idx=current_idx)
            self._search_sub_state  = 'MOVE_TO_WP'

    # ─────────────────────────────────────────────────────────────────
    #  LÓGICA DE SELECCIÓN DEL SIGUIENTE WAYPOINT
    # ─────────────────────────────────────────────────────────────────

    def _pick_next_wp(self, exclude_idx: int = -1) -> int:
        """
        Selecciona el índice del siguiente waypoint destino minimizando
        el COSTE REAL DE RUTA (longitud del path BFS en píxeles), no la
        distancia euclidiana.  Esto evita elegir waypoints "cercanos en
        línea recta" pero costosos de alcanzar por la topología del mapa.

        exclude_idx: índice a excluir de los candidatos (el waypoint
                     actual, para que un rojo no se reelija a sí mismo).

        Candidatos elegibles:
          • Todos los ROJOS  (nunca se marcan visitados) excepto exclude_idx
          • VERDES no visitados en este ciclo

        Si todos los verdes ya fueron visitados → reiniciar ciclo de verdes.
        """
        wps = self._search_wps_ordered

        green_indices = {i for i, (*_, ig) in enumerate(wps) if ig}
        red_indices   = {i for i, (*_, ig) in enumerate(wps) if not ig}

        unvisited_green = green_indices - self._green_visited
        if not unvisited_green:
            self.get_logger().info(
                'Todos los verdes visitados — reiniciando ciclo de búsqueda')
            self._green_visited.clear()
            unvisited_green = green_indices

        # Candidatos = rojos (sin el actual) + verdes no visitados
        candidates = list((red_indices - {exclude_idx}) | unvisited_green)
        if not candidates:
            # Fallback: todos excepto el actual
            candidates = [i for i in range(len(wps)) if i != exclude_idx]
        if not candidates:
            candidates = list(range(len(wps)))

        # Calcular coste real (longitud BFS en píxeles) para cada candidato
        start_px, start_py = self.conv.odom_to_pixel(self.robot_x, self.robot_y)
        navigable = self.route_masks['navigable']

        costs = {}
        for i in candidates:
            goal_ox, goal_oy = wps[i][0], wps[i][1]
            goal_px, goal_py = self.conv.odom_to_pixel(goal_ox, goal_oy)
            path = bfs_path_pixels(navigable, start_px, start_py, goal_px, goal_py)
            costs[i] = len(path) if path else float('inf')

        green_candidates = [i for i in candidates if wps[i][2]]
        red_candidates   = [i for i in candidates if not wps[i][2]]

        # Prioridad verdes: solo elegir rojo si supera al mejor verde por
        # mas de RED_PENALTY_PX pixeles de ruta — evita el bouncing al rojo
        # central entre dos verdes adyacentes.
        RED_PENALTY_PX = 40

        best_green_cost = min((costs[i] for i in green_candidates), default=float('inf'))
        best_red_cost   = min((costs[i] for i in red_candidates),   default=float('inf'))

        if green_candidates and best_green_cost <= best_red_cost + RED_PENALTY_PX:
            pool = green_candidates   # forzar verde
        else:
            pool = candidates         # sin verdes utiles, abrir a todos

        best_idx  = min(pool, key=lambda i: costs[i])
        best_cost = costs[best_idx]

        ox, oy, ig = wps[best_idx]
        self.get_logger().info(
            f'_pick_next_wp → idx={best_idx} {"verde" if ig else "rojo"} '
            f'({ox:.2f},{oy:.2f}) coste={best_cost}px | '
            f'verdes visitados={len(self._green_visited)}/{len(green_indices)} | '
            f'rojos={len(red_indices)}')
        return best_idx

    # ─────────────────────────────────────────────────────────────────
    #  PUBLICAR PATH VIA RUTA NAVEGABLE (BFS)
    # ─────────────────────────────────────────────────────────────────

    def _publish_path_to_wp(self, wp_idx: int):
        """
        BFS desde posición actual del robot hasta wp_idx usando la
        máscara 'navigable' (morado | verde | rojo | azul).
        Publica el resultado en /plan como nav_msgs/Path.
        Fallback a línea recta si BFS no encuentra camino.
        """
        wp = self._search_wps_ordered[wp_idx]
        goal_ox, goal_oy = wp[0], wp[1]

        start_px, start_py = self.conv.odom_to_pixel(self.robot_x, self.robot_y)
        goal_px,  goal_py  = self.conv.odom_to_pixel(goal_ox, goal_oy)

        pixel_path = bfs_path_pixels(
            self.route_masks['navigable'],
            start_px, start_py,
            goal_px,  goal_py)

        if not pixel_path:
            self.get_logger().warn(
                f'BFS sin camino navegable hacia wp {wp_idx} — línea recta')
            path_msg = Path()
            path_msg.header.frame_id = 'map'
            path_msg.header.stamp    = self.get_clock().now().to_msg()
            for ox, oy in [(self.robot_x, self.robot_y), (goal_ox, goal_oy)]:
                p = PoseStamped()
                p.header = path_msg.header
                p.pose.position.x = ox
                p.pose.position.y = oy
                p.pose.orientation.w = 1.0
                path_msg.poses.append(p)
            self.plan_pub.publish(path_msg)
            return

        path_msg = pixels_to_path_msg(
            pixel_path, self.conv, frame_id='map')
        path_msg.header.stamp = self.get_clock().now().to_msg()
        self.plan_pub.publish(path_msg)

        self.get_logger().info(
            f'Path publicado: {len(pixel_path)} px → '
            f'{len(path_msg.poses)} poses (RDP ε={RDP_EPSILON}px) '
            f'hacia wp {wp_idx} ({goal_ox:.2f},{goal_oy:.2f})')

    # ─────────────────────────────────────────────────────────────────
    #  HELPERS SUB-FSM ESCANEO
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_cardinal(theta: float) -> float:
        """Devuelve el múltiplo de 90° más cercano a theta (en radianes)."""
        cardinals = [0.0, math.pi/2, math.pi, -math.pi/2]
        return min(cardinals, key=lambda c: abs(wrap_angle(theta - c)))

    def _enter_scan_rotate_a(self):
        if self.current_mission == 'mission_1':
            self._scan_target_theta = MISSION1_SCAN_THETA
        else:
            # Misión 2: cardinal más cercano a la orientación actual
            self._scan_target_theta = wrap_angle(self._nearest_cardinal(self.robot_th) + math.pi / 2)
        self._search_sub_state = 'SCAN_ROTATE_A'
        self.get_logger().info(
            f'SCAN_ROTATE_A → θ={math.degrees(self._scan_target_theta):.1f}°')

    def _enter_scan_rotate_b(self):
        # Cardinal opuesto al lado A (siempre +180°, también cardinal)
        self._scan_target_theta = wrap_angle(self._scan_target_theta + math.pi)
        self._search_sub_state = 'SCAN_ROTATE_B'
        self.get_logger().info(
            f'SCAN_ROTATE_B → θ={math.degrees(self._scan_target_theta):.1f}°')

    def _do_scan_rotate(self, target_theta: float) -> bool:
        error = wrap_angle(target_theta - self.robot_th)
        if abs(error) < SCAN_ANGLE_TOL:
            return True
        omega = clamp(SCAN_KP_W * error, -SCAN_OMEGA, SCAN_OMEGA)
        cmd = Twist(); cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)
        return False

    def _stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    # ─────────────────────────────────────────────────────────────────
    #  CENTER_QR / LIFT / TRUCK / DROP
    # ─────────────────────────────────────────────────────────────────

    def _step_center_qr(self):
        if self._elapsed() < 0.15:
            enable_msg = Bool(); enable_msg.data = True
            self.enable_pub.publish(enable_msg)
            self.get_logger().info('center_qr habilitado')
            return
        if self.qr_lost_count >= QR_LOST_COUNT_THRESHOLD:
            self.get_logger().warn(
                f'QR perdido ({self.qr_lost_count} frames) → SEARCH_QR')
            self._disable_center_qr()
            self._transition('SEARCH_QR')
            return
        if self.qr_centered:
            self.get_logger().info('QR centrado → LIFT_PALLET')
            self._disable_center_qr()
            self._transition('LIFT_PALLET')

    def _step_lift_pallet(self):
        if self._elapsed() < 0.15:
            self.get_logger().info('Levantando pallet…'); return
        if self.forklift_state == 'DONE':
            self.get_logger().info('Pallet levantado → NAVIGATE_TO_TRUCK')
            self._transition('NAVIGATE_TO_TRUCK')

    def _step_navigate_to_truck(self):
        if self._elapsed() < 0.15:
            goal_msg = String(); goal_msg.data = 'descarga'
            self.goal_pub.publish(goal_msg)
            self.get_logger().info('Navegando a descarga'); return
        if self.nav_state == 'DONE':
            self.get_logger().info('Llegó a descarga → DROP_PALLET')
            self._transition('DROP_PALLET')

    def _step_drop_pallet(self):
        if self._elapsed() < 0.15:
            self.get_logger().info('Bajando pallet…'); return
        if self.forklift_state == 'DONE':
            self.get_logger().info('Pallet entregado → DONE')
            self._transition('DONE')

    # ─────────────────────────────────────────────────────────────────
    #  TRANSICIÓN
    # ─────────────────────────────────────────────────────────────────

    def _transition(self, new_state: str):
        self.get_logger().info(f'[FSM] {self.state} → {new_state}')
        self.state            = new_state
        self.state_start_time = self.get_clock().now()
        self.nav_state        = 'WAIT_PLAN'
        self.qr_lost_count    = 0

        if new_state in ('LIFT_PALLET', 'DROP_PALLET'):
            self.forklift_state = 'IDLE'

        if new_state == 'SEARCH_QR':
            wps = extract_search_waypoints(
                self.zone_masks[self.current_zone],
                self.route_masks['search_all'],
                self.conv,
                search_green_mask=self.route_masks['search'],
            )
            if not wps:
                self.get_logger().error(
                    f'Sin waypoints de búsqueda para {self.current_zone}')
                return

            # El más cercano al robot va primero; el resto mediante _pick_next_wp
            if self.pose_ok:
                first = min(range(len(wps)),
                            key=lambda i: math.hypot(
                                wps[i][0]-self.robot_x, wps[i][1]-self.robot_y))
                ordered = [wps[first]] + [wps[i] for i in range(len(wps)) if i != first]
            else:
                ordered = list(wps)

            self._search_wps_ordered = ordered
            self._search_wp_idx      = 0
            self._green_visited      = set()   # ningún verde visitado al inicio
            self._search_sub_state   = 'MOVE_TO_WP'
            self._move_wp_published  = False
            self._qr_scan_enabled    = False
            self._scan_sub_timer     = 0.0

            n_g = sum(1 for *_, ig in ordered if ig)
            n_r = len(ordered) - n_g
            self.get_logger().info(
                f'SEARCH_QR: {len(ordered)} wps '
                f'({n_g} verdes[ciclo], {n_r} rojos[siempre]) | '
                f'BFS sobre morado+verde+rojo+azul')

    # ─────────────────────────────────────────────────────────────────
    #  UTILIDADES
    # ─────────────────────────────────────────────────────────────────

    def _elapsed(self):
        return (self.get_clock().now() - self.state_start_time).nanoseconds * 1e-9

    def _robot_in_zone(self, zone):
        if not self.pose_ok: return False
        px, py = self.conv.odom_to_pixel(self.robot_x, self.robot_y)
        H, W   = self.zone_masks[zone].shape
        if not (0 <= px < W and 0 <= py < H): return False
        return bool(self.zone_masks[zone][py, px])

    def _cancel_navigation(self):
        msg = Bool(); msg.data = True; self.cancel_pub.publish(msg)

    def _disable_center_qr(self):
        msg = Bool(); msg.data = False; self.enable_pub.publish(msg)

    def _publish_current_image(self):
        if self.state in ('CENTER_QR','SEARCH_QR'):
            if self.latest_qr_msg:   self.img_pub.publish(self.latest_qr_msg)
        else:
            if self.latest_aruco_msg: self.img_pub.publish(self.latest_aruco_msg)

    def _publish_status(self):
        sub = self._search_sub_state if self.state == 'SEARCH_QR' else '-'
        n_g = sum(1 for *_, ig in self._search_wps_ordered if ig)
        msg = String()
        msg.data = (
            f'state={self.state} | sub={sub} | '
            f'mission={self.current_mission} | zone={self.current_zone} | '
            f'wp={self._search_wp_idx}/{len(self._search_wps_ordered)} '
            f'green_visited={len(self._green_visited)}/{n_g} | '
            f'qr={self.qr_detected} | qr_scan={self._qr_scan_enabled} | '
            f'nav={self.nav_state} | '
            f'x={self.robot_x:.2f} y={self.robot_y:.2f} '
            f'th={math.degrees(self.robot_th):.1f}')
        self.status_pub.publish(msg)


# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_vel_pub.publish(Twist())
            disable = Bool(); disable.data = False; node.enable_pub.publish(disable)
            cancel  = Bool(); cancel.data  = True;  node.cancel_pub.publish(cancel)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()