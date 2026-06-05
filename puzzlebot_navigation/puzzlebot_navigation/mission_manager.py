#!/usr/bin/env python3
"""
mission_manager.py
═══════════════════════════════════════════════════════════════════════
Nodo maestro — Orquestador de misiones del Puzzlebot

NUEVO ESQUEMA DE WAYPOINTS EN route_map
────────────────────────────────────────
  Verde  (0,255,0 BGR)  → solo búsqueda de QR
  Azul   (255,0,0 BGR)  → solo navegación de tránsito
  Rojo   (0,0,255 BGR)  → búsqueda Y navegación

  • La navegación entre zonas usa semantic_planner (goal_input);
    los waypoints azules y rojos del route_map son la referencia
    visual de por dónde se espera que el robot transite — el planner
    ya los conoce a través de su route_map propio.

  • Los waypoints de búsqueda (SEARCH_QR) son los píxeles
    verde+rojo intersectados con la zona de la misión.

COMPORTAMIENTO EN BÚSQUEDA DE QR
──────────────────────────────────
  El robot recorre los waypoints de búsqueda uno a uno.
  Al llegar a cada waypoint verde o rojo ejecuta la rutina de escaneo:

  Misión 1 (carga / conveyors):
    1. Girar a θ_abs = 180° en el frame del mapa.
    2. Esperar SCAN_WAIT_S segundos mirando hacia los conveyors.
    3. Si detecta QR → CENTER_QR; si no → siguiente waypoint.

  Misión 2 (racks):
    1. Girar θ_rel = +90° respecto a la orientación al llegar.
    2. Esperar SCAN_WAIT_S segundos.
    3. Girar θ_rel = -180° (mira al lado opuesto).
    4. Esperar SCAN_WAIT_S segundos.
    5. Si detecta QR en cualquier pausa → CENTER_QR; si no → siguiente wp.

  La detección de QR SOLO se considera válida mientras el robot
  está parado en un waypoint de búsqueda (estado SCAN_* interno).

Máquina de estados principal:
  IDLE → NAVIGATE_TO_ZONE → SEARCH_QR → CENTER_QR
       → LIFT_PALLET → NAVIGATE_TO_TRUCK → DROP_PALLET → DONE

Sub-FSM dentro de SEARCH_QR:
  MOVE_TO_WP → SCAN_ROTATE → SCAN_WAIT_A → [misión 2: SCAN_ROTATE_B → SCAN_WAIT_B] → NEXT_WP

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
  /goal_input           — std_msgs/String
  /plan                 — nav_msgs/Path
  /center_qr/enable     — std_msgs/Bool
  /nav/cancel           — std_msgs/Bool
  /mission/status       — std_msgs/String
"""

import rclpy
from rclpy.node import Node
from rclpy import qos

from std_msgs.msg import String, Bool
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np
import math

# ═══════════════════════════════════════════════════════════════════════
#  RUTAS DE MAPAS Y RESOLUCIÓN
# ═══════════════════════════════════════════════════════════════════════

SEMANTIC_MAP_PATH = "/home/juanjo/semantic_map.png"
ROUTE_MAP_PATH    = "/home/juanjo/waypoint_map.png"
MAP_RESOLUTION    = 0.05   # m/pixel — debe coincidir con semantic_planner

# ═══════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════

# Submuestreo de waypoints de búsqueda (cada N píxeles de ruta)
SEARCH_WP_SUBSAMPLE = 8

# Distancia mínima entre waypoints de búsqueda consecutivos (m)
SEARCH_WP_MIN_DIST = 0.20

# Radio para considerar que el robot ya está en la zona objetivo (m)
ZONE_ARRIVAL_RADIUS = 0.07

# Tolerancia de ángulo para considerar que el giro de escaneo terminó (rad)
SCAN_ANGLE_TOL = math.radians(3)

# Segundos de espera mirando en cada dirección durante el escaneo
SCAN_WAIT_S = 3.0

# Velocidad angular de giro durante escaneo (rad/s)
SCAN_OMEGA = 0.20

# Ganancias del control de ángulo durante el escaneo
SCAN_KP_W = 0.80
SCAN_KV_W = 0.05

# Cuántos mensajes False consecutivos de /qr/detected tras CENTER_QR
# indican que el QR se perdió (fallo de alineación)
QR_LOST_COUNT_THRESHOLD = 30

# Orientación absoluta (rad) a la que debe girar el robot en Misión 1
# para mirar hacia los conveyors (180° = mirando en dirección -X del mapa)
MISSION1_SCAN_THETA = math.pi   # 180°

# Giro relativo para Misión 2 (rad).  +90° = izquierda del robot
MISSION2_SCAN_DELTA_A = math.pi / 2.0    # +90°
MISSION2_SCAN_DELTA_B = -math.pi         # −180° (al lado opuesto)


# ═══════════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE COORDENADAS
# ═══════════════════════════════════════════════════════════════════════

class CoordConverter:
    def __init__(self, W: int, H: int, resolution: float):
        self.W         = W
        self.H         = H
        self.res       = resolution
        self.ox_offset = (W / 2) * resolution
        self.oy_offset = (H / 2) * resolution

    def pixel_to_world(self, px, py):
        return px * self.res, (self.H - 1 - py) * self.res

    def world_to_pixel(self, wx, wy):
        px = int(np.clip(int(wx / self.res),                   0, self.W - 1))
        py = int(np.clip(int((self.H - 1) - wy / self.res),    0, self.H - 1))
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
    """Extrae máscaras de zona desde el mapa semántico."""
    green = (semantic[:,:,1] > 200) & (semantic[:,:,0] < 80) & (semantic[:,:,2] < 80)
    blue  = (semantic[:,:,0] > 200) & (semantic[:,:,1] < 80) & (semantic[:,:,2] < 80)
    red   = (semantic[:,:,2] > 200) & (semantic[:,:,0] < 80) & (semantic[:,:,1] < 80)
    return {"carga": green, "racks": blue, "descarga": red}


def build_route_masks(route_img: np.ndarray):
    """
    Extrae tres máscaras del route_map según color de waypoint:
      verde (0,255,0 BGR) → solo búsqueda de QR
      azul  (255,0,0 BGR) → solo navegación
      rojo  (0,0,255 BGR) → búsqueda + navegación

    Devuelve dict: {'search': mask, 'nav': mask, 'both': mask}
    y máscaras combinadas:
      'search_all' = search | both   (todos los puntos de búsqueda)
      'nav_all'    = nav    | both   (todos los puntos de tránsito)
    """
    b = route_img[:,:,0].astype(np.int16)
    g = route_img[:,:,1].astype(np.int16)
    r = route_img[:,:,2].astype(np.int16)

    # Verde puro: G alto, R y B bajos
    mask_green = (g > 180) & (r < 80) & (b < 80)
    # Azul puro: B alto, G y R bajos
    mask_blue  = (b > 180) & (g < 80) & (r < 80)
    # Rojo puro: R alto, G y B bajos
    mask_red   = (r > 180) & (g < 80) & (b < 80)

    return {
        'search':     mask_green,
        'nav':        mask_blue,
        'both':       mask_red,
        'search_all': mask_green | mask_red,
        'nav_all':    mask_blue  | mask_red,
    }


# ═══════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE WAYPOINTS
# ═══════════════════════════════════════════════════════════════════════

def extract_search_waypoints(zone_mask: np.ndarray,
                              route_search_mask: np.ndarray,
                              conv: CoordConverter,
                              subsample: int = SEARCH_WP_SUBSAMPLE,
                              min_dist_m: float = SEARCH_WP_MIN_DIST):
    """
    Waypoints de búsqueda de QR para una zona:
    intersección de (verde | rojo) del route_map con la máscara de zona.
    """
    zone_route = route_search_mask & zone_mask
    ys, xs = np.where(zone_route)

    if len(xs) == 0:
        return []

    pixels = list(zip(xs[::subsample], ys[::subsample]))
    if not pixels:
        return []

    # Ordenar por nearest-neighbor desde el primer punto
    ordered = [pixels[0]]
    remaining = pixels[1:]
    while remaining:
        last = ordered[-1]
        dists = [math.hypot(p[0]-last[0], p[1]-last[1]) for p in remaining]
        ordered.append(remaining.pop(int(np.argmin(dists))))

    # Convertir a coordenadas odom y filtrar por distancia mínima
    waypoints = []
    last_ox, last_oy = None, None
    for px, py in ordered:
        ox, oy = conv.pixel_to_odom(px, py)
        if last_ox is None or math.hypot(ox-last_ox, oy-last_oy) >= min_dist_m:
            waypoints.append((ox, oy))
            last_ox, last_oy = ox, oy

    return waypoints


def reorder_from_nearest(waypoints: list, robot_x: float, robot_y: float):
    """Reordena la lista empezando desde el waypoint más cercano al robot."""
    if not waypoints:
        return waypoints
    dists = [math.hypot(wp[0]-robot_x, wp[1]-robot_y) for wp in waypoints]
    idx = int(np.argmin(dists))
    return waypoints[idx:] + waypoints[:idx]


# ═══════════════════════════════════════════════════════════════════════
#  UTILIDADES ANGULARES
# ═══════════════════════════════════════════════════════════════════════

def wrap_angle(a: float) -> float:
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
        self.conv = CoordConverter(W, H, MAP_RESOLUTION)

        self.zone_masks  = build_zone_masks(semantic_img)
        self.route_masks = build_route_masks(route_img)

        # Pre-calcular waypoints de búsqueda (verde+rojo) por zona
        self.search_waypoints = {}
        for zone in ('carga', 'racks'):
            wps = extract_search_waypoints(
                self.zone_masks[zone],
                self.route_masks['search_all'],
                self.conv
            )
            self.search_waypoints[zone] = wps
            self.get_logger().info(
                f'Zona {zone}: {len(wps)} waypoints de búsqueda (verde+rojo)')

        # ── Estado principal ──────────────────────────────────────────
        self.state           = 'IDLE'
        self.current_mission = None
        self.current_zone    = None

        self.robot_x  = 0.0
        self.robot_y  = 0.0
        self.robot_th = 0.0
        self.w_robot  = 0.0    # velocidad angular del robot (de odom)
        self.pose_ok  = False

        self.nav_state      = 'WAIT_PLAN'
        self.forklift_state = 'IDLE'

        # QR — solo se considera válido cuando el nodo lo habilita
        self._qr_scan_enabled = False
        self.qr_detected      = False
        self.qr_centered      = False
        self.qr_lost_count    = 0

        self.latest_aruco_msg = None
        self.latest_qr_msg    = None

        self.state_start_time = self.get_clock().now()

        # ── Sub-FSM búsqueda de QR ────────────────────────────────────
        # Estados: MOVE_TO_WP | SCAN_ROTATE_A | SCAN_WAIT_A |
        #          SCAN_ROTATE_B | SCAN_WAIT_B | NEXT_WP
        self._search_sub_state  = 'MOVE_TO_WP'
        self._search_wp_idx     = 0          # índice en self.search_waypoints[zone]
        self._search_wps_ordered = []        # lista reordenada al iniciar búsqueda
        self._scan_target_theta = 0.0        # ángulo objetivo del giro actual
        self._scan_sub_timer    = 0.0        # segundos transcurridos en SCAN_WAIT_*
        self._scan_sub_dt       = 0.1        # dt del timer principal (10 Hz)

        # ── Publishers ────────────────────────────────────────────────
        self.goal_pub    = self.create_publisher(String, '/goal_input',       10)
        self.plan_pub    = self.create_publisher(Path,   '/plan',             10)
        self.cmd_vel_pub = self.create_publisher(
            __import__('geometry_msgs.msg', fromlist=['Twist']).Twist,
            '/cmd_vel', 10)
        self.enable_pub  = self.create_publisher(Bool,   '/center_qr/enable', 10)
        self.cancel_pub  = self.create_publisher(Bool,   '/nav/cancel',       10)
        self.status_pub  = self.create_publisher(String, '/mission/status',   10)
        self.img_pub     = self.create_publisher(
            CompressedImage, '/mission_manager_image/compressed', 10)

        # ── Subscribers ───────────────────────────────────────────────
        self.mission_sub = self.create_subscription(
            String, '/mission', self._mission_cb, 10)
        self.status_sub = self.create_subscription(
            String, '/nav/status', self._nav_status_cb, 10)
        self.forklift_sub = self.create_subscription(
            String, '/forklift/status', self._forklift_status_cb, 10)
        self.qr_sub = self.create_subscription(
            Bool, '/qr/detected', self._qr_detected_cb, 10)
        self.qr_centered_sub = self.create_subscription(
            Bool, '/qr/centered', self._qr_centered_cb, 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/aruco/pose_centered', self._pose_cb, 10)
        self.aruco_cam = self.create_subscription(
            CompressedImage, '/aruco/image_detected/compressed',
            self._aruco_cam_cb, 10)
        self.qr_cam = self.create_subscription(
            CompressedImage, '/qr/image_detected/compressed',
            self._qr_cam_cb, 10)

        # ── Timer principal 10 Hz ─────────────────────────────────────
        self.timer = self.create_timer(0.1, self._fsm_step)

        self.get_logger().info(
            'mission_manager listo\n'
            '  ros2 topic pub --once /mission std_msgs/String "data: \'mission_1\'"')

    # ─────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────

    def _aruco_cam_cb(self, msg):
        self.latest_aruco_msg = msg

    def _qr_cam_cb(self, msg):
        self.latest_qr_msg = msg

    def _mission_cb(self, msg: String):
        mission = msg.data.strip().lower()
        if mission not in self.MISSION_MAP:
            self.get_logger().warn(f'Misión desconocida: "{mission}"')
            return
        if self.state not in ('IDLE', 'DONE'):
            self.get_logger().warn(
                f'Misión recibida pero el robot está en {self.state} — ignorada')
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
        # Solo propagar la detección si el escaneo está habilitado
        if self._qr_scan_enabled:
            self.qr_detected = raw
        else:
            self.qr_detected = False

        if self.state == 'CENTER_QR' and not raw:
            self.qr_lost_count += 1
        else:
            self.qr_lost_count = 0

    def _qr_centered_cb(self, msg: Bool):
        self.qr_centered = msg.data

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
        elapsed = self._elapsed()

        if elapsed < 0.15:
            if self.pose_ok and self._robot_in_zone(self.current_zone):
                self.get_logger().info(
                    f'Robot ya está en {self.current_zone} → SEARCH_QR directo')
                self._transition('SEARCH_QR')
                return
            goal_msg = String()
            goal_msg.data = self.current_zone
            self.goal_pub.publish(goal_msg)
            self.get_logger().info(f'Navegando a zona: {self.current_zone}')
            return

        if self.nav_state == 'DONE':
            self.get_logger().info(f'Llegó a {self.current_zone} → SEARCH_QR')
            self._transition('SEARCH_QR')

    # ─────────────────────────────────────────────────────────────────
    #  SEARCH_QR  (con sub-FSM)
    # ─────────────────────────────────────────────────────────────────

    def _step_search_qr(self):
        """
        Sub-FSM de búsqueda:
          MOVE_TO_WP    : publica el waypoint actual al waypoint_controller y espera DONE
          SCAN_ROTATE_A : gira al ángulo objetivo A (abs o rel según misión)
          SCAN_WAIT_A   : espera SCAN_WAIT_S con QR habilitado
          SCAN_ROTATE_B : (misión 2) gira al ángulo objetivo B
          SCAN_WAIT_B   : (misión 2) espera SCAN_WAIT_S
          NEXT_WP       : avanza al siguiente waypoint o reinicia el loop
        """
        ss = self._search_sub_state

        # ── Verificar QR detectado en cualquier sub-estado de espera ──
        if self.qr_detected and ss in ('SCAN_WAIT_A', 'SCAN_WAIT_B'):
            self.get_logger().info('QR detectado durante escaneo → CENTER_QR')
            self._qr_scan_enabled = False
            self._stop_robot()
            self._cancel_navigation()
            self._transition('CENTER_QR')
            return

        # ── MOVE_TO_WP ────────────────────────────────────────────────
        if ss == 'MOVE_TO_WP':
            # Primera vez en este sub-estado: publicar el waypoint
            if not hasattr(self, '_move_wp_published') or not self._move_wp_published:
                self._publish_single_wp(
                    self._search_wps_ordered[self._search_wp_idx])
                self._move_wp_published = True
                self._qr_scan_enabled   = False   # desactivar QR durante movimiento
                self.get_logger().info(
                    f'SEARCH_QR: yendo a wp {self._search_wp_idx+1}/'
                    f'{len(self._search_wps_ordered)}  '
                    f'{self._search_wps_ordered[self._search_wp_idx]}')
                return

            # Esperar que waypoint_controller llegue al waypoint
            if self.nav_state == 'DONE':
                self._move_wp_published = False
                self._enter_scan_rotate_a()

        # ── SCAN_ROTATE_A ─────────────────────────────────────────────
        elif ss == 'SCAN_ROTATE_A':
            reached = self._do_scan_rotate(self._scan_target_theta)
            if reached:
                self._stop_robot()
                self._search_sub_state = 'SCAN_WAIT_A'
                self._scan_sub_timer   = 0.0
                self._qr_scan_enabled  = True   # ← habilitar detección
                self.get_logger().info('SCAN_WAIT_A — esperando QR lado A')

        # ── SCAN_WAIT_A ───────────────────────────────────────────────
        elif ss == 'SCAN_WAIT_A':
            self._scan_sub_timer += self._scan_sub_dt
            if self._scan_sub_timer >= SCAN_WAIT_S:
                self._qr_scan_enabled = False
                if self.current_mission == 'mission_2':
                    # Misión 2: girar al lado opuesto
                    self._enter_scan_rotate_b()
                else:
                    # Misión 1: un solo lado de escaneo, ir al siguiente wp
                    self._search_sub_state = 'NEXT_WP'

        # ── SCAN_ROTATE_B (misión 2) ──────────────────────────────────
        elif ss == 'SCAN_ROTATE_B':
            reached = self._do_scan_rotate(self._scan_target_theta)
            if reached:
                self._stop_robot()
                self._search_sub_state = 'SCAN_WAIT_B'
                self._scan_sub_timer   = 0.0
                self._qr_scan_enabled  = True
                self.get_logger().info('SCAN_WAIT_B — esperando QR lado B')

        # ── SCAN_WAIT_B (misión 2) ────────────────────────────────────
        elif ss == 'SCAN_WAIT_B':
            self._scan_sub_timer += self._scan_sub_dt
            if self._scan_sub_timer >= SCAN_WAIT_S:
                self._qr_scan_enabled  = False
                self._search_sub_state = 'NEXT_WP'

        # ── NEXT_WP ───────────────────────────────────────────────────
        elif ss == 'NEXT_WP':
            self._search_wp_idx += 1
            if self._search_wp_idx >= len(self._search_wps_ordered):
                # Loop: reiniciar desde el más cercano
                self.get_logger().info(
                    'Recorrido de búsqueda completo sin QR — reiniciando loop')
                self._search_wps_ordered = reorder_from_nearest(
                    self._search_wps_ordered, self.robot_x, self.robot_y)
                self._search_wp_idx = 0

            self._move_wp_published = False
            self._search_sub_state  = 'MOVE_TO_WP'

    # ── Helpers sub-FSM ───────────────────────────────────────────────

    def _enter_scan_rotate_a(self):
        """Calcula el ángulo objetivo A y entra en SCAN_ROTATE_A."""
        if self.current_mission == 'mission_1':
            # Misión 1: orientación absoluta en mapa = 180°
            self._scan_target_theta = MISSION1_SCAN_THETA
        else:
            # Misión 2: +90° relativo a la orientación actual
            self._scan_target_theta = wrap_angle(
                self.robot_th + MISSION2_SCAN_DELTA_A)

        self._search_sub_state = 'SCAN_ROTATE_A'
        self._cancel_navigation()   # asegurar que el controller no interfiere
        self.get_logger().info(
            f'SCAN_ROTATE_A → θ_target={math.degrees(self._scan_target_theta):.1f}°')

    def _enter_scan_rotate_b(self):
        """Calcula el ángulo objetivo B (misión 2) y entra en SCAN_ROTATE_B."""
        self._scan_target_theta = wrap_angle(
            self._scan_target_theta + MISSION2_SCAN_DELTA_B)
        self._search_sub_state = 'SCAN_ROTATE_B'
        self.get_logger().info(
            f'SCAN_ROTATE_B → θ_target={math.degrees(self._scan_target_theta):.1f}°')

    def _do_scan_rotate(self, target_theta: float) -> bool:
        """
        Publica cmd_vel para girar hacia target_theta.
        Devuelve True cuando se llega a la tolerancia.
        Usa control P sobre el error angular con amortiguación.
        """
        from geometry_msgs.msg import Twist
        error = wrap_angle(target_theta - self.robot_th)

        if abs(error) < SCAN_ANGLE_TOL:
            return True

        omega = clamp(
            SCAN_KP_W * error - SCAN_KV_W * self.w_robot,
            -SCAN_OMEGA, SCAN_OMEGA
        )
        cmd = Twist()
        cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)
        return False

    def _stop_robot(self):
        from geometry_msgs.msg import Twist
        self.cmd_vel_pub.publish(Twist())

    def _publish_single_wp(self, wp):
        """Publica un único waypoint como nav_msgs/Path para el waypoint_controller."""
        ox, oy = wp
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp    = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = ox
        pose.pose.position.y = oy
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
        self.plan_pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────
    #  CENTER_QR
    # ─────────────────────────────────────────────────────────────────

    def _step_center_qr(self):
        elapsed = self._elapsed()

        if elapsed < 0.15:
            enable_msg = Bool()
            enable_msg.data = True
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

    # ─────────────────────────────────────────────────────────────────
    #  LIFT_PALLET
    # ─────────────────────────────────────────────────────────────────

    def _step_lift_pallet(self):
        if self._elapsed() < 0.15:
            self.get_logger().info('Levantando pallet…')
            return
        if self.forklift_state == 'DONE':
            self.get_logger().info('Pallet levantado → NAVIGATE_TO_TRUCK')
            self._transition('NAVIGATE_TO_TRUCK')

    # ─────────────────────────────────────────────────────────────────
    #  NAVIGATE_TO_TRUCK
    # ─────────────────────────────────────────────────────────────────

    def _step_navigate_to_truck(self):
        if self._elapsed() < 0.15:
            goal_msg = String()
            goal_msg.data = 'descarga'
            self.goal_pub.publish(goal_msg)
            self.get_logger().info('Navegando a descarga')
            return
        if self.nav_state == 'DONE':
            self.get_logger().info('Llegó a descarga → DROP_PALLET')
            self._transition('DROP_PALLET')

    # ─────────────────────────────────────────────────────────────────
    #  DROP_PALLET
    # ─────────────────────────────────────────────────────────────────

    def _step_drop_pallet(self):
        if self._elapsed() < 0.15:
            self.get_logger().info('Bajando pallet…')
            return
        if self.forklift_state == 'DONE':
            self.get_logger().info('Pallet entregado → DONE')
            self._transition('DONE')

    # ─────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _transition(self, new_state: str):
        self.get_logger().info(f'[FSM] {self.state} → {new_state}')
        self.state            = new_state
        self.state_start_time = self.get_clock().now()
        self.nav_state        = 'WAIT_PLAN'

        if new_state in ('LIFT_PALLET', 'DROP_PALLET'):
            self.forklift_state = 'IDLE'

        self.qr_lost_count = 0

        # Inicializar sub-FSM al entrar en SEARCH_QR
        if new_state == 'SEARCH_QR':
            wps = self.search_waypoints.get(self.current_zone, [])
            if not wps:
                self.get_logger().error(
                    f'Sin waypoints de búsqueda para zona {self.current_zone}')
                return
            if self.pose_ok:
                wps = reorder_from_nearest(wps, self.robot_x, self.robot_y)
            self._search_wps_ordered = wps
            self._search_wp_idx      = 0
            self._search_sub_state   = 'MOVE_TO_WP'
            self._move_wp_published  = False
            self._qr_scan_enabled    = False
            self._scan_sub_timer     = 0.0
            self.get_logger().info(
                f'Sub-FSM búsqueda iniciada: {len(wps)} waypoints')

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.state_start_time).nanoseconds * 1e-9

    def _robot_in_zone(self, zone: str) -> bool:
        if not self.pose_ok:
            return False
        px, py = self.conv.odom_to_pixel(self.robot_x, self.robot_y)
        H, W   = self.zone_masks[zone].shape
        if not (0 <= px < W and 0 <= py < H):
            return False
        return bool(self.zone_masks[zone][py, px])

    def _cancel_navigation(self):
        cancel_msg = Bool()
        cancel_msg.data = True
        self.cancel_pub.publish(cancel_msg)

    def _disable_center_qr(self):
        msg = Bool()
        msg.data = False
        self.enable_pub.publish(msg)

    def _publish_current_image(self):
        if self.state == 'CENTER_QR':
            if self.latest_qr_msg is not None:
                self.img_pub.publish(self.latest_qr_msg)
        else:
            if self.latest_aruco_msg is not None:
                self.img_pub.publish(self.latest_aruco_msg)

    def _publish_status(self):
        sub = self._search_sub_state if self.state == 'SEARCH_QR' else '-'
        msg = String()
        msg.data = (
            f'state={self.state} | sub={sub} | '
            f'mission={self.current_mission} | zone={self.current_zone} | '
            f'qr={self.qr_detected} | qr_scan={self._qr_scan_enabled} | '
            f'nav={self.nav_state} | '
            f'x={self.robot_x:.2f} y={self.robot_y:.2f} '
            f'th={math.degrees(self.robot_th):.1f}'
        )
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
            from geometry_msgs.msg import Twist
            node.cmd_vel_pub.publish(Twist())
            disable = Bool(); disable.data = False
            node.enable_pub.publish(disable)
            cancel = Bool(); cancel.data = True
            node.cancel_pub.publish(cancel)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
