#!/usr/bin/env python3
"""
mission_manager.py
═══════════════════════════════════════════════════════════════════════
Nodo maestro — Orquestador de misiones del Puzzlebot

Misiones disponibles:
  mission_1 → carga   → encontrar QR → ir a descarga
  mission_2 → racks   → encontrar QR → ir a descarga

Máquina de estados:
  IDLE
    └─ recibe /mission → NAVIGATE_TO_ZONE

  NAVIGATE_TO_ZONE
    └─ publica /goal_input y espera DONE en /nav/status
    └─ llega → SEARCH_QR

  SEARCH_QR
    └─ publica lista de waypoints de la zona en /plan
    └─ recorre en loop hasta recibir True en /qr/detected
    └─ detecta QR → cancela navegación → CENTER_QR

  CENTER_QR
    └─ habilita /center_qr/enable = True
    └─ espera a que center_qr termine (ratio >= stop_ratio, 
       se detecta por /qr/detected = False tras alineación)
    └─ termina → NAVIGATE_TO_TRUCK

  LIFT_PALLET
    └─ espera a que se complete (se detecta por /lift_pallet/status = DONE)
    └─ termina → NAVIGATE_TO_TRUCK

  NAVIGATE_TO_TRUCK
    └─ publica /goal_input = "descarga"
    └─ llega → DROP_PALLET

  DROP_PALLET
    └─ espera a que se complete (se detecta por /drop_pallet/status = DONE)
    └─ termina → DONE

  DONE
    └─ espera nueva misión

Tópicos suscritos:
  /mission            — std_msgs/String   ("mission_1" | "mission_2")
  /nav/status         — std_msgs/String   (estado waypoint_controller)
  /qr/detected        — std_msgs/Bool     (QR visible en cámara)
  /aruco/pose_centered — geometry_msgs/PoseWithCovarianceStamped (pose robot)

Tópicos publicados:
  /goal_input         — std_msgs/String   (para semantic_planner)
  /plan               — nav_msgs/Path     (waypoints de búsqueda directos)
  /center_qr/enable   — std_msgs/Bool     (activa center_qr)
  /nav/cancel         — std_msgs/Bool     (cancela waypoint_controller)
  /mission/status     — std_msgs/String   (estado para UI)
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
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

SEMANTIC_MAP_PATH = "/home/juanjo/semantic_map.png"
ROUTE_MAP_PATH    = "/home/juanjo/route_map.png"
MAP_RESOLUTION    = 0.05   # m/pixel — debe coincidir con semantic_planner

# Submuestreo de waypoints de búsqueda (cada N píxeles de ruta)
SEARCH_WP_SUBSAMPLE = 8

# Distancia mínima entre waypoints de búsqueda consecutivos (m)
# Evita waypoints demasiado juntos tras el submuestreo
SEARCH_WP_MIN_DIST = 0.20

# Radio para considerar que el robot ya está en la zona objetivo (m)
# Si está más cerca que esto al iniciar, salta directo a SEARCH_QR
ZONE_ARRIVAL_RADIUS = 0.20

# Cuántos mensajes False consecutivos de /qr/detected tras CENTER_QR
# indican que el QR se perdió (fallo de alineación)
QR_LOST_COUNT_THRESHOLD = 30


# ═══════════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE COORDENADAS  (igual que semantic_planner)
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
#  EXTRACCIÓN DE WAYPOINTS DE BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════

def build_zone_masks(semantic: np.ndarray):
    """Igual que en semantic_planner — extrae máscaras por zona."""
    green = (semantic[:,:,1] > 200) & (semantic[:,:,0] < 80) & (semantic[:,:,2] < 80)
    blue  = (semantic[:,:,0] > 200) & (semantic[:,:,1] < 80) & (semantic[:,:,2] < 80)
    red   = (semantic[:,:,2] > 200) & (semantic[:,:,0] < 80) & (semantic[:,:,1] < 80)
    return {"carga": green, "racks": blue, "descarga": red}


def build_route_mask(route_img: np.ndarray):
    """Extrae píxeles morados (rutas) del route_map."""
    b, g, r = route_img[:,:,0], route_img[:,:,1], route_img[:,:,2]
    return (b > 180) & (g < 80) & (r > 180)


def extract_search_waypoints(zone_mask: np.ndarray,
                              route_mask: np.ndarray,
                              conv: CoordConverter,
                              subsample: int = SEARCH_WP_SUBSAMPLE,
                              min_dist_m: float = SEARCH_WP_MIN_DIST):
    """
    Extrae waypoints de búsqueda para una zona:
      1. Intersecta píxeles de ruta con la máscara de zona
      2. Submuestrea cada `subsample` píxeles
      3. Ordena por nearest-neighbor para ruta coherente
      4. Filtra waypoints demasiado cercanos entre sí
      5. Devuelve lista de (odom_x, odom_y)
    """
    # Intersección ruta ∩ zona
    zone_route = route_mask & zone_mask
    ys, xs = np.where(zone_route)

    if len(xs) == 0:
        return []

    # Submuestreo uniforme por índice
    pixels = list(zip(xs[::subsample], ys[::subsample]))

    if len(pixels) == 0:
        return []

    # Ordenar por nearest-neighbor desde el primer punto
    ordered = [pixels[0]]
    remaining = pixels[1:]
    while remaining:
        last = ordered[-1]
        dists = [math.hypot(p[0]-last[0], p[1]-last[1]) for p in remaining]
        idx = int(np.argmin(dists))
        ordered.append(remaining.pop(idx))

    # Convertir a odom y filtrar por distancia mínima
    waypoints = []
    last_ox, last_oy = None, None
    for px, py in ordered:
        ox, oy = conv.pixel_to_odom(px, py)
        if last_ox is None or math.hypot(ox-last_ox, oy-last_oy) >= min_dist_m:
            waypoints.append((ox, oy))
            last_ox, last_oy = ox, oy

    return waypoints


def reorder_from_nearest(waypoints: list, robot_x: float, robot_y: float):
    """
    Reordena la lista de waypoints empezando desde el más cercano
    al robot. Así el recorrido siempre empieza desde donde está.
    """
    if not waypoints:
        return waypoints
    dists = [math.hypot(wp[0]-robot_x, wp[1]-robot_y) for wp in waypoints]
    idx = int(np.argmin(dists))
    return waypoints[idx:] + waypoints[:idx]


# ═══════════════════════════════════════════════════════════════════════
#  NODO
# ═══════════════════════════════════════════════════════════════════════

class MissionManagerNode(Node):

    # Mapeo misión → zona origen
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
            self.get_logger().error('No se pudieron cargar los mapas — verifica las rutas')
            raise RuntimeError('Mapas no encontrados')

        H, W, _ = semantic_img.shape
        self.conv = CoordConverter(W, H, MAP_RESOLUTION)

        self.zone_masks  = build_zone_masks(semantic_img)
        self.route_mask  = build_route_mask(route_img)

        # Pre-calcular waypoints de búsqueda por zona (excepto descarga)
        self.search_waypoints = {}
        for zone in ('carga', 'racks'):
            wps = extract_search_waypoints(
                self.zone_masks[zone],
                self.route_mask,
                self.conv
            )
            self.search_waypoints[zone] = wps
            self.get_logger().info(
                f'Zona {zone}: {len(wps)} waypoints de búsqueda extraídos')

        # ── Estado ────────────────────────────────────────────────────
        self.state          = 'IDLE'
        self.current_mission = None     # 'mission_1' | 'mission_2'
        self.current_zone   = None      # 'carga' | 'racks'

        self.robot_x    = 0.0
        self.robot_y    = 0.0
        self.robot_th   = 0.0
        self.pose_ok    = False

        self.nav_state       = 'WAIT_PLAN'   # último estado de waypoint_controller
        self.forklift_state  = 'IDLE'
        self.qr_detected     = False
        self.qr_centered     = False
        self.qr_lost_count   = 0

        self.latest_aruco_msg = None
        self.latest_qr_msg = None

        self.state_start_time = self.get_clock().now()

        # ── Publishers ────────────────────────────────────────────────
        self.goal_pub    = self.create_publisher(String, '/goal_input',      10)
        self.plan_pub    = self.create_publisher(Path,   '/plan',            10)
        self.enable_pub  = self.create_publisher(Bool,   '/center_qr/enable', 10)
        self.cancel_pub  = self.create_publisher(Bool,   '/nav/cancel',      10)
        self.status_pub  = self.create_publisher(String, '/mission/status',  10)
        self.img_pub_compressed = self.create_publisher(CompressedImage, '/mission_manager_image/compressed', 10)

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
            PoseWithCovarianceStamped, '/aruco/pose_centered',
            self._pose_cb, 10)
        
        self.aruco_cam = self.create_subscription(
            CompressedImage, '/aruco/image_detected/compressed',
            self.aruco_cam_cb, 10)
        
        self.qr_cam = self.create_subscription(
            CompressedImage, '/qr/image_detected/compressed',
            self.qr_cam_cb, 10)

        # ── Timer principal a 10 Hz ───────────────────────────────────
        self.timer = self.create_timer(0.1, self._fsm_step)

        self.get_logger().info(
            'mission_manager listo\n'
            '  Envía misión con:\n'
            '  ros2 topic pub --once /mission std_msgs/String "data: \'mission_1\'"')

    # ─────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────

    def aruco_cam_cb(self, msg: CompressedImage):
        self.latest_aruco_msg = msg

    def qr_cam_cb(self, msg: CompressedImage):
        self.latest_qr_msg = msg

    def _mission_cb(self, msg: String):
        mission = msg.data.strip().lower()
        if mission not in self.MISSION_MAP:
            self.get_logger().warn(
                f'Misión desconocida: "{mission}"  '
                f'Opciones: {list(self.MISSION_MAP.keys())}')
            return

        if self.state not in ('IDLE', 'DONE'):
            self.get_logger().warn(
                f'Misión recibida pero el robot está en estado {self.state} — ignorada')
            return

        self.current_mission = mission
        self.current_zone    = self.MISSION_MAP[mission]
        self.get_logger().info(
            f'Misión recibida: {mission} → zona: {self.current_zone}')
        self._transition('NAVIGATE_TO_ZONE')

    def _nav_status_cb(self, msg: String):
        # Extraer estado del string: "state=GO_TO_GOAL | wp=3/10 | ..."
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
        self.qr_detected = msg.data
        if self.state == 'CENTER_QR' and not msg.data:
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
        self.pose_ok = True

    # ─────────────────────────────────────────────────────────────────
    #  MÁQUINA DE ESTADOS
    # ─────────────────────────────────────────────────────────────────

    def _fsm_step(self):
        self._publish_status()
        self._publish_current_image()

        if self.state == 'IDLE':
            pass   # espera misión

        elif self.state == 'NAVIGATE_TO_ZONE':
            self._step_navigate_to_zone()

        elif self.state == 'SEARCH_QR':
            self._step_search_qr()

        elif self.state == 'CENTER_QR':
            self._step_center_qr()

        elif self.state == 'LIFT_PALLET':
            self._step_lift_pallet()

        elif self.state == 'NAVIGATE_TO_TRUCK':
            self._step_navigate_to_truck()

        elif self.state == 'DROP_PALLET':
            self._step_drop_pallet()

        elif self.state == 'DONE':
            pass   # espera nueva misión

    # ── NAVIGATE_TO_ZONE ──────────────────────────────────────────────

    def _step_navigate_to_zone(self):
        elapsed = self._elapsed()

        # Primera vez en este estado: publicar goal o verificar si ya está
        if elapsed < 0.15:
            if self.pose_ok and self._robot_in_zone(self.current_zone):
                self.get_logger().info(
                    f'Robot ya está en zona {self.current_zone} → saltando navegación')
                self._transition('SEARCH_QR')
                return

            goal_msg = String()
            goal_msg.data = self.current_zone
            self.goal_pub.publish(goal_msg)
            self.get_logger().info(
                f'Navegando a zona: {self.current_zone}')
            return

        # Waypoint_controller llegó al destino
        if self.nav_state == 'DONE':
            self.get_logger().info(f'Llegó a zona {self.current_zone} → SEARCH_QR')
            self._transition('SEARCH_QR')

    # ── SEARCH_QR ─────────────────────────────────────────────────────

    def _step_search_qr(self):
        elapsed = self._elapsed()

        # QR encontrado durante la navegación de búsqueda
        if self.qr_detected:
            self.get_logger().info('QR detectado → cancelando navegación → CENTER_QR')
            self._cancel_navigation()
            self._transition('CENTER_QR')
            return

        # Primera vez: publicar plan de búsqueda desde posición actual
        if elapsed < 0.15:
            self._publish_search_plan()
            return

        # Waypoint_controller terminó el recorrido y no encontró QR → loop
        if self.nav_state == 'DONE':
            self.get_logger().info(
                'Recorrido de búsqueda completo sin QR — repitiendo loop')
            self._publish_search_plan()   # reordena desde posición actual

    # ── CENTER_QR ─────────────────────────────────────────────────────

    def _step_center_qr(self):
        elapsed = self._elapsed()

        # Primera vez: habilitar center_qr
        if elapsed < 0.15:
            enable_msg = Bool()
            enable_msg.data = True
            self.enable_pub.publish(enable_msg)
            self.get_logger().info('center_qr habilitado')
            return

        # QR perdido por muchos frames consecutivos
        if self.qr_lost_count >= QR_LOST_COUNT_THRESHOLD:
            self.get_logger().warn(
                f'QR perdido ({self.qr_lost_count} frames) — volviendo a búsqueda')
            self._disable_center_qr()
            self._transition('SEARCH_QR')
            return

        if self.qr_centered:
            self.get_logger().info(
                'Alineación completada → NAVIGATE_TO_TRUCK')
            self._disable_center_qr()
            self._transition('LIFT_PALLET')

    # ── LIFT_PALLET ────────────────────────────────────────────────

    def _step_lift_pallet(self):
        elapsed = self._elapsed()

        if elapsed < 0.15:
            self.get_logger().info('Levantando pallet')
            return
        
        if self.forklift_state == 'DONE':
            self.get_logger().info('Pallet levantado → NAVIGATE_TO_TRUCK')
            self._transition('NAVIGATE_TO_TRUCK')        

    # ── NAVIGATE_TO_TRUCK ──────────────────────────────────────────

    def _step_navigate_to_truck(self):
        elapsed = self._elapsed()

        if elapsed < 0.15:
            goal_msg = String()
            goal_msg.data = 'descarga'
            self.goal_pub.publish(goal_msg)
            self.get_logger().info('Navegando a zona de descarga')
            return

        if self.nav_state == 'DONE':
            self.get_logger().info('Llegó a descarga → DROP_PALLET')
            self._transition('DROP_PALLET')

    # ── LIFT_PALLET ────────────────────────────────────────────────

    def _step_drop_pallet(self):
        elapsed = self._elapsed()

        if elapsed < 0.15:
            self.get_logger().info('Dejando pallet')
            return
        
        if self.forklift_state == 'DONE':
            self.get_logger().info('Pallet entregado → DONE')
            self._transition('DONE') 

    # ─────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _transition(self, new_state: str):
        self.get_logger().info(f'[FSM] {self.state} → {new_state}')
        self.state = new_state
        self.state_start_time = self.get_clock().now()
        self.nav_state    = 'WAIT_PLAN'   # reset para no disparar condiciones viejas
        if new_state in ('LIFT_PALLET', 'DROP_PALLET'):
            self.forklift_state = 'IDLE'
        self.qr_lost_count = 0

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.state_start_time).nanoseconds * 1e-9

    def _reset_state_timer(self):
        self.state_start_time = self.get_clock().now()

    def _robot_in_zone(self, zone: str) -> bool:
        """True si el robot ya está dentro de la zona (píxel navegable de esa zona)."""
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
        enable_msg = Bool()
        enable_msg.data = False
        self.enable_pub.publish(enable_msg)

    def _publish_search_plan(self):
        """
        Construye y publica el plan de búsqueda para la zona actual,
        reordenado desde la posición actual del robot.
        """
        wps = self.search_waypoints.get(self.current_zone, [])
        if not wps:
            self.get_logger().error(
                f'No hay waypoints de búsqueda para zona {self.current_zone}')
            return

        # Reordenar desde la posición actual del robot
        if self.pose_ok:
            wps = reorder_from_nearest(wps, self.robot_x, self.robot_y)

        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp    = self.get_clock().now().to_msg()

        for ox, oy in wps:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp    = msg.header.stamp
            pose.pose.position.x = ox
            pose.pose.position.y = oy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.plan_pub.publish(msg)
        self.get_logger().info(
            f'Plan de búsqueda publicado: {len(wps)} waypoints '
            f'(zona={self.current_zone})')
        
    def _publish_current_image(self):

        if self.state == 'CENTER_QR':
            if self.latest_qr_msg is not None:
                self.img_pub_compressed.publish(self.latest_qr_msg)
        else:
            if self.latest_aruco_msg is not None:
                self.img_pub_compressed.publish(self.latest_aruco_msg)

    def _publish_status(self):
        msg = String()
        msg.data = (
            f'state={self.state} | '
            f'mission={self.current_mission} | '
            f'zone={self.current_zone} | '
            f'qr={self.qr_detected} | '
            f'nav={self.nav_state} | '
            f'x={self.robot_x:.2f} y={self.robot_y:.2f} th={math.degrees(self.robot_th):.2f} | '
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
        # Asegurar que center_qr quede deshabilitado y el robot detenido
        try:
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
