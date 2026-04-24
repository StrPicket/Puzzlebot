"""
aruco_pose.py  -  Medición de posición con ArUcos para PuzzleBot
=================================================================
Solo detecta ArUcos y estima la posición del robot usando solvePnP.
Filtro de Kalman para suavizar la estimación de pose.

Publicaciones
-------------
  /aruco/image_detected   : imagen anotada con marcadores y ejes 3-D
  /aruco/pose             : PoseWithCovarianceStamped con la pose estimada

Ventanas OpenCV
---------------
  "ArUco Pose"  : imagen de la cámara con marcadores y ejes 3-D anotados
  "Mapa Pista"  : vista top-down con posición del robot y ArUcos visibles

Parámetros que debes ajustar
------------------------------
  ARUCO_MAP   : dict  {id: (x_m, y_m, θ_rad)}  - pose de cada ArUco en la pista
  MARKER_SIZE : tamaño real del marcador en metros

  KF_Q_XY    : ruido de proceso en posición (m²/paso)  — más alto = sigue más rápido la medición
  KF_Q_THETA : ruido de proceso en ángulo  (rad²/paso) — más alto = sigue más rápido la medición
  KF_R_XY    : ruido de medición en posición (m²)      — más alto = confía menos en ArUco
  KF_R_THETA : ruido de medición en ángulo  (rad²)     — más alto = confía menos en ArUco

Calibración incluida (cámara PuzzleBot)
  fx=133.87  fy=131.03  cx=157.77  cy=93.02
  dist=[-0.157, -0.618, -0.010, -0.007, 0.744]
"""

import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
from cv2 import aruco
from cv_bridge import CvBridge
import numpy as np
import math

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

# Calibración de la cámara
CAMERA_MATRIX = np.array([
    [133.87191654,   0.0,         157.76772928],
    [  0.0,         131.02895435,  93.02396443],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[-0.15698471, -0.61753973, -0.01000248, -0.00749885, 0.7441658]],
    dtype=np.float64
)

# Tamaño real del marcador ArUco en metros
MARKER_SIZE = 0.055   # ← ajusta al tamaño real de tus marcadores

# Mapa de ArUcos en la pista: {id: (x_global, y_global, yaw_rad)}
# El yaw es la rotación del frente del marcador respecto al eje X global
ARUCO_MAP = {
    1: (4.20,  3.70,  -math.pi/2),
    2: (4.20,  0.0,  math.pi / 2),
    3: (0.0,  3.70,  -math.pi/2),
    4: (0.6,  0.0, math.pi / 2),
}

# ─── Parámetros del Filtro de Kalman ──────────────────────────────────────────
#
#  Q (ruido de proceso): cuánto puede cambiar el estado entre frames.
#    Súbelo si el robot se mueve rápido y el filtro se queda atrás.
#    Bájalo si el robot está estático y quieres más suavizado.
#
#  R (ruido de medición): cuánto confías en la lectura ArUco.
#    Súbelo si las mediciones tienen mucho ruido (marcador lejos, ángulo oblicuo).
#    Bájalo si la cámara es precisa y el marcador está bien iluminado.
#
KF_Q_XY    = 0.001   # m²/paso   — ruido de proceso en x, y
KF_Q_THETA = 0.001   # rad²/paso — ruido de proceso en θ
KF_R_XY    = 0.05    # m²        — ruido de medición en x, y
KF_R_THETA = 0.05    # rad²      — ruido de medición en θ

# ─── Parámetros del mapa top-down ─────────────────────────────────────────────
MAP_SIZE    = 600          # píxeles del canvas cuadrado
MAP_PADDING = 60           # margen interior en píxeles
# Color BGR
COL_BG        = (30,  30,  30)
COL_GRID      = (55,  55,  55)
COL_ARUCO_UNK = (100, 100, 100)   # ArUco del mapa (no visto)
COL_ARUCO_VIS = (50,  220,  50)   # ArUco visible en este frame
COL_ROBOT     = (50,  180, 255)   # Robot
COL_LINE      = (100, 180, 100)   # Línea robot → ArUco

# Diccionario ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
try:
    det_params = aruco.DetectorParameters()
except AttributeError:
    det_params = aruco.DetectorParameters_create()


# ═══════════════════════════════════════════════════════════════════════════
#  FILTRO DE KALMAN  (estado: [x, y, θ])
# ═══════════════════════════════════════════════════════════════════════════

class PoseKalmanFilter:
    """
    Filtro de Kalman lineal de 3 estados: [x, y, θ].

    Modelo de movimiento: estado constante (sin odometría).
      x_k = x_{k-1} + ruido_proceso

    Modelo de medición: la medición es directamente el estado.
      z_k = x_k + ruido_medicion

    Esto es equivalente a un filtro paso-bajo adaptativo donde
    la ganancia (cuánto pesa la medición) se ajusta automáticamente
    según las covarianzas Q y R.
    """

    def __init__(self, q_xy: float, q_theta: float,
                 r_xy: float, r_theta: float):

        # ── Matrices de ruido ────────────────────────────────────────────
        # Q: covarianza del ruido de proceso (3×3)
        self.Q = np.diag([q_xy, q_xy, q_theta])

        # R: covarianza del ruido de medición (3×3)
        self.R = np.diag([r_xy, r_xy, r_theta])

        # ── Estado inicial ───────────────────────────────────────────────
        self.x = np.zeros(3)          # [x, y, θ]

        # P: covarianza del error de estimación (empieza alta = incertidumbre total)
        self.P = np.diag([10.0, 10.0, math.pi**2])

        # Flag: primera medición inicializa el estado directamente
        self.initialized = False

    def predict(self):
        """
        Paso de predicción.
        Modelo: estado constante  →  x_k|k-1 = x_{k-1|k-1}
                                      P_k|k-1 = P_{k-1|k-1} + Q
        """
        # Estado no cambia (sin modelo de movimiento)
        # Solo crece la incertidumbre con Q
        self.P = self.P + self.Q

    def update(self, z: np.ndarray):
        """
        Paso de corrección con una medición z = [x_m, y_m, θ_m].

        K  = P · Hᵀ · (H · P · Hᵀ + R)⁻¹     H = I  →  K = P·(P+R)⁻¹
        x  = x + K·(z - x)
        P  = (I - K)·P

        El ángulo se maneja con wrap_angle para evitar saltos en ±π.
        """
        if not self.initialized:
            # Primera medición: inicializar directamente sin filtrar
            self.x = z.copy()
            self.initialized = True
            return

        # H = I (la medición es el estado directamente)
        # Ganancia de Kalman: K = P · (P + R)⁻¹
        S = self.P + self.R                  # covarianza de innovación
        K = self.P @ np.linalg.inv(S)        # ganancia 3×3

        # Innovación (diferencia medición - predicción), con wrap en θ
        innov = z - self.x
        innov[2] = wrap_angle(innov[2])      # evitar salto ±π

        # Actualizar estado
        self.x = self.x + K @ innov
        self.x[2] = wrap_angle(self.x[2])   # mantener θ en [-π, π]

        # Actualizar covarianza  P = (I - K) · P
        I = np.eye(3)
        self.P = (I - K) @ self.P

    @property
    def state(self):
        """Devuelve (x, y, θ) del estado actual."""
        return float(self.x[0]), float(self.x[1]), float(self.x[2])


# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES GEOMÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def rvec_to_yaw(rvec: np.ndarray) -> float:
    R, _ = cv2.Rodrigues(rvec)
    yaw = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
    return yaw

def marker_side_px(corners_px: np.ndarray) -> float:
    """
    Calcula el lado medio del marcador en píxeles usando las 4 aristas.
    Más robusto que usar solo un lado porque promedia los 4 bordes.

    corners_px : (4, 2) en orden TL, TR, BR, BL
    """
    sides = [
        np.linalg.norm(corners_px[1] - corners_px[0]),  # top
        np.linalg.norm(corners_px[2] - corners_px[1]),  # right
        np.linalg.norm(corners_px[3] - corners_px[2]),  # bottom
        np.linalg.norm(corners_px[0] - corners_px[3]),  # left
    ]
    return float(np.mean(sides))


def dist_from_pixels(side_px: float) -> float:
    """
    Distancia frontal (m) a partir del tamaño aparente del marcador.

    Fórmula pin-hole:  d = f_mean * S_real / S_px
      f_mean = media de fx y fy  (más estable que usar solo uno)
      S_real = MARKER_SIZE en metros
      S_px   = lado medio en píxeles

    Esta estimación es mucho más estable que cam_z de solvePnP a distancias
    >40 cm, porque usa los 4 corners en conjunto y no amplifica el error
    de pose angular.
    """
    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    f_mean = (fx + fy) / 2.0
    if side_px < 2.0:          # marcador demasiado pequeño → ignorar
        return -1.0
    return f_mean * MARKER_SIZE / side_px


def estimate_robot_pose(tvec: np.ndarray, rvec: np.ndarray,
                        marker_id: int, corners_px: np.ndarray):
    """
    Estima la pose del robot en el frame global.

    Estrategia híbrida:
      - Distancia (cam_z)  → calculada desde tamaño aparente en píxeles.
                             Mucho más estable que solvePnP a distancia.
      - Lateral  (cam_x)   → de solvePnP invertido. Preciso cerca, aceptable lejos.
      - Yaw (robot_theta)  → de la matriz de rotación invertida (solvePnP).

    corners_px : (4, 2) — esquinas del marcador en píxeles (TL, TR, BR, BL)
    """

    if marker_id not in ARUCO_MAP:
        return None

    mx, my, m_yaw = ARUCO_MAP[marker_id]

    # ── 1. Matriz de rotación desde solvePnP ────────────────
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec

    # ── 2. Desplazamiento lateral de solvePnP (confiable) ───
    cam_x = -t_inv[0][0]

    # ── 3. Distancia frontal desde tamaño en píxeles ────────
    side_px = marker_side_px(corners_px)
    dist_px = dist_from_pixels(side_px)

    if dist_px < 0:
        return None   # marcador demasiado pequeño o mal detectado

    # cam_z robusto: distancia total proyectada al frente del marcador
    # Corregimos por el desplazamiento lateral para obtener solo la componente frontal
    cam_z = math.sqrt(max(dist_px**2 - cam_x**2, 0.0))

    # ── 4. Distancia horizontal y bearing ───────────────────
    dist_h  = math.sqrt(cam_x**2 + cam_z**2)
    bearing = math.atan2(cam_x, cam_z)

    # ── 5. Yaw del robot ────────────────────────────────────
    yaw_cam = math.atan2(R_inv[0, 2], R_inv[2, 2])
    robot_theta = wrap_angle(m_yaw - yaw_cam)

    # ── 6. Posición en frame global ─────────────────────────
    robot_x = mx + cam_z * math.cos(m_yaw) - cam_x * math.sin(m_yaw)
    robot_y = my + cam_z * math.sin(m_yaw) + cam_x * math.cos(m_yaw)

    # ── 7. Corrección cámara → centro del robot ─────────────
    dx = 0.12
    dy = 0.0
    robot_x_corr = robot_x - dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
    robot_y_corr = robot_y - dx * math.sin(robot_theta) - dy * math.cos(robot_theta)

    return robot_x_corr, robot_y_corr, robot_theta, dist_h, bearing

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN TOP-DOWN
# ═══════════════════════════════════════════════════════════════════════════

def _world_bounds():
    """Calcula min/max del mapa según las posiciones en ARUCO_MAP."""
    xs = [v[0] for v in ARUCO_MAP.values()]
    ys = [v[1] for v in ARUCO_MAP.values()]
    return min(xs), max(xs), min(ys), max(ys)

# Límites reales de la pista (en metros)
_WX_MIN, _WX_MAX = 0.0, 4.8
_WY_MIN, _WY_MAX = 0.0, 3.7

_WX_RANGE = _WX_MAX - _WX_MIN
_WY_RANGE = _WY_MAX - _WY_MIN

# Asegurar rango mínimo de 1 m para no dividir entre 0
_WX_RANGE = max(_WX_MAX - _WX_MIN, 1.0)
_WY_RANGE = max(_WY_MAX - _WY_MIN, 1.0)

def world_to_map(wx: float, wy: float) -> tuple:
    """Convierte coordenadas globales (m) a píxeles en el canvas del mapa."""
    draw_w = MAP_SIZE - 2 * MAP_PADDING
    draw_h = MAP_SIZE - 2 * MAP_PADDING
    scale = min(draw_w / _WX_RANGE, draw_h / _WY_RANGE)
    px = int(MAP_PADDING + (wx - _WX_MIN) * scale)
    py = int(MAP_SIZE - MAP_PADDING - (wy - _WY_MIN) * scale)
    return px, py

def draw_map(robot_x: float, robot_y: float, robot_theta: float,
             visible_ids: list) -> np.ndarray:
    """
    Dibuja el mapa top-down con:
      - Fondo oscuro y grid ligero
      - Todos los ArUcos del mapa (gris = no visible, verde = visible)
      - Líneas del robot hacia cada ArUco visible
      - Robot como flecha azul claro con su orientación
    """
    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), COL_BG, dtype=np.uint8)

    # ── Grid alineado al mundo real (cuadros de 0.6 m) ─────────────────────
    step_m = 0.6

    # Líneas verticales (X)
    x = 0.0
    while x <= _WX_MAX:
        gx, _ = world_to_map(x, _WY_MIN)
        cv2.line(canvas, (gx, MAP_PADDING), (gx, MAP_SIZE - MAP_PADDING),
                COL_GRID, 1)
        cv2.putText(canvas, f"{x:.1f}", (gx + 2, MAP_SIZE - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150,150,150), 1)
        x += step_m

    # Líneas horizontales (Y)
    y = 0.0
    while y <= _WY_MAX:
        _, gy = world_to_map(_WX_MIN, y)
        cv2.line(canvas, (MAP_PADDING, gy), (MAP_SIZE - MAP_PADDING, gy),
                COL_GRID, 1)
        cv2.putText(canvas, f"{y:.1f}", (5, gy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150,150,150), 1)
        y += step_m

    # ── Borde del área de juego ────────────────────────────────────────────
    p0 = world_to_map(_WX_MIN, _WY_MIN)
    p1 = world_to_map(_WX_MAX, _WY_MAX)
    cv2.rectangle(canvas, p0, p1, (80, 80, 80), 1)

    # ── ArUcos del mapa ───────────────────────────────────────────────────
    for mid, (mx, my, mth) in ARUCO_MAP.items():
        px, py = world_to_map(mx, my)
        color  = COL_ARUCO_VIS if mid in visible_ids else COL_ARUCO_UNK

        cv2.rectangle(canvas, (px - 10, py - 10), (px + 10, py + 10),
                      color, -1)
        cv2.rectangle(canvas, (px - 10, py - 10), (px + 10, py + 10),
                      (200, 200, 200), 1)

        arrow_len = 18
        ax = int(px + arrow_len * math.cos(mth))
        ay = int(py - arrow_len * math.sin(mth))
        cv2.arrowedLine(canvas, (px, py), (ax, ay), color, 2, tipLength=0.4)

        cv2.putText(canvas, f"ID{mid}", (px + 13, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # ── Líneas robot → ArUcos visibles ────────────────────────────────────
    rx, ry = world_to_map(robot_x, robot_y)
    for mid in visible_ids:
        if mid in ARUCO_MAP:
            mx, my, _ = ARUCO_MAP[mid]
            mpx, mpy  = world_to_map(mx, my)
            cv2.line(canvas, (rx, ry), (mpx, mpy), COL_LINE, 1, cv2.LINE_AA)

    # ── Robot ─────────────────────────────────────────────────────────────
    robot_r = 12
    cv2.circle(canvas, (rx, ry), robot_r, COL_ROBOT, -1)
    cv2.circle(canvas, (rx, ry), robot_r, (200, 230, 255), 1)

    arrow_len = 22
    fax = int(rx + arrow_len * math.cos(robot_theta))
    fay = int(ry - arrow_len * math.sin(robot_theta))
    cv2.arrowedLine(canvas, (rx, ry), (fax, fay), (255, 255, 255),
                    2, cv2.LINE_AA, tipLength=0.35)

    # ── Texto de pose del robot ───────────────────────────────────────────
    cv2.putText(canvas,
        f"x={robot_x:.2f}m  y={robot_y:.2f}m  "
        f"th={math.degrees(robot_theta):.1f}deg",
        (10, MAP_SIZE - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    return canvas


class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_pose')

        # ── Publishers ────────────────────────────────────────────────────
        self.image_pub = self.create_publisher(Image, '/aruco/image_detected', 10)
        self.pose_pub  = self.create_publisher(
            PoseWithCovarianceStamped, '/aruco/pose', 10)

        # ── Subscriber de imagen ──────────────────────────────────────────
        qos_img = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_img)

        # ── Cámara / ArUco ────────────────────────────────────────────────
        self.camera_width  = 320
        self.camera_height = 180
        self.bridge        = CvBridge()
        self.detector      = aruco.ArucoDetector(aruco_dict, det_params)
        self.latest_frame  = None
        self.latest_header = None

        # Puntos 3-D del marcador en su propio frame (centro en (0,0,0))
        half = MARKER_SIZE / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)

        # ── Filtro de Kalman ──────────────────────────────────────────────
        self.kf = PoseKalmanFilter(
            q_xy=KF_Q_XY, q_theta=KF_Q_THETA,
            r_xy=KF_R_XY, r_theta=KF_R_THETA
        )

        # Pose filtrada (publicada y dibujada en el mapa)
        self.aruco_x     = 0.0
        self.aruco_y     = 0.0
        self.aruco_theta = 0.0

        # ── Timer de procesamiento ────────────────────────────────────────
        self.timer = self.create_timer(1 / 50, self.process_and_publish)

        self.get_logger().info(
            f'aruco_pose (Kalman) iniciado | {len(ARUCO_MAP)} ArUcos en el mapa')

    # ─────────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────────
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.resize(frame, (self.camera_width, self.camera_height))
            self.latest_frame  = frame
            self.latest_header = msg.header
        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')
            self.latest_frame = None

    # ─────────────────────────────────────────────────────────────────────
    #  DETECCIÓN ArUco + ESTIMACIÓN DE POSE + KALMAN
    # ─────────────────────────────────────────────────────────────────────
    def process_and_publish(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        frame = cv2.resize(frame, (self.camera_width, self.camera_height))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        poses_estimadas = []

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for m_idx in range(len(corners)):
                pts = np.squeeze(corners[m_idx])
                if pts.shape != (4, 2):
                    continue

                marker_id = int(ids[m_idx][0])

                # ── Centro en píxeles (solo para visualización) ────────
                cx_m = int(np.mean(pts[:, 0]))
                cy_m = int(np.mean(pts[:, 1]))
                cv2.circle(frame, (cx_m, cy_m), 6, (0, 255, 0), -1)

                # ── solvePnP ──────────────────────────────────────────
                img_pts = pts.astype(np.float64)
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, img_pts,
                    CAMERA_MATRIX, DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    result = estimate_robot_pose(tvec, rvec, marker_id, pts)

                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS,
                                      rvec, tvec, MARKER_SIZE * 0.6)

                    t = tvec.flatten()
                    if result is not None:
                        rx, ry, rth, dist_h, bearing = result
                        poses_estimadas.append((rx, ry, rth))
                        label_color = (0, 255, 0)
                        cv2.putText(frame,
                            f"ID:{marker_id}  d={dist_h:.2f}m  b={math.degrees(bearing):.1f}deg",
                            (int(pts[0][0]), int(pts[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, label_color, 2)
                        cv2.putText(frame,
                            f"robot: x={rx:.2f} y={ry:.2f} th={math.degrees(rth):.1f}deg",
                            (int(pts[0][0]), int(pts[0][1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 0), 1)
                    else:
                        label_color = (0, 165, 255)
                        cv2.putText(frame,
                            f"ID:{marker_id} (no en mapa)",
                            (cx_m - 20, cy_m - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
                        cv2.putText(frame,
                            f"cam: x={t[0]:.2f} y={t[1]:.2f} z={t[2]:.2f}",
                            (int(pts[0][0]), int(pts[0][1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 0), 1)
                else:
                    cv2.putText(frame, f"ID:{marker_id} (solvePnP fail)",
                                (cx_m - 20, cy_m - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            if poses_estimadas:
                # ── Promedio de ArUcos visibles → medición para Kalman ──
                xs  = [p[0] for p in poses_estimadas]
                ys  = [p[1] for p in poses_estimadas]
                ths = [p[2] for p in poses_estimadas]
                meas_x = float(np.mean(xs))
                meas_y = float(np.mean(ys))
                sin_avg = np.mean([math.sin(t) for t in ths])
                cos_avg = np.mean([math.cos(t) for t in ths])
                meas_theta = float(math.atan2(sin_avg, cos_avg))

                # ── Kalman: predecir + corregir ─────────────────────────
                self.kf.predict()
                self.kf.update(np.array([meas_x, meas_y, meas_theta]))

                # Leer estado filtrado
                self.aruco_x, self.aruco_y, self.aruco_theta = self.kf.state

                self._publish_pose()
                self.get_logger().info(
                    f'KF pose: x={self.aruco_x:.2f}  y={self.aruco_y:.2f}  '
                    f'th={math.degrees(self.aruco_theta):.1f}deg')

            else:
                # Sin medición: solo predicción (mantiene el último estado)
                self.kf.predict()

        else:
            # Sin ArUcos: solo predicción
            self.kf.predict()
            cv2.putText(frame, 'No ArUco detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Publicar imagen anotada
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if self.latest_header is not None:
                out_msg.header = self.latest_header
            self.image_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'Error publicando imagen: {e}')

        # ── Ventana de mapa top-down ───────────────────────────────────────
        map_img = draw_map(self.aruco_x, self.aruco_y, self.aruco_theta,
                           [int(ids[m][0]) for m in range(len(corners))]
                           if ids is not None else [])
        cv2.imshow("Mapa Pista", map_img)
        cv2.waitKey(1)

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLICAR POSE
    # ─────────────────────────────────────────────────────────────────────
    def _publish_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = self.aruco_x
        msg.pose.pose.position.y = self.aruco_y
        msg.pose.pose.orientation.z = math.sin(self.aruco_theta / 2)
        msg.pose.pose.orientation.w = math.cos(self.aruco_theta / 2)

        # Publicar la covarianza del filtro (útil para nav2 / otros nodos)
        P = self.kf.P
        msg.pose.covariance[0]  = P[0, 0]   # var(x)
        msg.pose.covariance[7]  = P[1, 1]   # var(y)
        msg.pose.covariance[35] = P[2, 2]   # var(θ)

        self.pose_pub.publish(msg)


# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()