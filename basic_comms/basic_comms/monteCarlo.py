import sys
import select
import tty
import termios
import threading

import rclpy
from rclpy import qos
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_msgs.msg import Float32

import cv2
from cv2 import aruco
from cv_bridge import CvBridge
import numpy as np
import math

# Filtro de partículas (mismo directorio o en PYTHONPATH)
from particle_filter import ParticleFilter

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

# ── Cámara ──────────────────────────────────────────────────────────────────
CAMERA_MATRIX = np.array([
    [133.87191654,   0.0,         157.76772928],
    [  0.0,         131.02895435,  93.02396443],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[-0.15698471, -0.61753973, -0.01000248, -0.00749885, 0.7441658]],
    dtype=np.float64
)

# ── ArUcos ──────────────────────────────────────────────────────────────────
MARKER_SIZE = 0.055   # metros

ARUCO_MAP = {
    1: (4.20,  3.70,  -math.pi / 2),
    2: (4.20,  0.0,   math.pi  / 2),
    3: (0.6,   3.70,  -math.pi / 2),
    4: (0.6,   0.0,   math.pi  / 2),
}

# ── Teleoperación ────────────────────────────────────────────────────────────
VEL_LINEAR  = 0.15   # m/s
VEL_ANGULAR = 0.15   # rad/s

# ── Robot ────────────────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.0505   # m
WHEEL_BASE   = 0.183    # m
CAM_OFFSET_X = 0.12     # m (cámara adelante del centro)

# ── Filtro de partículas ──────────────────────────────────────────────────────
N_PARTICLES = 300

# ── Mapa top-down ────────────────────────────────────────────────────────────
MAP_W, MAP_H = 720, 540
MAP_PAD      = 50
WORLD_X_MAX  = 4.8
WORLD_Y_MAX  = 3.7

# Colores BGR
C_BG         = (20,  20,  20)
C_GRID       = (50,  50,  50)
C_BOUNDARY   = (80,  80,  80)
C_PARTICLE   = (30, 140, 255)    # partículas
C_ROBOT_MCL  = (0,  220, 100)    # pose MCL
C_ROBOT_ODOM = (80, 180, 255)    # pose odométrica
C_ARUCO_UNK  = (100,100,100)
C_ARUCO_VIS  = (50, 220,  50)
C_LINE       = (80, 160,  80)
C_TEXT       = (200,200,200)

# ── Detector ArUco ────────────────────────────────────────────────────────────
_aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
try:
    _det_params = aruco.DetectorParameters()
except AttributeError:
    _det_params = aruco.DetectorParameters_create()


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRÍA AUXILIAR
# ═══════════════════════════════════════════════════════════════════════════

def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def marker_side_px(pts: np.ndarray) -> float:
    sides = [
        np.linalg.norm(pts[1] - pts[0]),
        np.linalg.norm(pts[2] - pts[1]),
        np.linalg.norm(pts[3] - pts[2]),
        np.linalg.norm(pts[0] - pts[3]),
    ]
    return float(np.mean(sides))

def dist_from_pixels(side_px: float) -> float:
    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    f  = (fx + fy) / 2.0
    if side_px < 2.0:
        return -1.0
    return f * MARKER_SIZE / side_px

def estimate_robot_pose_aruco(tvec, rvec, marker_id, pts):
    """
    Devuelve (robot_x, robot_y, robot_theta, dist_h, bearing)
    o None si el marcador no está en el mapa.
    """
    if marker_id not in ARUCO_MAP:
        return None

    mx, my, m_yaw = ARUCO_MAP[marker_id]
    R, _  = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec

    cam_x   = -t_inv[0][0]
    side_px = marker_side_px(pts)
    dist_px = dist_from_pixels(side_px)
    if dist_px < 0:
        return None

    cam_z = math.sqrt(max(dist_px**2 - cam_x**2, 0.0))
    dist_h  = math.sqrt(cam_x**2 + cam_z**2)
    bearing = math.atan2(cam_x, cam_z)

    yaw_cam     = math.atan2(R_inv[0, 2], R_inv[2, 2])
    robot_theta = wrap(m_yaw - yaw_cam)

    robot_x = mx + cam_z * math.cos(m_yaw) - cam_x * math.sin(m_yaw)
    robot_y = my + cam_z * math.sin(m_yaw) + cam_x * math.cos(m_yaw)

    # Corrección cámara → centro del robot
    robot_x -= CAM_OFFSET_X * math.cos(robot_theta)
    robot_y -= CAM_OFFSET_X * math.sin(robot_theta)

    return robot_x, robot_y, robot_theta, dist_h, bearing


# ═══════════════════════════════════════════════════════════════════════════
#  MAPA TOP-DOWN
# ═══════════════════════════════════════════════════════════════════════════

def _w2m(wx, wy):
    """Coordenadas globales (m) → píxeles en el canvas."""
    dw = MAP_W - 2 * MAP_PAD
    dh = MAP_H - 2 * MAP_PAD
    sc = min(dw / WORLD_X_MAX, dh / WORLD_Y_MAX)
    px = int(MAP_PAD + wx * sc)
    py = int(MAP_H - MAP_PAD - wy * sc)
    return px, py

def draw_map(odom_x, odom_y, odom_th,
             mcl_x, mcl_y, mcl_th,
             particles: np.ndarray,
             visible_ids: list) -> np.ndarray:

    canvas = np.full((MAP_H, MAP_W, 3), C_BG, dtype=np.uint8)

    # ── Grid ────────────────────────────────────────────────────────────────
    step = 0.6
    x = 0.0
    while x <= WORLD_X_MAX:
        gx, _ = _w2m(x, 0)
        cv2.line(canvas, (gx, MAP_PAD), (gx, MAP_H - MAP_PAD), C_GRID, 1)
        cv2.putText(canvas, f"{x:.1f}", (gx + 2, MAP_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)
        x += step
    y = 0.0
    while y <= WORLD_Y_MAX:
        _, gy = _w2m(0, y)
        cv2.line(canvas, (MAP_PAD, gy), (MAP_W - MAP_PAD, gy), C_GRID, 1)
        cv2.putText(canvas, f"{y:.1f}", (2, gy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)
        y += step

    # Borde de la pista
    p0 = _w2m(0, 0)
    p1 = _w2m(WORLD_X_MAX, WORLD_Y_MAX)
    cv2.rectangle(canvas, p0, p1, C_BOUNDARY, 1)

    # ── Partículas ───────────────────────────────────────────────────────────
    if particles is not None and len(particles):
        for p in particles:
            ppx, ppy = _w2m(p[0], p[1])
            if 0 <= ppx < MAP_W and 0 <= ppy < MAP_H:
                cv2.circle(canvas, (ppx, ppy), 2, C_PARTICLE, -1)

    # ── ArUcos del mapa ──────────────────────────────────────────────────────
    for mid, (mx, my, mth) in ARUCO_MAP.items():
        px, py = _w2m(mx, my)
        col = C_ARUCO_VIS if mid in visible_ids else C_ARUCO_UNK
        cv2.rectangle(canvas, (px - 9, py - 9), (px + 9, py + 9), col, -1)
        cv2.rectangle(canvas, (px - 9, py - 9), (px + 9, py + 9), (180, 180, 180), 1)
        al = 16
        ax = int(px + al * math.cos(mth))
        ay = int(py - al * math.sin(mth))
        cv2.arrowedLine(canvas, (px, py), (ax, ay), col, 2, tipLength=0.4)
        cv2.putText(canvas, f"#{mid}", (px + 11, py - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    # ── Líneas robot → ArUcos visibles ───────────────────────────────────────
    mx_px, my_px = _w2m(mcl_x, mcl_y)
    for mid in visible_ids:
        if mid in ARUCO_MAP:
            apx, apy = _w2m(ARUCO_MAP[mid][0], ARUCO_MAP[mid][1])
            cv2.line(canvas, (mx_px, my_px), (apx, apy), C_LINE, 1, cv2.LINE_AA)

    # ── Pose odométrica (azul claro, más tenue) ──────────────────────────────
    ox, oy = _w2m(odom_x, odom_y)
    cv2.circle(canvas, (ox, oy), 7, C_ROBOT_ODOM, -1)
    al = 14
    cv2.arrowedLine(canvas, (ox, oy),
                    (int(ox + al * math.cos(odom_th)),
                     int(oy - al * math.sin(odom_th))),
                    C_ROBOT_ODOM, 1, tipLength=0.4)

    # ── Pose MCL (verde brillante) ───────────────────────────────────────────
    cv2.circle(canvas, (mx_px, my_px), 11, C_ROBOT_MCL, -1)
    cv2.circle(canvas, (mx_px, my_px), 11, (200, 255, 200), 1)
    al = 20
    cv2.arrowedLine(canvas, (mx_px, my_px),
                    (int(mx_px + al * math.cos(mcl_th)),
                     int(my_px - al * math.sin(mcl_th))),
                    (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)

    # ── Leyenda ──────────────────────────────────────────────────────────────
    cv2.circle(canvas, (MAP_W - 130, 18), 6, C_ROBOT_ODOM, -1)
    cv2.putText(canvas, "Odometría", (MAP_W - 120, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ROBOT_ODOM, 1)
    cv2.circle(canvas, (MAP_W - 130, 36), 6, C_ROBOT_MCL, -1)
    cv2.putText(canvas, "MCL", (MAP_W - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ROBOT_MCL, 1)
    cv2.circle(canvas, (MAP_W - 130, 50), 4, C_PARTICLE, -1)
    cv2.putText(canvas, "Partículas", (MAP_W - 120, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_PARTICLE, 1)

    # ── Texto de estado ──────────────────────────────────────────────────────
    cv2.putText(canvas,
        f"Odom  x={odom_x:.2f} y={odom_y:.2f} th={math.degrees(odom_th):.1f}deg",
        (8, MAP_H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_ROBOT_ODOM, 1)
    cv2.putText(canvas,
        f"MCL   x={mcl_x:.2f}  y={mcl_y:.2f}  th={math.degrees(mcl_th):.1f}deg",
        (8, MAP_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_ROBOT_MCL, 1)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  LECTURA DE TECLADO (no-bloqueante)
# ═══════════════════════════════════════════════════════════════════════════

class KeyboardReader:
    """Lee teclas en modo raw sin bloquear el hilo principal."""

    KEY_MAP = {
        '\x1b[A': 'UP',    '\x1b[B': 'DOWN',
        '\x1b[C': 'RIGHT', '\x1b[D': 'LEFT',
        'w': 'UP',    's': 'DOWN',
        'a': 'LEFT',  'd': 'RIGHT',
        ' ': 'STOP',
        'r': 'RESET', 'R': 'RESET',
        'q': 'QUIT',  '\x1b': 'QUIT',
    }

    def __init__(self):
        self._key   = None
        self._lock  = threading.Lock()
        self._old   = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while True:
            try:
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    # Posible secuencia de escape (flechas)
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch2 = sys.stdin.read(1)
                        r2, _, _ = select.select([sys.stdin], [], [], 0.02)
                        if r2:
                            ch3 = sys.stdin.read(1)
                            ch = ch + ch2 + ch3
                        else:
                            ch = ch + ch2
                with self._lock:
                    self._key = ch
            except Exception:
                break

    def get_key(self):
        with self._lock:
            k = self._key
            self._key = None
        return self.KEY_MAP.get(k, None) if k else None

    def restore(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


# ═══════════════════════════════════════════════════════════════════════════
#  NODO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class TeleopMCLNode(Node):

    def __init__(self):
        super().__init__('teleop_mcl')

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_pub   = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pose_pub  = self.create_publisher(
            PoseWithCovarianceStamped, '/mcl/pose', 10)
        self.image_pub = self.create_publisher(Image, '/aruco/image_detected', 10)

        # ── Subscribers ───────────────────────────────────────────────────
        qos_img = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Image,   '/image_raw',    self._cb_image, qos_img)
        self.create_subscription(Float32, 'VelocityEncR', self._cb_encR,
                                 qos.qos_profile_sensor_data)
        self.create_subscription(Float32, 'VelocityEncL', self._cb_encL,
                                 qos.qos_profile_sensor_data)

        # ── Cámara / ArUco ────────────────────────────────────────────────
        self.bridge    = CvBridge()
        self.detector  = aruco.ArucoDetector(_aruco_dict, _det_params)
        self.cam_w, self.cam_h = 320, 180
        self.obj_pts   = np.array([
            [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
            [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
            [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
            [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
        ], dtype=np.float64)

        self.latest_frame  = None
        self.latest_header = None

        # ── Encoders ──────────────────────────────────────────────────────
        self.wr = Float32()
        self.wl = Float32()

        # ── Odometría ─────────────────────────────────────────────────────
        self.odom_x     = 0.0
        self.odom_y     = 0.0
        self.odom_theta = 0.0

        # ── Filtro de partículas ──────────────────────────────────────────
        self.pf = ParticleFilter(
            n_particles  = N_PARTICLES,
            aruco_map    = ARUCO_MAP,
            x_min=0.0, x_max=WORLD_X_MAX,
            y_min=0.0, y_max=WORLD_Y_MAX,
            alpha1=0.05, alpha2=0.05,
            alpha3=0.02, alpha4=0.02,
            sigma_dist   = 0.3,
            sigma_bearing= 0.25,
            rand_frac    = 0.05,
        )
        self.pf.init_uniform()
        self.mcl_x, self.mcl_y, self.mcl_theta = self.pf.estimate()

        # ── Teleop ────────────────────────────────────────────────────────
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.kbd   = KeyboardReader()

        # ── Ventana de visualización ──────────────────────────────────────────
        cv2.namedWindow("MCL - Mapa Pista", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MCL - Mapa Pista", MAP_W, MAP_H)

        # ── Timers ────────────────────────────────────────────────────────
        self.t_odom   = self.create_timer(1 / 100, self._cb_odom)
        self.t_main   = self.create_timer(1 /  30, self._cb_main)

        self._t_odom_last = self.get_clock().now()

        self.get_logger().info(
            "TeleopMCL iniciado\n"
            "  W/S/A/D o flechas : mover\n"
            "  ESPACIO           : detener\n"
            "  R                 : reinicializar MCL\n"
            "  Q / ESC           : salir"
        )

    # ── Callbacks de sensores ───────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame  = cv2.resize(frame, (self.cam_w, self.cam_h))
            self.latest_header = msg.header
        except Exception as e:
            self.get_logger().error(f'image cb: {e}')

    def _cb_encR(self, msg: Float32): self.wr = msg
    def _cb_encL(self, msg: Float32): self.wl = msg

    # ── Odometría ───────────────────────────────────────────────────────────

    def _cb_odom(self):
        now = self.get_clock().now()
        dt  = (now - self._t_odom_last).nanoseconds * 1e-9
        self._t_odom_last = now
        if dt <= 0 or dt > 0.5:
            return

        vr = WHEEL_RADIUS * self.wr.data
        vl = WHEEL_RADIUS * self.wl.data
        v  = (vr + vl) / 2.0
        w  = (vr - vl) / WHEEL_BASE

        self.odom_x     += v * math.cos(self.odom_theta) * dt
        self.odom_y     += v * math.sin(self.odom_theta) * dt
        self.odom_theta  = wrap(self.odom_theta + w * dt)

        # Predicción del filtro de partículas con nueva odometría
        self.pf.predict_from_odom(self.odom_x, self.odom_y, self.odom_theta)

    # ── Ciclo principal (ArUcos + MCL + teleop + visualización) ─────────────

    def _cb_main(self):
        # ── 1. Teclado ───────────────────────────────────────────────────
        key = self.kbd.get_key()
        if key == 'QUIT':
            self._shutdown()
            return
        if key == 'RESET':
            self.pf.init_uniform()
            self.get_logger().info('MCL reinicializado (distribución uniforme)')
        if key == 'UP':
            self.v_cmd, self.w_cmd =  VEL_LINEAR, 0.0
        elif key == 'DOWN':
            self.v_cmd, self.w_cmd = -VEL_LINEAR, 0.0
        elif key == 'LEFT':
            self.v_cmd, self.w_cmd = 0.0,  VEL_ANGULAR
        elif key == 'RIGHT':
            self.v_cmd, self.w_cmd = 0.0, -VEL_ANGULAR
        elif key == 'STOP':
            self.v_cmd, self.w_cmd = 0.0, 0.0

        cmd = Twist()
        cmd.linear.x  = self.v_cmd
        cmd.angular.z = self.w_cmd
        self.cmd_pub.publish(cmd)

        # ── 2. Procesar ArUcos ───────────────────────────────────────────
        visible_ids = []
        observations = []   # [(id, dist_m, bearing_rad)]
        aruco_poses  = []   # [(x, y, theta)] para inicialización

        if self.latest_frame is not None:
            frame = self.latest_frame.copy()
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(frame, corners, ids)

                for m_idx in range(len(corners)):
                    pts = np.squeeze(corners[m_idx])
                    if pts.shape != (4, 2):
                        continue

                    mid = int(ids[m_idx][0])
                    visible_ids.append(mid)

                    cx_m = int(np.mean(pts[:, 0]))
                    cy_m = int(np.mean(pts[:, 1]))
                    cv2.circle(frame, (cx_m, cy_m), 5, (0, 255, 0), -1)

                    img_pts = pts.astype(np.float64)
                    ok, rvec, tvec = cv2.solvePnP(
                        self.obj_pts, img_pts,
                        CAMERA_MATRIX, DIST_COEFFS,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if ok:
                        result = estimate_robot_pose_aruco(tvec, rvec, mid, pts)
                        cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS,
                                          rvec, tvec, MARKER_SIZE * 0.6)

                        if result is not None:
                            rx, ry, rth, dist_h, bearing = result
                            aruco_poses.append((rx, ry, rth))

                            # Observación para el filtro de partículas
                            # bearing en frame del robot
                            observations.append((mid, dist_h, bearing))

                            col = (0, 255, 0)
                            cv2.putText(frame,
                                f"#{mid} d={dist_h:.2f}m b={math.degrees(bearing):.0f}°",
                                (int(pts[0][0]), int(pts[0][1]) - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                            cv2.putText(frame,
                                f"x={rx:.2f} y={ry:.2f} th={math.degrees(rth):.0f}°",
                                (int(pts[0][0]), int(pts[0][1]) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 0), 1)
                        else:
                            t = tvec.flatten()
                            cv2.putText(frame, f"#{mid} (no en mapa)",
                                        (cx_m - 20, cy_m - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1)
                    else:
                        cv2.putText(frame, f"#{mid} solvePnP fail",
                                    (cx_m - 20, cy_m - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)
            else:
                cv2.putText(frame, 'No ArUco', (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ── 3. Actualizar MCL con observaciones ──────────────────────
            if observations:
                # Primera vez con ArUco: inicializar el filtro cerca de la medición
                if not self.pf.initialized or \
                   (np.allclose(self.pf._estimate, 0) and aruco_poses):
                    avg_x = float(np.mean([p[0] for p in aruco_poses]))
                    avg_y = float(np.mean([p[1] for p in aruco_poses]))
                    sins  = [math.sin(p[2]) for p in aruco_poses]
                    coss  = [math.cos(p[2]) for p in aruco_poses]
                    avg_th = math.atan2(np.mean(sins), np.mean(coss))
                    self.pf.init_at(avg_x, avg_y, avg_th)
                    self.pf.reset_odom(self.odom_x, self.odom_y, self.odom_theta)

                self.pf.update(observations)

            self.mcl_x, self.mcl_y, self.mcl_theta = self.pf.estimate()

            # ── 4. Publicar pose MCL ─────────────────────────────────────
            self._publish_mcl_pose()

            # ── 5. Overlay HUD en imagen ─────────────────────────────────
            cv2.putText(frame,
                f"MCL x={self.mcl_x:.2f} y={self.mcl_y:.2f} "
                f"th={math.degrees(self.mcl_theta):.0f}deg",
                (4, self.cam_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_ROBOT_MCL, 1)

            cmd_txt = (f"v={self.v_cmd:+.2f} w={self.w_cmd:+.2f}"
                       f"  Neff={self.pf.neff:.0f}/{N_PARTICLES}")
            cv2.putText(frame, cmd_txt, (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Publicar imagen
            try:
                out = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                if self.latest_header:
                    out.header = self.latest_header
                self.image_pub.publish(out)
            except Exception as e:
                self.get_logger().error(f'img pub: {e}')
        else:
            # Sin frame: al menos predecir con odometría ya sucedió en _cb_odom
            self.mcl_x, self.mcl_y, self.mcl_theta = self.pf.estimate()

        # ── 6. Mapa top-down ─────────────────────────────────────────────
        mapa = draw_map(
            self.odom_x, self.odom_y, self.odom_theta,
            self.mcl_x,  self.mcl_y,  self.mcl_theta,
            self.pf.particles,
            visible_ids,
        )

        # Mostrar controles en el mapa
        controls = [
            "W/S: adelante/atras",
            "A/D: girar",
            "ESPACIO: parar",
            "R: reinicializar MCL",
            "Q/ESC: salir",
        ]
        for i, txt in enumerate(controls):
            cv2.putText(mapa, txt, (MAP_W - 190, 80 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 150, 150), 1)

        neff_ratio = self.pf.neff / N_PARTICLES
        bar_w = int(160 * neff_ratio)
        cv2.rectangle(mapa, (MAP_W - 190, 68), (MAP_W - 190 + 160, 76), (50,50,50), -1)
        cv2.rectangle(mapa, (MAP_W - 190, 68), (MAP_W - 190 + bar_w, 76),
                      (int(255*(1-neff_ratio)), int(255*neff_ratio), 50), -1)
        cv2.putText(mapa, f"Neff {self.pf.neff:.0f}/{N_PARTICLES}",
                    (MAP_W - 190, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160,160,160), 1)

        cv2.imshow("MCL - Mapa Pista", mapa)   # mismo nombre, sin em-dash
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self._shutdown()

    # ── Publicar pose ───────────────────────────────────────────────────────

    def _publish_mcl_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x    = self.mcl_x
        msg.pose.pose.position.y    = self.mcl_y
        msg.pose.pose.orientation.z = math.sin(self.mcl_theta / 2)
        msg.pose.pose.orientation.w = math.cos(self.mcl_theta / 2)
        P = self.pf.particles
        if P is not None:
            msg.pose.covariance[0]  = float(np.var(P[:, 0]))
            msg.pose.covariance[7]  = float(np.var(P[:, 1]))
            msg.pose.covariance[35] = float(np.var(P[:, 2]))
        self.pose_pub.publish(msg)

    # ── Cierre limpio ───────────────────────────────────────────────────────

    def _shutdown(self):
        stop = Twist()
        self.cmd_pub.publish(stop)
        self.kbd.restore()
        cv2.destroyAllWindows()
        self.get_logger().info('TeleopMCL apagado.')
        rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = TeleopMCLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._shutdown()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
