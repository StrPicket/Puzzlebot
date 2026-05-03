import rclpy
from rclpy import qos
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float32

import cv2
from cv2 import aruco
from cv_bridge import CvBridge
import numpy as np
import math

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

# ── Calibración de la cámara ───────────────────────────────────────────────
CAMERA_MATRIX = np.array([
    [133.87191654,   0.0,         157.76772928],
    [  0.0,         131.02895435,  93.02396443],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[-0.15698471, -0.61753973, -0.01000248, -0.00749885, 0.7441658]],
    dtype=np.float64
)

# ── ArUco ─────────────────────────────────────────────────────────────────
MARKER_SIZE = 0.095   # metros

# Mapa de ArUcos: {id: (x_global_m, y_global_m, yaw_rad)}
ARUCO_MAP = {
    4: (4.20,  3.70,  -math.pi / 2),
    3: (4.20,  0.0,   math.pi / 2),
    0: (0.6,   3.70,  -math.pi / 2),
    1: (0.6,   0.0,   math.pi / 2),
}

# ── Filtro de Kalman ───────────────────────────────────────────────────────
KF_Q_XY    = 0.001   # m²/paso   — confianza en la odometría
KF_Q_THETA = 0.001   # rad²/paso
KF_R_XY    = 0.1    # m²        — confianza en ArUco
KF_R_THETA = 0.1    # rad²

# ── Robot físico ────────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.0505   # m
WHEEL_BASE   = 0.183    # m  (distancia entre ruedas)

# ── Teclado ────────────────────────────────────────────────────────────────
VEL_LINEAR  = 0.15   # m/s  por pulsación W/S
VEL_ANGULAR = 0.15    # rad/s por pulsación A/D

# ── Mapa top-down ──────────────────────────────────────────────────────────
MAP_SIZE    = 650
MAP_PADDING = 60
COL_BG        = (30,  30,  30)
COL_GRID      = (55,  55,  55)
COL_ARUCO_UNK = (100, 100, 100)
COL_ARUCO_VIS = (50,  220,  50)
COL_ROBOT_ODO = (255, 180,  50)   # trayectoria odometría (naranja)
COL_ROBOT_KF  = (50,  180, 255)   # robot posición fusionada (azul)
COL_LINE      = (100, 180, 100)

# Límites de la pista
_WX_MIN, _WX_MAX = 0.0, 4.8
_WY_MIN, _WY_MAX = 0.0, 3.7
_WX_RANGE = max(_WX_MAX - _WX_MIN, 1.0)
_WY_RANGE = max(_WY_MAX - _WY_MIN, 1.0)

# ── Detector ArUco ─────────────────────────────────────────────────────────
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
try:
    det_params = aruco.DetectorParameters()
except AttributeError:
    det_params = aruco.DetectorParameters_create()


# ═══════════════════════════════════════════════════════════════════════════
#  FILTRO DE KALMAN  [x, y, θ]
# ═══════════════════════════════════════════════════════════════════════════

class PoseKalmanFilter:
    def __init__(self, q_xy, q_theta, r_xy, r_theta):
        self.Q = np.diag([q_xy, q_xy, q_theta])
        self.R = np.diag([r_xy, r_xy, r_theta])
        self.x = np.zeros(3)
        self.P = np.diag([10.0, 10.0, math.pi ** 2])
        self.initialized = False

    def predict_with_odometry(self, dx: float, dy: float, dtheta: float):
        # Ruido de proceso proporcional al desplazamiento
        dist = math.sqrt(dx ** 2 + dy ** 2)
        Q_dyn = self.Q + np.diag([
            dist * 0.0001,          # error lateral proporcional al avance
            dist * 0.0001,
            abs(dtheta) * 0.0005    # error angular proporcional al giro
        ])

        if not self.initialized:
            self.initialized = True
            # Estado inicial: origen
            self.x = np.array([dx, dy, dtheta])
            return

        # Propagar estado con el movimiento
        self.x[0] += dx
        self.x[1] += dy
        self.x[2]  = wrap_angle(self.x[2] + dtheta)
        self.P     = self.P + Q_dyn

    def update(self, z: np.ndarray):
        """Corrección con medición ArUco z = [x_m, y_m, θ_m]."""
        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return

        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        innov    = z - self.x
        innov[2] = wrap_angle(innov[2])

        self.x      = self.x + K @ innov
        self.x[2]   = wrap_angle(self.x[2])
        self.P      = (np.eye(3) - K) @ self.P

    @property
    def state(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])


# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES GEOMÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def marker_side_px(corners_px: np.ndarray) -> float:
    sides = [
        np.linalg.norm(corners_px[1] - corners_px[0]),
        np.linalg.norm(corners_px[2] - corners_px[1]),
        np.linalg.norm(corners_px[3] - corners_px[2]),
        np.linalg.norm(corners_px[0] - corners_px[3]),
    ]
    return float(np.mean(sides))

def dist_from_pixels(side_px: float) -> float:
    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    f_mean = (fx + fy) / 2.0
    if side_px < 2.0:
        return -1.0
    return f_mean * MARKER_SIZE / side_px

def estimate_robot_pose(tvec, rvec, marker_id, corners_px):
    if marker_id not in ARUCO_MAP:
        return None

    mx, my, m_yaw = ARUCO_MAP[marker_id]

    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec

    cam_x   = -t_inv[0][0]
    side_px = marker_side_px(corners_px)
    dist_px = dist_from_pixels(side_px)
    if dist_px < 0 or dist_px > 0.8:
        return None

    cam_z   = math.sqrt(max(dist_px ** 2 - cam_x ** 2, 0.0))
    dist_h  = math.sqrt(cam_x ** 2 + cam_z ** 2)
    bearing = math.atan2(cam_x, cam_z)

    yaw_cam     = math.atan2(R_inv[0, 2], R_inv[2, 2])
    robot_theta = wrap_angle(m_yaw - yaw_cam)

    robot_x = mx + cam_z * math.cos(m_yaw) - cam_x * math.sin(m_yaw)
    robot_y = my + cam_z * math.sin(m_yaw) + cam_x * math.cos(m_yaw)

    # Corrección cámara → centro del robot
    dx = 0.12
    robot_x -= dx * math.cos(robot_theta)
    robot_y -= dx * math.sin(robot_theta)

    return robot_x, robot_y, robot_theta, dist_h, bearing


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN TOP-DOWN
# ═══════════════════════════════════════════════════════════════════════════

def world_to_map(wx: float, wy: float) -> tuple:
    draw_w = MAP_SIZE - 2 * MAP_PADDING
    draw_h = MAP_SIZE - 2 * MAP_PADDING
    scale  = min(draw_w / _WX_RANGE, draw_h / _WY_RANGE)
    px = int(MAP_PADDING + (wx - _WX_MIN) * scale)
    py = int(MAP_SIZE - MAP_PADDING - (wy - _WY_MIN) * scale)
    return px, py

def draw_map(kf_x, kf_y, kf_theta,
             odo_x, odo_y, odo_theta,
             trail_kf, trail_odo,
             visible_ids, vel_lin, vel_ang) -> np.ndarray:
    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), COL_BG, dtype=np.uint8)

    # ── Grid ───────────────────────────────────────────────────────────────
    step_m = 0.6
    x = 0.0
    while x <= _WX_MAX:
        gx, _ = world_to_map(x, _WY_MIN)
        cv2.line(canvas, (gx, MAP_PADDING), (gx, MAP_SIZE - MAP_PADDING), COL_GRID, 1)
        cv2.putText(canvas, f"{x:.1f}", (gx + 2, MAP_SIZE - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        x += step_m

    y = 0.0
    while y <= _WY_MAX:
        _, gy = world_to_map(_WX_MIN, y)
        cv2.line(canvas, (MAP_PADDING, gy), (MAP_SIZE - MAP_PADDING, gy), COL_GRID, 1)
        cv2.putText(canvas, f"{y:.1f}", (5, gy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        y += step_m

    # Borde de la pista
    p0 = world_to_map(_WX_MIN, _WY_MIN)
    p1 = world_to_map(_WX_MAX, _WY_MAX)
    cv2.rectangle(canvas, p0, p1, (80, 80, 80), 1)

    # ── Trayectoria odometría ───────────────────
    for pt in trail_odo:
        px, py = world_to_map(pt[0], pt[1])
        cv2.circle(canvas, (px, py), 2, COL_ROBOT_ODO, -1)

    # ── Trayectoria filtrada  ───────────────────────────
    for pt in trail_kf:
        px, py = world_to_map(pt[0], pt[1])
        cv2.circle(canvas, (px, py), 2, COL_ROBOT_KF, -1)

    # ── ArUcos ────────────────────────────────────────────────────────────
    for mid, (mx, my, mth) in ARUCO_MAP.items():
        px, py = world_to_map(mx, my)
        color  = COL_ARUCO_VIS if mid in visible_ids else COL_ARUCO_UNK
        cv2.rectangle(canvas, (px - 10, py - 10), (px + 10, py + 10), color, -1)
        cv2.rectangle(canvas, (px - 10, py - 10), (px + 10, py + 10), (200, 200, 200), 1)
        ax = int(px + 18 * math.cos(mth))
        ay = int(py - 18 * math.sin(mth))
        cv2.arrowedLine(canvas, (px, py), (ax, ay), color, 2, tipLength=0.4)
        cv2.putText(canvas, f"ID{mid}", (px + 13, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # ── Líneas robot → ArUcos visibles ────────────────────────────────────
    rx_kf, ry_kf = world_to_map(kf_x, kf_y)
    for mid in visible_ids:
        if mid in ARUCO_MAP:
            mx, my, _ = ARUCO_MAP[mid]
            mpx, mpy  = world_to_map(mx, my)
            cv2.line(canvas, (rx_kf, ry_kf), (mpx, mpy), COL_LINE, 1, cv2.LINE_AA)

    # ── Robot odometría ────────────────────────────
    rx_o, ry_o = world_to_map(odo_x, odo_y)
    cv2.circle(canvas, (rx_o, ry_o), 7, COL_ROBOT_ODO, -1)
    fax_o = int(rx_o + 14 * math.cos(odo_theta))
    fay_o = int(ry_o - 14 * math.sin(odo_theta))
    cv2.arrowedLine(canvas, (rx_o, ry_o), (fax_o, fay_o),
                    (255, 220, 80), 1, cv2.LINE_AA, tipLength=0.4)

    # ── Robot KF ────────────────────────────────────────
    cv2.circle(canvas, (rx_kf, ry_kf), 12, COL_ROBOT_KF, -1)
    cv2.circle(canvas, (rx_kf, ry_kf), 12, (200, 230, 255), 1)
    fax_k = int(rx_kf + 22 * math.cos(kf_theta))
    fay_k = int(ry_kf - 22 * math.sin(kf_theta))
    cv2.arrowedLine(canvas, (rx_kf, ry_kf), (fax_k, fay_k),
                    (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

    # ── HUD ───────────────────────────────────────────────────────────────
    # Leyenda
    cv2.circle(canvas, (MAP_PADDING, 18), 5, COL_ROBOT_KF, -1)
    cv2.putText(canvas, "KF (fusionado)", (MAP_PADDING + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ROBOT_KF, 1)
    cv2.circle(canvas, (MAP_PADDING, 36), 5, COL_ROBOT_ODO, -1)
    cv2.putText(canvas, "Odometría", (MAP_PADDING + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ROBOT_ODO, 1)

    # Pose KF
    cv2.putText(canvas,
        f"KF:  x={kf_x:.2f}m  y={kf_y:.2f}m  th={math.degrees(kf_theta):.1f}deg",
        (10, MAP_SIZE - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_ROBOT_KF, 1)

    # Pose odometría
    cv2.putText(canvas,
        f"Odo: x={odo_x:.2f}m  y={odo_y:.2f}m  th={math.degrees(odo_theta):.1f}deg",
        (10, MAP_SIZE - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_ROBOT_ODO, 1)

    # Velocidades actuales desde teclado
    cv2.putText(canvas,
        f"v={vel_lin:+.2f}m/s  w={vel_ang:+.2f}rad/s  [W/S/A/D/SPC]",
        (10, MAP_SIZE - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  NODO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_pose')

        # ── Publishers ────────────────────────────────────────────────────
        self.image_pub   = self.create_publisher(Image, '/aruco/image_detected', 10)
        self.pose_pub    = self.create_publisher(PoseWithCovarianceStamped, '/aruco/pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # ── Subscribers ───────────────────────────────────────────────────
        qos_img = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_img)

        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_callback, qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_callback, qos.qos_profile_sensor_data)

        # ── Cámara / ArUco ────────────────────────────────────────────────
        self.camera_width  = 320
        self.camera_height = 180
        self.bridge        = CvBridge()
        self.detector      = aruco.ArucoDetector(aruco_dict, det_params)
        self.latest_frame  = None
        self.latest_header = None

        half = MARKER_SIZE / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)

        # ── Encoders ──────────────────────────────────────────────────────
        self.wr = Float32()   # velocidad angular rueda derecha  (rad/s)
        self.wl = Float32()   # velocidad angular rueda izquierda (rad/s)

        # ── Odometría pura ─────────────────
        self.odo_x     = 0.6
        self.odo_y     = 0.45
        self.odo_theta = -math.pi / 2   # orientación inicial 

        # ── Filtro de Kalman ──────────────────────────
        self.kf = PoseKalmanFilter(
            q_xy=KF_Q_XY, q_theta=KF_Q_THETA,
            r_xy=KF_R_XY, r_theta=KF_R_THETA)

        self.kf_x     = 0.0
        self.kf_y     = 0.0
        self.kf_theta = 0.0

        # ── Trayectorias para dibujar ─────────────────────────────────────
        self.trail_kf  = []   # [(x, y), ...]
        self.trail_odo = []
        self.MAX_TRAIL = 250

        # ── Teclado ───────────────────────────────────────────────────────
        self.vel_lin  = 0.0
        self.vel_ang  = 0.0
        self.running  = True

        # ── ArUcos visibles  ────────────────────────────────
        self.visible_ids = []

        # ── Timers ───────────────────────────────────────────────────────
        self.last_time_odom = self.get_clock().now()
        self.timer_odom = self.create_timer(1 / 100, self.odometria)
        self.timer_main = self.create_timer(1 / 30,  self.process_and_publish)

    # ─────────────────────────────────────────────────────────────────────
    #  CALLBACKS ENCODERS
    # ─────────────────────────────────────────────────────────────────────
    def encR_callback(self, msg: Float32):
        self.wr = msg

    def encL_callback(self, msg: Float32):
        self.wl = msg

    # ─────────────────────────────────────────────────────────────────────
    #  CALLBACK IMAGEN
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
    #  ODOMETRÍA  (timer 100 Hz)
    # ─────────────────────────────────────────────────────────────────────
    def odometria(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time_odom).nanoseconds * 1e-9
        self.last_time_odom = now

        if dt <= 0 or dt > 0.5:
            return

        vr = WHEEL_RADIUS * self.wr.data
        vl = WHEEL_RADIUS * self.wl.data

        v_avg = (vr + vl) / 2.0
        w     = (vr - vl) / WHEEL_BASE

        # Desplazamiento diferencial en el frame del robot
        dtheta = w * dt
        if abs(w) < 1e-6:
            dx_local = v_avg * dt
            dy_local = 0.0
        else:
            R_curv   = v_avg / w
            dx_local = R_curv * math.sin(dtheta)
            dy_local = R_curv * (1.0 - math.cos(dtheta))

        # Rotación al frame global
        dx = dx_local * math.cos(self.odo_theta) - dy_local * math.sin(self.odo_theta)
        dy = dx_local * math.sin(self.odo_theta) + dy_local * math.cos(self.odo_theta)

        # Integrar odometría pura
        self.odo_x     += dx
        self.odo_y     += dy
        self.odo_theta  = wrap_angle(self.odo_theta + dtheta)

        # Predicción Kalman con el desplazamiento odométrico
        self.kf.predict_with_odometry(dx, dy, dtheta)
        self.kf_x, self.kf_y, self.kf_theta = self.kf.state

        # Guardar trayectoria odometría
        self.trail_odo.append((self.odo_x, self.odo_y))
        if len(self.trail_odo) > self.MAX_TRAIL:
            self.trail_odo.pop(0)

    # ─────────────────────────────────────────────────────────────────────
    #  TECLADO
    # ─────────────────────────────────────────────────────────────────────
    def handle_keyboard(self, key: int):
        """Procesa tecla OpenCV y actualiza vel_lin / vel_ang."""
        key = key & 0xFF

        if key == ord('w'):
            self.vel_lin =  VEL_LINEAR
            self.vel_ang =  0.0
        elif key == ord('s'):
            self.vel_lin = -VEL_LINEAR
            self.vel_ang =  0.0
        elif key == ord('a'):
            self.vel_lin =  0.0
            self.vel_ang =  VEL_ANGULAR
        elif key == ord('d'):
            self.vel_lin =  0.0
            self.vel_ang = -VEL_ANGULAR
        elif key == ord(' '):
            self.vel_lin = 0.0
            self.vel_ang = 0.0
        elif key == ord('q'):
            self.running = False

    def publish_cmd_vel(self):
        cmd = Twist()
        cmd.linear.x  = self.vel_lin
        cmd.angular.z = self.vel_ang
        self.cmd_vel_pub.publish(cmd)

    # ─────────────────────────────────────────────────────────────────────
    #  CICLO PRINCIPAL: ArUco + Kalman + Mapa  (timer 30 Hz)
    # ─────────────────────────────────────────────────────────────────────
    def process_and_publish(self):
        if not self.running:
            # Detener el robot y apagar
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            raise KeyboardInterrupt

        # ── Publicar cmd_vel desde teclado ────────────────────────────────
        self.publish_cmd_vel()

        # ── Procesar imagen ───────────────────────────────────────────────
        if self.latest_frame is None:
            key = cv2.waitKey(1)
            if key != -1:
                self.handle_keyboard(key)
            return

        frame = self.latest_frame.copy()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        self.visible_ids = []
        poses_aruco = []

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
            self.visible_ids = [int(ids[m][0]) for m in range(len(ids))]

            for m_idx in range(len(corners)):
                pts       = np.squeeze(corners[m_idx])
                if pts.shape != (4, 2):
                    continue
                marker_id = int(ids[m_idx][0])

                cx_m = int(np.mean(pts[:, 0]))
                cy_m = int(np.mean(pts[:, 1]))
                cv2.circle(frame, (cx_m, cy_m), 6, (0, 255, 0), -1)

                img_pts = pts.astype(np.float64)
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, img_pts,
                    CAMERA_MATRIX, DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)

                if success:
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS,
                                      rvec, tvec, MARKER_SIZE * 0.6)
                    result = estimate_robot_pose(tvec, rvec, marker_id, pts)

                    if result is not None:
                        rx, ry, rth, dist_h, bearing = result
                        poses_aruco.append((rx, ry, rth))
                        cv2.putText(frame,
                            f"ID:{marker_id}  d={dist_h:.2f}m  b={math.degrees(bearing):.1f}°",
                            (int(pts[0][0]), int(pts[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
                        cv2.putText(frame,
                            f"KF: x={self.kf_x:.2f} y={self.kf_y:.2f} "
                            f"th={math.degrees(self.kf_theta):.1f}°",
                            (int(pts[0][0]), int(pts[0][1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 0), 1)
                    else:
                        cv2.putText(frame, f"ID:{marker_id} (no en mapa)",
                                    (cx_m - 20, cy_m - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                else:
                    cv2.putText(frame, f"ID:{marker_id} (PnP fail)",
                                (cx_m - 20, cy_m - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # ── Corrección Kalman con ArUco ───────────────────────────────
            if poses_aruco:
                xs   = [p[0] for p in poses_aruco]
                ys   = [p[1] for p in poses_aruco]
                ths  = [p[2] for p in poses_aruco]
                meas_x     = float(np.mean(xs))
                meas_y     = float(np.mean(ys))
                sin_avg    = np.mean([math.sin(t) for t in ths])
                cos_avg    = np.mean([math.cos(t) for t in ths])
                meas_theta = float(math.atan2(sin_avg, cos_avg))

                self.kf.update(np.array([meas_x, meas_y, meas_theta]))
                self.kf_x, self.kf_y, self.kf_theta = self.kf.state

                self._publish_pose()

        else:
            cv2.putText(frame, 'No ArUco', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── HUD en imagen ─────────────────────────────────────────────────
        cv2.putText(frame,
            f"KF  x={self.kf_x:.2f} y={self.kf_y:.2f} "
            f"th={math.degrees(self.kf_theta):.1f}deg",
            (5, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ROBOT_KF, 1)
        cv2.putText(frame,
            f"Odo x={self.odo_x:.2f} y={self.odo_y:.2f} "
            f"th={math.degrees(self.odo_theta):.1f}deg",
            (5, frame.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ROBOT_ODO, 1)

        # ── Publicar imagen ───────────────────────────────────────────────
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if self.latest_header is not None:
                out_msg.header = self.latest_header
            self.image_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'Error publicando imagen: {e}')

        # ── Trayectoria KF ────────────────────────────────────────────────
        self.trail_kf.append((self.kf_x, self.kf_y))
        if len(self.trail_kf) > self.MAX_TRAIL:
            self.trail_kf.pop(0)

        # ── Mapa top-down ─────────────────────────────────────────────────
        map_img = draw_map(
            self.kf_x, self.kf_y, self.kf_theta,
            self.odo_x, self.odo_y, self.odo_theta,
            list(self.trail_kf), list(self.trail_odo),
            self.visible_ids, self.vel_lin, self.vel_ang)

        # Instrucciones de teclado superpuestas en el mapa
        instructions = [
            "W: avanzar   S: retroceder",
            "A: girar izq  D: girar der",
            "SPC: detener   Q: salir",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(map_img, txt,
                        (MAP_SIZE - 230, MAP_PADDING + 16 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

        cv2.imshow("Mapa Pista", map_img)

        key = cv2.waitKey(1)
        if key != -1:
            self.handle_keyboard(key)

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLICAR POSE FUSIONADA
    # ─────────────────────────────────────────────────────────────────────
    def _publish_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x    = self.kf_x
        msg.pose.pose.position.y    = self.kf_y
        msg.pose.pose.orientation.z = math.sin(self.kf_theta / 2)
        msg.pose.pose.orientation.w = math.cos(self.kf_theta / 2)

        P = self.kf.P
        msg.pose.covariance[0]  = P[0, 0]
        msg.pose.covariance[7]  = P[1, 1]
        msg.pose.covariance[35] = P[2, 2]

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