import rclpy
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
import threading
import queue
from collections import deque
import time

import cv2
from cv2 import aruco
import numpy as np
import math

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

# Resolución de calibración (ajusta si tu calibración fue a otra resolución)
CALIB_W = 1280
CALIB_H = 720

CAMERA_MATRIX = np.array([
    [771.25742667,   0.0,         684.88203376],
    [  0.0,         773.15472704,  361.72143901],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[-4.12196743e-01,  2.39129843e-01,  9.29550695e-03,  6.35843547e-05, -7.68077937e-02]],
    dtype=np.float64
)

MARKER_SIZE    = 0.095
X_ARUCO_OFFSET = 0.608
P_MAX_XY       = 0.5    # m²
P_MAX_THETA    = 0.3    # rad²

ARUCO_MAP = {
    0:  (3.757 + X_ARUCO_OFFSET, 3.711, -math.pi/2),
    1:  (4.845 + X_ARUCO_OFFSET, 1.610,  math.pi),
    2:  (3.786 + X_ARUCO_OFFSET, 0.000,  math.pi/2),
    3:  (1.050 + X_ARUCO_OFFSET, 0.000,  math.pi/2),
    4:  (1.090 + X_ARUCO_OFFSET, 3.711, -math.pi/2),
    5:  (2.530 + X_ARUCO_OFFSET, 1.243,  math.pi),
    6:  (2.530 + X_ARUCO_OFFSET, 2.443,  math.pi),
    7:  (3.590 + X_ARUCO_OFFSET, 2.443,  0.0),
    8:  (3.590 + X_ARUCO_OFFSET, 1.243,  0.0),
    9:  (1.505 + X_ARUCO_OFFSET, 2.425,  math.pi/2),
    10: (1.505 + X_ARUCO_OFFSET, 1.365, -math.pi/2),
}

KF_Q_XY    = 0.001
KF_Q_THETA = 0.001
KF_R_XY    = 0.1
KF_R_THETA = 0.1

MAX_ARUCO_DIST = 1.5

# ── Ventana de presencia para estabilidad de detección ───────────────────
# Se requieren MIN_DETECTIONS frames con ArUco visible dentro de
# una ventana deslizante de PRESENCE_WINDOW para aceptar una corrección.
# Esto reemplaza el contador puntual same_id_count y es tolerante a
# frames perdidos por blur / oclusión parcial.
PRESENCE_WINDOW     = 6   # frames de la ventana deslizante (~0.4 s a 15 Hz)
MIN_DETECTIONS      = 4   # mínimo de frames con detección (67 %) — tolera 2 perdidos
MIN_HIST_FOR_UPDATE = 4   # mínimo de poses en historial para promediar

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
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
        dist = math.sqrt(dx**2 + dy**2)

        if not self.initialized:
            self.initialized = True
            self.x = np.array([dx, dy, dtheta])
            return

        self.x[0] += dx
        self.x[1] += dy
        self.x[2]  = wrap_angle(self.x[2] + dtheta)

        Q_dyn = np.diag([
            dist * 0.05 + abs(dtheta) * 0.005 + 1e-5,
            dist * 0.05 + abs(dtheta) * 0.005 + 1e-5,
            abs(dtheta) * 0.05 + dist * 0.005 + 1e-5,
        ])
        self.P = self.P + Q_dyn

        self.P[0, 0] = min(self.P[0, 0], P_MAX_XY)
        self.P[1, 1] = min(self.P[1, 1], P_MAX_XY)
        self.P[2, 2] = min(self.P[2, 2], P_MAX_THETA)

    def update(self, z: np.ndarray):
        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return

        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        innov    = z - self.x
        innov[2] = wrap_angle(innov[2])

        self.x    = self.x + K @ innov
        self.x[2] = wrap_angle(self.x[2])
        self.P    = (np.eye(3) - K) @ self.P

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

def estimate_robot_pose(tvec, rvec, marker_id):
    if marker_id not in ARUCO_MAP:
        return None

    mx, my, m_yaw = ARUCO_MAP[marker_id]

    R, _ = cv2.Rodrigues(rvec)
    cam_marker = -R.T @ tvec

    cam_x_marker = float(cam_marker[0, 0])
    cam_z_marker = float(cam_marker[2, 0])

    dist_h = math.sqrt(cam_x_marker**2 + cam_z_marker**2)

    marker_normal = R[:, 2]
    cos_view = abs(marker_normal[2])
    cos_view = np.clip(cos_view, -1.0, 1.0)
    view_angle = math.degrees(math.acos(cos_view))

    if view_angle > 45.0:
        return None

    yaw_cam = math.atan2(R[0, 2], R[2, 2])
    robot_theta = wrap_angle(m_yaw + math.pi - yaw_cam)

    robot_x = (
        mx
        + cam_z_marker * math.cos(m_yaw)
        - cam_x_marker * math.sin(m_yaw)
    )
    robot_y = (
        my
        + cam_z_marker * math.sin(m_yaw)
        + cam_x_marker * math.cos(m_yaw)
    )

    CAMERA_TO_CENTER = 0.10
    robot_x -= CAMERA_TO_CENTER * math.cos(robot_theta)
    robot_y -= CAMERA_TO_CENTER * math.sin(robot_theta)

    bearing = math.atan2(cam_x_marker, cam_z_marker)

    return (robot_x, robot_y, robot_theta, dist_h, bearing, view_angle)


# ═══════════════════════════════════════════════════════════════════════════
#  NODO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_pose')

        # ── Publishers ────────────────────────────────────────────────────
        self.image_pub         = self.create_publisher(
            CompressedImage, '/aruco/image_detected/compressed', 10)
        self.pose_centered_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/aruco/pose_centered', 10)

        self._annotated_queue = queue.Queue(maxsize=2)
        self._pub_thread = threading.Thread(target=self._image_pub_loop, daemon=True)
        self._pub_thread.start()

        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.aruco_markers_pub = self.create_publisher(MarkerArray, '/aruco/markers', latched_qos)

        # ── Locks y estado compartido ─────────────────────────────────────
        self._kf_lock    = threading.Lock()
        self._frame_lock = threading.Lock()
        self._vis_lock   = threading.Lock()

        # ── Odometría ────────────────────────────────────────────────────
        self.last_odom_x     = None
        self.last_odom_y     = None
        self.last_odom_theta = None
        self.odo_x = self.odo_y = self.odo_theta = 0.0

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Subscriptions ────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback,
            qos.qos_profile_sensor_data)
        self.mcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/mcl/pose',
            self._mcl_pose_callback, 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/video_source/compressed', self.image_callback, image_qos)

        # ── Cámara / ArUco ────────────────────────────────────────────────
        self.detector          = aruco.ArucoDetector(aruco_dict, det_params)
        self.latest_frame      = None
        self.latest_full_frame = None
        self.latest_header     = None
        self.latest_frame_time = None
        self.latest_raw_jpeg   = None
        self._new_raw_evt      = threading.Event()

        self.aruco_history  = {}   # {id: deque de (x, y, theta)}

        # ── FIX 2: ventana de presencia por ID ────────────────────────────
        # Reemplaza same_id_count. Tolera frames perdidos.
        self.aruco_presence = {}   # {id: deque de bool}

        half = MARKER_SIZE / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)

        # ── Kalman ───────────────────────────────────────────────────────
        self.kf       = PoseKalmanFilter(KF_Q_XY, KF_Q_THETA, KF_R_XY, KF_R_THETA)
        self.kf_x     = 0.0
        self.kf_y     = 0.0
        self.kf_theta = 0.0

        self.MAP_OFFSET_X = 2.825
        self.MAP_OFFSET_Y = 1.925

        # ── ArUcos visibles ──────────────────────────────────────────────
        self._visible_ids: list = []

        self._publish_aruco_markers()

        self._processing_thread = threading.Thread(
            target=self._detection_loop, daemon=True)
        self._processing_thread.start()

        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True)
        self._decode_thread.start()

        self.timer_main = self.create_timer(1 / 30, self._publish_only)

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLICADOR DE IMAGEN (hilo dedicado)
    # ─────────────────────────────────────────────────────────────────────
    def _image_pub_loop(self):
        last_pub      = 0.0
        MIN_INTERVAL  = 1.0 / 15.0
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 65]

        while rclpy.ok():
            try:
                annotated, header = self._annotated_queue.get(timeout=0.5)
                while not self._annotated_queue.empty():
                    try:
                        annotated, header = self._annotated_queue.get_nowait()
                    except queue.Empty:
                        break

                now = time.monotonic()
                if now - last_pub < MIN_INTERVAL:
                    continue
                last_pub = now

                pub = cv2.resize(annotated, (640, 360),
                                 interpolation=cv2.INTER_LINEAR)
                ok, buf = cv2.imencode('.jpg', pub, encode_params)
                if not ok:
                    continue

                out        = CompressedImage()
                out.header = header if header is not None else Header()
                out.format = 'jpeg'
                out.data   = buf.tobytes()
                self.image_pub.publish(out)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────────
    def _odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        if self.last_odom_x is None:
            self.last_odom_x     = x
            self.last_odom_y     = y
            self.last_odom_theta = theta
            self.odo_x = x;  self.odo_y = y;  self.odo_theta = theta
            return

        dx_odom = x - self.last_odom_x
        dy_odom = y - self.last_odom_y
        dtheta  = wrap_angle(theta - self.last_odom_theta)

        self.last_odom_x = x;  self.last_odom_y = y;  self.last_odom_theta = theta
        self.odo_x = x;        self.odo_y = y;         self.odo_theta = theta

        theta_before     = wrap_angle(theta - dtheta)
        dist             = math.hypot(dx_odom, dy_odom)
        move_angle_odom  = math.atan2(dy_odom, dx_odom)
        move_angle_local = wrap_angle(move_angle_odom - theta_before)

        with self._kf_lock:
            map_theta_before = self.kf.x[2]
            global_angle     = map_theta_before + move_angle_local + math.pi
            dx_map = dist * math.cos(global_angle)
            dy_map = dist * math.sin(global_angle)

            self.kf.predict_with_odometry(dx_map, dy_map, dtheta)
            self.kf_x, self.kf_y, self.kf_theta = self.kf.state

    def _mcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        if not self.kf.initialized:
            return

        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        r_xx = max(msg.pose.covariance[0],  1e-6)
        r_yy = max(msg.pose.covariance[7],  1e-6)
        r_tt = max(msg.pose.covariance[35], 1e-6)

        with self._kf_lock:
            old_R    = self.kf.R.copy()
            self.kf.R = np.diag([r_xx, r_yy, r_tt])
            self.kf.update(np.array([ox, oy, theta]))
            self.kf.R = old_R
            self.kf_x, self.kf_y, self.kf_theta = self.kf.state

        self.get_logger().debug(
            f'[MCL→KF] x={self.kf_x:.2f} y={self.kf_y:.2f} '
            f'th={math.degrees(self.kf_theta):.1f}°')

    def image_callback(self, msg: CompressedImage):
        with self._frame_lock:
            self.latest_frame    = None
            self.latest_raw_jpeg = bytes(msg.data)
            self.latest_header   = msg.header
            self.latest_frame_time = self.get_clock().now()
        self._new_raw_evt.set()

    # ─────────────────────────────────────────────────────────────────────
    #  HILO DECODE
    # ─────────────────────────────────────────────────────────────────────
    def _decode_loop(self):
        last_stamp_ns = -1

        while rclpy.ok():
            self._new_raw_evt.wait(timeout=0.2)
            self._new_raw_evt.clear()

            with self._frame_lock:
                raw   = getattr(self, 'latest_raw_jpeg', None)
                stamp = self.latest_frame_time

            if raw is None or stamp is None:
                continue

            stamp_ns = stamp.nanoseconds
            if stamp_ns == last_stamp_ns:
                continue

            np_arr     = np.frombuffer(raw, np.uint8)
            full_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if full_frame is None:
                continue

            # ── FIX 1: resize al tamaño de calibración para detección ─────
            # detection_loop recibe el frame ya en CALIB_W×CALIB_H.
            # solvePnP usará los corners directamente con CAMERA_MATRIX
            # que fue calibrada a esa misma resolución → sin error de escala.
            det_frame = cv2.resize(full_frame, (CALIB_W, CALIB_H),
                                   interpolation=cv2.INTER_LINEAR)

            with self._frame_lock:
                if self.latest_frame_time is not None and \
                   self.latest_frame_time.nanoseconds == stamp_ns:
                    self.latest_frame      = det_frame   # ← resolución de calibración
                    self.latest_full_frame = full_frame  # ← para anotación
            last_stamp_ns = stamp_ns

    # ─────────────────────────────────────────────────────────────────────
    #  HELPERS DE CALIDAD DE MARCADOR
    # ─────────────────────────────────────────────────────────────────────
    def _is_marker_usable(self, pts: np.ndarray) -> bool:
        v1   = pts[1] - pts[0]
        v2   = pts[3] - pts[0]
        area = abs(v1[0]*v2[1] - v1[1]*v2[0])

        top    = np.linalg.norm(pts[1] - pts[0])
        bottom = np.linalg.norm(pts[2] - pts[3])
        left   = np.linalg.norm(pts[3] - pts[0])
        right  = np.linalg.norm(pts[2] - pts[1])
        w_avg  = (top + bottom) / 2.0
        h_avg  = (left + right) / 2.0

        if w_avg < 1e-3 or h_avg < 1e-3:
            return False

        aspect = min(w_avg, h_avg) / max(w_avg, h_avg)
        return aspect >= 0.40 and area >= 400

    # ─────────────────────────────────────────────────────────────────────
    #  HILO DE DETECCIÓN ArUco (15 Hz)
    # ─────────────────────────────────────────────────────────────────────
    def _detection_loop(self):
        PERIOD = 1.0 / 15.0

        while rclpy.ok():
            t0 = time.monotonic()

            with self._frame_lock:
                frame      = self.latest_frame        # CALIB_W×CALIB_H
                full_frame = getattr(self, 'latest_full_frame', None)
                header     = self.latest_header
                frame_time = self.latest_frame_time

            if frame is None:
                time.sleep(PERIOD)
                continue

            if frame_time is not None:
                age = (self.get_clock().now() - frame_time).nanoseconds * 1e-9
                if age > 0.12:
                    _sleep_remainder(t0, PERIOD)
                    continue

            # ── Detección sobre frame de resolución de calibración ────────
            # Sin ROI: detectamos en el frame completo para no perder
            # marcadores en los bordes y evitar el offset de coordenadas.
            gray                    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _         = self.detector.detectMarkers(gray)

            # ── Escala de corners para anotación visual ───────────────────
            # Sólo para dibujar sobre full_frame (resolución nativa).
            # Los corners originales (coords CALIB_W×CALIB_H) se usan para solvePnP.
            h_full, w_full = full_frame.shape[:2] if full_frame is not None else (CALIB_H, CALIB_W)
            scale_x = w_full / CALIB_W
            scale_y = h_full / CALIB_H

            has_detections = ids is not None and len(ids) > 0

            # ── Actualizar ventana de presencia para TODOS los IDs ────────
            detected_ids_this_frame = set()
            if has_detections:
                for _id in ids.flatten():
                    detected_ids_this_frame.add(int(_id))

            # Añadir tick True/False a cada ID conocido
            all_tracked_ids = set(self.aruco_presence.keys()) | detected_ids_this_frame
            for mid in all_tracked_ids:
                if mid not in self.aruco_presence:
                    self.aruco_presence[mid] = deque(maxlen=PRESENCE_WINDOW)
                self.aruco_presence[mid].append(mid in detected_ids_this_frame)

            if not has_detections:
                with self._vis_lock:
                    self._visible_ids = []
                try:
                    pub = cv2.resize(frame, (640, 360))
                    cv2.putText(pub, 'No ArUco', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self._annotated_queue.full():
                        self._annotated_queue.get_nowait()
                    self._annotated_queue.put_nowait((pub, header))
                except (queue.Empty, queue.Full):
                    pass
                _sleep_remainder(t0, PERIOD)
                continue

            # ── Anotar sobre full_frame ───────────────────────────────────
            if full_frame is not None:
                annotated = full_frame.copy()
            else:
                annotated = frame.copy()

            # Corners escalados solo para dibujar
            corners_vis = [
                (c * np.array([scale_x, scale_y], dtype=np.float32))
                for c in corners
            ]
            aruco.drawDetectedMarkers(annotated, corners_vis, ids)

            new_visible = []
            best_aruco  = None
            best_score  = float('inf')

            for m_idx in range(len(corners)):
                # ── FIX 1: solvePnP usa corners en coords de calibración ──
                pts_calib = np.squeeze(corners[m_idx])   # CALIB_W×CALIB_H
                if pts_calib.shape != (4, 2):
                    continue

                marker_id = int(ids[m_idx][0])
                new_visible.append(marker_id)

                # Posición del centro en el frame anotado (para texto)
                pts_vis = np.squeeze(corners_vis[m_idx])
                cx_vis  = int(np.mean(pts_vis[:, 0]))
                cy_vis  = int(np.mean(pts_vis[:, 1]))

                if not self._is_marker_usable(pts_calib):
                    cv2.putText(annotated, f'ID:{marker_id} (skipped)',
                        (cx_vis - 20, cy_vis - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (128, 128, 0), 1)
                    continue

                cv2.circle(annotated, (cx_vis, cy_vis), 6, (0, 255, 0), -1)

                img_pts = pts_calib.astype(np.float64)
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, img_pts,
                    CAMERA_MATRIX, DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)

                if success:
                    result = estimate_robot_pose(tvec, rvec, marker_id)

                    if result is not None:
                        rx, ry, rth, dist_h, _, view_angle = result
                        if dist_h > MAX_ARUCO_DIST:
                            continue

                        # ── FIX 3: R proporcional a distancia y ángulo ────
                        # r_scale bajo = confiamos más en la medida.
                        # Aumenta con distancia y con ángulo de visión oblicua.
                        r_scale = (dist_h / MAX_ARUCO_DIST) ** 2 \
                                  + (view_angle / 45.0) ** 2 * 0.5
                        r_scale = max(0.2, min(r_scale, 3.0))
                        score   = dist_h * (1.0 + view_angle / 45.0)

                        if score < best_score:
                            best_score = score
                            best_aruco = (rx, ry, rth, r_scale, marker_id, dist_h)

                        self.get_logger().info(
                            f'[ArUco {marker_id}] robot=({rx:.2f}, {ry:.2f}) '
                            f'th={math.degrees(rth):.1f}° dist={dist_h:.2f}m '
                            f'view={view_angle:.1f}° r_scale={r_scale:.2f} '
                            f'kf=({self.kf_x:.2f}, {self.kf_y:.2f})')

                        cv2.putText(annotated,
                            f'ID:{marker_id}  d={dist_h:.2f}m  ang={view_angle:.0f}deg',
                            (int(pts_vis[0][0]), int(pts_vis[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
                    else:
                        cv2.putText(annotated, f'ID:{marker_id} (no en mapa)',
                            (cx_vis - 20, cy_vis - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            with self._vis_lock:
                self._visible_ids = new_visible

            # ── FIX 2: actualizar historial y decidir corrección KF ───────
            if best_aruco is not None:
                rx, ry, rth, r_scale, used_id, dist_h = best_aruco

                if used_id not in self.aruco_history:
                    self.aruco_history[used_id] = deque(maxlen=10)

                self.aruco_history[used_id].append((rx, ry, rth))

                presence = self.aruco_presence.get(used_id, deque())
                detections_in_window = sum(presence)

                self.get_logger().info(
                    f'[ARUCO] ID:{used_id} '
                    f'presencia={detections_in_window}/{len(presence)} '
                    f'(min={MIN_DETECTIONS})'
                )

                if (
                    detections_in_window >= MIN_DETECTIONS
                    and len(self.aruco_history[used_id]) >= MIN_HIST_FOR_UPDATE
                ):
                    hist = self.aruco_history[used_id]

                    mean_x   = np.mean([p[0] for p in hist])
                    mean_y   = np.mean([p[1] for p in hist])
                    mean_sin = np.mean([math.sin(p[2]) for p in hist])
                    mean_cos = np.mean([math.cos(p[2]) for p in hist])
                    mean_th  = math.atan2(mean_sin, mean_cos)

                    std_x = np.std([p[0] for p in hist])
                    std_y = np.std([p[1] for p in hist])

                    if std_x < 0.05 and std_y < 0.05:
                        self.get_logger().info(
                            f'[KF UPDATE] ID:{used_id} '
                            f'pos=({mean_x:.2f},{mean_y:.2f}) '
                            f'r_scale={r_scale:.2f}'
                        )

                        with self._kf_lock:
                            old_R = self.kf.R.copy()
                            # FIX 3: R bajo → confiamos más; R alto → medida ruidosa
                            self.kf.R = np.diag([
                                KF_R_XY    * r_scale,
                                KF_R_XY    * r_scale,
                                KF_R_THETA * r_scale,
                            ])
                            self.kf.update(np.array([mean_x, mean_y, mean_th]))
                            self.kf.R = old_R
                            self.kf_x, self.kf_y, self.kf_theta = self.kf.state

            try:
                if self._annotated_queue.full():
                    self._annotated_queue.get_nowait()
                self._annotated_queue.put_nowait((annotated, header))
            except (queue.Empty, queue.Full):
                pass

            _sleep_remainder(t0, PERIOD)

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLICAR MARKERS ARUCO EN RVIZ
    # ─────────────────────────────────────────────────────────────────────
    def _publish_aruco_markers(self):
        ma    = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        with self._vis_lock:
            vis = set(self._visible_ids)

        for marker_id, (mx, my, m_yaw) in ARUCO_MAP.items():
            cx = mx - self.MAP_OFFSET_X
            cy = my - self.MAP_OFFSET_Y

            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = 'odom'
            m.ns     = 'aruco'
            m.id     = marker_id
            m.type   = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x    = cx
            m.pose.position.y    = cy
            m.pose.position.z    = 0.1
            m.pose.orientation.z = math.sin(m_yaw / 2)
            m.pose.orientation.w = math.cos(m_yaw / 2)

            m.scale.x = 0.05;  m.scale.y = 0.05;  m.scale.z = 0.20

            if marker_id in vis:
                m.color = ColorRGBA(r=0.2, g=0.9, b=0.2, a=1.0)
            else:
                m.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.6)

            ma.markers.append(m)

            t = Marker()
            t.header             = m.header
            t.ns                 = 'aruco_labels'
            t.id                 = marker_id + 100
            t.type               = Marker.TEXT_VIEW_FACING
            t.action             = Marker.ADD
            t.pose.position.x    = cx
            t.pose.position.y    = cy
            t.pose.position.z    = 0.35
            t.pose.orientation.w = 1.0
            t.scale.z            = 0.12
            t.color              = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text               = f'ID{marker_id}'
            ma.markers.append(t)

        self.aruco_markers_pub.publish(ma)

    # ─────────────────────────────────────────────────────────────────────
    #  TIMER 30 Hz
    # ─────────────────────────────────────────────────────────────────────
    def _publish_only(self):
        self._publish_pose()

    def _publish_pose(self):
        with self._kf_lock:
            kf_x     = self.kf_x
            kf_y     = self.kf_y
            kf_theta = self.kf_theta
            p00      = self.kf.P[0, 0]
            p11      = self.kf.P[1, 1]
            p35      = self.kf.P[2, 2]

        stamp = self.get_clock().now().to_msg()

        cov     = [0.0] * 36
        cov[0]  = p00
        cov[7]  = p11
        cov[35] = p35

        msg_c = PoseWithCovarianceStamped()
        msg_c.header.stamp    = stamp
        msg_c.header.frame_id = 'map'
        msg_c.pose.pose.position.x = -(kf_x - self.MAP_OFFSET_X)
        msg_c.pose.pose.position.y = -(kf_y - self.MAP_OFFSET_Y)

        theta_pub = kf_theta
        msg_c.pose.pose.orientation.z = math.sin(theta_pub / 2)
        msg_c.pose.pose.orientation.w = math.cos(theta_pub / 2)
        msg_c.pose.covariance         = cov

        self.pose_centered_pub.publish(msg_c)


# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDAD: dormir el tiempo restante del período
# ═══════════════════════════════════════════════════════════════════════════

def _sleep_remainder(t_start: float, period: float):
    remaining = period - (time.monotonic() - t_start)
    if remaining > 0.002:
        time.sleep(remaining)


# ═══════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()