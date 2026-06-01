import rclpy
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import threading
import queue
import time

import cv2
from cv2 import aruco
import numpy as np
import math

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

# ── Calibración de la cámara ───────────────────────────────────────────────
CAMERA_MATRIX = np.array([
    [771.25742667,   0.0,         684.88203376],
    [  0.0,         773.15472704,  361.72143901],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[-4.12196743e-01,  2.39129843e-01,  9.29550695e-03,  6.35843547e-05,  -7.68077937e-02]],
    dtype=np.float64
)

# ── ArUco ─────────────────────────────────────────────────────────────────
MARKER_SIZE = 0.098   # metros

X_ARUCO_OFFSET = 0.608
# Mapa de ArUcos: {id: (x_global_m, y_global_m, yaw_rad)}

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

# ── Filtro de Kalman ───────────────────────────────────────────────────────
KF_Q_XY    = 0.001   # m²/paso   — confianza en la odometría
KF_Q_THETA = 0.001   # rad²/paso
KF_R_XY    = 0.1    # m²        — confianza en ArUco
KF_R_THETA = 0.1    # rad²

# ── Detector ArUco ─────────────────────────────────────────────────────────
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
    if dist_px < 0: 
        return None

    cam_z   = math.sqrt(max(dist_px ** 2 - cam_x ** 2, 0.0))
    dist_h  = math.sqrt(cam_x ** 2 + cam_z ** 2)
    bearing = math.atan2(cam_x, cam_z)

    yaw_cam     = math.atan2(R_inv[0, 2], R_inv[2, 2])
    robot_theta = wrap_angle(m_yaw - yaw_cam)

    robot_x = mx + cam_z * math.cos(m_yaw) - cam_x * math.sin(m_yaw)
    robot_y = my + cam_z * math.sin(m_yaw) + cam_x * math.cos(m_yaw)

    # Corrección cámara → centro del robot
    dx = 0.1
    robot_x -= dx * math.cos(robot_theta)
    robot_y -= dx * math.sin(robot_theta)

    return robot_x, robot_y, robot_theta, dist_h, bearing


# ═══════════════════════════════════════════════════════════════════════════
#  NODO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_pose')

        # ── Publishers ────────────────────────────────────────────────────
        self.image_pub   = self.create_publisher(CompressedImage, '/aruco/image_detected/compressed', 10)
        self.pose_centered_pub = self.create_publisher(PoseWithCovarianceStamped, '/aruco/pose_centered', 10)

        self._annotated_queue = queue.Queue(maxsize=2)  # máximo 2 frames pendientes

        self._pub_thread = threading.Thread(target=self._image_pub_loop, daemon=True)
        self._pub_thread.start()

        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.aruco_markers_pub = self.create_publisher(
            MarkerArray, '/aruco/markers', latched_qos)
        
        self._kf_lock = threading.Lock()

        self.last_odom_x     = None   # None indica que aún no recibimos el primer mensaje
        self.last_odom_y     = None
        self.last_odom_theta = None

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback,
            qos.qos_profile_sensor_data)
        
        self.mcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/mcl/pose',
            self._mcl_pose_callback, 10)

        self.image_sub = self.create_subscription(
            CompressedImage, '/video_source/compressed', self.image_callback, 10)


        # ── Cámara / ArUco ────────────────────────────────────────────────
        self.camera_width  = 1280
        self.camera_height = 720
        self.detector      = aruco.ArucoDetector(aruco_dict, det_params)
        self.latest_frame  = None
        self.latest_header = None
        self.ROI_MARGIN = 0.20 # margen para recortar bordes (20% de la imagen)
        self.latest_frame_time = None

        half = MARKER_SIZE / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)


        # ── Odometría pura ─────────────────
        self.odo_x     = 0.0
        self.odo_y     = 0.0
        self.odo_theta = 0.0

        # ── Filtro de Kalman ──────────────────────────
        self.kf = PoseKalmanFilter(
            q_xy=KF_Q_XY, q_theta=KF_Q_THETA,
            r_xy=KF_R_XY, r_theta=KF_R_THETA)

        self.kf_x     = 0.0
        self.kf_y     = 0.0
        self.kf_theta = 0.0

        self.MAP_OFFSET_X = 2.825
        self.MAP_OFFSET_Y = 1.925

        # ── ArUcos visibles  ────────────────────────────────
        self.visible_ids = []

        self._publish_aruco_markers()

        # ── Timers ───────────────────────────────────────────────────────
        self._frame_lock = threading.Lock()
        self._latest_gray = None
        self._processing_thread = threading.Thread(
            target=self._detection_loop, daemon=True)
        self._processing_thread.start()

        self.timer_main = self.create_timer(1 / 30,  self._publish_only)

    # ── 1. Throttle explícito al publicador de imagen ──────────────────────
    def _image_pub_loop(self):
        last_pub = 0.0
        MIN_INTERVAL = 1 / 15.0

        while rclpy.ok():
            try:
                # Drenar la cola — solo nos interesa el frame más reciente
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

                out = CompressedImage()
                out.header = header if header is not None else Header()
                out.format = 'jpeg'
                ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ok:
                    out.data = buf.tobytes()
                    self.image_pub.publish(out)
            except Exception:
                pass

    def _odom_callback(self, msg: Odometry):
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        q     = msg.pose.pose.orientation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Primer mensaje: solo guardar referencia, no predecir
        if self.last_odom_x is None:
            self.last_odom_x     = x
            self.last_odom_y     = y
            self.last_odom_theta = theta
            # Inicializar odometría visual con la pose inicial de /odom
            self.odo_x     = x
            self.odo_y     = y
            self.odo_theta = theta
            return

        # Delta de pose entre mensajes consecutivos
        dx     = x - self.last_odom_x
        dy     = y - self.last_odom_y
        dtheta = wrap_angle(theta - self.last_odom_theta)

        self.last_odom_x     = x
        self.last_odom_y     = y
        self.last_odom_theta = theta

        # Actualizar odometría visual (para el mapa top-down)
        self.odo_x     = x
        self.odo_y     = y
        self.odo_theta = theta

        # Predicción Kalman con el delta odométrico
        with self._kf_lock:
            self.kf.predict_with_odometry(dx, dy, dtheta)
            self.kf_x, self.kf_y, self.kf_theta = self.kf.state

    def _mcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        Corrección Kalman con pose MCL.
        Solo se llama cuando ArUco no está visible (lo filtra mcl_localizer).
        """
        if not self.kf.initialized:
            return

        ox    = msg.pose.pose.position.x
        oy    = msg.pose.pose.position.y
        q     = msg.pose.pose.orientation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Usar covarianza del mensaje MCL para ajustar la ganancia
        # (MCL es menos preciso que ArUco, así que R es mayor)

        with self._kf_lock:
            old_R = self.kf.R.copy()
            self.kf.R = np.diag([
                msg.pose.covariance[0],    # MCL_COV_XY
                msg.pose.covariance[7],    # MCL_COV_XY
                msg.pose.covariance[35]    # MCL_COV_THETA
            ])
            self.kf.update(np.array([ox, oy, theta]))
            self.kf.R = old_R
            self.kf_x, self.kf_y, self.kf_theta = self.kf.state
            
        self.get_logger().debug(
            f'[MCL→KF] x={self.kf_x:.2f} y={self.kf_y:.2f} '
            f'th={math.degrees(self.kf_theta):.1f}°')

    # ─────────────────────────────────────────────────────────────────────
    #  CALLBACK IMAGEN
    # ─────────────────────────────────────────────────────────────────────
    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with self._frame_lock:
                self.latest_frame  = frame
                self.latest_header = msg.header
                self.latest_frame_time = self.get_clock().now()  # ── NUEVO
        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')

    # ─────────────────────────────────────────────────────────────────────
    #  CICLO PRINCIPAL: ArUco + Kalman + Mapa  (timer 30 Hz)
    # ─────────────────────────────────────────────────────────────────────

    def _get_roi(self, frame: np.ndarray):
        """
        Devuelve (roi_frame, ox, oy) donde ox,oy es el offset del ROI
        en el frame original, para poder convertir coordenadas de vuelta.
        """
        h, w = frame.shape[:2]
        mx = int(w * self.ROI_MARGIN)
        my = int(h * self.ROI_MARGIN)
        roi = frame[my:h-my, mx:w-mx]
        return roi, mx, my

    def _is_marker_usable(self, pts: np.ndarray) -> bool:
        """
        Rechaza marcadores muy oblicuos o muy pequeños.
        pts: (4,2) — esquinas TL, TR, BR, BL
        """
        # Área del marcador en px² (cross product)
        v1 = pts[1] - pts[0]
        v2 = pts[3] - pts[0]
        area = abs(v1[0]*v2[1] - v1[1]*v2[0])
        
        # Ratio de aspecto: si está muy aplastado, está muy de lado
        top    = np.linalg.norm(pts[1] - pts[0])
        bottom = np.linalg.norm(pts[2] - pts[3])
        left   = np.linalg.norm(pts[3] - pts[0])
        right  = np.linalg.norm(pts[2] - pts[1])
        
        w_avg = (top + bottom) / 2.0
        h_avg = (left + right) / 2.0
        
        if w_avg < 1e-3 or h_avg < 1e-3:
            return False
        
        aspect_ratio = min(w_avg, h_avg) / max(w_avg, h_avg)
        
        # Rechazar si está más de ~55° de lado (aspect < 0.57 ≈ cos(55°))
        MIN_ASPECT  = 0.40   # ajustable — más alto = más estricto
        MIN_AREA_PX = 400    # píxeles cuadrados mínimos
        
        return aspect_ratio >= MIN_ASPECT and area >= MIN_AREA_PX
    
    def _marker_quality_weight(self, pts: np.ndarray) -> float:
        """
        Devuelve un multiplicador para R: 1.0 = perfecto, >1 = más incertidumbre.
        Penaliza marcadores oblicuos.
        """
        top   = np.linalg.norm(pts[1] - pts[0])
        left  = np.linalg.norm(pts[3] - pts[0])
        w_avg = (top + np.linalg.norm(pts[2] - pts[3])) / 2.0
        h_avg = (left + np.linalg.norm(pts[2] - pts[1])) / 2.0
        
        if w_avg < 1e-3 or h_avg < 1e-3:
            return 10.0
        
        aspect = min(w_avg, h_avg) / max(w_avg, h_avg)
        # aspect=1.0 → peso 1x,  aspect=0.4 → peso ~6x
        return 1.0 / max(aspect ** 2, 0.04)

    def _detection_loop(self):
        """Hilo dedicado a detección ArUco — no bloquea ROS."""
        last_process_time = 0.0
        DETECT_SCALE = 0.5   # detectar a 640×360 en vez de 1280×720

        while rclpy.ok():
            # ── Throttle: procesar máximo a 15 Hz ──────────────────────────
            now = time.monotonic()
            elapsed = now - last_process_time
            if elapsed < 0.066:
                time.sleep(0.066 - elapsed)
                continue
            last_process_time = time.monotonic()

            # ── Tomar siempre el frame más reciente ─────────────────────────
            with self._frame_lock:
                frame      = self.latest_frame
                header     = self.latest_header
                frame_time = self.latest_frame_time

            if frame is None:
                continue

            # ── Descartar frame viejo ────────────────────────────────────────
            if frame_time is not None:
                age = (self.get_clock().now() - frame_time).nanoseconds * 1e-9
                if age > 0.12:
                    continue

            # ── Detección a resolución reducida ─────────────────────────────
            frame_small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                                    interpolation=cv2.INTER_LINEAR)
            gray_small  = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray_roi, roi_ox, roi_oy = self._get_roi(gray_small)
            corners, ids, _ = self.detector.detectMarkers(gray_roi)

            # ── Escalar corners + offset de vuelta a resolución original ────
            if ids is not None and corners:
                offset = np.array([roi_ox, roi_oy], dtype=np.float32)
                corners = [(c + offset) / DETECT_SCALE for c in corners]

            # ── Copiar frame original SOLO si hay detecciones o hace falta anotar
            annotated = frame.copy()

            self.visible_ids  = []
            poses_aruco       = []

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(annotated, corners, ids)
                self.visible_ids = [int(ids[m][0]) for m in range(len(ids))]

                for m_idx in range(len(corners)):
                    pts = np.squeeze(corners[m_idx])
                    if pts.shape != (4, 2):
                        continue

                    marker_id = int(ids[m_idx][0])
                    cx_m = int(np.mean(pts[:, 0]))
                    cy_m = int(np.mean(pts[:, 1]))

                    if not self._is_marker_usable(pts):
                        continue

                    cv2.circle(annotated, (cx_m, cy_m), 6, (0, 255, 0), -1)

                    img_pts = pts.astype(np.float64)
                    success, rvec, tvec = cv2.solvePnP(
                        self.obj_points, img_pts,
                        CAMERA_MATRIX, DIST_COEFFS,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)

                    if success:
                        result = estimate_robot_pose(tvec, rvec, marker_id, pts)
                        if result is not None:
                            rx, ry, rth, dist_h, bearing = result
                            quality = self._marker_quality_weight(pts)
                            poses_aruco.append((rx, ry, rth, quality))
                            cv2.putText(annotated,
                                f"ID:{marker_id}  d={dist_h:.2f}m",
                                (int(pts[0][0]), int(pts[0][1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
                        else:
                            cv2.putText(annotated, f"ID:{marker_id} (no en mapa)",
                                (cx_m - 20, cy_m - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                    else:
                        cv2.putText(annotated, f"ID:{marker_id} (PnP fail)",
                            (cx_m - 20, cy_m - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                if poses_aruco:
                    best = min(poses_aruco,
                            key=lambda p: math.hypot(p[0] - self.kf_x, p[1] - self.kf_y)
                                            * p[3])
                    with self._kf_lock:
                        old_R = self.kf.R.copy()
                        weight = best[3]
                        self.kf.R = np.diag([KF_R_XY*weight, KF_R_XY*weight, KF_R_THETA*weight])
                        self.kf.update(np.array([best[0], best[1], best[2]]))
                        self.kf.R = old_R
                        self.kf_x, self.kf_y, self.kf_theta = self.kf.state

            else:
                cv2.putText(annotated, 'No ArUco', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ── Encolar frame anotado (resolución reducida para ahorrar ancho de banda)
            try:
                pub_frame = cv2.resize(annotated, (640, 360),
                                    interpolation=cv2.INTER_LINEAR)
                self._annotated_queue.put_nowait((pub_frame, header))
            except queue.Full:
                pass

    def _publish_aruco_markers(self):
        """
        Publica los ArUcos como cubos en RViz, en coordenadas centradas
        (mismo origen que /aruco/pose_center).
        """
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for marker_id, (mx, my, m_yaw) in ARUCO_MAP.items():
            # Convertir a coordenadas centradas
            cx = mx - self.MAP_OFFSET_X
            cy = my - self.MAP_OFFSET_Y

            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = 'odom'   # mismo frame que usa el planner
            m.ns     = 'aruco'
            m.id     = marker_id
            m.type   = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = cx
            m.pose.position.y = cy
            m.pose.position.z = 0.1   # ligeramente sobre el suelo

            # Orientación del marcador
            m.pose.orientation.z = math.sin(m_yaw / 2)
            m.pose.orientation.w = math.cos(m_yaw / 2)

            m.scale.x = 0.05
            m.scale.y = 0.05
            m.scale.z = 0.20

            # Verde si fue visto recientemente, gris si no
            if marker_id in self.visible_ids:
                m.color = ColorRGBA(r=0.2, g=0.9, b=0.2, a=1.0)
            else:
                m.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.6)

            ma.markers.append(m)

            # Texto con el ID encima
            t = Marker()
            t.header = m.header
            t.ns     = 'aruco_labels'
            t.id     = marker_id + 100
            t.type   = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = cx
            t.pose.position.y = cy
            t.pose.position.z = 0.35
            t.pose.orientation.w = 1.0
            t.scale.z = 0.12
            t.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text    = f'ID{marker_id}'
            ma.markers.append(t)

        self.aruco_markers_pub.publish(ma)

    def _publish_only(self):
        """Timer de ROS — solo publica pose, no procesa imagen."""
        self._publish_pose()

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLICAR POSE FUSIONADA
    # ─────────────────────────────────────────────────────────────────────
    def _publish_pose(self):
        with self._kf_lock:
            kf_x, kf_y, kf_theta = self.kf_x, self.kf_y, self.kf_theta
            P = self.kf.P.copy()
            # usar kf_x, kf_y, kf_theta, P en lugar de self.kf_x etc.
            msg = PoseWithCovarianceStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.pose.position.x    = kf_x
            msg.pose.pose.position.y    = kf_y
            msg.pose.pose.orientation.z = math.sin(kf_theta / 2)
            msg.pose.pose.orientation.w = math.cos(kf_theta / 2)
            P = self.kf.P
            msg.pose.covariance[0]  = P[0, 0]
            msg.pose.covariance[7]  = P[1, 1]
            msg.pose.covariance[35] = P[2, 2]

            # Versión centrada: resta el offset para que el origen quede en el centro
            msg_c = PoseWithCovarianceStamped()
            msg_c.header = msg.header
            msg_c.pose.covariance = msg.pose.covariance
            msg_c.pose.pose.position.x    = kf_x - self.MAP_OFFSET_X
            msg_c.pose.pose.position.y    = kf_y - self.MAP_OFFSET_Y
            msg_c.pose.pose.orientation   = msg.pose.pose.orientation
            self.pose_centered_pub.publish(msg_c)


# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
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