"""
mcl_aruco.py  -  Localización Monte Carlo con ArUcos para PuzzleBot
====================================================================
Arquitectura
------------
  1. Odometría diferencial  →  predicción de partículas (motion model)
  2. solvePnP sobre cada ArUco visto  →  posición 3-D del marcador en
     el frame de la cámara  →  transformada a coordenadas globales de pista
  3. Pesos de partículas proporcionales al error entre la posición global
     medida y la posición conocida del ArUco en el mapa
  4. Resampling sistemático  →  estimado final [x, y, θ]

Parámetros que debes ajustar
------------------------------
  ARUCO_MAP   : dict  {id: (x_m, y_m, θ_rad)}  -  pose de cada ArUco en la pista
  MARKER_SIZE : tamaño real del marcador en metros
  N_PARTICLES : número de partículas
  SIGMA_* / ALPHA_* : ruido del modelo de movimiento y sensor

Calibración incluida (cámara PuzzleBot)
  fx=133.87  fy=131.03  cx=157.77  cy=93.02
  dist=[-0.157, -0.618, -0.010, -0.007, 0.744]
"""

import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, PoseArray, Pose, PoseWithCovarianceStamped
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
# Ejemplo: marcador en esquina (0,0) mirando al eje +X → yaw=0.0
ARUCO_MAP = {
    1: (0.0,  0.0,  0.0),
    2: (1.0,  0.0,  math.pi / 2),
    3: (1.0,  1.0,  math.pi),
    4: (0.0,  1.0, -math.pi / 2),
}

# ─── Parámetros del filtro de partículas ───────────────────────────────────
N_PARTICLES  = 300      # Número de partículas
SIGMA_OBS_XY = 0.15     # Desviación estándar de observación en XY (metros)
SIGMA_OBS_TH = 0.20     # Desviación estándar de observación en θ (rad)

# Modelo de movimiento (ruido proporcional al movimiento)
ALPHA1 = 0.05   # ruido rotacional por rotación
ALPHA2 = 0.01   # ruido rotacional por traslación
ALPHA3 = 0.05   # ruido traslacional por traslación
ALPHA4 = 0.01   # ruido traslacional por rotación

# ─── Transformación cámara → base del robot ────────────────────────────────
# Ajusta según el montaje físico de la cámara en el PuzzleBot
# Traslación: la cámara está X_cam metros al frente y Z_cam metros arriba
CAM_TX = 0.08   # offset longitudinal (adelante +)  [m]
CAM_TY = 0.0    # offset lateral                    [m]
CAM_TZ = 0.06   # offset vertical                   [m]
CAM_PITCH = 0.0 # inclinación de la cámara          [rad]  (hacia abajo +)

# Diccionario ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
try:
    det_params = aruco.DetectorParameters()
except AttributeError:
    det_params = aruco.DetectorParameters_create()


# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES GEOMÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def rvec_to_yaw(rvec: np.ndarray) -> float:
    """Convierte un vector de rotación de solvePnP al ángulo yaw (en el plano XZ de la cámara)."""
    R, _ = cv2.Rodrigues(rvec)
    # yaw alrededor del eje Y de la cámara (profundidad)
    yaw = math.atan2(R[0, 2], R[2, 2])
    return yaw

def cam_to_global(tvec: np.ndarray, rvec: np.ndarray,
                  robot_x: float, robot_y: float, robot_theta: float):
    """
    Dada la traslación/rotación del marcador respecto a la cámara (solvePnP),
    devuelve la pose estimada del robot en coordenadas globales usando
    la posición conocida del marcador en el mapa.

    Retorna (x_est, y_est, theta_est) o None si el id no está en el mapa.
    """
    # Posición del marcador en frame cámara [metros]
    t = tvec.flatten()          # [x_cam, y_cam, z_cam]
    cam_x =  t[0]               # lateral   (der +)
    cam_z =  t[2]               # profundidad (adelante +)

    # Distancia horizontal al marcador (ignoramos altura)
    dist_h = math.sqrt(cam_x**2 + cam_z**2)

    # Ángulo horizontal del marcador respecto al eje óptico
    bearing = math.atan2(cam_x, cam_z)  # + = marcador a la derecha

    return dist_h, bearing, rvec_to_yaw(rvec)


def observation_from_pose(robot_x, robot_y, robot_theta,
                          marker_x, marker_y):
    """
    Predice {distancia, bearing} que vería el robot desde (robot_x, robot_y, robot_theta)
    hacia el marcador en (marker_x, marker_y).
    """
    dx = marker_x - robot_x
    dy = marker_y - robot_y
    dist = math.sqrt(dx**2 + dy**2)
    bearing = wrap_angle(math.atan2(dy, dx) - robot_theta)
    return dist, bearing


# ═══════════════════════════════════════════════════════════════════════════
#  NODO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class MCLArucoNode(Node):
    def __init__(self):
        super().__init__('mcl_aruco')

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub   = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_pub     = self.create_publisher(Image, '/aruco/image_detected', 10)
        self.particles_pub = self.create_publisher(PoseArray, '/mcl/particles', 10)
        self.pose_pub      = self.create_publisher(
            PoseWithCovarianceStamped, '/mcl/pose', 10)

        # ── Subscribers ───────────────────────────────────────────────────
        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_callback, qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_callback, qos.qos_profile_sensor_data)

        qos_img = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_img)

        # ── Timers ────────────────────────────────────────────────────────
        self.timer_odom = self.create_timer(1 / 100, self.odometria)
        self.timer_ctrl = self.create_timer(1 / 50,  self.control)

        # ── Odometría ─────────────────────────────────────────────────────
        self.x = 0.0; self.y = 0.0; self.theta = 0.0
        self.wr = Float32(); self.wl = Float32()
        self.v_robot = 0.0; self.w_robot = 0.0

        self.radio  = 0.0505
        self.lenght = 0.183

        self.last_time_odom    = self.get_clock().now()
        self.last_time_control = self.get_clock().now()

        # Delta de movimiento para el modelo de movimiento (acumulado entre updates)
        self._delta_trans = 0.0
        self._delta_rot1  = 0.0
        self._delta_rot2  = 0.0
        self._last_odom   = np.array([0.0, 0.0, 0.0])  # [x, y, θ]

        # ── Control (heredado de centerAruco) ─────────────────────────────
        self.Kp_v = 0.15; self.Ki_v = 0.25
        self.int_error_v = 0.0
        self.Kp_w = 0.08; self.Kv_w = 0.05
        self.stop_ratio   = 0.25
        self.close_enough = False
        self.cx = None; self.cy = None
        self.ratio = 0.0

        # ── Cámara / ArUco ────────────────────────────────────────────────
        self.camera_width  = 640
        self.camera_height = 480
        self.img_width     = self.camera_width
        self.bridge   = CvBridge()
        self.detector = aruco.ArucoDetector(aruco_dict, det_params)
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

        # ── Partículas MCL ────────────────────────────────────────────────
        # Inicializa alrededor del origen (0,0,0) con dispersión pequeña
        self.particles = self._init_particles(0.0, 0.0, 0.0, spread=0.3)
        self.weights   = np.ones(N_PARTICLES) / N_PARTICLES

        # Estimado actual del filtro
        self.mcl_x = 0.0; self.mcl_y = 0.0; self.mcl_theta = 0.0

        self.get_logger().info(
            f'mcl_aruco iniciado | {N_PARTICLES} partículas | '
            f'{len(ARUCO_MAP)} ArUcos en el mapa')

    # ── Inicialización de partículas ──────────────────────────────────────
    def _init_particles(self, x0, y0, th0, spread=0.5):
        pts = np.zeros((N_PARTICLES, 3))
        pts[:, 0] = np.random.normal(x0,  spread,      N_PARTICLES)
        pts[:, 1] = np.random.normal(y0,  spread,      N_PARTICLES)
        pts[:, 2] = np.random.normal(th0, spread / 2,  N_PARTICLES)
        pts[:, 2] = np.vectorize(wrap_angle)(pts[:, 2])
        return pts

    # ─────────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────────
    def encR_callback(self, msg): self.wr = msg
    def encL_callback(self, msg): self.wl = msg

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
    #  ODOMETRÍA
    # ─────────────────────────────────────────────────────────────────────
    def odometria(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time_odom).nanoseconds * 1e-9
        self.last_time_odom = now
        if dt <= 0:
            return

        v_r = self.radio * self.wr.data
        v_l = self.radio * self.wl.data
        V   = (v_r + v_l) / 2.0
        W   = (v_r - v_l) / self.lenght

        self.v_robot = V
        self.w_robot = W

        self.x     += V * math.cos(self.theta) * dt
        self.y     += V * math.sin(self.theta) * dt
        self.theta += W * dt
        self.theta  = wrap_angle(self.theta)

    # ─────────────────────────────────────────────────────────────────────
    #  MODELO DE MOVIMIENTO (para MCL)
    # ─────────────────────────────────────────────────────────────────────
    def _motion_update(self):
        """
        Compara odometría actual vs. última pose guardada y calcula
        (delta_rot1, delta_trans, delta_rot2) según el modelo probabilístico
        de Thrun et al.  Luego propaga las partículas con ruido.
        """
        odom_now = np.array([self.x, self.y, self.theta])
        dx    = odom_now[0] - self._last_odom[0]
        dy    = odom_now[1] - self._last_odom[1]
        dth   = wrap_angle(odom_now[2] - self._last_odom[2])
        dtrans = math.sqrt(dx**2 + dy**2)

        if dtrans < 1e-4 and abs(dth) < 1e-4:
            return   # sin movimiento significativo

        drot1 = wrap_angle(math.atan2(dy, dx) - self._last_odom[2]) if dtrans > 1e-4 else 0.0
        drot2 = wrap_angle(dth - drot1)

        self._last_odom = odom_now.copy()

        # Propagar cada partícula con ruido gaussiano
        def noise(scale): return np.random.normal(0, abs(scale) + 1e-6, N_PARTICLES)

        std_r1 = ALPHA1 * abs(drot1) + ALPHA2 * dtrans
        std_tr = ALPHA3 * dtrans     + ALPHA4 * (abs(drot1) + abs(drot2))
        std_r2 = ALPHA1 * abs(drot2) + ALPHA2 * dtrans

        r1 = drot1  + noise(std_r1)
        tr = dtrans + noise(std_tr)
        r2 = drot2  + noise(std_r2)

        self.particles[:, 0] += tr * np.cos(self.particles[:, 2] + r1)
        self.particles[:, 1] += tr * np.sin(self.particles[:, 2] + r1)
        self.particles[:, 2] += r1 + r2
        self.particles[:, 2]  = np.vectorize(wrap_angle)(self.particles[:, 2])

    # ─────────────────────────────────────────────────────────────────────
    #  MODELO SENSOR (pesos con observaciones ArUco)
    # ─────────────────────────────────────────────────────────────────────
    def _sensor_update(self, observations):
        """
        observations : lista de dicts con keys
            'dist_h'  - distancia horizontal medida al marcador [m]
            'bearing' - ángulo horizontal medido                [rad]
            'marker_id' - id del ArUco detectado
        Actualiza self.weights.
        """
        if not observations:
            return

        log_weights = np.zeros(N_PARTICLES)

        for obs in observations:
            mid = obs['marker_id']
            if mid not in ARUCO_MAP:
                continue

            mx, my, _ = ARUCO_MAP[mid]
            meas_dist    = obs['dist_h']
            meas_bearing = obs['bearing']

            # Para cada partícula, calcular la observación esperada
            for k in range(N_PARTICLES):
                px, py, pth = self.particles[k]
                pred_dist, pred_bearing = observation_from_pose(px, py, pth, mx, my)

                err_d = meas_dist    - pred_dist
                err_b = wrap_angle(meas_bearing - pred_bearing)

                # Log-verosimilitud gaussiana
                log_weights[k] += (
                    -0.5 * (err_d / SIGMA_OBS_XY) ** 2
                    -0.5 * (err_b / SIGMA_OBS_TH) ** 2
                )

        # Convertir de log a probabilidades (evita underflow)
        log_weights -= np.max(log_weights)
        self.weights  = np.exp(log_weights)
        self.weights += 1e-300
        self.weights /= self.weights.sum()

    # ─────────────────────────────────────────────────────────────────────
    #  RESAMPLING SISTEMÁTICO
    # ─────────────────────────────────────────────────────────────────────
    def _resample(self):
        cumsum = np.cumsum(self.weights)
        step   = 1.0 / N_PARTICLES
        start  = np.random.uniform(0, step)
        indices = []
        j = 0
        for i in range(N_PARTICLES):
            u = start + i * step
            while cumsum[j] < u:
                j += 1
            indices.append(j)
        self.particles = self.particles[indices]
        self.weights   = np.ones(N_PARTICLES) / N_PARTICLES

    # ─────────────────────────────────────────────────────────────────────
    #  ESTIMADO FINAL (media ponderada circular para θ)
    # ─────────────────────────────────────────────────────────────────────
    def _estimate(self):
        self.mcl_x = float(np.average(self.particles[:, 0], weights=self.weights))
        self.mcl_y = float(np.average(self.particles[:, 1], weights=self.weights))
        sin_avg = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_avg = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        self.mcl_theta = float(math.atan2(sin_avg, cos_avg))

    # ─────────────────────────────────────────────────────────────────────
    #  DETECCIÓN ArUco + POSE ESTIMATION
    # ─────────────────────────────────────────────────────────────────────
    def process_aruco(self):
        """
        Detecta ArUcos, estima su pose con solvePnP y retorna lista de observaciones.
        También actualiza self.cx, self.cy, self.ratio, self.close_enough para el control.
        """
        self.cx, self.cy = None, None
        observations = []

        if self.latest_frame is None:
            return observations

        frame = self.latest_frame.copy()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for m_idx in range(len(corners)):
                pts = np.squeeze(corners[m_idx])
                if pts.shape != (4, 2):
                    continue

                marker_id = int(ids[m_idx][0])

                # ── Centro en píxeles ──────────────────────────────────
                cx_m = int(np.mean(pts[:, 0]))
                cy_m = int(np.mean(pts[:, 1]))
                cv2.circle(frame, (cx_m, cy_m), 6, (0, 255, 0), -1)

                # ── Ratio para control de acercamiento ────────────────
                avg_side_px = np.mean([
                    np.linalg.norm(pts[0] - pts[1]),
                    np.linalg.norm(pts[1] - pts[2]),
                    np.linalg.norm(pts[2] - pts[3]),
                    np.linalg.norm(pts[3] - pts[0]),
                ])
                self.ratio = avg_side_px / self.img_width

                if self.cx is None:
                    self.cx, self.cy  = cx_m, cy_m
                    self.close_enough = self.ratio >= self.stop_ratio

                # ── solvePnP ──────────────────────────────────────────
                img_pts = pts.astype(np.float64)
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, img_pts,
                    CAMERA_MATRIX, DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    dist_h, bearing, marker_yaw = cam_to_global(
                        tvec, rvec, self.mcl_x, self.mcl_y, self.mcl_theta)

                    observations.append({
                        'marker_id': marker_id,
                        'dist_h':    dist_h,
                        'bearing':   bearing,
                        'tvec':      tvec.flatten(),
                        'rvec':      rvec.flatten(),
                    })

                    # Dibujar ejes 3-D sobre el marcador
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS,
                                      rvec, tvec, MARKER_SIZE * 0.6)

                    # Información de posición en pantalla
                    t = tvec.flatten()
                    label_color = (0, 255, 0) if marker_id in ARUCO_MAP else (0, 165, 255)
                    cv2.putText(frame,
                        f"ID:{marker_id}  d={dist_h:.2f}m  b={math.degrees(bearing):.1f}°",
                        (int(pts[0][0]), int(pts[0][1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, label_color, 2)
                    cv2.putText(frame,
                        f"cam: x={t[0]:.2f} y={t[1]:.2f} z={t[2]:.2f}",
                        (int(pts[0][0]), int(pts[0][1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 0), 1)
                else:
                    cv2.putText(frame, f"ID:{marker_id} (solvePnP fail)",
                                (cx_m - 20, cy_m - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # Estimado MCL en imagen
            cv2.putText(frame,
                f"MCL  x={self.mcl_x:.2f}  y={self.mcl_y:.2f}  "
                f"th={math.degrees(self.mcl_theta):.1f}deg",
                (10, self.camera_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'No ArUco detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame,
                f"MCL  x={self.mcl_x:.2f}  y={self.mcl_y:.2f}  "
                f"th={math.degrees(self.mcl_theta):.1f}deg",
                (10, self.camera_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # Publicar imagen anotada
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if self.latest_header is not None:
                out_msg.header = self.latest_header
            self.image_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'Error publicando imagen: {e}')

        cv2.imshow("MCL ArUco", frame)
        cv2.waitKey(1)

        return observations

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLICAR PARTÍCULAS (para visualizar en RViz)
    # ─────────────────────────────────────────────────────────────────────
    def _publish_particles(self):
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            # Convertir yaw a quaternion (solo z, w)
            pose.orientation.z = float(math.sin(p[2] / 2))
            pose.orientation.w = float(math.cos(p[2] / 2))
            msg.poses.append(pose)
        self.particles_pub.publish(msg)

    def _publish_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = self.mcl_x
        msg.pose.pose.position.y = self.mcl_y
        msg.pose.pose.orientation.z = math.sin(self.mcl_theta / 2)
        msg.pose.pose.orientation.w = math.cos(self.mcl_theta / 2)
        # Covarianza diagonal simplificada
        cov = np.zeros(36)
        var_xy = float(np.average(
            (self.particles[:, 0] - self.mcl_x)**2 +
            (self.particles[:, 1] - self.mcl_y)**2,
            weights=self.weights))
        var_th = float(np.average(
            (self.particles[:, 2] - self.mcl_theta)**2,
            weights=self.weights))
        cov[0]  = var_xy
        cov[7]  = var_xy
        cov[35] = var_th
        msg.pose.covariance = cov.tolist()
        self.pose_pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────────
    #  LOOP PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────
    def control(self):
        # 1. Detectar ArUcos y obtener observaciones
        observations = self.process_aruco()

        # ── MCL pipeline ─────────────────────────────────────────────────
        # 2. Motion update (predicción con odometría)
        self._motion_update()

        # 3. Sensor update (pesos con ArUcos vistos)
        if observations:
            self._sensor_update(observations)
            self._resample()

        # 4. Estimado
        self._estimate()

        # 5. Publicar
        self._publish_particles()
        self._publish_pose()

        # ── Control de velocidad (igual que centerAruco) ──────────────────
        cmd = Twist()
        now = self.get_clock().now()
        dt  = (now - self.last_time_control).nanoseconds * 1e-9
        self.last_time_control = now
        dt  = min(dt, 0.1)

        if self.cx is None:
            self.cmd_vel_pub.publish(cmd)
            return

        error_v = self.stop_ratio - self.ratio
        error_w = (self.cx - self.img_width / 2.0) / (self.img_width / 2.0)

        self.int_error_v += error_v * dt
        self.int_error_v  = max(min(self.int_error_v, 1.0), -1.0)

        if abs(error_w) < 0.05 and self.close_enough:
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(
                f'ArUco centrado — MCL: ({self.mcl_x:.2f}, {self.mcl_y:.2f}, '
                f'{math.degrees(self.mcl_theta):.1f}°)')
            return

        u_v = self.Ki_v * self.int_error_v - self.Kp_v * self.v_robot
        u_v = max(min(u_v, 0.4), -0.4)
        u_w = self.Kp_w * error_w - self.Kv_w * self.w_robot
        u_w = max(min(u_w, 0.2), -0.2)

        if abs(error_w) > 0.12:
            u_v = 0.0
            self.int_error_v = 0.0

        cmd.linear.x  =  u_v
        cmd.angular.z = -u_w
        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info(
            f'MCL({self.mcl_x:.2f},{self.mcl_y:.2f},{math.degrees(self.mcl_theta):.0f}°) '
            f'| ev={error_v:+.2f} ew={error_w:+.2f} '
            f'| uv={u_v:+.2f} uw={u_w:+.2f}')


# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = MCLArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()