#!/usr/bin/env python3
"""
slam_node.py  --  corre en la LAPTOP
══════════════════════════════════════════════════════════════════════
SLAM 2D con filtro de partículas (estilo CoreSLAM/BreezySLAM)

Pipeline (ver slides Semana 3):
  1. Recibe scan LIDAR  →  /scan
  2. Recibe odometría   →  /odom  (publicada por slam_odom.py en la Jetson)
  3. Predice partículas con dead reckoning (motion model)
  4. Puntúa partículas comparando scan contra mapa actual
  5. Remuestrea partículas (las malas mueren, las buenas se replican)
  6. La mejor partícula actualiza el mapa (stitching / ray casting)
  7. Publica: /slam/pose, /slam/map, /slam/particles, TF map→odom

Suscripciones
─────────────
  /scan   sensor_msgs/LaserScan
  /odom   nav_msgs/Odometry

Publicaciones
─────────────
  /slam/pose       geometry_msgs/PoseWithCovarianceStamped
  /slam/map        nav_msgs/OccupancyGrid   (mapa que se va construyendo)
  /slam/particles  geometry_msgs/PoseArray  (ver en rviz)
  TF dinámico:  map → odom

Requisitos
──────────
  pip3 install numpy opencv-python --break-system-packages
  ros-humble: nav_msgs geometry_msgs sensor_msgs tf2_ros

Parámetros ajustables (sección CONFIG abajo)
────────────────────────────────────────────
  N_PARTICLES    número de partículas (50–500)
  MAP_SIZE_M     tamaño del mapa cuadrado en metros
  MAP_RES        resolución del mapa en metros/celda
  SIGMA_XY       ruido de posición en el motion model (m)
  SIGMA_THETA    ruido de ángulo en el motion model (rad)
  SIGMA_HIT      std del modelo de sensor (m) — qué tan "preciso" es el LIDAR
  P_HIT          peso del rayo que toca obstáculo
  P_RAND         peso del rayo aleatorio (ruido)
  LIDAR_MAX_M    rango máximo del LIDAR en metros
  LIDAR_MIN_M    rango mínimo del LIDAR en metros
  N_RAYS         cuántos rayos del scan usar (submuestreo para velocidad)
"""

import math
import time
import numpy as np
import cv2

import rclpy
from rclpy import qos as ros_qos
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import (PoseWithCovarianceStamped, PoseArray,
                                Pose, TransformStamped)
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as RosTime


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

N_PARTICLES = 200         # número de partículas

# Mapa
MAP_SIZE_X  = 6         # metros — ancho del mapa
MAP_SIZE_Y  = 4.5         # metros — alto del mapa
MAP_RES     = 0.05        # metros por celda

# Motion model — ruido gaussiano en la predicción de partículas
SIGMA_XY    = 0.03        # m    — más alto = partículas más dispersas
SIGMA_THETA = 0.02        # rad

# Sensor model — cómo puntuar un scan contra el mapa
SIGMA_HIT   = 0.15        # m   — std de la gaussiana de coincidencia
P_HIT       = 0.85        # peso del término "rayo toca obstáculo"
P_RAND      = 0.05        # peso del término "rayo aleatorio"
# P_MAX = 1 - P_HIT - P_RAND  → peso del término "rayo sin retorno"

# LIDAR
LIDAR_MAX_M = 12.0        # metros — descartar rayos más largos
LIDAR_MIN_M = 0.10        # metros — descartar rayos muy cortos
N_RAYS      = 60          # cuántos rayos usar por scan (submuestreo)

# Mapa de ocupación: umbral para considerar una celda ocupada
OCC_THRESH  = 50          # 0–100 en OccupancyGrid

# Valor de ocupación que se añade al mapa en cada hit/miss (CoreSLAM style)
LOG_OCC_HIT  =  15        # aumenta probabilidad de ocupado
LOG_OCC_MISS = -5         # disminuye probabilidad de ocupado
LOG_OCC_MAX  =  100
LOG_OCC_MIN  = -100

# Ventana de visualización OpenCV
SHOW_MAP = False           # False si corres en headless


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES GEOMÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

def wrap(a: float) -> float:
    """Normaliza ángulo a [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_to_quat(yaw: float):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return 0.0, 0.0, sy, cy


def quat_to_yaw(qz: float, qw: float) -> float:
    return 2.0 * math.atan2(qz, qw)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAPA DE OCUPACIÓN (log-odds, estilo CoreSLAM)
# ═══════════════════════════════════════════════════════════════════════════════

class OccupancyMap:
    """
    Mapa 2D en log-odds.

    Internamente usa un array int16 de log-odds escalados.
    Para publicar en ROS se convierte a OccupancyGrid (0-100, -1=desconocido).

    El origen del mapa (celda [0,0]) corresponde a la coordenada global
    (-MAP_SIZE_M/2, -MAP_SIZE_M/2).  El robot arranca en el centro del mapa.
    """

    def __init__(self, size_x: float, size_y: float, res: float):
        self.res    = res                                # m / celda
        self.size_x = size_x
        self.size_y = size_y
        self.cells_x = int(size_x / res)
        self.cells_y = int(size_y / res)
        self.origin_x = 0.0      # esquina inferior izquierda del mapa real
        self.origin_y = 0.0
        self.logodds = np.zeros((self.cells_y, self.cells_x), dtype=np.int16)

    # ── Conversión coordenadas globales ↔ celda ─────────────────────────────
    def world_to_cell(self, wx: float, wy: float):
        cx = int((wx - self.origin_x) / self.res)
        cy = int((wy - self.origin_y) / self.res)
        return cx, cy

    def cell_to_world(self, cx: int, cy: int):
        wx = cx * self.res + self.origin_x + self.res / 2.0
        wy = cy * self.res + self.origin_y + self.res / 2.0
        return wx, wy

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.cells_x and 0 <= cy < self.cells_y

    # ── Actualizar mapa con un scan (ray casting Bresenham) ─────────────────
    def update(self, robot_x: float, robot_y: float, robot_theta: float,
               scan_ranges: np.ndarray, scan_angles: np.ndarray):
        """
        Para cada rayo válido:
          - Traza la línea desde el robot hasta el punto de impacto con Bresenham.
          - Celdas en la línea (excepto la última) → miss  (LOG_OCC_MISS)
          - Celda de impacto → hit  (LOG_OCC_HIT)
        """
        rx, ry = robot_x, robot_y
        crx, cry = self.world_to_cell(rx, ry)

        for i, r in enumerate(scan_ranges):
            if not (LIDAR_MIN_M < r < LIDAR_MAX_M):
                continue
            angle = robot_theta + scan_angles[i]
            hit_x = rx + r * math.cos(angle)
            hit_y = ry + r * math.sin(angle)
            chx, chy = self.world_to_cell(hit_x, hit_y)

            # Bresenham sobre el rayo
            cells_on_ray = self._bresenham(crx, cry, chx, chy)

            # Celdas previas al impacto → libre
            for (cx, cy) in cells_on_ray[:-1]:
                if self.in_bounds(cx, cy):
                    self.logodds[cy, cx] = max(
                        self.logodds[cy, cx] + LOG_OCC_MISS, LOG_OCC_MIN)

            # Celda de impacto → ocupada
            if self.in_bounds(chx, chy):
                self.logodds[chy, chx] = min(
                    self.logodds[chy, chx] + LOG_OCC_HIT, LOG_OCC_MAX)

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Devuelve lista de celdas (x,y) entre (x0,y0) y (x1,y1)."""
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x   += sx
            if e2 < dx:
                err += dx
                y   += sy
        return cells

    # ── Consulta: valor de ocupación en una coordenada global ───────────────
    def get_occ(self, wx: float, wy: float) -> float:
        """Devuelve probabilidad de ocupación [0,1]. 0.5 si desconocido."""
        cx, cy = self.world_to_cell(wx, wy)
        if not self.in_bounds(cx, cy):
            return 0.5
        lo = float(self.logodds[cy, cx])
        # log-odds → probabilidad: p = 1/(1+exp(-lo/10))
        return 1.0 / (1.0 + math.exp(-lo / 10.0))

    # ── Convertir a OccupancyGrid de ROS ────────────────────────────────────
    def to_ros_msg(self, stamp, frame_id: str) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.info.resolution = self.res
        msg.info.width      = self.cells_x
        msg.info.height     = self.cells_y
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        # Convertir log-odds a 0-100 (-1 = desconocido)
        data = np.full((self.cells_y, self.cells_x), -1, dtype=np.int8)
        known = self.logodds != 0
        # probabilidad de ocupación
        prob = 1.0 / (1.0 + np.exp(-self.logodds[known].astype(np.float32) / 10.0))
        data[known] = (prob * 100).astype(np.int8)
        msg.data = data.flatten().tolist()
        return msg

    # ── Imagen OpenCV para visualización ────────────────────────────────────
    def to_image(self) -> np.ndarray:
        """Mapa en escala de grises: blanco=libre, negro=ocupado, gris=desconocido."""
        img = np.full((self.cells_y, self.cells_x), 128, dtype=np.uint8)
        known = self.logodds != 0
        prob = 1.0 / (1.0 + np.exp(-self.logodds[known].astype(np.float32) / 10.0))
        img[known] = ((1.0 - prob) * 255).astype(np.uint8)
        return cv2.flip(img, 0)   # flip Y para visualización estándar


# ═══════════════════════════════════════════════════════════════════════════════
#  FILTRO DE PARTÍCULAS
# ═══════════════════════════════════════════════════════════════════════════════

class ParticleFilter:
    """
    Filtro de partículas para SLAM 2D.

    Cada partícula es [x, y, theta] con su peso w.

    Pasos por iteración:
      predict()  → motion model con ruido gaussiano
      weight()   → sensor model: ray casting sobre el mapa
      resample() → remuestreo sistemático
    """

    def __init__(self, n: int):
        self.n = n
        # Inicializar partículas cerca del origen (donde arranca el robot)
        self.particles = np.zeros((n, 3))   # [x, y, theta]

        self.particles[:, 0] = np.random.uniform((MAP_SIZE_X/2)-0.3, (MAP_SIZE_X/2)+0.3, n)
        self.particles[:, 1] = np.random.uniform((MAP_SIZE_Y/2)-0.3, (MAP_SIZE_Y/2)+0.3, n)
        self.particles[:, 2] = np.random.normal(0.0, 0.0, n)
        self.weights = np.ones(n) / n

    # ── 1. Motion model ─────────────────────────────────────────────────────
    def predict(self, dv: float, dw: float, dt: float):
        """
        Propaga cada partícula con el modelo cinemático diferencial
        más ruido gaussiano.

          x'     = x + (v + noise_v) * cos(theta) * dt
          y'     = y + (v + noise_v) * sin(theta) * dt
          theta' = theta + (w + noise_w) * dt
        """
        n   = self.n
        nv  = np.random.normal(0.0, SIGMA_XY,    n)
        nw  = np.random.normal(0.0, SIGMA_THETA, n)
        nxy = np.random.normal(0.0, SIGMA_XY / 3.0, (n, 2))

        v = dv + nv
        w = dw + nw

        self.particles[:, 0] += v * np.cos(self.particles[:, 2]) * dt + nxy[:, 0]
        self.particles[:, 1] += v * np.sin(self.particles[:, 2]) * dt + nxy[:, 1]
        self.particles[:, 2]  = np.array([wrap(a) for a in
                                          self.particles[:, 2] + w * dt])

    # ── 2. Sensor model ─────────────────────────────────────────────────────
    def weight(self, occ_map: OccupancyMap,
               scan_ranges: np.ndarray, scan_angles: np.ndarray):
        """
        Puntúa cada partícula comparando el scan LIDAR contra el mapa.

        Para cada rayo válido calcula el punto de impacto esperado y consulta
        la probabilidad de ocupación en el mapa.  La puntuación total de la
        partícula es el producto de los scores individuales de cada rayo
        (en log para evitar underflow).

        Score de un rayo:
          - Si la celda de impacto está ocupada (prob > umbral): P_HIT * gauss(0, SIGMA_HIT)
          - Si no:  P_RAND  (ruido uniforme)
        """
        log_weights = np.zeros(self.n)

        valid_idx = np.where(
            (scan_ranges > LIDAR_MIN_M) & (scan_ranges < LIDAR_MAX_M)
        )[0]

        if len(valid_idx) == 0:
            return

        for pi in range(self.n):
            px, py, pth = self.particles[pi]
            log_w = 0.0

            for i in valid_idx:
                r     = scan_ranges[i]
                angle = pth + scan_angles[i]
                hx    = px + r * math.cos(angle)
                hy    = py + r * math.sin(angle)

                occ = occ_map.get_occ(hx, hy)

                if occ > 0.5:
                    # Celda ocupada: score proporcional a gaussiana centrada en 0
                    score = P_HIT * math.exp(-0.5 * (0.0 / SIGMA_HIT) ** 2)
                else:
                    # Celda libre o desconocida: sólo ruido de fondo
                    score = P_RAND

                score = max(score, 1e-300)
                log_w += math.log(score)

            log_weights[pi] = log_w

        # Normalizar en escala log para estabilidad numérica
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.n) / self.n

    # ── 3. Resampling (sistemático) ─────────────────────────────────────────
    def resample(self):
        """
        Remuestreo sistemático: O(N), sin sesgo, estándar en MCL.

        Genera N nuevas partículas sorteadas con probabilidad proporcional
        a su peso.  Las partículas con peso bajo tienden a desaparecer;
        las de peso alto se replican.
        """
        positions = (np.arange(self.n) + np.random.uniform()) / self.n
        cumsum    = np.cumsum(self.weights)
        i, j      = 0, 0
        indices   = np.zeros(self.n, dtype=int)
        while i < self.n:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j = min(j + 1, self.n - 1)
        self.particles = self.particles[indices]
        self.weights   = np.ones(self.n) / self.n

    # ── Estimación de pose (media ponderada) ─────────────────────────────────
    def estimate(self):
        """Devuelve (x, y, theta) como media ponderada de las partículas."""
        x     = float(np.average(self.particles[:, 0], weights=self.weights))
        y     = float(np.average(self.particles[:, 1], weights=self.weights))
        sin_t = float(np.average(np.sin(self.particles[:, 2]), weights=self.weights))
        cos_t = float(np.average(np.cos(self.particles[:, 2]), weights=self.weights))
        theta = math.atan2(sin_t, cos_t)
        return x, y, theta

    # ── Partícula de mayor peso ───────────────────────────────────────────────
    def best(self):
        """Devuelve la partícula con mayor peso (para actualizar el mapa)."""
        idx = int(np.argmax(self.weights))
        return self.particles[idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  NODO ROS
# ═══════════════════════════════════════════════════════════════════════════════

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # ── Mapa y filtro ──────────────────────────────────────────────────
        self.occ_map = OccupancyMap(MAP_SIZE_X, MAP_SIZE_Y, MAP_RES)
        self.pf      = ParticleFilter(N_PARTICLES)

        # ── Estado de odometría ────────────────────────────────────────────
        self.last_odom_x     = 0.0
        self.last_odom_y     = 0.0
        self.last_odom_theta = 0.0
        self.odom_ready      = False

        self.last_v = 0.0   # velocidad lineal del último /odom
        self.last_w = 0.0   # velocidad angular del último /odom
        self.last_odom_time = None

        # ── Scan LIDAR más reciente ────────────────────────────────────────
        self.latest_scan = None

        # ── Publishers ────────────────────────────────────────────────────
        self.pose_pub      = self.create_publisher(
            PoseWithCovarianceStamped, '/slam/pose', 10)
        self.map_pub       = self.create_publisher(
            OccupancyGrid, '/slam/map', 10)
        self.particles_pub = self.create_publisher(
            PoseArray, '/slam/particles', 10)

        self.tf_br = TransformBroadcaster(self)

        # ── Subscribers ───────────────────────────────────────────────────
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(Odometry,   '/odom', self.odom_cb,  qos_sensor)
        self.create_subscription(LaserScan,  '/scan', self.scan_cb,  qos_sensor)

        # ── Timer principal: corre el pipeline SLAM a ~5 Hz ───────────────
        self.slam_timer = self.create_timer(0.2, self.slam_step)

        # ── Timer de publicación del mapa: cada 2 s ────────────────────────
        self.map_timer  = self.create_timer(2.0, self.publish_map)

        self.get_logger().info(
            f'slam_node listo | {N_PARTICLES} partículas | '
            f'mapa {MAP_SIZE_X}x{MAP_SIZE_Y} m @ {MAP_RES} m/celda')

    # ─────────────────────────────────────────────────────────────────────────
    #  CALLBACKS DE SENSORES
    # ─────────────────────────────────────────────────────────────────────────

    def odom_cb(self, msg: Odometry):
        """Guarda velocidades y computa delta de pose desde el último mensaje."""
        self.last_v = msg.twist.twist.linear.x
        self.last_w = msg.twist.twist.angular.z

        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        theta = quat_to_yaw(msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w)

        if not self.odom_ready:
            self.last_odom_x     = x
            self.last_odom_y     = y
            self.last_odom_theta = theta
            self.odom_ready      = True

        self.last_odom_x     = x
        self.last_odom_y     = y
        self.last_odom_theta = theta
        self.last_odom_time  = self.get_clock().now()

    def scan_cb(self, msg: LaserScan):
        """Guarda el scan más reciente, submuestreado a N_RAYS rayos."""
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.array([msg.angle_min + i * msg.angle_increment
                           for i in range(len(ranges))], dtype=np.float32)

        # Submuestreo uniforme a N_RAYS
        idx = np.linspace(0, len(ranges) - 1, N_RAYS, dtype=int)
        self.latest_scan = (ranges[idx], angles[idx])

    # ─────────────────────────────────────────────────────────────────────────
    #  PIPELINE SLAM  (se ejecuta a ~5 Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def slam_step(self):
        if not self.odom_ready or self.latest_scan is None:
            return

        scan_ranges, scan_angles = self.latest_scan
        dt = 0.2   # período del timer

        # ── 1. PREDICCIÓN: motion model con velocidades actuales ───────────
        self.pf.predict(self.last_v, self.last_w, dt)

        # ── 2. PONDERACIÓN: sensor model ──────────────────────────────────
        # Solo puntuar si el mapa ya tiene algo de información
        if np.any(self.occ_map.logodds != 0):
            self.pf.weight(self.occ_map, scan_ranges, scan_angles)

        # ── 3. REMUESTREO ─────────────────────────────────────────────────
        # Número efectivo de partículas: N_eff = 1/sum(w²)
        n_eff = 1.0 / (np.sum(self.pf.weights ** 2) + 1e-10)
        if n_eff < self.pf.n / 2:
            self.pf.resample()

        # ── 4. ESTIMACIÓN de pose ─────────────────────────────────────────
        est_x, est_y, est_theta = self.pf.estimate()

        # ── 5. ACTUALIZAR MAPA con la mejor partícula ─────────────────────
        best = self.pf.best()
        self.occ_map.update(best[0], best[1], best[2],
                            scan_ranges, scan_angles)

        # ── 6. PUBLICAR pose, partículas y TF ─────────────────────────────
        stamp = self.get_clock().now().to_msg()
        self._publish_pose(est_x, est_y, est_theta, stamp)
        self._publish_particles(stamp)
        self._publish_tf(est_x, est_y, est_theta, stamp)

        # ── 7. Visualización OpenCV (opcional) ───────────────────────────
        if SHOW_MAP:
            self._show_map(est_x, est_y, est_theta)

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLICADORES
    # ─────────────────────────────────────────────────────────────────────────

    def _publish_pose(self, x, y, theta, stamp):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        qx, qy, qz, qw = yaw_to_quat(theta)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        # Covarianza diagonal simplificada
        msg.pose.covariance[0]  = 0.1
        msg.pose.covariance[7]  = 0.1
        msg.pose.covariance[35] = 0.2
        self.pose_pub.publish(msg)

    def _publish_particles(self, stamp):
        msg = PoseArray()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'map'
        for p in self.pf.particles:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            qx, qy, qz, qw = yaw_to_quat(float(p[2]))
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            msg.poses.append(pose)
        self.particles_pub.publish(msg)

    def _publish_tf(self, x, y, theta, stamp):
        """Publica TF map → odom.  El TF odom → base_link lo publica la Jetson."""
        qx, qy, qz, qw = yaw_to_quat(theta)
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = 'map'
        tf.child_frame_id  = 'odom'
        # La corrección SLAM es la diferencia entre la pose estimada y la odometría
        # Para simplificar: publicamos map→odom como la pose estimada MCL
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_br.sendTransform(tf)

    def publish_map(self):
        """Publica el mapa de ocupación en ROS (para rviz / nav2)."""
        stamp = self.get_clock().now().to_msg()
        msg   = self.occ_map.to_ros_msg(stamp, 'map')
        self.map_pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────────────
    #  VISUALIZACIÓN OpenCV
    # ─────────────────────────────────────────────────────────────────────────

    def _show_map(self, robot_x: float, robot_y: float, robot_theta: float):
        img = self.occ_map.to_image()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cells_x = self.occ_map.cells_x
        cells_y = self.occ_map.cells_y
        res     = self.occ_map.res
        orig_x  = self.occ_map.origin_x
        orig_y  = self.occ_map.origin_y

        def w2p(wx, wy):
            cx = int((wx - orig_x) / res)
            cy = cells_y - int((wy - orig_y) / res)
            return cx, cy

        # Partículas
        for p in self.pf.particles:
            px, py = w2p(p[0], p[1])
            if 0 <= px < cells_x and 0 <= py < cells_y:
                cv2.circle(img, (px, py), 1, (0, 200, 255), -1)

        # Robot
        rx, ry = w2p(robot_x, robot_y)
        cv2.circle(img, (rx, ry), 5, (255, 80, 0), -1)
        ax = int(rx + 15 * math.cos(robot_theta))
        ay = int(ry - 15 * math.sin(robot_theta))
        cv2.arrowedLine(img, (rx, ry), (ax, ay), (255, 255, 255), 2, tipLength=0.4)

        # Info de pose
        cv2.putText(img,
            f'x={robot_x:.2f}m  y={robot_y:.2f}m  th={math.degrees(robot_theta):.1f}deg',
            (10, cells_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Escalar imagen para que sea visible
        scale = max(1, 600 // max(cells_x, cells_y))
        if scale > 1:
            img = cv2.resize(img, (cells_x * scale, cells_y * scale),
                            interpolation=cv2.INTER_NEAREST)

        cv2.imshow('SLAM Map', img)
        cv2.waitKey(1)

# ═══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if SHOW_MAP:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
