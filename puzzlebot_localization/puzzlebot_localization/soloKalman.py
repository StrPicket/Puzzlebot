#!/usr/bin/env python3
"""
poseKalman.py — EKF de localizacion Puzzlebot (robot REAL)

Portado de lo que sirvio en sim: modelo de observacion RANGE-BEARING (sin
gamma en las correcciones). Cada marcador es un landmark de posicion conocida;
medimos rango y bearing y corregimos. Eso arregla el bug del "orbitar".

Adaptado al robot real:
  - Odometria desde encoders (/VelocityEncR, /VelocityEncL, rad/s).
  - Publica en /odom (el EKF ES la odometria del robot real) + twist + cov.
  - Init sin suponer heading: del primer marcador saca pose completa
    (heading via gamma de solvePnP, posicion via rango+bearing). Gamma se usa
    SOLO en el instante del init, no en las correcciones continuas.
  - Offset de camara cam_forward=0.066: la medicion sale de la camara, que va
    adelante del eje de giro, asi que el modelo de observacion predice desde
    la posicion de la camara.
  - Piso al sigma de rango (el detector sub-estima de cerca).
  - Gate chi-cuadrado para tirar lecturas basura.

Sub: /aruco/detections (String/JSON), /VelocityEncR, /VelocityEncL (Float32)
Pub: /odom (Odometry)
"""

import json
import math

import numpy as np
import rclpy
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseWithCovarianceStamped


# mapa real medido (esquina inferior izquierda = origen). (x, y, yaw_normal)
# arena 3.6645 x 4.863 m
MAP_OX = 2.825
MAP_OY = 1.925
X_OFF = 0.608
Y_OFF = 3.6645
ARUCO_MAP = {
    0:  (3.757 + X_OFF, Y_OFF - 0.0000, -math.pi / 2),
    1:  (4.845 + X_OFF, Y_OFF - 2.0300,  math.pi),
    2:  (3.786 + X_OFF, Y_OFF - 3.6645,  math.pi / 2),
    3:  (1.050 + X_OFF, Y_OFF - 3.6645,  math.pi / 2),
    4:  (1.090 + X_OFF, Y_OFF - 0.0000, -math.pi / 2),
    5:  (2.530 + X_OFF, Y_OFF - 2.3795,  math.pi),
    6:  (2.530 + X_OFF, Y_OFF - 1.2430,  math.pi),
    7:  (3.590 + X_OFF, Y_OFF - 1.2430,  0.0),
    8:  (3.590 + X_OFF, Y_OFF - 2.3795,  0.0),
    9:  (0.000 + X_OFF, Y_OFF - 0.3650,  0.0),
    10: (0.000 + X_OFF, Y_OFF - 2.8755,  0.0),
} 

# robot real
WHEEL_RADIUS = 0.0475
WHEEL_BASE   = 0.19
CAM_FORWARD  = 0.12          # lente adelante del eje de giro (m)

# ruido de proceso
Q_TRANS_K, Q_TRANS_B = 0.10, 0.005
Q_ROT_K,   Q_ROT_B   = 0.15, 0.01

# ruido de medicion
SIGMA_R_FLOOR = 0.025         # piso del sigma de rango (m) ~ error real de cerca
SIGMA_BEARING = math.radians(2.5)
SIGMA_HEADING_BASE  = math.radians(8.0)   # heading desde gamma: sigma de cerca
SIGMA_HEADING_SLOPE = math.radians(10.0)  # + rad de sigma por metro (lejos pesa menos)
BEARING_SIGN  = -1.0          # yaw_deg(+)=marcador a la derecha -> bearing(-)
GATE_CHI2_1D  = 6.63          # gate heading (1 GL, 99%)
GATE_CHI2_2D  = 9.21          # gate rango-bearing (2 GL, 99%)
MAX_USE_RANGE = 4.0
HEADING_MAX_RANGE = 3.0       # arriba de esto ni se usa el gamma para heading
INIT_MAX_RANGE    = 1.5       # init confiable solo con marcadores asi de cerca

# re-localizacion (recuperarse si el filtro se pierde)
RELOC_STD_M  = 1.0            # si la incertidumbre de posicion supera esto -> relocaliza
RELOC_STREAK = 10             # ciclos seguidos con marcador visible pero todo rechazado


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ══════════════════════════════════════════════════════════════════
#  EKF range-bearing con offset de camara
# ══════════════════════════════════════════════════════════════════

class EKF:
    def __init__(self):
        self.x = np.zeros(3)
        self.P = np.diag([1.0, 1.0, 1.0])
        self.initialized = False

    def init_pose(self, x, y, th, pos_sigma=0.10, th_sigma=math.radians(15)):
        self.x = np.array([x, y, th], dtype=float)
        self.P = np.diag([pos_sigma ** 2, pos_sigma ** 2, th_sigma ** 2])
        self.initialized = True

    def predict(self, v, w, dt):
        if not self.initialized or dt <= 0:
            return
        th = self.x[2]
        if abs(w) < 1e-6:
            dx = v * dt * math.cos(th)
            dy = v * dt * math.sin(th)
            dth = 0.0
            Fx = np.array([[1, 0, -v * dt * math.sin(th)],
                           [0, 1,  v * dt * math.cos(th)],
                           [0, 0, 1]])
        else:
            r = v / w
            dth = w * dt
            th2 = th + dth
            dx = r * (math.sin(th2) - math.sin(th))
            dy = r * (math.cos(th) - math.cos(th2))
            Fx = np.array([[1, 0, r * (math.cos(th2) - math.cos(th))],
                           [0, 1, r * (math.sin(th2) - math.sin(th))],
                           [0, 0, 1]])
        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = wrap_angle(th + dth)
        trans, rot = abs(v) * dt, abs(w) * dt
        Qd = np.diag([(Q_TRANS_K * trans + Q_TRANS_B) ** 2,
                      (Q_TRANS_K * trans + Q_TRANS_B) ** 2,
                      (Q_ROT_K * rot + Q_ROT_B) ** 2])
        self.P = Fx @ self.P @ Fx.T + Qd

    def update_landmark(self, mx, my, z_r, z_b, sigma_r):
        if not self.initialized:
            return None
        x, y, th = self.x
        L = CAM_FORWARD
        c, s = math.cos(th), math.sin(th)
        cx = x + L * c          # posicion de la camara, no del centro del robot
        cy = y + L * s
        dx, dy = mx - cx, my - cy
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        if r < 1e-3:
            return None
        z_hat = np.array([r, wrap_angle(math.atan2(dy, dx) - th)])
        H = np.array([
            [-dx / r, -dy / r,  L * (dx * s - dy * c) / r],
            [ dy / q, -dx / q, -L * (dx * c + dy * s) / q - 1.0],
        ])
        R = np.diag([sigma_r ** 2, SIGMA_BEARING ** 2])
        innov = np.array([z_r - z_hat[0], wrap_angle(z_b - z_hat[1])])
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        if float(innov @ Sinv @ innov) > GATE_CHI2_2D:
            return False
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ innov
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True

    def update_heading(self, z_th, sigma_th):
        # mide theta directo desde la orientacion del marcador (m_yaw + gamma).
        # solo afecta theta (H=[0,0,1]) -> hace el heading observable con 1 marcador
        # sin reconstruir posicion con gamma (por eso no regresa el bug del orbitar).
        if not self.initialized:
            return None
        H = np.array([[0.0, 0.0, 1.0]])
        R = np.array([[sigma_th ** 2]])
        innov = np.array([wrap_angle(z_th - self.x[2])])
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        if float(innov @ Sinv @ innov) > GATE_CHI2_1D:
            return False
        K = self.P @ H.T @ Sinv
        self.x = self.x + (K @ innov)
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True

    @property
    def state(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])


# ══════════════════════════════════════════════════════════════════
#  Nodo
# ══════════════════════════════════════════════════════════════════

class PoseKalmanNode(Node):
    def __init__(self):
        super().__init__('pose_kalman')

        self.declare_parameter('debug', False)
        self.debug = bool(self.get_parameter('debug').value)
        self._dbg_n = 0

        self.pose_pub = self.create_publisher(Odometry, '/odom', 10)
        self.kf_pose_pub = self.create_publisher(PoseWithCovarianceStamped,'/aruco/pose_centered', 10)
        self.create_subscription(String, '/aruco/detections', self.aruco_cb, 10)
        self.create_subscription(Float32, '/VelocityEncR', self.encR_cb, qos.qos_profile_sensor_data)
        self.create_subscription(Float32, '/VelocityEncL', self.encL_cb, qos.qos_profile_sensor_data)

        self.wr = 0.0
        self.wl = 0.0
        self.last_dets = []

        self.ekf = EKF()
        self.odo = np.zeros(3)
        self.cur_v = 0.0
        self.cur_w = 0.0
        self.reject_streak = 0
        self._dr_dist = 0.0      # dead-reckoning acumulado (para calibrar escala)
        self._dr_rot = 0.0

        self.last_time = self.get_clock().now()
        self.create_timer(0.05, self.loop)
        self.get_logger().info('PoseKalman REAL (range-bearing) | esperando primer Aruco')

    def encR_cb(self, msg): self.wr = msg.data
    def encL_cb(self, msg): self.wl = msg.data

    def aruco_cb(self, msg):
        try:
            self.last_dets = json.loads(msg.data)
        except json.JSONDecodeError:
            self.last_dets = []

    def predict_step(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0 or dt > 0.5:
            return
        vr = WHEEL_RADIUS * self.wr
        vl = WHEEL_RADIUS * self.wl
        v = (vr + vl) / 2.0
        w = (vr - vl) / WHEEL_BASE
        self.cur_v, self.cur_w = v, w
        self._dr_dist += v * dt
        self._dr_rot += w * dt

        if self.debug:
            self._dbg_n += 1
            if self._dbg_n % 10 == 0:
                self.get_logger().info(
                    f'wr={self.wr:+.2f} wl={self.wl:+.2f} -> v={v:+.3f} w={math.degrees(w):+.1f}deg/s '
                    f'| acum: dist={self._dr_dist:+.3f}m rot={math.degrees(self._dr_rot):+.1f}deg')

        self.ekf.predict(v, w, dt)
        th = self.odo[2]
        if abs(w) < 1e-6:
            self.odo[0] += v * dt * math.cos(th)
            self.odo[1] += v * dt * math.sin(th)
        else:
            r = v / w
            th2 = th + w * dt
            self.odo[0] += r * (math.sin(th2) - math.sin(th))
            self.odo[1] += r * (math.cos(th) - math.cos(th2))
            self.odo[2] = wrap_angle(th2)

    def valid_dets(self):
        out = []
        for d in self.last_dets:
            mid, rng, yaw = d.get('id'), d.get('distance_m'), d.get('yaw_deg')
            if mid in ARUCO_MAP and rng is not None and yaw is not None \
               and 0.05 < rng <= MAX_USE_RANGE:
                out.append(d)
        return out

    def try_init(self, dets):
        # pose completa del/los marcador(es), sin suponer heading.
        # preferir CERCANOS (gamma y rango confiables). si solo hay lejanos,
        # inicializa igual pero con mucha incertidumbre para que se corrija al
        # moverse, en vez de comprometerse a un gamma lejano (que sale mal).
        ds = sorted(dets, key=lambda d: d['distance_m'])
        close = [d for d in ds if d['distance_m'] <= INIT_MAX_RANGE]
        use = close if close else ds[:1]
        confident = bool(close)
        xs, ys, ss, cc = [], [], [], []
        for d in use:
            mx, my, m_yaw = ARUCO_MAP[d['id']]
            rng = d['distance_m']
            gamma = d.get('gamma_deg')
            if gamma is None:
                continue
            th = wrap_angle(m_yaw + math.radians(gamma))
            beta = BEARING_SIGN * math.radians(d['yaw_deg'])
            world_dir = th + beta                 # rumbo camara->marcador
            cam_x = mx - rng * math.cos(world_dir)
            cam_y = my - rng * math.sin(world_dir)
            xs.append(cam_x - CAM_FORWARD * math.cos(th))
            ys.append(cam_y - CAM_FORWARD * math.sin(th))
            ss.append(math.sin(th)); cc.append(math.cos(th))
        if not xs:
            return
        x0 = float(np.median(xs))
        y0 = float(np.median(ys))
        th0 = math.atan2(float(np.mean(ss)), float(np.mean(cc)))
        pos_sigma = 0.10 if confident else 0.30
        th_sigma = math.radians(15) if confident else math.radians(35)
        self.ekf.init_pose(x0, y0, th0, pos_sigma, th_sigma)
        self.odo = np.array([x0, y0, th0], dtype=float)
        self.get_logger().info(
            f'INICIALIZADO desde {[d["id"] for d in use]} '
            f'({"cercano" if confident else "LEJANO, baja confianza"}) | '
            f'x={x0:.3f} y={y0:.3f} th={math.degrees(th0):.1f}deg')

    def correct_step(self, dets):
        accepted = 0
        for d in dets:
            mid = d['id']
            rng = d['distance_m']
            z_b = BEARING_SIGN * math.radians(d['yaw_deg'])
            sigma_r = d.get('range_sigma_m')
            if sigma_r is None:
                sigma_r = SIGMA_R_FLOOR
            sigma_r = max(float(sigma_r), SIGMA_R_FLOOR)   # piso
            mx, my, m_yaw = ARUCO_MAP[mid]
            if self.ekf.update_landmark(mx, my, rng, z_b, sigma_r):
                accepted += 1
            # heading desde la orientacion del marcador, solo si es confiable
            gamma = d.get('gamma_deg')
            if gamma is not None and rng <= HEADING_MAX_RANGE:
                z_th = wrap_angle(m_yaw + math.radians(gamma))
                sigma_th = SIGMA_HEADING_BASE + SIGMA_HEADING_SLOPE * rng
                self.ekf.update_heading(z_th, sigma_th)

        # divergencia -> re-localizar desde los marcadores visibles
        if dets:
            self.reject_streak = self.reject_streak + 1 if accepted == 0 else 0
            std_pos = math.sqrt(self.ekf.P[0, 0] + self.ekf.P[1, 1])
            if self.reject_streak >= RELOC_STREAK or std_pos > RELOC_STD_M:
                self.get_logger().warn(
                    f'divergencia (rechazos={self.reject_streak}, std={std_pos:.2f}m) '
                    f'-> re-localizando desde {[d["id"] for d in dets]}')
                self.try_init(dets)
                self.reject_streak = 0

    def publish_pose(self):
        if not self.ekf.initialized:
            return
        kx, ky, kth = self.ekf.state
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        msg.pose.pose.position.x = kx - MAP_OX
        msg.pose.pose.position.y = ky - MAP_OY
        msg.pose.pose.orientation.z = math.sin(kth / 2)
        msg.pose.pose.orientation.w = math.cos(kth / 2)
        msg.twist.twist.linear.x = self.cur_v
        msg.twist.twist.angular.z = self.cur_w
        P = self.ekf.P
        cov = [0.0] * 36
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        msg.pose.covariance = cov
        self.pose_pub.publish(msg)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp    = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.pose.position.x = kx - MAP_OX
        pose_msg.pose.pose.position.y = ky - MAP_OY

        pose_msg.pose.pose.orientation.z = math.sin(kth / 2.0)
        pose_msg.pose.pose.orientation.w = math.cos(kth / 2.0)

        pose_msg.pose.covariance = cov

        self.kf_pose_pub.publish(pose_msg)
        

    def loop(self):
        self.predict_step()
        dets = self.valid_dets()

        if not self.ekf.initialized:
            if dets:
                self.try_init(dets)
        else:
            if dets:
                self.correct_step(dets)

        self.publish_pose()



def main(args=None):
    rclpy.init(args=args)
    node = PoseKalmanNode()
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