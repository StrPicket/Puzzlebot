import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
from cv2 import aruco
from cv_bridge import CvBridge
import numpy as np
import math

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
parameters = aruco.DetectorParameters()


class centerAruco(Node):
    def __init__(self):
        super().__init__('center_aruco')

        # Publishers / Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_callback, qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_callback, qos.qos_profile_sensor_data)

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/puzzlebot/camera/image_raw/compressed',
            self.image_callback,
            qos_profile
        )

        # Timers
        self.timer_odom = self.create_timer(1 / 100, self.odometria)   # 100 Hz
        self.timer_ctrl = self.create_timer(1 / 50,  self.control)     #  50 Hz

        # ── Robot state ──────────────────────────────────────────────
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0          # rad
        self.wr = Float32()
        self.wl = Float32()
        self.w_robot = 0.0        # velocidad angular filtrada

        self.radio   = 0.0505     # m
        self.lenght  = 0.183      # m (wheel base)

        # ── Control gains ────────────────────────────────────────────
        self.Kp = 0.6             # ganancia proporcional (angular)
        self.Kv = 0.12            # ganancia derivativa (amortiguamiento con w)

        # ── Imagen ───────────────────────────────────────────────────
        self.camera_width  = 640
        self.camera_height = 480
        self.img_width     = self.camera_width

        self.latest_frame = None  # último frame recibido por ROS
        self.cx = None            # centro X del marcador detectado
        self.cy = None

        # ── Odometría ────────────────────────────────────────────────
        self.last_time_control = self.get_clock().now()

        self.get_logger().info('centerAruco node iniciado ✔')

    # ─────────────────────────────────────────────────────────────────
    # Callbacks de sensores
    # ─────────────────────────────────────────────────────────────────

    def image_callback(self, msg: CompressedImage):
        """Decodifica el CompressedImage y lo guarda para process_aruco."""
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame = cv2.resize(frame, (self.camera_width, self.camera_height))
        except Exception as e:
            self.get_logger().error(f'Error en image_callback: {e}')
            self.latest_frame = None

    def encR_callback(self, msg: Float32):
        self.wr = msg

    def encL_callback(self, msg: Float32):
        self.wl = msg

    # ─────────────────────────────────────────────────────────────────
    # Detección ArUco  (llamada desde el timer de control)
    # ─────────────────────────────────────────────────────────────────

    def process_aruco(self):
        """Detecta marcadores ArUco en el último frame disponible."""
        self.cx, self.cy = None, None

        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            for m_idx in range(len(corners)):
                pts = np.squeeze(corners[m_idx])
                if pts.shape != (4, 2):
                    continue

                # Esquinas
                for i, p in enumerate(pts):
                    px, py = int(p[0]), int(p[1])
                    cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(i), (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Centro del marcador
                cx_m = int(np.mean(pts[:, 0]))
                cy_m = int(np.mean(pts[:, 1]))
                cv2.circle(frame, (cx_m, cy_m), 6, (0, 255, 0), -1)

                aruco.drawDetectedMarkers(frame, [corners[m_idx]], ids[m_idx:m_idx + 1])
                cv2.putText(frame, f"ID:{int(ids[m_idx][0])}", (cx_m + 8, cy_m + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Guardamos solo el primer marcador encontrado
                if self.cx is None:
                    self.cx, self.cy = cx_m, cy_m

        cv2.imshow("Marker Detection", frame)
        cv2.waitKey(1)

    # ─────────────────────────────────────────────────────────────────
    # Odometría  (100 Hz)
    # ─────────────────────────────────────────────────────────────────

    def odometria(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_control).nanoseconds * 1e-9
        self.last_time_control = current_time

        if dt <= 0:
            return

        v_r    = self.radio * self.wr.data
        v_l    = self.radio * self.wl.data
        V_avg  = (v_r + v_l) / 2.0
        W_robot = (v_r - v_l) / self.lenght

        # Filtro de paso bajo sobre la velocidad angular
        self.w_robot = 0.2 * self.w_robot + 0.8 * W_robot

        self.x     += V_avg * math.cos(self.theta) * dt
        self.y     += V_avg * math.sin(self.theta) * dt
        self.theta += W_robot * dt
        self.theta  = (self.theta + math.pi) % (2 * math.pi) - math.pi

    # ─────────────────────────────────────────────────────────────────
    # Control  (50 Hz)
    # ─────────────────────────────────────────────────────────────────

    def control(self):
        # Primero actualizamos la detección con el frame más reciente
        self.process_aruco()

        cmd = Twist()

        if self.cx is None:
            # Sin marcador → detener
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        # Error normalizado en [-1, 1]  (0 = centrado)
        error = (self.cx - self.img_width / 2.0) / (self.img_width / 2.0)

        # Zona muerta: si ya está centrado, detener
        if abs(error) < 0.05:
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info('ArUco centrado ✔ — robot detenido')
            return

        # Control PD: proporcional al error + amortiguamiento con w_robot
        u = self.Kp * error - self.Kv * self.w_robot
        u = max(min(u, 0.5), -0.5)   # saturar a ±0.5 rad/s

        cmd.linear.x  = 0.0
        cmd.angular.z = -u            # signo según convención del robot

        self.cmd_vel_pub.publish(cmd)

        theta_deg = math.degrees(self.theta) % 360
        self.get_logger().info(
            f'Error: {error:+.3f} | θ: {theta_deg:.1f}° | '
            f'w_robot: {self.w_robot:.3f} | u: {u:+.3f}'
        )


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = centerAruco()
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