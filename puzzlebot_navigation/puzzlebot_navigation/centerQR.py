import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np
import math

# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRACIÓN DE CÁMARA
# ═══════════════════════════════════════════════════════════════════════════

CAMERA_MATRIX = np.array([
    [771.25742667,   0.0,         684.88203376],
    [  0.0,         773.15472704,  361.72143901],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[-4.12196743e-01,  2.39129843e-01,  9.29550695e-03,  6.35843547e-05, -7.68077937e-02]],
    dtype=np.float64
)

# Lado físico del QR en metros — ajusta según tu código impreso
QR_W = 0.09
QR_H = 0.09

class centerQR(Node):
    def __init__(self):
        super().__init__('center_qr')

        # ── Publishers / Subscribers ──────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_pub = self.create_publisher(CompressedImage, '/qr/image_detected/compressed', 10)
        self.qr_detected_pub = self.create_publisher(Bool, '/qr/detected', 10)
        self.qr_centered_pub = self.create_publisher(Bool, '/qr/centered', 10)

        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_callback, qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_callback, qos.qos_profile_sensor_data)

        self.image_sub = self.create_subscription(
            CompressedImage, '/video_source/compressed', self.image_callback,
            qos.qos_profile_sensor_data)
        
        self.enable_sub = self.create_subscription(
            Bool, '/center_qr/enable', self._enable_cb, 10)

        self.timer_odom = self.create_timer(1 / 100, self.odometria)
        self.timer_ctrl = self.create_timer(1 / 20,  self.control)

        self.enabled = False

        # ── Estado odométrico ─────────────────────────────────────────────
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.wr = Float32()
        self.wl = Float32()
        self.w_robot = 0.0
        self.v_robot = 0.0

        self.radio  = 0.0505
        self.lenght = 0.183

        # ── Ganancias control lineal (ratio QR) ───────────────────────────
        self.Kp_v = 0.15
        self.Ki_v = 0.25
        self.int_error_r = 0.0

        # ── Ganancias control angular ─────────────────────────────────────
        # Kp_w  → error de centrado en píxeles (igual que antes)
        # Kp_angle → error de perpendicularidad (ángulo rvec)
        self.Kp_w     = 0.08
        self.Kv_w     = 0.05
        self.Kp_angle = 0.30   # ganancia perpendicularidad (ajustable)

        # ── Parámetros de detención ───────────────────────────────────────
        # El robot se detiene cuando:
        #   ratio      >= stop_ratio           (está suficientemente cerca)
        #   |error_w|  <  center_tol           (centrado en X)
        #   |error_ang|<  angle_tol            (perpendicular al QR)
        self.stop_ratio   = 0.2
        self.center_tol   = 0.3          # fracción del semi-ancho de imagen
        self.angle_tol    = math.radians(3)  # 3°

        # ── Imagen ───────────────────────────────────────────────────────
        self.camera_width  = 1280
        self.camera_height = 720
        self.img_width     = self.camera_width
        self.new_frame = False
        self.frame_count = 0

        # ── Detector QR ───────────────────────────────────────────────────
        self.qr_detector = cv2.QRCodeDetector()

        # Puntos 3-D del QR en su propio marco (mismo orden que OpenCV devuelve):
        #   0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
        hw = QR_W / 2.0
        hh = QR_H / 2.0
        self.obj_points = np.array([
            [-hw,  hh, 0.0],
            [ hw,  hh, 0.0],
            [ hw, -hh, 0.0],
            [-hw, -hh, 0.0],
        ], dtype=np.float64)

        # ── Estado de detección ───────────────────────────────────────────
        self.latest_frame  = None
        self.latest_header = None
        self.cx            = None
        self.cy            = None
        self.ratio         = 0.0
        self.close_enough  = False
        self.error_angle   = 0.0   # ángulo perpendicularidad en radianes

        self.last_time_odom    = self.get_clock().now()
        self.last_time_control = self.get_clock().now()

        self.get_logger().info('centerQR node iniciado')

    # ── Callbacks imagen / encoders ───────────────────────────────────────

    def _enable_cb(self, msg: Bool):
        self.enabled = msg.data
        if not self.enabled:
            self.cx = None  # limpiar estado al desactivar

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.latest_frame  = frame
            self.latest_header = msg.header
            self.new_frame = True
        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')
            self.latest_frame  = None
            self.latest_header = None

    def encR_callback(self, msg: Float32):
        self.wr = msg

    def encL_callback(self, msg: Float32):
        self.wl = msg

    # ── Detección y estimación de pose del QR ────────────────────────────

    def process_qr(self):
        """
        Detecta el QR, calcula:
          - self.cx / self.cy : centroide en píxeles
          - self.ratio        : tamaño relativo (lado_px / img_width)
          - self.close_enough : True si ratio >= stop_ratio
          - self.error_angle  : ángulo Y de perpendicularidad (rad)
        """
        self.cx, self.cy = None, None
        self.error_angle  = 0.0

        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()

        scale = 0.5
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(small)

        # Detección — OpenCV QRCodeDetector.detect devuelve (bool, points)
        # points tiene forma (1, 4, 2) si se detecta, None si no.

        if retval and points is not None:
            pts = points[0]/ scale   # (4, 2) float32: TL, TR, BR, BL

            # Dibujar contorno y esquinas
            pts_int = pts.astype(int)
            cv2.polylines(frame, [pts_int.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            corner_colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)]
            for i, p in enumerate(pts_int):
                cv2.circle(frame, tuple(p), 5, corner_colors[i], -1)
                cv2.putText(frame, str(i), (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, corner_colors[i], 1)

            # Centroide
            cx_m = int(np.mean(pts[:, 0]))
            cy_m = int(np.mean(pts[:, 1]))
            cv2.circle(frame, (cx_m, cy_m), 6, (0, 255, 0), -1)

            # Lado medio en píxeles → ratio de tamaño
            avg_side_px = float(np.mean([
                np.linalg.norm(pts[0] - pts[1]),  # top
                np.linalg.norm(pts[3] - pts[2]),  # bottom
            ]))
            self.ratio        = avg_side_px / self.img_width
            self.close_enough = self.ratio >= self.stop_ratio

            # ── solvePnP para estimar perpendicularidad ───────────────────
            img_pts = pts.astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points, img_pts,
                CAMERA_MATRIX, DIST_COEFFS,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:

                # pts:
                # 0 = top-left
                # 1 = top-right

                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]

                yaw = math.atan2(dy, dx)

                if yaw > math.pi / 2:
                    yaw -= math.pi
                elif yaw < -math.pi / 2:
                    yaw += math.pi

                # Diferencia angular
                diff = abs(yaw - self.error_angle)

                # Normalización circular
                if diff > math.pi:
                    diff = 2 * math.pi - diff

                # Alpha adaptativo
                # Cambios pequeños -> mucho filtrado
                # Cambios grandes -> respuesta rápida
                if diff < math.radians(1):
                    alpha = 0.3
                elif diff < math.radians(3):
                    alpha = 0.1               
                else:
                    alpha = 0.01

                # Low-pass adaptativo
                self.error_angle = (
                    alpha * self.error_angle
                    + (1.0 - alpha) * yaw
                )

                # Dibujar ejes de pose
                cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS,
                                  rvec, tvec, QR_W * 0.5)

                angle_deg = math.degrees(self.error_angle)
            else:
                angle_deg = float('nan')

            # HUD sobre el QR
            label_color = (0, 255, 0) if self.close_enough else (0, 255, 255)
            cv2.putText(frame,
                f"QR ratio:{self.ratio:.2f}/{self.stop_ratio:.2f}  "
                f"angle:{angle_deg:.1f}deg",
                (pts_int[0][0], pts_int[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

            self.cx, self.cy = cx_m, cy_m

            qr_text = decoded_info[0] if decoded_info and len(decoded_info) > 0 else "QR detectado"

            cv2.putText(frame, qr_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        else:
            cv2.putText(frame, 'No QR detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 2)

        # Publicar imagen anotada
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            try:
                msg = CompressedImage()
                msg.header = self.latest_header if self.latest_header is not None else Header()
                msg.format = 'jpeg'
                msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
                self.image_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f'Error publicando imagen: {e}')


    # ── Odometría ─────────────────────────────────────────────────────────

    def odometria(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_odom).nanoseconds * 1e-9
        self.last_time_odom = current_time

        if dt <= 0:
            return

        v_r     = self.radio * self.wr.data
        v_l     = self.radio * self.wl.data
        V_avg   = (v_r + v_l) / 2.0
        W_robot = (v_r - v_l) / self.lenght

        self.v_robot = 0.15 * self.v_robot + 0.85 * V_avg
        self.w_robot = 0.15 * self.w_robot + 0.85 * W_robot

        self.x     += V_avg * math.cos(self.theta) * dt
        self.y     += V_avg * math.sin(self.theta) * dt
        self.theta += W_robot * dt
        self.theta  = (self.theta + math.pi) % (2 * math.pi) - math.pi

    # ── Control ───────────────────────────────────────────────────────────

    def control(self):
        if self.new_frame:          # ← solo procesar si hay frame nuevo
            self.process_qr()
            self.new_frame = False

        # Publicar detección siempre
        det_msg = Bool()
        det_msg.data = self.cx is not None
        self.qr_detected_pub.publish(det_msg)
        
        # Solo controlar si está habilitado
        if not self.enabled:
            return

        cmd = Twist()

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_control).nanoseconds * 1e-9
        self.last_time_control = current_time
        dt = min(dt, 0.1)

        # Sin detección → frenar
        if self.cx is None:
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        # Error de centrado horizontal normalizado [-1, 1]
        error_w = (self.cx - self.img_width / 2.0) / (self.img_width / 2.0)

        # Error de tamaño: positivo → robot lejos del QR (avanzar)
        error_r = self.stop_ratio - self.ratio

        # ── Condición de parada: centrado + cerca + perpendicular ─────────
        if (abs(error_w) < self.center_tol
                and self.close_enough
                and abs(self.error_angle) < self.angle_tol):
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            self.int_error_r = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.qr_centered_pub.publish(Bool(data=True))
            self.get_logger().info(
                'QR centrado y perpendicular — robot detenido')
            return
        else:
            self.qr_centered_pub.publish(Bool(data=False))

        # ── Velocidad lineal ──────────────────────────────────────────────
        # Sólo avanzamos si el QR está razonablemente centrado Y alineado.
        # Si hay mucho error angular o de centrado, giramos primero.
        angle_or_center_large = (abs(error_w) > 0.12
                                 or abs(self.error_angle) > math.radians(15))

        if angle_or_center_large:
            u_v = 0.0
            self.int_error_r = 0.0
        else:
            self.int_error_r += error_r * dt
            self.int_error_r  = max(min(self.int_error_r, 1.0), -1.0)

            u_v = self.Ki_v * self.int_error_r - self.Kp_v * self.ratio
            u_v = max(min(u_v, 0.4), -0.4)

        # ── Velocidad angular ─────────────────────────────────────────────
        # Combina dos fuentes de error:
        #   1) error_w     : centrado en píxeles (centrar el QR en imagen)
        #   2) error_angle : perpendicularidad  (quedar de frente al QR)
        # El signo de cmd.angular.z es -u_w porque la cámara ve al revés.
        u_w = (self.Kp_w * error_w
               + self.Kp_angle * self.error_angle
               - self.Kv_w * self.w_robot)
        u_w = max(min(u_w, 0.25), -0.25)

        cmd.linear.x  = u_v
        cmd.angular.z = -u_w   # signo igual que en centerAruco original

        self.cmd_vel_pub.publish(cmd)

        theta_deg = math.degrees(self.theta) % 360
        self.get_logger().info(
            f'Error_r: {error_r:+.3f} | ratio: {self.ratio:.3f} | u_v: {u_v:+.3f}')
        self.get_logger().info(
            f'Error_w: {error_w:+.3f} | Error_angle: {math.degrees(self.error_angle):+.1f}° | '
            f'θ: {theta_deg:.1f}° | u_w: {u_w:+.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = centerQR()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
