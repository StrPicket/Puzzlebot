import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
from cv2 import aruco
from cv_bridge import CvBridge
import numpy as np
import math

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)

try:
    parameters = aruco.DetectorParameters()
except AttributeError:
    parameters = aruco.DetectorParameters_create()


class centerAruco(Node):
    def __init__(self):
        super().__init__('center_aruco')

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, '/aruco/image_detected', 10)

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
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile
        )

        self.timer_odom = self.create_timer(1 / 100, self.odometria)
        self.timer_ctrl = self.create_timer(1 / 50,  self.control)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.wr = Float32()
        self.wl = Float32()
        self.w_robot = 0.0
        self.v_robot = 0.0

        self.radio   = 0.0505
        self.lenght  = 0.183

        self.Kp_v = 0.15
        self.Ki_v = 0.4

        self.int_error_v = 0.0

        self.Kp_w = 0.1
        self.Kv_w = 0.05

        self.stop_ratio   = 0.25 
        self.close_enough = False

        self.camera_width  = 640
        self.camera_height = 480
        self.img_width     = self.camera_width

        self.latest_frame  = None
        self.latest_header = None
        self.cx = None
        self.cy = None

        self.last_time_odom = self.get_clock().now()
        self.last_time_control = self.get_clock().now()

        self.get_logger().info('centerAruco node iniciado')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.resize(frame, (self.camera_width, self.camera_height))
            frame = cv2.flip(frame, -1)
            self.latest_frame  = frame
            self.latest_header = msg.header
        except Exception as e:
            self.get_logger().error(f'Error en image_callback: {e}')
            self.latest_frame  = None
            self.latest_header = None

    def encR_callback(self, msg: Float32):
        self.wr = msg

    def encL_callback(self, msg: Float32):
        self.wl = msg

    def process_aruco(self):
        self.cx, self.cy = None, None

        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for m_idx in range(len(corners)):
                pts = np.squeeze(corners[m_idx])
                if pts.shape != (4, 2):
                    continue

                for i, p in enumerate(pts):
                    px, py = int(p[0]), int(p[1])
                    cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(i), (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                cx_m = int(np.mean(pts[:, 0]))
                cy_m = int(np.mean(pts[:, 1]))
                cv2.circle(frame, (cx_m, cy_m), 6, (0, 255, 0), -1)

                avg_side_px = np.mean([
                    np.linalg.norm(pts[0] - pts[1]),
                    np.linalg.norm(pts[1] - pts[2]),
                    np.linalg.norm(pts[2] - pts[3]),
                    np.linalg.norm(pts[3] - pts[0]),
                ])
                self.ratio = avg_side_px / self.img_width
 
                label_color = (0, 255, 0) if self.ratio >= self.stop_ratio else (0, 255, 255)
                cv2.putText(frame,
                            f"ID:{int(ids[m_idx][0])}  ratio:{self.ratio:.2f}/{self.stop_ratio:.2f}",
                            (int(pts[0][0]), int(pts[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2)

                if self.cx is None:
                    self.cx, self.cy = cx_m, cy_m
                    self.close_enough = self.ratio >= self.stop_ratio
        else:
            cv2.putText(frame, 'No ArUco detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if self.latest_header is not None:
                out_msg.header = self.latest_header
            self.image_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'Error publicando imagen anotada: {e}')

        cv2.imshow("Marker Detection", frame)
        cv2.waitKey(1)

    def odometria(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_odom).nanoseconds * 1e-9
        self.last_time_odom = current_time

        if dt <= 0:
            return

        v_r    = self.radio * self.wr.data
        v_l    = self.radio * self.wl.data
        V_avg  = (v_r + v_l) / 2.0
        W_robot = (v_r - v_l) / self.lenght

        self.v_robot = 0.15 * self.v_robot + 0.85 * V_avg
        self.w_robot = 0.15 * self.w_robot + 0.85 * W_robot

        self.x     += V_avg * math.cos(self.theta) * dt
        self.y     += V_avg * math.sin(self.theta) * dt
        self.theta += W_robot * dt
        self.theta  = (self.theta + math.pi) % (2 * math.pi) - math.pi

    def control(self):
        self.process_aruco()

        cmd = Twist()

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_control).nanoseconds * 1e-9
        self.last_time_control = current_time
        dt = min(dt, 0.1)

        if self.cx is None:
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        error_v = self.stop_ratio - self.ratio
        error_w = (self.cx - self.img_width / 2.0) / (self.img_width / 2.0)

        self.int_error_v += error_v * dt
        self.int_error_v = max(min(self.int_error_v, 1.0), -1.0)

        if abs(error_w) < 0.05 and self.close_enough:
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            self.int_error_d = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info('ArUco centrado y cerca — robot detenido')
            return

        u_v = self.Ki_v * self.int_error_v - self.Kp_v * self.v_robot
        u_v = max(min(u_v, 0.7), -0.7)

        u_w = self.Kp_w * error_w - self.Kv_w * self.w_robot
        u_w = max(min(u_w, 0.2), -0.2)

        if abs(error_w) > 0.12:
            cmd.linear.x = 0.0
            cmd.angular.z = -u_w
        else:
            cmd.linear.x = u_v
            cmd.angular.z = -u_w

        self.cmd_vel_pub.publish(cmd)

        theta_deg = math.degrees(self.theta) % 360
        self.get_logger().info(
            f'Error_v: {error_v:+.3f} | v_robot: {self.v_robot:.3f} | u_v: {u_v:+.3f}')
        self.get_logger().info(
            f'Error_w: {error_w:+.3f} | θ: {theta_deg:.1f}° | '
            f'w_robot: {self.w_robot:.3f} | u_w: {u_w:+.3f}'
        )


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
