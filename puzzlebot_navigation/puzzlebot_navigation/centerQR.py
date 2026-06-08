#!/usr/bin/env python3
import math
import numpy as np
import cv2
import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist


CAMERA_MATRIX = np.array([
    [1.03795641e+03, 0.0, 6.36200746e+02],
    [0.0, 1.03634881e+03, 3.81386102e+02],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST_COEFFS = np.array([[0.00383057, 0.1087906, -1.68623574, 3.76464743]], dtype=np.float64)

QR_SIZE = 0.09
READY_DIST = 0.27

WHITELIST = {'Emezon', 'Wolmar', 'Popsi'}

TURN_SIGN = -1.0

K_DIST = 0.45
K_BEARING = 0.35
K_LAT = 0.35

V_MAX = 0.055
V_REV_MAX = 0.025
W_MAX = 0.04

BEARING_TOL = math.radians(4.5)
LAT_TOL = 0.020
DIST_TOL = 0.025

MIN_MARKER_PX = 20.0
CAMERA_PITCH_DEG = 0.0


class CenterQRVisual(Node):

    def __init__(self):
        super().__init__('center_qr_visual')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.det_pub = self.create_publisher(Bool, '/qr/detected', 10)
        self.cen_pub = self.create_publisher(Bool, '/qr/centered', 10)
        self.dist_pub = self.create_publisher(Float32, '/forklift/distance', 10)
        self.mark_pub = self.create_publisher(String, '/qr/mark_detected', 10)
        self.img_pub = self.create_publisher(
            CompressedImage,
            '/qr/image_detected/compressed',
            10
        )

        self.create_subscription(
            CompressedImage,
            '/video_source/compressed',
            self.image_cb,
            qos.qos_profile_sensor_data
        )

        self.create_subscription(
            Bool,
            '/center_qr/enable',
            self.enable_cb,
            10
        )

        self.enabled = True
        self.qr = cv2.QRCodeDetector()

        self.K = CAMERA_MATRIX
        self.D = DIST_COEFFS

        h = QR_SIZE / 2.0
        self.obj_pts = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float32)

        p = math.radians(CAMERA_PITCH_DEG)
        self.R_level = np.array([
            [1, 0, 0],
            [0, math.cos(p), -math.sin(p)],
            [0, math.sin(p),  math.cos(p)]
        ], dtype=np.float64)

        self.locked_id = ''
        self.last_center = None
        self.lost_count = 0
        self.center_count = 0
        self.center_need = 6

        self.last_w = 0.0
        self.last_v = 0.0
        self.prev_bearing = 0.0

        self.finished = False

        self.get_logger().info('center_qr_visual listo')

    def enable_cb(self, msg):
        self.enabled = msg.data

        if self.enabled:
            self.finished = False
            self.center_count = 0

        else:
            self.stop()
            self.center_count = 0
            self.finished = False

    def stop(self):
        self.cmd_pub.publish(Twist())

    def send_cmd(self, v, w):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    def detect_qr(self, gray):
        gray_u = cv2.fisheye.undistortImage(gray, self.K, self.D, Knew=self.K)

        try:
            retval, infos, points, _ = self.qr.detectAndDecodeMulti(gray_u)
        except Exception as e:
            self.get_logger().warn(f'QR detect error: {e}')
            return None, ''

        if not retval or points is None:
            return None, ''

        candidates = []

        for i in range(len(points)):
            pts = np.asarray(points[i], dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 4:
                continue

            text = ''
            if infos is not None and i < len(infos):
                text = infos[i].strip()

            center = pts[:4].mean(axis=0)
            candidates.append((text, pts[:4], center))

        if not candidates:
            return None, ''

        for text, pts, center in candidates:
            if text in WHITELIST:
                self.locked_id = text
                self.last_center = center
                return pts, text

        if self.last_center is not None:
            best = min(
                candidates,
                key=lambda c: np.linalg.norm(c[2] - self.last_center)
            )
            text, pts, center = best
            self.last_center = center
            return pts, text

        text, pts, center = candidates[0]
        self.last_center = center
        return pts, text

    def compute_pose(self, corners):
        img_pts = corners.reshape(4, 2).astype(np.float32)

        edge = (
            np.linalg.norm(img_pts[2] - img_pts[1]) +
            np.linalg.norm(img_pts[0] - img_pts[3])
        ) / 2.0

        if edge < MIN_MARKER_PX:
            return None

        img_pts_undist = cv2.fisheye.undistortPoints(
            img_pts.reshape(-1, 1, 2),
            self.K,
            self.D,
            P=self.K
        ).reshape(-1, 2).astype(np.float32)

        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_pts,
                img_pts_undist,
                self.K,
                None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
        except Exception:
            return None

        if n == 0:
            return None

        best_i = None
        for i in range(n):
            if tvecs[i][2, 0] > 0:
                if best_i is None or reproj[i] < reproj[best_i]:
                    best_i = i

        if best_i is None:
            best_i = int(np.argmin(np.array(reproj).ravel()))

        rvec = rvecs[best_i]
        tvec = tvecs[best_i]

        t_lvl = (self.R_level @ tvec.reshape(3)).ravel()

        tx = float(t_lvl[0])
        ty = float(t_lvl[1])
        tz = float(t_lvl[2])

        dist = math.sqrt(tx * tx + tz * tz)
        bearing = math.atan2(tx, tz)

        return {
            'tx': tx,
            'ty': ty,
            'tz': tz,
            'dist': dist,
            'bearing': bearing,
            'rvec': rvec,
            'tvec': tvec,
            'pts': img_pts,
            'edge': edge
        }

    def control_qr(self, pose):
        if self.finished:
            self.stop()
            return

        dist = pose['dist']
        bearing = pose['bearing']
        tx = pose['tx']

        e_dist = dist - READY_DIST
        e_lat = tx

        centered_ang = abs(bearing) < BEARING_TOL
        centered_lat = abs(e_lat) < LAT_TOL
        at_dist = abs(e_dist) < DIST_TOL

        self.dist_pub.publish(Float32(data=float(dist)))
        self.mark_pub.publish(String(data=self.locked_id))

        if centered_ang and centered_lat and at_dist:
            self.center_count += 1
            self.stop()

            if self.center_count >= self.center_need:
                self.cen_pub.publish(Bool(data=True))
                self.finished = True
                self.stop()

                self.get_logger().info(
                    f'QR CENTRADO | dist={dist:.3f} m | tx={tx:+.3f} | '
                    f'bearing={math.degrees(bearing):+.2f} deg | id={self.locked_id}',
                    throttle_duration_sec=1.0
                )
            return

        self.center_count = 0
        self.cen_pub.publish(Bool(data=False))

        # Si está muy descentrado, primero gira sin avanzar
        if abs(bearing) > math.radians(12.0):
            v = 0.0
        else:
            v = K_DIST * e_dist
            v = float(np.clip(v, -V_REV_MAX, V_MAX))

        # Control angular combinado
        w_raw = TURN_SIGN * (K_BEARING * bearing + K_LAT * e_lat)

        # Zona muerta: evita micro-oscilaciones
        if abs(bearing) < math.radians(3.0) and abs(e_lat) < 0.018:
            w = 0.0

        # Cerca del QR, giro más suave
        elif dist < 0.32:
            w = float(np.clip(w_raw, -0.025, 0.025))

        # Lejos, giro normal
        else:
            w = float(np.clip(w_raw, -W_MAX, W_MAX))

        self.send_cmd(v, w)

        self.get_logger().info(
            f'ALIGN | dist={dist:.3f} e_dist={e_dist:+.3f} '
            f'tx={tx:+.3f} bearing={math.degrees(bearing):+.2f} '
            f'v={v:+.3f} w={w:+.3f}',
            throttle_duration_sec=0.3
        )

    def draw_debug(self, frame, pose):
        h, w = frame.shape[:2]
        cx = int(self.K[0, 2])

        cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)

        if pose is not None:
            pts = pose['pts'].astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            try:
                cv2.drawFrameAxes(
                    frame,
                    self.K,
                    None,
                    pose['rvec'],
                    pose['tvec'],
                    QR_SIZE * 0.5,
                    2
                )
            except cv2.error:
                pass

            text = (
                f'd={pose["dist"]:.2f}m '
                f'tx={pose["tx"]:+.2f} '
                f'yaw={math.degrees(pose["bearing"]):+.1f}'
            )

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        try:
            _, buffer = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )

            img_msg = CompressedImage()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.format = 'jpeg'
            img_msg.data = buffer.tobytes()
            self.img_pub.publish(img_msg)

        except Exception as e:
            self.get_logger().warn(f'publish image error: {e}')

    def image_cb(self, msg):
        try:
            frame = cv2.imdecode(
                np.frombuffer(msg.data, np.uint8),
                cv2.IMREAD_COLOR
            )
        except Exception as e:
            self.get_logger().warn(f'decode error: {e}')
            return

        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        corners, text = self.detect_qr(gray)
        pose = self.compute_pose(corners) if corners is not None else None

        detected = pose is not None
        self.det_pub.publish(Bool(data=detected))

        if not self.enabled:
            self.stop()
            self.draw_debug(frame, pose)
            return

        if pose is None:
            self.lost_count += 1
            self.center_count = 0
            self.cen_pub.publish(Bool(data=False))

            if self.lost_count < 10:
                self.stop()
            else:
                # búsqueda lenta
                self.send_cmd(0.0, 0.025)

            self.get_logger().warn(
                'QR no detectado',
                throttle_duration_sec=0.5
            )

        else:
            self.lost_count = 0
            if text:
                self.locked_id = text
            self.control_qr(pose)

        self.draw_debug(frame, pose)


def main(args=None):
    rclpy.init(args=args)
    node = CenterQRVisual()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()