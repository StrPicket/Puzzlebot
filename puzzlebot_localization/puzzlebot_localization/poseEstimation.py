import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
from cv2 import aruco
from cv_bridge import CvBridge
import numpy as np
import math

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

CAMERA_MATRIX = np.array([
    [133.87191654,   0.0,         157.76772928],
    [  0.0,         131.02895435,  93.02396443],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[-0.15698471, -0.61753973, -0.01000248, -0.00749885, 0.7441658]],
    dtype=np.float64
)

MARKER_SIZE = 0.055

ARUCO_MAP = {
    1: (4.20,  3.70,  -math.pi/2),
    2: (4.20,  0.0,   math.pi / 2),
    3: (0.0,   3.70,  -math.pi/2),
    4: (0.6,   0.0,   math.pi / 2),
}

# ─── Parámetros del mapa ──────────────────────────────────────────────────
MAP_SIZE    = 600
MAP_PADDING = 60
COL_BG        = (30,  30,  30)
COL_GRID      = (55,  55,  55)
COL_ARUCO_UNK = (100, 100, 100)
COL_ARUCO_VIS = (50,  220,  50)
COL_ROBOT     = (50,  180, 255)
COL_LINE      = (100, 180, 100)

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


def estimate_robot_pose(tvec: np.ndarray, rvec: np.ndarray,
                        marker_id: int, corners_px: np.ndarray):
    if marker_id not in ARUCO_MAP:
        return None

    mx, my, m_yaw = ARUCO_MAP[marker_id]

    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec

    cam_x = -t_inv[0][0]

    side_px = marker_side_px(corners_px)
    dist_px = dist_from_pixels(side_px)

    if dist_px < 0:
        return None

    cam_z = math.sqrt(max(dist_px**2 - cam_x**2, 0.0))

    dist_h  = math.sqrt(cam_x**2 + cam_z**2)
    bearing = math.atan2(cam_x, cam_z)

    yaw_cam = math.atan2(R_inv[0, 2], R_inv[2, 2])
    robot_theta = wrap_angle(m_yaw - yaw_cam)

    robot_x = mx + cam_z * math.cos(m_yaw) - cam_x * math.sin(m_yaw)
    robot_y = my + cam_z * math.sin(m_yaw) + cam_x * math.cos(m_yaw)

    dx = 0.12
    dy = 0.0
    robot_x_corr = robot_x - dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
    robot_y_corr = robot_y - dx * math.sin(robot_theta) - dy * math.cos(robot_theta)

    return robot_x_corr, robot_y_corr, robot_theta, dist_h, bearing


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN TOP-DOWN
# ═══════════════════════════════════════════════════════════════════════════

_WX_MIN, _WX_MAX = 0.0, 4.8
_WY_MIN, _WY_MAX = 0.0, 3.7
_WX_RANGE = max(_WX_MAX - _WX_MIN, 1.0)
_WY_RANGE = max(_WY_MAX - _WY_MIN, 1.0)


def world_to_map(wx: float, wy: float) -> tuple:
    draw_w = MAP_SIZE - 2 * MAP_PADDING
    draw_h = MAP_SIZE - 2 * MAP_PADDING
    scale = min(draw_w / _WX_RANGE, draw_h / _WY_RANGE)
    px = int(MAP_PADDING + (wx - _WX_MIN) * scale)
    py = int(MAP_SIZE - MAP_PADDING - (wy - _WY_MIN) * scale)
    return px, py


def draw_map(robot_x: float, robot_y: float, robot_theta: float,
             visible_ids: list) -> np.ndarray:
    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), COL_BG, dtype=np.uint8)

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

    p0 = world_to_map(_WX_MIN, _WY_MIN)
    p1 = world_to_map(_WX_MAX, _WY_MAX)
    cv2.rectangle(canvas, p0, p1, (80, 80, 80), 1)

    for mid, (mx, my, mth) in ARUCO_MAP.items():
        px, py = world_to_map(mx, my)
        color = COL_ARUCO_VIS if mid in visible_ids else COL_ARUCO_UNK
        cv2.rectangle(canvas, (px - 10, py - 10), (px + 10, py + 10), color, -1)
        cv2.rectangle(canvas, (px - 10, py - 10), (px + 10, py + 10), (200, 200, 200), 1)
        arrow_len = 18
        ax = int(px + arrow_len * math.cos(mth))
        ay = int(py - arrow_len * math.sin(mth))
        cv2.arrowedLine(canvas, (px, py), (ax, ay), color, 2, tipLength=0.4)
        cv2.putText(canvas, f"ID{mid}", (px + 13, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    rx, ry = world_to_map(robot_x, robot_y)
    for mid in visible_ids:
        if mid in ARUCO_MAP:
            mx, my, _ = ARUCO_MAP[mid]
            mpx, mpy = world_to_map(mx, my)
            cv2.line(canvas, (rx, ry), (mpx, mpy), COL_LINE, 1, cv2.LINE_AA)

    robot_r = 12
    cv2.circle(canvas, (rx, ry), robot_r, COL_ROBOT, -1)
    cv2.circle(canvas, (rx, ry), robot_r, (200, 230, 255), 1)
    arrow_len = 22
    fax = int(rx + arrow_len * math.cos(robot_theta))
    fay = int(ry - arrow_len * math.sin(robot_theta))
    cv2.arrowedLine(canvas, (rx, ry), (fax, fay), (255, 255, 255),
                    2, cv2.LINE_AA, tipLength=0.35)

    cv2.putText(canvas,
        f"x={robot_x:.2f}m  y={robot_y:.2f}m  th={math.degrees(robot_theta):.1f}deg",
        (10, MAP_SIZE - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  NODO ROS 2
# ═══════════════════════════════════════════════════════════════════════════

class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_pose')

        self.image_pub = self.create_publisher(Image, '/aruco/image_detected', 10)
        self.pose_pub  = self.create_publisher(PoseWithCovarianceStamped, '/aruco/pose', 10)

        qos_img = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_img)

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

        # Pose actual (promedio directo de ArUcos visibles)
        self.aruco_x     = 0.0
        self.aruco_y     = 0.0
        self.aruco_theta = 0.0

        self.timer = self.create_timer(1 / 50, self.process_and_publish)

        self.get_logger().info(
            f'aruco_pose iniciado | {len(ARUCO_MAP)} ArUcos en el mapa')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.resize(frame, (self.camera_width, self.camera_height))
            self.latest_frame  = frame
            self.latest_header = msg.header
        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')
            self.latest_frame = None

    def process_and_publish(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        frame = cv2.resize(frame, (self.camera_width, self.camera_height))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        poses_estimadas = []

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for m_idx in range(len(corners)):
                pts = np.squeeze(corners[m_idx])
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
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    result = estimate_robot_pose(tvec, rvec, marker_id, pts)

                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS,
                                      rvec, tvec, MARKER_SIZE * 0.6)

                    t = tvec.flatten()
                    if result is not None:
                        rx, ry, rth, dist_h, bearing = result
                        poses_estimadas.append((rx, ry, rth))
                        cv2.putText(frame,
                            f"ID:{marker_id}  d={dist_h:.2f}m  b={math.degrees(bearing):.1f}deg",
                            (int(pts[0][0]), int(pts[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
                        cv2.putText(frame,
                            f"robot: x={rx:.2f} y={ry:.2f} th={math.degrees(rth):.1f}deg",
                            (int(pts[0][0]), int(pts[0][1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 0), 1)
                    else:
                        cv2.putText(frame,
                            f"ID:{marker_id} (no en mapa)",
                            (cx_m - 20, cy_m - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                        cv2.putText(frame,
                            f"cam: x={t[0]:.2f} y={t[1]:.2f} z={t[2]:.2f}",
                            (int(pts[0][0]), int(pts[0][1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 0), 1)
                else:
                    cv2.putText(frame, f"ID:{marker_id} (solvePnP fail)",
                                (cx_m - 20, cy_m - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            if poses_estimadas:
                # Promedio directo de todos los ArUcos visibles
                self.aruco_x = float(np.mean([p[0] for p in poses_estimadas]))
                self.aruco_y = float(np.mean([p[1] for p in poses_estimadas]))
                sin_avg = np.mean([math.sin(p[2]) for p in poses_estimadas])
                cos_avg = np.mean([math.cos(p[2]) for p in poses_estimadas])
                self.aruco_theta = float(math.atan2(sin_avg, cos_avg))

                self._publish_pose()
                self.get_logger().info(
                    f'pose: x={self.aruco_x:.2f}  y={self.aruco_y:.2f}  '
                    f'th={math.degrees(self.aruco_theta):.1f}deg')

        else:
            cv2.putText(frame, 'No ArUco detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if self.latest_header is not None:
                out_msg.header = self.latest_header
            self.image_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'Error publicando imagen: {e}')

        map_img = draw_map(self.aruco_x, self.aruco_y, self.aruco_theta,
                           [int(ids[m][0]) for m in range(len(corners))]
                           if ids is not None else [])
        cv2.imshow("Mapa Pista", map_img)
        cv2.waitKey(1)

    def _publish_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = self.aruco_x
        msg.pose.pose.position.y = self.aruco_y
        msg.pose.pose.orientation.z = math.sin(self.aruco_theta / 2)
        msg.pose.pose.orientation.w = math.cos(self.aruco_theta / 2)
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
