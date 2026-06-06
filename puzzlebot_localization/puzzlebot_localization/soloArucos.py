#!/usr/bin/env python3
"""
aruco_detector.py — Detector ArUco para el Puzzlebot REAL

Misma logica que aruco_detector_sim (la que funciono):
  - solvePnP (IPPE_SQUARE) para distancia y bearing.
  - distance_m = rango HORIZONTAL en el piso.
  - range_sigma_m que crece con la distancia (para la R del EKF).
  - subpixel + desambiguacion de las 2 soluciones de IPPE.

Adaptaciones al robot real:
  - Entrada CompressedImage (/video_source/compressed), JPEG.
  - Calibracion real medida (RMS 0.834 px @ 1280x720), CON distorsion.
  - Correccion de PITCH: la camara va inclinada ~20deg hacia arriba, asi que
    se rota el tvec a un frame nivelado antes de sacar el rango horizontal.
  - equalizeHist ON (la imagen real con JPEG/iluminacion lo agradece).
  - params de deteccion con umbral adaptativo (camara real lo necesita).

Marker_size = 0.096 (cuadro negro impreso, medido).

Publica /aruco/detections (std_msgs/String, JSON list).
""" 

import json
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# ── calibracion REAL (23 imgs, RMS 0.834 px @ 1280x720) ────────────
# fx,fy subidos ~4% (x1.04) tras verificar distancias: el tablero no media
# exactamente 3.0cm, lo que dejaba fx ~4% bajo (distancias ~5% cortas).
CAMERA_MATRIX = np.array([
    [1.03795641e+03,   0.0,         6.36200746e+02],
    [  0.0,         1.03634881e+03,  3.81386102e+02],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array([[0.00383057, 0.1087906, -1.68623574, 3.76464743]], dtype=np.float64)

ARUCO_DICT = cv2.aruco.DICT_4X4_50
VALID_IDS  = set(range(0, 11)) | {19, 31}


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        self.declare_parameter('image_topic', '/video_source/compressed')
        self.declare_parameter('detections_topic', '/aruco/detections')
        self.declare_parameter('marker_size', 0.096)   # cuadro negro medido
        self.declare_parameter('camera_pitch_deg', 0.0)  # inclinacion arriba (refinar)
        self.declare_parameter('flip_mode', 0)         # gstreamer ya publica derecha
        self.declare_parameter('equalize', True)
        self.declare_parameter('min_marker_px', 8.0)
        self.declare_parameter('max_range_m', 0.0)
        self.declare_parameter('pixel_noise_px', 1.0)
        self.declare_parameter('gt_distance', 0.0)

        self.image_topic    = self.get_parameter('image_topic').value
        self.det_topic      = self.get_parameter('detections_topic').value
        self.marker_size    = float(self.get_parameter('marker_size').value)
        pitch_deg           = float(self.get_parameter('camera_pitch_deg').value)
        self.flip_mode      = int(self.get_parameter('flip_mode').value)
        self.equalize       = bool(self.get_parameter('equalize').value)
        self.min_marker_px  = float(self.get_parameter('min_marker_px').value)
        self.max_range_m    = float(self.get_parameter('max_range_m').value)
        self.pixel_noise_px = float(self.get_parameter('pixel_noise_px').value)
        self.gt_distance    = float(self.get_parameter('gt_distance').value)

        self.K = CAMERA_MATRIX
        self.D = DIST_COEFFS

        # rotacion para nivelar el frame de la camara (pitch arriba)
        p = math.radians(pitch_deg)
        self.R_level = np.array([
            [1, 0,           0          ],
            [0, math.cos(p), -math.sin(p)],
            [0, math.sin(p),  math.cos(p)],
        ], dtype=np.float64)

        self._build_obj_pts()
        self._setup_detector()

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_cb, sensor_qos)
        self.pub = self.create_publisher(String, self.det_topic, 10)
        self.pub_img = self.create_publisher(CompressedImage, '/aruco/image_detected/compressed', 10)

        self.get_logger().info(
            f'aruco_detector REAL listo | marker={self.marker_size*100:.1f}cm | '
            f'pitch={pitch_deg:.0f}deg | fx={self.K[0,0]:.0f} | flip={self.flip_mode}')
        if self.gt_distance > 0:
            self.get_logger().warn(
                f'>> DIAGNOSTICO ON: deja UN marcador a {self.gt_distance:.3f} m y revisa el log')

    def _undistort_points(self, img_pts):
        """Proyecta puntos fisheye al plano normalizado pinhole equivalente."""
        pts = img_pts.reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.fisheye.undistortPoints(pts, self.K, self.D, P=self.K)
        return undist.reshape(-1, 2).astype(np.float32)

    def _build_obj_pts(self):
        h = self.marker_size / 2.0
        self.obj_pts = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float32)

    def _setup_detector(self):
        def tune(p):
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 53
            p.adaptiveThreshWinSizeStep = 4
            p.minMarkerPerimeterRate = 0.01
            p.maxMarkerPerimeterRate = 4.0
            p.polygonalApproxAccuracyRate = 0.03
            p.minCornerDistanceRate = 0.03
            p.minDistanceToBorder = 2
            try:
                p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                p.cornerRefinementWinSize = 5
                p.cornerRefinementMaxIterations = 30
                p.cornerRefinementMinAccuracy = 0.01
            except Exception:
                pass
            return p
        try:
            d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            self.detector = cv2.aruco.ArucoDetector(d, tune(cv2.aruco.DetectorParameters()))
            self.use_new_api = True
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
            self.aruco_params = tune(cv2.aruco.DetectorParameters_create())
            self.detector = None
            self.use_new_api = False

    def _fix_orientation(self, frame):
        m = self.flip_mode
        if m == 1: return cv2.rotate(frame, cv2.ROTATE_180)
        if m == 2: return cv2.flip(frame, 1)
        if m == 3: return cv2.flip(frame, 0)
        if m == 4: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if m == 5: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def image_cb(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            frame = self._fix_orientation(frame)
        except Exception as e:
            self.get_logger().warn(f'decode error: {e}')
            return
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.equalize:
            gray = cv2.equalizeHist(gray)

        if self.use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

        detections = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if int(marker_id) not in VALID_IDS:
                    continue
                det = self._solve_one(corners[i], int(marker_id))
                if det is not None:
                    detections.append(det)

        out = String()
        out.data = json.dumps([{k: v for k, v in d.items() if not k.startswith('_')}
                               for d in detections])
        self.pub.publish(out)

        for d in detections:
            self._draw(frame, d)
        try:
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            img_msg = CompressedImage()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.format = "jpeg"
            img_msg.data = buffer.tobytes()
            self.pub_img.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f'publish image error: {e}')

        

    def _solve_one(self, marker_corners, marker_id):
        img_pts = marker_corners.reshape((4, 2)).astype(np.float32)
        size_px = self._size_px(img_pts)
        if size_px < self.min_marker_px:
            return None
        
        img_pts_undist = self._undistort_points(img_pts)

        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_pts, img_pts_undist,   # ← puntos corregidos
                self.K, None,                   # ← D=None (ya sin distorsion)
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception:
            return None
        if n == 0:
            return None

        errs = np.array(reproj).ravel()
        order = np.argsort(errs)
        rvec, tvec = rvecs[order[0]], tvecs[order[0]]
        reproj_err = float(errs[order[0]])

        # nivelar el tvec por el pitch de la camara -> frame con z horizontal
        t_lvl = (self.R_level @ tvec.reshape(3)).ravel()
        tx, ty, tz = float(t_lvl[0]), float(t_lvl[1]), float(t_lvl[2])

        range_xz = math.sqrt(tx * tx + tz * tz)        # horizontal real (piso)
        range_3d = float(np.linalg.norm(tvec))          # 3D total (no depende del pitch)
        if self.max_range_m > 0 and range_xz > self.max_range_m:
            return None

        yaw_deg = float(np.degrees(np.arctan2(tx, tz)))  # bearing horizontal
        gamma_deg = self._gamma_of(rvec)
        gamma_alt_deg = self._gamma_of(rvecs[order[1]]) if n > 1 else -gamma_deg

        fx = self.K[0, 0]
        range_sigma = (range_xz ** 2) / (fx * self.marker_size) * self.pixel_noise_px

        if self.gt_distance > 0:
            ratio = range_xz / self.gt_distance
            implied = self.marker_size / ratio if ratio > 1e-6 else float('nan')
            self.get_logger().info(
                f'ID{marker_id} | medido={range_xz:.3f}m gt={self.gt_distance:.3f}m '
                f'ratio={ratio:.3f} -> marker_size real ~= {implied*100:.2f}cm | '
                f'size_px={size_px:.1f} | reproj={reproj_err:.3f}')

        return {
            'id':            marker_id,
            'corners':       img_pts.tolist(),
            'distance_m':    round(range_xz, 4),     # horizontal -> poseKalman
            'range_3d_m':    round(range_3d, 4),
            'range_sigma_m': round(range_sigma, 4),
            'yaw_deg':       round(yaw_deg, 2),
            'gamma_deg':     round(gamma_deg, 2),
            'gamma_alt_deg': round(gamma_alt_deg, 2),
            'aruco_x_cam':   round(tx, 3),           # ya nivelado
            'aruco_y_cam':   round(ty, 3),
            'aruco_z_cam':   round(tz, 3),
            'size_px':       round(size_px, 2),
            'reproj_err':    round(reproj_err, 4),
            '_rvec': rvec, '_tvec': tvec, '_pts': img_pts,
        }


    def _gamma_of(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        wrapped = (np.arctan2(R[0, 2], R[2, 2]) + math.pi) % (2 * math.pi) - math.pi
        return float(np.degrees(wrapped))

    def _size_px(self, img_pts):
        right = np.linalg.norm(img_pts[2] - img_pts[1])
        left  = np.linalg.norm(img_pts[0] - img_pts[3])
        return float((right + left) / 2.0)

    def _draw(self, frame, d):
        pts = d['_pts'].astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        cv2.drawFrameAxes(frame, self.K, None, d['_rvec'], d['_tvec'], self.marker_size * 0.5, 2)
        cv2.putText(frame, f"ID{d['id']} {d['distance_m']:.2f}m yaw={d['yaw_deg']:.0f}",
                    (max(cx - 90, 5), max(cy - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
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