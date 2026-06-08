#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YoloDistance — Distancia al objeto YOLO con corrección fisheye y calibración.

Correcciones aplicadas:
  1. Escala del bbox (BBOX_SCALE=0.90): compensa el padding que YOLO agrega.
  2. Undistort fisheye antes de solvePnP.
  3. Corrección lineal calibrada sobre la salida de solvePnP.

Calibración (medidas reales vs pnp promedio):
  real=0.55 m → pnp≈0.560 m
  real=0.77 m → pnp≈0.743 m
  => dist_real = 1.2022 * dist_pnp - 0.1232

Suscripciones:
  /Yolov8_Inference    (Yolov8Inference)

Publicaciones:
  /yolo/distance       (Float32)  — última distancia corregida en metros
  /yolo/detected       (Bool)
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from yolov8_msgs.msg import Yolov8Inference

# ── Parámetros de cámara fisheye ──────────────────────────────────────────────
CAMERA_MATRIX = np.array([
    [1.03795641e+03,   0.0,         6.36200746e+02],
    [  0.0,         1.03634881e+03,  3.81386102e+02],
    [  0.0,           0.0,           1.0           ]
], dtype=np.float64)

# cv2.fisheye requiere D con shape (4, 1)
DIST_COEFFS = np.array(
    [0.00383057, 0.1087906, -1.68623574, 3.76464743],
    dtype=np.float64
).reshape(4, 1)

# ── Objeto objetivo ───────────────────────────────────────────────────────────
OBJ_SIZE    = 0.10    # lado real del objeto cuadrado en metros (10 cm)
MIN_BBOX_PX = 20.0    # lado mínimo del bbox en píxeles para procesar

# ── Corrección 1: escala del bbox ─────────────────────────────────────────────
BBOX_SCALE = 0.90

# ── Corrección 2: calibración lineal sobre pnp ───────────────────────────────
CALIB_SLOPE     =  1.2022
CALIB_INTERCEPT = -0.1232

# ── Clases objetivo ───────────────────────────────────────────────────────────
TARGET_CLASSES = {'Amazon'}   # set vacío = todas las clases


class YoloDistance(Node):

    def __init__(self):
        super().__init__('yolo_distance')

        # ── Publishers ────────────────────────────────────────────────────
        self.dist_pub = self.create_publisher(Float32, '/yolo/distance', 10)
        self.det_pub  = self.create_publisher(Bool,    '/yolo/detected',  10)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Yolov8Inference, 'Yolov8_Inference',
            self._inference_cb, qos.qos_profile_sensor_data)

        # ── Cámara ────────────────────────────────────────────────────────
        self.K = CAMERA_MATRIX
        self.D = DIST_COEFFS

        # ── Puntos 3D del objeto plano (Z = 0), orden TL TR BR BL ─────────
        h = OBJ_SIZE / 2.0
        self.obj_pts = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float32)

        self.prev_gamma   = None
        self.last_distance = 0.0   # última medición válida (publicada siempre)

        self.get_logger().info(
            f'YoloDistance iniciado | clases={TARGET_CLASSES or "todas"} | '
            f'obj={OBJ_SIZE*100:.0f} cm | bbox_scale={BBOX_SCALE:.2f} | '
            f'calib=({CALIB_SLOPE:.4f}·pnp + {CALIB_INTERCEPT:.4f})'
        )

    # ══════════════════════════════════════════════════════════════════════
    # CORRECCIÓN FISHEYE
    # ══════════════════════════════════════════════════════════════════════

    def _undistort_points(self, img_pts: np.ndarray) -> np.ndarray:
        pts    = img_pts.reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.fisheye.undistortPoints(pts, self.K, self.D, P=self.K)
        return undist.reshape(-1, 2).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # ESCALA DEL BBOX
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _scale_bbox(left, top, right, bottom, scale: float):
        cx = (left  + right)  / 2.0
        cy = (top   + bottom) / 2.0
        hw = (right  - left)  / 2.0 * scale
        hh = (bottom - top)   / 2.0 * scale
        return cx - hw, cy - hh, cx + hw, cy + hh

    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _ang_diff(self, a: float, b: float) -> float:
        return abs(self._wrap(a - b))

    def _gamma_rad(self, rvec) -> float:
        R, _ = cv2.Rodrigues(rvec)
        return self._wrap(math.atan2(R[0, 2], R[2, 2]) + math.pi)

    @staticmethod
    def _bbox_to_corners(left, top, right, bottom) -> np.ndarray:
        return np.array([
            [left,  top],
            [right, top],
            [right, bottom],
            [left,  bottom],
        ], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # ESTIMACIÓN DE DISTANCIA
    # ══════════════════════════════════════════════════════════════════════

    def _compute_distance(self, left, top, right, bottom) -> float | None:
        if (right - left) < MIN_BBOX_PX or (bottom - top) < MIN_BBOX_PX:
            self.get_logger().warn(
                f'Bbox demasiado pequeño ({right-left:.0f}×{bottom-top:.0f} px)',
                throttle_duration_sec=1.0)
            return None

        sl, st, sr, sb = self._scale_bbox(left, top, right, bottom, BBOX_SCALE)
        img_pts_und    = self._undistort_points(self._bbox_to_corners(sl, st, sr, sb))

        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_pts,
                img_pts_und,
                self.K,
                np.zeros((4, 1), dtype=np.float64),
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception as e:
            self.get_logger().warn(f'solvePnP: {e}', throttle_duration_sec=1.0)
            return None

        if n == 0:
            return None

        if n >= 2 and self.prev_gamma is not None:
            gammas = [self._gamma_rad(rvecs[i]) for i in range(n)]
            best_i = min(range(n),
                         key=lambda i: self._ang_diff(gammas[i], self.prev_gamma))
        else:
            best_i = int(np.argmin(np.array(reproj).ravel()))

        self.prev_gamma = self._gamma_rad(rvecs[best_i])

        t          = tvecs[best_i].reshape(3)
        dist_pnp   = math.hypot(float(t[0]), float(t[2]))
        dist_final = CALIB_SLOPE * dist_pnp + CALIB_INTERCEPT

        self.get_logger().info(
            f'  pnp={dist_pnp:.3f} m  ->  final={dist_final:.3f} m',
            throttle_duration_sec=0.4)

        return max(dist_final, 0.01)

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACK DE INFERENCIA YOLO
    # ══════════════════════════════════════════════════════════════════════

    def _inference_cb(self, msg: Yolov8Inference):
        dets = msg.yolov8_inference

        if TARGET_CLASSES:
            dets = [d for d in dets if d.class_name in TARGET_CLASSES]

        best      = None
        best_area = 0.0
        for d in dets:
            area = (d.right - d.left) * (d.bottom - d.top)
            if area > best_area:
                best_area = area
                best = d

        dist     = None
        if best is not None:
            dist = self._compute_distance(
                best.left, best.top, best.right, best.bottom)

        detected = dist is not None

        # Actualizar última medición válida y siempre publicar
        if detected:
            self.last_distance = dist

        self.det_pub.publish(Bool(data=detected))
        self.dist_pub.publish(Float32(data=self.last_distance))

        if detected:
            self.get_logger().info(
                f'[DIST] clase={best.class_name}  dist={self.last_distance:.3f} m',
                throttle_duration_sec=0.4)
        else:
            self.get_logger().info(
                'Sin detección válida', throttle_duration_sec=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = YoloDistance()
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