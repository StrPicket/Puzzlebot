#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YoloDistance — Calcula la distancia al objeto detectado por YOLO usando solvePnP.

Suscripciones:
  /Yolov8_Inference    (Yolov8Inference)  — bboxes con left/top/right/bottom

Publicaciones:
  /yolo/distance       (Float32)          — distancia en metros (0.0 si no hay deteccion)
  /yolo/detected       (Bool)             — True si hay objeto detectado
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from yolov8_msgs.msg import Yolov8Inference

# ── Parametros de camara (fish-eye, calibracion propia) ──────────────────────
CAMERA_MATRIX = np.array([
    [1.03795641e+03,   0.0,         6.36200746e+02],
    [  0.0,         1.03634881e+03,  3.81386102e+02],
    [  0.0,           0.0,           1.0           ]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[0.00383057, 0.1087906, -1.68623574, 3.76464743]],
    dtype=np.float64
)

# ── Objeto objetivo ───────────────────────────────────────────────────────────
OBJ_SIZE       = 0.10          # lado del objeto cuadrado en metros (10 cm)
MIN_BBOX_PX    = 20.0          # lado mínimo del bbox en píxeles para procesar

# ── Clases objetivo ───────────────────────────────────────────────────────────
TARGET_CLASSES = {'Amazon'}    # set vacío = todas las clases


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

        # ── Pose estimation: esquinas 3D del objeto plano (Z = 0) ─────────
        # Orden TL, TR, BR, BL — debe coincidir con _bbox_to_corners()
        h = OBJ_SIZE / 2.0
        self.obj_pts = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float32)

        self.prev_gamma = None   # para selección consistente de solución IPPE

        self.get_logger().info(
            f'YoloDistance iniciado | '
            f'clases={TARGET_CLASSES or "todas"} | obj={OBJ_SIZE * 100:.0f} cm'
        )

    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _ang_diff(self, a: float, b: float) -> float:
        return abs(self._wrap(a - b))

    def _gamma_rad(self, rvec) -> float:
        """Extrae el ángulo de rotación principal del vector de Rodrigues."""
        R, _ = cv2.Rodrigues(rvec)
        return self._wrap(math.atan2(R[0, 2], R[2, 2]) + math.pi)

    # ══════════════════════════════════════════════════════════════════════
    # CONVERSIÓN BBOX → ESQUINAS IMAGEN
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _bbox_to_corners(left, top, right, bottom) -> np.ndarray:
        """Devuelve las 4 esquinas del bbox en orden TL, TR, BR, BL."""
        return np.array([
            [left,  top],
            [right, top],
            [right, bottom],
            [left,  bottom],
        ], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # ESTIMACIÓN DE DISTANCIA vía solvePnP
    # ══════════════════════════════════════════════════════════════════════

    def _compute_distance(self, left, top, right, bottom) -> float | None:
        """
        Estima la distancia (metros) al centro del objeto desde la cámara
        usando solvePnPGeneric con IPPE_SQUARE.

        Devuelve None si el bbox es demasiado pequeño o solvePnP falla.

        La distancia se calcula como:
            dist = sqrt(tx² + tz²)
        donde tx y tz son los componentes horizontal y frontal del vector
        de traslación en el marco de la cámara.
        """
        bbox_w = right  - left
        bbox_h = bottom - top
        if bbox_w < MIN_BBOX_PX or bbox_h < MIN_BBOX_PX:
            self.get_logger().warn(
                f'Bbox demasiado pequeño ({bbox_w:.0f}x{bbox_h:.0f} px) — ignorando',
                throttle_duration_sec=1.0)
            return None

        img_pts = self._bbox_to_corners(left, top, right, bottom)

        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_pts, img_pts,
                CAMERA_MATRIX, DIST_COEFFS,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception as e:
            self.get_logger().warn(
                f'solvePnP error: {e}', throttle_duration_sec=1.0)
            return None

        if n == 0:
            return None

        # Seleccionar la solución más consistente con el frame anterior
        if n >= 2 and self.prev_gamma is not None:
            gammas = [self._gamma_rad(rvecs[i]) for i in range(n)]
            best_i = min(range(n),
                         key=lambda i: self._ang_diff(gammas[i], self.prev_gamma))
        else:
            # Primera vez: tomar la solución con menor error de reproyección
            best_i = int(np.argmin(np.array(reproj).ravel()))

        self.prev_gamma = self._gamma_rad(rvecs[best_i])

        t  = tvecs[best_i].reshape(3)
        tx = float(t[0])   # desplazamiento lateral
        tz = float(t[2])   # desplazamiento frontal (profundidad)

        dist = math.hypot(tx, tz)
        return dist

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACK DE INFERENCIA YOLO
    # ══════════════════════════════════════════════════════════════════════

    def _inference_cb(self, msg: Yolov8Inference):
        dets = msg.yolov8_inference

        # Filtrar por clase objetivo
        if TARGET_CLASSES:
            dets = [d for d in dets if d.class_name in TARGET_CLASSES]

        # Seleccionar el bbox con mayor área (objeto más grande / más cercano)
        best      = None
        best_area = 0.0
        for d in dets:
            area = (d.right - d.left) * (d.bottom - d.top)
            if area > best_area:
                best_area = area
                best = d

        # Calcular distancia
        dist = None
        if best is not None:
            dist = self._compute_distance(
                best.left, best.top, best.right, best.bottom)

        detected = dist is not None

        # Publicar
        self.det_pub.publish(Bool(data=detected))
        self.dist_pub.publish(Float32(data=float(dist) if detected else 0.0))

        # Log
        if detected:
            self.get_logger().info(
                f'[DIST] clase={best.class_name}  dist={dist:.3f} m',
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