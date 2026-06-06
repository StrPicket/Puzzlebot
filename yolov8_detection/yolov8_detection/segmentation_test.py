#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YoloV8Detection — ROS 2 node for instance segmentation / object detection.

Fixes vs. previous version
───────────────────────────
1. rqt_image_view fix   → publica en 'rgb8' (rqt no siempre renderiza bgr8)
2. Mask alpha acumulativo → overlay se calcula desde frame original cada iteración
3. Timer leak           → destroy_timer() antes de crear uno nuevo
4. Warm-up del modelo   → inferencia dummy en _load_model() para eliminar lag inicial
5. Confidence en msg    → InferenceResult ahora incluye el campo confidence
6. NMS tuneable         → parámetro iou_threshold expuesto
7. Log spam reducido    → logger de detecciones en DEBUG, no INFO
8. Header sync          → imagen de salida reutiliza el header del mensaje entrante
"""

import sys
import os
import threading

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy import qos
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CompressedImage
from yolov8_msgs.msg import InferenceResult, Yolov8Inference


# ── Paleta de colores BGR (hasta 20 clases) ───────────────────────────────────
_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 77, 255,  97), ( 77, 204, 255),
    (163,  77, 255), (255,  77, 204), ( 51, 255, 255), (204, 255,  77),
    (255, 204,  51), ( 77,  77, 255), (204,  77, 255), (255,  77,  77),
    ( 51, 204, 255), ( 77, 255, 204), (255, 255,  51), (255,  51, 204),
    (204, 255, 204), (255, 204, 204), (204, 204, 255), (153, 255, 153),
]

def _class_color(cls_id: int) -> tuple:
    return _PALETTE[cls_id % len(_PALETTE)]


class YoloV8Detection(Node):

    def __init__(self):
        super().__init__('yolov8_detection')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('image_topic',          '/image_raw')
        self.declare_parameter('use_compressed',       False)
        self.declare_parameter('model_name',           'best.pt')
        self.declare_parameter('confidence_threshold', 0.45)   # ligeramente bajo para no perder detecciones
        self.declare_parameter('iou_threshold',        0.45)   # NMS — reducir si hay muchas superposiciones
        self.declare_parameter('update_rate',          15.0)
        self.declare_parameter('mask_alpha',           0.40)
        self.declare_parameter('agnostic_nms',         False)  # True: NMS entre clases
        self.declare_parameter('half_precision',       False)  # True: FP16 (requiere CUDA)

        self._read_all_params()

        # ── Estado interno ────────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self.model        = None
        self._image       = None
        self._image_lock  = threading.Lock()
        # Guardamos el header del último mensaje para sincronizar el output
        self._last_header = None

        # ── Cargar modelo ─────────────────────────────────────────────────────
        self._load_model()

        # ── Publishers ────────────────────────────────────────────────────────
        self.yolov8_pub = self.create_publisher(
            Yolov8Inference, 'Yolov8_Inference', qos.qos_profile_sensor_data)

        # FIX: publicamos en 'rgb8' para compatibilidad con rqt_image_view
        self.image_pub = self.create_publisher(
            Image, 'inference_result', qos.qos_profile_sensor_data)

        # ── Subscribers ───────────────────────────────────────────────────────
        self._create_subscriber()

        # ── Timer de inferencia ───────────────────────────────────────────────
        self._timer = self.create_timer(1.0 / self.update_rate, self._timer_callback)

        # ── Callback de parámetros dinámicos ──────────────────────────────────
        self.add_on_set_parameters_callback(self._param_callback)

        self.get_logger().info(
            f'YoloV8Detection listo | modelo={self.model_name} | '
            f'conf={self.confidence_threshold} | iou={self.iou_threshold} | '
            f'{self.update_rate} Hz'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS DE INICIALIZACIÓN
    # ══════════════════════════════════════════════════════════════════════════

    def _read_all_params(self):
        self.image_topic          = self.get_parameter('image_topic').value
        self.use_compressed       = self.get_parameter('use_compressed').value
        self.model_name           = self.get_parameter('model_name').value
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold        = float(self.get_parameter('iou_threshold').value)
        self.update_rate          = float(self.get_parameter('update_rate').value)
        self.mask_alpha           = float(self.get_parameter('mask_alpha').value)
        self.agnostic_nms         = bool(self.get_parameter('agnostic_nms').value)
        self.half_precision       = bool(self.get_parameter('half_precision').value)

    def _create_subscriber(self):
        if self.use_compressed:
            self.create_subscription(
                CompressedImage,
                f'{self.image_topic}/compressed',
                self._image_callback,
                qos.qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                Image,
                self.image_topic,
                self._image_callback,
                qos.qos_profile_sensor_data,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # CARGA DE MODELO
    # ══════════════════════════════════════════════════════════════════════════

    def _load_model(self):
        try:
            pkg_dir    = get_package_share_directory('yolov8_detection')
            model_path = os.path.join(pkg_dir, 'models', self.model_name)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Modelo no encontrado: {model_path}')

            self.model = YOLO(model_path)

            if self.half_precision:
                self.model.model.half()

            self.get_logger().info(
                f'Modelo cargado: {self.model_name} | '
                f'clases ({len(self.model.names)}): '
                + ', '.join(f'{k}={v}' for k, v in sorted(self.model.names.items()))
            )

            task = getattr(self.model, 'task', None)
            if task and task != 'segment':
                self.get_logger().warn(
                    f'task="{task}" (se esperaba "segment"). '
                    'Las máscaras pueden no estar disponibles.'
                )

            # FIX: warm-up para eliminar el lag de la primera inferencia real
            self.get_logger().info('Realizando warm-up del modelo...')
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, conf=self.confidence_threshold, verbose=False)
            self.get_logger().info('Warm-up completado.')

        except Exception as e:
            self.get_logger().error(f'Error cargando modelo: {e}')
            raise

    # ══════════════════════════════════════════════════════════════════════════
    # CALLBACK DE IMAGEN
    # ══════════════════════════════════════════════════════════════════════════

    def _image_callback(self, msg):
        try:
            if self.use_compressed:
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            with self._image_lock:
                self._image       = frame
                self._last_header = msg.header   # FIX: guardamos header original

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')
        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')

    # ══════════════════════════════════════════════════════════════════════════
    # TIMER — INFERENCIA
    # ══════════════════════════════════════════════════════════════════════════

    def _timer_callback(self):
        with self._image_lock:
            if self._image is None or self.model is None:
                return
            frame        = self._image.copy()
            last_header  = self._last_header

        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                agnostic_nms=self.agnostic_nms,
                verbose=False,
            )

            annotated      = self._draw_results(frame, results)
            inference_msg  = self._build_inference_msg(results)

            # FIX: convertir a RGB antes de publicar → rqt_image_view lo renderiza correctamente
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            out_msg = self.bridge.cv2_to_imgmsg(annotated_rgb, encoding='rgb8')

            # FIX: reutilizar header original para mantener sincronía de timestamps
            if last_header is not None:
                out_msg.header = last_header
            else:
                out_msg.header.stamp    = self.get_clock().now().to_msg()
                out_msg.header.frame_id = 'camera_frame'

            self.image_pub.publish(out_msg)
            self.yolov8_pub.publish(inference_msg)

        except Exception as e:
            self.get_logger().error(f'timer_callback: {e}')

    # ══════════════════════════════════════════════════════════════════════════
    # DIBUJO
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_results(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Dibuja máscaras, contornos, bounding boxes y etiquetas.

        FIX: el overlay se calcula siempre desde 'frame' original, no desde
        'output' acumulado → evita distorsión de color cuando hay múltiples
        detecciones superpuestas.
        """
        output = frame.copy()
        h, w   = frame.shape[:2]

        # ── Paso 1: aplicar TODOS los rellenos de máscara de una sola vez ──────
        if True:  # bloque para claridad
            overlay = frame.copy()
            for r in results:
                if r.boxes is None or r.masks is None:
                    continue
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls[0])
                    color  = _class_color(cls_id)

                    if i < len(r.masks.data):
                        mask_raw = r.masks.data[i].cpu().numpy()
                        mask_bin = cv2.resize(mask_raw, (w, h),
                                              interpolation=cv2.INTER_LINEAR) > 0.5
                        overlay[mask_bin] = color

            # FIX: blend desde el frame original (no acumulativo)
            output = cv2.addWeighted(overlay, self.mask_alpha,
                                     frame,   1.0 - self.mask_alpha, 0)

        # ── Paso 2: contornos, bboxes y etiquetas (sin alpha) ─────────────────
        for r in results:
            if r.boxes is None:
                continue
            for i, box in enumerate(r.boxes):
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = f'{self.model.names[cls_id]} {conf:.2f}'
                color  = _class_color(cls_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Contorno de máscara
                if r.masks is not None and i < len(r.masks.data):
                    mask_raw = r.masks.data[i].cpu().numpy()
                    mask_bin = cv2.resize(mask_raw, (w, h),
                                          interpolation=cv2.INTER_LINEAR) > 0.5
                    contours, _ = cv2.findContours(
                        mask_bin.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(output, contours, -1, color, 2)

                # Bounding box
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                # Etiqueta con fondo
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(output,
                               (x1, y1 - th - baseline - 4),
                               (x1 + tw + 4, y1),
                               color, cv2.FILLED)
                cv2.putText(output, label,
                            (x1 + 2, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 2, cv2.LINE_AA)

        return output

    # ══════════════════════════════════════════════════════════════════════════
    # MENSAJE DE INFERENCIA
    # ══════════════════════════════════════════════════════════════════════════

    def _build_inference_msg(self, results) -> Yolov8Inference:
        msg = Yolov8Inference()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                coords = box.xyxy[0].cpu().numpy()

                det            = InferenceResult()
                det.class_name = self.model.names[cls_id]
                det.left       = int(coords[0])
                det.top        = int(coords[1])
                det.right      = int(coords[2])
                det.bottom     = int(coords[3])
                # FIX: publicar confidence si el campo existe en el mensaje
                if hasattr(det, 'confidence'):
                    det.confidence = conf
                msg.yolov8_inference.append(det)

                # FIX: nivel DEBUG para no saturar la consola en producción
                self.get_logger().debug(
                    f'[{det.class_name}] conf={conf:.2f} '
                    f'bbox=({det.left},{det.top},{det.right},{det.bottom})'
                )

        return msg

    # ══════════════════════════════════════════════════════════════════════════
    # PARÁMETROS DINÁMICOS
    # ══════════════════════════════════════════════════════════════════════════

    def _param_callback(self, params: list) -> SetParametersResult:
        model_changed = False
        rate_changed  = False

        for p in params:
            if p.name == 'image_topic':
                if not isinstance(p.value, str) or not p.value.strip():
                    return SetParametersResult(successful=False,
                                               reason='image_topic no puede estar vacío.')
                self.image_topic = p.value

            elif p.name == 'use_compressed':
                if not isinstance(p.value, bool):
                    return SetParametersResult(successful=False,
                                               reason='use_compressed debe ser bool.')
                self.use_compressed = p.value

            elif p.name == 'model_name':
                if not isinstance(p.value, str) or not p.value.strip():
                    return SetParametersResult(successful=False,
                                               reason='model_name no puede estar vacío.')
                if p.value != self.model_name:
                    self.model_name = p.value
                    model_changed   = True

            elif p.name == 'confidence_threshold':
                if not (isinstance(p.value, (int, float)) and 0.0 <= p.value <= 1.0):
                    return SetParametersResult(successful=False,
                                               reason='confidence_threshold debe estar entre 0 y 1.')
                self.confidence_threshold = float(p.value)

            elif p.name == 'iou_threshold':
                if not (isinstance(p.value, (int, float)) and 0.0 <= p.value <= 1.0):
                    return SetParametersResult(successful=False,
                                               reason='iou_threshold debe estar entre 0 y 1.')
                self.iou_threshold = float(p.value)

            elif p.name == 'update_rate':
                if not (isinstance(p.value, (int, float)) and p.value > 0.0):
                    return SetParametersResult(successful=False,
                                               reason='update_rate debe ser > 0.')
                self.update_rate = float(p.value)
                rate_changed     = True

            elif p.name == 'mask_alpha':
                if not (isinstance(p.value, (int, float)) and 0.0 <= p.value <= 1.0):
                    return SetParametersResult(successful=False,
                                               reason='mask_alpha debe estar entre 0 y 1.')
                self.mask_alpha = float(p.value)

            elif p.name == 'agnostic_nms':
                if not isinstance(p.value, bool):
                    return SetParametersResult(successful=False,
                                               reason='agnostic_nms debe ser bool.')
                self.agnostic_nms = p.value

        # FIX: destruir el timer antiguo correctamente antes de crear uno nuevo
        if rate_changed and hasattr(self, '_timer') and self._timer:
            self.destroy_timer(self._timer)
            self._timer = self.create_timer(1.0 / self.update_rate, self._timer_callback)

        if model_changed:
            try:
                self._load_model()
            except Exception as e:
                return SetParametersResult(successful=False,
                                           reason=f'No se pudo cargar el modelo: {e}')

        return SetParametersResult(successful=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    try:
        node = YoloV8Detection()
    except Exception as e:
        print(f'[FATAL] No se pudo inicializar el nodo: {e}', file=sys.stderr)
        rclpy.shutdown()
        return

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