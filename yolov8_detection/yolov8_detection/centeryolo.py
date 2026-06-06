#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YoloCenterControl — Tres estados:
  SCAN    → gira buscando la clase objetivo
  CENTRAR → una vez detectada, centra y mantiene 3 s estables
  AVANZAR → avanza recto 2 s
  FIN     → se detiene y publica /yolo/centered=True
"""

import math
import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist
from yolov8_msgs.msg import Yolov8Inference

# ── Ajusta según tu setup ─────────────────────────────────────────────────────
TARGET_CLASSES   = {'Amazon'}   # clases a seguir; set() = todas
IMG_W_REF        = 640.0        # ancho de referencia para normalizar el bbox

SCAN_OMEGA       = 0.06       # velocidad angular de búsqueda (rad/s)
CENTER_OMEGA_MAX = 0.08         # velocidad angular máxima en centrado (rad/s)
K_BEARING        = 1.2          # ganancia proporcional sobre error_x
CENTER_TOL       = 0.08         # |error_x| < este valor = "centrado"
CENTERED_HOLD    = 3.0          # segundos que debe estar centrado (continuo)

FORWARD_SPEED    = 0.08         # velocidad lineal en AVANZAR (m/s)
FORWARD_TIME     = 2.0          # segundos de avance

TURN_SIGN        = -1.0         # cambiar a 1.0 si gira al revés


class YoloCenterControl(Node):

    def __init__(self):
        super().__init__('yolo_center_control')

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cen_pub = self.create_publisher(Bool,  '/yolo/centered', 10)
        self.det_pub = self.create_publisher(Bool,  '/yolo/detected', 10)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Yolov8Inference, 'Yolov8_Inference',
            self._inference_cb, qos.qos_profile_sensor_data)
        self.create_subscription(
            Float32, '/detection/error_x',
            self._error_x_cb, qos.qos_profile_sensor_data)
        self.create_subscription(
            Bool, '/center_yolo/enable', self._enable_cb, 10)
        self.create_subscription(
            Bool, '/center_yolo/stop',   self._stop_cb,   10)

        # ── Estado ────────────────────────────────────────────────────────
        self.state    = 'SCAN'
        self.enabled  = True
        self.estop    = False

        self.detected = False
        self.error_x  = 0.0

        self._centered_since = None   # timestamp del primer frame centrado
        self._forward_t0     = None   # timestamp del inicio del avance

        # ── Scan bidireccional ────────────────────────────────────────────
        self.scan_dir   = 1.0
        self.scan_first = True
        self.scan_t0    = self.get_clock().now()
        self.scan_period = 4.0

        self.create_timer(0.02, self._tick)   # 50 Hz
        self.get_logger().info(
            f'YoloCenterControl listo | clases={TARGET_CLASSES or "todas"} | '
            f'hold={CENTERED_HOLD}s | avance={FORWARD_TIME}s')

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    def _inference_cb(self, msg: Yolov8Inference):
        dets = msg.yolov8_inference
        if TARGET_CLASSES:
            dets = [d for d in dets if d.class_name in TARGET_CLASSES]
        self.detected = bool(dets)
        det_msg = Bool()
        det_msg.data = self.detected
        self.det_pub.publish(det_msg)

    def _error_x_cb(self, msg: Float32):
        self.error_x = float(msg.data)

    def _enable_cb(self, msg: Bool):
        self.enabled = msg.data
        if not self.enabled:
            self._stop()
            self.state = 'SCAN'
            self._centered_since = None
            self._forward_t0     = None

    def _stop_cb(self, msg: Bool):
        self.estop = bool(msg.data)
        if self.estop:
            self._stop()

    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _send(self, v: float, w: float):
        c = Twist()
        c.linear.x  = float(v)
        c.angular.z = float(w)
        self.cmd_pub.publish(c)

    # ══════════════════════════════════════════════════════════════════════
    # TICK PRINCIPAL — 50 Hz
    # ══════════════════════════════════════════════════════════════════════

    def _tick(self):
        if self.estop or not self.enabled:
            self._stop()
            return

        s = self.state
        if   s == 'SCAN':    self._st_scan()
        elif s == 'CENTRAR': self._st_centrar()
        elif s == 'AVANZAR': self._st_avanzar()
        elif s == 'FIN':     self._stop()

    # ══════════════════════════════════════════════════════════════════════
    # ESTADO 1 — SCAN: gira buscando la clase
    # ══════════════════════════════════════════════════════════════════════

    def _st_scan(self):
        if self.detected:
            self.get_logger().info('Objeto detectado — pasando a CENTRAR')
            self._centered_since = None
            self.state = 'CENTRAR'
            return

        # Giro bidireccional (igual que CenterQR)
        now = self.get_clock().now()
        elapsed = (now - self.scan_t0).nanoseconds * 1e-9
        period  = self.scan_period * (0.5 if self.scan_first else 1.0)
        if elapsed > period:
            self.scan_dir   *= -1.0
            self.scan_first  = False
            self.scan_t0     = now
        self._send(0.0, self.scan_dir * SCAN_OMEGA)

    # ══════════════════════════════════════════════════════════════════════
    # ESTADO 2 — CENTRAR: centra el bbox y mantiene 3 s estables
    # ══════════════════════════════════════════════════════════════════════

    def _st_centrar(self):
        # Si se pierde la detección, volver a buscar
        if not self.detected:
            self.get_logger().info('Objeto perdido — volviendo a SCAN')
            self._centered_since = None
            self.state = 'SCAN'
            return

        now      = self.get_clock().now()
        centrado = abs(self.error_x) < CENTER_TOL

        if centrado:
            # Iniciar o mantener el temporizador de estabilidad
            if self._centered_since is None:
                self._centered_since = now
                self.get_logger().info('Centrado — iniciando cuenta de estabilidad')
            held = (now - self._centered_since).nanoseconds * 1e-9
            self._stop()
            self.get_logger().info(
                f'[CENTRAR] error_x={self.error_x:+.3f} '
                f'estable={held:.1f}/{CENTERED_HOLD:.1f}s',
                throttle_duration_sec=0.5)
            if held >= CENTERED_HOLD:
                self.get_logger().info('Centrado estable — pasando a AVANZAR')
                self._centered_since = None
                self._forward_t0     = None
                self.state = 'AVANZAR'
        else:
            # Salió de la tolerancia: resetear temporizador y corregir
            if self._centered_since is not None:
                self.get_logger().info('Centrado perdido — reseteando cuenta')
            self._centered_since = None
            w = TURN_SIGN * float(
                max(-CENTER_OMEGA_MAX,
                    min(CENTER_OMEGA_MAX, K_BEARING * self.error_x)))
            self._send(0.0, w)
            self.get_logger().info(
                f'[CENTRAR] error_x={self.error_x:+.3f} w={w:+.3f}',
                throttle_duration_sec=0.4)

    # ══════════════════════════════════════════════════════════════════════
    # ESTADO 3 — AVANZAR: avanza recto 2 s y termina
    # ══════════════════════════════════════════════════════════════════════

    def _st_avanzar(self):
        now = self.get_clock().now()
        if self._forward_t0 is None:
            self._forward_t0 = now
        elapsed = (now - self._forward_t0).nanoseconds * 1e-9
        self.get_logger().info(
            f'[AVANZAR] {elapsed:.1f}/{FORWARD_TIME:.1f}s',
            throttle_duration_sec=0.5)
        if elapsed < FORWARD_TIME:
            self._send(FORWARD_SPEED, 0.0)
        else:
            self._stop()
            cen_msg = Bool()
            cen_msg.data = True
            self.cen_pub.publish(cen_msg)
            self.state = 'FIN'
            self.get_logger().info('FIN — objeto centrado y alcanzado')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = YoloCenterControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()