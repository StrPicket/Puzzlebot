#!/usr/bin/env python3
"""
center_qr_visual.py
────────────────────────────────────────────────────────────────────────────────
Control de alineación al QR en 6 fases (flujo geométrico):

  FASE 0 – BUSCAR:      No hay QR → gira lentamente hasta detectar uno.

  FASE 1 – CENTRAR ANG: QR detectado → solo giro hasta bearing ≈ 0.
                        Al DETECTAR el QR (FASE0→1) se guarda:
                          α = bearing_inicial (con signo)
                          sign_alpha = +1 derecha / -1 izquierda
                        Al TERMINAR (bearing ≈ 0) se calcula:
                          d_adj = h · cos(α)   (cateto adyacente)

  FASE 4 – GIRAR θ:     Gira sign_alpha * (90° − |α|) usando odometría.
                        Robot queda perpendicular al QR, pierde visión.
                        Ej: QR izquierda α=−10° → gira +80° → robot en 180°

  FASE 3 – AVANZAR d:   Avanza en línea recta d_adj usando odometría.
                        Al terminar, el robot está frente al QR alineado.

  FASE 4b – GIRAR 90°:  Gira −sign_alpha * 90° para quedar de frente al QR.
                        Ej: izquierda → gira −90° → robot en 90° → apunta al QR

  FASE 5 – APROXIMAR:   Avanza hasta READY_DIST con corrección visual.

NOTA: Las fases 3-4 usan odometría de ruedas (lazo abierto).
      La fase 5 re-engancha la visión para el ajuste fino.
────────────────────────────────────────────────────────────────────────────────
"""
import math
import numpy as np
import cv2
import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist


# ─── Calibración de cámara ────────────────────────────────────────────────────
CAMERA_MATRIX = np.array([
    [1.03795641e+03, 0.0,            6.36200746e+02],
    [0.0,            1.03634881e+03, 3.81386102e+02],
    [0.0,            0.0,            1.0           ],
], dtype=np.float64)

DIST_COEFFS = np.array(
    [[0.00383057, 0.1087906, -1.68623574, 3.76464743]],
    dtype=np.float64,
)

# ─── QR ───────────────────────────────────────────────────────────────────────
QR_SIZE    = 0.09    # lado del marcador (m)
READY_DIST = 0.3    # distancia objetivo en fase final (m)
WHITELIST  = {'Emezon', 'Wolmar', 'Popsi'}

# ─── Robot ────────────────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.0505   # igual que slam_odom.py
WHEEL_BASE   = 0.183    # igual que slam_odom.py

TURN_SIGN = -1.0   # +1 / -1 según montaje de cámara

# ─── Ganancias ────────────────────────────────────────────────────────────────
K_DIST    = 0.45
K_BEARING = 0.35

# ─── Saturaciones ─────────────────────────────────────────────────────────────
V_MAX     = 0.055
V_REV_MAX = 0.025
W_MAX     = 0.04    # rad/s para fases de giro visual (fases 0 y 1)

# ─── Búsqueda ─────────────────────────────────────────────────────────────────
W_SEARCH = 0.04    # rad/s velocidad de giro buscando QR (fase 0)

# ─── Tolerancias ──────────────────────────────────────────────────────────────
BEARING_TOL   = math.radians(4.5)   # fase 1: centrado angular OK
DIST_TOL      = 0.015               # ±1.5 cm tolerancia distancia final
DIST_TOL_NEAR = 0.008               # tolerancia fina de avance en fase 3

# ─── Giros odométricos ────────────────────────────────────────────────────────
ODO_ANGLE_TOL = math.radians(2.5)
W_TURN        = 0.08    # rad/s velocidad angular en giro odométrico
W_TURN_SLOW   = 0.06    # rad/s al acercarse al objetivo de giro

# ─── Avance odométrico ────────────────────────────────────────────────────────
V_ADVANCE     = 0.045   # m/s velocidad de avance en fase 3

# ─── Detección ────────────────────────────────────────────────────────────────
MIN_MARKER_PX    = 20.0
CAMERA_PITCH_DEG = 0.0


# ─── Fases del estado máquina ─────────────────────────────────────────────────
class Phase:
    SEARCH      = 0   # Busca QR girando
    CENTER_ANG  = 1   # Centra angularmente (bearing → 0)
    TURN_90     = 4   # Gira sign*(90°−|α|) para quedar perpendicular (odometría)
    ADVANCE_ADJ = 3   # Avanza cateto adyacente (odometría)
    TURN_FINAL  = 7   # Gira −sign*90° para quedar de frente al QR (odometría)
    APPROACH    = 5   # Aproximación final visual
    DONE        = 6


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class CenterQRVisual(Node):

    def __init__(self):
        super().__init__('center_qr_visual')

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_pub  = self.create_publisher(Twist,           '/cmd_vel',                      10)
        self.det_pub  = self.create_publisher(Bool,            '/qr/detected',                  10)
        self.cen_pub  = self.create_publisher(Bool,            '/qr/centered',                  10)
        self.dist_pub = self.create_publisher(Float32,         '/forklift/distance',            10)
        self.mark_pub = self.create_publisher(String,          '/qr/mark_detected',             10)
        self.img_pub  = self.create_publisher(CompressedImage, '/qr/image_detected/compressed', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            CompressedImage, '/video_source/compressed',
            self.image_cb, qos.qos_profile_sensor_data,
        )
        self.create_subscription(Bool, '/center_qr/enable', self.enable_cb, 10)
        self.create_subscription(
            Float32, 'VelocityEncR', self._encR_cb,
            qos.qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32, 'VelocityEncL', self._encL_cb,
            qos.qos_profile_sensor_data,
        )

        # ── Cámara ────────────────────────────────────────────────────────────
        self.enabled = False
        self.qr      = cv2.QRCodeDetector()
        self.K       = CAMERA_MATRIX
        self.D       = DIST_COEFFS

        h = QR_SIZE / 2.0
        self.obj_pts = np.array([
            [-h,  h, 0], [ h,  h, 0],
            [ h, -h, 0], [-h, -h, 0],
        ], dtype=np.float32)

        p = math.radians(CAMERA_PITCH_DEG)
        self.R_level = np.array([
            [1,          0,           0          ],
            [0, math.cos(p), -math.sin(p)],
            [0, math.sin(p),  math.cos(p)],
        ], dtype=np.float64)

        # ── Estado ────────────────────────────────────────────────────────────
        self.locked_id    = ''
        self.last_center  = None
        self.lost_count   = 0
        self.center_count = 0
        self.center_need  = 6

        # Máquina de estados
        self.phase = Phase.SEARCH

        # Parámetros calculados al detectar QR (FASE 0 → 1)
        self.alpha_rad   = 0.0   # bearing en el momento de detección (con signo)
        self.sign_alpha  = 0     # dirección del QR: +1 derecha, -1 izquierda
        self.h_original  = 0.0   # distancia al QR en el momento de primera detección (FASE 0)
        self.d_adj       = 0.0   # cateto adyacente = h_original·cos(α)

        # Objetivos odométricos
        self.odo_yaw_target     = 0.0
        self.odo_dist_start_xy  = (0.0, 0.0)

        # ── Odometría ─────────────────────────────────────────────────────────
        self._wr_raw        = 0.0
        self._wl_raw        = 0.0
        self._last_enc_time = 0.0          # nanoseconds como float
        self._STALE_TIMEOUT = 0.1          # 100ms sin encoder → velocidad 0
        self.odo       = [0.0, 0.0, 0.0]   # [x, y, yaw]
        self.last_time = self.get_clock().now()
        self.cur_v     = 0.0
        self.cur_w     = 0.0

        self.get_logger().info(
            f'center_qr_visual listo | READY_DIST={READY_DIST*100:.0f} cm'
        )

        # Timer de odometría a 50Hz (igual que slam_odom)
        self.create_timer(1.0 / 50.0, self.predict_step)

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def enable_cb(self, msg: Bool):
        self.enabled = msg.data
        if self.enabled:
            self._reset_state()
        else:
            self.stop()
            self.phase = Phase.SEARCH

    def _encR_cb(self, msg: Float32):
        self._wr_raw = msg.data
        self._last_enc_time = self.get_clock().now().nanoseconds * 1e-9

    def _encL_cb(self, msg: Float32):
        self._wl_raw = msg.data
        self._last_enc_time = self.get_clock().now().nanoseconds * 1e-9

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.phase        = Phase.SEARCH
        self.locked_id    = ''
        self.last_center  = None
        self.lost_count   = 0
        self.center_count = 0
        self.alpha_rad    = 0.0
        self.sign_alpha   = 0
        self.h_original   = 0.0
        self.d_adj        = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Odometría
    # ─────────────────────────────────────────────────────────────────────────

    def predict_step(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0 or dt > 0.5:
            return

        # Stale detection igual que slam_odom: si no llegan encoders → v=0
        now_s = now.nanoseconds * 1e-9
        if self._last_enc_time == 0.0 or (now_s - self._last_enc_time) > self._STALE_TIMEOUT:
            wr, wl = 0.0, 0.0
        else:
            wr, wl = self._wr_raw, self._wl_raw

        vr = WHEEL_RADIUS * wr
        vl = WHEEL_RADIUS * wl
        v  = (vr + vl) / 2.0
        w  = (vr - vl) / WHEEL_BASE
        self.cur_v, self.cur_w = v, w

        # ── LOG DIAGNÓSTICO (quitar cuando odometría funcione) ────────────────
        self.get_logger().info(
            f'ODO | wr={wr:.3f} wl={wl:.3f} '
            f'v={v:.4f} w={w:.4f} '
            f'x={self.odo[0]:.4f} y={self.odo[1]:.4f} '
            f'yaw={math.degrees(self.odo[2]):.2f}°',
            throttle_duration_sec=0.5,
        )

        th = self.odo[2]
        if abs(w) < 1e-6:
            self.odo[0] += v * dt * math.cos(th)
            self.odo[1] += v * dt * math.sin(th)
        else:
            r   = v / w
            th2 = th + w * dt
            self.odo[0] += r * (math.sin(th2) - math.sin(th))
            self.odo[1] += r * (math.cos(th) - math.cos(th2))
            self.odo[2]  = wrap_angle(th2)

    def odo_yaw(self) -> float:
        return self.odo[2]

    def odo_dist_traveled(self) -> float:
        dx = self.odo[0] - self.odo_dist_start_xy[0]
        dy = self.odo[1] - self.odo_dist_start_xy[1]
        return math.sqrt(dx * dx + dy * dy)

    # ─────────────────────────────────────────────────────────────────────────
    # Movimiento
    # ─────────────────────────────────────────────────────────────────────────

    def stop(self):
        self.cmd_pub.publish(Twist())

    def send_cmd(self, v: float, w: float):
        cmd = Twist()
        cmd.linear.x  = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    # ─────────────────────────────────────────────────────────────────────────
    # Detección QR
    # ─────────────────────────────────────────────────────────────────────────

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
            text = infos[i].strip() if (infos is not None and i < len(infos)) else ''
            center = pts[:4].mean(axis=0)
            candidates.append((text, pts[:4], center))

        if not candidates:
            return None, ''

        for text, pts, center in candidates:
            if text in WHITELIST:
                self.locked_id   = text
                self.last_center = center
                return pts, text

        if self.last_center is not None:
            best = min(candidates, key=lambda c: np.linalg.norm(c[2] - self.last_center))
            text, pts, center = best
            self.last_center = center
            return pts, text

        text, pts, center = candidates[0]
        self.last_center = center
        return pts, text

    # ─────────────────────────────────────────────────────────────────────────
    # Pose 6-DOF
    # ─────────────────────────────────────────────────────────────────────────

    def compute_pose(self, corners):
        img_pts = corners.reshape(4, 2).astype(np.float32)

        edge = (
            np.linalg.norm(img_pts[2] - img_pts[1]) +
            np.linalg.norm(img_pts[0] - img_pts[3])
        ) / 2.0

        if edge < MIN_MARKER_PX:
            return None

        img_pts_undist = cv2.fisheye.undistortPoints(
            img_pts.reshape(-1, 1, 2), self.K, self.D, P=self.K,
        ).reshape(-1, 2).astype(np.float32)

        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_pts, img_pts_undist, self.K, None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
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
        tx, ty, tz = float(t_lvl[0]), float(t_lvl[1]), float(t_lvl[2])

        dist    = math.sqrt(tx * tx + tz * tz)
        bearing = math.atan2(tx, tz)

        return {
            'tx': tx, 'ty': ty, 'tz': tz,
            'dist': dist, 'bearing': bearing,
            'rvec': rvec, 'tvec': tvec,
            'pts': img_pts, 'edge': edge,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Máquina de estados principal
    # ─────────────────────────────────────────────────────────────────────────

    def run_state_machine(self, pose):
        """Ejecuta la fase actual. pose puede ser None en fases odométricas."""

        # ══════════════════════════════════════════════════════════════════════
        # FASE 0 – Buscar QR girando
        # ══════════════════════════════════════════════════════════════════════
        if self.phase == Phase.SEARCH:
            if pose is not None:
                if self.alpha_rad == 0.0:
                    # ── Primera detección: guarda bearing y distancia originales ─
                    self.alpha_rad  = pose['bearing']
                    self.sign_alpha = 1 if self.alpha_rad >= 0.0 else -1
                    self.h_original = pose['dist']
                    self.get_logger().info(
                        f'FASE0→1: QR detectado | α={math.degrees(self.alpha_rad):+.2f}° '
                        f'sign={self.sign_alpha} h_orig={self.h_original:.3f} m'
                    )
                else:
                    # ── Redetección tras perder QR en FASE1: NO pisa alpha_rad ─
                    self.get_logger().info(
                        f'FASE0→1: QR recuperado | α_orig={math.degrees(self.alpha_rad):+.2f}° '
                        f'bearing_actual={math.degrees(pose["bearing"]):+.2f}°'
                    )
                self.phase = Phase.CENTER_ANG
                self.run_state_machine(pose)   # transición inmediata
                return
            self.send_cmd(0.0, W_SEARCH)
            return

        # ══════════════════════════════════════════════════════════════════════
        # FASE 1 – Centrar angularmente (solo giro visual, sin avanzar)
        # ══════════════════════════════════════════════════════════════════════
        if self.phase == Phase.CENTER_ANG:
            if pose is None:
                self.get_logger().warn('FASE1: QR perdido → SEARCH')
                self.phase = Phase.SEARCH
                return

            bearing = pose['bearing']
            dist    = pose['dist']

            if abs(bearing) <= BEARING_TOL:
                # d_adj = cateto opuesto = distancia lateral al QR
                # = h_orig × sin(|α|)  (NO cos: eso sería el cateto adyacente frontal)
                self.d_adj = self.h_original * math.sin(abs(self.alpha_rad))
                self.get_logger().info(
                    f'FASE1→4: centrado | α_orig={math.degrees(self.alpha_rad):+.2f}° '
                    f'h_orig={self.h_original:.3f} m  d_adj={self.d_adj:.3f} m  sign={self.sign_alpha}'
                )
                self._start_turn_90()
                return

            # Todavía centrando: solo giro visual
            w = TURN_SIGN * float(np.clip(K_BEARING * bearing, -W_MAX, W_MAX))
            self.send_cmd(0.0, w)
            self.get_logger().info(
                f'FASE1-ANG | bearing={math.degrees(bearing):+.2f}° '
                f'dist={dist:.3f} w={w:+.3f}',
                throttle_duration_sec=0.3,
            )
            return

        # ══════════════════════════════════════════════════════════════════════
        # FASE 4 – Girar sign*(90°−|α|) para quedar perpendicular (odometría)
        # ══════════════════════════════════════════════════════════════════════
        if self.phase == Phase.TURN_90:
            err = wrap_angle(self.odo_yaw_target - self.odo_yaw())
            if abs(err) <= ODO_ANGLE_TOL:
                self.stop()
                self.get_logger().info(
                    f'FASE4→3: giro completo | yaw={math.degrees(self.odo_yaw()):.2f}°'
                )
                self._start_advance_adj()
                return
            w_sign = 1.0 if err > 0 else -1.0
            speed  = W_TURN_SLOW if abs(err) < math.radians(10) else W_TURN
            self.send_cmd(0.0, w_sign * speed)
            self.get_logger().info(
                f'FASE4-GIRO | err={math.degrees(err):+.2f}° '
                f'yaw={math.degrees(self.odo_yaw()):+.2f}° '
                f'target={math.degrees(self.odo_yaw_target):.2f}°',
                throttle_duration_sec=0.3,
            )
            return

        # ══════════════════════════════════════════════════════════════════════
        # FASE 3 – Avanzar cateto adyacente (odometría)
        # ══════════════════════════════════════════════════════════════════════
        if self.phase == Phase.ADVANCE_ADJ:
            traveled  = self.odo_dist_traveled()
            remaining = self.d_adj - traveled
            if remaining <= DIST_TOL_NEAR:
                self.stop()
                self.get_logger().info(
                    f'FASE3→4b: avance completo | traveled={traveled:.3f} m'
                )
                self._start_turn_final()
                return
            v = V_ADVANCE if remaining > 0.05 else V_ADVANCE * 0.5
            self.send_cmd(v, 0.0)
            self.get_logger().info(
                f'FASE3-AVZ | traveled={traveled:.3f}/{self.d_adj:.3f} m rem={remaining:.3f}',
                throttle_duration_sec=0.3,
            )
            return

        # ══════════════════════════════════════════════════════════════════════
        # FASE 4b – Girar −sign*90° para quedar de frente al QR (odometría)
        # ══════════════════════════════════════════════════════════════════════
        if self.phase == Phase.TURN_FINAL:
            err = wrap_angle(self.odo_yaw_target - self.odo_yaw())
            if abs(err) <= ODO_ANGLE_TOL:
                self.stop()
                self.get_logger().info(
                    f'FASE4b→5: giro final completo | yaw={math.degrees(self.odo_yaw()):.2f}°'
                )
                self.phase = Phase.APPROACH
                return
            w_sign = 1.0 if err > 0 else -1.0
            speed  = W_TURN_SLOW if abs(err) < math.radians(10) else W_TURN
            self.send_cmd(0.0, w_sign * speed)
            self.get_logger().info(
                f'FASE4b-GIRO | err={math.degrees(err):+.2f}° '
                f'yaw={math.degrees(self.odo_yaw()):+.2f}° '
                f'target={math.degrees(self.odo_yaw_target):.2f}°',
                throttle_duration_sec=0.3,
            )
            return

        # ══════════════════════════════════════════════════════════════════════
        # FASE 5 – Aproximación final visual
        # ══════════════════════════════════════════════════════════════════════
        if self.phase == Phase.APPROACH:
            if pose is None:
                self.lost_count += 1
                if self.lost_count < 8:
                    self.stop()
                else:
                    self.send_cmd(0.02, 0.0)
                self.get_logger().warn('FASE5: QR no visible', throttle_duration_sec=0.5)
                return

            self.lost_count = 0
            dist    = pose['dist']
            bearing = pose['bearing']
            e_dist  = dist - READY_DIST

            if abs(e_dist) <= DIST_TOL and abs(bearing) <= BEARING_TOL:
                self.center_count += 1
                self.stop()
                if self.center_count >= self.center_need:
                    self.phase = Phase.DONE
                    self.cen_pub.publish(Bool(data=True))
                    self.get_logger().info(
                        f'QR CENTRADO ✓ | dist={dist:.3f} m | '
                        f'bearing={math.degrees(bearing):+.2f}° | id={self.locked_id}'
                    )
                return

            self.center_count = 0
            self.cen_pub.publish(Bool(data=False))

            v = float(np.clip(K_DIST * e_dist, -V_REV_MAX, V_MAX))
            w = TURN_SIGN * float(np.clip(K_BEARING * bearing, -0.25, 0.25))

            if abs(bearing) < math.radians(2.5):
                w = 0.0

            self.send_cmd(v, w)
            self.get_logger().info(
                f'FASE5-APROX | dist={dist:.3f} e={e_dist:+.3f} '
                f'bearing={math.degrees(bearing):+.2f}° '
                f'v={v:+.3f} w={w:+.3f}',
                throttle_duration_sec=0.3,
            )
            return

        # ══════════════════════════════════════════════════════════════════════
        # FASE DONE
        # ══════════════════════════════════════════════════════════════════════
        if self.phase == Phase.DONE:
            self.stop()
            self.cen_pub.publish(Bool(data=True))

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers de transición
    # ─────────────────────────────────────────────────────────────────────────

    def _start_advance_adj(self):
        """Fase 3: avanza d_adj metros en línea recta (odometría)."""
        self.odo_dist_start_xy = (self.odo[0], self.odo[1])
        self.phase = Phase.ADVANCE_ADJ
        self.get_logger().info(
            f'  → ADVANCE_ADJ | d={self.d_adj:.3f} m '
            f'desde ({self.odo[0]:.3f}, {self.odo[1]:.3f})'
        )

    def _start_turn_90(self):
        """
        Fase 4: gira sign*(90°−|α|) para quedar perpendicular al QR.

        El robot siempre parte a 90°.
        En FASE1 giró |α| en dirección sign_alpha.
        Ahora gira el ángulo restante en la misma dirección.

        Ejemplos (robot inicia a 90°):
          α = −10° (QR izquierda):
            FASE1 giró +10° → robot en 100°
            FASE4 gira +(90°−10°) = +80° → robot en 180° → perpendicular al QR ✓
          α = +10° (QR derecha):
            FASE1 giró −10° → robot en 80°
            FASE4 gira −(90°−10°) = −80° → robot en 0° → perpendicular al QR ✓
        """
        remaining_deg = math.pi / 2.0 - abs(self.alpha_rad)
        delta_yaw     = -self.sign_alpha * remaining_deg
        self.odo_yaw_target = wrap_angle(self.odo_yaw() + delta_yaw)
        self.phase = Phase.TURN_90
        self.get_logger().info(
            f'  → TURN_90 | α={math.degrees(self.alpha_rad):+.2f}° '
            f'restante={math.degrees(remaining_deg):.2f}° '
            f'delta={math.degrees(delta_yaw):+.2f}° '
            f'yaw_actual={math.degrees(self.odo_yaw()):.2f}° '
            f'target_yaw={math.degrees(self.odo_yaw_target):.2f}°'
        )

    def _start_turn_final(self):
        """
        Fase 4b: gira +sign_alpha * 90° para quedar de frente al QR.

        Tras avanzar el cateto, el robot apunta perpendicularmente al QR.
        Girando 90° en la misma dirección de sign_alpha queda de frente.

        Ejemplos:
          izquierda (sign=−1): robot en 180°, gira −90° → queda en 90° → apunta al QR ✓
          derecha   (sign=+1): robot en   0°, gira +90° → queda en 90° → apunta al QR ✓
        """
        delta_yaw = self.sign_alpha * (math.pi / 2.0)
        self.odo_yaw_target = wrap_angle(self.odo_yaw() + delta_yaw)
        self.phase = Phase.TURN_FINAL
        self.get_logger().info(
            f'  → TURN_FINAL | sign={self.sign_alpha} '
            f'delta={math.degrees(delta_yaw):+.2f}° '
            f'yaw_actual={math.degrees(self.odo_yaw()):.2f}° '
            f'target_yaw={math.degrees(self.odo_yaw_target):.2f}°'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Debug visual
    # ─────────────────────────────────────────────────────────────────────────

    def draw_debug(self, frame, pose):
        h, w = frame.shape[:2]
        cx = int(self.K[0, 2])
        cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)

        phase_names = {
            Phase.SEARCH:      'FASE0-SEARCH',
            Phase.CENTER_ANG:  'FASE1-CENTER_ANG',
            Phase.TURN_90:     'FASE4-TURN_90',
            Phase.ADVANCE_ADJ: 'FASE3-ADVANCE_ADJ',
            Phase.TURN_FINAL:  'FASE4b-TURN_FINAL',
            Phase.APPROACH:    'FASE5-APPROACH',
            Phase.DONE:        'DONE',
        }
        cv2.putText(frame, phase_names.get(self.phase, '?'), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        if pose is not None:
            pts = pose['pts'].astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            try:
                cv2.drawFrameAxes(
                    frame, self.K, None,
                    pose['rvec'], pose['tvec'],
                    QR_SIZE * 0.5, 2,
                )
            except cv2.error:
                pass

            line1 = (
                f"d={pose['dist']:.2f}m  tx={pose['tx']:+.2f}  "
                f"bearing={math.degrees(pose['bearing']):+.1f}deg"
            )
            line2 = (
                f"alpha={math.degrees(self.alpha_rad):+.1f}  "
                f"d_adj={self.d_adj:.3f}m  sign={self.sign_alpha}"
            )
            cv2.putText(frame, line1, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
            cv2.putText(frame, line2, (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2)

        try:
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            img_msg = CompressedImage()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.format = 'jpeg'
            img_msg.data   = buf.tobytes()
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f'publish image error: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Callback de imagen
    # ─────────────────────────────────────────────────────────────────────────

    def image_cb(self, msg: CompressedImage):
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn(f'decode error: {e}')
            return

        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        corners, text = self.detect_qr(gray)
        pose = self.compute_pose(corners) if corners is not None else None

        self.det_pub.publish(Bool(data=pose is not None))

        if pose is not None:
            self.dist_pub.publish(Float32(data=float(pose['dist'])))
            self.mark_pub.publish(String(data=self.locked_id))
            if text:
                self.locked_id = text

        if not self.enabled:
            self.stop()
            self.draw_debug(frame, pose)
            return

        if self.phase == Phase.DONE:
            self.stop()
            self.cen_pub.publish(Bool(data=True))
            self.draw_debug(frame, pose)
            return

        self.run_state_machine(pose)
        self.draw_debug(frame, pose)


# ─── Entry point ──────────────────────────────────────────────────────────────

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
