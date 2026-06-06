#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YoloCenterControl — Alineacion con objeto YOLO usando estimacion de pose.

Replica la logica de CenterQR pero para objetos detectados por YOLO:
  - solvePnP sobre las 4 esquinas del bounding box (objeto 10x10 cm cuadrado)
  - Odometria local por encoders (dead reckoning)
  - Flujo: SCAN -> OBSERVAR -> GIRO1 -> AVANCE -> GIRO2 -> AJUSTE -> FIN

IMPORTANTE: usar las esquinas del bbox como puntos 3D es una aproximacion valida
para objetos planos vistos de frente. El angulo psi tendra mas ruido que con QR.
Se recomienda usar IPPE_SQUARE con objeto cuadrado para maxima estabilidad.

Suscripciones:
  /Yolov8_Inference    (Yolov8Inference)  — bboxes con left/top/right/bottom
  /VelocityEncR        (Float32)          — velocidad angular rueda derecha [rad/s]
  /VelocityEncL        (Float32)          — velocidad angular rueda izquierda [rad/s]
  /center_yolo/enable  (Bool)
  /center_yolo/stop    (Bool)

Publicaciones:
  /cmd_vel             (Twist)
  /yolo/detected       (Bool)
  /yolo/centered       (Bool)
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist
from yolov8_msgs.msg import Yolov8Inference

# ── Parametros de camara (AJUSTAR con tu calibracion) ────────────────────────
CAMERA_MATRIX = np.array([
    [1.03795641e+03,   0.0,         6.36200746e+02],
    [  0.0,         1.03634881e+03,  3.81386102e+02],
    [  0.0,           0.0,           1.0      ]
], dtype=np.float64)

DIST_COEFFS = np.array([[0.00383057, 0.1087906, -1.68623574, 3.76464743]], dtype=np.float64)

# ── Objeto objetivo ───────────────────────────────────────────────────────────
OBJ_SIZE     = 0.10          # lado del objeto cuadrado en metros (10 cm)

# ── Distancias de maniobra ────────────────────────────────────────────────────
READY_DIST   = 0.35          # distancia final de alineacion (metros)
STANDOFF     = 0.55          # punto G: distancia de aproximacion para maniobra
CAM_AHEAD    = 0.066         # offset camara respecto al eje de giro del robot

# ── Tolerancias del ajuste final ──────────────────────────────────────────────
BEARING_TOL  = math.radians(5)
DIST_TOL     = 0.04
PSI_TOL      = math.radians(6)
LAT_TOL      = 0.03

# ── Encoders / odometria ──────────────────────────────────────────────────────
WHEEL_RADIUS = 0.0391
WHEEL_BASE   = 0.180

# ── Signos de giro (ajustar si los giros salen al reves) ─────────────────────
TURN_SIGN    = -1.0
GIRO_SIGN    =  1.0
PERP_SIGN    =  1.0

# ── Clases objetivo ───────────────────────────────────────────────────────────
TARGET_CLASSES = {'Amazon'}     # set vacio = todas las clases

# ── Estados que maneja el timer (sin camara) ──────────────────────────────────
TIMER_STATES = {'GIRO1', 'AVANCE', 'GIRO2'}


class YoloCenterControl(Node):

    def __init__(self):
        super().__init__('yolo_center_control')

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.det_pub = self.create_publisher(Bool,  '/yolo/detected', 10)
        self.cen_pub = self.create_publisher(Bool,  '/yolo/centered', 10)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Yolov8Inference, 'Yolov8_Inference',
            self._inference_cb, qos.qos_profile_sensor_data)
        self.create_subscription(
            Bool,    '/center_yolo/enable', self._enable_cb, 10)
        self.create_subscription(
            Bool,    '/center_yolo/stop',   self._stop_cb,   10)
        self.create_subscription(
            Float32, '/VelocityEncR', self._encR, qos.qos_profile_sensor_data)
        self.create_subscription(
            Float32, '/VelocityEncL', self._encL, qos.qos_profile_sensor_data)

        # ── Estado general ────────────────────────────────────────────────
        self.enabled  = True
        self.estop    = False
        self.state    = 'SCAN'

        # ── Odometria local (dead reckoning) ──────────────────────────────
        self.wr = self.wl = 0.0
        self.have_enc  = False
        self.dr_x = self.dr_y = self.dr_yaw = 0.0
        self.dr_last = self.get_clock().now()

        # ── Pose estimation ───────────────────────────────────────────────
        h = OBJ_SIZE / 2.0
        # Esquinas 3D del objeto plano (en coordenadas del objeto, Z=0)
        # Orden: TL, TR, BR, BL  (igual que QRCodeDetector)
        self.obj_pts = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float32)
        self.K = CAMERA_MATRIX
        self.D = DIST_COEFFS
        self.prev_gamma = None
        self.min_bbox_px = 20.0   # lado minimo del bbox en pixeles para procesar

        # ── Deteccion actual ──────────────────────────────────────────────
        self.current_pose  = None   # dict con dist/bearing/psi/e_lat/...
        self.detected      = False

        # ── Scan ──────────────────────────────────────────────────────────
        self.scan_omega  = 0.06          # rad/s — lento para que YOLO detecte
        self.scan_period = 52.0          # 180deg / 0.06 rad/s ≈ 52 s por barrido
        self.scanning    = False
        self.scan_dir    = 1.0
        self.scan_first  = True
        self.scan_t0     = self.get_clock().now()

        # Confirmacion en SCAN: N frames consecutivos detectados antes de
        # transicionar a OBSERVAR. Evita falsos positivos por frames sueltos.
        self.scan_confirm_need  = 4    # frames consecutivos requeridos
        self.scan_confirm_count = 0    # contador actual

        # Coast eliminado: el robot para inmediatamente al detectar.
        # La transicion a OBSERVAR solo ocurre si bearing < scan_entry_bearing.
        # Esto evita que el objeto salga del FOV por inercia del giro.
        self.scan_entry_bearing = math.radians(40)  # max bearing para entrar a OBSERVAR

        # ── Observacion ───────────────────────────────────────────────────
        self.bearing_obs    = math.radians(7)
        self.observe_frames = 8
        self.obs_buf        = []

        # ── Maniobra (GIRO1/AVANCE/GIRO2) ────────────────────────────────
        self.Gx = self.Gy = self.Ghead = 0.0
        self.giro_speed  = 0.08          # 5x mas lento que antes (era 0.4)
        self.drive_speed = 0.024         # 5x mas lento que antes (era 0.12)
        self.yaw_tol     = math.radians(4)
        self.pos_tol     = 0.04

        # ── Ajuste final (lazo visual cerrado) ────────────────────────────
        self.k_bearing  = 1.2
        self.k_v        = 0.5
        self.k_psi      = 0.5
        self.k_e        = 0.7
        self.v_max      = 0.024          # 5x mas lento (era 0.12)
        self.w_max      = 0.10           # 5x mas lento (era 0.5)
        self.v_rev      = 0.008          # 5x mas lento (era 0.04)
        self.perp_cap   = 0.06           # 5x mas lento (era 0.30)
        self.listo_need  = 5
        self.listo_count = 0
        self.retries     = 0
        self.max_retries = 2
        self.remaneuver_perp = math.radians(25)

        # ── Hold de bearing: N segundos estable antes de pasar a dist ──────
        self.bearing_hold_time  = 3.0          # segundos requeridos estable
        self.bearing_hold_t0    = None         # timestamp inicio del hold
        self.bearing_hold_ok    = False        # ya cumplio el hold

        # ── Re-adquisicion ────────────────────────────────────────────────
        self.qr_lost     = 999
        self.relost_scan = 60

        # ── Debug ─────────────────────────────────────────────────────────
        self.dbg = dict(state='SCAN', dist=0.0, bearing=0.0,
                        psi=0.0, e_lat=0.0, v=0.0, w=0.0, gerr=0.0)

        self.create_timer(0.02, self._ctrl_timer)   # 50 Hz
        self.get_logger().info(
            f'YoloCenterControl (pose solvePnP) iniciado | '
            f'clases={TARGET_CLASSES or "todas"} | obj={OBJ_SIZE*100:.0f}cm')

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS DE ENCODERS Y CONTROL
    # ══════════════════════════════════════════════════════════════════════

    def _encR(self, m): self.wr = m.data; self.have_enc = True
    def _encL(self, m): self.wl = m.data; self.have_enc = True

    def _enable_cb(self, msg: Bool):
        self.enabled = msg.data
        if not self.enabled:
            self._reset()
            self._stop()

    def _stop_cb(self, msg: Bool):
        self.estop = bool(msg.data)
        if self.estop:
            self._stop()
            self.get_logger().warn('PARO DE EMERGENCIA activado')

    def _reset(self):
        self.state              = 'SCAN'
        self.scanning           = False
        self.qr_lost            = 999
        self.obs_buf            = []
        self.retries            = 0
        self.listo_count        = 0
        self.prev_gamma         = None
        self.scan_confirm_count = 0
        self.bearing_hold_t0    = None
        self.bearing_hold_ok    = False


    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _ang_diff(self, a, b):
        return abs(self._wrap(a - b))

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _send(self, v, w):
        c = Twist()
        c.linear.x  = float(v)
        c.angular.z = float(w)
        self.cmd_pub.publish(c)
        self.dbg.update(v=v, w=w)

    # ══════════════════════════════════════════════════════════════════════
    # ESTIMACION DE POSE desde bbox YOLO
    # ══════════════════════════════════════════════════════════════════════

    def _bbox_to_corners(self, left, top, right, bottom):
        """
        Convierte bbox (l,t,r,b) en 4 esquinas imagen en orden TL,TR,BR,BL.
        Este orden debe coincidir con obj_pts para que solvePnP sea correcto.
        """
        return np.array([
            [left,  top],
            [right, top],
            [right, bottom],
            [left,  bottom],
        ], dtype=np.float32)

    def _gamma_rad(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        return self._wrap(math.atan2(R[0, 2], R[2, 2]) + math.pi)

    def compute_pose(self, left, top, right, bottom):
        """
        Estima dist, bearing, psi, e_lat a partir del bbox YOLO.
        Tambien devuelve cx_norm: posicion horizontal del centro del bbox
        normalizada [-1,1], usada en SCAN/OBSERVAR donde bearing de solvePnP
        es inestable (objeto visto de lado o muy cerca).
        Devuelve dict con la pose o None si el bbox es demasiado pequeno.
        """
        img_pts = self._bbox_to_corners(left, top, right, bottom)

        # Verificar que el bbox tiene un tamano razonable
        bbox_w = right - left
        bbox_h = bottom - top
        if bbox_w < self.min_bbox_px or bbox_h < self.min_bbox_px:
            return None

        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_pts, img_pts, self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception as e:
            self.get_logger().warn(f'solvePnP: {e}', throttle_duration_sec=1.0)
            return None

        if n == 0:
            return None

        # Seleccionar la solucion mas consistente con el frame anterior
        if n >= 2 and self.prev_gamma is not None:
            g = [self._gamma_rad(rvecs[i]) for i in range(n)]
            best_i = min(range(n), key=lambda i: self._ang_diff(g[i], self.prev_gamma))
        else:
            best_i = int(np.argmin(np.array(reproj).ravel()))

        rvec = rvecs[best_i]
        tvec = tvecs[best_i]

        gamma = self._gamma_rad(rvec)
        self.prev_gamma = gamma

        # Transformar al plano horizontal (pitch de camara)
        t = tvec.reshape(3).ravel()
        tx, tz = float(t[0]), float(t[2])

        dist    = math.hypot(tx, tz)
        bearing = math.atan2(tx, tz)

        # Normal del plano del objeto apuntando hacia la camara
        R, _ = cv2.Rodrigues(rvec)
        nrm = R[:, 2].astype(float)
        if (nrm[0] * (-tx) + nrm[2] * (-tz)) < 0:
            nrm = -nrm
        nx, nz = nrm[0], nrm[2]
        ln = math.hypot(nx, nz) + 1e-9
        nx, nz = nx / ln, nz / ln

        psi   = math.atan2(-nx, -nz)          # angulo perpendicular
        e_lat = (-tx * nz + tz * nx)          # error lateral (m)

        # cx_norm: error horizontal normalizado desde el centro de imagen [-1, 1]
        # Se usa en SCAN/OBSERVAR en lugar del bearing de solvePnP (mas estable)
        cx_px   = float((left + right) / 2.0)   # NOTE: necesitamos left/right aqui
        img_cx  = float(self.K[0, 2])            # cx de la camara
        img_w2  = float(self.K[0, 0])            # fx como proxy de mitad de imagen
        cx_norm = (cx_px - img_cx) / img_w2      # normalizado: 0=centro, +-1=borde

        return dict(dist=dist, bearing=bearing, psi=psi, e_lat=e_lat,
                    tx=tx, tz=tz, nx=nx, nz=nz,
                    rvec=rvec, tvec=tvec,
                    cx_norm=cx_norm)

    # ══════════════════════════════════════════════════════════════════════
    # GEOMETRIA DEL OBJETIVO (marco DR local)
    # ══════════════════════════════════════════════════════════════════════

    def _goal_in_robot(self, pose):
        """Calcula el punto G (standoff) en el marco del robot."""
        qx_r = CAM_AHEAD + pose['tz']
        qy_r = -pose['tx']
        nf, nl = pose['nz'], -pose['nx']
        ln = math.hypot(nf, nl) + 1e-9
        nf, nl = nf / ln, nl / ln
        gx_r = qx_r + nf * STANDOFF
        gy_r = qy_r + nl * STANDOFF
        gh_r = math.atan2(-nl, -nf)
        return gx_r, gy_r, gh_r

    def _lock_goal(self):
        """Promedia el buffer de observaciones y convierte al marco global DR."""
        arr = np.array(self.obs_buf)
        gx_r = arr[:, 0].mean()
        gy_r = arr[:, 1].mean()
        gh_r = arr[:, 2].mean()
        c, s = math.cos(self.dr_yaw), math.sin(self.dr_yaw)
        self.Gx    = self.dr_x + gx_r * c - gy_r * s
        self.Gy    = self.dr_y + gx_r * s + gy_r * c
        self.Ghead = self._wrap(self.dr_yaw + gh_r)

    # ══════════════════════════════════════════════════════════════════════
    # TIMER 50 Hz — odometria + maniobra DR
    # ══════════════════════════════════════════════════════════════════════

    def _ctrl_timer(self):
        # ── Integrar odometria ────────────────────────────────────────────
        now = self.get_clock().now()
        dt  = (now - self.dr_last).nanoseconds * 1e-9
        self.dr_last = now
        if 0 < dt < 0.5 and self.have_enc:
            v = WHEEL_RADIUS * (self.wr + self.wl) / 2.0
            w = WHEEL_RADIUS * (self.wr - self.wl) / WHEEL_BASE
            self.dr_x   += v * math.cos(self.dr_yaw) * dt
            self.dr_y   += v * math.sin(self.dr_yaw) * dt
            self.dr_yaw  = self._wrap(self.dr_yaw + w * dt)

        if self.estop:
            self._stop()
            return
        if not self.enabled:
            return

        s = self.state
        if s == 'GIRO1':  self._st_giro1()
        elif s == 'AVANCE': self._st_avance()
        elif s == 'GIRO2':  self._st_giro2()

    def _st_giro1(self):
        tgt = math.atan2(self.Gy - self.dr_y, self.Gx - self.dr_x)
        err = self._wrap(tgt - self.dr_yaw)
        self.dbg.update(gerr=err)
        if abs(err) < self.yaw_tol:
            self.state = 'AVANCE'
            self._stop()
            return
        self._send(0.0, GIRO_SIGN * self.giro_speed * (1.0 if err > 0 else -1.0))

    def _st_avance(self):
        dx = self.Gx - self.dr_x
        dy = self.Gy - self.dr_y
        d  = math.hypot(dx, dy)
        self.dbg.update(gerr=d)
        if d < self.pos_tol:
            self.state = 'GIRO2'
            self._stop()
            return
        herr = self._wrap(math.atan2(dy, dx) - self.dr_yaw)
        v    = min(self.drive_speed, self.k_v * d + 0.04)
        self._send(v, GIRO_SIGN * float(np.clip(1.5 * herr, -0.4, 0.4)))

    def _st_giro2(self):
        err = self._wrap(self.Ghead - self.dr_yaw)
        self.dbg.update(gerr=err)
        if abs(err) < self.yaw_tol:
            self.state = 'REVERIFICAR'
            self.qr_lost = 999
            self._stop()
            return
        self._send(0.0, GIRO_SIGN * self.giro_speed * (1.0 if err > 0 else -1.0))

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACK DE INFERENCIA YOLO
    # ══════════════════════════════════════════════════════════════════════

    def _inference_cb(self, msg: Yolov8Inference):
        if self.estop or not self.enabled:
            self._stop()
            return

        # ── Filtrar por clase objetivo ────────────────────────────────────
        dets = msg.yolov8_inference
        if TARGET_CLASSES:
            dets = [d for d in dets if d.class_name in TARGET_CLASSES]

        # ── Seleccionar el bbox Amazon mas grande ────────────────────────
        # Doble filtro: TARGET_CLASSES ya filtro arriba, pero verificamos
        # explicitamente que sea Amazon para no procesar detecciones erroneas.
        best = None
        best_area = 0.0
        for d in dets:
            if d.class_name not in TARGET_CLASSES:   # paranoia: ignorar otras clases
                continue
            area = (d.right - d.left) * (d.bottom - d.top)
            if area > best_area:
                best_area = area
                best = d

        # ── Calcular pose si hay deteccion ────────────────────────────────
        pose = None
        if best is not None:
            pose = self.compute_pose(best.left, best.top, best.right, best.bottom)

        self.detected     = pose is not None
        self.current_pose = pose

        self.det_pub.publish(Bool(data=self.detected))
        if pose:
            self.dbg.update(dist=pose['dist'], bearing=pose['bearing'],
                            psi=pose['psi'], e_lat=pose['e_lat'])

        # ── Maquina de estados (vision) ───────────────────────────────────
        s = self.state
        if s in TIMER_STATES:
            pass   # manejado por el timer DR

        elif s == 'SCAN':
            if pose is not None:
                # Para inmediatamente — no girar mas, el objeto esta en camara.
                self._stop()

                bearing_abs = abs(pose['bearing'])

                # Usar cx_norm (posicion pixel) en lugar de bearing de solvePnP.
                # bearing de solvePnP es inestable cuando el objeto esta de lado
                # o muy cerca; cx_norm solo depende del centro del bbox y es robusto.
                cx_norm = pose['cx_norm']          # [-1,1]: negativo=izq, positivo=der
                cx_tol  = 0.25                     # +-25% del semiancho = "centrado"

                if abs(cx_norm) > cx_tol:
                    # Objeto al borde: girar hacia el centro a velocidad proporcional
                    w_mag = float(np.clip(abs(cx_norm) * self.w_max * 2.0,
                                          self.w_max * 0.15, self.w_max))
                    w = TURN_SIGN * math.copysign(w_mag, cx_norm)
                    self._send(0.0, w)
                    self.scan_confirm_count = 0
                    self.get_logger().info(
                        f'[SCAN] Centrando: cx_norm={cx_norm:+.2f} w={w:+.4f}',
                        throttle_duration_sec=0.3)
                else:
                    # Centro del bbox dentro de tolerancia: confirmar con robot parado
                    self._stop()
                    self.scan_confirm_count += 1
                    self.get_logger().info(
                        f'[SCAN] Confirmando {self.scan_confirm_count}'
                        f'/{self.scan_confirm_need} cx={cx_norm:+.2f}...',
                        throttle_duration_sec=0.2)
                    if self.scan_confirm_count >= self.scan_confirm_need:
                        self.scanning           = False
                        self.scan_confirm_count = 0
                        self.obs_buf            = []
                        self.state              = 'OBSERVAR'
                        self.get_logger().info('Objeto confirmado — pasando a OBSERVAR')
            else:
                # Perdio la deteccion: reiniciar contador y seguir girando
                self.scan_confirm_count = 0
                self.qr_lost += 1
                self._scan()

        elif s == 'OBSERVAR':
            self._st_observe(pose)

        elif s == 'REVERIFICAR':
            self._st_reverificar(pose)

        elif s == 'AJUSTE':
            self._st_ajuste(pose)

        elif s in ('FIN',):
            self._stop()

        # ── Log de debug ──────────────────────────────────────────────────
        self.dbg.update(state='PARO' if self.estop else self.state)
        self.get_logger().info(
            f"[DBG] {self.dbg['state']} det={self.detected} "
            f"dist={self.dbg['dist']:.3f} "
            f"bearing={math.degrees(self.dbg['bearing']):+.1f}deg "
            f"psi={math.degrees(self.dbg['psi']):+.1f}deg "
            f"elat={self.dbg['e_lat']*100:+.1f}cm "
            f"dr=({self.dr_x:+.2f},{self.dr_y:+.2f},{math.degrees(self.dr_yaw):+.0f}deg) "
            f"gerr={self.dbg['gerr']:+.3f}",
            throttle_duration_sec=0.4)

    # ══════════════════════════════════════════════════════════════════════
    # ESTADOS DE VISION
    # ══════════════════════════════════════════════════════════════════════

    def _scan(self):
        """Giro bidireccional lento buscando el objeto.
        scan_omega bajo (0.12 rad/s) para que YOLO tenga tiempo de detectar
        en cada posicion antes de que el objeto salga del frame.
        """
        now = self.get_clock().now()
        if not self.scanning:
            self.scanning    = True
            self.scan_dir    = 1.0
            self.scan_first  = True
            self.scan_t0     = now
        period = self.scan_period * (0.5 if self.scan_first else 1.0)
        if (now - self.scan_t0).nanoseconds * 1e-9 > period:
            self.scan_dir   *= -1.0
            self.scan_first  = False
            self.scan_t0     = now
        self._send(0.0, self.scan_dir * self.scan_omega)

    def _st_observe(self, pose):
        """
        Acumula muestras de la pose del objetivo para calcular el punto G.
        Requiere que el objeto este aproximadamente centrado (bearing < bearing_obs).
        """
        if pose is None:
            self.obs_buf = []
            self.state   = 'SCAN'
            return

        # Centrar usando cx_norm (pixel), no bearing de solvePnP.
        # bearing de solvePnP es inestable a distancias cortas o angulos grandes.
        cx_norm  = pose['cx_norm']
        cx_tol_obs = 0.15    # tolerancia mas estricta que en SCAN para observar bien
        if abs(cx_norm) > cx_tol_obs:
            self.obs_buf = []
            w_mag = float(np.clip(abs(cx_norm) * self.w_max * 2.0,
                                  self.w_max * 0.15, self.w_max))
            w = TURN_SIGN * math.copysign(w_mag, cx_norm)
            self._send(0.0, w)
            self.get_logger().info(
                f'[OBSERVAR] Centrando: cx_norm={cx_norm:+.2f} w={w:+.4f}',
                throttle_duration_sec=0.3)
            return

        self.obs_buf.append(self._goal_in_robot(pose))
        self._stop()

        if len(self.obs_buf) >= self.observe_frames:
            if not self.have_enc:
                self.get_logger().warn(
                    'Sin encoders — no puedo ejecutar maniobra DR',
                    throttle_duration_sec=2.0)
                self.obs_buf = []
                return
            self._lock_goal()
            self.obs_buf = []
            self.state   = 'GIRO1'
            self.get_logger().info(
                f'Goal bloqueado: G=({self.Gx:.2f},{self.Gy:.2f}) '
                f'head={math.degrees(self.Ghead):.1f}deg')

    def _st_reverificar(self, pose):
        """
        Despues de GIRO2, verifica que el objeto sea visible y el bearing sea
        aceptable antes de entrar al ajuste visual fino.
        """
        if pose is not None and abs(pose['bearing']) < math.radians(15):
            self.state = 'AJUSTE'
            return
        self.qr_lost += 1
        if self.qr_lost < self.relost_scan:
            self._scan()
        else:
            self._scan()

    def _st_ajuste(self, pose):
        """
        Ajuste visual fino SECUENCIAL por eje (inspirado en el centrado del Tello).

        Logica: corregir UN eje a la vez en orden de prioridad, parar cuando
        cada eje entra en tolerancia antes de pasar al siguiente. Esto elimina
        las oscilaciones cruzadas del lazo simultaneo.

        Orden de correccion:
          1. bearing  — apuntar hacia el objeto (w solamente)
          2. dist     — acercarse/alejarse a READY_DIST (v solamente)
          3. psi+lat  — cuadrar perpendicular y lateral (w solamente)

        Una vez los tres ejes estan en tolerancia N frames seguidos -> FIN.
        """
        if pose is None:
            self.qr_lost += 1
            if self.qr_lost < self.relost_scan:
                self._scan()
            else:
                self.state = 'REVERIFICAR'
            return

        self.qr_lost = 0
        bearing = pose['bearing']
        dist    = pose['dist']
        psi     = pose['psi']
        e_lat   = pose['e_lat']
        e_dist  = dist - READY_DIST

        centered = abs(bearing) < BEARING_TOL
        at_dist  = abs(e_dist)  < DIST_TOL
        square   = abs(psi)     < PSI_TOL
        on_axis  = abs(e_lat)   < LAT_TOL

        # Re-maniobra si quedo muy perpendicular
        if abs(psi) > self.remaneuver_perp and self.retries < self.max_retries:
            self.retries    += 1
            self.obs_buf     = []
            self.listo_count = 0
            self.state       = 'OBSERVAR'
            self._stop()
            self.get_logger().warn(
                f'Re-maniobra #{self.retries}: psi={math.degrees(psi):.1f}deg')
            return

        # Todo en tolerancia: contar frames estables
        if centered and at_dist and square and on_axis:
            self.listo_count += 1
            self._stop()
            self.get_logger().info(
                f'[AJUSTE] ESTABLE {self.listo_count}/{self.listo_need}',
                throttle_duration_sec=0.3)
            if self.listo_count >= self.listo_need:
                self.cen_pub.publish(Bool(data=True))
                self.state = 'FIN'
                self.get_logger().info('FIN — objeto alineado correctamente')
            return

        # Cualquier eje salio: reiniciar estabilidad
        self.listo_count = 0
        self.cen_pub.publish(Bool(data=False))

        # PASO 1 — bearing (solo w)
        # Velocidad proporcional con ramp suave: evita el salto brusco inicial.
        # Referencia: error maximo esperado al entrar = scan_entry_bearing (40deg).
        # A 40deg -> w_max; a BEARING_TOL -> w_min (15% de w_max); interpolacion lineal.
        if not centered or not self.bearing_hold_ok:
            now = self.get_clock().now()

            if centered:
                # Dentro de tolerancia: mantener parado y contar hold de 3 s.
                self._stop()
                if self.bearing_hold_t0 is None:
                    self.bearing_hold_t0 = now
                held = (now - self.bearing_hold_t0).nanoseconds * 1e-9
                self.get_logger().info(
                    f'[AJUSTE-1-BEARING] HOLD {held:.1f}/{self.bearing_hold_time:.1f}s '
                    f'bearing={math.degrees(bearing):+.1f}deg',
                    throttle_duration_sec=0.3)
                if held >= self.bearing_hold_time:
                    self.bearing_hold_ok = True
                    self.get_logger().info('[AJUSTE-1-BEARING] Hold cumplido -> PASO 2')
            else:
                # Fuera de tolerancia: resetear hold y aplicar velocidad proporcional.
                self.bearing_hold_t0 = None
                self.bearing_hold_ok = False
                w_min  = self.w_max * 0.15
                # Rango: desde BEARING_TOL (w_min) hasta scan_entry_bearing (w_max)
                span   = self.scan_entry_bearing - BEARING_TOL
                scale  = float(np.clip((abs(bearing) - BEARING_TOL) / span, 0.0, 1.0))
                w_mag  = w_min + (self.w_max - w_min) * scale
                w      = TURN_SIGN * math.copysign(w_mag, bearing)
                self._send(0.0, w)
                self.get_logger().info(
                    f'[AJUSTE-1-BEARING] bearing={math.degrees(bearing):+.1f}deg '
                    f'scale={scale:.2f} w={w:+.4f}',
                    throttle_duration_sec=0.3)
            return

        # PASO 2 — distancia (solo v)
        if not at_dist:
            v = float(np.clip(self.k_v * e_dist, -self.v_rev, self.v_max))
            self._send(v, 0.0)
            self.get_logger().info(
                f'[AJUSTE-2-DIST] dist={dist:.3f} e_dist={e_dist:+.3f} v={v:+.4f}',
                throttle_duration_sec=0.3)
            return

        # PASO 3 — perpendicular + lateral (solo w, velocidad proporcional)
        if not square or not on_axis:
            perp_raw = PERP_SIGN * (self.k_psi * psi + self.k_e * e_lat)
            # Escala proporcional: cerca de 0 -> mas lento
            p_min = self.perp_cap * 0.15
            scale = float(np.clip(abs(perp_raw) / self.perp_cap, 0.0, 1.0))
            p_mag = p_min + (self.perp_cap - p_min) * scale
            perp  = math.copysign(p_mag, perp_raw)
            w = TURN_SIGN * float(np.clip(perp, -self.perp_cap, self.perp_cap))
            self._send(0.0, w)
            self.get_logger().info(
                f'[AJUSTE-3-PERP] psi={math.degrees(psi):+.1f}deg '
                f'e_lat={e_lat*100:+.1f}cm scale={scale:.2f} w={w:+.4f}',
                throttle_duration_sec=0.3)
            return


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