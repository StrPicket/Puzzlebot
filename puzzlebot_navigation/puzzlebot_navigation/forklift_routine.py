#!/usr/bin/env python3
"""
forklift_rutine.py
═════════════════════════════════════════════════════════════════════
Rutina para el manejo del forklift del Puzzlebot

Mission 1 (zona de carga):
    - Pre-elevar horquillas vacías a altura del pallet
    - Avanzar reaching_dist (0.5 m) para meter horquillas
    - Levantar pallet (LIFTING)
    - Retroceder reaching_dist (LEAVING)
    --------------------------------------------------
    - Avanzar entering_dist al truck (REACHING con DROP_PALLET)
    - Bajar pallet (DROPPING)
    - Retroceder entering_dist para salir del truck (LEAVING)

Mission 2 (zona de racks):
    - Pre-elevar horquillas vacías a altura del pallet
    - Avanzar reaching_dist (0.3 m) para meter horquillas
    - Levantar pallet (LIFTING)
    - Retroceder reaching_dist (LEAVING)
    --------------------------------------------------
    - Avanzar entering_dist al truck (REACHING con DROP_PALLET)
    - Bajar pallet (DROPPING)
    - Retroceder entering_dist para salir del truck (LEAVING)

Flujo de estados internos:
    IDLE → REACHING → LIFTING → LEAVING → DONE
    IDLE → REACHING → DROPPING → LEAVING → DONE

Separación de responsabilidades:
    control()   — solo mueve el robot, señala waypoint_reached
    _fsm_step() — gestiona transiciones de estado y forklift
"""

import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Float32, String, Bool
from geometry_msgs.msg import Twist

import math

from forklift_interfaces.srv import ForkliftCommand

class ForkliftRutine(Node):
    def __init__(self):
        super().__init__('forklift_rutine')

        # ── Publishers ───────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.state_pub   = self.create_publisher(String, 'forklift/status', 10)

        # ── Subscribers ──────────────────────────────────────────
        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_callback,
            qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_callback,
            qos.qos_profile_sensor_data)
        self.sub_mission_status = self.create_subscription(
            String, 'mission/status', self.mission_status_callback, 10)
        
        self.forklift_client = self.create_client(
            ForkliftCommand,
            '/forklift_command'
        )

        # ── Estado odométrico ────────────────────────────────────
        self.x      = 0.0
        self.y      = 0.0
        self.theta  = 0.0
        self.wr     = Float32()
        self.wl     = Float32()
        self.w_robot = 0.0
        self.v_robot = 0.0

        self.radio  = 0.0505
        self.lenght = 0.183

        # ── Parámetros de distancia ──────────────────────────────
        # reaching_dist : distancia a avanzar para meter horquillas
        #                 (diferente por misión, se asigna en _on_mission_state_changed)
        # entering_dist : distancia al truck — igual para ambas misiones
        self.waypoint_dist  = 0.0
        self.reaching_dist  = 0.0
        self.entering_dist  = 1.5   # m — distancia de entrada al truck

        # ── Comandos de forklift ─────────────────────────────────
        # forklift_pre_cmd  : altura para meter horquillas vacías (REACHING)
        # forklift_lift_cmd : altura para levantar el pallet cargado (LIFTING)
        # forklift_drop_cmd : posición para bajar el pallet (DROPPING)
        # Se asignan en _on_mission_state_changed según la misión.
        self.forklift_pre_cmd  = None
        self.forklift_lift_cmd = None
        self.forklift_drop_cmd = None

        # ── Ganancias control lineal (PI) ────────────────────────
        self.Kp_v        = 0.15
        self.Ki_v        = 0.25
        self.int_error_l = 0.0

        # ── Ganancias control angular (P + amortiguación) ────────
        self.Kp_w = 0.08
        self.Kv_w = 0.05

        # ── Estado de la misión (recibido por tópico) ─────────────
        self.current_mission       = None   # 'mission_1' | 'mission_2'
        self.current_mission_state = None   # 'LIFT_PALLET' | 'DROP_PALLET' | …

        # ── Máquina de estados interna ───────────────────────────
        # IDLE     : esperando instrucción
        # REACHING : avanzando hasta waypoint_dist
        # LIFTING  : levantando pallet (espera por tiempo)
        # DROPPING : bajando pallet (espera por tiempo)
        # LEAVING  : retrocediendo -waypoint_dist
        # DONE     : secuencia completa
        self.current_state = 'IDLE'

        # Flag que control() activa cuando llega al waypoint;
        # _fsm_step() lo consume para hacer la transición de estado.
        self.waypoint_reached = False
        self.command_sent = False

        # ── Timers ───────────────────────────────────────────────
        self.last_time_odom    = self.get_clock().now()
        self.last_time_control = self.get_clock().now()
        self.state_start_time  = self.get_clock().now()

        self.timer_odom = self.create_timer(1 / 100, self.odometria)
        self.timer_ctrl = self.create_timer(1 / 20,  self.control)
        self.timer_fsm  = self.create_timer(1 / 10,  self._fsm_step)

        self.get_logger().info('forklift_rutine listo — esperando mission/status')

    # ── Callbacks encoders ────────────────────────────────────────────────

    def encR_callback(self, msg: Float32):
        self.wr = msg

    def encL_callback(self, msg: Float32):
        self.wl = msg

    # ── Callback misión ───────────────────────────────────────────────────

    def mission_status_callback(self, msg: String):
        """
        Parsea el string 'mission=X | state=Y' y dispara _on_mission_state_changed
        solo cuando el estado de misión cambia efectivamente.
        """
        for part in msg.data.split('|'):
            part = part.strip()
            if part.startswith('mission='):
                self.current_mission = part.split('=')[1].strip()
            elif part.startswith('state='):
                new_state = part.split('=')[1].strip()
                if new_state != self.current_mission_state:
                    self.current_mission_state = new_state
                    self._on_mission_state_changed()

    # ── Odometría relativa ────────────────────────────────────────────────

    def _reset_odom(self):
        """Resetea la odometría local (origen = posición actual del robot)."""
        self.x           = 0.0
        self.y           = 0.0
        self.theta       = 0.0
        self.int_error_l = 0.0

    def odometria(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_odom).nanoseconds * 1e-9
        self.last_time_odom = current_time

        if dt <= 0:
            return

        v_r   = self.radio * self.wr.data
        v_l   = self.radio * self.wl.data
        V_avg = (v_r + v_l) / 2.0
        W     = (v_r - v_l) / self.lenght

        self.v_robot = 0.15 * self.v_robot + 0.85 * V_avg
        self.w_robot = 0.15 * self.w_robot + 0.85 * W

        self.x     += V_avg * math.cos(self.theta) * dt
        self.y     += V_avg * math.sin(self.theta) * dt
        self.theta += W * dt
        self.theta  = (self.theta + math.pi) % (2 * math.pi) - math.pi

    # ── Control de movimiento ─────────────────────────────────────────────

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.state_start_time).nanoseconds * 1e-9

    def control(self):
        """
        Responsabilidad única: mover el robot hacia waypoint_dist.
        Cuando llega, activa waypoint_reached y resetea la odometría.
        NO cambia current_state — eso lo hace _fsm_step().
        """
        cmd = Twist()

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_control).nanoseconds * 1e-9
        self.last_time_control = current_time
        dt = min(dt, 0.1)

        if self.current_state in ('REACHING', 'LEAVING'):
            error_x     = self.waypoint_dist - self.x
            error_theta = -self.theta   # mantener recto (theta=0)

            if abs(error_x) < 0.05:
                # Waypoint alcanzado — señalar a _fsm_step y limpiar
                self._reset_odom()
                self.waypoint_dist    = 0.0
                self.waypoint_reached = True
            else:
                self.int_error_l += error_x * dt
                v_cmd = self.Ki_v * self.int_error_l - self.Kp_v * self.x
                w_cmd = self.Kp_w * error_theta - self.Kv_w * self.w_robot

                cmd.linear.x  = max(min(v_cmd, 0.20), -0.20)
                cmd.angular.z = max(min(w_cmd, 0.20), -0.20)

        # En LIFTING, DROPPING, IDLE, DONE: robot quieto
        self.cmd_vel_pub.publish(cmd)

    # ── Inicialización de transición ──────────────────────────────────────

    def _on_mission_state_changed(self):
        """
        Se llama UNA sola vez cuando mission_state cambia.
        Configura distancias y alturas de forklift según misión y estado,
        y reinicia la FSM interna desde REACHING.
        """
        if self.current_mission_state == 'LIFT_PALLET':
            if self.current_mission == 'mission_1':
                # Zona de carga: pallet en conveyor
                self.reaching_dist    = 0.5   # m — distancia a meter horquillas
                self.forklift_pre_cmd = {
                    "cmd": 1,
                    "speed": 255,
                    "duration": 1.30
                }
                self.forklift_lift_cmd = {
                    "cmd": 1,
                    "speed": 255,
                    "duration": 0.5
                }
            elif self.current_mission == 'mission_2':
                # Zona de racks: pallet en rack
                self.reaching_dist    = 0.3
                self.forklift_pre_cmd = {
                    "cmd": 1,
                    "speed": 255,
                    "duration": 0.5
                }
                self.forklift_lift_cmd = {
                    "cmd": 1,
                    "speed": 255,
                    "duration": 0.3
                }
            self.forklift_drop_cmd = None   # no aplica en este estado

        elif self.current_mission_state == 'DROP_PALLET':
            # Misma distancia de entrada al truck en ambas misiones
            self.reaching_dist    = self.entering_dist
            self.forklift_pre_cmd  = None  # no pre-elevar al entrar al truck
            if self.current_mission == 'mission_1':
                self.forklift_drop_cmd = {
                    "cmd": 2,
                    "speed": 255,
                    "duration": 1.2
                }  # bajar pallet
            elif self.current_mission == 'mission_2':
                self.forklift_drop_cmd = {
                    "cmd": 2,
                    "speed": 255,
                    "duration": 0.55
                }  # bajar pallet
            self.forklift_lift_cmd = None  # no aplica en este estado

        else:
            # Estado de misión no reconocido — no hacer nada
            self.get_logger().warn(
                f'mission_state desconocido: {self.current_mission_state}')
            return

        # Reiniciar FSM interna
        self._reset_odom()
        self.waypoint_dist    = 0.0
        self.waypoint_reached = False
        self.command_sent = False
        self.current_state    = 'REACHING'
        self.state_start_time = self.get_clock().now()

        self.get_logger().info(
            f'[FSM] Nueva secuencia: mission={self.current_mission} '
            f'state={self.current_mission_state} '
            f'reaching_dist={self.reaching_dist:.2f}m')

    # ── Máquina de estados ────────────────────────────────────────────────

    def _send_forklift_cmd(self, cmd_data):

        req = ForkliftCommand.Request()
        req.cmd = cmd_data["cmd"]
        req.speed = cmd_data["speed"]
        req.duration = cmd_data["duration"]

        self.forklift_client.call_async(req)


    def _fsm_step(self):
        """
        Gestiona transiciones de estado y comandos de forklift.
        Consume waypoint_reached producido por control().
        """
        self._publish_status()

        # ── REACHING ──────────────────────────────────────────────
        if self.current_state == 'REACHING':

            if self.current_mission_state == 'LIFT_PALLET':
                # 1. Pre-elevar horquillas vacías a la altura del pallet
                if self.forklift_pre_cmd is not None and not self.command_sent:
                    self._send_forklift_cmd(self.forklift_pre_cmd)
                    self.command_sent = True

                # 2. Esperar 2 s a que el forklift suba antes de avanzar
                if self._elapsed() > 2.0 and self.waypoint_dist == 0.0:
                    self.waypoint_dist = self.reaching_dist
                    self.get_logger().info(
                        f'[REACHING] Forklift listo — avanzando {self.reaching_dist:.2f}m')

            elif self.current_mission_state == 'DROP_PALLET':
                # Entrar al truck directo, sin pre-elevación
                if self.waypoint_dist == 0.0:
                    self.waypoint_dist = self.reaching_dist
                    self.get_logger().info(
                        f'[REACHING] Entrando al truck — {self.reaching_dist:.2f}m')

            # Waypoint alcanzado → transición según misión
            if self.waypoint_reached:
                self.waypoint_reached = False
                self.state_start_time = self.get_clock().now()

                if self.current_mission_state == 'LIFT_PALLET':
                    self.current_state = 'LIFTING'
                    self.command_sent = False
                    self.get_logger().info('[FSM] REACHING → LIFTING')
                elif self.current_mission_state == 'DROP_PALLET':
                    self.current_state = 'DROPPING'
                    self.command_sent = False
                    self.get_logger().info('[FSM] REACHING → DROPPING')

        # ── LIFTING ───────────────────────────────────────────────
        elif self.current_state == 'LIFTING':
            if self.forklift_lift_cmd is not None and not self.command_sent:
                self._send_forklift_cmd(self.forklift_lift_cmd)
                self.command_sent = True

            if self._elapsed() > 2.0:
                self.current_state    = 'LEAVING'
                self.waypoint_dist    = -self.reaching_dist   # retroceder
                self.state_start_time = self.get_clock().now()
                self.get_logger().info(
                    f'[FSM] LIFTING → LEAVING ({self.waypoint_dist:.2f}m)')

        # ── DROPPING ──────────────────────────────────────────────
        elif self.current_state == 'DROPPING':
            if self.forklift_drop_cmd is not None and not self.command_sent:
                self._send_forklift_cmd(self.forklift_drop_cmd)
                self.command_sent = True

            if self._elapsed() > 2.0:
                self.current_state    = 'LEAVING'
                self.waypoint_dist    = -self.reaching_dist   # salir del truck
                self.state_start_time = self.get_clock().now()
                self.get_logger().info(
                    f'[FSM] DROPPING → LEAVING ({self.waypoint_dist:.2f}m)')

        # ── LEAVING ───────────────────────────────────────────────
        elif self.current_state == 'LEAVING':
            if self.waypoint_reached:
                self.waypoint_reached = False
                self.current_state    = 'DONE'
                self.get_logger().info('[FSM] LEAVING → DONE')

        # ── DONE / IDLE ───────────────────────────────────────────
        elif self.current_state in ('DONE', 'IDLE'):
            pass   # esperar nueva instrucción de mission/status

    def _publish_status(self):
        msg = String()
        msg.data = (
            f'state={self.current_state} | '
            f'mission={self.current_mission} | '
            f'mission_state={self.current_mission_state} | '
            f'wp={self.waypoint_dist:.2f} | '
            f'x={self.x:.2f}'
        )
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ForkliftRutine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())   # frenar al salir
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()