"""
waypoint_controller.py
═══════════════════════════════════════════════════════════════════════
Nodo 2 de 3 — Control de waypoints

Se suscribe al plan publicado por semantic_planner y sigue cada
waypoint con control PI (velocidad lineal) + P (angular).
Cuando el nodo bug2_monitor detecta un obstáculo, este nodo cede
el control: deja de publicar cmd_vel y espera a que bug2 termine.

Tópicos suscritos:
  /plan           — nav_msgs/Path       (waypoints del planificador)
  /odom           — nav_msgs/Odometry   (pose del robot, desde Jetson)
  /bug2/active    — std_msgs/Bool       (True = Bug2 tomó el control)

Tópicos publicados:
  /cmd_vel        — geometry_msgs/Twist
  /nav/status     — std_msgs/String     (estado actual para debug)
"""

import rclpy
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from tf_transformations import euler_from_quaternion

import math

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

# Ganancias de control
KP_V      = 0.23    # proporcional velocidad lineal
KI_V      = 0.20    # integral velocidad lineal
KP_W      = 0.20    # proporcional velocidad angular
KV_W      = 0.05    # amortiguación velocidad angular (usa w_robot de odom)

# Límites de velocidad
V_MAX     = 0.22    # m/s
OMEGA_MAX = 0.30    # rad/s

# Tolerancias
GOAL_DIST_TOL  = 0.05   # m — radio para considerar waypoint alcanzado
ANGLE_PRIORITY = math.radians(3)  # rad (~3°) — umbral para activar movimiento lineal

    
# ═══════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ═══════════════════════════════════════════════════════════════════════
#  NODO
# ═══════════════════════════════════════════════════════════════════════

class WaypointControllerNode(Node):
    """
    Máquina de estados interna:
      WAIT_PLAN → ROTATE_TO_GOAL → GO_TO_GOAL → DONE
    (WAIT_PLAN persiste si Bug2 tiene el control o no hay plan aún)
    """

    def __init__(self):
        super().__init__('waypoint_controller')

        # ── Publishers ────────────────────────────────────────────────
        self.cmd_vel_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub   = self.create_publisher(String, '/nav/status', 10)

        # ── Subscribers ───────────────────────────────────────────────
        self.plan_sub = self.create_subscription(
            Path, '/plan', self._plan_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback,
            qos.qos_profile_sensor_data)

        self.bug2_sub = self.create_subscription(
            Bool, '/bug2/active', self._bug2_callback, 10)

        # ── Estado ────────────────────────────────────────────────────
        self.waypoints    = []   # lista de (x, y)
        self.wp_idx       = 0

        self.x      = 0.0
        self.y      = 0.0
        self.theta  = 0.0
        self.w_robot = 0.0   # velocidad angular del robot (de /odom)

        self.int_error_d = 0.0
        self.x_start     = 0.0
        self.y_start     = 0.0

        self.bug2_active = False   # True = Bug2 tiene el control

        self.state = 'WAIT_PLAN'

        # ── Timer de control a 50 Hz ──────────────────────────────────
        self.last_time = self.get_clock().now()
        self.timer     = self.create_timer(1 / 50, self._control_loop)

        self.get_logger().info('waypoint_controller listo — esperando /plan y /odom')

    # ─────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────

    def _plan_callback(self, msg: Path):
        if not msg.poses:
            return
        # Actualizar waypoints solo si el plan cambió o aún no teníamos
        new_wps = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if new_wps == self.waypoints:
            return
        self.waypoints = new_wps
        self.wp_idx    = 0
        self.int_error_d = 0.0
        if self.state in ('WAIT_PLAN', 'DONE'):
            self.state = 'ROTATE_TO_GOAL'
            self.get_logger().info(
                f'Plan recibido: {len(self.waypoints)} waypoints — iniciando navegación')

    def _odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.theta = yaw

        # Velocidad angular del robot (útil para amortiguación)
        self.w_robot = msg.twist.twist.angular.z

    def _bug2_callback(self, msg: Bool):
        prev = self.bug2_active
        self.bug2_active = msg.data
        if prev and not self.bug2_active:
            # Bug2 acaba de liberar el control: reorientar antes de avanzar
            self.int_error_d = 0.0
            self.x_start     = self.x
            self.y_start     = self.y
            if self.state not in ('DONE', 'WAIT_PLAN'):
                self.state = 'ROTATE_TO_GOAL'
                self.get_logger().info('Bug2 liberó el control → ROTATE_TO_GOAL')

    # ─────────────────────────────────────────────────────────────────
    #  LOOP DE CONTROL
    # ─────────────────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now()
        dt  = clamp((now - self.last_time).nanoseconds * 1e-9, 1e-4, 0.1)
        self.last_time = now

        # Publicar estado
        status = String()
        status.data = (
            f'state={self.state} | wp={self.wp_idx}/{len(self.waypoints)} | '
            f'bug2={self.bug2_active} | '
            f'x={self.x:.2f} y={self.y:.2f} th={math.degrees(self.theta):.1f}°'
        )
        self.status_pub.publish(status)

        cmd = Twist()

        # Ceder el control si Bug2 está activo
        if self.bug2_active:
            self.cmd_vel_pub.publish(cmd)
            return

        if self.state == 'WAIT_PLAN':
            self.cmd_vel_pub.publish(cmd)
            return

        if self.state == 'DONE':
            self.cmd_vel_pub.publish(cmd)
            return

        if not self.waypoints or self.wp_idx >= len(self.waypoints):
            self.state = 'DONE'
            self.get_logger().info('✓ Todos los waypoints alcanzados')
            self.cmd_vel_pub.publish(cmd)
            return

        wx, wy = self.waypoints[self.wp_idx]
        dist_to_goal  = math.hypot(wx - self.x, wy - self.y)
        angle_to_goal = math.atan2(wy - self.y, wx - self.x)
        error_theta   = wrap_angle(angle_to_goal - self.theta)

        if self.state == 'ROTATE_TO_GOAL':
            cmd = self._rotate(error_theta)

        elif self.state == 'GO_TO_GOAL':
            cmd = self._go_to_goal(dist_to_goal, error_theta, dt)

        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info(
            f'[{self.state}] wp {self.wp_idx} | '
            f'ed={dist_to_goal:.3f}m eθ={math.degrees(error_theta):.1f}° | '
            f'v={cmd.linear.x:+.3f} w={cmd.angular.z:+.3f}')

    # ─────────────────────────────────────────────────────────────────
    #  ESTADOS
    # ─────────────────────────────────────────────────────────────────

    def _rotate(self, error_theta) -> Twist:
        cmd = Twist()
        if abs(error_theta) < ANGLE_PRIORITY:
            self.x_start     = self.x
            self.y_start     = self.y
            self.int_error_d = 0.0
            self.state = 'GO_TO_GOAL'
            self.get_logger().info(f'→ GO_TO_GOAL (wp {self.wp_idx})')
            return cmd
        cmd.angular.z = clamp(
            KP_W * error_theta - KV_W * self.w_robot,
            -OMEGA_MAX, OMEGA_MAX)
        return cmd

    def _go_to_goal(self, dist, error_theta, dt) -> Twist:
        cmd = Twist()

        # Waypoint alcanzado
        if dist < GOAL_DIST_TOL:
            self.wp_idx      += 1
            self.int_error_d  = 0.0
            if self.wp_idx >= len(self.waypoints):
                self.state = 'DONE'
                self.get_logger().info('✓ Último waypoint alcanzado — DONE')
            else:
                self.state = 'ROTATE_TO_GOAL'
                self.get_logger().info(
                    f'Waypoint {self.wp_idx} alcanzado → ROTATE_TO_GOAL')
            return cmd

        # Prioridad angular: si el error de ángulo es grande, solo girar
        if abs(error_theta) > ANGLE_PRIORITY:
            self.int_error_d = 0.0
            cmd.angular.z = clamp(
                KP_W * error_theta - KV_W * self.w_robot,
                -OMEGA_MAX, OMEGA_MAX)
            return cmd

        # Control PI lineal + P angular
        self.int_error_d = clamp(self.int_error_d + dist * dt, -2.0, 2.0)
        dr  = math.hypot(self.x - self.x_start, self.y - self.y_start)
        u_v = clamp(KI_V * self.int_error_d - KP_V * dr, 0.0, V_MAX)
        u_w = clamp(KP_W * error_theta - KV_W * self.w_robot, -OMEGA_MAX, OMEGA_MAX)

        cmd.linear.x  = u_v
        cmd.angular.z = u_w
        return cmd


# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = WaypointControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
