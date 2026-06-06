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
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from tf_transformations import euler_from_quaternion

import math

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

# Ganancias de control
KP_V      = 0.30    # proporcional velocidad lineal
KV_V      = 0.01    # integral velocidad lineal
KP_W      = 0.30    # proporcional velocidad angular
KV_W      = 0.02    # amortiguación velocidad angular (usa w_robot de odom)

# Límites de velocidad
V_MAX     = 0.15    # m/s
OMEGA_MAX = 0.15    # rad/s

# Tolerancias
GOAL_DIST_TOL  = 0.15   # m — radio para considerar waypoint alcanzado
ANGLE_PRIORITY = math.radians(5)  # rad (~3°) — umbral para activar movimiento lineal

    
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
        self.markers_pub = self.create_publisher(MarkerArray, '/nav/waypoints_viz', 10)

        # ── Subscribers ───────────────────────────────────────────────
        self.cancel_sub = self.create_subscription(Bool, '/nav/cancel', self._cancel_cb, 10)

        self.plan_sub = self.create_subscription(
            Path, '/plan', self._plan_callback, 10)
        
        self.kf_pose_active = False

        self.kf_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/aruco/pose_centered',
            self._kf_pose_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback,
            qos.qos_profile_sensor_data)
        
        self.mission_status_sub = self.create_subscription(
            String, '/mission/status', self._mission_status_cb, 10)

        self.bug2_sub = self.create_subscription(
            Bool, '/bug2/active', self._bug2_callback, 10)

        # ── Estado ────────────────────────────────────────────────────
        self.waypoints    = []   # lista de (x, y)
        self.wp_idx       = 0

        self.x      = 0.0
        self.y      = 0.0
        self.theta  = 0.0
        self.v_robot = 0.0
        self.w_robot = 0.0   # velocidad angular del robot (de /odom)

        self.u_v = 0.0
        self.u_w = 0.0

        self.int_error_d = 0.0
        self.x_start     = 0.0
        self.y_start     = 0.0

        self.bug2_active = False   # True = Bug2 tiene el control
        self.mission_status = None
        self.mission_sub_status = None

        self.state = 'WAIT_PLAN'

        # ── Timer de control a 50 Hz ──────────────────────────────────
        self.last_time = self.get_clock().now()
        self.timer     = self.create_timer(1 / 50, self._control_loop)

        self.get_logger().info('waypoint_controller listo — esperando /plan y /odom')

    # ─────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────

    def _mission_status_cb(self, msg:String):
        for part in msg.data.split('|'):
            part = part.strip()
            if part.startswith('state='):
                self.mission_status = part.split('=')[1].strip()
            elif part.startswith('sub='):
                self.mission_sub_status = part.split('=')[1].strip()

    def _cancel_cb(self, msg: Bool):
        if msg.data:
            self.waypoints = []
            self.wp_idx = 0
            self.int_error_d = 0.0
            self.state = 'WAIT_PLAN'

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
            
    def _kf_pose_callback(self, msg: PoseWithCovarianceStamped):
        from tf_transformations import euler_from_quaternion
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.theta = yaw
        self.kf_pose_active = True
        # Nota: w_robot no viene en PoseWithCovarianceStamped,
        # se sigue tomando de /odom para la amortiguación angular

    def _odom_callback(self, msg: Odometry):
        # w_robot siempre se toma de /odom (velocidad instantánea)
        self.v_robot = msg.twist.twist.linear.x
        self.w_robot = msg.twist.twist.angular.z

        if self.kf_pose_active:
            return   # pose ya actualizada por KF

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.theta = yaw

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

    def _other_node_has_control(self) -> bool:
        scanning    = self.mission_sub_status in ('SCAN_ROTATE','SCAN_ROTATE_A', 'SCAN_ROTATE_B')
        forklift_op = self.mission_status in ('CENTER_QR', 'LIFT_PALLET', 'DROP_PALLET')
        return scanning or forklift_op

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
            f'u_v={self.u_v} u_w={self.u_w}' 
        )
        
        self.status_pub.publish(status)

        if self._other_node_has_control():
            return

        if self.waypoints:
            self._publish_waypoint_markers()

        cmd = Twist()

        # Ceder el control si Bug2 está activo
        if self.bug2_active:
            self.cmd_vel_pub.publish(cmd)
            return

        if self.state in ('WAIT_PLAN', 'DONE'):
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
        #self.get_logger().info(
        #    f'[{self.state}] wp {self.wp_idx} | '
        #    f'ed={dist_to_goal:.3f}m eθ={math.degrees(error_theta):.1f}° | '
        #    f'v={cmd.linear.x:+.3f} w={cmd.angular.z:+.3f}')

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

            if dist < GOAL_DIST_TOL:
                self.wp_idx += 1
                if self.wp_idx >= len(self.waypoints):
                    self.state = 'DONE'
                    self.get_logger().info('✓ Último waypoint alcanzado — DONE')
                else:
                    self.state = 'ROTATE_TO_GOAL'
                    self.get_logger().info(
                        f'Waypoint {self.wp_idx} alcanzado → ROTATE_TO_GOAL')
                return cmd

            if abs(error_theta) > ANGLE_PRIORITY:
                cmd.angular.z = clamp(
                    KP_W * error_theta - KV_W * self.w_robot,
                    -OMEGA_MAX, OMEGA_MAX)
                return cmd

            self.u_v = clamp(KP_V * dist - KV_V * self.v_robot, 0.0, V_MAX)
            self.u_w = clamp(KP_W * error_theta - KV_W * self.w_robot, -OMEGA_MAX, OMEGA_MAX)

            cmd.linear.x  = self.u_v
            cmd.angular.z = self.u_w
            return cmd
    
    def _publish_waypoint_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i, (wx, wy) in enumerate(self.waypoints):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'odom'
            m.ns = 'waypoints'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.08

            if i < self.wp_idx:
                # Ya visitados — gris
                m.color = ColorRGBA(r=0.4, g=0.4, b=0.4, a=0.5)
            elif i == self.wp_idx:
                # Waypoint activo — verde brillante, más grande
                m.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=1.0)
                m.scale.x = m.scale.y = m.scale.z = 0.15
            else:
                # Pendientes — azul
                m.color = ColorRGBA(r=0.3, g=0.6, b=1.0, a=0.8)

            ma.markers.append(m)

        # Borrar markers sobrantes de planes anteriores más largos
        for i in range(len(self.waypoints), len(self.waypoints) + 20):
            d = Marker()
            d.header.stamp = stamp
            d.header.frame_id = 'odom'
            d.ns = 'waypoints'
            d.id = i
            d.action = Marker.DELETE
            ma.markers.append(d)

        self.markers_pub.publish(ma)


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