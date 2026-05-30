"""
bug2_monitor.py
═══════════════════════════════════════════════════════════════════════
Nodo 3 de 3 — Evasión de obstáculos Bug 2

Monitorea el LiDAR continuamente. Cuando detecta un obstáculo frente
al robot, toma el control de /cmd_vel y publica /bug2/active = True.
El waypoint_controller cede el paso mientras este flag esté activo.

Una vez resuelto el obstáculo (robot de regreso en la M-line y más
cerca de la meta), publica /bug2/active = False y devuelve el control.

Tópicos suscritos:
  /scan           — sensor_msgs/LaserScan
  /odom           — nav_msgs/Odometry   (pose actual, desde Jetson)
  /plan           — nav_msgs/Path       (para conocer la meta actual)
  /nav/status     — std_msgs/String     (para saber el wp_idx activo)

Tópicos publicados:
  /cmd_vel        — geometry_msgs/Twist (solo cuando bug2_active=True)
  /bug2/active    — std_msgs/Bool
"""

import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from tf_transformations import euler_from_quaternion

import math

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — LiDAR
# ═══════════════════════════════════════════════════════════════════════

# Offset del LiDAR montado hacia atrás (180° = apunta al frente del robot)
ANGLE_OFFSET_DEG = 180.0

SECTOR_FRONT_CENTER = 0.0
SECTOR_FRONT_HW     = 15.0   # ±15° alrededor del frente
SECTOR_LEFT_CENTER  = 90.0
SECTOR_LEFT_HW      = 45.0
SECTOR_RIGHT_CENTER = 270.0
SECTOR_RIGHT_HW     = 45.0

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — BUG 2
# ═══════════════════════════════════════════════════════════════════════

DANGER_DIST = 0.25   # m — distancia mínima al frente para activar Bug2
WALL_DIST   = 0.20   # m — distancia lateral deseada durante wall-follow

# Ganancias wall-follow
KP_WF_LAT = 1.5     # corrección lateral
KP_W      = 0.10
KV_W      = 0.05

V_WF      = 0.08    # m/s — velocidad lineal en wall-follow
OMEGA_MAX = 0.30
V_MAX     = 0.22

# Tolerancia M-line
MLINE_TOL = 0.06    # m

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

class Bug2MonitorNode(Node):
    """
    Estados internos del nodo:
      IDLE        — sin obstáculo, no publica cmd_vel
      WALL_FOLLOW — esquivando obstáculo con Bug2
    """

    def __init__(self):
        super().__init__('bug2_monitor')

        # ── Publishers ────────────────────────────────────────────────
        self.cmd_vel_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.active_pub   = self.create_publisher(Bool, '/bug2/active', 10)

        # ── Subscribers ───────────────────────────────────────────────
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback,
            qos.qos_profile_sensor_data)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback,
            qos.qos_profile_sensor_data)

        self.plan_sub = self.create_subscription(
            Path, '/plan', self._plan_callback, 10)

        self.status_sub = self.create_subscription(
            String, '/nav/status', self._status_callback, 10)

        # ── Pose del robot ────────────────────────────────────────────
        self.x       = 0.0
        self.y       = 0.0
        self.theta   = 0.0
        self.w_robot = 0.0

        # ── LiDAR ─────────────────────────────────────────────────────
        self.scan_msg   = None
        self.front_dist = math.inf
        self.left_dist  = math.inf
        self.right_dist = math.inf

        # ── Plan y waypoint activo ────────────────────────────────────
        self.waypoints = []   # lista de (x, y)
        self.wp_idx    = 0    # índice activo (extraído de /nav/status)

        # ── Estado Bug 2 ──────────────────────────────────────────────
        self.state     = 'IDLE'
        self.wall_side = 1   # +1 izquierda, -1 derecha

        self.bug2_start_x     = 0.0
        self.bug2_start_y     = 0.0
        self.hit_x            = None
        self.hit_y            = None
        self.hit_dist_to_goal = math.inf

        # ── Timer de control a 50 Hz ──────────────────────────────────
        self.timer = self.create_timer(1 / 50, self._control_loop)

        self.get_logger().info('bug2_monitor listo')

    # ─────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────

    def _scan_callback(self, msg: LaserScan):
        self.scan_msg = msg
        self._update_sectors()

    def _odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.theta   = yaw
        self.w_robot = msg.twist.twist.angular.z

    def _plan_callback(self, msg: Path):
        self.waypoints = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]

    def _status_callback(self, msg: String):
        # Extraer wp_idx del string de estado publicado por waypoint_controller
        # Formato: "state=... | wp=N/M | ..."
        try:
            for part in msg.data.split('|'):
                part = part.strip()
                if part.startswith('wp='):
                    nums = part.split('=')[1].split('/')
                    self.wp_idx = int(nums[0])
                    break
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    #  LiDAR — sectores
    # ─────────────────────────────────────────────────────────────────

    def _get_sector(self, center_deg, half_width_deg) -> float:
        if self.scan_msg is None:
            return math.inf
        msg = self.scan_msg
        n   = len(msg.ranges)
        if n == 0:
            return math.inf

        center_rad = math.radians(center_deg + ANGLE_OFFSET_DEG)
        hw_rad     = math.radians(half_width_deg)
        angle_min  = msg.angle_min
        angle_inc  = msg.angle_increment
        range_min  = max(msg.range_min, 0.01)
        range_max  = msg.range_max if msg.range_max > 0 else 12.0

        vals = []
        for k in range(n):
            angle_k = angle_min + k * angle_inc
            if abs(wrap_angle(angle_k - center_rad)) <= hw_rad:
                r = msg.ranges[k]
                if math.isfinite(r) and range_min <= r <= range_max:
                    vals.append(r)
        return min(vals) if vals else math.inf

    def _update_sectors(self):
        self.front_dist = self._get_sector(SECTOR_FRONT_CENTER, SECTOR_FRONT_HW)
        self.left_dist  = self._get_sector(SECTOR_LEFT_CENTER,  SECTOR_LEFT_HW)
        self.right_dist = self._get_sector(SECTOR_RIGHT_CENTER, SECTOR_RIGHT_HW)

    # ─────────────────────────────────────────────────────────────────
    #  LOOP PRINCIPAL
    # ─────────────────────────────────────────────────────────────────

    def _control_loop(self):
        active_msg = Bool()

        if self.state == 'IDLE':
            # ¿Hay obstáculo al frente?
            if self.front_dist < DANGER_DIST:
                self.wall_side = 1 if self.left_dist >= self.right_dist else -1
                self.hit_x            = self.x
                self.hit_y            = self.y
                self.bug2_start_x     = self.x
                self.bug2_start_y     = self.y
                self.hit_dist_to_goal = self._dist_to_goal()
                self.state = 'WALL_FOLLOW'
                self.get_logger().info(
                    f'Obstáculo detectado (frente={self.front_dist:.2f}m) → '
                    f'WALL_FOLLOW ({"izq" if self.wall_side==1 else "der"})')

            active_msg.data = False
            self.active_pub.publish(active_msg)
            return   # No publicar cmd_vel en IDLE

        # ── WALL_FOLLOW ───────────────────────────────────────────────
        active_msg.data = True
        self.active_pub.publish(active_msg)

        cmd = self._wall_follow_cmd()
        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info(
            f'[WALL_FOLLOW] F={self.front_dist:.2f} '
            f'L={self.left_dist:.2f} R={self.right_dist:.2f} | '
            f'v={cmd.linear.x:+.3f} w={cmd.angular.z:+.3f}')

    # ─────────────────────────────────────────────────────────────────
    #  WALL FOLLOW
    # ─────────────────────────────────────────────────────────────────

    def _wall_follow_cmd(self) -> Twist:
        cmd     = Twist()
        lateral = self.left_dist if self.wall_side == 1 else self.right_dist
        lat_err = lateral - WALL_DIST

        # Sub-caso 1: esquina — giro in-situ
        if self.front_dist < DANGER_DIST * 0.7:
            cmd.linear.x  = 0.0
            cmd.angular.z = -self.wall_side * 0.15
            return cmd

        # Sub-caso 2: frente cercano — frenar y girar suave
        if self.front_dist < DANGER_DIST * 1.2:
            cmd.linear.x  = -0.04
            cmd.angular.z = -self.wall_side * 0.20
            return cmd

        # Sub-caso 3: avance normal con corrección lateral
        cmd.linear.x  = V_WF
        cmd.angular.z = clamp(
            self.wall_side * KP_WF_LAT * lat_err,
            -OMEGA_MAX, OMEGA_MAX)

        # ── Condición de salida Bug 2 ──────────────────────────────────
        dist_to_goal = self._dist_to_goal()
        if (self._on_mline()
                and self.hit_x is not None
                and dist_to_goal < self.hit_dist_to_goal - 0.05
                and self.front_dist > DANGER_DIST
                and math.hypot(self.x - self.hit_x, self.y - self.hit_y) > 0.10):
            self._exit_wall_follow()

        return cmd

    # ─────────────────────────────────────────────────────────────────
    #  HELPERS Bug 2
    # ─────────────────────────────────────────────────────────────────

    def _dist_to_goal(self) -> float:
        if not self.waypoints or self.wp_idx >= len(self.waypoints):
            return math.inf
        gx, gy = self.waypoints[self.wp_idx]
        return math.hypot(gx - self.x, gy - self.y)

    def _on_mline(self) -> bool:
        if not self.waypoints or self.wp_idx >= len(self.waypoints):
            return False
        gx, gy = self.waypoints[self.wp_idx]
        sx, sy = self.bug2_start_x, self.bug2_start_y
        lx, ly = gx - sx, gy - sy
        length = math.hypot(lx, ly)
        if length < 1e-6:
            return False
        nx, ny = -ly / length, lx / length
        rx, ry = self.x - sx, self.y - sy
        perp   = abs(rx * nx + ry * ny)
        proj   = rx * (lx / length) + ry * (ly / length)
        return perp < MLINE_TOL and 0.0 <= proj <= length

    def _exit_wall_follow(self):
        self.state    = 'IDLE'
        self.hit_x    = None
        self.hit_y    = None
        self.hit_dist_to_goal = math.inf
        self.get_logger().info('Bug2 resuelto → IDLE (devolviendo control)')


# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = Bug2MonitorNode()
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
