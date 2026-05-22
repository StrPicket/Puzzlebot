"""
waypoints_bug.py — Waypoints + Bug 0 / Bug 2  (PuzzleBot + LiDAR /scan)
"""

import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import math

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

BUG_MODE = "BUG2"   # "BUG0" | "BUG2"

WAYPOINTS_X = [1.5]
WAYPOINTS_Y = [0.0]
WAYPOINTS_T = [0.0]

WHEEL_RADIUS = 0.0505
WHEEL_BASE   = 0.183

# Ganancias
KP_V      = 0.15
KP_W      = 0.10
KV_W      = 0.05
KP_WF_LAT = 1.5    # corrección lateral wall-follow

# LiDAR
DANGER_DIST      = 0.25    # m — trigger wall-follow / giro urgente
WALL_DIST        = 0.20    # m — distancia lateral deseada
ANGLE_OFFSET_DEG = 180.0   # 180° porque el LiDAR apunta hacia atrás

SECTOR_FRONT_CENTER = 0.0
SECTOR_FRONT_HW     = 15.0
SECTOR_LEFT_CENTER  = 90.0
SECTOR_LEFT_HW      = 45.0
SECTOR_RIGHT_CENTER = 270.0
SECTOR_RIGHT_HW     = 45.0

# Límites
V_MAX     = 0.22
OMEGA_MAX = 0.30
V_WF      = 0.08   # velocidad lineal en wall-follow normal

GOAL_DIST_TOL  = 0.05
ANGLE_PRIORITY = 0.05   # rad ≈ 3°


# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ═══════════════════════════════════════════════════════════════════════════
#  NODO
# ═══════════════════════════════════════════════════════════════════════════

class WaypointsBugNode(Node):
    """
    Máquina de estados:
      ROTATE_TO_GOAL → GO_TO_GOAL ↔ WALL_FOLLOW → GOAL_REACHED → DONE
    """

    def __init__(self):
        super().__init__('waypoints_bug')

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self._encR_cb, qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self._encL_cb, qos.qos_profile_sensor_data)
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, qos.qos_profile_sensor_data)

        self.timer_odom = self.create_timer(1 / 100, self._odometria)
        self.timer_ctrl = self.create_timer(1 / 50,  self._control)

        self.x = self.y = self.theta = 0.0
        self.wr = Float32()
        self.wl = Float32()
        self.v_robot = self.w_robot = 0.0

        self.last_time_odom    = self.get_clock().now()
        self.last_time_control = self.get_clock().now()

        self.waypoints   = list(zip(WAYPOINTS_X, WAYPOINTS_Y, WAYPOINTS_T))
        self.wp_idx      = 0
        self.int_error_d = 0.0
        self.x_start     = 0.0
        self.y_start     = 0.0

        self.scan_msg   = None
        self.front_dist = math.inf
        self.left_dist  = math.inf
        self.right_dist = math.inf

        self.bug_mode  = BUG_MODE
        self.state     = "ROTATE_TO_GOAL"
        self.wall_side = 1   # +1 pared izquierda, -1 pared derecha

        # Bug 2
        self.bug2_start_x     = 0.0
        self.bug2_start_y     = 0.0
        self.hit_x            = None
        self.hit_y            = None
        self.hit_dist_to_goal = math.inf

        self.get_logger().info(
            f'WaypointsBug iniciado | modo={self.bug_mode} | '
            f'{len(self.waypoints)} waypoints')

    # ── Callbacks ────────────────────────────────────────────────────────

    def _encR_cb(self, msg): self.wr = msg
    def _encL_cb(self, msg): self.wl = msg

    def _scan_cb(self, msg):
        self.scan_msg = msg
        self._update_sectors()

    # ── LiDAR ────────────────────────────────────────────────────────────

    def _get_sector(self, center_deg, half_width_deg):
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

    # ── Odometría ─────────────────────────────────────────────────────────

    def _odometria(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time_odom).nanoseconds * 1e-9
        self.last_time_odom = now
        if dt <= 0:
            return
        v_r = WHEEL_RADIUS * self.wr.data
        v_l = WHEEL_RADIUS * self.wl.data
        v_avg   = (v_r + v_l) / 2.0
        w_robot = (v_r - v_l) / WHEEL_BASE
        self.v_robot = v_avg
        self.w_robot = w_robot
        self.x     += v_avg * math.cos(self.theta) * dt
        self.y     += v_avg * math.sin(self.theta) * dt
        self.theta  = wrap_angle(self.theta + w_robot * dt)

    # ── Control principal ─────────────────────────────────────────────────

    def _control(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time_control).nanoseconds * 1e-9
        self.last_time_control = now
        dt = clamp(dt, 1e-4, 0.1)

        cmd = Twist()
        if self.state == "DONE":
            self.cmd_vel_pub.publish(cmd)
            return

        wx, wy, wt_deg = self.waypoints[self.wp_idx]
        wt_rad        = math.radians(wt_deg)
        dist_to_goal  = math.hypot(wx - self.x, wy - self.y)
        angle_to_goal = math.atan2(wy - self.y, wx - self.x)
        error_theta   = wrap_angle(angle_to_goal - self.theta)

        if   self.state == "ROTATE_TO_GOAL":
            cmd = self._state_rotate(error_theta)
        elif self.state == "GO_TO_GOAL":
            cmd = self._state_go_to_goal(dist_to_goal, error_theta, dt)
        elif self.state == "WALL_FOLLOW":
            cmd = self._state_wall_follow(wx, wy, dist_to_goal)
        elif self.state == "GOAL_REACHED":
            cmd = self._state_goal_reached(wt_rad)

        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info(
            f'[{self.state}] wp={self.wp_idx} '
            f'x={self.x:.2f} y={self.y:.2f} th={math.degrees(self.theta):.1f}° '
            f'F={self.front_dist:.2f} L={self.left_dist:.2f} R={self.right_dist:.2f}')

    # ── Estados ───────────────────────────────────────────────────────────

    def _state_rotate(self, error_theta):
        cmd = Twist()
        if abs(error_theta) < ANGLE_PRIORITY:
            self.state = "GO_TO_GOAL"
            self._reset_bug2()
            self.get_logger().info('→ GO_TO_GOAL')
            return cmd
        cmd.angular.z = clamp(KP_W * error_theta - KV_W * self.w_robot,
                               -OMEGA_MAX, OMEGA_MAX)
        return cmd

    def _state_go_to_goal(self, dist, error_theta, dt):
        cmd = Twist()

        if dist < GOAL_DIST_TOL:
            self.state = "GOAL_REACHED"
            self.get_logger().info(f'→ GOAL_REACHED (wp {self.wp_idx})')
            return cmd

        if self.front_dist < DANGER_DIST:
            self.wall_side = 1 if self.left_dist >= self.right_dist else -1
            if self.bug_mode == "BUG2":
                self.hit_x            = self.x
                self.hit_y            = self.y
                self.hit_dist_to_goal = dist
            self.state = "WALL_FOLLOW"
            self.get_logger().info(
                f'→ WALL_FOLLOW ({"izq" if self.wall_side==1 else "der"})')
            return cmd

        if abs(error_theta) > ANGLE_PRIORITY:
            self.int_error_d = 0.0
            cmd.angular.z = clamp(KP_W * error_theta - KV_W * self.w_robot,
                                   -OMEGA_MAX, OMEGA_MAX)
            return cmd

        self.int_error_d = clamp(self.int_error_d + dist * dt, -2.0, 2.0)
        wx, wy, _ = self.waypoints[self.wp_idx]
        dr  = math.hypot(wx - self.x_start, wy - self.y_start)
        u_v = clamp(KP_V * dr, 0.0, V_MAX)
        u_w = clamp(KP_W * error_theta - KV_W * self.w_robot, -OMEGA_MAX, OMEGA_MAX)
        cmd.linear.x  = u_v
        cmd.angular.z = u_w
        return cmd

    def _state_wall_follow(self, goal_x, goal_y, dist_to_goal):
        """
        Wall-follow con 3 sub-casos para manejar esquinas sin chocar el lateral.

        Sub-caso 1 — CORNER  (front < DANGER * 0.7)
          Giro in-situ en dirección OPUESTA a wall_side para doblar la esquina.
          linear.x = 0 para no avanzar contra el lateral.

        Sub-caso 2 — CLOSE FRONT  (front < DANGER * 1.2)
          Pequeño retroceso + giro suave. Transición antes de llegar a esquina.

        Sub-caso 3 — NORMAL
          Avance a V_WF con corrección proporcional al error lateral.
          wall_side=+1 (izq): lat_err>0 → gira izq (+), lat_err<0 → gira der (-)
          wall_side=-1 (der): signo invertido.
        """
        cmd = Twist()
        lateral = self.left_dist if self.wall_side == 1 else self.right_dist
        lat_err = lateral - WALL_DIST   # positivo = lejos de la pared

        # ── Sub-caso 1: esquina — giro in-situ ────────────────────────────
        if self.front_dist < DANGER_DIST * 0.7:
            cmd.linear.x  = 0.0
            cmd.angular.z = -self.wall_side * 0.15   # gira hacia el lado libre
            return cmd

        # ── Sub-caso 2: frente cercano — frenar y girar suave ─────────────
        if self.front_dist < DANGER_DIST * 1.2:
            cmd.linear.x  = -0.04
            cmd.angular.z = -self.wall_side * 0.20
            return cmd

        # ── Sub-caso 3: avance normal con corrección lateral ───────────────
        u_w = clamp(self.wall_side * KP_WF_LAT * lat_err, -OMEGA_MAX, OMEGA_MAX)
        cmd.linear.x  = V_WF
        cmd.angular.z = u_w

        # ── Condición de salida ────────────────────────────────────────────
        angle_to_goal = math.atan2(goal_y - self.y, goal_x - self.x)
        error_theta   = wrap_angle(angle_to_goal - self.theta)

        if self.bug_mode == "BUG0":
            if (self.front_dist > DANGER_DIST * 1.5
                    and lateral > WALL_DIST
                    and abs(error_theta) < math.radians(15)):
                self._exit_wall_follow()
        else:
            if self._on_mline(goal_x, goal_y) and self.hit_x is not None:
                d_hit = math.hypot(self.x - self.hit_x, self.y - self.hit_y)
                if (dist_to_goal < self.hit_dist_to_goal - 0.05
                        and self.front_dist > DANGER_DIST
                        and d_hit > 0.10):
                    self._exit_wall_follow()

        return cmd

    def _state_goal_reached(self, target_theta):
        cmd = Twist()
        err_th = wrap_angle(target_theta - self.theta)
        if abs(err_th) > ANGLE_PRIORITY:
            cmd.angular.z = clamp(KP_W * err_th - KV_W * self.w_robot,
                                   -OMEGA_MAX, OMEGA_MAX)
            return cmd

        self.wp_idx      += 1
        self.int_error_d  = 0.0
        self.x_start      = self.x
        self.y_start      = self.y

        if self.wp_idx >= len(self.waypoints):
            self.state = "DONE"
            self.get_logger().info('✓ Todos los waypoints alcanzados — DONE')
        else:
            self.state = "ROTATE_TO_GOAL"
            self._reset_bug2()
            self.get_logger().info(
                f'→ ROTATE_TO_GOAL (wp {self.wp_idx}: '
                f'x={self.waypoints[self.wp_idx][0]:.2f} '
                f'y={self.waypoints[self.wp_idx][1]:.2f})')
        return cmd

    # ── Helpers Bug 2 ─────────────────────────────────────────────────────

    def _reset_bug2(self):
        self.bug2_start_x     = self.x
        self.bug2_start_y     = self.y
        self.hit_x            = None
        self.hit_y            = None
        self.hit_dist_to_goal = math.inf

    def _on_mline(self, goal_x, goal_y, tol=0.06):
        sx, sy = self.bug2_start_x, self.bug2_start_y
        lx, ly = goal_x - sx, goal_y - sy
        length = math.hypot(lx, ly)
        if length < 1e-6:
            return False
        nx, ny = -ly / length, lx / length
        rx, ry = self.x - sx, self.y - sy
        perp   = abs(rx * nx + ry * ny)
        proj   = rx * (lx / length) + ry * (ly / length)
        return perp < tol and 0.0 <= proj <= length

    def _exit_wall_follow(self):
        self.int_error_d = 0.0
        self.x_start     = self.x
        self.y_start     = self.y
        self.state       = "ROTATE_TO_GOAL"
        self.get_logger().info('→ ROTATE_TO_GOAL (salida wall-follow)')


# ═══════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = WaypointsBugNode()
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