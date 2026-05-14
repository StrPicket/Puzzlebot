#!/usr/bin/env python3
"""
Parametros ajustables al inicio de __init__:
  self.radio, self.lenght   -- geometria del robot  (igual que tu waypoints.py)
  self.v_lin, self.v_ang    -- velocidades de teleop
  laser_x/y/z               -- posicion del LiDAR respecto a base_link
"""

import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import math
import sys
import termios
import tty
import threading
import select


# ---------------------------------------------------------------------------
# Quaternion helper (sin dependencias extra)
# ---------------------------------------------------------------------------

def yaw_to_quat(yaw: float):
    """Convierte yaw (rad) -> (qx, qy, qz, qw)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return 0.0, 0.0, sy, cy


# ---------------------------------------------------------------------------
# Nodo principal
# ---------------------------------------------------------------------------

class SlamTeleop(Node):

    # Mapa tecla -> (eje, direccion)
    KEY_MAP = {
        'w': ('linear',  +1),
        's': ('linear',  -1),
        'a': ('angular', +1),
        'd': ('angular', -1),
    }
    STOP_KEYS = {' ', '\x1b'}   # espacio, ESC

    def __init__(self):
        super().__init__('slam_teleop')

        # ------------------------------------------------------------------ #
        # Parametros del robot  (ajusta segun tu hardware)
        # ------------------------------------------------------------------ #
        self.radio   = 0.0505   # m  radio de rueda
        self.lenght  = 0.183    # m  distancia entre ruedas (track)

        self.v_lin   = 0.03    # m/s   velocidad lineal de teleop
        self.v_ang   = 0.03  # rad/s velocidad angular de teleop

        # Offset del LiDAR respecto al centro de base_link
        # Mide desde el centro del robot hasta donde esta montado el lidar
        laser_x = 0.04
        laser_y = 0.0
        laser_z = 0.10          # altura del lidar sobre el suelo aprox
        # ------------------------------------------------------------------ #

        # Estado odometria
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0
        self.wr    = Float32()
        self.wl    = Float32()

        # Estado teleop (toggle)
        self.active_linear  = 0   # -1, 0 o +1
        self.active_angular = 0

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist,    'cmd_vel', 10)
        self.odom_pub    = self.create_publisher(Odometry, 'odom',    10)

        # TF dinamico: odom -> base_link
        self.tf_br = TransformBroadcaster(self)

        # TF estatico: base_link -> laser  (se publica una sola vez y persiste)
        static_br = StaticTransformBroadcaster(self)
        st = TransformStamped()
        st.header.stamp    = self.get_clock().now().to_msg()
        st.header.frame_id = 'base_link'
        st.child_frame_id  = 'laser'
        st.transform.translation.x = laser_x
        st.transform.translation.y = laser_y
        st.transform.translation.z = laser_z
        st.transform.rotation.x = 0.0
        st.transform.rotation.y = 0.0
        st.transform.rotation.z = 0.0
        st.transform.rotation.w = 1.0
        static_br.sendTransform(st)
        self.get_logger().info(
            f'TF estatico publicado: base_link -> laser '
            f'[{laser_x}, {laser_y}, {laser_z}]')

        # Subscriptions a encoders
        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_cb,
            qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_cb,
            qos.qos_profile_sensor_data)

        # Timers
        self.last_odom_time = self.get_clock().now()
        self.create_timer(1.0 / 100.0, self.odometria_cb)  # 100 Hz
        self.create_timer(1.0 /  20.0, self.cmd_vel_cb)    #  20 Hz

        # Hilo de teclado (no bloquea el executor de ROS)
        self._running = True
        self._kb_thread = threading.Thread(target=self._keyboard_loop,
                                           daemon=True)
        self._kb_thread.start()

        self._print_help()
        self.get_logger().info('slamTools listo')

    # ---------------------------------------------------------------------- #
    # Callbacks encoders
    # ---------------------------------------------------------------------- #

    def encR_cb(self, msg: Float32):
        self.wr = msg

    def encL_cb(self, msg: Float32):
        self.wl = msg

    # ---------------------------------------------------------------------- #
    # Odometria + TF dinamico + /odom
    # ---------------------------------------------------------------------- #

    def odometria_cb(self):
        now = self.get_clock().now()
        dt  = (now - self.last_odom_time).nanoseconds * 1e-9
        self.last_odom_time = now
        if dt <= 0.0:
            return

        v_r = self.radio * self.wr.data
        v_l = self.radio * self.wl.data
        v   = (v_r + v_l) / 2.0
        w   = (v_r - v_l) / self.lenght

        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.theta  = (self.theta + math.pi) % (2.0 * math.pi) - math.pi

        qx, qy, qz, qw = yaw_to_quat(self.theta)
        stamp = now.to_msg()

        # -- TF: odom -> base_link --
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id  = 'base_link'
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_br.sendTransform(tf)

        # -- /odom --
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x    = self.x
        odom.pose.pose.position.y    = self.y
        odom.pose.pose.position.z    = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x    = v
        odom.twist.twist.angular.z   = w
        self.odom_pub.publish(odom)

    # ---------------------------------------------------------------------- #
    # Publicacion periodica de cmd_vel segun estado toggle
    # ---------------------------------------------------------------------- #

    def cmd_vel_cb(self):
        cmd = Twist()
        cmd.linear.x  = self.v_lin * self.active_linear
        cmd.angular.z = self.v_ang * self.active_angular
        self.cmd_vel_pub.publish(cmd)

    # ---------------------------------------------------------------------- #
    # Hilo de teclado  (raw mode, no-blocking read)
    # ---------------------------------------------------------------------- #

    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running and rclpy.ok():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                key = sys.stdin.read(1).lower()

                # Ctrl-C
                if key == '\x03':
                    self._stop_all()
                    self._running = False
                    break

                self._handle_key(key)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str):
        # Teclas de parada
        if key in self.STOP_KEYS:
            self._stop_all()
            return

        # Tecla no mapeada -> ignorar
        if key not in self.KEY_MAP:
            return

        axis, direction = self.KEY_MAP[key]

        if axis == 'linear':
            if self.active_linear == direction:
                # Segunda pulsacion de la misma tecla -> para
                self.active_linear = 0
                label = 'STOP lineal'
            else:
                self.active_linear  = direction
                self.active_angular = 0        # cancela giro al ir recto
                label = 'ADELANTE' if direction > 0 else 'ATRAS'

        else:  # angular
            if self.active_angular == direction:
                self.active_angular = 0
                label = 'STOP angular'
            else:
                self.active_angular = direction
                self.active_linear  = 0        # cancela avance al girar
                label = 'IZQ' if direction > 0 else 'DER'

        self.get_logger().info(
            f'[{key.upper()}] {label:<14} '
            f'v={self.active_linear  * self.v_lin:+.2f} m/s  '
            f'w={self.active_angular * self.v_ang:+.2f} rad/s')

    def _stop_all(self):
        self.active_linear  = 0
        self.active_angular = 0
        stop = Twist()
        self.cmd_vel_pub.publish(stop)
        self.get_logger().info('[STOP] robot detenido')

    # ---------------------------------------------------------------------- #
    # Ayuda en consola
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _print_help():
        print("""
+-----------------------------------------------+
|           SLAM TELEOP  --  controles           |
|                                                |
|   W  ->  toggle avanzar                        |
|   S  ->  toggle retroceder                     |
|   A  ->  toggle girar izquierda                |
|   D  ->  toggle girar derecha                  |
|  SPC ->  detener todo                          |
|  ESC / Ctrl-C  ->  salir                       |
|                                                |
|  Presionar la misma tecla DOS veces = STOP     |
|  Presionar tecla diferente = cambio directo    |
+-----------------------------------------------+
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SlamTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        # Asegura que el robot quede detenido al salir
        stop = Twist()
        node.cmd_vel_pub.publish(stop)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()