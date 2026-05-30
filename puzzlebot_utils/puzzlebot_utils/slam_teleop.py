#!/usr/bin/env python3
"""
slam_teleop.py  --  corre en la LAPTOP
Publica: /cmd_vel
Sin odometria, sin TF, sin encoders.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import sys
import termios
import tty
import threading
import select


class SlamTeleop(Node):

    KEY_MAP = {
        'w': ('linear',  +1),
        's': ('linear',  -1),
        'a': ('angular', +1),
        'd': ('angular', -1),
    }
    STOP_KEYS = {' ', '\x1b'}

    def __init__(self):
        super().__init__('slam_teleop')

        # ------------------------------------------------------------------ #
        # Velocidades de teleop
        # ------------------------------------------------------------------ #
        self.v_lin = 0.03    # m/s
        self.v_ang = 0.03    # rad/s
        # ------------------------------------------------------------------ #

        self.active_linear  = 0
        self.active_angular = 0

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(1.0 / 20.0, self.cmd_vel_cb)  # 20 Hz

        self._running = True
        self._kb_thread = threading.Thread(target=self._keyboard_loop,
                                           daemon=True)
        self._kb_thread.start()

        self._print_help()
        self.get_logger().info('slam_teleop listo (Laptop)')

    def cmd_vel_cb(self):
        cmd = Twist()
        cmd.linear.x  = self.v_lin * self.active_linear
        cmd.angular.z = self.v_ang * self.active_angular
        self.cmd_vel_pub.publish(cmd)

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

                if key == '\x03':
                    self._stop_all()
                    self._running = False
                    break

                self._handle_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str):
        if key in self.STOP_KEYS:
            self._stop_all()
            return

        if key not in self.KEY_MAP:
            return

        axis, direction = self.KEY_MAP[key]

        if axis == 'linear':
            if self.active_linear == direction:
                self.active_linear = 0
                label = 'STOP lineal'
            else:
                self.active_linear  = direction
                self.active_angular = 0
                label = 'ADELANTE' if direction > 0 else 'ATRAS'
        else:
            if self.active_angular == direction:
                self.active_angular = 0
                label = 'STOP angular'
            else:
                self.active_angular = direction
                self.active_linear  = 0
                label = 'IZQ' if direction > 0 else 'DER'

        self.get_logger().info(
            f'[{key.upper()}] {label:<14} '
            f'v={self.active_linear  * self.v_lin:+.2f} m/s  '
            f'w={self.active_angular * self.v_ang:+.2f} rad/s')

    def _stop_all(self):
        self.active_linear  = 0
        self.active_angular = 0
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info('[STOP] robot detenido')

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


def main(args=None):
    rclpy.init(args=args)
    node = SlamTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()