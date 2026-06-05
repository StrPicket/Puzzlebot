#!/usr/bin/env python3
"""
slam_odom.py  --  corre en la JETSON
Publica: /odom (nav_msgs/Odometry) + TF dinamico odom -> base_link

Cambios vs version anterior:
  1. Timer reducido a 50Hz (de 100Hz): micro_ROS tipicamente envia encoders
     a 20-50Hz; un timer mas rapido que la fuente solo quema CPU publicando
     odometria con datos identicos (dt real != 0 pero v/w estale).
  2. Deteccion de datos stale: si los encoders no se han actualizado en
     mas de 100ms, se asume velocidad 0 (robot detenido) para no integrar
     drift con datos viejos.
  3. QoS de suscripcion documentado: sensor_data es correcto para
     micro_ROS (best-effort, volatile).
  4. Sin cambios en la logica de odometria (no era el problema).
"""

import math
import time
import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return 0.0, 0.0, sy, cy


class SlamOdom(Node):
    def __init__(self):
        super().__init__('slam_odom')

        self.radio  = 0.0505
        self.lenght = 0.183

        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        self._wr_data: float = 0.0
        self._wl_data: float = 0.0

        # Timestamp del ultimo encoder recibido (para deteccion de stale)
        self._last_enc_time: float = time.monotonic()
        self._STALE_TIMEOUT: float = 0.1   # 100ms sin encoder -> velocidad 0

        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_br    = TransformBroadcaster(self)

        self.get_logger().info(
            'slam_odom: TFs estaticos desactivados (los publica la laptop)')

        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self._encR_cb,
            qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self._encL_cb,
            qos.qos_profile_sensor_data)

        self._last_odom_time = self.get_clock().now()

        # 50Hz es suficiente para encoders de microcontrolador a 20-50Hz.
        # Bajar de 100Hz libera ~0.5ms/ciclo de CPU para la camara.
        self.create_timer(1.0 / 50.0, self._odometria_cb)

        self.get_logger().info('slam_odom listo @ 50Hz (Jetson)')

    # ------------------------------------------------------------------
    # Callbacks de encoder
    # ------------------------------------------------------------------
    def _encR_cb(self, msg: Float32) -> None:
        self._wr_data = msg.data
        self._last_enc_time = time.monotonic()

    def _encL_cb(self, msg: Float32) -> None:
        self._wl_data = msg.data
        self._last_enc_time = time.monotonic()

    # ------------------------------------------------------------------
    # Timer de odometria
    # ------------------------------------------------------------------
    def _odometria_cb(self) -> None:
        now = self.get_clock().now()
        dt  = (now - self._last_odom_time).nanoseconds * 1e-9
        self._last_odom_time = now

        if dt <= 0.0 or dt > 0.5:
            return

        # Si los encoders llevan mas de STALE_TIMEOUT sin actualizarse,
        # el robot esta detenido (o micro_ROS perdio conexion).
        if time.monotonic() - self._last_enc_time > self._STALE_TIMEOUT:
            wr = 0.0
            wl = 0.0
        else:
            wr = self._wr_data
            wl = self._wl_data

        v_r = self.radio * wr
        v_l = self.radio * wl
        v   = (v_r + v_l) / 2.0
        w   = (v_r - v_l) / self.lenght

        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.theta  = (self.theta + math.pi) % (2.0 * math.pi) - math.pi

        qx, qy, qz, qw = yaw_to_quat(self.theta)
        stamp = now.to_msg()

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
        odom.pose.covariance[0]  = 0.01
        odom.pose.covariance[7]  = 0.01
        odom.pose.covariance[35] = 0.05
        odom.twist.covariance[0]  = 0.01
        odom.twist.covariance[35] = 0.05
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = SlamOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
