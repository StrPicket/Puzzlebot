import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

import numpy as np
import math

class waypoints(Node):
    def __init__(self):
        super().__init__('waypoints')

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_callback, qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_callback, qos.qos_profile_sensor_data)

        self.timer_odom = self.create_timer(1 / 100, self.odometria)
        self.timer_ctrl = self.create_timer(1 / 50,  self.control)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.wr = Float32()
        self.wl = Float32()
        self.w_robot = 0.0
        self.v_robot = 0.0

        self.radio   = 0.0505
        self.lenght  = 0.183

        self.Kp_d = 0.15
        self.Ki_d = 0.4

        self.int_error_d = 0.0

        self.Kp_t = 0.1
        self.Kv_t = 0.05

        self.last_time_odom = self.get_clock().now()
        self.last_time_control = self.get_clock().now()

        self.x_d = [1.0, 1.0, 0.0, 0.0]
        self.y_d = [1.0, 0.0, 0.0, 1.0]

        self.i = 0

        self.get_logger().info('waypoints node iniciado')

    def encR_callback(self, msg: Float32):
        self.wr = msg

    def encL_callback(self, msg: Float32):
        self.wl = msg

    def odometria(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_odom).nanoseconds * 1e-9
        self.last_time_odom = current_time

        if dt <= 0:
            return

        v_r    = self.radio * self.wr.data
        v_l    = self.radio * self.wl.data
        V_avg  = (v_r + v_l) / 2.0
        W_robot = (v_r - v_l) / self.lenght

        self.v_robot = 0.15 * self.v_robot + 0.85 * V_avg
        self.w_robot = 0.15 * self.w_robot + 0.85 * W_robot

        self.x     += V_avg * math.cos(self.theta) * dt
        self.y     += V_avg * math.sin(self.theta) * dt
        self.theta += W_robot * dt
        self.theta  = (self.theta + math.pi) % (2 * math.pi) - math.pi

    def control(self):
        cmd = Twist()

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_control).nanoseconds * 1e-9
        self.last_time_control = current_time
        dt = min(dt, 0.1)
    
        error_d = math.sqrt((self.x_d[self.i] - self.x) ** 2 + (self.y_d[self.i] - self.y) ** 2)

        error_theta = math.atan2(self.y_d[self.i] - self.y, self.x_d[self.i] - self.x) - self.theta
        error_theta = (error_theta + math.pi) % (2 * math.pi) - math.pi

        self.int_error_d += error_d * dt
        self.int_error_d = max(min(self.int_error_d, 1.0), -1.0)

        if error_d < 0.05:
            self.i += 1
            self.int_error_d = 0.0
            if self.i >= len(self.x_d):
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
                self.get_logger().info('Todos los waypoints alcanzados — robot detenido')
                return
            else:
                self.get_logger().info(f'Waypoint {self.i+1} alcanzado, avanzando al siguiente')

        u_d = self.Ki_d * self.int_error_d - self.Kp_d * error_d
        u_d = max(min(u_d, 0.7), -0.7)

        u_theta = self.Kp_t * error_theta - self.Kv_t * self.w_robot
        u_theta = max(min(u_theta, 0.2), -0.2)

        if abs(error_theta) > 1.0:
            cmd.linear.x = 0.0
            cmd.angular.z = -u_theta
        else:
            cmd.linear.x = u_d
            cmd.angular.z = -u_theta

        self.cmd_vel_pub.publish(cmd)

        theta_deg = math.degrees(self.theta) % 360
        self.get_logger().info(
            f'Error_d: {error_d:+.3f} | x_robot: {self.x:.3f} | y_robot: {self.y:.3f} | u_d: {u_d:+.3f}')
        self.get_logger().info(
            f'Error_theta: {error_theta:+.3f} | θ: {theta_deg:.1f}°  | u_theta: {u_theta:+.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = waypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    