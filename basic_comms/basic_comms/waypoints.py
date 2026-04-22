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

        self.Kp_v = 0.15
        self.Ki_v = 0.25

        self.int_error_d = 0.0

        self.Kp_w = 0.1
        self.Kv_w = 0.05

        self.last_time_odom = self.get_clock().now()
        self.last_time_control = self.get_clock().now()

        self.x_d = [1.0, 1.0, 0.0, 0.0]
        self.y_d = [0.0, 1.0, 1.0, 0.0]
        self.t_d = [0.0, 90.0, 180.0, 270.0]

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

        self.v_robot = V_avg
        self.w_robot = W_robot

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

        error_theta = math.radians(self.t_d[self.i]) - self.theta
        error_theta = (error_theta + math.pi) % (2 * math.pi) - math.pi

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


        if abs(error_theta) > math.radians(5):
            u_v = 0.0
            self.int_error_d = 0.0
        else:
            self.int_error_d += error_d * dt
            self.int_error_d = max(min(self.int_error_d, 1.0), -1.0)

            u_v = self.Ki_v * self.int_error_d - self.Kp_v * self.x
            u_v = max(min(u_v, 0.5), -0.5)

        u_w = self.Kp_w * error_theta - self.Kv_w * self.w_robot
        u_w = max(min(u_w, 0.2), -0.2)
            
        cmd.linear.x = u_v
        cmd.angular.z = u_w

        self.cmd_vel_pub.publish(cmd)

        theta_deg = math.degrees(self.theta) % 360
        self.get_logger().info(
            f'Error_d: {error_d:+.3f} | x_robot: {self.x:.3f} | y_robot: {self.y:.3f} | u_v: {u_v:+.3f}')
        self.get_logger().info(
            f'Error_theta: {error_theta:+.3f} | θ: {theta_deg:.1f}°  | u_w: {u_w:+.3f}')

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