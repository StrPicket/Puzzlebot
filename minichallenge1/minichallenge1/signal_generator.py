import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float32


class SignalGenerator(Node):
    def __init__(self):
        super().__init__('signal_generator')

        # Publishers
        self.signal_publish= self.create_publisher(Float32, '/signal', 10)
        self.time_publish = self.create_publisher(Float32, '/time', 10)

        # Timer
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_cb)
        self.t = 0.0
        self.dt = 0.1

    def timer_cb(self):
        signal = math.sin(self.t)

        signal_msg = Float32()
        signal_msg.data = signal

        time_msg = Float32()
        time_msg.data = self.t

        self.signal_publish.publish(signal_msg)
        self.time_publish.publish(time_msg)

        self.get_logger().info(f't={self.t:.2f}, signal={signal:.3f}')
        self.t += self.dt

def main(args=None):
    rclpy.init(args=args)

    signal_publisher = SignalGenerator()

    try:
        rclpy.spin(signal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        signal_publisher.destroy_node()