import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import math


class SignalProcessor(Node):
    def __init__(self):
        super().__init__('process')

        # Subscribers
        self.signal_sub = self.create_subscription(Float32,'/signal',
            self.signal_cb,10)

        self.time_sub = self.create_subscription(Float32,'/time',
            self.time_cb,10)

        # Publisher
        self.proc_pub = self.create_publisher(Float32, '/proc_signal', 10)

        # Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Variables
        self.signal = None
        self.time = None

        # Phase shift (hardcoded)
        self.phi = math.pi / 4

    def signal_cb(self, msg):
        self.signal = msg.data

    def time_cb(self, msg):
        self.time = msg.data

    def timer_cb(self):
        # Asegurarse de que ya hay datos
        if self.signal is None or self.time is None:
            return

        # Procesamiento
        processed = 0.5 * (math.sin(self.time + self.phi) + 1)

        # Mensaje
        msg = Float32()
        msg.data = float(processed)

        # Publicar
        self.proc_pub.publish(msg)

        # Log
        self.get_logger().info(
            f't={self.time:.2f}, original={self.signal:.3f}, processed={processed:.3f}'
        )


def main(args=None):
    rclpy.init(args=args)

    processed_publisher = SignalProcessor()

    try:
        rclpy.spin(processed_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        processed_publisher.destroy_node()
        rclpy.shutdown()