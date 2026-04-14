import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber (Node):
    def __init__(self):
        super().__init__('listener_node')

        self.subscription = self.create_subscription(
            String, 'chatter', self.listener_callback, 10) 

    def listener_callback(self,msg):
        self.get_logger().info('I heard "%s"' % msg.data)           




def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        minimal_subscriber.destroy_node()