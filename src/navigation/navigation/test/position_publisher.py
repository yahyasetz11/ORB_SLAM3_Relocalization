#!/usr/bin/env python3
# TEST ONLY — publishes a fake current_position shuttling between two points.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

# Positions in user coordinate system: origin bottom-left, X upward, Y rightward
# user_x = map_height - 1 - pixel_row,  user_y = pixel_col
START = (25,  15)   # was pixel (col=15,  row=85)  on a 111-tall map
END   = (21, 132)   # was pixel (col=132, row=89)
HZ    = 10.0       # publish rate
STEPS = 60         # ticks to travel one way  (6 s per leg)


class PositionPublisher(Node):

    def __init__(self):
        super().__init__('position_publisher')
        self._pub    = self.create_publisher(PointStamped, 'current_position', 10)
        self._tick   = 0
        self._fwd    = True
        self.create_timer(1.0 / HZ, self._publish)
        self.get_logger().info(
            f'PositionPublisher ready  |  {START} ↔ {END}  steps={STEPS}'
        )

    def _publish(self):
        t = self._tick / STEPS
        x = START[0] + (END[0] - START[0]) * t
        y = START[1] + (END[1] - START[1]) * t

        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x         = x
        msg.point.y         = y
        msg.point.z         = 0.0
        self._pub.publish(msg)

        self._tick += 1 if self._fwd else -1
        if self._tick >= STEPS:
            self._fwd = False
        elif self._tick <= 0:
            self._fwd = True


def main(args=None):
    rclpy.init(args=args)
    node = PositionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
