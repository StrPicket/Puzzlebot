#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import threading


class PuzzlebotAutoCapture(Node):
    def __init__(self):
        super().__init__('puzzlebot_auto_capture')

        self.total_photos = 20
        self.interval = 2.0
        self.save_dir = "calib_puzzlebot_logitec"
        self.image_topic = "/image_raw"

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()   # <-- protege acceso al frame
        self.image_count = 0
        self.last_capture_time = time.time()

        self.fps = 0.0
        self.fps_frame_count = 0
        self.fps_last_time = time.time()

        self._shutdown = False

        os.makedirs(self.save_dir, exist_ok=True)

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        # Timer solo para guardar, ya NO para imshow
        self.timer = self.create_timer(self.interval, self.save_callback)

        self.get_logger().info("=" * 40)
        self.get_logger().info("  Puzzlebot Auto Capture started")
        self.get_logger().info(f"  Target:   {self.total_photos} images")
        self.get_logger().info(f"  Interval: {self.interval}s | Dir: {self.save_dir}")
        self.get_logger().info("  [q] Quit   [s] Force capture")
        self.get_logger().info("=" * 40)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.flip(frame, -1)# -1 = flip horizontal y vertical (180°)
            self.fps_frame_count += 1
            now = time.time()
            if now - self.fps_last_time >= 1.0:
                self.fps = self.fps_frame_count / (now - self.fps_last_time)
                self.fps_frame_count = 0
                self.fps_last_time = now

            with self.frame_lock:
                self.latest_frame = frame

        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def save_frame(self, force=False):
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is None:
            return

        filename = os.path.join(self.save_dir, f"img_{self.image_count:04d}.jpg")
        if cv2.imwrite(filename, frame):
            self.image_count += 1
            tag = "[MANUAL]" if force else "[AUTO]  "
            self.get_logger().info(f"{tag} {self.image_count:>3}/{self.total_photos} → {filename}")
        else:
            self.get_logger().error(f"Failed to save: {filename}")

    def save_callback(self):
        """Timer callback: solo guarda, no toca GUI."""
        if self.latest_frame is None or self._shutdown:
            return

        if self.image_count >= self.total_photos:
            self.get_logger().info("✓ Capture completed!")
            self._shutdown = True
            return

        self.save_frame()

    def draw_overlay(self, frame):
        display = frame.copy()
        h, w = display.shape[:2]

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        time_to_next = max(0.0, self.interval - (time.time() - self.last_capture_time))

        cv2.putText(display, f"FPS: {self.fps:.1f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"Captured: {self.image_count}/{self.total_photos}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"next in {time_to_next:.1f}s",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.putText(display, "[q] quit  [s] save now",
                    (w - 230, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        bar_y = h - 8
        cv2.rectangle(display, (0, bar_y), (w, h), (50, 50, 50), -1)
        progress_w = int(w * self.image_count / self.total_photos)
        cv2.rectangle(display, (0, bar_y), (progress_w, h), (0, 220, 100), -1)

        return display

    def run_display_loop(self):
        """Loop de visualización en el hilo principal."""
        cv2.namedWindow("Puzzlebot Camera", cv2.WINDOW_AUTOSIZE)

        while rclpy.ok() and not self._shutdown:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is not None:
                display = self.draw_overlay(frame)
                cv2.imshow("Puzzlebot Camera", display)

            key = cv2.waitKey(30) & 0xFF  # ~30ms → ~33 FPS display

            if key == ord('q'):
                self.get_logger().info("Quit by user")
                self._shutdown = True
                break
            elif key == ord('s') and self.latest_frame is not None:
                self.save_frame(force=True)

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotAutoCapture()

    # ROS2 spin en hilo secundario
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # imshow en el hilo principal
        node.run_display_loop()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down via Ctrl+C")
    finally:
        node._shutdown = True
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()