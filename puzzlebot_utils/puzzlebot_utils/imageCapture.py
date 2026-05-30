#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os


class CameraCalibrationCapture(Node):
    def __init__(self):
        super().__init__('camera_calibration_capture')

        # Configuración
        self.chessboard_size = (7, 5)
        self.square_size = 0.025  # metros (opcional pero recomendado)
        self.total_photos = 30
        self.save_dir = "calibration_dataset"
        self.image_topic = "/camera/image_raw"

        self.bridge = CvBridge()
        self.latest_frame = None
        self.image_count = 0

        os.makedirs(self.save_dir, exist_ok=True)

        # Preparar puntos 3D
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        self.objpoints = []
        self.imgpoints = []

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info("=== Camera Calibration Mode ===")
        self.get_logger().info("Press ENTER to capture ONLY if chessboard is detected")
        self.get_logger().info("Press 'q' to quit")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Conversion error: {e}")

    def run(self):
        while rclpy.ok():

            if self.latest_frame is None:
                rclpy.spin_once(self)
                continue

            frame = self.latest_frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar chessboard
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK
            )

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners2, ret)

                status_text = "Chessboard DETECTED"
                color = (0, 255, 0)
            else:
                status_text = "Chessboard NOT detected"
                color = (0, 0, 255)

            # Overlay info
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"Images: {self.image_count}/{self.total_photos}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF

            # Salir
            if key == ord('q'):
                break

            # ENTER para guardar
            if key == 13 or key == 10:
                if not ret:
                    self.get_logger().warn("Chessboard NOT detected - image skipped")
                    continue

                if self.image_count >= self.total_photos:
                    self.get_logger().info("Dataset complete")
                    break

                filename = os.path.join(self.save_dir, f"img_{self.image_count:04d}.jpg")

                cv2.imwrite(filename, self.latest_frame)

                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)

                self.image_count += 1

                self.get_logger().info(f"Saved {self.image_count}/{self.total_photos}")

            rclpy.spin_once(self)

        self.calibrate()
        self.shutdown()

    def calibrate(self):
        if len(self.objpoints) < 10:
            self.get_logger().warn("Not enough images for calibration")
            return

        h, w = self.latest_frame.shape[:2]

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            (w, h),
            None,
            None
        )

        # Error de reproyección
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error /= len(self.objpoints)

        self.get_logger().info(f"Calibration error: {mean_error:.4f}")

        print("\n=== RESULTADOS ===")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)

        np.savez("calibration_result.npz",
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs)

        self.get_logger().info("Calibration saved to calibration_result.npz")

    def shutdown(self):
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationCapture()

    try:
        node.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()