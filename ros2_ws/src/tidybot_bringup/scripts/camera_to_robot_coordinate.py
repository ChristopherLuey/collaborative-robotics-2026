""""
ROS2 node to subscribe to the RGB-D camera and 
"""

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import time
import cv2


class YellowPointCloude(Node):
    def __init__(self):
        super().__init__('yp')

        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.rgb = None
        self.depth = None

        self.sub_info = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_cb, 10)
        self.sub_rgb  = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_cb, 10)
        self.sub_depth = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_cb, 10)

        self.pub_cloud = self.create_publisher(PointCloud2, '/banana_points', 10)

    def info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
    
    def rgb_cb(self, msg: Image):
        self.rgb = msg
        self.calculate()
    
    def depth_cb(self, msg: Image):
        self.depth = msg
        self.calculate()
    
    def calculate(self):
        if self.rgb is None or self.depth is None:
            self.get_logger().info("Any of depth and rgb image not received.")
            return
        if None in (self.fx, self.fy, self.cx, self.cy):
            self.get_logger().info("camera info not found")
            return   
        
        rgb = self.bridge.imgmsg_to_cv2(self.rgb, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(self.depth, desired_encoding='passthrough')

        # Depth units: common RealSense is 16UC1 in mm
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        # Yellow segmentation in HSV (tune these)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 100, 100], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.medianBlur(mask, 5)
        cv2.imwrite("bannana.jpg", mask)
        ts = time.time()
        # stop_for = 1
        # self.get_logger().info(f"Processing and publishing point cloud for {stop_for} seconds...")
        # while time.time() - ts < stop_for:
        #     rclpy.spin_once(self, timeout_sec=0.05)

        # self.get_logger().info(f"we did it")
        ys, xs = np.where(mask > 0)
        ys = ys.reshape(-1)
        self.get_logger().info(f"uv: ({xs.shape}, {ys.shape})")
        if xs.size == 0:
            return
        
        z_axis = depth[ys, xs]
        self.get_logger().info(f"depth: ({z_axis[0:20]})")
        valid_z_axis = (z_axis > 100) & (z_axis < 5000)
        # valid_z_axis = np.ones_like(xs)
        self.get_logger().info(f"found {len(mask[mask > 0])} points")
        self.get_logger().info(f"found {np.sum(valid_z_axis)} valid points")
        # if (valid_z_axis) == 0:
        #     self.get_logger().info("No valid depth points found.")
        #     return

        self.get_logger().info(f"cx: {self.cx}, cy: {self.cy}, fx: {self.fx}, fy: {self.fy}")
        self.get_logger().info(f"before: ({xs.shape}, {ys.shape}, {z_axis.shape})")
        xs, ys, z_axis = xs[valid_z_axis], ys[valid_z_axis], z_axis[valid_z_axis]
        # Back-project to camera frame
        self.get_logger().info(f"after: ({xs.shape}, {ys.shape}, {z_axis.shape})")
        try:
            X = (xs.astype(np.float32) - self.cx * np.ones_like(xs)) * z_axis / self.fx
            Y = (ys.astype(np.float32) - self.cy * np.ones_like(ys)) * z_axis / self.fy
            points = np.stack([X, Y, z_axis], axis=1)
        except Exception as e:
            self.get_logger().error(f"Error in back-projection: {e}")
            return

        
        self.get_logger().info(f"the number of 3 points are {X.shape}")

        header = self.rgb.header  # use color stamp/frame; ensure depth is aligned to color!
        cloud = point_cloud2.create_cloud_xyz32(header, points.tolist())
        self.pub_cloud.publish(cloud)

def main():
    rclpy.init()
    node = YellowPointCloude()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
