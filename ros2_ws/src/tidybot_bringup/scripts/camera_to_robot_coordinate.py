""""
ROS2 node to subscribe to the RGB-D camera and 
"""

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge

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

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return
        
        z_axis = depth[ys, xs]
        valid_z_axis = np.isfinte(z_axis) & (z_axis > 0.1) & (z_axis < 5.0)
        
        xs, ys, z_axis = xs[valid_z_axis], ys[valid_z_axis], z_axis[valid_z_axis]
        # Back-project to camera frame
        X = (xs.astype(np.float32) - self.cx) * valid_z_axis / self.fx
        Y = (ys.astype(np.float32) - self.cy) * valid_z_axis / self.fy

        points = np.stack([X, Y, valid_z_axis], axis=1)

        self.get_logger.info(f"the number of 3 pointsa are {X.shape}")

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
