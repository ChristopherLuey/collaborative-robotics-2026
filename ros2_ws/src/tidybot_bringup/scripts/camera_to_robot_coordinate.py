""""
ROS2 node to subscribe to the RGB-D camera and 
"""

import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation as R
from message_filters import Subscriber, ApproximateTimeSynchronizer

BANANA_LOW = (13,170,70)
BANANA_HIGH = (34,255,193)

class PointCloudMask(Node):
    def __init__(self):
        super().__init__('pointcloud_mask_node')

        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # TODO: test if the aligned_depth_to_color topic is published on our realsense!
        #       if so, we don't have to do physical alignment
        self.sub_info = self.create_subscription(CameraInfo, '/camera/realsense/aligned_depth_to_color/camera_info', self.info_cb, 10)
        self.sub_rgb  = Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = Subscriber(self, Image, '/camera/realsense/aligned_depth_to_color/image_raw')
        
        self.pub_cloud = self.create_publisher(PointCloud2, '/banana_points', 10)

        ## we need to listen for transforms between the camera and the arms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # a time synchronizer ensures that our rgb and depth frames are time aligned
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth],
            queue_size=10,
            slop=0.05
        )

        self.ts.registerCallback(self.publish_isolated_pc)

    def info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
    
    def publish_isolated_pc(self, rgb_msg, depth_msg):
        """
        method to take synced rgb and depth messages, and apply an HSV mask with some other
        operations to extract just the local pointcloud representation of the object view.
        """

        if rgb_msg is None or depth_msg is None:
            self.get_logger().info("Any of depth and rgb image not received.")
            return
        if None in (self.fx, self.fy, self.cx, self.cy):
            self.get_logger().info("camera info not found")
            return   
        
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        banana_mask = get_clean_mask(rgb, BANANA_LOW, BANANA_HIGH, 5)   
        if banana_mask is None:
            self.get_logger().info("Could not find any contours, no blobs in frame.")  
            return
        
        # extract banana pixels
        ys, xs = np.where(banana_mask > 0)

        if xs.size == 0:
            self.get_logger().info("No banana points found.")
            return
        
        # get corresponding z points and filter them
        z_axis = depth[ys, xs]
        valid_z_axis = (z_axis > 100) & (z_axis < 5000)
        xs, ys, z_axis = xs[valid_z_axis], ys[valid_z_axis], z_axis[valid_z_axis]

        # log the number of identified points
        mask_pts = len(banana_mask[banana_mask > 0])
        depth_valid_pts = np.sum(valid_z_axis)
        
        self.get_logger().info(f"found {mask_pts} mask points")
        self.get_logger().info(f"   of these, found {depth_valid_pts} points within depth limits")

        if depth_valid_pts <= 30:
            self.get_logger().info(f"Could not find enough valid points, no object detected.")
            return

        if (self.cx is None) or (self.cy is None) or (self.fx is None) or (self.fy is None):
            self.get_logger().info("Camera intrinsics not available yet.")
            return

        # Back-project to camera frame
        try:
            X = ((xs.astype(np.float32) - self.cx) * z_axis) / self.fx
            Y = ((ys.astype(np.float32) - self.cy) * z_axis) / self.fy
            points = np.stack([X, Y, z_axis], axis=1)
            points = points.astype(np.float32) * 0.001       # convert to meters

        except Exception as e:
            self.get_logger().error(f"Error in back-projection: {e}")
            return
        
        self.get_logger().info(f"Local Camera PC distributions (m)")
        self.get_logger().info(f"X: {np.mean(points[:, 0])} +/- {np.std(points[:, 0])}")
        self.get_logger().info(f"Y: {np.mean(points[:, 1])} +/- {np.std(points[:, 1])}")
        self.get_logger().info(f"Z: {np.mean(points[:, 2])} +/- {np.std(points[:, 2])}")    

        header = rgb_msg.header  # use color stamp/frame; ensure depth is aligned to color!
        self.get_logger().info("local pointcloud is in frame:", header.frame_id)
        cloud = point_cloud2.create_cloud_xyz32(header, points)
        self.pub_cloud.publish(cloud)

def main():
    rclpy.init()
    node = PointCloudMask()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def get_clean_mask(bgr_im, hsv_low, hsv_high, kernel_sz):
    """
    Function used to isolate the largest object falling within a given mask found in an image.
    see comments for process, but uses a combination of classic cv methods.
    """
    hsv = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2HSV)

    # create mask w tuned mask params
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    # perform erosion to eat away at edges of the mask (remove stray pixels) and then
    # dilate to bring back anything still connected to a blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # find contours to isolate the remaining blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # pick out the largest contour, hopefully this is the banana
    largest_contour = max(contours, key=cv2.contourArea)
    
    # draw a filled version of this contour, which should be our mask
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED) 
    return final_mask


if __name__ == '__main__':
    main()
