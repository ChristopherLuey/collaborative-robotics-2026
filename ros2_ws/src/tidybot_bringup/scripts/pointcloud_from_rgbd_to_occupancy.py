#!/usr/bin/env python3
"""
pointcloud_from_rgbd_to_occupancy.py

Build a pointcloud from synchronized RGB + aligned depth (+ CameraInfo) and
accumulate it into a 2D occupancy grid using odometry (with MuJoCo yaw correction).

This version applies the camera->base_link TF (if available) before using odom to
place points into the world/map frame.

Save to: ros2_ws/src/tidybot_bringup/scripts/pointcloud_from_rgbd_to_occupancy.py
chmod +x and add to tidybot_bringup CMakeLists install(PROGRAMS ...)

Parameters (ROS params):
 - map_width_m (float)        : width of map in meters (x axis)
 - map_height_m (float)       : height of map in meters (y axis)
 - resolution (float)         : meters per cell (e.g. 0.05)
 - min_z (float)              : min z (meters) to consider (e.g. 0.05)
 - max_z (float)              : max z (meters) to consider (e.g. 1.5)
 - min_depth (float)          : min depth to accept (meters) (optional override)
 - max_depth (float)          : max depth to accept (meters)
 - hit_threshold (int)        : how many hits to mark a cell occupied
 - decay_rate (float)         : per-second decay (0 = none)
 - rgb_topic (str)            : rgb image topic
 - depth_topic (str)          : aligned depth topic
 - camera_info_topic (str)    : camera info topic
 - odom_topic (str)           : odometry topic
 - map_frame (str)            : frame id for OccupancyGrid (default 'map')
 - stride (int)               : sample stride in pixels for backprojection (default 4)
 - publish_pointcloud (bool)  : whether to publish aggregated pointcloud of occupied cells
"""
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import Pose

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import tf2_ros
from scipy.spatial.transform import Rotation as SciR


def quat_to_yaw(qx, qy, qz, qw):
    t = 2.0 * (qw * qz + qx * qy)
    u = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(t, u)


class PointCloudFromRgbdToOccupancy(Node):
    def __init__(self):
        super().__init__('pointcloud_from_rgbd_to_occupancy')

        # params
        self.declare_parameter('map_width_m', 10.0)
        self.declare_parameter('map_height_m', 10.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('min_z', 0.05)
        self.declare_parameter('max_z', 1.5)
        self.declare_parameter('min_depth', 0.0)   # 0 means auto-detect from image type
        self.declare_parameter('max_depth', 5.0)
        self.declare_parameter('hit_threshold', 3)
        self.declare_parameter('decay_rate', 0.0)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('map_frame', 'odom')
        self.declare_parameter('stride', 4)
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('camera_to_base_timeout_s', 0.5)

        # load params
        self.map_w_m = float(self.get_parameter('map_width_m').value)
        self.map_h_m = float(self.get_parameter('map_height_m').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)
        self.min_depth_param = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.hit_threshold = int(self.get_parameter('hit_threshold').value)
        self.decay_rate = float(self.get_parameter('decay_rate').value)
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.stride = int(self.get_parameter('stride').value)
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.camera_to_base_timeout = float(self.get_parameter('camera_to_base_timeout_s').value)

        # derived
        self.width_cells = int(math.ceil(self.map_w_m / self.resolution))
        self.height_cells = int(math.ceil(self.map_h_m / self.resolution))
        self.occ_counts = np.zeros((self.height_cells, self.width_cells), dtype=np.int32)
        self.last_update_time = time.time()

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # odom (store x,y,yaw,z)
        self.latest_odom = None  # (x, y, yaw, z)
        self.origin_set = False
        self.origin_x = 0.0
        self.origin_y = 0.0

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # bridge
        self.bridge = CvBridge()

        # publishers
        self.pub_map = self.create_publisher(OccupancyGrid, '/local_occupancy_grid', 10)
        if self.publish_pointcloud:
            self.pub_accum_pc = self.create_publisher(PointCloud2, '/accumulated_points', 10)

        # camera_info subscription (normal rclpy sub)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_cb, 10)

        # message_filters subscribers for rgb + depth
        self.sub_rgb = Subscriber(self, Image, self.rgb_topic)
        self.sub_depth = Subscriber(self, Image, self.depth_topic)

        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.rgbd_callback)

        # odom
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)

        self.get_logger().info(f"Started: rgb={self.rgb_topic} depth={self.depth_topic} info={self.camera_info_topic}")

    def info_cb(self, msg: CameraInfo):
        # extract intrinsics
        try:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])
        except Exception as e:
            self.get_logger().warning(f"Failed to parse camera_info: {e}")

    def odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        # store x,y,yaw and z (z included for completeness)
        self.latest_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw, msg.pose.pose.position.z)

        if not self.origin_set:
            self.origin_x = self.latest_odom[0] - (self.map_w_m / 2.0)
            self.origin_y = self.latest_odom[1] - (self.map_h_m / 2.0)
            self.origin_set = True
            self.get_logger().info(f"Map origin set: origin_x={self.origin_x:.3f}, origin_y={self.origin_y:.3f}")

    def world_to_cell(self, x, y):
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return row, col

    def lookup_camera_to_base(self, camera_frame, stamp):
        """
        Query TF for transform: base_link <- camera_frame.
        Returns (trans_vec (3,), rotation (scipy Rotation)) or None on failure.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', camera_frame, rclpy.time.Time(), timeout=Duration(seconds=self.camera_to_base_timeout)
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            trans = np.array([t.x, t.y, t.z], dtype=np.float32)
            rot = SciR.from_quat([r.x, r.y, r.z, r.w])
            return trans, rot
        except Exception as e:
            # transform not available
            self.get_logger().debug(f"TF lookup failed for camera frame '{camera_frame}': {e}")
            return None

    def rgbd_callback(self, rgb_msg: Image, depth_msg: Image):
        # requires camera_info + odom
        if self.latest_odom is None:
            self.get_logger().debug("No odom yet; skipping frame.")
            return
        if None in (self.fx, self.fy, self.cx, self.cy):
            self.get_logger().debug("No camera intrinsics yet; skipping frame.")
            return

        # convert depth to numpy
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")
            return

        depth_np = np.array(depth)
        if depth_np.size == 0:
            return

        # sample by stride for speed
        h, w = depth_np.shape[:2]
        ys = np.arange(0, h, self.stride)
        xs = np.arange(0, w, self.stride)
        grid_x, grid_y = np.meshgrid(xs, ys)
        xs_flat = grid_x.ravel()
        ys_flat = grid_y.ravel()

        z_vals = depth_np[ys_flat, xs_flat]

        # If depth is integer and large -> assume mm and convert
        if np.issubdtype(z_vals.dtype, np.integer) or (z_vals.max() > 1000.0):
            z_m = z_vals.astype(np.float32) * 0.001
        else:
            z_m = z_vals.astype(np.float32)

        # optional param override for min depth (if param is >0 use that)
        min_depth = self.min_depth_param if self.min_depth_param > 0.0 else 0.0

        valid_mask = (z_m > min_depth) & (z_m <= self.max_depth)
        if not np.any(valid_mask):
            self.get_logger().debug("No valid depth samples in frame.")
            return

        xs_valid = xs_flat[valid_mask].astype(np.float32)
        ys_valid = ys_flat[valid_mask].astype(np.float32)
        z_valid = z_m[valid_mask].astype(np.float32)

        # backproject -> camera coordinates (meters)
        X = ((xs_valid - self.cx) * z_valid) / self.fx
        Y = ((ys_valid - self.cy) * z_valid) / self.fy
        Z = z_valid
        points_cam = np.stack([X, Y, Z], axis=1)  # shape (N,3), meters

        # Lookup camera->base_link TF using the depth image header frame_id
        camera_frame = depth_msg.header.frame_id if depth_msg.header.frame_id else rgb_msg.header.frame_id
        cam_to_base = self.lookup_camera_to_base(camera_frame, depth_msg.header.stamp)
        if cam_to_base is None:
            # if TF is not available, skip this frame (avoids using identity incorrectly)
            self.get_logger().debug(f"Skipping frame: missing camera->base TF for frame '{camera_frame}'")
            return

        trans_cam_base, rot_cam_base = cam_to_base

        # Transform points from camera frame -> base_link frame
        # rot_cam_base converts points expressed in camera_frame into base_link frame when applied
        try:
            points_base = rot_cam_base.apply(points_cam) + trans_cam_base  # (N,3)
        except Exception as e:
            self.get_logger().error(f"Error applying camera->base rotation: {e}")
            return

        # Transform points from base_link -> world/map using odom (rx,ry,ryaw,rz)
        rx, ry, ryaw, r_z = self.latest_odom
        c = math.cos(ryaw); s = math.sin(ryaw)
        R2 = np.array([[c, -s], [s, c]], dtype=np.float32)

        xy_base = points_base[:, :2]
        world_xy = (xy_base.dot(R2.T) + np.array([rx, ry], dtype=np.float32))

        # z in world = points_base[:,2] + odom z (if nonzero)
        z_base = points_base[:, 2]
        z_world = z_base + r_z

        # filter by Z (height) using min_z/max_z (use z_world)
        z_mask = (z_world >= self.min_z) & (z_world <= self.max_z)
        if not np.any(z_mask):
            self.get_logger().debug("No points within z height limits (after transforms).")
            return

        world_xy = world_xy[z_mask]
        #points_cam = points_cam[z_mask]  # not needed further
        #z_world = z_world[z_mask]

        # update occupancy counters
        rows_cols = [self.world_to_cell(float(x), float(y)) for (x, y) in world_xy]
        for (r, c) in rows_cols:
            if 0 <= r < self.height_cells and 0 <= c < self.width_cells:
                self.occ_counts[r, c] += 1

        # decay occupancy
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        if self.decay_rate > 0 and dt > 0:
            decay_amount = int(self.decay_rate * dt)
            if decay_amount > 0:
                self.occ_counts = np.maximum(self.occ_counts - decay_amount, 0)

        # publish occupancy grid and aggregated points
        self.publish_map(rgb_msg.header.stamp)
        if self.publish_pointcloud:
            self.publish_aggregated_points(rgb_msg.header)

    def publish_map(self, stamp):
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = stamp
        grid.header.frame_id = self.map_frame

        info = MapMetaData()
        info.resolution = float(self.resolution)
        info.width = int(self.width_cells)
        info.height = int(self.height_cells)
        info.origin = Pose()
        info.origin.position.x = float(self.origin_x)
        info.origin.position.y = float(self.origin_y)
        info.origin.position.z = 0.0
        info.origin.orientation.x = 0.0
        info.origin.orientation.y = 0.0
        info.origin.orientation.z = 0.0
        info.origin.orientation.w = 1.0
        grid.info = info

        flat = np.full((self.height_cells * self.width_cells,), -1, dtype=np.int8)
        occ_mask = self.occ_counts >= self.hit_threshold
        flat[occ_mask.flatten()] = 100
        grid.data = flat.tolist()
        self.pub_map.publish(grid)

    def publish_aggregated_points(self, header):
        occupied = np.argwhere(self.occ_counts >= self.hit_threshold)
        pts = []
        for r, c in occupied:
            x = self.origin_x + (c + 0.5) * self.resolution
            y = self.origin_y + (r + 0.5) * self.resolution
            z = 0.5 * (self.min_z + self.max_z)
            pts.append((x, y, z))
        if not pts:
            return
        header_msg = header
        header_msg.frame_id = self.map_frame
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc2 = point_cloud2.create_cloud(header_msg, fields, pts)
        self.pub_accum_pc.publish(pc2)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFromRgbdToOccupancy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()