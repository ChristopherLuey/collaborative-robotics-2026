#!/usr/bin/env python3
"""
SAM3 Object Pose Service Node

Exposes a ROS2 service /sam3/get_object_pose (tidybot_msgs/GetObjectPose):
  - Request:  string prompt  (e.g. "apple")
  - Response: bool success, geometry_msgs/PoseStamped pose, string message

On each service call the node:
  1. Appends "." to the prompt  -> "apple."
  2. Runs SAM3 on the latest synced (RGB, depth) frame
  3. Back-projects the segmentation mask to camera-frame 3D points
  4. Transforms them to the base frame via TF
  5. Computes centroid (position) + PCA major axis (orientation)
  6. Returns the resulting PoseStamped

Subscriptions
-------------
  /camera/color/image_raw
  /camera/realsense/aligned_depth_to_color/image_raw
  /camera/realsense/aligned_depth_to_color/camera_info

Parameters
----------
  sam3_confidence  float   SAM3 detection confidence threshold.  Default: 0.5
  min_depth_mm     int     Minimum valid depth in mm.            Default: 100
  max_depth_mm     int     Maximum valid depth in mm.            Default: 5000
  min_valid_pts    int     Min depth-valid points before pose.   Default: 30
  base_frame       str     TF frame for the returned pose.       Default: "odom"
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import TransformListener, Buffer

from tidybot_msgs.srv import GetObjectPose
import zmq

REMOTE_IP = "100.77.113.90"
PORT = 5555


class SAM3ObjectPoseNode(Node):
    """
    ROS2 service node: /sam3/get_object_pose.

    Keeps the latest synced camera frame in memory and, on each service call,
    runs SAM3 segmentation and returns the object pose in the base frame.
    """

    def __init__(self):
        super().__init__('sam3_object_pose_node')

        # ------------------------------------------------------------------ #
        # Parameters                                                           #
        # ------------------------------------------------------------------ #
        self.declare_parameter('min_depth_mm',    100)
        self.declare_parameter('max_depth_mm',    5000)
        self.declare_parameter('min_valid_pts',   30)
        self.declare_parameter('base_frame',      'base_link')

        self.min_z      = self.get_parameter('min_depth_mm').value
        self.max_z      = self.get_parameter('max_depth_mm').value
        self.min_pts    = self.get_parameter('min_valid_pts').value
        self.base_frame = self.get_parameter('base_frame').value

        # ------------------------------------------------------------------ #
        # Camera state                                                         #
        # ------------------------------------------------------------------ #
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_frame = None
        self.bridge = CvBridge()
        self.latest_rgb   = None
        self.latest_depth = None
        self._using_hardware = False  # True once real hardware topics arrive

        # ------------------------------------------------------------------ #
        # TF                                                                   #
        # ------------------------------------------------------------------ #
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------ #
        # Subscriptions (prefer real hardware topics, fall back to sim)         #
        # ------------------------------------------------------------------ #
        # Camera info: prefer real hardware, fall back to sim
        self.create_subscription(
            CameraInfo,
            '/camera/realsense/aligned_depth_to_color/camera_info',
            self._info_cb_hardware,
            10,
        )
        self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._info_cb_sim,
            10,
        )

        # Synced RGB+depth: real hardware topics (preferred)
        sub_rgb_real   = Subscriber(self, Image, '/camera/color/image_raw')
        sub_depth_real = Subscriber(self, Image,
                               '/camera/realsense/aligned_depth_to_color/image_raw')
        self._ts_real = ApproximateTimeSynchronizer(
            [sub_rgb_real, sub_depth_real], queue_size=5, slop=0.05)
        self._ts_real.registerCallback(self._image_cb_hardware)

        # Synced RGB+depth: sim topics (fallback)
        sub_rgb_sim   = Subscriber(self, Image, '/camera/color/image_raw')
        sub_depth_sim = Subscriber(self, Image, '/camera/depth/image_raw')
        self._ts_sim = ApproximateTimeSynchronizer(
            [sub_rgb_sim, sub_depth_sim], queue_size=5, slop=0.1)
        self._ts_sim.registerCallback(self._image_cb_sim)

        # ------------------------------------------------------------------ #
        # Service                                                              #
        # ------------------------------------------------------------------ #
        self.srv = self.create_service(
            GetObjectPose,
            '/sam3/get_object_pose',
            self._handle_request,
        )

        self.get_logger().info(
            "SAM3ObjectPoseNode ready.  Call /sam3/get_object_pose to segment."
        )

        self.setup_zmq_connection()

    def setup_zmq_connection(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)

        address = f"tcp://{REMOTE_IP}:{PORT}"
        self.get_logger().info(f"Connecting to {address}...")

        self.sock.connect(address)

    # ---------------------------------------------------------------------- #
    # Callbacks                                                                #
    # ---------------------------------------------------------------------- #

    def _info_cb_hardware(self, msg: CameraInfo):
        if not self._using_hardware:
            self.get_logger().info("Using real hardware camera topics.")
        self._using_hardware = True
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_frame = msg.header.frame_id

    def _info_cb_sim(self, msg: CameraInfo):
        # Only use sim intrinsics if hardware hasn't been seen
        if self._using_hardware:
            return
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_frame = msg.header.frame_id

    def _image_cb_hardware(self, rgb_msg: Image, depth_msg: Image):
        if not self._using_hardware:
            self.get_logger().info("Using real hardware camera topics.")
        self._using_hardware = True
        had_frames = self.latest_rgb is not None
        self.latest_rgb   = rgb_msg
        self.latest_depth = depth_msg
        if not had_frames:
            self.get_logger().info("Received first synced RGB+depth frames (hardware).")

    def _image_cb_sim(self, rgb_msg: Image, depth_msg: Image):
        # Only use sim frames if hardware hasn't been seen
        if self._using_hardware:
            return
        had_frames = self.latest_rgb is not None
        self.latest_rgb   = rgb_msg
        self.latest_depth = depth_msg
        if not had_frames:
            self.get_logger().info("Received first synced RGB+depth frames (sim).")

    # ---------------------------------------------------------------------- #
    # Service handler                                                          #
    # ---------------------------------------------------------------------- #

    def _handle_request(self, request: GetObjectPose.Request,
                        response: GetObjectPose.Response):
        
        prompt_text = request.prompt.strip()
        self.get_logger().info(
            f"Service call: '{request.prompt}' -> prompt='{prompt_text}'"
        )

        if None in (self.fx, self.fy, self.cx, self.cy):
            response.success = False
            response.message = "Camera intrinsics not yet available."
            return response

        if self.latest_rgb is None or self.latest_depth is None:
            response.success = False
            response.message = "No camera frames received yet."
            return response

        # Decode ----------------------------------------------------------------
        bgr   = self.bridge.imgmsg_to_cv2(self.latest_rgb,   desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')

        rgb_image = bgr[:, :, ::-1].copy().astype(np.uint8)

        # SAM3 segmentation via remote server ----------------------------------
        self.sock.send_json({"shape": rgb_image.shape, "dtype": str(rgb_image.dtype), "prompt": prompt_text}, zmq.SNDMORE)
        self.sock.send(rgb_image.tobytes())

        meta = self.sock.recv_json()
        data = self.sock.recv()
        mask_np = np.frombuffer(data, dtype=meta["dtype"]).reshape(meta["shape"])
        self.get_logger().info(f"SAM3 mask shape: {mask_np.shape}")

        # Back-project to camera-frame points ----------------------------------
        points_cam = self._mask_to_points(mask_np, depth)
        if points_cam is None:
            response.success = False
            response.message = (
                f"Too few valid depth points for '{prompt_text}' "
                f"(need >= {self.min_pts})."
            )
            return response

        # Transform to base frame ----------------------------------------------
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                self.latest_rgb.header.stamp,
            )
            points_base = _transform_pointcloud(points_cam, tf_msg)
        except Exception as e:
            response.success = False
            response.message = f"TF lookup failed: {e}"
            return response

        # Centroid + PCA orientation -------------------------------------------
        centroid    = points_base.mean(axis=0)
        pca         = PCA(n_components=3)
        pca.fit(points_base)
        major_axis  = pca.components_[0]

        stamp = self.get_clock().now().to_msg()
        response.success = True
        response.pose    = _compute_pose(centroid, major_axis, self.base_frame, stamp)
        response.message = (
            f"Detected '{prompt_text}' with {len(points_base)} points."
        )
        self.get_logger().info(response.message)
        return response

    # ---------------------------------------------------------------------- #
    # Depth back-projection                                                    #
    # ---------------------------------------------------------------------- #

    def _mask_to_points(self, mask: np.ndarray, depth: np.ndarray):
        """Back-projects mask pixels to camera-frame 3D points (metres)."""
        ys, xs = np.where(mask)
        if xs.size == 0:
            self.get_logger().warn("Mask has zero pixels.")
            return None

        z_vals = depth[ys, xs].astype(np.float32)
        valid  = (z_vals > self.min_z) & (z_vals < self.max_z)

        n_mask = xs.size
        n_valid = int(valid.sum())
        n_zero = int((z_vals == 0).sum())
        if n_mask > 0:
            z_min, z_max = float(z_vals.min()), float(z_vals.max())
            self.get_logger().info(
                f"Depth stats: mask_px={n_mask}, valid={n_valid}, zero={n_zero}, "
                f"range=[{z_min:.0f}, {z_max:.0f}]mm, filter=[{self.min_z}, {self.max_z}]mm"
            )

        xs, ys, z_vals = xs[valid], ys[valid], z_vals[valid]

        if n_valid < self.min_pts:
            return None

        X = ((xs - self.cx) * z_vals) / self.fx
        Y = ((ys - self.cy) * z_vals) / self.fy
        return np.stack([X, Y, z_vals], axis=1).astype(np.float32) * 0.001


# --------------------------------------------------------------------------- #
# Module-level helpers (mirror get_grasp_pose.py)                             #
# --------------------------------------------------------------------------- #

def _transform_pointcloud(points: np.ndarray, tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    q = tf.transform.rotation
    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T        = np.eye(4)
    T[:3, :3] = rot
    T[:3,  3] = [t.x, t.y, t.z]
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (T @ points_h.T).T[:, :3]


def _compute_pose(centroid: np.ndarray, axis: np.ndarray,
                  frame_id: str, stamp) -> PoseStamped:
    pose = PoseStamped()
    pose.header.stamp    = stamp
    pose.header.frame_id = frame_id

    pose.pose.position.x = float(centroid[0])
    pose.pose.position.y = float(centroid[1])
    pose.pose.position.z = float(centroid[2])

    # Align gripper: x-axis along world -Z, y-axis along minor axis
    z = np.array([0.0, 0.0, 1.0])
    y = axis - np.dot(axis, z) * z
    norm = np.linalg.norm(y)
    y = y / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    rot_mat = np.column_stack((x, y, z))
    quat    = R.from_matrix(rot_mat).as_quat()

    pose.pose.orientation.x = float(quat[0])
    pose.pose.orientation.y = float(quat[1])
    pose.pose.orientation.z = float(quat[2])
    pose.pose.orientation.w = float(quat[3])

    return pose


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = SAM3ObjectPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
