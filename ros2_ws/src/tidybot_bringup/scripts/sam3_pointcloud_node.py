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
import cv2
from PIL import Image as PILImage
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

        self.server_setup = False
        # ------------------------------------------------------------------ #
        # Parameters                                                           #
        # ------------------------------------------------------------------ #
        self.declare_parameter('min_depth_mm',    100)
        self.declare_parameter('max_depth_mm',    5000)
        self.declare_parameter('min_valid_pts',   30)
        self.declare_parameter('base_frame',      'odom')

        self.min_z      = self.get_parameter('min_depth_mm').value
        self.max_z      = self.get_parameter('max_depth_mm').value
        self.min_pts    = self.get_parameter('min_valid_pts').value
        self.base_frame = self.get_parameter('base_frame').value

        # ------------------------------------------------------------------ #
        # Initialize the connection to the sam3 server                         #
        # ------------------------------------------------------------------ #
        
        ##### THIS WILL BE ON SERVER ##### 

        # ------------------------------------------------------------------ #
        # Camera state                                                         #
        # ------------------------------------------------------------------ #
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_frame = None
        self.bridge = CvBridge()
        self.latest_rgb   = None
        self.latest_depth = None

        # ------------------------------------------------------------------ #
        # TF                                                                   #
        # ------------------------------------------------------------------ #
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------ #
        # Subscriptions                                                        #
        # ------------------------------------------------------------------ #
        self.create_subscription(
            CameraInfo,
            '/camera/realsense/aligned_depth_to_color/camera_info',
            self._info_cb,
            10,
        )

        sub_rgb   = Subscriber(self, Image, '/camera/color/image_raw')
        sub_depth = Subscriber(self, Image,
                               '/camera/realsense/aligned_depth_to_color/image_raw')
        ts = ApproximateTimeSynchronizer([sub_rgb, sub_depth], queue_size=5, slop=0.05)
        ts.registerCallback(self._image_cb)

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
        print(f"Connecting to {address}...")

        self.sock.connect(address)

    # ---------------------------------------------------------------------- #
    # Callbacks                                                                #
    # ---------------------------------------------------------------------- #

    def _info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_frame = msg.header.frame_id

    def _image_cb(self, rgb_msg: Image, depth_msg: Image):
        self.get_logger().info("Received synced RGB and depth frames.")
        self.latest_rgb   = rgb_msg
        self.latest_depth = depth_msg

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

        # pil_img = PILImage.fromarray(bgr[:, :, ::-1].copy().astype(np.uint8))
        rgb_image = bgr[:, :, ::-1].copy().astype(np.uint8)  # numpy RGB array


        # # SAM3 segmentation + back-projection + TF transform -------------------
        # points_base, err = self._get_points_base(rgb_image, depth, prompt_text)
        # if points_base is None:
        #     response.success = False
        #     response.message = err
        #     return response

        # Compute grasp pose given the prompt and point cloud ------------------
        stamp = self.get_clock().now().to_msg()
        if prompt_text == "door":
            drawer_points, drawer_err = self._get_points_base(rgb_image, depth, "drawer")
            knob_points, knob_err = self._get_points_base(rgb_image, depth, "door knob")
            if drawer_points is None or knob_points is None:
                response.success = False
                response.message = f"Failed to get door or drawer points: {drawer_err}, {knob_err}"
                return response
            
            response.pose = _compute_door_pose(drawer_points, knob_points, self.base_frame, 0.0, stamp)
        else:
            points_base, err = self._get_points_base(rgb_image, depth, prompt_text)
            if points_base is None:
                response.success = False
                response.message = err
                return response
        
            response.pose = _compute_object_pose(points_base, self.base_frame, 0.0, stamp)

        response.success = True
        response.message = f"Detected '{prompt_text}' with {len(points_base)} points."
        self.get_logger().info(response.message)
        return response

    def _get_points_base(self, rgb_image: np.ndarray, depth: np.ndarray,
                         prompt_text: str):
        """
        Send image + prompt to SAM3 server, back-project the returned mask to
        camera-frame 3D points, then transform to the base frame.

        Returns (points_base, None) on success, or (None, error_str) on failure.
        """
        # Send request to SAM3 server ------------------------------------------
        self.sock.send_json(
            {"shape": list(rgb_image.shape), "dtype": str(rgb_image.dtype), "prompt": prompt_text},
            zmq.SNDMORE,
        )
        self.sock.send(rgb_image.tobytes())

        meta   = self.sock.recv_json()
        data   = self.sock.recv()
        mask_np = np.frombuffer(data, dtype=meta["dtype"]).reshape(meta["shape"])
        self.get_logger().info(f"Mask shape: {mask_np.shape}, objects: {meta.get('num_objects', '?')}")

        # Back-project to camera-frame points ----------------------------------
        points_cam = self._mask_to_points(mask_np, depth)
        if points_cam is None:
            return None, (
                f"Too few valid depth points for '{prompt_text}' "
                f"(need >= {self.min_pts})."
            )

        # Transform to base frame ----------------------------------------------
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                self.latest_rgb.header.stamp,
            )
            points_base = _transform_pointcloud(points_cam, tf_msg)
        except Exception as e:
            return None, f"TF lookup failed: {e}"

        return points_base, None

    # ---------------------------------------------------------------------- #
    # Depth back-projection                                                    #
    # ---------------------------------------------------------------------- #

    def _mask_to_points(self, mask: np.ndarray, depth: np.ndarray):
        """Back-projects mask pixels to camera-frame 3D points (metres)."""
        ys, xs = np.where(mask)
        if xs.size == 0:
            return None

        z_vals = depth[ys, xs].astype(np.float32)
        valid  = (z_vals > self.min_z) & (z_vals < self.max_z)
        xs, ys, z_vals = xs[valid], ys[valid], z_vals[valid]

        if int(valid.sum()) < self.min_pts:
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


def _compute_door_pose(drawer_points: np.ndarray, knob_points: np.ndarray,
                  frame_id: str, offset: float, stamp) -> PoseStamped:

    # Position: centered on the knob
    # Orientation: aligned along the drawer pull direction (PCA of drawer points)

    centroid = knob_points.mean(axis=0)

    pca = PCA(n_components=3)
    pca.fit(drawer_points)
    axis = pca.components_[2]  # normal to the drawer face = pull direction
    # is this primary axis normal to the plan formed by the drawer points?

    pose = PoseStamped()
    pose.header.stamp    = stamp
    pose.header.frame_id = frame_id

    pose.pose.position.x = float(centroid[0])
    pose.pose.position.y = float(centroid[1])
    pose.pose.position.z = float(centroid[2]) + offset

    # # Gripper y-axis along drawer pull direction, z-axis up
    # z = np.array([0.0, 0.0, 1.0])
    # y = axis - np.dot(axis, z) * z
    # norm = np.linalg.norm(y)
    # y = y / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    # x = np.cross(y, z)
    # x /= np.linalg.norm(x)

    # Gripper x-axis into the drawer (along normal), z-axis up
    z = np.array([0.0, 0.0, 1.0])
    x = axis.copy()
    x[2] = 0.0  # flatten to XY plane
    norm = np.linalg.norm(x)
    x = x / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    rot_mat = np.column_stack((x, y, z))
    quat    = R.from_matrix(rot_mat).as_quat()

    pose.pose.orientation.x = float(quat[0])
    pose.pose.orientation.y = float(quat[1])
    pose.pose.orientation.z = float(quat[2])
    pose.pose.orientation.w = float(quat[3])

    return pose

def _compute_object_pose(points_base: np.ndarray,
                  frame_id: str, offset: float, stamp) -> PoseStamped:
    
    centroid    = points_base.mean(axis=0)
    pca         = PCA(n_components=3)
    pca.fit(points_base)
    axis  = pca.components_[0]

    pose = PoseStamped()
    pose.header.stamp    = stamp
    pose.header.frame_id = frame_id

    pose.pose.position.x = float(centroid[0])
    pose.pose.position.y = float(centroid[1])
    pose.pose.position.z = float(centroid[2]) + offset

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
