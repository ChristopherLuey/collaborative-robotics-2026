#!/usr/bin/env python3
"""
TidyBot2 Gripper Pose Generation Node

Generates a target gripper Pose from a point cloud of local object points. 
The pose is computed by:

1. Computing the centroid of the point cloud for the gripper position.
2. Performing PCA on the points to find the minor axis for orientation.
3. Aligning the gripper:
   - X-axis along world -Z
   - Y-axis along the minor axis
   - Z-axis orthogonal to both

The resulting Pose is published to `/gripper_pose_cmd` for the arm to follow.

Subscriptions:
- /object_points (PointCloud2): Input object point cloud

Publications:
- /gripper_pose_cmd (PoseStamped): Target gripper pose

Usage:
    # Terminal 1: Start simulation
    ros2 launch tidybot_bringup sim.launch.py

    # Terminal 2: Run this node
    ros2 run tidybot_bringup get_grasp_pose.py
"""

import rclpy
from rclpy.node import Node
import time
from sklearn.decomposition import PCA
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.spatial.transform import Rotation as R

class GripperPoseNode(Node):
    """Arm control demo - move from home to forward position."""

    def __init__(self):
        super().__init__('gripper_pose_node')

        # Publish the goal pose of the gripper (position + orientation) for the arm to follow
        self.pose_pub = self.create_publisher(PoseStamped, '/gripper_pose_cmd', 10)
        

        # Subscriber for joint states
        self.local_pc = None
        self.local_pc_sub = self.create_subscription(
            PointCloud2, '/banana_points', self.local_points_callback, 10
        )

        # Transformation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_timer(0.05, self.publish_pose)
        
        # Names of the transforms you want to listen to
        self.camera_frame = 'pan_link'
        self.base_frame = 'odom'

    def local_points_callback(self, msg):
        """Callback for local points topic."""
        points = pc2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        )

        # Convert structured output to regular Nx3 float32 array
        self.local_pc = np.array(
            [[p[0], p[1], p[2]] for p in points],
            dtype=np.float32
        )

        # Extract the frame of the point cloud
        self.camera_frame = msg.header.frame_id

        self.get_logger().info(
            f"Received {self.local_pc.shape[0]} points in frame '{self.camera_frame}'"
        )

    def get_centroid(self, points):
        """Calculate the centroid of a list of points."""
        if points is None or len(points) == 0:
            return None
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        z = sum(p[2] for p in points) / len(points)
        return (x, y, z)
    
    def get_axis(self, points):
        """Calculate major and minor axes of the point cloud using PCA."""
        if points is None or len(points) == 0:
            return None, None
        points_np = np.array(points, dtype=np.float32)
        pca = PCA(n_components=3)
        pca.fit(points_np)
        
        # Principal axes
        major_axis = pca.components_[0]  # direction of maximum variance
        minor_axis = pca.components_[1]  # direction of second-largest variance
        
        return (major_axis, minor_axis)

    def calculate_gripper_pose(self, centroid: np.ndarray, axis: np.ndarray):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.base_frame

        # Position
        pose.pose.position.x = float(centroid[0])
        pose.pose.position.y = float(centroid[1])
        pose.pose.position.z = float(centroid[2])

        # Orientation
        z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        y = axis - np.dot(axis, z)*z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        x /= np.linalg.norm(x)

        rot_mat = np.column_stack((x, y, z))
        r = R.from_matrix(rot_mat)
        quat = r.as_quat()

        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])

        return pose

    def publish_pose(self):
        if self.local_pc is None or len(self.local_pc) == 0:
            return

        # Transform the entire point cloud from camera frame to base frame
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time()
            )
            points_base = transform_pointcloud(self.local_pc, tf_msg)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # Compute centroid and PCA in base frame
        centroid = self.get_centroid(points_base)
        if centroid is None:
            return

        major_axis, minor_axis = self.get_axis(points_base)
        if major_axis is None:
            return

        # Compute PoseStamped in base frame
        pose_msg = self.calculate_gripper_pose(np.array(centroid, dtype=np.float32),
                                            np.array(major_axis, dtype=np.float32))

        # Publish
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GripperPoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def transform_pointcloud(points:np.ndarray, tf: TransformStamped):
    # Extract translation and rotation from TransformStamped
    t = tf.transform.translation
    q = tf.transform.rotation
    R = quaternion_to_rot_matrix(q.x, q.y, q.z, q.w)
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = [t.x, t.y, t.z]

    # Apply transform
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # Nx4
    points_transformed = (T @ points_h.T).T[:, :3]  # Nx3
    
    return points_transformed

def quaternion_to_rot_matrix(qx, qy, qz, qw):
    """
    Convert quaternion to 3x3 rotation matrix.
    """
    # Normalizing quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R
    
if __name__ == '__main__':
    main()

    