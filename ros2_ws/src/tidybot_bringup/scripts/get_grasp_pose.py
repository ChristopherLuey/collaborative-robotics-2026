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
- /gripper_pose_cmd (Pose): Target gripper pose

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
from geometry_msgs.msg import TransformStamped, Pose
from tf2_ros import TransformListener, Buffer
import numpy as np

class GripperPoseNode(Node):
    """Arm control demo - move from home to forward position."""

    def __init__(self):
        super().__init__('gripper_pose_node')

        # Publish the goal pose of the gripper (position + orientation) for the arm to follow
        self.pose_pub = self.create_publisher(Pose, '/gripper_pose_cmd', 10)
        

        # Subscriber for joint states
        self.local_pc = None
        self.local_pc_sub = self.create_subscription(
            PointCloud2, '/object_points', self.local_points_callback, 10
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
        self.local_pc = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        self.get_logger().info(f'Received local points: {self.local_pc}')
    
    def get_centroid(self, points):
        """Calculate the centroid of a list of points."""
        if not points:
            return None
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        z = sum(p[2] for p in points) / len(points)
        return (x, y, z)
    
    def get_axis(self, points):
        """Calculate major and minor axes of the point cloud using PCA."""
        if not points:
            return None, None
        points_np = np.array(points)
        pca = PCA(n_components=3)
        pca.fit(points_np)
        
        # Principal axes
        major_axis = pca.components_[0]  # direction of maximum variance
        minor_axis = pca.components_[1]  # direction of second-largest variance
        
        return (major_axis, minor_axis)

    def calculate_gripper_pose(self, centroid:np.ndarray, axis:np.ndarray):
        """Calculate the desired gripper pose based on the centroid and minor axis."""
        pose = Pose()
        pose.position.x = centroid[0]
        pose.position.y = centroid[1]
        pose.position.z = centroid[2]

        # x align with -z world
        # y align with minor axis
        # z orthogonal to both
        x = np.array([0, 0, -1])
        y = axis / np.linalg.norm(axis)
        z = np.cross(x, y)
        z = z / np.linalg.norm(z)
        R = np.column_stack([x, y, z])
        r = R.from_matrix(R)

        quat = r.as_quat()
        
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        return pose

    def publish_pose(self):
        if self.local_pc == None:
            return
        centroid = self.get_centroid(self.local_pc)
        major_axis, minor_axis = self.get_axis(self.local_pc)
        
        pose_msg = self.calculate_gripper_pose(centroid, minor_axis)

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

    