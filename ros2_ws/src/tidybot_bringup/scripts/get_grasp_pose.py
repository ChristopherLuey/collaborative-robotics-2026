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

        # Subscriber for local point cloud
        self.local_pc = None
        self.local_pc_sub = self.create_subscription(
            PointCloud2, '/banana_points', self.local_points_callback, 10
        )

        # Transformation setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_timer(0.05, self.publish_pose)

        # Names of the transforms you want to listen to
        self.camera_frame = None
        self.base_frame = 'odom'  # Set base frame as 'odom' (change if needed)

    def local_points_callback(self, msg):
        """Callback for local points topic."""
        self.camera_frame = msg.header.frame_id  # Store the frame of the incoming point cloud
        self.local_pc = msg
        #self.local_pc = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        #self.get_logger().info(f'Received {len(self.local_pc)} points in {self.camera_frame} frame')

    def transform_pointcloud(self, pointcloud, source_frame, target_frame):
        """Transform point cloud from source_frame to target_frame."""
        try:
            points = list(pc2.read_points(pointcloud, field_names=("x", "y", "z"), skip_nans=True))

            # Get the transformation from the source frame to the target frame
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())

            # Extract translation and rotation from the transform
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # Convert quaternion to rotation matrix
            rotation_matrix = quaternion_to_rot_matrix(rotation.x, rotation.y, rotation.z, rotation.w)

            # Create a 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [translation.x, translation.y, translation.z]

            # Now apply this transformation to each point in the point cloud
            transformed_points = []
            for point in points:
                # Convert each point to homogeneous coordinates (x, y, z, 1)
                homogeneous_point = np.array([point[0], point[1], point[2], 1.0])
                
                # Apply the transformation matrix (4x4 matrix * 4x1 point)
                transformed_point = transform_matrix @ homogeneous_point
                
                # Append the transformed point to the list
                transformed_points.append(transformed_point[:3])  # We only care about x, y, z

            return transformed_points

        except Exception as e:
            self.get_logger().error(f"Failed to transform point cloud: {e}")
            return []
        
    def transform_point(self, point: np.ndarray, header, target_frame: str) -> np.ndarray:
        """
        Transform a single 3D point from header.frame_id to target_frame.

        Args:
            point: np.array([x, y, z])
            header: std_msgs.msg.Header of the original point
            target_frame: frame to transform the point into

        Returns:
            np.array([x, y, z]) in target_frame, or None on failure
        """
        try:
            # Lookup transform from source frame (header.frame_id) to target_frame at the timestamp
            transform = self.tf_buffer.lookup_transform(
                target_frame, header.frame_id, rclpy.time.Time()
            )

            # Extract translation and rotation
            t = transform.transform.translation
            q = transform.transform.rotation

            # Convert quaternion to rotation matrix
            R_mat = quaternion_to_rot_matrix(q.x, q.y, q.z, q.w)

            # Construct homogeneous 4x4 transform
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = [t.x, t.y, t.z]

            # Transform the point
            point_h = np.hstack([point, 1.0])  # Make it homogeneous
            transformed_point = (T @ point_h)[:3]

            return transformed_point

        except Exception as e:
            self.get_logger().error(f"Failed to transform point: {e}")
            return None
        
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
        # Convert list of tuples into a simple 2D array (shape: Nx3)
        points_np = np.array([list(p) for p in points])  # Convert to plain float64 array
        pca = PCA(n_components=3)
        pca.fit(points_np)

        # Principal axes
        major_axis = pca.components_[0]  # direction of maximum variance
        minor_axis = pca.components_[1]  # direction of second-largest variance

        return (major_axis, minor_axis)

    def calculate_gripper_pose(self, centroid: np.ndarray, axis: np.ndarray):
        """Calculate the desired gripper pose based on the centroid and minor axis."""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()  # Add timestamp
        pose.header.frame_id = self.base_frame  # Set to base frame (odom)

        pose.pose.position.x = centroid[0]
        pose.pose.position.y = centroid[1]
        pose.pose.position.z = centroid[2]

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

        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        return pose

    def publish_pose(self):
        if self.local_pc is None:
            return
        # in the direction camera is facing, we want max depth to try and get center of the object
        # in the xy camera directions, we want centroid.
        # challenging because we only see a half-shell of the object, pure centroid returns values
        # too close to the camera.

        # Apply transformation to local point cloud

        transformed_pc = self.transform_pointcloud(self.local_pc, self.camera_frame, self.base_frame)

        local_pc = list(pc2.read_points(self.local_pc, field_names=("x", "y", "z"), skip_nans=True))
        local_centroid = list(self.get_centroid(local_pc))
        top_z = sorted(p[2] for p in local_pc)[-20:]          # Take largest 20 z-values
        local_centroid[2] = float(np.median(top_z))  
        transformed_centroid = self.transform_point(local_centroid, self.local_pc.header, self.base_frame)
        
        if transformed_pc and transformed_centroid is not None:

            major_axis, _ = self.get_axis(transformed_pc)

            pose_msg = self.calculate_gripper_pose(transformed_centroid, major_axis)
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

    