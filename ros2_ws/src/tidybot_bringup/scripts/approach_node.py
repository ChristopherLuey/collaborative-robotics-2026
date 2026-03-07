#!/usr/bin/env python3
"""
approach_node.py (service-only)

- Creates service /approach_pose of type tidybot_msgs/srv/ApproachPose
  Request: geometry_msgs/Pose2D pose, bool relative
  Response: (empty)

- When called, converts relative Pose2D -> world (using /odom) if relative=True,
  otherwise uses pose as world coords.

- Publishes Pose2D target to /base/target_pose (phoenix6_base_node interface).
- Listens to /base/goal_reached (Bool) and logs "Goal reached" when seen.
- Non-blocking service handler (returns immediately after publishing target).
"""

import math
import sys
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

# try to import the custom service; error and exit if not present (explicit failure)
try:
    from tidybot_msgs.srv import ApproachPose
except Exception as e:
    # We want a clear failure if the service type isn't built yet.
    # The user will run colcon build to generate the srv types.
    print("ERROR: tidybot_msgs.srv.ApproachPose not available. Have you added the .srv and built the workspace?")
    print("Exception:", e)
    sys.exit(1)


def rotate_xy(x, y, theta):
    """Rotate (x,y) by theta radians."""
    xr = math.cos(theta) * x - math.sin(theta) * y
    yr = math.sin(theta) * x + math.cos(theta) * y
    return xr, yr


class ApproachNode(Node):
    def __init__(self):
        super().__init__('approach_node_service')

        # params
        self.declare_parameter('service_name', '/approach_pose')
        self.declare_parameter('publish_target_topic', '/base/target_pose')
        self.declare_parameter('odom_topic', '/odom')

        self.service_name = self.get_parameter('service_name').value
        self.publish_target_topic = self.get_parameter('publish_target_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value

        # state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_th = 0.0
        self.odom_seen = False

        # publisher to base controller
        self.target_pub = self.create_publisher(Pose2D, self.publish_target_topic, 10)

        # subscribe to odom and to goal_reached (only for logging)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)
        self.create_subscription(Bool, '/base/goal_reached', self._goal_reached_cb, 10)

        # create service
        self.srv = self.create_service(ApproachPose, self.service_name, self._srv_cb)

        self.get_logger().info(f"ApproachNode (service) ready. Service: {self.service_name} -> publishes Pose2D to {self.publish_target_topic}")
        self.get_logger().info("Use the service to send approach commands (non-blocking).")

    # odom callback
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_x = float(p.x)
        self.current_y = float(p.y)
        self.current_th = float(yaw) - math.pi/2
        self.odom_seen = True

    # goal_reached (log only)
    def _goal_reached_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Goal reached (received from /base/goal_reached).")

    # service callback (non-blocking)
    def _srv_cb(self, request, response):
        """
        Request fields:
            request.pose (geometry_msgs/Pose2D)
            request.relative (bool)
        Publish Pose2D to /base/target_pose (phoenix6_base_node expected format)
        """
        # Read request
        pose2d = request.pose
        relative = bool(request.relative)

        self.get_logger().info(
            f"Service request received: pose=(x={pose2d.x:.3f}, y={pose2d.y:.3f}, th={pose2d.theta:.3f}), relative={relative}"
        )

        # Compute world pose
        world_x, world_y, world_th = self._compute_world_pose(pose2d, relative)

        # Publish target (convert world -> Pose2D expected by phoenix6_base_node)
        self._publish_target_from_world(world_x, world_y, world_th)

        # Return empty response (non-blocking)
        return response

    def _compute_world_pose(self, pose2d: Pose2D, relative: bool):
        """
        If relative==False: treat pose2d as world-frame pose and return it.
        If relative==True: treat pose2d as a local displacement (x forward, y left, theta)
        and convert to world-frame pose by rotating the (x,y) local vector by current_th
        and adding to the current world position. The returned world_th is current_th + pose2d.theta.
        If odom not available when relative=True, warn and treat the request as world coords.
        """
        if not relative:
            return float(pose2d.x), float(pose2d.y), float(pose2d.theta)

        # relative == True => interpret pose2d as local displacement (forward,left,delta_theta)
        if not self.odom_seen:
            self.get_logger().warn(
                "Relative pose requested but /odom not seen yet — treating request.pose as world coords."
            )
            return float(pose2d.x), float(pose2d.y), float(pose2d.theta)

        # local displacement (lx forward, ly left)
        lx = float(pose2d.x)
        ly = float(pose2d.y)
        lth = float(pose2d.theta)

        # rotate local displacement into world-frame delta using current heading
        # (dx,dy) = R(current_th) * (lx, ly)
        dx = math.cos(self.current_th) * lx - math.sin(self.current_th) * ly
        dy = math.sin(self.current_th) * lx + math.cos(self.current_th) * ly

        world_x = self.current_x + dx
        world_y = self.current_y + dy
        world_th = self.current_th + lth

        self.get_logger().info(
            f"Converted relative pose -> world: local=({lx:.3f},{ly:.3f},th={lth:.3f}) "
            f"delta=({dx:.3f},{dy:.3f}), world=({world_x:.3f},{world_y:.3f},th={world_th:.3f})"
        )
        return world_x, world_y, world_th

    def _publish_target_from_world(self, world_x: float, world_y: float, world_th: float):
        """
        phoenix6_base_node expects incoming Pose2D such that it does:
          target_pose.x = msg.y
          target_pose.y = -msg.x
        So inverse mapping:
          msg.x = -world_y
          msg.y = world_x
          ## IGNORE INVERSE MAPPING FOR NOW
        """
        msg = Pose2D()
        msg.x = float(world_x)
        msg.y = float(world_y)
        msg.theta = float(world_th)

        self.get_logger().info(f"Got a pose, moving -> publishing Pose2D to {self.publish_target_topic}: front={msg.x:.3f}, left={msg.y:.3f}, theta={msg.theta:.3f}")
        self.target_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()