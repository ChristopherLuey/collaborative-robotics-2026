#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose

class StateManager(Node):
    """A generic ROS2 node that subscribes to /transcription topic."""

    def __init__(self):
        super().__init__('transcription_subscriber')

        # Subscriber
        self.subscription = self.create_subscription(
            String,              # Message type
            '/transcription',    # Topic name
            self.transcription_callback,  # Callback
            10                   # QoS depth
        )
        self.subscription  # prevent unused variable warning

        self.get_logger().info('Transcription subscriber node started.')

        self.rest_pose = Pose()
        self.rest_pose.position.x = 0.0
        self.rest_pose.position.y = 0.0
        self.rest_pose.position.z = 0.0
        self.rest_pose.orientation.x = 0.0
        self.rest_pose.orientation.y = 0.0
        self.rest_pose.orientation.z = 0.0
        self.rest_pose.orientation.w = 1.0

    def transcription_callback(self, msg: String):
        """Called whenever a message is published to /transcription."""
        self.get_logger().info(f'Received transcription: "{msg.data}"')


    # ---------- high level functions ----------
    def pickup_object(self, object_label:str, max_attempts:int=3):
        """
        This is task #1
        """
        object_pose = self.get_object_pose(object_label, allow_search=True)
        if object_pose is None:
            self.get_logger().error(f'Could not find pose for object: {object_label}')
            return
        
        attempts = 0

        while attempts < max_attempts:
            result = self.grasp_and_return(grasp_pose=object_pose, end_pose=self.rest_pose)

            if not result["success"]:
                self.get_logger().error(f'Failed to grasp object: {object_label}')
                
            attempts += 1


        
    def stow_object(self, object_label:str, target_label:str, max_attempts:int=3):
        """
        This is task #2
        """
        object_pose = self.get_object_pose(object_label, allow_search=True)
        if object_pose is None:
            self.get_logger().error(f'Could not find pose for object: {object_label}')
            return
        
        target_pose = self.get_object_pose(target_label, allow_search=True)
        if target_pose is None:
            self.get_logger().error(f'Could not find pose for target location: {target_label}')
            return
        
        attempts = 0

        while attempts < max_attempts:
            result = self.grasp_and_return(grasp_pose=object_pose, end_pose=self.rest_pose)

            if not result["success"]:
                self.get_logger().error(f'Failed to grasp object: {object_label}')
                
            attempts += 1
        
        while attempts < max_attempts:
            result = self.release_and_return(release_pose=target_pose, end_pose=target_pose)
            
            if not result["success"]:
                self.get_logger().error(f'Failed to release object: {object_label}')
                
            attempts += 1

    def open_door(self, max_attempts:int=3):
        """
        This is task #3.
        We try to find a door, we move to that door, we attempt a grasp and pull, then we attempt a release and
        return to rest.
        """
        
        # estimate our current pose, and the pose of the door
        current_pose = self.get_current_pose()
        door_pose = self.get_object_pose("door", allow_search=True)
        if door_pose is None:
            self.get_logger().error('Could not find pose for door')
            return

        # we want to get be in front of the door by 30 cm, so it
        # is in range of our grasper. TODO
        near_door_pose = door_pose - current_pose
        self.move_base(target_pose=near_door_pose)
        
        # we attempt to find and grasp the door handle, then pull it by 15cm
        attempts = 0
        while attempts < max_attempts:

            door_handle_pose = self.get_object_pose("handle", allow_search=False)
            if door_handle_pose is None:
                self.get_logger().error('Could not find pose for door handle')
                attempts += 1
                continue
            
            # we pull the door handle towards us by 15 cm:
            retracted_pose = door_handle_pose.copy() # TODO check frames
            retracted_pose.position.x -= 0.15
            result = self.grasp_and_return(grasp_pose=door_handle_pose, end_pose=retracted_pose)

            if not result["success"]:
                self.get_logger().error('Failed to grasp door handle')
                attempts += 1
            else:
                break
        
        # now we try releasing at our current position, and moving back to rest pose
        attempts = 0
        while attempts < max_attempts:
            
            result = self.release_and_return(release_pose=retracted_pose, end_pose=self.rest_pose)

            if not result["success"]:
                self.get_logger().error('Failed to open door')
                attempts += 1
            else:
                break


    # ---------- tools ----------
    def get_object_pose(self, label:str, allow_search=False):  
        """
        Ask the perception system to find an object of a given string label
        returns a PoseStamped ros datatype

        allow_search, if true, allows the robot to turn left and right, to try
        and find the object.

        if the object cannot be found, returns None
        """

        self.get_logger().info(f'Getting pose for object: {label}')
        return None  # Replace with actual pose data structure

    def grasp_and_return(self, grasp_pose:Pose, end_pose:Pose):
        """
        Execute a grasp at the given pose and return to the end pose.
        """
        self.get_logger().info(f'Executing grasp at {grasp_pose} and returning to {end_pose}')
        # Implement grasping logic here

    def release_and_return(self, release_pose:Pose, end_pose:Pose):
        """
        Execute a release at the given pose and return to the end pose.
        """
        self.get_logger().info(f'Executing release at {release_pose} and returning to {end_pose}')
        # Implement releasing logic here

    def move_base(self, target_pose:Pose):
        """
        Move the robot base to the target pose.
        """
        self.get_logger().info(f'Moving base to {target_pose}')
        # Implement navigation logic here

    def rotate_base(self, angle:float):
        """
        Rotate the robot base by a given angle in radians.
        """
        self.get_logger().info(f'Rotating base by {angle:.2f} radians')
        # Implement rotation logic here

    def get_current_pose(self):
        """
        Get the current pose of the robot.
        """
        self.get_logger().info('Getting current robot pose')
        return None  # Replace with actual pose data structure

def main(args=None):
    rclpy.init(args=args)
    node = StateManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()