#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from tidybot_msgs.srv import GetObjectPose, RequestArmMotion
import numpy as np


class StateManager(Node):
    """A generic ROS2 node that subscribes to /transcription topic."""

    def __init__(self):
        super().__init__('state_manager')

        # Subscriber
        self.transcription_sub = self.create_subscription(
                                    String,              # Message type
                                    '/transcription',    # Topic name
                                    self._transcription_cb,  # Callback
                                    10                   # QoS depth
                                )
        
        self.current_request = ""
        
        self.get_logger().info('Transcription subscriber node started.')

        self.rest_pose = Pose()
        self.rest_pose.position.x = 0.0
        self.rest_pose.position.y = 0.0
        self.rest_pose.position.z = 0.0
        self.rest_pose.orientation.x = 0.0
        self.rest_pose.orientation.y = 0.0
        self.rest_pose.orientation.z = 0.0
        self.rest_pose.orientation.w = 1.0

        # create our connections to relevant services:
        self.vision_client = self.create_client(GetObjectPose, '/get_object_pose')
        while not self.vision_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /get_object_pose not available, waiting...')

        self.arm_client = self.create_client(RequestArmMotion, '/request_arm_motion')
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /request_arm_motion not available, waiting...')

        # TODO: add once service implemented
        #self.base_client = self.create_client(RequestArmMotion, '/request_base_motion')
        #while not self.base_client.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info('Service /request_base_motion not available, waiting...')

        self.base_moving_sub = self.create_subscription(
                                    Bool,              # Message type
                                    '/base/goal_reached',    # Topic name
                                    self._base_status_cb,  # Callback
                                    10                   # QoS depth
                                )
        self.base_ready = False
        

        self.arms_moving_sub = self.create_subscription(
                                    Bool,              # Message type
                                    '/arm_queue_full',    # Topic name
                                    self._arm_status_cb,  # Callback
                                    10                   # QoS depth
                                )
        self.arms_ready = False

        self.timer_cb = self.create_timer(0.1, self._timer_cb)
    
    def _timer_cb(self):
        """
        This is called at 10hz, and is where we will check if we have a new request, and if so, execute it.
        """
        if self.current_request == "":
            return
        
        if not self.base_ready:
            self.get_logger().info('Waiting for base to be ready...')
            return
        
        if not self.arms_ready:
            self.get_logger().info('Waiting for arms to be ready...')
            return

        # we have a request, and our base and arms are ready, so we can execute the request!
        self.get_logger().info(f'Executing request: "{self.current_request}"')

        # TODO parse the request and execute the appropriate high level function


    def _base_status_cb(self, msg: Bool):
        """Called whenever a message is published to /base/goal_reached."""
        if msg.data:
            self.get_logger().info('Base has reached its goal.')            
        else:
            self.get_logger().info('Base is moving towards its goal.')

        self.base_ready = msg.data

    def _arm_status_cb(self, msg: Bool):
        """Called whenever a message is published to /arm_queue_full."""
        if msg.data:
            self.get_logger().info('Arm command queue is full.')
        else:
            self.get_logger().info('Arm command queue has space.')

        self.arms_ready = not msg.data

    def _transcription_cb(self, msg: String):
        """Called whenever a message is published to /transcription."""
        self.get_logger().info(f'Received transcription: "{msg.data}"')
        self.current_request = msg.data

    # ---------- high level functions ----------
    def pickup_object(self, object_label:str, max_attempts:int=3):
        """
        This is task #1
        TODO: add base motion - what if we're too far away?
        """
        object_pose = self.get_object_pose(object_label, allow_search=True)
        if object_pose is None:
            self.get_logger().error(f'Could not find pose for object: {object_label}')
            return
        
        attempts = 0

        while attempts < max_attempts:
            result = self.grasp_and_hold(grasp_pose=object_pose, end_pose=self.rest_pose)

            if not result["success"]:
                self.get_logger().error(f'Failed to grasp object: {object_label}')
                
            attempts += 1
        
    def stow_object(self, object_label:str, target_label:str, max_attempts:int=3):
        """
        This is task #2
        TODO: add base motion - what if we're too far away?
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
            result = self.grasp_and_hold(grasp_pose=object_pose, end_pose=self.rest_pose)

            if not result["success"]:
                self.get_logger().error(f'Failed to grasp object: {object_label}')
                
            attempts += 1
        
        while attempts < max_attempts:
            result = self.release_and_hold(release_pose=target_pose, end_pose=target_pose)
            
            if not result["success"]:
                self.get_logger().error(f'Failed to release object: {object_label}')
                
            attempts += 1

    def open_door(self, max_attempts:int=3):
        """
        This is task #3.
        We try to find a door, we move to that door, we attempt a grasp and pull, then we attempt a release and
        return to rest.
        TODO: add base motion - what if we're too far away?
        """
        
        # estimate our current pose, and the pose of the door
        current_position = self.get_base_pose()
        door_pose = self.get_object_pose("door", allow_search=True)
        if door_pose is None:
            self.get_logger().error('Could not find pose for door')
            return

        # we want to get be in front of the door by 30 cm, so it
        # is in range of our grasper. TODO: finish this
        near_door_pose = door_pose - current_position
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
            retracted_pose = door_handle_pose.copy() # TODO check frames fix this
            retracted_pose.position.x -= 0.15
            result = self.grasp_and_hold(grasp_pose=door_handle_pose, end_pose=retracted_pose)

            if not result["success"]:
                self.get_logger().error('Failed to grasp door handle')
                attempts += 1
            else:
                break
        
        # now we try releasing at our current position, and moving back to rest pose
        attempts = 0
        while attempts < max_attempts:
            
            result = self.release_and_hold(release_pose=retracted_pose, end_pose=self.rest_pose)

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

        calls itself recursively if search is allowed.

        if the object cannot be found, returns None
        """

        self.get_logger().info(f'Getting pose for object: {label}')

        # prepare service request:
        request = GetObjectPose.Request()
        request.prompt = label

        response = self.vision_client.call(request)
        if response.success:
            self.get_logger().info(f'Found pose for object: {label}')
            return response.pose
        else:
            # we try to look around, and see if object is in view.
            if allow_search:
                self.rotate_base(angle=np.pi/4)  # Rotate 0.5 radians to the right and try again
                pose = self.get_object_pose(label, allow_search=False) 
                if pose is not None:
                    return pose
                self.rotate_base(angle=-np.pi/2)  # Rotate 0.5 radians to the left and try again
                pose = self.get_object_pose(label, allow_search=False)
                if pose is not None:
                    return pose
                
            self.get_logger().error(f'Could not find pose for object: {label}')

        return None
    
    def grasp_and_hold(self, arm_name:str, grasp_pose:Pose, end_pose:Pose):
        """
        Execute a grasp at the given pose and return to the end pose.
        TODO: this needs to be blocking and is not, use the arm status topic.
        """
        self.get_logger().info(f'Executing grasp at {grasp_pose} and returning to {end_pose}')
        response = self.arm_client.call(RequestArmMotion.Request(arm_name=arm_name, motion_type="grab", target_pose=grasp_pose))
        response = self.arm_client.call(RequestArmMotion.Request(arm_name=arm_name, motion_type="reach", target_pose=end_pose))

    def release_and_hold(self, arm_name:str, release_pose:Pose, end_pose:Pose):
        """
        Execute a release at the given pose and return to the end pose.
        TODO: this needs to be blocking and is not, use the arm status topic.
        """
        self.get_logger().info(f'Executing release at {release_pose} and returning to {end_pose}')
        response = self.arm_client.call(RequestArmMotion.Request(arm_name=arm_name, motion_type="release", target_pose=release_pose))
        response = self.arm_client.call(RequestArmMotion.Request(arm_name=arm_name, motion_type="reach", target_pose=end_pose))

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

    def get_base_pose(self):
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