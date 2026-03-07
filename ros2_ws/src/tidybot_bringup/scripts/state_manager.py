#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Pose2D
from tidybot_msgs.srv import GetObjectPose, RequestArmMotion
from tidybot_msgs.msg import TaskRequest
import numpy as np


class StateManager(Node):
    """A generic ROS2 node that subscribes to /transcription topic."""

    def __init__(self):
        super().__init__('state_manager')

        # Subscriber
        self.transcription_sub = self.create_subscription(
                                    TaskRequest,              # Message type
                                    '/task_request',    # Topic name
                                    self._transcription_cb,  # Callback
                                    10                   # QoS depth
                                )
        
        # internal state variables:
        self.current_request = ""
        self.inner_state = "" # this can be used to track progress within a task, e.g. "searching for object", "moving to object", "grasping object", etc.
        self.object = "" # this can be used to track the current object of interest, e.g. "bottle", "door", etc.
        self.target = "basket" #the thing we are trying to place object in, for task 2
        self.object_pose = None
        self.target_pose = None
        self.base_home = Pose2D() # TODO set this to the actual home pose of the robot
        self.object_grasp_thresh = 0.15 # how close we need to be to an object to attempt a grasp
        self.object_release_thresh = 0.15 # how close we need to be to a target location to attempt a release
        self.get_logger().info('Transcription subscriber node started.')

        #rest pose for our arms to return to after a grasp/release action, can be updated later if needed
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
        
        if self.current_request == "Task2":

            if self.inner_state == "far search":
                self.object_pose = self.get_object_pose(self.object, allow_search=True)
                if self.object_pose is not None:
                    self.inner_state = "Moving to object"
                    des_base = self._find_base_coordinates(self.object_pose, self.object_grasp_thresh)
                    self.move_base(des_base) #get us started moving!
                else:
                    self.get_logger().error('Could not find object in search.')
                    return
            if self.inner_state == "Moving to object":
                if not self.base_ready:
                    self.get_logger().info('Waiting for base to be ready...')
                    return
                else: #this means we arrived!
                    self.inner_state = "Grasping object"
                    #self.get_object_pose(self.object, allow_search=False) #get an updated pose for grasping, since we should be closer now.
                    self.pickup_object(self.object)
            if self.inner_state == "Grasping object":
                if not self.arms_ready:
                    self.get_logger().info('Waiting for arms to be ready...')
                    return
                else: #this means we should have the object grasped!
                    self.get_logger().info('Transitioning to searching for target.')
                    self.inner_state = "Searching for target"
            if self.inner_state == "Searching for target":
                self.target_pose = self.get_object_pose(self.target, allow_search=True) #find basket pose
                if self.target_pose is not None:
                    des_base = self._find_base_coordinates(self.target_pose, self.object_release_thresh)
                    self.move_base(des_base) #get us started moving!
                    self.inner_state = "Moving to target"
                else:
                    self.get_logger().error('Could not find target in search.')
                    return
            if self.inner_state == "Moving to target":
                if not self.base_ready:
                    self.get_logger().info('Waiting for base to be ready...')
                    return
                else: #this means we arrived!
                    #self.get_object_pose(self.target, allow_search=False) #get an updated pose for releasing, since we should be closer now.
                    self.stow_object(self.object, self.target)
                    self.inner_state = "Releasing object"
            if self.inner_state == "Releasing object":
                if not self.arms_ready:
                    self.get_logger().info('Waiting for arms to be ready...')
                    return
                else: #this means we should have the object released!
                    self.get_logger().info('Returning to home position.')
                    self.move_base(self.base_home) #move back to rest pose
                    self.inner_state = "Returning to Home"
            if self.inner_state == "Returning to Home":
                if not self.base_ready:
                    self.get_logger().info('Waiting for base to be ready...')
                    return
                else: #this means we are back home and done with the task!
                    self.get_logger().info('Arrived at home, waiting for next command!')
                    self.inner_state = "idle"
                    self.current_request = "idle" #transition to idle state, waiting for next request
            
            return


        if self.current_request == "Task3":
            self.open_door()
            self.current_request = ""
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

    def _find_base_coordinates(self, target_pose:Pose, distance_threshold:float):
        """
        This is a helper function that takes in a target pose (6 DOF), and returns the coordinates that the base should move to (X, Y)
        """
        current_base = self.get_base_pose() #current pose of the base
        object_x = target_pose.position.x
        object_y = target_pose.position.y
        Vbo = np.array([object_x - current_base.position.x, object_y - current_base.position.y]) #vector from base to object
        Vbo_unit = Vbo / np.linalg.norm(Vbo) if np.linalg.norm(Vbo) != 0 else np.array([0, 0]) #unit vector from base to object
        desired_base_coordinates = Vbo - distance_threshold * Vbo_unit
        global_desired_base_coordinates = desired_base_coordinates + np.array([current_base.position.x, current_base.position.y])
        desired_heading = np.arctan2(Vbo[1], Vbo[0]) #angke the base should be facing to look at the object in world
        des_base_pose = Pose2D()
        des_base_pose.x = global_desired_base_coordinates[0]
        des_base_pose.y = global_desired_base_coordinates[1]
        des_base_pose.theta = desired_heading
        return des_base_pose

        return relative_pose
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

    def _transcription_cb(self, msg: TaskRequest):
        """Called whenever a message is published to /transcription."""
        self.get_logger().info(f'Received transcription: "{msg.data}"')
        self.base_home = self.get_base_pose()
        self.current_request = msg.TaskType #e.g. "Task1", "Task2", "Task3"
        self.object = msg.Object #what we are looking for
        self.inner_state = "far search"

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

    def move_base(self, target_pose:Pose2d):
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