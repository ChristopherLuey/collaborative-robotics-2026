#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Pose2D, PoseStamped
from nav_msgs.msg import Odometry
from tidybot_msgs.srv import GetObjectPose, RequestArmMotion, ApproachPose
from tidybot_msgs.msg import TaskRequest
import numpy as np
from scipy.spatial.transform import Rotation as rot_obj


class StateManager(Node):
    """A generic ROS2 node that subscribes to /transcription topic."""

    def __init__(self):
        super().__init__('state_manager')

        # Subscriber
        self.transcription_sub = self.create_subscription(
                                    TaskRequest,              # Message type
                                    '/task_request',    # Topic name
                                    self._request_cb,  # Callback
                                    10                   # QoS depth
                                )
        
        # internal state variables:
        self.current_request = ""
        self.inner_state = "" # this can be used to track progress within a task, e.g. "searching for object", "moving to object", "grasping object", etc.
        self.object = "" # this can be used to track the current object of interest, e.g. "bottle", "door", etc.
        self.target = "basket" #the thing we are trying to place object in, for task 2
        self.object_pose = None
        self.target_pose = None
        self.base_home = None
        self.object_grasp_thresh = 0.65 # how close we need to be to an object to attempt a grasp
        self.object_release_thresh = 0.8 # how close we need to be to a target location to attempt a release
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
        self.vision_client = self.create_client(GetObjectPose, '/sam3/get_object_pose')
        while not self.vision_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /sam3/get_object_pose not available, waiting...')

        self.arm_client = self.create_client(RequestArmMotion, '/request_arm_motion')
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /request_arm_motion not available, waiting...')

        self.base_client = self.create_client(ApproachPose, '/approach_pose')
        while not self.base_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /approach_pose not available, waiting...')

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

        self.odom_sub = self.create_subscription(
                                    Odometry,              # Message type
                                    '/odom',    # Topic name
                                    self._odom_cb,  # Callback
                                    10                   # QoS depth
                                )

        self.timer_cb = self.create_timer(0.1, self._timer_cb)

        self.vision_future = None
    
    def _timer_cb(self):
        """
        This is called at 10hz, and is where we will check if we have a new request, and if so, execute it.
        """
        self.get_logger().info(f'Current request: {self.current_request}, inner state: {self.inner_state}')
        if self.current_request == "":
            return
        
        if self.current_request == "Task2":

            if self.inner_state == "far search":
                self.object_pose = self.get_object_pose(self.object, allow_search=False)
                if self.object_pose is not None:
                    self.inner_state = "Moving to object"
                    des_base = self._find_base_coordinates(self.object_pose, self.object_grasp_thresh)
                    self.move_base(des_base) #get us started moving!
                else:
                    #self.get_logger().error('Could not find object in search.')
                    self.get_logger().info('Waiting on vision...')
                    return
                
            if self.inner_state == "Moving to object":
                if not self.base_ready:
                    self.get_logger().info('Waiting for base to be ready...')
                    return
                else: #this means we arrived!
                    #self.get_object_pose(self.object, allow_search=False) #get an updated pose for grasping, since we should be closer now.
                    success = self.pickup_object(self.object)
                    if success:
                        self.inner_state = "Grasping object"

            if self.inner_state == "Grasping object":
                if not self.arms_ready:
                    self.get_logger().info('Waiting for arms to be ready...')
                    return
                else: #this means we should have the object grasped!
                    self.get_logger().info('Transitioning to searching for target.')
                    #self.inner_state = "Searching for target"
                    # TODO REMOVEEEEEE
                    self.inner_state = "idle"
                    self.current_request = "idle" #transition to idle state, waiting for next request




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
            # TODO write door logic
            self.open_door()
            self.current_request = ""
            return

    def _find_base_coordinates(self, target_pose:Pose, distance_threshold:float):
        """
        This is a helper function that takes in a target pose (6 DOF), and returns the coordinates that the base should move to (X, Y)
        """
        current_base = self.get_base_pose() #current pose of the base
        object_x = target_pose.position.x
        object_y = target_pose.position.y
        Vbo = np.array([object_x - current_base.x, object_y - current_base.y]) #vector from base to object
        Vbo_unit = Vbo / np.linalg.norm(Vbo) if np.linalg.norm(Vbo) != 0 else np.array([0, 0]) #unit vector from base to object
        desired_base_coordinates = Vbo - distance_threshold * Vbo_unit
        global_desired_base_coordinates = desired_base_coordinates + np.array([current_base.x, current_base.y])
        desired_heading = np.arctan2(Vbo[1], Vbo[0]) #angke the base should be facing to look at the object in world
        des_base_pose = Pose2D()
        des_base_pose.x = global_desired_base_coordinates[0]
        des_base_pose.y = global_desired_base_coordinates[1]
        des_base_pose.theta = desired_heading
        return des_base_pose

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

        self.arms_ready = not msg.data

    def _odom_cb(self, msg: Odometry):
        """Called whenever a message is published to /odom."""
        # We can use this callback to update our internal state about the robot's current position, which can be useful for navigation and decision making.
        #self.get_logger().info(f'Current robot pose updated')
        self.current_pose = msg.pose.pose

    def _request_cb(self, msg: TaskRequest):
        """Called whenever a message is published to /task_request."""
        self.get_logger().info(f'Received request')
        self.base_home = self.get_base_pose()
        self.current_request = msg.task_type.data #e.g. "Task1", "Task2", "Task3"
        self.object = msg.object.data #what we are looking for
        self.inner_state = "far search"

    # ---------- high level functions ----------
    def pickup_object(self, object_label:str, max_attempts:int=3):
        """
        This is task #1
        """
        object_pose = self.get_object_pose(object_label, allow_search=False)
        if object_pose is None:
            #self.get_logger().error(f'Could not find pose for object: {object_label}')
            self.get_logger().info("waiting on vision")
            return False
        self.get_logger().info(f'Found object pose at {object_pose}, attempting grasp...')
        result = self.grasp_and_hold(arm_name = "right", grasp_pose=object_pose, end_pose=self.rest_pose)
        return result['success']
        
    def stow_object(self, object_label:str, target_label:str, max_attempts:int=3):
        """
        This is task #2
        """
        object_pose = self.get_object_pose(object_label, allow_search=False)

        if object_pose is None:
            self.get_logger().error(f'Could not find pose for object: {object_label}')
            return
        
        target_pose = self.get_object_pose(target_label, allow_search=False)

        if target_pose is None:
            self.get_logger().error(f'Could not find pose for target location: {target_label}')
            return
        
        result = self.grasp_and_hold(arm_name = "right", grasp_pose=object_pose, end_pose=self.rest_pose)
        result = self.release_and_hold(arm_name = "right", release_pose=target_pose, end_pose=target_pose)
            

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
            result = self.grasp_and_hold(arm_name = "right", grasp_pose=door_handle_pose, end_pose=retracted_pose)

            if not result["success"]:
                self.get_logger().error('Failed to grasp door handle')
                attempts += 1
            else:
                break
        
        # now we try releasing at our current position, and moving back to rest pose
        attempts = 0
        while attempts < max_attempts:
            
            result = self.release_and_hold(arm_name = "right", release_pose=retracted_pose, end_pose=self.rest_pose)

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

        if self.vision_future is None:
            # prepare service request:
            request = GetObjectPose.Request()
            request.prompt = label

            self.vision_future = self.vision_client.call_async(request)
            return None
        
        if not self.vision_future.done():
            return None
        
        response = self.vision_future.result()
        self.vision_future = None
        if response.success:
            self.get_logger().info(f'Found pose for object: {label}')
            return response.pose.pose
        
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
        """
        self.get_logger().info(f'Executing grasp at {grasp_pose} and returning to {end_pose}')
        response = self.arm_client.call(RequestArmMotion.Request(arm_name=arm_name, motion_type="grab", target_pose=grasp_pose))
        response = self.arm_client.call(RequestArmMotion.Request(arm_name=arm_name, motion_type="reach", target_pose=end_pose))

    def release_and_hold(self, arm_name:str, release_pose:Pose, end_pose:Pose):
        """
        Execute a release at the given pose and return to the end pose.
        """
        self.get_logger().info(f'Executing release at {release_pose} and returning to {end_pose}')
        response = self.arm_client.call(RequestArmMotion.Request(arm_name=arm_name, motion_type="release", target_pose=release_pose))
        response = self.arm_client.call(RequestArmMotion.Request(arm_name=arm_name, motion_type="reach", target_pose=end_pose))

    def move_base(self, target_pose:Pose2D):
        """
        Move the robot base to the target pose.
        """
        self.get_logger().info(f'Moving base to {target_pose}')
        self.base_client.call_async(ApproachPose.Request(pose=target_pose, 
                                                   relative=False))
        # Implement navigation logic here

    def rotate_base(self, angle:float):
        """
        Rotate the robot base by a given angle in radians.
        """
        self.get_logger().info(f'Rotating base by {angle:.2f} radians')
        target_pose = Pose2D()
        target_pose.x = 0.0
        target_pose.y = 0.0
        target_pose.theta = angle

        self.base_client.call_async(ApproachPose.Request(pose=target_pose, 
                                                   relative=True))
        # Implement rotation logic here

    def get_base_pose(self):
        """
        Get the current pose of the robot.
        """
        self.get_logger().info('Getting current robot pose')
        current_pose_2d = Pose2D()
        current_pose_2d.x = self.current_pose.position.x
        current_pose_2d.y = self.current_pose.position.y

        orientation = rot_obj.from_quat([self.current_pose.orientation.x, 
                           self.current_pose.orientation.y, 
                           self.current_pose.orientation.z, 
                           self.current_pose.orientation.w])
        
        euler_angles = orientation.as_euler('xyz', degrees=False)
        
        current_pose_2d.theta = euler_angles[2]
        return current_pose_2d  # Replace with actual pose data structure

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