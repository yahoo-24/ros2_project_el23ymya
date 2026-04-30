"""
BlueBoxFinder ROS2 Node

This module implements a ROS2 node for autonomous robot navigation and colored box detection.
The robot explores an environment using a greedy search algorithm, spins to detect colored
boxes (blue, red, green) using HSV color filtering, and navigates toward detected blue boxes.

The system uses:
- Nav2 for autonomous navigation to waypoints
- OpenCV for image processing and color detection
- A min-pooling approach for map decomposition and path planning

Author: Yahia Abuhelweh (Mostly)
A few bits of code were taken from University of Leeds COMP3631 module
"""

import threading
import sys
import time
import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import rclpy.subscription
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal
import math
from .GreedySearch import PathPlanner
import matplotlib.pyplot as plt

# Maximum angular velocity for rotation (rad/s)
MAX_ANGULAR_VEL = 2.84

# Maximum linear velocity for forward movement (m/s)
MAX_VEL = 0.20

# Minimum contour area threshold to consider a blue box "found"
AREA_THRESHOLD = 10_000


class BlueBoxFinder(Node):
    """
    A ROS2 node that autonomously explores an environment to find colored boxes.
    
    The robot navigates through waypoints, performs 360-degree scans at each location,
    and uses color detection to identify blue, red, and green boxes. When a blue box
    is detected, the robot centers it in the camera view and approaches it.
    
    Attributes:
        action_client: Nav2 action client for sending navigation goals.
        publisher: Publisher for velocity commands on /cmd_vel.
        sensitivity: HSV color range sensitivity for detection.
        hsv_*_lower/upper: HSV color range bounds for blue, red, and green detection.
        red_found, green_found, blue_found: Flags indicating if respective colors were detected.
        position: Current robot position as numpy array [x, y].
        yaw: Current robot orientation in degrees (0-360).
        arrived: Flag indicating if the robot has reached its navigation goal.
        target: Current navigation target position.
        image: Occupancy grid map loaded from PGM file.
        coord: Coordinate mapping from pixel indices to world coordinates.
        explored: List of explored grid cells.
        unexplored: List of unexplored grid cells.
        blue_mask: Current blue color mask from camera image.
    """
    
    def __init__(self):
        """
        Initialize the BlueBoxFinder node.
        
        Sets up ROS2 publishers, subscribers, action clients, and initializes
        HSV color thresholds for blue, red, and green detection.
        """
        super().__init__('BBF')
        
        # Action client for Nav2 navigation goals
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Publisher for robot velocity commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for camera images
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        
        # Subscriber for odometry data (robot position and orientation)
        self.subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        
        # Rate limiter for control loops (10 Hz)
        self.rate = self.create_rate(10)
        
        # HSV color detection sensitivity (range around target hue)
        self.sensitivity = 20
        
        # Blue color HSV bounds (hue ~120 in OpenCV's 0-179 scale)
        self.hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        self.hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])
        
        # Red color HSV bounds (red wraps around 0/180, so two ranges needed)
        self.hsv_red_lower1 = np.array([180 - self.sensitivity, 100, 100])
        self.hsv_red_upper1 = np.array([179, 255, 255])
        self.hsv_red_upper2 = np.array([0 + self.sensitivity, 255, 255])
        self.hsv_red_lower2 = np.array([0, 100, 100])
        
        # Green color HSV bounds (hue ~60 in OpenCV's scale)
        self.hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        self.hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        
        # Detection state flags
        self.red_found = False
        self.green_found = False
        self.blue_found = False
        
        # Robot state
        self.position = None  # Will be set by odom_callback
        self.arrived = False  # Navigation completion flag
        
    def odom_callback(self, msg):
        """
        Process odometry messages to update robot position and orientation.
        
        Extracts the robot's x, y position and converts quaternion orientation
        to yaw angle in degrees (0-360 range).
        
        Args:
            msg: Odometry message containing pose information.
        """
        # Extract position from odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.position = np.array([x, y])
        
        # Extract quaternion orientation and convert to yaw angle
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        
        # Convert quaternion to yaw (only z and w needed for 2D rotation)
        self.yaw = math.degrees(math.atan2(z, w) * 2)
        
        # Normalize yaw to 0-360 range
        if self.yaw < 0:
            self.yaw += 360

    def send_goal(self, x, y, yaw):
        """
        Send a navigation goal to the Nav2 action server.
        
        Creates a NavigateToPose goal with the specified position and orientation,
        then sends it asynchronously to the navigation stack.
        
        Args:
            x: Target x coordinate in map frame.
            y: Target y coordinate in map frame.
            yaw: Target orientation in radians.
        """
        self.get_logger().info(f'Moving to: [x: {x}, y: {y}, yaw: {yaw}]')
        self.arrived = False
        
        # Construct the navigation goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set target position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Convert yaw angle to quaternion (only z and w components for 2D)
        goal_msg.pose.pose.orientation.z = np.sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = np.cos(yaw / 2)

        # Wait for the action server and send goal asynchronously
        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle the response from the Nav2 action server after sending a goal.
        
        If the goal is accepted, registers a callback for the result.
        
        Args:
            future: Future containing the goal handle response.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Handle the navigation result when the robot reaches its goal.
        
        Sets the arrived flag to True when navigation completes.
        
        Args:
            future: Future containing the navigation result.
        """
        result = future.result().result
        self.arrived = True
        self.get_logger().info(f'Navigation result: {result}')

    def feedback_callback(self, feedback_msg):
        """
        Process navigation feedback during goal execution.
        
        Checks if the robot is close enough to the target to consider
        navigation complete (within 0.2m threshold).
        
        Args:
            feedback_msg: Feedback message containing current navigation state.
        """
        # Access the current pose from feedback
        current_pose = feedback_msg.feedback.current_pose
        
        # Calculate distance to target and mark arrived if close enough
        distance = np.linalg.norm(self.target - self.position)
        if distance < 0.2:
            self.arrived = True
        
    def image_callback(self, data):
        """
        Process incoming camera images for color detection.
        
        Converts the image to HSV color space and applies color masks to detect
        blue, red, and green objects. Updates detection flags and displays
        filtered images for debugging.
        
        Args:
            data: ROS Image message from the camera.
        """
        # Convert ROS Image to OpenCV format
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        # Convert BGR to HSV color space for color detection
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply blue color mask
        self.blue_mask = cv2.inRange(hsv_image, self.hsv_blue_lower, self.hsv_blue_upper)
        
        # Create filtered image showing only blue regions
        filtered_img = cv2.bitwise_and(image, image, mask=self.blue_mask)
        
        # Check if blue is detected and above area threshold
        if np.any(self.blue_mask):
            area = self.find_area()
            if area > AREA_THRESHOLD:
                self.blue_found = True
        else:
            self.blue_found = False
        
        # Apply red color masks (two masks needed because red wraps around hue 0/180)
        red_mask_1 = cv2.inRange(hsv_image, self.hsv_red_lower1, self.hsv_red_upper1)
        red_mask_2 = cv2.inRange(hsv_image, self.hsv_red_lower2, self.hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        
        # Display red detection window
        red_filtered = cv2.bitwise_and(image, image, mask=red_mask)
        cv2.namedWindow('Red_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('Red_Feed', red_filtered)
        cv2.resizeWindow('Red_Feed', 320, 240)
        cv2.waitKey(3)
        
        # Log red detection (only first time)
        if not self.red_found:
            self.red_found = np.any(red_mask)
            if self.red_found:
                self.get_logger().info("Red Box is found")
        
        # Apply green color mask
        green_mask = cv2.inRange(hsv_image, self.hsv_green_lower, self.hsv_green_upper)
        
        # Display green detection window
        green_filtered = cv2.bitwise_and(image, image, mask=green_mask)
        cv2.namedWindow('Green_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('Green_Feed', green_filtered)
        cv2.resizeWindow('Green_Feed', 320, 240)
        cv2.waitKey(3)
        
        # Log green detection (only first time)
        if not self.green_found:
            self.green_found = np.any(green_mask)
            if self.green_found:
                self.get_logger().info("Green Box is found")
            
        # Display main camera feed with blue filter
        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', filtered_img)
        cv2.resizeWindow('camera_Feed', 320, 240)
        cv2.waitKey(3)
        
    def rotate(self, value):
        """
        Rotate the robot at a specified angular velocity.
        
        Publishes a Twist message with the given angular velocity and
        waits one rate cycle.
        
        Args:
            value: Angular velocity in rad/s (positive = counterclockwise).
        """
        desired_rotation = Twist()
        desired_rotation.angular.z = value
        
        self.publisher.publish(desired_rotation)
        self.rate.sleep()
        
    def rotate30(self):
        """
        Rotate the robot approximately 30 degrees.
        
        Rotates at pi/6 rad/s for 1 second, resulting in approximately
        30 degrees of rotation (pi/6 radians).
        """
        desired_rotation = Twist()
        desired_rotation.angular.z = np.pi / 6  # 30 deg/s
        
        # Rotate for 10 cycles at 10Hz = 1 second
        for _ in range(10):
            self.publisher.publish(desired_rotation)
            self.rate.sleep()
            
    def spin_360(self):
        """
        Perform a full 360-degree scan looking for blue boxes.
        
        Rotates the robot in 30-degree increments, checking for blue
        detection at each step. Stops early if blue is found.
        
        Returns:
            bool: True if a blue box was found during the scan, False otherwise.
        """
        for i in range(12):  # 12 x 30 degrees = 360 degrees
            print(f"Spun to {30 * i} degrees")
            self.rotate30()
            if self.blue_found:
                print("Blue Box Found!!!")
                return True
        print("Box Not Found :(")
        return False
    
    def move_forward(self, error):
        """
        Move the robot forward at a velocity proportional to the error.
        
        Args:
            error: Desired linear velocity in m/s.
        """
        desired_velocity = Twist()
        desired_velocity.linear.x = error
        
        # Publish for 3 cycles to ensure command is received
        for _ in range(3):
            self.publisher.publish(desired_velocity)
            self.rate.sleep()
    
    def stop(self):
        """
        Stop all robot motion.
        
        Publishes zero velocity commands for both linear and angular
        motion, repeated to ensure the robot stops.
        """
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.0
        desired_velocity.angular.z = 0.0
        
        # Publish stop command for 10 cycles to ensure it takes effect
        for _ in range(10):
            self.publisher.publish(desired_velocity)
            self.rate.sleep()

    def read_image(self):
        """
        Load and process the occupancy grid map from a PGM file.
        
        Reads the map image, thresholds it to create a binary occupancy grid
        where 1 represents free space and 0 represents obstacles.
        """
        file = '/uolstore/home/users/el23ymya/ros2_ws/src/ros2_project_el23ymya/map/map.pgm'
        self.image = cv2.imread(file, 0)
        
        # Threshold: pixels < 250 are obstacles (0), others are free space (1)
        self.image[self.image < 250] = 0
        self.image[self.image != 0] = 1

    def generate_coordinates(self, x_end, x_start, y_start, y_end):
        """
        Generate a coordinate mapping from pixel indices to world coordinates.
        
        Creates a 2D array where each element contains the [x, y] world
        coordinates corresponding to that pixel position in the map.
        
        Args:
            x_end: Maximum x coordinate in world frame.
            x_start: Minimum x coordinate in world frame.
            y_start: Maximum y coordinate in world frame.
            y_end: Minimum y coordinate in world frame.
        """
        shape = self.image.shape
        
        # Create evenly spaced coordinate arrays
        x_axis = np.linspace(x_start, x_end, shape[1]) - 3  # Offset adjustment
        y_axis = np.linspace(y_start, y_end, shape[0])
        
        # Build coordinate grid
        self.coord = np.zeros((shape[0], shape[1], 2))
        for index, y in enumerate(y_axis):
            self.coord[index] = np.array([x_axis, np.ones(shape[1]) * y]).T
            
    def expand_point(self, current_index, decomposed_image):
        """
        Expand explored region from the current position.
        
        Marks neighboring cells as explored by expanding in all four cardinal
        directions until hitting obstacles. This implements a flood-fill style
        expansion of the explored area.
        
        Args:
            current_index: [row, col] index of current position in decomposed grid.
            decomposed_image: The min-pooled occupancy grid.
        """
        max_index_x = decomposed_image.shape[1]
        max_index_y = decomposed_image.shape[0]
        
        # Mark current position as explored
        if current_index not in self.explored and current_index in self.unexplored:
            self.explored.append(current_index)
            self.unexplored.remove(current_index)
        
        # Expand in each cardinal direction until hitting an obstacle
        
        # Left expansion
        idx = 0
        stop = False
        while not stop:
            for i in range(current_index[1] - 1 - idx, current_index[1] + 2 + idx):
                node = np.array([current_index[0] - idx, i])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    node = node.tolist()
                    if free.any() and node not in self.explored:
                        print("Node Added!")
                        self.explored.append(node)
                        self.unexplored.remove(node)
                    elif not free.any():
                        stop = True
            idx += 1
            
        # Right expansion
        idx = 0
        stop = False
        while not stop:
            for i in range(current_index[1] - 1 - idx, current_index[1] + 2 + idx):
                node = np.array([current_index[0] + idx, i])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    node = node.tolist()
                    if free.any() and node not in self.explored:
                        print("Node Added!")
                        self.explored.append(node)
                        self.unexplored.remove(node)
                    elif not free.any():
                        stop = True
            idx += 1
            
        # Up expansion
        idx = 0
        stop = False
        while not stop:
            for i in range(current_index[0] - 1 - idx, current_index[0] + 2 + idx):
                node = np.array([i, current_index[1] - idx])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    node = node.tolist()
                    if free.any() and node not in self.explored:
                        print("Node Added!")
                        self.explored.append(node)
                        self.unexplored.remove(node)
                    elif not free.any():
                        stop = True
            idx += 1
            
        # Down expansion
        idx = 0
        stop = False
        while not stop:
            for i in range(current_index[0] - 1 - idx, current_index[0] + 2 + idx):
                node = np.array([i, current_index[1] + idx])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    node = node.tolist()
                    if free.any() and node not in self.explored:
                        print("Node Added!")
                        self.explored.append(node)
                        self.unexplored.remove(node)
                    elif not free.any():
                        stop = True
            idx += 1
       
    def find_centre(self):
        """
        Find the centroid x-coordinate of the largest blue contour.
        
        Uses contour detection on the blue mask to find the largest
        blue region and calculates its center of mass.
        
        Returns:
            int: X-coordinate of the blue region centroid, or 0 if no contours found.
        """
        contours, _ = cv2.findContours(
            self.blue_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        cx = 0

        if len(contours) > 0:
            # Find the largest contour by area
            c = max(contours, key=cv2.contourArea)

            # Calculate centroid using image moments
            M = cv2.moments(c)
            cx, _ = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            
        return cx
    
    def find_area(self):
        """
        Find the area of the largest blue contour.
        
        Returns:
            float: Area of the largest blue contour in pixels, or -1 if no contours found.
        """
        contours, _ = cv2.findContours(
            self.blue_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        area = -1

        if len(contours) > 0:
            # Find the largest contour by area
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
        return area
       
    def centre_blue_box(self):
        """
        Rotate the robot to center the blue box in the camera frame.
        
        Uses proportional control to adjust angular velocity based on the
        difference between the blue box centroid and the image center.
        """
        cx = self.find_centre()
        TARGET_CENTRE = 478  # Target x-coordinate (image center)
        
        # Rotate until the box is centered within tolerance
        while cx < TARGET_CENTRE - 8 or cx > TARGET_CENTRE + 8:
            # Proportional control: error scaled by max velocity, divided by 5 for smoothness
            error = ((TARGET_CENTRE - cx) / TARGET_CENTRE * MAX_ANGULAR_VEL) / 5
            self.rotate(error)
            print(f"Rotating: {error}")
            cx = self.find_centre()
            
    def move_towards_box(self):
        """
        Navigate toward the detected blue box.
        
        First centers the box in the camera view, then moves forward using
        proportional control based on the contour area (larger area = closer).
        Includes gradual acceleration and periodic re-centering.
        """
        self.centre_blue_box()
        area = self.find_area()
        print(area)
        
        start_counter = 0  # For gradual acceleration
        readjust_counter = 0  # For periodic re-centering
        GRADUAL_ACC_THRESHOLD = 15  # Steps for full acceleration ramp
        TARGET_AREA = 390_000  # Target contour area (~1m distance)
        READJUST_THRESHOLD = 50  # Steps between re-centering
        
        # Move until target area is reached (within tolerance)
        while area < TARGET_AREA - 5_000 or area > TARGET_AREA + 5_000:
            # Proportional control based on area error
            error = (TARGET_AREA - area) / TARGET_AREA * MAX_VEL
            
            # Gradual acceleration to avoid jerky motion
            if start_counter < GRADUAL_ACC_THRESHOLD:
                start_counter += 1
                error = min(error, MAX_VEL * start_counter / GRADUAL_ACC_THRESHOLD)
                
            self.move_forward(error)
            print(f"Moving: {error}, Current Area = {area}")
            area = self.find_area()
            
            # Periodically re-center the box
            readjust_counter += 1
            if readjust_counter >= READJUST_THRESHOLD:
                self.centre_blue_box()
                readjust_counter = 0
                    
    def controller(self):
        """
        Main control loop for autonomous exploration and box finding.
        
        Implements the exploration algorithm:
        1. Wait for initial pose
        2. Load and decompose the map
        3. At each location, perform a 360-degree scan
        4. If blue box found, approach it
        5. Otherwise, plan path to furthest unexplored cell and navigate there
        6. Repeat until blue box is found
        """
        # Wait for odometry data to be available
        while self.position is None:
            print("Waiting For Pose...")
            time.sleep(1)
            
        self.explored = []
        
        # Load and process the map
        print("Reading Image...")
        self.read_image()
        
        print("Generating Coordinates...")
        self.generate_coordinates(9.93, -12.2, 6.39, -15.8)
        
        # Decompose map into larger cells for planning
        print("Decomposing Image...")
        decomposed_image = min_pool(self.image, 27)
        
        # Initialize unexplored cells as all free cells in decomposed map
        self.unexplored = np.argwhere(decomposed_image).tolist()
        
        # Generate coordinate mapping for decomposed grid
        print("Decomposing Coordinates...")
        decomposed_coord = redefine_values(self.coord, 27, 2)
        
        # Find starting position in decomposed grid
        print("Finding Current Position...")
        current_index = find_current_node(
            self.position,
            decomposed_coord[decomposed_image > 0],
            decomposed_coord
        )
        
        print(f"Current position is {decomposed_coord[current_index[0], current_index[1], :]}, Index {current_index}")
        
        blue_found = False
        operation_complete = False
        
        # Main exploration loop
        while not operation_complete:
            # Scan for blue box
            blue_found = self.spin_360()
            self.stop()
            
            if blue_found or self.blue_found:
                # Blue box found - approach it
                self.move_towards_box()
                self.stop()
                operation_complete = True
            else:
                # Mark current area as explored
                print("Expanding Node...")
                self.expand_point(current_index, decomposed_image)
                
                # Find next exploration target (furthest unexplored cell)
                print("Choosing Next Node...")
                next_node_index = find_furthest_node(current_index, self.unexplored)
                print(f"Node Chosen is {decomposed_coord[next_node_index[0], next_node_index[1]]}, Index {next_node_index}")
                
                # Plan path to next node using greedy search
                planner = PathPlanner(current_index, next_node_index, decomposed_image)
                path = planner.plan()
                start = path.pop(0)  # Remove start node from path
                
                # Navigate through each waypoint in the path
                for node_index in path:
                    # Visualize the current path
                    plt.imshow(decomposed_image, cmap='gray')
                    path_print = np.array(path)
                    un_print = np.array(self.explored)
                    plt.scatter(un_print[:, 1], un_print[:, 0], label='Explored')
                    plt.scatter(path_print[:, 1], path_print[:, 0], marker='X', label='Path')
                    plt.scatter(start[1], start[0], marker='v', label='Start')
                    plt.scatter(path_print[-1, 1], path_print[-1, 0], marker='^', label='End')
                    plt.legend()
                    plt.show(block=False)
                    plt.pause(2)
                    
                    # Get world coordinates for this waypoint
                    next_node = decomposed_coord[node_index[0], node_index[1]]
                
                    # Navigate to waypoint
                    self.target = next_node
                    self.send_goal(float(next_node[0]), float(next_node[1]), 0.0)
                    
                    # Wait for navigation to complete
                    while not self.arrived:
                        time.sleep(1)
                        
                    # Scan at this location
                    self.spin_360()
                    self.expand_point(node_index, decomposed_image)
                    
                    if self.blue_found:
                        break
                        
                    current_index = node_index
                    
        # Keep node alive after operation completes
        while True:
            time.sleep(10)
    
    
def min_pool(frame, kernel_size):
    """
    Apply min pooling to downsample an image.
    
    Reduces image resolution by taking the minimum value within each
    kernel-sized block. Useful for conservative obstacle expansion in
    occupancy grids (any obstacle in a block makes the whole block an obstacle).
    
    Args:
        frame: 2D numpy array (grayscale image).
        kernel_size: Size of the pooling kernel (square).
        
    Returns:
        numpy.ndarray: Downsampled image, or -1 if frame is smaller than kernel.
    """
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    # Calculate output dimensions
    result_size_col = np.ceil(frame.shape[1] / kernel_size)
    result_size_row = np.ceil(frame.shape[0] / kernel_size)
    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32)))
    
    # Apply min pooling
    row_index = 0
    while row_index < result_size_row:
        row_start = row_index * kernel_size
        row_end = row_start + kernel_size
        col_index = 0
        while col_index < result_size_col:
            col_start = col_index * kernel_size
            col_end = col_start + kernel_size
            result_arr[row_index, col_index] = np.min(frame[row_start:row_end, col_start:col_end])
            col_index += 1
        row_index += 1
        
    return result_arr


def redefine_values(frame, kernel_size, dimensions):
    """
    Downsample a coordinate array by taking center values of each block.
    
    Used to create a coordinate mapping for the min-pooled grid by taking
    the coordinate at the center of each pooling block.
    
    Args:
        frame: 3D numpy array of shape (rows, cols, dimensions).
        kernel_size: Size of the pooling kernel (square).
        dimensions: Number of coordinate dimensions (typically 2 for x, y).
        
    Returns:
        numpy.ndarray: Downsampled coordinate array, or -1 if frame is smaller than kernel.
    """
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    # Calculate center position within kernel
    centre = (kernel_size + 1) / 2
    
    # Calculate output dimensions
    result_size_col = np.ceil(frame.shape[1] / kernel_size).astype(np.int32)
    result_size_row = np.ceil(frame.shape[0] / kernel_size).astype(np.int32)

    # Handle edge cases for partial blocks at boundaries
    rem_col = frame.shape[1] % kernel_size
    rem_row = frame.shape[0] % kernel_size
    rem_centre_col = np.floor(rem_col / 2) if rem_col != 0 else centre
    rem_centre_row = np.floor(rem_row / 2) if rem_row != 0 else centre

    result_arr = np.zeros((result_size_row, result_size_col, int(dimensions)))
    
    # Extract center values for each block
    row_index = 0
    while row_index < result_size_row:
        # Use remainder center for last row
        if row_index == result_size_row - 1:
            row_centre = row_index * kernel_size + rem_centre_row
        else:
            row_centre = row_index * kernel_size + centre
        col_index = 0

        while col_index < result_size_col:
            # Use remainder center for last column
            if col_index == result_size_col - 1:
                col_centre = col_index * kernel_size + rem_centre_col
            else:
                col_centre = col_index * kernel_size + centre
            result_arr[row_index, col_index] = frame[int(row_centre), int(col_centre)]
            col_index += 1
        row_index += 1

    return result_arr


def find_furthest_node(current, nodes):
    """
    Find the node furthest from the current position.
    
    Used to select the next exploration target to maximize coverage.
    
    Args:
        current: [row, col] index of current position.
        nodes: List of [row, col] indices of candidate nodes.
        
    Returns:
        list: [row, col] index of the furthest node.
    """
    distances = np.linalg.norm(np.array(nodes) - np.array(current), axis=1)
    node = nodes[np.argmax(distances)]
    return node


def validate_index(index, max_x, max_y):
    """
    Check if an index is within valid grid bounds.
    
    Args:
        index: [row, col] index to validate.
        max_x: Maximum valid column index.
        max_y: Maximum valid row index.
        
    Returns:
        bool: True if index is within bounds, False otherwise.
    """
    if index[0] < 0 or index[0] > max_x or index[1] < 0 or index[1] > max_y:
        return False
    return True

            
def find_current_node(position, filtered_coords, decomposed_coords):
    """
    Find the grid cell index corresponding to a world position.
    
    Searches through the coordinate mapping to find which grid cell
    the robot's current world position falls into.
    
    Args:
        position: [x, y] world coordinates of current position.
        filtered_coords: Array of valid (free space) world coordinates.
        decomposed_coords: Full coordinate mapping array.
        
    Returns:
        list: [row, col] index in the decomposed grid.
    """
    # Calculate distances to all valid coordinates
    distances = np.linalg.norm(filtered_coords - position, axis=1)
    
    # Find the closest valid coordinate
    closest_index = np.argmin(distances)
    closest = filtered_coords[closest_index]
    
    # Find the grid indices for this coordinate
    row_index, col_index = np.where(np.all(decomposed_coords == closest, axis=2))
    
    # Return as [row, col] list
    return [row_index[0], col_index[0]]


def main():
    """
    Main entry point for the BlueBoxFinder node.
    
    Initializes ROS2, creates the node, and runs the control loop.
    Handles graceful shutdown on SIGINT (Ctrl+C).
    """
    def signal_handler(sig, frame):
        """Handle SIGINT by stopping the robot and shutting down ROS2."""
        bbx.stop()
        rclpy.shutdown()

    # Initialize ROS2
    rclpy.init(args=None)
    bbx = BlueBoxFinder()

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run ROS2 spin in a separate thread to allow concurrent control
    thread = threading.Thread(target=rclpy.spin, args=(bbx,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            bbx.controller()
    except ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
