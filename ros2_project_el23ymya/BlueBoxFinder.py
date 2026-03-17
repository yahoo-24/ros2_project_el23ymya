import threading
import sys, time
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
from .ArtificialPotentialField import ArtificialPotentialField
from .GreedySearch import PathPlanner
import matplotlib.pyplot as plt

MAX_ANGULAR_VEL = 2.84
MAX_VEL = 0.20
AREA_THRESHOLD = 30_000

class BlueBoxFinder(Node):
    def __init__(self):
        super().__init__('BBF')
        
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)
        self.subscription = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.rate = self.create_rate(10)  # 10 Hz
        
        self.sensitivity = 20
        self.hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        self.hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])
        
        self.hsv_red_lower1 = np.array([180 - self.sensitivity, 100, 100])
        self.hsv_red_upper1 = np.array([179, 255, 255])
        self.hsv_red_upper2 = np.array([0 + self.sensitivity, 255, 255])
        self.hsv_red_lower2 = np.array([0, 100, 100])
        
        self.hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        self.hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        
        self.red_found = False
        self.green_found = False
        self.blue_found = False
        self.position = None
        self.arrived = False
        
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.position = np.array([x, y])  
        
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        self.yaw = math.degrees(math.atan2(z, w) * 2)
        if self.yaw < 0:
            self.yaw += 360

    def send_goal(self, x, y, yaw):
        self.get_logger().info(f'Moving to: [x: {x}, y: {y}, yaw: {yaw}]')
        self.arrived = False
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = np.sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = np.cos(yaw / 2)

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.arrived = True
        self.get_logger().info(f'Navigation result: {result}')

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # NOTE: if you want, you can use the feedback while the robot is moving.
        #       uncomment to suit your need.

        ## Access the current pose
        current_pose = feedback_msg.feedback.current_pose
        distance = np.linalg.norm(self.target - self.position)
        if distance < 0.2:
            self.arrived = True
        #orientation = current_pose.pose.orientation

        ## Access other feedback fields
        #navigation_time = feedback_msg.feedback.navigation_time
        #distance_remaining = feedback_msg.feedback.distance_remaining

        ## Print or process the feedback data
        #self.get_logger().info(f'Current Pose: [x: {position.x}, y: {position.y}, z: {position.z}]')
        #self.get_logger().info(f'Distance Remaining: {distance_remaining}')
        
    def image_callback(self, data):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.blue_mask = cv2.inRange(hsv_image, self.hsv_blue_lower, self.hsv_blue_upper)
                
        filtered_img = cv2.bitwise_and(image, image, mask=self.blue_mask)
        if np.any(self.blue_mask):
            area = self.find_area()
            print(area)
            if area > AREA_THRESHOLD:
                self.blue_found = True
        else:
            self.blue_found = False
        
        red_mask_1 = cv2.inRange(hsv_image, self.hsv_red_lower1, self.hsv_red_upper1)
        red_mask_2 = cv2.inRange(hsv_image, self.hsv_red_lower2, self.hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        red_filtered = cv2.bitwise_and(image, image, mask=red_mask)
        cv2.namedWindow('Red_Feed',cv2.WINDOW_NORMAL) 
        cv2.imshow('Red_Feed', red_filtered)
        cv2.resizeWindow('Red_Feed', 320, 240) 
        cv2.waitKey(3)
        if not self.red_found:
            self.red_found = np.any(red_mask)
            if self.red_found:
                self.get_logger().info("Red Box is found") 
        
        green_mask = cv2.inRange(hsv_image, self.hsv_green_lower, self.hsv_green_upper)
        green_filtered = cv2.bitwise_and(image, image, mask=green_mask)
        cv2.namedWindow('Green_Feed',cv2.WINDOW_NORMAL) 
        cv2.imshow('Green_Feed', green_filtered)
        cv2.resizeWindow('Green_Feed', 320, 240) 
        cv2.waitKey(3)
        if not self.green_found:
            self.green_found = np.any(green_mask)
            if self.green_found:
                self.get_logger().info("Green Box is found")
            
        cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL) 
        cv2.imshow('camera_Feed', filtered_img)
        cv2.resizeWindow('camera_Feed', 320, 240) 
        cv2.waitKey(3)
        # print(self.blue_found)
        
    # def move(self, goal):
    #     current_pos = self.position
    #     print(f"Goal Position is {goal}, Current Position is {self.position}")
    #     direction = goal - current_pos
        
    #     obstacles_row, obstacles_column = np.where(self.image == 0)
    #     obstacles_coord = self.coord[obstacles_row, obstacles_column, :]
    #     apf = ArtificialPotentialField(
    #         targets=goal,
    #         obstacles=obstacles_coord,
    #         att_gain=0.1,
    #         rep_gain=0.1,
    #         max_distance=0.5
    #     )
    #     apf_direction = apf.resultant_force(current_pos)
    #     print(f"APF Direction {apf_direction}")
        
    #     def adjust_angle(direction):
    #         print(f"Direction {direction}")
    #         angle = self.find_angle(direction)
            
    #         angle_error = angle - self.yaw
    #         if angle_error > 180:
    #             angle_error -= 360
    #         elif angle_error < -180:
    #             angle_error += 360
                
    #         print(f"Current Angle {self.yaw}, Goal Angle {angle}, Error {angle_error}")
            
    #         if abs(angle_error) > 5:
    #             while True:
    #                 angle_error = angle - self.yaw
    #                 if angle_error > 180:
    #                     angle_error -= 360
    #                 elif angle_error < -180:
    #                     angle_error += 360
                        
    #                 print(f"Current Angle {self.yaw}, Goal Angle {angle}, Error {angle_error}")
                        
    #                 if abs(angle_error) < 1:
    #                     print("Correct Angle Reached!")
    #                     self.stop()
    #                     break
    #                 else:
    #                     angle_control = MAX_ANGULAR_VEL * (angle_error / 180)
    #                     self.rotate(angle_control)
                        
    #     adjust_angle(direction)
    #     counter = 0
                    
    #     error = np.linalg.norm(goal - self.position)
    #     if error > 0.1:
    #         while True:
    #             print(f"Current Position {self.position}, Goal Position {goal}")
    #             error = np.linalg.norm(goal - self.position)
    #             control = error * 2
    #             control = np.clip(control, -0.2, 0.2)
    #             self.move_forward(float(control))
    #             if error < 0.1:
    #                 print("Correct Distance Reached!")
    #                 self.stop()
    #                 break
    #             elif error < 0.4 and counter > 8:
    #                 counter = 0
    #                 self.stop()
    #                 adjust_angle(goal - self.position)
    #             counter += 1
    #             if counter == 40:
    #                 self.stop()
    #                 adjust_angle(goal - self.position)
    #                 counter = 0
        
    # def find_angle(self, direction):
    #     angle = math.atan2(direction[1], direction[0])
    #     angle = math.degrees(angle)
    #     if angle < 0:
    #         angle += 360
    #     return angle
    
    def rotate(self, value):
        desired_rotation = Twist()
        desired_rotation.angular.z = value  # Rotate with (pi / 6) rad/s --> 30 deg/s
        
        self.publisher.publish(desired_rotation)
        self.rate.sleep()
        
    def rotate30(self):
        desired_rotation = Twist()
        desired_rotation.angular.z = np.pi / 3  # Rotate with (pi / 6) rad/s --> 30 deg/s
        
        for _ in range(5):  # Stop for a brief moment
            self.publisher.publish(desired_rotation)
            self.rate.sleep() # 10 * 0.1s = 1s --> theta = (pi / 6) * 1s = (pi / 6) radians
            
    def spin_360(self):
        for i in range(12):
            print(f"Spun to {30 * i} degrees")
            self.rotate30()
            if self.blue_found:
                print("Blue Box Found!!!")
                return True
        print("Box Not Found :(")
        return False
    
    def move_forward(self, error):
        desired_velocity = Twist()
        desired_velocity.linear.x = error
        for _ in range(3):  # Stop for a brief moment
            self.publisher.publish(desired_velocity)
            self.rate.sleep()
    
    def stop(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.0  # Send zero velocity to stop the robot
        desired_velocity.angular.z = 0.0  # Send zero velocity to stop the robot
        for _ in range(10):
            self.publisher.publish(desired_velocity)
            self.rate.sleep()

    def read_image(self):
        file = '/uolstore/home/users/el23ymya/ros2_ws/src/ros2_project_el23ymya/map/map.pgm'
        self.image = cv2.imread(file, 0)
        # print(type(self.image))
        self.image[self.image < 250] = 0
        self.image[self.image != 0] = 1
        # print(type(self.image))

    def generate_coordinates(self, x_end, x_start, y_start, y_end):
        shape = self.image.shape
        x_axis = np.linspace(x_start, x_end, shape[1]) - 3
        y_axis = np.linspace(y_start, y_end, shape[0])
        self.coord = np.zeros((shape[0], shape[1], 2))
        for index, y in enumerate(y_axis):
            self.coord[index] = (np.array([x_axis, np.ones(shape[1]) * y]).T)
            
    def expand_point(self, current_index, decomposed_image):
        max_index_x = decomposed_image.shape[1]
        max_index_y = decomposed_image.shape[0]
        
        if current_index not in self.explored and current_index in self.unexplored:  
            self.explored.append(current_index)
            self.unexplored.remove(current_index)
        
        # nodes_to_expand = []
        # x, y = np.where(nodes == current)
        # current_index = np.array([x, y])
        
        # up = current_index[1] + 1
        # down = current_index[1] - 1
        # left = current_index[0] - 1
        # right = current_index[0] + 1
            
        # for i in range(current_index[0] - 1, current_index[0] + 2):
        #     for j in range(current_index[1] - 1, current_index[1] + 2):
        #         node = np.array([i, j])
        #         if validate_index(node, max_index_x, max_index_y):
        #             free = decomposed_image[node[0], node[1]]
        #             print(free, node)
        #             node = node.tolist()
        #             if free.any() and node not in self.explored:
        #                 # print(node.tolist())
        #                 print(self.unexplored)
        #                 print("Node Added!")
        #                 self.explored.append(node)
        #                 self.unexplored.remove(node)
                        
        # Left
        idx = 0
        stop = False
        while not stop:
            for i in range(current_index[1] - 1 - idx, current_index[1] + 2 + idx):
                node = np.array([current_index[0] - idx, i])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    node = node.tolist()
                    if free.any() and node not in self.explored:
                        # print(node.tolist())
                        #print(self.unexplored)
                        print("Node Added!")
                        self.explored.append(node)
                        self.unexplored.remove(node)
                    elif not free.any():
                        stop = True
            idx += 1
            
        # Right
        idx = 0
        stop = False
        while not stop:
            for i in range(current_index[1] - 1 - idx, current_index[1] + 2 + idx):
                node = np.array([current_index[0] + idx, i])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    node = node.tolist()
                    if free.any() and node not in self.explored:
                        # print(node.tolist())
                        #print(self.unexplored)
                        print("Node Added!")
                        self.explored.append(node)
                        self.unexplored.remove(node)
                    elif not free.any():
                        stop = True
            idx += 1
            
        # Up
        idx = 0
        stop = False
        while not stop:
            for i in range(current_index[0] - 1 - idx, current_index[0] + 2 + idx):
                node = np.array([i, current_index[1] - idx])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    node = node.tolist()
                    if free.any() and node not in self.explored:
                        # print(node.tolist())
                        #print(self.unexplored)
                        print("Node Added!")
                        self.explored.append(node)
                        self.unexplored.remove(node)
                    elif not free.any():
                        stop = True
            idx += 1
            
        # Down
        idx = 0
        stop = False
        while not stop:
            for i in range(current_index[0] - 1 - idx, current_index[0] + 2 + idx):
                node = np.array([i, current_index[1] + idx])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    node = node.tolist()
                    if free.any() and node not in self.explored:
                        # print(node.tolist())
                        #print(self.unexplored)
                        print("Node Added!")
                        self.explored.append(node)
                        self.unexplored.remove(node)
                    elif not free.any():
                        stop = True
            idx += 1
       
    def find_centre(self):
        contours, _ = cv2.findContours(self.blue_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        cx = 0

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)

            #Moments can calculate the center of the contour
            M = cv2.moments(c)
            cx, _ = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            
        return cx
    
    def find_area(self):
        contours, _ = cv2.findContours(self.blue_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        area = -1

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(c)
            
        return area
       
    def centre_blue_box(self):
        cx = self.find_centre()
        TARGET_CENTRE = 478
        while cx < TARGET_CENTRE - 8 or cx > TARGET_CENTRE + 8:
            # The speed of rotation is dependent on the error. An addition division by 5 is added to reduce speed
            error = ((TARGET_CENTRE - cx) / TARGET_CENTRE * MAX_ANGULAR_VEL) / 5
            self.rotate(error)
            print(f"Rotating: {error}")
            cx = self.find_centre()
            
    def move_towards_box(self):
        self.centre_blue_box()
        area = self.find_area()
        print(area)
        start_counter = 0
        readjust_counter = 0
        GRADUAL_ACC_THRESHOLD = 15
        TARGET_AREA = 390_000 # The area where we are 1 m away from the box
        READJUST_THRESHOLD = 50
        while area < TARGET_AREA - 5_000 or area > TARGET_AREA + 5_000:
            error = (TARGET_AREA - area) / TARGET_AREA * MAX_VEL
            if start_counter < GRADUAL_ACC_THRESHOLD:
                # To avoid a large and instantaneous acceleration, a counter is added which limits the speed
                start_counter += 1
                error = min(error, MAX_VEL * start_counter / GRADUAL_ACC_THRESHOLD) # This is meant for a more gradual change in speed
                
            self.move_forward(error) # Move towards the box to reduce the error
            print(f"Moving: {error}, Current Area = {area}")
            area = self.find_area() # Recalculate the area
            readjust_counter += 1
            if readjust_counter >= READJUST_THRESHOLD:
                # Occasionally readjust such that the blue box is in the centre
                self.centre_blue_box()
                    
    def controller(self):
        while self.position is None:
            print("Waiting For Pose...")
            time.sleep(1)
        self.explored = []
        print("Reading Image...")
        self.read_image()
        
        print("Generating Coordinates...")
        self.generate_coordinates(9.93, -12.2, 6.39, -15.8)
        
        print("Decomposing Image...")
        decomposed_image = min_pool(self.image, 27)
        # x_axis = np.linspace(0, decomposed_image.shape[1] - 1, decomposed_image.shape[1])
        # y_axis = np.linspace(0, decomposed_image.shape[0] - 1, decomposed_image.shape[0])
        # self.unexplored = np.zeros((decomposed_image.shape[0], decomposed_image.shape[1], 2))
        # for index, y in enumerate(y_axis):
        #     self.unexplored[index] = (np.array([np.ones(decomposed_image.shape[1]) * y, x_axis]).T)
        # self.unexplored = self.unexplored.reshape(-1, 2).tolist()
        # self.unexplored = self.unexplored.astype(np.int32)
        self.unexplored = np.argwhere(decomposed_image).tolist()
        
        print("Decomposing Coordinates...")
        decomposed_coord = redefine_values(self.coord, 27, 2)
        
        print("Finding Current Position...")
        # print(f"{decomposed_coord.shape}, {self.position}")
        current_index = find_current_node(self.position, decomposed_coord[decomposed_image > 0], decomposed_coord)
        
        print(f"Current position is {decomposed_coord[current_index[0], current_index[1], :]}, Index {current_index}")
        blue_found = False
        operation_complete = False
        while not operation_complete:
            blue_found = self.spin_360()
            self.stop()
            if blue_found or self.blue_found:
                self.move_towards_box()
                self.stop()
                operation_complete = True
            else:
                print("Expanding Node...")
                self.expand_point(current_index, decomposed_image)
                
                print("Choosing Next Node...")
                next_node_index = find_furthest_node(current_index, self.unexplored)
                print(f"Node Chosen is {decomposed_coord[next_node_index[0], next_node_index[1]]}, Index {next_node_index}")
                # next_node = decomposed_coord[next_node_index[0], next_node_index[1]]
                
                planner = PathPlanner(current_index, next_node_index, decomposed_image)
                path = planner.plan()
                start = path.pop(0)
                for node_index in path:
                    plt.imshow(decomposed_image, cmap='gray')
                    path_print = np.array(path)
                    un_print = np.array(self.explored)
                    plt.scatter(un_print[:, 1], un_print[:, 0])
                    plt.scatter(path_print[:, 1], path_print[:, 0], marker='X')
                    plt.scatter(start[1], start[0], marker='v')
                    plt.scatter(path_print[-1, 1], path_print[-1, 0], marker='^')
                    plt.show(block=False)
                    plt.pause(2)
                    
                    next_node = decomposed_coord[node_index[0], node_index[1]]
                # path.pop(0) # Remove the start node
                # for next_index in path:
                #     next_node = decomposed_coord[next_index[0], next_index[1]]
                #     print(f"Going To Node {next_node}, Index {next_index}")
                #     self.move(next_node)
                
                    self.target = next_node
                    self.send_goal(float(next_node[0]), float(next_node[1]), 0.0)  # example coordinates
                    while not self.arrived:
                        time.sleep(1)
                        
                    self.spin_360()
                    self.expand_point(node_index, decomposed_image)
                    if self.blue_found:
                        break
                    current_index = node_index
                # current_index = next_node_index
        while True:
                time.sleep(10)
    
    
def min_pool(frame, kernel_size):
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    result_size_col = np.ceil(frame.shape[1] / kernel_size)
    result_size_row = np.ceil(frame.shape[0] / kernel_size)
    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32)))
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
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    centre = (kernel_size + 1) / 2
    
    result_size_col = np.ceil(frame.shape[1] / kernel_size).astype(np.int32)
    result_size_row = np.ceil(frame.shape[0] / kernel_size).astype(np.int32)

    rem_col = frame.shape[1] % kernel_size
    rem_row = frame.shape[0] % kernel_size
    if rem_col != 0:
        rem_centre_col = np.floor(rem_col / 2)
    else:
        rem_centre_col = centre
    if rem_row != 0:
        rem_centre_row = np.floor(rem_row / 2)
    else:
        rem_centre_row = centre

    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32), int(dimensions)))
    row_index = 0
    while row_index < result_size_row:
        if row_index == result_size_row - 1:
            row_centre = row_index * kernel_size + rem_centre_row
        else:
            row_centre = row_index * kernel_size + centre
        col_index = 0

        while col_index < result_size_col:
            if col_index == result_size_col - 1:
                col_centre = col_index * kernel_size + rem_centre_col
            else:
                col_centre = col_index * kernel_size + centre
            result_arr[row_index, col_index] = frame[int(row_centre), int(col_centre)]
            col_index += 1
        row_index += 1

    return result_arr

def find_furthest_node(current, nodes):
    distances = np.linalg.norm(np.array(nodes) - np.array(current), axis=1)
    node = nodes[np.argmax(distances)]
    return node

def validate_index(index, max_x, max_y):
    if index[0] < 0 or index[0] > max_x or index[1] < 0 or index[1] > max_y:
        return False
    return True

    
    # if validate_index(up, max_index_x, max_index_y):
    #     free = image[up]
    #     if free and up not in explored:
    #         nodes_to_expand.append(up)
            
    # if validate_index(down, max_index_x, max_index_y):
    #     free = image[down]
    #     if free and down not in explored:
    #         nodes_to_expand.append(down)
            
    # if validate_index(right, max_index_x, max_index_y):
    #     free = image[right]
    #     if free and right not in explored:
    #         nodes_to_expand.append(right)
            
    # if validate_index(left, max_index_x, max_index_y):
    #     free = image[left]
    #     if free and left not in explored:
    #         nodes_to_expand.append(left)
            
def find_current_node(position, filtered_coords, decomposed_coords):
    distances = np.linalg.norm(filtered_coords - position, axis=1) # Find the distances
    # print(distances)
    # print(distances.shape)
    closest_index = np.argmin(distances) # Find the index of the shortest distance
    closest = filtered_coords[closest_index] # Use the index to find the closest point to the current position
    row_index, col_index = np.where(np.all(decomposed_coords == closest, axis=2)) # Find the row and column in the coordinates array
    # print(row_index, col_index)
    return [row_index[0], col_index[0]] # The index of 0 avoids a 2D list of [[row], [col]] and instead gives [row, col]

def main():
    def signal_handler(sig, frame):
        bbx.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    bbx = BlueBoxFinder()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(bbx,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            bbx.controller()
    except ROSInterruptException:
        pass


if __name__ == "__main__":
    main()