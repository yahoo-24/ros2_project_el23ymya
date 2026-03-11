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
        self.position = None
        
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.position = np.array([x, y])  

    # def send_goal(self, x, y, yaw):
    #     self.get_logger().info(f'Moving to: [x: {x}, y: {y}, yaw: {yaw}]')
    #     self.arrived = False
        
    #     goal_msg = NavigateToPose.Goal()
    #     goal_msg.pose.header.frame_id = 'map'
    #     goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

    #     # Position
    #     goal_msg.pose.pose.position.x = x
    #     goal_msg.pose.pose.position.y = y

    #     # Orientation
    #     goal_msg.pose.pose.orientation.z = np.sin(yaw / 2)
    #     goal_msg.pose.pose.orientation.w = np.cos(yaw / 2)

    #     self.action_client.wait_for_server()
    #     self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
    #     self.send_goal_future.add_done_callback(self.goal_response_callback)

    # def goal_response_callback(self, future):
    #     goal_handle = future.result()
    #     if not goal_handle.accepted:
    #         self.get_logger().info('Goal rejected')
    #         return

    #     self.get_logger().info('Goal accepted')
    #     self.get_result_future = goal_handle.get_result_async()
    #     self.get_result_future.add_done_callback(self.get_result_callback)

    # def get_result_callback(self, future):
    #     result = future.result().result
    #     self.get_logger().info(f'Navigation result: {result}')

    # # def feedback_callback(self, feedback_msg):
    #     feedback = feedback_msg.feedback
    #     # NOTE: if you want, you can use the feedback while the robot is moving.
    #     #       uncomment to suit your need.

    #     ## Access the current pose
    #     current_pose = feedback_msg.feedback.current_pose
    #     self.position = current_pose.pose.position
    #     distance = np.linalg.norm(self.target - self.position)
    #     if distance < 0.2:
    #         self.arrived = True
    #     #orientation = current_pose.pose.orientation

    #     ## Access other feedback fields
    #     #navigation_time = feedback_msg.feedback.navigation_time
    #     #distance_remaining = feedback_msg.feedback.distance_remaining

    #     ## Print or process the feedback data
    #     #self.get_logger().info(f'Current Pose: [x: {position.x}, y: {position.y}, z: {position.z}]')
    #     #self.get_logger().info(f'Distance Remaining: {distance_remaining}')
        
    def image_callback(self, data):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.hsv_blue_lower, self.hsv_blue_upper)
        # print(blue_mask)
                
        filtered_img = cv2.bitwise_and(image, image, mask=blue_mask)
        self.blue_found = np.any(blue_mask)
        cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL) 
        cv2.imshow('camera_Feed', filtered_img)
        cv2.resizeWindow('camera_Feed', 320, 240) 
        cv2.waitKey(3)
        # print(self.blue_found)
        
    def move(self, x, y, yaw):
        current_x = self.position[0]
        current_y = self.position[1]
        
    def rotate30(self):
        desired_rotation = Twist()
        desired_rotation.angular.z = np.pi / 6  # Rotate with (pi / 6) rad/s --> 30 deg/s
        
        for _ in range(10):  # Stop for a brief moment
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
    
    def stop(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.0  # Send zero velocity to stop the robot
        self.publisher.publish(desired_velocity)

    def read_image(self):
        file = '/uolstore/home/users/el23ymya/ros2_ws/src/ros2_project_el23ymya/map/map.pgm'
        self.image = cv2.imread(file, 0)
        # print(type(self.image))
        self.image[self.image < 250] = 0
        self.image[self.image != 0] = 1
        # print(type(self.image))

    def generate_coordinates(self, x_start, x_end, y_start, y_end):
        shape = self.image.shape
        x_axis = np.linspace(x_start, x_end, shape[1])
        y_axis = np.linspace(y_start, y_end, shape[0])
        self.coord = np.zeros((shape[0], shape[1], 2))
        for index, y in enumerate(y_axis):
            self.coord[index] = (np.array([x_axis, np.ones(shape[1]) * y]).T)
            
    def expand_point(self, current_index, decomposed_image):
        max_index_x = decomposed_image.shape[1]
        max_index_y = decomposed_image.shape[0]
        
        self.explored.append(current_index.tolist())
        
        # nodes_to_expand = []
        # x, y = np.where(nodes == current)
        # current_index = np.array([x, y])
        
        # up = current_index[1] + 1
        # down = current_index[1] - 1
        # left = current_index[0] - 1
        # right = current_index[0] + 1
            
        for i in range(current_index[0] - 1, current_index[0] + 2):
            for j in range(current_index[1] - 1, current_index[1] + 2):
                node = np.array([i, j])
                if validate_index(node, max_index_x, max_index_y):
                    free = decomposed_image[node[0], node[1]]
                    print(free)
                    if free.any():
                        # print(node.tolist())
                        # print(self.unexplored)
                        self.explored.append(node)
                        self.unexplored.remove(node.tolist())
                    
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
        decomposed_image = min_pool(self.image, 25)
        x_axis = np.linspace(0, decomposed_image.shape[1] - 1, decomposed_image.shape[1])
        y_axis = np.linspace(0, decomposed_image.shape[0] - 1, decomposed_image.shape[0])
        self.unexplored = np.zeros((decomposed_image.shape[0], decomposed_image.shape[1], 2))
        for index, y in enumerate(y_axis):
            self.unexplored[index] = (np.array([np.ones(decomposed_image.shape[1]) * y, x_axis]).T)
        self.unexplored = self.unexplored.reshape(-1, 2).tolist()
        
        print("Decomposing Coordinates...")
        decomposed_coord = redefine_values(self.coord, 25, 2)
        
        print("Finding Current Position...")
        # print(f"{decomposed_coord.shape}, {self.position}")
        current_index = find_current_node(self.position, decomposed_coord)
        
        print(f"Current position is {decomposed_coord[current_index[0], current_index[1], :]}")
        blue_found = False
        while not blue_found:
            blue_found = self.spin_360()
            self.stop()
            if blue_found:
                while True:
                    self.stop()
            else:
                print("Expanding Node...")
                self.expand_point(current_index, decomposed_image)
                
                print("Choosing Next Node...")
                next_node_index = find_nearest_node(current_index, self.unexplored)
                row_index = next_node_index // decomposed_coord.shape[1]
                col_index = next_node_index % decomposed_coord.shape[1]
                next_node_index = np.array([row_index, col_index])
            
                next_node = decomposed_coord[next_node_index[0], next_node_index[1]]
                
                print(f"Node Chosen is {next_node}")
                self.move(next_node[0], next_node[1], 0.0)
                current_index = next_node
                while not self.arrived:
                    time.sleep(1)
    
    
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

def find_nearest_node(current, nodes):
    distances = np.linalg.norm(nodes - current, axis=1)
    return np.argmin(distances)

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
            
def find_current_node(position, image):
    distances = np.linalg.norm(image - position, axis=2)
    # print(distances)
    # print(distances.shape)
    index_flat = np.argmin(distances)
    row_index = index_flat // image.shape[1]
    col_index = index_flat % image.shape[1]
    return np.array([row_index, col_index])

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