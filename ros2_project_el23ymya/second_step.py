# Exercise 2 - Display an image of the camera feed to the screen

#from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
import rclpy.subscription
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.callback, 10)
        
        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        # We covered which topic to subscribe to should you wish to receive image data

        self.subscription  # prevent unused variable warning
        self.sensitivity = 20
        self.hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        self.hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        self.hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        self.hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])
        self.hsv_red_lower1 = np.array([180 - self.sensitivity, 100, 100])
        self.hsv_red_upper1 = np.array([179, 255, 255])
        self.hsv_red_upper2 = np.array([0 + self.sensitivity, 255, 255])
        self.hsv_red_lower2 = np.array([0, 100, 100])
        
    def callback(self, data):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL) 
        cv2.imshow('camera_Feed', image)
        cv2.resizeWindow('camera_Feed', 320, 240) 
        cv2.waitKey(3)
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_image, self.hsv_green_lower, self.hsv_green_upper)
        blue_mask = cv2.inRange(hsv_image, self.hsv_blue_lower, self.hsv_blue_upper)
        red_mask1 = cv2.inRange(hsv_image, self.hsv_red_lower1, self.hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.hsv_red_lower2, self.hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        rg_mask = cv2.bitwise_or(red_mask, blue_mask)
        rgb_mask = cv2.bitwise_or(green_mask, rg_mask)
        
        filtered_img = cv2.bitwise_and(image, image, mask=rgb_mask)
        cv2.namedWindow('filtered_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('filtered_feed', filtered_img)
        cv2.resizeWindow('filtered_feed', 320, 240)
        cv2.waitKey(3)
        
        green_img = cv2.bitwise_and(image, image, mask=green_mask)
        cv2.namedWindow('green_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('green_feed', green_img)
        cv2.resizeWindow('green_feed', 320, 240)
        cv2.waitKey(3)
        
        blue_img = cv2.bitwise_and(image, image, mask=blue_mask)
        cv2.namedWindow('blue_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('blue_feed', blue_img)
        cv2.resizeWindow('blue_feed', 320, 240)
        cv2.waitKey(3)
        
        red_img = cv2.bitwise_and(image, image, mask=red_mask)
        cv2.namedWindow('red_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('red_feed', red_img)
        cv2.resizeWindow('red_feed', 320, 240)
        cv2.waitKey(3)

        
        return
        # Convert the received image into a opencv image
        # But remember that you should always wrap a call to this conversion method in an exception handler
        # Show the resultant images you have created.
        

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():

    def signal_handler(sig, frame):
        rclpy.shutdown()
    # Instantiate your class
    # And rclpy.init the entire node
    rclpy.init(args=None)
    cI = colourIdentifier()


    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()
    

# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
