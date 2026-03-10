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

def BlueBoxFinder(Node):
    def __init__(self):
        super.__init__('BBF')
        
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

def validate_index(index, max):
    if index < 0 or index > max:
        return False
    return True

def expand_point(current_index, image, explored):
    max_index_x = image.shape[0]
    max_index_y = image.shape[1]
    
    nodes_to_expand = []
    # x, y = np.where(nodes == current)
    # current_index = np.array([x, y])
    
    up = current_index[1] + 1
    down = current_index[1] - 1
    left = current_index[0] - 1
    right = current_index[0] + 1
    
    if validate_index(up, max_index_y):
        free = image[up]
        if free and up not in explored:
            nodes_to_expand.append(up)
            
    if validate_index(down, max_index_y):
        free = image[down]
        if free and down not in explored:
            nodes_to_expand.append(down)
            
    if validate_index(right, max_index_x):
        free = image[right]
        if free and right not in explored:
            nodes_to_expand.append(right)
            
    if validate_index(left, max_index_x):
        free = image[left]
        if free and left not in explored:
            nodes_to_expand.append(left)
            
    