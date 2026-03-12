import numpy as np
import matplotlib.pyplot as plt
import cv2

class PathPlanner():
    def __init__(self, start, goal, image):
        self.explored = []
        self.nodes = []
        self.to_expand = [start]
        self.parent_nodes = [None]
        self.goal = goal
        self.image = image
        self.connections = dict()

    def validate(self, node):
        if node[0] > (self.image.shape[0] - 1) or node[0] < 0 or node[1] > (self.image.shape[1] - 1) or node[1] < 0:
            return False
        return True

    def expand(self, current):
        self.explored.append(current)
        if current == self.goal:
            return True
        
        for row in range(current[0] - 1, current[0] + 2):
            for column in range(current[1] - 1, current[1] + 2):
                node = np.array([row, column]).tolist()
                if not self.validate(node):
                    continue
                if self.image[row, column] != 0 and node not in self.explored:
                    self.to_expand.append(node)
                    self.parent_nodes.append(current)

        return False

    def select_next_node(self):
        nodes = np.array(self.to_expand)
        distances = np.linalg.norm(nodes - self.goal, axis=1)
        index = np.argmin(distances)
        next_node, parent_node = self.to_expand.pop(index), self.parent_nodes.pop(index)
        return next_node, parent_node
    
    def backtrack(self):
        path = []
        current_node = self.goal
        while current_node is not None:
            key = current_node[0] * self.image.shape[1] + current_node[1]
            path.append(current_node)
            current_node = self.connections[key]

        return path[::-1]
    
    def plan(self):        
        goal_found = False
        while not goal_found:
            if len(self.to_expand) == 0:
                return None
        
            next_node, parent_node = self.select_next_node()
            if next_node in self.explored:
                continue

            key = next_node[0] * self.image.shape[1] + next_node[1]
            self.connections[key] = parent_node

            goal_found = self.expand(next_node)

        path = self.backtrack()
        print(path)
        return path

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

def read_image():
        file = 'map.pgm'
        image = cv2.imread(file, 0)
        image[image < 250] = 0
        image[image != 0] = 1
        decomposed_image = min_pool(image, 25)
        return decomposed_image


if __name__ == "__main__":
    image = read_image()
    planner = PathPlanner(
        start=[2, 12],
        goal=[11, 7],
        image=image
    )
    path = planner.plan()

    plt.imshow(image, cmap='gray')
    if path is not None:
        path = np.array(path)
        plt.scatter(path[:, 1], path[:, 0])
    plt.show()