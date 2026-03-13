import numpy as np

class ArtificialPotentialField():
    def __init__(self, targets, obstacles, att_gain, rep_gain, max_distance):
        """
        Docstring for __init__
        
        :param targets: A NumPy nx3 array of n targets with their x, y and z positions
        :param obstacles: A NumPy mx3 array of m obstacles with their x, y and z positions
        :param att_gain: A multiplier increasing the strength of attraction
        :param rep_gain: A multiplier increasing the strength of repulsion
        :param max_distance: The maximum distance from an object where repulsion force can act
        """
        self.targets = targets
        self._update_len_targets()
        self.obstacles = obstacles
        self._update_len_obstacles()
        if obstacles is not None:
            self.len_obstacles = len(obstacles)
        else:
            self.len_obstacles = 1
        self.att_gain = abs(att_gain)
        self.rep_gain = abs(rep_gain)
        self.maximum_distance = abs(max_distance)

    def _update_len_targets(self):
        if self.targets is not None:
            self.len_targets = len(self.targets)
        else:
            self.len_targets = 1

    def _update_len_obstacles(self):
        if self.obstacles is not None:
            self.len_obstacles = len(self.obstacles)
        else:
            self.len_obstacles = 1

    def attraction(self, position):
        if self.targets is None:
            return 0
        differences = self.targets - position # nx3 np array - 1x3 np array --> nx3 np array
        att_force = self.att_gain * differences
        att_force = np.sum(att_force, axis=0)

        return att_force

    def repulsion(self, position):
        if self.obstacles is None:
            return 0
        differences = position - self.obstacles # (1x3 numpy array - mx3 numpy array) --> mx3 numpy array
        distances = np.linalg.norm(differences, axis=1) # 1xm numpy array

        remain = distances < self.maximum_distance
        differences = differences[remain]
        distances = distances[remain]
        unit_vector_diff = (differences.T / (distances + 1e-12)).T # Convert the vectors into a unit vector
        d = (1 / (distances + 1e-12)) - (1 / (self.maximum_distance + 1e-12))
        rep_force = self.rep_gain * (unit_vector_diff.T * d).T
        rep_force = np.sum(rep_force, axis=0)

        return rep_force

    def resultant_force(self, position, targets=None, obstacles=None):
        if targets is not None:
            self.targets = targets
        if obstacles is not None:
            self.obstacles = obstacles
        attraction_force = self.attraction(position)
        repulsion_force = self.repulsion(position)

        self._update_len_obstacles()
        self._update_len_targets()

        return attraction_force / (self.len_targets + 1e-12) + repulsion_force / (self.len_obstacles + 1e-12)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    targets_x = np.linspace(1, 10, 10) / 10
    targets_y = targets_x
    targets_z = np.zeros(10)
    targets = np.array([targets_x, targets_y, targets_z]).T
    obstacles = targets.copy()
    obstacles[:, 2] -= 0.05

    #print(targets)
    #print(obstacles)

    model = ArtificialPotentialField(
        targets=targets,
        obstacles=obstacles,
        att_gain=0.01,
        rep_gain=0.01,
        max_distance=0.15
    )

    position = np.array([0.0, 0.0, 0.0])
    # force = model.resultant_force(position)
    # position += force
    # print(f"Force is in the direction: {force} \nNew position is now at {position}")

    # force = model.resultant_force(position)
    # position += force
    # print(f"Force is in the direction: {force} \nNew position is now at {position}")

    # force = model.resultant_force(position)
    # position += force
    # print(f"Force is in the direction: {force} \nNew position is now at {position}")

    # force = model.resultant_force(position)
    # position += force
    # print(f"Force is in the direction: {force} \nNew position is now at {position}")

    previous_position = np.array([-1, -1, -1])
    counter = 1
    while np.linalg.norm(np.abs(position - previous_position)) > 0.001:
        previous_position = position.copy()
        force = model.resultant_force(position)
        position += force
        print(f"At time {counter}:\n Force is in the direction: {force} \nNew position is now at {position}\n")
        counter += 1

