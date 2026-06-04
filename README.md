# ros2-project
A lab project combines what has been taught in this module, to complete a task 
The map contains a red, green and blue box. The task is to locate the blue box and stop roughly one metre away from it.

The algorithm takes the map and performs Cell Decomposition marking cells where there are any obstacles as occupied.

The algorithm is designed to explore the reachable map systematically by selecting frontier-like positions (furthest points or cells from current position) and repeatedly scanning the environment. This ensures that all accessible regions are eventually observed.

The robot performs periodic 360-degree scans, allowing detection from multiple viewpoints as exploration progresses. The termination condition is based on detecting the target object and stopping 1 metre away from it.

The exploration process does not repeatedly visit the same regions indefinitely, as the planner progresses through previously unobserved areas of the map. This ensures that the robot does not get stuck in loops and will eventually cover all reachable free space.

Once the blue box is found, the robot adjusts its orientation using proportional control to keep the blue box centred. The robot then moves towards the blue box using proportional control periodically stopping to correct its orientation again.
# ros2_project_el23ymya
[![Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-blue)](https://github.com/yahoo-24/ros2_project_el23ymya/releases/download/v1.0/Video.mp4)
