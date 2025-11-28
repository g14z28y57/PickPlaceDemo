import numpy as np
from util import normalize_vector, compute_distance
FPS = 30


def get_action(robot_arm_catch_state,  # 1 for catching target, 0 for not catching target
               robot_arm_position,
               robot_arm_height,
               object_position,
               object_height,
               bin_position,
               speed):
    target_position = object_position.copy()
    target_position[2] += object_height + 0.5 * robot_arm_height

    if not robot_arm_catch_state and compute_distance(target_position, robot_arm_position) < 0.1:
        robot_arm_catch_state = True

    if robot_arm_catch_state:
        target_position = bin_position.copy()
        target_position[2] += 0.5 * robot_arm_height

    robot_arm_action = speed * normalize_vector(target_position - robot_arm_position)
    return robot_arm_catch_state, robot_arm_action
