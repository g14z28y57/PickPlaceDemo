import numpy as np
from tqdm import trange
from util import read_json, save_json

img1_list = []
img2_list = []
state_list = []
action_list = []
catch_state_list = []
task_state_list = []

for episode_index in trange(400):
    state_path = f"episodes/episode_{episode_index}/robot_state.json"
    states = read_json(state_path)["robot_arm_state"]
    num_frames = len(states)
    for frame_index in range(num_frames - 1):
        img1_path = f"episodes/episode_{episode_index}/camera_1/{frame_index}.png"
        img2_path = f"episodes/episode_{episode_index}/camera_2/{frame_index}.png"
        img1_list.append(img1_path)
        img2_list.append(img2_path)
        state = states[frame_index]
        state_list.append(state)
        next_state = states[frame_index + 1]
        action = [next_state[i] - state[i] for i in range(3)]
        action_list.append(action)
        catch_state = next_state[3]
        task_state = next_state[4]
        catch_state_list.append(catch_state)
        task_state_list.append(task_state)

data = {
    "img1": img1_list,
    "img2": img2_list,
    "state": state_list,
    "action": action_list,
    "catch_state": catch_state_list,
    "task_state": task_state_list
}

save_json(data, "train_data.json")
