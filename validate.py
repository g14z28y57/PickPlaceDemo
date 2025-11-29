import json
import os
import random
import torch
import cv2
import numpy as np
from util import normalize_vector
from scene import SimScene
from model import VisionActionModel


def normalize_image(img, device):
    img = np.transpose(img, [2, 0, 1])
    img = img / 255.0 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32, device=device)
    return img


@torch.inference_mode()
def inference(episode_save_dir, scene, model, device, action_min, action_max):
    action_min = torch.tensor(action_min, dtype=torch.float32, device=device)
    action_max = torch.tensor(action_max, dtype=torch.float32, device=device)

    os.makedirs(os.path.join(episode_save_dir, "camera_1"), exist_ok=True)
    os.makedirs(os.path.join(episode_save_dir, "camera_2"), exist_ok=True)
    robot_arm_speed = 0.1
    robot_arm_state_record = []

    frame_count = 0
    catch_state = 0  # 1 for catching, 0 for not catching
    task_state = 0  # 1 for complete, 0 for not complete

    while frame_count < 400:
        state = list(scene.robot_arm.location)
        state.append(catch_state)
        state.append(task_state)
        state = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, -1)

        save_path_1 = os.path.join(episode_save_dir, "camera_1", f"{frame_count}.png")
        scene.shot_1(save_path_1)
        img1 = cv2.imread(save_path_1)
        img1 = normalize_image(img1, device)
        img1 = torch.unsqueeze(img1, dim=0)

        save_path_2 = os.path.join(episode_save_dir, "camera_2", f"{frame_count}.png")
        scene.shot_2(save_path_2)
        img2 = cv2.imread(save_path_2)
        img2 = normalize_image(img2, device)
        img2 = torch.unsqueeze(img2, dim=0)

        action = model(img1, img2, state)
        action = (action * 0.5 + 0.5) * (action_max - action_min) + action_min
        action = action.squeeze(0).cpu().numpy()
        robot_arm_position_delta = action[:3].tolist()

        pos_delta = robot_arm_speed * normalize_vector(robot_arm_position_delta)
        scene.move_robot_arm(dx=pos_delta[0], dy=pos_delta[1], dz=pos_delta[2])
        if catch_state == 1:
            scene.move_object(dx=pos_delta[0], dy=pos_delta[1], dz=pos_delta[2])
        frame_count += 1

        catch_state = round(action[3].item())
        task_state = round(action[4].item())

    data = {
        "robot_arm_state": robot_arm_state_record,
        "object_init_x": scene.object_init_x,
        "object_init_y": scene.object_init_y
    }

    state_save_path = os.path.join(episode_save_dir, "robot_state.json")
    with open(state_save_path, "w") as f:
        json.dump(data, f)


def main():
    stat_path = "stat.npz"
    stat = np.load(stat_path, allow_pickle=True)
    state_min = stat["state_min"]
    state_max = stat["state_max"]
    action_min = stat["action_min"]
    action_max = stat["action_max"]
    device = "cuda"
    model = VisionActionModel().to(device)
    model.eval()
    init_object_x = random.uniform(-4, 4)
    init_object_y = random.uniform(-4, 4)
    scene = SimScene(object_init_x=init_object_x, object_init_y=init_object_y)
    episode_save_dir = os.path.join(os.path.dirname(__file__), "real_time_test")
    inference(episode_save_dir, scene, model, device, state_min, state_max, action_min, action_max)


if __name__ == "__main__":
    main()
