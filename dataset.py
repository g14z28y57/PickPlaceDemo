import cv2
import numpy as np
from torch.utils.data import Dataset
from util import read_json


class PickPlaceDataset(Dataset):
    def __init__(self, json_path, stat_path):
        super().__init__()
        data = read_json(json_path)

        self.img1 = data["img1"]
        self.img2 = data["img2"]

        self.state = np.array(data["state"], dtype=np.float32)

        self.action = np.array(data["action"], dtype=np.float32)
        self.action_min = np.min(self.action, axis=0)
        self.action_max = np.max(self.action, axis=0)
        self.normalize_action()

        self.catch_state = np.array(data["catch_state"], dtype=np.float32).reshape(-1, 1)
        self.task_state = np.array(data["task_state"], dtype=np.float32).reshape(-1, 1)
        self.length = self.state.shape[0]
        self.save_stat(stat_path)

    def normalize_action(self):
        action_min = self.action_min.reshape(1, -1)
        action_max = self.action_max.reshape(1, -1)
        self.action = (self.action - action_min) / (action_max - action_min) * 2 - 1

    def __len__(self):
        return self.length

    @staticmethod
    def normalize_image(img_path):
        img = cv2.imread(img_path)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.0 * 2 - 1
        return img.astype(np.float32)

    def save_stat(self, stat_path):
        print("action_min", self.action_min)
        print("action_max", self.action_max)
        np.savez(stat_path, action_min=self.action_min, action_max=self.action_max)

    def __getitem__(self, index):
        return (self.normalize_image(self.img1[index]),
                self.normalize_image(self.img2[index]),
                self.state[index],
                self.action[index],
                self.catch_state[index],
                self.task_state[index])

