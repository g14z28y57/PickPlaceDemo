import os
import random
from scene import SimScene


def main():
    init_object_x = random.uniform(-2, 2)
    init_object_y = random.uniform(-2, 2)
    scene = SimScene(object_init_x=init_object_x, object_init_y=init_object_y)
    save_path = os.path.join(os.path.dirname(__file__), "ScreenShot1.png")
    scene.shot_1(save_path)
    scene.move_robot_arm(dx=3, dy=0, dz=1)
    save_path = os.path.join(os.path.dirname(__file__), "ScreenShot2.png")
    scene.shot_1(save_path)


if __name__ == "__main__":
    main()
