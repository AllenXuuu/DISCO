import numpy as np
import json
import pickle as pkl
import os
from utils import utils
from collections import defaultdict
import threading
from env.thor_env import ThorEnv
import gen.constants as constants
import argparse


if __name__ == '__main__':
    all_scene_numbers = sorted(constants.TRAIN_SCENE_NUMBERS + constants.TEST_SCENE_NUMBERS)
    errors = []
    for scene_num in all_scene_numbers:
        scene_name = ('FloorPlan%d') % scene_num
        official_nav_pts = np.load(f'./gen/layouts/FloorPlan{scene_num}-layout.npy')
        our_gen_path = f'./data/scene_parse/nav/{scene_name}.pkl'
        if os.path.exists(our_gen_path):
            our_gen_pts = pkl.load(open(our_gen_path,'rb'))
            print(f"scene {scene_name}. check_valid {len(our_gen_pts)}. AlfredRelease {len(official_nav_pts)}")
        else:
            print(f'scene {scene_name}. No File {our_gen_path}')
            errors.append(scene_num)
            
    print(f'errors {errors}')
