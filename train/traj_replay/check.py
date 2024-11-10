from utils import arguments, utils
import os
import torch.multiprocessing as mp
import importlib
from env import thor_env
import time
from utils import image_util
import cv2
import pickle as pkl
import shutil
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
import numpy as np

def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)
    args.remove_repeat = True

    tasks = utils.load_alfred_tasks(args)
    for ii,task_info in enumerate(tasks):
        root = os.path.join('./data/replay', args.split, task_info['task'])
        rgb_root = os.path.join(root,'rgb')
        depth_root = os.path.join(root,'depth')
        seg_root = os.path.join(root,'seg')
        aff_int_root = os.path.join(root,'interaction')
        aff_nav_root = os.path.join(root,'navigation')
        
        expert_actions = [a['discrete_action'] for a in task_info['plan']['low_actions'] ]
        
        ok = True
        if not os.path.exists(rgb_root) or len(os.listdir(rgb_root)) != len(expert_actions) + 1:
            ok = False
        rgb_count = len(os.listdir(rgb_root)) if os.path.exists(rgb_root) else 0
            
        if not os.path.exists(depth_root) or len(os.listdir(depth_root)) != len(expert_actions) + 1:
            ok = False
        dep_count = len(os.listdir(depth_root)) if os.path.exists(depth_root) else 0

        if not os.path.exists(seg_root) or len(os.listdir(seg_root)) != len(expert_actions) + 1:
            ok = False
        seg_count = len(os.listdir(seg_root)) if os.path.exists(seg_root) else 0

        if not os.path.exists(aff_int_root) or len(os.listdir(aff_int_root)) != len(expert_actions) + 1:
            ok = False
        aff_int_count = len(os.listdir(aff_int_root)) if os.path.exists(aff_int_root) else 0
    
        if not os.path.exists(aff_nav_root) or len(os.listdir(aff_nav_root)) != len(expert_actions) + 1:
            ok = False
        aff_nav_count = len(os.listdir(aff_nav_root)) if os.path.exists(aff_nav_root) else 0
            
        task_index = task_info['task_index']   
        task_name = task_info['task']
        
        
        if not ok:
            print(f'{utils.timestr()} Rank {args.rank} {task_index}. {task_name}. Not OK!. {rgb_count} {dep_count} {seg_count} {aff_int_count} {aff_nav_count} / {len(expert_actions) + 1}')

        

            # cmd = f'scp -r -P 41300 rhos@79g70f0614.goho.co:/Disk4/xxy/VLNIDIFF/data/replay/valid_seen/{task_name} data/replay/valid_seen/{task_name}'
            # print(cmd)
        
if __name__ == '__main__':
    main()