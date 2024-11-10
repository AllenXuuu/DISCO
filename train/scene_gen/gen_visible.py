import sys
import numpy as np
import json
import pickle as pkl
import os
from utils import utils
from collections import defaultdict
import threading
from env.thor_env import ThorEnv
import gen.constants as constants
from utils import arguments
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS


def reset(env,scene_name):
    env.reset(  
        scene_name,
        render_image=False,
        render_depth_image=False,
        render_class_image=False,
        render_object_image=True)


def main(args, rank):
    env = ThorEnv(x_display = args.x_display)
    while True:
        lock.acquire()
        if len(all_scene_numbers) >0:
            scene_num = all_scene_numbers.pop()
        else:
            scene_num = None
        lock.release()
        if scene_num is None:
            break
        else:
            scene_name = ('FloorPlan%d') % scene_num
            reset(env, scene_name)
        
        agent_height = env.last_event.metadata['agent']['position']['y']
        navigation_pt = pkl.load(open('./data/scene_parse/nav/FloorPlan%d.pkl' % scene_num,'rb'))
        id2type = {}
        for obj in env.last_event.metadata['objects']:
            id2type[obj['objectId']] = obj['objectType']
        
        visiblepose = {}
        visiblepose_save = f'./data/scene_parse/visible/{scene_name}.pkl'
        print(f'{utils.timestr()} Rank {rank} {scene_name} pts {len(navigation_pt)}')
        for _,(x,z) in enumerate(navigation_pt):
            for r in [0,90,180,270]:
                for h in [45,0]:
                    action = {
                        'action': 'TeleportFull',
                        'x': x,
                        'y': agent_height,
                        'z': z,
                        'rotateOnTeleport': True,
                        'rotation': r,
                        'horizon': h
                        }
                    event = env.step(action)
                    if not event.metadata['lastActionSuccess']:
                        continue
                    
                    for objectId,mask in env.last_event.instance_masks.items():
                        if objectId not in id2type:
                            continue
                        objectType = id2type[objectId]
                        if objectType not in ALFRED_INTEREST_OBJECTS:
                            continue
                        if objectId not in visiblepose:
                            visiblepose[objectId] = []
                        visiblepose[objectId].append([x,z,r,h])
                            
        
        with open(visiblepose_save,'wb') as f:
            pkl.dump(visiblepose,f)
    

if __name__ == '__main__':
    args = arguments.parse_args()
    if args.task_scene is None:
        all_scene_numbers = sorted(constants.TRAIN_SCENE_NUMBERS + constants.TEST_SCENE_NUMBERS, reverse=True)
    else:
        all_scene_numbers = args.task_scene
    N_PROCS = args.n_proc
    
    lock = threading.Lock()
    if N_PROCS == 1:
        main(args,0)
    else:
        threads = []
        for n in range(N_PROCS):
            thread = threading.Thread(target=main,args=(args,n))
            threads.append(thread)
            thread.start()
