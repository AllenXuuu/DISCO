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
import copy

def reset(env,scene_name):
    env.reset(  
        scene_name,
        render_image=False,
        render_depth_image=False,
        render_class_image=False,
        render_object_image=False)

def main(args, rank):
    env = ThorEnv(x_display = args.x_display)
    eps = 0.0001
    while True:
        lock.acquire()
        if len(all_scene_numbers) >0:
            scene_num = all_scene_numbers.pop()
        else:
            scene_num = None
        lock.release()
        if scene_num is None:
            print(f'{utils.timestr()} Rank {rank} done')
            env.stop_unity()
            break
        else:
            scene_name = ('FloorPlan%d') % scene_num
            reset(env, scene_name)
        
        agent_height = env.last_event.metadata['agent']['position']['y']
        navigation_pt = pkl.load(open('./data/scene_parse/nav/FloorPlan%d.pkl' % scene_num,'rb'))
        id2type = {}
        for obj in env.last_event.metadata['objects']:
            id2type[obj['objectId']] = obj['objectType']
        
        pickpose         = {}
        visiblepose      = {}
        visiblesimpose   = {}
        visiblesimpose_save = f'./data/scene_parse/visible_sim/{scene_name}.pkl'
        visiblepose_save    = f'./data/scene_parse/visible/{scene_name}.pkl'
        pickpose_save       = f'./data/scene_parse/pick/{scene_name}.pkl'
        
        print(f'{utils.timestr()} Rank {rank} {scene_name} pts {len(navigation_pt)}')
        for _,(x,z) in enumerate(navigation_pt):
            for r in [0,90,180,270]:
                for h in [45,0]:
                    teleportaction = {
                        'action': 'TeleportFull',
                        'x': x,
                        'y': agent_height,
                        'z': z,
                        'rotateOnTeleport': True,
                        'rotation': r,
                        'horizon': h
                        }
                    event = env.step(teleportaction)
                    if not event.metadata['lastActionSuccess']:
                        continue
                    # assert np.abs(env.last_event.metadata['agent']['position']['x'] - x ) < eps
                    # assert np.abs(env.last_event.metadata['agent']['position']['z'] - z ) < eps
                    # assert np.abs(env.last_event.metadata['agent']['rotation']['y'] - r) < eps
                    # assert np.abs(env.last_event.metadata['agent']['cameraHorizon'] - h)  < eps
                    
                    # for objectId,mask in env.last_event.instance_masks.items():
                    #     if objectId not in id2type:
                    #         continue
                    #     objectType = id2type[objectId]
                    #     if objectType not in ALFRED_INTEREST_OBJECTS:
                    #         continue
                    #     if objectId not in visiblepose:
                    #         visiblepose[objectId] = []
                    #     visiblepose[objectId].append([x,z,r,h])
                    
                    
                    object_metadata = copy.copy(env.last_event.metadata['objects'])
                    for obj in object_metadata:
                        objectId = obj['objectId']
                        objectType = obj['objectType']
                        if objectType in ALFRED_INTEREST_OBJECTS and obj['visible']:
                            if objectId not in visiblesimpose:
                                visiblesimpose[objectId] = []
                            visiblesimpose[objectId].append([x,z,r,h])
                        
                        if objectType in ALFRED_INTEREST_OBJECTS and obj['visible'] and obj['pickupable']:
                            action = {'action': 'PickupObject', 'objectId': obj['objectId']}
                            env.step(action)
                            if env.last_event.metadata['lastActionSuccess']:
                                reset(env, scene_name)
                                event = env.step(teleportaction)
                                pick_ok = True
                            else:
                                pick_ok = False
                                
                            # print(pick_ok, obj['visible'],env.last_event.metadata['errorMessage'])
                            # assert pick_ok == obj['visible']
                            # pick_ok = obj['visible']
                            
                            if pick_ok:
                                if objectId not in pickpose:
                                    pickpose[objectId] = []
                                pickpose[objectId].append([x,z,r,h])
                            
 
        
        with open(visiblesimpose_save,'wb') as f:
            pkl.dump(visiblesimpose,f)
            
        # with open(visiblepose_save,'wb') as f:
        #     pkl.dump(visiblepose,f)
        
        with open(pickpose_save,'wb') as f:
            pkl.dump(pickpose,f)
    
        print(f'{utils.timestr()} Rank {rank} {scene_name} OK')

if __name__ == '__main__':
    args = arguments.parse_args()
    if args.task_scene is None:
        all_scene_numbers = sorted(constants.TRAIN_SCENE_NUMBERS + constants.TEST_SCENE_NUMBERS, reverse=True)
    else:
        all_scene_numbers = args.task_scene
    # all_scene_numbers = sorted(list(range(401, 431)), reverse=True)
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
