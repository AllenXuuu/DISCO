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
import time

def reset(env,scene_name):
    env.reset(  
        scene_name,
        render_image=False,
        render_depth_image=False,
        render_class_image=False,
        render_object_image=False)
    
def pick_obj(env, objectid, pickpose, scene_name):
    if len(env.last_event.metadata['inventoryObjects']) > 0:
        inv_id = env.last_event.metadata['inventoryObjects'][0]['objectId']
        if inv_id == objectid:
            return True
    if objectid not in pickpose:
        return False
    reset(env,scene_name)    
    for x,z,r,h in pickpose[objectid]:
        action = {
            'action': 'TeleportFull',
            'x': x,
            'y': env.last_event.metadata['agent']['position']['y'],
            'z': z,
            'rotateOnTeleport': True,
            'rotation': r,
            'horizon': h
            }
        env.step(action)
        assert env.last_event.metadata['lastActionSuccess']
            
        
        action = {'action': 'PickupObject', 'objectId': objectid}
        env.step(action)
        assert env.last_event.metadata['lastActionSuccess'], env.last_event.metadata['errorMessage']
        return True
    
    return False
    


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
        object2xz   = {}
        for obj in env.last_event.metadata['objects']:
            id2type[obj['objectId']] = obj['objectType']
            object2xz[obj['objectId']] = (obj['position']['x'],obj['position']['z'])
        
        putpose = {}
        visiblepose_save = f'./data/scene_parse/visible_sim/{scene_name}.pkl'
        pickpose_save    = f'./data/scene_parse/pick/{scene_name}.pkl'
        togglepose_save  = f'./data/scene_parse/toggle/{scene_name}.pkl'
        putpose_save     = f'./data/scene_parse/put/{scene_name}.pkl'
        visiblepose = pkl.load(open(visiblepose_save,'rb'))
        pickpose    = pkl.load(open(pickpose_save,'rb'))
        recep_obj_pair = json.load(open('./utils/recep_object_pairs.json'))
        recep_obj_pair = set([tuple(a) for a in recep_obj_pair])
        
        for recepId in visiblepose:
            poses = visiblepose[recepId]
            newposes = []
            for x,z,r,h in poses:
                if 'Microwave' not in recepId and h==0:
                    continue
                gtr = np.arctan2(object2xz[recepId][0] - x, object2xz[recepId][1] - z)
                gtr = np.rad2deg(gtr)
                gtr = int(np.round(gtr / 90))
                gtr = (gtr * 90) % 360
                # if r!=gtr:
                #     continue
                newposes.append([x,z,r,h])
            visiblepose[recepId] = newposes

            
        
        
        recep_type_count = defaultdict(int)
        recepable_pickupable_id_pair = []
        for recepId, recepType in id2type.items():
            if recepType not in ALFRED_INTEREST_OBJECTS:
                continue
            if recepType not in constants.VAL_RECEPTACLE_OBJECTS:
                continue
            if recepId not in visiblepose:
                continue
            recep_type_count[recepType] += 1
            if recep_type_count[recepType] > 3:
                continue           
            pickup_type_count = defaultdict(int)
            recepable_pickupids = []
            for objectId, objectType in id2type.items():
                # if objectType in constants.VAL_RECEPTACLE_OBJECTS[recepType] and objectId in pickpose:
                if (recepType,objectType) in recep_obj_pair and objectId in pickpose:
                    pickup_type_count[objectType] += 1
                    if pickup_type_count[objectType] > 1:
                        continue
                    recepable_pickupids.append(objectId)
                # if len(recepable_pickupids) > 2:
                #     break
            if (recepable_pickupids) == 0:
                continue
            np.random.shuffle(recepable_pickupids)
            recepable_pickupids = recepable_pickupids[:3]
            for pickupid in recepable_pickupids:
                recepable_pickupable_id_pair.append((recepId, pickupid))
        
        pts = sum([len(visiblepose[r]) for r,p in recepable_pickupable_id_pair])
        print(f'{utils.timestr()} Rank {rank} {scene_name} pair {len(recepable_pickupable_id_pair)} pts {pts}')    
        pair_save = f'./data/scene_parse/recep_pairs/{scene_name}.pkl'
        with open(pair_save,'wb') as f:
            pkl.dump(recepable_pickupable_id_pair,f)
                
                
                
            
            
    

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
