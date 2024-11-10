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
        if len(all_todos) >0:
            todo = all_todos.pop()
        else:
            todo = None
        lock.release()
        if todo is None:
            print(f'{utils.timestr()} Rank {rank} done')
            env.stop_unity()
            break
        else:
            scene_num, todo_recepId, todo_pickId = todo
            scene_name = ('FloorPlan%d') % scene_num
            reset(env, scene_name)
        
        agent_height = env.last_event.metadata['agent']['position']['y']
        navigation_pt = pkl.load(open('./data/scene_parse/nav/FloorPlan%d.pkl' % scene_num,'rb'))
        id2type = {}
        object2xz   = {}
        for obj in env.last_event.metadata['objects']:
            id2type[obj['objectId']] = obj['objectType']
            object2xz[obj['objectId']] = (obj['position']['x'],obj['position']['z'])
        
        pairpose = []
        visiblepose_save = f'./data/scene_parse/visible_sim/{scene_name}.pkl'
        pickpose_save    = f'./data/scene_parse/pick/{scene_name}.pkl'
        togglepose_save  = f'./data/scene_parse/toggle/{scene_name}.pkl'
        putpose_save     = f'./data/scene_parse/put/{scene_name}.pkl'
        pairpose_save    = f'./data/scene_parse/put_pairs/{scene_name}_{todo_recepId}_{todo_pickId}.pkl'
        visiblepose = pkl.load(open(visiblepose_save,'rb'))
        pickpose    = pkl.load(open(pickpose_save,'rb'))
        
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
            
        recepId, pickId = todo_recepId, todo_pickId
        print(f'{utils.timestr()} Rank {rank} {scene_name}-{recepId}-{pickId}')    
        recepType = id2type[recepId]
        picksuccess = pick_obj(env,pickId,pickpose,scene_name)
        assert picksuccess
        for x,z,r,h in visiblepose[recepId]:
            picksuccess = pick_obj(env,pickId,pickpose,scene_name)
            assert picksuccess
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
            
            recep = None
            object_metadata = env.last_event.metadata['objects']
            for obj in object_metadata:
                if obj['objectId'] == recepId:
                    recep = copy.copy(obj)
                    break
            assert recep is not None
            if not recep['visible']:
                continue
            
            is_openable = recepType in ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe']
            need_repick = recepType in ['Fridge', 'Microwave', 'SinkBasin']
            if is_openable and recep['isOpen']:
                env.step({'action': 'CloseObject','objectId': recepId})
            
            # open
            if is_openable:
                env.step({'action': 'OpenObject','objectId': recepId})
                if not env.last_event.metadata['lastActionSuccess']:
                    continue
                
            # put
            env.step({'action': 'PutObject','objectId': pickId,'receptacleObjectId': recepId,'forceAction': True, 'placeStationary': True})
            if not env.last_event.metadata['lastActionSuccess']:
                if is_openable:
                    env.step({'action': 'CloseObject','objectId': recepId})
                continue
            
            # pick
            env.step({'action': 'PickupObject','objectId': pickId})
            if need_repick and (not env.last_event.metadata['lastActionSuccess']):
                if is_openable:
                    env.step({'action': 'CloseObject','objectId': recepId})
                continue
            
            # close
            if is_openable:
                env.step({'action': 'CloseObject','objectId': recepId})
                if not env.last_event.metadata['lastActionSuccess']:
                    continue
                
            
            pairpose.append([x,z,r,h])
        
        with open(pairpose_save,'wb') as f:
            pkl.dump(pairpose,f)
                
                
                
            
            
    

if __name__ == '__main__':
    args = arguments.parse_args()
    all_scene_numbers = sorted(constants.TRAIN_SCENE_NUMBERS + constants.TEST_SCENE_NUMBERS)
    all_todos = []
    for scene in all_scene_numbers:
        pairs = pkl.load(open(f'./data/scene_parse/recep_pairs/FloorPlan{scene}.pkl','rb'))        
        for r, p in pairs:
            pairpose_save    = f'./data/scene_parse/put_pairs/FloorPlan{scene}_{r}_{p}.pkl'
            if not os.path.exists(pairpose_save):
                print(pairpose_save)
                all_todos.append([scene,r,p])
            
                
            # print(scene,r,p)
    # exit(0)
    all_todos = sorted(all_todos, reverse=True)
    print(f'=======================TODO: {len(all_todos)}=======================')
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
