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

def is_lamp(objectType):
    return objectType in ['DeskLamp','FloorLamp']

def reset(env,scene_name):
    env.reset(  
        scene_name,
        render_image=False,
        render_depth_image=False,
        render_class_image=False,
        render_object_image=False)
    if any([is_lamp(obj['objectType']) for obj in env.last_event.metadata['objects']]):
        env.step(dict(action='SetObjectToggles', 
                    objectToggles=[
                        {'objectType': 'DeskLamp',  'isOn': False},
                        {'objectType': 'FloorLamp', 'isOn': False}]
                    ))


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
            break
        else:
            scene_name = ('FloorPlan%d') % scene_num
            reset(env,scene_name)

        agent_height = env.last_event.metadata['agent']['position']['y']
        navigation_pt = pkl.load(open('./data/scene_parse/nav/FloorPlan%d.pkl' % scene_num,'rb'))
        id2type = {}
        for obj in env.last_event.metadata['objects']:
            id2type[obj['objectId']] = obj['objectType']
    
        
        togglepose = {}
        visiblepose_save = f'./data/scene_parse/visible/{scene_name}.pkl'
        pickpose_save    = f'./data/scene_parse/pick/{scene_name}.pkl'
        togglepose_save  = f'./data/scene_parse/toggle/{scene_name}.pkl'
        
        visiblepose = pkl.load(open(visiblepose_save,'rb'))
        pts = [len(visiblepose[objectId]) for objectId,objectType in id2type.items() if is_lamp(objectType)]
        pts = sum(pts)
        print(f'{utils.timestr()} Rank {rank} {scene_name} pts {pts}')
        
        for objectId, objectType in id2type.items():
            if not is_lamp(objectType):
                continue
            for x,z,r,h in visiblepose[objectId]:
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
                    
                action = {'action': 'ToggleObjectOn', 'objectId': objectId}
                lamp_visible = None
                object_metadata = env.last_event.metadata['objects']
                for obj in object_metadata:
                    if obj['objectId'] == objectId:
                        lamp_visible = obj['visible']
                        break
                assert lamp_visible is not None
                
                action = {'action': 'ToggleObjectOn', 'objectId': objectId}
                env.step(action)
                if env.last_event.metadata['lastActionSuccess']:
                    reset(env,scene_name)
                    event = env.step(teleportaction)
                    toggle_ok = True
                else:
                    toggle_ok = False
                
                assert toggle_ok == lamp_visible
                
                if toggle_ok:
                    if objectId not in togglepose:
                        togglepose[objectId] = []
                    togglepose[objectId].append([x,z,r,h])
                
                            
        with open(togglepose_save,'wb') as f:
            pkl.dump(togglepose,f)
    

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
