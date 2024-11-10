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
import multiprocessing as mp
import cv2
def is_lamp(objectType):
    return objectType in ['DeskLamp','FloorLamp']

def reset(env,scene_name):
    env.reset(  
        scene_name,
        render_image=True,
        render_depth_image=True,
        render_class_image=False,
        render_object_image=True)
    if any([is_lamp(obj['objectType']) for obj in env.last_event.metadata['objects']]):
        env.step(dict(action='SetObjectToggles', 
                    objectToggles=[
                        {'objectType': 'DeskLamp',  'isOn': False},
                        {'objectType': 'FloorLamp', 'isOn': False}]
                    ))

    
def pick_obj(env, objectid, scene_name):
    pickpose = pkl.load(open(f'./data/scene_parse/pick/{scene_name}.pkl','rb'))
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

def main(args,rank,tasks,start,end):
    env = ThorEnv(x_display = args.x_display)
    eps = 0.0001
    for t in tasks[start:end]:
        idx = t['index']
        scene_name = 'FloorPlan%d' % t['scene']
        interactId = t['interactId']
        pickId = t['pickId']
        x,z,r,h = t['pose']
        reset(env,scene_name)
        if pickId is not None:
            success = pick_obj(env,pickId,scene_name)
        
        action = {
            'action': 'TeleportFull',
            'x': x,
            'y': env.last_event.metadata['agent']['position']['y'],
            'z': z,
            'rotateOnTeleport': True,
            'rotation': r,
            'horizon': h
            }
        event = env.step(action)
        if not event.metadata['lastActionSuccess']:
           # assert False
            continue
        if idx % 400 == 0:
            print(f'{utils.timestr()} Rank {rank} {idx}/{end-start} [start={start},end={end}]')   
        if interactId not in env.last_event.instance_masks:
            continue
        root = './data/scene_parse/fine_action_train_frames'
        cv2.imwrite(os.path.join(root,'%08d_rgb.png' % idx), cv2.cvtColor(env.last_event.frame,cv2.COLOR_BGR2RGB))
        depth_image = env.last_event.depth_frame
        assert depth_image.max() <= 5000
        depth_image = depth_image * (255 / 5000)
        depth_image = depth_image.astype(np.uint8)
        cv2.imwrite(os.path.join(root,'%08d_depth.png' % idx), depth_image)
        mask = env.last_event.instance_masks[interactId]
        mask = np.tile(np.expand_dims(mask,-1),3)
        mask = mask.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(root,'%08d_mask.png' % idx), mask)
    env.stop_unity()

if __name__ == '__main__':
    args = arguments.parse_args()
    
    N_PROCS = args.n_proc
    
    tasks = json.load(open('./data/scene_parse/fine_action_train.json'))
    for i,t in enumerate(tasks):
        t['index'] = i

    if N_PROCS == 1:
        main(args,0,tasks ,0, len(tasks))
    else:
        ps = []
        for n in range(N_PROCS):
            start = len(tasks) * n // N_PROCS
            end  = len(tasks) * (n + 1) // N_PROCS
            proc = mp.Process(target=main,args=(args,n,tasks,start,end))
            ps.append(proc)
            proc.start()
        for p in ps:
            p.join()