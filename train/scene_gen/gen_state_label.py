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
import time,tqdm
from skimage.morphology import binary_dilation


    
def step_xz(x,z,yaw,size = 1):
    yaw = yaw % 360
    if yaw == 0:
        return np.array([x, size + z],dtype=int)
    elif yaw == 90:
        return np.array([x + size, z],dtype=int)
    elif yaw == 180:
        return np.array([x, z - size],dtype=int)
    elif yaw == 270:
        return np.array([x - size, z],dtype=int)
    else:
        raise NotImplementedError(yaw)
def main():
    all_scene_numbers = sorted(constants.TRAIN_SCENE_NUMBERS + constants.TEST_SCENE_NUMBERS)
    good_poses = {} # scene, interactId, pickId - > poses
    for scene_num in all_scene_numbers:
        # navigation_pt = pkl.load(open('./data/scene_parse/nav/FloorPlan%d.pkl' % scene_num,'rb'))
        
        # visible = f'./data/scene_parse/visible/FloorPlan{scene_num}.pkl'
        # visible = pkl.load(open(visible,'rb'))
        
        pickable = f'./data/scene_parse/pick/FloorPlan{scene_num}.pkl'
        pickable = pkl.load(open(pickable,'rb'))
        for interactId, pose in pickable.items():
            if len(pose) == 0:
                continue
            good_poses[(scene_num, interactId, None)] = pose
        
        toogleable = f'./data/scene_parse/toggle/FloorPlan{scene_num}.pkl'
        toogleable = pkl.load(open(toogleable,'rb'))
        for interactId, pose in toogleable.items():
            if len(pose) == 0:
                continue
            good_poses[(scene_num, interactId, None)] = pose
        
        
        for fn in os.listdir(f'./data/scene_parse/put_pairs'):
            sname, rid, pid = fn.replace('.pkl','').split('_')
            if sname == f'FloorPlan{scene_num}':
                poses = pkl.load(open(os.path.join(f'./data/scene_parse/put_pairs',fn),'rb'))
                # print(poses)
                if len(poses) == 0:
                    continue
                good_poses[(scene_num, rid, pid)] = poses
                
    print(len(good_poses))
    
    todos_frames = []
    for (scene_num, interactId, pickeId), good_pose in tqdm.tqdm(good_poses.items(),ncols = 100):
        
        if scene_num in [9, 10, 29, 215, 219, 226, 308, 315, 325, 404, 424, 425]:
            continue
        
        navigation_pt = pkl.load(open('./data/scene_parse/nav/FloorPlan%d.pkl' % scene_num,'rb'))
        visible = f'./data/scene_parse/visible/FloorPlan{scene_num}.pkl'
        
        visible = pkl.load(open(visible,'rb'))
        if interactId not in visible:
            continue
        visible = visible[interactId]
        
        
        navi_map = np.zeros((100,100))
        offset = 40
        for x,z in navigation_pt:
            x,z = round(x * 4) + offset, round(z * 4)+ offset
            navi_map[x,z] = 1
        dist_map = np.zeros_like(navi_map) - 1
        for x,z,r,h in good_pose:
            x,z = round(x * 4) + offset, round(z * 4)+ offset
            dist_map[x,z] = 0
        
        
        for d in range(1,3):
            dil = np.array([[0,1,0],[1,1,1],[0,1,0]])
            neighbor = binary_dilation(dist_map>=0, dil)
            neighbor[navi_map == 0] = 0
            neighbor[dist_map >= 0] = 0
            if not np.any(neighbor == 1):
                break
            dist_map[neighbor == 1] = d
            
        for fpx,fpz,r,h in visible:
            if h == 0 and 'Microwave' not in interactId:
                    continue
            x,z = round(fpx * 4) + offset, round(fpz * 4)+ offset
            if dist_map[x,z] == -1:
                continue
            if dist_map[x,z] > 0:
                next_dist = []
                yaw = [0,270,90,180]
                move_step = ['MoveAhead','MoveLeft','MoveRight','MoveBack']
                action_label = None
                for dr, action in zip(yaw, move_step):
                    next_x, next_z = step_xz(x, z, r + dr)
                    if dist_map[next_x, next_z] == dist_map[x,z] - 1:
                        action_label = action
                assert action_label is not None

            else:
                good_r = []
                good_h = []
                for gx,gz,gr,gh in good_pose:
                   if gx==fpx and gz ==fpz:
                       good_r.append(gr)
                   if gx==fpx and gz ==fpz and gr == r:
                       good_h.append(gh)
                assert len(good_r) > 0
                if r not in good_r:
                    if (r+90) % 360 in good_r:
                        action_label = 'RotateRight'
                    elif (r+270) % 360 in good_r:
                        action_label = 'RotateLeft'
                    elif (r+180) % 360 in good_r:
                        action_label = 'RotateRight'
                    else:
                        raise NotImplementedError
                elif h not in good_h:
                    if h == 0:
                        assert 45 in good_h
                        action_label = 'LookDown'
                    elif h == 45:
                        assert 0 in good_h
                        action_label = 'LookUp'
                    else:
                        raise NotImplementedError
                else:
                    action_label = 'Interact'
                
            todos_frames.append({
                'scene' : scene_num,
                'interactId' : interactId,
                'pickId' : pickeId,
                'action' : action_label,
                'pose' : (fpx,fpz,r,h)
            })
                
    print(len(todos_frames))
    count = defaultdict(int)
    for info in todos_frames:
        count[info['action']] += 1
    print(count)
    with open('./data/scene_parse/fine_action_train.json','w') as f:
        json.dump(todos_frames,f)
if __name__ == '__main__':
    main()