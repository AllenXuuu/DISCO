from .perceiver import Agent as BaseAgent
from utils import image_util
import time
import torch
import torchvision
import numpy as np
import cv2
from easydict import EasyDict
from models.depthunet import SimpleUNET
from models.affunet import AffordanceUNET
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS,OBJECT_PROPS
from utils.alfred_interest_objects import ALFRED_AFFORDANCE_LIST
import pickle as pkl
from collections import defaultdict
from utils import utils
import os
import cv2

from skimage.morphology import binary_dilation
from skimage.measure import label as connectivity_label
from models.resnet_policy import FinePolicyNet


class Agent(BaseAgent):
    def __init__(self, args, env):
        super(Agent,self).__init__(args, env)
        self.fine_action_classes = ['Interact','MoveAhead','MoveLeft','MoveRight','MoveBack','RotateLeft','RotateRight','LookUp','LookDown']
        self.fine_policy_net = FinePolicyNet(num_input = 5,num_classes = len(self.fine_action_classes), num_object = self.n_obj, cosine=args.cosine).to(self.device)
        self.fine_policy_net.load_state_dict(torch.load('./weights/finepolicy.pth',map_location='cpu')['model'])
        
    def finetune_action(self, target_box,object_classes):
        x1,z1,x2,z2 = target_box
        best_rotation = np.rad2deg(np.arctan2((x1+x2)/2-self.pose[0], (z1+z2)/2 - self.pose[1]))
        best_rotation = round(best_rotation % 360)
        
        candidate_rotation = np.array([0,90,180,270])
        rotation_offset = (candidate_rotation - best_rotation) % 360
        rotation_offset = np.minimum(rotation_offset, 360 - rotation_offset)
        approax_rotation = candidate_rotation[np.argmin(rotation_offset)]
        
        self.set_rotation(approax_rotation)
        self.perceive()

        for i in range(4):
            candidate_idx = [i for i in range(len(self.segementation_frame['labels'])) if self.segementation_frame['labels'][i] in object_classes]
            candidate_idx = sorted(candidate_idx, key = lambda x: self.segementation_frame['masks'][x].sum(), reverse= True)
            masks = [self.segementation_frame['masks'][idx] for idx in candidate_idx]
            labels = [self.segementation_frame['labels'][idx].item() for idx in candidate_idx]
            if len(masks) == 0:
                break
            mask = masks[0].float()
            label = labels[0]
            rgb = torch.from_numpy(self.env.last_event.frame.copy()).to(self.device).float().permute(2,0,1) / 255
            depth = self.depth_frame / 5000
            inp = torch.cat([rgb,depth.unsqueeze(0),mask.unsqueeze(0)],0)
            logits = self.fine_policy_net(inp.unsqueeze(0), [label])[0]
            action = self.fine_action_classes[logits.argmax().item()]
            self.log(f'FineAction: {action}')
            if action == 'Interact':
                break
            else:
                success = self.step(action)
            if not success:
                break

    def run(self):
        self.init_perception()
        
        # self.gt_nav_map = np.zeros([self.map_size, self.map_size],dtype=bool)
        # nav_pt = pkl.load(open(f"./data/scene_parse/nav/{self.task_info['scene']['floor_plan']}.pkl",'rb'))
        # for x,z in nav_pt:
        #     x = round( (x - self.task_info['scene']['init_action']['x']) * 4) + self.map_size // 2
        #     z = round( (z - self.task_info['scene']['init_action']['z']) * 4) + self.map_size // 2
        #     self.gt_nav_map[x, z] = True


        task_type, mrecep_target, object_target, parent_target, sliced = self.load_language_parsing()
        self.log(f'task_type: {task_type}')
        self.log(f'mrecep_target: {mrecep_target}')
        self.log(f'object_target: {object_target}')
        self.log(f'parent_target: {parent_target}')
        self.log(f'sliced: {sliced}')
        plans = self.make_plan(task_type, mrecep_target, object_target, parent_target, sliced)    
        self.log('=========================')
        for p in plans:
            self.log(p)
        self.log('=========================')
        
        self.look_down()
        self.perceive()
        self.rotate_left()
        self.perceive()
        self.rotate_left()
        self.perceive()
        self.rotate_left()
        self.perceive()
        
        
        for subgoal, object_arg in plans:
            self.log(f'============================================ Subgoal: {subgoal} {object_arg}')
            if self.is_terminate():
                break
            self.look_down()
            if subgoal == 'PickupObject':
                self.pickup(object_arg)
            elif subgoal == 'ToggleObject':
                self.toggle(object_arg)
            elif subgoal == 'PutObject':
                self.put(object_arg)
            elif subgoal == 'PutPickObject':
                self.putpick(object_arg)
            elif subgoal == 'SliceObject':
                self.slice(object_arg)
            elif subgoal == 'CoolObject':
                self.cool(object_arg)
            elif subgoal == 'HeatObject':
                self.heat(object_arg)
            elif subgoal == 'CleanObject':
                self.clean(object_arg)
            elif subgoal == 'FindSecond':
                self.find_second(object_arg)
            elif subgoal == 'PickSecond':
                self.pick_second(object_arg)
            else:
                raise NotImplementedError(subgoal)
    
    def obj_str_to_ambigious_class(self,obj_str):
        if obj_str == 'AnyKnife':
            return [self.object2index['Knife'],self.object2index['ButterKnife']]
        elif obj_str == 'Lamp':
            return [self.object2index['DeskLamp'],self.object2index['FloorLamp']]
        elif isinstance(obj_str,list):
            return [self.object2index[o] for o in obj_str]
        else:
            return [self.object2index[obj_str]]
    
    def query_navigable_map(self,connected = True):
        # if self.args.use_gt_aff:
        #     assert self.gt_nav_map[self.pose[0],self.pose[1]]
        #     return self.gt_nav_map
        
        navigable = self.affordance_map[:,:,0] > 0.6
        navigable[self.collision_map == 1] = False
        navigable[self.visited_map == 1] = True
        navigable = navigable.float().cpu().numpy()
        assert navigable[self.pose[0],self.pose[1]]
        

        
        if self.args.vis:
            cur_x, cur_z = self.pose[:2]
            vis = np.tile(np.expand_dims(navigable,-1),(3))
            vis[cur_x,cur_z] = [0,0,1]
            cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_navmap_full.png'), vis * 255) 
            
        if connected:
            connection = connectivity_label(navigable,connectivity=1)
            navigable = connection == connection[self.pose[0],self.pose[1]]
            navigable = navigable.astype(float)
            if self.args.vis:
                cur_x, cur_z = self.pose[:2]
                vis = np.tile(np.expand_dims(navigable,-1),(3))
                vis[cur_x,cur_z] = [0,0,1]
                cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_navmap_connected.png'), vis * 255) 
            
        return navigable
    
    def is_waypoint_reachable(self,waypoint,navigable):
        tgt_x,tgt_z = waypoint
        assert navigable[self.pose[0],self.pose[1]]
        if not navigable[tgt_x,tgt_z]:
            return False
        connection = connectivity_label(navigable,connectivity=1)
        return connection[tgt_x,tgt_z] == connection[self.pose[0],self.pose[1]]
    
    def bfs_distance_from_source(self,navigable):
        assert connectivity_label(navigable,connectivity=1).max() == 1
        distance = np.zeros_like(navigable) - 1
        distance[self.pose[0], self.pose[1]] = 0
        dil = np.array([[0,1,0],[1,1,1],[0,1,0]])
        cur_dis = 1
        while True:
            neighbor = binary_dilation(distance>=0, dil)
            neighbor[distance>=0]=0
            neighbor[navigable==0]=0
            if not np.any(neighbor):
                break
            distance[neighbor==1] = cur_dis
            cur_dis += 1
        assert np.all(np.logical_xor(navigable, distance == -1))
        
        return distance
    
    def bfs_distance_from_target_map(self,goal_map,navigable):
        goal_map[navigable == 0] = 0
        assert np.any(goal_map == 1)
        distance = np.zeros_like(navigable) - 1
        distance[goal_map == 1] = 0
        dil = np.array([[0,1,0],[1,1,1],[0,1,0]])
        cur_dis = 1
        while distance[self.pose[0],self.pose[1]] == -1:
            neighbor = binary_dilation(distance>=0, dil)
            neighbor[distance>=0]=0
            neighbor[navigable==0]=0
            assert np.any(neighbor)
            distance[neighbor==1] = cur_dis
            cur_dis += 1
        distance[distance==-1] = distance.max() + 1
        return distance
    
    def bfs_distance_from_target_point(self, tgt_x, tgt_z, navigable):
        goal_map = np.zeros_like(navigable)
        goal_map[tgt_x,tgt_z] = 1
        return self.bfs_distance_from_target_map(goal_map,navigable)
    
    def plan_step_to_waypoint(self, waypoint,navigable):
        if self.args.vis:
            vis = np.tile(np.expand_dims(navigable,-1),(3))
            vis[self.pose[0],self.pose[1]] = [0,0,1]
            vis[waypoint[0],waypoint[1]] = [1,0,0]
            # cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_waypoint_planning.png'), vis * 255) 
            cv2.imshow('planning',vis * 255)
            cv2.waitKey(50)
        
        distance_matrix = self.bfs_distance_from_target_point(waypoint[0],waypoint[1],navigable)
        max_dist = distance_matrix.max()
        cur_dist = distance_matrix[self.pose[0],self.pose[1]]
        assert max_dist == cur_dist + 1
        for diff_yaw in [0,90,270,180]:
            next_x, next_z = self.pose[:2] + self.yaw2dir(diff_yaw + self.pose[2])
            if not (0<=next_x<self.map_size and 0<=next_z<self.map_size):
                continue
            if distance_matrix[next_x,next_z] == cur_dist - 1:
                if diff_yaw == 0:
                    return 'MoveAhead'
                elif diff_yaw == 90:
                    return 'RotateRight'
                else:
                    return 'RotateLeft'
        raise NotImplementedError('BFS Fail')
     
    def step(self,action):
        if action == 'MoveAhead':
            success = self.move_ahead()
            self.perceive()
        elif action == 'RotateRight':
            success = self.rotate_right()
            self.perceive()
        elif action == 'RotateLeft':
            success = self.rotate_left()
            self.perceive()
        elif action == 'MoveLeft':
            success = self.move_left()
            self.perceive()
        elif action == 'MoveRight':
            success = self.move_right()
            self.perceive()
        elif action == 'MoveBack':
            success = self.move_back()
            self.perceive()
        elif action == 'LookUp':
            success = self.look_up()
            self.perceive()
        elif action == 'LookDown':
            success = self.look_down()
            self.perceive()
        else:
            raise NotImplementedError(action)
        return success

            
    def convert_mask_to_rectangle(self,mask):
        a = np.argwhere(mask==1)
        xmin, xmax = a[:,0].min(),a[:,0].max()
        zmin, zmax = a[:,1].min(),a[:,1].max()
        box = np.array([xmin,zmin,xmax,zmax])
        mask[xmin:xmax+1, zmin:zmax+1] = 1
        return mask, box
    
    def dilate(self,tgt_box, tgt_cls, navigable, failed_waypoint):
        prop = OBJECT_PROPS[ALFRED_INTEREST_OBJECTS[tgt_cls]]
        self.log(f'Fails: {failed_waypoint}')
        if prop['region'] == 'point':
            tgt_x,tgt_z,_,_ = tgt_box
            Z, X = np.meshgrid(np.arange(self.map_size), np.arange(self.map_size))
            yaw_offset = np.rad2deg(np.arctan2(     tgt_x-X,    tgt_z-Z      ))
            yaw_offset = yaw_offset % 90
            yaw_offset = np.minimum(yaw_offset, 90- yaw_offset)

            for expand in range(prop['min_expand'], prop['max_expand'] + 1):
                dilation_map = np.zeros_like(navigable)
                dilation_map[tgt_x-expand:tgt_x+expand+1, tgt_z-expand:tgt_z+expand+1] = 1
                dilation_map[yaw_offset > prop['yaw_thresh']] = 0
                dilation_map[tgt_x, tgt_z] = 0
                dilation_map[navigable == 0] = 0
                for pt in failed_waypoint:
                    dilation_map[pt[0], pt[1]] = 0
                if np.any(dilation_map==1):
                    return dilation_map
            
            for expand in range(prop['min_expand'], prop['max_expand'] + 1):
                dilation_map = np.zeros_like(navigable)
                dilation_map[tgt_x-expand:tgt_x+expand+1, tgt_z-expand:tgt_z+expand+1] = 1
                dilation_map[yaw_offset > 30 ] = 0
                dilation_map[tgt_x, tgt_z] = 0
                dilation_map[navigable == 0] = 0
                for pt in failed_waypoint:
                    dilation_map[pt[0], pt[1]] = 0
                if np.any(dilation_map==1):
                    return dilation_map
            
            distance = np.abs(tgt_x-X) + np.abs(tgt_z-Z)
            distance[navigable == 0] = 1000000 
            for pt in failed_waypoint:
                distance[pt[0], pt[1]] = 1000000
            min_dist = distance.min()
            dilation_map = (distance == min_dist).astype(float)
            return dilation_map
             
            # raise NotImplementedError
            
        elif prop['region'] == 'area':
            xmin,zmin,xmax,zmax = tgt_box
            for expand in range(prop['min_expand'], prop['max_expand'] + 1):
                dilation_map = np.zeros_like(navigable)
                dilation_map[xmin - expand : xmax + expand + 1, zmin : zmax + 1] = 1
                dilation_map[xmin : xmax + 1, zmin - expand : zmax + expand + 1] = 1
                
                exdist = 1
                dilation_map[xmin - exdist : xmax + exdist + 1, zmin : zmax + 1] = 0
                dilation_map[xmin : xmax + 1, zmin - exdist : zmax + exdist + 1] = 0
                
                dilation_map[navigable == 0] = 0
                for pt in failed_waypoint:
                    dilation_map[pt[0], pt[1]] = 0
                if np.any(dilation_map==1):
                    return dilation_map
            
            
            for expand in range(prop['min_expand'], prop['max_expand'] + 1):
                dilation_map = np.zeros_like(navigable)
                dilation_map[xmin - expand : xmax + expand + 1, zmin - expand : zmax + expand + 1] = 1
                
                exdist = 1
                dilation_map[xmin - exdist : xmax + exdist + 1, zmin - exdist : zmax + exdist + 1] = 0
                
                dilation_map[navigable == 0] = 0
                for pt in failed_waypoint:
                    dilation_map[pt[0], pt[1]] = 0
                if np.any(dilation_map==1):
                    return dilation_map
                
            
            Z, X = np.meshgrid(np.arange(self.map_size), np.arange(self.map_size))
            x_dist = np.maximum(xmin - X, X - xmax)
            x_dist = np.maximum(x_dist, 0)
            z_dist = np.maximum(zmin - Z, Z - zmax)
            z_dist = np.maximum(z_dist, 0)
            distance = x_dist + z_dist
            
            distance[navigable == 0] = 1000000 
            for pt in failed_waypoint:
                distance[pt[0], pt[1]] = 1000000
            min_dist = distance.min()
            dilation_map = (distance == min_dist).astype(float)
            return dilation_map
        
        else:
            raise NotImplementedError(prop['region'])        
     
    def sample_random_waypoint(self,navigable):
        waypoints = navigable.copy()
        obstacle = binary_dilation(navigable == 0, np.array([[0,1,0],[1,1,1],[0,1,0]]))
        
        waypoints[obstacle == 0] = 0
        waypoints[self.visited_map.cpu().numpy().astype(bool)] = 0
        x,z = self.pose[:2]
        waypoints[x,z] = 0
        
        pts = np.argwhere(waypoints==1)
        if len(pts) == 0:
            error= f'Sample Random Waypoint: Fail! Stucked Agent'
            self.log(error) 
            raise RuntimeError(error)
        else:
            index = np.random.choice(len(pts))
            return np.array([pts[index][0], pts[index][1]])
    
    def process_target(self, tgt_x, tgt_z, tgt_cls, now_object_map_bin):
        target_map = np.zeros((self.map_size,self.map_size))
        tgt_name = ALFRED_INTEREST_OBJECTS[tgt_cls]
        region_type = OBJECT_PROPS[tgt_name]['region']
        if region_type == 'point':
            target_map[tgt_x,tgt_z] = 1
            target_box = np.array([tgt_x,tgt_z,tgt_x,tgt_z])
        elif region_type == 'area':
            # target_map[tgt_x,tgt_z] = 1
            # target_box = np.array([tgt_x,tgt_z,tgt_x,tgt_z])
            connection = connectivity_label(now_object_map_bin, connectivity=2)
            target_map = (connection == connection[tgt_x, tgt_z])
            target_map, target_box = self.convert_mask_to_rectangle(target_map)
        else:
            raise NotImplementedError
        
        return target_map, target_box
    
    def sample_object_waypoint(self, tgt_box, tgt_cls, navigable, failed_waypoint):
        # tgt_map_dilation,yaw_offset = self.dilate(navigable, tgt_map, box, failed_waypoint,expand, yaw_thresh)
        dilation_map = self.dilate(tgt_box, tgt_cls, navigable, failed_waypoint)
        candidate_waypoints = np.argwhere(dilation_map)
        candidate_x, candidate_z = candidate_waypoints.transpose(1,0)
        distance = self.bfs_distance_from_source(navigable)
        distance = distance[candidate_x, candidate_z]
        
        idx = distance.argmin()
        waypoint = np.array([candidate_x[idx],  candidate_z[idx]])
        
        if self.args.vis:
            waypoint_plan_map = np.tile(navigable[:,:,None],3) * 255
            waypoint_plan_map[tgt_box[0] : tgt_box[2] + 1, tgt_box[1] : tgt_box[3] + 1] = [255, 0, 0]
            waypoint_plan_map[dilation_map == 1] = [120, 120, 0]
            waypoint_plan_map[waypoint[0], waypoint[1]] = [0, 255, 0]
            waypoint_plan_map[self.pose[0], self.pose[1]] = [0,0,255]
            cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_waypoint_plan.png'), waypoint_plan_map) 
            
        return waypoint
    
    def common(self, object_classes, affordance_class=None, region_mask = False):
        is_high_pos = len(set(object_classes).intersection(self.high_object_class)) > 0
        obj_thresh = 0.7
        resample_waypoint = True
        
        #### target info
        target_map = np.zeros((self.map_size,self.map_size))
        target_cls,target_x,target_z = -1, -1, -1
        target_box = np.array([0,0,0,0])
        
        #### waypoint info
        waypoint = np.array([-1, -1])
        
        mask_map = np.ones((self.map_size,self.map_size))
        failed_waypoint = []
        
        while not self.is_terminate():
            if any([o in self.segementation_frame['labels'] for o in object_classes]):
                self.log(f'Visible: {object_classes}')
            
            now_object_map = self.object_map[..., object_classes].data.cpu().numpy() * mask_map[:,:,None]
            if np.max(now_object_map) <= obj_thresh and target_map.sum() == 0:
                navigable = self.query_navigable_map()
                pt_x, pt_z = waypoint
                
                if resample_waypoint or (pt_x == self.pose[0] and pt_z == self.pose[1]) or (not self.is_waypoint_reachable(waypoint,navigable)):
                    resample_waypoint = True
                
                if resample_waypoint:
                    waypoint = self.sample_random_waypoint(navigable)
                    self.log(f'Sample Random Waypoint: {waypoint}')
                        
                    resample_waypoint = False
                action = self.plan_step_to_waypoint(waypoint,navigable)
                self.step(action)
                
                if (pt_x == self.pose[0] and pt_z == self.pose[1]):
                    for i in range(3):
                        self.rotate_left()
                        self.perceive()
                    if is_high_pos:
                        self.look_up()
                        self.perceive()
                        for i in range(3):
                            self.rotate_left()
                            self.perceive()
                        self.look_down()
                        self.perceive()
                
            else:
                navigable = self.query_navigable_map()
                if resample_waypoint or target_map.sum() == 0 or (not self.is_waypoint_reachable(waypoint,navigable)):
                    resample_waypoint = True
                
                if resample_waypoint:
                    resample_waypoint = False
                    self.log('========>>>')
                    
                    if np.max(now_object_map) >obj_thresh:
                        now_object_map[now_object_map < obj_thresh] = 0.
                        if affordance_class is not None:
                            now_object_map *= self.affordance_map[..., [affordance_class]].data.cpu().numpy()
                        index = np.argmax(now_object_map)
                        target_x =   index // (now_object_map.shape[1]    * now_object_map.shape[2])
                        target_z = ( index // (now_object_map.shape[2]) ) % now_object_map.shape[1]
                        target_cls = object_classes[index % now_object_map.shape[2]]
                        
                        now_object_map = self.object_map[..., target_cls].data.cpu().numpy() * mask_map
                        now_object_map = now_object_map > obj_thresh
                        self.log(f'Tgt x={target_x} z={target_z} cls={target_cls}.')
                    
                        target_map, target_box = self.process_target(target_x, target_z, target_cls, now_object_map)
                        
                        self.log(f'ReSample Target. Box: {target_box} .')
                    
                    waypoint = self.sample_object_waypoint(target_box, target_cls, navigable, failed_waypoint)
                    self.log(f'Sample Object {target_box} Waypoint {waypoint}.')
                
                if not (waypoint[:2] == self.pose[:2]).all():
                    action = self.plan_step_to_waypoint(waypoint,navigable)
                    self.step(action)                
                else:
                    if len(failed_waypoint) == 0:
                        self.finetune_action(target_box, object_classes)
                    x1,z1,x2,z2 = target_box
                    best_rotation = np.rad2deg(np.arctan2((x1+x2)/2-self.pose[0], (z1+z2)/2 - self.pose[1]))
                    best_rotation = round(best_rotation % 360)
                    
                    candidate_rotation = np.array([0,90,180,270])
                    rotation_offset = (candidate_rotation - best_rotation) % 360
                    rotation_offset = np.minimum(rotation_offset, 360 - rotation_offset)
                    approax_rotation = candidate_rotation[np.argmin(rotation_offset)]
                    
                    self.log(f'Arrived at waypoint. approax rotation {approax_rotation}')
                    self.set_rotation(approax_rotation)
                    self.perceive()

                    candidate_idx = [i for i in range(len(self.segementation_frame['labels'])) if self.segementation_frame['labels'][i] in object_classes]
                    candidate_idx = sorted(candidate_idx, key = lambda x: self.segementation_frame['masks'][x].sum(), reverse= True)
                    masks = [self.segementation_frame['masks'][idx].cpu().numpy() for idx in candidate_idx]
                    labels = [self.segementation_frame['labels'][idx].cpu().numpy() for idx in candidate_idx]
                    
                    for mask,label in zip(masks,labels):
                        yield mask, label
                        
                    if len(masks) ==0 and is_high_pos:
                        self.look_up()
                        self.perceive()
                    
                        candidate_idx = [i for i in range(len(self.segementation_frame['labels'])) if self.segementation_frame['labels'][i] in object_classes]
                        candidate_idx = sorted(candidate_idx, key = lambda x: self.segementation_frame['masks'][x].sum(), reverse= True)
                        masks = [self.segementation_frame['masks'][idx].cpu().numpy() for idx in candidate_idx]
                        labels = [self.segementation_frame['labels'][idx].cpu().numpy() for idx in candidate_idx]
                    
                        for mask,label in zip(masks,labels):
                            yield mask, label
                        self.look_down()
                    
                        
                    failed_waypoint.append(self.pose[:2].tolist())
                    self.log(f'Waypoint {self.pose} fail')
                    self.log('<<<<<<<<<<<<<<<<<<<<<<<<')
                    self.perceive()
                    
                    resample_waypoint = True
                    if region_mask:
                        mask_map[x1 - 5 : x2 + 5 +1, z1 -5 : z2 + 5 + 1] = 0
                    target_map = np.zeros((self.map_size,self.map_size))
                    target_cls,target_x,target_z = -1, -1, -1
                    target_box = np.array([0,0,0,0])
        
                    # return 
                         
    def put(self,object_arg):
        if not object_arg:
            return
        object_class =self.obj_str_to_ambigious_class(object_arg)
        affordance_class = self.affordance2index['PutObject']
        success = False
        for mask ,label in self.common(object_class, affordance_class):
            open_aff = self.affordance_frame[..., ALFRED_AFFORDANCE_LIST.index('OpenObject')][mask == 1]
            need_open = open_aff.float().mean() > 0.5
            
            if not need_open:
                success, _, _, error, _ = self.va_interact('PutObject', mask)
            else:
                open_success, _, _, _, _ = self.va_interact('OpenObject', mask)
                if not open_success:
                    continue
                put_success, _, _, _, _  = self.va_interact('PutObject', mask)
                close_success, _, _, _, _ = self.va_interact('CloseObject', mask)
                if not put_success:
                    continue
                else:
                    success = True
                
            if success:
                self.picked_object = None
                break
               
    def pickup(self, object_arg):
        if not object_arg:
            return
        
        affordance_class = self.affordance2index['PickupObject']
        object_class =self.obj_str_to_ambigious_class(object_arg)
        for mask ,label in self.common(object_class, affordance_class):
            success, _, _, _, _ = self.va_interact('PickupObject', mask)
            if success:
                self.picked_object = label
                break

    def putpick(self,object_arg):
        if not object_arg:
            return
        object_class =self.obj_str_to_ambigious_class(object_arg)
        for mask ,label in self.common(object_class):
            success, _, _, error, _ = self.va_interact('PutObject', mask)
            if not success:
                self.picked_object = None
                continue
            success, _, _, error, _ = self.va_interact('PickupObject', mask)
            if success:
                self.picked_object = label
                break
            
    def slice(self,object_arg):
        if not object_arg:
            return
        affordance_class = self.affordance2index['SliceObject']
        object_class =self.obj_str_to_ambigious_class(object_arg)
        for mask ,label in self.common(object_class, affordance_class):
            success, _, _, error, _ = self.va_interact('SliceObject', mask)
            if success:
                self.perceive()
                break

    def toggle(self, object_arg):
        if not object_arg:
            return
        
        affordance_class = self.affordance2index['ToggleObject']
        object_class =self.obj_str_to_ambigious_class(object_arg)
        success = False
        for mask ,label in self.common(object_class, affordance_class, region_mask = False):
            success, _, _, error, _ = self.va_interact('ToggleObjectOn', mask)
            if 'can\'t toggle object on if it\'s already on!' in error:
                    self.log('>>>>>>>>>>>>ToggleObjectOn: Wrong Instance <<<<<<<<<<<<')
                    self.log(f'>>>>>>>>>>>>Remove {label} {self.object_list[label]} <<<<<<<<<<<<')
                    object_class.remove(label)
                    object_arg = [self.object_list[o] for o in object_class]
                    self.log(f'>>>>>>>>>>>>Remain {object_class} {object_arg} <<<<<<<<<<<<')
                    assert len(object_class) == 1
                    self.toggle(object_arg)
                    return
            if success:
                break
    
    def cool(self, object_arg):
        if not object_arg:
            return
        assert object_arg == 'Fridge'
        
        affordance_class = self.affordance2index['CoolObject']
        orig_pick = self.picked_object
        fridge_class = self.obj_str_to_ambigious_class(object_arg)
        for fridge_mask, label in self.common(fridge_class, affordance_class):
            open_success, _, _, _, _ = self.va_interact('OpenObject', fridge_mask)
            if not open_success:
                continue
            put_success, _, _, _, _ = self.va_interact('PutObject', fridge_mask)
            if not put_success:
                self.va_interact('CloseObject', fridge_mask)
                continue
            self.picked_object = None
            _, _, _, _, _ = self.va_interact('CloseObject', fridge_mask)
            _, _, _, _, _ = self.va_interact('OpenObject', fridge_mask)
            self.perceive(learn= False)
            object_idx = [i for i in range(len(self.segementation_frame['labels'])) if self.segementation_frame['labels'][i].item() == orig_pick]
            object_idx = sorted(object_idx, key = lambda x: self.segementation_frame['masks'][x].sum(), reverse= True)
            object_idx = object_idx[0]
            _, _, _, _, _ = self.va_interact('PickupObject', self.segementation_frame['masks'][object_idx].cpu().numpy())
            _, _, _, _, _ = self.va_interact('CloseObject', fridge_mask)
            
            self.picked_object = orig_pick
            break
    
    def heat(self, object_arg):
        if not object_arg:
            return
        assert object_arg == 'Microwave'
        
        affordance_class = self.affordance2index['HeatObject']
        orig_pick = self.picked_object
        microwave_class = self.obj_str_to_ambigious_class(object_arg)
        for microwave_mask, label in self.common(microwave_class, affordance_class):
            open_success, _, _, _, _ = self.va_interact('OpenObject', microwave_mask)
            if not open_success:
                continue
            put_success, _, _, _, _ = self.va_interact('PutObject', microwave_mask)
            if not put_success:
                self.va_interact('CloseObject', microwave_mask)
                continue
            self.picked_object = None
            _, _, _, _, _ = self.va_interact('CloseObject', microwave_mask)
            _, _, _, _, _ = self.va_interact('ToggleObjectOn', microwave_mask)
            _, _, _, _, _ = self.va_interact('ToggleObjectOff', microwave_mask)
            _, _, _, _, _ = self.va_interact('OpenObject', microwave_mask)
            self.perceive(learn= False)
            object_idx = [i for i in range(len(self.segementation_frame['labels'])) if self.segementation_frame['labels'][i].item() == orig_pick]
            object_idx = sorted(object_idx, key = lambda x: self.segementation_frame['masks'][x].sum(), reverse= True)
            object_idx = object_idx[0]
            _, _, _, _, _ = self.va_interact('PickupObject', self.segementation_frame['masks'][object_idx].cpu().numpy())
            _, _, _, _, _ = self.va_interact('CloseObject', microwave_mask)
            
            self.picked_object = orig_pick
            break
       
    def clean(self, object_arg):
        if not object_arg:
            return
        assert object_arg == 'SinkBasin'
        
        affordance_class = self.affordance2index['CleanObject']
        orig_pick = self.picked_object
        faucet_class = self.object2index['Faucet']
        sinkbasin_class = self.obj_str_to_ambigious_class(object_arg)
        for sinkbasin_mask, label in self.common(sinkbasin_class, affordance_class):
            faucet_idx = [i for i in range(len(self.segementation_frame['labels'])) if self.segementation_frame['labels'][i].item() == faucet_class]
            faucet_idx = sorted(faucet_idx, key = lambda x: self.segementation_frame['masks'][x].sum(), reverse= True)
            if len(faucet_idx) == 0:
                continue
            
            faucet_idx = faucet_idx[0]
            faucet_mask = self.segementation_frame['masks'][faucet_idx].cpu().numpy()
            
            put_success, _, _, _, _ = self.va_interact('PutObject', sinkbasin_mask)
            if not put_success:
                continue
            self.picked_object = None
            _,_,_,_,_ = self.va_interact('ToggleObjectOn', faucet_mask)
            _,_,_,_,_ = self.va_interact('ToggleObjectOff', faucet_mask)
            self.perceive(learn= False)
            object_idx = [i for i in range(len(self.segementation_frame['labels'])) if self.segementation_frame['labels'][i].item() == orig_pick]
            object_idx = sorted(object_idx, key = lambda x: self.segementation_frame['masks'][x].sum(), reverse= True)
            object_idx = object_idx[0]
            _, _, _, _, _ = self.va_interact('PickupObject', self.segementation_frame['masks'][object_idx].cpu().numpy())
            
            self.picked_object = orig_pick
            break

    def find_second(self,object_arg):
        if not object_arg:
            return
        object_class =self.obj_str_to_ambigious_class(object_arg)
        for mask ,label in self.common(object_class):
            self.second_find = True
            self.second_pose = self.pose.copy()
            self.second_mask = mask.copy()
            return
            
    def pick_second(self,object_arg):
        if not object_arg:
            return
        
        object_class =self.obj_str_to_ambigious_class(object_arg)
        while True:
            if any(self.second_pose[:2] != self.pose[:2]):
                navigable = self.query_navigable_map()
                action = self.plan_step_to_waypoint(self.second_pose[:2], navigable)
                self.step(action)
                continue
            self.set_rotation(self.second_pose[2])
            self.set_horizon(self.second_pose[3])
            success, _, _, _, _ = self.va_interact('PickupObject', self.second_mask)
            if success:
                self.picked_object = object_class[0]
                break
            else:
                raise RuntimeError('Pick Second Fail')