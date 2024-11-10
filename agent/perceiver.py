from .base import Agent as BaseAgent
from utils import image_util
import time
import torch
import torchvision
import numpy as np
import cv2
from easydict import EasyDict
from models.depthunet import SimpleUNET
from models.affunet import AffordanceUNET
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS, OBJECT_PROPS
from utils.alfred_interest_objects import ALFRED_AFFORDANCE_LIST
import pickle as pkl
from collections import defaultdict
from utils import utils
import os
from gen import constants

class Agent(BaseAgent):
    def __init__(self, args, env):
        super(Agent,self).__init__(args, env)
        
        self.depth_net_hor45 = SimpleUNET(EasyDict(max_depth=5000,depth_bins=50)).to(self.device).eval()
        self.depth_net_hor45.load_state_dict(torch.load('./weights/depth_horizon45.pth',map_location='cpu')['model'])
        
        self.depth_net_hor00 = SimpleUNET(EasyDict(max_depth=5000,depth_bins=50)).to(self.device).eval()
        self.depth_net_hor00.load_state_dict(torch.load('./weights/depth_horizon0.pth',map_location='cpu')['model'])
        
        self.affordance_net = AffordanceUNET(n_cls=len(ALFRED_AFFORDANCE_LIST)).to(self.device).eval()
        self.affordance_net.load_state_dict(torch.load('./weights/affordance.pth',map_location='cpu')['model'])
        
        self.mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(weights = None, weights_backbone = None, num_classes=len(ALFRED_INTEREST_OBJECTS) + 1).to(self.device).eval()
        self.mrcnn.load_state_dict(torch.load('./weights/mrcnn.pth',map_location='cpu')['model'])
        
        self.text_db = pkl.load(open('./weights/bert.pkl','rb'))
        
        self.object_list = ALFRED_INTEREST_OBJECTS
        self.object2index = {a : i for i, a in enumerate(self.object_list)}
        self.n_obj = len(self.object_list)
        
        self.affordance_list = ALFRED_AFFORDANCE_LIST
        self.affordance2index = {a : i for i, a in enumerate(self.affordance_list)}
        self.n_aff = len(self.affordance_list)
        
        self.map_size = self.args.map_size
        self.map_unit_mm = self.args.map_unit
        
        self.object_queries = torch.randn((self.args.dim_hidden,self.n_obj)).float().to(self.device).requires_grad_(True)
        self.affordance_queries = torch.randn((self.args.dim_hidden,self.n_aff)).float().to(self.device).requires_grad_(True)
        self.scene_map = torch.zeros((self.map_size, self.map_size, self.args.dim_hidden)).float().to(self.device).requires_grad_(True)
        
        self.openable_obj_names = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe']
        self.openable_classes = [self.object2index[o] for o in self.openable_obj_names]
        
        self.optimizer = torch.optim.SGD(
            [self.scene_map, self.object_queries, self.affordance_queries],
            lr = self.args.optim_lr, momentum=0, weight_decay=0
        )

        self.eps = 1e-4
        
        color = np.load('./utils/color.npy')
        self.obj_color = color[ : self.n_obj]
        self.aff_color = color[self.n_obj : self.n_obj + self.n_aff]
        self.high_object = ['Microwave']
        self.high_object_class = [self.object2index[o] for o in self.high_object]
        
    def va_interact(self, action, interact_mask=None, smooth_nav=None):
        success, _, target_instance_id, error_message, api_action = super().va_interact(action, interact_mask, smooth_nav)
        
        if success:
            if action == 'MoveAhead':
                step_dir = self.yaw2dir(self.pose[2])
                self.pose[:2] = self.pose[:2] + step_dir
            elif action == 'RotateRight':
                self.pose[2] = (self.pose[2] + 90) % 360
            elif action == 'RotateLeft':
                self.pose[2] = (self.pose[2] + 270) % 360
            elif action == 'LookUp':
                self.pose[3] -= 15
            elif action == 'LookDown':
                self.pose[3] += 15
            self.visited_map[self.pose[0], self.pose[1]] = 1
            self.collision_map[self.pose[0], self.pose[1]] = 0
        else:
            if action == 'MoveAhead':
                step_dir = self.yaw2dir(self.pose[2])
                next_x, next_z = self.pose[:2] + step_dir
                self.collision_map[next_x, next_z] = 1
                self.visited_map[next_x,next_z] = 0
                
        
        return success, _, target_instance_id, error_message, api_action
    
    def yaw2dir(self,yaw,size = 1):
        yaw = yaw % 360
        if yaw == 0:
            return np.array([0,size],dtype=int)
        elif yaw == 90:
            return np.array([size,0],dtype=int)
        elif yaw == 180:
            return np.array([0,-size],dtype=int)
        elif yaw == 270:
            return np.array([-size,0],dtype=int)
        else:
            raise NotImplementedError(yaw)
    
    def set_rotation(self,r):
        self.log(f'adjust rotation to {r}')
        relative_r = (r - self.pose[2]) % 360
        assert relative_r in [0,90,180,270]
        
        if relative_r == 90:
            self.rotate_right()
        elif relative_r == 180:
            self.rotate_right()
            self.rotate_right()
        elif relative_r == 270:
            self.rotate_left()
            
    def set_horizon(self,h):
        self.log(f'adjust horizon to {h}')
        assert h in [0,45]
        if h == 0:
            self.look_up()
        else:
            self.look_down()
    
    def rotate_left(self):
        success, _,_,_,_ = self.va_interact('RotateLeft', smooth_nav=False)    
        return success
    
    def rotate_right(self):
        success, _,_,_,_ = self.va_interact('RotateRight', smooth_nav=False)    
        return success
    
    def move_ahead(self):
        success, _,_,_,_ = self.va_interact('MoveAhead', smooth_nav=False)    
        return success
        
    def move_left(self):
        self.va_interact('RotateLeft',  smooth_nav=False)   
        success, _,_,_,_ = self.va_interact('MoveAhead', smooth_nav=False)     
        self.va_interact('RotateRight', smooth_nav=False)     
        return success
    
    def move_right(self):
        self.va_interact('RotateRight',  smooth_nav=False)  
        success, _,_,_,_ = self.va_interact('MoveAhead', smooth_nav=False)     
        self.va_interact('RotateLeft', smooth_nav=False)       
        return success
    
    def move_back(self):
        self.va_interact('RotateRight',  smooth_nav=False)    
        self.va_interact('RotateRight',  smooth_nav=False)    
        success, _,_,_,_ = self.va_interact('MoveAhead', smooth_nav=False)    
        self.va_interact('RotateLeft', smooth_nav=False)      
        self.va_interact('RotateLeft', smooth_nav=False)       
        return success 
    
    def look_down(self):
        assert self.pose[3] in [0,15,30,45]
        success = True
        while self.pose[3]<45 and success:
            success, _,_,_,_ = self.va_interact('LookDown', smooth_nav=False)  
        return success
    
    def look_up(self):
        assert self.pose[3] in [0,15,30,45]
        success = True
        while self.pose[3]>0 and success:      
            success, _,_,_,_ = self.va_interact('LookUp', smooth_nav=False)  
        return success

    def load_language_parsing(self):
        if self.args.use_gt_text:
            task_type = self.task_info['task_type']
            mrecep_target = self.task_info['pddl_params']['mrecep_target']
            object_target = self.task_info['pddl_params']['object_target']
            parent_target = self.task_info['pddl_params']['parent_target']
            toggle_target = self.task_info['pddl_params']['toggle_target']
            sliced = self.task_info['pddl_params']['object_sliced']
            # task_type, mrecep_target, object_target, parent_target, sliced
        
        else:
            if not self.args.text_append:
                key = self.task_info['turk_annotations']['anns'][self.task_info['repeat_idx']]['task_desc']
            else:
                key = ''.join(self.task_info['turk_annotations']['anns'][self.task_info['repeat_idx']]['high_descs'])
            text_label = self.text_db[key]
            
            task_type = text_label['task_type']
            mrecep_target = text_label['mrecep_target']
            object_target = text_label['object_target']
            parent_target = text_label['parent_target']
            sliced = text_label['sliced'] == 1
        return task_type, mrecep_target, object_target, parent_target, sliced
    
    def make_plan(self,task_type, mrecep_target, object_target, parent_target, sliced):
        plans = []
        if task_type == 'look_at_obj_in_light':
            toggle_target = 'Lamp'
            plans.append(['PickupObject', object_target])
            plans.append(['ToggleObject', toggle_target])
        elif task_type == 'pick_and_place_simple':
            if sliced:
                plans.append(['PickupObject',  'AnyKnife'])
                plans.append(['SliceObject',   object_target])
                plans.append(['PutObject',     parent_target])
                object_target += 'Sliced'
            plans.append(['PickupObject', object_target])
            plans.append(['PutObject',    parent_target])
        elif task_type == 'pick_two_obj_and_place':
            if sliced:
                plans.append(['PickupObject',  'AnyKnife'])
                plans.append(['SliceObject',   object_target])
                plans.append(['PutObject',     parent_target])
                object_target += 'Sliced'
            plans.append(['PickupObject',  object_target])
            plans.append(['FindSecond',    object_target])
            plans.append(['PutObject',     parent_target])
            plans.append(['PickSecond',  object_target])
            plans.append(['PutObject',     parent_target])
        elif task_type == 'pick_and_place_with_movable_recep':
            if sliced:
                plans.append(['PickupObject',  'AnyKnife'])
                plans.append(['SliceObject',   object_target])
                plans.append(['PutObject',     parent_target])
                object_target += 'Sliced'
            plans.append(['PickupObject',  object_target])
            plans.append(['PutPickObject', mrecep_target])
            plans.append(['PutObject',     parent_target])
        elif task_type == 'pick_cool_then_place_in_recep':
            if sliced:
                plans.append(['PickupObject',  'AnyKnife'])
                plans.append(['SliceObject',   object_target])
                plans.append(['PutObject',     parent_target])
                object_target += 'Sliced'
            plans.append(['PickupObject',  object_target])
            plans.append(['CoolObject',    'Fridge'])
            plans.append(['PutObject',     parent_target])
        elif task_type == 'pick_heat_then_place_in_recep':
            if sliced:
                plans.append(['PickupObject',  'AnyKnife'])
                plans.append(['SliceObject',   object_target])
                plans.append(['PutObject',     parent_target])
                object_target += 'Sliced'
            plans.append(['PickupObject',  object_target])
            plans.append(['HeatObject',    'Microwave'])
            plans.append(['PutObject',     parent_target])
        elif task_type == 'pick_clean_then_place_in_recep':
            if sliced:
                plans.append(['PickupObject',  'AnyKnife'])
                plans.append(['SliceObject',   object_target])
                plans.append(['PutObject',     parent_target])
                object_target += 'Sliced'
            plans.append(['PickupObject',  object_target])
            plans.append(['CleanObject',   'SinkBasin'])
            plans.append(['PutObject',     parent_target])
        else:
            raise NotImplementedError(task_type)
        return plans
    
    @torch.no_grad()
    def pred_depth(self):
        if self.args.use_gt_depth:
            depth = self.env.last_event.depth_frame.copy()
            depth = torch.from_numpy(depth).float().to(self.device)
        else:
            assert self.pose[3] in [0,45]
            if self.pose[3] == 0:
                depth_net = self.depth_net_hor00
            else:
                depth_net = self.depth_net_hor45
            rgb = torch.from_numpy(self.env.last_event.frame.copy()).permute(2,0,1).float().contiguous().unsqueeze(0).to(self.device)
            depth = depth_net(rgb)[0]   # mm
            depth = depth.detach()
            return depth      
    
    @torch.no_grad()
    def pred_segmentation(self):
        if self.args.use_gt_seg:
            id2type = {}
            for obj in self.last_event.metadata['objects']:
                objectId = obj['objectId']
                objectType = obj['objectType']
                id2type[objectId] = objectType
            
            out = {'boxes': [], 'labels': [], 'scores': [], 'masks': [], 'names': [], 'areas': []}
            for objectId, mask in self.env.last_event.instance_masks.items():
                objectType = id2type[objectId]
                if objectType in self.object2index:
                    xs, ys = np.meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
                    xs, ys = xs[mask], ys[mask]
                    out['boxes'].append([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
                    out['labels'].append(self.object2index[objectType])
                    out['scores'].append(1)
                    out['masks'].append(torch.from_numpy(mask.copy()).float().to(self.device))
                    out['names'].append(objectType)
                    out['areas'].append(mask.sum())
                    
        else:
            rgb = torch.from_numpy(self.env.last_event.frame.copy()).permute(2,0,1).float().contiguous().to(self.device) / 255
            segs = self.mrcnn([rgb])[0]
            segs['labels'] -= 1
            segs['masks'] = segs['masks'].squeeze(1)
            out = {'boxes': [], 'labels': [], 'scores': [], 'masks': [], 'names': [], 'areas': []}
            for i in range(len(segs['scores'])):
                if segs['scores'][i] < self.args.obj_thresh:
                    continue
                box = segs['boxes'][i].cpu().numpy()
                mask = (segs['masks'][i] > 0.5).float().detach()
                area = mask.sum().item()
                label = segs['labels'][i].item()
                
                
                if box[3] - box[1] < 5 or box[2] - box[0] < 5:
                    continue
                
                out['boxes'].append(box)
                out['labels'].append(label)
                out['scores'].append(segs['scores'][i].item())
                out['masks'].append(mask)
                out['names'].append(self.object_list[label])
                out['areas'].append(area)
            # self.object_spotted[label] = True    
        
        if len(out['masks']) == 0:
            return out
            
        ###
        
        def remove_object(index):
            for k, vals in out.items():
                out[k] = [v for i, v in enumerate(vals) if i!=index]
        
        self.picked_object_mask.fill_(0)
        if self.picked_object:
            candidates = [i for i in range(len(out['masks'])) if self.picked_object == out['labels'][i]]
            if len(candidates) > 0:
                # print(f'Remove {self.picked_object} {self.object_list[self.picked_object]} in Ego-segmentation')
                index = sorted(candidates, key=lambda x: out['areas'][x],reverse=True)[0]
                self.picked_object_mask = out['masks'][index]
                remove_object(index)
        
        if len(out['masks']) ==0:
            return out
        
        out['masks'] = torch.stack(out['masks'],0) # type: ignore
        out['labels'] = torch.tensor(out['labels'],device=self.device,dtype=torch.int64) # type: ignore
        
        
        return out
    
    @torch.no_grad()
    def pred_affordance(self):
        rgb = torch.from_numpy(self.env.last_event.frame.copy()).permute(2,0,1).float().contiguous().unsqueeze(0).to(self.device) / 255
        affordance_frame = self.affordance_net(rgb)[0]
        affordance_frame = (affordance_frame > 0).float().detach()
        # print(affordance_frame[...,0].float().mean())
        return affordance_frame
    
    def slam(self, depth, segmentation, affordance_label):
        height, width = depth.shape
        fov = 60
        Z = depth.float().to(self.device)
        X, Y = torch.meshgrid(torch.arange(width,device=self.device), torch.arange(height-1,-1,-1,device=self.device))
        X, Y = X.T, Y.T
        X = (X - width/2 + 0.5) / (width / 2 - 0.5)
        Y = (Y - height/2 + 0.5) / (height / 2 - 0.5)
        X = Z * np.tan(np.deg2rad(fov / 2)) * X
        Y = Z * np.tan(np.deg2rad(fov / 2)) * Y
        coords = torch.stack([X,Y,Z],-1)
        
        # print('==================== Init Coord ================')
        # print(coords[150,150])
        # print('X range',coords[...,0].min().item(),coords[...,0].max().item())
        # print('Y range',coords[...,1].min().item(),coords[...,1].max().item())
        # print('Z range',coords[...,2].min().item(),coords[...,2].max().item())
        
        sin_h = np.sin(np.deg2rad(self.pose[3]))
        cos_h = np.cos(np.deg2rad(self.pose[3]))
        coords = coords @ torch.tensor([
            [1,     0,      0],
            [0,     cos_h,  sin_h],
            [0,     -sin_h, cos_h],    
        ], device=self.device, dtype=torch.float32)
        
        
        # print('==================== H transform ================')
        # print(coords[150,150])
        # print('X range',coords[...,0].min().item(),coords[...,0].max().item())
        # print('Y range',coords[...,1].min().item(),coords[...,1].max().item())
        # print('Z range',coords[...,2].min().item(),coords[...,2].max().item())
        
        sin_r = np.sin(np.deg2rad(self.pose[2]))
        cos_r = np.cos(np.deg2rad(self.pose[2]))
        coords = coords @ torch.tensor([
            [cos_r, 0,      -sin_r],
            [0,     1,      0],
            [sin_r, 0,      cos_r],   
        ], device=self.device, dtype=torch.float32)
        
        
        
        # print('==================== R transform ================')
        # print(coords[150,150])
        # print('X range',coords[...,0].min().item(),coords[...,0].max().item())
        # print('Y range',coords[...,1].min().item(),coords[...,1].max().item())
        # print('Z range',coords[...,2].min().item(),coords[...,2].max().item())
        
        XZ = coords[..., [0,2]].reshape((height * width, 2))
        XZ = XZ // self.map_unit_mm
        XZ = torch.round(XZ).long()
        XZ[..., 0] += self.pose[0]
        XZ[..., 1] += self.pose[1]
        
        
        # assert XZ[...,0].max() < self.map_size,(XZ[...,0].min(),XZ[...,0].max(),XZ[...,1].min(),XZ[...,1].max(),)
        # assert XZ[...,1].max() < self.map_size,(XZ[...,0].min(),XZ[...,0].max(),XZ[...,1].min(),XZ[...,1].max(),)
        # assert XZ[...,0].min() >= 0,(XZ[...,0].min(),XZ[...,0].max(),XZ[...,1].min(),XZ[...,1].max(),)
        # assert XZ[...,1].min() >= 0,(XZ[...,0].min(),XZ[...,0].max(),XZ[...,1].min(),XZ[...,1].max(),)
        
        pt_grid_index = XZ[...,0] * self.map_size + XZ[...,1]
        pt_grid_index = pt_grid_index.flatten()
        
        valid_pts = \
            (torch.logical_and(XZ[...,0]>=0, XZ[...,0]< self.map_size)) & \
                (torch.logical_and(XZ[...,1]>=0, XZ[...,1]< self.map_size)) & \
                    (depth.flatten() > 600) & \
                        (self.picked_object_mask.flatten() == 0)
        # self.log(f'step {self.steps}. valid pt rate={valid_pts.float().sum()/valid_pts.shape[0]}')
        
        ######### pc density
        pt_grid_index = pt_grid_index[valid_pts]
        pt_density = torch.bincount(pt_grid_index, minlength=self.map_size * self.map_size)
        pt_density = pt_density.view(self.map_size , self.map_size)
        
        
        ######### object semantics
        pt_object_label = torch.zeros((self.n_obj,height,width),device=self.device)
        if len(segmentation['masks']) > 0:
            pt_object_label = pt_object_label.index_add(0, segmentation['labels'], segmentation['masks'])
        pt_object_label = (pt_object_label > 0).float()
        pt_object_label = pt_object_label.view(self.n_obj, height * width).permute(1,0)
        pt_object_label = pt_object_label[valid_pts]
        
        scene_object_label = torch.zeros((self.map_size * self.map_size, self.n_obj),device = self.device)
        scene_object_label = scene_object_label.index_add(0, pt_grid_index, pt_object_label)
        max_object_label = scene_object_label.max(0).values
        is_point = torch.tensor([OBJECT_PROPS[o]['region']=='point' for o in ALFRED_INTEREST_OBJECTS], device=self.device)
        localize_obj_thresh = torch.where(is_point, max_object_label - self.eps, max_object_label * 0.2)
        scene_object_label = (scene_object_label >= 1 ) & (scene_object_label >= localize_obj_thresh)
        scene_object_label = scene_object_label.view(self.map_size , self.map_size, self.n_obj).float()
        
        
        ########## affordance semantics
        pt_affordance_label = affordance_label.view(height * width, self.n_aff)
        pt_affordance_label = pt_affordance_label[valid_pts]
        
        scene_affordance_label = torch.zeros((self.map_size * self.map_size, self.n_aff),device = self.device)
        scene_affordance_label = scene_affordance_label.index_add(0, pt_grid_index, pt_affordance_label)
        scene_affordance_label = scene_affordance_label.view(self.map_size , self.map_size, self.n_aff).float()
        scene_affordance_label = (scene_affordance_label / (pt_density[:,:,None] + self.eps)).float()
        
        
        # pt_label = torch.cat([pt_object_label,pt_affordance_label],0).permute(1,0)
        # pt_label = pt_label[valid_pts]
        
        
        
        ######### scene rep
        # scene_label = torch.zeros((self.map_size * self.map_size, self.n_obj + self.n_aff),device = self.device)
        # scene_label = scene_label.index_add(0, pt_grid_index, pt_label)
        # scene_label = scene_label.view(self.map_size , self.map_size, self.n_obj + self.n_aff)
        
        # scene_label = scene_label / (pt_density.unsqueeze(-1) + self.eps)
        # scene_label = scene_label / (scene_label.max(0).values.max(0).values + self.eps)
        # scene_label = scene_label.detach()
        
        # scene_object_label = scene_label[:,:,:self.n_obj]
        # scene_affordance_label = scene_label[:,:,self.n_obj:]
        
        return scene_object_label, scene_affordance_label , pt_density
    
    def optimize_scene(self, object_label, affordance_label, pt_density):
        if self.pose[3] == 45:
            update_aff = True
        else:
            update_aff = False
        # print('===')
        for opt_step in range(self.args.optim_steps):
            loss = 0
            
            object_predition = torch.sigmoid(self.scene_map @ self.object_queries)        
            object_loss = - object_label * torch.log(object_predition+self.eps) - (1-object_label) * torch.log(1-object_predition+self.eps)
            object_mask = (pt_density > self.args.optim_min_pts_obj).float().detach()
            object_mask[object_label.max(-1).values == 1] = 1
            object_loss = object_loss.sum(-1) * object_mask
            loss += object_loss
            
            if update_aff:
                affordance_mask = (pt_density > self.args.optim_min_pts_aff).float().detach()
                affordance_mask[self.pose[0] - 1 : self.pose[0] + 2, self.pose[1]] = 0
                affordance_mask[self.pose[0] , self.pose[1] -1 : self.pose[1] + 2] = 0
                next_pos = self.yaw2dir(self.pose[2]) + self.pose[:2]
                affordance_mask[next_pos[0], next_pos[1]] = 1
                affordance_label = affordance_label / (affordance_label.max(0).values.max(0).values + self.eps)
                
                if pt_density[next_pos[0], next_pos[1]] <= self.args.optim_min_pts_aff:
                    affordance_label[next_pos[0], next_pos[1], 0] = 1
                
                if affordance_label[next_pos[0], next_pos[1], 0] > 0.3:
                    affordance_label[next_pos[0], next_pos[1], 0] = 1
                    
                    
                # affordance_label = affordance_label > 0.5
                
                # print(affordance_mask[self.pose[0]-1:self.pose[0]+2, self.pose[1]-1:self.pose[1]+2])
                # print(affordance_label[self.pose[0]-1:self.pose[0]+2, self.pose[1]-1:self.pose[1]+2,0])
                # print(pt_density[self.pose[0]-1:self.pose[0]+2, self.pose[1]-1:self.pose[1]+2])
                
                affordance_label[self.collision_map == 1][...,0] = 0
                affordance_label[self.visited_map == 1][...,0] = 1
                affordance_predition = torch.sigmoid(self.scene_map @ self.affordance_queries)
                affordance_loss = - affordance_label * torch.log(affordance_predition+self.eps) - (1-affordance_label) * torch.log(1-affordance_predition+self.eps)
                
                
                affordance_loss = affordance_loss.sum(-1) * affordance_mask
                loss+=affordance_loss
            
            loss = loss.sum()
            
            self.optimizer.zero_grad()    
            loss.backward()
            self.optimizer.step()
        
        self.object_map = torch.sigmoid(self.scene_map @ self.object_queries)
        self.affordance_map = torch.sigmoid(self.scene_map @ self.affordance_queries)
        # print('step',self.steps, object_label[:,:,33].max(), (object_label[:,:,33] * object_mask).max()) # type:ignore
    
    def perceive(self, learn = True):
        depth = self.pred_depth()
        segmentation = self.pred_segmentation()
        affordance_frame = self.pred_affordance() # h, w, 8
        
        
        self.depth_frame = depth
        self.segementation_frame = segmentation
        self.affordance_frame = affordance_frame
        
        if learn:
            scene_object_label, scene_affordance_label, pt_density = self.slam(depth,segmentation,affordance_frame)
            self.optimize_scene(scene_object_label, scene_affordance_label, pt_density)
        
        if self.args.vis and self.task_log_dir:
            x,z,r,h = self.pose
            cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_rgb.png'), self.env.last_event.frame[:,:,::-1])
            cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_depth.png'), depth.data.cpu().numpy() * 255 / 5000)
            seg_frame = utils.visualize_segmentation(
                rgb = self.env.last_event.frame,
                labels= segmentation['labels'],
                masks = segmentation['masks'],
                captions=segmentation["names"],
                color_map=self.obj_color
            )
            cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_seg.png'), seg_frame[:,:,::-1])
            cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_navframe.png'), affordance_frame[...,0].cpu().numpy() * 255)
            # if learn:
            #     navlabel = np.tile(np.expand_dims(scene_affordance_label[...,0].cpu().numpy() * 255, -1),3)
            #     affordance_mask = (pt_density > self.args.optim_min_pts_aff).float().detach().cpu().numpy()
            #     navlabel[affordance_mask == 0] = [255,0,0]
            #     cv2.imwrite(os.path.join(self.task_log_dir, f'step_{self.steps:05d}_navlabel.png'), navlabel)
            navmap = (self.affordance_map[...,0] > 0.6).unsqueeze(-1).tile(3).cpu().numpy().astype(float)
            navmap[x,z] = [0,0,1]
            navmap = navmap.astype(np.uint8) * 255
            
            
    

    def init_perception(self):
        self.object_queries = torch.randn((self.args.dim_hidden,self.n_obj)).float().to(self.device).requires_grad_(True)
        self.affordance_queries = torch.randn((self.args.dim_hidden,self.n_aff)).float().to(self.device).requires_grad_(True)
        self.scene_map = torch.zeros((self.map_size, self.map_size, self.args.dim_hidden)).float().to(self.device).requires_grad_(True)
        
        self.optimizer = torch.optim.SGD(
            [self.scene_map, self.object_queries, self.affordance_queries],
            lr = self.args.optim_lr, momentum=0, weight_decay=0
        )
        
        self.second_find = False
        self.second_pose = np.zeros([0,0,0,0])
        self.second_mask = np.zeros((constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH))
        
        self.picked_object = None
        self.picked_object_mask = torch.zeros((constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH) ,device = self.device)
        self.pose = np.array([self.map_size // 2, self.map_size // 2, 0, 30],dtype=np.int64)
        # self.pose[0] += round(self.task_info['scene']['init_action']['x'] * 1000 / self.map_unit_mm)
        # self.pose[1] += round(self.task_info['scene']['init_action']['z'] * 1000 / self.map_unit_mm)
        self.pose[2] = int(self.task_info['scene']['init_action']['rotation'])
    
        self.collision_map  = torch.zeros((self.map_size, self.map_size)).float().to(self.device)
        self.visited_map = torch.zeros((self.map_size, self.map_size)).float().to(self.device)
        self.visited_map[self.pose[0],self.pose[1]] = 1.
        
        # self.object_spotted = np.zeros(self.n_obj,dtype=bool)
        
    def run(self):
        self.init_perception()
        
        # task_type, mrecep_target, object_target, parent_target, sliced = self.load_language_parsing()
        # plans = self.make_plan(task_type, mrecep_target, object_target, parent_target, sliced)    
        
        self.look_down()
        

        while True:
            self.perceive()
            action = input('Type Action:    ').strip()
            if action == 'w':
                self.move_ahead()
            elif action == 'a':
                self.move_left()
            elif action == 's':
                self.move_back()
            elif action == 'd':
                self.move_right()
            elif action == 'q':
                self.rotate_left()
            elif action == 'e':
                self.rotate_right()
            elif action == 'u':
                self.look_up()
            elif action == 'n':
                self.look_down()
            else:
                print('Unknown input: %s' % action)
    