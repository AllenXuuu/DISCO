import torch
import json
import os
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset,DataLoader
import sys
from utils.utils import remove_alfred_repeat_tasks
import cv2
from utils.alfred_interest_objects import ALFRED_AFFORDANCE_LIST
from torchvision.transforms.functional import resize, to_pil_image
import time

def build_aff_dataloader(name, split, args):
    if name == 'AlfredAffordanceDataset':
        dataset    = AlfredAffordanceDataset(args,split)
        dataloader = DataLoader(
            dataset,
            batch_size = args.bz,
            shuffle = (split == 'train'),
            num_workers= args.num_workers,
            collate_fn=AlfredAffordanceDataset.collate_fn
        )
    elif name == 'AlfredAffordanceDatasetV2':
        dataset    = AlfredAffordanceDatasetV2(args,split)
        dataloader = DataLoader(
            dataset,
            batch_size = args.bz,
            shuffle = (split == 'train'),
            num_workers= args.num_workers,
            collate_fn=AlfredAffordanceDatasetV2.collate_fn
        )
    elif name == 'AlfredAffordanceDatasetV3':
        dataset    = AlfredAffordanceDatasetV3(args,split)
        dataloader = DataLoader(
            dataset,
            batch_size = args.bz,
            shuffle = (split == 'train'),
            num_workers= args.num_workers,
            collate_fn=AlfredAffordanceDatasetV3.collate_fn
        )
    else:
        raise NotImplementedError(name)
    
    return dataloader


class AlfredAffordanceDataset(Dataset):
    def __init__(self,args,split,data_root = './data') -> None:
        super().__init__()
        self.args = args
        self.split = split
        self.data_root = data_root
        self.tasks = json.load(open(os.path.join(data_root,'splits/oct21.json')))[split]
        self.tasks = remove_alfred_repeat_tasks(self.tasks)
        self.images = [] 
        for i,t in enumerate(self.tasks):
            traj_data = json.load(open(os.path.join('./data/json_2.1.0/', split, t['task'], 'traj_data.json')))
            expert_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] ]
            camera_horizon = [30]
            for a in expert_actions:
                if 'LookDown' in a['action']:
                    camera_horizon.append(camera_horizon[-1] + 15)
                elif 'LookUp' in a['action']:
                    camera_horizon.append(camera_horizon[-1] - 15)
                else:
                    camera_horizon.append(camera_horizon[-1])
            n_frame = len(camera_horizon)
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'rgb'))) == n_frame
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'seg'))) == n_frame
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'navigation'))) == n_frame
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'interaction'))) == n_frame


            for j in range(n_frame):
                self.images.append([t['task'], j, camera_horizon[j]])
        self.image_size = args.image_size
        self.affordance_class = [
            'Navigate',
            'PickupObject',
            'PutObject',
            'OpenObject',
            'CloseObject',
            'ToggleObjectOn',
            'ToggleObjectOff',
            'SliceObject',
        ]
        self.n_cls = len(self.affordance_class)
        
        print(f'Split = {split}. Tasks = {len(self.tasks)}. Images = {len(self.images)}. N_class = {self.n_cls}')

    def __len__(self):
        return len(self.images)
        # return 100
    
    def __getitem__(self, index):
        task, step, camera_horizon = self.images[index]
        rgb_path = os.path.join(self.data_root,'replay',self.split,task,'rgb','step_%05d.png' % step)
        seg_path = os.path.join(self.data_root,'replay',self.split,task,'seg','step_%05d.pkl' % step)
        nav_path = os.path.join(self.data_root,'replay',self.split,task,'navigation', 'step_%05d.png' % step)
        int_path = os.path.join(self.data_root,'replay',self.split,task,'interaction','step_%05d_int.pkl' % step)
        
        rgb = cv2.cvtColor(cv2.imread(rgb_path),cv2.COLOR_BGR2RGB)
        h,w, _ = rgb.shape
        # rgb = np.array(resize(to_pil_image(rgb), (self.image_size, self.image_size)))
        rgb = np.transpose(rgb, (2,0,1))
        rgb = torch.from_numpy(rgb).float().contiguous() / 255


        aff_label = []
        nav_label = cv2.imread(nav_path,cv2.IMREAD_GRAYSCALE) > 0
        aff_label.append(nav_label)
        segs = pkl.load(open(seg_path, 'rb'))

        objectid2mask = {}
        for seg in segs['object']:
            objectId = seg['objectId']
            mask = segs['mask'] == seg['label']
            objectid2mask[objectId] = mask
        
        aff_int_info = pkl.load(open(int_path,'rb'))
        for aff_int in ['PickupObject','PutObject','OpenObject','CloseObject','ToggleObjectOn','ToggleObjectOff','SliceObject']:
            aff_obj_ids = aff_int_info[aff_int]
            mask = np.zeros((h,w), dtype=bool)
            for objectId in aff_obj_ids:
                if objectId in objectid2mask:
                    mask = np.logical_or(mask, objectid2mask[objectId])
            aff_label.append(mask)

        aff_label = np.stack(aff_label, -1)
        aff_label = torch.from_numpy(aff_label).float()
        return rgb,aff_label

    @staticmethod
    def collate_fn(blobs):
        rgb, aff_label = zip(*blobs)
        rgb = torch.stack(rgb,0)
        aff_label = torch.stack(aff_label,0)
        return rgb, aff_label
    




class AlfredAffordanceDatasetV2(Dataset):
    def __init__(self,args,split,data_root = './data') -> None:
        super().__init__()
        self.args = args
        self.split = split
        self.data_root = data_root
        self.tasks = json.load(open(os.path.join(data_root,'splits/oct21.json')))[split]
        self.tasks = remove_alfred_repeat_tasks(self.tasks)
        self.images = [] 
        for i,t in enumerate(self.tasks):
            traj_data = json.load(open(os.path.join('./data/json_2.1.0/', split, t['task'], 'traj_data.json')))
            expert_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] ]
            camera_horizon = [30]
            for a in expert_actions:
                if 'LookDown' in a['action']:
                    camera_horizon.append(camera_horizon[-1] + 15)
                elif 'LookUp' in a['action']:
                    camera_horizon.append(camera_horizon[-1] - 15)
                else:
                    camera_horizon.append(camera_horizon[-1])
            n_frame = len(camera_horizon)
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'rgb'))) == n_frame
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'seg'))) == n_frame
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'navigation'))) == n_frame
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'interaction'))) == n_frame


            for j in range(n_frame):
                self.images.append([t['task'], j, camera_horizon[j]])
        self.image_size = args.image_size
        self.affordance_class = ALFRED_AFFORDANCE_LIST
        self.affordance2class = {name : i + 1 for i,name in enumerate(self.affordance_class)}
        self.n_cls = len(self.affordance_class)        
        print(f'Split = {split}. Tasks = {len(self.tasks)}. Images = {len(self.images)}. N_class = {self.n_cls}')

    def mask_to_box(self,mask):
        h,w = mask.shape
        xs, ys = np.meshgrid(np.arange(h),np.arange(w))
        ys = ys[mask]
        xs = xs[mask]
        return [xs.min(), ys.min(), xs.max()+1, ys.max()+1]

    def __len__(self):
        return len(self.images)
        # return 100
    
    def __getitem__(self, index):
        task, step, camera_horizon = self.images[index]
        rgb_path = os.path.join(self.data_root,'replay',self.split,task,'rgb','step_%05d.png' % step)
        seg_path = os.path.join(self.data_root,'replay',self.split,task,'seg','step_%05d.pkl' % step)
        nav_path = os.path.join(self.data_root,'replay',self.split,task,'navigation', 'step_%05d.png' % step)
        int_path = os.path.join(self.data_root,'replay',self.split,task,'interaction','step_%05d_int.pkl' % step)
        
        rgb = cv2.cvtColor(cv2.imread(rgb_path),cv2.COLOR_BGR2RGB)
        h,w, _ = rgb.shape
        # rgb = np.array(resize(to_pil_image(rgb), (self.image_size, self.image_size)))
        rgb = np.transpose(rgb, (2,0,1))
        rgb = torch.from_numpy(rgb).float().contiguous() / 255


        class_list = []
        mask_list  = []
        box_list   = []
        
        nav_mask = cv2.imread(nav_path,cv2.IMREAD_GRAYSCALE) > 0
        if np.any(nav_mask):
            class_list.append(self.affordance2class['Navigate'])
            mask_list.append(nav_mask.astype(float))
            box_list.append(self.mask_to_box(nav_mask))

        segs = pkl.load(open(seg_path, 'rb'))
        objectid2mask = {}
        for seg in segs['object']:
            objectId = seg['objectId']
            mask = segs['mask'] == seg['label']
            objectid2mask[objectId] = mask

        aff_int_info = pkl.load(open(int_path,'rb'))
        for aff_int in ['PickupObject','PutObject', 'SliceObject', 'OpenObject', 'CloseObject']:
            aff_obj_ids = aff_int_info[aff_int]
            # mask = np.zeros((h,w), dtype=bool)
            for objectId in aff_obj_ids:
                if objectId in objectid2mask:
                    mask = objectid2mask[objectId]
                    class_list.append(self.affordance2class[aff_int])
                    mask_list.append(mask.astype(float))
                    box_list.append(self.mask_to_box(mask))
    
        for objectId,mask in objectid2mask.items():
            if 'Lamp' in objectId:
                class_list.append(self.affordance2class['ToggleObject'])
                mask_list.append(mask.astype(float))
                box_list.append(self.mask_to_box(mask))
            if 'Fridge' in objectId:
                class_list.append(self.affordance2class['CoolObject'])
                mask_list.append(mask.astype(float))
                box_list.append(self.mask_to_box(mask))
            if 'Microwave' in objectId:
                class_list.append(self.affordance2class['HeatObject'])
                mask_list.append(mask.astype(float))
                box_list.append(self.mask_to_box(mask))
            if 'Sink' in objectId:
                class_list.append(self.affordance2class['CleanObject'])
                mask_list.append(mask.astype(float))
                box_list.append(self.mask_to_box(mask))

        if len(class_list) == 0:
            target = None
        else:
            target = {}
            target["boxes"] = torch.tensor(box_list,dtype=torch.int64)
            target["masks"] = torch.from_numpy(np.stack(mask_list,0)).float()
            target["labels"] = torch.tensor(class_list,dtype=torch.int64)

        return rgb, target

    @staticmethod
    def collate_fn(blobs):
        blobs = [(img,tgt) for img,tgt in blobs if tgt is not None]
        assert len(blobs) > 0
        images, targets = zip(*blobs)
        # images = torch.stack(images,0)
        images = list(images)
        for i,tgt in enumerate(targets):
            tgt["image_id"] = i
        return images, targets




class AlfredAffordanceDatasetV3(Dataset):
    def __init__(self,args,split,data_root = './data') -> None:
        super().__init__()
        self.args = args
        self.split = split
        self.data_root = data_root
        self.tasks = json.load(open(os.path.join(data_root,'splits/oct21.json')))[split]
        self.tasks = remove_alfred_repeat_tasks(self.tasks)
        self.images = [] 
        for i,t in enumerate(self.tasks):
            traj_data = json.load(open(os.path.join('./data/json_2.1.0/', split, t['task'], 'traj_data.json')))
            expert_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] ]
            camera_horizon = [30]
            for a in expert_actions:
                if 'LookDown' in a['action']:
                    camera_horizon.append(camera_horizon[-1] + 15)
                elif 'LookUp' in a['action']:
                    camera_horizon.append(camera_horizon[-1] - 15)
                else:
                    camera_horizon.append(camera_horizon[-1])
            n_frame = len(camera_horizon)
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'rgb'))) == n_frame
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'seg'))) == n_frame
            assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'aff'))) == n_frame
            for j in range(n_frame):
                self.images.append([t['task'], j, camera_horizon[j]])
        self.image_size = args.image_size
        self.affordance_class = ALFRED_AFFORDANCE_LIST
        self.affordance2class = {name : i + 1 for i,name in enumerate(self.affordance_class)}
        self.n_cls = len(self.affordance_class)        
        print(f'Split = {split}. Tasks = {len(self.tasks)}. Images = {len(self.images)}. N_class = {self.n_cls}')

    def mask_to_box(self,mask):
        h,w = mask.shape
        xs, ys = np.meshgrid(np.arange(h),np.arange(w))
        ys = ys[mask]
        xs = xs[mask]
        return [xs.min(), ys.min(), xs.max()+1, ys.max()+1]

    def __len__(self):
        return len(self.images)
        
        
    def __getitem__(self, index):
        task, step, camera_horizon = self.images[index]
        rgb_path = os.path.join(self.data_root,'replay',self.split,task,'rgb','step_%05d.png' % step)
        seg_path = os.path.join(self.data_root,'replay',self.split,task,'seg','step_%05d.pkl' % step)
        aff_path = os.path.join(self.data_root,'replay',self.split,task,'aff','step_%05d_aff.pkl' % step)
        
        rgb = cv2.cvtColor(cv2.imread(rgb_path),cv2.COLOR_BGR2RGB)
        h,w, _ = rgb.shape
        # rgb = np.array(resize(to_pil_image(rgb), (self.image_size, self.image_size)))
        rgb = np.transpose(rgb, (2,0,1))
        rgb = torch.from_numpy(rgb).float().contiguous() / 255

        affs = pkl.load(open(aff_path,'rb'))
        nav_mask = affs['Navigable']

        segs = pkl.load(open(seg_path, 'rb'))
        objectid2mask = {}
        for seg in segs['object']:
            objectId = seg['objectId']
            mask = segs['mask'] == seg['label']
            objectid2mask[objectId] = mask

        aff_label = [nav_mask]
        for aff_int in self.affordance_class[1:]:
            if aff_int == 'ToggleObject':
                mask = np.zeros((h,w), dtype=bool)                
                for seg in segs['object']:
                    objectType = seg['objectType']
                    if objectType in ['FloorLamp','DeskLamp']:
                        mask = np.logical_or(mask, segs['mask'] == seg['label'])            
            else:
                aff_obj_ids = affs[aff_int]
                mask = np.zeros((h,w), dtype=bool)
                for objectId in aff_obj_ids:
                    if objectId in objectid2mask:
                        mask = np.logical_or(mask, objectid2mask[objectId])
            aff_label.append(mask)
        
        aff_label = np.stack(aff_label,-1)
        aff_label = torch.from_numpy(aff_label).float()
        # print(aff_label.shape)
        return rgb, aff_label

    @staticmethod
    def collate_fn(blobs):
        rgb, aff_label = zip(*blobs)
        rgb = torch.stack(rgb,0)
        aff_label = torch.stack(aff_label,0)
        return rgb, aff_label


