import torch
import json
import os
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset,DataLoader
import sys
from utils.utils import remove_alfred_repeat_tasks
import cv2
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
from torchvision.transforms.functional import resize, to_pil_image
import time

def build_depth_dataloader(name, split, args):
    if name == 'AlfredDepthDataset':
        dataset    = AlfredDepthDataset(args,split)
        dataloader = DataLoader(
            dataset,
            batch_size = args.bz,
            shuffle = (split == 'train'),
            num_workers= args.num_workers,
            collate_fn=AlfredDepthDataset.collate_fn
        )
    else:
        raise NotImplementedError(name)
    
    return dataloader


class AlfredDepthDataset(Dataset):
    def __init__(self,args,split,data_root = './data') -> None:
        super().__init__()
        self.args = args
        self.split = split
        self.data_root = data_root
        self.tasks = json.load(open(os.path.join(data_root,'splits/oct21.json')))[split]
        self.tasks = remove_alfred_repeat_tasks(self.tasks)
        self.images = [] 
        
        if args.horizon == [0]:
            for i,t in enumerate(self.tasks):
                traj_data = json.load(open(os.path.join('./data/json_2.1.0/', split, t['task'], 'traj_data.json')))
                folder = os.path.join(data_root,'replay_depth0',split,t['task'],'rgb')
                frames = os.listdir(folder)
                n_frame = len(frames)
                for j in range(n_frame):
                    self.images.append([t['task'], j, args.horizon[0]])
        else:    
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
                assert len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'depth'))) == n_frame
                for j in range(n_frame):
                    self.images.append([t['task'], j, camera_horizon[j]])
        
        
        
        
        
        self.image_size = args.image_size
        
        if args.horizon is not None:
            self.images = [info for info in self.images if info[2] in args.horizon]
        
        print(f'Split = {split}. Tasks = {len(self.tasks)}. Images = {len(self.images)}.')

    def __len__(self):
        return len(self.images) 
        # return 2000
    
    def __getitem__(self, index):
        task, step, camera_horizon = self.images[index]
        
        if self.args.horizon == [0]:
            rgb_path   = os.path.join(self.data_root,'replay_depth0',self.split,task,'rgb','step_%05d.png' % step)
            depth_path = os.path.join(self.data_root,'replay_depth0',self.split,task,'depth','step_%05d.png' % step)
        else:
            rgb_path   = os.path.join(self.data_root,'replay',self.split,task,'rgb','step_%05d.png' % step)
            depth_path = os.path.join(self.data_root,'replay',self.split,task,'depth','step_%05d.png' % step)
        
        rgb = cv2.cvtColor(cv2.imread(rgb_path),cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE).astype(float) / 255 * 5000

        rgb = torch.from_numpy(rgb).permute(2,0,1).float().contiguous()
        depth = torch.from_numpy(depth).float().contiguous()

        return rgb, depth, camera_horizon

    @staticmethod
    def collate_fn(blobs):
        rgb, depth, camera_horizon = zip(*blobs)
        rgb = torch.stack(rgb,0)
        depth = torch.stack(depth,0)
        camera_horizon = torch.tensor(camera_horizon)
        return rgb,depth,camera_horizon