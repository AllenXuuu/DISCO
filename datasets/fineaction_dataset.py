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

def build_fineaction_dataloader(name, split, args):
    if name == 'AlfredFineActionDataset':
        dataset    = AlfredFineActionDataset(args,split)
        dataloader = DataLoader(
            dataset,
            batch_size = args.bz,
            shuffle = (split == 'train'),
            num_workers= args.num_workers,
            collate_fn=AlfredFineActionDataset.collate_fn
        )
    else:
        raise NotImplementedError(name)
    
    return dataloader


class AlfredFineActionDataset(Dataset):
    def __init__(self,args,split) -> None:
        super().__init__()
        self.states = json.load(open('./data/scene_parse/fine_action_train.json'))
        self.action_classes = ['Interact','MoveAhead','MoveLeft','MoveRight','MoveBack','RotateLeft','RotateRight','LookUp','LookDown']
        self.action2class = {act : i for i, act in enumerate(self.action_classes)}
        self.n_cls = len(self.action_classes)
        self.frames = []

        self.object_classes = ALFRED_INTEREST_OBJECTS
        self.object2class = {
           obj : i for i, obj in enumerate(ALFRED_INTEREST_OBJECTS)
        }
        self.n_obj = len(ALFRED_INTEREST_OBJECTS)
        for idx, st in enumerate(self.states):
            fn = f'./data/scene_parse/fine_action_train_frames/{idx:08d}_rgb.png'  
            if os.path.exists(fn):
                objectType = st['interactId'].split('|')[0]
                if objectType in ['Sink','Bathtub']:
                    objectType += 'Basin'
                if 'Slice' in st['interactId']:
                    objectType += 'Slice'
                self.frames.append({
                    'rgb'   : fn,
                    'depth' : fn.replace('_rgb','_depth'),
                    'mask'  : fn.replace('_rgb','_mask'),
                    'label' : self.action2class[st['action']],
                    'object_class' : self.object2class[objectType]
                })
        print(f'States: {len(self.states)} Frames {len(self.frames)}')

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,idx):
        item = self.frames[idx]
        rgb   = cv2.cvtColor(cv2.imread(item['rgb']),cv2.COLOR_RGB2BGR)/255
        depth = cv2.imread(item['depth'])[...,[0]]/255
        mask  = cv2.imread(item['mask'])[...,[0]]/255
        inp = np.concatenate([rgb,depth,mask],-1)
        inp = torch.from_numpy(inp).permute(2,0,1)

        label = item['label']
        object_class = item['object_class']
        return inp, label, object_class
    

    @staticmethod
    def collate_fn(blobs):
        inp, label, object_class = zip(*blobs)
        inp = torch.stack(inp,0).float()
        label = torch.tensor(label).long()
        object_class = torch.tensor(object_class).long()
        return inp, label, object_class