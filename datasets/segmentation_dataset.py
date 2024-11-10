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

def build_seg_dataloader(name, split, args):
    if name == 'AlfredSegPointDataset':
        dataset    = AlfredSegPointDataset(args,split)
        dataloader = DataLoader(
            dataset,
            batch_size = args.bz,
            shuffle = (split == 'train'),
            num_workers= args.num_workers,
            collate_fn=dataset.collate_fn
        )
    elif name == 'AlfredSegImageDataset':
        dataset    = AlfredSegImageDataset(args,split)
        dataloader = DataLoader(
            dataset,
            batch_size = args.bz,
            shuffle = (split == 'train'),
            num_workers= args.num_workers,
            collate_fn=dataset.collate_fn
        )
    else:
        raise NotImplementedError(name)
    
    return dataloader


class AlfredSegPointDataset(Dataset):
    def __init__(self,args,split,data_root = './data') -> None:
        super().__init__()
        self.args = args
        self.split = split
        self.data_root = data_root
        self.tasks = json.load(open(os.path.join(data_root,'splits/oct21.json')))[split]
        self.tasks = remove_alfred_repeat_tasks(self.tasks)
        self.images = []
        for i,t in enumerate(self.tasks):
            n_frame = len(os.listdir(os.path.join(data_root,'replay',split,t['task'],'rgb')))
            for j in range(n_frame):
                self.images.append([t['task'], j])

        self.image_size = args.image_size
        self.mask_size  = args.mask_size
        
        # class = 0 background
        # class = 1,2, .... object
        self.object2class = {
           obj : i + 1 for i, obj in enumerate(ALFRED_INTEREST_OBJECTS)
        }
        self.n_obj = len(self.object2class) 

        print(f'Split = {split}. Tasks = {len(self.tasks)}. Images = {len(self.images)}. Objects = {len(self.object2class)}')
    
    def __len__(self):
        return len(self.images)
        # return 1000
    
    def __getitem__(self, index):
        task, step = self.images[index]
        rgb_path = os.path.join(self.data_root,'replay',self.split,task,'rgb','step_%05d.png' % step)
        seg_path = os.path.join(self.data_root,'replay',self.split,task,'seg','step_%05d.pkl' % step)
        
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.array(resize(to_pil_image(image), (self.image_size, self.image_size)))
        image = np.transpose(image, (2,0,1))
        image = torch.from_numpy(image).float().contiguous()

        class_list = []
        mask_list  = []
        point_list = []
        seg_info = pkl.load(open(seg_path,'rb'))

        for seg in seg_info['object']:
            objectType = seg['objectType']
            objectClass = self.object2class[objectType]
            mask = (seg_info['mask'] == seg['label'])
            x1,y1,x2,y2 = self.mask_to_box(mask)
            if self.args.center_only:
                if x2 <= 0.1 * mask.shape[1] or x1 >= 0.9 * mask.shape[1] or y2 <= 0.1 * mask.shape[0] or y1 >= 0.9 * mask.shape[0]:
                    # print(x1,y1,x2,y2)
                    continue
            mask = mask.astype(np.uint8) * 255
            mask = np.array(resize(to_pil_image(mask), (self.mask_size, self.mask_size)))
            mask = mask > 0
            point = self.sample_points_from_mask(mask)
            class_list.append(objectClass)
            mask_list.append(mask)
            point_list.append(point)
        pos_size = len(point_list)


        negative_size   = int(self.args.neg_rate * len(point_list))
        negative_size   = np.clip(negative_size, self.args.neg_min, self.args.neg_max)
        # print('Pos', pos_size, 'totalObj', len(seg_info['object']),  'Neg', negative_size)
        if negative_size > 0:
            negative_mask = (seg_info['mask'] == 0).astype(np.uint8) * 255
            negative_mask = np.array(resize(to_pil_image(negative_mask), (self.mask_size, self.mask_size)))
            negative_mask = negative_mask >= 0
            negative_points = self.sample_points_from_mask(negative_mask, negative_size)
            for i in range(negative_size):
                class_list.append(-1)
                mask_list.append(negative_mask)
            point_list.append(negative_points)

        if len(class_list) > 0:
            class_list = torch.tensor(class_list)
            mask_list  = torch.from_numpy(np.stack(mask_list,0)).float()
            point_list = torch.from_numpy(np.concatenate(point_list,0)).float()
        else:
            class_list = torch.empty((0), dtype = torch.int64)
            mask_list  = torch.empty((0,self.mask_size,self.mask_size), dtype = torch.float32)
            point_list = torch.empty((0,2), dtype = torch.float32)

        return image, class_list, mask_list, point_list
    
    def mask_to_box(self,mask):
        h,w = mask.shape
        xs, ys = np.meshgrid(np.arange(h),np.arange(w))
        ys = ys[mask]
        xs = xs[mask]
        return [xs.min(), ys.min(), xs.max()+1, ys.max()+1]

    def sample_points_from_mask(self,mask,size = 1):
        h,w = mask.shape
        xs, ys = np.meshgrid(np.arange(h),np.arange(w))
        ys = ys[mask]
        xs = xs[mask]
        index = np.random.randint(0, len(xs),size=size)
        xs = (xs[index] + 0.5) / w
        ys = (ys[index] + 0.5) / h
        points = np.stack([xs,ys],1)
        return points
    
    @staticmethod
    def collate_fn(blobs):
        image, class_list, mask_list, point_list = zip(*blobs)
        image = torch.stack(image,0)
        repeat = [point.shape[0] for point in point_list]
        repeat = torch.tensor(repeat)
        class_list = torch.cat(class_list,0)
        mask_list = torch.cat(mask_list,0)
        point_list = torch.cat(point_list,0)
        
        return image, repeat, class_list, mask_list, point_list
    
    

class AlfredSegImageDataset(AlfredSegPointDataset):
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
    
    def __getitem__(self, index):
        task, step = self.images[index]
        rgb_path = os.path.join(self.data_root,'replay',self.split,task,'rgb','step_%05d.png' % step)
        seg_path = os.path.join(self.data_root,'replay',self.split,task,'seg','step_%05d.pkl' % step)
        
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.array(resize(to_pil_image(image), (self.image_size, self.image_size)))
        image = np.transpose(image, (2,0,1))
        image = torch.from_numpy(image).float().contiguous() / 255

        seg_info = pkl.load(open(seg_path,'rb'))
        class_list = []
        mask_list  = []
        box_list   = []
        # print(len(seg_info['object']))
        for seg in seg_info['object']:
            objectType = seg['objectType']
            objectClass = self.object2class[objectType]
            mask = (seg_info['mask'] == seg['label'])
            x1,y1,x2,y2 = self.mask_to_box(mask)
            box_list.append([x1,y1,x2,y2])
            mask = mask.astype(np.uint8) * 255
            mask = np.array(resize(to_pil_image(mask), (self.mask_size, self.mask_size)))
            mask = mask > 0
            class_list.append(objectClass)
            mask_list.append(mask)

        if len(class_list) == 0:
            target = None
        else:
            target = {}
            target["boxes"] = torch.tensor(box_list,dtype=torch.int64)
            target["masks"] = torch.from_numpy(np.stack(mask_list,0)).float()
            target["labels"] = torch.tensor(class_list,dtype=torch.int64)

        return image, target
   