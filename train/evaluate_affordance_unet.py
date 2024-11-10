import cv2
import numpy as np
import torch
from utils import utils, arguments
import tqdm
import torch.nn.functional as F
import cv2,time
from models.affunet import AffordanceUNET
from datasets.affordance_dataset import build_aff_dataloader
from utils.optimizer import build_lr_scheduler,build_optimizer
from itertools import chain
from collections import OrderedDict
from utils.alfred_interest_objects import ALFRED_AFFORDANCE_LIST
import os
import logging
from torch.nn.parallel._functions import Scatter
from tensorboardX import SummaryWriter


  
def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)

    args.name = __file__.split('/')[-1].split('.')[0]
    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir,exist_ok=True)

    logger = utils.build_logger(filepath=os.path.join(logging_dir,'log.txt'),verbose=True)
    device = torch.device('cuda:%d' % args.gpu[0])

    
    use_dp = len(args.gpu) > 1
    if use_dp:
        args.bz = args.bz * len(args.gpu)
    args.image_size  = 300
    args.num_workers = 8
    vs_dataloader  = build_aff_dataloader('AlfredAffordanceDataset', split='valid_seen', args=args)
    vu_dataloader  = build_aff_dataloader('AlfredAffordanceDataset', split='valid_unseen', args=args)
    print(f'VSeen   #Dataset {len(vs_dataloader.dataset)} #Dataloader {len(vs_dataloader)}')
    print(f'VUnseen #Dataset {len(vu_dataloader.dataset)} #Dataloader {len(vu_dataloader)}')
    
    n_cls = vs_dataloader.dataset.n_cls
    model = AffordanceUNET(n_cls)
    model = model.to(device)
    
    if use_dp:
        model = torch.nn.DataParallel(model,device_ids=args.gpu)
        
    assert args.resume is not None
    print('Load Resume from <===== %s' % args.resume)
    resume = torch.load(args.resume,map_location='cpu')
    if use_dp:
        model.module.load_state_dict(resume['model'])
    else:
        model.load_state_dict(resume['model'])
        
    
        
    model.eval()    
    for split, dataloader in [('valid_seen', vs_dataloader), ('valid_unseen', vu_dataloader)]:
    # for split, dataloader in [('valid_unseen', vu_dataloader)]:
        logger.info(f'=================== {split} Test Start ==================')
        tp, fp, fn = torch.ones(n_cls), torch.ones(n_cls), torch.ones(n_cls)
        for i, blobs in enumerate(tqdm.tqdm(dataloader,ncols=100,desc = split)):
            blobs = [x.to(device) for x in blobs]
            rgb, gt_label = blobs
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred_label = model(rgb)
                    thresh = [0,0,0,0,0,0,0,0]
                    thresh = torch.tensor(thresh).to(device)
                    pred_label = pred_label > thresh
            gt_label = gt_label > 0   
            
            tp += torch.logical_and(gt_label, pred_label).float().sum(0).sum(0).sum(0).cpu()
            fp += torch.logical_and(torch.logical_not(gt_label), pred_label).float().sum(0).sum(0).sum(0).cpu()
            fn += torch.logical_and(gt_label, torch.logical_not(pred_label)).float().sum(0).sum(0).sum(0).cpu()

        logger.info(f'=================== {split} Inference Done ==================')
        
        precision_per_class = tp / (tp + fp)* 100
        recall_per_class    = tp / (tp + fn) * 100
        
        precision = torch.mean(precision_per_class)
        recall = torch.mean(recall_per_class)
        
        for i,name in enumerate(ALFRED_AFFORDANCE_LIST):
            logger.info(f'{split} {i:03d}-{name:20s} Precision {precision_per_class[i]:.2f} Recall {recall_per_class[i]:.2f}')
        logger.info(f'{split} Total Precision {precision:.2f} Recall {recall:.2f}')
            

if __name__ == '__main__':
    main()
