import cv2
import numpy as np
import torch
from utils import utils, arguments
import tqdm
import torch.nn.functional as F
import cv2,time
from models.depthunet import SimpleUNET
from datasets.depth_dataset import build_depth_dataloader
from utils.optimizer import build_lr_scheduler,build_optimizer
from itertools import chain
from collections import OrderedDict
import os
import logging
from tensorboardX import SummaryWriter




def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)

    args.name = __file__.split('/')[-1].split('.')[0] + '_' + args.split
    logging_dir = os.path.join('logs', args.name)
    print('Logging to ====> %s' % logging_dir)
    os.makedirs(logging_dir,exist_ok=True)

    logger = utils.build_logger(filepath=os.path.join(logging_dir,'log.txt'),verbose=True)
    device = torch.device('cuda:%d' % args.gpu[0])
    use_dp = len(args.gpu) > 1
    if use_dp:
        args.bz = args.bz * len(args.gpu)
        
    args.image_size = 300

    dataloader = build_depth_dataloader('AlfredDepthDataset', split=args.split, args=args)

    model = SimpleUNET(args)
    model = model.to(device).eval()
    
    logger.info(f'Load Resume <==== {args.resume}')
    resume = torch.load(args.resume,map_location='cpu')
    model.load_state_dict(resume['model'])

    if use_dp:
        model = torch.nn.DataParallel(model,device_ids=args.gpu)

    all_pred = []
    all_gt   = []
    all_hor  = []
    for i, blobs in enumerate(tqdm.tqdm(dataloader,ncols=100,desc=f'{args.split} Inference')):
        blobs = [x.to(device) for x in blobs]
        rgb, gt_depth, camera_horizon = blobs
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_depth = model(rgb)
        blobs = [x.cpu() for x in blobs]
        rgb, gt_depth, camera_horizon = blobs
        pred_depth = pred_depth.cpu()
        
        all_pred.append(pred_depth)
        all_gt.append(gt_depth)
        all_hor.append(camera_horizon)

    all_pred = torch.cat(all_pred, 0)
    all_gt   = torch.cat(all_gt,   0)
    all_hor  = torch.cat(all_hor,  0)


    metrics = utils.get_depth_estimation_metrics(all_gt,all_pred,all_hor)

    logger.info(f'=================== Split [{args.split}] Depth Estimation Result =====================')
    for k,v in metrics.items():
        key = k if isinstance(k,str) else f'Horizon={k}'
        logger.info(f'{args.split}. {key:12s} Error {v:.2f}mm')


if __name__ == '__main__':
    main()