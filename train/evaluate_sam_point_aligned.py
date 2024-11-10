import cv2
import numpy as np
import torch
from models.segmentation.sam.sam_wrapper import SamSegTrainWrapper
from models.segmentation.sam.sam_dp_wrapper import SamDataParallelWrapper
from datasets.segmentation_dataset import build_seg_dataloader
from utils import utils, arguments
import tqdm
import torch.nn.functional as F
import cv2,time
from utils.optimizer import build_lr_scheduler,build_optimizer
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
from itertools import chain
from collections import OrderedDict
import os
import logging
from torch.nn.parallel._functions import Scatter
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
    args.image_size  = 1024
    args.mask_size   = 256
    args.center_only = True
    args.neg_rate = 0
    args.neg_min = 0
    dataloader = build_seg_dataloader('AlfredSegPointDataset', split=args.split, args=args)

    model = SamSegTrainWrapper('vit_b',n_cls=dataloader.dataset.n_obj + 1)
    model = model.to(device).eval()
    
    assert args.resume is not None
    logger.info('Load Resume from <===== %s' % args.resume)
    resume = torch.load(args.resume,map_location='cpu')
    model.load_state_dict(resume['model'])

    
    if use_dp:
        model = SamDataParallelWrapper(model,device_ids=args.gpu)

    
    all_pred = []
    all_gt   = []
    for i, blobs in enumerate(tqdm.tqdm(dataloader,ncols=100,desc=f'{args.split} Inference')):
        blobs = [ x.to(device) for x in blobs]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_iou, pred_masks, pred_classes = model(blobs)
                pred_classes = torch.argmax(pred_classes,-1)
                pred_masks = (pred_masks > 0).float()

        blobs = [ x.cpu() for x in blobs]
        pred_masks, pred_classes = pred_masks.cpu(), pred_classes.cpu()
        images, repeat, gt_classes, gt_masks, points = blobs
        repeat = repeat.numpy().tolist()

        pred_masks   = torch.split(pred_masks,   repeat, dim = 0)
        pred_classes = torch.split(pred_classes, repeat, dim = 0)
        gt_masks     = torch.split(gt_masks,     repeat, dim = 0)
        gt_classes   = torch.split(gt_classes,   repeat, dim = 0)

        for pred_mask, pred_class, gt_mask, gt_class in zip(pred_masks, pred_classes, gt_masks, gt_classes):
            gt_is_pos = gt_class > 0
            all_gt.append({
                'label' : gt_class[gt_is_pos].numpy() - 1,
                'mask'  : gt_mask[gt_is_pos].numpy()
            })
            all_pred.append({
                'label' : pred_class[gt_is_pos].numpy()  - 1,
                'mask'  : pred_mask[gt_is_pos].numpy()
            })
    avg_iou, recall, iou_per_class, recall_per_class = utils.get_point_segmentation_metrics(all_gt, all_pred, iou_threshold=0.5)

    logger.info(f'=================== Split [{args.split}] Segmentation Result =====================')
    for i,name in enumerate(ALFRED_INTEREST_OBJECTS):
        if i in iou_per_class or i in recall_per_class:
            iou = iou_per_class.get(i,'nan')
            rec = recall_per_class.get(i,'nan')
            logger.info(f'{args.split} {i:03d}-{name:20s} IoU {iou:.4f} Recall {rec:s}')
    logger.info(f'{args.split} IoU {avg_iou:.4f} Recall {recall:.2f}')
    
if __name__ == '__main__':
    main()