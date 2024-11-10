import cv2
import numpy as np
import torch
from models.segmentation.sam.sam_wrapper_fix_sampling import SamSegFixedPointWrapper
from models.segmentation.sam.sam_dp_wrapper import SamFixedPointInferenceDataParallelWrapper
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

    model = SamSegFixedPointWrapper(
        'vit_b',
        n_cls=dataloader.dataset.n_obj + 1, 
        device = device, 
        n_per_side = args.n_per_side)
    model = model.to(device).eval()
    
    assert args.resume is not None
    logger.info('Load Resume from <===== %s' % args.resume)
    resume = torch.load(args.resume,map_location='cpu')
    model.load_state_dict(resume['model'])

    
    if use_dp:
        model = SamFixedPointInferenceDataParallelWrapper(model,device_ids=args.gpu)

    
    all_pred = []
    all_gt   = []
    for i, blobs in enumerate(tqdm.tqdm(dataloader,ncols=100,desc=f'{args.split} Inference')):
        images, repeat, gt_classes, gt_masks, points = blobs
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                predictions = model(images.to(device))

        gt_classes = torch.split(gt_classes,repeat.numpy().tolist(),dim = 0)
        gt_masks   = torch.split(gt_masks,repeat.numpy().tolist(),dim = 0)
        for gt_class, gt_mask in zip(gt_classes, gt_masks):
            gt_mask = gt_mask[gt_class > 0]
            gt_class = gt_class[gt_class > 0] - 1
            all_gt.append({
                'label' : gt_class.numpy(),
                'mask'  : gt_mask.numpy()
            })
        for pred in predictions:
            for k in pred:
                if len(pred[k]) == 0:
                    continue
                pred[k] = pred[k].detach().data.cpu().numpy()
            all_pred.append(pred)
    precision, recall, precision_per_class, recall_per_class = utils.get_instance_segmentation_metrics(all_gt, all_pred)


    logger.info(f'=================== Split [{args.split}] Segmentation Result =====================')
    for i,name in enumerate(ALFRED_INTEREST_OBJECTS):
        if i in precision_per_class or i in recall_per_class:
            p = precision_per_class.get(i,'nan')
            r = recall_per_class.get(i,'nan')
            logger.info(f'{args.split} {i:03d}-{name:20s} Precision {p:20s} Recall {r:20s}')
    logger.info(f'{args.split} Precision {precision:.2f} Recall {recall:.2f}')

if __name__ == '__main__':
    main()