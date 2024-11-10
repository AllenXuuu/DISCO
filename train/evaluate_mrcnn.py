import cv2
import numpy as np
import torch,torchvision
from models.segmentation.sam.sam_wrapper import SamSegTrainWrapper
from models.segmentation.sam.sam_dp_wrapper import SamDataParallelWrapper
from datasets.segmentation_dataset import build_seg_dataloader
from utils import utils, arguments
import tqdm
import torch.nn.functional as F
import cv2,time
from utils.optimizer import build_lr_scheduler,build_optimizer
from itertools import chain
from collections import OrderedDict
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
import os
import logging
from torch.nn.parallel._functions import Scatter
from tensorboardX import SummaryWriter
from .train_mrcnn import MRCNNDataParallelWrapper
torch.multiprocessing.set_sharing_strategy('file_system')


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
    args.mask_size   = 300
    args.num_workers = args.num_workers // 8
    vs_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_seen', args=args)
    vu_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_unseen', args=args)
    print('>>> ValidSeen #Dataset %d #Dataloader %d' % (len(vs_dataloader.dataset), len(vs_dataloader)  ))
    print('>>> ValidUnseen #Dataset %d #Dataloader %d' % (len(vu_dataloader.dataset), len(vu_dataloader)  ))
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights = None, weights_backbone = None, num_classes=vs_dataloader.dataset.n_obj + 1, box_score_thresh = 0.5)
    

    model = model.to(device)
    if use_dp:
        model = MRCNNDataParallelWrapper(model,device_ids=args.gpu)

    
    
    assert args.resume is not None
    logger.info('Load Resume from <===== %s' % args.resume)
    resume = torch.load(args.resume,map_location='cpu')
    if use_dp:
        model.module.load_state_dict(resume['model'])
    else:
        model.load_state_dict(resume['model'])
        
    model.eval()
    for split, dataloader in [('valid_seen', vs_dataloader), ('valid_unseen', vu_dataloader)]:
        
        logger.info(f'=================== {split} Test Start ==================')
        all_pred, all_gt = [], []
        for i, blobs in enumerate(dataloader):
            images, targets = blobs
            images  = [img.to(device) for img in images]
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    output = model(images)
            for tgt in targets:
                all_gt.append({
                    'label' : tgt['labels'].numpy(),
                    'mask'  : tgt['masks'].numpy()
                })
            for pred in output:
                all_pred.append({
                    'label' : pred['labels'].data.cpu().numpy(),
                    'mask'  : pred['masks'].data.cpu().numpy(),
                    'score' : pred['scores'].data.cpu().numpy(),
                })
        logger.info(f'=================== {split} Inference Done ==================')
        precision, recall, precision_per_class, recall_per_class = utils.get_instance_segmentation_metrics(all_gt, all_pred)
        logger.info(f'=================== Split [{split}] Segmentation Result =====================')
        for i, name in enumerate(ALFRED_INTEREST_OBJECTS):
            class_idx = i + 1
            if class_idx in precision_per_class or class_idx in recall_per_class:
                p = precision_per_class.get(class_idx,'nan')
                r = recall_per_class.get(class_idx,'nan')
                logger.info(f'{split} {class_idx:03d}-{name:20s} Precision {p:20s} Recall {r:20s}')
        logger.info(f'{split} Precision {precision:.2f} Recall {recall:.2f}')

if __name__ == '__main__':
    main()
