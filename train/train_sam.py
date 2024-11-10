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
from itertools import chain
from collections import OrderedDict
import os
import logging
from torch.nn.parallel._functions import Scatter
from tensorboardX import SummaryWriter


def compute_loss(gt_mask, gt_class, pred_mask, pred_cls, pred_iou, gamma =2, eps = 1e-7):
    pred_mask = pred_mask[gt_class > 0]
    gt_mask   = gt_mask[gt_class > 0]
    pred_iou  = pred_iou[gt_class > 0]
    
    prob_mask = torch.sigmoid(pred_mask)
    prob_true = torch.where(gt_mask == 1, prob_mask, 1 - prob_mask)
    ########### dice loss
    dice_loss = 1 - 2 * (gt_mask * prob_mask).sum(-1).sum(-1) / (eps + gt_mask.sum(-1).sum(-1)  + prob_mask.sum(-1).sum(-1))
    dice_loss = dice_loss.mean()

    ########### focal loss    
    focal_loss = - (1 - prob_true).pow(gamma) * torch.log(prob_true + eps)
    # focal_loss = focal_loss.sum(-1).sum(-1)
    focal_loss = focal_loss.mean(-1).mean(-1)
    focal_loss = focal_loss.mean()

    ########### class loss
    class_loss = torch.nn.functional.cross_entropy(pred_cls, gt_class)

    ########### iou loss
    bin_mask = (pred_mask.detach() > 0).float()
    intersection = (gt_mask * bin_mask).sum(-1).sum(-1)
    union  = gt_mask.sum(-1).sum(-1) + bin_mask.sum(-1).sum(-1) - intersection
    gt_iou = intersection / union
    iou_loss = F.mse_loss(pred_iou, gt_iou)
    gt_iou = gt_iou.mean()

    return dice_loss, focal_loss, class_loss, iou_loss, gt_iou

  
def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)

    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir,exist_ok=True)

    logger = utils.build_logger(filepath=os.path.join(logging_dir,'log.txt'),verbose=True)
    writer = SummaryWriter(os.path.join(logging_dir,'tb'))
    device = torch.device('cuda:%d' % args.gpu[0])
    
    use_dp = len(args.gpu) > 1
    if use_dp:
        args.bz = args.bz * len(args.gpu)
    args.image_size  = 1024
    args.mask_size   = 256
    args.center_only = True
    dataloader = build_seg_dataloader('AlfredSegPointDataset', split='train', args=args)

    model = SamSegTrainWrapper('vit_b',n_cls=dataloader.dataset.n_obj + 1)
    model = model.to(device).train()
    if use_dp:
        model = SamDataParallelWrapper(model,device_ids=args.gpu)

    optimizer = build_optimizer(args, model.parameters())
    lr_scheduler = build_lr_scheduler(args,optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    
    if args.resume is not None:
        print('Load Resume from <===== %s' % args.resume)
        resume = torch.load(args.resume,map_location='cpu')
        # for pg in resume['optimizer']['param_groups']:
        #     pg['lr'] = args.base_lr
        optimizer.load_state_dict(resume['optimizer'])
        # lr_scheduler.load_state_dict(resume['lr_scheduler'])
        if use_dp:
            model.module.load_state_dict(resume['model'])
        else:
            model.load_state_dict(resume['model'])
        
    
    for ep in range(1, args.epoch + 1):
        for i, blobs in enumerate(dataloader):
            step = 1 + i + (ep - 1) * len(dataloader)
            blobs = [ x.to(device) for x in blobs]
            with torch.cuda.amp.autocast(enabled=args.amp):
                image, repeat, gt_class, gt_mask, points = blobs
                pred_iou, pred_mask, pred_cls = model(blobs)
                dice_loss, focal_loss, class_loss, iou_loss, gt_iou = compute_loss(gt_mask,gt_class,pred_mask,pred_cls,pred_iou)

                loss    = [dice_loss, focal_loss, class_loss, iou_loss]
                weights = [args.loss_dice, args.loss_focal, args.loss_class, args.loss_iou]
                loss    =  [a * b for a,b in zip(loss,weights)]
                dice_loss, focal_loss, class_loss, iou_loss = loss
                loss = sum(loss)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            result = OrderedDict([
                ['loss',       loss.item()],
                ['loss_dice',  dice_loss.item()],
                ['loss_focal', focal_loss.item()],
                ['loss_class', class_loss.item()],
                ['loss_iou',   iou_loss.item()],
                ['gt_iou',     gt_iou.item()],
                ['lr',         optimizer.param_groups[0]['lr']],
            ])


            for key,val in result.items():
                writer.add_scalar(key, val, step)
            if step % args.report_freq == 0:
                report = 'Epoch %.4f Step %d. LR %.2e  Loss %.4f. LossDice %.4f. LossFocal %.4f. LossClass %.4f. LossIoU %.4f' % (
                    step/len(dataloader), step , result['lr'], result['loss'], result['loss_dice'], result['loss_focal'], result['loss_class'], result['loss_iou']
                )
                logger.info(report)


        if ep % args.save_freq == 0:
            save_path = os.path.join(logging_dir, 'epoch_%d.pth' % ep)
            print('Save Resume to ====> %s' % save_path)
            torch.save({
                'model' : model.state_dict() if not use_dp else model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                }, save_path)


if __name__ == '__main__':
    main()