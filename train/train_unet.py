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
from torch.nn.parallel._functions import Scatter
from tensorboardX import SummaryWriter


  
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
    args.image_size  = 300
    
    train_dataloader = build_depth_dataloader('AlfredDepthDataset', split='train', args=args)
    args.num_workers = args.num_workers // 2
    vs_dataloader  = build_depth_dataloader('AlfredDepthDataset', split='valid_seen', args=args)
    vu_dataloader  = build_depth_dataloader('AlfredDepthDataset', split='valid_unseen', args=args)
    
    print(f'Train   #Dataset {len(train_dataloader.dataset)} #Dataloader {len(train_dataloader)}')
    print(f'VSeen   #Dataset {len(vs_dataloader.dataset)} #Dataloader {len(vs_dataloader)}')
    print(f'VUnseen #Dataset {len(vu_dataloader.dataset)} #Dataloader {len(vu_dataloader)}')
    
    model = SimpleUNET(args)
    model = model.to(device)
    
    if use_dp:
        model = torch.nn.DataParallel(model,device_ids=args.gpu)

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
        logger.info(f'=================== Epoch {ep} Train Start ==================')
        model.train()
        for i, blobs in enumerate(train_dataloader):
            step = 1 + i + (ep - 1) * len(train_dataloader)
            blobs = [x.to(device) for x in blobs]
            rgb, gt_depth, camera_horizon = blobs
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = model(rgb, gt_depth)
                loss = loss.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            result = OrderedDict([
                ['loss',       loss.item()],
                ['lr',         optimizer.param_groups[0]['lr']],
            ])

            for key,val in result.items():
                writer.add_scalar(key, val, step)
            if step % args.report_freq == 0 or i == len(train_dataloader) - 1:
                report = 'Epoch %.4f Step %d. LR %.2e  Loss %.4f.' % (
                    step/len(train_dataloader), step , result['lr'], result['loss']
                    )
                logger.info(report)
        logger.info(f'=================== Epoch {ep} Train Done ==================\n')
        
        if ep % args.test_freq == 0:
            logger.info(f'=================== Epoch {ep} Test Start ==================')
            model.eval()
            
            for split, dataloader in [('valid_seen', vs_dataloader), ('valid_unseen', vu_dataloader)]:
                all_gt, all_pred, all_hor = [], [], []
                for i, blobs in enumerate(dataloader):
                    step = 1 + i + (ep - 1) * len(dataloader)
                    blobs = [x.to(device) for x in blobs]
                    rgb, gt_depth, camera_horizon = blobs
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=args.amp):
                            pred_depth = model(rgb)
                            
                    all_pred.append(pred_depth.cpu())
                    all_gt.append(gt_depth.cpu())
                    all_hor.append(camera_horizon.cpu())
                
                all_pred = torch.cat(all_pred, 0)
                all_gt   = torch.cat(all_gt,   0)
                all_hor  = torch.cat(all_hor,  0)

                metrics = utils.get_depth_estimation_metrics(all_gt,all_pred,all_hor)

                # logger.info(f'=================== Epoch [{ep}] Split [{split}] Depth Estimation Result =====================')
                for k,v in metrics.items():
                    key = k if isinstance(k,str) else f'Horizon={k}'
                    logger.info(f'{split}. {key:12s} DepthError {v.item():.2f}mm')

            save_path = os.path.join(logging_dir, 'epoch_%d.pth' % ep)
            logger.info('Save Resume to ====> %s' % save_path)
            torch.save({
                'model' : model.state_dict() if not use_dp else model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                }, save_path)
            
            logger.info(f'=================== Epoch {ep} Test Done ==================\n')


if __name__ == '__main__':
    main()