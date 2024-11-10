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

    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir,exist_ok=True)

    logger = utils.build_logger(filepath=os.path.join(logging_dir,'log.txt'),verbose=True)
    writer = SummaryWriter(os.path.join(logging_dir,'tb'))
    device = torch.device('cuda:%d' % args.gpu[0])
    
    use_dp = len(args.gpu) > 1
    if use_dp:
        args.bz = args.bz * len(args.gpu)
    args.image_size  = 300
    
    train_dataloader = build_aff_dataloader('AlfredAffordanceDatasetV3', split='train', args=args)
    args.num_workers = args.num_workers // 8
    vs_dataloader  = build_aff_dataloader('AlfredAffordanceDatasetV3', split='valid_seen', args=args)
    vu_dataloader  = build_aff_dataloader('AlfredAffordanceDatasetV3', split='valid_unseen', args=args)
    
    print(f'Train   #Dataset {len(train_dataloader.dataset)} #Dataloader {len(train_dataloader)}')
    print(f'VSeen   #Dataset {len(vs_dataloader.dataset)} #Dataloader {len(vs_dataloader)}')
    print(f'VUnseen #Dataset {len(vu_dataloader.dataset)} #Dataloader {len(vu_dataloader)}')
    
    model = AffordanceUNET(train_dataloader.dataset.n_cls)
    model = model.to(device)
    
    if use_dp:
        model = torch.nn.DataParallel(model,device_ids=args.gpu)

    optimizer = build_optimizer(args, model.parameters())
    lr_scheduler = build_lr_scheduler(args,optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.resume is not None:
        print('Load Resume from <===== %s' % args.resume)
        resume = torch.load(args.resume,map_location='cpu')
        optimizer.load_state_dict(resume['optimizer'])
        if use_dp:
            model.module.load_state_dict(resume['model'])
        else:
            model.load_state_dict(resume['model'])
    
    data_time = []
    model_time = []
    for ep in range(1, args.epoch + 1):
        logger.info(f'=================== Epoch {ep} Train Start ==================')
        model.train()
        pos_count = np.zeros(len(ALFRED_AFFORDANCE_LIST_V3))
        all_pixel_count = 0
        t1 = time.time()
        for i, blobs in enumerate(train_dataloader):
            t2 = time.time()
            data_time.append(t2 - t1)
            step = 1 + i + (ep - 1) * len(train_dataloader)
            blobs = [x.to(device) for x in blobs]
            rgb, label = blobs
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = model(rgb, label)
                loss = loss.mean(0)
            pos_count += label.sum(0).sum(0).sum(0).cpu().numpy()
            all_pixel_count += label.shape[0] * label.shape[1] * label.shape[2]

            optimizer.zero_grad()
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            loss = loss.data.cpu().numpy().tolist()
            
            result = OrderedDict([
                ['lr',         optimizer.param_groups[0]['lr']],
                ['loss',       np.mean(loss)],
            ])
            for i, class_name in enumerate(ALFRED_AFFORDANCE_LIST_V3):
                result[class_name] = loss[i]
            for key,val in result.items():
                writer.add_scalar(key, val, step)

            t1 = time.time()
            model_time.append(t1 - t2)
            if step % args.report_freq == 0 or i == len(train_dataloader) - 1:
                data_time = np.mean(data_time)
                model_time = np.mean(model_time)
                report = 'Epoch %.4f Step %d. LR %.2e  Loss %.4f. DataTime %.4f ModelTime %.4f' % (
                    step/len(train_dataloader), step , result['lr'], result['loss'], data_time, model_time
                    )
                report += '\n                          Loss:     '
                for i, class_name in enumerate(ALFRED_AFFORDANCE_LIST_V3):
                    key = class_name.replace('Object','')
                    report += f'{key} {loss[i]:.4f} '
                # report = f'Epoch {step/len(train_dataloader):.4f}   Step {step:d}     Time:     Data {data_time}       Model {model_time}'
                
                logger.info(report)


                data_time = []
                model_time = []
        print(pos_count, all_pixel_count)
        pos_dist = ['%.4f' % (a/all_pixel_count) for a in pos_count]
        logger.info(f'Pos Distribution {pos_dist}')
        logger.info(f'=================== Epoch {ep} Train Done ==================\n')
        
        if ep % args.test_freq == 0:
            logger.info(f'=================== Epoch {ep} Test Start ==================')
            model.eval()
            
            for split, dataloader in [('valid_seen', vs_dataloader), ('valid_unseen', vu_dataloader)]:
                all_gt, all_pred = [], []
                for i, blobs in enumerate(dataloader):
                    rgb, gt_label = blobs
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=args.amp):
                            pred_label = model(rgb.to(device)) 
                            pred_label = pred_label > 0
                    gt_label = gt_label > 0
                    all_pred.append(pred_label.cpu())
                    all_gt.append(gt_label.cpu())
                
                all_pred = torch.cat(all_pred, 0).numpy()
                all_gt   = torch.cat(all_gt,   0).numpy()

                logger.info(f'=================== Epoch {ep} {split} Inference Done ==================')
                precision, recall, precision_per_class, recall_per_class = utils.get_affordance_metrics(all_gt,all_pred)
                for i,name in enumerate(ALFRED_AFFORDANCE_LIST_V3):
                    logger.info(f'{split} {i:03d}-{name:20s} Precision {precision_per_class[i]:.2f} Recall {recall_per_class[i]:.2f}')
                logger.info(f'{split} Total Precision {precision:.2f} Recall {recall:.2f}')
                

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