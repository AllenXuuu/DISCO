import cv2
import numpy as np
import torch
from utils import utils, arguments
import tqdm
import torch.nn.functional as F
import cv2,time
from models.resnet_policy import FinePolicyNet
from datasets.fineaction_dataset import build_fineaction_dataloader
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
    
    train_dataloader = build_fineaction_dataloader('AlfredFineActionDataset', split='train', args=args)
    args.num_workers = args.num_workers // 8
    logger.info(f'Train   #Dataset {len(train_dataloader.dataset)} #Dataloader {len(train_dataloader)}')
    
    model = FinePolicyNet(
        num_input = 5, 
        num_classes = train_dataloader.dataset.n_cls,
        num_object = train_dataloader.dataset.n_obj,
        cosine=args.cosine)
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
        
    
    for ep in range(1, args.epoch + 1):
        logger.info(f'=================== Epoch {ep} Train Start ==================')
        model.train()
        all_pred = []
        all_labels = []
        all_objects = []
        for i, blobs in enumerate(train_dataloader):
            step = 1 + i + (ep - 1) * len(train_dataloader)
            blobs = [x.to(device) for x in blobs]
            inputs, label, object_classes = blobs
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(inputs, object_classes)
                loss = torch.nn.functional.cross_entropy(logits,label)
            optimizer.zero_grad()
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            all_pred.append(torch.argmax(logits,dim=1).cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_objects.append(object_classes.cpu().numpy())
            loss = loss.data.cpu().numpy().tolist()
            
            result = OrderedDict([
                ['lr',         optimizer.param_groups[0]['lr']],
                ['loss',       np.mean(loss)],
            ])

            for key,val in result.items():
                writer.add_scalar(key, val, step)
            if step % args.report_freq == 0 or i == len(train_dataloader) - 1:
                report = 'Epoch %.4f Step %d. LR %.2e  Loss %.4f.' % (
                    step/len(train_dataloader), step , result['lr'], result['loss']
                    )
                logger.info(report)
        all_labels  = np.concatenate(all_labels,0)
        all_pred    = np.concatenate(all_pred,0)
        all_objects = np.concatenate(all_objects,0)

        for i,oname in enumerate(train_dataloader.dataset.object_classes):
            mask = (all_objects == i)
            if any(mask):
                acc = np.mean(all_pred[mask] == all_labels[mask]) * 100
                logger.info(f'{i:02d} {oname:20s} {mask.sum():05d} Acc {acc:.2f}')
            else:
                logger.info(f'{i:02d} {oname:20s} No Object')

        for c in range(train_dataloader.dataset.n_cls):
            recall    = (all_pred[all_labels == c] == c).sum() / ((all_labels == c).sum() + 1e-7) * 100
            precision = (all_labels[all_pred == c] == c).sum() / ((all_pred == c).sum() + 1e-7) * 100
            logger.info(f'{c} {train_dataloader.dataset.action_classes[c]} Precision {precision:.2f} Recall {recall:.2f}')
        acc = np.mean(all_pred == all_labels) * 100
        logger.info(f'Accuracy {acc:.2f}')
        save_path = os.path.join(logging_dir, 'epoch_%d.pth' % ep)
        logger.info('Save Resume to ====> %s' % save_path)
        torch.save({
            'model' : model.state_dict() if not use_dp else model.module.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler.state_dict(),
            }, save_path)
            
        logger.info(f'=================== Epoch {ep} Train Done ==================\n')
        
if __name__ == '__main__':
    main()