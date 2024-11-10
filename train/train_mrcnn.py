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
torch.multiprocessing.set_sharing_strategy('file_system')

class MRCNNDataParallelWrapper(torch.nn.DataParallel):
    def scatter(
        self,
        inputs,
        kwargs,
        device_ids,
    ):
        n_dev = len(device_ids)
        n_bz = len(inputs[0])
        size = [ (i + 1) * n_bz // n_dev  - i * n_bz // n_dev  for i in range(n_dev) ]
        size = [ s for s in size if s > 0]

        cur = 0
        new_inputs = []
        for i,s in enumerate(size):
            device = torch.device('cuda:%d' % device_ids[i])
            if len(inputs) == 1:
                images = inputs[0]
                for i in range(cur,cur+s):
                    images[i] = images[i].to(device)
                new_inputs.append([images[cur:cur+s]])
            else:
                images, targets = inputs
                for i in range(cur,cur+s):
                    images[i] = images[i].to(device)
                    for k in targets[i]:
                        if isinstance(targets[i][k],torch.Tensor):
                            targets[i][k] = targets[i][k].to(device)
                new_inputs.append([images[cur:cur+s], targets[cur:cur+s]])
            cur += s
        kwargs = [kwargs for _ in range(len(new_inputs))]
        return new_inputs,kwargs

    def gather(self, outputs, output_device):
        device = torch.device('cuda:%d' % output_device)
        if isinstance(outputs[0],list):
            final_out = []
            for output in outputs:
                for img_result in output:
                    for k in img_result:
                        img_result[k] = img_result[k].to(device)
                    final_out.append(img_result)
            return final_out
        elif isinstance(outputs[0],dict):
            final_out = {}
            keys = list(outputs[0].keys())
            for k in keys:
                assert 'loss' in k
                final_out[k] = [output[k].to(device) for output in outputs]
                final_out[k] = sum(final_out[k]) / len(final_out[k])
            return final_out
        else:
            raise NotImplementedError    
        



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
    args.mask_size   = 300
    train_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='train', args=args)
    args.num_workers = args.num_workers // 8
    vs_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_seen', args=args)
    vu_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_unseen', args=args)
    print('>>> Train #Dataset %d #Dataloader %d' % (len(train_dataloader.dataset), len(train_dataloader) ))
    print('>>> ValidSeen #Dataset %d #Dataloader %d' % (len(vs_dataloader.dataset), len(vs_dataloader)  ))
    print('>>> ValidUnseen #Dataset %d #Dataloader %d' % (len(vu_dataloader.dataset), len(vu_dataloader)  ))
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights = None, weights_backbone = None, num_classes=train_dataloader.dataset.n_obj + 1, box_score_thresh = 0.5)
    
    pretrained_ckpt_path = './weights/mrcnn_torchvision/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
    predtrained_ckpt = torch.load(pretrained_ckpt_path)
    predtrained_ckpt = {k:v for k,v in predtrained_ckpt.items() if 'roi_heads.box_predictor' not in k and 'roi_heads.mask_predictor.mask_fcn_logits' not in k}
    load_result = model.load_state_dict(predtrained_ckpt, strict=False)
    print(load_result)

    model = model.to(device)
    if use_dp:
        model = MRCNNDataParallelWrapper(model,device_ids=args.gpu)

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
            images, targets = blobs
            images  = [img.to(device) for img in images]
            targets = [{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
            # print(targets[0]['boxes'])
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images, targets)  # Returns losses and detections
            loss_weights = {
                'loss_classifier' : 1,
                'loss_box_reg' : 1,
                'loss_mask' : 1,
                'loss_objectness' : 5,
                'loss_rpn_box_reg' : 1,
            }

            output = {k:loss_weights[k] * v for k,v in output.items()}
            loss = sum(output.values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            for key,val in output.items():
                writer.add_scalar(key, val, step)
            if step % args.report_freq == 0 or i == len(train_dataloader) - 1:
                report = 'Epoch %.4f Step %d. LR %.2e  loss %.4f.' % (
                    step/len(train_dataloader), step , optimizer.param_groups[0]['lr'], loss,
                )
                logger.info(report)
        logger.info(f'=================== Epoch {ep} Train Done ==================\n')

        if ep % args.test_freq == 0:
            logger.info(f'=================== Epoch {ep} Test Start ==================')
            model.eval()
            # for split, dataloader in [('valid_seen', vs_dataloader), ('valid_unseen', vu_dataloader)]:
            for split, dataloader in [('valid_unseen', vu_dataloader)]:            
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
                precision, recall, precision_per_class, recall_per_class = utils.get_instance_segmentation_metrics(all_gt, all_pred)
                logger.info(f'=================== Split [{split}] Segmentation Result =====================')
                for i, name in enumerate(ALFRED_INTEREST_OBJECTS):
                    class_idx = i + 1
                    if class_idx in precision_per_class or class_idx in recall_per_class:
                        p = precision_per_class.get(class_idx,'nan')
                        r = recall_per_class.get(class_idx,'nan')
                        logger.info(f'{split} {class_idx:03d}-{name:20s} Precision {p:20s} Recall {r:20s}')
                logger.info(f'{split} Precision {precision:.2f} Recall {recall:.2f}')

            save_path = os.path.join(logging_dir, 'epoch_%d.pth' % ep)
            logger.info('Save Resume to ====> %s' % save_path)
            torch.save({
                'model' : model.state_dict() if not use_dp else model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                }, save_path)


if __name__ == '__main__':
    main()
