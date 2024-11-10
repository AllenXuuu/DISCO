import random
import numpy as np
import torch
import json
import os
from collections import OrderedDict
from datetime import datetime
import tqdm
import logging
from torch.nn.parallel._functions import Scatter
import multiprocessing as mp
import imgviz
from imgviz import utils as imgviz_utils

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def timestr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def build_logger(filepath=None, verbose=False):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    if verbose:
        handler1 = logging.StreamHandler()
        handler1.setLevel(logging.DEBUG)
        handler1.setFormatter(formatter)
        logger.addHandler(handler1)
    if filepath is not None:
        handler2 = logging.FileHandler(filename=filepath, mode='w')
        handler2.setFormatter(formatter)
        handler2.setLevel(logging.DEBUG)
        logger.addHandler(handler2)

    return logger

################################ alfred utils

def remove_alfred_repeat_tasks(tasks):
    names = set()
    out = []
    for t in tasks:
        if t['task'] in names:
            continue
        else:
            names.add(t['task'])
            out.append(t)
    return out


def load_alfred_tasks(args):
    tasks = json.load(open("data/splits/oct21.json"))[args.split]
    for i in range(len(tasks)):
        tasks[i]['split'] = {
            'train'        : 'TR',
            'valid_seen'   : 'VS',
            'valid_unseen' : 'VU',
            'tests_seen'   : 'TS',
            'tests_unseen' : 'TU',
        }[args.split]
        tasks[i]['task_index'] = tasks[i]['split'] + '%05d' % i
        if args.task_from is not None and i < args.task_from:
            tasks[i] = None
            continue        
        if args.task_to   is not None and i > args.task_to:
            tasks[i] = None
            continue
        if args.task_name is not None and args.task_name not in tasks[i]['task']:
            tasks[i] = None
            continue        
        if args.task_index is not None and i not in args.task_index:
            tasks[i] = None
            continue
    tasks = [t for t in tasks if t is not None]
    
    if args.remove_repeat:
        tasks = remove_alfred_repeat_tasks(tasks)
    
    for i in tqdm.tqdm(range(len(tasks)),ncols=100,desc='Load Alfred Tasks'):
        tasks[i].update(json.load(open(os.path.join(
                './data/json_2.1.0/', args.split, tasks[i]['task'], 'traj_data.json'))))
          
        if args.task_type is not None and tasks[i]['task_type'] not in args.task_type:
            tasks[i] = None
            continue
        
        if args.task_scene is not None and int(tasks[i]['scene']['floor_plan'].replace('FloorPlan','')) not in args.task_scene:
            tasks[i] = None
            continue
        
    tasks = [t for t in tasks if t is not None]

    return tasks



def visualize_segmentation(rgb, labels, bboxes=None, masks = None, captions = None, color_map = None, alpha = 0.5, line_width = 2, font_size=8):
    if len(labels) == 0:
        return rgb
    
    if isinstance(labels,torch.Tensor):
        labels = labels.cpu().numpy()
        
    if isinstance(bboxes,torch.Tensor):
        bboxes = bboxes.cpu().numpy()
        
    if isinstance(masks,torch.Tensor):
        masks = masks.cpu().numpy()
        
    masks = masks.astype(bool) # type: ignore
        
    if color_map is None:
        color_map = imgviz.label.label_colormap()
    
    
    n_instances = len(labels)
    areas = []
    

    
    
    assert bboxes is not None or masks is not None
    for i in range(n_instances):
        if masks is not None:
            areas.append(masks[i].sum())
        elif bboxes is not None:
            areas.append( (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1]) )

    if bboxes is None:
        bboxes = np.zeros((len(labels), 4), dtype=float)
        for i, mask in enumerate(masks): # type: ignore
            if mask.sum() == 0:
                continue
            where = np.argwhere(mask)
            (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
            bbox = y1, x1, y2, x2
            bboxes[i] = bbox
    
    
    dst = rgb.copy()
    if masks is not None:
        for i in sorted(range(n_instances),key=lambda x : areas[x]): # type: ignore
            color = color_map[labels[i]  % len(color_map)]
            mask = masks[i]
            maskviz = masks[i][:, :, None] * color.astype(float)
            dst = dst.copy()
            dst[mask] = (1 - alpha) * rgb[mask].astype(float) + alpha * maskviz[mask]
    
    dst = imgviz_utils.numpy_to_pillow(dst)
    for instance_id in range(n_instances):
        label = labels[instance_id]
        color = color_map[label % len(color_map)]
        
        y1, x1, y2, x2 = bboxes[instance_id]
        if (y2 - y1) * (x2 - x1) == 0:
            continue
        aabb1 = np.array([y1, x1], dtype=int)
        aabb2 = np.array([y2, x2], dtype=int)
        imgviz.draw.rectangle_(
            dst,
            aabb1,
            aabb2,
            outline=color,
            width=line_width,
        )

        if captions is not None:
            loc = None
            for loc in ["lt+", "lt"]:
                y1, x1, y2, x2 = imgviz.draw.text_in_rectangle_aabb(
                    img_shape=(dst.height, dst.width),
                    loc=loc,
                    text=captions[instance_id],
                    size=font_size,
                    aabb1=aabb1,
                    aabb2=aabb2,
                    font_path=None,
                )
                if y1 >= 0 and x1 >= 0 and y2 < dst.height and x2 < dst.width:
                    break
            imgviz.draw.text_in_rectangle_(
                img=dst,
                loc=loc,
                text=captions[instance_id],
                size=font_size,
                background=color,
                aabb1=aabb1,
                aabb2=aabb2,
                font_path=None,
            )
    return imgviz_utils.pillow_to_numpy(dst)




def get_alfred_metrics(tasks, results):
    def _get_metrics(results):
        NALL = len(results)
        results = [a for a in results if a is not None]
        if len(results) == 0:
            return {}
        
        N = len(results)
        S = sum([a['success'] for a in results])
        SR = S / N
        GC0 = sum([a['completed_goal_conditions'] for a in results])
        GC1 = sum([a['total_goal_conditions'] for a in results])
        GC = GC0 / GC1
        
        TOTAL_LEN = sum([a['path_len_expert'] for a in results]) 
        PLW_SR = sum([a['success'] * min(1, a['path_len_expert']/a['path_len_agent']) * a['path_len_expert'] for a in results]) / TOTAL_LEN
        PLW_GC = sum([a['completed_goal_conditions'] / a['total_goal_conditions'] * min(1, a['path_len_expert']/a['path_len_agent']) * a['path_len_expert'] for a in results]) / TOTAL_LEN
        
        ret = {
            'N' : N,
            'NALL' : NALL,
            'SR' : SR * 100,
            'GC' : GC * 100,
            'PLW_SR' : PLW_SR * 100,
            'PLW_GC' : PLW_GC * 100
        }
        return ret
        
    assert len(tasks) == len(results)
    if 'task_type' in tasks[0]:
        for key in [
            'look_at_obj_in_light',
            'pick_and_place_simple',
            'pick_two_obj_and_place',
            'pick_and_place_with_movable_recep',
            'pick_cool_then_place_in_recep',
            'pick_heat_then_place_in_recep',
            'pick_clean_then_place_in_recep'
        ]:
            # print(tasks[0])
            metrics = _get_metrics([r for t,r in zip(tasks,results) if t['task_type'] == key])   
            if len(metrics) == 0:
                # print(key, 'nan')
                pass
            else:
                key = key + f" [{metrics['N']}/{metrics['NALL']}]"
                print(f"{key:45s} SR={metrics['SR']:.2f} GC={metrics['GC']:.2f} PLW_SR={metrics['PLW_SR']:.2f} PLW_GC={metrics['PLW_GC']:.2f}")
    
    metrics = _get_metrics(results)
    key = f"ALL [{metrics['N']}/{metrics['NALL']}]"
    print(f"{key:45s} SR={metrics['SR']:.2f} GC={metrics['GC']:.2f} PLW_SR={metrics['PLW_SR']:.2f} PLW_GC={metrics['PLW_GC']:.2f}")
    
    







############################## segmentation_metrics

def get_point_segmentation_metrics(all_gt,all_pred,iou_threshold = 0.5):
    all_gt_mask    = np.concatenate([a['mask'] for a in all_gt   if len(a['mask'])>0],0)
    all_pred_mask  = np.concatenate([a['mask'] for a in all_pred if len(a['mask'])>0],0)
    all_gt_label   = np.concatenate([a['label'] for a in all_gt   if len(a['label'])>0],0)
    all_pred_label = np.concatenate([a['label'] for a in all_pred if len(a['label'])>0],0)
    
    intersection = (all_gt_mask * all_pred_mask).sum(-1).sum(-1)
    union = all_gt_mask.sum(-1).sum(-1) + all_pred_mask.sum(-1).sum(-1) - intersection
    iou = intersection / union
    
    mean_iou = iou.mean()
    match = np.logical_and(iou > iou_threshold, all_gt_label == all_pred_label)
    recall = match.mean() * 100
    
    iou_per_class = {}
    recall_per_class = {}

    for i in np.unique(all_gt_label):
        iou_per_class[i] = iou[all_gt_label == i].mean()
        instances = match[all_gt_label == i]
        recall_per_class[i] = '%d / %d = %.2f' % (instances.sum(),instances.shape[0],instances.mean()*100) + '%'

    return mean_iou, recall, iou_per_class, recall_per_class


def get_instance_segmentation_metrics(all_gt,all_pred,iou_threshold = 0.5):
    
    def instance_match(blobs):
        gts,preds,iou_threshold = blobs
        n_gt,n_pred = len(gts['label']), len(preds['label'])
        
        if n_gt == 0 or n_pred == 0:
            return np.zeros(n_pred), np.zeros(n_gt)
        
        class_match_matrix = preds['label'][:,None] == gts['label'][None,:]
        
        pred_match = np.zeros(n_pred)
        gt_match = np.zeros(n_gt)
        
        for i in sorted(range(n_pred),key=lambda x : - preds['score'][x]):
            for j in range(n_gt):
                if gt_match[j]:
                    continue
                if not class_match_matrix[i,j]:
                    continue
                intersection = (preds['mask'][i] * gts['mask'][j]).sum()
                union = preds['mask'][i].sum() + gts['mask'][j].sum() - intersection
                iou = intersection / union
                if iou >= iou_threshold:
                    pred_match[i] = 1
                    gt_match[j] = 1
        return pred_match, gt_match
    
    iou_threshold = [iou_threshold for _ in range(len(all_gt))]
    blobs = list(zip(all_gt,all_pred,iou_threshold))
    blobs = tqdm.tqdm(blobs,ncols=100,desc='Match',leave=False)
    match_result = list(map(instance_match, blobs))

    pred_match, gt_match = zip(*match_result)
    pred_match = np.concatenate(pred_match,0)
    gt_match   = np.concatenate(gt_match,0)

    pred_class = np.concatenate([pred['label'] for pred in all_pred],0)
    gt_class  = np.concatenate([gt['label'] for gt in all_gt],0)
    
    recall    = gt_match.mean() * 100
    precision = pred_match.mean() * 100
    
    precision_per_class = {}
    recall_per_class = {}
    
    for i in np.unique(gt_class):
        instances = gt_match[gt_class == i]
        recall_per_class[i] = '%d / %d = %.2f' % (instances.sum(),instances.shape[0],instances.mean()*100) + '%'
    for i in np.unique(pred_class):
        if i < 0:
            continue
        instances = pred_match[pred_class == i]
        precision_per_class[i] = '%d / %d = %.2f' % (instances.sum(),instances.shape[0],instances.mean()*100) + '%'

    return precision, recall, precision_per_class, recall_per_class




############################## depth_metrics


def get_depth_estimation_metrics(gt,pred,horizon=None):
    keys = []
    if horizon is not None:
        keys += [k for k in sorted(np.unique(horizon).tolist()) ]
    if len(keys) > 1:
        keys.append('total')
    result = OrderedDict()
    for k in keys:
        if k == 'total':
            error = np.abs((gt - pred)).mean()
        else:
            mask = horizon == k
            error = np.abs((gt[mask] - pred[mask])).mean()
        result [k] = error

    return result




############################## depth_metrics


def get_affordance_metrics(gt,pred):
    b,h,w,c = pred.shape
    precision_per_class = []
    recall_per_class = []
    for i in range(c):
        tp = np.logical_and(gt[...,i], pred[...,i]).sum()
        fp = np.logical_and(np.logical_not(gt[...,i]), pred[...,i]).sum()
        fn = np.logical_and(gt[...,i], np.logical_not(pred[...,i])).sum()
        precision_per_class.append(tp / (tp + fp) * 100)
        recall_per_class.append(tp / (tp + fn) * 100)
    
    recall = np.mean(recall_per_class)
    precision = np.mean(precision_per_class)
    return precision, recall, precision_per_class, recall_per_class
