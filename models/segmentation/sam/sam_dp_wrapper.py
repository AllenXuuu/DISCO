import torch
import torch.nn as nn
from torch.nn.parallel._functions import Scatter
from ..maskrcnn.maskrcnn_dp_wrapper import MaskRCNNInferenceDataParallelWrapper

SamFixedPointInferenceDataParallelWrapper = MaskRCNNInferenceDataParallelWrapper

class SamDataParallelWrapper(torch.nn.DataParallel):
    def scatter(
        self,
        inputs,
        kwargs,
        device_ids,
    ):
        blobs = inputs[0]
        images, repeat, gt_class, gt_mask, points = blobs
        images = Scatter.apply(device_ids, None, 0, images)
        repeat = Scatter.apply(device_ids, None, 0 ,repeat)
        chunksize = [a.sum().item() for a in repeat]
        gt_class = [data.to(images[idx].device) for idx, data in enumerate(torch.split(gt_class, chunksize, dim=0))]
        gt_mask  = [data.to(images[idx].device) for idx, data in enumerate(torch.split(gt_mask, chunksize, dim=0))]
        points   = [data.to(images[idx].device) for idx, data in enumerate(torch.split(points, chunksize, dim=0))]

        inputs = list(zip(images, repeat, gt_class, gt_mask, points))
        inputs = [[a] for a in inputs]
        kwargs = [kwargs for _ in range(len(inputs))]
        return inputs,kwargs
