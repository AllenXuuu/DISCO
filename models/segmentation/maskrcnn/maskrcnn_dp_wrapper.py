import torch
from torch.nn.parallel._functions import Scatter

class MaskRCNNInferenceDataParallelWrapper(torch.nn.DataParallel):
    def gather(self, outputs, output_device):
        
        gather_out = []
        for out in outputs:
            gather_out += out
        for res in gather_out:
            for k in res:
                if len(res[k]) == 0:
                    continue
                res[k] = res[k].to('cuda:%d' % output_device)
        return gather_out
