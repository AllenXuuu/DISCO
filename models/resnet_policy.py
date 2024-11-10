import torch
import torch.nn as nn
import torchvision

class CosineClassifier(nn.Module):
    def __init__(self,dimin,dimout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dimin,dimout),requires_grad=True)
        self.t = nn.Parameter(torch.tensor(0.),requires_grad=True)
        
    def forward(self,feat):
        feat   = feat   / torch.norm(feat,dim=-1,p=2).unsqueeze(-1)
        weight = self.weight
        weight = weight / torch.norm(weight,dim=-1,p=2).unsqueeze(-1)
        
        return self.t.exp() * feat @ weight


class FinePolicyNet(nn.Module):
    def __init__(self, num_input, num_classes, num_object, cosine) -> None:
        super().__init__()
        self.n_action = num_classes
        self.n_obj = num_object
        self.backbone = torchvision.models.resnet50(weights=None,num_classes=num_classes * num_object)
        self.backbone.conv1 = nn.Conv2d(num_input, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if cosine:
            self.backbone.fc = CosineClassifier(2048, num_classes * num_object)

    def forward(self,x, object_classes):
        logits = self.backbone(x)
        logits = [logits[i, ocls * self.n_action : (ocls + 1) * self.n_action] for i, ocls in enumerate(object_classes)]
        logits = torch.stack(logits,0)
        return logits
    