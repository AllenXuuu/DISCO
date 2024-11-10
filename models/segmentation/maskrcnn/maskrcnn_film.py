import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN

from torchvision.models.resnet import resnet50
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection._utils import overwrite_eps
from torchvision.ops import misc as misc_nn_ops
import pickle
from collections import OrderedDict
import os
import sys
#import alfworld.gen.constants as constants

object_detector_objs = ['AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
receptacle_detector_objs = ['ArmChair', 'Bathtub', 'BathtubBasin', 'Bed', 'Cabinet', 'Cart', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'Desk', 'DiningTable', 'Drawer', 'Dresser', 'Fridge', 'GarbageCan', 'HandTowelHolder', 'LaundryHamper', 'Microwave', 'Ottoman', 'PaintingHanger', 'Safe', 'Shelf', 'SideTable', 'Sink', 'SinkBasin', 'Sofa', 'StoveBurner', 'TVStand', 'Toaster', 'Toilet', 'ToiletPaperHanger', 'TowelHolder']
    
def get_model_instance_segmentation(score_thresh,num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,box_score_thresh=score_thresh)

    anchor_generator = AnchorGenerator(
        sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
    model.rpn.anchor_generator = anchor_generator

    # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def load_pretrained_model(path, which_type, score_thresh=0.5):
    device = 'cpu'
    if which_type == 'obj':
        categories = len(object_detector_objs)
    elif which_type =='recep':
        categories = len(receptacle_detector_objs)
    else:
        raise NotImplementedError(which_type)
    mask_rcnn = get_model_instance_segmentation(score_thresh,categories+1)
    mask_rcnn.load_state_dict(torch.load(path, map_location=device))
    return mask_rcnn


class FILM_MaskRCNN_Seg(torch.nn.Module):
    def __init__(self, threshold, object_list_final) -> None:
        super().__init__()
        self.large_objs = receptacle_detector_objs
        self.small_objs = object_detector_objs
        self.all_detectable_objs = object_detector_objs + receptacle_detector_objs
        self.obj_to_idx = {a:i for i,a in enumerate(object_list_final)}
        self.threshold = threshold
        self.large_model = load_pretrained_model('./weights/film_maskrcnn/receps_lr5e-3_003.pth','recep')
        self.small_model = load_pretrained_model('./weights/film_maskrcnn/objects_lr5e-3_005.pth','obj')

        print('Undetectable: %s' % (set(object_list_final) - set(self.all_detectable_objs)))
        print('Redundant Detection: %s' % (set(self.all_detectable_objs) - set(object_list_final)))
        

    def forward(self,images):
        images = images / 255
        segmentation_large_batch = self.large_model(images)
        segmentation_small_batch = self.small_model(images)
        batch_output = []
        for seg_large, seg_small in zip(segmentation_large_batch,segmentation_small_batch):
            label_large,mask_large,score_large = self.process_seg(seg_large, self.large_objs)
            label_small,mask_small,score_small = self.process_seg(seg_small, self.small_objs)
            labels = label_large + label_small
            masks  = mask_large + mask_small
            scores = score_large + score_small

            if len(labels) > 0:
                labels = torch.tensor(labels)
                scores = torch.tensor(scores)
                masks  = torch.stack(masks,0)

            output = {
                'label' : labels,
                'mask'  : masks,
                'score' : scores, 
            }
            batch_output.append(output)
        return batch_output

    def process_seg(self,seg_result, class_names):
        label_out,mask_out,score_out = [],[],[]

        for i in range(len(seg_result['labels'])):
            if seg_result['labels'][i].item() == len(class_names):
                continue
            obj_name  = class_names[seg_result['labels'][i].item()]
            if obj_name not in self.obj_to_idx:
                continue
            obj_class = self.obj_to_idx[obj_name]
            if seg_result['scores'][i].item() <= self.threshold:
                continue
            mask = (seg_result['masks'][i] > 0.5).float()
            area = mask.sum()
            if area <= 0:
                continue
            
            label_out.append(obj_class)
            mask_out.append(mask)
            score_out.append(seg_result['scores'][i].item())
            
        return label_out, mask_out, score_out