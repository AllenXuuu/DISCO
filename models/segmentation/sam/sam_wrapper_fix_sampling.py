import cv2
import numpy as np
import torch
import torch.nn as nn
from .sam_wrapper import SamSegTrainWrapper
from .segment_anything.build_sam import sam_model_registry
from .segment_anything.utils.amg import (
   build_point_grid,
   batched_mask_to_box
)
from torchvision.ops.boxes import batched_nms


class SamSegFixedPointWrapper(SamSegTrainWrapper):
    def __init__(self, name, n_cls, device, n_per_side = 32, nms_thresh = 0.5) -> None:
        super(SamSegFixedPointWrapper,self).__init__(name,n_cls)
        self.to(device)
        self.n_per_side = n_per_side
        self.nms_thresh = nms_thresh

        with torch.no_grad():
            points = build_point_grid(n_per_side=self.n_per_side)
            points = torch.from_numpy(points * self.image_size).unsqueeze(1).to(device)
            points_labels = torch.ones(points.shape[0], dtype = torch.int, device=device).unsqueeze(1)
            n_pt = points.shape[0]
            self.sparse_embeddings, self.dense_embeddings = self.sam_model.prompt_encoder(
                points=(points, points_labels),
                boxes=None,
                masks=None,
            )
            self.sparse_embeddings = self.sparse_embeddings.detach().cpu()
            self.dense_embeddings = self.dense_embeddings.detach().cpu()
            
            self.image_pe = self.sam_model.prompt_encoder.get_dense_pe().detach().cpu()
    
    
    def forward(self,images):
        n_pt  = self.sparse_embeddings.shape[0]
        n_img = images.shape[0]
        c = self.sparse_embeddings.shape[-1]
        device = images.device

        ########################### image side
        with torch.no_grad():
            images = self.sam_model.preprocess(images)
            images = self.sam_model.image_encoder(images)
            # N_img * C * H * W

            images = torch.repeat_interleave(images, n_pt, dim=0)
            images += torch.tile(self.dense_embeddings.to(device),(n_img,1,1,1))
            images = images.detach()
            _,_,h,w = images.shape

        image_pe = torch.repeat_interleave(self.image_pe, n_pt * n_img, dim=0).to(device)

        tokens = torch.cat([
            self.sam_model.mask_decoder.iou_token.weight, 
            self.sam_model.mask_decoder.mask_tokens.weight[:1]
            ], dim=0)
        tokens = tokens.unsqueeze(0).expand(n_pt, tokens.shape[0], c)
        tokens = torch.cat((tokens, self.sparse_embeddings.to(device)), dim=1)
        tokens = torch.cat((tokens, self.class_token.weight.expand(n_pt, 1, c)), dim=1) # N_pt * N_token * c
        tokens = torch.tile(tokens, (n_img,1,1))
        
        hs, src = self.sam_model.mask_decoder.transformer(images, image_pe, tokens)
        iou_token_out   = hs[:, 0,  :]
        mask_tokens_out = hs[:, 1,  :]
        class_token_out = hs[:,-1,  :]
        
        src = src.transpose(1, 2).view(n_pt * n_img , c, h, w)
        src = self.sam_model.mask_decoder.output_upscaling(src)
        b,c,h,w = src.shape
        
        # hyper_in_list = []
        # for i in range(self.sam_model.mask_decoder.num_mask_tokens):
        #     hyper_in_list.append(self.sam_model.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # hyper_in = torch.stack(hyper_in_list, dim=1)

        hyper_in = self.sam_model.mask_decoder.output_hypernetworks_mlps[0](mask_tokens_out)
        hyper_in = hyper_in.unsqueeze(1)

        pred_masks = (hyper_in @ src.view(n_img * n_pt, c, h * w))
        pred_masks = pred_masks.view(n_img, n_pt, h, w)

        pred_iou = self.sam_model.mask_decoder.iou_prediction_head(iou_token_out)
        pred_iou = pred_iou.view(n_img, n_pt, self.sam_model.mask_decoder.num_mask_tokens)[:,:,0]


        pred_scores = self.class_prediction_head(class_token_out).view(n_img,n_pt,self.n_cls)
        pred_scores = torch.softmax(pred_scores, -1)
        pred_scores, pred_classes = torch.max(pred_scores, dim = -1)


        batched_output = []
        pred_masks = pred_masks > 0
        boxes = batched_mask_to_box(pred_masks)
        for i in range(n_img):
            is_pos = pred_classes[i] > 0

            pos_box   = boxes[i][is_pos].float()
            pos_score = pred_scores[i][is_pos]
            pos_class = pred_classes[i][is_pos]
            pos_mask  = pred_masks[i][is_pos]

            keep_by_nms = batched_nms(
                pos_box,
                pos_score,
                pos_class,  # categories
                iou_threshold=self.nms_thresh,
            )
            if torch.any(keep_by_nms):
                output = {
                    'label' : pos_class[keep_by_nms] - 1,
                    'mask'  : pos_mask[keep_by_nms],
                    'score' : pos_score[keep_by_nms], 
                }
            else:
                output = {
                    'label' : [],
                    'mask'  : [],
                    'score' : [], 
                }
            batched_output.append(output)


        return batched_output