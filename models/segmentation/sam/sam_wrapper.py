import cv2
import numpy as np
import torch
import torch.nn as nn
from .segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from .segment_anything.build_sam import sam_model_registry
from .segment_anything.utils.transforms import ResizeLongestSide
from .segment_anything.utils.amg import (
   build_point_grid
)


class SamSegTrainWrapper(nn.Module):
    def __init__(self, name, n_cls) -> None:
        super(SamSegTrainWrapper,self).__init__()
        self.n_cls = n_cls
        sam_ckpt_path = {
            'vit_b' : './weights/ckpt/sam_vit_b_01ec64.pth',
            'vit_l' : './weights/ckpt/sam_vit_l_0b3195.pth',
            'vit_h' : './weights/ckpt/sam_vit_h_4b8939.pth',
        }
        self.sam_model = sam_model_registry[name](sam_ckpt_path[name])
        for n,p in self.sam_model.named_parameters():
            if 'encoder' in n:
                p.requires_grad_(False)
        
        self.sam_model.mask_decoder.output_hypernetworks_mlps = self.sam_model.mask_decoder.output_hypernetworks_mlps[:1]

        self.image_size = self.sam_model.image_encoder.img_size
        
        # self.class_token = nn.Parameter(
        #     torch.randn((1, self.sam_model.mask_decoder.transformer_dim)), requires_grad=True)

        self.class_token = nn.Embedding(1,self.sam_model.mask_decoder.transformer_dim)
        self.class_prediction_head = nn.Linear(self.sam_model.mask_decoder.transformer_dim, n_cls)
        
        # self.sparse_embeddings: N_pt * 2 * c
        # self.dense_embeddings : N_pt * C * h * w
    
    def forward(self,blobs):
        device = self.class_token.weight.device
        images, repeat, classes, masks, points = blobs
        # return None,None,None
        n_pt = points.shape[0]
        ########################### image side
        with torch.no_grad():
            points = points * self.image_size
            points_labels = torch.ones(points.shape[0], dtype=torch.int, device=device)

            points = points.unsqueeze(1)
            points_labels = points_labels.unsqueeze(1)

            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=(points, points_labels),
                boxes=None,
                masks=None,
            )
            sparse_embeddings, dense_embeddings = sparse_embeddings.detach(), dense_embeddings.detach()
            image_pe = self.sam_model.prompt_encoder.get_dense_pe().detach()
            image_pe = torch.repeat_interleave(image_pe, n_pt, dim=0)

            images = self.sam_model.preprocess(images)
            images = self.sam_model.image_encoder(images)

            images = torch.repeat_interleave(images, repeat, dim=0)
            images += dense_embeddings
            images = images.detach()
            n_pt, c, h, w = images.shape


        tokens = torch.cat([
            self.sam_model.mask_decoder.iou_token.weight, 
            self.sam_model.mask_decoder.mask_tokens.weight[:1]
            ], dim=0)
        tokens = tokens.unsqueeze(0).expand(n_pt, tokens.shape[0], c)
        tokens = torch.cat((tokens, sparse_embeddings), dim=1)
        tokens = torch.cat((tokens, self.class_token.weight.expand(n_pt, 1, c)), dim=1) # N_pt * N_token * c



        hs, src = self.sam_model.mask_decoder.transformer(images, image_pe, tokens)
        iou_token_out   = hs[:, 0,  :]
        mask_tokens_out = hs[:, 1,  :]
        class_token_out = hs[:,-1,  :]
        
        src = src.transpose(1, 2).view(n_pt , c, h, w)
        src = self.sam_model.mask_decoder.output_upscaling(src)
        
        # hyper_in_list = []
        # for i in range(self.sam_model.mask_decoder.num_mask_tokens):
        #     hyper_in_list.append(self.sam_model.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # hyper_in = torch.stack(hyper_in_list, dim=1)

        hyper_in = self.sam_model.mask_decoder.output_hypernetworks_mlps[0](mask_tokens_out)
        hyper_in = hyper_in.unsqueeze(1)
        b, c, h, w = src.shape

        pred_masks = (hyper_in @ src.view(b, c, h * w))
        pred_masks = pred_masks.view(n_pt, h, w)

        pred_iou = self.sam_model.mask_decoder.iou_prediction_head(iou_token_out)
        pred_iou = pred_iou.view(n_pt, self.sam_model.mask_decoder.num_mask_tokens)[:,0]

        pred_cls = self.class_prediction_head(class_token_out)
        
        return pred_iou, pred_masks, pred_cls