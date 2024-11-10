from typing import Dict, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, cmid=None, stride2=1):
        super(DoubleConv, self).__init__()
        if cmid is None:
            cmid = cin
        self.conv1 = nn.Conv2d(cin, cmid, k, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(cmid, cout, k, stride=stride2, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        return x

class UpscaleDoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(UpscaleDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)
        #self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(cout, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2(x)
        if x.shape[2] > output_size[2]:
            x = x[:, :, :output_size[2], :]
        if x.shape[3] > output_size[3]:
            x = x[:, :, :, :output_size[3]]
        return x

class AffordanceUNET(torch.nn.Module):
    def __init__(self, n_cls):
        super(AffordanceUNET, self).__init__()

        self.n_cls = n_cls
        class objectview(object):
            def __init__(self, d):
                self.__dict__ = d
        params = {
            "in_channels": 3,
            "hc1": 256,
            "hc2": 256,
            "out_channels": self.n_cls,
            "stride": 2
        }

        self.p = objectview(params)
        DeconvOp = UpscaleDoubleConv
        ConvOp = DoubleConv

        self.conv1 = ConvOp(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv6 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv1 = DeconvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv5 = DeconvOp(self.p.hc1 * 2, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv6 = DeconvOp(self.p.hc1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        self.norm6 = nn.InstanceNorm2d(self.p.hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()
        #self.deconv5.init_weights()

    def forward(self, rgb, pixel_label = None):
        x1 = self.norm2(self.act(self.conv1(rgb)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))

        x3 = self.dropout(x3)

        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.norm6(self.act(self.conv5(x4)))
        x6 = self.act(self.conv6(x5))

        x6 = self.dropout(x6)

        y5 = self.act(self.deconv1(x6, output_size=x5.size()))
        xy5 = torch.cat([x5, y5], 1)

        y4 = self.dnorm3(self.act(self.deconv2(xy5, output_size=x4.size())))
        xy4 = torch.cat([x4, y4], 1)
        y3 = self.dnorm4(self.act(self.deconv3(xy4, output_size=x3.size())))
        xy3 = torch.cat([x3, y3], 1)
        y2 = self.dnorm4(self.act(self.deconv4(xy3, output_size=x2.size())))
        xy2 = torch.cat([x2, y2], 1)

        xy2 = self.dropout(xy2)

        y1 = self.dnorm5(self.act(self.deconv5(xy2, output_size=x1.size())))
        xy1 = torch.cat([x1, y1], 1)
        scores = self.deconv6(xy1, output_size=rgb.size())
        scores = scores.permute(0, 2, 3, 1)

        
        # return bins
        if pixel_label is None:
            # pred_label = torch.sigmoid(scores)
            # pred_label = scores > 0
            pred_label = scores
            return pred_label
        else:

            b,w,h,c = scores.shape
            assert (b,w,h,c) == pixel_label.shape
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, pixel_label, reduction='none')
            loss = loss.mean(1).mean(1)
            return loss