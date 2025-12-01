"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

import timm

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, Interpolate, _make_encoder, _make_scratch

REPO_DIR = '/localscratch/ctc32/research/misc/dinov3'

class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, activation='sigmoid', pretrained=False, features=256, input_channels=3, output_channels=1, group_width=8, last_residual=False, aa=False):
        """Init. Changed by Chris to add input_channels and output_channels

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
            input_channels (int, optional): number of input channels for the encoder
        """

        super(MidasNet, self).__init__()

        self.out_chan = output_channels
        self.last_res = last_residual

        self.pretrained, self.scratch = _make_encoder(
            backbone="resnext101_wsl", 
            features=features, 
            use_pretrained=pretrained, 
            in_chan=input_channels, 
            group_width=group_width,
            aa=aa)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        if activation == 'sigmoid':
            out_act = nn.Sigmoid()
        elif activation == 'relu':
            out_act = nn.ReLU()
        else:
            out_act = nn.Identity()

        res_dim = 128 + (input_channels if last_residual else 0)
        self.scratch.output_conv = nn.ModuleList([
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(res_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            out_act
        ])

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # out = self.scratch.output_conv(path_1)

        out = self.scratch.output_conv[0](path_1)
        out = self.scratch.output_conv[1](out)

        if self.last_res:
            out = torch.cat((out, x), dim=1)

        out = self.scratch.output_conv[2](out)
        out = self.scratch.output_conv[3](out)
        out = self.scratch.output_conv[4](out)
        out = self.scratch.output_conv[5](out)


        # if self.out_chan == 1:
        #     return torch.squeeze(out, dim=1)
        # else:
        #     return out
        return out


class DINOEncoderDecoder(BaseModel):

    def __init__(self, activation='sigmoid', input_channels=3, output_channels=1, features=256, last_residual=False):

        super(DINOEncoderDecoder, self).__init__()

        self.out_chan = output_channels
        self.last_res = last_residual

        # the model name can be changed (convnext large, small, etc) but we need to 
        # also change the values passed to _make_scratch based on the dino feature channels
        self.pretrained = timm.create_model(
            'convnext_base.dinov3_lvd1689m', 
            pretrained=True, 
            features_only=True,
            in_chans=input_channels
        )

        self.scratch = _make_scratch([128, 256, 512, 1024], features)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        if activation == 'sigmoid':
            out_act = nn.Sigmoid()
        elif activation == 'relu':
            out_act = nn.ReLU()
        else:
            out_act = nn.Identity()

        res_dim = 128 + (input_channels if last_residual else 0)
        self.scratch.output_conv = nn.ModuleList([
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(res_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            out_act
        ])

    def forward(self, x):

        layer_1, layer_2, layer_3, layer_4 = self.pretrained(x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv[0](path_1)
        out = self.scratch.output_conv[1](out)

        if self.last_res:
            out = torch.cat((out, x), dim=1)

        out = self.scratch.output_conv[2](out)
        out = self.scratch.output_conv[3](out)
        out = self.scratch.output_conv[4](out)
        out = self.scratch.output_conv[5](out)

        return out
    
