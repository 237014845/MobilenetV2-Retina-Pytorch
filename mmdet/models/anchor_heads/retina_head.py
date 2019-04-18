import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from .anchor_head import AnchorHead
from ..registry import HEADS
from ..utils import bias_init_with_prob


@HEADS.register_module
class RetinaHead(AnchorHead):
    #     num_classes=2,
    #     in_channels=256,
    #     stacked_convs=4,
    #     feat_channels=256,
    #     octave_base_scale=4,
    #     scales_per_octave=3,
    #     anchor_ratios=[0.5, 1.0, 2.0],
    #     anchor_strides=[8, 16, 32, 64, 128],
    #     target_means=[.0, .0, .0, .0],
    #     target_stds=[1.0, 1.0, 1.0, 1.0]
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 **kwargs):
        self.stacked_convs = stacked_convs # 4
        self.octave_base_scale = octave_base_scale # 4
        self.scales_per_octave = scales_per_octave # 3
        # octave_scales = [1, 2^(1/3), 2^(2/3)]
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        # anchor_scales=[4, 4*2^(1/3), 4*2^(2/3)]
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_scales=anchor_scales,
            use_sigmoid_cls=True,
            use_focal_loss=True,
            **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # 俩组并行的 subnet ： 由4个 conv(256, 256, 3, s=1, p=1) 组成
        # fro i in range(4)
        for i in range(self.stacked_convs):
            # 256
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
            self.reg_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m, std=0.01)
        for m in self.reg_convs:
            normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = self.relu(cls_conv(cls_feat))
        for reg_conv in self.reg_convs:
            reg_feat = self.relu(reg_conv(reg_feat))
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
