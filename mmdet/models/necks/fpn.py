import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 # in_channels=[256, 512, 1024, 2048]
                 # in_channels=[32, 96, 160, 320] (mobile)
                 in_channels,
                 # out_channels=256
                 out_channels,
                 # num_outs=5
                 num_outs,
                 # start_level=1
                 start_level=0,
                 end_level=-1,
                 # add_extra_convs=True
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # for i in range(1, 4)
        for i in range(self.start_level, self.backbone_end_level):
            # conv(512, 256, 1)
            # conv(1024, 256, 1)
            # conv(2048, 256, 1)

            # mobile
                # conv(96, 256, 1)
                # conv(160, 256, 1)
                # conv(320, 256, 1)
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            # conv(256, 256, 3)
            # conv(256, 256, 3)
            # conv(256, 256, 3)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        # extra_levels = 5-4+1 = 2
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        # add_extra_convs: P6, P7
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                # 2048, 256
                # 320, 250(mobile)
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                # conv(2048, 256)
                # conv(256, 256)
                # fpn_conv = [conv(256, 256, 3), conv(256, 256, 3),conv(256, 256, 3),
                #             conv(2048, 256, k=3, s=2, p=1), conv(256, 256, k=3, s=2, p=1)]


                # (mobile): # fpn_conv = [conv(256, 256, 3), conv(256, 256, 3),conv(256, 256, 3),
                #             conv(320, 256, k=3, s=2, p=1), conv(256, 256, k=3, s=2, p=1)]
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        # C3, C4, P5
        # laterals = [conv(512, 256, 1), conv(1024, 256, 1), conv(2048, 256, 1)]
        # (mobile)laterals = [conv(96, 256, 1), conv(160, 256, 1), conv(320, 256, 1)]
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        # used_backbone_levels = 3
        used_backbone_levels = len(laterals)
        # for i in range(2, 0, -1)
        # i = 2, 1
        for i in range(used_backbone_levels - 1, 0, -1):
            # P4 = C4 + up_sample(p5) (25*25) (19*19)
            # P3 = C3 + up_sample(p4) (50*50) (38*38)
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                # P6
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                # P7
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        # (P3(75x75), P4(38x38), P5(19x19), P6(10x10), P7(5x5))
        return tuple(outs)
