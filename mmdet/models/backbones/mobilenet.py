import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_norm_layer




def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class MobileNetV2(nn.Module):
    def __init__(self,
                 out_indices=(1, 3, 6, 10, 13, 16, 17),
                 # out_indices=(1, 3, 6, 10, 13, 16, 17),
                 width_mult=1.,
                 ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # assert input_size % 32 == 0
        # 32
        input_channel = int(input_channel * width_mult)
        # 1280
        self.out_indices = out_indices
        # self.zero_init_residual = zero_init_residual
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    # 1)InvertedResidual(32, 16, s=1, expand_ratio=1)
                    # 2)InvertedResidual(16, 24, s=2, expand_ratio=6)
                    # 3)InvertedResidual(24, 32, s=2, expand_ratio=6)
                    # 4)InvertedResidual(32, 64, s=2, expand_ratio=6)
                    # 5)InvertedResidual(64, 96, s=1, expand_ratio=6)
                    # 6)InvertedResidual(96, 160, s=2, expand_ratio=6)
                    # 7)InvertedResidual(160, 320, s=1, expand_ratio=6)
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    # 2)InvertedResidual(24, 24, s=1, expand_ratio=6)
                    # 3)InvertedResidual(32, 32, s=1, expand_ratio=6)
                    # 3)InvertedResidual(32, 32, s=1, expand_ratio=6)
                    # 4)InvertedResidual(64, 64, s=1, expand_ratio=6)
                    # 4)InvertedResidual(64, 64, s=1, expand_ratio=6)
                    # 4)InvertedResidual(64, 64, s=1, expand_ratio=6)
                    # 5)InvertedResidual(96, 96, s=1, expand_ratio=6)
                    # 5)InvertedResidual(96, 96, s=1, expand_ratio=6)
                    # 6)InvertedResidual(160, 160, s=1, expand_ratio=6)
                    # 6)InvertedResidual(160, 160, s=1, expand_ratio=6)
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        # self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential

        # self.features = nn.ModuleList(self.features)

        self.features = nn.Sequential(*self.features)



        # self._initialize_weights()
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

            # if self.dcn is not None:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck) and hasattr(
            #                 m, 'conv2_offset'):
            #             constant_init(m.conv2_offset, 0)

            # if self.zero_init_residual:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck):
            #             constant_init(m.norm3, 0)
            #         elif isinstance(m, BasicBlock):
            #             constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        if mode:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()



    # def forward(self, x):
    #     outs = []
    #     for i, layer in enumerate(self.features):
    #         # print(i, layer)
    #         x = layer(x)
    #         if i in self.out_indices:
    #             outs.append(x)
    #     if len(outs) == 1:
    #         return outs[0]
    #     else:
    #         return tuple(outs)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
# #
# from torchsummary import summary
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MobileNetV2().to(device)
# a = summary(model, (3, 224, 224))
# # print(a)
# input = torch.randn(2, 3, 224, 224)
# model = MobileNetV2((3,))
# print(model(input).shape)
