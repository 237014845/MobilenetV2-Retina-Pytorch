import torch
import torch.nn.functional as F
import numpy as np
import math

# base_size = 8, 16, 32, 64, 128
# anchor_scales=[4, 4*2^(1/3), 4*2^(2/3)]
# anchor_ratios=[0.5, 1.0, 2.0]
class AnchorGenerator(object):
    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)
        # [xmin,ymin,xmax,ymax]
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid



anchor_generators = []
anchor_base_sizes = [8, 16, 32, 64, 128]
scales = [4.0, 5.04, 6.35]
scales = torch.Tensor(scales)
ratios = [0.5, 1.0, 2.0]
ratios = torch.Tensor(ratios)
h_ratios = torch.sqrt(ratios)
w_ratios = 1 / h_ratios


for anchor_base in anchor_base_sizes:
    anchor_generators.append(
        AnchorGenerator(anchor_base, scales, ratios))
print(anchor_generators[0].gen_base_anchors(),
      anchor_generators[1].gen_base_anchors(),
      anchor_generators[2].gen_base_anchors(),
      anchor_generators[3].gen_base_anchors(),
      anchor_generators[4].gen_base_anchors(),
      )
# print(anchor_generators[1].grid_anchor.s((5, 5), stride=128, device='cpu').shape)


def meshgrid( x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx
base_anchors = [[-19.,  -7.,  26.,  14.],
                [-25., -10.,  32.,  17.],
                [-32., -14.,  39.,  21.],
                [-12., -12.,  19.,  19.],
                [-16., -16.,  23.,  23.],
                [-21., -21.,  28.,  28.],
                [ -7., -19.,  14.,  26.],
                [-10., -25.,  17.,  32.],
                [-14., -32.,  21.,  39.]]
base_anchors = torch.Tensor(base_anchors)
# featmap_size: (75, 75), (38,38), (19,19), (10,10), (5,5)
# stride : (8, 16, 32, 64, 128)
stride = 16
feat_h, feat_w = (38,38)
shift_x = torch.arange(0, feat_w) * stride
shift_y = torch.arange(0, feat_h) * stride
shift_xx, shift_yy = meshgrid(shift_x, shift_y)
shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
shifts = shifts.type_as(base_anchors)
# print(shifts, shifts.shape)
# first feat_w elements correspond to the first row of shifts
# add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
# shifted anchors (K, A, 4), reshape to (K*A, 4)
# [xmin,ymin,xmax,ymax]
# base_anchors[None, :, :]相当于有1444个base_anchors   （1444， 4， 4）
# shiifts[:, None, :] 相当于shifts每一行都重复四次       （1444， 4， 4）
all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
all_anchors = all_anchors.view(-1, 4)
# first A rows correspond to A anchors of (0, 0) in feature map,
# then (0, 1), (0, 2), ...
# (指featmap：38x38)  前四行表示featmap的(0,0)cell的产生的4个anchor的(xmin,ymin,xmax,ymax),
#                    接着(0, 1)...(0, 38) 再到(1, 0)...(1, 38)...(38, 38)
