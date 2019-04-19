import logging
import torch.nn as nn
import numpy as np
import torch


class AnchorGenerator(object):
    # ctr = (3.5, 3.5)、(7.5, 7.5)、 (15.5, 15.5)、(31.5, 31.5)、
    # (49.5, 49.5)、 (149.5, 149.5)、 
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
anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2])
anchor_strides=(8, 16, 32, 64, 100, 300)
basesize_ratio_range=(0.2, 0.9)
min_ratio, max_ratio = basesize_ratio_range
min_ratio = int(min_ratio * 100)
max_ratio = int(max_ratio * 100)
input_size = 300
step = int(np.floor(max_ratio - min_ratio) / (6 - 2))

min_sizes = []
max_sizes = []
for r in range(int(min_ratio), int(max_ratio) + 1, step):
    # min_sizes = [30, 90, 150, 210, 270]
    # max_sizes = [90, 150, 210, 270, 330]
    min_sizes.append(int(input_size * r / 100))
    max_sizes.append(int(input_size * (r + step) / 100))
if input_size == 300:
    if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
        min_sizes.insert(0, int(input_size * 7 / 100))
        max_sizes.insert(0, int(input_size * 15 / 100))
    elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
        min_sizes.insert(0, int(input_size * 10 / 100))
        max_sizes.insert(0, int(input_size * 20 / 100))
for k in range(len(anchor_strides)):
    base_size = min_sizes[k] # 30, 60, 111, 162, 213, 264
    stride = anchor_strides[k] # 8, 16, 32, 64, 100, 300
    ctr = ((stride - 1) / 2., (stride - 1) / 2.)
    scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
    ratios = [1.]

    for r in anchor_ratios[k]:
        # [1.0, 0.5, 2]
        # [1.0, 0.5, 2]
        # [1.0, 0.5, 2, 0.3333333333333333, 3]
        # [1.0, 0.5, 2]
        # [1.0, 0.5, 2, 0.3333333333333333, 3]
        # [1.0, 0.5, 2]
        # [1.0, 0.5, 2, 0.3333333333333333, 3]
        # [1.0, 0.5, 2]
        # [1.0, 0.5, 2]
        ratios += [1 / r, r]
    anchor_generator = AnchorGenerator(
        base_size, scales, ratios, scale_major=False, ctr=ctr)
    # print(anchor_generator)
    indices = list(range(len(ratios)))
    # indices : [0, 3, 1, 2]
    #           [0, 5, 1, 2, 3, 4]
    #           [0, 5, 1, 2, 3, 4]
    #           [0, 5, 1, 2, 3, 4]
    #           [0, 3, 1, 2]
    #           [0, 3, 1, 2]
    indices.insert(1, len(indices))
    anchor_generator.base_anchors = torch.index_select(
        anchor_generator.base_anchors, 0, torch.LongTensor(indices))
    anchor_generators.append(anchor_generator)
    print(anchor_generator.grid_anchors((38, 38)), anchor_generator.grid_anchors((38, 38)).shape)
#     # scales: [1.0, 1.4142135623730951] [1.0, 1.3601470508735443] [1.0, 1.2080808993852437]
#     #         [1.0, 1.1466537466972386] [1.0, 1.1132998786123665] [1.0, 1.0923286218816286]
#     scales = torch.Tensor(scales)
#     # ratios : [1.0, 0.5, 2]
#     #          [1.0, 0.5, 2, 0.3333333333333333, 3]
#     #          [1.0, 0.5, 2, 0.3333333333333333, 3]
#     #          [1.0, 0.5, 2, 0.3333333333333333, 3]
#     #          [1.0, 0.5, 2]
#     #          [1.0, 0.5, 2]
#     ratios = torch.Tensor(ratios)
#     #  base_size:30, 60, 111, 162, 213, 264
#     w = base_size
#     h = base_size
#     # ctr : (3.5, 3.5)、(7.5, 7.5)、 (15.5, 15.5)、(31.5, 31.5)、 (49.5, 49.5)、 (149.5, 149.5)
#     x_ctr, y_ctr = ctr
#     # h_ratios : tensor([1.0000, 0.7071, 1.4142])
#     #            tensor([1.0000, 0.7071, 1.4142, 0.5774, 1.7321])
#     #            tensor([1.0000, 0.7071, 1.4142, 0.5774, 1.7321])
#     #            tensor([1.0000, 0.7071, 1.4142, 0.5774, 1.7321])
#     #            tensor([1.0000, 0.7071, 1.4142])
#     #            tensor([1.0000, 0.7071, 1.4142])
#     h_ratios = torch.sqrt(ratios)
#     # w_ratios : tensor([1.0000, 1.4142, 0.7071])
#                # tensor([1.0000, 1.4142, 0.7071, 1.7321, 0.5774])
#                # tensor([1.0000, 1.4142, 0.7071, 1.7321, 0.5774])
#                # tensor([1.0000, 1.4142, 0.7071, 1.7321, 0.5774])
#                # tensor([1.0000, 1.4142, 0.7071])
#                # tensor([1.0000, 1.4142, 0.7071])
#     w_ratios = 1 / h_ratios
#     ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
#     hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)
# base_anchors: 
# tensor([[-11., -11.,  18.,  18.],
#         [-17.,  -7.,  24.,  14.],
#         [ -7., -17.,  14.,  24.],
#         [-17., -17.,  24.,  24.],
#         [-26., -11.,  33.,  18.],
#         [-11., -26.,  18.,  33.]]) torch.Size([6, 4])
# tensor([[-22., -22.,  37.,  37.],
#         [-34., -13.,  49.,  28.],
#         [-13., -34.,  28.,  49.],
#         [-44.,  -9.,  59.,  24.],
#         [ -9., -44.,  24.,  59.],
#         [-33., -33.,  48.,  48.],
#         [-50., -21.,  65.,  36.],
#         [-21., -50.,  36.,  65.],
#         [-63., -16.,  78.,  31.],
#         [-16., -63.,  31.,  78.]]) torch.Size([10, 4])
# tensor([[ -40.,  -40.,   70.,   70.],
#         [ -62.,  -23.,   93.,   54.],
#         [ -23.,  -62.,   54.,   93.],
#         [ -80.,  -16.,  111.,   47.],
#         [ -16.,  -80.,   47.,  111.],
#         [ -51.,  -51.,   82.,   82.],
#         [ -79.,  -31.,  110.,   62.],
#         [ -31.,  -79.,   62.,  110.],
#         [-100.,  -23.,  131.,   54.],
#         [ -23., -100.,   54.,  131.]]) torch.Size([10, 4])
# tensor([[ -49.,  -49.,  112.,  112.],
#         [ -83.,  -25.,  146.,   88.],
#         [ -25.,  -83.,   88.,  146.],
#         [-108.,  -15.,  171.,   78.],
#         [ -15., -108.,   78.,  171.],
#         [ -61.,  -61.,  124.,  124.],
#         [ -99.,  -34.,  162.,   97.],
#         [ -34.,  -99.,   97.,  162.],
#         [-129.,  -22.,  192.,   85.],
#         [ -22., -129.,   85.,  192.]]) torch.Size([10, 4])
# tensor([[ -56.,  -56.,  156.,  156.],
#         [-101.,  -25.,  200.,  124.],
#         [ -25., -101.,  124.,  200.],
#         [ -69.,  -69.,  168.,  168.],
#         [-118.,  -34.,  217.,  133.],
#         [ -34., -118.,  133.,  217.]]) torch.Size([6, 4])
# tensor([[ 18.,  18., 281., 281.],
#         [-37.,  57., 336., 242.],
#         [ 57., -37., 242., 336.],
#         [  6.,   6., 293., 293.],
#         [-54.,  48., 353., 251.],
#         [ 48., -54., 251., 353.]]) torch.Size([6, 4])
#     base_anchors = torch.stack(
#         [
#             x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
#             x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
#         ],
#         dim=-1).round()
#     print(base_anchors, base_anchors.shape)



