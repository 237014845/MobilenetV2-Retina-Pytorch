import torch
import numpy as np

#
# base_anchors = torch.Tensor([[-11., -11., 18., 18.],
#                     [-17., -17., 24., 24.],
#                     [-17., -7., 24., 14.],
#                     [-7., -17., 14., 24.]])
# num_base_anchors = base_anchors.size(0)
# print(num_base_anchors)
#
# def meshgrid(x, y, row_major=True):
#     xx = x.repeat(len(y))
#     yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
#     if row_major:
#         return xx, yy
#     else:
#         return yy, xx
#
# featmap_size = (38, 38)
# valid_size = (37, 37)
#
# feat_h, feat_w = featmap_size
# valid_h, valid_w = valid_size
# assert valid_h <= feat_h and valid_w <= feat_w
# valid_x = torch.zeros(feat_w, dtype=torch.uint8)
# valid_y = torch.zeros(feat_h, dtype=torch.uint8)
# # print(valid_x)
# valid_x[:valid_w] = 1
# valid_y[:valid_h] = 1
# valid_xx, valid_yy = meshgrid(valid_x, valid_y)
# # print(valid_xx[:100])
# # print(valid_yy[-100:])
# valid = valid_xx & valid_yy
# print(valid.shape)
# valid = valid[:, None].expand(
#     valid.size(0), num_base_anchors).contiguous().view(-1)
# print(valid[:200], valid.shape)


# x = torch.randn(8,4)
# print(x)
# a = torch.tensor([0, 1, 1, 1, 0, 1, 1, 1], dtype=torch.uint8)
# print(a.shape)
# c = x[a,:]
# print(c, c.shape)
torch.manual_seed(1314)
x = torch.rand(4, 8)
print(x)
assigned_gt_inds = x.new_full((8,), -1, dtype=torch.long)
print(assigned_gt_inds)
max_overlaps, argmax_overlaps = x.max(dim=0)
print(max_overlaps, argmax_overlaps)
assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.5)] = 0
print(assigned_gt_inds)
pos_inds = max_overlaps >= 0.5
print(pos_inds)
print(assigned_gt_inds[pos_inds])
print(argmax_overlaps[pos_inds])
assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
print(assigned_gt_inds[pos_inds])
gt_max_overlaps, gt_argmax_overlaps = x.max(dim=1)
print(gt_max_overlaps, gt_argmax_overlaps)
print()
for i in range(4):
            if gt_max_overlaps[i] >= 0.:
                 assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
print(assigned_gt_inds)
assigned_labels = assigned_gt_inds.new_zeros((8, ))
print(assigned_labels)
pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze(-1).unique()
print(pos_inds, pos_inds.shape)
gt_labels = torch.LongTensor([1, 1, 1, 1])
if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
print(assigned_labels)
pos_assigned_gt_inds = assigned_gt_inds[pos_inds]-1
print(pos_assigned_gt_inds)
