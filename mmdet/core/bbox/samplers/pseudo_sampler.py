import torch

from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


class PseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        # torch.nonzero(input) 返回 input的 > 0 元素的索引值
        # assigned_gt_inds中正样本值对应的anchor索引值的从大到小排列
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        # assigned_gt_inds中负样本值对应的anchor索引值的从大到小排列
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        # ft_flags.shape: [8732]
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result

# bboxes = torch.randn(8, 4)
# gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
# pos_inds = torch.LongTensor([7, 5, 3, 1])
# print(gt_flags, gt_flags.shape)
# print(gt_flags[pos_inds])