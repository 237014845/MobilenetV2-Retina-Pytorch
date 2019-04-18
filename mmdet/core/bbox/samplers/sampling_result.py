import torch


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        # assigned_gt_inds中正样本值对应的anchor索引值的从大到小排列
        self.pos_inds = pos_inds
        # assigned_gt_inds中负样本值对应的anchor索引值的从大到小排列
        self.neg_inds = neg_inds
        # anchor中正样本的坐标
        self.pos_bboxes = bboxes[pos_inds]
        # anchor中负样本的坐标
        self.neg_bboxes = bboxes[neg_inds]
        # self.pos_is_gt： 选出来正样本的anchor，值为0
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        # self.pos_assigned_gt_inds 表示正样本anchor对应真实label值的索引
        # 也就是正样本anchor对应的正负样本值 - 1
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        # self.pos_gt_bboxes 表示从gt中选出正样本anchor对应的gt的[xmin, ymin, xmax, ymax]
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            # self.pos_gt_labels 表示从gt_label中选出正样本anchor对应的真值label
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
