import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import (AnchorGenerator, anchor_target, weighted_smoothl1,
                        multi_apply)
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class SSDHead(AnchorHead):

    def __init__(self,
                 input_size=300,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 # featmap_size: (38,38), (19,19), (10,10), (5,5), (3,3), (1,1)
                 # anchor_strides指的是对应featmap的每个cell映射到原图的坐标之间的步长
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        # num_anchors= [4, 6, 6, 6, 4, 4]
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        min_ratio, max_ratio = basesize_ratio_range
        # min_ratio = 20
        # max_ratio = 90
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        # step = 17
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        # for r in range(20, 91, 17)
        # r = 10, 30, 50, 70, 90
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            # min_sizes = [60, 111, 162, 213, 264]
            # max_sizes = [111, 162, 213, 264, 315]
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        # min_sizes = [30, 60, 111, 162, 213, 264]
        # max_sizes = [60, 111, 162, 213, 264, 315]
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        # for k in range(6):
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k] # 30, 60, 111, 162, 213, 264
            stride = anchor_strides[k] # 8, 16, 32, 64, 100, 300
            # ctr : 中心点坐标（cx, cy）
            # ctr : (3.5, 3.5)、(7.5, 7.5)、 (15.5, 15.5)、(31.5, 31.5)、 (49.5, 49.5)、 (149.5, 149.5)
            ctr = ((stride - 1) / 2., (stride - 1) / 2.)
            # scales: [1.0, 1.4142135623730951] [1.0, 1.3601470508735443] [1.0, 1.2080808993852437]
            #         [1.0, 1.1466537466972386] [1.0, 1.1132998786123665] [1.0, 1.0923286218816286]
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            # ratios : [1.0, 0.5, 2]
            #          [1.0, 0.5, 2, 0.3333333333333333, 3]
            #          [1.0, 0.5, 2, 0.3333333333333333, 3]
            #          [1.0, 0.5, 2, 0.3333333333333333, 3]
            #          [1.0, 0.5, 2]
            #          [1.0, 0.5, 2]
            ratios = [1.]
            # r = [2] 或者 [2, 3]
            for r in anchor_ratios[k]:
                # r=2  ratios=[1.0, 0.5, 2]
                # r=3, ratios=[1.0, 0.5, 2, 0.3333333333333333, 3]
                ratios += [1 / r, r]  # 4 or 6 ratio

            # 根据6个anchor_strides、base_size、scales、ratios、ctr产生每个anchor_strides对应产生的
            # 不同种类anchor的坐标：torch.Size([6, 4]、[10, 4]、[10, 4]、[10, 4]、[6, 4]、[6, 4])
            anchor_generator = AnchorGenerator(
                base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            # indices : [0, 3, 1, 2]
            #           [0, 5, 1, 2, 3, 4]
            #           [0, 5, 1, 2, 3, 4]
            #           [0, 5, 1, 2, 3, 4]
            #           [0, 3, 1, 2]
            #           [0, 3, 1, 2]
            indices.insert(1, len(indices))
            # 将anchor_generator.base_anchors产生的base_anchors按照indices选出来
            # 此时anchor_generator.base_anchors ：torch.Size([4, 4]、[6, 4]、[6, 4]、[6, 4]、[4, 4]、[4, 4])
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            # self.anchor_generators 表示每个 anchor_strides 下对应的 AnchorGenerator
            # 将anchor_generator.base_anchors产生的base_anchors按照indices选出来
            # 此时anchor_generator.base_anchors ：torch.Size([4, 4]、[6, 4]、[6, 4]、[6, 4]、[4, 4]、[4, 4])
            # tensor([[-11., -11., 18., 18.],
            #         [-17., -17., 24., 24.],
            #         [-17., -7., 24., 14.],
            #         [-7., -17., 14., 24.]])
            # tensor([[-22., -22., 37., 37.],
            #         [-33., -33., 48., 48.],
            #         [-34., -13., 49., 28.],
            #         [-13., -34., 28., 49.],
            #         [-44., -9., 59., 24.],
            #         [-9., -44., 24., 59.]])
            # tensor([[-40., -40., 70., 70.],
            #         [-51., -51., 82., 82.],
            #         [-62., -23., 93., 54.],
            #         [-23., -62., 54., 93.],
            #         [-80., -16., 111., 47.],
            #         [-16., -80., 47., 111.]])
            # tensor([[-49., -49., 112., 112.],
            #         [-61., -61., 124., 124.],
            #         [-83., -25., 146., 88.],
            #         [-25., -83., 88., 146.],
            #         [-108., -15., 171., 78.],
            #         [-15., -108., 78., 171.]])
            # tensor([[-56., -56., 156., 156.],
            #         [-69., -69., 168., 168.],
            #         [-101., -25., 200., 124.],
            #         [-25., -101., 124., 200.]])
            # tensor([[18., 18., 281., 281.],
            #         [6., 6., 293., 293.],
            #         [-37., 57., 336., 242.],
            #         [57., -37., 242., 336.]])
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.use_focal_loss = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        # cls_scores表示每个anchor对应每个类别的分数
        # [B, 84(126), w, h]
        cls_scores = []
        # bbox_preds表示每个anchor四个坐标预测的值
        # [B, 16(24), w, h]
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds
    # all_cls_scores:[B, 8732, 21]
    # all_bbox_preds:[B, 8732, 4]
    # all_labels:[B, 8732]
    # all_label_weights:[B, 8732]
    # all_bbox_targets: [B, 8732, 4]
    # all_bbox_weights: [B, 8732, 4]
    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # nonzero(): 符合条件的索引值
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        # cfg.neg_pos_ratio = 3
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        # topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # 如果维度没有指定，则选择为输入的最后一个维度
        # 选出分类损失中由大到小的前k个负样本的分类损失
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        # loss_cls_pos：将所有正样本的分类损失相加
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        # loss_cls_neg：将前k个负样本的分类损失相加
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        # 位置误差仅针对正样本进行计算
        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        # return : loss_reg: [1, B, num_pos, 4]
        #          loss_cls[None]: [1, B, 8732]
        return loss_cls[None], loss_reg

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        # img_meta = dict(
        #     ori_shape=ori_shape,
        #     img_shape=img_shape,
        #     pad_shape=pad_shape,
        #     scale_factor=scale_factor,
        #     flip=flip)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            sampling=False,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        # cls_scores:[B((num_images)), C, H, W] => [B, H, W, C] => [B, self.num_anchors x H x W, self.cls_out_channels]
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        # all_labels : [num_images, 8732]
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        # all_label_weights : [num_images, 8732]
        all_label_weights = torch.cat(label_weights_list, -1).view(
            num_images, -1)
        # bbox_preds: [B((num_images)), C, H, W] => [B, H, W, C] => [B, self.num_anchors x H x W (8732), 4]
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        # all_bbox_targets:[num_images, 8732, 4]
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(
            num_images, -1, 4)
        # all_bbox_weights:[num_images, 8732, 4]
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(
            num_images, -1, 4)

        losses_cls, losses_reg = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_reg=losses_reg)
