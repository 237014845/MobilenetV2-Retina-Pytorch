class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        use_sigmoid_cls (bool): Whether to use sigmoid loss for classification.
            (softmax by default)
        use_focal_loss (bool): Whether to use focal loss for classification.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 # anchor_scales=[4, 4*2^(1/3), 4*2^(2/3)]
                 anchor_scales=[8, 16, 32],
                 # anchor_ratios=[0.5, 1.0, 2.0]
                 anchor_ratios=[0.5, 1.0, 2.0],
                 # anchor_strides=[8, 16, 32, 64, 128]
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 # True
                 use_sigmoid_cls=False,
                 # True
                 use_focal_loss=False):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = use_sigmoid_cls
        self.use_focal_loss = use_focal_loss

        self.anchor_generators = []
        # self.anchor_base_sizes:[8, 16, 32, 64, 128]
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes

        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        # 图片数量
        num_imgs = len(img_metas)
        # 6
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            # 每个 featmap 对应的 all_anchors
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        # anchor_list: 所有 featmap 对应的 all_anchors 列表
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        # 如果有四张图同时训练
        # anchor_list.shape:[ [[5776, 4], [2166, 4], [600, 4], [150, 4], [36, 4], [4, 4]],
        #                     [[5776, 4], [2166, 4], [600, 4], [150, 4], [36, 4], [4, 4]],
        #                     [[5776, 4], [2166, 4], [600, 4], [150, 4], [36, 4], [4, 4]],
        #                     [[5776, 4], [2166, 4], [600, 4], [150, 4], [36, 4], [4, 4]] ]
        # valid_flag_list.shape:[ [[5776], [2166], [600], [150], [36], [4]],
        #                         [[5776], [2166], [600], [150], [36], [4]],
        #                         [[5776], [2166], [600], [150], [36], [4]],
        #                         [[5776], [2166], [600], [150], [36], [4]] ]
        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        if self.use_sigmoid_cls:
            labels = labels.reshape(-1, self.cls_out_channels)
            label_weights = label_weights.reshape(-1, self.cls_out_channels)
        else:
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            if self.use_focal_loss:
                cls_criterion = weighted_sigmoid_focal_loss
            else:
                cls_criterion = weighted_binary_cross_entropy
        else:
            if self.use_focal_loss:
                raise NotImplementedError
            else:
                cls_criterion = weighted_cross_entropy
        if self.use_focal_loss:
            loss_cls = cls_criterion(
                cls_score,
                labels,
                label_weights,
                gamma=cfg.gamma,
                alpha=cfg.alpha,
                avg_factor=num_total_samples)
        else:
            loss_cls = cls_criterion(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls, loss_reg

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

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        sampling = False if self.use_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # cls_reg_target :(labels, label_weights, bbox_targets, bbox_weights, pos_inds,
        #     neg_inds)
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
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos if self.use_focal_loss else
                             num_total_pos + num_total_neg)
        losses_cls, losses_reg = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_reg=losses_reg)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels



import torch


class AnchorGenerator(object):
    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size # [8, 16, 32, 64, 128]
        # anchor_scales=[4, 4*2^(1/3), 4*2^(2/3)]
        # anchor_scales=[4, 5.04, 6.35]
        self.scales = torch.Tensor(scales)
        # anchor_ratios=[0.5, 1.0, 2.0]
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        # self.base_size :[8, 16, 32, 64, 128]
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            # x_ctr, y_ctr = [ 3.5,  7.5, 15.5, 31.5, 63.5]
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr
        # h_ratios : tensor([0.7071, 1.0000, 1.4142])
        h_ratios = torch.sqrt(self.ratios)
        # W_ratios : tensor([1.4142, 1.0000, 0.7071])
        w_ratios = 1 / h_ratios
        if self.scale_major:
            # ws 表示该anchor_strides下anchor对应的宽的种类
            # w_ratios[:, None].shape: [3,1]    tensor([[1.4142],
            #                                           [1.0000],
            #                                           [0.7071]])
            # self.scales[None, :].shape:[1,3]   tensor([[4.0000, 5.0400, 6.3500]])
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            # hs 表示该anchor_strides下anchor对应的高的种类
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)
        # torch.round(input, out=None) → Tensor 返回一个新的张量，每个输入元素四舍五入到最接近的整数。
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
        # base_anchors:
        # [xmin, ymin, xmax, ymax]
        # tensor([[-19.,  -7.,  26.,  14.],
        #         [-25., -10.,  32.,  17.],
        #         [-32., -14.,  39.,  21.],
        #         [-12., -12.,  19.,  19.],
        #         [-16., -16.,  23.,  23.],
        #         [-21., -21.,  28.,  28.],
        #         [ -7., -19.,  14.,  26.],
        #         [-10., -25.,  17.,  32.],
        #         [-14., -32.,  21.,  39.]]) torch.Size([9, 4])
        # tensor([[-37., -15.,  52.,  30.],
        #         [-49., -21.,  64.,  36.],
        #         [-64., -28.,  79.,  43.],
        #         [-24., -24.,  39.,  39.],
        #         [-32., -32.,  47.,  47.],
        #         [-43., -43.,  58.,  58.],
        #         [-15., -37.,  30.,  52.],
        #         [-21., -49.,  36.,  64.],
        #         [-28., -64.,  43.,  79.]]) torch.Size([9, 4])
        # tensor([[ -75.,  -29.,  106.,   60.],
        #         [ -98.,  -41.,  129.,   72.],
        #         [-128.,  -56.,  159.,   87.],
        #         [ -48.,  -48.,   79.,   79.],
        #         [ -65.,  -65.,   96.,   96.],
        #         [ -86.,  -86.,  117.,  117.],
        #         [ -29.,  -75.,   60.,  106.],
        #         [ -41.,  -98.,   72.,  129.],
        #         [ -56., -128.,   87.,  159.]]) torch.Size([9, 4])
        # tensor([[-149.,  -59.,  212.,  122.],
        #         [-196.,  -82.,  259.,  145.],
        #         [-255., -112.,  318.,  175.],
        #         [ -96.,  -96.,  159.,  159.],
        #         [-129., -129.,  192.,  192.],
        #         [-171., -171.,  234.,  234.],
        #         [ -59., -149.,  122.,  212.],
        #         [ -82., -196.,  145.,  259.],
        #         [-112., -255.,  175.,  318.]]) torch.Size([9, 4])
        # tensor([[-298., -117.,  425.,  244.],
        #         [-392., -164.,  519.,  291.],
        #         [-511., -223.,  638.,  350.],
        #         [-192., -192.,  319.,  319.],
        #         [-259., -259.,  386.,  386.],
        #         [-342., -342.,  469.,  469.],
        #         [-117., -298.,  244.,  425.],
        #         [-164., -392.,  291.,  519.],
        #         [-223., -511.,  350.,  638.]]) torch.Size([9, 4])
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        # 此时假定stride=8， featmap_size: (38,38)
        # 对应的base_anchors = torch.Tensor([[-11., -11.,  18.,  18.],
        #                                   [-17., -17.,  24.,  24.],
        #                                   [-17.,  -7.,  24.,  14.],
        #                                   [ -7., -17.,  14.,  24.]])torch.Size([4, 4]
        base_anchors = self.base_anchors.to(device)
        # featmap_size: (75, 75), (38,38), (19,19), (10,10), (5,5)
        # stride : (8, 16, 32, 64, 128)
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        # shifts:[[  0,   0,   0,   0],
        #         [  8,   0,   8,   0],
        #         [ 16,   0,  16,   0],
        #         ...,
        #         [576, 592, 576, 592],
        #         [584, 592, 584, 592],
        #         [592, 592, 592, 592]]) torch.Size([5625, 4])
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)
        # [xmin,ymin,xmax,ymax]
        # base_anchors[None, :, :]相当于有5625个base_anchors   (5625, 4, 4)
        # shiifts[:, None, :] 相当于shifts每一行都重复四次       (5625, 4, 4)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        # (指featmap：38x38)  前四行表示featmap的(0,0)cell的产生的9个anchor的(xmin,ymin,xmax,ymax),
        #                    接着(0, 1)...(0, 75) 再到(1, 0)...(1, 75)...(75, 75)
        
        # (75, 75):[50625, 4], 
        # (38, 38):[12996, 4], 
        # (19, 19):[3249, 4].
        # (10, 10):[900, 4].
        # (5, 5)  :[225, 4]
        # 一张图产生的all_anchor.shape: [67995, 4]
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        # 如(38x38)
        feat_h, feat_w = featmap_size
        # (37, 37)
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        # uint8相当于byte类型，不能进行乘法操作 因为是ByteTensor,这里0,1代表每一个位置是否有效
        # 如果这里的 dtype=torch.LongTensor 或者 torch.Tensor,则0,1代表序号(即索引值index)
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        # [1, 1, 1, 1, ..., 1, 0]
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        # valid xx: [1,1, ..., 1,0, 1,1, ..., 1,0, 1,1, ..., 1,0, ...]  shape:[1444]
        # valid_yy: [1,1, ..., 1,1, ..., 1,1, ..., 1,1, 0,0, ..., 0,0]  shape:[1444]
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        # valid_xx 和 valid_yy 其中只要有一个不在有效值内就无效
        # 总共有1444个值表示 38*38 featmap每个cell的值
        valid = valid_xx & valid_yy
        # 因为每个cell在 38x38 featmap 下产生4个anchor，所以1444x4=5776
        # 表示该featmap下产生的所有anchor的valid值(有效为1，无效为0)
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
print(anchor_generators)