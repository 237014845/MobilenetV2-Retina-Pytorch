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
        # self.base_size :[30, 60, 111, 162, 213, 264]
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            # ctr : (3.5, 3.5)、(7.5, 7.5)、 (15.5, 15.5)、(31.5, 31.5)、 (49.5, 49.5)、 (149.5, 149.5)
            x_ctr, y_ctr = self.ctr
        # h_ratios : tensor([1.0000, 0.7071, 1.4142])
        #            tensor([1.0000, 0.7071, 1.4142, 0.5774, 1.7321])
        #            tensor([1.0000, 0.7071, 1.4142, 0.5774, 1.7321])
        #            tensor([1.0000, 0.7071, 1.4142, 0.5774, 1.7321])
        #            tensor([1.0000, 0.7071, 1.4142])
        #            tensor([1.0000, 0.7071, 1.4142])
        h_ratios = torch.sqrt(self.ratios)
        # w_ratios : tensor([1.0000, 1.4142, 0.7071])
        #            tensor([1.0000, 1.4142, 0.7071, 1.7321, 0.5774])
        #            tensor([1.0000, 1.4142, 0.7071, 1.7321, 0.5774])
        #            tensor([1.0000, 1.4142, 0.7071, 1.7321, 0.5774])
        #            tensor([1.0000, 1.4142, 0.7071])
        #            tensor([1.0000, 1.4142, 0.7071])
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            # self.scales[:, None] torch.Size([2, 1])
            # self.scales[:, None] : tensor([[1.0000],
            #                                [1.4142]])
            #                        tensor([[1.0000],
            #                                [1.3601]])
            #                        tensor([[1.0000],
            #                                [1.2081]])
            #                        tensor([[1.0000],
            #                                [1.1467]])
            #                        tensor([[1.0000],
            #                                [1.1133]])
            #                        tensor([[1.0000],
            #                                [1.0923]])

            # w_ratios[None, :] :tensor([[1.0000, 1.4142, 0.7071]]) torch.Size([1, 3])
            #                    tensor([[1.0000, 1.4142, 0.7071, 1.7321, 0.5774]]) torch.Size([1, 5])
            #                    tensor([[1.0000, 1.4142, 0.7071, 1.7321, 0.5774]]) torch.Size([1, 5])
            #                    tensor([[1.0000, 1.4142, 0.7071, 1.7321, 0.5774]]) torch.Size([1, 5])
            #                    tensor([[1.0000, 1.4142, 0.7071]]) torch.Size([1, 3])
            #                    tensor([[1.0000, 1.4142, 0.7071]]) torch.Size([1, 3])

            # ws(无view（-1）) : tensor([[30.0000, 42.4264, 21.2132],
            #              [42.4264, 60.0000, 30.0000]]) torch.Size([2, 3])
            #      tensor([[ 60.0000,  84.8528,  42.4264, 103.9230,  34.6410],
            #              [ 81.6088, 115.4123,  57.7061, 141.3506,  47.1169]]) torch.Size([2, 5])
            #      tensor([[111.0000, 156.9777,  78.4889, 192.2576,  64.0859],
            #              [134.0970, 189.6418,  94.8209, 232.2628,  77.4209]]) torch.Size([2, 5])
            #      tensor([[162.0000, 229.1026, 114.5513, 280.5923,  93.5307],
            #              [185.7579, 262.7014, 131.3507, 321.7422, 107.2474]]) torch.Size([2, 5])
            #      tensor([[213.0000, 301.2275, 150.6137],
            #              [237.1329, 335.3565, 167.6783]]) torch.Size([2, 3])
            #      tensor([[264.0000, 373.3524, 186.6762],
            #              [288.3748, 407.8235, 203.9117]]) torch.Size([2, 3])
            # ws 表示该anchor_strides下anchor对应的宽的种类
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            # hs : tensor([30.0000, 21.2132, 42.4264, 42.4264, 30.0000, 60.0000]) torch.Size([6])
            #      tensor([ 60.0000,  42.4264,  84.8528,  34.6410, 103.9230,  81.6088,  57.7061,
            #               115.4123,  47.1169, 141.3506]) torch.Size([10])
            #      tensor([111.0000,  78.4889, 156.9777,  64.0859, 192.2576, 134.0970,  94.8209,
            #              189.6418,  77.4209, 232.2628]) torch.Size([10])
            #      tensor([162.0000, 114.5513, 229.1026,  93.5307, 280.5922, 185.7579, 131.3507,
            #              262.7014, 107.2474, 321.7421]) torch.Size([10])
            #      tensor([213.0000, 150.6137, 301.2275, 237.1329, 167.6783, 335.3565]) torch.Size([6])
            #      tensor([264.0000, 186.6762, 373.3524, 288.3748, 203.9117, 407.8235]) torch.Size([6])
            # hs 表示该anchor_strides下anchor对应的高的种类
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
        # featmap_size: (38,38), (19,19), (10,10), (5,5), (3,3), (1,1)
        # stride : (8, 16, 32, 64, 100, 300)
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        # shifts:[[0., 0., 0., 0.],
                # [8., 0., 8., 0.],
                # [16., 0., 16., 0.],
                # ...,
                # [280., 296., 280., 296.],
                # [288., 296., 288., 296.],
                # [296., 296., 296., 296.]]) torch.Size([1444, 4])
        shifts = shifts.type_as(base_anchors)
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

