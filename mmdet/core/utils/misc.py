from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    #  map()是 Python 内置的高阶函数,它接收一个函数 f 和一个 list,
    # 并通过把函数 f 依次作用在 list 的每个元素上,得到一个新的 list 并返回
    map_results = map(pfunc, *args)
    # >>>a = [1,2,3]
    # >>> b = [4,5,6]
    # >>> zipped = zip(a,b)     # 打包为元组的列表
    # [(1, 4), (2, 5), (3, 6)]
    # 就是将每张图产生的(all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
    #  pos_inds_list, neg_inds_list) 打包成元组的列表,再打包成元组
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret

