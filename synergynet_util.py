"""SynergyNet utils"""
import random
import pickle
import mindspore
from mindspore import ops
from mindspore import Tensor
import os
import numpy as np
import argparse


class Crop:
    """
    Input:
        Tensor: shape(3, 120, 120),
    Return:
        Tensor: shape(3, 120, 120)
    """

    def __init__(self, maximum, std=None, prob=0.01, mode='test'):
        self.maximum = maximum
        self.std = std
        self.prob = prob
        self.type_li = [1, 2, 3, 4, 5, 6, 7]
        self.switcher = {
            1: self.lup,
            2: self.rup,
            3: self.ldown,
            4: self.rdown,
            5: self.lhalf,
            6: self.rhalf,
            7: self.center
        }
        self.mode = mode
        self.zero = ops.Zeros()

    def get_params(self, img):
        h = img.shape[1]
        w = img.shape[2]
        crop_margins = self.maximum
        rand = random.random()

        return crop_margins, h, w, rand

    def lup(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :h // 2, :w // 2] = img[:, :h // 2, :w // 2]
        return new_img

    def rup(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :h // 2, w // 2:] = img[:, :h // 2, w // 2:]
        return new_img

    def ldown(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, h // 2:, :w // 2] = img[:, h // 2:, :w // 2]
        return new_img

    def rdown(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :h // 2, :w // 2] = img[:, :h // 2, :w // 2]
        return new_img

    def lhalf(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :, :w // 2] = img[:, :, :w // 2]
        return new_img

    def rhalf(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :, w // 2:] = img[:, :, w // 2:]
        return new_img

    def center(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, h // 4: -h // 4, w // 4: -w // 4] = img[:, h // 4: -h // 4, w // 4: -w // 4]
        return new_img

    def __call__(self, img, gt=None):
        img_tensor = Tensor(img, dtype=mindspore.float32)
        crop_margins, h, w, rand = self.get_params(img_tensor)
        crop_backgnd = self.zero((3, h, w), mindspore.float32)

        crop_backgnd[:, crop_margins:h - 1 * crop_margins, crop_margins:w - 1 * crop_margins] = \
            img_tensor[:, crop_margins: h - crop_margins, crop_margins: w - crop_margins]
        # random center crop
        if (rand < self.prob) and (self.mode == 'train'):
            func = self.switcher.get(random.randint(1, 7))
            crop_backgnd = func(crop_backgnd, h, w)

        # center crop
        if self.mode == 'test':
            crop_backgnd[:, crop_margins:h - 1 * crop_margins, crop_margins:w - 1 * crop_margins] = \
                img_tensor[:, crop_margins: h - crop_margins, crop_margins: w - crop_margins]
        crop_backgnd = crop_backgnd.asnumpy()
        return crop_backgnd


def make_abs_path(d):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), d)


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))  # 和 dump() 函数相对应，用于将二进制对象文件转换成 Python 对象


def mkdir(d):
    """only works on *nix system"""
    if not os.path.isdir(d) and not os.path.exists(d):
        os.system('mkdir -p {}'.format(d))


class Compose_GT(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        for t in self.transforms:
            img = t(img)
        img = Tensor.from_numpy(img)
        return img, gt


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ParamsPack:
    """Parameter package"""

    def __init__(self):
        try:
            d = make_abs_path('E:/LAB/threeD/SynergyNet-main/3dmm_data')
            self.keypoints = _load(os.path.join(d, 'keypoints_sim.npy'))

            # PCA basis for shape, expression, texture
            self.w_shp = _load(os.path.join(d, 'w_shp_sim.npy'))
            self.w_exp = _load(os.path.join(d, 'w_exp_sim.npy'))
            # param_mean and param_std are used for re-whitening
            meta = _load(os.path.join(d, 'param_whitening.pkl'))
            self.param_mean = meta.get('param_mean')
            self.param_std = meta.get('param_std')
            # mean values
            self.u_shp = _load(os.path.join(d, 'u_shp.npy'))
            self.u_exp = _load(os.path.join(d, 'u_exp.npy'))
            self.u = self.u_shp + self.u_exp
            self.w = np.concatenate((self.w_shp, self.w_exp), axis=1)
            # base vector for landmarks
            self.w_base = self.w[self.keypoints]
            self.w_norm = np.linalg.norm(self.w, axis=0)
            self.w_base_norm = np.linalg.norm(self.w_base, axis=0)
            self.u_base = self.u[self.keypoints].reshape(-1, 1)
            self.w_shp_base = self.w_shp[self.keypoints]
            self.w_exp_base = self.w_exp[self.keypoints]
            self.std_size = 120
            self.dim = self.w_shp.shape[0] // 3
        except:
            raise RuntimeError('Missing data')
