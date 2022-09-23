"""SynergyNet"""

import numpy as np
import scipy.io as sio
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from models.blocks.i2p import I2P
from models.blocks.synergynet_mlp import ForwardProcess, ReverseProcess

# All data parameters import
from utils.synergynet_util import ParamsPack

__all__ = ["SynergyNet"]

param_pack = ParamsPack()


def parse_param_62(param):
    """Work for only tensor"""
    reshape = ops.Reshape()
    p_ = reshape(param[:, :12], (-1, 3, 4))
    p = p_[:, :, :3]
    offset = reshape(p_[:, :, -1], (-1, 3, 1))
    alpha_shp = reshape(param[:, 12:52], (-1, 40, 1))
    alpha_exp = reshape(param[:, 52:62], (-1, 10, 1))
    return p, offset, alpha_shp, alpha_exp


class SynergyNet(nn.Cell):
    def __init__(self, img_size=120, mode="test"):
        super(SynergyNet, self).__init__()
        self.triangles = sio.loadmat('E:/LAB/threeD/SynergyNet-main//3dmm_data/tri.mat')['tri'] - 1  # 读取数据
        self.triangles = self.triangles.astype(np.int64)
        self.mode = mode
        # self.triangles = Tensor(img_norm, dtype=mindspore.int64)
        # mindspore.context.set_context(device_target="GPU")
        # mindspore.context.set_context(device_id=0)
        self.img_size = img_size
        # Image-to-parameter
        self.I2P = I2P()
        # Forward
        self.forwardDirection = ForwardProcess(68)
        # Reverse
        self.reverseDirection = ReverseProcess(68)

        param_mean = Tensor(param_pack.param_mean, dtype=mindspore.float32)
        self.param_mean = param_mean.asnumpy()
        param_std = Tensor(param_pack.param_std, dtype=mindspore.float32)
        self.param_std = param_std.asnumpy()
        w_shp = Tensor(param_pack.w_shp, dtype=mindspore.float32)
        self.w_shp = w_shp.asnumpy()
        u = Tensor(param_pack.u, dtype=mindspore.float32)
        self.u = u.asnumpy()
        w_exp = Tensor(param_pack.w_exp, dtype=mindspore.float32)
        self.w_shp = w_exp.asnumpy()

        # If doing only offline evaluation, use these
        # self.u_base = Tensor(param_pack.u_base, dtype=mindspore.float32)
        # self.w_shp_base = Tensor(param_pack.w_shp_base, dtype=mindspore.float32)
        # self.w_exp_base = Tensor(param_pack.w_exp_base, dtype=mindspore.float32)

        # Online training needs these to parallel
        u_base = Tensor(param_pack.u_base, dtype=mindspore.float32)
        self.u_base = u_base.asnumpy()
        w_shp_base = Tensor(param_pack.w_shp_base, dtype=mindspore.float32)
        self.w_shp_base = w_shp_base.asnumpy()
        w_exp_base = Tensor(param_pack.w_exp_base, dtype=mindspore.float32)
        self.w_exp_base = w_exp_base.asnumpy()
        self.keypoints = Tensor(param_pack.keypoints, dtype=mindspore.int64)
        self.data_param = [self.param_mean, self.param_std, self.w_shp_base, self.u_base, self.w_exp_base]

    def reconstruct_vertex_62(self, param, lmk_pts=68):
        if param.shape[1] == 62:
            param_ = param * self.param_std[:62] + self.param_mean[:62]
        else:
            raise RuntimeError('length of params mismatch')

        p, offset, alpha_shp, alpha_exp = parse_param_62(param_)
        transpose = ops.Transpose()

        """For 68 pts"""
        vertex1 = ops.matmul(self.w_shp_base, alpha_shp)
        vertex2 = ops.matmul(self.w_exp_base, alpha_exp)
        vertex = self.u_base + vertex1 + vertex2
        vertex = vertex.view(-1, lmk_pts, 3)
        vertex = transpose(vertex, (0, 2, 1))
        vertex = ops.matmul(p, vertex)
        vertex = vertex + offset

        # transform to image coordinate space
        vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        return vertex

    def construct_train(self, x, target):
        """training time construct"""
        _3D_attr, _3D_attr_GT, avgpool = self.I2P(x, target)
        vertex_lmk1 = self.reconstruct_vertex_62(_3D_attr)
        vertex_GT_lmk = self.reconstruct_vertex_62(_3D_attr_GT)

        # global point feature
        point_residual = self.forwardDirection(vertex_lmk1, avgpool, _3D_attr[:, 12:52], _3D_attr[:, 52:62])
        # low-level point feature
        vertex_lmk2 = vertex_lmk1 + 0.05 * point_residual
        _3D_attr_S2 = self.reverseDirection(vertex_lmk2)
        return _3D_attr, _3D_attr_GT, vertex_lmk1, vertex_GT_lmk, vertex_lmk2, _3D_attr_S2

    def construct(self, x, target=None):
        _3D_attr = 0
        if self.mode == "test":
            _3D_attr = self.I2P.construct_test(x)
            return _3D_attr
        elif self.mode == "train":
            _3D_attr, _3D_attr_GT, vertex_lmk1, vertex_GT_lmk, vertex_lmk2, _3D_attr_S2 = self.construct_train(x,
                                                                                                               target)
            return _3D_attr, _3D_attr_GT, vertex_lmk1, vertex_GT_lmk, vertex_lmk2, _3D_attr_S2
        return -1
