"""SynergyNet loss definition."""
import math
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.loss.loss import LossBase


class WingLoss(LossBase):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.log_term = math.log(1 + self.omega / self.epsilon)

    def construct(self, pred, target, kp=False):
        n_points = pred.shape[2]
        transpose = ops.Transpose()
        pred = transpose(pred, (1, 2))
        pred = pred.view(-1, 3 * n_points)
        # pred = pred.transpose(1, 2).contiguous()
        target = transpose(target, (1, 2))
        target = target.view(target, (-1, 3 * n_points))
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        log = ops.Log()
        loss1 = self.omega * log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * self.log_term
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class ParamLoss(LossBase):
    """Input and target are all 62-d param"""

    def __init__(self):
        super(ParamLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')  # 均方误差
        self.sqrt = ops.Sqrt()

    def construct(self, input, target, mode='normal'):
        if mode == 'normal':
            loss = self.criterion(input[:, :12], target[:, :12]).mean(1) + self.criterion(input[:, 12:],
                                                                                          target[:, 12:]).mean(1)
            return self.sqrt(loss)
        elif mode == 'only_3dmm':
            loss = self.criterion(input[:, :50], target[:, 12:62]).mean(1)
            return self.sqrt(loss)
        # return self.sqrt(loss.mean(1))


class Synergynet_Loss(LossBase):
    """SynergynetLosses"""

    def __init__(self, reduction="mean"):
        super(Synergynet_Loss, self).__init__(reduction)
        self.LMKLoss_3D = WingLoss()
        self.ParamLoss = ParamLoss()

        self.loss = {'loss_LMK_f0': 0.0,
                     'loss_LMK_pointNet': 0.0,
                     'loss_Param_In': 0.0,
                     'loss_Param_S2': 0.0,
                     'loss_Param_S1S2': 0.0,
                     }

    def get_losses(self):
        return self.loss.keys()

    def construct(self, _3D_attr, _3D_attr_GT, vertex_lmk1, vertex_GT_lmk, vertex_lmk2, _3D_attr_S2):
        self.loss['loss_LMK_f0'] = 0.05 * self.LMKLoss_3D(vertex_lmk1, vertex_GT_lmk, kp=True)
        self.loss['loss_Param_In'] = 0.02 * self.ParamLoss(_3D_attr, _3D_attr_GT)
        self.loss['loss_LMK_pointNet'] = 0.05 * self.LMKLoss_3D(vertex_lmk2, vertex_GT_lmk, kp=True)
        self.loss['loss_Param_S2'] = 0.02 * self.ParamLoss(_3D_attr_S2, _3D_attr_GT, mode='only_3dmm')
        self.loss['loss_Param_S1S2'] = 0.001 * self.ParamLoss(_3D_attr_S2, _3D_attr, mode='only_3dmm')
        return self.get_loss(self.loss)
