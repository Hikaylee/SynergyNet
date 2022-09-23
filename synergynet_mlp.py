"""SynergyNet MLP modules"""

import mindspore.nn as nn
import mindspore.ops as ops


class ForwardProcess(nn.Cell):
    def __init__(self, num_pts):
        super(ForwardProcess, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1, has_bias=True)
        self.conv2 = nn.Conv1d(64, 64, 1, has_bias=True)
        self.conv3 = nn.Conv1d(64, 64, 1, has_bias=True)
        self.conv4 = nn.Conv1d(64, 128, 1, has_bias=True)
        self.conv5 = nn.Conv1d(128, 1024, 1, has_bias=True)
        self.conv6 = nn.Conv1d(2418, 512, 1, has_bias=True)  # 1024 + 64 + 1280 = 2368
        self.conv7 = nn.Conv1d(512, 256, 1, has_bias=True)
        self.conv8 = nn.Conv1d(256, 128, 1, has_bias=True)
        self.conv9 = nn.Conv1d(128, 3, 1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(3)
        self.num_pts = num_pts
        self.max_pool = nn.MaxPool1d(num_pts)
        self.relu = ops.ReLU()
        self.tile = ops.Tile()

    def construct(self, x, other_input1=None, other_input2=None, other_input3=None):
        out = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(out), -1))))
        point_features = out
        out = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(out), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn4(ops.ExpandDims()(self.conv4(out), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn5(ops.ExpandDims()(self.conv5(out), -1))))
        global_features = self.max_pool(out)
        global_features_repeated = self.tile(global_features, (1, 1, self.num_pts))

        # 3DMMImg
        avgpool = other_input1
        expand_dims = ops.ExpandDims()
        avgpool = expand_dims(avgpool, 2)
        avgpool = self.tile(avgpool, (1, 1, self.num_pts))

        shape_code = other_input2
        shape_code = expand_dims(shape_code, 2)
        shape_code = self.tile(shape_code, (1, 1, self.num_pts))

        expr_code = other_input3
        expr_code = expand_dims(expr_code, 2)
        expr_code = self.tile(expr_code, (1, 1, self.num_pts))

        concat_op = ops.Concat(1)
        cat_features = concat_op([point_features, global_features_repeated, avgpool, shape_code, expr_code])
        out = self.relu(ops.Squeeze(-1)(self.bn6(ops.ExpandDims()(self.conv6(cat_features), -1))))

        out = self.relu(ops.Squeeze(-1)(self.bn7(ops.ExpandDims()(self.conv7(out), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn8(ops.ExpandDims()(self.conv8(out), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn9(ops.ExpandDims()(self.conv9(out), -1))))
        return out


class ReverseProcess(nn.Cell):
    def __init__(self, num_pts):
        super(ReverseProcess, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1, has_bias=True)
        self.conv2 = nn.Conv1d(64, 64, 1, has_bias=True)
        self.conv3 = nn.Conv1d(64, 64, 1, has_bias=True)
        self.conv4 = nn.Conv1d(64, 128, 1, has_bias=True)
        self.conv5 = nn.Conv1d(128, 1024, 1, has_bias=True)
        self.conv6_1 = nn.Conv1d(1024, 12, 1, has_bias=True)
        self.conv6_2 = nn.Conv1d(1024, 40, 1, has_bias=True)
        self.conv6_3 = nn.Conv1d(1024, 10, 1, has_bias=True)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn6_1 = nn.BatchNorm2d(12)
        self.bn6_2 = nn.BatchNorm2d(40)
        self.bn6_3 = nn.BatchNorm2d(10)
        self.num_pts = num_pts
        self.max_pool = nn.MaxPool1d(num_pts)
        self.relu = ops.ReLU()

    def construct(self, x, other_input1=None, other_input2=None, other_input3=None):
        out = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(out), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(out), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn4(ops.ExpandDims()(self.conv4(out), -1))))
        out = self.relu(ops.Squeeze(-1)(self.bn5(ops.ExpandDims()(self.conv5(out), -1))))
        global_features = self.max_pool(out)

        # Global point feature
        out_rot = self.relu(ops.Squeeze(-1)(self.bn6_1(ops.ExpandDims()(self.conv6_1(global_features), -1))))
        out_shape = self.relu(ops.Squeeze(-1)(self.bn6_2(ops.ExpandDims()(self.conv6_2(global_features), -1))))
        out_expr = self.relu(ops.Squeeze(-1)(self.bn6_3(ops.ExpandDims()(self.conv6_3(global_features), -1))))

        concat_op = ops.Concat(1)
        out = concat_op([out_rot, out_shape, out_expr])
        squeeze = ops.Squeeze(2)
        out = squeeze(out)

        return out
