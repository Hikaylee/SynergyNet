"""i2p module."""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from models.backbones.mobilenet_v2 import MobileNetV2
from utils.mobilenetv2_utils import make_divisible
from models.neck.pooling import GlobalAvgPooling


# Image-to-parameter module
class I2P(nn.Cell):
    """
    i2p module. The input data is x(Tensor): shape(1, 3, C, N),
    where C and N are the size of the input image.

    Returns:
        Tensor: shape(1, 62)

    Examples:
        # >>> img_ori = cv2.imread("img.jpg")
        # >>> transform = mindspore.dataset.vision.py_transforms.ToTensor()
        # >>> img_tran = transform(img_ori)
        # >>> expand_dims = ops.ExpandDims()
        # >>> img = expand_dims(input_tensor, 0)
        # >>> i2p = I2P()
        # >>> out = i2p(img)
    """

    def __init__(self,
                 last_channel: int = 1280,
                 alpha: float = 1.0,
                 round_nearest: int = 8):
        super(I2P, self).__init__()

        self.feature_extractor = MobileNetV2()
        self.avgpool_op = GlobalAvgPooling(keep_dims=True)  # ops.AdaptiveAvgPool2D(1)
        self.last_channel = make_divisible(last_channel * max(1.0, alpha), round_nearest)

        self.num_ori = 12
        self.num_shape = 40
        self.num_exp = 10

        # building classifier(orientation/shape/expression)
        self.classifier_ori = nn.SequentialCell(
            nn.Dropout(0.2),
            nn.Dense(self.last_channel, self.num_ori),
        )
        self.classifier_shape = nn.SequentialCell(
            nn.Dropout(0.2),
            nn.Dense(self.last_channel, self.num_shape),
        )
        self.classifier_exp = nn.SequentialCell(
            nn.Dropout(0.2),
            nn.Dense(self.last_channel, self.num_exp),
        )

    def construct(self, x, target):
        """Training time construct"""
        x = self.feature_extractor(x)
        pool = self.avgpool_op(x)
        x = pool.reshape(x.shape[0], -1)
        avgpool = x
        x_ori = self.classifier_ori(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)
        concat_op = ops.Concat(1)
        _3D_attr = concat_op((x_ori, x_shape, x_exp))
        _3D_attr_GT = Tensor(target, dtype=mindspore.float32)
        # mindspore.context.set_context(device_target="GPU", device_id=0)
        return _3D_attr, _3D_attr_GT, avgpool

    def construct_test(self, x):
        """Testing time construct."""
        x = self.feature_extractor(x)
        pool = self.avgpool_op(x)
        x = pool.reshape(x.shape[0], -1)
        x_ori = self.classifier_ori(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)
        concat_op = ops.Concat(1)
        out = concat_op((x_ori, x_shape, x_exp))

        return out
