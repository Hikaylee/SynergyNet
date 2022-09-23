import os.path as osp
import numpy as np
from PIL import Image
from pathlib import Path
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.dataset.vision.c_transforms as transforms
from utils.synergynet_util import _load, Crop, Compose_GT


class DDFADataset:
    def __init__(self, root, filelists, param_fp, gt_transform=False):
        self.root = root
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = Tensor.from_numpy(_load(param_fp))
        self.gt_transform = gt_transform

    def _target_loader(self, index):
        target_param = self.params[index]
        target = target_param
        return target

    def __getitem__(self, index):
        index = int(index)
        path = osp.join(self.root, self.lines[index])
        img = Image.open(path)
        img = np.array(img.convert("RGB"))
        target = self._target_loader(index)

        colorjitter = transforms.RandomColorAdjust(0.4, 0.4, 0.4)
        transpose = transforms.HWC2CHW()
        crop = Crop(5, mode='train')
        mean_channel = [127.5]
        std_channel = [128]
        normalize_op = transforms.Normalize(mean=mean_channel, std=std_channel)
        self.trans = Compose_GT([colorjitter, transpose, crop, normalize_op])

        if self.gt_transform:
            img, target = self.trans(img, target)
        else:
            img = self.trans(img)
        return [img], [target]

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":
    dataset_generator = DDFADataset(root='E:/LAB/threeD/SynergyNet-main/3dmm_data/train_aug_120x120',
                                    filelists='E:/LAB/threeD/SynergyNet-main/3dmm_data/train_aug_120x120.list.train',
                                    param_fp='E:/LAB/threeD/SynergyNet-main/3dmm_data/param_all_norm_v201.pkl',
                                    gt_transform=True)
    # data = dataset_generator[0]
    print(dataset_generator.__len__())
    # print(dataset_generator.__getitem__(0)[0])
    # print(dataset_generator.__getitem__(0)[1])
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "target"], shuffle=False)
    dataset = dataset.batch(1, drop_remainder=True)
    for data in dataset.create_dict_iterator():
        img = data['data'].asnumpy()
        target = data['target'].asnumpy()
        print(target)
        sys.exit()
    print('data size:', dataset.get_dataset_size())
