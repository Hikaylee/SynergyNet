"""Load AFLW2000-3D dataset."""

from pathlib import Path
import os.path as osp
import numpy as np
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as transforms
import mindspore.dataset.transforms.c_transforms as c_transforms
from utils.synergynet_util import Crop

__all__ = ["AFLW2000"]


class AFLW2000:
    """
    A source dataset that reads, parses and augments the ModelNet40 dataset.

    The generated dataset has two columns :py:obj:`[image]`.
    The tensor of column :py:obj:`image` is a matrix of the float32 type.

    About AFLW2000 dataset:
    The AFLW2000 dataset contains 2,000 face images .

    You can unzip the original AFLW2000 dataset files into this directory structure and read them by
    MindSpore Vision's API.

    .. code-block::

        ./aflw2000_data/
        ├── AFLW2000-3D_crop
        │   ├── image00002.jpg
        │   ├── image00004.jpg
        │   └── ....
        ├── eval
        │   ├── AFLW2000-3D.pose.npy
        │   ├── AFLW2000-3D.pts68.npy
        │   └── ....
        └── AFLW2000-3D_crop.list

    Citation:

    .. code-block::
        @inproceedings{7780392,
                    author={Zhu, Xiangyu and Lei, Zhen and Liu, Xiaoming and Shi, Hailin and Li, Stan Z.},
                    booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                    title={Face Alignment Across Large Poses: A 3D Solution},
                    year={2016},
                    pages={146-155}
                }
    """

    def __init__(self, filelists, root='', transform=True):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = Image.open(path)
        img = np.array(img.convert("RGB"))

        if self.transform is True:
            transpose = transforms.HWC2CHW()
            crop = Crop(5, mode='test')
            mean_channel = [127.5]
            std_channel = [128]
            normalize_op = transforms.Normalize(mean=mean_channel, std=std_channel)
            trans = c_transforms.Compose([transpose, crop, normalize_op])

            img = trans(img)
        return [img]

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":
    root = 'E:/LAB/threeD/SynergyNet-main/aflw2000_data/AFLW2000-3D_crop'
    filelists = 'E:/LAB/threeD/SynergyNet-main/aflw2000_data/AFLW2000-3D_crop.list'

    dataset_generator = AFLW2000(filelists=filelists, root=root, transform=True)
    # data = dataset_generator[0]
    print(dataset_generator.__len__())
    print(dataset_generator.__getitem__(0)[0].shape)
    # print(dataset_generator.__getitem__(0)[1].shape)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    dataset = dataset.batch(1, drop_remainder=True)
    for data in dataset.create_dict_iterator():
        img = data['data'].asnumpy()
        print(img.shape)
    print('data size:', dataset.get_dataset_size())
