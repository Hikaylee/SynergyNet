#!/usr/bin/env python3
# coding: utf-8
import time
import numpy as np
import math
import argparse
from math import cos, atan2, asin
import logging
import os
import mindspore as ms
import mindspore.dataset as ds

from utils.synergynet_eval_utils import calc_nme as calc_nme_alfw2000
from utils.synergynet_eval_utils import ana_msg as ana_alfw2000
from example.synergy_net.synergynet_aflw2000_eval import reconstruct_vertex
from mindspore import context, Tensor, load_checkpoint, load_param_into_net
from utils.synergynet_util import ParamsPack
from dataset.aflw2000 import AFLW2000

param_pack = ParamsPack()


# Only work with numpy without batch
def parse_pose(param):
    """ Parse the parameters into 3x4 affine matrix and pose angles """
    param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle_corr(R)  # yaw, pitch, roll
    return P, pose


def P2sRt(P):
    """ Decompositing camera matrix P."""
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


# numpy
def matrix2angle_corr(R):
    """
    Compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    """

    if R[2, 0] != 1 and R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[1, 2] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[0, 1] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi

    return [rx, ry, rz]


def extract_param(model, root='', filelists=None,batch_size=8):
    dataset_generator = AFLW2000(filelists=filelists, root=root, transform=True)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    end = time.time()
    outputs = []
    for data in dataset.create_dict_iterator():
        inputs = data['data']
        output = model(inputs)
        param_prediction = output.asnumpy()
        outputs.append(param_prediction)
    outputs = np.concatenate(outputs, axis=0)
    print('Extracting params take {: .3f}s'.format(time.time() - end))
    return outputs


def benchmark_aflw2000_params(params, data_param):
    """
    Reconstruct the landmark points and calculate the statistics
    """
    outputs = []
    params = Tensor(params, dtype=ms.float32)

    batch_size = 50
    num_samples = params.shape[0]
    iter_num = math.floor(num_samples / batch_size)
    residual = num_samples % batch_size
    for i in range(iter_num + 1):
        if i == iter_num:
            if residual == 0:
                break
            batch_data = params[i * batch_size: i * batch_size + residual]
            lm = reconstruct_vertex(batch_data, data_param)
            lm = lm.asnumpy()
            for j in range(residual):
                outputs.append(lm[j, :2, :])
        else:
            batch_data = params[i * batch_size: (i + 1) * batch_size]
            lm = reconstruct_vertex(batch_data, data_param)
            lm = lm.asnumpy()
            for j in range(batch_size):
                outputs.append(lm[j, :2, :])
    return ana_alfw2000(calc_nme_alfw2000(outputs, option='ori'))


# 102
def benchmark_pipeline(model):
    """
    Run the benchmark validation pipeline for Facial Alignments: AFLW and AFLW2000, FOE: AFLW2000.
    """

    def aflw2000(data_param):
        root = './aflw2000_data/AFLW2000-3D_crop'
        filelists = './aflw2000_data/AFLW2000-3D_crop.list'

        if not os.path.isdir(root) or not os.path.isfile(filelists):
            raise RuntimeError(
                'The data is not properly downloaded from the S3 bucket. Please check your S3 bucket access permission')

        params = extract_param(
            root=root,
            batch_size=8,
            filelists=filelists)

        s2 = benchmark_aflw2000_params(params, data_param)
        logging.info(s2)

    aflw2000(model.data_param)


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc.pth.tar', type=str)
    args = parser.parse_args()

    benchmark_pipeline(args.checkpoint_fp)


if __name__ == '__main__':
    main()
