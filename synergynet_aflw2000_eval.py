""" SynergyNet eval script. """
import argparse
import time
import os
import glob
import math
import cv2
import numpy as np
import mindspore as ms
import mindspore.dataset as ds

from dataset.aflw2000 import AFLW2000
from mindspore import context, Tensor, load_checkpoint, load_param_into_net
import mindspore.ops as ops
from models.synergynet import SynergyNet
from utils.synergynet_eval_utils import calc_nme as calc_nme_alfw2000
from utils.synergynet_eval_utils import ana_msg as ana_alfw2000
from utils.synergynet_eval_utils import parse_pose, parsing

from utils.synergynet_util import ParamsPack

param_pack = ParamsPack()


def reconstruct_vertex(param, data_param, lmk_pts=68):
    """
    This function includes parameter de-whitening, reconstruction of landmarks,
    and transform from coordinate space (x,y) to image space (u,v)
    """
    param_mean, param_std, w_shp_base, u_base, w_exp_base = data_param
    param_mean = Tensor(param_mean, dtype=ms.float32)
    param_std = Tensor(param_std, dtype=ms.float32)
    w_shp_base = Tensor(w_shp_base, dtype=ms.float32)
    u_base = Tensor(u_base, dtype=ms.float32)
    w_exp_base = Tensor(w_exp_base, dtype=ms.float32)
    transpose = ops.Transpose()

    if param.shape[1] == 62:
        param = param * param_std[:62] + param_mean[:62]
    else:
        raise NotImplementedError("Parameter length must be 62")

    if param.shape[1] == 62:
        p, offset, alpha_shp, alpha_exp = parsing(param)
    else:
        raise NotImplementedError("Parameter length must be 62")

    vertex1 = ops.matmul(w_shp_base, alpha_shp)
    vertex2 = ops.matmul(w_exp_base, alpha_exp)
    vertex = u_base + vertex1 + vertex2
    vertex = vertex.view(-1, lmk_pts, 3)
    vertex = transpose(vertex, (0, 2, 1))
    vertex = ops.matmul(p, vertex)
    vertex = vertex + offset
    vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

    return vertex


def extract_param(checkpoint_fp, root='', batch_size=8, filelists=''):
    """

    Args:
        checkpoint_fp: Path of the check point file.
        root: Location of data.
        batch_size: batch size.
        filelists : dataset filelist.

    """

    model = SynergyNet(img_size=120, mode="test")
    checkpoint = load_checkpoint(checkpoint_fp)
    load_param_into_net(model, checkpoint)

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
    return outputs, model.data_param


# AFLW2000 facial alignment
img_list = sorted(glob.glob('./aflw2000_data/AFLW2000-3D_crop/*.jpg'))


def benchmark_aflw2000_params(params, data_param):
    '''Reconstruct the landmark points and calculate the statistics'''
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
            lm = reconstruct_vertex(batch_data, data_param, lmk_pts=68)
            lm = lm.asnumpy()
            for j in range(residual):
                outputs.append(lm[j, :2, :])
        else:
            batch_data = params[i * batch_size: (i + 1) * batch_size]
            lm = reconstruct_vertex(batch_data, data_param, lmk_pts=68)
            lm = lm.asnumpy()
            for j in range(batch_size):
                if i == 0:
                    # plot the first 50 samples for validation
                    bkg = cv2.imread(img_list[i * batch_size + j], -1)
                    lm_sample = lm[j]
                    c0 = np.clip((lm_sample[1, :]).astype(np.int), 0, 119)
                    c1 = np.clip((lm_sample[0, :]).astype(np.int), 0, 119)
                    for y, x, in zip([c0, c0, c0 - 1, c0 - 1], [c1, c1 - 1, c1, c1 - 1]):
                        bkg[y, x, :] = np.array([233, 193, 133])
                    cv2.imwrite(f'./results/{i * batch_size + j}.png', bkg)

                outputs.append(lm[j, :2, :])
    return ana_alfw2000(calc_nme_alfw2000(outputs, option='ori'))  # Calculate the error statistics


# AFLW2000 face orientation estimation
def benchmark_foe(params):
    """
    FOE benchmark validation. Only calculate the groundtruth of angles within [-99, 99]
    (following FSA-Net https://github.com/shamangary/FSA-Net)
    """

    # AFLW200 groundturh and indices for skipping, whose yaw angle lies outside [-99, 99]
    exclude_aflw2000 = './aflw2000_data/eval/ALFW2000-3D_pose_3ANG_excl.npy'
    skip_aflw2000 = './aflw2000_data/eval/ALFW2000-3D_pose_3ANG_skip.npy'

    if not os.path.isfile(exclude_aflw2000) or not os.path.isfile(skip_aflw2000):
        raise RuntimeError('Missing data')

    pose_gt = np.load(exclude_aflw2000)
    skip_indices = np.load(skip_aflw2000)
    pose_mat = np.ones((pose_gt.shape[0], 3))

    idx = 0
    for i in range(params.shape[0]):
        if i in skip_indices:
            continue
        _, angles = parse_pose(params[i])
        angles[0], angles[1], angles[2] = angles[1], angles[0], angles[2]  # we decode raw-ptich-yaw order
        pose_mat[idx, :] = np.array(angles)
        idx += 1

    pose_analyis = np.mean(np.abs(pose_mat - pose_gt), axis=0)  # pose GT uses [pitch-yaw-roll] order
    mae = np.mean(pose_analyis)
    yaw = pose_analyis[1]
    pitch = pose_analyis[0]
    roll = pose_analyis[2]
    msg = 'Mean MAE = %3.3f (in deg), [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]' % (mae, yaw, pitch, roll)
    print('\nFace orientation estimation:')
    print(msg)
    return msg


def synergynet_eval(args):
    '''synergynet benchmark validation pipeline'''

    def aflw2000():
        if not os.path.isdir(args.root):
            raise RuntimeError('check if the testing data exist')

        params, data_param = extract_param(
            checkpoint_fp=args.checkpoint_fp,
            root=args.root,
            batch_size=args.batch_size,
            filelists=args.filelists)

        info_out_fal = benchmark_aflw2000_params(params, data_param)
        print(info_out_fal)
        benchmark_foe(params)

    aflw2000()


def main():
    parser = argparse.ArgumentParser(description='SynergyNet eval.')
    parser.add_argument('--root', type=str, default='./aflw2000_data/AFLW2000-3D_crop', help='Location of data.')
    parser.add_argument('--checkpoint_fp', type=str, default='./synergynet.ckpt', help='Path of the check point file.')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch size.')
    parser.add_argument('--filelists', type=str, default='./aflw2000_data/AFLW2000-3D_crop.list')

    args = parser.parse_known_args()[0]
    synergynet_eval(args)


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE)
    main()
