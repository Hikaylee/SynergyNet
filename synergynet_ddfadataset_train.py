""" SynergyNet train script. """
import os.path as osp
from pathlib import Path
import argparse
import time
import logging
from mindspore import context, nn, load_checkpoint, load_param_into_net, save_checkpoint
import mindspore.dataset as ds
from dataset.DDFAdataset import DDFADataset
from utils.synergynet_util import str2bool, AverageMeter, mkdir
from models.synergynet import SynergyNet
from utils.synergynet_train_utils import benchmark_pipeline
from engine.ops.synergynet_loss import Synergynet_Loss

# global args (configuration)
args = None  # define the static training setting, which wouldn't and shouldn't be changed over the whole experiements.


def parse_args():
    parser = argparse.ArgumentParser(description='SynergyNet train')
    parser.add_argument('-j', '--workers', default=6, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('-vb', '--val-batch-size', default=32, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='E:/LAB/threeD/vision/mindvision/ms3d/models/synergynet.ckpt', type=str, metavar='PATH')
    parser.add_argument('--device_target', default="CPU", choices=["Ascend", "GPU", "CPU"], type=str)
    parser.add_argument('--filelists-train', default='E:/LAB/threeD/SynergyNet-main/3dmm_data/train_aug_120x120.list.train', type=str)
    parser.add_argument('--root', default='E:/LAB/threeD/SynergyNet-main/3dmm_data/train_aug_120x120', type=str)
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--milestones', default=None, type=str)
    parser.add_argument('--param-fp-train', default='E:/LAB/threeD/SynergyNet-main/3dmm_data/param_all_norm_v201.pkl', type=str)
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('--save_val_freq', default=10, type=int)

    global args
    args = parser.parse_args()

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(train_dataset, model, epoch, criterion):
    """Network training, loss updates, and backward calculation"""

    # AverageMeter for statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_list = []
    losses_name = list(criterion.get_losses())
    losses_name.append('loss_total')
    losses_meter = [AverageMeter() for i in range(len(losses_name))]

    end = time.time()

    for i, data in enumerate(train_dataset.create_dict_iterator()):
        input = data['data']
        target = data['target']
        target = target[:, :62]

        params = model(input, target)
        losses = Synergynet_Loss(params)
        loss_num = losses.asnumpy()
        loss_list.append(loss_num)

        data_time.update(time.time() - end)

        loss_total = 0
        for j, name in enumerate(losses):
            mean_loss = losses[name].mean()
            losses_meter[j].update(mean_loss, input.size(0))
            loss_total += mean_loss

        losses_meter[j + 1].update(loss_total, input.size(0))

        # compute gradient and do SGD step

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            msg = 'Epoch: [{}][{}/{}]\t'.format(epoch, i, len(train_dataset)) + \
                  'Time: {:.3f} ({:.3f})\t'.format(batch_time.val, batch_time.avg)
            for k in range(len(losses_meter)):
                msg += '{}: {:.4f} ({:.4f})\t'.format(losses_name[k], losses_meter[k].val, losses_meter[k].avg)
            logging.info(msg)


def main():
    """ Main funtion for the training process"""
    parse_args()  # parse global argsl

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )

    print_args(args)  # print args

    # step1: define the model structure
    model = SynergyNet(img_size=args.img_size, mode="train")
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    # step2: optimization: loss and optimization method
    # Set learning rate scheduler.
    milestone = list(range(80, 20000, 80))
    lr_rate = [args.learning_rate * 0.2 ** x for x in range(249)]
    lr = nn.piecewise_constant_lr(milestone, lr_rate)

    optimizer = nn.SGD(model.trainable_params(),
                       learning_rate=lr,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       nesterov=True)

    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')
            checkpoint = load_checkpoint(args.resume)
            load_param_into_net(model, checkpoint, strict_load=False)

        else:
            logging.info(f'=> no checkpoint found at {args.resume}')

    # step 2.2 define loss
    criterion = Synergynet_Loss(reduction="mean")

    net_with_loss = nn.WithLossCell(model, criterion)
    net_train = nn.TrainOneStepCell(net_with_loss, optimizer)
    net_train.set_train()

    # step3: data
    train_dataset = DDFADataset(root=args.root,
                                filelists=args.filelists_train,
                                param_fp=args.param_fp_train,
                                gt_transform=True)
    train_ds = ds.GeneratorDataset(train_dataset, ["data", "target"], shuffle=False)
    train_ds = train_ds.batch(8, drop_remainder=True)

    # step4: run
    for epoch in range(args.start_epoch, args.epochs + 1):
        net_train.set_train(True)
        # train for one epoch
        train(train_ds, model, epoch, criterion)

        filename = f'{args.snapshot}_checkpoint_epoch_{epoch}.pth.tar'
        # save checkpoints and current model validation
        if (epoch % args.save_val_freq == 0) or (epoch == args.epochs):
            save_checkpoint(model, filename)
            logging.info(f'Save checkpoint to {filename}')
            logging.info('\nVal[{}]'.format(epoch))
            benchmark_pipeline(model)
    print("training completed...")


if __name__ == '__main__':
    main()
