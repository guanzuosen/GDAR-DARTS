import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                    help='dataset name')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='num of warmup epochs where we only train weights')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--cpu_ids', type=str, default='0,1,2,3', help='List of CPU cores to use, e.g., "0,1,2,3"')
args = parser.parse_args()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.set == 'cifar10':
    CLASSES = 10
elif args.set == 'cifar100':
    CLASSES = 100
elif args.set == 'ImageNet16-120':
    CLASSES = 120


def main():

    if args.cpu_ids is not None:
        cpu_list = [int(x) for x in args.cpu_ids.split(',')]
        logging.info(f"Attempting to use specific CPUs: {cpu_list}")


        torch.set_num_threads(len(cpu_list))

        if hasattr(os, 'sched_setaffinity'):

            try:
                os.sched_setaffinity(0, set(cpu_list))
                logging.info(f"Success: Process affinity set to CPUs {cpu_list} (via os.sched_setaffinity)")
            except Exception as e:
                logging.warning(f"Failed to set affinity via os: {e}")
        elif psutil:

            try:
                p = psutil.Process()
                p.cpu_affinity(cpu_list)
                logging.info(f"Success: Process affinity set to CPUs {cpu_list} (via psutil)")
            except Exception as e:
                logging.warning(f"Failed to set CPU affinity via psutil: {e}")
        else:
            logging.warning("Cannot set CPU affinity: os.sched_setaffinity not available (are you on Windows?) "
                            "and psutil not installed.")
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.set == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.set == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    '''elif args.set == 'ImageNet16-120':
        train_transform, valid_transform = utils._data_transforms_imagenet16(args)
        train_data = ImageNet16(root=os.path.join(args.data, 'ImageNet16'),
                               train=True,
                               transform=train_transform,
                               use_num_of_class_only=120)'''

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=1)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]


        if epoch < args.warmup_epochs:
            logging.info('epoch %d lr %e (Warmup Phase: Weights Only)', epoch, lr)
        else:
            logging.info('epoch %d lr %e (Search Phase: Weights + Arch)', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print("Normal Alphas:", F.softmax(model.alphas_normal, dim=-1))
        print("Reduce Alphas:", F.softmax(model.alphas_reduce, dim=-1))


        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args, epoch)
        logging.info('train_acc %f', train_acc)


        valid_acc, valid_obj = infer(valid_queue, model, criterion, args)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    valid_queue_iter = iter(valid_queue)

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except StopIteration:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)


        if epoch >= args.warmup_epochs:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,
                           epoch=epoch,
                           n_epochs=args.epochs)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(non_blocking=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()