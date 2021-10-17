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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import uuid
from torch.autograd import Variable
from model import NetworkCIFAR as Network

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    CIFAR_CLASSES = 10

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    valid_data = dset.CIFAR10(root=args.dataset_path, train=False, download=True, transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                            pin_memory=True, num_workers=8)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model.drop_path_prob=0
    model = model.cuda()
    state=torch.load(args.model_path)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    val_top1, val_loss = infer(valid_queue, model, criterion, args)
    print(f"Val Top-1: {val_top1}, Val Loss: {val_loss}")


def infer(valid_queue, model, criterion, args):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)

        logits, _ = model(input)
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
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--dataset-path', type=str, default='/home/work/dataset/cifar/', help='location of the data corpus')
    parser.add_argument('--model-path', type=str, required=True, help='path to model')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
    args = parser.parse_args()
    main(args)
