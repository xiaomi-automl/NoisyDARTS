# encoding: utf-8
import os
import sys
import time
import glob
import numpy as np
from numpy import linalg as LA
import logging
import argparse
import codecs
from copy import deepcopy
import json
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from thop import profile
from torch.autograd import Variable

import utils as utils
from model_search import Network
from model import NetworkCIFAR
from architect import Architect
from analyze import Analyzer

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/work/dataset/cifar/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--exp_name', type=str, default='normal-noisy', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--noise_mean', type=float, default=0.0, help='noise mean, shared by skip and pooling')
parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'uniform'], help='noise type')
parser.add_argument('--noise_mixture', type=str, default='additive', choices=['additive', 'multiplicative'], help='noise mixture type')
parser.add_argument('--noise_decay', action='store_true', default=False, help='add noise decay or not')
parser.add_argument('--add_noise_skip', action='store_true', default=False, help='add noise to skip')
parser.add_argument('--factor_skip', type=float, default=0.2, help='noise factor of std to skip')
parser.add_argument('--compute_hessian', action='store_true', default=False, help='compute or not Hessian')
parser.add_argument('--report_freq_hessian', type=int, default=50, help='report frequency hessian')
parser.add_argument('--pretrained_weights', type=str, default=None, help='path to weights')
parser.add_argument('--frozen-epochs', default=0, type=int, help='frozen epochs before alpha update')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.exp_name, time.strftime("%Y%m%d"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available, terminating')
        sys.exit(0)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args=args)
    if torch.cuda.is_available():
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # RobustDARTS curvature analyser
    analyser = Analyzer(args, model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    if args.pretrained_weights is not None:
        utils.load(model,  os.path.join(args.pretrained_weights, 'weights.pt'))
        # don't load since arch weights are softmaxed
        # utils.load_arch_weights(model, args.pretrained_weights)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1), model.alphas_normal)
        print(F.softmax(model.alphas_reduce, dim=-1), model.alphas_reduce)
        model.update_history()

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, analyser)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        utils.save_file(recoder=model.alphas_normal_history, path=os.path.join(args.save, 'normal'))
        utils.save_file(recoder=model.alphas_reduce_history, path=os.path.join(args.save, 'reduce'))


    print('-' * 100)
    print('final alphas_normal: \n', F.softmax(model.alphas_normal, dim=-1))
    print('final alphas_reduce: \n', F.softmax(model.alphas_reduce, dim=-1))
    print('final genotype: \n', model.genotype())
    model = NetworkCIFAR(36, CIFAR_CLASSES, 20, True, model.genotype(), args.drop_path_prob)
    flops, _ = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
    params = utils.count_parameters_in_MB(model)
    print('flops: {}, params: {}'.format(flops / 1e6, params))

    
def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, analyser):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # import time

    def valid_generator():
        while True:
            for x, t in valid_queue:
                yield x, t

    valid_gen = valid_generator()
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda(non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        # t_0 = time.time()
        if epoch >= args.frozen_epochs:
            input_search, target_search = next(valid_gen)  # next(iter(valid_queue))
            # t_1 = time.time()
            # print("step %d" % step, " cost: ", t_1 - t_0)
            input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
            target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,
                           epoch=epoch)


        optimizer.zero_grad()
        logits = model(input, epoch)
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

    if args.compute_hessian:
        if (epoch != 0 and epoch % args.report_freq_hessian == 0) or (epoch == (args.epochs - 1)):
            compute_hessian(train_queue, valid_gen, lr, optimizer, analyser, epoch, args)

    return top1.avg, objs.avg


def compute_hessian(train_queue, valid_gen, lr, optimizer, analyser, epoch, args):
    logging.info(f"Epoch {epoch}: Start RDARTS Eigenvector Calculation...")
    _data_loader = deepcopy(train_queue)
    _input, target = next(iter(_data_loader))
    _input = Variable(_input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    input_search, target_search = next(valid_gen)
    input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    H = analyser.compute_Hw(_input, target, input_search, target_search,
                            lr, optimizer, False, epoch)
    g = analyser.compute_dw(_input, target, input_search, target_search,
                            lr, optimizer, False, epoch)
    g = torch.cat([x.view(-1) for x in g])

    del _data_loader
    ev = max(LA.eigvals(H.cpu().data.numpy()))
    logging.info(f'Epoch {epoch}: CURRENT MAX EV: {ev}')

    state = {'epoch': epoch,
            'H': H.cpu().data.numpy().tolist(),
            'g': g.cpu().data.numpy().tolist(),
            #'g_train': float(grad_norm),
            #'eig_train': eigenvalue,
            }

    with codecs.open(os.path.join(args.save,
                                'derivatives_{}_{}.json'.format(epoch, args.exp_name)),
                                'a', encoding='utf-8') as file:
        json.dump(state, file, separators=(',', ':'))
        file.write('\n')


def infer(valid_queue, model, criterion, epoch):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda(non_blocking=True)
            target = Variable(target).cuda(non_blocking=True)

            logits = model(input, epoch)
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

