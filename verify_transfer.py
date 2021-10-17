import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import CIFAR10
from thop import profile

from timm.models import EfficientNet
from timm.models.efficientnet_builder import decode_arch_def
from timm.models.efficientnet_blocks import resolve_bn_args

from utils import AverageMeter, accuracy, _data_transforms_imagenet

def get_args():
	parser = argparse.ArgumentParser("Transfer Learning")
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to transfer')
	parser.add_argument('--dataset-path', type=str, default='/home/work/dataset/cifar', help='dataset to transfer')
	parser.add_argument('--model-path', type=str, default='pretrained/noisy_darts_a_transfer.pt.tar', help='transferred model path')
	parser.add_argument('--batch_size', type=int, default=256, help='batch size')
	parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
	parser.add_argument('--lr_scheduler', type=str, default='cos', help='cos or step')
	parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
	parser.add_argument('--epochs', type=int, default=200, help='fine-tune epochs')
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
						help='number of label classes (default: 1000)')
	parser.add_argument('--dropout', type=float, default=0.1, metavar='DROP', help='Dropout rate (default: 0.)')
	parser.add_argument('--drop_connect', type=float, default=0.1, metavar='DROP',
						help='Drop connect rate (default: 0.1)')

	args = parser.parse_args()
	print(args)
	return args

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	val_transform = _data_transforms_imagenet()

	
	val_data = CIFAR10(root=args.dataset_path, train=False, download=True, transform=val_transform)
	val_queue = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
											pin_memory=True, num_workers=8)

	# NoisyDARTS-A model specification
	s_r = 0.25
	arch_def = [
				# stage 0, 112x112 in
				['ds_r1_k3_s1_e1_c16'],  # relu
				# stage 1, 112x112 in
				['ir_r1_k3_s2_e6_c32_se%f_nsw' % s_r],
				# stage 2, 56x56 in
				['ir_r1_k3_s1_e3_c32_se%f_nsw' % s_r],  # swish
				# stage 3, 28x28 in
				['ir_r1_k5_s2_e6_c40_se%f_nsw' % s_r, 'ir_r3_k3_s1_e6_c40_se%f_nsw' % s_r],  # swish
				# stage 4, 14x14in
				['ir_r1_k5_s2_e6_c80_se%f_nsw' % s_r, 'ir_r1_k7_s1_e6_c80_se%f_nsw' % s_r, 'ir_r2_k3_s1_e6_c80_se%f_nsw' % s_r,
				'ir_r4_k3_s1_e6_c96_se%f_nsw' % s_r],  # swish
				# stage 5, 7x7in
				['ir_r1_k3_s2_e6_c192_se%f_nsw' % s_r, 'ir_r3_k7_s1_e6_c192_se%f_nsw' % s_r,
				'ir_r1_k7_s1_e6_c320_se%f_nsw' % s_r],  # swish
	]

	model_kwargs = dict(
		block_args=decode_arch_def(arch_def, 1.0, depth_trunc='round'),
		num_features=1280,
		stem_size=32,
		channel_multiplier=1.0,
		act_layer=nn.ReLU
	)
	
	model = EfficientNet(**model_kwargs)
	model.classifier = nn.Sequential(
							nn.Dropout(args.dropout),
							nn.Linear(1280, 10),
						)
	model.load_state_dict(torch.load(args.model_path)['model_state'])
	model = model.to(device)
	input = torch.randn(1, 3, 224, 224).cuda()
	flops, params = profile(model, inputs=(input,), verbose=False)
	print('flops: {}M, params: {}M'.format(flops / 1e6, params / 1e6))

	model.eval()
	criterion = nn.CrossEntropyLoss().to(device)

	infer(val_queue, model, criterion, device)

def infer(val_queue, model, criterion, device):
	top1 = AverageMeter()
	top5 = AverageMeter()
	loss_ = 0.
	with torch.no_grad():
		for step, (inputs, labels) in enumerate(val_queue):
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels).mean()
			loss_ += loss
			prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
			n = inputs.size(0)
			top1.update(prec1.item(), n)
			top5.update(prec5.item(), n)

	print('Valiate. loss: {}, top1: {}, top5: {}'.format(loss_ / (step + 1), top1.avg, top5.avg))


if __name__ == '__main__':
	args = get_args()
	main(args)
