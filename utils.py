import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from genotypes import PRIMITIVES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

class AverageMeter(object):
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0
	
	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)
	
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res


class Cutout(object):
	def __init__(self, length):
		self.length = length
	
	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)
		
		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)
		
		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img


def _data_transforms_cifar10(args):
	CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
	CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
	
	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	if args.cutout:
		train_transform.transforms.append(Cutout(args.cutout_length))
	
	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	return train_transform, valid_transform

def _data_transforms_imagenet():
	# imagenet
	MEAN = [0.485, 0.456, 0.406]
	STD = [0.229, 0.224, 0.225]

	val_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(MEAN, STD),
	])
	return val_transform

def count_parameters_in_MB(model):
	return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
	filename = os.path.join(save, 'checkpoint.pth.tar')
	torch.save(state, filename)
	if is_best:
		best_filename = os.path.join(save, 'model_best.pth.tar')
		shutil.copyfile(filename, best_filename)


def save(model, model_path):
	torch.save(model.state_dict(), model_path)


def load(model, model_path, map_location='cuda:0'):
	model.load_state_dict(torch.load(model_path, map_location))


def drop_path(x, drop_prob):
	if drop_prob > 0.:
		keep_prob = 1. - drop_prob
		mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
		x.div_(keep_prob)
		x.mul_(mask)
	return x


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))
	
	if scripts_to_save is not None:
		subpath = os.path.join(path, 'scripts')
		if not os.path.exists(subpath):
			os.mkdir(subpath)
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)


def save_file(recoder, size = (14, len(PRIMITIVES)), path='./'):
	fig, axs = plt.subplots(*size, figsize=(36, 98))
	num_ops = size[1]
	row = 0
	col = 0
	for (k, v) in recoder.items():
		axs[row, col].set_title(k)
		axs[row, col].plot(v, 'r+')
		if col == num_ops-1:
			col = 0
			row += 1
		else:
			col += 1
	if not os.path.exists(path):
		os.makedirs(path)
	fig.savefig(os.path.join(path, 'output.png'), bbox_inches='tight')
	plt.tight_layout()
	print('save history weight in {}'.format(os.path.join(path, 'output.png')))
	with open(os.path.join(path, 'history_weight.json'), 'w') as outf:
		json.dump(recoder, outf)
		print('save history weight in {}'.format(os.path.join(path, 'history_weight.json')))

def convert_arch_weights_to_numpy(arch_weights, k=14, num_ops=len(PRIMITIVES)):
	# key = ['edge: (%d, %d), op: %s' %(i,j,op) for op in PRIMITIVES for i in range(3) for j in range(5)]
	# get last epoch alpha
	alphas=np.zeros((k, num_ops))
	mm = 0
	last_id = 1
	node_id = 0
	for i in range(k):
		for j in range(num_ops):
			alphas[i,j] = arch_weights['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])][-1]
		if mm == last_id:
			mm = 0
			last_id += 1
			node_id += 1
		else:
			mm += 1
    
	return alphas
    		

def load_arch_weights(model, arch_weights_dir):
	for sub in ['normal', 'reduce']:
		_file = os.path.join(arch_weights_dir, sub, 'history_weight.json')
		with open(_file, 'r') as f:
			arch_weights = json.load(f)
			arch_weights = convert_arch_weights_to_numpy(arch_weights)
			if sub == 'normal':
				d_ = model.alphas_normal.detach()
				d_.copy_(torch.as_tensor(arch_weights))
			else:
				d_ = model.alphas_reduce.detach()
				d_.copy_(torch.as_tensor(arch_weights))