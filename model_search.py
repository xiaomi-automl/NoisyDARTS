import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
from utils import drop_path


class CosineDecayScheduler(object):
	def __init__(self, base_lr=1.0, last_iter=0, T_max=50):
		self.base_lr = base_lr
		self.last_iter = last_iter
		self.T_max = T_max
		self.cnt = 0

	def decay_rate(self, step):
		self.last_iter = step
		decay_rate = self.base_lr * (1 + math.cos(math.pi * self.last_iter / self.T_max)) / 2.0 if self.last_iter <= self.T_max else 0
		return decay_rate

class NoiseOp(nn.Module):
	def __init__(self, noise_type, factor, mean, noise_mixture, decay_scheduler=None, add_noise=True, args=None):
		super(NoiseOp, self).__init__()
		self.noise_type = noise_type
		self.factor = factor # factor for std
		self.mean = mean
		self.noise_mixture = noise_mixture
		self.decay_scheduler = decay_scheduler
		self.add_noise = add_noise
		self.args=args

	def forward(self, x, step):
		if self.training and self.add_noise:
			if self.noise_type == 'uniform':
				# uniform variance is (b-a)^2/12, so a = sqrt(3*factor)
				# uniform takes between (-1,1) * a
				a = np.sqrt(3*self.factor)
				noise = self.mean + (-2 * torch.rand_like(x) + 1) * a
			elif self.noise_type == 'gaussian':
				# normal distribution
				std = x.std() * self.factor if self.noise_mixture == 'additive' else self.factor
				means = self.mean + torch.zeros_like(x, device=torch.device("cuda"), requires_grad=False)
				noise = torch.normal(means, std, out=None).cuda()
			else:
				assert False, 'Not supported noise type'

			if self.decay_scheduler is not None:
				decay_rate = self.decay_scheduler.decay_rate(step)
			else:
				decay_rate = 1
			
			if self.noise_mixture == 'additive':
				x = x + noise * decay_rate
				# x = noise
			elif self.noise_mixture == 'multiplicative':
				x = x * noise * decay_rate
			else:
				assert False, 'Not supported noise mixture'

		return x

class MixedOp(nn.Module):
	def __init__(self, C, stride, drop_prob, reduction, args):
		super(MixedOp, self).__init__()
		self.reduction = reduction
		self.args = args
		self.drop_prob = drop_prob
		self._ops = nn.ModuleList()
		if self.args.noise_decay:
			noise_decay_schedular = CosineDecayScheduler()
		else:
			noise_decay_schedular = None

		self.noise_skip = NoiseOp(self.args.noise_type, self.args.factor_skip, self.args.noise_mean, self.args.noise_mixture,
						decay_scheduler=noise_decay_schedular, add_noise=self.args.add_noise_skip, args=self.args)

		for primitive in PRIMITIVES:
			op = OPS[primitive](C, stride, False)
			if 'max_pool' in primitive:
				self.bn1 = nn.BatchNorm2d(C, affine=False)
			if 'avg_pool' in primitive:
				self.bn2 = nn.BatchNorm2d(C, affine=False)
			self._ops.append(op)

	def forward(self, x, weights, step):
		temp = []
		for i, (w, primitive, op) in enumerate(zip(weights, PRIMITIVES, self._ops)):
			if 'max_pool' in primitive:
				temp.append(w * self.bn1(op(x)))
			elif 'avg_pool' in primitive:
				temp.append(w * self.bn2(op(x)))
			elif 'skip' in primitive:
				temp.append(w * drop_path(self.noise_skip(op(x), step), self.drop_prob))
			else:
				temp.append(w * op(x))
		res = sum(temp)
		return res


class Cell(nn.Module):
	def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, drop_prob, args):
		super(Cell, self).__init__()
		self.reduction = reduction
		self.args = args
		
		if reduction_prev:
			self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
		else:
			self.preprocess0 = ReLUConvBN(C_prev_prev, C, kernel_size=1, stride=1, padding=0, affine=False)
		self.preprocess1 = ReLUConvBN(C_prev, C, kernel_size=1, stride=1, padding=0, affine=False)
		self._steps = steps
		self._multiplier = multiplier
		
		self._ops = nn.ModuleList()
		self._bns = nn.ModuleList()
		for i in range(self._steps):
			for j in range(2+i):
				stride = 2 if reduction and j < 2 else 1
				op = MixedOp(C, stride, drop_prob, reduction, self.args)
				self._ops.append(op)

	def forward(self, s0, s1, weights, epoch):
		s0 = self.preprocess0(s0)
		s1 = self.preprocess1(s1)
		
		states = [s0, s1]
		offset = 0
		for i in range(self._steps):
			s = sum(self._ops[offset+j](h, weights[offset+j], epoch) for j, h in enumerate(states))
			offset += len(states)
			states.append(s)
		
		return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
	def __init__(self, C, num_classes, layers, criterion, args=None, steps=4, multiplier=4, stem_multiplier=3, drop_prob=0.3):
		super(Network, self).__init__()
		self._C = C
		self._num_classes = num_classes
		self._layers = layers
		self._criterion = criterion
		self._steps = steps
		self._multiplier = multiplier
		self._drop_prob = drop_prob
		self.args = args
		
		C_curr = stem_multiplier*C
		self.stem = nn.Sequential(
			nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
			nn.BatchNorm2d(C_curr)
		)

		C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
		self.cells = nn.ModuleList()
		reduction_prev = False
		for i in range(layers):
			if i in [layers//3, 2*layers//3]:
				C_curr *= 2
				reduction = True
			else:
				reduction = False
			cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction,
						reduction_prev, drop_prob, args)
			reduction_prev = reduction
			self.cells += [cell]
			C_prev_prev, C_prev = C_prev, multiplier*C_curr
		
		self.global_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(C_prev, num_classes)
		
		self._initialize_alphas()

	def new(self):
		model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data.copy_(y.data)
		return model_new

	def forward(self, input, epoch):
		s0 = s1 = self.stem(input)
		for i, cell in enumerate(self.cells):
			if cell.reduction:
				weights = F.softmax(self.alphas_reduce, dim=-1)
			else:
				weights = F.softmax(self.alphas_normal, dim=-1)
			s0, s1 = s1, cell(s0, s1, weights, epoch)
		out = self.global_pooling(s1)
		logits = self.classifier(out.view(out.size(0),-1))
		return logits

	def _loss(self, input, target, epoch):
		logits = self(input, epoch)
		return self._criterion(logits, target)

	def _initialize_alphas(self, alphas=None):
		k = sum(1 for i in range(self._steps) for n in range(2+i))
		num_ops = len(PRIMITIVES)
		
		if alphas is None:
			self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
			self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
		else:
			self.alphas_normal = Variable(torch.Tensor(alphas[0]).cuda(), requires_grad=True)
			self.alphas_reduce = Variable(torch.Tensor(alphas[1]).cuda(), requires_grad=True)

		# init the history
		self.alphas_normal_history = {}
		self.alphas_reduce_history = {}
		mm = 0
		last_id = 1
		node_id = 0
		for i in range(k):
			for j in range(num_ops):
				self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])] = []
				self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])] = []
			if mm == last_id:
				mm = 0
				last_id += 1
				node_id += 1
			else:
				mm += 1

		self._arch_parameters = [
			self.alphas_normal,
			self.alphas_reduce,
		]

	def arch_parameters(self):
		return self._arch_parameters

	def genotype(self):
		def _parse(weights):
			gene = []
			n = 2
			start = 0
			for i in range(self._steps):
				end = start + n
				W = weights[start:end].copy()
				if 'none' in PRIMITIVES:
					edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
				else:
					edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
				for j in edges:
					k_best = None
					for k in range(len(W[j])):
						if k_best is None or W[j][k] > W[j][k_best]:
							if 'none' in PRIMITIVES:
								k_best = k_best if k == PRIMITIVES.index('none') else k
							else:
								k_best = k
					gene.append((PRIMITIVES[k_best], j))
				start = end
				n += 1
			return gene

		gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
		gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
		
		concat = range(2+self._steps-self._multiplier, self._steps+2)
		genotype = Genotype(
			normal=gene_normal, normal_concat=concat,
			reduce=gene_reduce, reduce_concat=concat
		)
		return genotype

	def update_history(self):
		mm = 0
		last_id = 1
		node_id = 0
		weights1 = F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
		weights2 = F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy()
		
		k, num_ops = weights1.shape
		for i in range(k):
			for j in range(num_ops):
				self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(
					float(weights1[i][j]))
				self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(
					float(weights2[i][j]))
			if mm == last_id:
				mm = 0
				last_id += 1
				node_id += 1
			else:
				mm += 1
