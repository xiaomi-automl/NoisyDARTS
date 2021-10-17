from __future__ import print_function, division
import os
import argparse
import torch
import torch.nn as nn
from thop import clever_format, profile
from timm.models import EfficientNet
from timm.models.efficientnet_builder import decode_arch_def
from timm.models.efficientnet_blocks import resolve_bn_args

from dataloader import get_imagenet_dataset
from utils import accuracy

def get_args():
    parser = argparse.ArgumentParser("Evaluate NoisyDARTS-SE Models on ImageNet")
    parser.add_argument('--model', type=str, required=True, help='model to evaluate')
    parser.add_argument('--model-path', type=str, default='pretrained/noisy_darts_a_se.pth.tar', help='dir to models')
    parser.add_argument('--se-ratio', default=0.25, type=float, help='squeeze-and-excitation ratio')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch-size', default=100, type=int, help='val batch size')
    parser.add_argument('--val-dataset-root', default='/Your_Root/ILSVRC2012', help="val dataset root path")
    args = parser.parse_args()
    return args

def evaluate(args):
    torch.cuda.set_device(args.gpu)
        
    # NoisyDARTS-A model specification
    s_r = args.se_ratio
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

    device = torch.device(args.device)
    if args.device == 'cuda':
        model.cuda()
    state = torch.load(f'{args.model_path}',  map_location=device)
    model.load_state_dict(state)

    _input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(_input,), verbose=False)
    print('Model: {}, params: {}M, flops: {}M'.format(args.model, params / 1e6, flops / 1e6))
    
    model.eval()
    val_dataloader = get_imagenet_dataset(batch_size=args.batch_size,
                                          dataset_root=args.val_dataset_root,
                                          dataset_tpye="valid")

    print("Start to evaluate ...")
    total_top1 = 0.0
    total_top5 = 0.0
    total_counter = 0.0
    for image, label in val_dataloader:
        image, label = image.to(device), label.to(device)
        result = model(image)
        top1, top5 = accuracy(result, label, topk=(1, 5))
        if device.type == 'cuda':
            total_counter += image.cpu().data.shape[0]
            total_top1 += top1.cpu().data.numpy()
            total_top5 += top5.cpu().data.numpy()
        else:
            total_counter += image.data.shape[0]
            total_top1 += top1.data.numpy()
            total_top5 += top5.data.numpy()
    mean_top1 = total_top1 / total_counter
    mean_top5 = total_top5 / total_counter
    print('Evaluate Result: Total: %d\tmTop1: %.4f\tmTop5: %.6f' % (total_counter, mean_top1, mean_top5))


if __name__ == '__main__':
    args = get_args()
    evaluate(args)
