#!/bin/sh

# verify CIFAR-10 models
python verify.py --auxiliary --arch noisy_darts_a --model-path pretrained/noisy_darts_a.pt

# verify ImageNet models
python verify_se.py  --model noisy_darts_a --model-path pretrained/noisy_darts_a_se.pth.tar

# verify Transferred models on CIFAR-10
python verify_transfer.py --model-path/pretrained/noisy_darts_a_transfer.pt.tar