#!/bin/sh

# e.g.
nohup python -u train.py --auxiliary --cutout --save noisy_darts_a --arch noisy_darts_a --gpu 0 > noisy_darts_a.log 2>&1 &