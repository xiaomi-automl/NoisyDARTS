#!/bin/sh

# Gaussian, Additive, lambda=0.2, mean=0
nohup python -u train_search.py --exp_name noisy_darts_a --factor_skip 0.2 --add_noise_skip --gpu 0 --seed 10 > noisy_darts_a.log 2>&1 &

# Gaussian, Additive, lambda=0.1, mean=0
nohup python -u train_search.py --exp_name noisy_darts_b --factor_skip 0.1 --add_noise_skip --gpu 1 --seed 10 > noisy_darts_b.log 2>&1 &

# Uniform, Additive, lambda=0.2, mean=0
nohup python -u train_search.py --exp_name noisy_darts_c --factor_skip 0.2 --add_noise_skip --noise_type uniform --gpu 2 --seed 10 > noisy_darts_c.log 2>&1 &

# Uniform, Additive, lambda=0.1, mean=0
nohup python -u train_search.py --exp_name noisy_darts_d --factor_skip 0.1 --add_noise_skip --noise_type uniform --gpu 3 --seed 10 > noisy_darts_d.log 2>&1 &

# Gaussian, Multiplicative, lambda=0.2, mean=1
nohup python -u train_search.py --exp_name noisy_darts_e --factor_skip 0.2 --add_noise_skip --noise_mixture multiplicative --noise_mean 1 --gpu 4 --seed 10 > noisy_darts_e.log 2>&1 &

# Gaussian, Multiplicative, lambda=0.1, mean=1
nohup python -u train_search.py --exp_name noisy_darts_f --factor_skip 0.1 --add_noise_skip --noise_mixture multiplicative --noise_mean 1 --gpu 5 --seed 10 > noisy_darts_f.log 2>&1 &

# Gaussian, Additive, lambda=0.2, mean=0.5
nohup python -u train_search.py --exp_name noisy_darts_g --factor_skip 0.2 --add_noise_skip --noise_mean 0.5 --noise_decay --gpu 6 --seed 10 > noisy_darts_g.log 2>&1 &

# Gaussian, Additive, lambda=0.2, mean=1.0
nohup python -u train_search.py --exp_name noisy_darts_h --factor_skip 0.2 --add_noise_skip --noise_mean 1.0 --noise_decay --gpu 7 --seed 10 > noisy_darts_h.log 2>&1 &

# Gaussian, Additive, lambda=0.1, mean=0.5
nohup python -u train_search.py --exp_name noisy_darts_i --factor_skip 0.1 --add_noise_skip --noise_mean 0.5 --noise_decay --gpu 0 --seed 10 > noisy_darts_i.log 2>&1 &

# Gaussian, Additive, lambda=0.1, mean=1.0
nohup python -u train_search.py --exp_name noisy_darts_j --factor_skip 0.1 --add_noise_skip --noise_mean 1.0 --noise_decay --gpu 1 --seed 10 > noisy_darts_j.log 2>&1 &

# Original DARTS
nohup python -u train_search.py --exp_name darts_1 --gpu 2 --seed 10 > darts_1.log 2>&1 &
