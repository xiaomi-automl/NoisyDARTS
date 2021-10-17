
# Noisy Differentiable Architecture Search [BMVC 2021]

This repository includes the implementation of NoisyDARTS. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

We use CIFAR-10 and ImageNet datasets, which can be downloaded by `torchvision` automatically. We follow the standard preprocessing for both datasets.

## Searching

To perform a standard NoisyDARTS search, execute for example:

```searching
nohup python -u train_search.py --exp_name noisy_darts_a --factor_skip 0.2 --add_noise_skip --gpu 0 --seed 10 > noisy_darts_a.log 2>&1 &

```

## Training

To train the model(s) in the paper, run for example:

```train
nohup python -u train.py --auxiliary --cutout --save noisy_darts_a --arch noisy_darts_a --gpu 0 > noisy_darts_a.log 2>&1 &
```


## Evaluation

To evaluate CIFAR models, run for example:

```eval
python verify.py --auxiliary --arch noisy_darts_a --model-path pretrained/noisy_darts_a.pt
```

More evalutaion commands can be found in `scripts/run_verify.sh`.

## Pre-trained Models

You can download pretrained models here:

- All [NoisyDARTS models](https://drive.google.com/drive/folders/1sf5D-7Le0_MLE5w39rZ5ErY2Wd42dEs8?usp=sharing) can be downloaded from this link.

## Results

### NoisyDARTS Models Searched on CIFAR-10

| Models         | Search Strategy | Multiply-adds (M)| Parameters (M) | Top-1 (%) | 
| -------------- | ----------------| ----------- | ----------- | ---- |
| NoisyDARTS-a   | Gaussian, lambda=0.2 |   534   | 3.25 | **97.61** |
| NoisyDARTS-b   | Gaussian, lambda=0.1 |   511   | 3.09 | 97.53 |
| NoisyDARTS-c   | Uniform, lambda=0.2  |   539   | 3.33 | 97.40  |
| NoisyDARTS-d   | Uniform, lambda=0.1  |   501   | 3.06 | 97.42 |
| NoisyDARTS-e   | Gaussian, Multiplicative, std=0.1 |   539   | 3.24 | 97.55 |
| NoisyDARTS-f   | Gaussian, Multiplicative, std=0.2 | 443     | 2.68 | 97.18 |
| NoisyDARTS-g   | Gaussian, lambda=0.2, mean=0.5  | 549     | 3.32 | 97.49 |
| NoisyDARTS-h   | Gaussian, lambda=0.2, mean=1.0 | 511     | 3.01 | 97.35 |
| NoisyDARTS-i   | Gaussian, lambda=0.1, mean=0.5 | 495     | 3.07 | 97.28 |
| NoisyDARTS-j   | Gaussian, lambda=0.1, mean=1.0 | 476     | 2.94 | 97.21 |

Our model achieves the following performance on :

### Image Classification on ImageNet

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| NoisyDARTS-A       |     77.9%       |      94.0%     |



## Citation

```
@inproceedings{chu2021noisy,
  title={Noisy Differentiable Architecture Search},
  author={Chu, Xiangxiang and Zhang, Bo},
  booktitle={BMVC},
  year={2021}
}
```

