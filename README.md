# Spherical Semi-Orthogonal Convolution

> A PyTorch-based implementation of **Spherical Semi-Orthogonal (SSO) Convolutional Layers**, designed for robustness and redundancy reduction in deep neural networks.  
> This work is based on our ICML submission, which introduces a novel transformation inspired by geometric principles in the real projective space.

---

## Overview

Convolutional layers with orthogonal constraints have been widely studied to enhance adversarial robustness and mitigate feature redundancy.  
While **semi-orthogonal layers** serve as a practical alternative when weight matrices are non-square, they fall short in effectively reducing covariance among features.

In this project, we introduce a new transformation called **Spherical Semi-Orthogonal (SSO)**, which extends semi-orthogonality by promoting **even distribution of layer weights in the real projective space**.  
We demonstrate its superior performance over existing methods such as **Cayley**, **SOC**, and **ECO** in both **certified robustness** and **feature diversity**.

---

## Key Contributions

- **Theoretical Foundation**: We define SSO transformations using geometric properties of real projective spaces.
- **SSO Convolutional Layers**: Implementations in both spatial and frequency domains using 2D DFT.
- **Redundancy Metrics**: New energy-based and correlation-based metrics to evaluate feature overlap.
- **Empirical Gains**: Demonstrated improvements in standard accuracy and robustness across multiple datasets and architectures.

---

## Key Updates
- Jan 5, 2022
      - Completed experiment setup (Adam optimizer, LR_max: 0.01, always use standardization, etc.)
      - When only --conv is specified, the corresponding default linear function is automatically selected.
      - Fixed bug in SESLinear.
      - PlainConv can be used with all backbones except LipConvNet.
      - Added support for CIFAR-100. Use --dataset cifar100.
      - For the SES method, an additional argument --lam (default: 1.7) is available.
      - For LipConvNet, use --n_lip (default: 1), which corresponds to LipConvNet-5 when set to 1.
- Jan 6, 2022
      - Updated SES-T method. Now compatible with all backbones.
      - When using SESConv2dFT, if --linear is not specified, SESLinearT is used by default.
- Jan 21, 2022
      - Updated SES implementation. Renamed ses_t.py to ses.py.
      - Added --scale argument. Scaling is now applied in extract_SESLoss instead of inside SESLinear.
- Jan 14, 2022
      - Added hyperparameter tuning using python hyperparam_ray_tune.py.
      - Introduced main_for_tune.py.
      - Changed default loss from Margin Loss to Cross Entropy (CE).
- Jan 16, 2022
      - Removed gradient clipping.
      - Updated defaults: --lam 2.0, --scale 2.0.
- Jan 20, 2022
      - Updated optimal defaults to --lam 10.0, --scale 2.0.
      - For LipConvNet-n, use learning rate 0.0001 when n ≥ 30; otherwise use 0.001.
      - Added ses_new.py, a more concise version.

## Commands
- Important Arguments

```sh
## Baseline
python main.py --gpu 0 --exp_name Exp --conv PlainConv --backbone KWLarge --seed 1

## Cayley
python main.py --gpu 0 --exp_name Exp --conv CayleyConv --backbone ResNet9 --seed 1

## SES
python main.py --gpu 0 --exp_name Exp --conv SESConv --backbone ResNet9 --seed 1
## SES + CayleyLinear, LipConvNet-10
python main.py --gpu 0 --exp_name Exp --conv SESConv --linear CayleyLinear --lam 1.7 --backbone LipConvNet --n_lip 2 --seed 1

```

- If `--linear` is not explicitly set (default: 'none'), the predefined linear layer corresponding to the given conv is used automatically

'PlainConv'    : 'Linear',         
'BCOP' : 'BCOP', 'SOC' : 'SOC', 'ECO': 'ECO',
'CayleyConv'   : 'CayleyLinear',
'SESConv'      : 'SESLinear', 'SESConv1x1'    : 'SESLinear', 

- For argument details, refer to utils/option.py.

---
## Tensorboard
Experiment results are stored under exps/EXP_NAME, including TensorBoard logs.

```sh
# Launch tensorboard
tensorboard --logdir exps --host 0.0.0.0 --port 6006
```


---

## Precaution & TODO List
- The kernel size for all conv layers is 1 by default.
