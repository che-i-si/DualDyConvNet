# DualDyConvNet

Published paper:

    Jang, Y., Jeong, J., Kim, Y.K., Kim, D.H., Park, W., Kim, L., Kim, Y.H. and Lee, M., 2025. 
    DualDyConvNet: Dual-Stream Dynamic Convolution Network 
    via Parameter-Efficient Fine-Tuning for Predicting Motor Prognosis in Subacute Stroke. 
    IEEE Transactions on Neural Systems and Rehabilitation Engineering.

---
# Data
## Input data
Stacked topological maps of EEG power spectral density ($X\in\mathbb{R}^{-1\times F\times H\times W}$).
- $F$: the number of frequency bands
- $H\times W$: the size of each topological map

## Training dataset
(input data $X_\text{train}$, true labels on target 1 $Y^1_\text{train}$, true labels on target 2 $Y^2_\text{train}$)
- In the paper, target 1 and target 2 are `pre-score` and `post-score`, respectively

## Fine-tuning and test dataset
(input data $X_\text{test}$, true labels on target 1 $Y^1_\text{test}$)
