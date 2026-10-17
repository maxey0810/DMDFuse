# DMDFuse: A dynamic Mamba decoupling network for infrared and visible image fusion
Cheng Fei, Zhisheng Chen∗, Jun Zhang, Haorui Peng, Huan Zhang

Published in: Infrared Physics & Technology

- [paper](https://doi.org/10.1016/j.infrared.2026.106941)

## Abstract
Benefiting from long-range modeling capabilities comparable to Transformers and linear computational complexity, Mamba is emerging as a core foundational architecture in computer vision. However, in the infrared and visible image fusion tasks, the tremendous potential of Mamba for frequency-domain perception and explicit structural modeling remains underexplored. This paper introduces DMDFuse, a novel spatial-frequency dynamic decoupling fusion network. Specifically, we first propose the Spectral Contextual Mamba Block (SCMB) to purify the high-frequency fine-grained structures of the source image, providing directionally decoupled, high-fidelity inputs for the fusion network. Secondly, to strengthen the fusion backbone, we design the Dynamic Routing Mamba Block (DRMB) featuring a structure-aware local dynamic scanning strategy. This mechanism enables data-driven token sequence reorganization and effectively enhances the network's causal modeling capacity for long-range semantics. Finally, the Discrepancy-Guided Attention Fusion (DGAF) module leverages cross-modal feature residuals as spatial priors to precisely compensate for information gaps between modalities. Coupled with a custom-designed correlation discrepancy loss, it establishes an enhanced theoretical constraint for cross-modal feature interaction. Extensive experiments on multiple benchmark datasets and downstream detection tasks demonstrate that DMDFuse achieves advanced performance in both subjective visual quality and objective quantitative evaluations.

## Framework
<img width="2434" height="1459" alt="2" src="https://github.com/user-attachments/assets/f599fb98-f8e5-40df-a8aa-bd53a95faa91" />


## Recommended Environment

 - [x] python 3.11.13
 - [x] torch 2.8.0+cu128
 - [x] mamba-ssm 2.3.1
 - [x] causal-conv1d 1.6.1

## To Train
We train our model on the MSRS dataset. Please prepare the training dataset in `./datasets/MSRS` and run the training program.

    python train.py

## To Test
Place the pre-trained model weights (`epoch_39.pth`) in the root directory, and then run the testing program.

    python test.py --root_path ./datasets/MSRS

## Citation

@article{FEI2027106941,
title = {DMDFuse: A dynamic Mamba decoupling network for infrared and visible image fusion},
journal = {Infrared Physics & Technology},
volume = {160},
pages = {106941},
year = {2027},
issn = {1350-4495},
doi = {https://doi.org/10.1016/j.infrared.2026.106941},
author = {Cheng Fei and Zhisheng Chen and Jun Zhang and Haorui Peng and Huan Zhang}
}
