
<!---
# Depth-Completion
[![Depth Completion](link)](#)
-->
<p align="center">
  <h1 align="center"><img src="assets/DC.png" width="85"></ins><br>Adversarial Learning for Unguided Single Depth Map Completion of Indoor Scenes</h1>
  <p align="center">
    <a href="#">Moushumi&nbsp;Medhi*</a>
    ·
    <a href="https://www.iitkgp.ac.in/department/EE/faculty/ee-rajiv">Rajiv Ranjan&nbsp;Sahay</a>
    <br>   
  </p>
<p align="center">
    <img src="assets/IIT_Kharagpur_Logo.svg.png" alt="Logo" height="60">
</p>
<p align="center">
    <strong>IIT Kharagpur</strong>
</p>

---
##
🚨 This repository hosts the source code for unguided single-depth map completion of indoor scenes—a lightweight design for restoring noisy depth maps in a realistic manner using a generative adversarial network (GAN).
## 📖 Abstract
Depth map completion without guidance from color images is a challenging, ill-posed problem. Conventional methods rely on computationally intensive optimization processes. This work proposes a deep adversarial learning approach to estimate missing depth information directly from a single degraded observation, without requiring RGB guidance or postprocessing.

### Key Features

- Handles different types of depth map degradations:
  - Simulated random and textual missing pixels
  - Holes found in Kinect depth maps
- With only a maximum of 1.8 million parameters, our model is edge-friendly and compact.
- Generalization capability across various indoor depth datasets without additional fine-tuning.
- Adaptable to existing works, supporting diverse computer vision applications.
![Teaser Image](assets/teaser.png)
## 🎓 BibTeX Citation

The BibTeX citation will be provided soon.


## 🙏 Acknowledgement

We thank the anonymous reviewers for their constructive reviews. 
Some part of the training code is adapted from an initial fork of [Soumith's DCGAN](https://github.com/soumith/dcgan.torch) implementation. We'd like to thank the authors for making these frameworks available.
## Installation and Running
### Prerequisites
Install torch: http://torch.ch/docs/getting-started.html
   
Install the `matio` package using the following command:
```bash
luarocks install --server=https://luarocks.org/dev matio 
```
**CUDA (Optional)**:
   - For GPU support, ensure CUDA is installed and configured correctly.
   - 
**Setup dependencies**

Ensure you have Torch7 installed along with the following required packages:

```lua
require 'nn'
require 'optim'
require 'image'
require 'cunn'
require 'autograd'
require 'torch'
require 'ffi'
require('pl.class')
```
Other dependencies required:
- `cunn`
- `optim`
- `autograd`
- `threads`


Clone the repository
  ```Shell
  git clone https://github.com/Moushumi9medhi/Depth-Completion.git
  cd Depth-Completion
  ```
### Demo
If you want to run  quick demos for depth completion corresponding to the two cases of our depth degradations: 90% random missing depth values and real Kinect depth degradation, please download our pre-trained models.
 
All models are trained on a single GeForce GTX 1080 Ti GPU with the largest possible batch size.

| Degradation type            | Models           | Params | Performance |
|-----------------------------|:-------:|:-------:|:-------:|
| Simulated random (90%)      | [GAN-RM(90%)](https://www.dropbox.com/scl/fi/ce2wxefifs4vf1gkwdmmb/DC_chk_90.t7?rlkey=uadjku5hqdkb1gs0fmfoac1je&st=pxuj4j5o&dl=1) | 1.45M | 33.70 dB [PSNR/[Middlebury](https://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf)]                                                     |
| Real Kinect depth map holes | [GAN-Real](https://www.dropbox.com/scl/fi/12nmxojuljmwk8km39jmc/DC_chk_Real.t7?rlkey=vgqcf8o00orsguab34lsgb2bq&st=i23va6n7&dl=1)     | 1.81M | 40.77 dB [PSNR/K_deg], 1.54m [RMSE/[Matterport3D-500](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8374622)], 1.49m [RMSE/[Matterport3D-474](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8374622)]|


The entries in the **Performance** column specify the evaluation metric followed by the dataset used for testing, as indicated inside the brackets. For example, in "PSNR/Middlebury," the first part refers to the evaluation metric, in this case, PSNR (Peak Signal-to-Noise Ratio), while the second part specifies the dataset, Middlebury dataset. Similarly, "PSNR/K_deg" refers to the PSNR metric evaluated on a dataset with structural Kinect-like degradation. Metrics such as RMSE (Root Mean Square Error) followed by datasets like Matterport-500 and Matterport-474 indicate RMSE values measured in meters for the respective subsets of the Matterport3D dataset. This format ensures that both the metric and the dataset used are clearly associated in each entry.


Download and save the `pretrained model(s)` to `./chk`.

Execute the testing script from the command line:
  
  ```Shell
 # Test the depth completion model for 90% randomly missing depth values
 th test_realKinectholes.lua

# Test the depth completion model for real Kinect holes
 th test_randommissing.lua

# Note: If you are running on cpu, use gpu=0
  ```

---
Feel free to reach out [✉️](mailto:medhi.moushumi@gmail.com) if you encounter any issues!
### Training

2. Train the unguided indoor depth completion model in order to obtain a discriminator network and a generator network. I have already trained the model on the [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).... and put the corresponding networks into the `checkpoints` folder. If you want to train it again or use a different dataset run
```
DATA_ROOT=<dataset_folder> name=<whatever_name_you_want> th main.lua
```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Inpainting using context encoder trained jointly with reconstruction and adversarial loss.
XXXXXXXXXXXXXXXXXXXXX
Train the model
  ```Shell
  # Train your own pixel interpolation model
  cd pixelInterpolation
  DATA_ROOT=../dataset/my_train_set name=pixel niter=250 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
  ```
 
xxxxxxxxxxxxxxxxxxxxxxxxxxxx
  Train the model
  ```Shell
  # Train your own image deblurring model
  cd deblurring
  DATA_ROOT=../dataset/my_train_set name=deblur niter=250 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
  ```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Train the model
  ```Shell
  # Train your own image denoising model
  cd denoising
  DATA_ROOT=../dataset/my_train_set name=denoise niter=1500 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
  ```
 
 

    
## 📜 License
This project is licensed under the The MIT License (MIT).

---

** :envelope: For any queries, feel free to raise an issue or contact us directly via [email](mailto:medhi.moushumi@iitkgp.ac.in).**
2222222222222222222222 start
https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
nyu data
### Results
Quanlitative Evaluation On NYU online test dataset

<img src="./results/NYU.png" width = "536" height = "300" alt="NYU" />

Quantitative Evaluation On NYU online test dataset

<img src="./results/NYU_results.jpg" width = "556" height = "336" alt="NYU_table" />

Quanlitative Evaluation On KITTI online test dataset

<img src="./results/KITTI.png" width = "930" height = "530" alt="KITTI" />

Quantitative Evaluation On KITTI online test dataset

<img src="./results/KITTI_results.jpg" width = "600" height = "400" alt="KITTI_table" />



## ⚙️ Setup

## 💾 Datasets
We used two datasets for training and evaluation.

## ⏳ Training


## 📊 Testing


## 🔭 Introduction


## 💻 Requirements
The code has been trained on:
- Ubuntu 20.04
- CUDA 11.3
- Python 3.9.18
- Pytorch 1.12.1
- GeForce RTX 4090 $\times$ 2.

## 🔧 Installation


## 💾 Datasets
We used two datasets for training and three datasets for evaluation.



## 🚅 Pretrained model

## ✏️ Test


## 💡 Citation

## :clapper: Introduction

## :inbox_tray: Pretrained Models

### :hammer_and_wrench: Setup Instructions

## :floppy_disk: Datasets
We used two datasets for training and evaluation.
## :art: Qualitative Results

In this section, we present illustrative examples that demonstrate the effectiveness of our proposal.

<br>

<p float="left">
  <img src="./images/competitors.png" width="800" />
</p>
 

## :envelope: Contacts
## :pray: Acknowledgements


## Download Dataset \& Pretrained Model
 
 

## Training
**RoofDiffusion**

Use `roof_completion.json` for training the RoofDiffusion model with the footprint version, where in each footprint image, a pixel value of 1 denotes the building footprint and 0 denotes areas outside the footprint. 
```console
python run.py -p train -c config/roof_completion.json
```

**No-FP RoofDiffusion**

Use `roof_completion_no_footprint.json` for training with footprint images where all pixels are set to 1, indicating no distinction between inside and outside footprint areas.
```console
python run.py -p train -c config/roof_completion_no_footprint.json
```

See training progress
```console
tensorboard --logdir experiments/train_roof_completion_XXXXXX_XXXXXX
```

## Inference
**RoofDiffusion**
```console
python run.py -p test -c config/roof_completion.json \
    --resume ./pretrained/w_footprint/260 \
    --n_timestep 500 \
    --data_root ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/img.flist \
    --footprint_root ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/footprint.flist
```

**No-FP RoofDiffusion**
```console
python run.py -p test -c config/roof_completion_no_footprint.json \
    --resume ./pretrained/wo_footprint/140 \
    --n_timestep 500 \
    --data_root ./dataset/PoznanRD/benchmark/wo_footprint/s95_i30/img.flist \
    --footprint_root ./dataset/PoznanRD/benchmark/wo_footprint/s95_i30/footprint.flist
```

> Tested on NVIDIA RTX3090. Please adjust `batch_size` in JSON file if out of GPU memory.


## Customize Data Synthesis for Training
Modify JSON config file:
- `"down_res_pct"` controls sparsity.
- `"local_remove"` adjusts local incompleteness (Please refer to paper for details).
- `"noise_config"` injects senser/environmental noise.
- `"height_scale_probability"` randomly scales the distance between the min-max nonzero roof height.
- `"tree"` introduce tree noise into height maps





22222222222222222222222222222 end
mmmmmmmmmmmmmmmmmmmmmmmmmm start
<!-- # magic-edit.github.io -->

<p align="center">

  <h2 align="center">MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=-4iADzMAAAAJ&hl=en"><strong>Zhongcong Xu</strong></a>
    ·
    <a href="http://jeff95.me/"><strong>Jianfeng Zhang</strong></a>
    ·
    <a href="https://scholar.google.com.sg/citations?user=8gm-CYYAAAAJ&hl=en"><strong>Jun Hao Liew</strong></a>
    ·
    <a href="https://hanshuyan.github.io/"><strong>Hanshu Yan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=stQQf7wAAAAJ&hl=en"><strong>Jia-Wei Liu</strong></a>
    ·
    <a href="https://zhangchenxu528.github.io/"><strong>Chenxu Zhang</strong></a>
    ·
    <a href="https://sites.google.com/site/jshfeng/home"><strong>Jiashi Feng</strong></a>
    ·
    <a href="https://sites.google.com/view/showlab"><strong>Mike Zheng Shou</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2311.16498"><img src='https://img.shields.io/badge/arXiv-MagicAnimate-red' alt='Paper PDF'></a>
        <a href='https://showlab.github.io/magicanimate'><img src='https://img.shields.io/badge/Project_Page-MagicAnimate-green' alt='Project Page'></a>
        <a href='https://huggingface.co/spaces/zcxu-eric/magicanimate'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
    <br>
    <b>National University of Singapore &nbsp; | &nbsp;  ByteDance</b>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/teaser/t4.gif">
    </td>
    <td>
      <img src="assets/teaser/t2.gif">
    </td>
    </tr>
  </table>


## 🏃‍♂️ Getting Started



## ⚒️ Installation


## 💃 Inference


## 🎨 Gradio Demo 

 
### 2) Train Context Encoders

If you could successfully run the above demo, run following steps to train your own context encoder model for image inpainting.

0. [Optional] Install Display Package as follows. If you don't want to install it, then set `display=0` in `train.lua`.
  ```Shell
  luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
  cd ~
  th -ldisplay.start 8000
  # if working on server machine create tunnel: ssh -f -L 8000:localhost:8000 -N server_address.com
  # on client side, open in browser: http://localhost:8000/
  ```

1. Make the dataset folder.
  ```Shell
  mkdir -p /path_to_wherever_you_want/mydataset/train/images/
  # put all training images inside mydataset/train/images/
  mkdir -p /path_to_wherever_you_want/mydataset/val/images/
  # put all val images inside mydataset/val/images/
  cd context-encoder/
  ln -sf /path_to_wherever_you_want/mydataset dataset
  ```

2. Train the model
  ```Shell
  # For training center region inpainting model, run:
  DATA_ROOT=dataset/train display_id=11 name=inpaintCenter overlapPred=4 wtl2=0.999 nBottleneck=4000 niter=500 loadSize=350 fineSize=128 gpu=1 th train.lua

  # For training random region inpainting model, run:
  DATA_ROOT=dataset/train display_id=11 name=inpaintRandomNoOverlap useOverlapPred=0 wtl2=0.999 nBottleneck=4000 niter=500 loadSize=350 fineSize=128 gpu=1 th train_random.lua
  # or use fineSize=64 to train to generate 64x64 sized image (results are better):
  DATA_ROOT=dataset/train display_id=11 name=inpaintRandomNoOverlap useOverlapPred=0 wtl2=0.999 nBottleneck=4000 niter=500 loadSize=350 fineSize=64 gpu=1 th train_random.lua
  ```
 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm end

