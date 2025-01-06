
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

### 🗝️ Key Features

- Handles different types of depth map degradations:
  - Simulated random and textual missing pixels
  - Holes found in Kinect depth maps
- With only a maximum of 1.8 million parameters, our model is edge-friendly and compact.
- Generalization capability across various indoor depth datasets without additional fine-tuning.
- Adaptable to existing works, supporting diverse computer vision applications.
![Teaser Image](assets/teaser.png)

## :hammer_and_wrench: Installation and Running
### 💻 Prerequisites
Install torch: http://torch.ch/docs/getting-started.html
   
Install the `matio` package using the following command:
```bash
luarocks install --server=https://luarocks.org/dev matio 
```
**CUDA (Optional)**:
   - For GPU support, ensure CUDA is installed and configured correctly.
  
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
### 💃 Demo
If you want to run  quick demos for depth completion corresponding to the two cases of our depth degradations: 90% random missing depth values and real Kinect depth degradation, please download our pre-trained models.
 
All models are trained on a single GeForce GTX 1080 Ti GPU with the largest possible batch size.

| Degradation type            | :inbox_tray:Models           | Params | Performance |
|-----------------------------|:-------:|:-------:|:-------:|
| Simulated random (90%)      | [GAN-RM(90%)](https://www.dropbox.com/scl/fi/ce2wxefifs4vf1gkwdmmb/DC_chk_90.t7?rlkey=uadjku5hqdkb1gs0fmfoac1je&st=pxuj4j5o&dl=1) | 1.45M | 33.70 dB [PSNR/[Middlebury](https://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf)]                                                     |
| Real Kinect depth map holes | [GAN-Real](https://www.dropbox.com/scl/fi/12nmxojuljmwk8km39jmc/DC_chk_Real.t7?rlkey=vgqcf8o00orsguab34lsgb2bq&st=i23va6n7&dl=1)     | 1.81M | 40.77 dB [PSNR/K_deg], 1.54m [RMSE/[Matterport3D-500](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8374622)], 1.49m [RMSE/[Matterport3D-474](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8374622)]|


The entries in the **Performance** column specify the evaluation metric followed by the dataset used for testing, as indicated inside the brackets. For example, in "PSNR/Middlebury," the first part refers to the evaluation metric, in this case, PSNR (Peak Signal-to-Noise Ratio), while the second part specifies the dataset, Middlebury dataset. Similarly, "PSNR/K_deg" refers to the PSNR metric evaluated on a dataset with structural Kinect-like degradation. Metrics such as RMSE (Root Mean Square Error) followed by datasets like Matterport-500 and Matterport-474 indicate RMSE values measured in meters for the respective subsets of the Matterport3D dataset. This format ensures that both the metric and the dataset used are clearly associated in each entry.


Download and save the `pretrained model(s)` to `./chk`.

Execute the testing script from the command line:
  
  ```Shell
# Test the depth completion model for real Kinect holes
 th test_realKinectholes.lua

# Test the depth completion model for 90% randomly missing depth values
 th test_randommissing.lua

# Note: If you are running on cpu, use gpu=0
  ```

### ⏳ Training
We used the :floppy_disk:[NYU-Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) dataset for all training purposes and have provided the trained models above. Please refer to our paper for more details. If you want to train your own generator and discriminator models or to train on a different dataset, run the training file `train.lua`.
```Shell
 th train.lua
```
## 🎓 BibTeX Citation

The BibTeX citation will be provided soon.


## 🙏 Acknowledgement

We thank the anonymous reviewers for their constructive reviews. 
Some part of the training code is adapted from an initial fork of [Soumith's DCGAN](https://github.com/soumith/dcgan.torch) implementation. We'd like to thank the authors for making these frameworks available.

## 📜 License
This project is licensed under the The MIT License (MIT).

---
Feel free to reach out [✉️](mailto:medhi.moushumi@gmail.com) if you encounter any issues!

 

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



##  Testing


## 🔭 Introduction


##  Requirements

## 🔧 Installation



## ✏️ Test


## :clapper: Introduction

##  Pretrained Models

###  Setup Instructions

## :art: Qualitative Results

In this section, we present illustrative examples that demonstrate the effectiveness of our proposal.

<br>

<p float="left">
  <img src="./images/competitors.png" width="800" />
</p>
 

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


##  Getting Started


## 🎨 Gradio Demo 


mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm end

