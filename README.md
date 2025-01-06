
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

  ```Shell
 # Test the depth completion model for 90% randomly missing depth values
 th test_realKinectholes.lua

# Test the depth completion model for real Kinect holes
 th test_randommissing.lua

# Note: If you are running on cpu, use gpu=0
  ```
### Training
1. Choose a RGB-Depth dataset and create a folder with its name (ex: `mkdir celebA`). Inside this folder create a folder `images` containing your images.  .....swee how training  image folder was created...............check this........not right.....

 Download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [SUN397](http://vision.cs.princeton.edu/projects/2010/SUN/) dataset, or prepare your own dataset. 


*Note:* for the `celebA` dataset, run
```
DATA_ROOT=celebA th data/crop_celebA.lua
```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
1) Place inputs into the `input` folder. An input image and corresponding sparse metric depth map are expected:

    ```bash
    input
    ├── image                   # RGB image
    │   ├── <timestamp>.png
    │   └── ...
    └── sparse_depth            # sparse metric depth map
        ├── <timestamp>.png     # as 16b PNG
        └── ...
    ```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
We used preprocessed NYUv2 HDF5 dataset provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).

```bash
$ cd PATH_TO_DOWNLOAD
$ wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
$ tar -xvf nyudepthv2.tar.gz
```

After that, you will get a data structure as follows:

```
nyudepthv2
├── train
│    ├── basement_0001a
│    │    ├── 00001.h5
│    │    └── ...
│    ├── basement_0001b
│    │    ├── 00001.h5
│    │    └── ...
│    └── ...
└── val
    └── official
        ├── 00001.h5
        └── ...
```

Download the  that the original full NYUv2 dataset is available at the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

After preparing the dataset, you should generate a json file containing paths to individual images.

```bash
$ cd THIS_PROJECT_ROOT/utils
$ python generate_json_NYUDepthV2.py --path_root PATH_TO_NYUv2
```

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
- The Places models were trained on the [Places2 dataset](http://places2.csail.mit.edu/) and thus best performance is for natural outdoor images.
- While the Places2 models work on images of any size with arbitrary holes, we trained them on images with the smallest edges in the [256, 384] pixel range and random holes in the [96, 128] pixel range. Our models will work best on images with holes of those sizes.
- Significantly large holes or extrapolation when the holes are at the border of images may fail to be filled in due to limited spatial support of the model.
xxxxxxxxxxxxxxxxxxxxxxxxx
1. Make the dataset folder.
  ```Shell
  mkdir -p /path_to_wherever_you_want/mydataset/train/images/
  # put all training images inside mydataset/train/images/
  mkdir -p /path_to_wherever_you_want/mydataset/val/images/
  # put all val images inside mydataset/val/images/
  cd context-encoder/
  ln -sf /path_to_wherever_you_want/mydataset dataset
  ```
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





3) The `--save-output` flag enables saving outputs to the `output` folder. By default, the following outputs will be saved per sample:

    ```bash
    output
    ├── ga_depth                # metric depth map after global alignment
    │   ├── <timestamp>.pfm     # as PFM
    │   ├── <timestamp>.png     # as 16b PNG
    │   └── ...
    └── sml_depth               # metric depth map output by SML
        ├── <timestamp>.pfm     # as PFM
        ├── <timestamp>.png     # as 16b PNG
        └── ...
    ```


## ⚙️ Setup

## 💾 Datasets
We used two datasets for training and evaluation.

## ⏳ Training
$ python main.py --dir_data PATH_TO_KITTI_DC --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
    --patch_height 240 --patch_width 1216 --gpus 0,1,2,3 --loss 1.0*L1+1.0*L2 --lidar_lines 64 \
    --batch_size 3 --max_depth 90.0 --lr 0.001 --epochs 250 --milestones 150 180 210 240 \
    --top_crop 100 --test_crop --log_dir ../experiments/ --save NAME_TO_SAVE \
```

Please refer to the config.py for more options. 
Then you can access the tensorboard via http://YOUR_SERVER_IP:6006

## 📊 Testing

**Pretrained Checkpoints**: [NYUv2](https://drive.google.com/drive/folders/1GlMVhI1Auo9noimR6NN0S-QLwL04ypCb?usp=sharing), [KITTI_DC](https://drive.google.com/drive/folders/1Tp1XAU7D7HOMq_iLEGzvt4I15g_HGBeM?usp=sharing)!


## 👩‍⚖️ Acknowledgement
Besides, we also thank [DySPN](https://arxiv.org/abs/2202.09769) for providing their evalution results on KITTI DC.

<h2>
<a href="https://whu-usi3dv.github.io/SparseDC/" target="_blank">SparseDC: Depth completion from sparse and non-uniform inputs</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **SparseDC: Depth completion from sparse and non-uniform inputs**<br/>
> [Chen Long](https://chenlongwhu.github.io/), [Wenxiao Zhang](https://github.com/XLechter), [Zhe Chen](https://github.com/ChenZhe-Code), [Haiping Wang](https://hpwang-whu.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Peiling Tong](https://3s.whu.edu.cn/info/1028/1961.htm), [Zhen Cao](https://github.com/a4152684), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm)<br/>
> *Information Fusion 2024*<br/>
> [**Paper**](https://doi.org/10.1016/j.inffus.2024.102470) | [**Project-page**]() | [**Video**]()


## 🔭 Introduction
<p align="center" style="font-size:18px">
<strong>SparseDC: Depth completion from sparse and non-uniform inputs</strong>
</p>
<img src="media/teaser.png" alt="Network" style="zoom:50%;">

<p align="justify">
<strong>Abstract:</strong> We propose SparseDC, a model for <strong>D</strong>epth <strong>C</strong>ompletion of <strong>Sparse</strong> and non-uniform depth inputs. Unlike previous methods focusing on completing fixed distributions on benchmark datasets (e.g., NYU with 500 points, KITTI with 64 lines), SparseDC is specifically designed to handle depth maps with poor quality in real usage.
The key contributions of SparseDC are two-fold.
First, we design a simple strategy, called SFFM, to improve the robustness under sparse input by explicitly filling the unstable depth features with stable image features.
Second, we propose a two-branch feature embedder to predict both the precise local geometry of regions with available depth values and accurate structures in regions with no depth. The key of the embedder is an uncertainty-based fusion module called UFFM to balance the local and long-term information extracted by CNNs and ViTs. Extensive indoor and outdoor experiments demonstrate the robustness of our framework when facing sparse and non-uniform input depths.
</p>

## 🆕 News
- 2024-04-10: [SparseDC](https://doi.org/10.1016/j.inffus.2024.102470) is accepted by Information Fusion! 🎉
- 2023-12-04: Code, [Preprint paper](https://arxiv.org/pdf/2312.00097) are available! 🎉

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

#### NYU Depth V2 (NYUv2)

We used preprocessed NYUv2 HDF5 dataset provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).

```bash
$ cd PATH_TO_DOWNLOAD
$ wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
$ tar -xvf nyudepthv2.tar.gz
```
Note that the original full NYUv2 dataset is available at the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).


Then, you should generate a json file containing paths to individual images. We use the data lists for NYUv2 borrowed from the [NLSPN repository](https://github.com/zzangjinsun/NLSPN_ECCV20/blob/master/data_json/nyu.json). You can put this json into your data dir.

After that, you will get a data structure as follows:

```
nyudepthv2
├── nyu.json
├── train
│    ├── basement_0001a
│    │    ├── 00001.h5
│    │    └── ...
│    ├── basement_0001b
│    │    ├── 00001.h5
│    │    └── ...
│    └── ...
└── val
    └── official
        ├── 00001.h5
        └── ...
```

## 🚅 Pretrained model

## ✏️ Test


## 💡 Citation
## :bookmark_tabs: Table of Contents
- [:bookmark\_tabs: Table of Contents](#bookmark_tabs-table-of-contents)
- [:clapper: Introduction](#clapper-introduction)
- [:inbox\_tray: Pretrained Models](#inbox_tray-pretrained-models)
- [:memo: Code](#memo-code)
  - [:hammer\_and\_wrench: Setup Instructions](#hammer_and_wrench-setup-instructions)
- [:floppy\_disk: Datasets](#floppy_disk-datasets)
- [:rocket: Test](#rocket-test)
- [:art: Qualitative Results](#art-qualitative-results)
- [:envelope: Contacts](#envelope-contacts)
- [:pray: Acknowledgements](#pray-acknowledgements)

</div>

## :clapper: Introduction

## :inbox_tray: Pretrained Models

### :hammer_and_wrench: Setup Instructions


## :floppy_disk: Datasets
We used two datasets for training and evaluation.

### NYU Depth V2 (NYUv2)

We used preprocessed NYUv2 HDF5 dataset provided by [Andrea Conti](https://github.com/andreaconti/sparsity-agnostic-depth-completion).

```bash
$ cd PATH_TO_DOWNLOAD
$ wget https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_img_gt.h5
$ wget https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_500.h5
```

After that, you will get a data structure as follows:

```
nyudepthv2
├── nyu_img_gt.h5
└── nyu_pred_with_500.h5
```

Note that the original full NYUv2 dataset is available at the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).


## :art: Qualitative Results

In this section, we present illustrative examples that demonstrate the effectiveness of our proposal.

<br>

<p float="left">
  <img src="./images/competitors.png" width="800" />
</p>
 
**Synth-to-real generalization.** Given an NYU Depth V2 frame and 500 sparse depth points (a), our framework with RAFT-Stereo trained only on the Sceneflow synthetic dataset (e) outperforms the generalization capability of state-of-the-art depth completion networks [NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20) (b), [SpAgNet](https://github.com/andreaconti/sparsity-agnostic-depth-completion) (c), and [CompletionFormer](https://github.com/youmi-zym/CompletionFormer) (d) – all trained on the same synthetic dataset.
 
<br>

<p float="left">
  <img src="./images/indoor2outdoor.png" width="800" />
</p>

**From indoor to outdoor.** When it comes to pre-training on SceneFlow and train on indoor data then run the model outdoor, significant domain shift occurs. NLPSN and CompletionFormer seem unable to generalize to outdoor data, while SpAgNet can produce some meaningful depth maps, yet far from being accurate. Finally, VPP4DC can improve the results even further thanks to the pre-training process.

<br>

<p float="left">
  <img src="./images/outdoor2indoor.png" width="800" />
</p>

**From outdoor to indoor.** We consider the case complementary to the previous one – i.e., with models pre-trained on SceneFlow and trained outdoor then tested indoor. NLSPN, CompletionFormer and SpAgNet can predict a depth map that is reasonable to some extent. Our approach instead predicts very accurate results on regions covered by depth hints, yet failing where these are absent.

## :envelope: Contacts

For questions, please send an email to luca.bartolomei5@unibo.it


## :pray: Acknowledgements


## Download Dataset \& Pretrained Model
Download [here](https://drive.google.com/drive/folders/1o_I4Z-9xRT7PqBgXQQgVUlcDOwOTT9Qj?usp=drive_link) .

Unzip and place dataset under the `RoofDiffusion/dataset` of repo e.g. `RoofDiffusion/dataset/PoznanRD`

Place RoofDiffusion pretrained model at `RoofDiffusion/pretrained/w_footprint/260_Network.pth`

Or place No-NF RoofDiffusion pretrained model at `RoofDiffusion/pretrained/wo_footprint/140_Network.pth`

> The height maps are in uint16 format, where the actual roof height (meter) = pixel value / 256. (same as [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php))

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

## Context Encoders: Feature Learning by Inpainting

![teaser](images/teaser.jpg "Sample inpainting results on held-out images")




2. [Train Context Encoders](#2-train-context-encoders)
3. [Download Features Caffemodel](#3-download-features-caffemodel)
4. [TensorFlow Implementation](#4-tensorflow-implementation)
5. [Project Website](#5-project-website)
6. [Download Dataset](#6-paris-street-view-dataset)




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

3. Test the model
  ```Shell
  # For training center region inpainting model, run:
  DATA_ROOT=dataset/val net=checkpoints/inpaintCenter_500_net_G.t7 name=test_patch overlapPred=4 manualSeed=222 batchSize=30 loadSize=350 gpu=1 th test.lua
  DATA_ROOT=dataset/val net=checkpoints/inpaintCenter_500_net_G.t7 name=test_full overlapPred=4 manualSeed=222 batchSize=30 loadSize=129 gpu=1 th test.lua

  # For testing random region inpainting model, run (with fineSize=64 or 124, same as training):
  DATA_ROOT=dataset/val net=checkpoints/inpaintRandomNoOverlap_500_net_G.t7 name=test_patch_random useOverlapPred=0 manualSeed=222 batchSize=30 loadSize=350 gpu=1 th test_random.lua
  DATA_ROOT=dataset/val net=checkpoints/inpaintRandomNoOverlap_500_net_G.t7 name=test_full_random useOverlapPred=0 manualSeed=222 batchSize=30 loadSize=129 gpu=1 th test_random.lua
  ```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
We used NYU Depth Dataset V2 as our dataset. We used Labeled dataset (~2.8 GB) of NYU Depth Dataset which provides 1449 densely labeled pairs of aligned RGB and depth images. We divided labeled dataset into three parts (Training - 1024, Validation - 224, Testing - 201) for our project. NYU Dataset also provides Raw dataset (~428 GB) on which we couldn't train due to machine capacity.
We used [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) as our dataset. We used Labeled dataset (~2.8 GB) of NYU Depth Dataset which provides 1449 densely labeled pairs of aligned RGB and depth images. We divided labeled dataset into three parts (Training - 1024, Validation - 224, Testing - 201) for our project. NYU Dataset also provides Raw dataset (~428 GB) on which we couldn't train due to machine capacity.

### 3) 

### 4) TensorFlow Implementation

Checkout the TensorFlow implementation of our paper by Taeksoo [here](https://github.com/jazzsaxmafia/Inpainting). However, it does not implement full functionalities of our paper.

### 5) Project Website

Click [here](https://www.cs.cmu.edu/~dpathak/context_encoder/).

### 6) Paris Street-View Dataset

Please email me if you need the dataset and I will share a private link with you. I can't post the public link to this dataset due to the policy restrictions from Google Street View.



mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm end

gpt startttttttttttttttttttttttttttttttttttttttttttttt
# Instructions to Run the Test Code

This document provides step-by-step instructions to prepare the NYU depth data, set up the environment, and execute the provided Lua test script.

---

## 1. **Environment Setup**

### Prerequisites
Ensure the following dependencies are installed:

1. **Torch**:
   - Install Torch by following the instructions at [Torch GitHub](https://github.com/torch/distro).
2. **Required Torch Packages**:
   ```bash
   luarocks install nn
   luarocks install optim
   luarocks install image
   luarocks install matio
   luarocks install paths
   luarocks install cunn
   luarocks install cudnn  # Optional, for GPU acceleration
   ```
3. **CUDA (Optional)**:
   - For GPU support, ensure CUDA is installed and configured correctly.

---

## 2. **Prepare NYU Data**

1. **Dataset Requirements**:
   - The script expects input files in the `.mat` format.
   - The `.mat` files should contain two channels:
     - **Channel 1**: Depth data (raw input).
     - **Channel 2**: Mask.

2. **Input Directory**:
   - Place all `.mat` files in the directory:
     ```
     /home/rrs/ONS_Mou/Fromeeserver_ubuntu/NYU_singleimage/
     ```
   - Ensure the directory exists and contains valid `.mat` files.

---

## 3. **Running the Code**

### Steps:

1. Open the Lua script (`test_nyu_model.lua`) in your preferred text editor or IDE.
2. Ensure the following paths in the script are correct:
   - Pretrained model path:
     ```lua
     /home/rrs/ONS_Mou/Fromeeserver_ubuntu/jbgjh/old/
     ```
   - Input directory:
     ```lua
     /home/rrs/ONS_Mou/Fromeeserver_ubuntu/NYU_singleimage/
     ```
   - Output directory:
     ```lua
     /home/rrs/ONS_Mou/Fromeeserver_ubuntu/NYU_results_forcroppedinput/
     ```

3. Execute the script from the command line:
   ```bash
   th test_nyu_model.lua
   ```

---

## 4. **Output**

### Results:

1. **Predicted Outputs**:
   - Predicted depth images will be saved in `.mat` format in:
     ```
     /home/rrs/ONS_Mou/Fromeeserver_ubuntu/NYU_results_forcroppedinput/
     ```
   - File naming convention: `overlaid_<input_filename>.mat`.

2. **Visual Outputs**:
   - Predicted depth maps (`.png`) will also be saved in the same directory with names like:
     - `<input_filename>.png`: Raw predictions.
     - `overlaid_<input_filename>.png`: Overlay of predictions on the original image.

---

## 5. **Notes**

- Make sure the `.mat` files conform to the expected structure to avoid errors.
- GPU usage is enabled by default. To disable GPU, set the `gpu` option to `0` in the script:
  ```lua
  opt = {
      nc = 1,       -- Number of input channels
      gpu = 0,      -- Disable GPU by setting to 0
  }
  ```
- If using GPU, ensure that your CUDA version matches the `cudnn` library.

---

Feel free to reach out if you encounter any issues!

For 90-% missing:
Below is the updated code with **clear comments** and proper structure to ensure it is easy to understand for the public. Each section is explained with appropriate comments.

```lua
-- Required libraries
require 'image'
require 'nn'
util = paths.dofile('util.lua')  -- Utility functions for loading models and other operations
torch.setdefaulttensortype('torch.FloatTensor')  -- Set the default tensor type
local matio = require 'matio'  -- Library for loading/saving .mat files

-- Options for the script
opt = {
    net = './chk/DC_chk_90.t7',    -- Path to the pre-trained model
    gpu = 1,                      -- GPU mode: 0 = CPU, 1 = GPU
    nc = 1,                       -- Number of channels in the input images
}

-- Overwrite options with environment variables if they exist
for k, v in pairs(opt) do 
    opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
end
print(opt)

-- Load the pre-trained model
net = util.load(opt.net, opt.gpu)
net:evaluate() -- Set the network to evaluation mode

-- Directory containing corrupted depth maps (input images)
local input_dir = '/path-to-corrupted depth maps'
-- Directory to save the results
local output_dir = '/Results'
-- Mask file path
local mask_path = 'MASK90.png'

-- Collect all files with '.png' extension in the input directory
files = {}
for file in paths.files(input_dir) do
    if file:find('png' .. '$') then
        table.insert(files, input_dir .. '/' .. file)
    end
end

-- Check if any valid files were found
if #files == 0 then
    error('The given directory does not contain any files of type: png')
end

-- Load all images from the file list
images = {}
for i, file in ipairs(files) do
    -- Load each image as a float tensor
    table.insert(images, image.load(file, opt.nc, 'float'))
end

-- Load the binary mask (assumes mask is the same for all images)
local mask = image.load(mask_path)
mask = mask:byte() -- Convert the mask to binary (byte tensor)

-- Process each image
for ldimg = 1, #images do  
    local image_ctx = images[ldimg]  -- Get the current image
    print(ldimg .. '   :' .. paths.basename(files[ldimg]))  -- Print progress

    local imgheight = image_ctx:size(2)
    local imgwidth = image_ctx:size(3)

    -- Create a tensor for the input image
    local input_image_ctx = torch.Tensor(opt.nc, imgheight, imgwidth)

    -- Move the model and input to GPU if GPU mode is enabled
    if opt.gpu > 0 then  
        require 'cunn'
        if pcall(require, 'cudnn') then  
            require 'cudnn'
            net = util.cudnn(net)  -- Use cuDNN for better performance
        end
        net:cuda()
        input_image_ctx = input_image_ctx:cuda()
    else
        net:float()  -- Use CPU
    end

    -- Clone the image as the input
    input_image_ctx = image_ctx:clone()

    -- Apply the mask: set masked pixels to -0.5
    input_image_ctx[{{1}, {}, {}}][mask] = -0.5  

    -- Perform prediction
    local pred_dep
    if opt.noiseGen then
        pred_dep = net:forward({input_image_ctx, noise})
    else
        input_image_ctx = input_image_ctx:cuda()  -- Ensure input is on GPU
        pred_dep = net:forward(input_image_ctx)
    end

    -- Resize the predicted output to match the dimensions of the input image
    pred_dep = pred_dep:resize(pred_dep:size(2), pred_dep:size(3), pred_dep:size(4))

    -- Combine the predicted depth with the original image for visualization
    local pred_depth = image_ctx:clone()
    pred_depth[{{1}, {}, {}}][mask] = pred_dep[{{1}, {}, {}}][mask]:float()

    -- Save results
    local basename = paths.basename(files[ldimg])
    matio.save(output_dir .. '/' .. basename:sub(1, #basename - 4) .. '.mat', {
        GT = image_ctx,             -- Ground truth image
        OutputN = pred_dep:float(), -- Predicted depth map (only masked area)
        OutputO = pred_depth:float(), -- Predicted depth map (overlayed on original)
        Maskedimage = input_image_ctx:float() -- Input image with mask applied
    })

    -- Clear variables to free memory
    pred_dep = nil
    pred_depth = nil
    input_image_ctx = nil
    image_ctx = nil
end

print('Processing completed successfully!')
```

---

### **Improvements and Features in the Code**
1. **Comments**: Detailed comments explain each part of the code.
2. **Error Handling**: Checks if the directory contains any valid `.png` files and throws an error otherwise.
3. **Mask Loading**: Added a specific explanation of how the binary mask is loaded and used.
4. **Memory Management**: Variables are cleared at the end of each iteration to free up memory.
5. **Output Structure**: Clearly saves the results into `.mat` files in the `/Results` directory.

---

### **Important Instructions for Public Usage**
You can create a `README.md` file with the following information:

1. **Dependencies**:
   - Torch
   - Libraries: `image`, `nn`, `cunn`, `cudnn`, `matio`, and `util.lua`.

2. **Input Data**:
   - Corrupted depth maps should be stored in a folder, and their path should replace `/path-to-corrupted depth maps`.
   - The binary mask file (`MASK90.png`) should be placed in the working directory.

3. **How to Run**:
   - Edit the `opt` table to include your model path and GPU/CPU configuration.
   - Execute the script using:
     ```bash
     th script_name.lua
     ```

4. **Output**:
   - Results (including predictions and overlaid depth maps) are saved as `.mat` files in the `/Results` directory.

Let me know if you'd like help creating the accompanying `README.md` file!
gpt endddddddddddddddddddddddddddddddddddddddddddddd
local directory ='/home/cvlab/data/test/NYU/' .......for loading real test images
