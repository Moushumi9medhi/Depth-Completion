
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
This repository hosts the source code for unguided single-depth map completion of indoor scenes—a lightweight design for restoring noisy depth maps in a realistic manner using a generative adversarial network (GAN).
## 📖 Abstract
Single depth map completion in the absence of any guidance from color images is a challenging,
ill-posed problem in computer vision. Most of the conventional depth map completion approaches
rely on information extracted from the corresponding color image and require heavy computations
and optimization-based postprocessing functions, which cannot yield results in real time. Successful
application of generative adversarial networks has led to significant progress in several computer vision
problems including, color image inpainting. However, contrasting local and non-local features of depth
maps compared to color images prevents the direct application of deep learning models designed for
color image inpainting to depth map completion. In this work we
propose to use deep adversarial learning to derive plausible estimates of missing depth information
in a single degraded observation without any guidance from the corresponding RGB frame and any
postprocessing. 

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

1. __A dataset__: multi-sensor data streams captured by AR devices and laser scanners
2. __scantools__: a processing pipeline to register different user sessions together
3. __A benchmark__: a framework to evaluate algorithms for localization and mapping

See our [ECCV 2022 tutorial](https://lamar.ethz.ch/tutorial-eccv2022/) for an overview of LaMAR and of the state of the art of localization and mapping for AR.

## Overview

This codebase is composed of the following modules:

- <a href="#benchmark">`lamar`</a>: evaluation pipeline and baselines for localization and mapping
- <a href="#processing-pipeline">`scantools`</a>: data API, processing tools and pipeline
- [ScanCapture](apps/ScanCapture_iOS): a data recording app for Apple devices

## Data format

We introduce a new data format, called *Capture*, to handle multi-session and multi-sensor data recorded by different devices. A Capture object corresponds to a capture location. It is composed of multiple sessions and each of them corresponds to a data recording by a given device. Each sessions stores the raw sensor data, calibration, poses, and all assets generated during the processing.



## Directory Structure

- `server`: contains server-side code and relevant information for reproducing the server-side experimental results.
- `client`: contains an Unity3D-based application. This application was developed using LitAR client/server APIs, and can be used for reproducing the remaining experimental results.



## 🙏 Acknowledgement

We thank the anonymous reviewers for their constructive reviews. 
## Installation and Running
### Prerequisites
1. Install `torch`: http://torch.ch/docs/getting-started.html
..............MATIO
```shell
luarocks install cv
```
**Dependencies**

- [Torch7](http://torch.ch/docs/getting-started.html)
- [nn](https://github.com/torch/nn)
- [image](https://github.com/torch/image)
  
The packages of `nn`, `image`, and `nngraph` should be a part of a standard Torch7 install.
2. Clone the repository
  ```Shell
  git clone https://github.com/Moushumi9medhi/Depth-Completion.git
  cd Depth-Completion
  ```
3. Run the following single command in your terminal to download the pretrained model:

```bash
bash download_model.sh
# It will download the pre-trained model into the `models` directory.
```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
 We have released the following two models:
- `completionnet_places2_freeform.t7`: An image completion model trained with free-form holes on the [Places2 dataset](http://places2.csail.mit.edu/), which will work better than the model trained with rectangular holes, even without post-processing. We used a part of the [context encoder [Pathak et al. 2016]](https://github.com/pathak22/context-encoder) implementation to generate the random free-form holes for training.
- `completionnet_celeba.t7`: A face completion model trained with rectangular holes on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). This model was trained on face images with the smallest edges in the [160, 178], and thus it will work best on images of similar sizes.

These models can be downloaded via `download_model.sh`.

### Training
1. Choose a RGB-Depth dataset and create a folder with its name (ex: `mkdir celebA`). Inside this folder create a folder `images` containing your images.  .....swee how training  image folder was created...............check this........not right.....

 Download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [SUN397](http://vision.cs.princeton.edu/projects/2010/SUN/) dataset, or prepare your own dataset. 
  ```Shell
  # put all training images inside my_train_set/images/
  mkdir -p /your_path/my_train_set/images/
  # put all validation images inside my_val_set/images/
  mkdir -p /your_path/my_val_set/images/
  # put all testing images inside my_test_set/images/
  mkdir -p /your_path/my_test_test/images/
  cd on-demand-learning/
  ln -sf /your_path dataset
  ```


*Note:* for the `celebA` dataset, run
```
DATA_ROOT=celebA th data/crop_celebA.lua
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
  xxxxxxxxxxxxxxxxxxxxxxxxxxx
  
First, download the models by running the download script:

```
bash download_model.sh
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
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```Shell
  DATA_ROOT=dataset/train display_id=11 name=inpaintCenter overlapPred=4 wtl2=0.999 nBottleneck=4000 niter=500 loadSize=350 fineSize=128 gpu=1 th train.lua
  ```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  net=models/inpaintCenter/paris_inpaintCenter.t7 name=paris_result imDir=images/paris overlapPred=4 manualSeed=222 batchSize=21 gpu=1 th demo.lua
  net=models/inpaintCenter/imagenet_inpaintCenter.t7 name=imagenet_result imDir=images/imagenet overlapPred=0 manualSeed=222 batchSize=21 gpu=1 th demo.lua
  net=models/inpaintCenter/paris_inpaintCenter.t7 name=ucberkeley_result imDir=images/ucberkeley overlapPred=4 manualSeed=222 batchSize=4 gpu=1 th demo.lua
### Inference
1. Complete your images. You may want to choose another dataset to avoid completing images you used for training.
```
DATA_ROOT=<dataset_folder> name=<whatever_name_you_want> net=<prefix_of_net_in_checkpoints> th inpainting.lua
```
*Example:*
```
DATA_ROOT=celebA noise=normal net=celebA-normal name=inpainting-celebA display=2929 th inpainting.lua
```
### Display images in a browser

If you want, install the `display` package (`luarocks install display`) and run
```
th -ldisplay.start <PORT_NUMBER> 0.0.0.0
```
to launch a server on the port you chose. You can access it in your browser with the url http://localhost:PORT_NUMBER.

To train your network or for completion add the variable `display=<PORT_NUMBER>` to the list of options.

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
 [Optional] Install the Display Package, which enables you to track the training progress. If you don't want to install it, please set `display=0` in `train.lua`.
  ```Shell
  luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
  #start the display server
  th -ldisplay.start 8000
  # on client side, open in browser: http://localhost:8000/
  # You can then see the training progress in your browser window.
  ```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Install Display Package as follows. If you don't want to install it, then set `display=0` in `train.lua`.
  ```Shell
  luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
  cd ~
  th -ldisplay.start 8000
  # if working on server machine create tunnel: ssh -f -L 8000:localhost:8000 -N server_address.com
  # on client side, open in browser: http://localhost:8000/
  ```
### Optional parameters

In your command line instructions you can specify several parameters (for example the display port number), here are some of them:
+ `noise` which can be either `uniform` or `normal` indicates the prior distribution from which the samples are generated
+ `batchSize` is the size of the batch used for training or the number of images to reconstruct
+ `name` is the name you want to use to save your networks or the generated images
+ `gpu` specifies if the computations are done on the GPU or not. Set it to 0 to use the CPU (not recommended, see below) and to n to use the nth GPU you have (1 is the default value)
+ `lr` is the learning rate
+ `loadSize` is the size to use to scale the images. 0 means no rescale
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Other options:

- `--model`: Model to be used. Defaults to 'completionnet_places2_freeform.t7'.
- `--gpu`: Use GPU for the computation. [cunn](https://github.com/torch/cunn) is required to use this option. Defaults to false.
- `--maxdim`: Long edge dimension of the input image. Defaults to 600.
- `--postproc`: Perform the post-processing. Defaults to false. If you fail to install the `torch-opencv`, do not use this option to avoid using the package.

For example:

### Demo
If you want to run a quick demo for the four image restoration tasks, please download our pre-trained models using the following script.
  ```Shell
  cd models
  #download image inpainting model
  bash download_model.sh inpainting
  #download pixel interpolation model
  bash download_model.sh pixelInterpolation
  #download image deblurring model
  bash download_model.sh deblurring
  #download image denoising model
  bash download_model.sh denoising
  #download our image denoising model equipped to denoise images of any size
  bash download_model.sh denoise_anysize
  ```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Demo
  ```Shell
  # Test the image inpainting model on various corruption levels
  cd inpainting
  DATA_ROOT=../dataset/my_test_set name=inpaint_demo net=../models/inpainting_net_G.t7 manualSeed=333 gpu=1 display=1 th demo.lua
  # Demo results saved as inpaint_demo.png
  ```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  Demo
  ```Shell
  # Test the image deblurring model on various corruption levels
  cd deblurring
  DATA_ROOT=../dataset/my_test_set name=deblur_demo net=../models/deblurring_net_G.t7 manualSeed=333 gpu=1 display=1 th demo.lua
  # Demo results saved as deblur_demo.png
  ```
xxxxxxxxxxxxxxxxxxxxxxx
 Demo
  ```Shell
  # Test the image denoising model on various corruption levels
  cd deblurring
  DATA_ROOT=../dataset/my_test_set name=denoise_demo net=../models/denoising_net_G.t7 sigma=25 manualSeed=333 gpu=1 display=1 th demo.lua
  # Demo results saved as denoise_demo.png
  ```
xxxxxxxxxxxxxxxxxxxxxxxx
Image Denoising of Arbitrary Sizes
Denoising/DB11 contains 11 classic images commonly used to evaluate image denoising algorithms. Because the input of our network is of size 64 x 64, given an image of arbitrary size (assuming larger than 64 x 64), we use a sliding-window approach to denoise each patch separately, then average outputs at overlapping pixels.
  ```Shell
  # Denoise classic image Lena from DB11 dataset
  cd denoise_anysize
  img_path=DB11/Lena.png name=denoise net=../models/denoise_anysize_net_G.t7 sigma=25 stepSize=3 gpu=1 th denoise.lua
  # Denoising results saved as denoise.png
  ```

xxxxxxxxxxxxxxxxxxxxxxxx

```
th inpaint.lua --input example.png --mask example_mask.png
```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```Shell
  # you can either use demo.lua to display the result or use test.lua using following commands:
  DATA_ROOT=dataset/val net=checkpoints/inpaintCenter_500_net_G.t7 name=test_patch overlapPred=4 manualSeed=222 batchSize=30 loadSize=350 gpu=1 th test.lua
  DATA_ROOT=dataset/val net=checkpoints/inpaintCenter_500_net_G.t7 name=test_full overlapPred=4 manualSeed=222 batchSize=30 loadSize=129 gpu=1 th test.lua
  ```

## Benchmark

We evaluate the performance of multiple state-of-the-art and popular stereo matching methods, both for standard and 360° images. All models are trained on a single NVIDIA A100 GPU with
the largest possible batch size to ensure comparable use of computational resources.

| Method             | Type           | Disp-MAE (°) | Disp-RMSE (°) | Disp-MARE | Depth-MAE (m) | Depth-RMSE (m) | Depth-MARE |
|--------------------|----------------|--------------|---------------|-----------|---------------|----------------|----------------|
| [PSMNet](https://arxiv.org/abs/1803.08669)           | Stereo        | 0.33         | 0.54          | 0.20      | 2.79          | 6.17           | 0.29           |
| [360SD-Net](https://arxiv.org/abs/1911.04460)        | 360° Stereo   | 0.21         | 0.42          | 0.18      | 2.14          | 5.12           | 0.15           |
| [IGEV-Stereo](https://arxiv.org/abs/2303.06615)      | Stereo        | 0.22         | 0.41          | 0.17      | 1.85          | 4.44           | 0.15           |
| [360-IGEV-Stereo](https://arxiv.org/abs/2411.18335)    | 360° Stereo   | **0.18**     | **0.39**      | **0.15**  | **1.77**      | **4.36**       | **0.14**       |


Download and save the `pretrained model` to `./checkpoints`

| Pretrained Model                                                                                    | Blocks    | Channels | Drop rate |
| --------------------------------------------------------------------------------------------------- |:-------:|:--------:|:-------:|
| [SPNet-Tiny](https://drive.google.com/file/d/1ivmCX-i9lej4uJhT0Yyk2Nq9ZmoQlsB9/view?usp=drive_link)    | [3,3,9,3]  | [96,192,384,768]    | 0.0  |
| [SPNet-Small](https://drive.google.com/file/d/1Ba-W3oX62lCjx5MvvGkn91LXP6SuCnV6/view?usp=drive_link)   | [3,3,27,3] | [96,192,384,768]    | 0.1  | 
| [SPNet-Base](https://drive.google.com/file/d/1B9uPRVPGm1F8F-isVDVzEdHgxXmp43hn/view?usp=drive_link)    | [3,3,27,3] | [128,256,512,1024]  | 0.1  | 
| [SPNet-Large](https://drive.google.com/file/d/11dujPviL4pKLEXytXK0mEmPBNQDqgEak/view?usp=drive_link)   | [3,3,27,3] | [192,384,768,1536]  | 0.2  | 

Run `test.py`

```python
# SPNet-Tiny
python test.py --dims=[3,3,9,3] --depths=[96,192,384,768] --dp_rate=0.0 --model_dir='checkpoints/Tiny.pth'
# SPNet-Small
python test.py --dims=[3,3,27,3] --depths=[96,192,384,768] --dp_rate=0.1 --model_dir='checkpoints/Small.pth'
# SPNet-Base
python test.py --dims=[3,3,27,3] --depths=[128,256,512,1024] --dp_rate=0.1 --model_dir='checkpoints/Base.pth'
# SPNet-Large
python test.py --dims=[3,3,27,3] --depths=[192,384,768,1536] --dp_rate=0.2 --model_dir='checkpoints/Large.pth'
```

## 📜 License
This project is licensed under the The MIT License (MIT).

---

**For any queries, feel free to raise an issue or contact us directly via [email](mailto:medhi.moushumi@iitkgp.ac.in).**
2222222222222222222222 start

### Results
Quanlitative Evaluation On NYU online test dataset

<img src="./results/NYU.png" width = "536" height = "300" alt="NYU" />

Quantitative Evaluation On NYU online test dataset

<img src="./results/NYU_results.jpg" width = "556" height = "336" alt="NYU_table" />

Quanlitative Evaluation On KITTI online test dataset

<img src="./results/KITTI.png" width = "930" height = "530" alt="KITTI" />

Quantitative Evaluation On KITTI online test dataset

<img src="./results/KITTI_results.jpg" width = "600" height = "400" alt="KITTI_table" />


### Enviroment Config
- pytorch=1.11 CUDA=11.6 python=3.9
- pip install einops tqdm matplotlib numpy opencv-python pandas scikit-image scikit-learn h5py
#### NVIDIA Apex

We used NVIDIA Apex for multi-GPU training.

Apex can be installed as follows:

```bash
$ cd PATH_TO_INSTALL
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir ./ 
```

#### Dataset

We used NYU Depth V2 (indoor) and KITTI Depth Completion datasets for training and evaluation.

#### NYU Depth V2 

download NYU Depth Dataset 
```bash
$ cd PATH_TO_DOWNLOAD
$ wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
$ tar -xvf nyudepthv2.tar.gz
```

#### KITTI Depth Completion (KITTI DC)

KITTI DC dataset get from the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).

For KITTI Raw dataset is get from the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php).

```bash
$ cd NLSPN_ROOT/utils
$ python prepare_KITTI_DC.py --path_root_dc PATH_TO_KITTI_DC --path_root_raw PATH_TO_KITTI_RAW
```
To get the train and test data, and the data structure as follows:

```
.
├── depth_selection
│    ├── test_depth_completion_anonymous
│    │    ├── image
│    │    ├── intrinsics
│    │    └── velodyne_raw
│    ├── test_depth_prediction_anonymous
│    │    ├── image
│    │    └── intrinsics
│    └── val_selection_cropped
│        ├── groundtruth_depth
│        ├── image
│        ├── intrinsics
│        └── velodyne_raw
├── train
│    ├── 2011_09_26_drive_0001_sync
│    │    ├── image_02
│    │    │     └── data
│    │    ├── image_03
│    │    │     └── data
│    │    ├── oxts
│    │    │     └── data
│    │    └── proj_depth
│    │        ├── groundtruth
│    │        └── velodyne_raw
│    └── ...
└── val
    ├── 2011_09_26_drive_0002_sync
    └── ...
```

#### Training

```bash
$ cd SDformer/src

# for NYU Depth v2 dataset training
$ python main.py --gpus 0,1  --epochs 25 --batch_size 8 --save NYU

# for KITTI Depth Completion dataset training
$ python main.py --gpus 0,1 --epochs 20 --batch_size 4 --test_crop --save KITTI
```

#### Testing

```bash
$ cd SDformer/src

$ python main.py --test_only --pretrain pretrain-path --save test_on_NYU

$ python main.py --test_only --pretrain pretrain-path --save test_on_KITTI
```

KITTI DC Online evaluation data:

```bash
$ cd SDformer/src
$ python main.py --split_json ../data_json/kitti_dc_test.json --test_only --pretrain pretrain --save_image --save_result_only --save online_kitti
```

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
Download the pretrained base models for [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [MSE-finetuned VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse).

Download our MagicAnimate [checkpoints](https://huggingface.co/zcxu-eric/MagicAnimate).



## ⚒️ Installation


## 💃 Inference


## 🎨 Gradio Demo 

## Context Encoders: Feature Learning by Inpainting

![teaser](images/teaser.jpg "Sample inpainting results on held-out images")



### Contents
1. [Semantic Inpainting Demo](#1-semantic-inpainting-demo)
2. [Train Context Encoders](#2-train-context-encoders)
3. [Download Features Caffemodel](#3-download-features-caffemodel)
4. [TensorFlow Implementation](#4-tensorflow-implementation)
5. [Project Website](#5-project-website)
6. [Download Dataset](#6-paris-street-view-dataset)

### 1) Semantic Inpainting Demo




  net=models/inpaintCenter/paris_inpaintCenter.t7 name=paris_result imDir=images/paris overlapPred=4 manualSeed=222 batchSize=21 gpu=1 th demo.lua
  net=models/inpaintCenter/imagenet_inpaintCenter.t7 name=imagenet_result imDir=images/imagenet overlapPred=4 manualSeed=222 batchSize=21 gpu=1 th demo.lua
  net=models/inpaintCenter/paris_inpaintCenter.t7 name=ucberkeley_result imDir=images/ucberkeley overlapPred=4 manualSeed=222 batchSize=4 gpu=1 th demo.lua
  # Note: If you are running on cpu, use gpu=0
  # Note: samples given in ./images/* are held-out images
  ```

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

### 3) Download Features Caffemodel

Features for context encoder trained with reconstruction loss.

- [Prototxt](https://www.cs.cmu.edu/~dpathak/context_encoder/resources/ce_features.prototxt)
- [Caffemodel](https://www.cs.cmu.edu/~dpathak/context_encoder/resources/ce_features.caffemodel)

### 4) TensorFlow Implementation

Checkout the TensorFlow implementation of our paper by Taeksoo [here](https://github.com/jazzsaxmafia/Inpainting). However, it does not implement full functionalities of our paper.

### 5) Project Website

Click [here](https://www.cs.cmu.edu/~dpathak/context_encoder/).

### 6) Paris Street-View Dataset

Please email me if you need the dataset and I will share a private link with you. I can't post the public link to this dataset due to the policy restrictions from Google Street View.



mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm end
