
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
## Usage
### Prerequisites
1. Install `torch`: http://torch.ch/docs/getting-started.html
..............MATIO
```shell
luarocks install cv
```
2. Clone the repository
  ```Shell
  git clone https://github.com/Moushumi9medhi/Depth-Completion.git
  cd Depth-Completion
  ```
3. Run the following single command in your terminal to download the pretrained model:

```bash
bash -c "$(wget -qO- https://your-script-url.com/download_model.sh)"
# It will download the pre-trained model into the `models` directory.
```
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

2. Train the unguided indoor depth completion model in order to obtain a discriminator network and a generator network. I have already trained the model on the [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).... and put the corresponding networks into the `checkpoints` folder. If you want to train it again or use a different dataset run
```
DATA_ROOT=<dataset_folder> name=<whatever_name_you_want> th main.lua
```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Train the model
  ```Shell
  # Train your own pixel interpolation model
  cd pixelInterpolation
  DATA_ROOT=../dataset/my_train_set name=pixel niter=250 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
  ```
  xxxxxxxxxxxxxxxxxxxxxxxxxxx
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
### Optional parameters

In your command line instructions you can specify several parameters (for example the display port number), here are some of them:
+ `noise` which can be either `uniform` or `normal` indicates the prior distribution from which the samples are generated
+ `batchSize` is the size of the batch used for training or the number of images to reconstruct
+ `name` is the name you want to use to save your networks or the generated images
+ `gpu` specifies if the computations are done on the GPU or not. Set it to 0 to use the CPU (not recommended, see below) and to n to use the nth GPU you have (1 is the default value)
+ `lr` is the learning rate
+ `loadSize` is the size to use to scale the images. 0 means no rescale

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
## 📜 License
This project is licensed under the The MIT License (MIT).

---

**For any queries, feel free to raise an issue or contact us directly via [email](mailto:medhi.moushumi@iitkgp.ac.in).**
222222222222222222222222222222 start
# [Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/)

[Satoshi Iizuka](http://iizuka.cs.tsukuba.ac.jp/index_eng.html), [Edgar Simo-Serra](https://esslab.jp/~ess/), [Hiroshi Ishikawa](http://www.f.waseda.jp/hfs/indexE.html)

![Teaser Image](teaser.png)

## News
**09/17/2020 Update:** We have released the following two models:
- `completionnet_places2_freeform.t7`: An image completion model trained with free-form holes on the [Places2 dataset](http://places2.csail.mit.edu/), which will work better than the model trained with rectangular holes, even without post-processing. We used a part of the [context encoder [Pathak et al. 2016]](https://github.com/pathak22/context-encoder) implementation to generate the random free-form holes for training.
- `completionnet_celeba.t7`: A face completion model trained with rectangular holes on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). This model was trained on face images with the smallest edges in the [160, 178], and thus it will work best on images of similar sizes.

These models can be downloaded via `download_model.sh`.

## Overview

This code provides an implementation of the research paper:

```
  "Globally and Locally Consistent Image Completion"
  Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa
  ACM Transaction on Graphics (Proc. of SIGGRAPH 2017), 2017
```
We learn to inpaint missing regions with a deep convolutional network.
Our network completes images of arbitrary resolutions by filling in
missing regions of any shape. We use global and local context discriminators
to train the completion network to provide both locally and globally consistent results.
See our [project page](http://iizuka.cs.tsukuba.ac.jp/projects/completion/) for more detailed information.

## License

```
  Copyright (C) <2017> <Satoshi Iizuka, Edgar Simo-Serra, Hiroshi Ishikawa>

  This work is licensed under the Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy
  of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or
  send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

  Satoshi Iizuka, University of Tsukuba
  iizuka@cs.tsukuba.ac.jp, http://iizuka.cs.tsukuba.ac.jp/index_eng.html
  
  Edgar Simo-Serra, Waseda University
  ess@waseda.jp, https://esslab.jp/~ess/
```


## Dependencies

- [Torch7](http://torch.ch/docs/getting-started.html)
- [nn](https://github.com/torch/nn)
- [image](https://github.com/torch/image)
- [nngraph](https://github.com/torch/nngraph)
- [torch-opencv](https://github.com/VisionLabs/torch-opencv) (optional, required for post-processing)

The packages of `nn`, `image`, and `nngraph` should be a part of a standard Torch7 install.
For information on how to install Torch7 please see the [official torch documentation](http://torch.ch/docs/getting-started.html)
on the subject. The `torch-opencv` is OpenCV bindings for LuaJit+Torch, which can be installed via 
`luarocks install cv` after installing OpenCV 3.1. Please see the [instruction page](https://github.com/VisionLabs/torch-opencv/wiki/Installation) for more detailed information.

**17/09/2020 Note:** If you fail to install Torch7 with current GPUs, please try the [self-contained Torch installation repository (unofficial)]( https://github.com/nagadomi/distro) by [@nadadomi](https://github.com/nagadomi), which supports CUDA10.1, Volta, and Turing.

## Usage

First, download the models by running the download script:

```
bash download_model.sh
```

Basic usage is:

```
th inpaint.lua --input <input_image> --mask <mask_image>
```
The mask is a binary image (1 for pixels to be completed, 0 otherwise) and should be the same size as the input image. If the mask is not specified, a mask with randomly generated holes will be used.

Other options:

- `--model`: Model to be used. Defaults to 'completionnet_places2_freeform.t7'.
- `--gpu`: Use GPU for the computation. [cunn](https://github.com/torch/cunn) is required to use this option. Defaults to false.
- `--maxdim`: Long edge dimension of the input image. Defaults to 600.
- `--postproc`: Perform the post-processing. Defaults to false. If you fail to install the `torch-opencv`, do not use this option to avoid using the package.

For example:

```
th inpaint.lua --input example.png --mask example_mask.png
```

### Best Performance

- The Places models were trained on the [Places2 dataset](http://places2.csail.mit.edu/) and thus best performance is for natural outdoor images.
- While the Places2 models work on images of any size with arbitrary holes, we trained them on images with the smallest edges in the [256, 384] pixel range and random holes in the [96, 128] pixel range. Our models will work best on images with holes of those sizes.
- Significantly large holes or extrapolation when the holes are at the border of images may fail to be filled in due to limited spatial support of the model.

### Notes

- This is developed on a Linux machine running Ubuntu 16.04 during late 2016.
- Provided model and sample code is under a non-commercial creative commons license.

## Citing

If you use this code please cite:

```
@Article{IizukaSIGGRAPH2017,
  author = {Satoshi Iizuka and Edgar Simo-Serra and Hiroshi Ishikawa},
  title = {{Globally and Locally Consistent Image Completion}},
  journal = "ACM Transactions on Graphics (Proc. of SIGGRAPH)",
  year = 2017,
  volume = 36,
  number = 4,
  pages = 107:1--107:14,
  articleno = 107,
}
```




2222222222222222222222222222 end
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
