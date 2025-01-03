
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
1. Clone the repository:
   ```sh
   git clone https://github.com/Moushumi9medhi/Depth-Completion.git
## 📜 License
This project is licensed under the The MIT License (MIT).

---

**For any queries, feel free to raise an issue or contact us directly via [email](mailto:medhi.moushumi@iitkgp.ac.in).**
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




mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm end
