
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
This repository hosts the source code for unguided single-depth map completion of indoor scenes—a lightweight design for restoring noisy depth maps in a realistic manner using a generative adversarial network (GAN)

### Key Features

- Handles different types of depth map degradations:
  - Simulated random and textual missing pixels
  - Holes found in Kinect depth maps
- With only a maximum of 1.8 million parameters, our model is edge-friendly and compact.
- Generalization capability across various indoor depth datasets without additional fine-tuning.
- Adaptable to existing works, supporting diverse computer vision applications.

## BibTeX Citation

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


## Code
Code will be uploaded soon. For any pressing inquiries, contact: Moushumi Medhi @ [medhi.moushumi@iitkgp.ac.in](mailto:medhi.moushumi@iitkgp.ac.in)


## Features
- **Depth Completion**

## Directory Structure

- `server`: contains server-side code and relevant information for reproducing the server-side experimental results.
- `client`: contains an Unity3D-based application. This application was developed using LitAR client/server APIs, and can be used for reproducing the remaining experimental results.



## Acknowledgement

We thank the anonymous reviewers for their constructive reviews. 
## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/Moushumi9medhi/Depth-Completion.git
