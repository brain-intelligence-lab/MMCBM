#  MMCBM 
[![arXiv](https://img.shields.io/badge/arXiv-MMCBM-b31b1b.svg)](https://arxiv.org/abs/2403.05606)


## Introduction
This is the official repository for **MMCBM: Interpretable Diagnosis of Choroid Neoplasias via the Multimodal Concept
Bottleneck Model.**

![model](images/Fig1_v2.png)

## Installation

1. Create a conda environment from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. Install from the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

## Preparation

1. download
   the [model checkpoint](https://drive.google.com/drive/folders/1YwDhqC_M9ACBnGjn_8IZouWHgJx1ue5Q?usp=drive_link) place
   it.
    + backbone:  at 
      ```bash
       work_dir/result/Efficientb0_SCLS_attnscls_CrossEntropy_32
      ```
    + MMCBM: at 
      ```bash
      work_dir/result/CAV_m2CBM_sigmoid_C0.1CrossEntropy_32_report_strict_add_aow_zero_MM_max
      ```

### configuration

1. ChatGPT
   fill in the `openai_info` section in `params.py` with your own `api_key` and `api_base` from OpenAI.
2. Tencent Translation (Optional)
   fill in the `tencent_info` section in `params.py` with your own `app_id` and `app_key` from Tencent Cloud.

## Usage
### Web Interface
1. Web Interface using Gradio (Recommended), our web interface is available
   at [Interface Link](https://mmcbm.liuy.site).
2. you can also run this website locally by running the following command in the terminal:
   ```bash
   python web_interface.py
   ```
   then open the browser and enter `http://127.0.0.1:7860/` to access the website.
3. you can also build the docker image and run it:
   ```bash
   docker build -t mmcbm .
   docker run --name mmcbm -p 7860:7860 mmcbm
   ```
   then open the browser and enter `http://127.0.0.1:7890` to access the website.
### Command Line
Command Line without Gradio. We also provide a bash script to run the model inference from the command line:
   ```bash
   python mmcmb_inference.py
   ```