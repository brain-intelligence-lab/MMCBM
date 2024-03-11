# MMCBM

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
1. download the [model checkpoint](https://drive.google.com/drive/folders/1YwDhqC_M9ACBnGjn_8IZouWHgJx1ue5Q?usp=drive_link) place it.
   + backbone:  at work_dir/result/Efficientb0_SCLS_attnscls_CrossEntropy_32
   + MMCBM: at work_dir/result/CAV_m2CBM_sigmoid_C0.1CrossEntropy_32_report_strict_add_aow_zero_MM_max
### configuration
1. ChatGPT
   fill in the `openai_info` section in `params.py` with your own `api_key` and `api_base` from OpenAI.
2. Tencent Translation (Optional)
   fill in the `tencent_info` section in `params.py` with your own `app_id` and `app_key` from Tencent Cloud.
## Usage

1. Web Interface using Gradio (Recommended), our web interface is available
   at [Interface](https://intervention.liuy.site).

   ```bash
   python interface.py
   ```
   we also provide docker image for the web interface, you can pull the image from docker hub:
   ```bash
    docker run -d --restart=always --name intervention-web -p 7860:7860 ly1998117/intervention:latest
    ```

2. Command Line

   ```bash
   python mmcmb_inference.py
   ```