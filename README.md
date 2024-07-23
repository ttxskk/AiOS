<style>
  .publication-authors {
    text-align: center;
  }
  .author-block {
    display: inline-block;
    margin-right: 5px;
  }
</style>
<div align="center">
    <h2>
      AiOS: All-in-One-Stage Expressive Human Pose and Shape Estimation
    </h2>
</div>
<div class="is-size-5 publication-authors">
  <span class="author-block">
    <a href="https://github.com/ttxskk" target="_blank">Qingping Sun</a><sup>1, 2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://github.com/WYJSJTU" target="_blank">Yanjun Wang</a><sup>1</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://ailingzeng.site/" target="_blank">Ailing Zeng</a><sup>3</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=zlIJwBEAAAAJ" target="_blank">Wanqi Yin</a><sup>1</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://www.linkedin.com/in/chen-wei-weic0006/" target="_blank">Chen Wei</a><sup>1</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://wenjiawang0312.github.io/" target="_blank">Wenjia Wang</a><sup>5</sup>,&nbsp;
  </span>
  <br>
  <span class="author-block">
    <a href="https://haiyi-mei.com" target="_blank">Haiyi Mei</a><sup>1</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://ttxskk.github.io/AiOS/" target="_blank">Chi Sing Leung</a><sup>2</sup>,&nbsp;
    <span class="author-block">
      <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>4</sup>,&nbsp;
    </span>
  </span>
  <span class="author-block">
    <a href="https://yanglei.me/" target="_blank">Lei Yang</a><sup>1, 5</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://caizhongang.github.io/" target="_blank">Zhongang Cai</a><sup>✉, 1, 4, 5</sup>,&nbsp;
  </span>
</div>
<br>
<div class="is-size-5 publication-authors">
  <span class="author-block"><sup>1</sup>SenseTime Research</span>,
  <span class="author-block"><sup>2</sup>City University of Hong Kong</span>,
  <br>
  <span class="author-block"><sup>3</sup>International Digital Economy Academy (IDEA)</span>,
  <br>
  <span class="author-block"><sup>4</sup>S-Lab, Nanyang Technological University</span>,
  <span class="author-block"><sup>5</sup>Shanghai AI Laboratory</span>
</div>
<br>
<div align="center">
    <a href="https://ttxskk.github.io/AiOS/"><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href="https://arxiv.org/abs/2403.17934"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
    <a href="https://huggingface.co/spaces/ttxskk/AiOS"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue'></a> 
</div>

---
<img width="1195" alt="method" src="https://github.com/ttxskk/AiOS/assets/24960075/40177dd2-886e-4f17-addc-ba4729bcc58e">

<div class="columns is-centered has-text-centered">
  <div class="column">
    <div class="content has-text-justified">
      <p>
        AiOS performs human localization and SMPL-X estimation in a progressive manner.
        It is composed of (1) the body localization stage that predicts coarse human location;
        (2) the Body refinement stage that refines body features and produces face and
        hand locations; (3) the Whole-body Refinement stage that refines whole-body features and regress SMPL-X
        parameters.
      </p>
    </div>
  </div>
</div>






## Preparation
- download all datasets
  - [AGORA](https://agora.is.tue.mpg.de/index.html)       
  - [BEDLAM](https://bedlam.is.tue.mpg.de/index.html)   
  - [MSCOCO](https://cocodataset.org/#home) 
  - [UBody](https://github.com/IDEA-Research/OSX)
  - [ARCTIC](https://arctic.is.tue.mpg.de/) 
  - [EgoBody](https://sanweiliti.github.io/egobody/egobody.html)
  - [EHF](https://smpl-x.is.tue.mpg.de/index.html)
- process all datasets into [HumanData](https://github.com/open-mmlab/mmhuman3d/blob/main) format. We provided the proccessed npz file, which can be download from [here](https://huggingface.co/datasets/ttxskk/AiOS_Train_Data).
- download [SMPL-X](https://smpl-x.is.tue.mpg.de/)
- download AiOS [checkpoint](https://huggingface.co/ttxskk/AiOS/tree/main)

The file structure should be like:
```text
AiOS/
├── config/
└── data
    ├── body_models
    |   ├── smplx
    |   |   ├──MANO_SMPLX_vertex_ids.pkl
    |   |   ├──SMPL-X__FLAME_vertex_ids.npy
    |   |   ├──SMPLX_NEUTRAL.pkl
    |   |   ├──SMPLX_to_J14.pkl
    |   |   ├──SMPLX_NEUTRAL.npz
    |   |   ├──SMPLX_MALE.npz
    |   |   └──SMPLX_FEMALE.npz
    |   └── smpl
    |       ├──SMPL_FEMALE.pkl
    |       ├──SMPL_MALE.pkl
    |       └──SMPL_NEUTRAL.pkl
    ├── preprocessed_npz
    │   └── cache
    |       ├──agora_train_3840_w_occ_cache_2010.npz
    |       ├──bedlam_train_cache_080824.npz
    |       ├──...
    |       └──coco_train_cache_080824.npz
    ├── checkpoint
    │   └── aios_checkpoint.pth
    ├── datasets
    │   ├── agora
    |   │    └──3840x2160
    │   │        ├──train
    │   │        └──test
    │   ├── bedlam
    │   │     ├──train_images
    │   │     └──test_images
    │   ├── ARCTIC
    │   │     ├──s01
    │   │     ├──s02
    │   │     ├──...   
    │   │     └──s10
    │   ├── EgoBody
    │   │     ├──egocentric_color
    │   │     └──kinect_color
    │   └── UBody
    |       └──images
    └── checkpoint
        ├── edpose_r50_coco.pth
        └── aios_checkpoint.pth

```
# Installtion

```shell
# Create a conda virtual environment and activate it.
conda create -n aios python=3.8 -y
conda activate aios

# Install PyTorch and torchvision.
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install Pytorch3D
git clone -b v0.6.1 https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -v -e .
cd ..

# Install MMCV, build from source
git clone -b v1.6.1 https://github.com/open-mmlab/mmcv.git
cd mmcv
export MMCV_WITH_OPS=1
export FORCE_MLU=1
pip install -v -e .
cd ..

# Install other dependencies
conda install -c conda-forge ffmpeg
pip install -r requirements.txt 

# Build deformable detr
cd models/aios/ops
python setup.py build install
cd ../../..
```

## Inference 
- Place the mp4 video for inference under `AiOS/demo/`
- Prepare the pretrained models to be used for inference under `AiOS/data/checkpoint`
- Inference output will be saved in `AiOS/demo/{INPUT_VIDEO}_out` 

```bash
# CHECKPOINT: checkpoint path
# INPUT_VIDEO: input video path
# OUTPUT_DIR: output path
# NUM_PERSON: num of person. This parameter sets the expected number of persons to be detected in the input (image or video). 
#   The default value is 1, meaning the algorithm will try to detect at least one person. If you know the maximum number of persons
#   that can appear simultaneously, you can set this variable to that number to optimize the detection process (a lower threshold is recommended as well).
# THRESHOLD: socre threshold. This parameter sets the score threshold for person detection. The default value is 0.5. 
#   If the confidence score of a detected person is lower than this threshold, the detection will be discarded. 
#   Adjusting this threshold can help in filtering out false positives or ensuring only high-confidence detections are considered.
# GPU_NUM: GPU num. 
sh scripts/inference.sh {CHECKPOINT} {INPUT_VIDEO} {OUTPUT_DIR} {NUM_PERSON} {THRESHOLD} {THRESHOLD}

# For inferencing short_video.mp4 with output directory of demo/short_video_out
sh scripts/inference.sh data/checkpoint/aios_checkpoint.pth short_video.mp4 demo 2 0.1 8
```
# Test

<table>
 <tr>
   <th></th>
   <th colspan="2">NMVE</th>
   <th colspan="2">NMJE</th>
   <th colspan="4">MVE</th>
   <th colspan="4">MPJPE</th>
 </tr>
 <tr>
   <th>DATASETS</th>
   <th>FB</th>
   <th>B</th>
   <th>FB</th>
   <th>B</th>
   <th>FB</th>
   <th>B</th>
   <th>F</th>
   <th>LH/RH</th>
   <th>FB</th>
   <th>B</th>
   <th>F</th>
   <th>LH/RH</th>
 </tr>
 <tr>
   <td>BEDLAM</td>
   <td>87.6</td>
   <td>57.7</td>
   <td>85.8</td>
   <td>57.7</td>
   <td>83.2</td>
   <td>54.8</td>
   <td>26.2</td>
   <td>28.1/30.8</td>
   <td>81.5</td>
   <td>54.8</td>
   <td>26.2</td>
   <td>25.9/28.0</td>
 </tr>
 <tr>
   <td>AGORA-Test</td>
   <td>102.9</td>
   <td>63.4</td>
   <td>100.7</td>
   <td>62.5</td>
   <td>98.8</td>
   <td>60.9</td>
   <td>27.7</td>
   <td>42.5/43.4</td>
   <td>96.7</td>
   <td>60.0</td>
   <td>29.2</td>
   <td>40.1/41.0</td>
   </tr>
  <tr>
   <td>AGORA-Val</td>
   <td>105.1</td>
   <td>60.9</td>
   <td>102.2</td>
   <td>61.4</td>
   <td>100.9</td>
   <td>60.9</td>
   <td>30.6</td>
   <td>43.9/45.6</td>
   <td>98.1</td>
   <td>58.9</td>
   <td>32.7</td>
   <td>41.5/43.4</td>
 </tr>
</table>


a. Make test_result dir 
```shell
mkdir test_result
```


b. AGORA Validatoin

Run the following command and it will generate a 'predictions/' result folder which can evaluate with the [agora evaluation tool](https://github.com/pixelite1201/agora_evaluation)    

```shell
sh scripts/test_agora_val.sh data/checkpoint/aios_checkpoint.pth agora_val
```


b. AGORA Test Leaderboard


Run the following command and it will generate a 'predictions.zip' which can be submitted to AGORA Leaderborad
```shell
sh scripts/test_agora.sh data/checkpoint/aios_checkpoint.pth agora_test
```


c. BEDLAM


Run the following command and it will generate a 'predictions.zip' which can be submitted to BEDLAM Leaderborad
```shell
sh scripts/test_bedlam.sh data/checkpoint/aios_checkpoint.pth bedlam_test
```


# Acknowledge

Some of the codes are based on [`MMHuman3D`](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/install.md), [`ED-Pose`](https://github.com/IDEA-Research/ED-Pose/tree/master) and [`SMPLer-X`](https://github.com/caizhongang/SMPLer-X).
