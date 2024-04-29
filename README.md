<div align="center">
    <h2>
      AiOS: All-in-One-Stage Expressive Human Pose and Shape Estimation
    </h2>
</div>

<div align="center">
    <a href="https://ttxskk.github.io/AiOS/" class="button"><b>[Homepage]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://arxiv.org/abs/2403.17934" class="button"><b>[arXiv]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://ttxskk.github.io/AiOS/" class="button"><b>[Code]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
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
- download datasets for evaluation
  - [AGORA](https://agora.is.tue.mpg.de/index.html)       
  - [BEDLAM](https://bedlam.is.tue.mpg.de/index.html)      
- download [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models.
- download AiOS [checkpoint]()
- download AGORA validation set [Humandata](https://drive.google.com/file/d/1cjCVwrFdZ9qMXsA_yaZa3_plYYK8uyPU/view?usp=sharing)

Organize them according to this datastructure:
```text
AiOS/
├── config/
└── data
    ├── body_models
        └── smplx
    |       ├──MANO_SMPLX_vertex_ids.pkl
    |       ├──SMPL-X__FLAME_vertex_ids.npy
    |       ├──SMPLX_NEUTRAL.pkl
    |       ├──SMPLX_to_J14.pkl
    |       ├──SMPLX_NEUTRAL.npz
    |       ├──SMPLX_MALE.npz
    |       └──SMPLX_FEMALE.npz
    ├── cache
    ├── checkpoint
    │   └── aios_checkpoint.pth
    ├── datasets
    │   ├── agora
    │   └── bedlam
    └── multihuman_data
        └── agora_validation_multi_3840_1010.npz
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
cd pythorch3d
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
cd main
sh scripts/inference.sh {INPUT_VIDEO} {OUTPUT_DIR} 

# For inferencing short_video.mp4 with output directory of demo/short_video_out
sh scripts/inference.sh short_video demo
```
# Test
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
sh scripts/test_bedlam.sh
```

# Train (Coming Soon)



# Acknowledge

Some of the codes are based on [`MMHuman3D`](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/install.md), [`ED-Pose`](https://github.com/IDEA-Research/ED-Pose/tree/master) and [`SMPLer-X`](https://github.com/caizhongang/SMPLer-X).
