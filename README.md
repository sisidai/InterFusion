<div align="center">

<h1>[ECCV 2024] InterFusion: Text-Driven Generation of 3D Human-Object Interaction</h1>

![](./assets/teaser.gif)

<h4>InterFusion can generate diverse 3D scenes of human-object interaction (3D HOI) given texts.</h4>

<h4 align="center">
  <a href="https://sisidai.github.io/InterFusion/" target='_blank'>[Project Page]</a> •
  <a href="https://arxiv.org/abs/2403.15612" target='_blank'>[arXiv]</a> •
  <a href="https://arxiv.org/pdf/2403.15612.pdf" target='_blank'>[PDF]</a>
</h4>

<!-- This repository contains the official implementation of InterFusion. -->

</div>

## :fire: Updates
[12/2024] Code released!

[07/2024] InterFusion is accepted to ECCV 2024!

## :hammer: Installation

For flexibility, we provide two separate environments, s1 and s2, corresponding to the two stages of InterFusion. We recommend using anaconda to manage the environments. Additionally, we provide `INSTALL.MD` which summarizes potential issues and solutions that may arise during environment installation. If you encounter any issues not mentioned in the document, please submit an issue.

#### Enviroment for pose generation
Set up the environment:
```
conda create -n interfusion python=3.8
conda activate interfusion
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install necessary dependencies:
```
pip install -r requirements-s1.txt
```

Install neural mesh renderer library:
```
git clone https://github.com/adambielski/neural_renderer.git
cd neural_renderer
python setup.py install
cd ..
rm -rf neural_renderer
```

Install osmesa library to support offscreen rendering:
```
conda install -c menpo osmesa
```

We adopt the [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) integrated in the Hugging Face Diffusers library. To use it:
1. Make sure you have a [Hugging Face account](https://huggingface.co/join) and are logged in;
2. Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0);
3. Run the login function in a Python shell to login locally, and enter your [Hugging Face Hub access token](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens):
```
from huggingface_hub import login

login()
```

#### Enviroment for interaction generation
Set up the environment:
```
conda create -n threestudio python=3.8
conda activate threestudio
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
Install tiny-cuda-nn:
```
pip install --force-reinstall  git+https://github.com/NVlabs/tiny-cuda-nn@78a14fe8c292a69f54e6d0d47a09f52b777127e1#subdirectory=bindings/torch
```
Install other dependencies:
```
pip install -r requirements-s2.txt
```
Moreover, please replace `.../anaconda3/envs/threestudio/lib/python3.8/site-packages/nerfacc/estimators/occ_grid.py` with `./occ_grid.py`.

## :wrench: Data Preparation
We utilize the PIXIE model for human pose estimation and adopt its implementation in [ICON](https://github.com/YuliangXiu/ICON/tree/master):
```
cd PoseGen
git clone https://github.com/YuliangXiu/ICON.git
mv estimate.py ICON/apps
```
Make sure you register the dependencies: [SMPL](http://smpl.is.tue.mpg.de/), [SMPLIFY](http://smplify.is.tue.mpg.de/), [SMPL-X](http://smpl-x.is.tue.mpg.de/), [ICON](https://icon.is.tue.mpg.de/), [PIXIE](https://pixie.is.tue.mpg.de/). Then download the required models with the account (username and password) and our data:
```
bash fetch_data.sh
```
The final data folder structure should look like:
```
PoseGen/
├── ICON/
│   ├── data
│   │   ├── ckpt/
│   │   │   └── ... 
│   │   ├── HPS/
│   │   │   └── ... 
│   │   ├── smpl_related/
│   │   │   ├── models/
│   │   │   │   ├── smpl/
│   │   │   │   │   └── ...   
│   │   │   │   ├── smplx/
│   │   │   │   │   └── ...  
│   │   │   ├── smpl_data/
│   │   │   │   └── ...  
...
data/
├── smplx_model/
│   ├── smplx/
        └── ...
├── smplx_uv/
│   │   ├── f_02_alb.002.png
│   │   ├── smpl_uv.mtl
│   │   └── smpl_uv.obj
├── vposer/
│   ├── snapshots/
│   │   └── ...
│   ├── V02_05.log
│   └── V02_05.yaml
└── codebook.pth
```

## :closed_book: Pose Generation
Codes for this part are located in `PoseGen`:
```
conda activate interfusion
cd PoseGen
```
If you'd like to construct your own codebook tailored to your task, please follow the step-by-step instructions below. Otherwise, for a quick start, you can skip to the final step and directly use our provided codebook `data/codebook.pth`. Note that this pose codebook is interaction-biased, as it was built based on the interaction-biased texts from `PoseGen/prompt.txt`. 
#### Step 1: Generate images from text prompts
```
python gen_image.py -in_path prompt.txt -out_path results/images -num_images_per_text 250
```
#### Step 2: Estimate human poses from generated images
```
cd ICON
python -m apps.estimate -cfg ./configs/icon-filter.yaml -gpu 0 -in_dir ../results/images/ -out_dir ../results/estimated_poses -hps_type pixie
cd ..
```
#### Step 3: Cluster estimated poses and create the codebook
```
python gen_codebook.py -in_path results/estimated_poses -out_path results/codebook.pth -cluster_size 2048
```
#### Step 4: Use the codebook to generate pose from the input text
```
python gen_pose.py -in_path ../data/codebook.pth -out_path results/interfusion_poses -inter_text "riding a bike" -topk 7
```
(Optional) With the rendered TopK poses, e.g. `PosGen/results/interfusion_poses/***.png`, ask GPT-4V to select the most precise pose. Here is an example prompt:
```
Here are seven poses with indexes, please give me the index that which one best physically matches the human-object interaction "***"? 
```
We encourage you to have a try to construct your own pose codebook, as it is a cost-effective approach that can yield diverse poses (pseudo). Here are some tips that may be helpful for your constrcuting:
1. The greater the number of poses generated (in step 1 and step 2), the more varied the resulting pose codebook will be. 
2. The codebook is created based on the similarity between the text feature and image features rendered from multiple views. As a result, rendering factors, such as camera positions and human body texture, can influence the final outcome. You can experiment with these factors to observe different results.

## :orange_book: Interaction Generation
Make sure Step 4 of `Pose Generation` has been executed. Codes for `Interaction Generation` are located in `InterGen`:
```
conda activate threestudio
cd InterGen
```
#### Start training
We provide example configurations in `configs/interfusion-if/`. Choose one to start training:
```
python launch.py --config configs/interfusion-if/sitting_on_a_chair.yaml --train --gpu 0
```
#### Resume from checkpoint
To resume training from the last checkpoint, replace the `path/to/trial` with the saved path:
```
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
```
If the training has completed and you wanna continue training for a longer time, set the `trainer.max_steps`:
```
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt trainer.max_steps=20000
```
Resuming the last checkpoint for testing:
```
python launch.py --config path/to/trial/dir/configs/parsed.yaml --test --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
```
Load weights from checkpoint but do not resume training (i.e. do not load optimizer state):
```
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 system.weights=path/to/trial/dir/ckpts/last.ckpt
```
Note that the above commands use parsed configuration files from previous trials, which will continue using the same trial directory. If you wanna save to a new trial directory, replace `parsed.yaml` with `raw.yaml` in the command.
#### Make new configuration
- Relace the pose path in `system.guide_shape`.
- Relace prompts for object style, human style, interaction descriptions, ***head of*** human style in `system.prompt_processor_o.prompt`, `system.prompt_processor_h.prompt`, `system.prompt_processor_i.prompt`, `system.prompt_processor_hh.prompt`, respectively.
- To get more satisfying results, consider augemntations, e.g. setting `system.aug_rot` to True or setting `system.aug_trans` to True. Additionally, you can modify `system.loss.shape_weights` or `data.camera_distance_range`.
- Besides the prompts in the [paper](https://arxiv.org/pdf/2403.15612.pdf), you can refer to [interfusion_more.pdf](https://sisidai.github.io/Supplementary_Materials/interfusion_more.pdf) for additional prompts.

## :bulb: Citation
If you find InterFusion useful for your research, please consider citing the paper:
```
@inproceedings{dai2024interfusion,
  title={InterFusion: Text-Driven Generation of 3D Human-Object Interaction},
  author={Dai, Sisi and Li, Wenhao and Sun, Haowen and Huang, Haibin and Ma, Chongyang and Huang, Hui and Xu, Kai and Hu, Ruizhen},
  booktitle={ECCV},
  year={2024}
}
```

## :raised_hands: Acknowledgements
Our repository is built upon the shoulders of giants:
- [threestudio](https://github.com/threestudio-project/threestudio), 
- [AvatarCLIP](https://github.com/hongfz16/AvatarCLIP), 
- [DeepFloyd/IF](https://github.com/deep-floyd/IF), 
- [ICON](https://github.com/YuliangXiu/ICON/tree/master). 

We sincerely thank them for their contributions.