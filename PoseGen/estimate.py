# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

from tqdm.auto import tqdm
from lib.common.render import query_color, image2vid
from lib.renderer.mesh import compute_normal_batch
from lib.common.config import cfg
from lib.common.cloth_extraction import extract_cloth
from lib.dataset.mesh_util import (
    load_checkpoint, update_mesh_shape_prior_losses, get_optim_grid_image, blend_rgb_norm, unwrap,
    remesh, tensor2variable, rot6d_to_rotmat
)

from lib.dataset.TestDataset import TestDataset
from lib.net.local_affine import LocalAffine
from pytorch3d.structures import Meshes
from apps.ICON import ICON

import os
from termcolor import colored
import argparse
import numpy as np
from PIL import Image
import trimesh
import pickle
import numpy as np
import torch

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-colab", action="store_true")
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=100)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-vis_freq", "--vis_freq", type=int, default=1000)
    parser.add_argument("-loop_cloth", "--loop_cloth", type=int, default=200)
    parser.add_argument("-hps_type", "--hps_type", type=str, default="pixie")
    parser.add_argument("-export_video", action="store_true")
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument('-seg_dir', '--seg_dir', type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/icon-filter.yaml")

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymaf/configs/pymaf_config.yaml")

    cfg_show_list = [
        "test_gpus", [args.gpu_device], "mcube_res", 256, "clean_mesh", True, "test_mode", True,
        "batch_size", 1
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # gpus available :)
    device = torch.device(f"cuda:{args.gpu_device}")

    os.makedirs(args.out_dir, exist_ok=True)
    
    # load model and dataloader
    model = ICON(cfg)
    model = load_checkpoint(model, cfg)

    dataset_param = {
        'image_dir': args.in_dir,
        'seg_dir': args.seg_dir,
        'colab': args.colab,
        'has_det': True,    # w/ or w/o detection
        'hps_type':
            args.hps_type    # pymaf/pare/pixie
    }

    if args.hps_type == "pixie" and "pamir" in args.config:
        print(colored("PIXIE isn't compatible with PaMIR, thus switch to PyMAF", "red"))
        dataset_param["hps_type"] = "pymaf"

    dataset = TestDataset(dataset_param, device)

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    for data in pbar:

        pbar.set_description(f"{data['name']}")

        in_tensor = {"smpl_faces": data["smpl_faces"], "image": data["image"]}

        # The optimizer and variables
        optimed_pose = torch.tensor(
            data["body_pose"], device=device, requires_grad=True
        )    # [1,23,3,3]
        optimed_trans = torch.tensor(data["trans"], device=device, requires_grad=True)    # [3]
        optimed_betas = torch.tensor(data["betas"], device=device, requires_grad=True)    # [1,10]
        optimed_orient = torch.tensor(
            data["global_orient"], device=device, requires_grad=True
        )    # [1,1,3,3]

        optimizer_smpl = torch.optim.Adam(
            [optimed_pose, optimed_trans, optimed_betas, optimed_orient],
            lr=1e-3,
            amsgrad=True,
        )
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )

        losses = {
            "normal": {
                "weight": 1e0,
                "value": 0.0
            },    # Body: Normal_pred - Normal_smpl
            "silhouette": {
                "weight": 1e0,
                "value": 0.0
            },    # Body: Silhouette_pred - Silhouette_smpl
        }

        # smpl optimization

        loop_smpl = tqdm(range(args.loop_smpl))

        per_data_lst = []

        for i in loop_smpl:

            per_loop_lst = []

            optimizer_smpl.zero_grad()

            # 6d_rot to rot_mat
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).unsqueeze(0)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).unsqueeze(0)

            if dataset_param["hps_type"] != "pixie":
                smpl_out = dataset.smpl_model(
                    betas=optimed_betas,
                    body_pose=optimed_pose_mat,
                    global_orient=optimed_orient_mat,
                    transl=optimed_trans,
                    pose2rot=False,
                )

                smpl_verts = smpl_out.vertices * data["scale"]
                smpl_joints = smpl_out.joints * data["scale"]
            else:
                smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
                    shape_params=optimed_betas,
                    expression_params=tensor2variable(data["exp"], device),
                    body_pose=optimed_pose_mat,
                    global_pose=optimed_orient_mat,
                    jaw_pose=tensor2variable(data["jaw_pose"], device),
                    left_hand_pose=tensor2variable(data["left_hand_pose"], device),
                    right_hand_pose=tensor2variable(data["right_hand_pose"], device),
                )

                smpl_verts = (smpl_verts + optimed_trans) * data["scale"]
                smpl_joints = (smpl_joints + optimed_trans) * data["scale"]

            smpl_joints *= torch.tensor([1.0, 1.0, -1.0]).to(device)

            if data["type"] == "smpl":
                in_tensor["smpl_joint"] = smpl_joints[:, :24, :]
            elif data["type"] == "smplx" and dataset_param["hps_type"] != "pixie":
                in_tensor["smpl_joint"] = smpl_joints[:, dataset.smpl_joint_ids_24, :]
            else:
                in_tensor["smpl_joint"] = smpl_joints[:, dataset.smpl_joint_ids_24_pixie, :]

            # render optimized mesh (normal, T_normal, image [-1,1])
            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device), in_tensor["smpl_faces"]
            )
            T_mask_F, T_mask_B = dataset.render.get_silhouette_image()

            with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = model.netG.normal_filter(in_tensor)

            diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
            diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

            losses["normal"]["value"] = (diff_F_smpl + diff_B_smpl).mean()

            # silhouette loss
            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)[0]
            gt_arr = torch.cat([in_tensor["normal_F"][0], in_tensor["normal_B"][0]],
                               dim=2).permute(1, 2, 0)
            gt_arr = ((gt_arr + 1.0) * 0.5).to(device)
            bg_color = (torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device))
            gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()
            diff_S = torch.abs(smpl_arr - gt_arr)
            losses["silhouette"]["value"] = diff_S.mean()

            # Weighted sum of the losses
            smpl_loss = 0.0
            pbar_desc = "Body Fitting --- "
            for k in ["normal", "silhouette"]:
                pbar_desc += f"{k}: {losses[k]['value'] * losses[k]['weight']:.3f} | "
                smpl_loss += losses[k]["value"] * losses[k]["weight"]
            pbar_desc += f"Total: {smpl_loss:.3f}"
            loop_smpl.set_description(pbar_desc)

            if i % args.vis_freq == 0:

                per_loop_lst.extend(
                    [
                        in_tensor["image"],
                        in_tensor["T_normal_F"],
                        in_tensor["normal_F"],
                        diff_F_smpl / 2.0,
                        diff_S[:, :512].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
                    ]
                )
                per_loop_lst.extend(
                    [
                        in_tensor["image"],
                        in_tensor["T_normal_B"],
                        in_tensor["normal_B"],
                        diff_B_smpl / 2.0,
                        diff_S[:, 512:].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
                    ]
                )
                per_data_lst.append(get_optim_grid_image(per_loop_lst, None, nrow=5, type="smpl"))

            smpl_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(smpl_loss)
            in_tensor["smpl_verts"] = smpl_verts * \
                torch.tensor([1.0, 1.0, -1.0]).to(device)

        smpl_info = {
            'betas': optimed_betas,
            'pose': optimed_pose,
            'orient': optimed_orient,
            'trans': optimed_trans
        }

        np.save(
            f"{args.out_dir}/{data['name']}_smpl.npy", smpl_info, allow_pickle=True
        )
