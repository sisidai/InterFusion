import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import ShapeLoss, binary_cross_entropy, dot
from threestudio.utils.typing import *
import imageio
import torch
import numpy as np
import smplx
from coap import attach_coap
from threestudio.utils.ops import get_rays, transform, to_numpy, to_tensor, get_body_verts, params_encapsulate, rot6d_to_rotation_matrix, rotation_matrix_to_angle_axis


@threestudio.register("interfusion-system")
class InterFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        guide_shape: Optional[str] = None
        aug_rot: bool = False
        aug_trans: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

        to_device = torch.device("cuda")
        self.center: Float[Tensor, "1 3"] = torch.zeros((1, 3)).to(to_device)
        smpl_model = smplx.create('/root/test/models', model_type='smplx', gender='neutral', num_pca_comps=12, ext='npz').to(to_device)
        smpl_model = attach_coap(smpl_model, pretrained=True, device=to_device)
        assert smpl_model.joint_mapper is None, 'Valid joints are required!!!'

        load_pose = to_tensor(np.load(self.cfg.guide_shape), to_device).reshape(1, -1)

        print(self.cfg.aug_rot, self.cfg.aug_trans)
        assert not (self.cfg.aug_rot and self.cfg.aug_trans), 'Valid augmentations are required!!!'
        if not self.cfg.aug_rot and not self.cfg.aug_trans:
            body_verts, _, _ = get_body_verts(params_encapsulate(torch.zeros([1, 3]).to(to_device), torch.zeros([1, 3]).to(to_device), load_pose, params_to_zero=[]), smpl_model) # [B, N, 3]
            params_dict = params_encapsulate(-torch.mean(body_verts, dim=1), torch.zeros([1, 3]).to(to_device), load_pose, params_to_zero=[])
        elif self.cfg.aug_rot:
            body_verts, _, _ = get_body_verts(params_encapsulate(torch.zeros([1, 3]).to(to_device), torch.tensor([[-np.pi/6, 0, 0]]).to(to_device), load_pose, params_to_zero=[]), smpl_model) # [B, N, 3]
            params_dict = params_encapsulate(-torch.mean(body_verts, dim=1), torch.tensor([[-np.pi/6, 0, 0]]).to(to_device), load_pose, params_to_zero=[])
        elif self.cfg.aug_trans:
            _, t_jnts, _ = get_body_verts(params_encapsulate(torch.zeros([1, 3]).to(to_device), torch.zeros([1, 3]).to(to_device), torch.zeros([1, 63]).to(to_device), params_to_zero = []), smpl_model) # [B, N, 3]
            body_root = t_jnts[:, 0, :].detach() # Tensor [1, 3]        
            params_dict = params_encapsulate(-body_root, torch.zeros([1, 3]).to(to_device), load_pose, params_to_zero=[])

        _, jnts, smpl_output = get_body_verts(params_dict, smpl_model)

        matrix_rot = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        self.head = transform(jnts[:, 15, :].clone().detach(), matrix_rot)

        if self.cfg.guide_shape is not None:
            self.shape_loss = ShapeLoss(smpl_model, smpl_output)

    def forward(self, batch: Dict[str, Any], is_train=True) -> Dict[str, Any]:
        if self.true_global_step % 1000 == 0 and self.true_global_step > 0:
            self.center = self.geometry.isocenter()

        if is_train:
            o_camera_positions = 0.7 * batch["camera_positions"] + self.center
        else:
            o_camera_positions = batch["camera_positions"] + self.center
        o_light_positions = batch["light_positions"] + self.center        
        o_c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [batch["c_mtx"], o_camera_positions[:, :, None]],
            dim=-1,
        )
        o_c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [o_c2w3x4, torch.zeros_like(o_c2w3x4[:, :1])], dim=1
        )
        o_c2w[:, 3, 3] = 1.0
        o_rays_o, o_rays_d = get_rays(batch["directions"], o_c2w, keepdim=True)
        o_batch = {
            "rays_o": o_rays_o,
            "rays_d": o_rays_d,
            "light_positions": o_light_positions,
        }

        if is_train:
            i_camera_positions = 0.8 * batch["camera_positions"] + self.center/2
        else:
            i_camera_positions = batch["camera_positions"] + self.center/2
        i_light_positions = batch["light_positions"] + self.center/2        
        i_c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [batch["c_mtx"], i_camera_positions[:, :, None]],
            dim=-1,
        )
        i_c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [i_c2w3x4, torch.zeros_like(i_c2w3x4[:, :1])], dim=1
        )
        i_c2w[:, 3, 3] = 1.0
        i_rays_o, i_rays_d = get_rays(batch["directions"], i_c2w, keepdim=True)
        i_batch = {
            "rays_o": i_rays_o,
            "rays_d": i_rays_d,
            "light_positions": i_light_positions,
        }

        if is_train:
            h_camera_positions = 0.7 * batch["camera_positions"]
        else:
            h_camera_positions = batch["camera_positions"]
        h_light_positions = batch["light_positions"]
        h_c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [batch["c_mtx"], h_camera_positions[:, :, None]],
            dim=-1,
        )
        h_c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [h_c2w3x4, torch.zeros_like(h_c2w3x4[:, :1])], dim=1
        )
        h_c2w[:, 3, 3] = 1.0
        h_rays_o, h_rays_d = get_rays(batch["directions"], h_c2w, keepdim=True)
        h_batch = {
            "rays_o": h_rays_o,
            "rays_d": h_rays_d,
            "light_positions": h_light_positions,
        }

        hh_camera_positions = 0.2 * batch["camera_positions"] + self.head
        hh_light_positions = batch["light_positions"] + self.head
        hh_c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [batch["c_mtx"], hh_camera_positions[:, :, None]],
            dim=-1,
        )
        hh_c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [hh_c2w3x4, torch.zeros_like(hh_c2w3x4[:, :1])], dim=1
        )
        hh_c2w[:, 3, 3] = 1.0
        hh_rays_o, hh_rays_d = get_rays(batch["directions"], hh_c2w, keepdim=True)
        hh_batch = {
            "rays_o": hh_rays_o,
            "rays_d": hh_rays_d,
            "light_positions": hh_light_positions,
        }

        render_out_o = self.renderer_o(**o_batch)
        render_out_h = self.renderer_h(**h_batch)
        render_out_i = self.renderer_i(**i_batch)
        render_out_hh = self.renderer_h(**hh_batch)
        return {**render_out_o,}, {**render_out_h,}, {**render_out_i,}, {**render_out_hh,}

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor_o = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor_o
        )
        self.prompt_processor_h = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor_h
        )
        self.prompt_processor_i = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor_i
        )
        self.prompt_processor_hh = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor_hh
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):

        data_o, data_h, data_i, data_hh = self(batch)

        prompt_utils_o = self.prompt_processor_o()
        prompt_utils_h = self.prompt_processor_h()
        prompt_utils_i = self.prompt_processor_i()
        prompt_utils_hh = self.prompt_processor_hh()

        loss = 0.0

        weight = (self.true_global_step//1000) / 10
        guidance_out_o = self.guidance(
            data_o["comp_rgb"], prompt_utils_o, **batch, rgb_as_latents=False
        )
        guidance_out_i = self.guidance(
            data_i["comp_rgb"], prompt_utils_i, **batch, rgb_as_latents=False
        )

        guidance_out_h = self.guidance(
            data_h["comp_rgb"], prompt_utils_h, **batch, rgb_as_latents=False
        )

        guidance_out_hh = self.guidance(
            data_hh["comp_rgb"], prompt_utils_hh, **batch, rgb_as_latents=False
        )

        for name, value in guidance_out_o.items():
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")]) * weight
        
        for name, value in guidance_out_i.items():
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        for name, value in guidance_out_h.items():
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        for name, value in guidance_out_hh.items():
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in data_o:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient_o = (
                data_o["weights"].detach()
                * dot(data_o["normal"], data_o["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (data_o["opacity"] > 0).sum()
            # self.log("train/loss_orient_o", loss_orient_o)
            loss += loss_orient_o * self.C(self.cfg.loss.lambda_orient)
    
        loss_sparsity_o = (data_o["opacity"] ** 2 + 0.01).sqrt().mean()
        # self.log("train/loss_sparsity_o", loss_sparsity_o)
        loss += loss_sparsity_o * self.C(self.cfg.loss.lambda_sparsity)
    
        opacity_clamped_o = data_o["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque_o = binary_cross_entropy(opacity_clamped_o, opacity_clamped_o)
        # self.log("train/loss_opaque_o", loss_opaque_o)
        loss += loss_opaque_o * self.C(self.cfg.loss.lambda_opaque)
    
        if (
            self.cfg.guide_shape is not None
            and self.C(self.cfg.loss.lambda_shape) > 0
            and data_o["points"].shape[0] > 0
        ):
            loss_shape_o = self.shape_loss(data_o["points"], data_o["density"], "type3")
            # self.log("train/loss_shape_o", loss_shape_o)
            if self.true_global_step < 1000 or self.true_global_step > 9000:
                shape_weight = self.cfg.loss.shape_weights[0]
            elif self.true_global_step < 2000 or self.true_global_step > 8000:
                shape_weight = self.cfg.loss.shape_weights[1]
            else:
                shape_weight = self.cfg.loss.shape_weights[2]
            loss += loss_shape_o * self.C(self.cfg.loss.lambda_shape) * shape_weight

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in data_h:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient_h = (
                data_h["weights"].detach()
                * dot(data_h["normal"], data_h["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (data_h["opacity"] > 0).sum()
            # self.log("train/loss_orient_h", loss_orient_h)
            loss += loss_orient_h * self.C(self.cfg.loss.lambda_orient)
    
        loss_sparsity_h = (data_h["opacity"] ** 2 + 0.01).sqrt().mean()
        # self.log("train/loss_sparsity_h", loss_sparsity_h)
        loss += loss_sparsity_h * self.C(self.cfg.loss.lambda_sparsity)
    
        opacity_clamped_h = data_h["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque_h = binary_cross_entropy(opacity_clamped_h, opacity_clamped_h)
        # self.log("train/loss_opaque_h", loss_opaque_h)
        loss += loss_opaque_h * self.C(self.cfg.loss.lambda_opaque)
    
        if (
            self.cfg.guide_shape is not None
            and self.C(self.cfg.loss.lambda_shape) > 0
            and data_h["points"].shape[0] > 0
        ):
            loss_shape_h = self.shape_loss(data_h["points"], data_h["density"], "type2")
            # self.log("train/loss_shape_h", loss_shape_h)
            loss += loss_shape_h * self.C(self.cfg.loss.lambda_shape)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out_o, out_h, out_i, out_hh = self(batch)
        
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out_o["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out_o["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_o
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_o["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out_h["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out_h["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_h
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_h["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out_i["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out_i["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_i
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_i["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            # + [
            #     {
            #         "type": "rgb",
            #         "img": out_hh["comp_rgb"][0],
            #         "kwargs": {"data_format": "HWC"},
            #     },
            # ]
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out_hh["comp_normal"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal" in out_hh
            #     else []
            # )
            # + [
            #     {
            #         "type": "grayscale",
            #         "img": out_hh["opacity"][0, :, :, 0],
            #         "kwargs": {"cmap": None, "data_range": (0, 1)},
            #     },
            # ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out_o, out_h, out_i, out_hh = self(batch, is_train=False)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": (out_o["comp_rgb_fg"] + (1.0 - out_o["opacity"]))[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": (out_o["comp_normal"] + (1.0 - out_o["opacity"]))[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_o
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_o["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": (out_h["comp_rgb_fg"] + (1.0 - out_h["opacity"]))[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": (out_h["comp_normal"] + (1.0 - out_h["opacity"]))[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_h
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_h["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": (out_i["comp_rgb_fg"] + (1.0 - out_i["opacity"]))[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": (out_i["comp_normal"] + (1.0 - out_i["opacity"]))[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_i
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_i["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            # + [
            #     {
            #         "type": "rgb",
            #         "img": (out_hh["comp_rgb_fg"] + (1.0 - out_hh["opacity"]))[0],
            #         "kwargs": {"data_format": "HWC"},
            #     },
            # ]
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out_hh["comp_normal"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal" in out_hh
            #     else []
            # ),
            # + [
            #     {
            #         "type": "grayscale",
            #         "img": out_hh["opacity"][0, :, :, 0],
            #         "kwargs": {"cmap": None, "data_range": (0, 1)},
            #     },
            # ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
