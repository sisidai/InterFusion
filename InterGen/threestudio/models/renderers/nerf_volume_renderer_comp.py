from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRendererComp
from threestudio.utils.ops import chunk_batch
from threestudio.utils.typing import *
import random


@threestudio.register("nerf-volume-renderer-comp")
class NeRFVolumeRenderer(VolumeRendererComp):
    @dataclass
    class Config(VolumeRendererComp.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        grid_prune: bool = True
        prune_alpha_threshold: bool = True
        return_comp_normal: bool = False
        return_normal_perturb: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
        geometry_: BaseImplicitGeometry,
        material_: BaseMaterial,
        background_: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background, geometry_, material_, background_)
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=self.bbox.view(-1), resolution=32, levels=1
        )
        if not self.cfg.grid_prune:
            self.estimator.occs.fill_(True)
            self.estimator.binaries.fill_(True)
        self.render_step_size = (
            1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        )
        self.randomized = self.cfg.randomized

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * t_positions
            if self.training:
                sigma = self.geometry.forward_density(positions)[..., 0]
            else:
                sigma = chunk_batch(
                    self.geometry.forward_density,
                    self.cfg.eval_chunk_size,
                    positions,
                )[..., 0]
            return sigma

        def sigma_fn_(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * t_positions
            if self.training:
                sigma = self.geometry_.forward_density(positions)[..., 0]
            else:
                sigma = chunk_batch(
                    self.geometry_.forward_density,
                    self.cfg.eval_chunk_size,
                    positions,
                )[..., 0]
            return sigma

        if not self.cfg.grid_prune:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    early_stop_eps=0,
                )
        else:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling_(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                    sigma_fn_=sigma_fn_ if self.cfg.prune_alpha_threshold else None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                )

        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        # TODO: still proceed if the scene is empty

        if self.training:
            geo_out = self.geometry(
                positions, output_normal=self.material.requires_normal
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d_flatten)
            geo_out_ = self.geometry_(
                positions, output_normal=self.material_.requires_normal
            )
            rgb_fg_all_ = self.material_(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out_,
                **kwargs
            )
            comp_rgb_bg_ = self.background_(dirs=rays_d_flatten)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d_flatten
            )
            geo_out_ = chunk_batch(
                self.geometry_,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material_.requires_normal,
            )
            rgb_fg_all_ = chunk_batch(
                self.material_,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out_
            )
            comp_rgb_bg_ = chunk_batch(
                self.background_, self.cfg.eval_chunk_size, dirs=rays_d_flatten
            )

        transmittance, alphas = nerfacc.render_transmittance_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        transmittance_, alphas_ = nerfacc.render_transmittance_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out_["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        weights: Float[Tensor, "Nr 1"]

        # gradient truncation of alphas_ will make guidance more focused, thus better performance
        alphas_comp = alphas + alphas_.clone().detach()
        alphas_comp = torch.clamp(alphas_comp, 0, 1)

        alphas_weight = (alphas / (alphas + alphas_.clone().detach() + 1e-10 * torch.ones_like(alphas))).unsqueeze(-1)
        ones = torch.ones_like(alphas_weight)
        rgbs = alphas_weight * rgb_fg_all + (ones - alphas_weight) * rgb_fg_all_.clone().detach()

        # implemention if no gradient truncation
        # alphas_comp = alphas + alphas_
        # alphas_comp = torch.clamp(alphas_comp, 0, 1)

        # alphas_weight = (alphas / (alphas + alphas_ + 1e-10 * torch.ones_like(alphas))).unsqueeze(-1)
        # ones = torch.ones_like(alphas_weight)
        # rgbs = alphas_weight * rgb_fg_all + (ones - alphas_weight) * rgb_fg_all_

        weights_, _ = nerfacc.render_weight_from_alpha(
            alphas_comp, 
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgbs, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        t_depth = depth[ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        if bg_color is None:
            if random.random() > 0.5:
                bg_color = comp_rgb_bg
            else:
                bg_color = comp_rgb_bg_
        else:
            if bg_color.shape == (batch_size, height, width, 3):
                bg_color = bg_color.reshape(-1, 3)

        # augmentation for background
        if random.random() > 0.5:
            comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)
        else:
            comp_rgb = comp_rgb_fg + (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "n_rays": n_rays,
                    "rgb_fg_all": rgb_fg_all,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    # values=geo_out["normal"],
                    values=alphas_weight * geo_out["normal"] + (ones - alphas_weight) * geo_out_["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                        "weights": weights,
                        "ray_indices": ray_indices,
                        "n_rays": n_rays,
                        "rgb_fg_all": rgb_fg_all,
                        "points": positions,
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.grid_prune:

            def occ_eval_fn(x):
                density = self.geometry.forward_density(x) + self.geometry_.forward_density(x)
                # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                return density * self.render_step_size

            if self.training and not on_load_weights:
                self.estimator.update_every_n_steps(
                    step=global_step, occ_eval_fn=occ_eval_fn
                )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()
