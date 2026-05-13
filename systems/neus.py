import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, SSIM, LPIPS, binary_cross_entropy


@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'ssim': SSIM(),
            'lpips': LPIPS(),
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        self.ema_variance = None   # lazy-initialized on first training batch

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            # --- lazy EMA buffer init (needs dataset dims, available here) ---
            if self.ema_variance is None:
                N_img = len(self.dataset.all_images)
                # start at near-zero variance → near-max confidence at activation
                self.ema_variance = torch.full(
                    (N_img, self.dataset.h, self.dataset.w), 1e-4,
                    device=self.rank
                )

            conf_cfg = self.config.model.get('conf', None)
            _use_ray_sampling = (
                conf_cfg is not None
                and conf_cfg.get('enabled', False)
                and conf_cfg.get('use_ray_sampling', False)
                and self.model.conf_active
            )

            if _use_ray_sampling:
                # two-stage sampling to stay under torch.multinomial's 2^24 limit:
                # stage 1 — sample images proportional to mean per-image confidence
                img_conf = (1.0 / (self.ema_variance + 1e-4)).mean(dim=[1, 2])  # (N_img,)
                img_probs = img_conf / img_conf.sum()
                index = torch.multinomial(img_probs, self.train_num_rays, replacement=True)
                # stage 2 — sample pixels uniformly within each selected image
                x = torch.randint(0, self.dataset.w, size=(self.train_num_rays,), device=self.rank)
                y = torch.randint(0, self.dataset.h, size=(self.train_num_rays,), device=self.rank)
            else:
                x = torch.randint(
                    0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
                )
                y = torch.randint(
                    0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
                )

            # expand scalar image index to per-ray for EMA indexing
            ray_idx = index.expand(self.train_num_rays) if index.shape[0] == 1 else index

            c2w = self.dataset.all_c2w[index]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })
        if stage in ['train']:
            batch.update({
                'ray_idx': ray_idx,
                'ray_x': x,
                'ray_y': y,
            })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        conf_cfg = self.config.model.get('conf', None)
        _conf_on = (
            conf_cfg is not None
            and conf_cfg.get('enabled', False)
            and self.model.conf_active
            and 'eikonal_points' in out
        )

        # --- EMA variance update (detached, no grad) ---
        if _conf_on:
            with torch.no_grad():
                residuals = (out['comp_rgb_full'] - batch['rgb']).abs().mean(-1)
                alpha = conf_cfg.get('ema_alpha', 0.99)
                idx, ry, rx = batch['ray_idx'], batch['ray_y'], batch['ray_x']
                self.ema_variance[idx, ry, rx] = (
                    alpha * self.ema_variance[idx, ry, rx] + (1.0 - alpha) * residuals
                )

        # --- eikonal loss (importance-sampled when enabled) ---
        if _conf_on and conf_cfg.get('use_eikonal_importance', True):
            with torch.no_grad():
                conf = self.model.get_confidence(out['eikonal_points'])
                temp = conf_cfg.get('sample_temperature', 3.0)
                probs = conf.pow(1.0 / temp)
                probs = probs / (probs.sum() + 1e-8)
                ratio = conf_cfg.get('eikonal_sample_ratio', 0.5)
                n_sub = max(1, int(len(conf) * ratio))
                eik_idx = torch.multinomial(probs, n_sub, replacement=False)
            grad_sub = out['sdf_grad_samples'][eik_idx]
            loss_eikonal = ((torch.linalg.norm(grad_sub, ord=2, dim=-1) - 1.)**2).mean()
        else:
            loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()

        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        # --- confidence supervision loss Lconf ---
        if _conf_on:
            conf_pred = self.model.get_confidence(out['eikonal_points'])
            with torch.no_grad():
                # map per-ray EMA variance to per-sample via ray_indices
                ray_var = self.ema_variance[batch['ray_idx'], batch['ray_y'], batch['ray_x']]
                sample_var = ray_var[out['ray_indices']]
                target = (1.0 / (sample_var + 1e-4))
                target = (target / (target.max() + 1e-8)).clamp(0.0, 1.0)
            loss_conf = F.mse_loss(conf_pred, target)
            self.log('train/loss_conf', loss_conf)
            loss += loss_conf * self.C(self.config.system.loss.lambda_conf)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        W, H = self.dataset.img_wh
        rgb_pred = out['comp_rgb_full'].to(batch['rgb'])
        psnr = self.criterions['psnr'](rgb_pred, batch['rgb'])
        # reshape to (1, 3, H, W) for SSIM and LPIPS
        pred_img = rgb_pred.view(H, W, 3).permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        gt_img   = batch['rgb'].view(H, W, 3).permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        ssim  = self.criterions['ssim'](pred_img, gt_img)
        lpips = self.criterions['lpips'](pred_img, gt_img)
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips,
            'index': batch['index']
        }


    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {
                        'psnr': step_out['psnr'],
                        'ssim': step_out['ssim'],
                        'lpips': step_out['lpips'],
                    }
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {
                            'psnr': step_out['psnr'][oi],
                            'ssim': step_out['ssim'][oi],
                            'lpips': step_out['lpips'][oi],
                        }
            psnr  = torch.mean(torch.stack([o['psnr']  for o in out_set.values()]))
            ssim  = torch.mean(torch.stack([o['ssim']  for o in out_set.values()]))
            lpips = torch.mean(torch.stack([o['lpips'] for o in out_set.values()]))
            self.log('val/psnr',  psnr,  prog_bar=True, rank_zero_only=True)
            self.log('val/ssim',  ssim,  prog_bar=True, rank_zero_only=True)
            self.log('val/lpips', lpips, prog_bar=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        out = self(batch)
        W, H = self.dataset.img_wh
        rgb_pred = out['comp_rgb_full'].to(batch['rgb'])
        psnr = self.criterions['psnr'](rgb_pred, batch['rgb'])
        pred_img = rgb_pred.view(H, W, 3).permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        gt_img   = batch['rgb'].view(H, W, 3).permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        ssim  = self.criterions['ssim'](pred_img, gt_img)
        lpips = self.criterions['lpips'](pred_img, gt_img)
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips,
            'index': batch['index']
        }

    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {
                        'psnr': step_out['psnr'],
                        'ssim': step_out['ssim'],
                        'lpips': step_out['lpips'],
                    }
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {
                            'psnr': step_out['psnr'][oi],
                            'ssim': step_out['ssim'][oi],
                            'lpips': step_out['lpips'][oi],
                        }
            psnr  = torch.mean(torch.stack([o['psnr']  for o in out_set.values()]))
            ssim  = torch.mean(torch.stack([o['ssim']  for o in out_set.values()]))
            lpips = torch.mean(torch.stack([o['lpips'] for o in out_set.values()]))
            self.log('test/psnr',  psnr,  prog_bar=True, rank_zero_only=True)
            self.log('test/ssim',  ssim,  prog_bar=True, rank_zero_only=True)
            self.log('test/lpips', lpips, prog_bar=True, rank_zero_only=True)

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            
            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )        
