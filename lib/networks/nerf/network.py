import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg
import ipdb
from .nerf import NeRF

class Network(nn.Module):
    def __init__(self, ):
        super(Network, self).__init__()
        self.coarse_model = NeRF()
        self.is_fine = False
        self.cascade_samples = cfg.task_arg.cascade_samples

        if len(self.cascade_samples) == 2:
            self.is_fine = True
        self.fine_model = NeRF() if self.is_fine else None
        self.model_dict = {
            'coarse': self.coarse_model,
            'fine': self.fine_model
        }   ## 抱歉最近又去回顾了一下论文，发现是用两个network进行训练的，所以把nerf拆出来单独作为一个类
            ## we simultaneously optimize two networks: one "coarse" and one "fine".
            ## 所以对其进行了一些修改

    def nerf_forward(self, r, dir, type):
        model = self.model_dict[type]
        sigma, color = model(r, dir)
        return sigma, color

    def volume_rendering(self, rays_o, rays_d, t, sigma, color, sample):

        sigma = sigma.reshape(-1, sample)
        color = color.reshape(-1, sample, 3)
        # ipdb.set_trace()
        delta = torch.cat([t[..., 1:] - t[..., :-1],
                           (torch.broadcast_to(torch.tensor(1e10), t[..., 0].shape).to(rays_o.device)).unsqueeze(1)],
                          -1)  ## N N_sample
        ## consider delta * |rays_d|
        delta = delta * torch.norm(rays_d[..., None, :], dim=-1)  ###?
        alpha = 1. - torch.exp(-sigma * delta)
        Ti = torch.cumprod(1. - alpha + 1e-10, -1)
        Ti = torch.roll(Ti, 1, -1)
        Ti[..., 0] = 1.
        Ti_alpha = alpha * Ti  ##
        rgb = torch.sum(Ti_alpha[..., None] * color, -2)
        depth = torch.sum(Ti_alpha * t, -1)
        acc = torch.sum(Ti_alpha, -1)
        return rgb, depth, acc, Ti_alpha

    def render(self, rays_o, rays_d, batch):

        near, far = batch['near'], batch['far']
        # ipdb.set_trace()
        coarse_sample = self.cascade_samples[0]
        if self.is_fine:
            fine_sample = self.cascade_samples[1]

        t = torch.linspace(0, 1, coarse_sample, device=rays_o.device)
        t = near.item() * (1 - t) + far.item() * t
        shape = list(rays_o.shape[:-1]) + [coarse_sample]
        t = t.expand(shape)

        # if self.perturb > 0:
        #     t_mid = 0.5 * (t[:, :-1] + t[:, 1:])
        #     upper = torch.cat([t_mid, t[:, -1:]], -1)
        #     lower = torch.cat([t[:, :1], t_mid], -1)
        #
        #     perturb_rand = self.perturb * torch.rand(t.shape, device=rays_o.device)
        #     t = lower + (upper - lower) * perturb_rand

        r = rays_o[..., None, :] + rays_d[..., None, :] * t[..., :, None]  ## N 1 3, N 1 3, N sample 1 -> broadcast
        dir = rays_d.unsqueeze(1)
        dir = dir.expand(-1, coarse_sample, -1)
        dir = dir.reshape(-1, 3)
        r = r.reshape(-1, 3)  ## x, y, z

        sigma, color = self.nerf_forward(r, dir, type = 'coarse')

        rgb_coarse, depth_coarse, acc_coarse, Ti_alpha = self.volume_rendering(rays_o, rays_d, t, sigma, color, coarse_sample)

        ### fine sample
        if self.is_fine:
            t_mid = .5 * (t[..., 1:] - t[..., :-1])
            t_sample = self.sample_pdf(t_mid, Ti_alpha[..., 1:-1], fine_sample).detach()
            t, _ = torch.sort(torch.cat([t, t_sample], -1), -1)

            r = rays_o[..., None, :] + rays_d[..., None, :] * t[..., :, None]
            dir = rays_d.unsqueeze(1)
            dir = dir.expand(-1, coarse_sample + fine_sample, -1)
            dir = dir.reshape(-1, 3)
            r = r.reshape(-1, 3)

            sigma, color = self.nerf_forward(r, dir, type = 'fine')

            rgb_fine, depth_fine, acc_fine, _ = self.volume_rendering(rays_o, rays_d, t, sigma, color,
                                                                                   coarse_sample + fine_sample)

        ret = {'rgb_coarse': rgb_coarse, 'depth_coarse': depth_coarse, 'acc_coarse': acc_coarse}
        if self.is_fine:
            ret.update({'rgb_fine': rgb_fine, 'depth_fine': depth_fine, 'acc_fine': acc_fine})

        return ret

    def batchify(self, rays_o, rays_d, batch):
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        # ipdb.set_trace()
        for i in range(0, rays_o.shape[0], chunk):
            ret = self.render(rays_o[i:i + chunk], rays_d[i:i + chunk], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret

    def forward(self, batch):

        B, N_rays, C = batch['rays_o'].shape
        ret = self.batchify(batch['rays_o'].reshape(-1, C), batch['rays_d'].reshape(-1, C), batch)
        return {k: ret[k].reshape(B, N_rays, -1) for k in ret}

    def sample_pdf(self, bins, weights, N_samples):  ## 这段代码待需理解
        # pass
        eps = 1e-5
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive


        # u = torch.rand(N_rays, N_samples, device=bins.device)
        u = torch.linspace(0, 1, N_samples, device=bins.device)
        u = u.expand(N_rays, N_samples)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, side='right')
        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_samples)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_samples, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_samples, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])

        return samples