import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg
import ipdb

class NeRF(nn.Module):
    def __init__(self, ):
        super(NeRF, self).__init__()
        net_cfg = cfg.network
        self.xyz_encoder, input_xyz = get_encoder(net_cfg.xyz_encoder)
        self.dir_encoder, input_dir = get_encoder(net_cfg.dir_encoder)

        D, W = net_cfg.nerf.D, net_cfg.nerf.W
        self.skips = 5
        self.backbone_layer1 = nn.ModuleList(
            [nn.Linear(input_xyz, W)] + [nn.Linear(W, W) for i in range(self.skips - 1)]
        )
        self.backbone_layer2 = nn.ModuleList(
            [nn.Linear(input_xyz + W, W)] + [nn.Linear(W, W) for i in range(D - self.skips)]
        )

        self.output_sigma = nn.Linear(W, 1)
        self.output_feature = nn.Linear(W, W)
        self.output_color = nn.Sequential(
            nn.Linear(W + input_dir, W // 2),
            nn.ReLU(),
            nn.Linear(W // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, r, dir):

        xyz_encoding = self.xyz_encoder(r)
        dir_encoding = self.dir_encoder(dir)
        x = xyz_encoding
        for i, l in enumerate(self.backbone_layer1):
            # ipdb.set_trace()
            x = self.backbone_layer1[i](x)
            x = F.relu(x)

        x = torch.cat([xyz_encoding, x], dim=-1)

        for i, l in enumerate(self.backbone_layer2):
            x = self.backbone_layer2[i](x)
            x = F.relu(x)
        # print(self.output_sigma(x))
        sigma = F.relu(self.output_sigma(x))
        # sigma = self.output_sigma(x)
        x = self.output_feature(x)
        x = torch.cat([dir_encoding, x], dim=-1)
        color = self.output_color(x)

        return sigma, color