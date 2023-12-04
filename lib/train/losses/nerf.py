import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg
import ipdb

class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def forward(self, batch):
        output = self.net(batch)

        scalar_stats = {}
        loss = 0
        # ipdb.set_trace()
        color_fine_loss = self.color_crit(output['rgb_fine'], batch['rgb'])
        scalar_stats.update({'color_fine_mse': color_fine_loss})
        color_coarse_loss = self.color_crit(output['rgb_coarse'], batch['rgb'])
        scalar_stats.update({'color_coarse_mse': color_coarse_loss})
        loss += color_fine_loss + color_coarse_loss

        psnr_fine = -10. * torch.log(color_fine_loss.detach()) / \
                torch.log(torch.Tensor([10.]).to(color_fine_loss.device))
        psnr_coarse = -10. * torch.log(color_coarse_loss.detach()) / \
                    torch.log(torch.Tensor([10.]).to(color_coarse_loss.device))
        scalar_stats.update({'psnr_fine': psnr_fine})
        scalar_stats.update({'psnr_coarse': psnr_coarse})
        scalar_stats.update({'total_loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
