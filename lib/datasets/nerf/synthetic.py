import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
import ipdb

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_rays

        # read image
        imgs = []
        c2ws = []

        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(split))))

        for frame in json_info['frames']:
            image_path = os.path.join(self.data_root, frame['file_path'][2:] + '.png')
            imgs.append(imageio.imread(image_path)/255.)
            c2ws.append(frame['transform_matrix'])

        # ipdb.set_trace()

        imgs = np.array(imgs).astype(np.float32)
        c2ws = np.array(c2ws).astype(np.float32)

        H, W = imgs[0].shape[:2]
        camera_angle_x = json_info['camera_angle_x']
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        if cfg.task_arg.white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])

        if self.input_ratio != 1.:
            imgs_temp = []
            for img in imgs:
                imgs_temp.append(cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA))
            focal /= 2.
            H = H // 2
            W = W // 2
            imgs = np.array(imgs_temp).astype(np.float32)

        self.imgs = imgs # N H W 3
        rays_o, rays_d = [], []

        for c2w in c2ws:
            ray_o, ray_d = self.get_rays(H, W, focal, c2w)
            rays_o.append(ray_o)
            rays_d.append(ray_d)

        rays_o = np.array(rays_o).astype(np.float32) ## N H W 3
        rays_d = np.array(rays_d).astype(np.float32) ## N H W 3

        self.near = 2.
        self.far = 6.

        # rays_o, rays_d = self.get_ndc_rays(H, W, focal, rays_o, rays_d) ## 如果是处理街景深度场景时使用
        self.rays_o = rays_o
        self.rays_d = rays_d
        # ipdb.set_trace()



        # if self.split != 'train':
        #     self.rays_o = np.expand_dims(self.rays_o, axis=0)
        #     self.rays_d = np.expand_dims(self.rays_d, axis=0)

        # set image
        # self.img = np.array(img).astype(np.float32)
        # set uv
        # H, W = img.shape[:2]
        # X, Y = np.meshgrid(np.arange(W), np.arange(H))
        # u, v = X.astype(np.float32) / (W-1), Y.astype(np.float32) / (H-1)
        # self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)

    def __getitem__(self, index):
        if self.split == 'train':
            ids1 = np.random.choice(len(self.rays_o))
            ## 也可以按顺序训练 不采用random choice 就直接用index
            rays_o = self.rays_o[ids1].reshape(-1, 3)
            rays_d = self.rays_d[ids1].reshape(-1, 3)
            ids2 = np.random.choice(len(rays_o), self.batch_size, replace=False)
            rays_o = rays_o[ids2]
            rays_d = rays_d[ids2]
            rgb = self.imgs[ids1].reshape(-1, 3)[ids2]

        else:
            rays_o = self.rays_o[index].reshape(-1, 3)
            rays_d = self.rays_d[index].reshape(-1, 3)
            rgb = self.imgs[index].reshape(-1, 3)

        ret = {'rays_o': rays_o, 'rays_d': rays_d, 'rgb': rgb} # input and output. they will be sent to cuda
        ret.update({'meta': {'H': self.imgs[0].shape[0], 'W': self.imgs[0].shape[1]}}) # meta means no need to send to cuda
        ret.update({'near': self.near, 'far': self.far})

        return ret

    def __len__(self):

        return len(self.rays_o)

    def get_rays(self, H, W, focal,c2w):
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        X, Y = X.astype(np.float32), Y.astype(np.float32)
        dirs = np.stack([(X-W*.5)/focal, -(Y-H*.5)/focal, -np.ones_like(X)], -1) ## 获得相机平面坐标(..)
        # ipdb.set_trace()
        ray_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) ##
        ray_d = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True) ## 归一化
        ray_o = np.broadcast_to(c2w[:3, -1], np.shape(ray_d))
        ## 用np.tile也可实现
        return ray_o, ray_d

    def get_ndc_rays(self, H, W, focal, rays_o, rays_d):
        # ipdb.set_trace()
        t = -(self.near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d ## shift to -n(z) plane
        ## NeRF论文附录有公式推导，也可以参考一下 https://zhuanlan.zhihu.com/p/628675070 讲的很好
        ax = -1./W/(2.*focal)
        ay = -1./H/(2.*focal)

        ox_oz = rays_o[..., 0] / rays_o[..., 2]
        oy_oz = rays_o[..., 1] / rays_o[..., 2]

        dx_dz = rays_d[..., 0] / rays_d[..., 2]
        dy_dz = rays_d[..., 1] / rays_d[..., 2]

        ox = ax * ox_oz
        oy = ay * oy_oz
        oz = 1. + 2. * self.near / rays_o[..., 2]

        dx = ax * (dx_dz - ox_oz)
        dy = ay * (dy_dz - oy_oz)
        dz = 1. - oz

        rays_o = np.stack([ox, oy, oz], -1)
        rays_d = np.stack([dx, dy, dz], -1)

        return rays_o, rays_d
