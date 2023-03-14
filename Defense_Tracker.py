from __future__ import absolute_import, division, print_function

import torch
import numpy as np
from tracker import Tracker
from Tracker_Network import DCFNet
from util.config import Config
from util import util

__all__ = ['Defense_Tracker']

class Defense_Tracker(Tracker):
    def __init__(self, net_path1=None, net_path2=None):
        super(Defense_Tracker, self).__init__('Network', True)

        self.config = Config()

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        cos = np.outer(np.hanning(self.config.crop_sz), np.hanning(self.config.crop_sz))
        self.cos_window = torch.Tensor(cos).to(self.device)

        self.net = DCFNet(self.config, net_path1=net_path1, net_path2=net_path2, cos_window=self.cos_window)

        self.net = self.net.to(self.device)



    @torch.no_grad()
    def init(self, img, box):
        self.box = box

        self.target_pos, self.target_sz = util.rect1_2_cxy_wh(box)
        self.target_sz_gt = self.target_sz

        self.min_sz = np.maximum(self.config.min_scale_factor * self.target_sz, 4)
        self.max_sz = np.minimum(img.shape[:2], self.config.max_scale_factor * self.target_sz)

        self.window_sz = self.target_sz * (1 + self.config.padding)

        self.bbox = util.cxy_wh_2_bbox(self.target_pos, self.window_sz)
        self.patch = util.crop_chw(img, self.bbox, self.config.crop_sz)
        self.target = self.patch - self.config.net_average_image

        self.net.DCF_update(torch.Tensor(np.expand_dims(self.target, axis=0)).to(self.device))

        self.patch_crop = np.zeros(
            (self.config.num_scale, self.patch.shape[0], self.patch.shape[1], self.patch.shape[2]), np.float32)
        self.patch_crop_valid = np.zeros(
            (self.config.num_scale, self.patch.shape[0], self.patch.shape[1], self.patch.shape[2]), np.float32)

    @torch.no_grad()
    def update(self, img):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            self.window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            self.bbox = util.cxy_wh_2_bbox(self.target_pos, self.window_sz)
            self.window_sz_valid = (self.target_sz + np.array([8, 8]) * self.config.scale_factor[i])
            self.bbox_valid = util.cxy_wh_2_bbox(self.target_pos, self.window_sz_valid)
            self.patch_crop_valid[i, :] = util.crop_chw(img, self.bbox_valid, self.config.crop_sz)
            self.patch_crop[i, :] = util.crop_chw(img, self.bbox, self.config.crop_sz)

        self.search = self.patch_crop - self.config.net_average_image
        self.search_valid = self.patch_crop_valid - self.config.net_average_image

        self.response = self.net(torch.Tensor(self.search).to(self.device))

        self.peak, self.idx = torch.max(self.response.view(self.config.num_scale, -1), 1)

        self.peak = self.peak.data.cpu().numpy() * self.config.scale_penalties
        self.idx = self.idx.data.cpu().numpy()
        self.best_scale = np.argmax(self.peak)

        self.r_max, self.c_max = np.unravel_index(self.idx[self.best_scale], self.config.net_input_size)

        if self.r_max > self.config.net_input_size[0] / 2:
            self.r_max = self.r_max - self.config.net_input_size[0]
        if self.c_max > self.config.net_input_size[1] / 2:
            self.c_max = self.c_max - self.config.net_input_size[1]
        self.window_sz = self.target_sz * (self.config.scale_factor[self.best_scale] * (1 + self.config.padding))

        self.target_pos = self.target_pos + np.array(
            [self.c_max, self.r_max]) * self.window_sz / self.config.net_input_size
        self.target_sz = np.minimum(np.maximum(self.window_sz / (1 + self.config.padding), self.min_sz), self.max_sz)

        self.window_sz = self.target_sz * (1 + self.config.padding)
        self.bbox = util.cxy_wh_2_bbox(self.target_pos, self.window_sz)
        self.patch = util.crop_chw(img, self.bbox, self.config.crop_sz)
        self.target = self.patch - self.config.net_average_image

        self.net.DCF_update(torch.Tensor(np.expand_dims(self.target, axis=0)).to(self.device), lr=self.config.interp_factor)

        self.total_pos = np.array(
            [self.target_pos[0] - self.target_sz[0] / 2, self.target_pos[1] - self.target_sz[1] / 2, self.target_sz[0],
             self.target_sz[1]])
        self.box = self.total_pos

        return self.box