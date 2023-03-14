import numpy as np
import torch
from util import util

class Config:
    def __init__(self):
        self.feature_path = 'param.pth'

        self.crop_sz = 107  # 107

        self.lambda0 = 1e-4
        self.neg_lambda0 = 0.1
        self.padding = 2
        self.output_sigma_factor = 0.1
        self.interp_factor = 0.01

        self.num_scale = 5  # 5#3
        self.scale_step = 1.02  # 1.02#1.0275
        self.scale_factor = self.scale_step ** (np.arange(self.num_scale) - self.num_scale / 2)

        self.min_scale_factor = 0.2
        self.max_scale_factor = 5
        self.scale_penalty = 0.9925

        self.scale_penalties = self.scale_penalty ** (np.abs((np.arange(self.num_scale) - self.num_scale / 2)))

        self.net_input_size = [self.crop_sz, self.crop_sz]

        self.net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)

        output_sigma = self.crop_sz / (1 + self.padding) * self.output_sigma_factor
        y = util.gaussian_shaped_labels(output_sigma, self.net_input_size)

        self.yf = torch.rfft(torch.Tensor(y).view(1, 1, self.crop_sz, self.crop_sz).cuda(), signal_ndim=2)