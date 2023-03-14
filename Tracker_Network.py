import torch.nn as nn
import torch

from torch.functional import F

def LocalResponseNorm(input, size, alpha, beta, k):
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)

    if dim == 3:
        div = F.pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = F.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = F.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = F.avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)
    return input / div

def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.Shared_Encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(13*13*64, 512)  ## 13
        self.fc2 = nn.Linear(13*13*64, 512)
        self.fc3 = nn.Linear(512, 13*13*64)
        self.relu = nn.ReLU()

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def encode(self, x):
        x = self.Shared_Encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.Decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.relu(self.fc3(z))
        z = z.view(z.size(0), 64, 13, 13)
        decoded = self.decode(z)
        return F.interpolate(decoded, size=[107, 107]), mu, logvar


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.85, k=1), #0.85 #DCF=0.75 --> sequential에서 제거
        )

    def forward(self, x):
        x = self.feature(x)
        x = LocalResponseNorm(input=x, size=5, alpha=0.0001, beta=0.85, k=1) ## 함수로 추가
        return x


class DCFNet(nn.Module):
    def __init__(self, config=None, net_path1=None, net_path2=None, cos_window=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.recon_feature = VAE()

        if net_path1 is not None:
            self.feature.load_state_dict(torch.load(net_path1, map_location=lambda storage, loc: storage))

        if net_path2 is not None:
            self.recon_feature.load_state_dict(torch.load(net_path2, map_location=lambda storage, loc: storage))

        self.model_alphaf = []
        self.model_zf = []
        self.model_zf_n = []

        self.config = config

        self.cos_window = cos_window

        self.init_feat = []

    def forward(self, x):
        ################ DCF #################################
        feat = self.feature(x)

        xp = feat * self.cos_window
        xf = torch.rfft(xp, signal_ndim=2)

        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response = torch.irfft(complex_mul(kxzf, self.model_alphaf), signal_ndim=2)

        return response

    def DCF_update(self, z, lr=1.):

        feat = self.feature(z)

        feat = feat * self.cos_window

        feat_a = torch.cat((feat, feat, feat, feat, feat, feat, feat, feat), 0)  ## original 8 ## 3and /2 is better
        feat2 = self.recon_feature(feat_a)
        feat2 = torch.sum(feat2[0], dim=0, keepdim=True)

        zp = (feat) + (feat2/2)
        zf = torch.rfft(zp, signal_ndim=2)

        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0)

        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:

            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data





