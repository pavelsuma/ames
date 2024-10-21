import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BinarizationLayer(nn.Module):

    def __init__(self, file_name=None, dims=None, bits=None, sigma=1e-6, pretrained=False, trainable=False):
        super(BinarizationLayer, self).__init__()
        self.bits = bits
        self.dims = dims
        self.sigma = sigma
        self.trainable = trainable
        self.W = None
        if file_name is not None:
            self.load(file_name)
        elif pretrained:
            pretrained_url = 'https://mever.iti.gr/distill-and-select/models/itq_resnet50W_dns100k_1M.pth'
            weights = torch.hub.load_state_dict_from_url(pretrained_url)
            self.init_params(weights['proj'])
        elif dims is not None:
            self.bits = bits if bits is not None else dims
            self.init_params()

    def save(self, file_name):
        np.savez_compressed(file_name, proj=self.W.detach().cpu().numpy())

    def load(self, file_name):
        white = np.load(file_name)
        proj = torch.from_numpy(white['proj']).float()
        self.init_params(proj)

    def init_params(self, proj=None):
        if proj is None:
            proj = torch.randn(self.dims, self.bits)
        self.W = nn.Parameter(proj, requires_grad=self.trainable)
        self.dims, self.bits = self.W.shape

    @staticmethod
    def _itq_rotation(v, n_iter, bit):
        r = np.random.randn(bit, bit)
        u11, s2, v2 = np.linalg.svd(r)

        r = u11[:, :bit]

        for _ in range(n_iter):
            z = np.dot(v, r)
            ux = np.ones(z.shape) * (-1.)
            ux[z >= 0] = 1
            c = np.dot(ux.transpose(), v)
            ub, sigma, ua = np.linalg.svd(c)
            r = np.dot(ua, ub.transpose())
        z = np.dot(v, r)
        b = np.ones(z.shape) * -1.
        b[z >= 0] = 1
        return b, r

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        x = torch.matmul(x, self.W)
        if self.training and self.trainable:
            x = torch.erf(x / np.sqrt(2 * self.sigma))
        else:
            x = torch.sign(x)
        return x

    def __repr__(self, ):
        return '{}(dims={}, bits={})'.format(self.__class__.__name__, self.W.shape[0], self.W.shape[1])