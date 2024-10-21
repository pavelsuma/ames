import math
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    '''

    def __init__(self, in_c, bn_eps=1e-5, bn_mom=0.1, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 1, 1)
        self.bn = nn.BatchNorm2d(in_c, eps=bn_eps, momentum=bn_mom)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_c, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20)  # use default setting.

        for conv in [self.conv1, self.conv2]:
            conv.apply(init_weights)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score
        '''
        x = self.conv1(x)
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, dim=1, eps=1e-15)

        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        return feature_map_norm, att_score

    def __repr__(self):
        return self.__class__.__name__

