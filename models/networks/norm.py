
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type = norm_type

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer




class ANLs(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(norm_nc, affine=False)
        self.conv = spectral_norm(nn.Conv2d(norm_nc, norm_nc, kernel_size=3, padding=1))

        self.shared = nn.Sequential(nn.Conv2d(label_nc, 128, kernel_size=3, padding=1), nn.ReLU())
        self.gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def norm(self, x, parsing):
        bn = self.batch_norm(x)

        parsing = F.interpolate(parsing, size=x.size()[2:], mode='nearest')
        tmp_feature = self.shared(parsing)
        gamma = self.gamma(tmp_feature)
        beta = self.beta(tmp_feature)

        att_map = torch.sigmoid(gamma)            # attention map
        out = bn * (1 + att_map) + beta

        return out


    def forward(self, x, segmap):
        x_tmp = x

        # step 1
        norm = self.norm(x, segmap)
        # step 2
        act = self.actvn(norm)
        dx = self.conv(act)

        out = x_tmp + dx

        return out


