import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.norm import get_norm_layer, ANLs
from models.networks.partialconv2d import PartialConv2d

"""
StageI: Free-form Parsing Network
"""


class ParsingGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='batch')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.input_nc  # script: 26
        norm_layer = get_norm_layer(opt, opt.norm_G)  # script: batch
        activation = nn.ReLU(False)

        model = []
        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


"""
StageII: Parsing-aware Inpainting Network
"""


class MultiAtt3Generator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super(MultiAtt3Generator, self).__init__()
        self.opt = opt
        ngf = opt.ngf
        pconv_input_nc = 3
        parsing_input_nc = 20
        output_nc = opt.output_nc
        block_num = 8
        self.down1 = DownBlock(pconv_input_nc, ngf, kernel_size=7, padding=0, return_mask=True, multi_channel=True)
        self.down2 = DownBlock(ngf, 2 * ngf, kernel_size=4, stride=2, padding=1, return_mask=True, multi_channel=True)
        self.down3 = DownBlock(2 * ngf, 4 * ngf, kernel_size=4, stride=2, padding=1, return_mask=True,
                               multi_channel=True)

        self.parsing_encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=parsing_input_nc, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

        )

        blocks = []
        for _ in range(block_num):
            block = MultiAttResnetBlock(256 * 2, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.tmp_block = MultiAttResnetBlock2(dim_in=256 * 2, dim_out=256)

        self.up2 = UpBlock(4 * ngf, 2 * ngf, kernel_size=4, stride=2, padding=1)
        self.up3 = UpBlock(2 * ngf, ngf, kernel_size=4, stride=2, padding=1)
        self.up4 = UpBlock(ngf, output_nc, kernel_size=7, padding=0)

        self.up_1 = ANLs(4 * ngf, opt.input_ANLs_nc)
        self.up_2 = ANLs(2 * ngf, opt.input_ANLs_nc)
        self.up_3 = ANLs(1 * ngf, opt.input_ANLs_nc)

    def forward(self, x):
        # print("generator x.shape:")  torch.Size([1, 29, 512, 320])
        # print(x.shape)
        parsing = x[:, :20, :, :]
        incompleted_image = x[:, 20:23, :, :]
        part_mask = x[:, 23:24, :, :]
        edge_color = x[:, 24:, :, :]  # including noise

        mask = torch.cat([part_mask for _ in range(3)], dim=1)

        x1, mask1 = self.down1(incompleted_image, mask)
        x2, mask2 = self.down2(x1, mask1)
        x3, mask3 = self.down3(x2, mask2)

        en_parsing = self.parsing_encoder(parsing)
        x4_parsing = torch.cat([x3, en_parsing], dim=1)

        xm = self.middle(x4_parsing)
        xm = self.tmp_block(xm)

        u1 = self.up_1(xm, edge_color)
        u2 = self.up2(u1)
        u2 = self.up_2(u2, edge_color)
        u3 = self.up3(u2)
        u3 = self.up_3(u3, edge_color)
        u4 = self.up4(u3)

        return u4


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class MultiAttResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(MultiAttResnetBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        return out


class MultiAttResnetBlock2(nn.Module):
    def __init__(self, dim_in, dim_out, dilation=1, use_spectral_norm=False):
        super(MultiAttResnetBlock2, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(
                nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, padding=0, dilation=dilation,
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim_out, track_running_stats=False)
        )

    def forward(self, x):
        out = self.conv_block(x)

        return out


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, return_mask=True,
                 multi_channel=True):
        super(DownBlock, self).__init__()
        self.kernel_size = kernel_size
        self.ReflectionPad2d = nn.ReflectionPad2d(3)
        self.conv = PartialConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=False, return_mask=return_mask, multi_channel=multi_channel)
        self.InstanceNorm2d = nn.InstanceNorm2d(out_planes, track_running_stats=False)

    def forward(self, x, mask):
        if self.kernel_size == 7:
            x = self.ReflectionPad2d(x)
            mask = self.ReflectionPad2d(mask)

        x, mask = self.conv(x, mask)
        x = self.InstanceNorm2d(x)
        output = F.relu(x)

        return output, mask


class UpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0):
        super(UpBlock, self).__init__()
        self.kernel_size = kernel_size
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                               stride=stride, padding=padding),
            nn.InstanceNorm2d(out_planes, track_running_stats=False),
            nn.ReLU(True),
        )
        self.last_up = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_planes, out_channels=3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        if self.kernel_size == 7:
            output = self.last_up(x)
        else:
            output = self.up(x)

        return output


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out
