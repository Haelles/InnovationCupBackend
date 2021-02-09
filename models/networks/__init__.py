
import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import *
import util.util as util

from models.networks.loss2 import StyleLoss


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename  # parsing + generator; multiscale + discriminator
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)  # StageI: class ParsingGenerator(BaseNetwork)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')  # trainI: parsing
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:  # 得到了生成器之后构造判别器
        netD_cls = find_network_using_name(opt.netD, 'discriminator')  # netD default='multiscale'
        parser = netD_cls.modify_commandline_options(parser, is_train)  # StageI: class MultiscaleDiscriminator(
        # BaseNetwork)

    return parser


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_G(opt, filename='generator'):
    netG_cls = find_network_using_name(opt.netG, filename)  # StageI得到class ParsingGenerator(BaseNetwork)
    return create_network(netG_cls, opt)


def define_D(opt, filename='discriminator'):
    netD_cls = find_network_using_name(opt.netD, filename)
    return create_network(netD_cls, opt)

