
import re
import importlib
import torch
import numpy as np
from PIL import Image
import os
import argparse
import dill as pickle
import random

import scipy.io as sio
CMAP = sio.loadmat('./util/colormap.mat')['colormap']
CMAP = (CMAP * 256).astype(np.uint8)

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here
def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow  # rowPadding == 3
    if rowPadding > 0:
        # (8, 512, 960, 3)
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):  # (0, 5, 4)
        # 注意在line43已经把填充补齐排面的图加到imgs里面了
        # imgs[j]: (512, 960, 3) 这里还是沿着width拼接多个图片
        # append的是(512, 3840, 3)
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)  # (1024, 3840, 3)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):  # 0 -> batch_size - 1
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()  # .dim() == 3
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def label_2_onehot_batch(b_parsing_tensor, parsing_label_nc=20):
    size = b_parsing_tensor.size()
    oneHot_size = (size[0], parsing_label_nc, size[2], size[3])
    b_parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    b_parsing_label = b_parsing_label.scatter_(1, b_parsing_tensor.long().cuda(), 1.0)

    return b_parsing_label


def label_2_onehot(b_parsing_tensor, parsing_label_nc=20):
    size = b_parsing_tensor.size()  # torch.Size([1, 512, 320])
    oneHot_size = (parsing_label_nc, size[1], size[2])
    b_parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    b_parsing_label = b_parsing_label.scatter_(0, b_parsing_tensor.long().cuda(), 1.0)

    return b_parsing_label

def parsing2im(parsing, imtype=np.uint8):
    parsing_numpy = parsing.detach().cpu().float().numpy()
    # trainI: parsing_numpy (20, 512, 960)
    # testI: (512, 320)
    image_index = np.argmax(parsing_numpy, axis=0)  # 可能性最大的作为类别
    parsing_numpy = np.zeros((image_index.shape[0], image_index.shape[1], 3))
    for h in range(image_index.shape[0]):
        for w in range(image_index.shape[1]):
            parsing_numpy[h, w, :] = CMAP[image_index[h, w]]

    return parsing_numpy.astype(imtype)

def parsing2im_batch(parsing_tensor, parsing_label_nc=20, tile=False):
    # transform each parsing_tensor in the batch
    images_np = []
    for b in range(parsing_tensor.size(0)):
        one_parsing = parsing_tensor[b]  # [512, 320]
        one_parsing_np = parsing2im(label_2_onehot(one_parsing, parsing_label_nc=parsing_label_nc))
        images_np.append(one_parsing_np.reshape(1, *one_parsing_np.shape))
    images_np = np.concatenate(images_np, axis=0)

    # images_tiled = tile_images(images_np)
    # return images_tiled
    if tile:
        images_tiled = tile_images(images_np)
        return images_tiled
    else:
        images_np = images_np[0]
        return images_np

def parsing2im_batch_by20chnl(parsing_tensor, tile=False):
    # transform each parsing_tensor in the batch
    images_np = []
    for b in range(parsing_tensor.size(0)):
        one_parsing = parsing_tensor[b]  # trainI: one_parsing (20, 512, 960)
        one_parsing_np = parsing2im(one_parsing)  # (512, 960, 3)
        images_np.append(one_parsing_np.reshape(1, *one_parsing_np.shape))
    images_np = np.concatenate(images_np, axis=0)  # (5, 512, 960, 3)

    # images_tiled = tile_images(images_np)
    # return images_tiled
    if tile:  # tile = self.opt.batchSize > 1
        images_tiled = tile_images(images_np)
        return images_tiled  # (1024, 3840, 3)
    else:
        images_np = images_np[0]
        return images_np



def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    # image_pil.save(image_path.replace('.jpg', '.png'))
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    print(target_cls_name)
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if N == 182:  # COCO
            important_colors = {
                'sea': (54, 62, 167),
                'sky-other': (95, 219, 255),
                'tree': (140, 104, 47),
                'clouds': (170, 170, 170),
                'grass': (29, 195, 49)
            }
            for i in range(N):
                name = util.coco.id2label(i)
                if name in important_colors:
                    color = important_colors[name]
                    cmap[i] = np.array(list(color))

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def get_random_color_img(height, width):
    color = lambda : [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    c = color()
    img = np.zeros([height, width,3],dtype=np.uint8)
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            img[i, j] = c

    return img

def parsing_2_onechannel(parsing, imtype=np.uint8):
    parsing_numpy = parsing.cpu().float().numpy()  # torch.Size([20, 512, 320])
    image_index = np.argmax(parsing_numpy, axis=0)
    parsing_numpy = np.zeros((image_index.shape[0], image_index.shape[1], 3))
    for h in range(image_index.shape[0]):
        for w in range(image_index.shape[1]):
            parsing_numpy[h, w, :] = image_index[h, w]

    return parsing_numpy.astype(imtype)[:, :, 0:1]
