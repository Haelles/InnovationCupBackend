
import random
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image

from skimage.feature import canny
from skimage.color import rgb2gray
from scipy.misc import imread, imresize
import torch

from util.util import get_random_color_img


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, edge_paths, color_paths, mask_paths, color_mask_paths = self.get_paths(opt)
        mu1 = len(image_paths) / len(mask_paths)  # 6106 / 9600
        mu2 = len(image_paths) / len(color_mask_paths)  # 6106 / 50000
        mask_paths = mask_paths * (int(mu1) + 1)  # 9600
        color_mask_paths = color_mask_paths * (int(mu2) + 1)  # 50000

        self.label_paths = label_paths[:opt.max_dataset_size]  # 2^63 - 1
        self.image_paths = image_paths[:opt.max_dataset_size]
        self.edge_paths = edge_paths[:opt.max_dataset_size]
        self.color_paths = color_paths[:opt.max_dataset_size]
        # 以上4个都是6106
        self.mask_paths = mask_paths[:opt.max_dataset_size]
        self.color_mask_paths = color_mask_paths[:opt.max_dataset_size]

        size = len(self.label_paths)  # 6106
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        mask_paths = []
        color_mask_paths = []
        color_paths = []
        edge_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, edge_paths, color_paths, mask_paths, color_mask_paths

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)  # 只进行toTensor
        label_tensor = transform_label(label) * 255.0

        # input image (real images)
        image_path = self.image_paths[index]
        img_numpy = imread(image_path)
        image = Image.fromarray(img_numpy)
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)  # 进行toTensor和normalize
        image_tensor = transform_image(image)

        # load mask
        h, w = img_numpy.shape[0:2]
        mask_onehot = self.load_mask(h, w, index)  # 得到320 * 512 load_mask里进行了resize和(mask > 0).astype(np.uint8)
        mask_arr = mask_onehot * 255
        mask = Image.fromarray(mask_arr).convert('RGB')
        mask_tensor = transform_image(mask)  # 进行toTensor和normalize

        # incompleted label
        original_label = imread(label_path)
        print("original_label.shape:" + str(original_label.shape))
        print("mask_onehot.shape:" + str(mask_onehot.shape))
        incompleted_label = original_label * (1 - mask_onehot)
        incompleted_label = Image.fromarray(incompleted_label)
        incompleted_label_tensor = transform_label(incompleted_label) * 255.0  # 只进行toTensor

        # incompleted image = 原图 * mask处理1 - mask处理2
        incompleted_image_tensor = image_tensor * (1 - mask_tensor) / 2.0 - (1 + mask_tensor) / 2.0

        # load edge
        # edge = self.load_edge(img_numpy) * 255.0
        edge_path = self.edge_paths[index]
        img_numpy = imread(edge_path)
        edge = Image.fromarray(img_numpy).convert('RGB')
        edge_tensor = transform_image(edge)  # 进行toTensor和normalize
        # mask_edge_tensor = edge_tensor * (1 - mask_tensor) / 2.0 + (1 + mask_tensor) / 2.0
        mask_edge_tensor = edge_tensor * (1 + mask_tensor) / 2.0 - (1 - mask_tensor) / 2.0

        # load noise
        # noise = self.load_noise(h, w) * 255.0
        # noise = Image.fromarray(noise).convert('RGB')
        # noise_tensor = transform_image(noise)
        noise_tensor = torch.randn_like(image_tensor)
        mask_noise_tensor = noise_tensor * (1 + mask_tensor) / 2.0 - (1 - mask_tensor) / 2.0

        # load color mask
        h, w = img_numpy.shape[0:2]
        color_mask = self.load_color_mask(h, w, index)
        color_mask = (color_mask == 0).astype(np.uint8) * 255
        color_mask = Image.fromarray(color_mask).convert('RGB')
        color_mask_tensor = transform_image(color_mask)  # 进行toTensor和normalize

        if self.opt.test_color:
            img_numpy = get_random_color_img(h, w)  # 随机生成颜色 320*512*3
        else:
            color_path = self.color_paths[index]  # 数据集中的color
            img_numpy = imread(color_path)
        color = Image.fromarray(img_numpy).convert('RGB')

        color_tensor_0 = transform_image(color)  # 进行toTensor和normalize
        color_tensor = color_tensor_0 * (1 + mask_tensor) / 2.0 - (1 - mask_tensor) / 2.0
        color_tensor = color_tensor * (1 + color_mask_tensor) / 2.0 - (1 - color_mask_tensor) / 2.0

        # incompleted label
        bg_mask_numpy = (original_label == 0).astype(np.uint8)
        part_mask_numpy = (mask_onehot * (1 - bg_mask_numpy) + bg_mask_numpy).astype(np.uint8)
        part_mask_numpy = 1 - part_mask_numpy   # partial conv on the non-mask region
        part_mask = Image.fromarray(part_mask_numpy)
        part_mask_tensor =  transform_label(part_mask)  * 255

        face_mask_numpy = (original_label == 13).astype(np.uint8)
        face_mask_img = Image.fromarray(face_mask_numpy)
        face_mask_tensor = transform_label(face_mask_img)  * 255
        mask_onehot_img = Image.fromarray(mask_onehot)
        mask_onehot_tensor = transform_label(mask_onehot_img)  * 255

        # face_mask = Image.fromarray(face_mask_numpy * 255).convert('RGB')
        # mask_onehot = Image.fromarray(mask_onehot * 255).convert('RGB')
        part_mask = Image.fromarray(part_mask_numpy * 255).convert('RGB')
        # face_RGB = transform_image(face_mask)
        # mask_onehot_RGB = transform_image(mask_onehot)
        part_RGB = transform_image(part_mask)


        ## for wihte mask on image
        original_im = imread(image_path)
        mask_onehot_tmp = np.expand_dims(mask_onehot, 2)
        incompleted_im_numpy = original_im * (1 - mask_onehot_tmp) + mask_onehot_tmp*255
        incompleted_im = Image.fromarray(incompleted_im_numpy).convert('RGB')
        incompleted_image_2_tensor = transform_image(incompleted_im)

        input_dict = {
                      'label': label_tensor,
                      'incompleted_label': incompleted_label_tensor,
                      'original_image': image_tensor,
                      'incompleted_image_2': incompleted_image_2_tensor,
                      'incompleted_image': incompleted_image_tensor,
                      'mask': mask_tensor[0:1, :, :],
                      'mask_edge': mask_edge_tensor[0:1, :, :],
                      'mask_noise': mask_noise_tensor[0:1, :, :],
                      'mask_color': color_tensor,
                      # 'color': color_tensor_0,
                      # 'bg_mask': bg_mask_tensor,
                      'part_mask': part_mask_tensor,
                      'face_mask': face_mask_tensor,
                      'mask_onehot': mask_onehot_tensor,
                      # 'face_RGB': face_RGB,
                      # 'mask_onehot_RGB': mask_onehot_RGB,
                      'part_RGB': part_RGB,
                      'img_path': image_path
                      }

        return input_dict

    def load_mask(self, h, w, index):
        if self.opt.phase == 'train':
            mask_index = random.randint(0, len(self.mask_paths) - 1)
            mask = imread(self.mask_paths[mask_index])
            mask = self.resize(mask, h, w)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = imread(self.mask_paths[index])
            mask = self.resize(mask, h, w, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8)
            return mask

        return mask

    def load_color_mask(self, h, w, index):
        if self.opt.phase == 'train':
            mask_index = random.randint(0, len(self.color_mask_paths) - 1)
            mask = imread(self.color_mask_paths[mask_index])
            mask = self.resize(mask, h, w)
            # mask = (mask > 0).astype(np.uint8) * 255  # threshold due to interpolation
            mask = (mask > 0).astype(np.uint8) # threshold due to interpolation
        else:
            mask = imread(self.color_mask_paths[index])
            mask = self.resize(mask, h, w, centerCrop=False)
            mask = rgb2gray(mask)
            # mask = (mask > 0).astype(np.uint8) * 255
            mask = (mask > 0).astype(np.uint8)
            return mask

        return mask

    def load_edge(self, img):
        img_gray = rgb2gray(img)
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        # mask = None if self.opt.phase == 'train' else (1 - mask / 255).astype(np.bool)
        mask = None
        # return canny(img_gray, sigma=0, mask=mask).astype(np.float)
        # return canny(img_gray, sigma=2, mask=mask).astype(np.uint8)
        return canny(img_gray, sigma=1, mask=mask).astype(np.uint8)  # 1 is the best

    def resize(self, img, height, width, centerCrop=True):
        h, w = img.shape[0:2]  # 对mask来说,尺寸为512*512，不需要centerCrop
        if centerCrop and h != w:
            # center crop
            side = np.minimum(h, w)  # 320
            j = (h - side) // 2  # 0
            i = (w - side) // 2  # 96
            img = img[j:j + side, i:i + side, ...]
        img = imresize(img, [height, width])

        return img

    def __len__(self):
        return self.dataset_size
