import base64
import io
import json
import os
import random

import flask
import torch
from PIL import Image
from PyQt5.QtGui import QImage
from flask import send_from_directory, Flask, request
from imageio import imread
import numpy as np
import cv2
from skimage.color import rgb2gray
from torchvision.transforms import transforms
from scipy.misc import imresize
from werkzeug.utils import secure_filename

from data.base_dataset import get_params, get_transform
from models.stageII_multiatt3_model import StageII_MultiAtt3_Model
from models.stageI_parsing_model import StageI_Parsing_Model
from options.test_options import TestOptions
from util.util import parsing2im_batch_by20chnl, parsing_2_onechannel, tensor2im

app = flask.Flask(__name__)

UPLOAD_FOLDER = './api/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model1 = None
model2 = None
config1 = None
config2 = None
name_list = ['original.jpg', 'sketch.png', 'mask.png', 'stroke.png']
names = ['original', 'sketch', 'mask', 'stroke']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/result/<path:file_name>', methods=['GET', 'POST'])
def index(file_name):
    if os.path.isdir(file_name):
        return '文件夹无法下载'
    else:
        return send_from_directory('./result', file_name, as_attachment=True)


@app.route("/generate", methods=["POST"])
def generate():
    """
    需要前端提供原图original、sketch、mask和stroke
    :return: json
    """
    result = {"success": False}

    if request.method == 'POST':
        data = json.loads(request.get_data(as_text=True))['data']
        for i in range(4):
            with open(UPLOAD_FOLDER + name_list[i], 'wb') as decode_image:
                decode_image.write(base64.b64decode(str(data.get(names[i]))))

        original = cv2.imread(UPLOAD_FOLDER + 'original.jpg')
        sketch = cv2.imread(UPLOAD_FOLDER + 'sketch.png')
        mask = cv2.imread(UPLOAD_FOLDER + 'mask.png')
        stroke = cv2.imread(UPLOAD_FOLDER + 'stroke.png')
        result['result'] = "http://gpu193.mistgpu.xyz:30324/result/" + get_result(original, sketch, mask, stroke)
        result["success"] = True

    return flask.jsonify(result)


def test_api():
    original = cv2.imread(UPLOAD_FOLDER + 'original.jpg')
    sketch = cv2.imread(UPLOAD_FOLDER + 'sketch.png')
    mask = cv2.imread(UPLOAD_FOLDER + 'mask.png')

    stroke = cv2.imread(UPLOAD_FOLDER + 'stroke.png')
    result = get_result(original, sketch, mask, stroke)
    print("done")


def load_model():
    global model1
    global model2
    global config1
    global config2

    opt = TestOptions().parse('model1')
    opt.dataset_mode = 'fashionE'
    opt.netG = 'parsing'
    opt.norm_G = 'batch'
    opt.stage = 1
    opt.input_nc = 26
    opt.output_nc = 20
    opt.name = 'fashionE_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520'
    opt.which_epoch = 20
    model1 = StageI_Parsing_Model(opt)
    model1.eval()
    config1 = opt

    opt2 = TestOptions().parse('model2')
    opt2.dataset_mode = 'fashionE'
    opt2.output_nc = 3
    opt2.input_nc = 29
    opt2.stage = 25
    opt2.netG = 'multiatt3'
    opt2.norm_G = 'batch'
    opt2.ngf = 64
    opt2.input_ANLs_nc = 5
    opt2.which_epoch = 20
    opt2.name = 'fasionE_stageII_matt3_noFeat_noL1_tv10_style200_20190520'
    model2 = StageII_MultiAtt3_Model(opt2)
    model2.eval()
    config2 = opt2


def get_result(original, sketch, mask, stroke):
    """

    :param original: 原图
    :param sketch: 手绘草图
    :param mask: 掩码
    :param stroke: 颜色喷漆
    :return: 生成的图片
    """
    load_model()
    global model1
    global model2
    global config1
    global config1
    # TODO
    # 这里命名realname = "1"是因为初步搭建框架，功能不完善，后续我们需要在后端将输入的图片处理一下获得parsing map
    # 初步阶段就先默认我们有parsing map和parsing_gray.png了
    realname = "1"
    file = "./model_input/1_image.jpg"
    cv2.imwrite('./model_input/' + realname + '_image.jpg', original)
    cv2.imwrite('./model_input/' + realname + '_mask_final.png', mask)
    noise = make_noise() * mask
    mask_input1 = mask  # 没经过asarray
    mask_3 = np.asarray(mask[:, :, 0] / 255, dtype=np.uint8)
    mask_3 = np.expand_dims(mask_3, axis=2)
    sketch = sketch
    stroke = stroke * mask_3

    cv2.imwrite("./model_input/" + realname + "_noise.png", noise)
    cv2.imwrite("./model_input/" + realname + "_sketch.png", sketch)
    cv2.imwrite("./model_input/" + realname + "_stroke.png", stroke)
    img_path = file
    # TODO
    # 就是这里，realname_parsing_gray.png需要后端生成，年前我们先不管这个
    parsing_path = img_path.replace('image', 'parsing').replace('.jpg', '_gray.png')

    # 下面这些基本都是复制的demo.py
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    transform_image = transforms.Compose(transform_list)

    label = imread(parsing_path)
    params = get_params(config1, label.shape)
    transform_label = get_transform(config1, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0

    incompleted_image = original * (1 - mask)
    cv2.imwrite("./model_input/" + realname + "_incom.png", incompleted_image)
    mask_onehot = rgb2gray(imread("./model_input/" + realname + "_mask_final.png"))
    mask_onehot = (mask_onehot > 0).astype(np.uint8)
    mask_arr = mask_onehot * 255
    mask_2 = Image.fromarray(mask_arr).convert('RGB')
    mask_tensor = transform_image(mask_2)

    original_label = imread(parsing_path)
    incompleted_label = original_label * (1 - mask_onehot)
    incompleted_label = Image.fromarray(incompleted_label)
    incompleted_label_tensor = transform_label(incompleted_label) * 255.0

    # incompleted image

    incompleted_image_1 = imread("./model_input/" + realname + "_incom.png")
    # print(incompleted_image_1.shape)
    incompleted_image_1 = Image.fromarray(incompleted_image_1)
    incompleted_image_tensor = transform_image(incompleted_image_1)
    # print(incompleted_image_tensor.size())
    image_tensor = transform_image(original)

    noise_tensor = torch.randn_like(image_tensor)
    mask_noise_tensor = noise_tensor * (1 + mask_tensor) / 2.0 - (1 - mask_tensor) / 2.0

    more_edge_numpy = cv2.imread("./model_input/" + realname + "_sketch.png")
    edge_final = more_edge_numpy
    cv2.imwrite("./model_input/" + realname + "_edge_masked.png", edge_final)
    edge_masked_final = imread("./model_input/" + realname + "_edge_masked.png")

    edge_masked_final_1 = Image.fromarray(edge_masked_final).convert('RGB')
    mask_edge_tensor = transform_image(edge_masked_final_1)

    color = imread("./model_input/" + realname + "_stroke.png")
    color_1 = Image.fromarray(color).convert('RGB')
    # cv2.imwrite("./model_input/" + realname + "_stroke1.png", color_1)
    color_tensor = transform_image(color_1)

    # incompleted label
    bg_mask_numpy = (original_label == 0).astype(np.uint8)
    part_mask_numpy = (mask_onehot * (1 - bg_mask_numpy) + bg_mask_numpy).astype(np.uint8)
    part_mask_numpy = 1 - part_mask_numpy
    part_mask = Image.fromarray(part_mask_numpy)
    part_mask_tensor = transform_label(part_mask) * 255

    face_mask_numpy = (original_label == 13).astype(np.uint8)
    face_mask_img = Image.fromarray(face_mask_numpy)
    face_mask_tensor = transform_label(face_mask_img) * 255
    mask_onehot_img = Image.fromarray(mask_onehot)
    mask_onehot_tensor = transform_label(mask_onehot_img) * 255

    part_mask = Image.fromarray(part_mask_numpy * 255).convert('RGB')
    part_RGB = transform_image(part_mask)

    label_tensor = label_tensor.unsqueeze(0)
    incompleted_label_tensor = incompleted_label_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)
    incompleted_image_tensor = incompleted_image_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0)
    mask_edge_tensor = mask_edge_tensor.unsqueeze(0)
    mask_noise_tensor = mask_noise_tensor.unsqueeze(0)
    color_tensor = color_tensor.unsqueeze(0)
    face_mask_tensor = face_mask_tensor.unsqueeze(0)
    mask_onehot_tensor = mask_onehot_tensor.unsqueeze(0)
    part_mask_tensor = part_mask_tensor.unsqueeze(0)
    part_RGB = part_RGB.unsqueeze(0)

    data_1 = {
        'label': label_tensor,
        'incompleted_label': incompleted_label_tensor,
        'original_image': image_tensor,
        'incompleted_image': incompleted_image_tensor,
        'mask': mask_tensor[:, 0:1, :, :],
        'mask_edge': mask_edge_tensor[:, 0:1, :, :],
        'mask_noise': mask_noise_tensor[:, 0:1, :, :],
        'mask_color': color_tensor,
        'part_mask': part_mask_tensor,
        'face_mask': face_mask_tensor,
        'mask_onehot': mask_onehot_tensor,
        'part_RGB': part_RGB,
    }

    result = model1(data_1, mode='inference')  # (20, 320, 512)

    result1 = parsing2im_batch_by20chnl(result, 0)

    second_input = parsing_2_onechannel(result[0])

    cv2.imwrite("./model_input/" + realname + "_model1_output.png", second_input)

    label1 = Image.open("./model_input/" + realname + "_model1_output.png")
    params = get_params(config2, label1.size)
    transform_label = get_transform(config2, params, method=Image.NEAREST, normalize=False)
    label_tensor1 = transform_label(label1) * 255.0

    original_label1 = imread("./model_input/" + realname + "_model1_output.png")
    incompleted_label1 = original_label1 * (1 - mask_onehot)
    incompleted_label1 = Image.fromarray(incompleted_label1)
    incompleted_label_tensor1 = transform_label(incompleted_label1) * 255.0

    label_tensor1 = label_tensor1.unsqueeze(0)
    incompleted_label_tensor1 = incompleted_label_tensor1.unsqueeze(0)

    data_2 = {
        'label': label_tensor1,
        'incompleted_label': incompleted_label_tensor1,
        'original_image': image_tensor,
        'incompleted_image': incompleted_image_tensor,
        'mask': mask_tensor[:, 0:1, :, :],
        'mask_edge': mask_edge_tensor[:, 0:1, :, :],
        'mask_noise': mask_noise_tensor[:, 0:1, :, :],
        'mask_color': color_tensor,
        'part_mask': part_mask_tensor,
        'face_mask': face_mask_tensor,
        'mask_onehot': mask_onehot_tensor,
        'part_RGB': part_RGB,
    }

    result2 = model2(data_2, mode='inference')

    result_final = tensor2im(result2[0])

    result_final = np.concatenate([result_final[:, :, :1], result_final[:, :, 1:2], result_final[:, :, 2:3]], axis=2)
    result_show_1 = np.concatenate([result1[:, :, 2:3], result1[:, :, 1:2], result1[:, :, :1]], axis=2)
    result_show = np.concatenate([result_final[:, :, 2:3], result_final[:, :, 1:2], result_final[:, :, :1]], axis=2)
    cv2.imwrite("./result/" + realname + "result1.png", result_show_1)
    cv2.imwrite("./result/" + realname + "result2.png", result_show)
    f = (edge_masked_final > 100).astype(np.uint8) * 255.0
    g = (mask_arr > 240).astype(np.uint8) * 255.0
    g = np.expand_dims(g, axis=2)
    e = rgb2gray(color)
    e = np.expand_dims(e, axis=2)
    e = (e > 0).astype(np.uint8) * 255.0
    q = np.zeros((512, 320, 3))
    image_mid = q + g - e + color + incompleted_image_1 - f
    image_mid = np.concatenate((image_mid[:, :, 2:3], image_mid[:, :, 1:2], image_mid[:, :, :1]), axis=2)
    cv2.imwrite("./model_input/" + realname + "_image_mid.png", image_mid)

    first = np.pad(original, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))
    second = np.pad(image_mid, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))
    third = np.pad(result_show, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))
    fourth = np.pad(result_show_1, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))

    final = np.concatenate((first, second, third, fourth), axis=1)
    final3 = np.concatenate((first, second, third), axis=1)

    cv2.imwrite("./full_pic/" + realname + "_full4pic.png", final)
    cv2.imwrite("./full_pic/" + realname + "_full3pic.png", final3)

    cv2.imwrite("./result/" + realname + "_result.png", final3)
    return realname + "_result.png"


def make_noise():
    noise = np.zeros([512, 320, 1], dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    return noise


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=30324)
    # test_api()
