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
# from scipy.misc import imresize
from werkzeug.utils import secure_filename

from data.base_dataset import get_params, get_transform
from models.stageII_multiatt3_model import StageII_MultiAtt3_Model
from models.stageI_parsing_model import StageI_Parsing_Model
from options.test_options import TestOptions
from util.util import parsing2im_batch_by20chnl, parsing_2_onechannel, tensor2im

import torchfcn
import copy


app = flask.Flask(__name__)

# parsing part
model_original = None

# fashion editing part
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
    需要前端提供名字name、原图original、sketch、mask和stroke
    :return: json
    """
    result = {"success": False}

    if request.method == 'POST':
        data = json.loads(request.get_data(as_text=True))['data']
        for i in range(4):
            with open(UPLOAD_FOLDER + name_list[i], 'wb') as decode_image:
                decode_image.write(base64.b64decode(str(data.get(names[i]))))

        name = str(data.get('name'))
        print(name)
        size = (320, 512)
        original = cv2.imread(UPLOAD_FOLDER + 'original.jpg')
        original = cv2.resize(original, dsize=size)
        cv2.imwrite(UPLOAD_FOLDER + 'original.jpg', original)

        sketch = cv2.imread(UPLOAD_FOLDER + 'sketch.png')
        sketch = cv2.resize(sketch, dsize=size)
        cv2.imwrite(UPLOAD_FOLDER + 'sketch.png', sketch)

        mask = cv2.imread(UPLOAD_FOLDER + 'mask.png')
        mask = cv2.resize(mask, dsize=size)
        cv2.imwrite(UPLOAD_FOLDER + 'mask.png', mask)

        stroke = cv2.imread(UPLOAD_FOLDER + 'stroke.png')
        stroke = cv2.resize(stroke, dsize=size)
        cv2.imwrite(UPLOAD_FOLDER + 'stroke.png', stroke)

        result['result'] = "mist@gpu82.mistgpu.xyz:30524/result/" + get_result(name, original, sketch, mask, stroke)
        result["success"] = True

    return flask.jsonify(result)


def test_api():
    path = '0AE81A01V/0AE81A01V-A11@12.jpg'  # image目录下的相对路径
    original = cv2.imread('../datasets/fashionE/fashionE_320_512_image/' + path)
    sketch = cv2.imread(UPLOAD_FOLDER + 'sketch.png')
    mask = cv2.imread(UPLOAD_FOLDER + 'mask.png')

    stroke = cv2.imread(UPLOAD_FOLDER + 'stroke.png')
    result = get_result(path, original, sketch, mask, stroke)
    print("done")


def load_model():
    global model1
    global model2
    global config1
    global config2

    global model_original

    model_original = torchfcn.models.FCN8sAtOnce(n_class=20).cuda()
    model_original_file = os.path.expanduser('~/fcn8s_at_once_epoch25.pth.tar')

    print('==> Loading %s model file: %s' %
          (model_original.__class__.__name__, model_original_file))
    model_data = torch.load(model_original_file)
    try:
        model_original.load_state_dict(model_data)
    except Exception:
        model_original.load_state_dict(model_data['model_state_dict'])
    model_original.eval()

    opt = TestOptions().parse('model1')
    opt.dataset_mode = 'fashionE'
    opt.netG = 'parsing'
    opt.norm_G = 'batch'
    opt.stage = 1
    opt.input_nc = 26
    opt.output_nc = 20
    opt.name = 'stage1_01'
    opt.which_epoch = 'latest'
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
    opt2.which_epoch = 'latest'
    opt2.name = 'stage2_01'
    model2 = StageII_MultiAtt3_Model(opt2)
    model2.eval()
    config2 = opt2


def get_result(name, original, sketch, mask, stroke):
    """
    :param real_name: 对应数据集中图片名
    :param original: 原图
    :param sketch: 手绘草图
    :param mask: 掩码
    :param stroke: 颜色喷漆
    :return: 生成的图片
    """

    global model1
    global model2
    global config1
    global config1

    global model_original

    real_name = name.replace('.jpg', '')

    mean_bgr = np.array([187.4646117, 190.3556895, 198.6592035])
    original_copy = copy.deepcopy(np.array(original).astype(np.float64))
    original_copy -= mean_bgr
    original_copy = original_copy.transpose(2, 0, 1)
    original_copy_tensor = torch.from_numpy(original_copy).cuda().unsqueeze(0).float()
    parsing_gray = model_original(original_copy_tensor).data.max(1)[1].squeeze(0)
    parsing_gray = parsing_gray.cpu().numpy()
    cv2.imwrite('./model_input/' + real_name + '_image_parsing_gray.png', parsing_gray)

    cv2.imwrite('./model_input/' + real_name + '_image.jpg', original)
    cv2.imwrite('./model_input/' + real_name + '_mask_final.png', mask)
    noise = make_noise() * mask
    mask_input1 = mask  # 没经过asarray
    mask_3 = np.asarray(mask[:, :, 0] / 255, dtype=np.uint8)
    mask_3 = np.expand_dims(mask_3, axis=2)
    sketch = sketch
    stroke = stroke * mask_3

    cv2.imwrite("./model_input/" + real_name + "_noise.png", noise)
    cv2.imwrite("./model_input/" + real_name + "_sketch.png", sketch)
    cv2.imwrite("./model_input/" + real_name + "_stroke.png", stroke)
    parsing_path = "./model_input/" + real_name + "_image_parsing_gray.png"

    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    transform_image = transforms.Compose(transform_list)

    label = imread(parsing_path)
    params = get_params(config1, label.shape)
    transform_label = get_transform(config1, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0

    mask_onehot = rgb2gray(imread("./model_input/" + real_name + "_mask_final.png"))
    mask_onehot = (mask_onehot > 0).astype(np.uint8)  # 0和1
    # 1 - mask后，原mask白色的区域应该对应0
    # mask中白色为255 np.uint8(1 - 255) == 2
    temp = 1 - mask
    temp[temp == 2] = 0
    incompleted_image = original * temp  # mask的部分变为0即黑色 1 - mask得到1或2，似有误
    cv2.imwrite("./model_input/" + real_name + "_incom.png", incompleted_image)
    mask_arr = mask_onehot * 255  # 还原成了0和255的mask，不过是单通道的，而mask是三通道的
    mask_2 = Image.fromarray(mask_arr).convert('RGB')
    mask_tensor = transform_image(mask_2)

    original_label = imread(parsing_path)
    incompleted_label = original_label * (1 - mask_onehot)
    incompleted_label = Image.fromarray(incompleted_label)
    incompleted_label_tensor = transform_label(incompleted_label) * 255.0

    # incompleted image

    incompleted_image_1 = imread("./model_input/" + real_name + "_incom.png")
    # print(incompleted_image_1.shape)
    incompleted_image_1 = Image.fromarray(incompleted_image_1)
    incompleted_image_tensor = transform_image(incompleted_image_1)
    # print(incompleted_image_tensor.size())
    image_tensor = transform_image(original)

    noise_tensor = torch.randn_like(image_tensor)
    mask_noise_tensor = noise_tensor * (1 + mask_tensor) / 2.0 - (1 - mask_tensor) / 2.0

    more_edge_numpy = cv2.imread("./model_input/" + real_name + "_sketch.png")
    edge_final = more_edge_numpy
    cv2.imwrite("./model_input/" + real_name + "_edge_masked.png", edge_final)
    edge_masked_final = imread("./model_input/" + real_name + "_edge_masked.png")

    edge_masked_final_1 = Image.fromarray(edge_masked_final).convert('RGB')
    mask_edge_tensor = transform_image(edge_masked_final_1)

    color = imread("./model_input/" + real_name + "_stroke.png")
    color_1 = Image.fromarray(color).convert('RGB')
    # cv2.imwrite("./model_input/" + real_name + "_stroke1.png", color_1)
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

    result = model1(data_1, mode='inference').unsqueeze(0)  # (1, 20, 320, 512)

    result1 = parsing2im_batch_by20chnl(result, 0)

    second_input = parsing_2_onechannel(result[0])

    cv2.imwrite("./model_input/" + real_name + "_model1_output.png", second_input)

    label1 = Image.open("./model_input/" + real_name + "_model1_output.png")
    params = get_params(config2, label1.size)
    transform_label = get_transform(config2, params, method=Image.NEAREST, normalize=False)
    label_tensor1 = transform_label(label1) * 255.0

    original_label1 = imread("./model_input/" + real_name + "_model1_output.png")
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
    # mask_tensor -1黑 1白
    # original = cv2.imread('./model_input/' + '1_image.jpg')
    #
    # mask = cv2.imread('./model_input/' + 'mask_final.png')

    mask_tensor = mask_tensor[0].cuda()
    image_tensor = image_tensor[0].cuda()
    generated_image_mul_mask = tensor2im(result2[0] * (1 + mask_tensor) / 2.0 - (1 - mask_tensor) / 2.0)
    original_image_mul_mask = tensor2im(image_tensor * (1 - mask_tensor) / 2.0 - (1 + mask_tensor) / 2.0)
    # cv2.imwrite("./result/" + real_name + "use_mask.png", (generated_image_mul_mask + original_image_mul_mask))


    result_final = tensor2im(result2[0])  # 得到HWC ndarray
    # print(type(1 - mask))
    # print(type(result_final * mask))

    result_final = np.concatenate([result_final[:, :, :1], result_final[:, :, 1:2], result_final[:, :, 2:3]], axis=2)
    # BGR -> RGB
    result_show_1 = np.concatenate([result1[:, :, 2:3], result1[:, :, 1:2], result1[:, :, :1]], axis=2)
    result_show = np.concatenate([result_final[:, :, 2:3], result_final[:, :, 1:2], result_final[:, :, :1]], axis=2)
    result_show = result_show * (1 - temp) + original * temp
    cv2.imwrite("./result/" + real_name + "result1.png", result_show_1)  # model1的结果
    cv2.imwrite("./result/" + real_name + "result2.png", result_show)  # model2的结果
    f = (edge_masked_final > 100).astype(np.uint8) * 255.0
    g = (mask_arr > 240).astype(np.uint8) * 255.0
    g = np.expand_dims(g, axis=2)
    e = rgb2gray(color)
    e = np.expand_dims(e, axis=2)
    e = (e > 0).astype(np.uint8) * 255.0
    q = np.zeros((512, 320, 3))
    image_mid = q + g - e + color + incompleted_image_1 - f
    image_mid = np.concatenate((image_mid[:, :, 2:3], image_mid[:, :, 1:2], image_mid[:, :, :1]), axis=2)
    cv2.imwrite("./model_input/" + real_name + "_image_mid.png", image_mid)

    first = np.pad(original, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))
    second = np.pad(image_mid, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))
    third = np.pad(result_show, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))
    fourth = np.pad(result_show_1, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))

    final = np.concatenate((first, second, third, fourth), axis=1)
    final3 = np.concatenate((first, second, third), axis=1)

    cv2.imwrite("./full_pic/" + real_name + "_full4pic.png", final)
    cv2.imwrite("./full_pic/" + real_name + "_full3pic.png", final3)

    cv2.imwrite("./result/" + real_name + "_result.png", final3)
    return real_name + "_result.png"


def make_noise():
    noise = np.zeros([512, 320, 1], dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    return noise


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=60504)
    # test_api()



