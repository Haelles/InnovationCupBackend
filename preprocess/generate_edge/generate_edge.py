
import os
import numpy as np
import cv2 as cv

from skimage.feature import canny
from skimage.color import rgb2gray
from scipy.misc import imread

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

### deepfashion
# root = 'I:/datasets/DeepFashion/In-shop Clothes Retrieval Benchmark/Img/img_highres/img_320_512'
# root = '/data1/datasets/In-shop_HD/img_320_512_image'
# root = '/data1/datasets/multi-pose-try-on/MPV_320_512/MPV_320_512_image'
root = '/data1/datasets/fashionE/fashionE_320_512_image'

# sub_filenames = open('../deepfashion_train_test.txt').readlines()
# sub_filenames = open('../mpv_train_test.txt').readlines()
sub_filenames = open('../filter_image_by_keypoint/fashionE_allfile_list.txt').readlines()
count = 0
for name in sub_filenames:
    filepath = os.path.join(root, name.split()[0])
    output_filepath = os.path.join(root.replace('320_512_image', '320_512_edge'), name.split()[0])
    img = imread(filepath)
    img_gray = rgb2gray(img)
    edge = canny(img_gray, sigma=1, mask=None).astype(np.uint8) * 255

    subfold_path = os.path.split(output_filepath)[0]
    if not os.path.exists(subfold_path):
        mkdirs(subfold_path)
    if count % 100 == 0:
        print(count)
    count = count + 1

    cv.imwrite(output_filepath, edge)

print("END!!!")

