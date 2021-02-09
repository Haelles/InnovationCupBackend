
import os
import numpy as np
import cv2 as cv

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mean_color_mask(image, mask):
    mean = cv.mean(image, mask=mask)[:3]
    return mean


def get_color(image, label, n):
    mask = (label == n).astype(np.uint8)
    if n == 0:
        mean = [0, 0, 0]
    else:
        mean = mean_color_mask(image, mask)
    mask = np.expand_dims(mask, axis=2)
    m1 = mask * mean[0]
    m2 = mask * mean[1]
    m3 = mask * mean[2]
    mask_color = np.concatenate((m1, m2, m3), axis=2).astype(np.uint8)
    return mask_color


def get_color_domain(image, label):
    mask_color = np.zeros((label.shape[0], label.shape[1], 3), np.uint8)
    for i in range(20):
        mask_color = mask_color + get_color(image, label, i)
    return mask_color

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
    img_filepath = os.path.join(root, name.split()[0])
    label_filepath = os.path.join(root.replace('320_512_image', '320_512_parsing'), name.split()[0]).replace('.jpg', '_gray.png')
    output_filepath = os.path.join(root.replace('320_512_image', '320_512_color'), name.split()[0])

    image = cv.imread(img_filepath, 1)  # real image
    label = cv.imread(label_filepath, 0)  # parsing gray

    median_f = cv.medianBlur(image, 3)
    median_filtered_f = cv.bilateralFilter(median_f, 7, 20.0, 20.0)
    color_domain = get_color_domain(median_filtered_f, label)

    subfold_path = os.path.split(output_filepath)[0]
    if not os.path.exists(subfold_path):
        mkdirs(subfold_path)
    if count % 100 == 0:
        print(count)
    count = count + 1

    cv.imwrite(output_filepath, color_domain)

print("END!!!")



### MPV
