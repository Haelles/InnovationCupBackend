
import cv2 as cv
import numpy as np

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

image = cv.imread("4VI21D020-Q11@13=person_half_front.jpg", 1)
label = cv.imread("4VI21D020-Q11@13=person_half_front_gray.png", 0)


# step 1: blur
median_f = cv.medianBlur(image, 3)
# cv.imwrite("median_f.jpg", median_f)

# step 2: filter
median_filtered_f = cv.bilateralFilter(median_f, 7, 20.0, 20.0)
# cv.imwrite("median_filtered_f.jpg", median_filtered_f)

# step 3: get mean color of each part
color_domain = get_color_domain(image, label)
cv.imwrite("mean_color.jpg", color_domain)

# cv.imshow('mask_color', color_domain)
# cv.waitKey()
# cv.destroyAllWindows()
