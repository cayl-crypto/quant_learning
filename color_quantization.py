import cv2
import numpy as np
from tqdm import tqdm
import os
import glob

# Loading our image
img = cv2.imread('original/seg_test/seg_test/buildings/20057.jpg', 1)
# cv2.imshow('original', img)
# cv2.waitKey(0)

seg_pred = glob.glob("original/seg_pred/seg_pred/*.jpg")
seg_test = glob.glob("original/seg_test/seg_test/*/*.jpg")
seg_train = glob.glob("original/seg_train/seg_train/*/*.jpg")
quantize = [2, 4, 8, 16, 32, 64, 128, 256]
# cv2.imwrite('file.jpg', img)
# if not os.path.exists('2/my_folder/abc'):
#     os.makedirs('2/my_folder/abc')
print(seg_pred[0].split(os.path.sep))
a = seg_pred[0].split(os.path.sep)
a[0] = str(2)
print(a)
print(os.path.join(*a))


def color_quantization(image, K):
    # Defining input data for clustering
    data = np.float32(image).reshape((-1, 3))
    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Applying cv2.kmeans function
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result


def quantize_all(im_list, quantize_list):
    for im_path in tqdm(im_list):
        # read image
        im = cv2.imread(im_path, 1)
        for k in quantize_list:
            # quantize image
            quant_im = color_quantization(im, k)

            # save path

            save_path_list = im_path.split(os.path.sep)
            save_path_list[0] = str(k)
            save_path = os.path.join(*save_path_list[:-1])

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_file_path = os.path.join(*save_path_list)
            # save quantized image
            cv2.imwrite(save_file_path, quant_im)


quantize_all(seg_test, quantize)
quantize_all(seg_pred, quantize)
quantize_all(seg_train, quantize)
# Applying color quantization with different values for K
# color_2 = color_quantization(img, 2)
# cv2.imshow('2', color_2)
# cv2.waitKey(0)
# color_4 = color_quantization(img, 4)
# cv2.imshow('4', color_4)
# cv2.waitKey(0)
# color_8 = color_quantization(img, 8)
# cv2.imshow('8', color_8)
# cv2.waitKey(0)
# color_16 = color_quantization(img, 16)
# cv2.imshow('16', color_16)
# cv2.waitKey(0)
# color_32 = color_quantization(img, 32)
# cv2.imshow('32', color_32)
# cv2.waitKey(0)
# color_64 = color_quantization(img, 64)
# cv2.imshow('64', color_64)
# cv2.waitKey(0)
# color_128 = color_quantization(img, 128)
# cv2.imshow('128', color_128)
# cv2.waitKey(0)
# color_256 = color_quantization(img, 256)
# cv2.imshow('256', color_256)
# cv2.waitKey(0)
