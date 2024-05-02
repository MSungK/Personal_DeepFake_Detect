
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob as g
import cv2
from tqdm import tqdm
from shutil import copy, move


def resize_img(input_name , output_name, target_width = 640):
    # print(input_name)
    im = cv2.imread(input_name)
    h,w,_  = im.shape
    target_height = int(h / w * target_width)
    im = cv2.resize(im , (target_width , target_height), interpolation = cv2.INTER_AREA)
    cv2.imwrite(output_name , im)


def resize_all_imgs(imgs_path):
    for img in tqdm(imgs_path):
        resize_img(img, img)


if __name__ == '__main__':
    imgs_dir = 'wider_yolo_format/images/**/*.jpg'
    lbls_dir = 'wider_yolo_format/labels/**/*.txt'
    names = g(lbls_dir, recursive=True)
    print(f'Threre are {len(names)}  images')
    imgs_path = g(imgs_dir, recursive=True)
    assert len(imgs_path) == len(names)
    print(imgs_path[0])
    resize_all_imgs(imgs_path)
