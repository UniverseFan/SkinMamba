import base64
import json
import os
import platform

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

FOLDER = '/home/sql/data0222/ISIC2018'


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_files(folder):
    files = [item for item in os.listdir(folder) if item[0] != '.']
    files.sort()
    files = [f for f in files if '.png' in f or '.jpg' in f]
    files = [os.path.join(folder, f) for f in files]
    return files

for item in os.listdir(FOLDER):
    if item[0] != '.' and 'ISIC' in item:
        images = get_files(os.path.join(FOLDER,item))


        i = 0
        for image in images:
            split_tag = "/"
            if platform.system() == 'Windows':
                split_tag = "\\"
            file_name = image.split(split_tag)[-1]

            image_arr = cv.imread(image)
            image_arr = cv.resize(image_arr, (256,256), interpolation=cv.INTER_AREA)
            pre = 'train'
            if 'Valid' in image:
                pre = 'valid'
            elif 'Test' in image:
                pre = 'test'
            sec = 'images'
            if 'Ground' in image:
                sec = 'masks'
            # print(os.path.join(FOLDER,pre,sec,file_name))
            cv.imwrite(os.path.join(FOLDER,pre,sec,file_name), image_arr)
