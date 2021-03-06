# !/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import func_1 as f1
import csv

# 定义最大灰度级数
gray_level = 16

def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    # print(height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


def get_results(path,image_name):
    img = cv2.imread(path+'/'+image_name)
    # img_shape = img.shape
    # img = cv2.resize(img, (img_shape[1] / 2, img_shape[0] / 2), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_0 = getGlcm(img_gray, 1, 0)
    # glcm_1=getGlcm(src_gray, 0,1)
    # glcm_2=getGlcm(src_gray, 1,1)
    # glcm_3=getGlcm(src_gray, -1,1)
    asm, con, eng, idm = feature_computer(glcm_0)
    re = [image_name,asm, con, eng, idm]
    print(re)
    outfile = "fuc2_t.csv"
    with open(outfile, 'a', newline='') as out:
        csv_writer = csv.writer(out, dialect='excel')
        csv_writer.writerow(re)
    return 0

def get_one_result(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_0 = getGlcm(img_gray, 1, 0)
    asm, con, eng, idm = feature_computer(glcm_0)
    re = [asm, con, eng, idm]
    return re


if __name__ == '__main__':
    result = get_results("total","004_0092.jpg")
    print(result)





