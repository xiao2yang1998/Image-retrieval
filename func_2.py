from __future__ import division
from PIL import Image, ImageDraw
from ImageProcess import ImageBlock
from ImageProcess import ImageTexture
import os, sys

# %matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
import codecs
import csv
import math
import heapq
import func_1 as f1

hlist = [20, 40, 75, 155, 190, 270, 290, 316, 360]
slist = [0.25, 0.7, 1.0]
vlist = [0.3, 0.8, 1.0]

def initial_csv_color(path):
    picture_names = os.listdir(path)
    picture_names = picture_names[1:]
    sign = 0
    for image_name in picture_names:
        try:
            if image_name == "007_0061.jpg":
                sign = 1
            if sign == 1:
                im = ImageBlock.ImageBlock(path,image_name)
                im.get_results
        except OSError:
            print(image_name + u'图像文件发生错误')


def initial_csv_texture(path):
    picture_names = os.listdir(path)
    picture_names = picture_names[1:]
    for image_name in picture_names:
        try:
            im = ImageTexture.get_results(path, image_name)
        except OSError:
            print(image_name + u'图像文件发生错误')


def get_hsv(l):
    h = l // 9
    l = l % 9
    s = l // 3
    l = l % 3
    v = l % 3
    return hlist[h], slist[s], vlist[v]


def color_similarities(l1, l2):
    h1, s1, v1 = get_hsv(l1)
    #     print(h1,s1,v1)
    h2, s2, v2 = get_hsv(l2)
    #     print(math.pow((s1*(math.sin(h1))-s2*(math.sin(h2))),2))
    d = 1 - math.sqrt(
        math.pow((s1 * (math.cos(h1)) - s2 * (math.cos(h2))), 2) + math.pow((s1 * (math.sin(h1)) - s2 * (math.sin(h2))),
                                                                            2) + math.pow((v1 - v2), 2))
    if d < 0:
        d = 0
    return d


def s_color(p_1, p_2):
    s_color_type = []
    p = []
    q = []
    for i in p_1:
        if i not in s_color_type:
            s_color_type.append(i)
            p.append(p_1[i])
            if i in p_2:
                q.append(p_2[i])
            else:
                q.append(0)
    for i in p_2:
        if i not in s_color_type:
            s_color_type.append(i)
            q.append(p_2[i])
            if i in p_1:
                p.append(p_1[i])
            else:
                p.append(0)

    length = len(s_color_type)
    s_color_type = np.array(s_color_type).reshape(1, len(s_color_type))
    s_color = np.ones((length, length), np.float16)
    for i in range(len(s_color[0])):
        for j in range(len(s_color[0])):
            s_color[i][j] = color_similarities(s_color_type[0][i], s_color_type[0][j])
    p = np.array(p)
    q = np.array(q)
    psubq = p - q
    re = psubq.dot(s_color.dot(psubq.T))
    re = float(re)
    return re


def s_texture(p_1,p_2):
    re = 0
    for i in range(4):
        re+=math.pow((p_1[0]-p_2[0]),2)
    re = math.sqrt(re)
    return re


def testone():
    # select the target image
    im_index = random.randint(0, len(total_color))
    target_image = total_color_title[im_index]
    path = 'total'
    color_feature = total_color[im_index]
    texture_feature = total_texture[im_index]
    tagrget_cato = f1.getCategory(target_image)

    re = []

    for i in range(len(total_color)):
        re_temp = [total_color_title[i]]
        dist = 0.5*s_color(total_color[i],color_feature)+0.5*s_texture(total_texture[i],texture_feature)
        re_temp.append(dist)
        re_temp.append(i)
        re.append(re_temp)

    # 距离越小，越相近
    result = heapq.nsmallest(10, re, key=lambda s: s[1])
    right = 0
    for i in range(10):
        # img = Image.open('total/' + result[i][0])
        # print(result[i][1])
        # img.save('result/similar' + str(i) + '.jpg')
        if (f1.getCategory(result[i][0]) == tagrget_cato):
            right += 1;
    ratio = right/10
    print(ratio)
    return ratio
    # Image.open('total/' + target_image).show()

if __name__ == '__main__':
    # initial_csv('total')
    # initial_csv_texture('total')
    with open('fuc2_c.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=':', quotechar='|')
        total_color = []
        total_color_title = []
        for row in reader:
            a = list(filter(None, (row[0].split(','))))
            total_color_title.append(a[0])
            a = a[1:]
            temp_dict = {}
            temp = 0.0
            for i in range(len(a)):

                if (i % 2 == 0):
                    temp_dict[int(a[i])] = float(a[i + 1])
                    temp += temp_dict[int(a[i])]
            for j in temp_dict:
                temp_dict[j] /= temp
                temp_dict[j] = float("%.02f" % float(temp_dict[j]))
            total_color.append(temp_dict)
    csvfile.close()

    with open('fuc2_t.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=':', quotechar='|')
        total_texture = []
        total_texture_title = []
        for row in reader:
            a = (row[0].split(','))
            total_texture_title.append(a[0])
            a = a[1:]
            a = [float("%.02f" % float(i)) for i in a]
            total_texture.append(a)
    csvfile.close()
    re = 0
    for i in range(100):
        re+= testone()
    re = re/100
    print("total",re)




