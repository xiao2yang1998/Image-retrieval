from __future__ import division
from PIL import Image, ImageDraw

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


def rgb22hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    import math
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v


def quantilize(h, s, v):
    '''hsv直方图量化
    '''
    hlist = [20, 40, 75, 155, 190, 270, 290, 316, 360]
    slist = [0.25, 0.7, 1.0]
    vlist = [0.3, 0.8, 1.0]

    for i in range(len(hlist)):
        if h <= hlist[i]:
            h = i % 8
            break
    for i in range(len(slist)):
        if s <= slist[i]:
            s = i
            break
    for i in range(len(vlist)):
        if v <= vlist[i]:
            v = i
            break

    return 9 * h + 3 * s + v


def getColorFeature(filename):
    img = Image.open(filename)

    re = img.split()
    if len(re) == 1:
        r = re[0];
        g = re[0];
        b = re[0]
    else:
        r = re[0];
        g = re[1];
        b = re[2]
    r = np.array(r).flatten()
    g = np.array(g).flatten()
    b = np.array(b).flatten()
    h = []
    s = []
    v = []
    for i in range(r.shape[0]):
        h_t, s_t, v_t = rgb22hsv(r[i], g[i], b[i])
        h.append(h_t)
        s.append(s_t)
        v.append(v_t)

    h = np.array(h).flatten()
    s = np.array(s).flatten()
    v = np.array(v).flatten()

    calc = [0 for i in range(0, 80)]
    re = []
    for i in range(r.shape[0]):
        temp = quantilize(h[i], s[i], v[i])
        re.append(temp)
        calc[temp] += 1;

    return np.array(calc)


def getDistance_Eruo(im1, im2):
    distance = 0
    for i in range(im1.shape[0]):
        distance += (im1[i] - im2[i]) ** 2
    distance = math.sqrt(distance)
    return distance


def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def getDistance_Pear(x, y):
    x = np.array(x, dtype="int64")
    y = np.array(y, dtype="int64")
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = math.sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den


class DataMomemt():
    def __init__(self, path):
        self.picture_path = path
        self.picture_names = os.listdir(self.picture_path)
        self.picture_names = self.picture_names[1:]

    def moment(self):
        color_features = []

        for image_name in self.picture_names:
            try:
                color_feature = []
                color_feature.append(image_name)
                color_feature.extend(getColorFeature(self.picture_path + '/' + image_name))
                color_features.append(color_feature)
            except OSError:
                print(image_name + u'图像文件发生错误')
        return color_features


def processFolder(path):
    outfile = 'data/resutl' + path + '.csv'
    data = DataMomemt(path)
    feature = data.moment()
    with open(outfile, 'w', newline='') as out:
        csv_writer = csv.writer(out, dialect='excel')
        for f in feature:
            csv_writer.writerow(f)


def initial_csv():
    # for i in range(1,10):
    #   if(i!=5):
    #       processFolder('0'+str(i))
    #   print('finish'+str(i))

    for i in range(1, 10):
        if (i != 5):
            path = '0' + str(i);
            datafile = 'data/resutl' + path + '.csv'
            data1 = pd.read_csv(datafile, header=None)
            if (i == 1):
                data = data1
            else:
                data = data.append(data1, ignore_index=True)
    data.to_csv('data/total.csv', index=False)


def single2total():
    data = []
    for i in range(1, 9):
        if i != 5:
            path = '0' + str(i);
        datafile = 'data/resutl' + path + '.csv'
        data1 = pd.read_csv(datafile, header=None)
        if (i == 1):
            data = data1
        else:
            data = data.append(data1, ignore_index=True)
    data.to_csv('data/total.csv', index=False)


def getCategory(s):
    category = ['001_', '002_', '003_', '004_', '005_',
                '006_', '007_', '008_', '009_', '010_']

    for i in range(len(category)):
        if category[i] in s:
            return i;


def getRight_Ratio():
    # select the target image
    im_index = random.randint(0, len(data))
    target_image = title[im_index]
    print(target_image)
    print('target' + str(getCategory(target_image)))
    tagrget_cato = getCategory(target_image)
    target_feature = getColorFeature('total/' + target_image)

    re = []

    for i in range(len(data)):
        re_temp = [title[i]]
        dist = getDistance_Pear(target_feature, list(data.iloc[i, :]))
        re_temp.append(dist)
        re_temp.append(i)
        re.append(re_temp)

    result = heapq.nlargest(10, re, key=lambda s: s[1])

    right = 0;
    n = 10;

    for i in range(10):
        # img = Image.open('total/'+result[i][0])
        # img.save('result/'+result[i][0]+'_similar'+str(i)+'.jpg')
        if (getCategory(result[i][0]) == tagrget_cato):
            right += 1;
    return (right / n)


def testone():
    data = pd.read_csv('data/total.csv', header=None)
    title = data[data.columns[0]]
    data = data[data.columns[1:]]

    # select the target image
    im_index = random.randint(0, len(data))
    print(im_index)
    print(len(data))
    target_image = title[im_index]
    print(target_image)
    target_feature = getColorFeature('total/' + target_image)

    re = []

    for i in range(len(data)):
        re_temp = [title[i]]
        dist = getDistance_Pear(target_feature, list(data.iloc[i, :]))
        re_temp.append(dist)
        re_temp.append(i)
        re.append(re_temp)

    ##use pearson correlation
    result = heapq.nlargest(10, re, key=lambda s: s[1])
    ##use Eruo distance
    # result = heapq.nsmallest(10,re,key = lambda s:s[1])

    for i in range(10):
        img = Image.open('total/' + result[i][0])
        print(result[i][1])
        img.save('result/similar' + str(i) + '.jpg')
    Image.open('total/' + target_image).show()


if __name__ == '__main__':
    data = pd.read_csv('data/total.csv', header=None)
    title = data[data.columns[0]]
    data = data[data.columns[1:]]
    re = 0
    for i in range(100):
        re += getRight_Ratio()
        print(i)
    re = re / 100
    print(re)