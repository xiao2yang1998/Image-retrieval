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


if __name__ == '__main__':
    # initial_csv('total')
    initial_csv_texture('total')
