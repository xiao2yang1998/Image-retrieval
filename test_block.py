from __future__ import division
from PIL import Image, ImageDraw
from ImageProcess import ImageBlock
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


if __name__ == '__main__':
    filename = '004_0092.jpg'
    im = ImageBlock.ImageBlock("total",filename)
    re = im.get_results

    print(re)
