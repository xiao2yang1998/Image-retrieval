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



if __name__ == '__main__':
    filename = 'total/001_0043.jpg'
    im = ImageBlock.ImageBlock(filename)
    re = im.get_corners()
    print(re)
