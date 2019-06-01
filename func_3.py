# -*- coding: utf-8 -*-

from extract_cnn_vgg16_keras import VGGNet
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


def query(img,num):
    # read in indexed images' feature vectors and corresponding image names
    h5f = h5py.File("featureCNN.total", 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    # read and show query image
    queryDir = img
    queryImg = mpimg.imread(queryDir)

    # init VGGNet16 model
    model = VGGNet()

    # extract query image's feature, compute simlarity score and sort
    queryVec = model.extract_feat(queryDir)
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # print rank_ID
    # print rank_score

    # number of top retrieved images to show
    maxres = num
    imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
    re = [ str(im, 'utf-8') for im in imlist]
    return re

if __name__ == '__main__':
    img = 'total/006_0109.jpg'
    query(img,10)

