# -*- coding: utf-8 -*-
from PIL import Image
import os, sys
import func_1 as f1
import func_2 as f2
import func_3 as f3
import os
import h5py
import numpy as np
import argparse
from PIL import Image
from extract_cnn_vgg16_keras import VGGNet


def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.bmp') or f.endswith('.png')]


def import_images(folder_path):
    db = "total"
    folder = folder_path
    img_list = get_imlist(folder)
    output = "featureCNN.total"

    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        img = Image.open(folder + '/' + img_name)
        img.save(db + '/' + img_name)
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))


    h5f = h5py.File(output, 'r')
    names_np = h5f['dataset_2'][:]
    names_np = [str(i, 'utf-8') for i in names_np]
    print(len(names_np))
    names_np.extend(names)
    print(len(names_np))

    features_np = h5f['dataset_1'][:]
    features_np = list(features_np)
    print(len(features_np))
    features_np.extend(feats)
    print(len(features_np))
    features_np = np.array(features_np)
    h5f.close()

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=features_np)
    h5f.create_dataset('dataset_2', data=np.string_(names_np))
    h5f.close()

    f1.processFolder(folder)
    f2.processFolder(folder)


if __name__ == '__main__':
    img = 'total/006_0002.jpg'
    #查询，img为图片路径，第二个参数为查询张数
    re3 = f3.query(img,10)
    print(re3)
    re2 = f2.query(img,10)
    print(re2)
    re1 = f1.query(img,10)
    print(re1)
    # 导入图片
    # import_images('11_1')
