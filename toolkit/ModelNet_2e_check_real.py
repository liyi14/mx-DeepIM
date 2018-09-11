# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import numpy as np
import os
from lib.pair_matching.RT_transform import *
from lib.utils.mkdir_if_missing import mkdir_if_missing
import cv2
import matplotlib.pyplot as plt

classes = ['airplane', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'guitar',
           'laptop',
           'mantel', #'dresser',
             'piano', 'range_hood', 'sink', 'stairs',
           'stool', 'tent', 'toilet', 'tv_stand',
           'door', 'glass_box', 'wardrobe', 'plant', 'xbox']
print(classes)


# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]) # LM
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000


# init render machines
# brightness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4] ###################
modelnet_root = '/data/wanggu/Downloads/modelnet'  # NB: change to your dir
modelnet40_root = os.path.join(modelnet_root, 'ModelNet40')
data_dir = os.path.join(modelnet_root, 'modelnet_render_v1/data/real')
model_set_dir = os.path.join(modelnet_root, 'model_set')

example_render_dir = os.path.join(modelnet_root, 'example_render')
mkdir_if_missing(example_render_dir)

for cls_i, cls_name in enumerate(classes):
    if not cls_name in ['door', 'glass_box', 'wardrobe', 'plant', 'xbox']:
        continue
    print(cls_name)
    class_real_dir = os.path.join(data_dir, cls_name)
    for set in ['train', 'test']:
        image_list = [os.path.join(class_real_dir, set, fn)
                      for fn in os.listdir(os.path.join(class_real_dir, set)) if '0000-color' in fn]
        image_list.sort()

        with open(os.path.join(model_set_dir, '{}_{}.txt'.format(cls_name, set))) as f:
            model_list = [line.strip() for line in f.readlines()]

        for model_name in model_list:
            image_path = os.path.join(class_real_dir, set, '{}_0000-color.png'.format(model_name))
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # print(im.shape)
            fig = plt.figure()
            plt.subplot(1, 1, 1)
            plt.imshow(im[:, :, [2,1,0]])
            plt.title(image_path[image_path.find('real'):])
            # plt.show()
            plt.savefig(os.path.join(example_render_dir, '{}_{}'.format(set, os.path.basename(image_path))))



# models that has problem
# car
# train
# 5, 7,



















