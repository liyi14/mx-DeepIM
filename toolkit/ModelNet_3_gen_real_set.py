# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import numpy as np
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '..'))
from lib.utils.mkdir_if_missing import mkdir_if_missing

classes = ['airplane', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'guitar',
           'laptop',
           'mantel', #'dresser',
             'piano', 'range_hood', 'sink', 'stairs',
           'stool', 'tent', 'toilet', 'tv_stand',
           'door', 'glass_box', 'wardrobe', 'plant', 'xbox',
           'bathtub', 'table', 'monitor', 'sofa', 'night_stand']
print(classes)
test_classes = ['range_hood',  #'stairs',
             'tv_stand',
            'bookshelf',
             'mantel',
               'door', 'glass_box',
               'guitar',
               'wardrobe', # test
               # 'plant',
               'xbox',

           'bathtub', 'table', 'monitor', 'sofa', 'night_stand']

# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]) # LM
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000


# init render machines
# brightness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4] ###################
modelnet_root = os.path.join(cur_dir, '../data/ModelNet')
modelnet40_root = os.path.join(modelnet_root, 'ModelNet40')
data_dir = os.path.join(modelnet_root, 'modelnet_render_v1/data/real')

real_set_dir = os.path.join(modelnet_root, 'modelnet_render_v1/image_set/real')
mkdir_if_missing(real_set_dir)


for cls_i, cls_name in enumerate(classes):
    if not cls_name in test_classes:
        continue
    print(cls_name)
    class_dir = os.path.join(data_dir, cls_name)

    all_indices = []
    train_indices = []
    test_indices = []
    for set in ['train', 'test']:
        real_indices = [fn.split('-')[0] for fn in os.listdir(os.path.join(class_dir, set)) if 'color' in fn]
        real_indices.sort()

        for real_idx in real_indices:
            # all_indices.append('{}'.format(set, real_idx))
            if set == 'train':
                train_indices.append('{}'.format(real_idx))
            elif set == 'test':
                test_indices.append('{}'.format(real_idx))

    # all_indices.sort()
    train_indices.sort()
    test_indices.sort()
    train_indices = ['{}/train/{}'.format(cls_name, idx) for idx in train_indices]
    test_indices = ['{}/test/{}'.format(cls_name, idx) for idx in test_indices]
    all_indices = train_indices + test_indices

    with open(os.path.join(real_set_dir, '{}_{}_real.txt'.format(cls_name, 'train')), 'w') as f:
        for line in train_indices:
            f.write(line + '\n')

    with open(os.path.join(real_set_dir, '{}_{}_real.txt'.format(cls_name, 'test')), 'w') as f:
        for line in test_indices:
            f.write(line + '\n')

    with open(os.path.join(real_set_dir, '{}_{}_real.txt'.format(cls_name, 'all')), 'w') as f:
        for line in all_indices:
            f.write(line + '\n')




















