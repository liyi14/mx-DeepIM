# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import numpy as np
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '..'))

import random
import subprocess

classes = ['airplane', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'guitar',
           'laptop', 'mantel', 'piano', 'range_hood', 'sink', 'stairs',
           'stool', 'tent', 'toilet', 'tv_stand',
           'door', 'glass_box', 'wardrobe', 'plant', 'xbox',
           'bathtub', 'table', 'monitor', 'sofa', 'night_stand']
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
modelnet_root = os.path.join(cur_dir, '../data/ModelNet')
modelnet40_root = os.path.join(modelnet_root, 'ModelNet40')

seed_dict = {'airplane': 1, 'car': 2, 'chair': 3,
              'bed': 4, 'bench': 5, 'bookshelf': 6,
             'guitar': 7, 'laptop': 8, 'mantel': 9,
            'piano': 10, 'range_hood': 11, 'sink': 12,
             'stairs': 13,
            'stool': 14, 'tent': 15, 'toilet': 16, 'tv_stand': 17,
            'door':18, 'glass_box':19, 'wardrobe':20, 'plant':21, 'xbox':22,
            'bathtub':23, 'table':24, 'monitor':25, 'sofa':26, 'night_stand':27
} # airplane



for cls_i, cls_name in enumerate(classes):
    if not cls_name in ['bathtub', 'table', 'monitor', 'sofa', 'night_stand']: # airplane
        continue
    print(cls_name)
    seed = seed_dict[cls_name]
    random.seed(seed)
    np.random.seed(seed)
    seed_list = [random.randint(0, 10000) for i in range(100000)]
    class_dir = os.path.join(modelnet40_root, cls_name)

    for set in ['train', 'test']:
        model_list = [fn for fn in os.listdir(os.path.join(class_dir, set)) if '.obj' in fn]
        model_list.sort()
        model_folder = os.path.join(class_dir, set)
        for model_i, model_name in enumerate(model_list):
            # print(model_name)
            model_prefix = model_name.split('.')[0]


            model_path = '{}/{}.obj'.format(model_folder, model_prefix)

            texture_path = os.path.join(modelnet_root, 'gray_texture.png')
            cmd = 'python toolkit/ModelNet_2b_renderer.py --model_path {} --texture_path {} ' \
                  '--seed {}'.format(model_path, texture_path, seed_list[model_i])
            # os.system(cmd)
            output = subprocess.check_output(cmd.split())
            print(output)





















