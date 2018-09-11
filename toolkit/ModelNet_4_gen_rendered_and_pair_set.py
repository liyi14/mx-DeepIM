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

from lib.pair_matching.RT_transform import *
from lib.utils.mkdir_if_missing import mkdir_if_missing
from tqdm import tqdm
import random
import subprocess


classes = ['airplane', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'guitar',
           'laptop',
           'mantel', #'dresser',
             'piano', 'range_hood', 'sink', 'stairs',
           'stool', 'tent', 'toilet', 'tv_stand',
           'door', 'glass_box', 'wardrobe', 'plant', 'xbox',
           'bathtub', 'table', 'monitor', 'sofa', 'night_stand'
           ]
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
# seed_dict = {'airplane': 1, 'car': 2, 'chair': 3} # airplane
seed_dict = {'airplane': 1, 'car': 2, 'chair': 3,
              'bed': 4, 'bench': 5, 'bookshelf': 6,
             'guitar': 7, 'laptop': 8, 'mantel': 9,
            'piano': 10, 'range_hood': 11, 'sink': 12,
             'stairs': 13,
            'stool': 14, 'tent': 15, 'toilet': 16, 'tv_stand': 17,
            'door':18, 'glass_box':19, 'wardrobe':20, 'plant':21, 'xbox':22,
            'bathtub':23, 'table':24, 'monitor':25, 'sofa':26, 'night_stand':27
} # airplane

# init render machines
# brightness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4] ###################
modelnet_root = os.path.join(cur_dir, '../data/ModelNet')
modelnet40_root = os.path.join(modelnet_root, 'ModelNet40')
data_dir = os.path.join(modelnet_root, 'modelnet_render_v1/data/real')
real_set_dir = os.path.join(modelnet_root, 'modelnet_render_v1/image_set/real')
image_set_dir = os.path.join(modelnet_root, 'modelnet_render_v1/image_set')

rendered_data_dir = os.path.join(modelnet_root, 'modelnet_render_v1/data/rendered')
mkdir_if_missing(rendered_data_dir)

for cls_i, cls_name in enumerate(classes):
    if not cls_name in test_classes:
        continue
    print(cls_name)
    seed = seed_dict[cls_name]
    random.seed(seed)
    np.random.seed(seed)
    seed_list = [random.randint(0, 10000) for i in range(100000)]

    with open(os.path.join(real_set_dir, '{}_{}_real.txt'.format(cls_name, 'all'))) as f:
        all_indices = [line.strip() for line in f.readlines()]

    pair_set_dict = {'train': {}, 'my_val': {}}
    gen_images = False ##########################################

    for real_i, real_idx in enumerate(tqdm(all_indices)):
        cls_name_, set, prefix = real_idx.split('/')

        real_pose_path = os.path.join(data_dir, cls_name, set, '{}-pose.txt'.format(prefix))
        rendered_pose_dir = os.path.join(rendered_data_dir, cls_name, set)
        mkdir_if_missing(rendered_pose_dir)
        rendered_pose_path = os.path.join(rendered_pose_dir, '{}_0-pose.txt'.format(prefix))

        model_name = prefix.split('_')[0] + '_' + prefix.split('_')[1]
        if prefix.split('_')[0] in ['range', 'tv', 'glass', 'night']:
            model_name = prefix.split('_')[0] + '_' + prefix.split('_')[1] + '_' +prefix.split('_')[2]
        model_path = os.path.join(modelnet40_root, cls_name, set, '{}.obj'.format(model_name))
        # print(model_name)
        texture_path = os.path.join(modelnet_root, 'gray_texture.png')

        num_real = 50
        real_number = prefix.split('_')[-1]
        if real_number == '0000' and gen_images:
            cmd = 'python toolkit/ModelNet_4b_renderer.py --model_path {} --texture_path {} ' \
                  '--real_pose_path {} --rendered_pose_path {} ' \
                  '--num_real {} --seed {}'.format(
                        model_path, texture_path, real_pose_path,
                        rendered_pose_path, num_real, seed_list[real_i])
            output = subprocess.check_output(cmd.split())
            print(output)

        if set == 'train':
            if not model_name in pair_set_dict['train'].keys():
                pair_set_dict['train'][model_name] = ['{} {}'.format(real_idx, real_idx+'_0')]
            else:
                pair_set_dict['train'][model_name].append('{} {}'.format(real_idx, real_idx+'_0'))
        elif set == 'test':
            if not model_name in pair_set_dict['my_val'].keys():
                pair_set_dict['my_val'][model_name] = ['{} {}'.format(real_idx, real_idx+'_0')]
            else:
                pair_set_dict['my_val'][model_name].append('{} {}'.format(real_idx, real_idx + '_0'))

    for set, model_idx_dict in pair_set_dict.items():
        for model_name, indices in model_idx_dict.items():
            with open(os.path.join(image_set_dir, '{}_{}.txt'.format(set, model_name)), 'w') as f:
                for line in indices:
                    f.write(line + '\n')






















