# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
generate pair set
'''
from __future__ import print_function, division
import numpy as np
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))

from lib.utils.mkdir_if_missing import *

import random
random.seed(2333)
np.random.seed(2333)
from tqdm import tqdm

origin_set_dir = os.path.join(cur_path, '../data/occLM_render_v1/image_set')

image_set_dir = os.path.join(cur_path, '..',
                             'data/LINEMOD_6D/LM6d_occ_render_v1/image_set')

idx2class = {1: 'ape',
            # 2: 'benchviseblue',
            # 4: 'camera',
            5: 'can',
            6: 'cat',
            8: 'driller',
            9: 'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            # 13: 'iron',
            # 14: 'lamp',
            # 15: 'phone'
}
classes = idx2class.values()
classes.sort()


def class2idx(class_name, idx2class=idx2class):
    for k,v in idx2class.items():
        if v == class_name:
            return k


def main():
    for class_idx, class_name in enumerate(tqdm(classes)):
        print("start ", class_idx, class_name)
        if class_name in ['__back_ground__']:
            continue

        for set_type in ['train']:
            f_origin = os.path.join(origin_set_dir, '{}_{}.txt'.format(set_type, class_name))
            pairs = []
            with open(f_origin, 'r') as f:
                for line in f:
                    real_idx, rendered_idx = line.strip('\r\n').split()
                    video_name, real_prefix = real_idx.split('/')
                    if video_name == 'test':
                        new_video_name = '02'
                    else:
                        cls_idx = class2idx(class_name)
                        new_video_name = '{:02d}'.format(cls_idx)
                    new_real_idx = '{}/{}'.format(new_video_name, real_prefix)

                    rendered_video, rendered_prefix = rendered_idx.split('/')
                    rendered_prefix = rendered_prefix.replace(class_name+'_', '')
                    new_rendered_idx = '{}/{}'.format(rendered_video, rendered_prefix)

                    pairs.append('{} {}'.format(new_real_idx, new_rendered_idx))


            with open(os.path.join(image_set_dir, '{}_{}.txt'.format(set_type, class_name)), 'w') as f:
                for pair in pairs:
                    f.write(pair+'\n')


if __name__ == "__main__":
    main()
