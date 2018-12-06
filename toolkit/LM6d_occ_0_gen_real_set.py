# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
generate real set
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

image_set_dir = os.path.join(cur_path, '..',
                             'data/LINEMOD_6D/LM6d_occ_render_v1/image_set')
real_set_dir = os.path.join(cur_path, '..',
                            'data/LINEMOD_6D/LM6d_occ_render_v1/image_set/real')
idx2class = {1: 'ape',
            # 2: 'benchvise',
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
        print("start ", class_name)
        if class_name in ['__back_ground__']:
            continue

        for set_type in ['all', 'train', 'val']:
            f_real_origin = os.path.join(real_set_dir, 'occLM_{}_real_{}.txt'.format(set_type, class_name))
            real_indices = []
            with open(f_real_origin, 'r') as f:
                for line in f:
                    video_name, prefix = line.strip('\r\n').split('/')
                    if video_name == 'test':
                        new_video_name = '02'
                    else:
                        cls_idx = class2idx(class_name)
                        new_video_name = '{:02d}'.format(cls_idx)
                    new_real_idx = '{}/{}'.format(new_video_name, prefix)
                    real_indices.append(new_real_idx)

            with open(os.path.join(real_set_dir, '{}_{}.txt'.format(class_name, set_type)), 'w') as f:
                for real_index in real_indices:
                    f.write(real_index+'\n')




if __name__ == "__main__":
    main()
