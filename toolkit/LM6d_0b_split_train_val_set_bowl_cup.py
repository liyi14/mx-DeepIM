# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang
# --------------------------------------------------------
'''
generate real set
'''
from __future__ import print_function, division
import numpy as np
import os
import sys
from shutil import copyfile

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))

from lib.utils.mkdir_if_missing import *
import random
random.seed(2333)
np.random.seed(2333)
from tqdm import tqdm

real_dir = os.path.join(cur_path, '..',
                             'data/LINEMOD_6D/LM6d_render_v1/data/real')

image_set_dir = os.path.join(cur_path, '..',
                             'data/LINEMOD_6D/LM6d_render_v1/image_set')
real_set_dir = os.path.join(cur_path, '..',
                            'data/LINEMOD_6D/LM6d_render_v1/image_set/real')
idx2class = {1: 'ape',
            2: 'benchviseblue',
            3: 'bowl',
            4: 'camera',
            5: 'can',
            6: 'cat',
            7: 'cup',
            8: 'driller',
            9: 'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            13: 'iron',
            14: 'lamp',
            15: 'phone'
}
classes = idx2class.values()
classes = sorted(classes)


def class2idx(class_name, idx2class=idx2class):
    for k,v in idx2class.items():
        if v == class_name:
            return k


def main():
    for class_idx, class_name in enumerate(tqdm(classes)):
        print("start ", class_name)
        if class_name in ['__back_ground__']:
            continue
        if not class_name in ['bowl', 'cup']:
            continue

        real_indices = ['{}/{}'.format("{:02d}".format(class2idx(class_name)), fn.split('-')[0])
                        for fn in os.listdir(os.path.join(real_dir, "{:02d}".format(class2idx(class_name))))
                        if 'color' in fn]
        real_indices = sorted(real_indices)

        random.shuffle(real_indices)

        train_num = int(0.15 * len(real_indices))
        train_indices = real_indices[:train_num]
        train_indices = sorted(train_indices)

        val_indices = []
        for real_idx in real_indices:
            if not real_idx in train_indices:
                val_indices.append(real_idx)

        print('class: {}, all: {}, val: {}, train: {}, train/all: {}'.format(class_name,
                                                                           len(real_indices),
                                                                           len(val_indices),
                                                                           len(train_indices),
                                                                           len(train_indices)/len(real_indices)))

        with open(os.path.join(real_set_dir, '{}_train.txt'.format(class_name)), 'w') as f:
            for real_index in train_indices:
                f.write(real_index+'\n')

        with open(os.path.join(real_set_dir, '{}_all.txt'.format(class_name)), 'w') as f:
            for real_index in real_indices:
                f.write(real_index+'\n')

        with open(os.path.join(real_set_dir, '{}_test.txt'.format(class_name)), 'w') as f:
            for real_index in val_indices:
                f.write(real_index+'\n')


if __name__ == "__main__":
    main()
