# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
generate train_10k pairs
'''
from __future__ import print_function, division
import numpy as np
import os
import sys
from shutil import copyfile

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))
from tqdm import tqdm



if __name__=='__main__':
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


    # output_path
    version = 'v1' # -------------------------------------------

    LINEMOD_root = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted')

    pair_set_dir = os.path.join(LINEMOD_root, 'LM6d_data_syn_light/image_set')

    for class_idx, class_name in enumerate(tqdm(classes)):
        # val_pair = []
        print("start ", class_idx, class_name)
        if class_name in ['__back_ground__']:
            continue
        if class_name not in ['ape']:
            continue

        for set_type in ['train']:
            pair_set_file = os.path.join(pair_set_dir, "train_{}_{}.txt".format(version, class_name))
            with open(pair_set_file, "r") as text_file:
                train_pairs = [line.strip('\r\n') for line in text_file.readlines()]
                train_10k = train_pairs[:10000]

            pair_set_10k_file = os.path.join(pair_set_dir, "train_{}_10k_{}.txt".format(version, class_name))
            with open(pair_set_10k_file, "w") as text_file:
                for line in train_10k:
                    text_file.write(line + '\n')

        print(class_name, " done")
