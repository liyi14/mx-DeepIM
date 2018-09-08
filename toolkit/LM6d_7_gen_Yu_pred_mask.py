# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import os
from lib.utils.mkdir_if_missing import *
import scipy.io as sio
import cv2

if __name__=='__main__':
    class_name_list = ['__back_ground__', 'ape', 'benchviseblue',
                      'camera', 'can', 'cat', 'driller',
                      'duck', 'eggbox', 'glue', 'holepuncher',
                      'iron', 'lamp', 'phone']
    big_idx2class = {
        1: 'ape',
        2: 'benchviseblue',
        4: 'camera',
        5: 'can',
        6: 'cat',
        8: 'driller',
        9: 'duck',
        10: 'eggbox',
        11: 'glue',
        12: 'holepuncher',
        13: 'iron',
        14: 'lamp',
        15: 'phone'
    }

    class2big_idx = {}
    for key in big_idx2class:
        class2big_idx[big_idx2class[key]] = key

    cur_path = os.path.abspath(os.path.dirname(__file__))

    # config for Yu's results
    keyframe_path = "%s/{}_test.txt"%(os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/image_set/real'))
    yu_pred_dir = os.path.join(cur_path, '../data/LINEMOD_6D/Yu_LINEMOD_SIXD_results')

    # config for renderer
    width = 640
    height = 480
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    ZNEAR = 0.25
    ZFAR = 6.0

    # output_path
    version = 'mask_Yu_v01'
    real_root_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/data/real')
    real_meta_path = "%s/{}-meta.mat"%(real_root_dir)
    rendered_root_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/data', version)
    pair_set_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/image_set')
    mkdir_if_missing(rendered_root_dir)
    mkdir_if_missing(pair_set_dir)
    all_pair = []
    for small_class_idx, class_name in enumerate(class_name_list):
        if small_class_idx == 0:
            continue
        big_class_idx = class2big_idx[class_name]
        with open(keyframe_path.format(class_name)) as f:
            real_index_list = [x.strip() for x in f.readlines()]
        video_name_list = [x.split('/')[0] for x in real_index_list]
        real_prefix_list = [x.split('/')[1] for x in real_index_list]

        # init render
        model_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/models/{}'.format(class_name))

        all_pair = []
        for idx, real_index in enumerate(real_index_list):
            rendered_dir = os.path.join(rendered_root_dir, video_name_list[idx], class_name)
            mkdir_if_missing(rendered_dir)
            label_file = os.path.join(rendered_dir, '{}-label.png'.format(real_prefix_list[idx]))
            if os.path.exists(label_file):
                continue
            yu_idx = idx
            yu_pred_file = os.path.join(yu_pred_dir, class_name, "{:04d}.mat".format(yu_idx))
            yu_pred = sio.loadmat(yu_pred_file)
            labels = yu_pred['labels']
            labels[labels==1] = big_class_idx
            cv2.imwrite(label_file, labels)

        print(class_name, " done")