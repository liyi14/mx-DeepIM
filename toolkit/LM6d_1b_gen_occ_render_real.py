# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division

import sys, os
from pprint import pprint
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '..'))
from glumpy import app, gl, gloo, glm, data, log
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.render_glumpy.render_py import Render_Py
import copy
import struct
import numpy as np
import scipy.io as sio
import cv2
from lib.pair_matching import RT_transform
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copyfile

'''
our data structure:
(render real)
ape/
    000001-color.png
    000001-depth.png
    000001-label.png
    000001-pose.txt
'''

# =================== global settings ======================
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


def class2idx(class_name, idx2class=idx2class):
    for k,v in idx2class.items():
        if v == class_name:
            return k

width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0

DEPTH_FACTOR = 1000

real_set_dir = os.path.join(cur_dir, '../data/LINEMOD_6D/LM6d_render_v1/image_set/real')
real_data_root = os.path.join(cur_dir, '../data/LINEMOD_6D/LM6d_render_v1/data/real')
LM6d_root = os.path.join(cur_dir, '../data/LINEMOD_6D')

# render real
render_real_root = os.path.join(cur_dir, '../data/LINEMOD_6D/LM6d_render_v1/data/render_real/occ_test/')
mkdir_if_missing(render_real_root)

# ==========================================================
def write_pose_file(pose_file, class_idx, pose_ori_m):
    text_file = open(pose_file, 'w')
    text_file.write("{}\n".format(class_idx))
    pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
        .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
    text_file.write(pose_str)


def gen_render_real():
    for cls_idx, cls_name in idx2class.items():
        print(cls_idx, cls_name)
        if cls_name == 'driller':
            continue
        with open(os.path.join(real_set_dir, 'occLM_val_real_{}.txt'.format(cls_name)), 'r') as f:
            all_indices = [line.strip('\r\n') for line in f.readlines()]

        # render machine
        model_dir = os.path.join(LM6d_root, 'models', cls_name)
        render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for real_idx in tqdm(all_indices):
            video_name, prefix = real_idx.split('/') # video name is "test"
            # read pose -------------------------------------
            real_meta_path = os.path.join(real_data_root, "02/{}-meta.mat".format(prefix))
            meta_data = sio.loadmat(real_meta_path)
            inner_id = np.where(np.squeeze(meta_data['cls_indexes']) == cls_idx)
            if len(meta_data['poses'].shape) == 2:
                pose = meta_data['poses']
            else:
                pose = np.squeeze(meta_data['poses'][:, :, inner_id])

            new_pose_path = os.path.join(render_real_root, cls_name, "{}-pose.txt".format(prefix))
            mkdir_if_missing(os.path.join(render_real_root, cls_name))
            # write pose
            write_pose_file(new_pose_path, cls_idx, pose)

            # ----------------------render color, depth ------------
            rgb_gl, depth_gl = render_machine.render(RT_transform.mat2quat(pose[:3, :3]), pose[:, -1])
            rgb_gl = rgb_gl.astype('uint8')
            render_color_path = os.path.join(render_real_root, cls_name, "{}-color.png".format(prefix))
            cv2.imwrite(render_color_path, rgb_gl)

            # depth
            depth_save = depth_gl * DEPTH_FACTOR
            depth_save = depth_save.astype('uint16')
            render_depth_path = os.path.join(render_real_root, cls_name, "{}-depth.png".format(prefix))
            cv2.imwrite(render_depth_path, depth_save)

            #--------------------- render label ----------------------------------
            render_label = depth_gl != 0
            render_label = render_label.astype('uint8')

            # write label
            label_path = os.path.join(render_real_root, cls_name, "{}-label.png".format(prefix))
            cv2.imwrite(label_path, render_label)


def check_render_real():
    cls_name = 'driller'
    real_indices = ['000001', '000003', '000010']
    for real_idx in real_indices:
        render_color_path = os.path.join(render_real_root, cls_name, "{}-color.png".format(real_idx))
        render_color = cv2.imread(render_color_path)
        render_depth_path = os.path.join(render_real_root, cls_name, "{}-depth.png".format(real_idx))
        render_depth = cv2.imread(render_depth_path, cv2.IMREAD_UNCHANGED)
        render_label_path = os.path.join(render_real_root, cls_name, "{}-label.png".format(real_idx))
        render_label = cv2.imread(render_label_path, cv2.IMREAD_UNCHANGED)

        cls_idx = class2idx(cls_name)
        cls_idx_str = "{:02d}".format(cls_idx)
        real_color_path = os.path.join(real_data_root, cls_idx_str, "{}-color.png".format(real_idx))
        real_color = cv2.imread(real_color_path)
        real_depth_path = os.path.join(real_data_root, cls_idx_str, "{}-depth.png".format(real_idx))
        real_depth = cv2.imread(real_depth_path, cv2.IMREAD_UNCHANGED)
        real_label_path = os.path.join(real_data_root, cls_idx_str, "{}-label.png".format(real_idx))
        real_label = cv2.imread(real_label_path, cv2.IMREAD_UNCHANGED)


        render_depth[0,0] = real_depth.max()
        plt.subplot(2, 3, 1)
        plt.imshow(render_color[:,:,[2,1,0]])

        plt.subplot(2, 3, 2)
        plt.imshow(render_depth)

        plt.subplot(2, 3, 3)
        plt.imshow(render_label)

        plt.subplot(2, 3, 4)
        plt.imshow(real_color[:,:,[2,1,0]])

        plt.subplot(2, 3, 5)
        plt.imshow(real_depth)

        plt.subplot(2, 3, 6)
        plt.imshow(real_label)
        plt.show()



if __name__ == "__main__":
    gen_render_real()
    check_render_real()
    pass