# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
input: render real poses
generate rendered poses,
'''
from __future__ import division, print_function
import numpy as np
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))

from lib.pair_matching.RT_transform import *
from math import pi
from lib.utils.mkdir_if_missing import mkdir_if_missing
from tqdm import tqdm
np.random.seed(2333)

# =================== global settings ======================
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

LINEMOD_root = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted')
# input render real poses
data_dir = os.path.join(LINEMOD_root, 'LM6d_data_syn_light', 'data', 'render_real')
image_set_root = os.path.join(LINEMOD_root, 'LM6d_data_syn_light/image_set/real')

# output: generated rendered poses
pose_dir = os.path.join(LINEMOD_root, 'data_syn_rendered_poses')
mkdir_if_missing(pose_dir)


sel_classes = classes
num_rendered_per_real = 1 # 10
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
# version = 'v1'
# generate a set of my val
version_params = {'v1': [15.0, 45.0, 0.01, 0.01, 0.05],
                  'v2': [20.0, 60.0, 0.01, 0.01, 0.05],
                  'v3': [30.0, 90.0, 0.01, 0.01, 0.05],
                  'v4': [40.0, 120.0, 0.01, 0.01, 0.05],
                  'v5': [60.0, 180.0, 0.01, 0.01, 0.05],

                  'v6': [15.0, 45.0, 0.02, 0.02, 0.05],
                  'v7': [15.0, 45.0, 0.03, 0.03, 0.05],
                  'v8': [15.0, 45.0, 0.04, 0.04, 0.05],
                  'v9': [15.0, 45.0, 0.05, 0.05, 0.05]}

version = 'v1'
angle_std, angle_max, x_std, y_std, z_std = version_params[version]
print('version: ', version)
print(angle_std, angle_max, x_std, y_std, z_std)
image_set = 'train'
for cls_idx, cls_name in enumerate(tqdm(sel_classes)):
    print(cls_idx, cls_name)
    rd_stat = []
    td_stat = []
    pose_real = []
    pose_rendered = []

    cls_idx_in_all = class2idx(cls_name)  #classes.index(cls_name)

    sel_set_file = os.path.join(image_set_root, "LM6d_data_syn_{}_real_{}.txt".format(image_set, cls_name))
    with open(sel_set_file) as f:
        image_list = [x.strip() for x in f.readlines()]

    for real_idx in tqdm(image_list):
        pose_real_path = os.path.join(data_dir, '{}-pose.txt'.format(real_idx))
        src_pose_m = np.loadtxt(pose_real_path, skiprows=1)

        src_euler = np.squeeze(mat2euler(src_pose_m[:3, :3]))
        src_quat = euler2quat(src_euler[0], src_euler[1], src_euler[2]).reshape(1, -1)
        src_trans = src_pose_m[:, 3]
        pose_real.append((np.hstack((src_quat, src_trans.reshape(1, 3)))))

        for rendered_idx in range(num_rendered_per_real):
            tgt_euler = src_euler + np.random.normal(0, angle_std/180*pi, 3)
            x_error = np.random.normal(0, x_std, 1)[0]
            y_error = np.random.normal(0, y_std, 1)[0]
            z_error = np.random.normal(0, z_std, 1)[0]
            tgt_trans = src_trans + np.array([x_error, y_error, z_error])
            tgt_pose_m = np.hstack((euler2mat(tgt_euler[0], tgt_euler[1], tgt_euler[2]),  tgt_trans.reshape((3, 1))))
            r_dist, t_dist = calc_rt_dist_m(tgt_pose_m, src_pose_m)
            transform = np.matmul(K, tgt_trans.reshape(3, 1))
            center_x = transform[0] / transform[2]
            center_y = transform[1] / transform[2]
            count = 0
            while (r_dist>angle_max or not(16<center_x<(640-16) and 16<center_y<(480-16))):
                tgt_euler = src_euler + np.random.normal(0, angle_std/180*pi, 3)
                x_error = np.random.normal(0, x_std, 1)[0]
                y_error = np.random.normal(0, y_std, 1)[0]
                z_error = np.random.normal(0, z_std, 1)[0]
                tgt_trans = src_trans + np.array([x_error, y_error, z_error])
                tgt_pose_m = np.hstack((euler2mat(tgt_euler[0], tgt_euler[1], tgt_euler[2]), tgt_trans.reshape((3, 1))))
                r_dist, t_dist = calc_rt_dist_m(tgt_pose_m, src_pose_m)
                transform = np.matmul(K, tgt_trans.reshape(3, 1))
                center_x = transform[0] / transform[2]
                center_y = transform[1] / transform[2]
                count += 1
                if count == 100:
                    print(rendered_idx)
                    print("{}: {}, {}, {}, {}".format(real_idx, r_dist, t_dist, center_x, center_y))
                    print("count: {}, image_path: {}, rendered_idx: {}".format(count,
                                                                           pose_real_path.replace('pose.txt', 'color.png'),
                                                                           rendered_idx))

            tgt_quat = euler2quat(tgt_euler[0], tgt_euler[1], tgt_euler[2]).reshape(1, -1)
            pose_rendered.append(np.hstack((tgt_quat, tgt_trans.reshape(1,3))))
            rd_stat.append(r_dist)
            td_stat.append(t_dist)
    rd_stat = np.array(rd_stat)
    td_stat = np.array(td_stat)
    print("r dist: {} +/- {}".format(np.mean(rd_stat), np.std(rd_stat)))
    print("t dist: {} +/- {}".format(np.mean(td_stat), np.std(td_stat)))


    output_file_name = os.path.join(pose_dir, 'LM6d_data_syn_{}_{}_rendered_pose_{}.txt'.format(version, image_set, cls_name))
    with open(output_file_name, "w") as text_file:
        for x in pose_rendered:
            text_file.write("{}\n".format(' '.join(map(str, np.squeeze(x)))))

