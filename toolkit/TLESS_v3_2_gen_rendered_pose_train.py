# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import sys, os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '..'))
import numpy as np
from lib.pair_matching.RT_transform import *
from math import pi
from lib.utils.mkdir_if_missing import mkdir_if_missing
from tqdm import tqdm

np.random.seed(1234)


# =================== global settings ======================
class_list = ['{:02d}'.format(i) for i in range(1, 31)]
sel_classes = ['05', '06']


TLESS_root = os.path.join(cur_dir, '../data/TLESS')
origin_data_root = os.path.join(cur_dir, '../data/TLESS/train_render_reconst')

K = np.array([[1075.65091572, 0, 320.], [0, 1073.90347929, 240.], [0, 0, 1]]) # Primesense
ZNEAR = 0.25
ZFAR = 6.0

DEPTH_FACTOR = 10000

real_set_root = os.path.join(TLESS_root, 'TLESS_render_v3/image_set/real')

real_data_root = os.path.join(TLESS_root, 'TLESS_render_v3/data/real')



# output path
pose_dir = os.path.join(TLESS_root, 'syn_poses_v3')
mkdir_if_missing(pose_dir)

num_rendered_per_real = 10
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
for cls_idx, cls_name in enumerate(class_list):
    if not cls_name in sel_classes:
        continue
    print(cls_idx, cls_name)
    rd_stat = []
    td_stat = []
    pose_real = []
    pose_rendered = []

    sel_set_file = os.path.join(real_set_root, "{}_{}.txt".format(cls_name, image_set))
    with open(sel_set_file) as f:
        image_list = [x.strip() for x in f.readlines()]

    for real_idx in tqdm(image_list):
        # get src_pose_m
        video_name, prefix = real_idx.split('/')
        pose_path = os.path.join(real_data_root, cls_name, "{}-pose.txt".format(prefix))
        src_pose_m = np.loadtxt(pose_path, skiprows=1)

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

            tgt_quat = euler2quat(tgt_euler[0], tgt_euler[1], tgt_euler[2]).reshape(1, -1)
            pose_rendered.append(np.hstack((tgt_quat, tgt_trans.reshape(1,3))))
            rd_stat.append(r_dist)
            td_stat.append(t_dist)
    rd_stat = np.array(rd_stat)
    td_stat = np.array(td_stat)
    print("r dist: {} +/- {}".format(np.mean(rd_stat), np.std(rd_stat)))
    print("t dist: {} +/- {}".format(np.mean(td_stat), np.std(td_stat)))


    output_file_name = os.path.join(pose_dir, 'TLESS_{}_{}_rendered_pose_{}.txt'.format(version, image_set, cls_name))
    with open(output_file_name, "w") as text_file:
        for x in pose_rendered:
            text_file.write("{}\n".format(' '.join(map(str, np.squeeze(x)))))