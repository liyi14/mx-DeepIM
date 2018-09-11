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


render_real_dir = os.path.join(cur_path, '..', 'data',
                               'LINEMOD_6D/LM6d_occ_ds_multi/data/render_real')
real_set_dir = os.path.join(cur_path, '..',
                            'data/LINEMOD_6D/LM6d_occ_ds_multi/image_set/real')

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

# output dir
pose_dir = os.path.join(cur_path, '..', 'data/LINEMOD_6D/LM6d_occ_ds_multi/ds_rendered_poses')
mkdir_if_missing(pose_dir)

sel_classes = classes
num_rendered_per_real = 1 # 10
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
version = 'v1'
image_set = 'NDtrain'

def main():
    for cls_name in tqdm(sel_classes):
        print(cls_name)
        # if cls_name != 'driller':
        #     continue
        rd_stat = []
        td_stat = []
        pose_real = []
        pose_rendered = []

        real_set_file = os.path.join(real_set_dir, "NDtrain_real_{}.txt".format(cls_name))
        with open(real_set_file) as f:
            image_list = [x.strip() for x in f.readlines()]

        for data in image_list:
            pose_real_path = os.path.join(render_real_dir, cls_name, data+'-pose.txt')
            src_pose_m = np.loadtxt(pose_real_path, skiprows=1)

            src_euler = np.squeeze(mat2euler(src_pose_m[:, :3]))
            src_quat = euler2quat(src_euler[0], src_euler[1], src_euler[2]).reshape(1, -1)
            src_trans = src_pose_m[:, 3]
            pose_real.append((np.hstack((src_quat, src_trans.reshape(1, 3)))))

            for rendered_idx in range(num_rendered_per_real):
                tgt_euler = src_euler + np.random.normal(0, 15.0/180*pi, 3)
                x_error = np.random.normal(0, 0.01, 1)[0]
                y_error = np.random.normal(0, 0.01, 1)[0]
                z_error = np.random.normal(0, 0.05, 1)[0]
                tgt_trans = src_trans + np.array([x_error, y_error, z_error])
                tgt_pose_m = np.hstack((euler2mat(tgt_euler[0], tgt_euler[1], tgt_euler[2]),  tgt_trans.reshape((3, 1))))
                r_dist, t_dist = calc_rt_dist_m(tgt_pose_m, src_pose_m)
                transform = np.matmul(K, tgt_trans.reshape(3, 1))
                center_x = transform[0] / transform[2]
                center_y = transform[1] / transform[2]
                count = 0
                while (r_dist>45 or not(48<center_x<(640-48) and 48<center_y<(480-48))):
                    tgt_euler = src_euler + np.random.normal(0, 15.0/180*pi, 3)
                    x_error = np.random.normal(0, 0.01, 1)[0]
                    y_error = np.random.normal(0, 0.01, 1)[0]
                    z_error = np.random.normal(0, 0.05, 1)[0]
                    tgt_trans = src_trans + np.array([x_error, y_error, z_error])
                    tgt_pose_m = np.hstack((euler2mat(tgt_euler[0], tgt_euler[1], tgt_euler[2]), tgt_trans.reshape((3, 1))))
                    r_dist, t_dist = calc_rt_dist_m(tgt_pose_m, src_pose_m)
                    transform = np.matmul(K, tgt_trans.reshape(3, 1))
                    center_x = transform[0] / transform[2]
                    center_y = transform[1] / transform[2]
                    count += 1
                    if count == 100:
                        print(rendered_idx)
                        print("{}: {}, {}, {}, {}".format(data, r_dist, t_dist, center_x, center_y))
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


        output_file_name = os.path.join(pose_dir, 'LM6d_occ_dsm_{}_NDtrain_rendered_pose_{}.txt'.format(version, cls_name))
        with open(output_file_name, "w") as text_file:
            for x in pose_rendered:
                text_file.write("{}\n".format(' '.join(map(str, np.squeeze(x)))))


if __name__ == '__main__':
    main()
