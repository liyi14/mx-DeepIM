# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
generate rendered from rendered poses
generate (real rendered) pair list file for training (or test)
'''
from __future__ import print_function, division
import numpy as np
import os
import sys
from shutil import copyfile

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))

from lib.utils.mkdir_if_missing import *
from lib.render_glumpy.render_py import Render_Py
import lib.pair_matching.RT_transform as se3
import cv2
import random
random.seed(2333)
np.random.seed(2333)
from tqdm import tqdm

image_set_dir = os.path.join(cur_path, '..',
                             'data/LINEMOD_6D/LM6d_occ_ds_multi/image_set')
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

rendered_pose_dir = os.path.join(cur_path, '..',
                             'data/LINEMOD_6D/LM6d_occ_ds_multi/ds_rendered_poses')
# output dir
rendered_root_dir = os.path.join(cur_path, '..',
                             'data/LINEMOD_6D/LM6d_occ_ds_multi/data/rendered')
mkdir_if_missing(rendered_root_dir)

# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000

# output_path
version = 'v1'

def main():
    gen_images = True ##########################################################################
    for class_idx, class_name in enumerate(tqdm(classes)):
        train_pair = []
        print("start ", class_name)
        if class_name in ['__back_ground__']:
            continue
        # if class_name != 'driller':
        #     continue

        if gen_images:
            # init render machine
            model_dir = os.path.join(cur_path, '../data/LINEMOD_6D/models/{}'.format(class_name))
            render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for set_type in ['NDtrain']:
            # real index list
            real_list_path = os.path.join(real_set_dir, "NDtrain_real_{}.txt".format(class_name))
            with open(real_list_path, 'r') as f:
                real_list = [x.strip() for x in f.readlines()]

            # rendered poses
            rendered_pose_path = os.path.join(rendered_pose_dir,
                                          "LM6d_occ_dsm_{}_NDtrain_rendered_pose_{}.txt".format(version,
                                                                                   class_name))
            with open(rendered_pose_path, 'r') as f:
                str_rendered_pose_list = [x.strip().split(' ') for x in f.readlines()]
            rendered_pose_list = np.array([[float(x) for x in each_pose] for each_pose in str_rendered_pose_list])

            rendered_per_real = 1
            assert(len(rendered_pose_list) == 1*len(real_list)), '{} vs {}'.format(len(rendered_pose_list), len(real_list))

            for idx, real_index in enumerate(tqdm(real_list)):
                video_name, real_prefix = real_index.split('/') # ./prefix
                rendered_dir = os.path.join(rendered_root_dir, video_name)
                mkdir_if_missing(rendered_dir)
                rendered_dir = os.path.join(rendered_dir, class_name)
                mkdir_if_missing(rendered_dir)

                for inner_idx in range(rendered_per_real):
                    if gen_images:
                        image_file = os.path.join(rendered_dir,
                                                  '{}_{}-color.png'.format(real_prefix, inner_idx))
                        depth_file = os.path.join(rendered_dir,
                                                  '{}_{}-depth.png'.format(real_prefix, inner_idx))
                        # if os.path.exists(image_file) and os.path.exists(depth_file):
                        #     continue
                        rendered_idx = idx*rendered_per_real + inner_idx
                        pose_rendered_q = rendered_pose_list[rendered_idx]

                        rgb_gl, depth_gl = render_machine.render(pose_rendered_q[:4], pose_rendered_q[4:])
                        rgb_gl = rgb_gl.astype('uint8')

                        depth_gl = (depth_gl*depth_factor).astype(np.uint16)

                        cv2.imwrite(image_file, rgb_gl)
                        cv2.imwrite(depth_file, depth_gl)

                        pose_rendered_file = os.path.join(rendered_dir, '{}_{}-pose.txt'.format(real_prefix, inner_idx))
                        text_file = open(pose_rendered_file, 'w')
                        text_file.write("{}\n".format(class2idx(class_name)))
                        pose_rendered_m = np.zeros((3, 4))
                        pose_rendered_m[:, :3] = se3.quat2mat(pose_rendered_q[:4])
                        pose_rendered_m[:, 3] = pose_rendered_q[4:]
                        pose_ori_m = pose_rendered_m
                        pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}"\
                            .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                                    pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                                    pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
                        text_file.write(pose_str)

                    train_pair.append("{} {}/{}_{}".format(real_index,
                                                  class_name, real_prefix, inner_idx))

            pair_set_file = os.path.join(image_set_dir, "train_{}.txt".format(class_name))
            train_pair.sort()
            with open(pair_set_file, "w") as text_file:
                for x in train_pair:
                    text_file.write("{}\n".format(x))

        print(class_name, " done")

def check_real_rendered():
    from lib.utils.utils import read_img
    import matplotlib.pyplot as plt

    real_dir = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D/LM6d_occ_ds_multi/data/real')

    for class_idx, class_name in enumerate(tqdm(classes)):
        if class_name != 'driller':
            continue
        print(class_name)
        real_list_path = os.path.join(real_set_dir, "NDtrain_real_{}.txt".format(class_name))
        with open(real_list_path, 'r') as f:
            real_list = [x.strip() for x in f.readlines()]
        for idx, real_index in enumerate(real_list):
            print(real_index)
            prefix = real_index.split('/')[1]
            color_real = read_img(os.path.join(real_dir, prefix+'-color.png'), 3)
            color_rendered = read_img(os.path.join(rendered_root_dir, class_name, prefix + '_0-color.png'), 3)
            fig = plt.figure()
            plt.axis('off')
            plt.subplot(1,2,1)
            plt.imshow(color_real[:,:,[2,1,0]])
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.imshow(color_rendered[:,:,[2,1,0]])
            plt.axis('off')

            plt.show()



if __name__ == "__main__":
    # main()
    check_real_rendered()
