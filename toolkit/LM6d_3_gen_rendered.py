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
from lib.render_glumpy.render_py import Render_Py
import copy
import struct
import numpy as np
from shutil import copyfile
from lib.utils.mkdir_if_missing import *
from lib.render_glumpy.render_py import Render_Py
from lib.pair_matching import RT_transform
import cv2
from tqdm import tqdm



idx2class = {1: 'ape',
            2: 'benchviseblue',
            # 3: 'bowl',
            4: 'camera',
            5: 'can',
            6: 'cat',
            # 7: 'cup',
            8: 'driller',
            9: 'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            13: 'iron',
            14: 'lamp',
            15: 'phone'
}
# config for render machine
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000


version = 'v1' ################################################################################# version
print('version', version)

LM6d_root = os.path.join(cur_dir, '../data/LINEMOD_6D/LM6d_converted')
real_set_root = os.path.join(cur_dir, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/image_set/real')
rendered_pose_path = "%s/LM6d_%s_{}_rendered_pose_{}.txt"%(os.path.join(LM6d_root, 'syn_poses'),
                                                   version)

# output_path
rendered_root_dir = os.path.join(cur_dir, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/data/rendered_{}'.format(version))
pair_set_dir = os.path.join(LM6d_root, 'LM6d_render_v1/image_set')
mkdir_if_missing(rendered_root_dir)
mkdir_if_missing(pair_set_dir)

def main():
    gen_images = True ##################################################################### gen_images
    for class_idx, class_name in idx2class.items():
        train_pair = []
        val_pair = []
        print("start ", class_name)
        if class_name in ['__back_ground__']:
            continue

        if gen_images:
            # init render
            model_dir = os.path.join(LM6d_root, 'models', class_name)
            render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for set_type in ['all']:
            with open(os.path.join(real_set_root, '{}_{}.txt'.format(class_name, 'all')), 'r') as f:
                all_real_list = [x.strip() for x in f.readlines()]
            with open(os.path.join(real_set_root, '{}_{}.txt'.format(class_name, 'train')), 'r') as f:
                train_real_list = [x.strip() for x in f.readlines()]
            with open(os.path.join(real_set_root, '{}_{}.txt'.format(class_name, 'test')), 'r') as f:
                test_real_list = [x.strip() for x in f.readlines()]

            with open(rendered_pose_path.format(set_type, class_name)) as f:
                str_rendered_pose_list = [x.strip().split(' ') for x in f.readlines()]
            rendered_pose_list = np.array([[float(x) for x in each_pose] for each_pose in str_rendered_pose_list])
            rendered_per_real = 10
            assert (len(rendered_pose_list) == 10*len(all_real_list)), \
                  '{} vs {}'.format(len(rendered_pose_list), len(all_real_list))
            for idx, real_index in enumerate(tqdm(all_real_list)):
                video_name, real_prefix = real_index.split('/')
                rendered_dir = os.path.join(rendered_root_dir, class_name)
                mkdir_if_missing(rendered_dir)
                for inner_idx in range(rendered_per_real):
                    if gen_images:
                    # if gen_images and real_index in test_real_list and inner_idx == 0: # only generate my_val_v{}
                        image_file = os.path.join(rendered_dir,
                                                  '{}_{}-color.png'.format(real_prefix, inner_idx))
                        depth_file = os.path.join(rendered_dir,
                                                  '{}_{}-depth.png'.format(real_prefix, inner_idx))
                        rendered_idx = idx*rendered_per_real + inner_idx
                        pose_rendered_q = rendered_pose_list[rendered_idx]

                        rgb_gl, depth_gl = render_machine.render(pose_rendered_q[:4], pose_rendered_q[4:])
                        rgb_gl = rgb_gl.astype('uint8')

                        depth_gl = (depth_gl*depth_factor).astype(np.uint16)

                        cv2.imwrite(image_file, rgb_gl)
                        cv2.imwrite(depth_file, depth_gl)

                        pose_rendered_file = os.path.join(rendered_dir, '{}_{}-pose.txt'.format(real_prefix, inner_idx))
                        text_file = open(pose_rendered_file, 'w')
                        text_file.write("{}\n".format(class_idx))
                        pose_rendered_m = np.zeros((3, 4))
                        pose_rendered_m[:, :3] = RT_transform.quat2mat(pose_rendered_q[:4])
                        pose_rendered_m[:, 3] = pose_rendered_q[4:]
                        pose_ori_m = pose_rendered_m
                        pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}"\
                            .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                                    pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                                    pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
                        text_file.write(pose_str)

                    if real_index in test_real_list:
                        if inner_idx == 0:
                            val_pair.append("{} {}/{}_{}"
                                        .format(real_index,
                                                class_name, real_prefix, inner_idx))
                    else:
                        train_pair.append("{} {}/{}_{}"
                                          .format(real_index,
                                                  class_name, real_prefix, inner_idx))

            train_pair_set_file = os.path.join(pair_set_dir, "train_{}_{}.txt".format(version, class_name))
            train_pair = sorted(train_pair)
            with open(train_pair_set_file, "w") as text_file:
                for x in train_pair:
                    text_file.write("{}\n".format(x))

            test_pair_set_file = os.path.join(pair_set_dir, "my_val_{}_{}.txt".format(version, class_name))
            val_pair = sorted(val_pair)
            with open(test_pair_set_file, "w") as text_file:
                for x in val_pair:
                    text_file.write("{}\n".format(x))
        print(class_name, " done")

def check_real_rendered():
    from lib.utils.utils import read_img
    import matplotlib.pyplot as plt

    real_dir = os.path.join(cur_dir, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/data/real')

    for class_idx, class_name in idx2class.items():
        if class_name != 'duck':
            continue
        print(class_name)
        real_list_path = os.path.join(real_set_root, "{}_train.txt".format(class_name))
        with open(real_list_path, 'r') as f:
            real_list = [x.strip() for x in f.readlines()]
        for idx, real_index in enumerate(real_list):
            print(real_index)
            prefix = real_index.split('/')[1]
            color_real = read_img(os.path.join(real_dir, real_index + '-color.png'), 3)
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
    main()
    # check_real_rendered()
