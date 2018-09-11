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
from lib.utils.mkdir_if_missing import *
from lib.render_glumpy.render_py import Render_Py
import lib.pair_matching.RT_transform as se3
import cv2
from tqdm import tqdm


class_list = ['{:02d}'.format(i) for i in range(1, 31)]
sel_classes = ['05', '06']

# config for render machine
width = 640
height = 480
K = np.array([[1075.65091572, 0, 320.], [0, 1073.90347929, 240.], [0, 0, 1]]) # Primesense
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 10000


version = 'v1' ################################################################################# version
print('version', version)

TLESS_root = os.path.join(cur_dir, '../data/TLESS')
real_set_root = os.path.join(TLESS_root, 'TLESS_render_v3/image_set/real')
rendered_pose_path = "%s/TLESS_%s_{}_rendered_pose_{}.txt"%(os.path.join(TLESS_root, 'syn_poses_v3'),
                                                   version)

# output_path
rendered_root_dir = os.path.join(TLESS_root, 'TLESS_render_v3/data/rendered_{}'.format(version))
pair_set_dir = os.path.join(TLESS_root, 'TLESS_render_v3/image_set')
mkdir_if_missing(rendered_root_dir)
mkdir_if_missing(pair_set_dir)

def main():
    gen_images = True ##################################################################### gen_images
    for class_idx, class_name in enumerate(class_list):
        train_pair = []
        # val_pair = []
        if class_name in ['__back_ground__']:
            continue
        if not class_name in sel_classes:
            continue
        print("start ", class_name)

        if gen_images:
            # init render
            model_dir = os.path.join(TLESS_root, 'models', class_name)
            render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for set_type in ['train']:
            with open(os.path.join(real_set_root, '{}_{}.txt'.format(class_name, 'train')), 'r') as f:
                train_real_list = [x.strip() for x in f.readlines()]

            with open(rendered_pose_path.format(set_type, class_name)) as f:
                str_rendered_pose_list = [x.strip().split(' ') for x in f.readlines()]
            rendered_pose_list = np.array([[float(x) for x in each_pose] for each_pose in str_rendered_pose_list])
            rendered_per_real = 10
            assert (len(rendered_pose_list) == rendered_per_real*len(train_real_list)), \
                  '{} vs {}'.format(len(rendered_pose_list), len(train_real_list))
            for idx, real_index in enumerate(tqdm(train_real_list)):
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
                        pose_rendered_m[:, :3] = se3.quat2mat(pose_rendered_q[:4])
                        pose_rendered_m[:, 3] = pose_rendered_q[4:]
                        pose_ori_m = pose_rendered_m
                        pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}"\
                            .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                                    pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                                    pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
                        text_file.write(pose_str)


                    train_pair.append("{} {}/{}_{}"
                                          .format(real_index,
                                                  class_name, real_prefix, inner_idx))

            train_pair_set_file = os.path.join(pair_set_dir, "train_{}_{}.txt".format(version, class_name))
            train_pair.sort()
            with open(train_pair_set_file, "w") as text_file:
                for x in train_pair:
                    text_file.write("{}\n".format(x))


        print(class_name, " done")

def check_real_rendered():
    from lib.utils.utils import read_img
    import matplotlib.pyplot as plt

    real_dir = os.path.join(TLESS_root, 'TLESS_render_v3/data/real')

    for class_idx, class_name in enumerate(class_list):
        if not class_name in sel_classes:
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
    check_real_rendered()
