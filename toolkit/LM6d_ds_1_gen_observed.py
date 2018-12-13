# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
generate observed light from syn_poses
'''
from __future__ import division, print_function
import numpy as np
import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.render_glumpy.render_py_light import Render_Py_Light
import lib.pair_matching.RT_transform as se3
import cv2
from six.moves import cPickle
import random
from tqdm import tqdm
random.seed(2333)
np.random.seed(2333)

idx2class = {
    1: 'ape',
    2: 'benchvise',
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
classes = idx2class.values()
classes = sorted(classes)


def class2idx(class_name, idx2class=idx2class):
    for k, v in idx2class.items():
        if v == class_name:
            return k


# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899],
              [0, 0, 1]])  # for lm
ZNEAR = 0.25
ZFAR = 6.0

depth_factor = 1000

LINEMOD_syn_root = os.path.join(
    cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_refine_syn')
observed_pose_dir = os.path.join(LINEMOD_syn_root, 'poses')


def gen_observed():
    # output path
    observed_root_dir = os.path.join(LINEMOD_syn_root, 'data', 'observed')
    image_set_dir = os.path.join(LINEMOD_syn_root, 'image_set')
    mkdir_if_missing(observed_root_dir)
    mkdir_if_missing(image_set_dir)

    syn_poses_path = os.path.join(observed_pose_dir,
                                  'LM6d_ds_train_observed_pose_all.pkl')
    with open(syn_poses_path, 'rb') as f:
        syn_pose_dict = cPickle.load(f)

    for class_idx, class_name in enumerate(classes):
        if class_name == '__back_ground__':
            continue
        # uncomment here to only generate data for ape
        # if class_name not in ['ape']:
        #     continue

        # init render machines
        brightness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4]
        model_dir = os.path.join(LINEMOD_syn_root, 'models', class_name)
        render_machine = Render_Py_Light(model_dir, K, width, height, ZNEAR,
                                         ZFAR, brightness_ratios)

        syn_poses = syn_pose_dict[class_name]
        num_poses = syn_poses.shape[0]
        observed_index_list = [
            '{}/{:06d}'.format(class_name, i + 1) for i in range(num_poses)
        ]

        observed_set_path = os.path.join(
            image_set_dir,
            'observed/LM6d_data_syn_train_observed_{}.txt'.format(class_name))
        mkdir_if_missing(os.path.join(image_set_dir, 'observed'))
        f_observed_set = open(observed_set_path, 'w')

        for idx, observed_index in enumerate(tqdm(observed_index_list)):
            f_observed_set.write('{}\n'.format(observed_index))
            prefix = observed_index.split('/')[1]

            observed_dir = os.path.join(observed_root_dir, class_name)
            mkdir_if_missing(observed_dir)

            observed_color_file = os.path.join(observed_dir,
                                               prefix + "-color.png")
            observed_depth_file = os.path.join(observed_dir,
                                               prefix + "-depth.png")
            observed_pose_file = os.path.join(observed_dir,
                                              prefix + "-pose.txt")

            observed_label_file = os.path.join(observed_dir,
                                               prefix + "-label.png")

            pose_quat = syn_poses[idx, :]
            pose = se3.se3_q2m(pose_quat)

            # generate random light_position
            if idx % 6 == 0:
                light_position = [1, 0, 1]
            elif idx % 6 == 1:
                light_position = [1, 1, 1]
            elif idx % 6 == 2:
                light_position = [0, 1, 1]
            elif idx % 6 == 3:
                light_position = [-1, 1, 1]
            elif idx % 6 == 4:
                light_position = [-1, 0, 1]
            elif idx % 6 == 5:
                light_position = [0, 0, 1]
            else:
                raise Exception("???")
            light_position = np.array(light_position) * 0.5
            # inverse yz
            light_position[0] += pose[0, 3]
            light_position[1] -= pose[1, 3]
            light_position[2] -= pose[2, 3]

            # randomly adjust color and intensity for light_intensity
            colors = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                               [1, 0, 1], [1, 1, 0], [1, 1, 1]])
            intensity = np.random.uniform(0.9, 1.1, size=(3, ))
            colors_randk = random.randint(0, colors.shape[0] - 1)
            light_intensity = colors[colors_randk] * intensity

            # randomly choose a render machine
            rm_randk = random.randint(0, len(brightness_ratios) - 1)
            # get render result
            rgb_gl, depth_gl = render_machine.render(
                se3.mat2quat(pose[:3, :3]),
                pose[:, -1],
                light_position,
                light_intensity,
                brightness_k=rm_randk)
            rgb_gl = rgb_gl.astype('uint8')
            # gt_observed label
            label_gl = np.zeros(depth_gl.shape)
            # print('depth gl:', depth_gl.shape)
            label_gl[depth_gl != 0] = 1

            cv2.imwrite(observed_color_file, rgb_gl)
            depth_gl = (depth_gl * depth_factor).astype(np.uint16)
            cv2.imwrite(observed_depth_file, depth_gl)

            cv2.imwrite(observed_label_file, label_gl)

            text_file = open(observed_pose_file, 'w')
            text_file.write("{}\n".format(class_idx))
            pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
                .format(pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3],
                        pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3],
                        pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3])
            text_file.write(pose_str)

        print(class_name, " done")


if __name__ == '__main__':
    gen_observed()
    print("{} finished".format(__file__))
