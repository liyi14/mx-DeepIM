# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
generate gt observed from syn_poses
'''
from __future__ import division, print_function
import numpy as np
import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))
from lib.utils.mkdir_if_missing import *
from lib.render_glumpy.render_py import Render_Py
import lib.pair_matching.RT_transform as se3
import cv2
from six.moves import cPickle
import random
from tqdm import tqdm
random.seed(2333)
np.random.seed(2333)

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
classes = idx2class.values()
classes = sorted(classes)


def class2idx(class_name, idx2class=idx2class):
    for k,v in idx2class.items():
        if v == class_name:
            return k

# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]) # for lm
ZNEAR = 0.25
ZFAR = 6.0

depth_factor = 1000

LINEMOD_root = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_refine')
LINEMOD_syn_root = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_refine_syn')
syn_poses_path = os.path.join(LINEMOD_syn_root, 'poses/LM6d_ds_train_observed_pose_all.pkl')

# output path
gt_observed_root_dir = os.path.join(LINEMOD_syn_root, 'data', 'gt_observed')
mkdir_if_missing(gt_observed_root_dir)

def gen_gt_observed():
    with open(syn_poses_path, 'rb') as f:
        syn_pose_dict = cPickle.load(f)

    for class_idx, class_name in enumerate(classes):
        if class_name == '__back_ground__':
            continue
        # uncomment here to only generate data for ape
        # if class_name not in ['ape']:
        #     continue

        # init render machines
        # brightness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4] ###################
        model_dir = os.path.join(LINEMOD_syn_root, 'models/{}'.format(class_name))
        render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        # syn_poses_path = os.path.join(syn_poses_dir, 'LM6d_v1_all_rendered_pose_{}.txt'.format(class_name))
        # syn_poses = np.loadtxt(syn_poses_path)
        # print(syn_poses.shape) # nx7
        syn_poses = syn_pose_dict[class_name]
        num_poses = syn_poses.shape[0]
        observed_index_list = ['{}/{:06d}'.format(class_name, i+1) for i in range(num_poses)]

        # observed_set_path = os.path.join(image_set_dir, 'observed/LM_data_syn_train_observed_{}.txt'.format(class_name))
        # mkdir_if_missing(os.path.join(image_set_dir, 'observed'))
        # f_observed_set = open(observed_set_path, 'w')

        all_pair = []
        for idx, observed_index in enumerate(tqdm(observed_index_list)):
            # f_observed_set.write('{}\n'.format(observed_index))
            # continue # just generate observed set file
            prefix = observed_index.split('/')[1]
            video_name = observed_index.split('/')[0]

            gt_observed_dir = os.path.join(gt_observed_root_dir, class_name)
            mkdir_if_missing(gt_observed_dir)


            gt_observed_color_file = os.path.join(gt_observed_dir, prefix+"-color.png")
            gt_observed_depth_file = os.path.join(gt_observed_dir, prefix+"-depth.png")
            gt_observed_pose_file = os.path.join(gt_observed_dir, prefix+"-pose.txt")

            # observed_label_file = os.path.join(observed_root_dir, video_name, prefix + "-label.png")
            gt_observed_label_file = os.path.join(gt_observed_dir, prefix + "-label.png")

            pose_quat = syn_poses[idx, :]
            pose = se3.se3_q2m(pose_quat)

            # generate random light_position
            if idx%6 == 0:
                light_position = [1, 0, 1]
            elif idx%6 == 1:
                light_position = [1, 1, 1]
            elif idx%6 == 2:
                light_position = [0, 1, 1]
            elif idx%6 == 3:
                light_position = [-1, 1, 1]
            elif idx%6 == 4:
                light_position = [-1, 0, 1]
            elif idx%6 == 5:
                light_position = [0, 0, 1]
            else:
                raise Exception("???")
            # print( "light_position a: {}".format(light_position))
            light_position=np.array(light_position)*0.5
            # inverse yz
            light_position[0] += pose[0, 3]
            light_position[1] -= pose[1, 3]
            light_position[2] -= pose[2, 3]
            # print("light_position b: {}".format(light_position))

            # get render result
            rgb_gl, depth_gl = render_machine.render(pose[:3, :3], pose[:, 3], r_type='mat')
            rgb_gl = rgb_gl.astype('uint8')
            # gt_observed label
            label_gl = np.zeros(depth_gl.shape)
            # print('depth gl:', depth_gl.shape)
            label_gl[depth_gl!=0] = 1

            cv2.imwrite(gt_observed_color_file, rgb_gl)
            depth_gl = (depth_gl * depth_factor).astype(np.uint16)
            cv2.imwrite(gt_observed_depth_file, depth_gl)

            cv2.imwrite(gt_observed_label_file, label_gl)

            text_file = open(gt_observed_pose_file, 'w')
            text_file.write("{}\n".format(class_idx))
            pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
                .format(pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3],
                        pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3],
                        pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3])
            text_file.write(pose_str)

        print(class_name, " done")


if __name__=='__main__':
    gen_gt_observed()
