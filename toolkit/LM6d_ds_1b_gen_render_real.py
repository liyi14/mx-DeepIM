# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
generate render real from syn_poses 
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

# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]) # for lm
ZNEAR = 0.25
ZFAR = 6.0

depth_factor = 1000

LINEMOD_root = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted')


def gen_render_real():
    syn_poses_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/syn_poses_single/')

    # output path
    render_real_root_dir = os.path.join(LINEMOD_root, 'LM6d_data_syn_light', 'data', 'render_real')
    image_set_dir = os.path.join(LINEMOD_root, 'LM6d_data_syn_light/image_set')
    mkdir_if_missing(render_real_root_dir)
    mkdir_if_missing(image_set_dir)

    syn_poses_path = os.path.join(syn_poses_dir, 'LM6d_ds_v1_all_syn_pose.pkl')
    with open(syn_poses_path, 'rb') as f:
        syn_pose_dict = cPickle.load(f)

    for class_idx, class_name in enumerate(tqdm(classes)):
        if class_name == '__back_ground__':
            continue
        if class_name in ['ape']:
            continue

        # init render machines
        # brightness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4] ###################
        model_dir = os.path.join(LINEMOD_root, 'models/{}'.format(class_name))
        render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        # syn_poses_path = os.path.join(syn_poses_dir, 'LM6d_v1_all_rendered_pose_{}.txt'.format(class_name))
        # syn_poses = np.loadtxt(syn_poses_path)
        # print(syn_poses.shape) # nx7
        syn_poses = syn_pose_dict[class_name]
        num_poses = syn_poses.shape[0]
        real_index_list = ['{}/{:06d}'.format(class_name, i+1) for i in range(num_poses)]

        # real_set_path = os.path.join(image_set_dir, 'real/LM_data_syn_train_real_{}.txt'.format(class_name))
        # mkdir_if_missing(os.path.join(image_set_dir, 'real'))
        # f_real_set = open(real_set_path, 'w')

        all_pair = []
        for idx, real_index in enumerate(real_index_list):
            # f_real_set.write('{}\n'.format(real_index))
            # continue # just generate real set file
            prefix = real_index.split('/')[1]
            video_name = real_index.split('/')[0]

            render_real_dir = os.path.join(render_real_root_dir, class_name)
            mkdir_if_missing(render_real_dir)


            render_real_color_file = os.path.join(render_real_dir, prefix+"-color.png")
            render_real_depth_file = os.path.join(render_real_dir, prefix+"-depth.png")
            render_real_pose_file = os.path.join(render_real_dir, prefix+"-pose.txt")

            # real_label_file = os.path.join(real_root_dir, video_name, prefix + "-label.png")
            render_real_label_file = os.path.join(render_real_dir, prefix + "-label.png")

            if idx % 500 == 0:
                print('  ', class_name, idx, '/', len(real_index_list), ' ', real_index)

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

            # randomly adjust color and intensity for light_intensity
            colors = np.array([[0, 0, 1],
                               [0, 1, 0],
                               [0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 1],
                               [1, 1, 0],
                               [1, 1, 1]])
            intensity = np.random.uniform(0.9, 1.1, size=(3,))
            colors_randk = random.randint(0, colors.shape[0] - 1)
            light_intensity = colors[colors_randk] * intensity
            # print('light intensity: ', light_intensity)



            # print('brightness ratio:', brightness_ratios[rm_randk])
            # get render result
            rgb_gl, depth_gl = render_machine.render(pose[:3, :3], pose[:, 3], r_type='mat')
            rgb_gl = rgb_gl.astype('uint8')
            # render_real label
            label_gl = np.zeros(depth_gl.shape)
            # print('depth gl:', depth_gl.shape)
            label_gl[depth_gl!=0] = 1


            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # plt.axis('off')
            # fig.add_subplot(1, 3, 1)
            # plt.imshow(rgb_gl[:, :, [2,1,0]])
            #
            # fig.add_subplot(1, 3, 2)
            # plt.imshow(depth_gl)
            #
            # fig.add_subplot(1, 3, 3)
            # plt.imshow(label_gl)
            #
            # fig.suptitle('light position: {}\n light_intensity: {}\n brightness: {}'.format(light_position, light_intensity, brightness_ratios[rm_randk]))
            # plt.show()


            cv2.imwrite(render_real_color_file, rgb_gl)
            depth_gl = (depth_gl * depth_factor).astype(np.uint16)
            cv2.imwrite(render_real_depth_file, depth_gl)

            cv2.imwrite(render_real_label_file, label_gl)

            text_file = open(render_real_pose_file, 'w')
            text_file.write("{}\n".format(class_idx))
            pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
                .format(pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3],
                        pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3],
                        pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3])
            text_file.write(pose_str)

        print(class_name, " done")


if __name__=='__main__':
    gen_render_real()
