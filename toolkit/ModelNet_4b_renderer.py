# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import numpy as np
import os, sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))
from lib.pair_matching.RT_transform import *
from math import pi
import random
from lib.render_glumpy.render_py_light_modelnet import Render_Py_Light_ModelNet
import cv2


classes = ['airplane', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'guitar',
           'laptop',
           'mantel', #'dresser',
             'piano', 'range_hood', 'sink', 'stairs',
           'stool', 'tent', 'toilet', 'tv_stand',
           'door', 'glass_box', 'wardrobe', 'plant', 'xbox',
           'bathtub', 'table', 'monitor', 'sofa', 'night_stand']
# print(classes)


# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]) # LM
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000

modelnet_root = '/data/wanggu/Downloads/modelnet' # NB: change to your dir
modelnet40_root = os.path.join(modelnet_root, 'ModelNet40')

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='renderer')
    parser.add_argument('--model_path', required=True, help='model path')
    parser.add_argument('--texture_path', required=True, help='texture path')
    parser.add_argument('--real_pose_path', required=True, help='real pose path')
    parser.add_argument('--rendered_pose_path', required=True, help='rendered pose path')
    parser.add_argument('--num_real', required=True, type=int, help='num real per model')
    parser.add_argument('--seed', required=True, type=int, help='seed')

    args = parser.parse_args()
    return args

args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

model_path = args.model_path
texture_path = args.texture_path
real_pose_path = args.real_pose_path
rendered_pose_path = args.rendered_pose_path
num_real = args.num_real

def angle_axis_to_quat(angle, rot_axis):
    angle = angle % (2 * np.pi)
    # print(angle)
    q = np.zeros(4)
    q[0] = np.cos(0.5 * angle)
    q[1:] = np.sin(0.5 * angle) * rot_axis
    if q[0] < 0:
        q *= -1
    # print('norm of q: ', LA.norm(q))
    q = q / np.linalg.norm(q)
    # print('norm of q: ', LA.norm(q))
    return q

def angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))  # -> cosine of the angle
    rad = np.arccos(np.clip(c, -1, 1))
    deg = rad / np.pi * 180
    return deg





# init render machines
brightness_ratios = [0.7] #[0.6, 0.7, 0.8] ###################


render_machine = Render_Py_Light_ModelNet(model_path, texture_path, K, width, height, ZNEAR, ZFAR, brightness_ratios)



def gen_rendered():
    # -----------------
    model_folder_string_list = model_path.split('/')
    cls_name = model_folder_string_list[-3]
    cls_idx = classes.index(cls_name) + 1
    # set = model_folder_string_list[-2]

    # model_prefix = model_folder_string_list[-1].split('.')[0]
    real_dir = os.path.dirname(real_pose_path)
    real_base_name = os.path.basename(real_pose_path)
    # print(real_pose_path)
    model_name = real_base_name.split('-')[0]
    # print(model_name)
    if model_name.split('_')[0] in ['range', 'tv', 'glass', 'night']:
        model_name = model_name.split('_')[0] + '_' + model_name.split('_')[1] + '_' + model_name.split('_')[2]
    else:
        model_name = model_name.split('_')[0] + '_' + model_name.split('_')[1]


    rendered_dir = os.path.dirname(rendered_pose_path)
    rendered_base_name = os.path.basename(rendered_pose_path)
    rendered_model_name = rendered_base_name.split('-')[0]
    if rendered_model_name.split('_')[0] in ['range', 'tv', 'glass', 'night']:
        rendered_model_name = rendered_model_name.split('_')[0] + '_' + rendered_model_name.split('_')[1] + '_' + rendered_model_name.split('_')[2]
    else:
        rendered_model_name = rendered_model_name.split('_')[0] + '_' + rendered_model_name.split('_')[1]

    for real_i in range(num_real):
        new_real_pose_path = os.path.join(real_dir, model_name + '_' + '{:04d}-pose.txt'.format(real_i))
        new_rendered_pose_path = os.path.join(rendered_dir, rendered_model_name + '_' + '{:04d}_0-pose.txt'.format(real_i))
        src_pose_m = np.loadtxt(new_real_pose_path, skiprows=1)
        src_euler = np.squeeze(mat2euler(src_pose_m[:, :3]))
        # src_quat = euler2quat(src_euler[0], src_euler[1], src_euler[2]).reshape(1, -1)
        src_trans = src_pose_m[:, 3]

        tgt_euler = src_euler + np.random.normal(0, 15.0 / 180 * pi, 3)
        x_error = np.random.normal(0, 0.01, 1)[0]
        y_error = np.random.normal(0, 0.01, 1)[0]
        z_error = np.random.normal(0, 0.05, 1)[0]
        tgt_trans = src_trans + np.array([x_error, y_error, z_error])
        tgt_pose_m = np.hstack((euler2mat(tgt_euler[0], tgt_euler[1], tgt_euler[2]), tgt_trans.reshape((3, 1))))
        r_dist, t_dist = calc_rt_dist_m(tgt_pose_m, src_pose_m)
        transform = np.matmul(K, tgt_trans.reshape(3, 1))
        center_x = transform[0] / transform[2]
        center_y = transform[1] / transform[2]
        count = 0
        while (r_dist > 45 or not (48 < center_x < (640 - 48) and 48 < center_y < (480 - 48))):
            tgt_euler = src_euler + np.random.normal(0, 15.0 / 180 * pi, 3)
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
                print("{}: {}, {}, {}, {}".format(rendered_pose_path, r_dist, t_dist, center_x, center_y))
                print("count: {}, image_path: {}".format(count,
                                                                       real_pose_path.replace('pose.txt', 'color.png')))

        # ---------------------------------------------------------------------------------
        pose = tgt_pose_m.copy()

        idx = 2 #random.randint(0, 100)

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
        # print("light_position a: {}".format(light_position))
        light_position = np.array(light_position) * 0.5
        # inverse yz
        light_position[0] += pose[0, 3]
        light_position[1] -= pose[1, 3]
        light_position[2] -= pose[2, 3]
        # print("light_position b: {}".format(light_position))

        # randomly adjust color and intensity for light_intensity
        # colors = np.array([[0, 0, 1],
        #                    [0, 1, 0],
        #                    [0, 1, 1],
        #                    [1, 0, 0],
        #                    [1, 0, 1],
        #                    [1, 1, 0],
        #                    [1, 1, 1]])
        colors = np.array([1, 1, 1])  # white light
        intensity = np.random.uniform(0.9, 1.1, size=(3,))
        colors_randk = random.randint(0, colors.shape[0] - 1)
        light_intensity = colors[colors_randk] * intensity
        # print('light intensity: ', light_intensity)

        # randomly choose a render machine
        rm_randk = random.randint(0, len(brightness_ratios) - 1)
        # print('brightness ratio:', brightness_ratios[rm_randk])
        # get render result
        rgb_gl, depth_gl = render_machine.render(mat2quat(pose[:3, :3]), pose[:, -1],
                                                 light_position,
                                                 light_intensity,
                                                 brightness_k=rm_randk)
        rgb_gl = rgb_gl.astype('uint8')
        # render_real label
        label_gl = np.zeros(depth_gl.shape)
        # print('depth gl:', depth_gl.shape)
        label_gl[depth_gl != 0] = 1
        depth_res = (depth_gl * depth_factor).astype('uint16')


        # write results -------------------------------------
        rendered_color_file = new_rendered_pose_path.replace('pose.txt', 'color.png')
        rendered_depth_file = new_rendered_pose_path.replace('pose.txt', 'depth.png')
        rendered_label_file = new_rendered_pose_path.replace('pose.txt', 'label.png')

        cv2.imwrite(rendered_color_file, rgb_gl)
        cv2.imwrite(rendered_depth_file, depth_res)
        cv2.imwrite(rendered_label_file, label_gl)

        text_file = open(new_rendered_pose_path, 'w')
        text_file.write("{}\n".format(cls_idx))
        pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
            .format(pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3],
                    pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3],
                    pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3])
        text_file.write(pose_str)

        real_color = cv2.imread(new_real_pose_path.replace('pose.txt', 'color.png'), cv2.IMREAD_COLOR)

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(8, 6), dpi=200)
        # plt.axis('off')
        # fig.add_subplot(2, 3, 1)
        # plt.imshow(real_color[:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.title('color real')
        #
        # fig.add_subplot(2, 3, 2)
        # plt.imshow(rgb_gl[:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.title('color rendered')
        #
        # plt.show()


if __name__ == "__main__":
    gen_rendered()
    pass















