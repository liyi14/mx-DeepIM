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
from lib.utils.mkdir_if_missing import mkdir_if_missing
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

modelnet_root = os.path.join(cur_path, '../data/ModelNet')
modelnet40_root = os.path.join(modelnet_root, 'ModelNet40')

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='renderer')
    parser.add_argument('--model_path', required=True, help='model path')
    parser.add_argument('--texture_path', required=True, help='texture path')
    parser.add_argument('--seed', required=True, type=int, help='seed')

    args = parser.parse_args()
    return args

args = parse_args()



model_path = args.model_path
texture_path = args.texture_path
random.seed(args.seed)
np.random.seed(args.seed)

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
brightness_ratios = [0.7] ###################

render_machine = Render_Py_Light_ModelNet(model_path, texture_path, K, width, height, ZNEAR, ZFAR, brightness_ratios)


def gen_real():
    data_dir = os.path.join(modelnet_root, 'modelnet_render_v1/data/real')
    mkdir_if_missing(data_dir)
    real_set_dir = os.path.join(modelnet_root, 'modelnet_render_v1/image_set/real')
    mkdir_if_missing(real_set_dir)


    # -----------------
    model_folder_string_list = model_path.split('/')
    cls_name = model_folder_string_list[-3]
    cls_idx = classes.index(cls_name) + 1
    set = model_folder_string_list[-2]

    data_dir = os.path.join(data_dir, cls_name, set)
    mkdir_if_missing(data_dir)


    model_prefix = model_folder_string_list[-1].split('.')[0]

    pz_up_dict = {'car': np.array([0, -1, 0]), 'airplane': np.array([0, -1, 0]),
                  'chair': np.array([0, -1, 0])}
    p_front_dict = {'car': np.array([0, 0, -1]), 'airplane': np.array([0, 0, -1]),
                    'chair': np.array([0, 0, -1])}
    pz = np.array([0, 0, 1])
    pz_up = np.array([0, -1, 0]) # pz_up_dict[cls_name]
    p_front = np.array([0, 0, -1]) # p_front_dict[cls_name]


    up_deg_max = 45
    front_deg_min = 0
    front_deg_max = 90
    if cls_name == 'airplane':
        front_deg_max = 75
    elif cls_name in ['car', 'chair']:
        front_deg_max = 90

    num_real_per_model = 50

    # real_indices = []
    for real_i in range(num_real_per_model):

        gen_this_pose = True
        tgt_trans = np.array([0, 0, 0.5])


        # randomly generate a quat -------------------------------------------------
        tgt_quat = np.random.normal(0, 1, 4)
        tgt_quat = tgt_quat / np.linalg.norm(tgt_quat)
        if tgt_quat[0] < 0:
            tgt_quat *= -1

        tgt_rot_m = quat2mat(tgt_quat)
        new_pz = np.dot(tgt_rot_m, pz.reshape((-1, 1))).reshape((3,))
        up_deg = angle(new_pz, pz_up)
        front_deg = angle(new_pz, p_front)
        count = 0
        while not (up_deg <= up_deg_max and front_deg_min < front_deg < front_deg_max):
            tgt_quat = np.random.normal(0, 1, 4)
            tgt_quat = tgt_quat / np.linalg.norm(tgt_quat)
            if tgt_quat[0] < 0:
                tgt_quat *= -1

            tgt_rot_m = quat2mat(tgt_quat)
            new_pz = np.dot(tgt_rot_m, pz.reshape((-1, 1))).reshape((3,))
            up_deg = angle(new_pz, pz_up)
            front_deg = angle(new_pz, p_front)
            count += 1
            if count % 100 == 0:
                print(real_i, cls_name, count, "up_deg: {}, front_deg: {}".format(
                    up_deg, front_deg))

            if count == 5000:
                gen_this_pose = False
                break

        # ---------------------------------------------------------------------------------
        if gen_this_pose:
            # real_indices.append('{:04d}'.format(real_i))
            pose = np.zeros((3, 4))
            pose[:3, 3] = tgt_trans
            pose[:3, :3] = tgt_rot_m

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

            light_position = np.array(light_position) * 0.5
            # inverse yz
            light_position[0] += pose[0, 3]
            light_position[1] -= pose[1, 3]
            light_position[2] -= pose[2, 3]

            colors = np.array([1, 1, 1])  # white light
            intensity = np.random.uniform(0.9, 1.1, size=(3,))
            colors_randk = random.randint(0, colors.shape[0] - 1)
            light_intensity = colors[colors_randk] * intensity
            # print('light intensity: ', light_intensity)

            # randomly choose a render machine
            rm_randk = random.randint(0, len(brightness_ratios) - 1)

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
            prefix = '{}_{:04d}'.format(model_prefix, real_i)
            print(prefix)
            real_color_file = os.path.join(data_dir, prefix + '-color.png')
            real_depth_file = os.path.join(data_dir, prefix + '-depth.png')
            real_label_file = os.path.join(data_dir, prefix + '-label.png')
            pose_file = os.path.join(data_dir, prefix + '-pose.txt')

            cv2.imwrite(real_color_file, rgb_gl)
            cv2.imwrite(real_depth_file, depth_res)
            cv2.imwrite(real_label_file, label_gl)

            text_file = open(pose_file, 'w')
            text_file.write("{}\n".format(cls_idx))
            pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
                .format(pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3],
                        pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3],
                        pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3])
            text_file.write(pose_str)

        else:
            print('{} failed to generate pose'.format(real_i))


def test_rotate():
    pose = np.zeros((3, 4))

    pose[:, 3] = np.array([0, 0, 0.5])

    pose[:3, :3] = np.eye(3, 3)

    print(pose)

    idx = 2 # random.randint(0, 100)

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
    print("light_position a: {}".format(light_position))
    light_position = np.array(light_position) * 0.5
    # inverse yz
    light_position[0] += pose[0, 3]
    light_position[1] -= pose[1, 3]
    light_position[2] -= pose[2, 3]

    colors = np.array([1, 1, 1])  # white light
    intensity = np.random.uniform(0.9, 1.1, size=(3,))
    colors_randk = random.randint(0, colors.shape[0] - 1)
    light_intensity = colors[colors_randk] * intensity

    # randomly choose a render machine
    rm_randk = random.randint(0, len(brightness_ratios) - 1)
    # get render result

    def rotate(angle, rot_axis, pose_gt, render_machine, p_center=np.array([0, 0, 0])):
        rot_sym_q = angle_axis_to_quat(angle, rot_axis)
        rot_sym_m = quat2mat(rot_sym_q)
        rot_res = R_transform(pose_gt[:3, :3], rot_sym_m, rot_coord='model')
        rot_res_q = mat2quat(rot_res)

        rgb_gl, depth_gl = render_machine.render(rot_res_q, pose_gt[:, 3] + p_center, light_position,
                                                 light_intensity, brightness_k=rm_randk)
        rgb_gl = rgb_gl.astype('uint8')
        pose_res = np.zeros((3, 4))
        pose_res[:3, :3] = rot_res
        pose_res[:3, 3] = pose_gt[:3, 3]
        return rgb_gl, depth_gl, pose_res

    angle = np.pi/2
    rot_axis = np.array([0, 1, 0])
    im_rot, depth_rot, pose_rot = rotate(angle, rot_axis, pose, render_machine) #################

    angle = np.pi/2
    rot_axis = np.array([1, 0, 0])
    im_rot_1, depth_rot_1, pose_rot_1 = rotate(angle, rot_axis, pose_rot, render_machine) #################
    print(pose_rot_1)
    pz = np.array([0, 0, 1])
    new_pz = np.dot(pose_rot_1[:3, :3], pz.reshape((-1, 1))).reshape((3,))
    print('new_pz: {}'.format(new_pz)) # (0, -1, 0)

    rgb_gl, depth_gl = render_machine.render(mat2quat(pose[:3, :3]), pose[:, -1],
                                             light_position,
                                             light_intensity,
                                             brightness_k=rm_randk)
    rgb_gl = rgb_gl.astype('uint8')
    # render_real label
    label_gl = np.zeros(depth_gl.shape)
    # print('depth gl:', depth_gl.shape)
    label_gl[depth_gl!=0] = 1

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6), dpi=200)
    plt.axis('off')
    fig.add_subplot(2, 3, 1)
    plt.imshow(rgb_gl[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.title('color real')

    fig.add_subplot(2, 3, 2)
    plt.imshow(im_rot[:, :, [2,1,0]])
    plt.axis('off')
    plt.title('rot image')

    fig.add_subplot(2, 3, 3)
    plt.imshow(im_rot_1[:, :, [2,1,0]])
    plt.axis('off')
    plt.title('rot image 1')

    plt.show()


if __name__ == "__main__":
    # test_rotate()
    gen_real()
    pass















