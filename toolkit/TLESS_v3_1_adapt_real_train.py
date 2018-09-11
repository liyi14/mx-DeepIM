# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import sys, os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '..'))
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.render_glumpy.render_py import Render_Py
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import random
from lib.utils.image import resize
random.seed(1234)
np.random.seed(1234)

# =================== global settings ======================
class_list = ['{:02d}'.format(i) for i in range(1, 31)]
sel_classes = ['05', '06']


TLESS_root = os.path.join(cur_dir, '../data/TLESS')
test_data_root = os.path.join(cur_dir, '../data/TLESS/TLESS_render_v3/data/real')
ori_train_data_root = os.path.join(cur_dir, '../data/TLESS/t-less_v2/train_primesense')

width = 640
height = 480

K = np.array([[1075.65091572, 0, 320.], [0, 1073.90347929, 240.], [0, 0, 1]]) # Primesense
ZNEAR = 0.25
ZFAR = 6.0

DEPTH_FACTOR = 10000

real_set_dir = os.path.join(TLESS_root, 'TLESS_render_v3/image_set/real')


def read_img(path, n_channel=3):
    if n_channel == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    elif n_channel == 1:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        raise Exception("Unsupported n_channel: {}".format(n_channel))
    return img

# ==========================================================

def adapt_real_train():
    class_list = ['{:02d}'.format(i) for i in range(1, 31)]
    sel_classes = ['05', '06']

    width = 640  # 400
    height = 480  # 400
    depth_factor = 10000
    K_0 = np.array([[1075.65091572, 0, 320.], [0, 1073.90347929, 240.], [0, 0, 1]])  # Primesense
    new_data_root = os.path.join(TLESS_root, 'TLESS_render_v3/data/real')
    mkdir_if_missing(new_data_root)

    real_set_dir = os.path.join(TLESS_root, 'TLESS_render_v3/image_set/real')
    mkdir_if_missing(real_set_dir)

    for cls_idx, cls_name in enumerate(class_list):
        if not cls_name in sel_classes:
            continue
        print(cls_idx, cls_name)

        model_dir = os.path.join(TLESS_root, 'models', cls_name)

        render_machine = Render_Py(model_dir, K_0, width, height, ZNEAR, ZFAR)

        gt_path = os.path.join(TLESS_root, 't-less_v2/train_primesense/{}/gt.yml'.format(cls_name))
        gt_dict = load_gt(gt_path)
        info_path = os.path.join(TLESS_root, 't-less_v2/train_primesense/{}/info.yml'.format(cls_name))
        info_dict = load_info(info_path)

        real_indices = []

        for img_id in tqdm(gt_dict.keys()):
            R = np.array(gt_dict[img_id][0]['cam_R_m2c']).reshape((3, 3))
            t = np.array(gt_dict[img_id][0]['cam_t_m2c']) / 1000.
            K = np.array(info_dict[img_id]['cam_K']).reshape((3, 3))

            # K[0, 2] += 120 # cx
            # K[1, 2] += 40 # cy

            pose = np.zeros((3, 4))
            pose[:3, :3] = R
            pose[:3, 3] = t
            # print(pose)
            # print(K)
            K_diff = K_0 - K
            cx_diff = K_diff[0, 2]
            cy_diff = K_diff[1, 2]
            px_diff = int(np.round(cx_diff))
            py_diff = int(np.round(cy_diff))

            # pose ----------------
            pose_path = os.path.join(new_data_root, cls_name, '{:06d}-pose.txt'.format(img_id))
            mkdir_if_missing(os.path.join(new_data_root, cls_name))
            write_pose_file(pose_path, cls_idx, pose)

            rgb_gl, depth_gl = render_machine.render(pose[:3, :3], pose[:, -1],
                                                     r_type='mat', K=K_0)
            rgb_gl = rgb_gl.astype('uint8')

            # depth ------------------
            depth_gl = (depth_gl * depth_factor).astype(np.uint16)
            depth_path = os.path.join(new_data_root, cls_name, '{:06d}-depth.png'.format(img_id))
            cv2.imwrite(depth_path, depth_gl)

            # label ---------------------
            label_gl = np.zeros(depth_gl.shape)
            label_gl[depth_gl != 0] = 1
            label_path = os.path.join(new_data_root, cls_name, '{:06d}-label.png'.format(img_id))
            cv2.imwrite(label_path, label_gl)


            # real color ----------------------------
            color_real = read_img(os.path.join(ori_train_data_root, cls_name, 'rgb/{:04d}.png'.format(img_id)), 3)
            # print(color_real.max(), color_real.min())
            pad_real = np.zeros((480, 640, 3))
            xs = 0
            ys = 0
            pad_real[xs:400 + xs, ys:400 + ys, :] = color_real
            pad_real = pad_real.astype('uint8')

            # translate image
            M = np.float32([[1, 0, px_diff], [0, 1, py_diff]])
            pad_real = cv2.warpAffine(pad_real, M, (640, 480))

            color_path = os.path.join(new_data_root, cls_name, '{:06d}-color.png'.format(img_id))
            cv2.imwrite(color_path, pad_real)

            # real index
            real_indices.append("{}/{:06d}".format(cls_name, img_id))

        real_indices.sort()
        real_set_file = os.path.join(real_set_dir, '{}_train.txt'.format(cls_name))
        with open(real_set_file, 'w') as f:
            for real_idx in real_indices:
                f.write(real_idx + '\n')

def load_info(info_path):
    with open(info_path, 'r') as f:
        info_dict = yaml.load(f)
    return info_dict

def load_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt_dict = yaml.load(f)
    return gt_dict

def write_pose_file(pose_file, class_idx, pose_ori_m):
    text_file = open(pose_file, 'w')
    text_file.write("{}\n".format(class_idx))
    pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
        .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
    text_file.write(pose_str)




if __name__ == "__main__":
    adapt_real_train()