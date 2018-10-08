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
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.render_glumpy.render_py import Render_Py
import numpy as np
import scipy.io as sio
import cv2
from lib.pair_matching import RT_transform
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

'''
our data structure:
(render observed)
ape/
    000001-color.png
    000001-depth.png
    000001-label.png
    000001-pose.txt
'''

# =================== global settings ======================
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

def class2idx(class_name, idx2class=idx2class):
    for k,v in idx2class.items():
        if v == class_name:
            return k
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0

DEPTH_FACTOR = 1000

LM6d_root = os.path.join(cur_dir, '../data/LINEMOD_6D/LM6d_converted/LM6d_refine')
observed_set_dir = os.path.join(LM6d_root, 'image_set/observed')
observed_data_root = os.path.join(LM6d_root, 'data/observed')
gt_observed_root = os.path.join(LM6d_root, 'data/gt_observed')
print("target path: {}"
      .format(gt_observed_root))

# ==========================================================
def write_pose_file(pose_file, class_idx, pose_ori_m):
    text_file = open(pose_file, 'w')
    text_file.write("{}\n".format(class_idx))
    pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
        .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
    text_file.write(pose_str)


def gen_gt_observed():
    for cls_idx, cls_name in idx2class.items():
        print(cls_idx, cls_name)
        # uncomment here to only generate data for ape
        # if cls_name != 'ape':
        #     continue
        with open(os.path.join(observed_set_dir, '{}_all.txt'.format(cls_name)), 'r') as f:
            all_indices = [line.strip('\r\n') for line in f.readlines()]

        # render machine
        model_dir = os.path.join(LM6d_root, 'models', cls_name)
        render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for observed_idx in tqdm(all_indices):
            video_name, prefix = observed_idx.split('/')
            # read pose -------------------------------------
            observed_meta_path = os.path.join(observed_data_root, "{}-meta.mat".format(observed_idx))
            meta_data = sio.loadmat(observed_meta_path)
            inner_id = np.where(np.squeeze(meta_data['cls_indexes']) == cls_idx)
            if len(meta_data['poses'].shape) == 2:
                pose = meta_data['poses']
            else:
                pose = np.squeeze(meta_data['poses'][:, :, inner_id])

            new_pose_path = os.path.join(gt_observed_root, cls_name, "{}-pose.txt".format(prefix))
            mkdir_if_missing(os.path.join(gt_observed_root, cls_name))
            # write pose
            write_pose_file(new_pose_path, cls_idx, pose)

            # ----------------------render color, depth ------------
            rgb_gl, depth_gl = render_machine.render(RT_transform.mat2quat(pose[:3, :3]), pose[:, -1])
            if any([x in observed_idx for x in ['000128', '000256', '000512']]):
                rgb_gl = rgb_gl.astype('uint8')
                render_color_path = os.path.join(gt_observed_root, cls_name, "{}-color.png".format(prefix))
                cv2.imwrite(render_color_path, rgb_gl)

            # depth
            depth_save = depth_gl * DEPTH_FACTOR
            depth_save = depth_save.astype('uint16')
            render_depth_path = os.path.join(gt_observed_root, cls_name, "{}-depth.png".format(prefix))
            cv2.imwrite(render_depth_path, depth_save)

            #--------------------- render label ----------------------------------
            render_label = depth_gl != 0
            render_label = render_label.astype('uint8')

            # write label
            label_path = os.path.join(gt_observed_root, cls_name, "{}-label.png".format(prefix))
            cv2.imwrite(label_path, render_label)

def check_gt_observed():
    cls_name = 'ape'
    observed_indices = ['000128', '000256', '000512']
    for observed_idx in observed_indices:
        render_color_path = os.path.join(gt_observed_root, cls_name, "{}-color.png".format(observed_idx))
        render_color = cv2.imread(render_color_path)
        render_depth_path = os.path.join(gt_observed_root, cls_name, "{}-depth.png".format(observed_idx))
        render_depth = cv2.imread(render_depth_path, cv2.IMREAD_UNCHANGED)
        render_label_path = os.path.join(gt_observed_root, cls_name, "{}-label.png".format(observed_idx))
        render_label = cv2.imread(render_label_path, cv2.IMREAD_UNCHANGED)

        cls_idx = class2idx(cls_name)
        cls_idx_str = "{:02d}".format(cls_idx)
        observed_color_path = os.path.join(observed_data_root, cls_idx_str, "{}-color.png".format(observed_idx))
        observed_color = cv2.imread(observed_color_path)
        observed_depth_path = os.path.join(observed_data_root, cls_idx_str, "{}-depth.png".format(observed_idx))
        observed_depth = cv2.imread(observed_depth_path, cv2.IMREAD_UNCHANGED)
        observed_label_path = os.path.join(observed_data_root, cls_idx_str, "{}-label.png".format(observed_idx))
        observed_label = cv2.imread(observed_label_path, cv2.IMREAD_UNCHANGED)

        plt.subplot(2, 3, 1)
        plt.imshow(observed_color[:,:,[2,1,0]])

        plt.subplot(2, 3, 2)
        plt.imshow(observed_depth)

        plt.subplot(2, 3, 3)
        plt.imshow(observed_label)

        plt.subplot(2, 3, 4)
        plt.imshow(np.int_(render_color[:,:,[2,1,0]]*0.8+observed_color[:,:,[2,1,0]]*0.2))

        plt.subplot(2, 3, 5)
        a = observed_depth.astype(np.int)*observed_label-render_depth.astype(np.int)*render_label
        plt.imshow(np.repeat(a[:, :, np.newaxis]+128, 3, axis=2))

        plt.subplot(2, 3, 6)
        plt.imshow(np.stack([render_label, render_label, observed_label], axis=2)*255)

        plt.show()

if __name__ == "__main__":
    gen_gt_observed()
    # check_gt_observed()

    print("{} finished".format(__file__))
