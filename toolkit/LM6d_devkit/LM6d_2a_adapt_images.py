# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yu Xiang
# --------------------------------------------------------
from __future__ import print_function, division
import sys, os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))
import numpy as np
from lib.utils.mkdir_if_missing import mkdir_if_missing
import cv2
import yaml
from shutil import copyfile
from tqdm import tqdm
# from lib.utils import renderer, inout
import matplotlib.pyplot as plt
from lib.render_glumpy.render_py_multi import Render_Py
import scipy.io as sio

LM6d_origin_root = os.path.join(cur_dir, '../../data/LINEMOD_6D/LM6d_origin/test')
# following previous works, part of the observed images are used for training and only images.

LM6d_new_root = os.path.join(cur_dir, '../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/data/observed')
model_dir = os.path.join(cur_dir, '../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/models')
mkdir_if_missing(LM6d_new_root)
print("target path: {}".format(LM6d_new_root))

idx2class = {1: 'ape',
            2: 'benchvise',
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

width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0

DEPTH_FACTOR = 1000

def read_img(path, n_channel=3):
    if n_channel == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    elif n_channel == 1:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        raise Exception("Unsupported n_channel: {}".format(n_channel))
    return img

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

def main():
    sel_classes = classes
    render_machine = Render_Py(model_dir, classes, K, width, height, ZNEAR, ZFAR)
    for cls_idx, cls_name in enumerate(classes):
        if not cls_name in sel_classes:
            continue
        print(cls_idx, cls_name)
        observed_indices = []
        images = [fn for fn in os.listdir(os.path.join(LM6d_origin_root,
                                                       '{:02d}'.format(class2idx(cls_name)), 'rgb')) if '.png' in fn]
        images = sorted(images)

        gt_path = os.path.join(LM6d_origin_root, '{:02d}'.format(class2idx(cls_name)), 'gt.yml')
        gt_dict = load_gt(gt_path)

        info_path = os.path.join(LM6d_origin_root, '{:02d}'.format(class2idx(cls_name)), 'info.yml')
        info_dict = load_info(info_path)

        for observed_img in tqdm(images):
            old_color_path = os.path.join(LM6d_origin_root, '{:02d}'.format(class2idx(cls_name)), "rgb/{}".format(observed_img))
            assert os.path.exists(old_color_path), old_color_path
            old_depth_path = os.path.join(LM6d_origin_root, '{:02d}'.format(class2idx(cls_name)), "depth/{}".format(observed_img))
            assert os.path.exists(old_depth_path), old_depth_path
            img_id = int(observed_img.replace('.png', ''))
            new_img_id = img_id + 1

            # K
            # K = np.array(info_dict[img_id]['cam_K']).reshape((3, 3))
            color_img = cv2.imread(old_color_path, cv2.IMREAD_COLOR)

            ## depth
            depth = read_img(old_depth_path, 1)
            # print(np.max(depth), np.min(depth))

            # print(color_img.shape)

            new_color_path = os.path.join(LM6d_new_root, '{:02d}'.format(class2idx(cls_name)),
                                          "{:06d}-color.png".format(new_img_id))
            new_depth_path = os.path.join(LM6d_new_root, '{:02d}'.format(class2idx(cls_name)),
                                          "{:06d}-depth.png".format(new_img_id))
            mkdir_if_missing(os.path.dirname(new_color_path))

            copyfile(old_color_path, new_color_path)
            copyfile(old_depth_path, new_depth_path)

            # meta and label
            meta_dict = {}
            num_instance = len(gt_dict[img_id])
            meta_dict['cls_indexes'] = np.zeros((1, num_instance), dtype=np.int32)
            meta_dict['boxes'] = np.zeros((num_instance, 4), dtype='float32')
            meta_dict['poses'] = np.zeros((3,4,num_instance), dtype='float32')
            distances = []
            label_dict = {}
            for ins_id, instance in enumerate(gt_dict[img_id]):
                obj_id = instance['obj_id']
                meta_dict['cls_indexes'][0, ins_id] = obj_id
                obj_bb = np.array(instance['obj_bb'])
                meta_dict['boxes'][ins_id, :] = obj_bb
                # pose
                pose = np.zeros((3, 4))

                R = np.array(instance['cam_R_m2c']).reshape((3, 3))
                t = np.array(instance['cam_t_m2c']) / 1000.  # mm -> m
                pose[:3, :3] = R
                pose[:3, 3] = t
                distances.append(t[2])
                meta_dict['poses'][:,:,ins_id] = pose
                image_gl, depth_gl = render_machine.render(obj_id-1, pose[:3, :3], pose[:3, 3],
                                                            r_type='mat')
                image_gl = image_gl.astype('uint8')
                label = np.zeros(depth_gl.shape)
                label[depth_gl!=0] = 1
                label_dict[obj_id] = label
            meta_path = os.path.join(LM6d_new_root, '{:02d}'.format(class2idx(cls_name)),
                                          "{:06d}-meta.mat".format(new_img_id))
            sio.savemat(meta_path, meta_dict)

            dis_inds = sorted(range(len(distances)), key=lambda k: -distances[k]) # put deeper objects first
            # label
            res_label = np.zeros((480, 640))
            for dis_id in dis_inds:
                cls_id = meta_dict['cls_indexes'][0, dis_id]
                tmp_label = label_dict[cls_id]
                # label
                res_label[tmp_label == 1] = cls_id

            label_path = os.path.join(LM6d_new_root, '{:02d}'.format(class2idx(cls_name)),
                                      "{:06d}-label.png".format(new_img_id))
            cv2.imwrite(label_path, res_label)

            # observed idx
            observed_indices.append("{:02d}/{:06d}".format(class2idx(cls_name), new_img_id))

if __name__ == "__main__":
    main()
    print("{} finished".format(__file__))
