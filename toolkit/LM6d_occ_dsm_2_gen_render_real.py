# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import os
from lib.pair_matching.RT_transform import *
from lib.utils.mkdir_if_missing import mkdir_if_missing
from tqdm import tqdm
from lib.render_glumpy.render_py import Render_Py
import cv2
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.path.dirname(__file__))
model_root = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D/models')
real_root_dir = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D/LM6d_occ_ds_multi/data/real')
real_set_dir = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D/LM6d_occ_ds_multi/image_set/real')

# output dir
data_dir = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D/LM6d_occ_ds_multi/data/render_real')
mkdir_if_missing(data_dir)


idx2class = {1: 'ape',
            # 2: 'benchviseblue',
            # 4: 'camera',
            5: 'can',
            6: 'cat',
            8: 'driller',
            9: 'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            # 13: 'iron',
            # 14: 'lamp',
            # 15: 'phone'
}


def class2idx(class_name, idx2class=idx2class):
    for k,v in idx2class.items():
        if v == class_name:
            return k

classes = idx2class.values()
classes.sort()

# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000


sel_classes = classes
num_class = len(sel_classes)

def pose_q2m(pose_q):
    pose = np.zeros((3, 4), dtype='float32')
    pose[:3, :3] = quat2mat(pose_q[:4])
    pose[:3, 3] = pose_q[4:]
    return pose

def main():
    for cls_idx, cls_name in enumerate(tqdm(sel_classes)):
        print(cls_idx, cls_name)
        keyframe_path = os.path.join(real_set_dir, "train_real_{}.txt".format(cls_name))
        with open(keyframe_path) as f:
            real_index_list = [x.strip() for x in f.readlines()]
        video_name_list = [x.split('/')[0] for x in real_index_list]
        real_prefix_list = [x.split('/')[1] for x in real_index_list]

        # init renderer
        model_dir = os.path.join(model_root, cls_name)
        render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for idx, real_index in enumerate(tqdm(real_index_list)):
            prefix = real_prefix_list[idx]
            video_name = video_name_list[idx]

            render_real_dir = os.path.join(data_dir, cls_name)
            mkdir_if_missing(render_real_dir)
            render_real_dir = os.path.join(render_real_dir, video_name) # ./
            mkdir_if_missing(render_real_dir)

            # to be written
            render_real_color_file = os.path.join(render_real_dir, prefix + "-color.png")
            render_real_depth_file = os.path.join(render_real_dir, prefix + "-depth.png")
            render_real_pose_file = os.path.join(render_real_dir, prefix + "-pose.txt")

            real_label_file = os.path.join(real_root_dir, video_name, prefix + "-label.png")
            render_real_label_file = os.path.join(render_real_dir, prefix + "-label.png")

            real_pose_file = os.path.join(real_root_dir, video_name, prefix + "-poses.npy")
            real_poses = np.load(real_pose_file)
            real_pose_dict = real_poses.all()
            # pprint(real_pose_dict)
            if not cls_name in real_pose_dict.keys():
                continue
            pose = real_pose_dict[cls_name]
            rgb_gl, depth_gl = render_machine.render(mat2quat(pose[:3, :3]), pose[:, -1])

            rgb_gl = rgb_gl.astype('uint8')

            label_gl = np.zeros(depth_gl.shape)
            label_gl[depth_gl != 0] = 1

            depth_gl = depth_gl * depth_factor
            depth_gl = depth_gl.astype('uint16')

            # write results
            cv2.imwrite(render_real_color_file, rgb_gl)
            cv2.imwrite(render_real_depth_file, depth_gl)
            cv2.imwrite(render_real_label_file, label_gl)

            text_file = open(render_real_pose_file, 'w')
            text_file.write("{}\n".format(cls_idx))
            pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
                .format(pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3],
                        pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3],
                        pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3])
            text_file.write(pose_str)

    print('done')




def read_img(path, n_channel=3):
    if n_channel == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    elif n_channel == 1:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        raise Exception("Unsupported n_channel: {}".format(n_channel))
    return img

def check_real_render_real():
    real_dir = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D/LM6d_occ_ds_multi/data/real')
    real_set_dir = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D/LM6d_occ_ds_multi/image_set/real')
    render_real_dir = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D/LM6d_occ_ds_multi/data/render_real')
    cls_name = 'cat'
    indices = ["{:06d}".format(i) for i in range(1, 200)]

    for prefix in indices:
        render_real_color_file = os.path.join(render_real_dir, cls_name, prefix + "-color.png")
        render_real_depth_file = os.path.join(render_real_dir, cls_name, prefix + "-depth.png")
        render_real_label_file = os.path.join(render_real_dir, cls_name, prefix + "-label.png")
        # render_real_pose_file = os.path.join(render_real_dir, cls_name, prefix + "-pose.txt")

        real_color_file = os.path.join(real_dir, prefix + "-color.png")
        real_depth_file = os.path.join(real_dir, prefix + "-depth.png")
        real_label_file = os.path.join(real_dir, prefix + "-label.png")

        if not os.path.exists(render_real_depth_file):
            print("{} not exits".format(render_real_depth_file))
            continue

        color_r = read_img(real_color_file, 3)
        depth_r = read_img(real_depth_file, 1) / depth_factor
        label_r = read_img(real_label_file, 1)

        color_rr = read_img(render_real_color_file, 3)
        depth_rr = read_img(render_real_depth_file, 1) / depth_factor
        label_rr = read_img(render_real_label_file, 1)

        fig = plt.figure(figsize=(8, 6), dpi=200)
        plt.axis('off')
        fig.add_subplot(2, 3, 1)
        plt.imshow(color_r[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.title('color real')

        plt.subplot(2, 3, 2)
        plt.imshow(depth_r)
        plt.axis('off')
        plt.title('depth real')

        plt.subplot(2, 3, 3)
        plt.imshow(label_r)
        plt.axis('off')
        plt.title('label real')

        fig.add_subplot(2, 3, 4)
        plt.imshow(color_rr[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.title('color render real')

        plt.subplot(2, 3, 5)
        plt.imshow(depth_rr)
        plt.axis('off')
        plt.title('depth render real')

        plt.subplot(2, 3, 6)
        plt.imshow(label_rr)
        plt.axis('off')
        plt.title('label render real')

        plt.show()



if __name__ == "__main__":
    main()
    check_real_render_real()
