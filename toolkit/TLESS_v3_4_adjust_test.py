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
from lib.utils import renderer, inout
import numpy as np
import cv2
from tqdm import tqdm
import yaml

'''
origin data structure:
05/
    rgb/
    depth/
    info.yml
    gt.yml
-------
our data structure:
(real)
05/
    000001-color.png
    000001-depth.png
    000001-label.png
    # 000001-meta.mat
    000001-pose.txt
'''




# =================== global settings ======================
video_list = ['{:02d}'.format(i) for i in range(1, 21)]
sel_videos = ['02']

class_list = ['{:02d}'.format(i) for i in range(1, 31)]
sel_classes = ['05', '06']


TLESS_root = os.path.join(cur_dir, '../data/TLESS')
origin_data_root = os.path.join(cur_dir, '../data/TLESS/t-less_v2/test_primesense') # 20 videos


width = 720
height = 540
crop_width = 640 # crop 80
crop_height = 480 # crop 60


K_0 = np.array([[1075.65091572, 0, 320.], [0, 1073.90347929, 240.], [0, 0, 1]]) # Primesense

# in test set, K is different for each image
ZNEAR = 0.25
ZFAR = 6.0

DEPTH_FACTOR = 10000


# new data root -----------
new_data_root = os.path.join(TLESS_root, 'TLESS_render_v3/data/real/test')
mkdir_if_missing(new_data_root)

real_set_dir = os.path.join(TLESS_root, 'TLESS_render_v3/image_set/real')
mkdir_if_missing(real_set_dir)

def read_img(path, n_channel=3):
    if n_channel == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    elif n_channel == 1:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        raise Exception("Unsupported n_channel: {}".format(n_channel))
    return img

# ==========================================================
def mini_test():
    # cls_name = '05'
    video_name = '02'
    depth_path = os.path.join(origin_data_root, video_name, 'depth/0000.png')

    depth = read_img(depth_path, 1) / float(DEPTH_FACTOR)
    print(np.unique(depth))

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

def write_K_file(K_file, K):
    text_file = open(K_file, 'w')
    # text_file.write("{}\n".format(class_idx))
    K_str = "{} {} {}\n{} {} {}\n{} {} {}" \
        .format(K[0, 0], K[0, 1], K[0, 2],
                K[1, 0], K[1, 1], K[1, 2],
                K[2, 0], K[2, 1], K[2, 2])
    text_file.write(K_str)


def adapt_color_depth_pose_label():
    # real/test/cls_name/vidoe_name/06d-color.png
    for cls_idx, cls_name in enumerate(class_list):
        if not cls_name in sel_classes:
            continue
        print(cls_idx, cls_name)
        real_indices = []
        model_path = os.path.join(TLESS_root, 'models/{}/obj_{}_scaled.ply'.format(cls_name, cls_name))
        model = inout.load_ply(model_path)
        for video_name in sel_videos: # 02
            print('video name: {}'.format(video_name))
            images = [fn for fn in os.listdir(os.path.join(origin_data_root, video_name, 'rgb')) if '.png' in fn]
            images.sort()

            gt_path = os.path.join(origin_data_root, video_name, 'gt.yml')
            gt_dict = load_gt(gt_path)

            info_path = os.path.join(origin_data_root, video_name, 'info.yml')
            info_dict = load_info(info_path)

            for real_img in tqdm(images):
                old_color_path = os.path.join(origin_data_root, video_name, "rgb/{}".format(real_img))
                assert os.path.exists(old_color_path), old_color_path
                old_depth_path = os.path.join(origin_data_root, video_name, "depth/{}".format(real_img))
                assert os.path.exists(old_depth_path), old_depth_path
                img_id = int(real_img.replace('.png', ''))

                # K
                K = np.array(info_dict[img_id]['cam_K']).reshape((3, 3))


                K_diff = K_0 - K
                cx_diff = K_diff[0, 2]
                cy_diff = K_diff[1, 2]
                px_diff = int(np.round(cx_diff))
                py_diff = int(np.round(cy_diff))

                color_img = cv2.imread(old_color_path, cv2.IMREAD_COLOR)

                # translate
                M = np.float32([[1, 0, px_diff], [0, 1, py_diff]])
                color_img = cv2.warpAffine(color_img, M, (720, 540))

                # crop to (480, 640)
                crop_color = color_img[:480, :640, :]

                ## depth
                # translate
                depth = read_img(old_depth_path, 1)
                depth = cv2.warpAffine(depth, M, (720, 540))
                # crop
                crop_depth = depth[:480, :640]

                # print(color_img.shape)
                for ins_id, instance in enumerate(gt_dict[img_id]):
                    obj_id = instance['obj_id']
                    if obj_id == int(cls_name):
                        new_color_path = os.path.join(new_data_root, cls_name, video_name,
                                                      "{:06d}_{}-color.png".format(img_id, ins_id))
                        new_depth_path = os.path.join(new_data_root, cls_name, video_name,
                                                      "{:06d}_{}-depth.png".format(img_id, ins_id))
                        mkdir_if_missing(os.path.join(new_data_root, cls_name, video_name))
                        # save color img


                        cv2.imwrite(new_color_path, crop_color)
                        cv2.imwrite(new_depth_path, crop_depth)

                        # pose
                        pose = np.zeros((3, 4))
                        R = np.array(instance['cam_R_m2c']).reshape((3, 3))
                        t = np.array(instance['cam_t_m2c']) / 1000. # mm -> m
                        pose[:3, :3] = R
                        pose[:3, 3] = t
                        pose_path = os.path.join(new_data_root, cls_name, video_name,
                                                      "{:06d}_{}-pose.txt".format(img_id, ins_id))
                        write_pose_file(pose_path, cls_idx, pose)

                        # label
                        # depth = read_img(old_depth_path, 1)
                        surf_color = None # (1, 0, 0) # ?????
                        im_size = (640, 480) # (w, h)

                        ren_rgb, ren_depth = renderer.render(model, im_size, K_0, R, t, clip_near=ZNEAR, clip_far=ZFAR,
                                                             surf_color=surf_color, mode='rgb+depth')
                        ren_rgb = ren_rgb.astype('uint8')
                        # print('ren_rgb: ', ren_rgb.max(), ren_rgb.min())
                        # print('ren_depth: ', ren_depth.max(), ren_depth.min())
                        label = np.zeros((crop_height, crop_width))
                        label[ren_depth!=0] = 1
                        label_path = os.path.join(new_data_root, cls_name, video_name,
                                                      "{:06d}_{}-label.png".format(img_id, ins_id))
                        cv2.imwrite(label_path, label)
                        # def vis_check():
                        #     fig = plt.figure(figsize=(8, 6), dpi=120)
                        #     plt.subplot(2, 3, 1)
                        #
                        #     plt.imshow(crop_depth)
                        #     plt.title('crop_depth')
                        #
                        #     plt.subplot(2, 3, 2)
                        #     # plt.imshow(label)
                        #     # plt.title('label rendered')
                        #     depth_diff = crop_depth.copy()
                        #     depth_diff[ren_depth!=0] = 0
                        #     plt.imshow(depth_diff)
                        #     plt.title('depth_diff')
                        #
                        #     plt.subplot(2, 3, 3)
                        #     # color_img = read_img(old_color_path, 3)
                        #     plt.imshow(crop_color[:, :, [2, 1, 0]])
                        #     plt.title('color image')
                        #
                        #     plt.subplot(2, 3, 4)
                        #     plt.imshow(ren_rgb)
                        #     plt.title('ren_rgb')
                        #
                        #     plt.subplot(2, 3, 5)
                        #     plt.imshow(ren_depth)
                        #     plt.title('ren_depth')
                        #
                        #     plt.subplot(2, 3, 6)
                        #     color_diff = crop_color - ren_rgb
                        #     plt.imshow(color_diff)
                        #     plt.title('color_diff')
                        #
                        #     plt.show()
                        # vis_check()


                        # real idx
                        real_indices.append("test/{}/{}/{:06d}_{}".format(cls_name, video_name, img_id, ins_id))

            # one idx file for each video of each class
            real_idx_file = os.path.join(real_set_dir, "{}_{}_test.txt".format(cls_name, video_name))
            with open(real_idx_file, 'w') as f:
                for real_idx in real_indices:
                    f.write(real_idx + '\n')



if __name__ == "__main__":
    adapt_color_depth_pose_label()