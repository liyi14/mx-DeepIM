# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import sys, os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '..'))
from lib.utils import renderer, inout
import numpy as np
from lib.utils.mkdir_if_missing import *
import lib.pair_matching.RT_transform as se3
import cv2
from tqdm import tqdm


class_list = ['{:02d}'.format(i) for i in range(1, 31)]
sel_classes = ['05', '06']

video_list = ['{:02d}'.format(i) for i in range(1, 21)]
sel_videos = ['02']

# config for render machine
width = 640
height = 480
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 10000

K_0 = np.array([[1075.65091572, 0, 320.], [0, 1073.90347929, 240.], [0, 0, 1]]) # Primesense

version = 'v1' ################################################################################# version
print('version', version)

TLESS_root = os.path.join(cur_dir, '../data/TLESS')
real_set_root = os.path.join(TLESS_root, 'TLESS_render_v3/image_set/real')
real_data_dir = os.path.join(TLESS_root, 'TLESS_render_v3/data/real')

rendered_pose_path = "%s/TLESS_%s_{}_rendered_pose_{}_{}.txt"%(os.path.join(TLESS_root, 'syn_poses_v3'),
                                                   version)

# output_path
rendered_root_dir = os.path.join(TLESS_root, 'TLESS_render_v3/data/rendered_{}/test'.format(version))
pair_set_dir = os.path.join(TLESS_root, 'TLESS_render_v3/image_set')
mkdir_if_missing(rendered_root_dir)
mkdir_if_missing(pair_set_dir)

def main():
    gen_images = True ##################################################################### gen_images
    for class_idx, class_name in enumerate(class_list):
        # train_pair = []
        val_pair = []
        if class_name in ['__back_ground__']:
            continue
        if not class_name in sel_classes:
            continue
        print("start ", class_name)
        model_path = os.path.join(TLESS_root, 'models/{}/obj_{}_scaled.ply'.format(class_name, class_name))
        model = inout.load_ply(model_path)
        for video_name in sel_videos: # 02

            for set_type in ['test']:
                with open(os.path.join(real_set_root, '{}_{}_{}.txt'.format(class_name, video_name, 'test')), 'r') as f:
                    test_real_list = [x.strip() for x in f.readlines()]

                with open(rendered_pose_path.format(set_type, class_name, video_name)) as f:
                    str_rendered_pose_list = [x.strip().split(' ') for x in f.readlines()]
                rendered_pose_list = np.array([[float(x) for x in each_pose] for each_pose in str_rendered_pose_list])
                rendered_per_real = 1
                assert (len(rendered_pose_list) == 1*len(test_real_list)), \
                      '{} vs {}'.format(len(rendered_pose_list), len(test_real_list))
                for idx, real_index in enumerate(tqdm(test_real_list)):
                    _, cls_name, vid_name, real_prefix = real_index.split('/')

                    rendered_dir = os.path.join(rendered_root_dir, class_name, video_name)
                    mkdir_if_missing(rendered_dir)
                    for inner_idx in range(rendered_per_real):
                        if gen_images:
                            # if gen_images and real_index in test_real_list and inner_idx == 0: # only generate my_val_v{}
                            image_file = os.path.join(rendered_dir,
                                                      '{}-color.png'.format(real_prefix))
                            depth_file = os.path.join(rendered_dir,
                                                      '{}-depth.png'.format(real_prefix))

                            rendered_idx = idx*rendered_per_real + inner_idx
                            pose_rendered_q = rendered_pose_list[rendered_idx]
                            R = se3.quat2mat(pose_rendered_q[:4])
                            t = pose_rendered_q[4:]

                            surf_color = None  # (1, 0, 0) # ?????
                            im_size = (width, height)  # (w, h)

                            ren_rgb, ren_depth = renderer.render(model, im_size, K_0, R, t, clip_near=ZNEAR, clip_far=ZFAR,
                                                                 surf_color=surf_color, mode='rgb+depth')

                            ren_depth = (ren_depth*depth_factor).astype(np.uint16)

                            cv2.imwrite(image_file, ren_rgb)
                            cv2.imwrite(depth_file, ren_depth)

                            pose_rendered_file = os.path.join(rendered_dir, '{}-pose.txt'.format(real_prefix))
                            text_file = open(pose_rendered_file, 'w')
                            text_file.write("{}\n".format(class_idx))
                            pose_rendered_m = np.zeros((3, 4))
                            pose_rendered_m[:, :3] = se3.quat2mat(pose_rendered_q[:4])
                            pose_rendered_m[:, 3] = pose_rendered_q[4:]
                            pose_ori_m = pose_rendered_m
                            pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}"\
                                .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                                        pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                                        pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
                            text_file.write(pose_str)


                        val_pair.append("{} test/{}/{}/{}"
                                            .format(real_index,
                                                    class_name, video_name, real_prefix))



                test_pair_set_file = os.path.join(pair_set_dir, "my_val_{}_video{}_{}.txt".format(version, video_name, class_name))
                val_pair.sort()
                with open(test_pair_set_file, "w") as text_file:
                    for x in val_pair:
                        text_file.write("{}\n".format(x))
            print(class_name, video_name, " done")


def check_real_rendered():
    from lib.utils.utils import read_img
    import matplotlib.pyplot as plt

    real_dir = os.path.join(TLESS_root, 'TLESS_render_v3/data/real')

    for class_idx, class_name in enumerate(class_list):
        if not class_name in sel_classes:
            continue
        print(class_name)
        real_list_path = os.path.join(real_set_root, "{}_{}_test.txt".format(class_name, '02'))
        with open(real_list_path, 'r') as f:
            real_list = [x.strip() for x in f.readlines()]
        for idx, real_index in enumerate(real_list):
            print(real_index)
            prefix = real_index.split('/')[-1]
            color_real = read_img(os.path.join(real_dir, real_index + '-color.png'), 3)
            color_rendered = read_img(os.path.join(rendered_root_dir, class_name, '02', prefix + '-color.png'), 3)
            fig = plt.figure(figsize=(8, 6), dpi=120)
            # plt.axis('off')
            plt.subplot(1,3,1)
            plt.imshow(color_real[:,:,[2,1,0]])
            plt.title('real')

            plt.subplot(1,3,2)
            plt.imshow(color_rendered[:,:,[2,1,0]])
            plt.title('rendered')

            plt.subplot(1, 3, 3)
            color_diff = color_real - color_rendered
            plt.imshow(color_diff[:, :, [2, 1, 0]])
            plt.title('diff')
            plt.show()



if __name__ == "__main__":
    main()
    check_real_rendered()
