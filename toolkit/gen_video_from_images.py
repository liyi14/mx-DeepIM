# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang
# --------------------------------------------------------
from __future__ import division, print_function
import os
import sys
import numpy as np
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))
import cv2
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate video full')
    parser.add_argument('--exp_dir', required=True, help='exp_dir for generating video full, e.g.: '
        'output/deepim/deepim_flownet_occLM_v1_se3_ex_u2s16_driller_iter_v42_zoom_NF_NF_NF_NF_real_1gpus/yu_val_driller')
    args = parser.parse_args()
    return args

def my_cmp(x, y):
    x_idx = int(x.split('_')[0])
    y_idx = int(y.split('_')[0])
    return x_idx - y_idx

if __name__ == '__main__':
    args = parse_args()
    exp_dir = args.exp_dir
    print('exp_dir: ', exp_dir)

    before_ICP_dir = os.path.join(exp_dir, 'video_full/before_ICP')
    after_ICP_dir = os.path.join(exp_dir, 'video_full/after_ICP')
    ours_dir = os.path.join(exp_dir, 'video_full/ours')
    last_frame_dir = os.path.join(exp_dir, 'video_full/last_frame')

    before_ICP_list = os.listdir(before_ICP_dir)
    before_ICP_list.sort(cmp=my_cmp)

    after_ICP_list = os.listdir(after_ICP_dir)
    after_ICP_list.sort(cmp=my_cmp)

    ours_list = os.listdir(ours_dir)
    ours_list.sort(cmp=my_cmp)

    last_frame_list = os.listdir(last_frame_dir)
    last_frame_list.sort(cmp=my_cmp)

    N = len(before_ICP_list)
    # print(before_ICP_list)
    images_dict = {k:[] for k in ['before_ICP', 'after_ICP', 'ours', 'last_frame']}
    print('loading images...')
    for i in tqdm(range(N)):
        images_dict['before_ICP'].append(cv2.imread(os.path.join(before_ICP_dir, before_ICP_list[i]),
                                                    cv2.IMREAD_COLOR))
        images_dict['after_ICP'].append(cv2.imread(os.path.join(after_ICP_dir, after_ICP_list[i]),
                                                    cv2.IMREAD_COLOR))
        images_dict['ours'].append(cv2.imread(os.path.join(ours_dir, ours_list[i]),
                                                    cv2.IMREAD_COLOR))
        images_dict['last_frame'].append(cv2.imread(os.path.join(last_frame_dir, last_frame_list[i]),
                                                    cv2.IMREAD_COLOR))

    height, width, channel = images_dict['before_ICP'][0].shape
    print(height, width)
    width = 800
    height = 600

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(os.path.join(exp_dir, 'video_full/Before_After_Ours_LastFrame.avi'), fourcc, 5.0, (width, height))

    print('writing video...')
    for i in tqdm(range(N)):
        res_img_1 = images_dict['before_ICP'][i]
        res_img_2 = images_dict['after_ICP'][i]
        res_img_3 = images_dict['ours'][i]
        res_img_4 = images_dict['last_frame'][i]
        res_img = np.vstack((np.hstack((res_img_1, res_img_2)), np.hstack((res_img_3, res_img_4))))
        im_scale = 0.5
        res_img = cv2.resize(res_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)

        if res_img.shape[0] == 480:
            im_scale = 600.0/480.0
            res_img = cv2.resize(res_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        video.write(res_img)

    video.release()
    os.popen('ffmpeg -i {} -vcodec mpeg4 -acodec copy -preset placebo -crf 1 -b:v 1550k {}'.format(
        os.path.join(exp_dir, 'video_full/Before_After_Ours_LastFrame.avi'),
        os.path.join(exp_dir, 'video_full/Before_After_Ours_LastFrame_compressed.avi')))







