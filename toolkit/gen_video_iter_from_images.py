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
    parser = argparse.ArgumentParser(description='Generate video iter')
    parser.add_argument('--exp_dir', required=True, help='exp_dir for generating video iter, e.g.: '
        'output/deepim/deepim_flownet_occLM_v1_se3_ex_u2s16_driller_iter_v42_zoom_NF_NF_NF_NF_real_1gpus/yu_val_driller/video_iter')
    args = parser.parse_args()
    return args



def my_cmp(x, y):
    x_idx = int(x.split('/')[-1].split('.')[0].split('_')[-1])
    y_idx = int(y.split('/')[-1].split('.')[0].split('_')[-1])
    return x_idx - y_idx

if __name__ == '__main__':
    args = parse_args()
    exp_dir = args.exp_dir
    print('exp_dir: ', exp_dir)
    flow_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if 'flow' in d]
    flow_dirs = sorted(flow_dirs)
    pose_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if 'pose' in d]
    pose_dirs = sorted(pose_dirs)

    mask_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if 'mask' in d]
    mask_dirs = sorted(mask_dirs)


    flow_path_list = []
    for flow_dir in flow_dirs:
        files = [os.path.join(flow_dir, fn) for fn in os.listdir(flow_dir) if '.png' in fn]
        files.sort(cmp=my_cmp)
        for i in range(len(files)):
            if i == 0 or i == len(files)-2:
                for j in range(5):
                    flow_path_list.append(files[i])
            if i == len(files)-1:
                pass
            else:
                flow_path_list.append(files[i])

    pose_path_list = []
    for pose_dir in pose_dirs:
        files = [os.path.join(pose_dir, fn) for fn in os.listdir(pose_dir) if '.png' in fn]
        files.sort(cmp=my_cmp)
        for i in range(len(files)):
            if i == 0 or i == len(files)-2:
                for j in range(5):
                    pose_path_list.append(files[i])
            if i == len(files)-1:
                pass
            else:
                pose_path_list.append(files[i])

    mask_path_list = []
    for mask_dir in mask_dirs:
        files = [os.path.join(mask_dir, fn) for fn in os.listdir(mask_dir) if '.png' in fn]
        files.sort(cmp=my_cmp)
        for i in range(len(files)):
            mask_path_list.append(files[i])


    N = len(pose_path_list)
    images_dict = {k:[] for k in ['flow', 'pose', 'mask']}
    print('loading images...')

    N_mask = len(mask_path_list)
    for i in range(N_mask):
        if len(mask_path_list) != 0:
            images_dict['mask'].append(cv2.imread(mask_path_list[i],
                                                  cv2.IMREAD_COLOR))

    for i in tqdm(range(N)):
        if len(flow_path_list) != 0:
            images_dict['flow'].append(cv2.imread(flow_path_list[i],
                                                cv2.IMREAD_COLOR))
        images_dict['pose'].append(cv2.imread(pose_path_list[i],
                                            cv2.IMREAD_COLOR))

    height, width, channel = images_dict['pose'][0].shape
    print(height, width)
    width = 800
    height = 600

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if len(mask_path_list) != 0:
        video_mask = cv2.VideoWriter(os.path.join(exp_dir, '../video_full/mask.avi'), fourcc, 5.0, (width, height))

    if len(flow_path_list) != 0:
        video_flow = cv2.VideoWriter(os.path.join(exp_dir, '../video_full/flow.avi'), fourcc, 5.0, (width, height))
        video = cv2.VideoWriter(os.path.join(exp_dir, '../video_full/pose_flow.avi'), fourcc, 5.0, (width * 2, height))
    video_pose = cv2.VideoWriter(os.path.join(exp_dir, '../video_full/pose.avi'), fourcc, 10.0, (width, height))

    print('writing video...')
    for i in tqdm(range(N)):
        res_img_1 = images_dict['pose'][i]
        if res_img_1.shape[0] == 480:
            im_scale = 600.0/480.0
            res_img_1 = cv2.resize(res_img_1, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        video_pose.write(res_img_1)

        if len(flow_path_list) != 0:
            res_img_2 = images_dict['flow'][i]
            if res_img_2.shape[0] == 480:
                im_scale = 600.0 / 480.0
                res_img_2 = cv2.resize(res_img_2, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)

            res_img = np.hstack((res_img_1, res_img_2))

            video_flow.write(res_img_2)
            video.write(res_img)

    for i in range(N_mask):
        res_img = images_dict['mask'][i]
        if res_img.shape[0] == 480:
            im_scale = 600.0 / 480.0
            res_img = cv2.resize(res_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        video_mask.write(res_img)

    video_pose.release()
    if len(flow_path_list) != 0:
        video_flow.release()
        video.release()

    if N_mask != 0:
        video_mask.release()


    os.popen('ffmpeg -i {} -vcodec mpeg4 -acodec copy -preset placebo -crf 1 -b:v 1550k {}'.format(
        os.path.join(exp_dir, '../video_full/pose.avi'),
        os.path.join(exp_dir, '../video_full/pose_compressed.avi')))

    if N_mask != 0:
        os.popen('ffmpeg -i {} -vcodec mpeg4 -acodec copy -preset placebo -crf 1 -b:v 1550k {}'.format(
            os.path.join(exp_dir, '../video_full/mask.avi'),
            os.path.join(exp_dir, '../video_full/mask_compressed.avi')))

    if len(flow_path_list) != 0:
        os.popen('ffmpeg -i {} -vcodec mpeg4 -acodec copy -preset placebo -crf 1 -b:v 1550k {}'.format(
            os.path.join(exp_dir, '../video_full/flow.avi'),
            os.path.join(exp_dir, '../video_full/flow_compressed.avi')))
        os.popen('ffmpeg -i {} -vcodec mpeg4 -acodec copy -preset placebo -crf 1 -b:v 1550k {}'.format(
            os.path.join(exp_dir, '../video_full/pose_flow.avi'),
            os.path.join(exp_dir, '../video_full/pose_flow_compressed.avi')))








