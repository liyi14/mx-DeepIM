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
sys.path.insert(1, os.path.join(cur_path, ".."))
import cv2
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video iter")
    parser.add_argument(
        "--exp_dir",
        required=True,
        help="exp_dir for generating video iter, e.g.: "
        "output/deepim/deepim_flownet_occLM_v1_se3_ex_u2s16_driller_iter_v42_zoom_NF_NF_NF_NF_real_1gpus/yu_val_driller/video_iter",  # noqa:E501
    )
    args = parser.parse_args()
    return args


def my_cmp(x, y):
    x_idx = int(x.split("/")[-1].split(".")[0].split("_")[-1])
    y_idx = int(y.split("/")[-1].split(".")[0].split("_")[-1])
    return x_idx - y_idx


if __name__ == "__main__":
    args = parse_args()
    exp_dir = args.exp_dir
    print("exp_dir: ", exp_dir)
    pose_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if "pose" in d]
    pose_dirs = sorted(pose_dirs)

    zoom_mask_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if "zoom_mask" in d]
    zoom_mask_dirs = sorted(zoom_mask_dirs)

    z_image_real_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if "zoom_image_real" in d]
    z_image_real_dirs = sorted(z_image_real_dirs)

    z_image_rendered_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if "zoom_image_rendered" in d]
    z_image_rendered_dirs = sorted(z_image_rendered_dirs)

    pose_path_list = []
    for pose_dir in pose_dirs:
        files = [os.path.join(pose_dir, fn) for fn in os.listdir(pose_dir) if ".png" in fn]
        files.sort(cmp=my_cmp)
        for i in range(len(files)):
            if i == 0 or i == len(files) - 1:
                for j in range(5):
                    pose_path_list.append(files[i])
            else:
                pose_path_list.append(files[i])

    zoom_mask_path_list = []
    for z_mask_dir in zoom_mask_dirs:
        files = [os.path.join(z_mask_dir, fn) for fn in os.listdir(z_mask_dir) if ".png" in fn]
        files.sort(cmp=my_cmp)
        for i in range(len(files)):
            if i == 0 or i == len(files) - 1:
                for j in range(5):
                    zoom_mask_path_list.append(files[i])
            else:
                zoom_mask_path_list.append(files[i])

    z_image_real_path_list = []
    z_image_rendered_path_list = []
    for i in range(len(z_image_real_dirs)):
        z_image_real_dir = z_image_real_dirs[i]
        z_image_rendered_dir = z_image_rendered_dirs[i]

        real_files = [os.path.join(z_image_real_dir, fn) for fn in os.listdir(z_image_real_dir) if ".png" in fn]
        rendered_files = [
            os.path.join(z_image_rendered_dir, fn) for fn in os.listdir(z_image_rendered_dir) if ".png" in fn
        ]
        real_files.sort(cmp=my_cmp)
        rendered_files.sort(cmp=my_cmp)
        for i in range(len(real_files)):
            if i == 0 or i == len(real_files) - 1:
                for j in range(5):
                    z_image_real_path_list.append(real_files[i])
                    z_image_rendered_path_list.append(rendered_files[i])
            else:
                z_image_real_path_list.append(real_files[i])
                z_image_rendered_path_list.append(rendered_files[i])

    N = len(pose_path_list)
    images_dict = {k: [] for k in ["flow", "pose", "mask", "zoom_mask", "zoom_image_real", "zoom_image_rendered"]}
    print("loading images...")

    for i in tqdm(range(N)):
        images_dict["pose"].append(cv2.imread(pose_path_list[i], cv2.IMREAD_COLOR))
        images_dict["zoom_mask"].append(cv2.imread(zoom_mask_path_list[i], cv2.IMREAD_COLOR))
        images_dict["zoom_image_real"].append(cv2.imread(z_image_real_path_list[i], cv2.IMREAD_COLOR))
        images_dict["zoom_image_rendered"].append(cv2.imread(z_image_rendered_path_list[i], cv2.IMREAD_COLOR))

    height, width, channel = images_dict["pose"][0].shape
    print(height, width)
    width = 800
    height = 600

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_pose_mask_image = cv2.VideoWriter(
        os.path.join(exp_dir, "../video_full/zoom_image_mask_pose.avi"), fourcc, 2.0, (width, height)
    )

    print("writing video...")
    for i in tqdm(range(N)):
        res_img_1 = images_dict["zoom_image_real"][i]
        res_img_2 = images_dict["zoom_mask"][i]
        res_img_3 = images_dict["zoom_image_rendered"][i]
        res_img_4 = images_dict["pose"][i]
        res_img = np.vstack((np.hstack((res_img_1, res_img_2)), np.hstack((res_img_3, res_img_4))))
        im_scale = 0.5
        res_img = cv2.resize(res_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        if res_img.shape[0] == 480:
            im_scale = 600.0 / 480.0
            res_img = cv2.resize(res_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        video_pose_mask_image.write(res_img)

    video_pose_mask_image.release()

    os.popen(
        "ffmpeg -i {} -vcodec mpeg4 -acodec copy -preset placebo -crf 1 -b:v 1550k {}".format(
            os.path.join(exp_dir, "../video_full/zoom_image_mask_pose.avi"),
            os.path.join(exp_dir, "../video_full/zoom_image_mask_pose_compressed.avi"),
        )
    )
