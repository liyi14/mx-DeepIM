# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
"""
generate rendered from rendered poses
generate (real rendered) pair list file for training (or test)
"""
from __future__ import print_function, division
import numpy as np
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, ".."))

from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.render_glumpy.render_py import Render_Py
import lib.pair_matching.RT_transform as se3
import cv2
import random

random.seed(2333)
np.random.seed(2333)
from tqdm import tqdm

if __name__ == "__main__":
    idx2class = {
        1: "ape",
        2: "benchvise",
        # 3: 'bowl',
        4: "camera",
        5: "can",
        6: "cat",
        # 7: 'cup',
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
        13: "iron",
        14: "lamp",
        15: "phone",
    }
    classes = idx2class.values()
    classes = sorted(classes)

    def class2idx(class_name, idx2class=idx2class):
        for k, v in idx2class.items():
            if v == class_name:
                return k

    width = 640
    height = 480
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    ZNEAR = 0.25
    ZFAR = 6.0
    depth_factor = 1000

    # output_path
    LINEMOD_root = os.path.join(
        cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine"
    )
    LINEMOD_syn_root = os.path.join(
        cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine_syn"
    )
    rendered_pose_path = "%s/LM6d_ds_rendered_pose_{}.txt" % (
        os.path.join(LINEMOD_syn_root, "poses", "rendered_poses")
    )
    observed_list_path = "%s/LM6d_data_syn_{}_observed_{}.txt" % (
        os.path.join(LINEMOD_syn_root, "image_set/observed")
    )

    # output_path
    rendered_root_dir = os.path.join(LINEMOD_syn_root, "data", "rendered")
    pair_set_dir = os.path.join(LINEMOD_syn_root, "image_set")
    mkdir_if_missing(rendered_root_dir)
    mkdir_if_missing(pair_set_dir)

    gen_images = True

    for class_idx, class_name in enumerate(classes):
        train_pair = []
        print("start ", class_idx, class_name)
        if class_name in ["__back_ground__"]:
            continue
        # uncomment here to only generate data for ape
        # if class_name not in ['ape']:
        #     continue

        if gen_images:
            # init render
            model_dir = os.path.join(LINEMOD_root, "models", class_name)
            render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for set_type in ["train"]:
            with open(observed_list_path.format(set_type, class_name)) as f:
                real_list = [x.strip() for x in f.readlines()]
            with open(rendered_pose_path.format(class_name)) as f:
                str_rendered_pose_list = [x.strip().split(" ") for x in f.readlines()]
            rendered_pose_list = np.array(
                [[float(x) for x in each_pose] for each_pose in str_rendered_pose_list]
            )
            rendered_per_real = 1
            assert len(rendered_pose_list) == 1 * len(real_list), "{} vs {}".format(
                len(rendered_pose_list), len(real_list)
            )
            for idx, real_index in enumerate(tqdm(real_list)):
                video_name, real_prefix = real_index.split("/")
                rendered_dir = os.path.join(rendered_root_dir, video_name)
                mkdir_if_missing(rendered_dir)
                for inner_idx in range(rendered_per_real):
                    if gen_images:
                        image_file = os.path.join(
                            rendered_dir,
                            "{}_{}_{}-color.png".format(
                                class_name, real_prefix, inner_idx
                            ),
                        )
                        depth_file = os.path.join(
                            rendered_dir,
                            "{}_{}_{}-depth.png".format(
                                class_name, real_prefix, inner_idx
                            ),
                        )
                        # if os.path.exists(image_file) and os.path.exists(depth_file):
                        #     continue
                        rendered_idx = idx * rendered_per_real + inner_idx
                        pose_rendered_q = rendered_pose_list[rendered_idx]

                        rgb_gl, depth_gl = render_machine.render(
                            pose_rendered_q[:4], pose_rendered_q[4:]
                        )
                        rgb_gl = rgb_gl.astype("uint8")

                        depth_gl = (depth_gl * depth_factor).astype(np.uint16)

                        cv2.imwrite(image_file, rgb_gl)
                        cv2.imwrite(depth_file, depth_gl)

                        pose_file = os.path.join(
                            rendered_dir,
                            "{}_{}_{}-pose.txt".format(
                                class_name, real_prefix, inner_idx
                            ),
                        )
                        text_file = open(pose_file, "w")
                        text_file.write("{}\n".format(class_idx))
                        pose_rendered_m = np.zeros((3, 4))
                        pose_rendered_m[:, :3] = se3.quat2mat(pose_rendered_q[:4])
                        pose_rendered_m[:, 3] = pose_rendered_q[4:]
                        pose_ori_m = pose_rendered_m
                        pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}".format(
                            pose_ori_m[0, 0],
                            pose_ori_m[0, 1],
                            pose_ori_m[0, 2],
                            pose_ori_m[0, 3],
                            pose_ori_m[1, 0],
                            pose_ori_m[1, 1],
                            pose_ori_m[1, 2],
                            pose_ori_m[1, 3],
                            pose_ori_m[2, 0],
                            pose_ori_m[2, 1],
                            pose_ori_m[2, 2],
                            pose_ori_m[2, 3],
                        )
                        text_file.write(pose_str)

                    train_pair.append(
                        "{} {}/{}_{}_{}".format(
                            real_index, video_name, class_name, real_prefix, inner_idx
                        )
                    )

            pair_set_file = os.path.join(
                pair_set_dir, "train_{}.txt".format(class_name)
            )
            train_pair = sorted(train_pair)
            with open(pair_set_file, "w") as text_file:
                for x in train_pair:
                    text_file.write("{}\n".format(x))

        print(class_name, " done")
    print("{} finished".format(__file__))
