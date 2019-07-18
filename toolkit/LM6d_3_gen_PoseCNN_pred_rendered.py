# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, ".."))
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.render_glumpy.render_py import Render_Py
from lib.pair_matching import RT_transform
import scipy.io as sio
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    class_name_list = [
        "__back_ground__",
        "ape",
        "benchvise",
        "camera",
        "can",
        "cat",
        "driller",
        "duck",
        "eggbox",
        "glue",
        "holepuncher",
        "iron",
        "lamp",
        "phone",
    ]
    big_idx2class = {
        1: "ape",
        2: "benchvise",
        4: "camera",
        5: "can",
        6: "cat",
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
        13: "iron",
        14: "lamp",
        15: "phone",
    }

    class2big_idx = {}
    for key in big_idx2class:
        class2big_idx[big_idx2class[key]] = key

    cur_path = os.path.abspath(os.path.dirname(__file__))

    # config for renderer
    width = 640
    height = 480
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    ZNEAR = 0.25
    ZFAR = 6.0
    depth_factor = 1000

    gen_images = True
    version = "PoseCNN"  # you can change it to your own method
    LINEMOD_root = os.path.join(cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine")
    observed_root_dir = os.path.join(LINEMOD_root, "data/observed")
    observed_meta_path = "%s/{}-meta.mat" % (observed_root_dir)

    # config for external method results
    keyframe_path = "%s/{}_test.txt" % (os.path.join(LINEMOD_root, "image_set/observed"))
    exMethod_pred_dir = os.path.join(
        cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine/{}_LINEMOD_6D_results".format(version)
    )

    # output_path
    rendered_root_dir = os.path.join(LINEMOD_root, "data/rendered_val_{}".format(version))
    pair_set_dir = os.path.join(LINEMOD_root, "image_set")
    mkdir_if_missing(rendered_root_dir)
    mkdir_if_missing(pair_set_dir)

    all_pair = []
    for small_class_idx, class_name in enumerate(class_name_list):
        if class_name in ["__back_ground__"]:
            continue
        # uncomment here to only generate data for ape
        # if class_name not in ['ape']:
        #     continue
        big_class_idx = class2big_idx[class_name]

        # init render
        model_dir = os.path.join(LINEMOD_root, "models/{}".format(class_name))
        if gen_images:
            render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        with open(keyframe_path.format(class_name)) as f:
            observed_index_list = [x.strip() for x in f.readlines()]
        video_name_list = [x.split("/")[0] for x in observed_index_list]
        observed_prefix_list = [x.split("/")[1] for x in observed_index_list]

        all_pair = []
        for idx, observed_index in enumerate(tqdm(observed_index_list)):
            rendered_dir = os.path.join(rendered_root_dir, video_name_list[idx], class_name)
            mkdir_if_missing(rendered_dir)
            exMethod_idx = idx
            exMethod_pred_file = os.path.join(exMethod_pred_dir, class_name, "{:04d}.mat".format(exMethod_idx))
            exMethod_pred = sio.loadmat(exMethod_pred_file)
            labels = exMethod_pred["rois"][:, 1]  # 1: found; -1: not found
            if labels != -1:
                try:
                    meta_data = sio.loadmat(observed_meta_path.format(observed_index))
                except:  # noqa:E722
                    raise Exception(observed_index)

                proposal_idx = np.where(labels == 1)
                assert len(proposal_idx) == 1
                pose_ori_q = exMethod_pred["poses"][proposal_idx].reshape(7)
                pose_icp_q = exMethod_pred["poses_icp"][proposal_idx].reshape(7)
                pose_ori_m = RT_transform.se3_q2m(pose_ori_q)
                pose_ori_q[:4] = RT_transform.mat2quat(pose_ori_m[:, :3])
                pose_icp_m = RT_transform.se3_q2m(pose_icp_q)
                pose_icp_q[:4] = RT_transform.mat2quat(pose_icp_m[:, :3])

                pose_gt = meta_data["poses"]
                if len(pose_gt.shape) > 2:
                    pose_gt = pose_gt[:, :, list(meta_data["cls_indexes"][0]).index(big_class_idx)]
                if idx % 100 == 0:  # to check whether the pose representation is different
                    print("{}, {:04d}, {}".format(class_name, idx, RT_transform.calc_rt_dist_m(pose_ori_m, pose_gt)))

                pose_ori_file = os.path.join(
                    rendered_dir, "{}_{}_{}-pose.txt".format(class_name, observed_prefix_list[idx], 0)
                )
                pose_icp_file = os.path.join(
                    rendered_dir, "{}_{}_{}-pose_icp.txt".format(class_name, observed_prefix_list[idx], 0)
                )
                image_file = os.path.join(
                    rendered_dir, "{}_{}_{}-color.png".format(class_name, observed_prefix_list[idx], 0)
                )
                depth_file = os.path.join(
                    rendered_dir, "{}_{}_{}-depth.png".format(class_name, observed_prefix_list[idx], 0)
                )
                segmentation_file = os.path.join(
                    rendered_dir, "{}_{}_{}-label.png".format(class_name, observed_prefix_list[idx], 0)
                )

                if gen_images:
                    rgb_gl, depth_gl = render_machine.render(pose_ori_q[:4], pose_ori_q[4:])
                    depth_gl = (depth_gl * depth_factor).astype(np.uint16)
                    segmentation = exMethod_pred["labels"] * big_class_idx
                    cv2.imwrite(image_file, rgb_gl)
                    cv2.imwrite(depth_file, depth_gl)
                    cv2.imwrite(segmentation_file, segmentation)

                text_file = open(pose_ori_file, "w")
                text_file.write("{}\n".format(big_class_idx))
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

                text_file = open(pose_icp_file, "w")
                text_file.write("{}\n".format(big_class_idx))
                pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}".format(
                    pose_icp_m[0, 0],
                    pose_icp_m[0, 1],
                    pose_icp_m[0, 2],
                    pose_icp_m[0, 3],
                    pose_icp_m[1, 0],
                    pose_icp_m[1, 1],
                    pose_icp_m[1, 2],
                    pose_icp_m[1, 3],
                    pose_icp_m[2, 0],
                    pose_icp_m[2, 1],
                    pose_icp_m[2, 2],
                    pose_icp_m[2, 3],
                )
                text_file.write(pose_str)

                all_pair.append(
                    "{} {}/{}/{}_{}_{}".format(
                        observed_index, video_name_list[idx], class_name, class_name, observed_prefix_list[idx], 0
                    )
                )
            else:
                print("no exMethod_pred in {} {} {}".format(class_name, idx, observed_index))

            pair_set_file = os.path.join(pair_set_dir, "{}_val_{}.txt".format(version, class_name))
            with open(pair_set_file, "w") as text_file:
                for x in all_pair:
                    text_file.write("{}\n".format(x))

        print(class_name, " done")
