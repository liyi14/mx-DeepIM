# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
"""
generate syn_poses
"""
from __future__ import division, print_function
import numpy as np
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, ".."))
from lib.utils.mkdir_if_missing import mkdir_if_missing
import lib.pair_matching.RT_transform as se3
import random
from six.moves import cPickle
from tqdm import tqdm

random.seed(2333)
np.random.seed(2333)

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


# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  # for lm
ZNEAR = 0.25
ZFAR = 6.0

depth_factor = 1000

LINEMOD_root = os.path.join(cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine")
gt_observed_dir = os.path.join(LINEMOD_root, "data/gt_observed")
observed_set_dir = os.path.join(LINEMOD_root, "image_set/observed")

# output path
LINEMOD_syn_root = os.path.join(cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine_syn")
pose_dir = os.path.join(LINEMOD_syn_root, "poses")  # single object in each image
mkdir_if_missing(pose_dir)
print("target path: {}".format(pose_dir))

NUM_IMAGES = 10000


def angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))  # -> cosine of the angle
    rad = np.arccos(np.clip(c, -1, 1))
    deg = rad / np.pi * 180
    return deg


def stat_poses():
    pz = np.array([0, 0, 1])
    new_points = {}
    pose_dict = {}
    trans_stat = {}
    quat_stat = {}
    for cls_idx, cls_name in idx2class.items():
        # uncomment here to only generate data for ape
        # if cls_name != 'ape':
        #     continue
        new_points[cls_name] = {"pz": []}
        train_idx_file = os.path.join(observed_set_dir, "{}_train.txt".format(cls_name))
        with open(train_idx_file, "r") as f:
            observed_indices = [line.strip() for line in f.readlines()]

        num_observed = len(observed_indices)
        pose_dict[cls_name] = np.zeros((num_observed, 7))  # quat, translation
        trans_stat[cls_name] = {}
        quat_stat[cls_name] = {}
        for observed_i, observed_idx in enumerate(tqdm(observed_indices)):
            prefix = observed_idx.split("/")[1]
            pose_path = os.path.join(gt_observed_dir, cls_name, "{}-pose.txt".format(prefix))
            assert os.path.exists(pose_path), "path {} not exists".format(pose_path)
            pose = np.loadtxt(pose_path, skiprows=1)
            rot = pose[:3, :3]
            # print(rot)
            quat = np.squeeze(se3.mat2quat(rot))
            src_trans = pose[:3, 3]
            pose_dict[cls_name][observed_i, :4] = quat
            pose_dict[cls_name][observed_i, 4:] = src_trans

            new_pz = np.dot(rot, pz.reshape((-1, 1))).reshape((3,))
            new_points[cls_name]["pz"].append(new_pz)
        new_points[cls_name]["pz"] = np.array(new_points[cls_name]["pz"])
        new_points[cls_name]["pz_mean"] = np.mean(new_points[cls_name]["pz"], 0)
        new_points[cls_name]["pz_std"] = np.std(new_points[cls_name]["pz"], 0)

        trans_mean = np.mean(pose_dict[cls_name][:, 4:], 0)
        trans_std = np.std(pose_dict[cls_name][:, 4:], 0)
        trans_stat[cls_name]["trans_mean"] = trans_mean
        trans_stat[cls_name]["trans_std"] = trans_std

        quat_mean = np.mean(pose_dict[cls_name][:, :4], 0)
        quat_std = np.std(pose_dict[cls_name][:, :4], 0)
        quat_stat[cls_name]["quat_mean"] = quat_mean
        quat_stat[cls_name]["quat_std"] = quat_std

        print("new z: ", "mean: ", new_points[cls_name]["pz_mean"], "std: ", new_points[cls_name]["pz_std"])

        new_points[cls_name]["angle"] = []  # angle between mean vector and points
        pz_mean = new_points[cls_name]["pz_mean"]
        for p_i in range(new_points[cls_name]["pz"].shape[0]):
            deg = angle(pz_mean, new_points[cls_name]["pz"][p_i, :])
            new_points[cls_name]["angle"].append(deg)
        new_points[cls_name]["angle"] = np.array(new_points[cls_name]["angle"])

        print(
            "angle mean: ",
            np.mean(new_points[cls_name]["angle"]),
            "angle std: ",
            np.std(new_points[cls_name]["angle"]),
            "angle max: ",
            np.max(new_points[cls_name]["angle"]),
        )
        new_points[cls_name]["angle_max"] = np.max(new_points[cls_name]["angle"])
        print()

        def vis_points():
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa:F401

            sel_p = "pz"
            ax = plt.figure().add_subplot(111, projection="3d")
            ax.scatter(
                new_points[cls_name][sel_p][:, 0],
                new_points[cls_name][sel_p][:, 1],
                new_points[cls_name][sel_p][:, 2],
                c="r",
                marker="^",
            )
            ax.scatter(0, 0, 0, c="b", marker="o")
            ax.scatter(0, 0, 1, c="b", marker="o")
            ax.scatter(0, 1, 0, c="b", marker="o")
            ax.scatter(1, 0, 0, c="b", marker="o")
            ax.quiver(0, 0, 0, 0, 0, 1)
            pz_mean = new_points[cls_name]["pz_mean"]
            ax.quiver(0, 0, 0, pz_mean[0], pz_mean[1], pz_mean[2])

            ax.scatter(pz_mean[0], pz_mean[1], pz_mean[2], c="b", marker="o")
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            ax.set_zlabel("Z Label")
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            plt.title(cls_name + "-" + sel_p)
            plt.show()

        # vis_points()
    return pose_dict, quat_stat, trans_stat, new_points


def gen_poses():
    pz = np.array([0, 0, 1])
    pose_dict, quat_stat, trans_stat, new_points = stat_poses()
    observed_prefix_list = ["{:06d}".format(i + 1) for i in range(NUM_IMAGES)]
    sel_classes = classes
    observed_pose_dict = {cls_name: np.zeros((NUM_IMAGES, 7)) for cls_name in sel_classes}

    for cls_i, cls_name in enumerate(sel_classes):
        # uncomment here to only generate data for ape
        # if cls_name not in ['ape']:
        #     continue
        # src_quat_mean = quat_stat[cls_name]['quat_mean']
        # src_quat_std = quat_stat[cls_name]['quat_std']
        src_trans_mean = trans_stat[cls_name]["trans_mean"]
        src_trans_std = trans_stat[cls_name]["trans_std"]
        deg_max = new_points[cls_name]["angle_max"]

        for i in tqdm(range(NUM_IMAGES)):
            observed_prefix = observed_prefix_list[i]

            # randomly generate a pose
            tgt_quat = np.random.normal(0, 1, 4)
            tgt_quat = tgt_quat / np.linalg.norm(tgt_quat)
            if tgt_quat[0] < 0:
                tgt_quat *= -1
            tgt_trans = np.random.normal(src_trans_mean, src_trans_std)

            tgt_rot_m = se3.quat2mat(tgt_quat)
            new_pz = np.dot(tgt_rot_m, pz.reshape((-1, 1))).reshape((3,))
            pz_mean = new_points[cls_name]["pz_mean"]
            deg = angle(new_pz, pz_mean)

            # r_dist, t_dist = calc_rt_dist_q(tgt_quat, src_quat, tgt_trans, src_trans)
            transform = np.matmul(K, tgt_trans.reshape(3, 1))
            center_x = float(transform[0] / transform[2])
            center_y = float(transform[1] / transform[2])
            count = 0
            while deg > deg_max or not (48 < center_x < (640 - 48) and 48 < center_y < (480 - 48)):
                # randomly generate a pose
                tgt_quat = np.random.normal(0, 1, 4)
                tgt_quat = tgt_quat / np.linalg.norm(tgt_quat)
                if tgt_quat[0] < 0:
                    tgt_quat *= -1
                tgt_trans = np.random.normal(src_trans_mean, src_trans_std)

                tgt_rot_m = se3.quat2mat(tgt_quat)
                new_pz = np.dot(tgt_rot_m, pz.reshape((-1, 1))).reshape((3,))
                pz_mean = new_points[cls_name]["pz_mean"]
                deg = angle(new_pz, pz_mean)

                transform = np.matmul(K, tgt_trans.reshape(3, 1))
                center_x = float(transform[0] / transform[2])
                center_y = float(transform[1] / transform[2])
                count += 1
                if count % 100 == 0:
                    print(
                        observed_prefix,
                        cls_name,
                        count,
                        "deg < deg_max={}: {}, 48 < center_x < (640-48): {}, 48 < center_y < (480-48): {}".format(
                            deg_max, deg <= deg_max, 48 < center_x < (640 - 48), 48 < center_y < (480 - 48)
                        ),
                    )

            tgt_pose_q = np.zeros((7,), dtype="float32")
            tgt_pose_q[:4] = tgt_quat
            tgt_pose_q[4:] = tgt_trans
            observed_pose_dict[cls_name][i, :] = tgt_pose_q

    # write pose
    poses_file = os.path.join(pose_dir, "LM6d_ds_train_observed_pose_all.pkl")
    with open(poses_file, "wb") as f:
        cPickle.dump(observed_pose_dict, f, 2)


if __name__ == "__main__":
    # pose_dict, quat_stat, trans_stat, new_points = stat_poses()
    gen_poses()
    print("{} finished".format(__file__))
