# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, ".."))
import lib.pair_matching.RT_transform as se3
from tqdm import tqdm
from lib.utils.utils import read_img

idx2class = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
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

K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


def class2idx(class_name, idx2class=idx2class):
    for k, v in idx2class.items():
        if v == class_name:
            return k


version = "v1"

LINEMOD_root = os.path.join(cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine")
LINEMOD_syn_root = os.path.join(cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine_syn")
# input render observed poses
gt_observed_dir = os.path.join(LINEMOD_syn_root, "data", "gt_observed")
observed_dir = os.path.join(LINEMOD_syn_root, "data", "observed")
rendered_dir = os.path.join(LINEMOD_syn_root, "data", "rendered")

image_set_dir = os.path.join(LINEMOD_syn_root, "image_set")
observed_set_dir = os.path.join(LINEMOD_syn_root, "image_set/observed")


def check():
    cls_name = "ape"
    set_file = os.path.join(image_set_dir, "train_{}.txt".format(cls_name))
    with open(set_file, "r") as f:
        pairs = [line.strip() for line in f.readlines()]

    for pair in pairs:
        observed_idx, rendered_idx = pair.split()
        observed_color = read_img(os.path.join(observed_dir, "{}-color.png".format(observed_idx)), 3)
        gt_rendered_color = read_img(os.path.join(gt_observed_dir, "{}-color.png".format(observed_idx)), 3)  # noqa:F841
        rendered_color = read_img(os.path.join(rendered_dir, "{}-color.png".format(rendered_idx)), 3)

        fig = plt.figure(dpi=150)  # noqa:F841
        plt.subplot(2, 3, 1)
        plt.imshow(observed_color[:, :, [2, 1, 0]])  # BGR->RGB
        plt.title("observed")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(rendered_color[:, :, [2, 1, 0]])
        plt.title("rendered")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        observed_rendered_diff = np.abs(observed_color[:, :, [2, 1, 0]] - rendered_color[:, :, [2, 1, 0]])
        plt.imshow(observed_rendered_diff.astype(np.uint8))
        plt.title("observed - rendered")
        plt.axis("off")

        plt.show()


def angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))  # -> cosine of the angle
    rad = np.arccos(np.clip(c, -1, 1))
    deg = rad / np.pi * 180
    return deg


def check_rot_distribution():
    pz = np.array([0, 0, 1])
    new_points = {}
    pose_dict = {}
    trans_stat = {}
    quat_stat = {}
    for cls_idx, cls_name in idx2class.items():
        print(cls_name)
        if cls_name != "bowl":
            continue
        new_points[cls_name] = {"pz": []}
        train_idx_file = os.path.join(observed_set_dir, "LM6d_data_syn_train_observed_{}.txt".format(cls_name))
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

        vis_points()
    return pose_dict, quat_stat, trans_stat, new_points


def load_object_points(point_path):
    # print(point_path)
    assert os.path.exists(point_path), "Path does not exist: {}".format(point_path)
    points = np.loadtxt(point_path)
    return points


def points_to_2D(points, R, T, K):
    """
    :param points: (N, 3)
    :param R: (3, 3)
    :param T: (3, )
    :param K: (3, 3)
    :return:
    """
    points_in_world = np.matmul(R, points.T) + T.reshape((3, 1))  # (3, N)
    points_in_camera = np.matmul(K, points_in_world)  # (3, N)
    N = points_in_world.shape[1]
    points_2D = np.zeros((2, N))
    points_2D[0, :] = points_in_camera[0, :] / points_in_world[2, :]
    points_2D[1, :] = points_in_camera[1, :] / points_in_world[2, :]
    z = points_in_world[2, :]
    return points_2D, z


def check_model_points():
    height = 480
    width = 640
    LINEMOD_root = os.path.join(cur_path, "../data/LINEMOD_6D")
    model_root = os.path.join(LINEMOD_root, "models")
    cls_name = "bowl"
    points = np.loadtxt(os.path.join(model_root, cls_name, "points.xyz"))
    # points_1 = np.loadtxt(os.path.join(model_root, cls_name, '{}.xyz'.format(cls_name)))
    for cls_idx, cls_name in idx2class.items():
        print(cls_name)
        if cls_name != "bowl":
            continue

        train_idx_file = os.path.join(observed_set_dir, "LM6d_data_syn_train_observed_{}.txt".format(cls_name))
        with open(train_idx_file, "r") as f:
            observed_indices = [line.strip() for line in f.readlines()]

        for observed_i, observed_idx in enumerate(tqdm(observed_indices)):
            prefix = observed_idx.split("/")[1]
            pose_path = os.path.join(gt_observed_dir, cls_name, "{}-pose.txt".format(prefix))
            observed_color_path = os.path.join(gt_observed_dir, cls_name, "{}-color.png".format(prefix))
            observed_color = read_img(observed_color_path)
            assert os.path.exists(pose_path), "path {} not exists".format(pose_path)
            pose = np.loadtxt(pose_path, skiprows=1)
            R = pose[:3, :3]
            T = pose[:3, 3]
            points_2d, z = points_to_2D(points, R, T, K)
            points_2d_ = np.round(points_2d).astype(np.int64)
            im_p = np.zeros((height, width))
            im_p[points_2d_[1, :], points_2d_[0, :]] = 1
            fig = plt.figure()  # noqa:F401
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.imshow(im_p)

            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.imshow(observed_color[:, :, [2, 1, 0]])

            plt.show()


if __name__ == "__main__":
    check()
