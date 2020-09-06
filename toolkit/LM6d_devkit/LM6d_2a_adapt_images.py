# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yu Xiang
# --------------------------------------------------------
from __future__ import print_function, division
import sys
import os
import os.path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(1, osp.join(cur_dir, "../.."))
import numpy as np
import mmcv
import cv2
from shutil import copyfile
from tqdm import tqdm

import scipy.io as sio

LM6d_origin_root = osp.join(cur_dir, "../../data/LINEMOD_6D/LM6d_origin/test")
# following previous works, part of the observed images are used for training and only images.

LM6d_new_root = osp.join(cur_dir, "../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/data/observed")
model_dir = osp.join(cur_dir, "../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/models")
mmcv.mkdir_or_exist(LM6d_new_root)
print("target path: {}".format(LM6d_new_root))

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

class2idx = {cls_name: cls_idx for cls_idx, cls_name in idx2class.items()}


width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0

DEPTH_FACTOR = 1000


def write_pose_file(pose_file, class_idx, pose_ori_m):
    text_file = open(pose_file, "w")
    text_file.write("{}\n".format(class_idx))
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


def main():
    sel_classes = classes
    for cls_idx, cls_name in enumerate(classes):
        obj_id = class2idx[cls_name]
        if cls_name not in sel_classes:
            continue
        print(cls_idx, cls_name)
        observed_indices = []

        gt_path = osp.join(LM6d_origin_root, "{:06d}".format(obj_id), "scene_gt.json")
        gt_info_path = osp.join(LM6d_origin_root, "{:06d}".format(obj_id), "scene_gt_info.json")
        gt_dict = mmcv.load(gt_path)
        gt_info_dict = mmcv.load(gt_info_path)

        for str_im_id in tqdm(gt_dict):
            int_im_id = int(str_im_id)
            old_color_path = osp.join(LM6d_origin_root, "{:06d}/rgb/{:06d}.png".format(obj_id, int_im_id))
            assert osp.exists(old_color_path), old_color_path
            old_depth_path = osp.join(LM6d_origin_root, "{:06d}/depth/{:06d}.png".format(obj_id, int_im_id))
            assert osp.exists(old_depth_path), old_depth_path
            new_img_id = int_im_id + 1

            # depth
            # depth = mmcv.imread(old_depth_path, "unchanged")
            # print(np.max(depth), np.min(depth))

            # print(color_img.shape)

            new_color_path = osp.join(
                LM6d_new_root, "{:02d}/{:06d}-color.png".format(obj_id, new_img_id)
            )
            new_depth_path = osp.join(
                LM6d_new_root, "{:02d}/{:06d}-depth.png".format(obj_id, new_img_id)
            )
            mmcv.mkdir_or_exist(osp.dirname(new_color_path))

            copyfile(old_color_path, new_color_path)
            copyfile(old_depth_path, new_depth_path)

            # meta and label
            meta_dict = {}
            num_instance = len(gt_dict[str_im_id])
            meta_dict["cls_indexes"] = np.zeros((1, num_instance), dtype=np.int32)
            meta_dict["boxes"] = np.zeros((num_instance, 4), dtype="float32")
            meta_dict["poses"] = np.zeros((3, 4, num_instance), dtype="float32")
            distances = []
            label_dict = {}
            for ins_id, instance in enumerate(gt_dict[str_im_id]):
                cur_obj_id = instance["obj_id"]
                meta_dict["cls_indexes"][0, ins_id] = cur_obj_id
                bbox = np.array(gt_info_dict[str_im_id][ins_id]["bbox_visib"])
                meta_dict["boxes"][ins_id, :] = bbox
                # pose
                pose = np.zeros((3, 4))

                R = np.array(instance["cam_R_m2c"]).reshape((3, 3))
                t = np.array(instance["cam_t_m2c"]) / 1000.0  # mm -> m
                pose[:3, :3] = R
                pose[:3, 3] = t
                distances.append(t[2])
                meta_dict["poses"][:, :, ins_id] = pose
                mask_path = osp.join(LM6d_origin_root, "{:06d}/mask/{:06d}_{:06d}.png".format(obj_id, int_im_id, ins_id))
                label = mmcv.imread(mask_path, "unchanged").astype(np.bool).astype(np.uint8)
                label_dict[cur_obj_id] = label
            meta_path = osp.join(LM6d_new_root, "{:02d}/{:06d}-meta.mat".format(obj_id, new_img_id))
            sio.savemat(meta_path, meta_dict)

            # NOTE: for linemod, this is not necessary, because only one object is labeled in each image
            dis_inds = sorted(range(len(distances)), key=lambda k: -distances[k])  # put deeper objects first
            # label
            res_label = np.zeros((480, 640))
            for dis_id in dis_inds:
                cls_id = meta_dict["cls_indexes"][0, dis_id]
                tmp_label = label_dict[cls_id]
                # label
                res_label[tmp_label == 1] = cls_id

            label_path = osp.join(LM6d_new_root, "{:02d}/{:06d}-label.png".format(obj_id, new_img_id))
            cv2.imwrite(label_path, res_label)

            # observed idx
            observed_indices.append("{:02d}/{:06d}".format(obj_id, new_img_id))


if __name__ == "__main__":
    main()
    print("{} finished".format(__file__))
