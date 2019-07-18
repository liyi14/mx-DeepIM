# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import os
import cv2
import random
from PIL import Image
from lib.pair_matching.load_object_points import load_object_points, load_points_from_obj
from lib.utils.mask_dilate import mask_dilate
from lib.utils.get_min_rect import get_min_rect

cur_dir = os.path.abspath(os.path.dirname(__file__))
# TODO: This two functions should be merged with individual data loader


def get_segmentation_image(segdb, config):
    """
    propocess image and return segdb
    :param segdb: a list of segdb
    :return: list of img as mxnet format
    """
    num_images = len(segdb)
    assert num_images > 0, "No images"
    processed_ims = []
    processed_segdb = []
    processed_seg_cls_gt = []
    for i in range(num_images):
        seg_rec = segdb[i]
        print(seg_rec["image"])
        assert os.path.exists(seg_rec["image"]), "{} does not exist".format(seg_rec["image"])
        im = np.array(cv2.imread(seg_rec["image"]))

        new_rec = seg_rec.copy()

        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec["im_info"] = im_info

        seg_cls_gt = np.array(Image.open(seg_rec["seg_cls_path"]))
        seg_cls_gt, seg_cls_gt_scale = resize(seg_cls_gt, target_size, max_size, interpolation=cv2.INTER_NEAREST)
        seg_cls_gt_tensor = transform_seg_gt(seg_cls_gt)

        processed_ims.append(im_tensor)
        processed_segdb.append(new_rec)
        processed_seg_cls_gt.append(seg_cls_gt_tensor)

    return processed_ims, processed_seg_cls_gt, processed_segdb


def get_pair_image(pairdb, config, phase="train", random_k=18):
    """
    preprocess image and return processed pairdb
    :param pairdb: a list of pairdb
    :return: list of img as in mxnet format
    pairdb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_pairs = len(pairdb)
    processed_ims_observed = []
    processed_ims_rendered = []
    scale_ind_list = []
    for i in range(num_pairs):
        pair_rec = pairdb[i]

        scale_ind = random.randrange(len(config.SCALES))
        scale_ind_list.append(scale_ind)
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        if pairdb[i]["img_flipped"]:
            raise Exception("NOT_IMPLEMENTED")

        # process rgb image
        image_observed_path = pair_rec["image_observed"]
        # image_observed_path = pair_rec['image_observed']
        assert os.path.exists(image_observed_path), "{} does not exist".format(pair_rec["image_observed"])
        im_observed = cv2.imread(image_observed_path, cv2.IMREAD_COLOR)

        assert os.path.exists(pair_rec["image_rendered"]), "{} does not exist".format(pair_rec["image_rendered"])
        im_rendered = cv2.imread(pair_rec["image_rendered"], cv2.IMREAD_COLOR)

        im_observed, im_scale = resize(im_observed, target_size, max_size)
        im_rendered, im_scale_rendered = resize(im_rendered, target_size, max_size)
        assert im_scale == im_scale_rendered, "scale mismatch"

        # add random background to data_syn observed image
        if "data_syn" in pair_rec.keys() and phase == "train":
            if pair_rec["data_syn"] is True or (
                pair_rec["data_syn"] is False and np.random.rand() < config.TRAIN.REPLACE_OBSERVED_BG_RATIO
            ):
                VOC_root = os.path.join(config.dataset.root_path, "VOCdevkit/VOC2012")
                VOC_image_set_dir = os.path.join(VOC_root, "ImageSets/Main")
                VOC_bg_list_path = os.path.join(VOC_image_set_dir, "diningtable_trainval.txt")
                with open(VOC_bg_list_path, "r") as f:
                    VOC_bg_list = [
                        line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
                    ]
                height, width, channel = im_observed.shape
                target_size = min(height, width)
                max_size = max(height, width)
                observed_hw_ratio = float(height) / float(width)

                k = random.randint(0, len(VOC_bg_list) - 1)
                bg_idx = VOC_bg_list[k]
                bg_path = os.path.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx))
                bg_image = cv2.imread(bg_path, cv2.IMREAD_COLOR)
                bg_h, bg_w, bg_c = bg_image.shape
                bg_image_resize = np.zeros((height, width, channel), dtype="uint8")
                if (float(height) / float(width) < 1 and float(bg_h) / float(bg_w) < 1) or (
                    float(height) / float(width) >= 1 and float(bg_h) / float(bg_w) >= 1
                ):
                    if bg_h >= bg_w:
                        bg_h_new = int(np.ceil(bg_w * observed_hw_ratio))
                        if bg_h_new < bg_h:
                            bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                        else:
                            bg_image_crop = bg_image
                    else:
                        bg_w_new = int(np.ceil(bg_h / observed_hw_ratio))
                        if bg_w_new < bg_w:
                            bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                        else:
                            bg_image_crop = bg_image
                else:
                    if bg_h >= bg_w:
                        bg_h_new = int(np.ceil(bg_w * observed_hw_ratio))
                        bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                    else:  # bg_h < bg_w
                        bg_w_new = int(np.ceil(bg_h / observed_hw_ratio))
                        print(bg_w_new)
                        bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]

                bg_image_resize_0, _ = resize(bg_image_crop, target_size, max_size)
                h, w, c = bg_image_resize_0.shape
                bg_image_resize[0:h, 0:w, :] = bg_image_resize_0

                # add background to image_observed
                res_image = bg_image_resize.copy()
                if phase == "train":
                    fg_label = cv2.imread(pair_rec["mask_gt_observed"], cv2.IMREAD_UNCHANGED)

                fg_label = np.dstack([fg_label, fg_label, fg_label])
                res_image[fg_label != 0] = im_observed[fg_label != 0]

                im_observed = res_image

        im_observed_tensor = transform(im_observed, config.network.PIXEL_MEANS)
        im_rendered_tensor = transform(im_rendered, config.network.PIXEL_MEANS)

        processed_ims_observed.append(im_observed_tensor)
        processed_ims_rendered.append(im_rendered_tensor)

    return processed_ims_observed, processed_ims_rendered, scale_ind_list


def get_gt_observed_depth(pairdb, config, scale_ind_list, phase="train", random_k=18):
    num_pairs = len(pairdb)
    processed_depth_gt_observed = []
    for i in range(num_pairs):
        pair_rec = pairdb[i]

        scale_ind = scale_ind_list[i]
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        depth_gt_observed_path = pair_rec["depth_gt_observed"]
        assert os.path.exists(depth_gt_observed_path), "{} does not exist".format(pair_rec["depth_gt_observed"])
        depth_gt_observed = cv2.imread(depth_gt_observed_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        depth_gt_observed, _ = resize(depth_gt_observed, target_size, max_size)
        depth_gt_observed = depth_gt_observed / config.dataset.DEPTH_FACTOR

        depth_gt_observed_tensor = depth_gt_observed[np.newaxis, np.newaxis, :, :]

        processed_depth_gt_observed.append(depth_gt_observed_tensor)

    return processed_depth_gt_observed


def get_pair_depth(pairdb, config, scale_ind_list, phase="train", random_k=[]):
    num_pairs = len(pairdb)
    processed_depth_observed = []
    processed_depth_rendered = []
    for i in range(num_pairs):
        pair_rec = pairdb[i]

        scale_ind = scale_ind_list[i]
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        depth_observed_path = pair_rec["depth_observed"]
        assert os.path.exists(depth_observed_path), "{} does not exist".format(pair_rec["depth_observed"])
        depth_observed = cv2.imread(depth_observed_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if config.network.MASK_INPUTS:
            if config.TRAIN.MASK_SYN and phase == "train" and random_k[i] < config.TRAIN.MASK_SYN_RATIO:
                mask_observed = cv2.imread(pair_rec["mask_syn"], cv2.IMREAD_UNCHANGED)
            elif config.dataset.MASK_GT or (phase == "train" and not config.dataset.MASK_GT):
                mask_observed = cv2.imread(pair_rec["mask_gt_observed"], cv2.IMREAD_UNCHANGED)
            else:
                mask_observed = cv2.imread(pair_rec["mask_observed_est"], cv2.IMREAD_UNCHANGED)
            depth_observed *= mask_observed == pair_rec["mask_idx"]

        assert os.path.exists(pair_rec["depth_rendered"]), "{} does not exist".format(pair_rec["depth_rendered"])
        depth_rendered = cv2.imread(pair_rec["depth_rendered"], cv2.IMREAD_UNCHANGED).astype(np.float32)

        depth_observed, _ = resize(depth_observed, target_size, max_size)
        depth_rendered, _ = resize(depth_rendered, target_size, max_size)
        depth_observed = depth_observed / config.dataset.DEPTH_FACTOR
        depth_rendered = depth_rendered / config.dataset.DEPTH_FACTOR

        depth_observed_tensor = depth_observed[np.newaxis, np.newaxis, :, :]
        depth_rendered_tensor = depth_rendered[np.newaxis, np.newaxis, :, :]

        processed_depth_observed.append(depth_observed_tensor)
        processed_depth_rendered.append(depth_rendered_tensor)

    return processed_depth_observed, processed_depth_rendered


def get_pair_mask(pairdb, config, scale_ind_list, phase="train", random_k=[]):
    """
    get mask_observed, mask_gt_observed, mask_rendered
    :param pairdb:
    :param config:
    :param scale_ind_list:
    :param phase:
    :param random_k:
    :return:
    """
    num_pairs = len(pairdb)
    mask_observed_list = []
    mask_gt_observed_list = []
    mask_rendered_list = []
    for i in range(num_pairs):
        pair_rec = pairdb[i]
        scale_ind = scale_ind_list[i]
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        # prepare mask_observed and mask_gt_observed
        if phase == "train":
            # mask_gt_observed
            mask_gt_observed_path = pair_rec["mask_gt_observed"]
            assert os.path.exists(mask_gt_observed_path), "{} does not exist".format(pair_rec["mask_gt_observed"])
            mask_gt_observed = cv2.imread(mask_gt_observed_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            cur_mask_gt_observed = np.zeros(mask_gt_observed.shape)
            fg = mask_gt_observed == pair_rec["mask_idx"]
            cur_mask_gt_observed[fg] = 1.0
            cur_mask_gt_observed, _ = resize(cur_mask_gt_observed, target_size, max_size)
            cur_mask_gt_observed[cur_mask_gt_observed < 0.5] = 0.0  # binarize the resized result
            assert fg.any(), "NOT_VALID: {}, {}".format(mask_gt_observed_path, np.unique(fg))

            # mask_observed
            if config.TRAIN.INIT_MASK == "mask_gt":
                mask_observed = mask_gt_observed.copy()
            elif config.TRAIN.INIT_MASK == "box_gt":
                # mask_observed: use mask_gt_observed's bbox area
                mask_observed = np.zeros(mask_gt_observed.shape)
                x_start, y_start, x_end, y_end = get_min_rect(cur_mask_gt_observed)
                mask_observed[y_start:y_end, x_start:x_end] = 1.0  # rectangle
            elif config.TRAIN.INIT_MASK == "box_rendered":
                # mask_observed: use mask_rendered's bbox area
                assert os.path.exists(pair_rec["depth_rendered"]), "{} does not exist".format(
                    pair_rec["depth_rendered"]
                )
                depth_rendered = cv2.imread(pair_rec["depth_rendered"], cv2.IMREAD_UNCHANGED).astype(np.float32)
                depth_rendered, _ = resize(depth_rendered, target_size, max_size)
                depth_rendered = depth_rendered / config.dataset.DEPTH_FACTOR
                cur_mask_rendered = np.zeros(depth_rendered.shape)
                fg = depth_rendered > 0.2
                cur_mask_rendered[fg] = 1.0
                cur_mask_observed = np.zeros(cur_mask_rendered.shape)
                assert fg.any(), "NO POINT VALID IN INIT MASK: {}".format(pair_rec["depth_rendered"])
                x_start, y_start, x_end, y_end = get_min_rect(cur_mask_rendered)
                cur_mask_observed[y_start:y_end, x_start:x_end] = 1.0  # rectangle
            else:
                raise Exception("Unknown mask type: {}".format(config.TRAIN.INIT_MASK))

            if config.TRAIN.MASK_DILATE:
                mask_observed = mask_dilate(mask_observed)

            mask_observed = mask_observed[np.newaxis, np.newaxis, :, :]
            cur_mask_gt_observed = cur_mask_gt_observed[np.newaxis, np.newaxis, :, :]
            mask_observed_list.append(mask_observed)
            mask_gt_observed_list.append(cur_mask_gt_observed)

        else:  # test phase
            # in Yu_rendered, some objects are not detected
            assert os.path.exists(pair_rec["depth_rendered"]), "{} does not exist".format(pair_rec["depth_rendered"])
            depth_rendered = cv2.imread(pair_rec["depth_rendered"], cv2.IMREAD_UNCHANGED).astype(np.float32)
            if np.sum(depth_rendered) == 0:
                cur_mask_observed = np.zeros(depth_rendered.shape)
                print("NO POINT VALID IN INIT MASK")
            else:
                if config.TEST.INIT_MASK == "mask_gt_observed":
                    mask_observed_path = pair_rec["mask_gt_observed"]
                    assert os.path.exists(mask_observed_path), "{} does not exist".format(mask_observed_path)
                    mask_observed = cv2.imread(mask_observed_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    cur_mask_observed = np.zeros(mask_observed.shape)
                    cur_mask_observed[mask_observed == pair_rec["mask_idx"]] = 1.0
                    cur_mask_observed, _ = resize(cur_mask_observed, target_size, max_size)
                    cur_mask_observed[cur_mask_observed < 0.5] = 0.0  # binarize the resized result
                elif config.TEST.INIT_MASK == "mask_observed":
                    mask_observed_path = pair_rec["mask_observed"]
                    assert os.path.exists(mask_observed_path), "{} does not exist".format(mask_observed_path)
                    mask_observed = cv2.imread(mask_observed_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    cur_mask_observed = np.zeros(mask_observed.shape)
                    cur_mask_observed[mask_observed == pair_rec["mask_idx"]] = 1.0
                    cur_mask_observed, _ = resize(cur_mask_observed, target_size, max_size)
                    cur_mask_observed[cur_mask_observed < 0.5] = 0.0  # binarize the resized result
                elif config.TEST.INIT_MASK == "box_gt_observed":
                    # print("use box as mask")
                    mask_gt_observed = cv2.imread(pair_rec["mask_gt_observed"], cv2.IMREAD_UNCHANGED).astype(np.float32)
                    cur_mask_gt_observed = np.zeros(mask_gt_observed.shape)
                    cur_mask_gt_observed[mask_gt_observed == pair_rec["mask_idx"]] = 1.0
                    assert np.nonzero(cur_mask_gt_observed), pairdb
                    x_max = np.max(cur_mask_gt_observed, 0)
                    y_max = np.max(cur_mask_gt_observed, 1)
                    nz_x = np.nonzero(x_max)[0]
                    nz_y = np.nonzero(y_max)[0]
                    x_start = np.min(nz_x)
                    x_end = np.max(nz_x)
                    y_start = np.min(nz_y)
                    y_end = np.max(nz_y)
                    cur_mask_observed = np.zeros(cur_mask_gt_observed.shape)
                    cur_mask_observed[y_start:y_end, x_start:x_end] = 1.0  # rectangle
                elif config.TEST.INIT_MASK == "box_":
                    # print("use box as mask")
                    mask_observed = cv2.imread(pair_rec["mask_observed"], cv2.IMREAD_UNCHANGED).astype(np.float32)
                    cur_mask_rendered = np.zeros(mask_observed.shape)
                    cur_mask_rendered[mask_observed == pair_rec["mask_idx"]] = 1.0
                    cur_mask_observed = np.zeros(cur_mask_rendered.shape)
                    if np.count_nonzero(cur_mask_rendered) != 0:
                        x_max = np.max(cur_mask_rendered, 0)
                        y_max = np.max(cur_mask_rendered, 1)
                        nz_x = np.nonzero(x_max)[0]
                        nz_y = np.nonzero(y_max)[0]
                        x_start = np.min(nz_x)
                        x_end = np.max(nz_x)
                        y_start = np.min(nz_y)
                        y_end = np.max(nz_y)
                        cur_mask_observed[y_start:y_end, x_start:x_end] = 1.0  # rectangle
                    else:
                        print("NO POINT VALID IN INIT MASK")
                elif config.TEST.INIT_MASK == "box_rendered":
                    assert os.path.exists(pair_rec["depth_rendered"]), "{} does not exist".format(
                        pair_rec["depth_rendered"]
                    )
                    depth_rendered = cv2.imread(pair_rec["depth_rendered"], cv2.IMREAD_UNCHANGED).astype(np.float32)
                    depth_rendered, _ = resize(depth_rendered, target_size, max_size)
                    depth_rendered = depth_rendered / config.dataset.DEPTH_FACTOR
                    cur_mask_rendered = np.zeros(depth_rendered.shape)
                    cur_mask_rendered[depth_rendered > 0.2] = 1.0
                    cur_mask_observed = np.zeros(cur_mask_rendered.shape)
                    if np.count_nonzero(cur_mask_rendered) != 0:
                        x_max = np.max(cur_mask_rendered, 0)
                        y_max = np.max(cur_mask_rendered, 1)
                        nz_x = np.nonzero(x_max)[0]
                        nz_y = np.nonzero(y_max)[0]
                        x_start = np.min(nz_x)
                        x_end = np.max(nz_x)
                        y_start = np.min(nz_y)
                        y_end = np.max(nz_y)
                        cur_mask_observed[y_start:y_end, x_start:x_end] = 1.0  # rectangle
                    else:
                        print("NO POINT VALID IN INIT MASK")
                else:
                    raise Exception("Unknown init mask type: {}".format(config.TEST.INIT_MASK))

            if config.TEST.MASK_DILATE:
                cur_mask_observed = mask_dilate(cur_mask_observed, max_thickness=10)

            cur_mask_observed = cur_mask_observed[np.newaxis, np.newaxis, :, :]

            mask_observed_list.append(cur_mask_observed)
            # during test, there is NO mask_gt_observed, so assign it with mask_observed
            mask_gt_observed_list.append(cur_mask_observed)

        # prepare mask_rendered
        assert os.path.exists(pair_rec["depth_rendered"]), "{} does not exist".format(pair_rec["depth_rendered"])
        depth_rendered = cv2.imread(pair_rec["depth_rendered"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_rendered, _ = resize(depth_rendered, target_size, max_size)
        depth_rendered = depth_rendered / config.dataset.DEPTH_FACTOR
        depth_rendered[depth_rendered > 0.2] = 1
        mask_rendered = depth_rendered
        mask_rendered = mask_rendered[np.newaxis, np.newaxis, :, :]
        mask_rendered_list.append(mask_rendered)

    return mask_observed_list, mask_gt_observed_list, mask_rendered_list


def get_pair_flow(pairdb, config, scale_ind_list, phase="train", random_k=[]):
    from lib.pair_matching.flow import calc_flow

    num_pairs = len(pairdb)
    flow_tensor = []
    flow_weights_tensor = []
    X_rendered_valid_list = []
    X_observed_valid_list = []

    for i in range(num_pairs):
        pair_rec = pairdb[i]
        flow_depth_rendered = cv2.imread(pair_rec["depth_rendered"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        flow_depth_rendered /= config.dataset.DEPTH_FACTOR

        if "depth_gt_observed" in pair_rec:
            flow_depth_observed_raw = cv2.imread(pair_rec["depth_gt_observed"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        else:
            flow_depth_observed_raw = cv2.imread(pair_rec["depth_observed"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        flow_depth_observed_raw /= config.dataset.DEPTH_FACTOR

        flow_depth_observed = flow_depth_observed_raw

        if config.network.PRED_FLOW or (config.train_iter.SE3_PM_LOSS):
            flow, visible, X_rendered_valid = calc_flow(
                flow_depth_rendered,
                pair_rec["pose_rendered"],
                pair_rec["pose_observed"],
                config.dataset.INTRINSIC_MATRIX,
                flow_depth_observed,
                standard_rep=config.network.STANDARD_FLOW_REP,
            )
            # print('flow *'*20, flow.shape, np.unique(flow))
            # print('flow weights *' * 20, visible.shape, np.unique(visible))
            flow_tensor.append(flow.transpose((2, 0, 1))[np.newaxis, :, :, :])
            if config.TRAIN.FLOW_WEIGHT_TYPE == "all":
                flow_weights = np.ones(visible.shape, dtype=np.float32)
            elif config.TRAIN.FLOW_WEIGHT_TYPE == "viz":
                flow_weights = visible
            elif config.TRAIN.FLOW_WEIGHT_TYPE == "valid":
                flow_weights = np.logical_or(np.squeeze(flow_depth_rendered == 0), visible)
            flow_weights_tensor.append(np.tile(flow_weights[np.newaxis, np.newaxis, :, :], (1, 2, 1, 1)))
            # flow_weights_tensor.append(flow_weights[np.newaxis, np.newaxis, :, :])
            X_rendered_valid_list.append(X_rendered_valid)

    return (flow_tensor, flow_weights_tensor, X_rendered_valid_list, X_observed_valid_list)


point_cloud_dict = {}


def get_point_cloud_model(config, pairdb):
    if pairdb[0]["gt_class"] not in point_cloud_dict:
        if not config.dataset.dataset.startswith("ModelNet"):
            point_path = os.path.join(config.dataset.model_dir, pairdb[0]["gt_class"], "points.xyz")
            point_cloud_dict[pairdb[0]["gt_class"]] = load_object_points(point_path)
        else:
            obj_path = os.path.join(config.dataset.model_dir, pairdb[0]["gt_class"] + ".obj")
            point_cloud_dict[pairdb[0]["gt_class"]] = load_points_from_obj(obj_path)

    points_obj = point_cloud_dict[pairdb[0]["gt_class"]]
    num_points = points_obj.shape[0]
    num_sample = config.train_iter.NUM_3D_SAMPLE
    num_keep = min(num_points, num_sample)

    keep_idx = np.arange(num_points)
    np.random.shuffle(keep_idx)
    keep_idx = keep_idx[:num_keep]

    points_sample = np.zeros((num_sample, 3))
    points_sample[:num_keep, :] = points_obj[keep_idx, :]

    channel_num = 3
    points_weights = np.zeros((num_sample, channel_num))
    points_weights[:num_keep, :] = 1
    points_sample = np.expand_dims(points_sample.T, axis=0)
    points_weights = np.expand_dims(points_weights.T, axis=0)
    return [points_sample], [points_weights]


def get_point_cloud_observed(config, points_model, pose_observed):
    R = pose_observed[:, :3]
    T = pose_observed[:, 3]
    points_observed = np.dot(R, points_model) + T.reshape((3, 1))
    return points_observed


def get_point_cloud(pairdb, config, scale_ind_list, X_list=None, phase="train"):
    assert config.TRAIN.MASK_SYN is False, "NOT IMPLEMENTED"
    from lib.utils.projection import backproject_camera, se3_inverse, se3_mul

    num_pairs = len(pairdb)
    X_obj_tensor = []
    X_obj_weights_tensor = []
    for batch_idx in range(num_pairs):
        pair_rec = pairdb[batch_idx]
        if "depth_gt_observed" in pair_rec:
            depth_observed_raw = cv2.imread(pair_rec["depth_gt_observed"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        else:
            depth_observed_raw = cv2.imread(pair_rec["depth_observed"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_observed_raw /= config.dataset.DEPTH_FACTOR

        # needs to be checked !!!
        if "mask_gt_observed" in pair_rec and config.network.MASK_INPUTS:
            mask_observed_path = pair_rec["mask_gt_observed"]
            assert os.path.exists(mask_observed_path), "{} does not exist".format(pair_rec["mask_observed"])
            mask_observed = cv2.imread(mask_observed_path, cv2.IMREAD_UNCHANGED)
            depth_observed = np.zeros(depth_observed_raw.shape)
            depth_observed[mask_observed == pair_rec["mask_idx"]] = depth_observed_raw[
                mask_observed == pair_rec["mask_idx"]
            ]
        else:
            depth_observed = depth_observed_raw

        if X_list:
            X = X_list[batch_idx]
        else:
            X = backproject_camera(depth_observed, intrinsic_matrix=config.dataset.INTRINSIC_MATRIX)
        transform_r2i = se3_mul(pair_rec["pose_rendered"], se3_inverse(pair_rec["pose_observed"]))
        X_obj = np.matmul(transform_r2i, np.append(X, np.ones([1, X.shape[1]], dtype=np.float32), axis=0)).reshape(
            (1, 3, depth_observed.shape[0], depth_observed.shape[1])
        )
        X_obj_weights = (depth_observed != 0).astype(np.float32)
        X_obj_weights = np.tile(X_obj_weights[np.newaxis, np.newaxis, :, :], (1, 3, 1, 1))
        # X_obj_weights = X_obj_weights[np.newaxis, np.newaxis, :, :]
        X_obj_tensor.append(X_obj)
        X_obj_weights_tensor.append(X_obj_weights)

    return X_obj_tensor, X_obj_weights_tensor


def get_point_cloud_fast(config, valid_pc_list, sample_number, phase="train"):
    # type: (object, object, object) -> object
    X_obj_tensor = []
    X_obj_weights_tensor = []
    for idx, raw_pc in enumerate(valid_pc_list):
        number_proposals = raw_pc.shape[1]
        num_kept = min(sample_number, number_proposals)
        keep_idx = np.arange(number_proposals)
        np.random.shuffle(keep_idx)
        keep_idx = keep_idx[:num_kept]
        X_obj = np.zeros((1, 3, sample_number))
        X_obj[0, :, :num_kept] = np.array([c[keep_idx] for c in raw_pc])
        X_obj_tensor.append(X_obj)
        channel_num = 3
        X_obj_weights = np.zeros((1, channel_num, sample_number))
        X_obj_weights[0, :, :num_kept] = 1
        X_obj_weights_tensor.append(X_obj_weights)
    return X_obj_tensor, X_obj_weights_tensor


def resize(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        return padded_im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor


def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice : (ind + 1) * islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice : (ind + 1) * islice, : tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice : (ind + 1) * islice, : tensor.shape[1], : tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[
                ind * islice : (ind + 1) * islice, : tensor.shape[1], : tensor.shape[2], : tensor.shape[3]
            ] = tensor
    else:
        raise Exception("Sorry, unimplemented.")
    return all_tensor


def my_tensor_vstack(tensor_list):
    all_tensor = np.concatenate(tensor_list, axis=0)
    return all_tensor
