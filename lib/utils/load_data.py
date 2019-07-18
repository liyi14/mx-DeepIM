# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from lib.dataset import *  # noqa: F401, F403


def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None, flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def load_proposal_roidb(
    dataset_name, image_set_name, root_path, dataset_path, result_path=None, proposal="rpn", append_gt=True, flip=False
):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)

    gt_roidb = imdb.gt_roidb()
    roidb = eval("imdb." + proposal + "_roidb")(gt_roidb, append_gt)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def load_gt_sdsdb(
    dataset_name, image_set_name, root_path, dataset_path, result_path=None, flip=False, mask_size=21, binary_thresh=0.4
):
    """ load ground truth sdsdb """
    imdb = eval(dataset_name)(
        image_set_name, root_path, dataset_path, result_path, mask_size=mask_size, binary_thresh=binary_thresh
    )
    sdsdb = imdb.gt_sdsdb()
    if flip:
        sdsdb = imdb.append_flipped_images(sdsdb)
    return sdsdb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb, config):
    """ remove roidb entries without usable rois """

    def is_valid(entry):
        """ valid images have at least 1 fg or bg roi """
        overlaps = entry["max_overlaps"]
        fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
        bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print("filtered %d roidb entries: %d -> %d" % (num - num_after, num, num_after))

    return filtered_roidb


def load_gt_segdb(dataset_name, image_set_name, root_path, dataset_path, result_path=None, flip=False):
    """ load ground truth segdb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    segdb = imdb.gt_segdb()
    if flip:
        segdb = imdb.append_flipped_images_for_segmentation(segdb)
    return segdb


def merge_segdb(segdbs):
    """ segdb are list, concat them together """
    segdb = segdbs[0]
    for r in segdbs[1:]:
        segdb.extend(r)
    return segdb


def load_gt_pairdb(
    cfg,
    dataset_name,
    image_set_name,
    root_path,
    dataset_path,
    class_name,
    result_path=None,
    pair_flip=False,
    img_flip=False,
):
    print(image_set_name, root_path, dataset_path, result_path)
    imdb = eval(dataset_name)(
        cfg, image_set_name, root_path, dataset_path, class_name=class_name, result_path=result_path
    )
    pairdb = imdb.gt_pairdb()
    if pair_flip:
        pairdb = imdb.append_flipped_pairs(pairdb)
    assert not img_flip, "img_flip not supported"
    return pairdb


def merge_pairdb(pairdbs):
    """ pairdb are list, concat them together """
    pairdb = pairdbs[0]
    for r in pairdbs[1:]:
        pairdb.extend(r)
    return pairdb
