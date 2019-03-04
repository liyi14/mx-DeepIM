# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path.
As to training, it is necessary that ground truth and proposals are mixed together for training.
"""
from __future__ import print_function, division, absolute_import
import os
from six.moves import cPickle
import numpy as np
from PIL import Image


def get_flipped_entry_outclass_wrapper(IMDB_instance, seg_rec):
    return IMDB_instance.get_flipped_entry(seg_rec)


class IMDB(object):
    def __init__(self, name, image_set, root_path, dataset_path, result_path=None):
        """
        basic information about an image database
        :param name: name of image database will be used for any output
        :param root_path: root path store cache and proposal data
        :param dataset_path: dataset path store images and image lists
        """
        self.name = name + "_" + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self._result_path = result_path

        # abstract attributes
        self.all_classes = []
        self.num_classes = 0
        self.image_set_index = []
        self.num_images = 0

        self.config = {}

    def image_path_from_index(self, index):
        raise NotImplementedError

    def gt_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, detections):
        raise NotImplementedError

    def evaluate_segmentations(self, segmentations):
        raise NotImplementedError

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.root_path, "cache")
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    @property
    def result_path(self):
        if self._result_path and os.path.exists(self._result_path):
            return self._result_path
        else:
            return self.cache_path

    def image_path_at(self, index):
        """
        access image at index in image database
        :param index: image index in image database
        :return: image path
        """
        return self.image_path_from_index(self.image_set_index[index])

    def load_rpn_data(self, full=False):
        if full:
            rpn_file = os.path.join(
                self.result_path, "rpn_data", self.name + "_full_rpn.pkl"
            )
        else:
            rpn_file = os.path.join(
                self.result_path, "rpn_data", self.name + "_rpn.pkl"
            )
        print("loading {}".format(rpn_file))
        assert os.path.exists(rpn_file), "rpn data not found at {}".format(rpn_file)
        with open(rpn_file, "rb") as f:
            box_list = cPickle.load(f)
        return box_list

    def load_rpn_roidb(self, gt_roidb):
        """
        turn rpn detection boxes into roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        box_list = self.load_rpn_data()
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def rpn_roidb(self, gt_roidb, append_gt=False):
        """
        get rpn roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of rpn
        """
        if append_gt:
            print("appending ground truth annotations")
            rpn_roidb = self.load_rpn_roidb(gt_roidb)
            roidb = IMDB.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self.load_rpn_roidb(gt_roidb)
        return roidb

    def get_flipped_entry(self, seg_rec):
        return {
            "image": self.flip_and_save(seg_rec["image"]),
            "seg_cls_path": self.flip_and_save(seg_rec["seg_cls_path"]),
            "height": seg_rec["height"],
            "width": seg_rec["width"],
            "flipped": True,
        }

    def append_flipped_images_for_segmentation(self, segdb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param segdb: [image_index]['seg_cls_path', 'flipped']
        :return: segdb: [image_index]['seg_cls_path', 'flipped']
        """
        print("append flipped images to segdb...")
        assert self.num_images == len(segdb)

        segdb_flip = []
        for i in range(self.num_images):
            seg_rec = segdb[i]
            segdb_flip.append(self.get_flipped_entry(seg_rec))
        segdb += segdb_flip
        self.image_set_index *= 2
        print("done")
        return segdb

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print("append flipped images to roidb")
        assert self.num_images == len(roidb)
        for i in range(self.num_images):
            roi_rec = roidb[i]
            boxes = roi_rec["boxes"].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = roi_rec["width"] - oldx2 - 1
            boxes[:, 2] = roi_rec["width"] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {
                "image": roi_rec["image"],
                "height": roi_rec["height"],
                "width": roi_rec["width"],
                "boxes": boxes,
                "gt_classes": roidb[i]["gt_classes"],
                "gt_overlaps": roidb[i]["gt_overlaps"],
                "max_classes": roidb[i]["max_classes"],
                "max_overlaps": roidb[i]["max_overlaps"],
                "flipped": True,
            }

            # if roidb has mask
            if "cache_seg_inst" in roi_rec:
                [filename, extension] = os.path.splitext(roi_rec["cache_seg_inst"])
                entry["cache_seg_inst"] = os.path.join(filename + "_flip" + extension)

            roidb.append(entry)

        self.image_set_index *= 2
        return roidb

    def flip_and_save(self, image_path):
        """
        flip the image by the path and save the flipped image with suffix 'flip'
        :param path: the path of specific image
        :return: the path of saved image
        """
        [image_name, image_ext] = os.path.splitext(os.path.basename(image_path))
        image_dir = os.path.dirname(image_path)
        saved_image_path = os.path.join(image_dir, image_name + "_flip" + image_ext)
        try:
            flipped_image = Image.open(saved_image_path)
        except:  # noqa: E722
            flipped_image = Image.open(image_path)
            flipped_image = flipped_image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image.save(saved_image_path, "png")
        return saved_image_path

    def get_flipped_pairs_entry(self, pair_rec):
        return {
            "image_real": pair_rec["image_rendered"],
            "image_rendered": pair_rec["image_real"],
            "depth_real": pair_rec["depth_rendered"],
            "depth_rendered": pair_rec["depth_real"],
            "pose_real": pair_rec["pose_est"],
            "pose_est": pair_rec["pose_real"],
            "height": pair_rec["height"],
            "width": pair_rec["width"],
            "pair_flipped": True,
            "img_flipped": False,
            "gt_class": pair_rec["gt_class"],
        }

    def append_flipped_pairs(self, pairdb):
        """
        append flipped images to an pairdb
        exchange real and rendered image in pair
        :param pairdb: [pair_index]['*_real', '*_rendered', 'pair_flipped', 'img_flipped']
        :return: pairdb: [pair_index]['*_real', '*_rendered', 'pair_flipped', 'img_flipped']
        """
        print("append flipped images to pairdb...")
        assert self.num_pairs == len(pairdb), "{} vs {}".format(
            self.num_pairs, len(pairdb)
        )
        pair_flip = []
        for i in range(self.num_pairs):
            pair_rec = pairdb[i]
            pair_flip.append(self.get_flipped_pairs_entry(pair_rec))
        pairdb += pair_flip
        self.image_set_index *= 2
        print("done")
        return pairdb

    @staticmethod
    def merge_roidbs(a, b):
        """
        merge roidbs into one
        :param a: roidb to be merged into
        :param b: roidb to be merged
        :return: merged imdb
        """
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]["boxes"] = np.vstack((a[i]["boxes"], b[i]["boxes"]))
            a[i]["gt_classes"] = np.hstack((a[i]["gt_classes"], b[i]["gt_classes"]))
            a[i]["gt_overlaps"] = np.vstack((a[i]["gt_overlaps"], b[i]["gt_overlaps"]))
            a[i]["max_classes"] = np.hstack((a[i]["max_classes"], b[i]["max_classes"]))
            a[i]["max_overlaps"] = np.hstack(
                (a[i]["max_overlaps"], b[i]["max_overlaps"])
            )
        return a
