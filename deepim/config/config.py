# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.MXNET_VERSION = ""

config.output_path = ""
config.symbol = ""
config.SCALES = [(480, 640)]  # first is scale (the shorter side); second is max size

# default training
config.default = edict()
config.default.frequent = 1000
config.default.kvstore = "device"

# network related params
config.network = edict()
config.network.FIXED_PARAMS = []
config.network.PIXEL_MEANS = np.array([0, 0, 0])
config.network.pretrained = "../model/pretrained_model/flownet"
config.network.pretrained_epoch = 0
config.network.init_from_flownet = False
config.network.skip_initialize = False
config.network.INPUT_DEPTH = False
config.network.INPUT_MASK = False
config.network.PRED_MASK = False
config.network.PRED_FLOW = False
config.network.STANDARD_FLOW_REP = False
config.network.TRAIN_ITER = False
config.network.TRAIN_ITER_SIZE = 1
config.network.REGRESSOR_NUM = 1  # 1 or num_classes
config.network.ROT_TYPE = "QUAT"  # 'QUAT', 'EULER'
config.network.ROT_COORD = "CAMERA"
config.network.TRANS_LOSS_TYPE = "L2"  # 'L1', 'smooth_L1'

# dataset related params
config.dataset = edict()
config.dataset.dataset = "LINEMOD_REFINE"
config.dataset.dataset_path = "./data/LINEMOD_6D/LINEMOD_converted/LINEMOD_refine"
config.dataset.image_set = "train_ape"
config.dataset.root_path = "./data"
config.dataset.test_image_set = "val_ape"
config.dataset.model_dir = ""
config.dataset.model_file = "./data/ModelNet/render_v1/models.txt"  # optional, if too many classes
config.dataset.pose_file = "./data/ModelNet/render_v1/poses.txt"  # optional, if too many classes

config.dataset.DEPTH_FACTOR = 1000
config.dataset.NORMALIZE_FLOW = 1.0
config.dataset.NORMALIZE_3D_POINT = 0.1
config.dataset.INTRINSIC_MATRIX = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
config.dataset.ZNEAR = 0.25
config.dataset.ZFAR = 6.0

config.dataset.class_name_file = ""
config.dataset.class_name = []
config.dataset.trans_means = np.array([0.0, 0.0, 0.0])
config.dataset.trans_stds = np.array([1.0, 1.0, 1.0])

config.TRAIN = edict()
config.TRAIN.optimizer = "sgd"
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.lr = 0.0001
config.TRAIN.lr_step = "4, 6"
config.TRAIN.momentum = 0.975
config.TRAIN.wd = 0.0005
config.TRAIN.model_prefix = "deepim"
config.TRAIN.RESUME = False
config.TRAIN.SHUFFLE = True
config.TRAIN.BATCH_PAIRS = 1
config.TRAIN.FLOW_WEIGHT_TYPE = "all"  # 'all', 'viz', 'valid'
# config.TRAIN.VISUALIZE = False
config.TRAIN.TENSORBOARD_LOG = False
config.TRAIN.INIT_MASK = "box_gt"  # mask_gt, box_gt
config.TRAIN.UPDATE_MASK = "box_gt"
config.TRAIN.MASK_DILATE = False
config.TRAIN.REPLACE_OBSERVED_BG_RATIO = 0.0  # replace train images' bg with VOC

config.TEST = edict()
config.TEST.BATCH_PAIRS = 1
config.TEST.test_epoch = 0
config.TEST.VISUALIZE = False
config.TEST.test_iter = 1
config.TEST.INIT_MASK = "box_rendered"
config.TEST.UPDATE_MASK = "box_rendered"
config.TEST.FAST_TEST = False
config.TEST.PRECOMPUTED_ICP = False  # evaluate with ICP refinement
config.TEST.BEFORE_ICP = False  # evaluate without ICP refinement

# for iterative train
# se3 distance loss
config.train_iter = edict()
config.train_iter.SE3_DIST_LOSS = False
config.train_iter.LW_ROT = 0.0
config.train_iter.LW_TRANS = 0.0
config.train_iter.TRANS_LOSS_TYPE = "L2"  # 'L1', 'smooth_L1'
config.train_iter.TRANS_SMOOTH_L1_SCALAR = 3.0
# se3 point matching loss
config.train_iter.SE3_PM_LOSS = False
config.train_iter.LW_PM = 0.0
config.train_iter.SE3_PM_LOSS_TYPE = "L1"
config.train_iter.SE3_PM_SL1_SCALAR = 1.0
config.train_iter.NUM_3D_SAMPLE = -1
# flow loss
config.train_iter.LW_FLOW = 0.0
# segmentation loss
config.train_iter.LW_MASK = 0.0


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == "TRAIN":
                        if "BBOX_WEIGHTS" in v:
                            v["BBOX_WEIGHTS"] = np.array(v["BBOX_WEIGHTS"])
                    elif k == "network":
                        if "PIXEL_MEANS" in v:
                            v["PIXEL_MEANS"] = np.array(v["PIXEL_MEANS"])
                    elif k == "dataset":
                        # make elegant later
                        if "INTRINSIC_MATRIX" in v:
                            v["INTRINSIC_MATRIX"] = np.array(v["INTRINSIC_MATRIX"]).reshape([3, 3]).astype(np.float32)
                        if "trans_means" in v:
                            v["trans_means"] = np.array(v["trans_means"]).flatten().astype(np.float32)
                        if "trans_stds" in v:
                            v["trans_stds"] = np.array(v["trans_stds"]).flatten().astype(np.float32)
                        if "class_name_file" in v:
                            if v["class_name_file"] != "":
                                with open(v["class_name_file"]) as f:
                                    v["class_name"] = [line.strip() for line in f.readlines()]
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == "SCALES":
                        config[k][0] = tuple(v)
                    else:
                        config[k] = v
            else:
                raise ValueError("key: {} does not exist in config.py".format(k))
