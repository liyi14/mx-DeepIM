# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import os.path as osp
import logging
from lib.utils import logger
from lib.utils.fs import mkdir_p


def create_logger(root_output_path, cfg, image_set, temp_flie=False):
    # set up logger
    mkdir_p(root_output_path)
    assert osp.exists(root_output_path), "{} does not exist".format(root_output_path)

    cfg_name = osp.basename(cfg).split(".")[0]
    config_output_path = osp.join(root_output_path, "{}".format(cfg_name))
    mkdir_p(config_output_path)

    image_sets = [iset for iset in image_set.split("+")]
    final_output_path = osp.join(config_output_path, "{}".format("_".join(image_sets)))
    mkdir_p(final_output_path)

    if temp_flie:
        log_prefix = "temp_{}".format(cfg_name)
    else:
        log_prefix = "{}".format(cfg_name)
    logger.set_logger_dir(final_output_path, action='k', prefix=log_prefix)
    logger.setLevel(logging.INFO)

    return final_output_path
