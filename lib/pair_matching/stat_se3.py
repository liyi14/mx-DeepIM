# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from lib.pair_matching.RT_transform import calc_RT_delta, calc_rt_dist_m
from lib.utils.tictoc import tic, toc


def stat_se3(pairdb, config):
    """
    stat mean and std of se3 between real and rendered poses in a pairdb
    :param pairdb:
    :param config:
    :return:
    """
    num_pair = len(pairdb)
    tic()
    se3_i2r_tensor = (
        np.zeros([num_pair, 6])
        if config.network.SE3_TYPE == "EULER"
        else np.zeros([num_pair, 7])
    )
    se3_i2r_dist = np.zeros([num_pair, 2])
    for i in range(num_pair):
        rot_i2r, trans_i2r = calc_RT_delta(
            pairdb[i]["pose_rendered"],
            pairdb[i]["pose_real"],
            np.zeros(3),
            np.ones(3),
            config.network.ROT_COORD,
            config.network.SE3_TYPE,
        )
        se3_i2r_tensor[i, :-3] = rot_i2r
        se3_i2r_tensor[i, -3:] = trans_i2r

        R_dist, T_dist = calc_rt_dist_m(
            pairdb[i]["pose_rendered"], pairdb[i]["pose_real"]
        )
        se3_i2r_dist[i, 0] = R_dist
        se3_i2r_dist[i, 1] = T_dist

    print("stat finished, using {} seconds".format(toc()))
    se3_mean = np.mean(se3_i2r_tensor, axis=0)
    se3_std = np.std(se3_i2r_tensor, axis=0)
    print("mean: {}, \nstd: {}".format(se3_mean, se3_std))
    print(
        "R_max: {}, T_max: {}".format(
            np.max(se3_i2r_dist[:, 0]), np.max(se3_i2r_dist[:, 1])
        )
    )
    return se3_mean, se3_std
