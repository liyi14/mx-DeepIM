# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from lib.utils.projection import backproject_camera
from lib.pair_matching import RT_transform
import cv2


def flow2se3(depth_object, flow, mask_image, K):
    """
    give flow from object to image, calculate the pose

    :param depth_object: height x width, ndarray the depth map of object image.
    :param flow: height x width x (w, h) flow from object image to real image
    :param mask_image: height x width, the mask of real image
    :param K: 3x3 intrinsic matrix
    :return: se3: 3x4 matrix.
    """
    height = depth_object.shape[0]
    width = depth_object.shape[1]
    assert mask_image.shape == (height, width)
    valid_in_object = (depth_object != 0).flatten()
    all_op = backproject_camera(depth_object, intrinsic_matrix=K)
    # all_op = all_op.reshape((3, width, height))

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x += flow[:, :, 0]
    y += flow[:, :, 1]
    x = x.flatten()
    y = y.flatten()
    all_ip = np.vstack((x, y))

    valid_in_image = (mask_image != 0).flatten()

    valid = np.where(np.logical_and(valid_in_object, valid_in_image))[0]
    objectPoints = all_op[:, valid].astype(np.float64).transpose()
    imagePoints = all_ip[:, valid].astype(np.float64).transpose()
    convex, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints,
                                                     K, np.zeros(4))

    se3_q = np.zeros(7)
    if convex:
        R, _ = cv2.Rodrigues(rvec)
        se3_q[:4] = RT_transform.mat2quat(R)
        se3_q[4:] = tvec.flatten()
        return convex, se3_q
    else:
        se3_q[0] = 1
        return convex, se3_q
