# --------------------------------------------------------
# DA-RNN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np


# RT is a 3x4 matrix
def se3_inverse(RT):
    """
    return the inverse of a RT
    :param RT=[R,T], 4x3 np array
    :return: RT_new=[R,T], 4x3 np array
    """
    R = RT[0:3, 0:3]
    T = RT[0:3, 3].reshape((3, 1))
    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = R.transpose()
    RT_new[0:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def se3_mul(RT1, RT2):
    """
    concat 2 RT transform
    :param RT1=[R,T], 4x3 np array
    :param RT2=[R,T], 4x3 np array
    :return: RT_new = RT1 * RT2
    """
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3, 1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3, 1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new


# backproject pixels into 3D points in camera's coordinate system
def backproject_camera(depth, intrinsic_matrix, FLIP_X=False):
    # get intrinsic matrix
    K = intrinsic_matrix
    K = np.matrix(K)
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = Kinv * x2d.transpose()

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    return np.array(X)
