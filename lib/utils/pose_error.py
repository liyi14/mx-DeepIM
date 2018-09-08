# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Implementation of the pose error functions described in:
# Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW 2016
# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import os
import math
import numpy as np
from scipy import spatial

def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def transform_pts_Rt_2d(pts, R, t, K):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :param K: 3x3 intrinsic matrix 
    :return: nx2 ndarray with transformed 2D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1)) # 3xn
    pts_c_t = K.dot(pts_t)
    n = pts.shape[0]
    pts_2d = np.zeros((n, 2))
    pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
    pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

    return pts_2d


def arp_2d(R_est, t_est, R_gt, t_gt, pts, K):
    '''
    average re-projection error in 2d
    :param R_est: 
    :param t_est: 
    :param R_gt: 
    :param t_gt: 
    :param pts: 
    :param K: 
    :return: 
    '''
    pts_est_2d = transform_pts_Rt_2d(pts, R_est, t_est, K)
    pts_gt_2d = transform_pts_Rt_2d(pts, R_gt, t_gt, K)
    e = np.linalg.norm(pts_est_2d - pts_gt_2d, axis=1).mean()
    return e

def add(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

def adi(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e

def re(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose (3x1 vector).
    :param R_gt: Rotational element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi # [rad] -> [deg]
    return error

def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error


def load_object_points(point_path):
    print(point_path)
    assert os.path.exists(point_path), 'Path does not exist: {}'.format(point_path)
    points = np.loadtxt(point_path)
    return points

def load_object_extents(extent_path, num_classes):
    assert os.path.exists(extent_path), \
            'Path does not exist: {}'.format(extent_path)
    extents = np.zeros((num_classes, 3), dtype=np.float32)
    extents[1:, :] = np.loadtxt(extent_path) # assume class 0 is '__background__'
    return extents


if __name__=="__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lov_path = os.path.join(cur_dir, '../../data/LOV')
    point_file = os.path.join(lov_path, 'models', '003_cracker_box', 'points.xyz')
    points = load_object_points(point_file)
    print(points.min(0))
    print(points.max(0))
    print(points.max(0) - points.min(0))

    extent_file = os.path.join(lov_path, 'extents.txt')
    extents = load_object_extents(extent_file, num_classes=22) # 21 + 1

