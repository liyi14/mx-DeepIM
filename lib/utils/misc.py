# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague
from __future__ import print_function, division
import math
import os
import numpy as np
from scipy.spatial import distance


def ensure_dir(path):
    """
    Ensures that the specified directory exists.

    :param path: Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def depth_im_to_dist_im(depth_im, K):
    """
    Converts depth image to distance image.

    :param depth_im: Input depth image, where depth_im[y, x] is the Z coordinate
    of the 3D point [X, Y, Z] that projects to pixel [x, y], or 0 if there is
    no such 3D point (this is a typical output of the Kinect-like sensors).
    :param K: Camera matrix.
    :return: Distance image dist_im, where dist_im[y, x] is the distance from
    the camera center to the 3D point [X, Y, Z] that projects to pixel [x, y],
    or 0 if there is no such 3D point.
    """
    xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
    ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T

    Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
    Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])

    dist_im = np.linalg.norm(np.dstack((Xs, Ys, depth_im)), axis=2)
    return dist_im


def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def calc_pts_diameter(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set).

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    diameter = -1
    for pt_id in range(pts.shape[0]):
        if pt_id % 1000 == 0:
            print(pt_id)
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


def calc_pts_diameter2(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set). Faster but requires more memory than
    calc_pts_diameter.

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    dists = distance.cdist(pts, pts, "euclidean")
    diameter = np.max(dists)
    return diameter
