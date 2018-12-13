# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import os
import numpy as np


def load_object_points(point_path):
    assert os.path.exists(point_path), 'Path does not exist: {}'.format(
        point_path)
    points = np.loadtxt(point_path)
    return points


def load_points_from_obj(obj_path):
    from glumpy import data
    assert os.path.exists(obj_path), 'Path does not exist: {}'.format(obj_path)
    vertices, indices = data.objload("{}".format(obj_path), rescale=True)
    vertices['position'] = vertices['position'] / 10.

    points = np.array(vertices['position'])
    return points
