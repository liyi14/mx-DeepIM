# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
import numpy as np


def get_min_rect(mask):
    """
    input mask, return minimum rectangle including all non-zero elements
    :param mask:
    :return: x_start, y_start, x_end, y_end
    """
    x_max = np.max(mask, 0)
    y_max = np.max(mask, 1)
    nz_x = np.nonzero(x_max)[0]
    nz_y = np.nonzero(y_max)[0]
    x_start = np.min(nz_x)
    x_end = np.max(nz_x)
    y_start = np.min(nz_y)
    y_end = np.max(nz_y)
    return x_start, y_start, x_end, y_end
