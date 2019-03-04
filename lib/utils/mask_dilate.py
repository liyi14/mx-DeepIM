# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np


def mask_dilate(mask_origin, max_thickness=10):
    """

    :param pairdb:
    :param config:
    :param phase:
    :param random_k:
    :return:
    """
    direction = np.random.randint(10)

    mask_expand = np.copy(mask_origin)
    if direction not in [0, 1, 4]:
        thickness = np.random.randint(max_thickness) + 1
        mask_expand[thickness:, :] = (
            np.logical_and(
                mask_origin[:-thickness, :] != 0, mask_origin[thickness:, :] == 0
            )
            + mask_expand[thickness:, :]
        )
    if direction not in [1, 2, 5]:
        thickness = np.random.randint(max_thickness) + 1
        mask_expand[:-thickness, :] = (
            np.logical_and(
                mask_origin[thickness:, :] != 0, mask_origin[:-thickness, :] == 0
            )
            + mask_expand[:-thickness, :]
        )
    if direction not in [2, 3, 6]:
        thickness = np.random.randint(max_thickness) + 1
        mask_expand[:, thickness:] = (
            np.logical_and(
                mask_origin[:, :-thickness] != 0, mask_origin[:, thickness:] == 0
            )
            + mask_expand[:, thickness:]
        )
    if direction not in [0, 3, 7]:
        thickness = np.random.randint(max_thickness) + 1
        mask_expand[:, :-thickness] = (
            np.logical_and(
                mask_origin[:, thickness:] != 0, mask_origin[:, :-thickness] == 0
            )
            + mask_expand[:, :-thickness]
        )
    mask_expand[mask_expand > 1] = 1
    return mask_expand
