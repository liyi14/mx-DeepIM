# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np


def mask_augment(mask_origin, augment_type):
    """

    :param pairdb:
    :param config:
    :param phase:
    :param random_k:
    :return:
    """
    height, width = mask_origin.shape
    x_max = np.max(mask_origin, axis=0)
    y_max = np.max(mask_origin, axis=1)
    nz_x = np.nonzero(x_max)[0]
    nz_y = np.nonzero(y_max)[0]
    start_x = np.min(nz_x)
    end_x = np.max(nz_x)
    obj_width = end_x - start_x + 1.0
    start_y = np.min(nz_y)
    end_y = np.max(nz_y)
    obj_height = end_y - start_y + 1.0
    mask_expand = np.copy(mask_origin)

    augment_type = augment_type % 18
    x_start_ind = 0
    x_end_ind = 0
    y_start_ind = 0
    y_end_ind = 0
    if augment_type < 9 and augment_type != 4:  # not mask center
        x_start_ind = np.floor(augment_type / 3)
        x_end_ind = np.floor(augment_type / 3) + 1
        y_start_ind = augment_type % 3
        y_end_ind = augment_type % 3 + 1

    if augment_type == 9:
        x_start_ind = 0
        x_end_ind = 1
        y_start_ind = 0
        y_end_ind = 3
    elif augment_type == 10:
        x_start_ind = 1.2
        x_end_ind = 1.8
        y_start_ind = 0
        y_end_ind = 3
    elif augment_type == 11:
        x_start_ind = 2
        x_end_ind = 3
        y_start_ind = 0
        y_end_ind = 3
    elif augment_type == 12:
        x_start_ind = 0
        x_end_ind = 3
        y_start_ind = 0
        y_end_ind = 1
    elif augment_type == 13:
        x_start_ind = 0
        x_end_ind = 3
        y_start_ind = 1.2
        y_end_ind = 1.8
    elif augment_type == 14:
        x_start_ind = 0
        x_end_ind = 3
        y_start_ind = 2
        y_end_ind = 3
    elif augment_type == 15:
        x_start_ind = 0
        x_end_ind = 3
        y_start_ind = 2.3
        y_end_ind = 3
    patch_start_x = np.round(start_x + obj_width * x_start_ind / 3.0)
    patch_end_x = np.round(start_x + obj_width * x_end_ind / 3.0)
    patch_start_y = np.round(start_y + obj_height * y_start_ind / 3.0)
    patch_end_y = np.round(start_y + obj_height * y_end_ind / 3.0)

    patch_start_x = min(max(patch_start_x, 0), width).astype("int")
    patch_end_x = min(max(patch_end_x, 0), width).astype("int")
    patch_start_y = min(max(patch_start_y, 0), height).astype("int")
    patch_end_y = min(max(patch_end_y, 0), height).astype("int")

    mask_expand[patch_start_y:patch_end_y, patch_start_x:patch_end_x] = 0

    patch_xy = [patch_start_x, patch_end_x, patch_start_y, patch_end_y]
    if np.sum(mask_expand) / (np.sum(mask_origin) + 0.0) < 0.4:
        mask_expand = mask_origin
        patch_xy = None

    return mask_expand, patch_xy


if __name__ == "__main__":
    origin = np.zeros([20, 20])
    origin[6:15, 5:15] = 1
    import matplotlib.pyplot as plt

    for i in range(18):
        plt.subplot(3, 6, i + 1)
        mask_res, patch_xy = mask_augment(origin, i)
        plt.imshow(mask_res)
        print(patch_xy)
    plt.show()
