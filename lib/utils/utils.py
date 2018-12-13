# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import cv2


def read_img(path, n_channel=3):
    if n_channel == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    elif n_channel == 1:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        raise Exception("Unsupported n_channel: {}".format(n_channel))
    return img
