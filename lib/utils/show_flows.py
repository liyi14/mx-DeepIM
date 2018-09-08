# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np


MAXCOLS = 60
ncols = 0
colorwheel = None


def setcols(r, g, b, k):
    colorwheel[k][0] = r
    colorwheel[k][1] = g
    colorwheel[k][2] = b


def makecolorwheel():
    # relative lengths of color transitions:
    # these are chosen based on/home/davidm/DA-RNN/DA-RNN/lib/utils/test_triplet_flow_loss.py perceptual similarity
    # (e.g. one can distinguish more shades between red and yellow
    #  than between yellow and green)
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    global ncols, colorwheel
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3], dtype=np.float32)
    print("ncols = %d\n" % ncols)
    if (ncols > MAXCOLS):
        raise EnvironmentError("something went wrong?")
    k = 0
    for i in range(RY):
        setcols(1,    1.0*float(i)/RY,     0,           k)
        k += 1
    for i in range(YG):
        setcols(1.0-float(i)/YG, 1,        0,           k)
        k += 1
    for i in range(GC):
        setcols(0,       1,      float(i)/GC,     k)
        k += 1
    for i in range(CB):
        setcols(0,       1-float(i)/CB, 1,          k)
        k += 1
    for i in range(BM):
        setcols(float(i)/BM,      0,      1,           k)
        k += 1
    for i in range(MR):
        setcols(1,    0,      1-float(i)/MR, k)
        k += 1
makecolorwheel()




def sintel_compute_color(data_interlaced):
    # type: (np.ndarray) -> np.ndarray
    data_u_in, data_v_in = np.split(data_interlaced, 2, axis=2)
    data_u_in = np.squeeze(data_u_in)
    data_v_in = np.squeeze(data_v_in)
    # pre-normalize
    max_rad = np.max(np.sqrt(np.power(data_u_in, 2) + np.power(data_v_in, 2))) + 1E-10
    fx = data_u_in / max_rad
    fy = data_v_in / max_rad

    # now do the stuff done in computeColor()
    rad = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
    a = np.nan_to_num(np.arctan2(-fy, -fx) / np.pi)
    fk = (a + 1.0) / 2.0 * (ncols-1)
    k0 = fk.astype(np.int32)
    k1 = ((k0 + 1) % ncols).astype(np.int32)
    f = fk - k0
    h, w = k0.shape
    col0 = colorwheel[k0.reshape(-1)].reshape([h, w, 3])
    col1 = colorwheel[k1.reshape(-1)].reshape([h, w, 3])
    col = (1 - f[..., np.newaxis]) * col0 + f[..., np.newaxis] * col1
    # col = col0

    col = 1 - rad[..., np.newaxis] * (1 - col)  # increase saturation with radius
    return col
