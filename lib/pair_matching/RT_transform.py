# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from lib.utils.projection import se3_inverse, se3_mul
import math

from scipy.linalg import logm
import numpy.linalg as LA
from math import pi


def calc_RT_delta(pose_src,
                  pose_tgt,
                  T_means,
                  T_stds,
                  rot_coord='MODEL',
                  rot_type='MATRIX'):
    """
    project the points in source corrd to target corrd
    :param pose_src: pose matrix of soucre, [R|T], 3x4
    :param pose_tgt: pose matrix of target, [R|T], 3x4
    :param rot_coord: model/camera
    :param rot_type: quat/euler/matrix
    :return: Rm_delta
    :return: T_delta
    """
    if rot_coord.lower() == 'naive':
        se3_src2tgt = se3_mul(pose_tgt, se3_inverse(pose_src))
        Rm_delta = se3_src2tgt[:, :3]
        T_delta = se3_src2tgt[:, 3].reshape((3))
    else:
        Rm_delta = R_inv_transform(pose_src[:3, :3], pose_tgt[:3, :3],
                                   rot_coord)
        T_delta = T_inv_transform(pose_src[:, 3], pose_tgt[:, 3], T_means,
                                  T_stds, rot_coord)

    if rot_type.lower() == 'quat':
        r = mat2quat(Rm_delta)
    elif rot_type.lower() == 'euler':
        r = mat2euler(Rm_delta)
    elif rot_type.lower() == 'matrix':
        r = Rm_delta
    else:
        raise Exception("Unknown rot_type: {}".format(rot_type))
    t = np.squeeze(T_delta)

    return r, t


def R_transform(R_src, R_delta, rot_coord='MODEL'):
    """
    transform R_src use R_delta
    :param R_src: matrix
    :param R_delta:
    :param rot_coord:
    :return:
    """
    if rot_coord.lower() == 'model':
        R_output = np.dot(R_src, R_delta)
    elif rot_coord.lower() == 'camera' or rot_coord.lower(
    ) == 'naive' or rot_coord.lower() == 'camera_new':
        R_output = np.dot(R_delta, R_src)
    else:
        raise Exception(
            "Unknown rot_coord in R_transform: {}".format(rot_coord))
    return R_output


def R_inv_transform(R_src, R_tgt, rot_coord):
    if rot_coord.lower() == 'model':
        R_delta = np.dot(R_src.transpose(), R_tgt)
    elif rot_coord.lower() == 'camera' or rot_coord.lower() == 'camera_new':
        R_delta = np.dot(R_tgt, R_src.transpose())
    else:
        raise Exception(
            "Unknown rot_coord in R_inv_transform: {}".format(rot_coord))
    return R_delta


def T_transform(T_src, T_delta, T_means, T_stds, rot_coord):
    '''
    :param T_src: (x1, y1, z1)
    :param T_delta: (dx, dy, dz), normed
    :return: T_tgt: (x2, y2, z2)
    '''
    # print("T_src: {}".format(T_src))
    assert T_src[2] != 0, "T_src: {}".format(T_src)
    T_delta_1 = T_delta * T_stds + T_means
    T_tgt = np.zeros((3, ))
    z2 = T_src[2] / np.exp(T_delta_1[2])
    T_tgt[2] = z2
    if rot_coord.lower() == 'camera' or rot_coord.lower() == 'model':
        T_tgt[0] = z2 * (T_delta_1[0] + T_src[0] / T_src[2])
        T_tgt[1] = z2 * (T_delta_1[1] + T_src[1] / T_src[2])
    elif rot_coord.lower() == 'camera_new':
        T_tgt[0] = T_src[2] * T_delta_1[0] + T_src[0]
        T_tgt[1] = T_src[2] * T_delta_1[1] + T_src[1]
    else:
        raise Exception("Unknown: {}".format(rot_coord))

    return T_tgt


def T_transform_naive(R_delta, T_src, T_delta):
    T_src = T_src.reshape((3, 1))
    T_delta = T_delta.reshape((3, 1))
    T_new = np.dot(R_delta, T_src) + T_delta
    return T_new.reshape((3, ))


def T_inv_transform(T_src, T_tgt, T_means, T_stds, rot_coord):
    '''
    :param T_src:
    :param T_tgt:
    :param T_means:
    :param T_stds:
    :return: T_delta: delta in pixel
    '''
    T_delta = np.zeros((3, ))
    if rot_coord.lower() == 'camera_new':
        T_delta[0] = (T_tgt[0] - T_src[0]) / T_src[2]
        T_delta[1] = (T_tgt[1] - T_src[1]) / T_src[2]
    elif rot_coord.lower() == 'camera' or rot_coord.lower() == 'model':
        T_delta[0] = T_tgt[0] / T_tgt[2] - T_src[0] / T_src[2]
        T_delta[1] = T_tgt[1] / T_tgt[2] - T_src[1] / T_src[2]
    else:
        raise Exception("Unknown: {}".format(rot_coord))
    T_delta[2] = np.log(T_src[2] / T_tgt[2])
    T_delta_normed = (T_delta - T_means) / T_stds
    return T_delta_normed


def RT_transform(pose_src, r, t, T_means, T_stds, rot_coord='MODEL'):
    # r: 4(quat) or 3(euler) number
    # t: 3 number, (delta_x, delta_y, scale)
    r = np.squeeze(r)
    if r.shape[0] == 3:
        Rm_delta = euler2mat(r[0], r[1], r[2])
    elif r.shape[0] == 4:
        # QUAT
        quat = r / LA.norm(r)
        Rm_delta = quat2mat(quat)
    else:
        raise Exception("Unknown r shape: {}".format(r.shape))
    t_delta = np.squeeze(t)

    if rot_coord.lower() == 'naive':
        se3_mx = np.zeros((3, 4))
        se3_mx[:, :3] = Rm_delta
        se3_mx[:, 3] = t
        pose_est = se3_mul(se3_mx, pose_src)
    else:
        pose_est = np.zeros((3, 4))
        pose_est[:3, :3] = R_transform(pose_src[:3, :3], Rm_delta, rot_coord)
        pose_est[:3, 3] = T_transform(pose_src[:, 3], t_delta, T_means, T_stds,
                                      rot_coord)

    return pose_est


def calc_rt_dist_q(Rq_src, Rq_tgt, T_src, T_tgt):

    rd_rad = np.arccos(np.inner(Rq_src, Rq_tgt)**2 * 2 - 1)
    rd_deg = rd_rad / pi * 180
    td = LA.norm(T_tgt - T_src)
    return rd_deg, td


def calc_rt_dist_m(pose_src, pose_tgt):
    R_src = pose_src[:, :3]
    T_src = pose_src[:, 3]
    R_tgt = pose_tgt[:, :3]
    T_tgt = pose_tgt[:, 3]
    temp = logm(np.dot(np.transpose(R_src), R_tgt))
    rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = rd_rad / pi * 180

    td = LA.norm(T_tgt - T_src)

    return rd_deg, td


def calc_se3(pose_src, pose_tgt):
    """
    project the points in source corrd to target corrd
    :param pose_src: pose matrix of soucre, [R|T], 3x4
    :param pose_tgt: pose matrix of target, [R|T], 3x4
    :return: visible: whether points in source can be viewed in target
    """
    se3_src2tgt = se3_mul(pose_tgt, se3_inverse(pose_src))
    rotm = se3_src2tgt[:, :3]
    t = se3_src2tgt[:, 3].reshape((3))

    return rotm, t


def se3_q2m(se3_q):
    assert se3_q.size == 7
    se3_mx = np.zeros((3, 4))
    quat = se3_q[0:4] / LA.norm(se3_q[0:4])
    R = quat2mat(quat)
    se3_mx[:, :3] = R
    se3_mx[:, 3] = se3_q[4:]
    return se3_mx


# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0),
    'sxyx': (0, 0, 1, 0),
    'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0),
    'syzx': (1, 0, 0, 0),
    'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0),
    'syxy': (1, 1, 1, 0),
    'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0),
    'szyx': (2, 1, 0, 0),
    'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1),
    'rxyx': (0, 0, 1, 1),
    'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1),
    'rxzy': (1, 0, 0, 1),
    'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1),
    'ryxy': (1, 1, 1, 1),
    'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1),
    'rxyz': (2, 1, 0, 1),
    'rzyz': (2, 1, 1, 1)
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0

_MAX_FLOAT = np.maximum_sctype(np.float)
_FLOAT_EPS = np.finfo(np.float).eps


def euler2mat(ai, aj, ak, axes='sxyz'):
    """Return rotation matrix from Euler angles and axis sequence.
    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def mat2euler(mat, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    Note that many Euler angle triplets can describe one matrix.
    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS4:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS4:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quat_inverse(q):
    q = np.squeeze(q)
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z] / Nq)


def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.
    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                     [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                     [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def mat2quat(M):
    ''' Calculate quaternion corresponding to given rotation matrix

    Parameters
    ----------
    M : array-like
      3x3 rotation matrix

    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]

    Notes
    -----
    Method claimed to be robust to numerical errors in M

    Constructs quaternion by calculating maximum eigenvector for matrix
    K (constructed from input `M`).  Although this is not tested, a
    maximum eigenvalue of 1 corresponds to a valid rotation.

    A quaternion q*-1 corresponds to the same rotation as q; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).

    References
    ----------
    * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090

    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True

    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
    quaternion from a rotation matrix", AIAA Journal of Guidance,
    Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
    0731-5090

    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([[Qxx - Qyy - Qzz, 0, 0, 0], [
        Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0
    ], [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                  [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q


def euler2quat(ai, aj, ak, axes='sxyz'):
    """Return `quaternion` from Euler angles and axis sequence `axes`
    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format
    Examples
    --------
    >>> q = euler2quat(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0

    if q[0] < 0:
        q *= -1
    return q


def quat2euler(quaternion, axes='sxyz'):
    """Euler angles from `quaternion` for specified axis sequence `axes`
    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    Examples
    --------
    >>> angles = quat2euler([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True
    """
    return mat2euler(quat2mat(quaternion), axes)


def qmult(q1, q2):
    ''' Multiply two quaternions
    Parameters
    ----------
    q1 : 4 element sequence
    q2 : 4 element sequence
    Returns
    -------
    q12 : shape (4,) array
    Notes
    -----
    See : http://en.wikipedia.org/wiki/Quaternions#Hamilton_product
    '''
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q *= -1
    return q
