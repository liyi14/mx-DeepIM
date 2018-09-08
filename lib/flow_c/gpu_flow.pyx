# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_flow.hpp":
    void _flow(np.float32_t*, np.float32_t*, np.float32_t*, np.float32_t*,
               np.float32_t*, np.float32_t*,
               np.int32_t, np.int32_t, np.int32_t,
               np.int32_t);

# flow: n * 2 * h * w [h, w]
# depth_src: n * 1 * h * w
# depth_tgt: n * 1 * h * w
# KT: N * 3 * 4
# Kinv: 3 * 3

def gpu_flow(np.ndarray[np.float32_t, ndim=4] depth_src,
             np.ndarray[np.float32_t, ndim=4] depth_tgt,
             np.ndarray[np.float32_t, ndim=3] KT,
             np.ndarray[np.float32_t, ndim=2] Kinv,
             np.int32_t device_id = 0):
    cdef int batch_size = depth_src.shape[0]
    cdef int height = depth_src.shape[2]
    cdef int width = depth_src.shape[3]
    cdef np.ndarray[np.float32_t, ndim=4] \
        flow = np.zeros([batch_size, 2, height, width], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=4] \
        valid = np.zeros([batch_size, 1, height, width], dtype=np.float32)
    if batch_size>0:
        _flow(&flow[0, 0, 0, 0], &valid[0, 0, 0, 0],
              &depth_src[0, 0, 0, 0], &depth_tgt[0, 0, 0, 0],
              &KT[0, 0, 0], &Kinv[0, 0], batch_size,
              height, width, device_id)
    return flow, valid
