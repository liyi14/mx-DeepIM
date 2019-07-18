# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import sys
import os

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../external/mxnet/mxnet_v00_origin"))
import mxnet as mx
import numpy as np


class ZoomTransOperator(mx.operator.CustomOp):
    def __init__(self, b_inv_zoom, b_zoom_grad):
        super(ZoomTransOperator, self).__init__()
        self.b_inv_zoom = b_inv_zoom
        self.b_zoom_grad = b_zoom_grad

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        batch_size = in_data[0].shape[0]
        zoom_factor_array = in_data[0].asnumpy()
        trans_delta_array = in_data[1].asnumpy()
        zoom_trans_delta_array = np.zeros(trans_delta_array.shape)

        for batch_idx in range(batch_size):
            wx = zoom_factor_array[batch_idx][0]
            wy = zoom_factor_array[batch_idx][0]
            assert wx == wy
            delta_x, delta_y, delta_z = trans_delta_array[batch_idx]
            if self.b_inv_zoom:
                # zoom backward
                zoom_delta_x = delta_x * wx
                zoom_delta_y = delta_y * wy
            else:
                # zoom in
                zoom_delta_x = delta_x / wx  # wx = crop / origin
                zoom_delta_y = delta_y / wy
            zoom_delta_z = delta_z
            zoom_trans_delta_array[batch_idx, 0] = zoom_delta_x
            zoom_trans_delta_array[batch_idx, 1] = zoom_delta_y
            zoom_trans_delta_array[batch_idx, 2] = zoom_delta_z
        self.assign(out_data[0], req[0], mx.ndarray.array(zoom_trans_delta_array, ctx=ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        ctx = in_data[0].context
        batch_size = in_data[0].shape[0]
        zoom_factor_array = in_data[0].asnumpy()
        zoom_trans_grad_array = out_grad[0].asnumpy()
        trans_grad_array = np.zeros(zoom_trans_grad_array.shape)
        for batch_idx in range(batch_size):
            wx = zoom_factor_array[batch_idx][0]
            wy = zoom_factor_array[batch_idx][0]
            assert wx == wy
            zoom_grad_x, zoom_grad_y, zoom_grad_z = zoom_trans_grad_array[batch_idx]
            if self.b_zoom_grad:
                if self.b_inv_zoom:
                    grad_x = zoom_grad_x * wx
                    grad_y = zoom_grad_y * wy
                else:
                    grad_x = zoom_grad_x / wx
                    grad_y = zoom_grad_y / wy
            else:
                grad_x = zoom_grad_x
                grad_y = zoom_grad_y
            grad_z = zoom_grad_z
            trans_grad_array[batch_idx, 0] = grad_x
            trans_grad_array[batch_idx, 1] = grad_y
            trans_grad_array[batch_idx, 2] = grad_z
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], mx.ndarray.array(trans_grad_array, ctx=ctx))


@mx.operator.register("ZoomTrans")
class ZoomTransProp(mx.operator.CustomOpProp):
    def __init__(self, b_inv_zoom="False", b_zoom_grad="False"):
        super(ZoomTransProp, self).__init__(True)
        self.b_inv_zoom = b_inv_zoom.lower() == "true"
        self.b_zoom_grad = b_zoom_grad.lower() == "true"

    def list_arguments(self):
        input_list = ["zoom_factor", "trans_delta"]
        return input_list

    def list_outputs(self):
        output_list = ["zoom_trans_delta"]
        return output_list

    def infer_shape(self, in_shape):
        out_shape = [in_shape[1]]
        return in_shape, out_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        input_type = [dtype, dtype]
        output_type = [dtype]
        return input_type, output_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return ZoomTransOperator(self.b_inv_zoom, self.b_zoom_grad)


if __name__ == "__main__":
    # configs
    thresh = 1e-3
    step = 1e-4
    ctx = mx.gpu(0)
    batch_size = 8

    # initialize layer
    zoom_factor = mx.sym.Variable("zoom_factor")
    trans_delta = mx.sym.Variable("trans_delta")
    se3_trans = mx.sym.Variable("se3_trans")
    proj2d = mx.sym.Custom(
        zoom_factor=zoom_factor, trans_delta=trans_delta, name="updater", op_type="ZoomTrans", b_inv_zoom=False
    )
    v_zoom_factor = np.random.rand(batch_size, 4) * 2
    v_zoom_factor[:, 1] = v_zoom_factor[:, 0]
    v_trans_delta = np.random.rand(batch_size, 3) * 2

    exe1 = proj2d.simple_bind(ctx=ctx, zoom_factor=v_zoom_factor.shape, trans_delta=v_trans_delta.shape)

    # forward
    exe1.arg_dict["zoom_factor"][:] = mx.ndarray.array(v_zoom_factor, ctx=ctx)
    exe1.arg_dict["trans_delta"][:] = mx.ndarray.array(v_trans_delta, ctx=ctx)
    import time
    import matplotlib.pyplot as plt  # noqa:F401

    t = time.time()
    exe1.forward(is_train=True)
    zoom_trans_delta_mx = exe1.outputs[0].asnumpy()
    zoom_trans_delta_py = np.copy(v_trans_delta)
    for batch_idx in range(batch_size):
        zoom_trans_delta_py[batch_idx, :2] /= v_zoom_factor[batch_idx, 0]
        print("py: ", zoom_trans_delta_py[batch_idx])
        print("mx: ", zoom_trans_delta_mx[batch_idx])
    print(zoom_trans_delta_py - zoom_trans_delta_mx)

    proj2d = mx.sym.Custom(
        zoom_factor=zoom_factor, trans_delta=trans_delta, name="updater", op_type="ZoomTrans", b_inv_zoom=True
    )
    exe2 = proj2d.simple_bind(ctx=ctx, zoom_factor=v_zoom_factor.shape, trans_delta=v_trans_delta.shape)
    exe2.arg_dict["zoom_factor"][:] = mx.ndarray.array(v_zoom_factor, ctx=ctx)
    exe2.arg_dict["trans_delta"][:] = mx.ndarray.array(zoom_trans_delta_mx, ctx=ctx)
    exe2.forward(is_train=True)
    trans_delta_mx = exe2.outputs[0].asnumpy()
    for batch_idx in range(batch_size):
        print("py: ", v_trans_delta[batch_idx])
        print("mx: ", trans_delta_mx[batch_idx])

    print("trans_delta_mx-v_trans_delta:\n", trans_delta_mx - v_trans_delta)
