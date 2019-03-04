# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import sys
import os

sys.path.insert(0, os.path.join("../../external/mxnet/mxnet_v00_origin"))
sys.path.insert(0, "../../")
"""
Simple zoom for flow
"""

import cv2
import mxnet as mx
import numpy as np


class ZoomFlowOperator(mx.operator.CustomOp):
    def __init__(self, height, width, b_inv_zoom):
        super(ZoomFlowOperator, self).__init__()
        self.height = height
        self.width = width
        self.b_inv_zoom = b_inv_zoom

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        batch_size = in_data[0].shape[0]

        grid_array = mx.ndarray.zeros([batch_size, 2, self.height, self.width], ctx=ctx)
        zoom_factor_array = in_data[0].asnumpy()
        for batch_idx in range(batch_size):
            if self.b_inv_zoom:
                wx_in, wy_in, tx_in, ty_in = zoom_factor_array[batch_idx]
                wx = 1 / wx_in
                wy = 1 / wy_in
                crop_width = wx_in * self.width
                crop_height = wy_in * self.height
                obj_real_c_x = tx_in * 0.5 * self.width + 0.5 * self.width
                obj_real_c_y = ty_in * 0.5 * self.height + 0.5 * self.height
                tx = (self.width * 0.5 - obj_real_c_x) / crop_width * 2
                ty = (self.height * 0.5 - obj_real_c_y) / crop_height * 2
            else:
                wx, wy, tx, ty = zoom_factor_array[batch_idx]
            affine_matrix = mx.ndarray.array(
                [[wx, 0, tx], [0, wy, ty]], ctx=ctx
            ).reshape((1, 6))
            a = mx.ndarray.GridGenerator(
                data=affine_matrix,
                transform_type="affine",
                target_shape=(self.height, self.width),
            )
            grid_array[batch_idx] = a[0]

        flow_array = in_data[1]
        zoom_flow_array = mx.ndarray.BilinearSampler(flow_array, grid_array)
        for batch_idx in range(batch_size):
            wx = zoom_factor_array[batch_idx][0]
            wy = zoom_factor_array[batch_idx][1]
            assert wx == wy, "wx and wy should be equal"
            if self.b_inv_zoom:
                zoom_flow_array[batch_idx] *= wx
            else:
                zoom_flow_array[batch_idx] /= wx
        self.assign(out_data[0], req[0], zoom_flow_array)

        if not self.b_inv_zoom:
            flow_weights_array = in_data[2]
            zoom_flow_weights_array = mx.ndarray.BilinearSampler(
                flow_weights_array, grid_array
            )
            bin_zoom_flow_weights_array = mx.ndarray.round(
                zoom_flow_weights_array - 0.45
            )  # binarize zoomed flow_weights
            self.assign(out_data[1], req[1], bin_zoom_flow_weights_array)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("ZoomFlow")
class ZoomFlowProp(mx.operator.CustomOpProp):
    def __init__(self, width=640, height=480, b_inv_zoom="False"):
        super(ZoomFlowProp, self).__init__(True)
        self.height = int(height)
        self.width = int(width)
        self.b_inv_zoom = b_inv_zoom.lower() == "true"

    def list_arguments(self):
        if self.b_inv_zoom:
            input_list = ["zoom_factor", "flow"]
        else:
            input_list = ["zoom_factor", "flow", "flow_weights"]
        return input_list

    def list_outputs(self):
        if self.b_inv_zoom:
            output_list = ["zoom_flow"]
        else:
            output_list = ["zoom_flow", "zoom_flow_weights"]
        return output_list

    def infer_shape(self, in_shape):
        out_shape = in_shape[1:]
        return in_shape, out_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        in_type = [dtype, dtype] if self.b_inv_zoom else [dtype, dtype, dtype]
        out_type = [dtype] if self.b_inv_zoom else [dtype, dtype]
        return in_type, out_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return ZoomFlowOperator(self.height, self.width, self.b_inv_zoom)


if __name__ == "__main__":
    from lib.pair_matching.flow import calc_flow

    # configs
    thresh = 1e-3
    step = 1e-4
    ctx = mx.gpu(0)
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    height = 480
    width = 640
    b_zoom_se3 = False
    batch_size = 8
    img_idx = ["035_{:06}".format(x) for x in [11, 21, 31, 41, 51, 61, 71, 81]]
    sub_idx1 = ["0" for x in range(batch_size)]
    sub_idx2 = ["5" for x in range(batch_size)]
    pixel_means = 120

    # initialize layer
    zoom_factor = mx.sym.Variable("zoom_factor")
    flow_sym = mx.sym.Variable("flow")
    flow_weights_sym = mx.sym.Variable("flow_weights")

    v_image_real = np.zeros((batch_size, 3, height, width))
    v_image_rendered = np.zeros((batch_size, 3, height, width))
    v_depth_real = np.zeros((batch_size, 1, height, width))
    v_depth_rendered = np.zeros((batch_size, 1, height, width))
    v_zoom_factor = np.zeros((batch_size, 4), dtype=np.float32)
    v_flow = np.zeros((batch_size, 2, height, width), dtype=np.float32)
    v_flow_weights = np.zeros((batch_size, 2, height, width), dtype=np.float32)
    for idx in range(batch_size):
        v_image_rendered[idx] = cv2.imread(
            "../../data/render_v5/data/rendered/0006/{}_{}-color.png".format(
                img_idx[idx], sub_idx1[idx]
            ),
            cv2.IMREAD_COLOR,
        ).transpose([2, 0, 1])
        v_image_real[idx] = cv2.imread(
            "../../data/render_v5/data/rendered/0006/{}_{}-color.png".format(
                img_idx[idx], sub_idx2[idx]
            ),
            cv2.IMREAD_COLOR,
        ).transpose([2, 0, 1])
        v_depth_real[idx, 0] = (
            cv2.imread(
                "../../data/render_v5/data/rendered/0006/{}_{}-depth.png".format(
                    img_idx[idx], sub_idx1[idx]
                ),
                cv2.IMREAD_UNCHANGED,
            )
            / 10000.0
        )
        v_depth_rendered[idx, 0] = (
            cv2.imread(
                "../../data/render_v5/data/rendered/0006/{}_{}-depth.png".format(
                    img_idx[idx], sub_idx2[idx]
                ),
                cv2.IMREAD_UNCHANGED,
            )
            / 10000.0
        )
        pose_real_path = "../../data/render_v5/data/rendered/0006/{}_{}-pose.txt".format(
            img_idx[idx], sub_idx1[idx]
        )
        pose_est_path = "../../data/render_v5/data/rendered/0006/{}_{}-pose.txt".format(
            img_idx[idx], sub_idx2[idx]
        )
        pose_real = np.loadtxt(pose_real_path, skiprows=1)
        pose_est = np.loadtxt(pose_est_path, skiprows=1)

        flow, visible, _ = calc_flow(
            v_depth_rendered[idx, 0],
            pose_est,
            pose_real,
            K,
            v_depth_real[idx, 0],
            thresh=3e-3,
            standard_rep=False,
        )
        v_zoom_factor[:, :2] = 1.5
        v_zoom_factor[:, 2:] = 0.2
        v_flow[idx] = flow.transpose((2, 0, 1))
        v_flow_weights[idx] = np.tile(visible, (2, 1, 1))

    zoom_flow_mx = mx.sym.Custom(
        name="updater_in",
        op_type="ZoomFlow",
        height=height,
        width=width,
        flow=flow_sym,
        flow_weights=flow_weights_sym,
        zoom_factor=zoom_factor,
        b_inv_zoom=False,
    )
    exe0 = zoom_flow_mx.simple_bind(
        ctx=ctx,
        zoom_factor=v_zoom_factor.shape,
        flow=v_flow.shape,
        flow_weights=v_flow_weights.shape,
    )
    # forward
    exe0.arg_dict["zoom_factor"][:] = mx.ndarray.array(v_zoom_factor, ctx=ctx)
    exe0.arg_dict["flow"][:] = mx.ndarray.array(v_flow, ctx=ctx)
    exe0.arg_dict["flow_weights"][:] = mx.ndarray.array(v_flow_weights, ctx=ctx)
    exe0.forward(is_train=True)  # forward

    zoom_flow_mx, zoomf_flow_weights = mx.sym.Custom(
        name="updater_in",
        op_type="ZoomFlow",
        height=height,
        width=width,
        flow=flow_sym,
        flow_weights=flow_weights_sym,
        zoom_factor=zoom_factor,
        b_inv_zoom=False,
    )
    simple_zoom_flow = mx.sym.Custom(
        name="updater_out",
        op_type="ZoomFlow",
        height=height,
        width=width,
        flow=zoom_flow_mx,
        zoom_factor=zoom_factor,
        b_inv_zoom=True,
    )
    exe1 = simple_zoom_flow.simple_bind(
        ctx=ctx,
        zoom_factor=v_zoom_factor.shape,
        flow=v_flow.shape,
        flow_weights=v_flow_weights.shape,
    )

    # forward
    exe1.arg_dict["zoom_factor"][:] = mx.ndarray.array(v_zoom_factor, ctx=ctx)
    exe1.arg_dict["flow"][:] = mx.ndarray.array(v_flow, ctx=ctx)
    exe1.arg_dict["flow_weights"][:] = mx.ndarray.array(v_flow_weights, ctx=ctx)
    import time
    import matplotlib.pyplot as plt

    t = time.time()
    exe1.forward(is_train=True)  # forward

    zoom_flow = exe0.outputs[0].asnumpy()
    zoom_flow_weights = exe0.outputs[1].asnumpy()
    inv_zoom_flow = exe1.outputs[0].asnumpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x * 20))

    STANDARD_FLOW_REP = False

    for batch_idx in range(batch_size):
        im_real = v_image_real[batch_idx].transpose([1, 2, 0]).astype(np.uint8)
        im_rendered = v_image_rendered[batch_idx].transpose([1, 2, 0]).astype(np.uint8)

        flow = v_flow[batch_idx].transpose([1, 2, 0])
        flow_weights = v_flow_weights[batch_idx]  # (2,h,w)
        flow_inv = inv_zoom_flow[batch_idx].transpose([1, 2, 0])
        z_flow = zoom_flow[batch_idx].transpose([1, 2, 0])
        z_flow_weights = zoom_flow_weights[batch_idx]
        visible = flow_weights[0, :, :]
        z_visible = z_flow_weights[0, :, :]

        fig = plt.figure()
        tmp = fig.add_subplot(3, 4, 1)
        tmp.set_title("image_real")
        plt.axis("off")
        plt.imshow(im_real)

        tmp = fig.add_subplot(3, 4, 2)
        tmp.set_title("image_imagine")
        plt.axis("off")
        plt.imshow(im_rendered)

        mesh_src = np.zeros((height, width, 3), np.uint8)
        mesh_tgt = np.zeros((height, width, 3), np.uint8)
        mesh_back_src = np.zeros((height, width, 3), np.uint8)
        mesh_back_tgt = np.zeros((height, width, 3), np.uint8)

        for h in range(height):
            for w in range(width):
                if visible[h, w]:
                    cur_flow = flow[h, w, :].flatten()
                    if not STANDARD_FLOW_REP:
                        cur_flow = cur_flow[[1, 0]]
                    point_color = [
                        sigmoid(float(h) / height - 0.5) * 200 + 50,
                        sigmoid(0.5 - float(w) / width) * 200 + 50,
                        sigmoid(float(h) / height + float(w) / width - 1) * 200 + 50,
                    ]

                    mesh_src = cv2.circle(
                        mesh_src,
                        (np.round(w).astype(int), np.round(h).astype(int)),
                        1,
                        point_color,
                    )
                    mesh_tgt = cv2.circle(
                        mesh_tgt,
                        (
                            np.round(w + cur_flow[0]).astype(int),
                            np.round(h + cur_flow[1]).astype(int),
                        ),
                        1,
                        point_color,
                    )

                cur_flow_inv = flow_inv[h, w, :].flatten()
                if not STANDARD_FLOW_REP:
                    cur_flow_inv = cur_flow_inv[[1, 0]]
                if cur_flow_inv[0] != 0 or cur_flow_inv[1] != 0:
                    point_color = [
                        sigmoid(float(h) / height - 0.5) * 200 + 50,
                        sigmoid(0.5 - float(w) / width) * 200 + 50,
                        sigmoid(float(h) / height + float(w) / width - 1) * 200 + 50,
                    ]
                    mesh_back_src = cv2.circle(
                        mesh_back_src,
                        (np.round(w).astype(int), np.round(h).astype(int)),
                        1,
                        point_color,
                    )
                    mesh_back_tgt = cv2.circle(
                        mesh_back_tgt,
                        (
                            np.round(w + cur_flow_inv[0]).astype(int),
                            np.round(h + cur_flow_inv[1]).astype(int),
                        ),
                        1,
                        point_color,
                    )
        # zoom_flow
        z_mesh_tgt = np.zeros((height, width, 3), np.uint8)
        z_mesh_src = np.zeros((height, width, 3), np.uint8)
        for h in range(height):
            for w in range(width):
                if z_visible[h, w]:
                    cur_flow = z_flow[h, w, :].flatten()
                    if not STANDARD_FLOW_REP:
                        cur_flow = cur_flow[[1, 0]]
                    point_color = [
                        sigmoid(float(h) / height - 0.5) * 200 + 50,
                        sigmoid(0.5 - float(w) / width) * 200 + 50,
                        sigmoid(float(h) / height + float(w) / width - 1) * 200 + 50,
                    ]

                    z_mesh_src = cv2.circle(
                        z_mesh_src,
                        (np.round(w).astype(int), np.round(h).astype(int)),
                        1,
                        point_color,
                    )
                    z_mesh_tgt = cv2.circle(
                        z_mesh_tgt,
                        (
                            np.round(w + cur_flow[0]).astype(int),
                            np.round(h + cur_flow[1]).astype(int),
                        ),
                        1,
                        point_color,
                    )

        # fig = plt.figure()
        tmp = fig.add_subplot(3, 4, 5)
        tmp.set_title("mesh_src")
        plt.axis("off")
        plt.imshow(mesh_src)

        tmp = fig.add_subplot(3, 4, 6)
        tmp.set_title("mesh_tgt")
        plt.axis("off")
        plt.imshow(mesh_tgt)

        tmp = fig.add_subplot(3, 4, 7)
        tmp.set_title("z_mesh_src")
        plt.axis("off")
        plt.imshow(z_mesh_src)

        tmp = fig.add_subplot(3, 4, 8)
        tmp.set_title("z_mesh_tgt")
        plt.axis("off")
        plt.imshow(z_mesh_tgt)

        tmp = fig.add_subplot(3, 4, 9)
        tmp.set_title("mesh_tgt_inv")
        plt.axis("off")
        plt.imshow(mesh_back_src)

        tmp = fig.add_subplot(3, 4, 10)
        tmp.set_title("mesh_tgt_inv")
        plt.axis("off")
        plt.imshow(mesh_back_tgt)

        tmp = fig.add_subplot(3, 4, 11)
        tmp.set_title("z_visible")
        plt.axis("off")
        plt.imshow(z_visible)

        plt.show()
