# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import sys
import os

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../external/mxnet/mxnet_0303"))
"""
zoom mask, using mask_real_est and mask_rendered to get the zoom area, zoom factor
output: zoomed mask_real_est, zoomed mask_real_gt, zoomed mask_rendered
"""

import cv2
import mxnet as mx
import numpy as np


class ZoomMaskOperator(mx.operator.CustomOp):
    def __init__(self, K, height, width):
        super(ZoomMaskOperator, self).__init__()
        self.K = K
        self.height = height
        self.width = width

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        batch_size = in_data[0].shape[0]
        mask_real_est_array = in_data[0]
        mask_real_gt_array = in_data[1]
        mask_rendered_array = in_data[2]
        src_pose_array = in_data[3].asnumpy()
        valid_pixels = mask_real_gt_array.asnumpy()
        valid_real_array = np.sum(valid_pixels, axis=1) > 0.3

        mask_rendered_np = mask_rendered_array.asnumpy()
        mask_rendered_np[mask_rendered_np > 0.2] = 1  # if the mask_rendered input is depth
        mask_rendered_np[mask_rendered_np <= 0.2] = 0  # if the mask_rendered input is depth
        mask_rendered_array = mx.nd.array(mask_rendered_np, ctx=ctx)
        valid_rendered_array = np.sum(mask_rendered_np, axis=1) > 0.3

        grid_array = mx.nd.zeros([batch_size, 2, self.height, self.width], ctx=ctx)
        zoom_factor_array = mx.nd.zeros([batch_size, 4], ctx=ctx)
        for batch_idx in range(batch_size):
            valid_real = valid_real_array[batch_idx]
            src_pose = src_pose_array[batch_idx]

            x_max = np.max(valid_real, axis=0)
            y_max = np.max(valid_real, axis=1)
            nz_x = np.nonzero(x_max)[0]
            nz_y = np.nonzero(y_max)[0]
            obj_real_start_x = np.min(nz_x)
            obj_real_end_x = np.max(nz_x)
            obj_real_start_y = np.min(nz_y)
            obj_real_end_y = np.max(nz_y)
            obj_real_c_x = (obj_real_start_x + obj_real_end_x) * 0.5
            obj_real_c_y = (obj_real_start_y + obj_real_end_y) * 0.5

            valid_rendered = valid_rendered_array[batch_idx]
            x_max = np.max(valid_rendered, axis=0)
            y_max = np.max(valid_rendered, axis=1)
            nz_x = np.nonzero(x_max)[0]
            nz_y = np.nonzero(y_max)[0]
            obj_rendered_c = np.dot(self.K, src_pose[:, 3])
            obj_rendered_c_x = obj_rendered_c[0] / obj_rendered_c[2]
            obj_rendered_c_y = obj_rendered_c[1] / obj_rendered_c[2]
            if len(nz_x) == 0 or len(nz_y) == 0:
                print("NO POINT VALID IN MASK rendered")
                obj_rendered_start_x = obj_real_start_x
                obj_rendered_end_x = obj_real_end_x
                obj_rendered_start_y = obj_real_start_y
                obj_rendered_end_y = obj_real_end_y
                zoom_c_x = obj_real_c_x
                zoom_c_y = obj_real_c_y
            else:
                obj_rendered_start_x = np.min(nz_x)
                obj_rendered_end_x = np.max(nz_x)
                obj_rendered_start_y = np.min(nz_y)
                obj_rendered_end_y = np.max(nz_y)
                zoom_c_x = obj_rendered_c_x
                zoom_c_y = obj_rendered_c_y

            left_dist = max(zoom_c_x - obj_rendered_start_x, zoom_c_x - obj_real_start_x)
            right_dist = max(obj_rendered_end_x - zoom_c_x, obj_real_end_x - zoom_c_x)
            up_dist = max(zoom_c_y - obj_rendered_start_y, zoom_c_y - obj_real_start_y)
            down_dist = max(obj_real_end_y - zoom_c_y, obj_rendered_end_y - zoom_c_y)
            crop_height = np.max([0.75 * right_dist, 0.75 * left_dist, up_dist, down_dist]) * 1.4 * 2

            wx = crop_height / self.height
            wy = crop_height / self.height
            tx = zoom_c_x / self.width * 2 - 1
            ty = zoom_c_y / self.height * 2 - 1
            affine_matrix = mx.nd.array([[wx, 0, tx], [0, wy, ty]], ctx=ctx).reshape((1, 6))
            a = mx.nd.GridGenerator(data=affine_matrix, transform_type="affine", target_shape=(self.height, self.width))
            grid_array[batch_idx] = a[0]

            zoom_factor_array[batch_idx, 0] = wx
            zoom_factor_array[batch_idx, 1] = wy
            zoom_factor_array[batch_idx, 2] = tx
            zoom_factor_array[batch_idx, 3] = ty

        zoom_mask_real_est_array = mx.nd.round(mx.nd.BilinearSampler(mask_real_est_array, grid_array))
        zoom_mask_real_gt_array = mx.nd.round(mx.nd.BilinearSampler(mask_real_gt_array, grid_array))
        zoom_mask_rendered_array = mx.nd.round(mx.nd.BilinearSampler(mask_rendered_array, grid_array))

        self.assign(out_data[0], req[0], zoom_mask_real_est_array)
        self.assign(out_data[1], req[1], zoom_mask_real_gt_array)
        self.assign(out_data[2], req[2], zoom_mask_rendered_array)
        self.assign(out_data[3], req[3], zoom_factor_array)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)


@mx.operator.register("ZoomMask")
class ZoomMaskProp(mx.operator.CustomOpProp):
    def __init__(self, K, width=640, height=480):
        super(ZoomMaskProp, self).__init__(True)
        self.K = np.fromstring(K[1:-1], dtype=np.float32, sep=" ").reshape([3, 3])
        self.height = int(height)
        self.width = int(width)

    def list_arguments(self):
        input_list = ["mask_observed", "mask_gt_observed", "mask_rendered", "src_pose"]
        return input_list

    def list_outputs(self):
        output_list = ["zoom_mask_observed", "zoom_mask_gt_observed", "zoom_mask_rendered", "zoom_factor"]
        return output_list

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        out_shape = in_shape[:-1]
        out_shape.append([batch_size, 4])
        return in_shape, out_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        input_type = [dtype, dtype, dtype, dtype]
        output_type = [dtype, dtype, dtype, dtype]
        return input_type, output_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return ZoomMaskOperator(self.K, self.height, self.width)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # configs
    thresh = 1e-3
    step = 1e-4
    ctx = mx.gpu(0)
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    height = 480
    width = 640
    b_zoom_se3 = False
    batch_size = 8

    img_idx = [x for x in [1, 11, 21, 31, 41, 51, 61, 71]]
    sub_idx1 = [x + 200 for x in img_idx]
    sub_idx2 = [x + 400 for x in img_idx]
    sub_idx3 = [x + 600 for x in img_idx]

    # initialize layer
    mask_real_est = mx.sym.Variable("mask_real_est")
    mask_real_gt = mx.sym.Variable("mask_real_gt")
    mask_rendered = mx.sym.Variable("mask_rendered")
    src_pose = mx.sym.Variable("src_pose")

    proj2d = mx.sym.Custom(
        mask_real_est=mask_real_est,
        mask_real_gt=mask_real_gt,
        mask_rendered=mask_rendered,
        src_pose=src_pose,
        K=K.flatten(),
        name="updater",
        op_type="ZoomMask",
        height=height,
        width=width,
    )
    v_mask_real_est = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_mask_real_gt = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_mask_rendered = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_src_pose = np.zeros((batch_size, 3, 4), dtype=np.float32)
    for idx in range(batch_size):
        # mask_real_est
        tmp = (
            cv2.imread(
                os.path.join(
                    cur_dir,
                    "../../data/render_v5/data/render_real/002_master_chef_can/0012/{:06}-depth.png".format(
                        sub_idx1[idx]
                    ),
                ),
                cv2.IMREAD_UNCHANGED,
            )
            / 10000.0
        )
        tmp[tmp != 0] = 1
        tmp = tmp[np.newaxis, :, :]
        v_mask_real_est[idx] = tmp

        # mask_real_gt
        tmp = (
            cv2.imread(
                os.path.join(
                    cur_dir,
                    "../../data/render_v5/data/render_real/002_master_chef_can/0012/{:06}-depth.png".format(
                        sub_idx1[idx]
                    ),
                ),
                cv2.IMREAD_UNCHANGED,
            )
            / 10000.0
        )
        tmp[tmp != 0] = 1
        tmp = tmp[np.newaxis, :, :]
        v_mask_real_gt[idx] = tmp

        # mask_rendered
        tmp = (
            cv2.imread(
                os.path.join(
                    cur_dir,
                    "../../data/render_v5/data/render_real/002_master_chef_can/0012/{:06}-depth.png".format(
                        sub_idx2[idx]
                    ),
                ),  # img_idx[idx])),
                cv2.IMREAD_UNCHANGED,
            )
            / 10000.0
        )
        tmp[tmp != 0] = 1
        tmp = tmp[np.newaxis, :, :]
        v_mask_rendered[idx] = tmp

        # src_pose of rendered
        v_src_pose[idx] = np.loadtxt(
            os.path.join(
                cur_dir,
                "../../data/render_v5/data/render_real/002_master_chef_can/0012/{:06}-pose.txt".format(sub_idx2[idx]),
            ),
            skiprows=1,
        )

    exe1 = proj2d.simple_bind(
        ctx=ctx,
        mask_real_est=v_mask_real_est.shape,
        mask_real_gt=v_mask_real_gt.shape,
        mask_rendered=v_mask_rendered.shape,
        src_pose=v_src_pose.shape,
    )

    # forward
    def simple_forward(exe1, v_mask_real_est, v_mask_real_gt, v_mask_rendered, v_src_pose, ctx, is_train=False):
        exe1.arg_dict["mask_real_est"][:] = mx.ndarray.array(v_mask_real_est, ctx=ctx, dtype="float32")
        exe1.arg_dict["mask_real_gt"][:] = mx.ndarray.array(v_mask_real_gt, ctx=ctx, dtype="float32")
        exe1.arg_dict["mask_rendered"][:] = mx.ndarray.array(v_mask_rendered, ctx=ctx, dtype="float32")
        exe1.arg_dict["src_pose"][:] = mx.ndarray.array(v_src_pose, ctx=ctx, dtype="float32")
        exe1.forward(is_train=is_train)

    t = time.time()

    simple_forward(exe1, v_mask_real_est, v_mask_real_gt, v_mask_rendered, v_src_pose, ctx, is_train=True)
    zoom_mask_real_est = exe1.outputs[0].asnumpy()
    zoom_mask_real_gt = exe1.outputs[1].asnumpy()
    zoom_mask_rendered = exe1.outputs[2].asnumpy()
    zoom_factor = exe1.outputs[3].asnumpy()
    print("using time: {:.2f}".format(time.time() - t))

    for batch_idx in range(batch_size):
        im_real_est = np.squeeze(v_mask_real_est[batch_idx])
        im_real_gt = np.squeeze(v_mask_real_gt[batch_idx])
        im_rendered = np.squeeze(v_mask_rendered[batch_idx])
        z_im_real_est = np.squeeze(zoom_mask_real_est[batch_idx])
        z_im_real_gt = np.squeeze(zoom_mask_real_gt[batch_idx])
        z_im_rendered = np.squeeze(zoom_mask_rendered[batch_idx])
        fig = plt.figure()
        tmp = fig.add_subplot(2, 3, 1)
        tmp.set_title("mask_real_est")
        plt.axis("off")
        plt.imshow(im_real_est)

        tmp = fig.add_subplot(2, 3, 2)
        tmp.set_title("mask_real_gt")
        plt.axis("off")
        plt.imshow(im_real_gt)

        tmp = fig.add_subplot(2, 3, 3)
        tmp.set_title("mask_rendered")
        plt.axis("off")
        plt.imshow(im_rendered)

        tmp = fig.add_subplot(2, 3, 4)
        tmp.set_title("mask_real_est after zoom")
        plt.axis("off")
        plt.imshow(z_im_real_est)

        tmp = fig.add_subplot(2, 3, 5)
        tmp.set_title("mask_real_gt after zoom")
        plt.axis("off")
        plt.imshow(z_im_real_gt)

        tmp = fig.add_subplot(2, 3, 6)
        tmp.set_title("mask_rendered after zoom")
        plt.axis("off")
        plt.imshow(z_im_rendered)
        plt.show()
