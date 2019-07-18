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

import cv2
import mxnet as mx
import numpy as np


class ZoomImageOperator(mx.operator.CustomOp):
    def __init__(self, K, height, width, pixel_means):
        super(ZoomImageOperator, self).__init__()
        self.K = K
        self.height = height
        self.width = width
        self.pixel_means = pixel_means

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        if type(self.pixel_means) == np.ndarray:
            self.pixel_means = mx.ndarray.array(self.pixel_means, ctx=ctx)
        batch_size = in_data[0].shape[0]
        image_real_array = in_data[0] + self.pixel_means
        image_rendered_array = in_data[1] + self.pixel_means
        src_pose_array = in_data[2].asnumpy()
        valid_pixels = image_real_array.asnumpy()
        valid_real_array = np.sum(valid_pixels, axis=1) > 0.01
        valid_pixels = image_rendered_array.asnumpy()
        valid_rendered_array = np.sum(valid_pixels, axis=1) > 0.01

        grid_array = mx.ndarray.zeros([batch_size, 2, self.height, self.width], ctx=ctx)
        zoom_factor_array = mx.ndarray.zeros([batch_size, 4], ctx=ctx)
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
                print("NO POINT VALID IN rendered")
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
            affine_matrix = mx.ndarray.array([[wx, 0, tx], [0, wy, ty]], ctx=ctx).reshape((1, 6))
            a = mx.ndarray.GridGenerator(
                data=affine_matrix, transform_type="affine", target_shape=(self.height, self.width)
            )
            grid_array[batch_idx] = a[0]

            zoom_factor_array[batch_idx, 0] = wx
            zoom_factor_array[batch_idx, 1] = wy
            zoom_factor_array[batch_idx, 2] = tx
            zoom_factor_array[batch_idx, 3] = ty

        zoom_image_real_array = mx.ndarray.BilinearSampler(image_real_array, grid_array)
        zoom_image_real_array -= self.pixel_means
        zoom_image_rendered_array = mx.ndarray.BilinearSampler(image_rendered_array, grid_array)
        zoom_image_rendered_array -= self.pixel_means

        self.assign(out_data[0], req[0], zoom_image_real_array)
        self.assign(out_data[1], req[1], zoom_image_rendered_array)
        self.assign(out_data[2], req[2], zoom_factor_array)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("ZoomImage")
class ZoomImageProp(mx.operator.CustomOpProp):
    def __init__(self, K, width=640, height=480, pixel_means="[0 0 0]"):
        super(ZoomImageProp, self).__init__(True)
        self.K = np.fromstring(K[1:-1], dtype=np.float32, sep=" ").reshape([3, 3])
        self.height = int(height)
        self.width = int(width)
        self.pixel_means = np.fromstring(pixel_means[1:-1], dtype=np.float32, sep=" ").reshape(3)
        self.pixel_means = self.pixel_means[::-1]
        self.pixel_means = self.pixel_means[np.newaxis, :, np.newaxis, np.newaxis]

    def list_arguments(self):
        input_list = ["image_observed", "image_rendered", "src_pose"]
        return input_list

    def list_outputs(self):
        output_list = ["zoom_image_observed", "zoom_image_rendered", "zoom_factor"]
        return output_list

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        out_shape = in_shape[:]
        out_shape[2] = [batch_size, 4]
        return in_shape, out_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        input_type = [dtype, dtype, dtype]
        output_type = [dtype, dtype, dtype]
        return input_type, output_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return ZoomImageOperator(self.K, self.height, self.width, self.pixel_means)


if __name__ == "__main__":
    # configs
    thresh = 1e-3
    step = 1e-4
    ctx = mx.gpu(0)
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    height = 480
    width = 640
    b_zoom_se3 = False
    batch_size = 8
    img_idx = ["005_{:06}".format(x) for x in [1, 11, 21, 31, 41, 51, 61, 71]]
    sub_idx1 = ["0" for x in range(batch_size)]
    sub_idx2 = ["5" for x in range(batch_size)]
    pixel_means = 120

    # initialize layer
    image_real = mx.sym.Variable("image_real")
    image_rendered = mx.sym.Variable("image_rendered")
    src_pose = mx.sym.Variable("src_pose")
    proj2d = mx.sym.Custom(
        image_real=image_real,
        image_rendered=image_rendered,
        src_pose=src_pose,
        K=K.flatten(),
        pixel_means=np.array([pixel_means, pixel_means, pixel_means]).flatten(),
        name="updater",
        op_type="ZoomImage",
        height=height,
        width=width,
    )
    v_image_real = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    v_image_rendered = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    v_src_pose = np.zeros((batch_size, 3, 4), dtype=np.float32)
    for idx in range(batch_size):
        v_image_rendered[idx] = cv2.imread(
            os.path.join(
                cur_dir, "../../data/render_v5/data/rendered/0003/{}_{}-color.png".format(img_idx[idx], sub_idx1[idx])
            ),
            cv2.IMREAD_COLOR,
        ).transpose([2, 0, 1])
        v_image_real[idx] = cv2.imread(
            os.path.join(
                cur_dir, "../../data/render_v5/data/rendered/0003/{}_{}-color.png".format(img_idx[idx], sub_idx2[idx])
            ),
            cv2.IMREAD_COLOR,
        ).transpose([2, 0, 1])
        v_src_pose[idx] = np.loadtxt(
            os.path.join(
                cur_dir, "../../data/render_v5/data/rendered/0003/{}_{}-pose.txt".format(img_idx[idx], sub_idx1[idx])
            ),
            skiprows=1,
        )
    exe1 = proj2d.simple_bind(
        ctx=ctx, image_real=v_image_real.shape, image_rendered=v_image_rendered.shape, src_pose=v_src_pose.shape
    )

    # forward
    exe1.arg_dict["image_real"][:] = mx.ndarray.array(v_image_real - pixel_means, ctx=ctx)
    exe1.arg_dict["image_rendered"][:] = mx.ndarray.array(v_image_rendered - pixel_means, ctx=ctx)
    exe1.arg_dict["src_pose"][:] = mx.ndarray.array(v_src_pose, ctx=ctx)
    import time
    import matplotlib.pyplot as plt

    t = time.time()
    exe1.forward(is_train=True)
    zoom_image_real = exe1.outputs[0].asnumpy().astype(np.uint8)
    zoom_image_rendered = exe1.outputs[1].asnumpy().astype(np.uint8)
    for batch_idx in range(batch_size):
        im_real = v_image_real[batch_idx].transpose([1, 2, 0]).astype(np.uint8)
        im_rendered = v_image_rendered[batch_idx].transpose([1, 2, 0]).astype(np.uint8)
        z_im_real = zoom_image_real[batch_idx].transpose([1, 2, 0]) + pixel_means
        z_im_rendered = zoom_image_rendered[batch_idx].transpose([1, 2, 0]) + pixel_means
        fig = plt.figure()
        tmp = fig.add_subplot(2, 2, 1)
        tmp.set_title("image_real")
        plt.axis("off")
        plt.imshow(im_real)

        tmp = fig.add_subplot(2, 2, 2)
        tmp.set_title("image_imagine")
        plt.axis("off")
        plt.imshow(im_rendered)

        tmp = fig.add_subplot(2, 2, 3)
        tmp.set_title("image_real after zoom")
        plt.axis("off")
        plt.imshow(z_im_real)

        tmp = fig.add_subplot(2, 2, 4)
        tmp.set_title("image_imagine after zoom")
        plt.axis("off")
        plt.imshow(z_im_rendered)
        plt.show()
