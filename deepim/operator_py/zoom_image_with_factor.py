# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import sys
import os

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../external/mxnet/mxnet_v00_origin"))
"""
zoom image with zoom factor
"""

import cv2
import mxnet as mx
import numpy as np


class ZoomImageWithFactorOperator(mx.operator.CustomOp):
    def __init__(self, height, width, pixel_means, high_light_center):
        super(ZoomImageWithFactorOperator, self).__init__()
        self.height = height
        self.width = width
        self.pixel_means = pixel_means
        self.high_light_center = high_light_center
        self.center_map = None
        self.spot_radius = 5

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        if type(self.pixel_means) == np.ndarray:
            self.pixel_means = mx.nd.array(self.pixel_means, ctx=ctx)
        batch_size = in_data[0].shape[0]
        if self.center_map is None:
            self.center_map = np.zeros([batch_size, 3, self.height, self.width])
            spot_start_x = int(np.floor(float(self.width) / 2 - self.spot_radius))
            spot_end_x = int(np.ceil(float(self.width) / 2 + self.spot_radius))
            spot_start_y = int(np.floor(float(self.height) / 2 - self.spot_radius))
            spot_end_y = int(np.ceil(float(self.height) / 2 + self.spot_radius))
            self.center_map[
                :, 0, spot_start_y:spot_end_y, spot_start_x:spot_end_x
            ] = 255.0  # red center
            self.center_map = mx.ndarray.array(self.center_map, ctx=ctx)
        image_real_array = in_data[1] + self.pixel_means
        image_rendered_array = in_data[2] + self.pixel_means

        grid_array = mx.ndarray.zeros([batch_size, 2, self.height, self.width], ctx=ctx)
        zoom_factor = in_data[0].asnumpy()
        for batch_idx in range(batch_size):
            wx, wy, tx, ty = zoom_factor[batch_idx]
            affine_matrix = mx.ndarray.array(
                [[wx, 0, tx], [0, wy, ty]], ctx=ctx
            ).reshape((1, 6))
            a = mx.ndarray.GridGenerator(
                data=affine_matrix,
                transform_type="affine",
                target_shape=(self.height, self.width),
            )
            grid_array[batch_idx] = a[0]

        zoom_image_rendered_array = mx.nd.BilinearSampler(
            image_rendered_array, grid_array
        )
        zoom_image_real_array = mx.nd.BilinearSampler(image_real_array, grid_array)
        if self.high_light_center:
            zoom_image_rendered_array = mx.ndarray.maximum(
                zoom_image_rendered_array, self.center_map
            )
        zoom_image_real_array -= self.pixel_means
        zoom_image_rendered_array -= self.pixel_means

        self.assign(out_data[0], req[0], zoom_image_real_array)
        self.assign(out_data[1], req[1], zoom_image_rendered_array)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("ZoomImageWithFactor")
class ZoomImageWithFactorProp(mx.operator.CustomOpProp):
    def __init__(
        self, width=640, height=480, pixel_means="[0 0 0]", high_light_center="False"
    ):
        super(ZoomImageWithFactorProp, self).__init__(True)
        self.height = int(height)
        self.width = int(width)
        self.pixel_means = np.fromstring(
            pixel_means[1:-1], dtype=np.float32, sep=" "
        ).reshape(3)
        self.pixel_means = self.pixel_means[::-1]
        self.pixel_means = self.pixel_means[np.newaxis, :, np.newaxis, np.newaxis]
        self.hight_light_center = high_light_center.lower() == "true"

    def list_arguments(self):
        input_list = ["zoom_factor", "image_observed", "image_rendered"]
        return input_list

    def list_outputs(self):
        output_list = ["zoom_image_observed", "zoom_image_rendered"]
        return output_list

    def infer_shape(self, in_shape):
        out_shape = [in_shape[1], in_shape[2]]

        return in_shape, out_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        input_type = [dtype, dtype, dtype]
        output_type = [dtype, dtype]
        return input_type, output_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return ZoomImageWithFactorOperator(
            self.height, self.width, self.pixel_means, self.hight_light_center
        )


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
    zoom_factor = mx.sym.Variable("zoom_factor")
    proj2d = mx.sym.Custom(
        zoom_factor=zoom_factor,
        image_real=image_real,
        image_rendered=image_rendered,
        pixel_means=np.array([pixel_means, pixel_means, pixel_means]).flatten(),
        name="updater",
        op_type="ZoomImageWithFactor",
        height=height,
        width=width,
        high_light_center=True,
    )
    v_zoom_factor = np.ones((batch_size, 4)) * 0.8
    v_zoom_factor[:, 1] = v_zoom_factor[:, 0]
    v_zoom_factor[:, 2:] = 0.0
    v_image_real = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    v_image_rendered = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    for idx in range(batch_size):
        v_image_rendered[idx] = cv2.imread(
            os.path.join(
                cur_dir,
                "../../data/render_v5/data/rendered/0003/{}_{}-color.png".format(
                    img_idx[idx], sub_idx1[idx]
                ),
            ),
            cv2.IMREAD_COLOR,
        ).transpose([2, 0, 1])
        v_image_real[idx] = cv2.imread(
            os.path.join(
                cur_dir,
                "../../data/render_v5/data/rendered/0003/{}_{}-color.png".format(
                    img_idx[idx], sub_idx2[idx]
                ),
            ),
            cv2.IMREAD_COLOR,
        ).transpose([2, 0, 1])

    exe1 = proj2d.simple_bind(
        ctx=ctx,
        zoom_factor=v_zoom_factor.shape,
        image_real=v_image_real.shape,
        image_rendered=v_image_rendered.shape,
    )

    # forward
    exe1.arg_dict["zoom_factor"][:] = mx.nd.array(v_zoom_factor, ctx=ctx)
    exe1.arg_dict["image_real"][:] = mx.nd.array(v_image_real - pixel_means, ctx=ctx)
    exe1.arg_dict["image_rendered"][:] = mx.nd.array(
        v_image_rendered - pixel_means, ctx=ctx
    )
    import time
    import matplotlib.pyplot as plt

    t = time.time()
    exe1.forward(is_train=True)
    zoom_image_real = exe1.outputs[0].asnumpy().astype(np.uint8)
    zoom_image_rendered = exe1.outputs[1].asnumpy().astype(np.uint8)
    print("using time: {:.2f}s".format(time.time() - t))

    for batch_idx in range(batch_size):
        im_real = v_image_real[batch_idx].transpose([1, 2, 0]).astype(np.uint8)
        im_rendered = v_image_rendered[batch_idx].transpose([1, 2, 0]).astype(np.uint8)
        z_im_real = zoom_image_real[batch_idx].transpose([1, 2, 0]) + pixel_means
        z_im_rendered = (
            zoom_image_rendered[batch_idx].transpose([1, 2, 0]) + pixel_means
        )
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
