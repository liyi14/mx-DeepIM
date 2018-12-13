# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import sys
import os
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,
                os.path.join(cur_dir, '../../external/mxnet/mxnet_v00_origin'))
"""
zoom mask, using mask_real_est and mask_rendered to get the zoom area, zoom factor
output: zoomed mask_real_est, zoomed mask_real_gt, zoomed mask_rendered
"""

import cv2
import mxnet as mx
import numpy as np


class ZoomMaskWithFactorOperator(mx.operator.CustomOp):
    def __init__(self, height, width, b_inv_zoom):
        super(ZoomMaskWithFactorOperator, self).__init__()
        self.height = height
        self.width = width
        self.b_inv_zoom = b_inv_zoom

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        batch_size = in_data[0].shape[0]
        zoom_factor_array = in_data[0].asnumpy()
        mask_array = in_data[1]

        mask_np = mask_array.asnumpy()
        mask_np[mask_np > 0.2] = 1  # if the mask_input is depth
        mask_np[mask_np <= 0.2] = 0  # if the mask input is depth
        mask_array = mx.nd.array(mask_np, ctx=ctx)

        grid_array = mx.nd.zeros([batch_size, 2, self.height, self.width],
                                 ctx=ctx)

        for batch_idx in range(batch_size):
            if self.b_inv_zoom:
                wx_in, wy_in, tx_in, ty_in = zoom_factor_array[batch_idx]
                wx = 1 / wx_in
                wy = 1 / wy_in
                crop_height = wy_in * self.height
                crop_width = wx_in * self.width
                obj_real_c_x = tx_in * 0.5 * self.width + 0.5 * self.width
                obj_real_c_y = ty_in * 0.5 * self.height + 0.5 * self.height
                tx = (self.width * 0.5 - obj_real_c_x) / crop_width * 2
                ty = (self.height * 0.5 - obj_real_c_y) / crop_height * 2
            else:
                wx, wy, tx, ty = zoom_factor_array[batch_idx]

            affine_matrix = mx.ndarray.array([[wx, 0, tx], [0, wy, ty]],
                                             ctx=ctx).reshape((1, 6))
            a = mx.ndarray.GridGenerator(
                data=affine_matrix,
                transform_type='affine',
                target_shape=(self.height, self.width))
            grid_array[batch_idx] = a[0]

        zoom_mask_array = mx.nd.round(
            mx.nd.BilinearSampler(mask_array, grid_array))

        self.assign(out_data[0], req[0], zoom_mask_array)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('ZoomMaskWithFactor')
class ZoomMaskWithFactorProp(mx.operator.CustomOpProp):
    def __init__(self, width=640, height=480, b_inv_zoom='False'):
        super(ZoomMaskWithFactorProp, self).__init__(True)
        self.height = int(height)
        self.width = int(width)
        self.b_inv_zoom = b_inv_zoom.lower() == 'true'

    def list_arguments(self):
        input_list = ['zoom_factor', 'mask']
        return input_list

    def list_outputs(self):
        output_list = ['zoom_mask']
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
        return ZoomMaskWithFactorOperator(self.height, self.width,
                                          self.b_inv_zoom)


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from zoom_mask import *  # noqa:F401,F403
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

    # zoom mask ------------------------------------
    # initialize layer
    mask_real_est = mx.sym.Variable('mask_real_est')
    mask_real_gt = mx.sym.Variable('mask_real_gt')
    mask_rendered = mx.sym.Variable('mask_rendered')
    src_pose = mx.sym.Variable('src_pose')

    proj2d_1 = mx.sym.Custom(
        mask_real_est=mask_real_est,
        mask_real_gt=mask_real_gt,
        mask_rendered=mask_rendered,
        src_pose=src_pose,
        name='updater',
        op_type='ZoomMask',
        K=K.flatten(),
        height=height,
        width=width)
    print('zoom mask operator')
    v_mask_real_est = np.zeros((batch_size, 1, height, width),
                               dtype=np.float32)
    v_mask_real_gt = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_mask_rendered = np.zeros((batch_size, 1, height, width),
                               dtype=np.float32)
    v_src_pose = np.zeros((batch_size, 3, 4), dtype=np.float32)
    for idx in range(batch_size):
        tmp = cv2.imread(
            os.path.join(
                cur_dir,
                '../../data/render_v5/data/render_real/002_master_chef_can/0012/{:06}-depth.png'
                .format(sub_idx1[idx])), cv2.IMREAD_UNCHANGED) / 10000.
        tmp[tmp != 0] = 1
        tmp = tmp[np.newaxis, :, :]
        v_mask_real_est[idx] = tmp

        tmp = cv2.imread(
            os.path.join(
                cur_dir,
                '../../data/render_v5/data/render_real/002_master_chef_can/0012/{:06}-depth.png'
                .format(sub_idx1[idx])), cv2.IMREAD_UNCHANGED) / 10000.
        tmp[tmp != 0] = 1
        tmp = tmp[np.newaxis, :, :]
        v_mask_real_gt[idx] = tmp

        tmp = cv2.imread(
            os.path.join(
                cur_dir,
                '../../data/render_v5/data/render_real/002_master_chef_can/0012/{:06}-depth.png'
                .format(sub_idx2[idx])),  # img_idx[idx])),
            cv2.IMREAD_UNCHANGED) / 10000.
        tmp[tmp != 0] = 1
        tmp = tmp[np.newaxis, :, :]
        v_mask_rendered[idx] = tmp

        # src_pose of rendered
        v_src_pose[idx] = np.loadtxt(
            os.path.join(
                cur_dir,
                '../../data/render_v5/data/render_real/002_master_chef_can/0012/{:06}-pose.txt'
                .format(sub_idx2[idx])),
            skiprows=1)
    print('bind1')
    exe1 = proj2d_1.simple_bind(
        ctx=ctx,
        mask_real_est=v_mask_real_est.shape,
        mask_real_gt=v_mask_real_gt.shape,
        mask_rendered=v_mask_rendered.shape,
        src_pose=v_src_pose.shape)

    # forward
    def simple_forward_1(exe1,
                         v_mask_real_est,
                         v_mask_real_gt,
                         v_mask_rendered,
                         v_src_pose,
                         ctx,
                         is_train=False):
        exe1.arg_dict['mask_real_est'][:] = mx.ndarray.array(
            v_mask_real_est, ctx=ctx, dtype='float32')
        exe1.arg_dict['mask_real_gt'][:] = mx.ndarray.array(
            v_mask_real_gt, ctx=ctx, dtype='float32')
        exe1.arg_dict['mask_rendered'][:] = mx.ndarray.array(
            v_mask_rendered, ctx=ctx, dtype='float32')
        exe1.arg_dict['src_pose'][:] = mx.ndarray.array(
            v_src_pose, ctx=ctx, dtype='float32')
        exe1.forward(is_train=is_train)

    t = time.time()

    v_mask_real_est = v_mask_real_gt
    simple_forward_1(
        exe1,
        v_mask_real_est,
        v_mask_real_gt,
        v_mask_rendered,
        v_src_pose,
        ctx,
        is_train=True)
    zoom_mask_real_est = exe1.outputs[0].asnumpy()
    zoom_mask_real_gt = exe1.outputs[1].asnumpy()
    zoom_mask_rendered = exe1.outputs[2].asnumpy()
    zoom_factor = exe1.outputs[3].asnumpy()
    print('using time: {:.2f}'.format(time.time() - t))

    # inv_zoom_mask ------------------------------------------------------------
    # initialize layer
    mask_sym = mx.sym.Variable('mask')
    zoom_factor_sym = mx.sym.Variable('zoom_factor')
    proj2d_2 = mx.sym.Custom(
        zoom_factor=zoom_factor_sym,
        mask=mask_sym,
        name='updater1',
        op_type='ZoomMaskWithFactor',
        height=height,
        width=width,
        b_inv_zoom=True)
    v_mask = zoom_mask_rendered
    v_zoom_factor = zoom_factor

    print('v_zoom_factor: ', v_zoom_factor.shape)
    print('v_mask: ', v_mask.shape)

    exe2 = proj2d_2.simple_bind(
        ctx=ctx, zoom_factor=v_zoom_factor.shape, mask=v_mask.shape)

    # forward
    def simple_forward_2(exe, v_zoom_factor, v_mask, ctx, is_train=False):
        exe.arg_dict['mask'][:] = mx.nd.array(v_mask, ctx=ctx, dtype='float32')
        exe.arg_dict['zoom_factor'][:] = mx.nd.array(
            v_zoom_factor, ctx=ctx, dtype='float32')
        exe.forward(is_train=is_train)

    t = time.time()

    simple_forward_2(exe2, v_zoom_factor, v_mask, ctx, is_train=True)
    zoom_mask = exe2.outputs[0].asnumpy()
    print('using time: {:.2f}'.format(time.time() - t))

    for batch_idx in range(batch_size):

        im_real_est = np.squeeze(v_mask_real_est[batch_idx])
        im_real_gt = np.squeeze(v_mask_real_gt[batch_idx])
        im_rendered = np.squeeze(v_mask_rendered[batch_idx])
        z_im_real_est = np.squeeze(zoom_mask_real_est[batch_idx])
        z_im_real_gt = np.squeeze(zoom_mask_real_gt[batch_idx])
        z_im_rendered = np.squeeze(zoom_mask_rendered[batch_idx])

        fig = plt.figure()
        tmp = fig.add_subplot(3, 4, 1)
        tmp.set_title('mask_real_est')
        plt.axis('off')
        plt.imshow(im_real_est)

        tmp = fig.add_subplot(3, 4, 2)
        tmp.set_title('mask_real_gt')
        plt.axis('off')
        plt.imshow(im_real_gt)

        tmp = fig.add_subplot(3, 4, 3)
        tmp.set_title('mask_rendered')
        plt.axis('off')
        plt.imshow(im_rendered)

        tmp = fig.add_subplot(3, 4, 5)
        tmp.set_title('mask_real_est after zoom')
        plt.axis('off')
        plt.imshow(z_im_real_est)

        tmp = fig.add_subplot(3, 4, 6)
        tmp.set_title('mask_real_gt after zoom')
        plt.axis('off')
        plt.imshow(z_im_real_gt)

        tmp = fig.add_subplot(3, 4, 7)
        tmp.set_title('mask_rendered after zoom')
        plt.axis('off')
        plt.imshow(z_im_rendered)

        # ---------------------------
        im = np.squeeze(v_mask[batch_idx])
        z_im = np.squeeze(zoom_mask[batch_idx])

        tmp = fig.add_subplot(3, 4, 9)
        tmp.set_title('mask')
        plt.axis('off')
        plt.imshow(im)

        tmp = fig.add_subplot(3, 4, 10)
        tmp.set_title('zoom_mask')
        plt.axis('off')
        plt.imshow(z_im)

        plt.show()
