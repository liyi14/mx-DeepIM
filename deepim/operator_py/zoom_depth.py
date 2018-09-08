# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import sys, os
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_dir, '../../external/mxnet/mxnet_v00_origin'))

import cv2
import mxnet as mx
import numpy as np
class ZoomDepthOperator(mx.operator.CustomOp):
    def __init__(self, height, width):
        super(ZoomDepthOperator, self).__init__()
        self.height = height
        self.width = width

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        batch_size = in_data[0].shape[0]
        zoom_factor = in_data[0].asnumpy()

        grid_array = mx.ndarray.zeros([batch_size, 2, self.height, self.width], ctx=ctx)
        for batch_idx in range(batch_size):
            wx, wy, tx, ty = zoom_factor[batch_idx]
            print("wx: {}, wy: {}, tx: {}, ty: {}".format(wx, wy, tx, ty))
            affine_matrix = mx.ndarray.array([[wx, 0, tx], [0, wy, ty]], ctx=ctx).reshape((1, 6))
            a = mx.ndarray.GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(self.height, self.width))
            grid_array[batch_idx] = a[0]

        depth_real_array = in_data[1]
        depth_rendered_array = in_data[2]
        zoom_depth_real_array = mx.ndarray.BilinearSampler(depth_real_array, grid_array)
        zoom_depth_rendered_array = mx.ndarray.BilinearSampler(depth_rendered_array, grid_array)
        self.assign(out_data[0], req[0], zoom_depth_real_array)
        self.assign(out_data[1], req[1], zoom_depth_rendered_array)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

@mx.operator.register('ZoomDepth')
class ZoomDepthProp(mx.operator.CustomOpProp):
    def __init__(self, width=640, height=480):
        super(ZoomDepthProp, self).__init__(True)
        self.height = int(height)
        self.width = int(width)

    def list_arguments(self):
        input_list = ['zoom_factor', 'depth_real', 'depth_rendered']
        return input_list

    def list_outputs(self):
        output_list = ['zoom_depth_real', 'zoom_depth_rendered']
        return output_list

    def infer_shape(self, in_shape):
        out_shape = in_shape[1:]
        return in_shape, out_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        input_type = [dtype, dtype, dtype]
        output_type = [dtype, dtype]
        return input_type, output_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return ZoomDepthOperator(self.height, self.width)


if __name__ == '__main__':
    # configs
    thresh = 1e-3
    step = 1e-4
    ctx = mx.gpu(3)
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    height = 480
    width = 640
    b_zoom_se3 = False
    batch_size = 8
    img_idx = ['005_{:06}'.format(x) for x in [1, 11, 21, 31, 41, 51, 61, 71]]
    sub_idx1 = ['0' for x in range(batch_size)]
    sub_idx2 = ['5' for x in range(batch_size)]
    pixel_means = 120

    # initialize layer
    zoom_factor = mx.sym.Variable('zoom_factor')
    depth_real = mx.sym.Variable('depth_real')
    depth_rendered = mx.sym.Variable('depth_rendered')
    proj2d = mx.sym.Custom(zoom_factor=zoom_factor, depth_real=depth_real, depth_rendered=depth_rendered,
                           name='updater', op_type='ZoomDepth',
                           height=height, width=width)
    v_zoom_factor = np.ones((batch_size, 4))*2#np.random.rand(batch_size, 4) * 2
    v_zoom_factor[:, 1] = v_zoom_factor[:, 0]
    v_zoom_factor[:, 2:] = 0.
    v_depth_real = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_depth_rendered = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    for idx in range(batch_size):
        v_depth_rendered[idx] = cv2.imread(
            '/home/yili/PoseEst/mx-DeepPose/data/render_v5/data/rendered/0003/{}_{}-depth.png'.format(img_idx[idx], sub_idx1[idx]),
            cv2.IMREAD_COLOR).transpose([2, 0, 1])[np.newaxis, 0, :, :]/10000.0
        v_depth_real[idx] = cv2.imread(
            '/home/yili/PoseEst/mx-DeepPose/data/render_v5/data/rendered/0003/{}_{}-depth.png'.format(img_idx[idx], sub_idx2[idx]),
            cv2.IMREAD_COLOR).transpose([2, 0, 1])[np.newaxis, 0, :, :]/10000.0

    exe1 = proj2d.simple_bind(ctx=ctx, zoom_factor=v_zoom_factor.shape, depth_real=v_depth_real.shape, depth_rendered=v_depth_rendered.shape)

    # forward
    exe1.arg_dict['zoom_factor'][:] = mx.ndarray.array(v_zoom_factor, ctx=ctx)
    exe1.arg_dict['depth_real'][:] = mx.ndarray.array(v_depth_real, ctx=ctx)
    exe1.arg_dict['depth_rendered'][:] = mx.ndarray.array(v_depth_rendered, ctx=ctx)
    import time
    import matplotlib.pyplot as plt
    t = time.time()
    exe1.forward(is_train=True)
    zoom_depth_real = exe1.outputs[0].asnumpy()
    zoom_depth_rendered = exe1.outputs[1].asnumpy()
    for batch_idx in range(batch_size):
        d_real = np.squeeze(v_depth_real[batch_idx].transpose([1, 2, 0]))
        d_rendered = np.squeeze(v_depth_rendered[batch_idx].transpose([1, 2, 0]))
        z_d_real = np.squeeze(zoom_depth_real[batch_idx].transpose([1, 2, 0]))
        z_d_rendered = np.squeeze(zoom_depth_rendered[batch_idx].transpose([1, 2, 0]))
        fig = plt.figure()
        tmp=fig.add_subplot(2, 2, 1)
        tmp.set_title('image_real')
        plt.axis('off')
        plt.imshow(d_real)

        tmp =fig.add_subplot(2, 2, 2)
        tmp.set_title('image_imagine')
        plt.axis('off')
        plt.imshow(d_rendered)

        tmp =fig.add_subplot(2, 2, 3)
        tmp.set_title('image_real after zoom')
        plt.axis('off')
        plt.imshow(z_d_real)

        tmp =fig.add_subplot(2, 2, 4)
        tmp.set_title('image_imagine after zoom')
        plt.axis('off')
        plt.imshow(z_d_rendered)
        plt.show()