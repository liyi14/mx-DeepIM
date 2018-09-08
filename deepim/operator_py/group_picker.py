# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import sys, os
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_dir, '../../external/mxnet/mxnet_v00_0303'))
import cv2
import mxnet as mx
import numpy as np
class GroupPickerOperator(mx.operator.CustomOp):
    def __init__(self, group_num):
        super(GroupPickerOperator, self).__init__()
        self.group_num = group_num
        self.channels_in_group = -1

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        input_shape = in_data[0].shape
        batch_size = input_shape[0]
        bottom_data = in_data[0].asnumpy()
        pick_idx_data = in_data[1].asnumpy()
        output_shape = np.copy(input_shape)
        output_shape[1] /= self.group_num
        top_data = np.zeros(output_shape)

        if self.channels_in_group <= 0:
            assert input_shape[1] % self.group_num == 0
            self.channels_in_group = input_shape[1] / self.group_num

        for batch_idx in range(batch_size):
            g = int(np.squeeze(pick_idx_data[batch_idx]))
            assert g<self.group_num and g>=0
            top_data[batch_idx] = bottom_data[batch_idx][self.channels_in_group*g:self.channels_in_group*(g+1)]

        self.assign(out_data[0], req[0], mx.ndarray.array(top_data, ctx=ctx))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        ctx = in_data[0].context
        batch_size = in_data[0].shape[0]
        top_grad = out_grad[0].asnumpy()
        bottom_grad = np.zeros(in_data[0].shape)
        pick_idx_data = in_data[1].asnumpy()

        for batch_idx in range(batch_size):
            g = int(np.squeeze(pick_idx_data[batch_idx]))
            assert g<self.group_num and g>=0
            bottom_grad[batch_idx][self.channels_in_group*g:self.channels_in_group*(g+1)] = top_grad[batch_idx]

        self.assign(in_grad[0], req[0], mx.ndarray.array(bottom_grad, ctx=ctx))
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('GroupPicker')
class GroupPickerProp(mx.operator.CustomOpProp):
    def __init__(self, group_num):
        super(GroupPickerProp, self).__init__(True)
        self.group_num = int(group_num)

    def list_arguments(self):
        input_list = ['input_data', 'group_idx']
        return input_list

    def list_outputs(self):
        output_list = ['picked_data']
        return output_list

    def infer_shape(self, in_shape):
        picked_data_shape = np.copy(in_shape[0])
        picked_data_shape[1] /= self.group_num
        out_shape = [picked_data_shape]
        return in_shape, out_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        input_type = [dtype, dtype]
        output_type = [dtype]
        return input_type, output_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return GroupPickerOperator(self.group_num)


if __name__ == '__main__':
    # configs
    thresh = 1e-3
    step = 1e-4
    ctx = mx.gpu(3)
    batch_size = 8
    num_classes = 3

    # initialize layer
    rot_all_classes = mx.sym.Variable('rot_all_classes')
    class_index = mx.sym.Variable('class_index')
    proj2d = mx.sym.Custom(input_data=rot_all_classes, group_idx=class_index,
                           name='updater', op_type='GroupPicker', group_num=num_classes)
    v_rot_all_classes = np.random.rand(batch_size, 4*num_classes)
    v_class_index = np.random.randint(num_classes, size=[batch_size, 1])

    exe1 = proj2d.simple_bind(ctx=ctx, rot_all_classes=v_rot_all_classes.shape, class_index=v_class_index.shape)

    # forward
    exe1.arg_dict['rot_all_classes'][:] = mx.ndarray.array(v_rot_all_classes, ctx=ctx)
    exe1.arg_dict['class_index'][:] = mx.ndarray.array(v_class_index, ctx=ctx)
    import time
    import matplotlib.pyplot as plt
    t = time.time()
    exe1.forward(is_train=True)
    picked_data_mx = exe1.outputs[0].asnumpy()
    picked_data_py = np.zeros([batch_size, 4])
    for i in range(batch_size):
        g = int(v_class_index[i])
        picked_data_py[i] = v_rot_all_classes[i, g*4:(g+1)*4]

    for batch_idx in range(batch_size):
        print('py: ', picked_data_mx[batch_idx])
        print('mx: ', picked_data_py[batch_idx])
    print(picked_data_mx-picked_data_py)

    grad = mx.ndarray.ones_like(exe1.outputs[0])
    exe1.backward(grad)
    grad_est_all = exe1.grad_dict['rot_all_classes'].asnumpy()
    print(grad_est_all)

    print(np.squeeze(v_class_index))

