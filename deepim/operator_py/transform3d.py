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
"""
transform 3d
input: model points, pose src, R_delta, T_delta; T_means, T_stds
forward: (pose_src, RT_delta) -> pose_tgt, (model points, pose_tgt) -> points_tgt
ouput: points_tgt (transformed_3d_points)
"""

import mxnet as mx
import numpy as np
from lib.pair_matching.RT_transform import (
    RT_transform,
    quat2mat,
    R_transform,
    T_transform,
    T_transform_naive,
)
from distutils.util import strtobool


class transform3dOperator(mx.operator.CustomOp):
    def __init__(
        self, T_means=None, T_stds=None, rot_coord="MODEL", projection_2d=False
    ):
        super(transform3dOperator, self).__init__()
        self.T_means = T_means
        self.T_stds = T_stds
        self._projection_2d = projection_2d
        self.rot_coord = rot_coord
        assert not projection_2d, "NOT_IMPLEMENTED"

    def forward(self, is_train, req, in_data, out_data, aux):
        # bottom_3d_points: batch_size x 3 x spatial_dim
        # rotation: R_delta, batch_size x {3 or 4}
        # translation: T_delta, batch_size x 3
        # pose_src: (B, 3, 4)
        # ----
        # top_3d_points: batch_size x 3 x spatial_dim
        batch_size = in_data[0].shape[0]
        ctx = in_data[0].context
        # T_means = mx.nd.array(self.T_means, ctx=ctx)
        # T_stds = mx.nd.array(self.T_stds, ctx=ctx)

        bottom_3d_points = in_data[0]
        bottom_3d_points = bottom_3d_points.reshape((batch_size, 3, -1))
        rotation = in_data[1]
        T_delta = in_data[2]  # mx.nd.expand_dims(in_data[2], axis=2)
        pose_src = in_data[3]
        Rm_src = mx.nd.slice_axis(pose_src, axis=2, begin=0, end=3)  # (B, 3, 3)
        T_src = mx.nd.slice_axis(pose_src, axis=2, begin=3, end=4).reshape(
            (batch_size, 3)
        )  # (B, 3)

        assert (
            rotation.shape[0] == batch_size and T_delta.shape[0] == batch_size
        ), "rotation.shape[0]:{} vs batch_size:{}, translation.shape[0]:{} vs batch_size:{}".format(
            rotation.shape[0], batch_size, T_delta.shape[0], batch_size
        )

        # R_delta --quat2mat-->Rm_delta
        Rm_delta = mx.ndarray.zeros((batch_size, 3, 3), ctx=ctx, dtype="float32")
        for batch_idx in range(batch_size):
            if rotation.shape[1] == 4:
                Rm_delta[batch_idx] = self.quat2mat_forward(rotation[batch_idx])
            elif rotation.shape[1] == 3:
                raise Exception("NOT_IMPLEMENTED")
            else:
                raise Exception(
                    "UNKNOWN ROTATION REPRESENTATION {}".format(rotation.shape[1])
                )
        self.Rm_delta = Rm_delta

        # rot coord == 'model', dot(Rm_src, Rm_delta)==>Rm_tgt
        Rm_tgt = mx.nd.zeros((batch_size, 3, 3), ctx=ctx)
        for i in range(batch_size):
            Rm_tgt[i] = mx.nd.array(
                R_transform(
                    Rm_src[i].asnumpy(), Rm_delta[i].asnumpy(), rot_coord=self.rot_coord
                ),
                ctx=ctx,
            )

        # T_delta, T_src --> T_tgt
        T_tgt = mx.nd.zeros((batch_size, 3), ctx=ctx)
        for i in range(batch_size):
            if self.rot_coord.lower() == "naive":
                T_tgt[i] = mx.nd.array(
                    T_transform_naive(
                        Rm_delta[i].asnumpy(), T_src[i].asnumpy(), T_delta[i].asnumpy()
                    )
                )
            else:
                T_tgt[i] = mx.nd.array(
                    T_transform(
                        T_src[i].asnumpy(),
                        T_delta[i].asnumpy(),
                        self.T_means,
                        self.T_stds,
                        rot_coord=self.rot_coord,
                    ),
                    ctx=ctx,
                )
        T_tgt = mx.nd.expand_dims(T_tgt, axis=2)  # (B, 3, 1)
        top_3d_points = mx.nd.add(mx.nd.batch_dot(Rm_tgt, bottom_3d_points), T_tgt)

        output = top_3d_points.reshape(in_data[0].shape)
        for ind, val in enumerate([output]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        batch_size = in_data[0].shape[0]
        ctx = in_data[0].context
        T_means = mx.nd.array(self.T_means, ctx=ctx)
        T_stds = mx.nd.array(self.T_stds, ctx=ctx)

        bottom_3d_points = in_data[0].reshape((batch_size, 3, -1))
        rotation = in_data[1]
        T_delta = in_data[2]
        pose_src = in_data[3]

        Rm_src = mx.nd.slice_axis(pose_src, axis=2, begin=0, end=3)  # (B, 3, 3)
        T_src = mx.nd.slice_axis(pose_src, axis=2, begin=3, end=4).reshape(
            (batch_size, 3)
        )  # (B, 3)

        top_3d_diff = out_grad[0].reshape((batch_size, 3, -1))

        # T diff
        T_tgt_diff = mx.nd.sum(top_3d_diff, axis=2)  # (B,3)
        T_delta_diff = mx.nd.zeros((batch_size, 3), ctx=ctx)
        if self.rot_coord.lower() == "naive":
            T_delta_diff = T_tgt_diff
        else:
            for i in range(batch_size):
                T_delta_diff[i] = self.T_transform_backward(
                    T_tgt_diff[i],
                    T_src[i],
                    T_delta[i],
                    T_means,
                    T_stds,
                    ctx=ctx,
                    rot_coord=self.rot_coord,
                )

        # R diff
        Rm_tgt_diff = mx.nd.batch_dot(
            top_3d_diff, bottom_3d_points, transpose_b=True
        )  # (B,3,3)
        Rm_src_T = mx.nd.transpose(Rm_src, axes=(0, 2, 1))
        if self.rot_coord.lower() == "model":
            Rm_delta_diff = mx.nd.batch_dot(Rm_src_T, Rm_tgt_diff)
        elif (
            self.rot_coord.lower() == "camera" or self.rot_coord.lower() == "camera_new"
        ):
            Rm_delta_diff = mx.nd.batch_dot(Rm_tgt_diff, Rm_src_T)
        elif self.rot_coord.lower() == "naive":
            src_3d_points = mx.nd.add(
                mx.nd.batch_dot(Rm_src, bottom_3d_points),
                mx.nd.expand_dims(T_src, axis=2),
            )
            Rm_delta_diff = mx.nd.batch_dot(
                top_3d_diff, src_3d_points, transpose_b=True
            )  # (B,3,3)

        else:
            raise Exception(
                "Unknown rot_coord in transform3d operator: {}".format(self.rot_coord)
            )

        R_delta_diff = mx.nd.zeros_like(rotation, ctx=ctx, dtype="float32")

        for batch_idx in range(batch_size):
            # rotation(R_delta) --quat2mat--> Rm_delta
            R_delta_diff[batch_idx] = self.quat2mat_backward(
                Rm_delta_diff[batch_idx], rotation[batch_idx], self.Rm_delta[batch_idx]
            )

        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], R_delta_diff)
        self.assign(in_grad[2], req[2], T_delta_diff)
        self.assign(in_grad[3], req[3], 0)

    def T_transform_backward(
        self, T_tgt_diff, T_src, T_delta, T_means, T_stds, ctx, rot_coord="model"
    ):
        """dy_dx, y: (3,1), x: (1,3)
        :param T_src:
        :param T_delta:
        :param T_means:
        :param T_stds:
        :param ctx:
        :return:
        """
        assert T_src[2].asnumpy() != 0, "T_src: {}".format(T_src)
        D = T_tgt_diff
        T_delta_1 = T_delta * T_stds + T_means
        z2 = T_src[2] / mx.nd.exp(T_delta_1[2])

        T_delta_diff = mx.nd.zeros((3,), ctx=ctx)
        if rot_coord.lower() == "camera" or rot_coord.lower() == "model":
            T_delta_diff[0] = D[0] * (T_stds[0] * z2) + D[1] * 0 + D[2] * 0
            T_delta_diff[1] = D[0] * 0 + D[1] * (T_stds[1] * z2) + D[2] * 0
            share_term = -T_stds[2] * z2
            T_delta_diff[2] = (
                D[0] * (share_term * (T_delta_1[0] + T_src[0] / T_src[2]))
                + D[1] * (share_term * (T_delta_1[1] + T_src[1] / T_src[2]))
                + D[2] * (-T_stds[2] * z2)
            )  # noqa:E127
        elif rot_coord.lower() == "camera_new":
            T_delta_diff[0] = D[0] * (T_stds[0] * T_src[2]) + D[1] * 0 + D[2] * 0
            T_delta_diff[1] = D[0] * 0 + D[1] * (T_stds[1] * T_src[2]) + D[2] * 0
            share_term = -T_stds[2] * z2
            T_delta_diff[2] = D[0] * 0 + D[1] * 0 + D[2] * (-T_stds[2] * z2)

        return T_delta_diff

    def quat2mat_forward(self, q):
        w, x, y, z = q.asnumpy()
        Nq = w * w + x * x + y * y + z * z
        if not (-1e-2 < Nq - 1 < 1e-2):  # Nq < 1e-8:
            return mx.nd.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], ctx=q.context)

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
        return mx.nd.array(
            [
                [1.0 - (yY + zZ), xY - wZ, xZ + wY],
                [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
            ],
            ctx=q.context,
            dtype="float32",
        )

    def quat2mat_backward(self, mat_diff, q, M):
        w, x, y, z = q.asnumpy()
        Nq = w * w + x * x + y * y + z * z
        Nq_sqrt = np.sqrt(Nq)
        w_, x_, y_, z_ = q.asnumpy() / Nq_sqrt

        s = 2.0
        if not (-1e-4 < Nq - 1 < 1e-4):
            return mx.nd.zeros(4, ctx=q.context, dtype="float32")
        D = mat_diff.asnumpy()
        w_diff_ = (
            0 * D[0, 0]
            - z_ * D[0, 1]
            + y_ * D[0, 2]
            + z_ * D[1, 0]
            + 0 * D[1, 1]
            - x_ * D[1, 2]
            - y_ * D[2, 0]
            + x_ * D[2, 1]
            + 0 * D[2, 2]
        )  # noqa:E127
        w_diff_ *= s

        x_diff_ = (
            0 * D[0, 0]
            + y_ * D[0, 1]
            + z_ * D[0, 2]
            + y_ * D[1, 0]
            - 2 * x_ * D[1, 1]
            - w_ * D[1, 2]
            + z_ * D[2, 0]
            + w_ * D[2, 1]
            - 2 * x_ * D[2, 2]
        )  # noqa:E127
        x_diff_ *= s

        y_diff_ = (
            -2 * y_ * D[0, 0]
            + x_ * D[0, 1]
            + w_ * D[0, 2]
            + x_ * D[1, 0]
            + 0 * D[1, 1]
            + z_ * D[1, 2]
            - w_ * D[2, 0]
            + z_ * D[2, 1]
            - 2 * y_ * D[2, 2]
        )  # noqa:E127
        y_diff_ *= s

        z_diff_ = (
            -2 * z_ * D[0, 0]
            - w_ * D[0, 1]
            + x_ * D[0, 2]
            + w_ * D[1, 0]
            - 2 * z_ * D[1, 1]
            + y_ * D[1, 2]
            + x_ * D[2, 0]
            + y_ * D[2, 1]
            + 0 * D[2, 2]
        )  # noqa:E127
        z_diff_ *= s

        share_term = Nq_sqrt ** 3 * (
            w * w_diff_ + x * x_diff_ + y * y_diff_ + z * z_diff_
        )
        w_diff = Nq_sqrt * w_diff_ - w * share_term
        x_diff = Nq_sqrt * x_diff_ - x * share_term
        y_diff = Nq_sqrt * y_diff_ - y * share_term
        z_diff = Nq_sqrt * z_diff_ - z * share_term
        return mx.nd.array(
            [w_diff, x_diff, y_diff, z_diff], ctx=q.context, dtype="float32"
        )


@mx.operator.register("Transform3D")
class transform3dProp(mx.operator.CustomOpProp):
    def __init__(
        self, T_means=None, T_stds=None, rot_coord="MODEL", b_project_2d="False"
    ):
        super(transform3dProp, self).__init__(True)
        self.T_means = np.fromstring(T_means[1:-1], dtype=np.float32, sep=" ").reshape(
            (3,)
        )
        self.T_stds = np.fromstring(T_stds[1:-1], dtype=np.float32, sep=" ").reshape(
            (3,)
        )
        self._b_project_2d = strtobool(b_project_2d)
        self.rot_coord = rot_coord

    def list_arguments(self):
        return ["point_cloud", "rotation", "translation", "pose_src"]

    def list_outputs(self):
        return ["transformed_3d_points"]

    def infer_shape(self, in_shape):
        output_shape = in_shape[0]
        return in_shape, [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return transform3dOperator(
            self.T_means, self.T_stds, self.rot_coord, self._b_project_2d
        )


if __name__ == "__main__":

    def set_diff(top_diff):
        top_diff[:, :, :] = 1

    def get_output(y):
        return np.sum(y.astype(np.float32))

    def load_object_points(point_path):
        # print(point_path)
        assert os.path.exists(point_path), "Path does not exist: {}".format(point_path)
        points = np.loadtxt(point_path)
        return points

    a = 1
    print("random seed: ", a)
    np.random.seed(a)
    ctx = mx.gpu(0)
    rot_coord = "CAMERA"
    step = 1e-2
    thresh = 0.005
    batch_size = 8
    num_points = 3000
    T_means = np.array([0.0, 0.0, 0.0])
    T_stds = np.array([1.0, 1.0, 1.0])
    class_name = "035_power_drill"
    src_img_idx = ["{:06}".format(x * 100 + 1) for x in range(batch_size)]

    model_dir = os.path.join(cur_dir, "../../data/LOV/models/{}".format(class_name))
    point_file = os.path.join(model_dir, "points.xyz")
    points = load_object_points(point_file)
    points = points.T.astype("float32")

    pose_path = os.path.join(
        cur_dir,
        "../../data/render_v5/data/render_real/%s/0006/{}-pose.txt" % (class_name),
    )

    point_cloud = mx.sym.Variable("point_cloud")
    rotation = mx.sym.Variable("rotation")
    translation = mx.sym.Variable("translation")
    pose_src = mx.sym.Variable("pose_src")

    tran3d = mx.sym.Custom(
        point_cloud=point_cloud,
        rotation=rotation,
        translation=translation,
        pose_src=pose_src,
        name="tran3d",
        op_type="Transform3D",
        T_means=T_means,
        T_stds=T_stds,
        rot_coord=rot_coord,
    )

    v_point_cloud = np.tile(points[:, :num_points], (batch_size, 1, 1))
    print("v_point_cloud ", v_point_cloud.shape)
    v_rotation = np.random.randint
    v_rotation = np.random.randint(1e4, size=[batch_size, 4]) / 1e4
    v_rotation = np.array(
        [x / np.sqrt(np.dot(x, x)) for x in v_rotation], dtype="float32"
    )
    v_translation = np.random.randint(1e4, size=[batch_size, 3]) / 1e4
    v_translation = v_translation.astype("float32")
    print(v_translation)
    v_pose_src = np.tile(
        np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]]), (batch_size, 1, 1)
    )  # (B, 3, N)
    print("v_pose_src:", v_pose_src.shape)

    exe1 = tran3d.simple_bind(
        ctx=ctx,
        point_cloud=v_point_cloud.shape,
        rotation=v_rotation.shape,
        translation=v_translation.shape,
        pose_src=v_pose_src.shape,
    )

    def simple_forward(
        exe1, v_point_cloud, v_rotation, v_translation, v_pose_src, ctx, is_train=False
    ):
        exe1.arg_dict["point_cloud"][:] = mx.ndarray.array(
            v_point_cloud, ctx=ctx, dtype="float32"
        )
        exe1.arg_dict["rotation"][:] = mx.ndarray.array(
            v_rotation, ctx=ctx, dtype="float32"
        )
        exe1.arg_dict["translation"][:] = mx.ndarray.array(
            v_translation, ctx=ctx, dtype="float32"
        )
        exe1.arg_dict["pose_src"][:] = mx.ndarray.array(
            v_pose_src, ctx=ctx, dtype="float32"
        )
        exe1.forward(is_train=is_train)

    simple_forward(
        exe1,
        v_point_cloud,
        v_rotation,
        v_translation,
        v_pose_src,
        ctx=ctx,
        is_train=True,
    )
    y1_raw = exe1.outputs[0]
    y1 = y1_raw.asnumpy()
    sum_y1 = get_output(y1)
    # forward check
    for iter in range(batch_size):
        pc = np.squeeze(v_point_cloud[iter])
        rot_q_delta = v_rotation[iter]
        trans_delta = v_translation[iter]
        pose_src = v_pose_src[iter]
        rot_m_delta = quat2mat(rot_q_delta)

        pose_est = RT_transform(
            pose_src, rot_q_delta, trans_delta, T_means, T_stds, rot_coord
        )
        rot_m_tgt = pose_est[:3, :3]
        trans_tgt = pose_est[:, 3]

        pc_gt = np.matmul(rot_m_tgt, pc) + trans_tgt.reshape((3, 1))
        pc_est = y1[iter]
        assert (
            np.abs(pc_est - pc_gt).max(1) < 1e-4
        ).all(), "forward check failed, {} > {}".format(
            np.abs(pc_est - pc_gt).sum(), thresh
        )
    print("forward check succeed")

    top_grad = mx.nd.zeros_like(y1_raw)
    set_diff(top_grad)
    exe1.backward(top_grad)
    grad_rot = exe1.grad_dict["rotation"].asnumpy()
    rot_shape = grad_rot.shape
    grad_trans = exe1.grad_dict["translation"].asnumpy()
    trans_shape = grad_trans.shape

    for test_idx in range(int(v_translation.size / 2)):
        i = np.random.randint(trans_shape[0])
        j = np.random.randint(trans_shape[1])
        grad_est = grad_trans[i, j]
        stride = step * v_translation[i, j]

        v_translation_neg = np.copy(v_translation)
        v_translation_neg[i, j] -= stride

        v_translation_pos = np.copy(v_translation)
        v_translation_pos[i, j] += stride

        simple_forward(
            exe1, v_point_cloud, v_rotation, v_translation_neg, v_pose_src, ctx=ctx
        )
        y0 = exe1.outputs[0].asnumpy()
        sum_y0 = y0.sum()

        simple_forward(
            exe1, v_point_cloud, v_rotation, v_translation_pos, v_pose_src, ctx=ctx
        )

        y2 = exe1.outputs[0].asnumpy()
        sum_y2 = y2.sum()
        # forward check
        for iter in range(batch_size):
            pc = np.squeeze(v_point_cloud[iter])
            rot_q_delta = v_rotation[iter]
            trans_delta = v_translation_pos[iter]
            pose_src = v_pose_src[iter]
            rot_m_delta = quat2mat(rot_q_delta)

            pose_est = RT_transform(
                pose_src, rot_q_delta, trans_delta, T_means, T_stds, rot_coord
            )
            rot_m_tgt = pose_est[:3, :3]
            trans_tgt = pose_est[:, 3]

            pc_gt = np.matmul(rot_m_tgt, pc) + trans_tgt.reshape((3, 1))

            pc_est = y2[iter]
            assert (np.abs(pc_est - pc_gt).max(1) < thresh).all()

        print("trans forward check succeed")

        grad_gt = (sum_y2 - sum_y0) / (2 * stride)
        abs_diff = grad_est - grad_gt
        rel_diff = np.abs(grad_est - grad_gt) / max(np.abs(grad_gt), 1e-15)
        if not (abs_diff < thresh or rel_diff < thresh):
            print(
                "test_idx:{}, translation backward error, abs_diff:{}, rel_diff:{}, thresh:{}".format(
                    test_idx, abs_diff, rel_diff, thresh
                )
            )

        if abs_diff < thresh or rel_diff < thresh:
            print("trans backward check succeed")
        else:
            print(
                "sample_point: {}, grad_est: {}, grad_gt: {}, abs_diff: {}, rel_diff: {}, v_translation: {}".format(
                    (i, j),
                    grad_est,
                    grad_gt,
                    grad_est - grad_gt,
                    np.abs(grad_est - grad_gt) / max(np.abs(grad_gt), 1e-15),
                    v_translation[i],
                )
            )
    print("v_translation backward check pass")

    for i in range(v_rotation.shape[0]):
        for j in range(v_rotation.shape[1]):
            grad_est = grad_rot[i, j]

            v_rotation_neg = np.copy(v_rotation)
            v_rotation_neg[i, j] -= stride
            v_rotation_neg = np.array(
                [x / np.sqrt(np.dot(x, x)) for x in v_rotation_neg]
            )

            v_rotation_pos = np.copy(v_rotation)
            v_rotation_pos[i, j] += stride
            v_rotation_pos = np.array(
                [x / np.sqrt(np.dot(x, x)) for x in v_rotation_pos]
            )

            simple_forward(
                exe1, v_point_cloud, v_rotation_neg, v_translation, v_pose_src, ctx=ctx
            )
            y0 = exe1.outputs[0].asnumpy()
            sum_y0 = get_output(y0)

            simple_forward(
                exe1, v_point_cloud, v_rotation_pos, v_translation, v_pose_src, ctx=ctx
            )
            y2 = exe1.outputs[0].asnumpy()
            sum_y2 = get_output(y2)

            # forward check
            for iter in range(batch_size):
                pc = np.squeeze(v_point_cloud[iter])
                rot_q_delta = v_rotation_pos[iter]
                trans_delta = v_translation[iter]
                pose_src = v_pose_src[iter]
                rot_m_delta = quat2mat(rot_q_delta)

                pose_est = RT_transform(
                    pose_src, rot_q_delta, trans_delta, T_means, T_stds, rot_coord
                )
                rot_m_tgt = pose_est[:3, :3]
                trans_tgt = pose_est[:, 3]

                pc_gt = np.matmul(rot_m_tgt, pc) + trans_tgt.reshape((3, 1))

                pc_est = y2[iter]
                assert (np.abs(pc_est - pc_gt).max(1) < 1e-4).all()

            print("rot forward check succeed")
            grad_gt = (sum_y2 - sum_y0) / (2 * stride)
            abs_diff = grad_est - grad_gt
            rel_diff = np.abs(grad_est - grad_gt) / max(np.abs(grad_gt), 1e-15)
            if abs_diff < thresh or rel_diff < thresh:
                print("rotation backward succeed")
            else:
                print(
                    "sample_point: {}, grad_est: {}, grad_gt: {}, abs_diff: {}, rel_diff: {}, v_rotation: {}".format(
                        (i, j),
                        grad_est,
                        grad_gt,
                        grad_est - grad_gt,
                        np.abs(grad_est - grad_gt) / max(np.abs(grad_gt), 1e-15),
                        v_rotation[i],
                    )
                )

    print("v_rotation backward check pass")
