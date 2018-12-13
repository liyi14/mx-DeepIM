# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from glumpy import app, gl, gloo, data, log
import logging
import os

log.setLevel(logging.ERROR)  # ERROR, WARNING, DEBUG, INFO
from lib.pair_matching.RT_transform import quat2mat


class Render_Py():
    vertex = """
    uniform mat4   u_model;         // Model matrix
    uniform mat4   u_view;          // View matrix
    uniform mat4   u_projection;    // Projection matrix
    attribute vec3 position;
    attribute vec2 texcoord;
    varying vec2   v_texcoord;

    void main()
    {
        // Assign varying variables
        v_texcoord = texcoord;

        // Final position
        gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
    }
    """

    fragment = """
    uniform sampler2D u_texture;  // Texture
    varying vec2      v_texcoord; // Interpolated fragment texture coordinates (in)

    void main()
    {
        // Get texture color
        vec4 t_color = texture2D(u_texture, v_texcoord);

        // Final color
        gl_FragColor = t_color;
    }
    """

    def __init__(self,
                 model_dir,
                 classes,
                 K,
                 width=640,
                 height=480,
                 zNear=0.25,
                 zFar=6.0):
        self.width = width
        self.height = height
        self.zNear = zNear
        self.zFar = zFar
        self.K = K
        self.model_dir = model_dir

        self.rgb_buffer = np.zeros((self.height, self.width, 4),
                                   dtype=np.float32)
        self.depth_buffer = np.zeros((self.height, self.width),
                                     dtype=np.float32)

        log.info("Loading mesh")
        self.render_kernel_list = [[] for cls in classes]
        self.classes = classes
        self.cls_idx = 0
        for class_id, cur_class in enumerate(classes):
            model_folder = os.path.join(model_dir, cur_class)
            print("Loading {}".format(model_folder))
            vertices, indices = data.objload(
                "{}/textured.obj".format(model_folder), rescale=False)
            render_kernel = gloo.Program(self.vertex, self.fragment)
            render_kernel.bind(vertices)
            log.info("Loading texture")
            render_kernel['u_texture'] = np.copy(
                data.load(
                    "{}/texture_map.png".format(model_folder))[::-1, :, :])

            render_kernel['u_model'] = np.eye(4, dtype=np.float32)
            u_projection = self.my_compute_calib_proj(K, width, height, zNear,
                                                      zFar)
            render_kernel['u_projection'] = np.copy(u_projection)
            self.render_kernel_list[class_id] = render_kernel
        print('************Finish loading models*************')

        self.window = app.Window(width=width, height=height, visible=False)
        print("self.window: ", self.window)
        print("self.render_kernel at init: ", self.render_kernel_list)

        @self.window.event
        def on_draw(dt):
            self.window.clear()
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            self.render_kernel_list[self.cls_idx].draw(gl.GL_TRIANGLES)
            gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA,
                            gl.GL_FLOAT, self.rgb_buffer)
            gl.glReadPixels(0, 0, self.width, self.height,
                            gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT,
                            self.depth_buffer)

        @self.window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)

    def render(self, cls_idx, r, t, r_type='quat', K=None):
        if r_type == 'quat':
            R = quat2mat(r)
        elif r_type == 'mat':
            R = r

        self.cls_idx = cls_idx
        self.render_kernel_list[cls_idx]['u_view'] = self._get_view_mtx(R, t)

        if K is not None:
            u_projection = self.my_compute_calib_proj(
                K, self.width, self.height, self.zNear, self.zFar)
            self.render_kernel_list[cls_idx]['u_projection'] = np.copy(
                u_projection)

        # import time
        # t = time.time()
        app.run(framecount=0, framerate=0)
        # print("render {} seconds/image".format(time.time()-t))
        # app.run()

        rgb_gl = np.flipud(self.rgb_buffer)
        depth_gl = np.flipud(self.depth_buffer)

        bgr_gl = rgb_gl[:, :, [2, 1, 0]]  # convert to BGR format as cv2
        bgr_gl *= 255

        depth_bg = depth_gl == 1
        depth_gl = 2 * self.zFar * self.zNear / (self.zFar + self.zNear -
                                                 (self.zFar - self.zNear) *
                                                 (2 * depth_gl - 1))
        depth_gl[depth_bg] = 0
        return bgr_gl, depth_gl

    def __del__(self):
        self.window.close()

    def my_compute_calib_proj(self, K, w, h, zNear, zFar):
        u0 = K[0, 2] + 0.5
        v0 = K[1, 2] + 0.5
        fu = K[0, 0]
        fv = K[1, 1]
        L = +(u0) * zNear / -fu
        T = +(v0) * zNear / fv
        R = -(w - u0) * zNear / -fu
        B = -(h - v0) * zNear / fv
        proj = np.zeros((4, 4))
        proj[0, 0] = 2 * zNear / (R - L)
        proj[1, 1] = 2 * zNear / (T - B)
        proj[2, 2] = -(zFar + zNear) / (zFar - zNear)
        proj[2, 0] = (R + L) / (R - L)
        proj[2, 1] = (T + B) / (T - B)
        proj[2, 3] = -1.0
        proj[3, 2] = -(2 * zFar * zNear) / (zFar - zNear)
        return proj

    def _get_view_mtx(self, R, t):
        u_view = np.eye(4, dtype=np.float32)
        u_view[:3, :3], u_view[:3, 3] = R, t.squeeze()
        yz_flip = np.eye(4, dtype=np.float32)
        yz_flip[1, 1], yz_flip[2, 2] = -1, -1
        u_view = yz_flip.dot(u_view)  # OpenCV to OpenGL camera system
        u_view = u_view.T  # OpenGL expects column-wise matrix format
        return u_view


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    def mat2quat(M):
        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
        # Fill only lower half of symmetric matrix
        K = np.array(
            [[Qxx - Qyy - Qzz, 0, 0, 0], [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
             [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
             [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K)
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q = vecs[[3, 0, 1, 2], np.argmax(vals)]
        if q[0] < 0:
            q *= -1
        return q

    cur_dir = os.path.abspath(os.path.dirname(__file__))

    classes = ['driller']  # '002_master_chef_can'
    model_dir = os.path.join(
        cur_dir, '../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/models/')
    pose_path = os.path.join(
        cur_dir,
        '../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/data/gt_observed/{}/{}-pose.txt'
    )
    color_path = os.path.join(
        cur_dir,
        '../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/data/gt_observed/{}/{}-color.png'
    )
    depth_path = os.path.join(
        cur_dir,
        '../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/data/gt_observed/{}/{}-depth.png'
    )
    K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899],
                  [0.0, 0.0, 1.0]])

    width = 640
    height = 480
    ZNEAR = 0.25
    ZFAR = 6.0
    img_idx_list = ['000001', '000001']

    render_machine = Render_Py(model_dir, classes, K, width, height, ZNEAR,
                               ZFAR)
    for idx in range(len(classes)):
        # warm up
        bgr_gl, _ = render_machine.render(idx, (0.5, 0.5, 0.5, 0.5),
                                          np.array([0, 0, 1]))
        bgr_gl = bgr_gl.astype(np.uint8)
        fig = plt.figure()
        plt.axis('off')
        fig.add_subplot(2, 3, 1)
        plt.imshow(bgr_gl[:, :, [2, 1, 0]])
        plt.show()

        cur_class = classes[idx]
        cur_img_idx = img_idx_list[idx]
        pose_real_est = np.loadtxt(
            pose_path.format(cur_class, cur_img_idx), skiprows=1)
        r_quat = mat2quat(pose_real_est[:, :3])
        t = pose_real_est[:, 3]
        import time
        start_t = time.time()
        bgr_gl, depth_gl = render_machine.render(idx, r_quat, t)
        print("using {} seconds".format(time.time() - start_t))
        bgr_gl = bgr_gl.astype(np.uint8)
        c = np.dot(K, t)
        c_x = int(round(c[0] / c[2]))
        c_y = int(round(c[1] / c[2]))
        bgr_gl[c_y, c_x, :] = np.array([255, 0, 0])

        bgr_pa = cv2.imread(
            color_path.format(cur_class, cur_img_idx), cv2.IMREAD_COLOR)
        depth_pa = cv2.imread(
            depth_path.format(cur_class, cur_img_idx),
            cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        fig = plt.figure()
        plt.axis('off')
        fig.add_subplot(2, 3, 1)
        plt.imshow(bgr_gl[:, :, [2, 1, 0]])
        fig.add_subplot(2, 3, 2)
        plt.imshow(bgr_pa[:, :, [2, 1, 0]])
        fig.add_subplot(2, 3, 3)
        plt.imshow(bgr_gl - bgr_pa)
        fig.add_subplot(2, 3, 4)
        plt.imshow(depth_gl)
        fig.add_subplot(2, 3, 5)
        plt.imshow(depth_pa)
        fig.add_subplot(2, 3, 6)
        plt.imshow(depth_gl - depth_pa)
        plt.show()
