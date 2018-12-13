# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from glumpy import app, gl, gloo, data, log
import logging

log.setLevel(logging.WARNING)  # ERROR, WARNING, DEBUG, INFO
from lib.pair_matching.RT_transform import quat2mat

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


class Render_Py_depth():
    def __init__(self,
                 model_folder,
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
        self.model_folder = model_folder

        log.info("Loading mesh")
        vertices, indices = data.objload(
            "{}/textured.obj".format(model_folder), rescale=False)
        self.render_kernel = gloo.Program(vertex, fragment)
        self.render_kernel.bind(vertices)
        log.info("Loading texture")
        self.render_kernel['u_texture'] = np.copy(
            data.load("{}/texture_map.png".format(model_folder))[::-1, :, :])

        self.render_kernel['u_model'] = np.eye(4, dtype=np.float32)
        u_projection = self.my_compute_calib_proj(K, width, height, zNear,
                                                  zFar)
        self.render_kernel['u_projection'] = np.copy(u_projection)

        self.window = app.Window(width=width, height=height, visible=False)

        @self.window.event
        def on_draw(dt):
            global trans
            self.window.clear()
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            self.render_kernel.draw(gl.GL_TRIANGLES)

        @self.window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)

    def render(self, r, t, r_type='quat'):
        if r_type == 'quat':
            R = quat2mat(r)
        elif r_type == 'mat':
            R = r
        self.render_kernel['u_view'] = self._get_view_mtx(R, t)
        app.run(framecount=0)
        rgb_buffer = np.zeros((self.height, self.width, 4), dtype=np.float32)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT,
                        rgb_buffer)

        rgb_gl = np.copy(rgb_buffer)
        rgb_gl.shape = 480, 640, 4
        rgb_gl = rgb_gl[::-1, :]
        rgb_gl = np.round(rgb_gl[:, :, :3] * 255).astype(
            np.uint8)  # Convert to [0, 255]
        bgr_gl = rgb_gl[:, :, [2, 1, 0]]

        depth_buffer = np.zeros((self.height, self.width), dtype=np.float32)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_DEPTH_COMPONENT,
                        gl.GL_FLOAT, depth_buffer)
        depth_gl = np.copy(depth_buffer)
        depth_gl.shape = 480, 640
        depth_gl = depth_gl[::-1, :]
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
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1
        return q

    class_name = '002_master_chef_can'
    model_dir = '/home/yili/PoseEst/mx-DeepPose/data/LOV/models/{}'.format(
        class_name)
    pose_path = '/home/yili/PoseEst/render/synthesize/train/%s/{}_pose.txt' % (
        class_name)
    color_path = '/home/yili/PoseEst/render/synthesize/train/%s/{}_color.png' % (
        class_name)
    depth_path = '/home/yili/PoseEst/render/synthesize/train/%s/{}_depth.png' % (
        class_name)
    width = 640
    height = 480
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    ZNEAR = 0.25
    ZFAR = 6.0
    idx = '000000'

    render_machine = Render_Py_depth(model_dir, K, width, height, ZNEAR, ZFAR)
    pose_real_est = np.loadtxt(pose_path.format(idx), skiprows=1)
    r_quat = mat2quat(pose_real_est[:, :3])
    t = pose_real_est[:, 3]
    # warm up
    rgb_gl, _ = render_machine.render((1 / 2, 1 / 2, 1 / 2, 1 / 2), t)
    import time

    start_t = time.time()
    rgb_gl, depth_gl = render_machine.render(r_quat, t)
    print(depth_gl.shape, np.max(depth_gl))
    print("using {} seconds".format(time.time() - start_t))

    rgb_pa = cv2.imread(
        '/home/yili/PoseEst/render/synthesize/train/002_master_chef_can/{}_color.png'
        .format(idx), cv2.IMREAD_COLOR)
    rgb_pa = cv2.imread(color_path.format(idx), cv2.IMREAD_COLOR)
    depth_pa = cv2.imread(depth_path.format(idx), cv2.IMREAD_UNCHANGED).astype(
        np.float32) / 10000.0
    print(depth_pa.shape, np.max(depth_pa))
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.axis('off')
    fig.add_subplot(2, 3, 1)
    plt.imshow(rgb_gl)
    fig.add_subplot(2, 3, 2)
    plt.imshow(rgb_pa)
    fig.add_subplot(2, 3, 3)
    plt.imshow(rgb_gl - rgb_pa)
    fig.add_subplot(2, 3, 4)
    plt.imshow(depth_gl)
    fig.add_subplot(2, 3, 5)
    plt.imshow(depth_pa)
    fig.add_subplot(2, 3, 6)
    plt.imshow(depth_gl - depth_pa)
    plt.show()
