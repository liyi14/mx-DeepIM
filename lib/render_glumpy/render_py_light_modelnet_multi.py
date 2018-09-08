# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from glumpy import app, gl, gloo, glm, data, log
import logging

log.setLevel(logging.WARNING)  # ERROR, WARNING, DEBUG, INFO
from lib.pair_matching.RT_transform import quat2mat

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 position;
attribute vec3 normal;
attribute vec2 texcoord;
varying vec3   v_normal;
varying vec3   v_position;
varying vec2   v_texcoord;
void main()
{
    // Assign varying variables
    v_normal   = normal;
    v_position = position;
    v_texcoord = texcoord;
    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
}
"""


def get_fragment(brightness_ratio=0.4):
    fragment = """
    uniform mat4      u_model;           // Model matrix
    uniform mat4      u_view;            // View matrix
    uniform mat4      u_normal;          // Normal matrix
    uniform mat4      u_projection;      // Projection matrix
    uniform sampler2D u_texture;         // Texture 
    uniform vec3      u_light_position;  // Light position
    uniform vec3      u_light_intensity; // Light intensity

    varying vec3      v_normal;          // Interpolated normal (in)
    varying vec3      v_position;        // Interpolated position (in)
    varying vec2      v_texcoord;        // Interpolated fragment texture coordinates (in)
    void main()
    {{
        // Calculate normal in world coordinates
        vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

        // Calculate the location of this fragment (pixel) in world coordinates
        vec3 position = vec3(u_view*u_model * vec4(v_position, 1));
        //vec3 light_position = vec3(u_view*u_model * vec4(u_light_position, 1));
        // Calculate the vector from this pixels surface to the light source
        vec3 surfaceToLight = u_light_position - position;

        // Calculate the cosine of the angle of incidence (brightness)
        float brightness = dot(normal, surfaceToLight) /
                          (length(surfaceToLight) * length(normal));
        brightness = max(min(brightness,1.0),0.0);

        // Calculate final color of the pixel, based on:
        // 1. The angle of incidence: brightness
        // 2. The color/intensities of the light: light.intensities
        // 3. The texture and texture coord: texture(tex, fragTexCoord)

        // Get texture color
        vec4 t_color = vec4(texture2D(u_texture, v_texcoord).rgb, 1.0);

        // Final color
        gl_FragColor = t_color * (({} + {}*brightness) * vec4(u_light_intensity, 1));
    }}
    """.format(1 - brightness_ratio, brightness_ratio)
    return fragment


class Render_Py_Light_ModelNet_Multi():
    def __init__(self, model_path_list, texture_path, K, width=640, height=480, zNear=0.25, zFar=6.0, brightness_ratios=[0.7]):
        self.width = width
        self.height = height
        self.zNear = zNear
        self.zFar = zFar
        self.K = K
        self.model_path_list = model_path_list
        self.model_path = model_path_list[0]
        self.render_kernels = {model_path: [] for model_path in model_path_list}

        log.info("Loading mesh")
        for model_i, model_path in enumerate(model_path_list):
            print('loading model: {}/{}, {}'.format(model_i+1, len(model_path_list), model_path))
            vertices, indices = data.objload("{}"
                                             .format(model_path), rescale=True)
            vertices['position'] = vertices['position'] / 10.


            for brightness_ratio in brightness_ratios:
                fragment = get_fragment(brightness_ratio=brightness_ratio)
                render_kernel = gloo.Program(vertex, fragment)
                render_kernel.bind(vertices)

                log.info("Loading brain texture")
                render_kernel['u_texture'] = np.copy(data.load("{}"
                                                               .format(texture_path))[::-1, :, :])

                render_kernel['u_model'] = np.eye(4, dtype=np.float32)
                u_projection = self.my_compute_calib_proj(K, width, height, zNear, zFar)
                render_kernel['u_projection'] = np.copy(u_projection)

                render_kernel['u_light_intensity'] = 1, 1, 1
                self.render_kernels[model_path].append(render_kernel)
        print('************Finish loading models in Render_Py_Light_ModelNet_Multi********************')
        self.brightness_k = 0  # init

        self.window = app.Window(width=width, height=height, visible=False)

        @self.window.event
        def on_draw(dt):
            self.window.clear()
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            self.render_kernels[self.model_path][self.brightness_k].draw(gl.GL_TRIANGLES)

        @self.window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)

    def render(self, model_idx, r, t, light_position, light_intensity, brightness_k=0, r_type='quat'):
        '''
        :param r: 
        :param t: 
        :param light_position: 
        :param light_intensity: 
        :param brightness_k: choose which brightness in __init__ 
        :param r_type: 
        :return: 
        '''
        if r_type == 'quat':
            R = quat2mat(r)
        elif r_type == 'mat':
            R = r
        self.model_path =self.model_path_list[model_idx]
        self.brightness_k = brightness_k
        self.render_kernels[self.model_path][brightness_k]['u_view'] = self._get_view_mtx(R, t)
        self.render_kernels[self.model_path][brightness_k]['u_light_position'] = light_position
        self.render_kernels[self.model_path][brightness_k]['u_normal'] = np.array(
            np.matrix(np.dot(self.render_kernels[self.model_path][brightness_k]['u_view'].reshape(4, 4),
                             self.render_kernels[self.model_path][brightness_k]['u_model'].reshape(4, 4))).I.T)
        self.render_kernels[self.model_path][brightness_k]['u_light_intensity'] = light_intensity

        app.run(framecount=0)
        rgb_buffer = np.zeros((self.height, self.width, 4), dtype=np.float32)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, rgb_buffer)

        rgb_gl = np.copy(rgb_buffer)
        rgb_gl.shape = 480, 640, 4
        rgb_gl = rgb_gl[::-1, :]
        rgb_gl = np.round(rgb_gl[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255]
        rgb_gl = rgb_gl[:, :, [2, 1, 0]]

        depth_buffer = np.zeros((self.height, self.width), dtype=np.float32)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, depth_buffer)
        depth_gl = np.copy(depth_buffer)
        depth_gl.shape = 480, 640
        depth_gl = depth_gl[::-1, :]
        depth_bg = depth_gl == 1
        depth_gl = 2 * self.zFar * self.zNear / (self.zFar + self.zNear - (self.zFar - self.zNear) * (2 * depth_gl - 1))
        depth_gl[depth_bg] = 0
        return rgb_gl, depth_gl

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
    import os
    import sys
    from pprint import pprint

    cur_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(1, os.path.join(cur_path, '../..'))


    def mat2quat(M):
        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
        # Fill only lower half of symmetric matrix
        K = np.array([
            [Qxx - Qyy - Qzz, 0, 0, 0],
            [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
            [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
            [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]
        ) / 3.0
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

    model_dir = os.path.join(cur_path, '../../data/LOV/models/{}'.format(class_name))
    pose_path = os.path.join(cur_path, '../../data/render_v5/data/render_real/%s/0001/{}-pose.txt' % (class_name))
    color_path = os.path.join(cur_path, '../../data/render_v5/data/render_real/%s/0001/{}-color.png' % (class_name))
    width = 640
    height = 480
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    ZNEAR = 0.25
    ZFAR = 6.0
    idx = '000001'

    brightness_ratios = [0.3, 0.4]
    # light_position[0] += pose[0, 3]
    # light_position[1] -= pose[1, 3]
    # light_position[2] -= pose[2, 3]
    render_machine = Render_Py_Light(model_dir, K, width, height, ZNEAR, ZFAR, brightness_ratios=brightness_ratios)
    pose_real_est = np.loadtxt(pose_path.format(idx), skiprows=1)
    r_quat = mat2quat(pose_real_est[:, :3])
    t = pose_real_est[:, 3]
    # warm up
    rgb_gl, _ = render_machine.render((0.5, 0.5, 0.5, 0.5), t, light_position=[0, 0, 1], light_intensity=[1, 1, 1])
    import time

    start_t = time.time()
    rgb_gl_1, _ = render_machine.render(r_quat, t, light_position=[0, 0, -1], light_intensity=[1, 1, 1], brightness_k=0)
    rgb_gl_2, _ = render_machine.render(r_quat, t, light_position=[0, 0, -1],
                                        light_intensity=[1 * 1.5, 1 * 1.5, 1.2 * 1.5], brightness_k=1)
    rgb_gl_3, _ = render_machine.render(r_quat, t, light_position=[0, 0, -1], light_intensity=np.array([1, 1.2, 1]) * 2,
                                        brightness_k=0)
    rgb_gl_4, _ = render_machine.render(r_quat, t, light_position=[0, 0, -1], light_intensity=np.array([1.2, 1, 1]) * 3,
                                        brightness_k=1)
    rgb_gl_5, _ = render_machine.render(r_quat, t, light_position=[0, 0, -1], light_intensity=[1, 0, 1], brightness_k=0)
    rgb_gl_6, _ = render_machine.render(r_quat, t, light_position=[0, 0, -1], light_intensity=[0, 1, 1], brightness_k=0)
    rgb_gl_7, _ = render_machine.render(r_quat, t, light_position=[0, 0, -1], light_intensity=[10, 10, 10],
                                        brightness_k=0)
    print("using {} seconds".format(time.time() - start_t))
    rgb_pa = cv2.imread(color_path.format(idx),
                        cv2.IMREAD_COLOR)

    pprint(rgb_gl_1[rgb_gl_1.nonzero()])
    pprint(rgb_gl_2[rgb_gl_2.nonzero()])
    pprint(rgb_gl_3[rgb_gl_3.nonzero()])
    pprint(rgb_gl_4[rgb_gl_3.nonzero()])
    pprint(rgb_gl_5[rgb_gl_3.nonzero()])
    pprint(rgb_gl_6[rgb_gl_3.nonzero()])
    pprint(rgb_gl_7[rgb_gl_3.nonzero()])

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.axis('off')
    fig.add_subplot(2, 4, 1)
    plt.imshow(rgb_pa)
    fig.add_subplot(2, 4, 2)
    plt.imshow(rgb_gl_1)
    fig.add_subplot(2, 4, 3)
    plt.imshow(rgb_gl_2)
    fig.add_subplot(2, 4, 4)
    plt.imshow(rgb_gl_3)
    fig.add_subplot(2, 4, 5)
    plt.imshow(rgb_gl_4)
    fig.add_subplot(2, 4, 6)
    plt.imshow(rgb_gl_5)
    fig.add_subplot(2, 4, 7)
    plt.imshow(rgb_gl_6)
    fig.add_subplot(2, 4, 8)
    plt.imshow(rgb_gl_7)
    fig.suptitle('brightness_ratio: {}'.format(brightness_ratios[0]))
    # plt.imshow(rgb_gl - rgb_pa)

    plt.show()