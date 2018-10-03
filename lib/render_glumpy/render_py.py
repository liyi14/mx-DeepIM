# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from glumpy import app, gl, gloo, glm, data, log
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
    def __init__(self, model_folder, K, width=640, height=480, zNear=0.25, zFar=6.0):
        self.width = width
        self.height = height
        self.zNear = zNear
        self.zFar = zFar
        self.K = K
        self.model_folder = model_folder

        self.rgb_buffer = np.zeros((self.height, self.width, 4), dtype=np.float32)
        self.depth_buffer = np.zeros((self.height, self.width), dtype=np.float32)

        log.info("Loading brain mesh")
        vertices, indices = data.objload("{}/textured.obj"
                                         .format(model_folder), rescale=False)
        self.render_kernel = gloo.Program(self.vertex, self.fragment)
        self.render_kernel.bind(vertices)
        log.info("Loading brain texture")
        self.render_kernel['u_texture'] = np.copy(data.load("{}/texture_map.png"
                                                            .format(model_folder))[::-1, :, :])

        self.render_kernel['u_model'] = np.eye(4, dtype=np.float32)
        u_projection = self.my_compute_calib_proj(K, width, height, zNear, zFar)
        self.render_kernel['u_projection'] = np.copy(u_projection)

        self.window = app.Window(width=width, height=height, visible=False)
        print("self.window: ", self.window)
        print("self.render_kernel at init: ", self.render_kernel)
        @self.window.event
        def on_draw(dt):
            self.window.clear()
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            self.render_kernel.draw(gl.GL_TRIANGLES)
            gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, self.rgb_buffer)
            gl.glReadPixels(0, 0, self.width, self.height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, self.depth_buffer)

        @self.window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)

    def render(self, r, t, r_type='quat', K=None):
        if r_type == 'quat':
            R = quat2mat(r)
        elif r_type == 'mat':
            R = r
        self.render_kernel['u_view'] = self._get_view_mtx(R, t)
        if K is not None:
            u_projection = self.my_compute_calib_proj(K, self.width, self.height, self.zNear, self.zFar)
            self.render_kernel['u_projection'] = np.copy(u_projection)
        app.run(framecount=0, framerate=0)

        rgb_gl = np.flipud(self.rgb_buffer)
        depth_gl = np.flipud(self.depth_buffer)

        rgb_gl = rgb_gl[:, :, [2, 1, 0]]
        rgb_gl *= 255

        depth_bg = depth_gl == 1
        depth_gl = 2*self.zFar*self.zNear / (self.zFar+self.zNear-(self.zFar-self.zNear)*(2*depth_gl-1))
        depth_gl[depth_bg] = 0  # Convert to [0, 255]
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
