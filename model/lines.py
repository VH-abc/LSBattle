# -*- coding: utf8 -*-
# lines.py
import os

from OpenGL.GL import *

from program.utils import compile_program, boost


class Lines(object):
    def load_program(self):
        self.program_id = compile_program(
        """
        #version 120
        """ + boost.GLSL + """
        void main() {
            vec3 v = gl_Vertex.xyz;      // already relative to the player
            float r = length(v);
            vec3 d = v / r;
            vec4 vertex = lz_pos(vec4(v, -r), tmv_null(d, r, pb_n), pb_n, pb_u, pb_inv);
            vertex.w = 1.0;
            gl_Position = gl_ModelViewProjectionMatrix * vertex;
            float factor = max(1.0/5.0,
                                   min(1.0, 10.0/(gl_Position.w*gl_Position.w))
                                   );
            vec4 color = gl_Color;
            color.a *= factor;
            gl_FrontColor = color;
        }
        """,
        """
        #version 120
        void main() {
            gl_FragColor = gl_Color;
        }
        """)

    def __init__(self, color):
        self.color = color
        self.load_program()

    def draw(self, Xp, U, vertices_1, vertices_2, color=None):
        glUseProgram(self.program_id)
        if color is None:
            glColor(*self.color)
        else:
            glColor(*color)
        boost.set_uniforms(self.program_id, U)
        xp, yp, zp = Xp.x, Xp.y, Xp.z
        glBegin(GL_LINES)
        for i in range(len(vertices_1)//3):
            glVertex(vertices_1[i*3]-xp, vertices_1[i*3+1]-yp, vertices_1[i*3+2]-zp)
            glVertex(vertices_2[i*3]-xp, vertices_2[i*3+1]-yp, vertices_2[i*3+2]-zp)
        glEnd()
        # vertices = vertices_1 + vertices_2
        # a = len(vertices_1) / 3
        # indices = []
        # for i in xrange(a):
        #     indices += [i, i+a]
        # n = len(vertices)
        # glVertexPointer(3, GL_FLOAT, 0, (GLfloat*n)(*vertices))
        # glDrawElements(GL_LINES, n, GL_UNSIGNED_BYTE, (GLint*len(indices))(*indices))
        glUseProgram(0)
