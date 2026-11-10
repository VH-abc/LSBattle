# coding: utf8
# entity/wireframe.py
from OpenGL.GL import *

from go import Vector3, Vector4D, Matrix44, Lorentz
from program.utils import compile_program, boost
from program import script


class WireFrame(object):
    vertices = None
    def __init__(self, scale):
        self.L = script.world.wireframe.range * scale
        self.N = script.world.wireframe.div
        self.color = script.world.wireframe.color
        self.make_lattice(self.N, self.L)
        self.make_program()

    @classmethod
    def make_lattice(cls, N, L):
        if cls.vertices is not None:
            return
        def add(xx, yy, zz, a):
            vertices.extend([xx, yy, zz])
            if a < N*c:
                indices.append(len(vertices)//3-1)
                indices.append(len(vertices)//3)
        c = script.world.wireframe.inner_div
        vertices = []
        indices = []
        n = 2*N + 1
        for i in range(-N, N+1):
            xx = (i+0.5)*L/N
            for j in range(-N, N+1):
                yy = (j+0.5)*L/N
                for k in range(-N*c, N*c+1):
                    zz = (k+0.5*c)*L/(N*c)
                    add(xx, yy, zz, k)
        for i in range(-N, N+1):
            xx = (i+0.5)*L/N
            for j in range(-N, N+1):
                zz = (j+0.5)*L/N
                for k in range(-N*c, N*c+1):
                    yy = (k+0.5*c)*L/(N*c)
                    add(xx, yy, zz, k)
        for i in range(-N, N+1):
            zz = (i+0.5)*L/N
            for j in range(-N, N+1):
                yy = (j+0.5)*L/N
                for k in range(-N*c, N*c+1):
                    xx = (k+0.5*c)*L/(N*c)
                    add(xx, yy, zz, k)
        cls.vertices = (GLfloat*len(vertices))(*vertices)
        cls.indices = (GLint*len(indices))(*indices)
        cls.n = len(indices)
        cls.lattice_unit = L / N

    def make_program(self):
        self.program_id = compile_program(
            """
            #version 120
            uniform vec3 dXo;      // lattice origin minus player position (double on CPU)
            """ + boost.GLSL + """
            void main() {
                vec3 v = gl_Vertex.xyz + dXo;
                float r = length(v);
                vec3 d = v / r;
                vec4 vertex = lz_pos(vec4(v, -r), tmv_null(d, r, pb_n), pb_n, pb_u, pb_inv);
                vertex.w = 1.0;
                gl_Position = gl_ModelViewProjectionMatrix * vertex;
                float factor = max(0.0,
                                   min(1.0, 200.0/(gl_Position.w*gl_Position.w))
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
        self.dXo_local  = glGetUniformLocation(self.program_id, "dXo")
    
    def draw(self, Xp, L, U, color=None):
        glLineWidth(script.world.wireframe.line_width)
        glUseProgram(self.program_id)
        if color is None:
            r, g, b, a = self.color
        else:
            r, g, b, a = color
        glColor(r, g, b, min(a,10*a/L.get_gamma()))
        xp, yp, zp = Xp.get_lis3()
        xo = int(xp/self.lattice_unit) * self.lattice_unit
        yo = int(yp/self.lattice_unit) * self.lattice_unit
        zo = int(zp/self.lattice_unit) * self.lattice_unit
        glUniform3f(self.dXo_local, xo - xp, yo - yp, zo - zp)
        boost.set_uniforms(self.program_id, U)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glDrawElements(GL_LINES, self.n, GL_UNSIGNED_INT, self.indices)
        glUseProgram(0)
        glLineWidth(1)



