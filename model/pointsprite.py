# -*- coding: utf8 -*-
# pointsplite.py
from OpenGL.GL import *

from program.utils import compile_program, load_texture, DY_TEXTURE_KYU
from program.utils import boost
from go import Vector4D


_vshader = """
#version 120
%s float size;
""" + boost.GLSL + """
void main() {
    vec3 v = gl_Vertex.xyz;          // relative to the player (double on CPU)
    float rr = length(v);
    vec3 d = v / rr;
    vec4 vertex = lz_pos(vec4(v, -rr), tmv_null(d, rr, pb_n), pb_n, pb_u, pb_inv);
    float r = -vertex.w * 0.2;
    %svertex /= r;
    vertex.w = 1.0;
    gl_Position = gl_ModelViewProjectionMatrix * vertex;
    %s;
    vec4 color = gl_Color;
    %scolor.a = min(color.a, color.a/pow(r,2)*100);    
    gl_FrontColor = color;
}
"""
_fshader = """
#version 120
uniform sampler2D texture;
void main() {
    vec4 color = gl_Color * texture2D(texture, gl_PointCoord.xy);
    if(color.a == 0.0) discard;
    gl_FragColor = color;
}
"""

def _relative(vertices, Xp):
    xp, yp, zp = Xp.x, Xp.y, Xp.z
    out = []
    for i in range(len(vertices)//3):
        out += [vertices[i*3]-xp, vertices[i*3+1]-yp, vertices[i*3+2]-zp]
    return out

class PointSprite(object):

    def __init__(self, vertices=None, size=None, color=None, size_w=True, texture=DY_TEXTURE_KYU):
        if vertices is not None:
            n = len(vertices)
            self.vertices = (GLfloat*n)(*vertices)
            self.n = n / 3
        if size is None:
            size_type = "attribute"
            size_local_geter = glGetAttribLocation
        else:
            self.size = size
            size_type = "uniform"
            size_local_geter = glGetUniformLocation
        self._draw = getattr(self, "_draw_%i_%i"%(1 if vertices is None else 0,
                                                  1 if size is None else 0)
                            )

        self.color = color

        if size_w:
            psize = "gl_PointSize = size / gl_Position.w"
            color_decay = ""
            normalize_to_look = "//"
        else:
            psize = "gl_PointSize = size"
            color_decay = "//"
            normalize_to_look = ""
        self.program_id = compile_program(
            _vshader%(size_type,
                normalize_to_look, 
                psize, 
                color_decay),
            _fshader)
        self.size_local = size_local_geter(self.program_id, "size")
        self.tex_local  = glGetUniformLocation(self.program_id, "texture")

        self.texture_id = load_texture(texture)

    def draw(self, Xp, L, vertices=None, size=None, color=None):
        glUseProgram(self.program_id)
        if color is None:
            glColor(*self.color)
        else:
            glColor(*color)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(self.tex_local, 0)
        boost.set_uniforms(self.program_id, boost.U_of_L(L))
        if vertices is not None:
            vertices = _relative(vertices, Xp)
        self._draw(vertices, size)
        glUseProgram(0)

    def _draw_0_0(self, vertices, size):
        if size is None:
            glUniform1f(self.size_local, self.size)
        else:
            glUniform1f(self.size_local, size)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glDrawArrays(GL_POINTS, 0, self.n)
    def _draw_0_1(self, vertices, sizes):
        glEnableVertexAttribArray(self.size_local)
        glVertexAttribPointer(self.size_local, 1, GL_FLOAT, GL_FALSE, 0, (GLfloat*self.n)(*sizes))
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glDrawArrays(GL_POINTS, 0, self.n)
        glDisableVertexAttribArray(self.size_local)
    def _draw_1_0(self, vertices, size):
        if size is None:
            glUniform1f(self.size_local, self.size)
        else:
            glUniform1f(self.size_local, size)
        n = len(vertices)
        glVertexPointer(3, GL_FLOAT, 0, (GLfloat*n)(*vertices))
        glDrawArrays(GL_POINTS, 0, n//3)
    def _draw_1_1(self, vertices, sizes):
        n = len(sizes)
        glEnableVertexAttribArray(self.size_local)
        glVertexAttribPointer(self.size_local, 1, GL_FLOAT, GL_FALSE, 0, (GLfloat*n)(*sizes))
        glVertexPointer(3, GL_FLOAT, 0, (GLfloat*(n*3))(*vertices))
        glDrawArrays(GL_POINTS, 0, n)
        glDisableVertexAttribArray(self.size_local)

_vshader_doppler = """
#version 120
%s float size;
attribute vec4 UU;   // emitter 4-velocity in the player's frame (x,y,z,t)
""" + boost.GLSL + """
void main() {
    vec3 v = gl_Vertex.xyz;          // relative to the player (double on CPU)
    float rr = length(v);
    vec3 d = v / rr;
    vec4 vertex = lz_pos(vec4(v, -rr), tmv_null(d, rr, pb_n), pb_n, pb_u, pb_inv);
    // emitter frame: boost by its 4-velocity relative to the player
    float ue = length(UU.xyz);
    float ge = sqrt(1.0 + ue*ue);
    vec3 ne = ue > 0.0 ? UU.xyz / ue : vec3(0.0, 0.0, 1.0);
    float inve = 1.0 / (ge + ue);
    float re = -vertex.w;
    vec3 de = vertex.xyz / re;
    vec4 vertex_e = lz_pos(vertex, tmv_null(de, re, ne), ne, ue, inve);
    float ratio = vertex.w / vertex_e.w;
    float r = -vertex.w * 0.2;
    %svertex /= r;
    vertex.w = 1.0;
    gl_Position = gl_ModelViewProjectionMatrix * vertex;
    %s;
    vec4 color = gl_Color;
    %scolor.a = min(color.a, color.a/pow(r,2)*100);
    color *= 255.0;
    float l = max(max(color.r, color.g), color.b);
    float Tr, Tg, Tb;
    if (l > 1) {
        color *= 255.0 / l;
        Tb = 1000.0 + 904.495*exp(0.00721929 * color.b);
        Tr = 6000.0 + 8.01879e20 * pow(max(color.r, 1.0), -7.507239275877164);
        if (color.b > 254.0) {
            Tg = 6502.86;
        } else {
            if (color.b / max(color.r, 1.0) > 0.98084) {
                Tg = 505.192 * exp(0.0100532 * color.g);
            } else {
                Tg = 6000.0 + 3.55446e34 * pow(max(color.g, 1), -13.24242861627803);
            }
        }
        Tr *= ratio;
        Tg *= ratio;
        Tb *= ratio;
        if (1905.0 > Tb) {
            color.b = 0.0;
        } else if (6700.0 < Tb) {
            color.b = 255.0;
        } else {
            color.b = -305.045 + 138.518 * log(0.01 * Tb - 10.0);
        }
        if (6689.0 > Tr) {
            color.r = 255.0;
        } else {
            color.r = 608.873 * pow(Tr - 6000.0, -0.133205);
        }
        if (506.0 > Tg) {
            color.g = 0.0;
        } else if (6502.86 > Tg) {
            color.g = -619.2 + 99.4708 * log(Tg);
        } else {
            color.g = 406.534 * pow(Tg - 6000.0, -0.0755148);
        }
        color *= l / 255.0;
    }
    gl_FrontColor = color / 255.0;
}
"""
class PointSpriteDoppler(object):

    def __init__(self, vertices=None, size=None, color=None, size_w=True, texture=DY_TEXTURE_KYU):
        if vertices is not None:
            n = len(vertices)
            self.vertices = (GLfloat*n)(*vertices)
            self.n = n / 3
        if size is None:
            size_type = "attribute"
            size_local_geter = glGetAttribLocation
        else:
            self.size = size
            size_type = "uniform"
            size_local_geter = glGetUniformLocation
        self._draw = getattr(self, "_draw_%i_%i"%(1 if vertices is None else 0,
                                                  1 if size is None else 0)
                            )

        self.color = color

        if size_w:
            psize = "gl_PointSize = size / gl_Position.w"
            color_decay = ""
            normalize_to_look = "//"
        else:
            psize = "gl_PointSize = size"
            color_decay = "//"
            normalize_to_look = ""
        self.program_id = compile_program(
            _vshader_doppler%(size_type,
                normalize_to_look, 
                psize, 
                color_decay),
            _fshader)
        self.size_local = size_local_geter(self.program_id, "size")
        self.U_local  = glGetAttribLocation(self.program_id, "UU")
        self.tex_local  = glGetUniformLocation(self.program_id, "texture")

        self.texture_id = load_texture(texture)

    def draw(self, Xp, L, vertices=None, U=None, size=None, color=None):
        glUseProgram(self.program_id)
        if color is None:
            glColor(*self.color)
        else:
            glColor(*color)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(self.tex_local, 0)
        boost.set_uniforms(self.program_id, boost.U_of_L(L))
        if vertices is not None:
            vertices = _relative(vertices, Xp)
        glEnableVertexAttribArray(self.U_local)
        if U:
            Urel = []
            for i in range(len(U)//4):
                w = L.get_transform([U[i*4+3], U[i*4], U[i*4+1], U[i*4+2]])
                Urel += [w.x, w.y, w.z, w.t]
            glVertexAttribPointer(self.U_local, 4, GL_FLOAT, GL_FALSE, 0, (GLfloat*len(Urel))(*Urel))
        else:
            glVertexAttribPointer(self.U_local, 4, GL_FLOAT, GL_FALSE, 0,
                (GLfloat*(self.n*4))(*([0.0,0.0,0.0,1.0]*self.n)))
        self._draw(vertices, size)
        glUseProgram(0)

    def _draw_0_0(self, vertices, size):
        if size is None:
            glUniform1f(self.size_local, self.size)
        else:
            glUniform1f(self.size_local, size)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glDrawArrays(GL_POINTS, 0, self.n)
    def _draw_0_1(self, vertices, sizes):
        glEnableVertexAttribArray(self.size_local)
        glVertexAttribPointer(self.size_local, 1, GL_FLOAT, GL_FALSE, 0, (GLfloat*self.n)(*sizes))
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glDrawArrays(GL_POINTS, 0, self.n)
        glDisableVertexAttribArray(self.size_local)
    def _draw_1_0(self, vertices, size):
        if size is None:
            glUniform1f(self.size_local, self.size)
        else:
            glUniform1f(self.size_local, size)
        n = len(vertices)
        glVertexPointer(3, GL_FLOAT, 0, (GLfloat*n)(*vertices))
        glDrawArrays(GL_POINTS, 0, n//3)
    def _draw_1_1(self, vertices, sizes):
        n = len(sizes)
        glEnableVertexAttribArray(self.size_local)
        glVertexAttribPointer(self.size_local, 1, GL_FLOAT, GL_FALSE, 0, (GLfloat*n)(*sizes))
        glVertexPointer(3, GL_FLOAT, 0, (GLfloat*(n*3))(*vertices))
        glDrawArrays(GL_POINTS, 0, n)
        glDisableVertexAttribArray(self.size_local)

