# -*- coding: utf8 -*-
# polygon.py
import os

from OpenGL.GL import *

from go import Vector4D, Matrix44, Lorentz
from model.mqo.mqo2gpo import mqo2gpo
from program.utils import load_texture, search_imagefile, compile_program
from program.utils import blackbody, boost
from program.utils.lensing import Lensing, GLSL as LENS_GLSL
from program.box import BOX


class Material(object):

    def __init__(self, color, tex_name=None, texture=True, path=""):
        self.color = color
        if tex_name and texture:
            #tex_name = search_imagefile(self.tex_name)
            tex_name = os.path.join(path, tex_name)
            self.tex_name = tex_name
            self.texture_id = load_texture(tex_name)
        else:
            self.tex_name = None
            self.texture_id = None

    def reload_texture(self):
        if self.tex_name is not None:
            self.texture_id = load_texture(self.tex_name)

class MqoGpoPolygon(object):

    def load_gpo(self, name, func=lambda x,y,z:(x,y,z), texture=True):
        igpo = open(name)
        if next(igpo).strip() != "Game Polygon Object":
            raise IOError("%s is not gpo file"%name)
        path, name = os.path.split(name)
        self.path = path
        vertices = []
        texcoords = []
        objects = []
        for line in igpo:
            words = line.split()
            if not words:continue
            if words[0] == "p":
                x, y, z, u, v = list(map(float, words[1:]))
                vertices.extend(func(x, y, z))
                texcoords.extend([u, v])
            elif words[0] == "m":
                color = list(map(float, words[1:-1]))
                tex_name = words[-1][1:-1]
                objects.append([Material(color, tex_name, texture, path), []])
            elif words[0] == "i":
                objects[-1][1] = list(map(int, words[1:]))
                
        self.vertices = (GLfloat*len(vertices))(*vertices)
        self.texcoords = (GLfloat*len(texcoords))(*texcoords)
        
        self.objects = []
        for material, indices in objects:
            n = len(indices)
            obj = [material, n, (GLuint*n)(*indices)]
            self.objects.append(obj)
    
    def set_texture(self, tex_name):
        def _set(self, i, name):
            if os.path.isfile(os.path.join(self.path, name)):
                name = os.path.join(self.path, name)
                self.objects[i][0].texture_id = load_texture(name)
                self.objects[i][0].tex_name = name
            else:
                name = search_imagefile(name)
                self.objects[i][0].texture_id = load_texture(name)
                self.objects[i][0].tex_name = name
        if isinstance(tex_name, list) or isinstance(tex_name, tuple):
            for i, name in enumerate(tex_name):
                _set(self, i, name)
        else:
            _set(self, 0, tex_name)
            
    def make_gpo(self, name, ext=""):
        if ext.lower() in [".mqo", ".gpo"]:
            if os.path.isfile(name+ext):
                mqo2gpo(name+ext)
                return name + ".gpo"
            elif os.path.isdir(name):
                name = os.path.join(name, os.path.split(name)[1])
                return self.make_gpo(name, ext)
        elif ext == "":
            if os.path.isfile(name+".gpo"):
                if os.path.isfile(name+".mqo"):
                    if os.path.getmtime(name+".mqo") > os.path.getmtime(name+".gpo"):
                        mqo2gpo(name+".mqo")
                return name + ".gpo"
            elif os.path.isfile(name+".mqo"):
                mqo2gpo(name+".mqo")
                return name + ".gpo"
            elif os.path.isdir(name):
                name = os.path.join(name, os.path.split(name)[1])
                return self.make_gpo(name, ext)
            else:
                raise IOError("%s's model file is not exsit"%(os.path.split(name)[1]))
        raise IOError("%s's model file is not exsit"%(os.path.split(name)[1]+ext))

vs_T = """
#version 120
uniform vec4 ucolor;     // brightness/tint (fragment stage: gl_FrontColor is clamped)
uniform mat4 Rotate;     // emitter body -> emitter rest frame (incl. Wigner)
uniform vec3 xp_perp;    // player relative to emitter centre, emitter frame:
uniform float xp_par;    //   components perpendicular / parallel to rb_n
uniform vec3 rb_n; uniform float rb_u, rb_g, rb_inv;   // emitter U in player frame
uniform vec3 light_dir;  // unit vector body centre -> illuminating star, emitter frame
uniform float light_on;  // 0: self-luminous, 1: sphere (Lambert), 2: thin disc (both faces)
uniform float g_shift;   // gravitational redshift factor of the emitter (1 = none)
uniform float disc_Rs;   // >0: accretion disc; mesh units per Schwarzschild radius
uniform float disc_spin; // +1 / -1: sense of rotation about the mesh +y axis
uniform float lens_w;    // scale of the bending (1: a black hole's exact table; 0: none)
""" + boost.GLSL + LENS_GLSL + """
varying float ratio;
varying float shade;
void main() {
    // Each vertex is treated as its own light ray: the past-directed null
    // vector from the vertex to the player is formed in the emitter frame,
    // split along the boost axis so the (small) transverse part never
    // suffers cancellation, then boosted with the stable t+vpar form.
    // Boosting centre + offset separately loses ~gamma^2 digits once a
    // body behind the player wraps around the sky.
    vec3 xi = (Rotate * gl_Vertex).xyz;          // vertex rel. centre, emitter frame
    if (light_on > 1.5) {
        shade = abs(dot(normalize((Rotate * vec4(0.0, 1.0, 0.0, 0.0)).xyz), light_dir));
    } else if (light_on > 0.5) {
        shade = max(dot(normalize(xi), light_dir), 0.0);
    } else {
        shade = 1.0;
    }
    float xi_par = dot(xi, rb_n);
    vec3  v_perp = (xi - xi_par*rb_n) - xp_perp; // v = xi - xp
    float v_par  = xi_par - xp_par;
    float p2  = dot(v_perp, v_perp);
    float rho = sqrt(v_par*v_par + p2);          // emission distance, emitter frame
    // k = (v, -rho) -> player frame = Lorentz(-U_rel): tpv = t + vpar = -(rho - v_par)
    float tpv = (v_par > 0.0) ? -p2/(rho + v_par) : -(rho - v_par);
    float t1    = -rb_inv*rho   + rb_u*tpv;
    float vpar1 =  rb_inv*v_par + rb_u*tpv;
    float dv = 1.0;
    if (disc_Rs > 0.0) {
        // Keplerian gas: each vertex is its own emitter moving in the hole's
        // frame; a steadily rotating axisymmetric disc has the same shape at
        // every retarded time, so only the Doppler factor needs the velocity
        vec3 vm = gl_Vertex.xyz;
        float rr = max(length(vm.xz) / disc_Rs, 1.5);
        float sp = sqrt(0.5 / rr);
        vec3 tang = normalize((Rotate * vec4(-vm.z, 0.0, vm.x, 0.0)).xyz) * disc_spin;
        vec3 to_player = normalize(-(v_perp + v_par*rb_n));
        dv = sqrt(1.0 - sp*sp) / (1.0 - sp * dot(tang, to_player));
    }
    ratio = g_shift * dv * t1 / (-rho);
    vec3 P = v_perp + vpar1*rb_n;
    if (lens_w > 0.0 && lens_k > 0.0) {
        // beyond a lens: the vertex is seen where its primary image is
        float dist = length(P);
        // each vertex sees the fraction (d - D)/d of the deflection for its
        // own distance d beyond the lens at D: a disc's far side bends up
        // over the hole, its near side and the hole itself stay put
        if (dist > 0.0) {
            vec3 dl = lens_to_lab(P / dist);
            vec3 img = vec3(0.0);
            if (lens_finite > 0.5 && lens_w > 0.0) {
                // a source near the hole: its image from the real light path
                // between its true position and the ship (primary, or the
                // secondary round the far side on the second pass)
                img = lens_apparent_near(dl * dist, lens_branch);
            }
            if (dot(img, img) < 0.5) {
                if (lens_branch > 0.5 && lens_finite > 0.5) {
                    // no secondary image for this vertex: drop its triangles
                    P = vec3(0.0 / 0.0);
                    img = dl;
                } else {
                    // beyond the near table: a far source, the 1-D tables
                    float wv = lens_w * max(0.0, (dist - lens_D) / dist);
                    if (wv > 0.0) img = lens_apparent(dl, wv); else img = dl;
                }
            }
            if (P == P) P = lens_to_ship(img) * dist;
        }
    }
    gl_Position = gl_ModelViewProjectionMatrix * vec4(P, 1.0);
    %s
}
"""
fs_T = """
#version 120
%s
varying float ratio;
varying float shade;
void main() {
    %s
    color.rgb *= shade;
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
    gl_FragColor = color / 255.0;
}
"""
fs_A = """
#version 120
%s
varying float ratio;
varying float shade;
""" + blackbody.GLSL + """
void main() {
    %s
    color.rgb *= pow(shade, 1.0 / 2.2);     // shade is linear; colour is sRGB-encoded here
    if (bb_logT0 > 0.0 && bb_lit > 0.5) {
        // lit surface: albedo under a Planck illuminant, shifted as a spectrum
        gl_FragColor = vec4(doppler_spectral(color.rgb, ratio, exposure), color.a);
    } else {
        // emitter (or unknown temperature): the texel is the appearance at rest,
        // moved along the Planck locus; the spectral round trip is not exact
        // for very hot stars and left a magenta cast
        gl_FragColor = vec4(doppler_accurate(color.rgb, ratio, exposure), color.a);
    }
}
"""
def _fs():
    return fs_A if BOX.COLOR_MODE == "ACCURATE" else fs_T

_polygon_cache = {}
class Polygon(MqoGpoPolygon):

    def load_program_tex(self):
        self.program_tex = compile_program(
            vs_T%("""
                gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
                gl_FrontColor = vec4(1.0);
                """),
            _fs()%("uniform sampler2D texture;\nuniform vec4 ucolor;",
                  "vec4 color = texture2D(texture, gl_TexCoord[0].xy) * ucolor;")
        )
        self.accurate = (BOX.COLOR_MODE == "ACCURATE")
        self.temperature = 0.0
        self.light = None          # None or ((x, y, z) unit dir to the light, mode 1|2)
        self.light_scale = 1.0     # irradiance factor applied to the colour
        self.grav_shift = 1.0      # gravitational redshift of the emitter
        self.disc = None           # (mesh units per R_s, spin) for accretion discs
        self.t_disc_Rs = glGetUniformLocation(self.program_tex, "disc_Rs")
        self.t_disc_spin = glGetUniformLocation(self.program_tex, "disc_spin")
        self.t_gshift = glGetUniformLocation(self.program_tex, "g_shift")
        self.t_light_dir = glGetUniformLocation(self.program_tex, "light_dir")
        self.t_light_on = glGetUniformLocation(self.program_tex, "light_on")
        self.t_color_local = glGetUniformLocation(self.program_tex, "ucolor")
        self.t_R_local = glGetUniformLocation(self.program_tex, "Rotate")
        self.t_xpp_local = glGetUniformLocation(self.program_tex, "xp_perp")
        self.t_xpl_local = glGetUniformLocation(self.program_tex, "xp_par")
        self.tex_local = glGetUniformLocation(self.program_tex, "texture")
    
    def load_program_col(self):
        self.program_col = compile_program(
            vs_T%("gl_FrontColor = gl_Color;"),
            _fs()%("uniform vec4 ucolor;", "vec4 color = gl_Color * ucolor;")
        )
        self.c_disc_Rs = glGetUniformLocation(self.program_col, "disc_Rs")
        self.c_disc_spin = glGetUniformLocation(self.program_col, "disc_spin")
        self.c_gshift = glGetUniformLocation(self.program_col, "g_shift")
        self.c_light_dir = glGetUniformLocation(self.program_col, "light_dir")
        self.c_light_on = glGetUniformLocation(self.program_col, "light_on")
        self.c_color_local = glGetUniformLocation(self.program_col, "ucolor")
        self.c_R_local = glGetUniformLocation(self.program_col, "Rotate")
        self.c_xpp_local = glGetUniformLocation(self.program_col, "xp_perp")
        self.c_xpl_local = glGetUniformLocation(self.program_col, "xp_par")

    def __init__(self, name, func=lambda x,y,z:(x,y,z),
                 texture=True, color=(1.0,1.0,1.0,1.0)):
        self.load_program_col()
        self.load_program_tex()
        name, ext = os.path.splitext(name)
        if name in _polygon_cache:
            poly = _polygon_cache[name]
            if func(1.0, 1.0, 1.0) == poly.func(1.0, 1.0, 1.0):
                self.vertices = poly.vertices
                self.texcoords = poly.texcoords
                self.objects = poly.objects
                self.color = color
                self.texture = texture
                self.temperature = 0.0
                self.light = None
                self.light_scale = 1.0
                self.grav_shift = 1.0
                self.disc = None
                self.path = poly.path
                for obj in self.objects:
                    obj[0].reload_texture()
                return
        file_name = self.make_gpo(name, ext)
        self.load_gpo(file_name, func, texture)
        self.color = color
        self.func = func
        self.texture = texture
        _polygon_cache[name] = self

    def draw(self, Xp, L, LL, X=Vector4D(0,0,0,0), U=Vector4D(1,0,0,0), R=Matrix44(), scale=1.0, gshift=1.0):
        """
        Xp: player's X in background frame
        L: background to player frame
        LL: player to background frame
        U: enemy's U in background frame
        X: enemy's X in background frame
        R: enemy's rotation matrix
        """
        # dX: X-Xp in background frame, on the past light cone
        dX = X - Xp
        dX.t = -dX.length()
        # xp: Xp-X in the emitter frame
        xp = Lorentz(U).get_transform(-dX)
        # emitter 4-velocity in the player frame
        Urel = L.get_transform(U)
        if U.squared_length() > 0.0:
            # L*Lorentz(-U) = Lorentz(-Urel)*Rw : fold the Wigner rotation into
            # the body rotation so the shader only needs a pure boost
            Rw = Lorentz(Urel) * L * Lorentz(-U)
            R = Rw * R
            xp = Rw.get_transform(xp)
        # split xp along the boost axis (double precision) for the shader
        n = boost.params(Urel)[0]
        xp_par = xp.x*n[0] + xp.y*n[1] + xp.z*n[2]
        xp_perp = [xp.x - xp_par*n[0], xp.y - xp_par*n[1], xp.z - xp_par*n[2]]
        R = R.to_glsl()
        c = self.color
        ls = scale * (self.light_scale if self.light is not None else 1.0)
        color = (c[0]*ls, c[1]*ls, c[2]*ls, c[3])
        if self.light is not None:
            (lx, ly, lz), lmode = self.light
        else:
            lx = ly = lz = 0.0; lmode = 0
        dRs, dspin = self.disc if self.disc is not None else (0.0, 1.0)
        lens_w = 0.0 if getattr(self, "is_lens", False) else Lensing.wscale()
        lens_wbody = Lensing.weight(-dX.t)        # the body's own fraction (its centre)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glTexCoordPointer(2, GL_FLOAT, 0, self.texcoords)
        program = 0
        for material, n, indices in self.objects:
            if material.texture_id is None:
                glColor(*material.color)
                if self.program_col != program:
                    glUseProgram(self.program_col)
                    boost.set_uniforms(self.program_col, Urel, "rb")
                    Lensing.set_uniforms(self.program_col, lens_w, table_unit=4)
                    glUniformMatrix4fv(self.c_R_local, 1, GL_FALSE, R)
                    glUniform3fv(self.c_xpp_local, 1, xp_perp)
                    glUniform1f(self.c_xpl_local, xp_par)
                    glUniform4fv(self.c_color_local, 1, color)
                    glUniform1f(self.c_gshift, self.grav_shift * gshift)
                    glUniform1f(self.c_disc_Rs, dRs); glUniform1f(self.c_disc_spin, dspin)
                    glUniform3f(self.c_light_dir, lx, ly, lz)
                    glUniform1f(self.c_light_on, float(lmode))
                    if self.accurate:
                        blackbody.set_uniforms(self.program_col, BOX.exposure(), self.temperature, lmode > 0)
                    program = self.program_col
            else:
                glBindTexture(GL_TEXTURE_2D, material.texture_id)
                if self.program_tex != program:
                    glUseProgram(self.program_tex)
                    boost.set_uniforms(self.program_tex, Urel, "rb")
                    Lensing.set_uniforms(self.program_tex, lens_w, table_unit=4)
                    glUniformMatrix4fv(self.t_R_local, 1, GL_FALSE, R)
                    glUniform3fv(self.t_xpp_local, 1, xp_perp)
                    glUniform1f(self.t_xpl_local, xp_par)
                    glUniform4fv(self.t_color_local, 1, color)
                    glUniform1i(self.tex_local, 0)
                    glUniform1f(self.t_gshift, self.grav_shift * gshift)
                    glUniform1f(self.t_disc_Rs, dRs); glUniform1f(self.t_disc_spin, dspin)
                    glUniform3f(self.t_light_dir, lx, ly, lz)
                    glUniform1f(self.t_light_on, float(lmode))
                    if self.accurate:
                        blackbody.set_uniforms(self.program_tex, BOX.exposure(), self.temperature, lmode > 0)
                    program = self.program_tex
            glDrawElements(GL_TRIANGLES, n, GL_UNSIGNED_INT, indices)
            if (lens_wbody > 0.9 or Lensing.near_table()) and Lensing.secondary(lens_w) and not getattr(self, "is_lens", False):
                # a black hole's secondary image of the body, round its far side
                loc = glGetUniformLocation(program, "lens_branch")
                glUniform1f(loc, 1.0)
                glDrawElements(GL_TRIANGLES, n, GL_UNSIGNED_INT, indices)
                glUniform1f(loc, 0.0)
        glUseProgram(0)
