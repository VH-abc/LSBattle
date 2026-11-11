# coding: utf8
# entity/glare.py
"""
Glare post-effect modelled on the eye's point-spread function.

Passes (no framebuffer objects, so it runs on the legacy GL 2.1 context):
  1. copy the rendered scene into texture A;
  2. clear the colour buffer and re-draw only the bright emitters (stars),
     in grey scaled by their real brightness, against the existing depth
     buffer so occlusion is respected; copy that into texture B with
     mipmaps.  A white planet surface is as bright as the Sun once clamped
     to the display, so this is the only honest way to pick glare sources.
     A body that would saturate the buffer, or that is smaller than a pixel,
     is drawn enlarged with its radiance scaled down by the same area
     (Star.draw_glare), so a source's glare depends on its total light, not
     on how many pixels it covers or whether it clips;
  3. draw texture A back to restore the scene;
  4. add the glare: texture B sampled on rings of growing radius around each
     pixel, each ring from the mip level that matches its radius, with equal
     energy per logarithmic ring.  A point source then spreads as 1/theta^2,
     the measured wing of the human glare PSF, plus a faint six-ray starburst
     (ciliary corona) along the same rings.
"""
from math import cos, sin, pi

from OpenGL.GL import *

from program.box import BOX
from program import script
from program.utils import compile_program
from program.utils.playing import FlatScreen


VS = """
#version 120
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""

FS_LINEAR = """
#version 120
uniform sampler2D src;
uniform vec2 size;
void main() {
    vec3 c = texture2D(src, gl_FragCoord.xy / size).rgb;
    // display units -> linear radiance (relative to the Sun), capped
    gl_FragColor = vec4(min(pow(max(c, vec3(0.0)), vec3(2.2)), vec3(6.0)), 1.0);
}
"""

FS_COPY = """
#version 120
uniform sampler2D scene;
uniform vec2 size;
void main() {
    gl_FragColor = vec4(texture2D(scene, gl_FragCoord.xy / size).rgb, 1.0);
}
"""

N_RINGS = 12
N_ANG = 16

FS_GLARE = """
#version 120
uniform sampler2D bright;
uniform vec2 size;
uniform float strength;
uniform float r0;         // innermost ring radius (pixels)
uniform float q;          // ring radius ratio
uniform vec2 dirs[%d];
void main() {
    vec2 uv = gl_FragCoord.xy / size;
    vec3 halo = vec3(0.0);
    vec3 star = vec3(0.0);
    float r = r0;
    for (int k = 0; k < %d; k++) {
        float lod = max(log2(r * 0.4), 0.0);
        vec3 ring = vec3(0.0);
        for (int a = 0; a < %d; a++) {
            vec2 o = dirs[a] * r / size;
            ring += texture2D(bright, uv + o, lod).rgb;
        }
        halo += ring;
        // six rays (every third direction, both signs are in dirs)
        for (int a = 0; a < %d; a += 3) {
            vec2 o = dirs[a] * r / size;
            star += texture2D(bright, uv + o, max(lod - 1.0, 0.0)).rgb;
        }
        r *= q;
    }
    // each ring is one annulus of the PSF with equal energy per log-radius,
    // so rings add up rather than average out
    halo *= strength / float(%d);
    star *= strength * 0.25 / float(%d / 3);
    // slight warm tint in the far wings, as in real ocular scatter
    vec3 g = halo + star;
    g *= vec3(1.0, 0.93, 0.85);
    gl_FragColor = vec4(g, 1.0);
}
""" % (N_ANG, N_RINGS, N_ANG, N_ANG, N_ANG, N_ANG)


class Glare(object):

    def __init__(self):
        self.ready = False
        self.debug = False        # tests: record the total linear source energy of the last frame
        self.source_energy = 0.0

    def _init_gl(self):
        self.p_copy = compile_program(VS, FS_COPY)
        self.p_linear = compile_program(VS, FS_LINEAR)
        self.texC = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texC)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self.p_glare = compile_program(VS, FS_GLARE)
        self.texA = glGenTextures(1)
        self.texB = glGenTextures(1)
        for tex, mip in ((self.texA, False), (self.texB, True)):
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR_MIPMAP_LINEAR if mip else GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            if mip:
                glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(self.p_glare)
        dirs = []
        for a in range(N_ANG):
            ang = 2.0 * pi * a / N_ANG + 0.2
            dirs += [cos(ang), sin(ang)]
        glUniform2fv(glGetUniformLocation(self.p_glare, "dirs"), N_ANG, (GLfloat * len(dirs))(*dirs))
        glUseProgram(0)
        self.ready = True

    def _quad(self, W, H):
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0)
        glVertex2f(W, 0.0)
        glVertex2f(W, H)
        glVertex2f(0.0, H)
        glEnd()

    def apply(self, draw_sources, strength=1.0):
        """draw_sources(): draws the glare sources with the current 3-D
        matrices; called with the colour buffer cleared and depth writes off."""
        if not self.ready:
            self._init_gl()
        W, H = BOX.X, BOX.Y
        fmt = GL_RGBA16F if BOX.HDR else GL_RGBA8
        glActiveTexture(GL_TEXTURE0)
        # 1. scene -> A
        glBindTexture(GL_TEXTURE_2D, self.texA)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, fmt, 0, 0, W, H, 0)
        # 2. bright emitters only -> framebuffer -> B (mipmapped)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glClearColor(*script.ui.backimage.color)
        glDepthMask(GL_FALSE)
        draw_sources()
        glDepthMask(GL_TRUE)
        glBindTexture(GL_TEXTURE_2D, self.texC)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, fmt, 0, 0, W, H, 0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        FlatScreen.push()
        glColor(1.0, 1.0, 1.0, 1.0)
        # linearise (C -> framebuffer) and copy into B with mipmaps
        glUseProgram(self.p_linear)
        glUniform1i(glGetUniformLocation(self.p_linear, "src"), 0)
        glUniform2f(glGetUniformLocation(self.p_linear, "size"), float(W), float(H))
        self._quad(W, H)
        glBindTexture(GL_TEXTURE_2D, self.texB)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 0, 0, W, H, 0)
        if self.debug:
            import array as _array
            def _sum(tex):
                glBindTexture(GL_TEXTURE_2D, tex)
                a = _array.array('f', bytes(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT)))
                return float(sum(a)), sum(1 for v in a if v > 0.0)
            self.source_energy, self.source_pixels = _sum(self.texB)
            self.main_sum, self.main_pixels = _sum(self.texC)
            glBindTexture(GL_TEXTURE_2D, self.texB)
        # 3. restore the scene from A
        glBindTexture(GL_TEXTURE_2D, self.texA)
        glUseProgram(self.p_copy)
        glUniform1i(glGetUniformLocation(self.p_copy, "scene"), 0)
        glUniform2f(glGetUniformLocation(self.p_copy, "size"), float(W), float(H))
        self._quad(W, H)
        # 4. add the glare from B
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        glBindTexture(GL_TEXTURE_2D, self.texB)
        glUseProgram(self.p_glare)
        glUniform1i(glGetUniformLocation(self.p_glare, "bright"), 0)
        glUniform2f(glGetUniformLocation(self.p_glare, "size"), float(W), float(H))
        glUniform1f(glGetUniformLocation(self.p_glare, "strength"), strength)
        glUniform1f(glGetUniformLocation(self.p_glare, "r0"), 2.0)
        glUniform1f(glGetUniformLocation(self.p_glare, "q"), 1.55)
        self._quad(W, H)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        FlatScreen.pop()
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
