# coding: utf8
# entity/glare.py
"""
Glare post-effect modelled on the eye's point-spread function.

Passes (one framebuffer object, via the EXT extension the legacy GL 2.1
context provides, for the only stage that needs more than 8 bits):
  1. copy the rendered scene into texture A;
  2. clear the colour buffer and re-draw only the bright emitters (stars),
     in grey scaled by their real brightness, against the existing depth
     buffer so occlusion is respected; copy that into texture B with
     mipmaps.  A white planet surface is as bright as the Sun once clamped
     to the display, so this is the only honest way to pick glare sources.
     The bodies are drawn again at 1/16, 1/256, 1/4096 and 1/65536 of the
     exposure; where a pixel saturates in one pass the next one supplies
     its radiance, so a hot star or a blueshifted Sun keeps its true
     brightness up to ~60,000x the Sun's surface (the sky is only in the
     first pass and stays capped).  A
     body smaller than a few pixels is drawn enlarged with its radiance
     scaled down by the same area (Star.draw_glare).  Either way a source's
     glare depends on its total light, not on how many pixels it covers;
     The linearised radiance is rendered straight into the float texture B
     through a framebuffer object: the window's own buffer is 8-bit in SDR
     and would clip everything brighter than display white, which is the
     whole point of the ladder;
  3. draw texture A back to restore the scene;
  4. add the glare: texture B sampled on rings of growing radius around each
     pixel, each ring from the mip level that matches its radius, with equal
     energy per logarithmic ring.  A point source then spreads as 1/theta^2,
     the measured wing of the human glare PSF, plus a faint six-ray starburst
     (ciliary corona) along the same rings.
"""
from math import cos, sin, pi, tan, radians

from OpenGL.GL import *
from program.const import VIEW_ANGLE
from OpenGL.GL.EXT.framebuffer_object import (glGenFramebuffersEXT, glBindFramebufferEXT, glFramebufferTexture2DEXT,
                                              glCheckFramebufferStatusEXT, glGenerateMipmapEXT,
                                              GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_FRAMEBUFFER_COMPLETE_EXT)


def _avail(f):
    try:
        return bool(f)
    except Exception:
        return False


def _fbo_api():
    """The framebuffer-object entry points this context provides: the core
    (GL 3.0) names, else the EXT ones, else None.  Looked up once a
    context exists, since the pointers are resolved per context."""
    try:
        from OpenGL.GL import (glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
                               glCheckFramebufferStatus, GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_COMPLETE)
        if _avail(glGenFramebuffers) and _avail(glBindFramebuffer):
            return (glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D, glCheckFramebufferStatus,
                    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_COMPLETE)
    except Exception:
        pass
    if _avail(glGenFramebuffersEXT) and _avail(glBindFramebufferEXT):
        return (glGenFramebuffersEXT, glBindFramebufferEXT, glFramebufferTexture2DEXT, glCheckFramebufferStatusEXT,
                GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_FRAMEBUFFER_COMPLETE_EXT)
    return None


def _mipmap_func():
    """glGenerateMipmap (core), else the EXT one, else None (then the
    GL_GENERATE_MIPMAP texture flag makes glCopyTexImage2D build the chain)."""
    try:
        from OpenGL.GL import glGenerateMipmap
        if _avail(glGenerateMipmap):
            return glGenerateMipmap
    except Exception:
        pass
    if _avail(glGenerateMipmapEXT):
        return glGenerateMipmapEXT
    return None

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

EXPOSURES = (1.0, 1.0/16.0, 1.0/256.0, 1.0/4096.0, 1.0/65536.0)

def _fs_linear():
    """Linearise the exposure ladder: each pixel's radiance comes from the
    first pass that does not saturate; the sky is only in pass 0.  The body
    shaders decode their (scaled) input and encode their output with a 2.2
    power, so a pass at scale k holds (lin * k^2.2)^(1/2.2) and its gain is
    k^-2.2."""
    n = len(EXPOSURES)
    lines = ["#version 120"]
    lines += ["uniform sampler2D src%d;" % i for i in range(n)]
    lines += ["uniform vec2 size;", "void main() {", "    vec2 uv = gl_FragCoord.xy / size;"]
    for i, k in enumerate(EXPOSURES):
        lines.append("    vec3 c%d = texture2D(src%d, uv).rgb;" % (i, i))
        lines.append("    vec3 lin%d = pow(max(c%d, vec3(0.0)), vec3(2.2)) * %.6g;" % (i, i, (1.0 / k) ** 2.2))
        lines.append("    float m%d = max(max(c%d.r, c%d.g), c%d.b);" % (i, i, i, i))
    # linear radiance relative to the Sun's surface; pass 0 (with the sky) capped
    lines.append("    vec3 outc = min(lin0, vec3(6.0));")
    chain = "lin%d" % (n - 1)
    for i in range(n - 2, 0, -1):
        chain = "((m%d > 0.9) ? %s : lin%d)" % (i, chain, i)
    lines.append("    if (m0 > 0.9) outc = max(outc, min(%s, vec3(1.0e7)));   // 0 where only the sky saturates" % chain)
    lines += ["    gl_FragColor = vec4(outc, 1.0);", "}"]
    return "\n".join(lines)

FS_LINEAR = _fs_linear()

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
uniform sampler2D scene;     // the frame, display-encoded (2.2 power)
uniform vec2 size;
uniform float strength;
uniform float r0;         // innermost ring radius (pixels)
uniform float q;          // ring radius ratio
uniform float rmax;       // rings beyond this radius (pixels) are left out: a limit on the spread
uniform vec2 dirs[%d];
void main() {
    vec2 uv = gl_FragCoord.xy / size;
    vec3 halo = vec3(0.0);
    vec3 star = vec3(0.0);
    float r = r0;
    for (int k = 0; k < %d; k++) {
        if (r > rmax) break;
        float lod = max(log2(r * 0.7), 0.0);     // blur wide enough that the sample directions do not show
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
    // add in linear light: decode the frame, add, encode again
    vec3 lin = pow(max(texture2D(scene, uv).rgb, vec3(0.0)), vec3(2.2));
    gl_FragColor = vec4(pow(lin + g, vec3(1.0/2.2)), 1.0);
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
        self.texC = list(glGenTextures(len(EXPOSURES)))
        for tex in self.texC:
            glBindTexture(GL_TEXTURE_2D, tex)
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
        glBindTexture(GL_TEXTURE_2D, 0)
        # float render target for the linearised radiance
        self.fbo = None
        self.fbo_size = None
        self.fbo_api = _fbo_api()
        self.gen_mipmap = _mipmap_func()
        if self.gen_mipmap is None:
            glBindTexture(GL_TEXTURE_2D, self.texB)
            glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
            glBindTexture(GL_TEXTURE_2D, 0)
        if self.fbo_api is not None and self.gen_mipmap is not None:
            try:
                self.fbo = self.fbo_api[0](1)
            except Exception:
                self.fbo = None
        glUseProgram(self.p_glare)
        dirs = []
        for a in range(N_ANG):
            ang = 2.0 * pi * a / N_ANG + 0.2
            dirs += [cos(ang), sin(ang)]
        glUniform2fv(glGetUniformLocation(self.p_glare, "dirs"), N_ANG, (GLfloat * len(dirs))(*dirs))
        glUseProgram(0)
        self.ready = True

    def _bind_fbo(self, W, H):
        """Render into texture B (RGBA16F) through the framebuffer object;
        returns False when that is unavailable."""
        if self.fbo is None:
            return False
        gen, bind, attach, status, FB, ATT0, COMPLETE = self.fbo_api
        glBindTexture(GL_TEXTURE_2D, self.texB)
        if self.fbo_size != (W, H):
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, W, H, 0, GL_RGBA, GL_FLOAT, None)
            self.gen_mipmap(GL_TEXTURE_2D)        # allocate the chain
            self.fbo_size = (W, H)
        bind(FB, self.fbo)
        attach(FB, ATT0, GL_TEXTURE_2D, self.texB, 0)
        if status(FB) != COMPLETE:
            bind(FB, 0)
            self.fbo = None
            return False
        glBindTexture(GL_TEXTURE_2D, self.texC[0])
        return True

    def _quad(self, W, H):
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0)
        glVertex2f(W, 0.0)
        glVertex2f(W, H)
        glVertex2f(0.0, H)
        glEnd()

    def apply(self, draw_sources, strength=1.0):
        """draw_sources(k): draws the glare sources at brightness scale k
        (the sky only for k == 1) with the current 3-D matrices; called with
        the colour buffer cleared and depth writes off."""
        if not self.ready:
            self._init_gl()
        W, H = BOX.X, BOX.Y
        fmt = GL_RGBA16F if BOX.HDR else GL_RGBA8
        glActiveTexture(GL_TEXTURE0)
        # 1. scene -> A
        glBindTexture(GL_TEXTURE_2D, self.texA)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, fmt, 0, 0, W, H, 0)
        # 2. bright emitters only, at each exposure -> framebuffer -> C[i]
        for i, k in enumerate(EXPOSURES):
            glActiveTexture(GL_TEXTURE0)          # the bodies bind their textures on unit 0
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glClearColor(*script.ui.backimage.color)
            glDepthMask(GL_FALSE)
            draw_sources(k)
            glDepthMask(GL_TRUE)
            if self.debug:
                import array as _array
                a = _array.array('f', bytes(glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT)))
                self.fb_sums = getattr(self, "fb_sums", {}); self.fb_sums[i] = (float(sum(a)), sum(1 for v in a if v > 0.0), max(a))
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.texC[i])
            glCopyTexImage2D(GL_TEXTURE_2D, 0, fmt, 0, 0, W, H, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texC[0])
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        FlatScreen.push()
        glColor(1.0, 1.0, 1.0, 1.0)
        # linearise (C -> framebuffer) and copy into B with mipmaps
        glUseProgram(self.p_linear)
        for i in range(len(EXPOSURES)):
            loc = glGetUniformLocation(self.p_linear, "src%d" % i)
            if self.debug:
                self.src_locs = getattr(self, "src_locs", {}); self.src_locs[i] = loc
            glUniform1i(loc, i)
        glUniform2f(glGetUniformLocation(self.p_linear, "size"), float(W), float(H))
        use_fbo = self._bind_fbo(W, H)
        self._quad(W, H)
        for i in range(1, len(EXPOSURES)):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texB)
        if use_fbo:
            self.fbo_api[1](self.fbo_api[4], 0)
            self.gen_mipmap(GL_TEXTURE_2D)
        else:
            # no float target: the window's buffer clips at display white
            glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 0, 0, W, H, 0)
            if self.gen_mipmap is not None:
                self.gen_mipmap(GL_TEXTURE_2D)
        if self.debug:
            import array as _array
            def _sum(tex):
                glBindTexture(GL_TEXTURE_2D, tex)
                a = _array.array('f', bytes(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT)))
                return float(sum(a)), sum(1 for v in a if v > 0.0)
            self.source_energy, self.source_pixels = _sum(self.texB)
            self.main_sum, self.main_pixels = _sum(self.texC[0])
            self.pass_sums = [_sum(t) for t in self.texC]
            glBindTexture(GL_TEXTURE_2D, self.texB)
        # 3. the scene from A plus the glare from B, summed in linear light
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.texA)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texB)
        glUseProgram(self.p_glare)
        glUniform1i(glGetUniformLocation(self.p_glare, "bright"), 0)
        glUniform1i(glGetUniformLocation(self.p_glare, "scene"), 1)
        glUniform2f(glGetUniformLocation(self.p_glare, "size"), float(W), float(H))
        glUniform1f(glGetUniformLocation(self.p_glare, "strength"), strength)
        glUniform1f(glGetUniformLocation(self.p_glare, "r0"), 2.0)
        glUniform1f(glGetUniformLocation(self.p_glare, "q"), 1.55)
        # spread limit: degrees of view -> pixels (the ring radius is the
        # distance light is thrown, before the ring's own blur)
        deg = BOX.GLARE_SPREAD
        if deg > 0.0:
            px_per_deg = 0.5*H / tan(0.5*radians(VIEW_ANGLE)) * (pi/180.0)
            rmax = deg * px_per_deg
        else:
            rmax = 1.0e9
        glUniform1f(glGetUniformLocation(self.p_glare, "rmax"), rmax)
        self._quad(W, H)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        FlatScreen.pop()
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
