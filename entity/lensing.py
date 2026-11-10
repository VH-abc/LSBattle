# coding: utf8
# entity/lensing.py
"""
Gravitational light bending near one compact body, as a screen-space pass.

The scene is rendered as usual, then redrawn through a lens: every pixel
looks up where its light really came from.  For a pixel at angle theta
from the body, the ray passed it at impact parameter b = D theta and was
bent by alpha = 2 R_s/b + (15 pi/16) (R_s/b)^2 (Schwarzschild, to second
order), so it came from angle beta = theta - alpha * (D_s - D)/D_s, where
D_s is the depth of what that pixel shows: the sky and far bodies bend
fully, things in front of the lens not at all.  Inside beta < 0 the sky
appears inverted: the inner image, the Einstein ring at beta = 0.  For a
black hole, rays with b < 3 sqrt(3)/2 R_s are captured: the shadow.

The bending happens in the background (static) frame, so each pixel's
direction is first taken from the moving ship's frame to the background
frame (inverse aberration), bent there, and brought back; at high Lorentz
factor the whole forward sky is squeezed into a narrow cone and doing the
geometry in the ship's frame would exaggerate the bending enormously.

Only done when 2 R_s / D exceeds a threshold, so nothing is spent on it in
an ordinary planetary system.
"""
from math import pi, tan, radians

from OpenGL.GL import *

from program.box import BOX
from program.const import VIEW_ANGLE
from program.utils import compile_program, boost
from program.utils.playing import FlatScreen


VS = """
#version 120
void main() { gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; }
"""

FS = """
#version 120
uniform sampler2D scene;
uniform sampler2D depth;
uniform vec2 size;
uniform vec3 hdir;       // direction of the lens, background frame, eye axes
uniform float D;         // its distance (the scene's length unit)
uniform float Rs;        // its Schwarzschild radius
uniform float shadow;    // 1: black hole (capture region drawn black)
uniform float tanh_;     // tan(fov_y / 2)
uniform float aspect;
uniform float near_, far_;
uniform vec3 n_;         // ship velocity direction, eye axes
uniform float u_, inv_;  // |gamma beta|, 1/(gamma+u)
// direction toward a source seen by the ship -> the same ray in the background frame
vec3 to_lab(vec3 v) {
    float c = dot(v, n_);
    float tpv = (c > 0.0) ? 0.5*dot(v - n_, v - n_) : 1.0 - c;     // t + vpar of the photon (-v, 1)
    float vpar = -c;
    vec3 perp = -v - vpar*n_;
    vec3 xyz = perp + (inv_*vpar + u_*tpv)*n_;
    float t = inv_ + u_*tpv;
    return normalize(-xyz / t);
}
// and back: a background-frame source direction as the ship sees it
vec3 to_ship(vec3 d) {
    float c = dot(d, n_);
    float tmv = (c < 0.0) ? 0.5*dot(d + n_, d + n_) : 1.0 + c;     // t - vpar of the photon (-d, 1)
    float vpar = -c;
    vec3 perp = -d - vpar*n_;
    vec3 xyz = perp + (inv_*vpar - u_*tmv)*n_;
    float t = inv_ + u_*tmv;
    return normalize(-xyz / t);
}
void main() {
    vec2 uv = gl_FragCoord.xy / size;
    vec3 v = normalize(vec3((2.0*uv.x - 1.0)*tanh_*aspect, (2.0*uv.y - 1.0)*tanh_, -1.0));
    vec3 d = to_lab(v);
    float ct = clamp(dot(d, hdir), -1.0, 1.0);
    float th = acos(ct);
    float k = Rs / D;
    // what this pixel shows: in front of the lens -> no bending, and no
    // shadow either (the near side of an accretion disc crosses the shadow)
    float zb = texture2D(depth, uv).r;
    float w = 1.0;
    bool behind = true;
    if (zb < 0.99999) {
        float zn = 2.0*zb - 1.0;
        float Ds = 2.0*far_*near_ / ((far_ + near_) - zn*(far_ - near_));
        w = clamp((Ds - D) / Ds, 0.0, 1.0);
        behind = Ds > D;
    }
    if (shadow > 0.5 && behind && th < 2.598*k) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec3 e = d - ct*hdir;
    float el = length(e);
    if (el < 1e-6 || w <= 0.0) {
        gl_FragColor = texture2D(scene, uv);
        return;
    }
    e /= el;
    float alpha = (2.0*k/th + 2.945*k*k/(th*th)) * w;
    float beta = th - alpha;
    vec3 ds = hdir*cos(beta) + e*sin(beta);
    vec3 vs = to_ship(ds);
    if (vs.z > -1e-4) {
        gl_FragColor = texture2D(scene, uv);           // bent in from behind the camera: not rendered, keep the pixel
        return;
    }
    vec2 s = vec2(vs.x / (-vs.z*tanh_*aspect), vs.y / (-vs.z*tanh_));
    vec2 uv2 = 0.5*(s + 1.0);
    if (uv2.x < 0.0 || uv2.x > 1.0 || uv2.y < 0.0 || uv2.y > 1.0) {
        gl_FragColor = texture2D(scene, uv);           // source outside the rendered frame
        return;
    }
    gl_FragColor = texture2D(scene, uv2);
}
"""


class Lensing(object):
    THRESHOLD = 2.5e-5      # 2 R_s / D: the Einstein radius is 0.3 degrees here

    def __init__(self):
        self.ready = False

    @classmethod
    def pick(cls, solar, Xp):
        """The body whose lensing is strongest at Xp, with its distance, or None."""
        best, bs, bd = None, cls.THRESHOLD, 0.0
        for star in solar.stars.values():
            if star.GM <= 0.0 or star.hp <= 0:
                continue
            d = (star.X - Xp).length()
            if d <= 0.0:
                continue
            s = 4.0*star.GM / d                    # 2 R_s / D
            if s > bs:
                best, bs, bd = star, s, d
        return (best, bd) if best is not None else None

    def _init_gl(self):
        self.program = compile_program(VS, FS)
        self.texA, self.texD = glGenTextures(2)
        for tex in (self.texA, self.texD):
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR if tex == self.texA else GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR if tex == self.texA else GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.ready = True

    def apply(self, hdir, D, Rs, is_hole, n_eye=(0.0, 0.0, 1.0), u=0.0, inv=1.0):
        """hdir: unit direction of the lens in the background frame, on the
        eye axes; n_eye, u, inv: the ship's boost (direction on the eye axes,
        |gamma beta|, 1/(gamma+u))."""
        if not self.ready:
            self._init_gl()
        W, H = BOX.X, BOX.Y
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.texD)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 0, 0, W, H, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texA)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F if BOX.HDR else GL_RGBA8, 0, 0, W, H, 0)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        FlatScreen.push()
        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "scene"), 0)
        glUniform1i(glGetUniformLocation(self.program, "depth"), 1)
        glUniform2f(glGetUniformLocation(self.program, "size"), float(W), float(H))
        glUniform3f(glGetUniformLocation(self.program, "hdir"), *hdir)
        glUniform1f(glGetUniformLocation(self.program, "D"), D)
        glUniform1f(glGetUniformLocation(self.program, "Rs"), Rs)
        glUniform1f(glGetUniformLocation(self.program, "shadow"), 1.0 if is_hole else 0.0)
        glUniform1f(glGetUniformLocation(self.program, "tanh_"), tan(radians(VIEW_ANGLE) * 0.5))
        glUniform1f(glGetUniformLocation(self.program, "aspect"), W * 1.0 / H)
        glUniform1f(glGetUniformLocation(self.program, "near_"), BOX.near_clip)
        glUniform1f(glGetUniformLocation(self.program, "far_"), BOX.far_clip)
        glUniform3f(glGetUniformLocation(self.program, "n_"), *n_eye)
        glUniform1f(glGetUniformLocation(self.program, "u_"), u)
        glUniform1f(glGetUniformLocation(self.program, "inv_"), inv)
        glColor(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0); glVertex2f(W, 0.0); glVertex2f(W, H); glVertex2f(0.0, H)
        glEnd()
        glUseProgram(0)
        FlatScreen.pop()
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
