# coding: utf8
# program/utils/kerr.py
"""
A spinning black hole: the Kerr metric in Boyer-Lindquist coordinates,
G = c = 1, lengths in light-seconds, M = GM, a = (a/M) M.

    Sigma = r^2 + a^2 cos^2 th,  Delta = r^2 - 2 M r + a^2,
    A = (r^2 + a^2)^2 - a^2 Delta sin^2 th,
    omega = 2 M a r / A        (frame dragging: the ZAMO's angular velocity)
    alpha = sqrt(Delta Sigma / A)   (lapse: a ZAMO's clock against infinity)
    varpi = sqrt(A / Sigma) sin th  (the ZAMO's circumferential radius)

The local frame everything is reckoned in near the hole is the ZAMO's (zero
angular momentum observer: at rest in the Boyer-Lindquist grid except for
being dragged round at omega), which exists all the way down to the
horizon, where static observers do not (the ergosphere).  The ship's U in
the game's coordinates is its velocity relative to the ZAMO at its
position, with the tetrad directions r^, th^, ph^ mapped onto the game's
spherical unit vectors about the hole's spin axis.

Motion: the geodesic Hamiltonian H = g^{mu nu} p_mu p_nu / 2 with the
conserved E = -p_t and L = p_phi,
    2 Sigma H = Delta p_r^2 + p_th^2 + Theta(th) - R(r)/Delta + m^2 (r^2 + a^2 cos^2 th)
    Theta = (L - a E sin^2 th)^2 / sin^2 th,   R = ((r^2 + a^2) E - a L)^2
integrated in proper time (ship) or Mino time (light, on the GPU).

Light: the sky of a ZAMO is a 2-D table over its view directions, built on
the GPU each time the ship moves: every texel traces its photon back
through the metric to infinity (sky direction, frequency factor), into the
horizon (shadow) or into the equatorial accretion disc (radius, azimuth of
the gas, frequency factor against the orbiting gas).  The lookup is in
Lensing's GLSL (kerr_source).
"""
import math

from OpenGL.GL import *

from program.utils import compile_program


def horizon(M, a):
    return M + math.sqrt(max(M*M - a*a, 0.0))


def isco(M, a):
    """Innermost stable circular orbit (Bardeen, Press, Teukolsky): prograde
    for a > 0, retrograde for a < 0."""
    x = abs(a) / M
    z1 = 1.0 + (1.0 - x*x) ** (1.0/3.0) * ((1.0 + x) ** (1.0/3.0) + (1.0 - x) ** (1.0/3.0))
    z2 = math.sqrt(3.0*x*x + z1*z1)
    root = math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0*z2))
    return M * (3.0 + z2 - (root if a >= 0.0 else -root))


def omega_orbit(M, a, r):
    """Coordinate angular velocity of the prograde circular equatorial orbit."""
    return math.sqrt(M) / (r ** 1.5 + a * math.sqrt(M))


def ut_orbit(M, a, r):
    """dt/dtau of the gas on that orbit (its energy against infinity per unit
    rest mass is u_t; -p.u = u^t (E - Omega L) for a photon)."""
    sm = math.sqrt(M)
    d = r ** 1.5 - 3.0*M*math.sqrt(r) + 2.0*a*sm
    if d <= 0.0:
        return 1.0e9
    return (r ** 1.5 + a*sm) / (r ** 0.75 * math.sqrt(d))


def zamo(M, a, r, th):
    """(alpha, omega, varpi, Sigma, Delta) at (r, th)."""
    s = math.sin(th); c = math.cos(th)
    S = r*r + a*a*c*c
    D = r*r - 2.0*M*r + a*a
    A = (r*r + a*a) ** 2 - a*a*D*s*s
    om = 2.0*M*a*r / A
    al = math.sqrt(max(D*S / A, 1.0e-300))
    vp = math.sqrt(A / S) * s
    return al, om, vp, S, D


def circular_speed(M, a, r, prograde=True):
    """Speed of the circular equatorial orbit at r relative to the ZAMO
    there (what the ship's U is measured against); None where no such
    orbit exists."""
    sm = math.sqrt(M)
    Om = sm / (r ** 1.5 + a*sm) if prograde else -sm / (r ** 1.5 - a*sm)
    al, om, vp, S, D = zamo(M, a, r, 0.5*math.pi)
    v = (Om - om) * vp / al
    if abs(v) >= 1.0:
        return None
    return v


def axes(spin):
    """(e1, e2, s): an orthonormal basis with the spin axis last, so that
    (r, th, ph) are Boyer-Lindquist angles about the spin."""
    sx, sy, sz = spin
    n = math.sqrt(sx*sx + sy*sy + sz*sz)
    sx, sy, sz = sx/n, sy/n, sz/n
    # e1: the game's +x projected off the axis, else +z
    ex, ey, ez = 1.0 - sx*sx, -sx*sy, -sx*sz
    m = math.sqrt(ex*ex + ey*ey + ez*ez)
    if m < 1.0e-6:
        ex, ey, ez = -sz*sx, -sz*sy, 1.0 - sz*sz
        m = math.sqrt(ex*ex + ey*ey + ez*ez)
    ex, ey, ez = ex/m, ey/m, ez/m
    # e2 = s x e1
    fx, fy, fz = sy*ez - sz*ey, sz*ex - sx*ez, sx*ey - sy*ex
    return (ex, ey, ez), (fx, fy, fz), (sx, sy, sz)


def spherical(basis, d):
    """Boyer-Lindquist (r, th, ph) of the game-frame offset d from the hole,
    and the unit vectors r^, th^, ph^ there (game frame)."""
    (e1, e2, s) = basis
    x1 = d[0]*e1[0] + d[1]*e1[1] + d[2]*e1[2]
    x2 = d[0]*e2[0] + d[1]*e2[1] + d[2]*e2[2]
    x3 = d[0]*s[0] + d[1]*s[1] + d[2]*s[2]
    r = math.sqrt(x1*x1 + x2*x2 + x3*x3)
    if r <= 0.0:
        return None
    th = math.acos(max(-1.0, min(1.0, x3 / r)))
    ph = math.atan2(x2, x1)
    st, ct = math.sin(th), math.cos(th)
    cp, sp = math.cos(ph), math.sin(ph)
    def comb(a1, a2, a3):
        return (a1*e1[0] + a2*e2[0] + a3*s[0], a1*e1[1] + a2*e2[1] + a3*s[1], a1*e1[2] + a2*e2[2] + a3*s[2])
    er = comb(st*cp, st*sp, ct)
    et = comb(ct*cp, ct*sp, -st)
    ep = comb(-sp, cp, 0.0)
    return r, th, ph, er, et, ep


def ship_step(M, a, r, th, ph, pr, pth, E, L, ds):
    """One proper-time step of a massive geodesic: returns (r, th, ph, pr,
    pth, dt), dt the coordinate time elapsed.  4th-order Runge-Kutta on
    the Hamiltonian form (E and L conserved)."""
    a2 = a*a
    def deriv(r, th, pr, pth):
        s = math.sin(th); c = math.cos(th)
        if abs(s) < 1.0e-6:
            s = 1.0e-6 if s >= 0.0 else -1.0e-6
        s2 = s*s
        S = r*r + a2*c*c
        D = r*r - 2.0*M*r + a2
        P = (r*r + a2)*E - a*L
        Lm = L - a*E*s2
        dr = D*pr / S
        dth = pth / S
        dph = (Lm/s2 + a*P/D) / S
        dt = ((r*r + a2)*P/D + a*Lm) / S
        Dp = 2.0*r - 2.0*M
        dRD = (4.0*r*E*P*D - P*P*Dp) / (D*D)         # d/dr of R/Delta
        dGr = Dp*pr*pr - dRD + 2.0*r
        dGt = -2.0*L*L*c/(s2*s) + 2.0*a2*E*E*s*c - 2.0*a2*c*s
        return dr, dth, dph, dt, -0.5*dGr/S, -0.5*dGt/S
    k1 = deriv(r, th, pr, pth)
    k2 = deriv(r + 0.5*ds*k1[0], th + 0.5*ds*k1[1], pr + 0.5*ds*k1[4], pth + 0.5*ds*k1[5])
    k3 = deriv(r + 0.5*ds*k2[0], th + 0.5*ds*k2[1], pr + 0.5*ds*k2[4], pth + 0.5*ds*k2[5])
    k4 = deriv(r + ds*k3[0], th + ds*k3[1], pr + ds*k3[4], pth + ds*k3[5])
    def comb(i):
        return ds/6.0*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
    return (r + comb(0), th + comb(1), ph + comb(2), pr + comb(4), pth + comb(5), comb(3))


# ------------------------------------------------------------------ the sky
# The table's rows are view angles from the hole's direction (pi - vartheta,
# vartheta from the outward radial r^): a fraction WIN_FRAC of the rows
# cover the hole's neighbourhood, out to the angle `win` (the disc's images
# and the shadow, from this distance), uniformly; the rest the remaining
# sky out to the anti-hole direction.  The columns are the angle psi about
# r^ (from th^ toward ph^).
TABLE_W, TABLE_H = 384, 512
WIN_FRAC = 0.75
R_FAR = 1.0e4             # in M: where the ray is read as at infinity
MAX_STEPS = 2400


def window(r_M, r_out_M):
    """The angle (radians, from the hole's direction) the fine rows cover
    for an observer at r_M: the disc's widest images, with room."""
    return min(math.pi, 2.0*math.atan(1.5*max(r_out_M, 8.0) / r_M))


def row_angle(j, win, H=None):
    """Python twin of kerr_vth: the view angle from r^ of row j (texel centre)."""
    H = H or TABLE_H
    v = (j + 0.5) / H
    away = win*v/WIN_FRAC if v <= WIN_FRAC else win + (math.pi - win)*(v - WIN_FRAC)/(1.0 - WIN_FRAC)
    return math.pi - away


MAP_GLSL = """
uniform float kerr_win;    // the fine rows' extent (radians from the hole's direction)
// row coordinate v -> angle from the outward radial direction
float kerr_vth(float v) {
    float f = %g;
    float away = (v <= f) ? kerr_win * v / f : kerr_win + (3.14159265358979 - kerr_win) * (v - f) / (1.0 - f);
    return 3.14159265358979 - away;
}
// and back
float kerr_v(float vth) {
    float f = %g;
    float away = 3.14159265358979 - vth;
    return (away <= kerr_win) ? f * away / kerr_win : f + (1.0 - f) * (away - kerr_win) / (3.14159265358979 - kerr_win);
}
""" % (WIN_FRAC, WIN_FRAC)

BUILD_VS = """
#version 120
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""

# The tracer: shared by the table build and by the sky shader, which runs
# it for the pixels the table cannot resolve (the band along the shadow's
# edge and the disc's edges, where neighbouring texels disagree).
TRACE_GLSL = """
uniform float ka;          // a/M
uniform float kr;          // observer r/M
uniform float kth;         // observer theta
uniform float krfar;
uniform float kd_in;       // disc inner edge (M), 0: no disc
uniform float kd_out;      // outer edge (M)
uniform vec3 kd_phase;     // gas pattern phase of the three rings at the observer's time
uniform vec3 kd_omega;     // their pattern angular velocities (per M of time)
uniform vec2 kd_split;     // ring boundaries (M)
float Omega(float r) { return 1.0 / (pow(r, 1.5) + ka); }
float ut_gas(float r) {
    float d = pow(r, 1.5) - 3.0*sqrt(r) + 2.0*ka;
    return (pow(r, 1.5) + ka) / (pow(r, 0.75) * sqrt(max(d, 1e-6)));
}
// The photon arriving along d (the ZAMO's tetrad: r^, th^, ph^), traced
// backward.  Returns 1 for a ray from the sky, 0 for one from the horizon;
// dir: where the sky ray came from (hole frame: x the observer's azimuth,
// z the spin), invE: 1/E, its frequency factor; q: the ray's crossing of
// the equatorial plane as the table stores it (radius in M, the gas
// pattern's cos and sin, the frequency factor against the gas); hit: the
// crossing point relative to the observer (hole frame, M), or the hole's
// centre when the ray never crossed the plane.
int kerr_trace(vec3 d, out vec3 dir, out float invE, out vec4 q, out vec3 hit) {
    // the ZAMO at the observer
    float a = ka, a2 = ka*ka;
    float r = kr, th = kth;
    float s = sin(th), c = cos(th);
    float S = r*r + a2*c*c;
    float D = r*r - 2.0*r + a2;
    float A = (r*r + a2)*(r*r + a2) - a2*D*s*s;
    float om = 2.0*a*r/A;
    float al = sqrt(D*S/A);
    float vp = sqrt(A/S)*s;
    vec3 obs = vec3(r*s, 0.0, r*c);
    // the photon that arrives along d moves along -d; unit energy in the ZAMO frame
    float prh = -d.x, pth_ = -d.y, pph = -d.z;
    float E = al + om*vp*pph;
    float L = vp*pph;
    float pr = sqrt(S/D)*prh;
    float pt = sqrt(S)*pth_;
    float ph = 0.0, t = 0.0;
    float rplus = 1.0 + sqrt(max(1.0 - a2, 0.0));
    float rstop = rplus * 1.005;
    float cprev = c, rprev = r, phprev = ph, tprev = t;
    int state = 0;        // 0 running, 1 sky, 2 shadow
    // the ray's crossing of the equatorial plane: the first one inside the
    // disc if any, else the one nearest the disc's edges (a continuous
    // field across each edge, so that the lookup can interpolate it)
    float hit_r = -1.0, hit_ph = 0.0, hit_t = 0.0, hit_d = 1.0e9;
    bool hit_in = false;
    for (int i = 0; i < %d; i++) {
        // one RK4 step backward in Mino time
        float h;
        {
            float s_ = sin(th); float s2 = s_*s_;
            float D_ = r*r - 2.0*r + a2;
            float P = (r*r + a2)*E - a*L;
            float Lm = L - a*E*s2;
            float dph = Lm/s2 + a*P/D_;
            h = 0.02 / max(max(abs(D_*pr)/r, abs(pt)), max(abs(dph), 1e-9));
            h = -h;
        }
        float rr = r, tt = th, pp = pr, qq = pt;
        float k_r[4]; float k_t[4]; float k_p[4]; float k_q[4]; float k_f[4]; float k_s[4];
        for (int j = 0; j < 4; j++) {
            float s_ = sin(tt), c_ = cos(tt);
            if (abs(s_) < 1e-5) s_ = (s_ >= 0.0) ? 1e-5 : -1e-5;
            float s2 = s_*s_;
            float S_ = rr*rr + a2*c_*c_;
            float D_ = rr*rr - 2.0*rr + a2;
            float P = (rr*rr + a2)*E - a*L;
            float Lm = L - a*E*s2;
            float Dp = 2.0*rr - 2.0;
            float dRD = (4.0*rr*E*P*D_ - P*P*Dp) / (D_*D_);
            k_r[j] = D_*pp;
            k_t[j] = qq;
            k_f[j] = Lm/s2 + a*P/D_;
            k_s[j] = (rr*rr + a2)*P/D_ + a*Lm;
            k_p[j] = -0.5*(Dp*pp*pp - dRD);
            k_q[j] = L*L*c_/(s2*s_) - a2*E*E*s_*c_;
            float f = (j < 2) ? 0.5*h : h;
            if (j < 3) {
                rr = r + f*k_r[j]; tt = th + f*k_t[j]; pp = pr + f*k_p[j]; qq = pt + f*k_q[j];
            }
        }
        r  += h/6.0*(k_r[0] + 2.0*k_r[1] + 2.0*k_r[2] + k_r[3]);
        th += h/6.0*(k_t[0] + 2.0*k_t[1] + 2.0*k_t[2] + k_t[3]);
        pr += h/6.0*(k_p[0] + 2.0*k_p[1] + 2.0*k_p[2] + k_p[3]);
        pt += h/6.0*(k_q[0] + 2.0*k_q[1] + 2.0*k_q[2] + k_q[3]);
        ph += h/6.0*(k_f[0] + 2.0*k_f[1] + 2.0*k_f[2] + k_f[3]);
        t  += h/6.0*(k_s[0] + 2.0*k_s[1] + 2.0*k_s[2] + k_s[3]);
        float cc = cos(th);
        if (kd_in > 0.0 && cprev*cc < 0.0 && !hit_in) {
            float f = cprev / (cprev - cc);
            float rh = rprev + f*(r - rprev);
            float dd = max(max(kd_in - rh, rh - kd_out), 0.0);
            if (dd < hit_d) {
                hit_r = rh; hit_ph = phprev + f*(ph - phprev); hit_t = tprev + f*(t - tprev);
                hit_d = dd; hit_in = (dd <= 0.0);
            }
        }
        cprev = cc; rprev = r; phprev = ph; tprev = t;
        if (r < rstop) { state = 2; break; }
        if (r > krfar) { state = 1; break; }
    }
    dir = vec3(0.0); invE = 0.0;
    if (state == 1) {
        // the direction the light came from: minus its momentum direction,
        // read in the flat frame at the far point, as a vector in the hole's
        // frame (x toward the observer's azimuth, z the spin)
        float s_ = sin(th), c_ = cos(th);
        float S_ = r*r + a2*c_*c_;
        float D_ = r*r - 2.0*r + a2;
        float A_ = (r*r + a2)*(r*r + a2) - a2*D_*s_*s_;
        float vp_ = sqrt(A_/S_)*s_;
        vec3 m = vec3(sqrt(D_/S_)*pr, pt/sqrt(S_), L/vp_);
        m = -normalize(m);
        float cp = cos(ph), sp = sin(ph);
        vec3 er = vec3(s_*cp, s_*sp, c_);
        vec3 et = vec3(c_*cp, c_*sp, -s_);
        vec3 ep = vec3(-sp, cp, 0.0);
        dir = m.x*er + m.y*et + m.z*ep;
        invE = 1.0/E;
        if (hit_r < 0.0) hit_r = 1.0e4;          // never through the plane: beyond the disc
    } else {
        if (hit_r < 0.0) hit_r = 0.0;            // never through the plane: inside the disc's edge
    }
    hit = -obs;
    if (hit_r > 0.0 && hit_r < 1.0e3) {
        int ring = (hit_r < kd_split.x) ? 0 : ((hit_r < kd_split.y) ? 1 : 2);
        float phase = (ring == 0) ? kd_phase.x : ((ring == 1) ? kd_phase.y : kd_phase.z);
        float omr = (ring == 0) ? kd_omega.x : ((ring == 1) ? kd_omega.y : kd_omega.z);
        float pat = hit_ph - phase - omr*hit_t;
        // the gas's orbit exists from the inner edge out; inside it the
        // formula runs away, and the value only serves the edge's blend
        float rg = clamp(hit_r, kd_in, kd_out);
        float g = 1.0 / (ut_gas(rg) * max(E - Omega(rg)*L, 1e-3));
        q = vec4(hit_r, cos(pat), sin(pat), clamp(g, 0.01, 4.0));
        hit = vec3(hit_r*cos(hit_ph), hit_r*sin(hit_ph), 0.0) - obs;
    } else {
        q = vec4(hit_r, 1.0, 0.0, 1.0);
    }
    return state == 1 ? 1 : 0;
}
""" % MAX_STEPS

BUILD_FS = """
#version 120
uniform vec2 ksize;        // table size (texels)
""" + MAP_GLSL + TRACE_GLSL + """
void main() {
    // the view direction of this texel in the ZAMO's tetrad (r^, th^, ph^)
    vec2 uv = gl_FragCoord.xy / ksize;
    float psi = 6.28318530717959 * uv.x - 3.14159265358979;
    float vth = kerr_vth(uv.y);
    vec3 d = vec3(cos(vth), sin(vth)*cos(psi), sin(vth)*sin(psi));
    vec3 dir, hit; float invE; vec4 q;
    kerr_trace(d, dir, invE, q, hit);
    gl_FragData[0] = vec4(dir, invE);     // zero in the shadow
    gl_FragData[1] = q;
    gl_FragData[2] = vec4(hit, 0.0);
}
"""


class KerrTable(object):
    """The ZAMO's sky, built on the GPU into an RGBA32F texture."""

    def __init__(self):
        self.program = None
        self.tex = None
        self.tex_disc = None
        self.tex_hit = None
        self.params = None
        self.fbo = None
        self.api = None
        self.key = None
        self.win = math.pi
        self.ok = True

    def _init_gl(self):
        from entity.glare import _fbo_api
        self.api = _fbo_api()
        if self.api is None or not bool(glDrawBuffers):
            self.ok = False
            return
        self.program = compile_program(BUILD_VS, BUILD_FS)
        bound = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
        self.tex, self.tex_disc, self.tex_hit = [int(t) for t in glGenTextures(3)]
        for tex in (self.tex, self.tex_disc, self.tex_hit):
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TABLE_W, TABLE_H, 0, GL_RGBA, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, bound)
        self.fbo = self.api[0](1)

    def build(self, a_over_M, r_M, th, disc=None):
        """Trace the table for an observer at (r_M, th) (r in units of M);
        disc = (r_in, r_out, phases(3), omegas(3), split(2)) in M units, or
        None.  Skipped when nothing changed since the last build."""
        if not self.ok:
            return False
        # called from inside a draw (the sky's uniforms): keep its program,
        # texture unit and bindings
        prog = int(glGetIntegerv(GL_CURRENT_PROGRAM))
        unit = int(glGetIntegerv(GL_ACTIVE_TEXTURE))
        if self.program is None:
            self._init_gl()
            glUseProgram(prog)
            glActiveTexture(unit)
            if not self.ok:
                return False
        win = window(r_M, disc[1] if disc is not None else 0.0)
        key = (round(a_over_M, 6), round(r_M, 5), round(th, 6), round(win, 6),
               None if disc is None else tuple(round(x, 5) for x in (disc[0], disc[1]) + tuple(disc[2])))
        if key == self.key:
            return True
        glActiveTexture(GL_TEXTURE0)
        bound0 = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
        gen, bind, attach, status, FB, ATT0, COMPLETE = self.api
        bind(FB, self.fbo)
        attach(FB, ATT0, GL_TEXTURE_2D, self.tex, 0)
        attach(FB, ATT0 + 1, GL_TEXTURE_2D, self.tex_disc, 0)
        attach(FB, ATT0 + 2, GL_TEXTURE_2D, self.tex_hit, 0)
        glDrawBuffers(3, (GLenum * 3)(ATT0, ATT0 + 1, ATT0 + 2))
        if status(FB) != COMPLETE:
            bind(FB, 0)
            self.ok = False
            return False
        glPushAttrib(GL_VIEWPORT_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, TABLE_W, TABLE_H)
        glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND); glDisable(GL_CULL_FACE)
        glDepthMask(GL_FALSE)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0.0, TABLE_W, 0.0, TABLE_H, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glUseProgram(self.program)
        u = lambda n: glGetUniformLocation(self.program, n)
        glUniform1f(u("ka"), a_over_M)
        glUniform1f(u("kr"), r_M)
        glUniform1f(u("kth"), th)
        glUniform1f(u("kerr_win"), win)
        self.win = win
        glUniform2f(u("ksize"), float(TABLE_W), float(TABLE_H))
        glUniform1f(u("krfar"), max(R_FAR, 20.0*r_M))
        self.params = dict(ka=a_over_M, kr=r_M, kth=th, krfar=max(R_FAR, 20.0*r_M), disc=disc)
        if disc is None:
            glUniform1f(u("kd_in"), 0.0); glUniform1f(u("kd_out"), 0.0)
            glUniform3f(u("kd_phase"), 0.0, 0.0, 0.0); glUniform3f(u("kd_omega"), 0.0, 0.0, 0.0)
            glUniform2f(u("kd_split"), 0.0, 0.0)
        else:
            r_in, r_out, phases, omegas, split = disc
            glUniform1f(u("kd_in"), r_in); glUniform1f(u("kd_out"), r_out)
            glUniform3f(u("kd_phase"), *phases); glUniform3f(u("kd_omega"), *omegas)
            glUniform2f(u("kd_split"), *split)
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0); glVertex2f(TABLE_W, 0.0); glVertex2f(TABLE_W, TABLE_H); glVertex2f(0.0, TABLE_H)
        glEnd()
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()
        glPopAttrib()
        glDepthMask(GL_TRUE)
        glDrawBuffers(1, (GLenum * 1)(ATT0))
        bind(FB, 0)
        glBindTexture(GL_TEXTURE_2D, bound0)
        glUseProgram(prog)
        glActiveTexture(unit)
        self.key = key
        return True

    def bind(self, unit):
        """The sky table on `unit`, the disc table and the crossing points on the next two."""
        for k, tex in enumerate((self.tex, self.tex_disc, self.tex_hit)):
            glActiveTexture(GL_TEXTURE0 + unit + k)
            glBindTexture(GL_TEXTURE_2D, tex if self.tex is not None else 0)
        glActiveTexture(GL_TEXTURE0)

    def set_trace_uniforms(self, program):
        """The tracer's uniforms (TRACE_GLSL) on a program that includes it,
        as the last build set them."""
        if self.params is None:
            return
        u = lambda n: glGetUniformLocation(program, n)
        if u("ka") < 0:
            return
        pr = self.params
        glUniform1f(u("ka"), pr["ka"]); glUniform1f(u("kr"), pr["kr"]); glUniform1f(u("kth"), pr["kth"])
        glUniform1f(u("krfar"), pr["krfar"])
        disc = pr["disc"]
        if disc is None:
            glUniform1f(u("kd_in"), 0.0); glUniform1f(u("kd_out"), 0.0)
            glUniform3f(u("kd_phase"), 0.0, 0.0, 0.0); glUniform3f(u("kd_omega"), 0.0, 0.0, 0.0)
            glUniform2f(u("kd_split"), 0.0, 0.0)
        else:
            r_in, r_out, phases, omegas, split = disc
            glUniform1f(u("kd_in"), r_in); glUniform1f(u("kd_out"), r_out)
            glUniform3f(u("kd_phase"), *phases); glUniform3f(u("kd_omega"), *omegas)
            glUniform2f(u("kd_split"), *split)


# the lookup, included in Lensing's GLSL
GLSL = """
uniform float lens_kerr;        // 1: the lens is a Kerr hole: the sky from kerr_table
uniform sampler2D kerr_table;   // sky direction (hole frame; zero in the shadow), frequency factor
uniform sampler2D kerr_disc;    // the ray's crossing of the equatorial plane: radius (M), gas pattern cos/sin, frequency factor
uniform sampler2D kerr_hit;     // the crossing point relative to the observer (hole frame, M); the hole's centre when none
uniform vec2 kerr_size;         // the tables' size (texels)
uniform float kerr_M;           // the hole's M in light-seconds
uniform float kerr_rin, kerr_rout;        // the disc's edges (M); 0: no disc
uniform vec3 kerr_er, kerr_et, kerr_ep;   // the ZAMO's tetrad directions, background frame
uniform vec3 kerr_bx, kerr_by, kerr_bz;   // the hole's frame (x: observer's azimuth, z: spin), background frame
""" + MAP_GLSL + """
float lens_wsky = 1.0;          // after kerr_source: how much of the pixel is sky (the rest shadow)...
float lens_wdisc = 0.0;         // ...and how much is disc (opaque, over both)
float lens_gdisc = 1.0;         // disc: observed/emitted frequency against the orbiting gas
float lens_disc_r = 0.0;        // disc: radius in M
float lens_disc_u = 0.0;        // ...and the gas pattern's azimuth / 2 pi
vec3 lens_hit = vec3(0.0);      // the crossing point (or the hole) relative to the observer, hole frame, M
vec3 lens_dtet = vec3(1.0, 0.0, 0.0);   // the view direction in the ZAMO's tetrad (r^, th^, ph^)
vec2 lens_uv = vec2(0.0);       // its table coordinates
// the sky direction (background frame) of the ZAMO's view direction d
vec3 kerr_source(vec3 d) {
    float dr = clamp(dot(d, kerr_er), -1.0, 1.0);
    float dt = dot(d, kerr_et), dp = dot(d, kerr_ep);
    float vth = acos(dr);
    float psi = atan(dp, dt);
    float v = kerr_v(vth);
    float u = (psi + 3.14159265358979) / 6.28318530717959;
    vec2 uv = vec2(u, v);
    lens_dtet = vec3(dr, dt, dp); lens_uv = uv;
    vec4 t = texture2D(kerr_table, uv);
    // the direction is zero in the shadow, so its blended length is the sky's share of the pixel
    float ws = length(t.xyz);
    lens_wsky = clamp(ws, 0.0, 1.0);
    lens_gs = (ws > 1e-4) ? t.a / ws : 1.0;
    lens_wdisc = 0.0;
    if (kerr_rin > 0.0) {
        vec4 q = texture2D(kerr_disc, uv);
        // the crossing radius interpolates across the disc's edges (the sky
        // shader turns it into an edge a pixel wide with its derivatives)
        lens_disc_r = q.r;
        lens_wdisc = (q.r >= kerr_rin && q.r <= kerr_rout) ? 1.0 : 0.0;
        lens_gdisc = q.a;
        lens_disc_u = atan(q.b, q.g) / 6.28318530717959;
    }
    lens_hit = texture2D(kerr_hit, uv).xyz;
    if (ws <= 1e-4) return d;
    vec3 qd = t.xyz / ws;
    return normalize(qd.x*kerr_bx + qd.y*kerr_by + qd.z*kerr_bz);
}
"""
