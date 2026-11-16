# coding: utf8
# entity/lensing.py
"""
Gravitational light bending near one compact body.

Nothing is captured or resampled.  The deflection is applied where each
thing is drawn:

* the sky shader bends every pixel's direction before it looks up the sky;
* bodies beyond the lens have their vertices moved to the apparent
  (primary-image) direction in the vertex shader; bodies at or in front of
  the lens are not touched, so nothing can ghost;
* the lens body itself is drawn at its apparent size, R/sqrt(1 - R_s/R)
  for a star, the photon-capture radius 3 sqrt(3)/2 R_s (the shadow) for a
  black hole.

A ray ending at the observer at angle theta from the lens passed it at
impact parameter b = D sin(theta) and was bent by the part of the
Schwarzschild deflection 2 R_s/b + (15 pi/16)(R_s/b)^2 that lies before the
observer, the fraction (1 + cos theta)/2 -- nothing at all from straight
behind.  A source at distance D_s sees the fraction (D_s - D)/D_s of that
(the thin-lens equation), the sky all of it.  The bending is done in the
background frame: directions seen by the moving ship are first taken there
(inverse aberration) and brought back after.

Only done when 2 R_s / D exceeds a threshold, so nothing is spent on it in
an ordinary planetary system.
"""
from OpenGL.GL import *

from program.utils import boost


# GLSL shared by the sky and the bodies: uniforms and the bending.
# lens_h: unit direction of the lens, background frame; lens_k: R_s / D
# (0: no lens); lens_n, lens_u, lens_inv: the ship's boost (direction,
# |gamma beta|, 1/(gamma + u)).
GLSL = """
uniform vec3 lens_h;
uniform float lens_k;
uniform vec3 lens_n;
uniform float lens_u, lens_inv;
uniform float lens_exact;        // 1: a black hole; the sky map comes from the table below
uniform sampler1D lens_table;    // view angle from the hole (0..pi) -> (source angle, shadow flag)
// direction toward a source seen by the ship -> the same ray in the background frame
vec3 lens_to_lab(vec3 v) {
    float c = dot(v, lens_n);
    float tpv = (c > 0.0) ? 0.5*dot(v - lens_n, v - lens_n) : 1.0 - c;
    float vpar = -c;
    vec3 perp = -v - vpar*lens_n;
    vec3 xyz = perp + (lens_inv*vpar + lens_u*tpv)*lens_n;
    float t = lens_inv + lens_u*tpv;
    return normalize(-xyz / t);
}
// and back: a background-frame source direction as the ship sees it
vec3 lens_to_ship(vec3 d) {
    float c = dot(d, lens_n);
    float tmv = (c < 0.0) ? 0.5*dot(d + lens_n, d + lens_n) : 1.0 + c;
    float vpar = -c;
    vec3 perp = -d - vpar*lens_n;
    vec3 xyz = perp + (lens_inv*vpar - lens_u*tmv)*lens_n;
    float t = lens_inv + lens_u*tmv;
    return normalize(-xyz / t);
}
// deflection of a ray arriving at angle th from the lens, times w
float lens_alpha(float th, float w) {
    float st = max(sin(th), 1e-6);
    float k = lens_k;
    return (2.0*k/st + 2.945*k*k/(st*st)) * 0.5*(1.0 + cos(th)) * w;
}
// where a pixel's light came from: background-frame direction d -> source direction
// (a zero vector means the shadow of a black hole)
vec3 lens_source(vec3 d) {
    float ct = clamp(dot(d, lens_h), -1.0, 1.0);
    float th = acos(ct);
    vec3 e = d - ct*lens_h;
    float el = length(e);
    float beta;
    if (lens_exact > 0.5) {
        vec4 p = texture1D(lens_table, th * 0.31830988618379067);
        if (p.g > 0.5) return vec3(0.0);
        beta = p.r;
    } else {
        beta = th - lens_alpha(th, 1.0);
    }
    if (el < 1e-6) return d;
    e /= el;
    return lens_h*cos(beta) + e*sin(beta);
}
// where a source at background-frame direction d appears (primary image),
// for a source that sees the fraction w of the deflection
vec3 lens_apparent(vec3 d, float w) {
    float cb = clamp(dot(d, lens_h), -1.0, 1.0);
    float beta = acos(cb);
    vec3 e = d - cb*lens_h;
    float el = length(e);
    if (el < 1e-6) return d;
    e /= el;
    // solve th - alpha(th) w = beta outside the Einstein ring: the
    // first-order closed form, then Newton's method
    float th = 0.5*(beta + sqrt(beta*beta + 8.0*lens_k*w));
    for (int i = 0; i < 4; i++) {
        float f = th - lens_alpha(th, w) - beta;
        float fp = 1.0 - (lens_alpha(th + 1e-4, w) - lens_alpha(th - 1e-4, w)) * 5000.0;
        th -= f / max(fp, 0.2);
        th = max(th, 1e-5);
    }
    return lens_h*cos(th) + e*sin(th);
}
"""


from math import pi, sin, sqrt


def _simpson(f, a, b, n=40):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4.0 if i % 2 else 2.0) * f(a + i*h)
    return s * h / 3.0


def hole_table(Rs, r, N=256):
    """The sky of a static observer at radius r from a Schwarzschild black
    hole, exactly: for view angles theta from the hole's direction
    (0..pi), the angle beta from that direction the light really came from,
    and whether the ray is one that climbed out of the hole (the shadow).

    A ray at angle theta has impact parameter b = r sin(theta)/sqrt(1-Rs/r).
    Looking toward the hole (theta < pi/2) the photon is outgoing at the
    observer: it has passed its periapsis, so b must exceed the capture
    value 3 sqrt(3)/2 Rs, and it swept 2 Phi(b) - Psi(b, r); looking away it
    is incoming and swept Psi(b, r), where Phi is the sweep from infinity to
    periapsis and Psi from infinity to r (u = 1/r':
    d phi = du / sqrt(1/b^2 - u^2 + Rs u^3)).  beta = pi - sweep."""
    out = []
    f = 1.0 - Rs/r
    if f <= 0.0:
        return [(0.0, 1.0)] * N
    sf = sqrt(f)
    bc = 1.5*sqrt(3.0)*Rs
    def g(u, b):
        return 1.0/(b*b) - u*u + Rs*u*u*u
    def periapsis_u(b):
        # smallest positive root of g: from u = 1/b, where g is slightly positive
        u = 1.0/b
        for _ in range(40):
            gu = g(u, b)
            dg = -2.0*u + 3.0*Rs*u*u
            if abs(dg) < 1e-300:
                break
            un = u - gu/dg
            if un <= 0.0:
                un = 0.5*u
            if abs(un - u) < 1e-13*max(u, 1e-300):
                u = un
                break
            u = un
        return u
    for i in range(N):
        th = pi*(i + 0.5)/N
        b = r*sin(th)/sf
        outgoing = th < 0.5*pi
        if b < 1e-12:
            out.append((0.0 if not outgoing else 0.0, 1.0 if outgoing else 0.0))
            continue
        if outgoing and b <= bc*(1.0 + 1e-9):
            out.append((0.0, 1.0))
            continue
        if b > bc:
            up = periapsis_u(b)
            if up < (1.0 - 1e-9)/r:
                # periapsis outside the observer: no ray from infinity reaches
                # here with this impact parameter (inside the photon sphere
                # only the narrow window of b < b_c looks out)
                out.append((0.0, 1.0))
                continue
            # Phi: from u = 0 to u_p with u = u_p (1 - t^2) (regular integrand)
            def ph(t, b=b, up=up):
                u = up*(1.0 - t*t)
                gu = max(g(u, b), 1e-300)
                return 2.0*up*t / sqrt(gu)
            # the limit at t -> 0 from g ~ -g'(u_p) u_p t^2
            dg = -2.0*up + 3.0*Rs*up*up
            def ph0(t, b=b, up=up, dg=dg):
                return ph(t) if t > 1e-6 else 2.0*up / sqrt(max(-dg*up, 1e-300))
            Phi = _simpson(ph0, 0.0, 1.0)
            # Psi: from u = 0 (t = 1) to u = 1/r (t_r), same substitution
            tr2 = 1.0 - 1.0/(r*up)
            tr = sqrt(max(tr2, 0.0))
            Psi = _simpson(ph0, tr, 1.0)
            sweep = (2.0*Phi - Psi) if outgoing else Psi
        else:
            # no periapsis: a straight integral in u from 0 to 1/r
            Psi = _simpson(lambda u, b=b: 1.0/sqrt(max(g(u, b), 1e-300)), 0.0, 1.0/r)
            sweep = Psi
        out.append((pi - sweep, 0.0))
    return out


class Lensing(object):
    THRESHOLD = 2.5e-5      # 2 R_s / D: the Einstein radius is 0.3 degrees here
    TABLE_N = 256
    _table_tex = None
    _table_key = None

    # the lens in force for the frame being drawn, or None:
    # dict(body, D, Rs, R, h, is_hole, n, u, inv, cap)
    current = None

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

    @staticmethod
    def capture(R, Rs):
        """Apparent radius over R_s: the photon-capture radius for a black
        hole or a star inside it, else the star's own lensed radius."""
        if R <= 1.5*Rs:
            return 2.598
        return (R / Rs) / (1.0 - Rs / R) ** 0.5

    @classmethod
    def set_current(cls, lens, Xc, Xp, U):
        """lens = (body, D) or None; Xc: the body's (retarded) position;
        Xp, U: the player's position and 4-velocity, background frame."""
        if lens is None:
            cls.current = None
            return
        body, D = lens
        dX = Xc - Xp
        n = (dX.x*dX.x + dX.y*dX.y + dX.z*dX.z) ** 0.5
        if n <= 0.0:
            cls.current = None
            return
        bn, bu, bg, binv = boost.params(U)
        Rs = 2.0*body.GM
        cls.current = dict(body=body, D=D, Rs=Rs, R=body.radius, h=(dX.x/n, dX.y/n, dX.z/n),
                           is_hole=body.is_hole, n=bn, u=bu, inv=binv,
                           cap=2.598 if body.is_hole else cls.capture(body.radius, Rs))

    @classmethod
    def enlarge(cls, body):
        """How much larger than its radius the body is drawn: its apparent
        size when it is the lens, 1 otherwise."""
        c = cls.current
        if c is None or c["body"] is not body or c["R"] <= 0.0:
            return 1.0
        f = max(1.0 - c["Rs"] / c["D"], 0.0) ** 0.5     # as seen from D, not from infinity
        e = c["cap"] * c["Rs"] * f / c["R"]
        if c["is_hole"]:
            # the shadow itself is painted exactly by the sky; the globe only
            # hides what is behind the hole, so it never swallows the observer
            e = min(e, 0.9 * c["D"] / c["R"])
        return max(e, 1.0)

    @classmethod
    def weight(cls, Ds):
        """Fraction of the deflection seen by a body at distance Ds: the
        thin-lens (D_s - D)/D_s, 0 for anything at or inside the lens's
        distance (the lens itself, its planets and disc)."""
        c = cls.current
        if c is None or Ds <= c["D"] + 1.5*c["R"]:
            return 0.0
        return (Ds - c["D"]) / Ds

    @classmethod
    def bind_table(cls, unit=4):
        """For a black hole lens, build (when the distance changed by more
        than 0.3%) and bind the exact sky table on texture unit `unit`."""
        c = cls.current
        if c is None or not c["is_hole"]:
            return False
        key = (c["body"].name, round(c["D"] / c["Rs"], 3) if c["D"] < 1e3*c["Rs"] else None)
        if key != cls._table_key or cls._table_tex is None:
            if cls._table_tex is None:
                cls._table_tex = glGenTextures(1)
                glBindTexture(GL_TEXTURE_1D, cls._table_tex)
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            if key[1] is None:
                cls._table_key = key
                return False
            rows = hole_table(c["Rs"], c["D"], cls.TABLE_N)
            flat = []
            for beta, shadow in rows:
                flat += [beta, shadow, 0.0, 1.0]
            glBindTexture(GL_TEXTURE_1D, cls._table_tex)
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, cls.TABLE_N, 0, GL_RGBA, GL_FLOAT, (GLfloat*len(flat))(*flat))
            cls._table_key = key
        if key[1] is None:
            return False
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_1D, cls._table_tex)
        glActiveTexture(GL_TEXTURE0)
        return True

    @classmethod
    def set_uniforms(cls, program, w=1.0, table_unit=None):
        """Load the lens into a program that includes GLSL (k = 0: no lens;
        lens_w, where the program has it: the body's fraction)."""
        c = cls.current
        loc = glGetUniformLocation(program, "lens_w")
        if table_unit is not None:
            # always: a sampler1D left on unit 0 beside the program's sampler2D
            # makes every draw fail with GL_INVALID_OPERATION (no sky at all)
            glUniform1i(glGetUniformLocation(program, "lens_table"), table_unit)
        if c is None or w <= 0.0:
            glUniform1f(glGetUniformLocation(program, "lens_k"), 0.0)
            if loc >= 0:
                glUniform1f(loc, 0.0)
            return
        glUniform3f(glGetUniformLocation(program, "lens_h"), *c["h"])
        glUniform1f(glGetUniformLocation(program, "lens_k"), c["Rs"] / c["D"])
        glUniform3f(glGetUniformLocation(program, "lens_n"), *c["n"])
        glUniform1f(glGetUniformLocation(program, "lens_u"), c["u"])
        glUniform1f(glGetUniformLocation(program, "lens_inv"), c["inv"])
        exact = table_unit is not None and cls.bind_table(table_unit)
        glUniform1f(glGetUniformLocation(program, "lens_exact"), 1.0 if exact else 0.0)
        if loc >= 0:
            glUniform1f(loc, w)
