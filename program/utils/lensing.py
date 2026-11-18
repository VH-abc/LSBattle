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
uniform sampler1D lens_table;    // view angle from the hole (0..pi) -> (source angle, shadow flag, frequency factor)
uniform sampler1D lens_tinv;     // source angle (0..pi) -> view angle of its primary image (the table inverted)
uniform sampler1D lens_tinv2;    // source angle (0..pi) -> view angle of its secondary image (wrapped round the far side)
uniform float lens_branch;       // 0: primary image; 1: secondary (a second pass over the body)
float lens_gs = 1.0;             // set by lens_source: observed/emitted frequency of the sky in that direction (exact table only)
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
        lens_gs = p.b;
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
    if (lens_exact > 0.5) {
        // a black hole: the sky table inverted gives the primary image of a
        // source at infinity; a nearer source sees the fraction w of the shift
        if (lens_branch > 0.5) {
            // the secondary image: the ray went round the far side, so the
            // image is on the opposite side of the hole, inside the Einstein ring
            float th2 = texture1D(lens_tinv2, beta * 0.31830988618379067).r;
            return lens_h*cos(th2) - e*sin(th2);
        }
        float thx = texture1D(lens_tinv, beta * 0.31830988618379067).r;
        float thw = beta + (thx - beta) * w;
        return lens_h*cos(thw) + e*sin(thw);
    }
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


import math
from math import pi, sin, sqrt, acos


def _simpson(f, a, b, n=40):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4.0 if i % 2 else 2.0) * f(a + i*h)
    return s * h / 3.0


def _photon_source(Rs, r, b, outgoing):
    """A photon at radius r with impact parameter b, outgoing (past its
    periapsis) or ingoing: the angle from the hole's direction the light
    came from, and whether it is one that climbed out of the hole (the
    shadow).  With u = 1/r', d phi = du / sqrt(1/b^2 - u^2 + Rs u^3); Phi is
    the sweep from infinity to periapsis and Psi from infinity to r, so an
    outgoing photon swept 2 Phi - Psi, an ingoing one Psi, and the source
    angle is pi - sweep.  Outgoing photons need b above the capture value
    3 sqrt(3)/2 Rs; any photon whose periapsis lies outside r was never at
    infinity (it came up from the horizon), inside the horizon that is
    every photon with b above the capture value."""
    bc = 1.5*sqrt(3.0)*Rs
    def g(u):
        return 1.0/(b*b) - u*u + Rs*u*u*u
    if b < 1e-12:
        return (0.0, 1.0) if outgoing else (pi, 0.0)
    if outgoing and b <= bc*(1.0 + 1e-9):
        return (0.0, 1.0)
    if b > bc:
        # periapsis: smallest positive root of g, from u = 1/b (g slightly positive)
        u = 1.0/b
        for _ in range(40):
            gu = g(u)
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
        up = u
        if up < (1.0 - 1e-9)/r:
            return (0.0, 1.0)
        # Phi: from u = 0 to u_p with u = u_p (1 - t^2) (regular integrand)
        dg = -2.0*up + 3.0*Rs*up*up
        def ph(t):
            if t <= 1e-6:
                # the limit at t -> 0 from g ~ -g'(u_p) u_p t^2
                return 2.0*up / sqrt(max(-dg*up, 1e-300))
            u = up*(1.0 - t*t)
            return 2.0*up*t / sqrt(max(g(u), 1e-300))
        Phi = _simpson(ph, 0.0, 1.0)
        # Psi: from u = 0 (t = 1) to u = 1/r (t_r), same substitution
        tr = sqrt(max(1.0 - 1.0/(r*up), 0.0))
        Psi = _simpson(ph, tr, 1.0)
        sweep = (2.0*Phi - Psi) if outgoing else Psi
    else:
        # no periapsis: a straight integral in u from 0 to 1/r
        sweep = _simpson(lambda u: 1.0/sqrt(max(g(u), 1e-300)), 0.0, 1.0/r)
    return (pi - sweep, 0.0)


def hole_table(Rs, r, N=256):
    """The sky of a static observer at radius r (> Rs) from a Schwarzschild
    black hole, exactly: for view angles theta from the hole's direction
    (0..pi), (the angle beta from that direction the light really came
    from, shadow flag, observed/emitted frequency).  A ray at angle theta
    has impact parameter b = r sin(theta)/sqrt(1-Rs/r); looking toward the
    hole (theta < pi/2) the photon is outgoing at the observer.  The
    frequency factor is the uniform 1/sqrt(1-Rs/r) of a static observer."""
    f = 1.0 - Rs/r
    if f <= 0.0:
        return [(0.0, 1.0, 1.0)] * N
    sf = sqrt(f)
    out = []
    for i in range(N):
        th = pi*(i + 0.5)/N
        beta, shadow = _photon_source(Rs, r, r*sin(th)/sf, th < 0.5*pi)
        out.append((beta, shadow, 1.0/sf))
    return out


def rain_table(Rs, r, N=256):
    """The sky of a 'rain' observer (radial free fall from rest at
    infinity) at radius r, inside or outside the horizon, in the same form
    as hole_table.  The rain frame is regular across the horizon, where no
    static observer exists.  In Painleve-Gullstrand coordinates the rain
    observer moves at beta = sqrt(Rs/r); a photon of unit energy at
    infinity with impact parameter b and radial momentum p = +-sqrt(1 -
    b^2 f/r^2) has, in the rain frame,
        radial momentum p1 = (beta + p)/f,  energy E = (1 + beta p)/f,
    which for an ingoing photon (p < 0) is better written without the
    cancellation at f = 0 as
        p1 = (b^2/r^2 - 1)/(beta - p),  E = (1 + b^2 beta^2/r^2)/(1 - beta p).
    The photon is seen at cos(theta) = p1/E from the hole's direction.  The
    table is built by sampling b on each branch and interpolating onto the
    uniform theta grid; directions no photon from infinity reaches are the
    shadow.  Inside the horizon only ingoing photons exist and the sky
    from infinity fills the cone cos(theta) < sqrt(r/Rs) away from the
    hole, its edge (b near the capture value) ever more blueshifted."""
    f = 1.0 - Rs/r
    bv = sqrt(Rs/r)
    bc = 1.5*sqrt(3.0)*Rs
    M = 6*N
    branches = []
    # the ingoing branch: b from 0 up to the tangential photon (outside) or
    # to the limiting direction (inside, b -> infinity)
    if f > 0.0:
        bmax = r/sqrt(f)*(1.0 - 1e-12)
    else:
        bmax = 2.0e3*r
    bs = sorted(set([bmax*(i/float(M))**2 for i in range(M + 1)] +
                    [bc*(1.0 + sg*10.0**-k) for k in range(1, 10) for sg in (-1.0, 1.0)
                     if 0.0 < bc*(1.0 + sg*10.0**-k) < bmax]))
    samp = []
    for b in bs:
        pr = -sqrt(max(1.0 - b*b*f/(r*r), 0.0))
        p1 = (b*b/(r*r) - 1.0)/(bv - pr)
        E = (1.0 + b*b*bv*bv/(r*r))/(1.0 - bv*pr)
        beta, shadow = _photon_source(Rs, r, b, False)
        samp.append((max(-1.0, min(1.0, p1/E)), beta, shadow, E))
    branches.append(samp)
    if f > 0.0:
        # the outgoing branch, b from 0 (straight out of the hole) to the
        # tangential photon
        bs = sorted(set([bmax*(i/float(M)) for i in range(M + 1)] +
                        [bc*(1.0 + sg*10.0**-k) for k in range(1, 10) for sg in (-1.0, 1.0)
                         if 0.0 < bc*(1.0 + sg*10.0**-k) < bmax]))
        samp = []
        for b in bs:
            pr = sqrt(max(1.0 - b*b*f/(r*r), 0.0))
            p1 = (bv + pr)/f
            E = (1.0 + bv*pr)/f
            beta, shadow = _photon_source(Rs, r, b, True)
            samp.append((max(-1.0, min(1.0, p1/E)), beta, shadow, E))
        branches.append(samp)
    out = [None]*N
    for samp in branches:
        for (c0, b0, s0, e0), (c1, b1, s1, e1) in zip(samp[:-1], samp[1:]):
            th0, th1 = acos(c0), acos(c1)
            lo, hi = min(th0, th1), max(th0, th1)
            i0 = int(lo/pi*N - 0.5 + 1.0)
            i1 = int(hi/pi*N - 0.5)
            for i in range(max(i0, 0), min(i1, N - 1) + 1):
                th = pi*(i + 0.5)/N
                t = (th - th0)/(th1 - th0) if hi > lo else 0.0
                if s0 > 0.5 or s1 > 0.5:
                    near = (s0, b0, e0) if t < 0.5 else (s1, b1, e1)
                    out[i] = (near[1], near[0], near[2])
                else:
                    out[i] = (b0 + t*(b1 - b0), 0.0, e0 + t*(e1 - e0))
    return [row if row is not None else (0.0, 1.0, 1.0) for row in out]


def inverse_rows(rows, N=None):
    """A sky table inverted, on two branches: for source angles beta on a
    uniform grid over 0..pi, the view angle of the primary image, and of
    the secondary image (the ray went round the far side of the hole, so
    that image lies on the opposite side, between the Einstein ring and the
    shadow).  Walking the table inward from theta = pi (beta = pi, no
    deflection) beta falls monotonically: through 0 at the Einstein ring
    (end of the primary branch), on to -pi (the secondary branch, as -beta)
    and further round (higher orders, left out)."""
    n = len(rows)
    N = N or n
    prim, sec = [], []          # (beta, theta) along each branch
    last = None
    for i in range(n - 1, -1, -1):
        beta, shadow = rows[i][0], rows[i][1]
        th = pi*(i + 0.5)/n
        if shadow > 0.5 or (last is not None and beta >= last):
            break
        last = beta
        if beta >= 0.0:
            prim.append((beta, th))
        elif beta >= -pi:
            sec.append((-beta, th))
        else:
            break
    def grid(pts, fallback):
        if len(pts) < 2:
            return [fallback(j) for j in range(N)]
        pts = sorted(pts)
        bs = [b for b, _ in pts]; ts = [t for _, t in pts]
        out = []
        for j in range(N):
            b = pi*(j + 0.5)/N
            if b <= bs[0]:
                out.append(ts[0]); continue
            if b >= bs[-1]:
                out.append(ts[-1]); continue
            lo, hi = 0, len(bs) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if bs[mid] <= b:
                    lo = mid
                else:
                    hi = mid
            f = (b - bs[lo]) / max(bs[hi] - bs[lo], 1e-30)
            out.append(ts[lo] + f*(ts[hi] - ts[lo]))
        return out
    p_ = grid(prim, lambda j: pi*(j + 0.5)/N)
    # beyond the table's last primary point the source is undeflected
    s_ = grid(sec, lambda j: 0.0)
    return p_, s_


class Lensing(object):
    THRESHOLD = 2.5e-5      # 2 R_s / D: the Einstein radius is 0.3 degrees here
    TABLE_N = 256
    # Within this many Schwarzschild radii of a black hole the frame the
    # ship's velocity, the sky and the bodies are reckoned in is no longer
    # the static one (whose boost diverges at the horizon, where it ceases
    # to exist) but the 'rain' frame of radial free fall from rest at
    # infinity, regular all the way down (see rain_table, Player._geodesic_step).
    RAIN = 1.001
    BODY_KMIN, BODY_KMAX = 0.05, 0.1   # R_s/D range over which the thin-lens bending of bodies fades out
    _table_tex = None
    _inv_tex = None
    _inv2_tex = None
    _table_key = None
    _inv_rows = None       # the inverse table of the current hole (markers use it)

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
                           cap=2.598 if body.is_hole else cls.capture(body.radius, Rs),
                           rain=body.is_hole and D < cls.RAIN*Rs)

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
        if c is None or c["rain"] or Ds <= c["D"] + 1.5*c["R"]:
            return 0.0
        w = (Ds - c["D"]) / Ds
        if c["is_hole"]:
            return w            # bent with the exact table, like the sky
        # A star's thin lens (small deflection) is only good while the lens
        # is weak as seen from the ship, R_s/D well below 1; the bending of
        # bodies fades out between R_s/D = 0.05 and 0.1.
        k = c["Rs"] / c["D"]
        fade = max(0.0, min(1.0, (cls.BODY_KMAX - k) / (cls.BODY_KMAX - cls.BODY_KMIN)))
        return w * fade

    @classmethod
    def apparent(cls, v, w):
        """Python twin of the vertex shader's bending, for things drawn on
        the CPU (the navigation markers): v = unit direction toward a source
        as the ship sees it, w its fraction of the deflection (weight());
        returns the unit direction of its primary image."""
        c = cls.current
        if c is None or w <= 0.0:
            return v
        nx, ny, nz = c["n"]; u = c["u"]; inv = c["inv"]
        k = c["Rs"] / c["D"]
        hx, hy, hz = c["h"]
        # to the background frame (lens_to_lab)
        vx, vy, vz = v
        cc = vx*nx + vy*ny + vz*nz
        tpv = 0.5*((vx-nx)**2 + (vy-ny)**2 + (vz-nz)**2) if cc > 0.0 else 1.0 - cc
        vpar = -cc
        px, py, pz = -vx - vpar*nx, -vy - vpar*ny, -vz - vpar*nz
        a = inv*vpar + u*tpv
        t = inv + u*tpv
        dx, dy, dz = -(px + a*nx)/t, -(py + a*ny)/t, -(pz + a*nz)/t
        m = (dx*dx + dy*dy + dz*dz) ** 0.5
        dx, dy, dz = dx/m, dy/m, dz/m
        # the primary image (lens_apparent)
        cb = max(-1.0, min(1.0, dx*hx + dy*hy + dz*hz))
        beta = math.acos(cb)
        ex, ey, ez = dx - cb*hx, dy - cb*hy, dz - cb*hz
        el = (ex*ex + ey*ey + ez*ez) ** 0.5
        if el >= 1e-6 and c["is_hole"] and cls._inv_rows is not None:
            ex, ey, ez = ex/el, ey/el, ez/el
            tab = cls._inv_rows; N = len(tab)
            x = beta/pi*N - 0.5
            j = max(0, min(N - 2, int(x))); f = max(0.0, min(1.0, x - j))
            thx = tab[j] + f*(tab[j + 1] - tab[j])
            th = beta + (thx - beta)*w
            ct, st = math.cos(th), math.sin(th)
            dx, dy, dz = hx*ct + ex*st, hy*ct + ey*st, hz*ct + ez*st
        elif el >= 1e-6:
            ex, ey, ez = ex/el, ey/el, ez/el
            def alpha(th):
                st = max(math.sin(th), 1e-6)
                return (2.0*k/st + 2.945*k*k/(st*st)) * 0.5*(1.0 + math.cos(th)) * w
            th = 0.5*(beta + (beta*beta + 8.0*k*w) ** 0.5)
            for _ in range(4):
                f = th - alpha(th) - beta
                fp = 1.0 - (alpha(th + 1e-4) - alpha(th - 1e-4)) * 5000.0
                th -= f / max(fp, 0.2)
                th = max(th, 1e-5)
            ct, st = math.cos(th), math.sin(th)
            dx, dy, dz = hx*ct + ex*st, hy*ct + ey*st, hz*ct + ez*st
        # back to the ship (lens_to_ship)
        cc = dx*nx + dy*ny + dz*nz
        tmv = 0.5*((dx+nx)**2 + (dy+ny)**2 + (dz+nz)**2) if cc < 0.0 else 1.0 + cc
        vpar = -cc
        px, py, pz = -dx - vpar*nx, -dy - vpar*ny, -dz - vpar*nz
        a = inv*vpar - u*tmv
        t = inv + u*tmv
        vx, vy, vz = -(px + a*nx)/t, -(py + a*ny)/t, -(pz + a*nz)/t
        m = (vx*vx + vy*vy + vz*vz) ** 0.5
        return (vx/m, vy/m, vz/m)

    @classmethod
    def secondary(cls, w):
        """Whether a body with bending fraction w gets its secondary image:
        a black hole is the lens and the body is far beyond it."""
        c = cls.current
        return c is not None and c["is_hole"] and not c["rain"] and w > 0.5 and cls._inv_rows is not None

    @classmethod
    def bind_table(cls, unit=4):
        """For a black hole lens, build (when the distance changed by more
        than 0.3%) and bind the exact sky table on texture unit `unit`."""
        c = cls.current
        if c is None or not c["is_hole"]:
            return False
        key = (c["body"].name, round(c["D"] / c["Rs"], 3) if c["D"] < 1e3*c["Rs"] else None, c["rain"])
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
            table = rain_table if c["rain"] else hole_table
            rows = table(c["Rs"], c["D"], cls.TABLE_N)
            flat = []
            for beta, shadow, freq in rows:
                flat += [beta, shadow, freq, 1.0]
            glBindTexture(GL_TEXTURE_1D, cls._table_tex)
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, cls.TABLE_N, 0, GL_RGBA, GL_FLOAT, (GLfloat*len(flat))(*flat))
            # the inverse, for the bodies' images: primary and secondary
            inv, inv2 = inverse_rows(rows, cls.TABLE_N)
            cls._inv_rows = inv
            for attr, data in (("_inv_tex", inv), ("_inv2_tex", inv2)):
                if getattr(cls, attr) is None:
                    setattr(cls, attr, glGenTextures(1))
                    glBindTexture(GL_TEXTURE_1D, getattr(cls, attr))
                    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glBindTexture(GL_TEXTURE_1D, getattr(cls, attr))
                flat = []
                for th in data:
                    flat += [th, 0.0, 0.0, 1.0]
                glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, cls.TABLE_N, 0, GL_RGBA, GL_FLOAT, (GLfloat*len(flat))(*flat))
            cls._table_key = key
        if key[1] is None:
            return False
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_1D, cls._table_tex)
        glActiveTexture(GL_TEXTURE0 + unit + 1)
        glBindTexture(GL_TEXTURE_1D, cls._inv_tex)
        glActiveTexture(GL_TEXTURE0 + unit + 2)
        glBindTexture(GL_TEXTURE_1D, cls._inv2_tex)
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
            glUniform1i(glGetUniformLocation(program, "lens_tinv"), table_unit + 1)
            glUniform1i(glGetUniformLocation(program, "lens_tinv2"), table_unit + 2)
            glUniform1f(glGetUniformLocation(program, "lens_branch"), 0.0)
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
