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
from program.utils import kerr as kerrlib
from program.utils.kerr import GLSL as KERR_GLSL

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
float lens_gs = 1.0;             // set by lens_source: observed/emitted frequency of the sky in that direction (exact table only)
""" + KERR_GLSL + """
uniform sampler1D lens_tinv;     // source angle (0..pi) -> view angle of its primary image (the table inverted)
uniform sampler1D lens_tinv2;    // source angle (0..pi) -> view angle of its secondary image (wrapped round the far side)
uniform float lens_branch;       // 0: primary image; 1: secondary (a second pass over the body)
uniform float lens_D;            // distance of the lens from the ship (ship-frame units of the vertices)
uniform sampler2D lens_ray;      // (theta, log rho) -> (sweep on the inbound leg, sweep on the outbound leg), -1: none
uniform float lens_rs;           // the hole's Schwarzschild radius, same units
uniform float lens_rhomax;       // the table's largest rho over R_s (log grid from R_s)
uniform float lens_rhon;         // rows of the 2-D table (texel centres at (j + 0.5)/rows)
uniform float lens_finite;       // 1: the 2-D table is bound
// the sweep of the ray at view angle th when it reaches radius rho (0 < v < 1 along the log-rho grid)
vec2 lens_sweep(float th, float v) {
    return texture2D(lens_ray, vec2(th * 0.31830988618379067, v)).rg;
}
// view angle th of the image of a source at radius rho (log-grid coordinate v)
// and sweep angle target, on leg 0 (inbound, sweep rising with th, defined
// from th = 0 to the tangent ray) or leg 1 (outbound, sweep falling with th,
// defined from the shadow's edge to the tangent ray or to pi); bisection on
// that domain.  -1 when the target lies outside the leg's range; with edge,
// leg 0 returns its tangent end when the target lies beyond it.
float lens_solve(int leg, float v, float target, bool edge) {
    vec4 row = texture2D(lens_ray, vec2(0.5, v));
    float lo, hi;
    if (leg == 0) {
        if (row.b < 0.0) return -1.0;
        lo = 0.0062; hi = row.b;                       // from the first texel's centre
    } else {
        if (row.a < 0.0) return -1.0;
        lo = row.a; hi = (row.b > 0.0) ? row.b : 3.1354;
    }
    vec2 slo = lens_sweep(lo, v), shi = lens_sweep(hi, v);
    float flo = (leg == 0) ? slo.r : slo.g;
    float fhi = (leg == 0) ? shi.r : shi.g;
    if (flo < 0.0 || fhi < 0.0) return -1.0;
    if ((flo - target) * (fhi - target) > 0.0) {
        if (edge && leg == 0 && fhi < target) return hi;
        return -1.0;
    }
    for (int i = 0; i < 14; i++) {
        float mid = 0.5*(lo + hi);
        vec2 sw = lens_sweep(mid, v);
        float f = (leg == 0) ? sw.r : sw.g;
        if (f < 0.0) { hi = mid; continue; }
        if ((flo - target) * (f - target) <= 0.0) { hi = mid; fhi = f; } else { lo = mid; flo = f; }
    }
    return 0.5*(lo + hi);
}
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
// the background-frame distance of a point the ship sees at unit distance in
// direction v (the ship-frame null vector (-1, v) boosted; |xyz| = |t|)
float lens_lab_dist(vec3 v) {
    float c = dot(v, lens_n);
    float tpv = (c > 0.0) ? 0.5*dot(v - lens_n, v - lens_n) : 1.0 - c;
    return abs(lens_inv + lens_u*tpv);
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
    if (lens_kerr > 0.5) return kerr_source(d);
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
// the image of a source at position p (background frame, ship at the origin)
// near the hole: its radius rho from the hole and the angle psi at the hole
// between ship and source select the ray; primary (branch 0) on the inbound
// leg if there is one, else the outbound; secondary (branch 1) round the far
// side.  Returns the image direction, or the zero vector when the 2-D
// table has nothing (the caller then falls back).
vec3 lens_apparent_near(vec3 p, float branch) {
    vec3 q = p - lens_D * lens_h;               // source relative to the hole
    float rho = length(q);
    if (rho <= lens_rs || rho >= lens_rs * lens_rhomax) return vec3(0.0);
    vec3 qh = q / rho;
    float cpsi = clamp(dot(-lens_h, qh), -1.0, 1.0);
    float psi = acos(cpsi);
    vec3 e = qh + cpsi * lens_h;                // the source's side, perpendicular to h
    float el = length(e);
    if (el < 1e-6) e = vec3(0.0); else e /= el;
    float v = (log(rho / lens_rs) / log(lens_rhomax) * (lens_rhon - 1.0) + 0.5) / lens_rhon;
    float th = -1.0;
    if (branch < 0.5) {
        th = lens_solve(0, v, psi, false);
        if (th < 0.0) th = lens_solve(1, v, psi, false);
        if (th < 0.0) {
            // between the legs' stored domains: the texel or two at the
            // tangent ray, where the sweep runs as the square root of the
            // distance to it (inbound below, outbound above): a parabola in
            // th through both ends rather than a jump
            vec4 row = texture2D(lens_ray, vec2(0.5, v));
            if (row.b > 0.0) {
                vec2 sB = lens_sweep(row.b, v);
                if (sB.r >= 0.0 && sB.g >= 0.0 && psi >= sB.r && psi <= sB.g) {
                    float s = (2.0*psi - sB.r - sB.g) / max(sB.g - sB.r, 1e-6);
                    float tt = row.b + 0.0245;              // the tangent ray, ~2 texels on
                    th = tt - (tt - row.b) * s * s;
                }
            }
        }
        if (th < 0.0) th = lens_solve(0, v, psi, true);
        if (th < 0.0) return vec3(0.0);
        return lens_h*cos(th) + e*sin(th);
    }
    th = lens_solve(1, v, 6.28318530717959 - psi, false);
    if (th < 0.0) return vec3(0.0);
    return lens_h*cos(th) - e*sin(th);
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
            // image is on the opposite side of the hole, inside the Einstein
            // ring.  Only for sources well beyond the hole; nearer ones go
            // to their primary place (the pass then just overdraws it)
            if (w < 0.3) {
                float thw = beta + (texture1D(lens_tinv, beta * 0.31830988618379067).r - beta) * w;
                return lens_h*cos(thw) + e*sin(thw);
            }
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
    _ray_tex = None
    _ray_rows = None       # (phi_in, phi_out, rho) of the current hole
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
        kerr = getattr(body, "is_kerr", False)
        cls.current = dict(body=body, D=D, Rs=Rs, R=body.radius, h=(dX.x/n, dX.y/n, dX.z/n),
                           is_hole=body.is_hole, n=bn, u=bu, inv=binv,
                           cap=(body.r_plus/Rs if kerr else 2.598) if body.is_hole else cls.capture(body.radius, Rs),
                           rain=body.is_hole and not kerr and D < cls.RAIN*Rs, kerr=kerr)
        if kerr:
            # the ZAMO's tetrad at the ship and the hole's frame (x toward
            # the ship's azimuth, z the spin), all in the background frame
            sp = kerrlib.spherical(body.kerr_basis, (-dX.x, -dX.y, -dX.z))
            if sp is None:
                cls.current = None
                return
            r, th, ph, er, et, ep = sp
            e1, e2, sax = body.kerr_basis
            cp, s_ = math.cos(ph), math.sin(ph)
            bx = tuple(cp*e1[i] + s_*e2[i] for i in range(3))
            by = tuple(-s_*e1[i] + cp*e2[i] for i in range(3))
            M = body.GM
            cls.current.update(kr=max(r, body.r_plus*1.002), kth=th, kph=ph, er=er, et=et, ep=ep,
                               bx=bx, by=by, bz=sax, t=Xp.t, M=M, a=body.kerr_a)

    @classmethod
    def enlarge(cls, body):
        """How much larger than its radius the body is drawn: its apparent
        size when it is the lens, 1 otherwise."""
        c = cls.current
        if c is None or c["body"] is not body or c["R"] <= 0.0:
            return 1.0
        if c["kerr"]:
            # the shadow is painted by the ray table, and the disc's lensed
            # images lie where a globe would be: no globe at all
            return 1.0e-3
        f = max(1.0 - c["Rs"] / c["D"], 0.0) ** 0.5     # as seen from D, not from infinity
        e = c["cap"] * c["Rs"] * f / c["R"]
        if c["is_hole"]:
            # the shadow itself is painted exactly by the sky; the globe only
            # hides what is behind the hole, so it never swallows the observer
            e = min(e, 0.9 * c["D"] / c["R"])
        return max(e, 1.0)

    @classmethod
    def wscale(cls):
        """Scale of the bending of bodies: the per-vertex fraction of the
        deflection, (d - D)/d for a vertex at distance d beyond the lens at
        D (0 at or inside it: the lens itself, the near side of its disc),
        is multiplied by this.  1 for a black hole (the exact table), 0 in
        the rain frame; for a star's thin lens (only good while the lens is
        weak as seen from the ship) it fades out between R_s/D = 0.05 and
        0.1."""
        c = cls.current
        if c is None or c["rain"]:
            return 0.0
        if c["is_hole"]:
            return 1.0
        k = c["Rs"] / c["D"]
        return max(0.0, min(1.0, (cls.BODY_KMAX - k) / (cls.BODY_KMAX - cls.BODY_KMIN)))

    @classmethod
    def weight(cls, Ds):
        """Fraction of the deflection seen by a point source at distance Ds
        (the markers): (Ds - D)/Ds times wscale()."""
        c = cls.current
        if c is None or Ds <= c["D"]:
            return 0.0
        return (Ds - c["D"]) / Ds * cls.wscale()

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
    def ship_dir(cls, d):
        """A background-frame direction as the ship sees it (lens_to_ship)."""
        c = cls.current
        nx, ny, nz = c["n"]; u = c["u"]; inv = c["inv"]
        dx, dy, dz = d
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
    def image_near(cls, p, branch=0):
        """numpy twin of the shader's lens_apparent_near: p = source position
        in the background frame with the ship at the origin; the image
        direction, or None when the near table has nothing for it."""
        c = cls.current
        if c is None or cls._ray_rows is None:
            return None
        import numpy as np
        phi_in, phi_out, rho = cls._ray_rows
        hx, hy, hz = c["h"]; D = c["D"]; Rs = c["Rs"]
        qx, qy, qz = p[0] - D*hx, p[1] - D*hy, p[2] - D*hz
        r = (qx*qx + qy*qy + qz*qz) ** 0.5
        if r <= Rs or r >= Rs*RHO_MAX:
            return None
        qx, qy, qz = qx/r, qy/r, qz/r
        cpsi = max(-1.0, min(1.0, -(hx*qx + hy*qy + hz*qz)))
        psi = math.acos(cpsi)
        ex, ey, ez = qx + cpsi*hx, qy + cpsi*hy, qz + cpsi*hz
        el = (ex*ex + ey*ey + ez*ez) ** 0.5
        ex, ey, ez = (ex/el, ey/el, ez/el) if el > 1e-6 else (0.0, 0.0, 0.0)
        # bilinear row at this rho (log grid)
        NR = rho.size; N = phi_in.shape[0]
        v = math.log(r / Rs) / math.log(RHO_MAX) * (NR - 1)
        j = max(0, min(NR - 2, int(v))); f = v - j
        def row(arr):
            return arr[:, j]*(1.0 - f) + arr[:, j + 1]*f
        th = np.pi*(np.arange(N) + 0.5)/N
        def solve(col, target):
            ok = col >= 0.0
            d = col - target
            for i in range(N - 1):
                if ok[i] and ok[i + 1] and d[i]*d[i + 1] <= 0.0:
                    t = d[i] / (d[i] - d[i + 1]) if d[i] != d[i + 1] else 0.0
                    return th[i] + t*(th[i + 1] - th[i])
            return None
        if branch == 0:
            t_ = solve(row(phi_in), psi)
            if t_ is None:
                t_ = solve(row(phi_out), psi)
            if t_ is None:
                col = row(phi_in); ok = col >= 0.0
                idx = np.where(ok)[0]
                if idx.size and col[idx[-1]] < psi:
                    # the gap at the tangent ray (as the shader: a parabola
                    # between the inbound leg's end and the outbound leg's)
                    i_ = idx[-1]; cout = row(phi_out)
                    t_ = th[i_]
                    if cout[i_] >= 0.0 and psi <= cout[i_]:
                        s_ = (2.0*psi - col[i_] - cout[i_]) / max(cout[i_] - col[i_], 1e-6)
                        tt = th[i_] + 2.0*np.pi/N
                        t_ = tt - (tt - th[i_])*s_*s_
            if t_ is None:
                return None
            ct, st = math.cos(t_), math.sin(t_)
            return (hx*ct + ex*st, hy*ct + ey*st, hz*ct + ez*st)
        t_ = solve(row(phi_out), 2.0*math.pi - psi)
        if t_ is None:
            return None
        ct, st = math.cos(t_), math.sin(t_)
        return (hx*ct - ex*st, hy*ct - ey*st, hz*ct - ez*st)

    @classmethod
    def near_table(cls):
        """Whether the 2-D ray table for sources near the hole is bound."""
        c = cls.current
        return c is not None and c["is_hole"] and not c["rain"] and cls._ray_rows is not None

    @classmethod
    def secondary(cls, wscale):
        """Whether bodies get a second pass for their secondary image: a
        black hole is the lens (the shader keeps the pass to the vertices
        well beyond the hole)."""
        c = cls.current
        return c is not None and c["is_hole"] and not c["rain"] and wscale > 0.0 and cls._inv_rows is not None

    _kerr_table = None

    @classmethod
    def kerr_init(cls):
        """Create the Kerr sky table's GL objects (shader, texture,
        framebuffer) now, outside any draw: made lazily inside the sky's
        draw they leave that first frame without the disc."""
        if cls._kerr_table is None:
            cls._kerr_table = kerrlib.KerrTable()
            cls._kerr_table._init_gl()
            glUseProgram(0)

    @classmethod
    def _bind_kerr(cls, unit):
        """Build the ZAMO's sky for the current Kerr lens (when the ship
        moved) and bind it on texture unit `unit`."""
        c = cls.current
        if cls._kerr_table is None:
            cls._kerr_table = kerrlib.KerrTable()
        body = c["body"]; M = c["M"]
        disc = None
        if body.disk is not None:
            r_in, r_out = body.disk.inner*c["Rs"]/M, body.disk.outer*c["Rs"]/M      # in M
            # the gas pattern turns at the Keplerian rate of three rings; the
            # phase is reduced in double precision here, the shader only adds
            # the light-travel part
            split = [r_in*(r_out/r_in) ** (k/3.0) for k in range(4)]
            omegas = [kerrlib.omega_orbit(1.0, c["a"]/M, math.sqrt(split[k]*split[k+1])) for k in range(3)]
            tM = c["t"] / M
            phases = [math.fmod(om*tM, 2.0*math.pi) for om in omegas]
            disc = (r_in, r_out, phases, omegas, (split[1], split[2]))
        cls._kerr_table.build(c["a"]/M, c["kr"]/M, c["kth"], disc)
        cls._kerr_table.bind(unit)

    @classmethod
    def bind_table(cls, unit=4):
        """For a black hole lens, build (when the distance changed by more
        than 0.3%) and bind the exact sky table on texture unit `unit`."""
        c = cls.current
        if c is None or not c["is_hole"]:
            return False
        if c["kerr"]:
            cls._bind_kerr(unit + 4)
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
            # the 2-D ray table for sources near the hole
            rt = ray_table(c["Rs"], c["D"], cls.TABLE_N) if not c["rain"] else None
            cls._ray_rows = rt
            if rt is not None:
                import numpy as np
                phi_in, phi_out, rho = rt
                data = np.zeros((phi_in.shape[1], phi_in.shape[0], 4), dtype=np.float32)   # rows = rho, cols = theta
                data[:, :, 0] = phi_in.T; data[:, :, 1] = phi_out.T
                # per row, the legs' domains in theta (row-constant): B = the
                # inbound leg's last theta (the tangent ray; -1: none), A =
                # the outbound leg's first theta (the shadow's edge; -1: none)
                # ...conservative: the tightest of the row and its neighbours,
                # a texel inside, so that a bilinear sample at the limit (which
                # blends two rows) is still defined in both
                N_ = phi_in.shape[0]; thg = np.pi*(np.arange(N_) + 0.5)/N_; dth = np.pi/N_
                NR_ = phi_in.shape[1]
                last_in = np.array([thg[np.where(phi_in[:, j] >= 0.0)[0][-1]] if (phi_in[:, j] >= 0.0).any() else -1.0 for j in range(NR_)])
                first_out = np.array([thg[np.where(phi_out[:, j] >= 0.0)[0][0]] if (phi_out[:, j] >= 0.0).any() else 9.0 for j in range(NR_)])
                for j in range(NR_):
                    nb = slice(max(0, j - 1), min(NR_, j + 2))
                    b = last_in[nb].min(); a = first_out[nb].max()
                    data[j, :, 2] = (b - dth) if b > 0.0 and (b - dth) > 2.0*dth else -1.0
                    data[j, :, 3] = (a + dth) if a < 8.0 else -1.0
                glActiveTexture(GL_TEXTURE0 + unit + 3)       # never on unit 0: the sky's texture lives there
                if cls._ray_tex is None:
                    cls._ray_tex = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, cls._ray_tex)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glBindTexture(GL_TEXTURE_2D, cls._ray_tex)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, phi_in.shape[0], phi_in.shape[1], 0, GL_RGBA, GL_FLOAT, data)
                glActiveTexture(GL_TEXTURE0)
            cls._table_key = key
        if key[1] is None:
            return False
        if cls._ray_rows is not None:
            glActiveTexture(GL_TEXTURE0 + unit + 3)
            glBindTexture(GL_TEXTURE_2D, cls._ray_tex)
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
            glUniform1i(glGetUniformLocation(program, "lens_ray"), table_unit + 3)
            glUniform1i(glGetUniformLocation(program, "lens_tinv2"), table_unit + 2)
            glUniform1i(glGetUniformLocation(program, "lens_ray"), table_unit + 3)
            glUniform1i(glGetUniformLocation(program, "kerr_table"), table_unit + 4)
            glUniform1i(glGetUniformLocation(program, "kerr_disc"), table_unit + 5)
            glUniform1f(glGetUniformLocation(program, "lens_branch"), 0.0)
        kloc = glGetUniformLocation(program, "lens_kerr")
        if kloc >= 0:
            glUniform1f(kloc, 1.0 if (c is not None and c["kerr"] and table_unit is not None and cls._kerr_table is not None and cls._kerr_table.ok) else 0.0)
        if c is None or w <= 0.0:
            glUniform1f(glGetUniformLocation(program, "lens_k"), 0.0)
            if loc >= 0:
                glUniform1f(loc, 0.0)
            return
        if c["kerr"] and kloc >= 0 and table_unit is not None:
            body = c["body"]
            if body.disk is not None:
                glUniform1f(glGetUniformLocation(program, "kerr_rin"), body.disk.inner*c["Rs"]/c["M"])
                glUniform1f(glGetUniformLocation(program, "kerr_rout"), body.disk.outer*c["Rs"]/c["M"])
            else:
                glUniform1f(glGetUniformLocation(program, "kerr_rin"), 0.0)
            for name in ("er", "et", "ep", "bx", "by", "bz"):
                glUniform3f(glGetUniformLocation(program, "kerr_" + name), *c[name])
            if body.disk is not None:
                dloc = glGetUniformLocation(program, "kd_rin")
                if dloc >= 0:
                    glUniform1f(dloc, body.disk.inner*c["Rs"]/c["M"])
                    glUniform1f(glGetUniformLocation(program, "kd_rout"), body.disk.outer*c["Rs"]/c["M"])
                    glUniform1f(glGetUniformLocation(program, "kd_Tin"), body.disk.temperature)
                    glUniform1f(glGetUniformLocation(program, "kd_bin"), body.disk.brightness)
        glUniform3f(glGetUniformLocation(program, "lens_h"), *c["h"])
        glUniform1f(glGetUniformLocation(program, "lens_k"), c["Rs"] / c["D"])
        glUniform3f(glGetUniformLocation(program, "lens_n"), *c["n"])
        glUniform1f(glGetUniformLocation(program, "lens_u"), c["u"])
        glUniform1f(glGetUniformLocation(program, "lens_inv"), c["inv"])
        exact = table_unit is not None and cls.bind_table(table_unit)
        if c["kerr"] and kloc >= 0 and table_unit is not None:
            # after the build: the row window the table was traced with
            glUniform1f(glGetUniformLocation(program, "kerr_win"), cls._kerr_table.win)
        glUniform1f(glGetUniformLocation(program, "lens_exact"), 1.0 if exact else 0.0)
        glUniform1f(glGetUniformLocation(program, "lens_D"), c["D"])
        glUniform1f(glGetUniformLocation(program, "lens_rs"), c["Rs"])
        glUniform1f(glGetUniformLocation(program, "lens_rhomax"), RHO_MAX)
        glUniform1f(glGetUniformLocation(program, "lens_rhon"), float(RHO_N))
        glUniform1f(glGetUniformLocation(program, "lens_finite"), 1.0 if (exact and cls._ray_rows is not None) else 0.0)
        if loc >= 0:
            glUniform1f(loc, w)


# ----------------------------------------------------------- finite sources
# For bodies near the hole (its disc, its planets) a source is at a finite
# radius rho from the hole, and its image is found by connecting it to the
# observer with a real light path: ray_table gives, for each view angle
# theta and radius rho, the polar angle the ray has swept (about the hole,
# from the observer) when it reaches rho, on its inbound leg (before
# periapsis) and on its outbound leg.  The vertex shader then solves
# sweep(theta, rho) = psi for theta, psi being the angle at the hole
# between the observer and the source (primary image), and
# sweep_out(theta, rho) = 2 pi - psi for the secondary image round the
# far side.
RHO_N = 128
RHO_MAX = 80.0        # in Schwarzschild radii; farther sources use the 1-D tables


def ray_table(Rs, r_o, N=256, NR=RHO_N, rho_max=RHO_MAX):
    """numpy: arrays phi_in[N, NR], phi_out[N, NR] (radians; -100 where the
    leg does not reach that radius) on a log grid of rho from Rs to
    rho_max Rs, for a static observer at r_o."""
    import numpy as np
    f = 1.0 - Rs/r_o
    if f <= 0.0:
        return None
    sf = math.sqrt(f)
    bc = 1.5*math.sqrt(3.0)*Rs
    th = np.pi*(np.arange(N) + 0.5)/N
    b = r_o*np.sin(th)/sf
    u_o = 1.0/r_o
    rho = Rs*np.exp(np.log(rho_max)*np.arange(NR)/(NR - 1.0))      # Rs .. rho_max Rs
    u_grid = 1.0/rho                                                 # decreasing
    # undefined: -100, so a bilinear blend with a real value stays negative
    phi_in = -100.0*np.ones((N, NR)); phi_out = -100.0*np.ones((N, NR))
    M = 400
    s = (np.arange(M) + 0.5)/M
    def g(u, bb):
        return 1.0/(bb*bb) - u*u + Rs*u*u*u
    def periapsis(bb):
        u = 1.0/bb
        for _ in range(60):
            gu = g(u, bb); dg = -2.0*u + 3.0*Rs*u*u
            if abs(dg) < 1e-300: break
            un = u - gu/dg
            if un <= 0.0: un = 0.5*u
            if abs(un - u) < 1e-14*max(u, 1e-300):
                u = un; break
            u = un
        return u
    for i in range(N):
        bb = b[i]
        if th[i] < 0.5*np.pi:
            if bb > bc*(1.0 + 1e-9):
                up = periapsis(bb)
                if up < u_o:          # periapsis outside the observer: no such ray
                    continue
                # inbound: u from u_o to u_p, u = u_p - (u_p - u_o) s^2 (regular at u_p)
                uu = up - (up - u_o)*s*s
                integ = 2.0*(up - u_o)*s/np.sqrt(np.maximum(g(uu, bb), 1e-300))
                # cumulative from s = 0 (u = u_p) ... we want from u_o (s = 1): integrate from s=1 down
                cum = np.cumsum(integ[::-1])/M            # from s=1 (u_o) toward s=0 (u_p)
                phi_u = cum[::-1]                          # at each s: sweep from u_o to that u
                # at grid radii between r_o and r_p (u_o <= u <= u_p)
                mask = (u_grid >= u_o) & (u_grid <= up)
                phi_in[i, mask] = np.interp(u_grid[mask], uu[::-1], phi_u[::-1])
                tot_in = cum[-1]
                # outbound: u from u_p down to 0, u = u_p (1 - t^2)
                uu2 = up*(1.0 - s*s)
                integ2 = 2.0*up*s/np.sqrt(np.maximum(g(uu2, bb), 1e-300))
                cum2 = np.cumsum(integ2)/M                 # from u_p outward
                mask2 = u_grid <= up
                phi_out[i, mask2] = tot_in + np.interp(u_grid[mask2], uu2[::-1], cum2[::-1])
            else:
                # plunging: inbound from u_o to the horizon u = 1/Rs, no periapsis
                uh = 1.0/Rs
                uu = u_o + (uh - u_o)*s
                integ = (uh - u_o)/np.sqrt(np.maximum(g(uu, bb), 1e-300))
                phi_u = np.cumsum(integ)/M
                mask = (u_grid >= u_o) & (u_grid <= uh)
                phi_in[i, mask] = np.interp(u_grid[mask], uu, phi_u)
        else:
            # outbound from the start: u from u_o down to 0
            uu = u_o*(1.0 - s)
            integ = u_o/np.sqrt(np.maximum(g(uu, bb), 1e-300))
            phi_u = np.cumsum(integ)/M
            mask = u_grid <= u_o
            phi_out[i, mask] = np.interp(u_grid[mask], uu[::-1], phi_u[::-1])
    return phi_in, phi_out, rho
