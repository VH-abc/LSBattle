# coding: utf8
# utils/boost.py
"""
Numerically stable Lorentz boosts for the shaders.

At Lorentz factors in the millions a 4x4 boost matrix in float32 is useless:
its entries are ~gamma and the visible forward cone is ~1/gamma wide, so
every product cancels to rounding noise and the picture jitters.  Instead
each boost is written in terms of the velocity direction n, u = |gamma*beta|,
gamma and inv = 1/(gamma+u) = gamma - u (all computed in double on the CPU)
plus one small quantity, t -/+ v_par, which the caller forms without
cancellation.  Components perpendicular to n are untouched by the boost.
"""
from OpenGL.GL import *

from go import Vector4D

GLSL = """
// Lorentz(U) v : t' = g t - u vpar ; vpar' = g vpar - u t   (tmv = t - vpar)
vec4 lz_pos(vec4 v, float tmv, vec3 n, float u, float inv) {
    float vpar = dot(v.xyz, n);
    vec3 perp = v.xyz - vpar*n;
    return vec4(perp + (inv*vpar - u*tmv)*n, inv*v.w + u*tmv);
}
// Lorentz(-U) v : t' = g t + u vpar ; vpar' = g vpar + u t  (tpv = t + vpar)
vec4 lz_neg(vec4 v, float tpv, vec3 n, float u, float inv) {
    float vpar = dot(v.xyz, n);
    vec3 perp = v.xyz - vpar*n;
    return vec4(perp + (inv*vpar + u*tpv)*n, inv*v.w + u*tpv);
}
// past-directed null vector v = (r d, -r), d unit:
float tmv_null(vec3 d, float r, vec3 n) {    // t - vpar = -r (1 + c)
    float c = dot(d, n);
    if (c < 0.0) { vec3 e = d + n; return -0.5*r*dot(e, e); }
    return -r*(1.0 + c);
}
float tpv_null(vec3 d, float r, vec3 n) {    // t + vpar = -r (1 - c)
    float c = dot(d, n);
    if (c > 0.0) { vec3 e = d - n; return -0.5*r*dot(e, e); }
    return -r*(1.0 - c);
}
uniform vec3 pb_n; uniform float pb_u, pb_g, pb_inv;   // player's 4-velocity
"""

def params(U):
    """(n, u, g, inv) of a 4-velocity, computed in double."""
    ux, uy, uz = U.x, U.y, U.z
    u = (ux*ux + uy*uy + uz*uz) ** 0.5
    g = (1.0 + u*u) ** 0.5
    if u > 0.0:
        n = (ux/u, uy/u, uz/u)
    else:
        n = (0.0, 0.0, 1.0)
    return n, u, g, 1.0/(g + u)

def set_uniforms(program, U, prefix="pb"):
    n, u, g, inv = params(U)
    glUniform3f(glGetUniformLocation(program, prefix + "_n"), n[0], n[1], n[2])
    glUniform1f(glGetUniformLocation(program, prefix + "_u"), u)
    glUniform1f(glGetUniformLocation(program, prefix + "_g"), g)
    glUniform1f(glGetUniformLocation(program, prefix + "_inv"), inv)

def U_of_L(L):
    """4-velocity U from the matrix Lorentz(U): its first column is (g, -u)."""
    v = L.get_transform([1.0, 0.0, 0.0, 0.0])
    return Vector4D(v.t, -v.x, -v.y, -v.z)


def transform_null(U, X):
    """
    Lorentz(U) * X for a past-directed null vector X (X.t == -|X.xyz|), in
    double precision and without catastrophic cancellation at large gamma.
    Returns a new Vector4D.
    """
    n, u, g, inv = params(U)
    x, y, z = X.x, X.y, X.z
    r = (x*x + y*y + z*z) ** 0.5
    if r == 0.0:
        return Vector4D(X.t, 0.0, 0.0, 0.0)
    par = x*n[0] + y*n[1] + z*n[2]
    c = par / r
    if c < 0.0:
        ex, ey, ez = x/r + n[0], y/r + n[1], z/r + n[2]
        tmv = -0.5*r*(ex*ex + ey*ey + ez*ez)      # t - par = -r(1+c)
    else:
        tmv = -r*(1.0 + c)
    t = -r
    tp = inv*t + u*tmv
    pp = inv*par - u*tmv
    return Vector4D(tp, x + (pp - par)*n[0], y + (pp - par)*n[1], z + (pp - par)*n[2])
