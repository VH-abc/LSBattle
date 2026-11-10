# coding: utf8
# utils/blackbody.py
"""
Physically based Doppler colour/brightness for the "accurate" colour mode.

Model: every texel is treated as a grey-body emitter.  Its colour fixes a
blackbody temperature T0 (from the blue/red ratio on the Planckian locus)
and its luminance fixes an emissivity.  A photon with Doppler factor D
(observed / emitted frequency) from a blackbody at T0 is seen exactly as a
blackbody at D*T0, with spectral radiance Planck(D*T0) -- the full
D^3 I_nu transformation is built in.  So the observed texel is

    out = (texel_linear / Planck(T0)) * Planck(D*T0) * exposure

where Planck(T) is the linear-sRGB colour of a blackbody at T integrated
over the CIE 1931 observer.  For a true blackbody texel the ratio in front
is its emissivity; for anything else it preserves the texel's departure
from the locus.  Beyond the table, the Rayleigh-Jeans limit (luminance
linear in T, fixed hue) and the Wien limit (black) are used.

Two 1D float lookup textures are built:
  LUT_P: log T   -> (r, g, b, Y) linear sRGB of a blackbody, Y(6500K) = 1
  LUT_T: ln(b/r) -> log T   (the inverse map for estimating T0)
"""
from math import exp, log, log10, sqrt

from OpenGL.GL import *

T_MIN, T_MAX = 500.0, 1.0e7        # Kelvin
N = 512

# Wyman, Sloan, Shirley 2013: multi-lobe Gaussian fit of CIE 1931 2-deg CMFs.
def _g(x, mu, s1, s2):
    t = (x - mu) / (s1 if x < mu else s2)
    return exp(-0.5*t*t)
def xbar(l):
    return 1.056*_g(l, 599.8, 37.9, 31.0) + 0.362*_g(l, 442.0, 16.0, 26.7) - 0.065*_g(l, 501.1, 20.4, 26.2)
def ybar(l):
    return 0.821*_g(l, 568.8, 46.9, 40.5) + 0.286*_g(l, 530.9, 16.3, 31.1)
def zbar(l):
    return 1.217*_g(l, 437.0, 11.8, 36.0) + 0.681*_g(l, 459.0, 26.0, 13.8)

_H, _C, _K = 6.62607015e-34, 2.99792458e8, 1.380649e-23
def planck(l_nm, T):
    l = l_nm*1e-9
    x = _H*_C/(l*_K*T)
    if x > 700.0:
        return 0.0
    return 1.0/(l**5 * (exp(x) - 1.0))

_LAMBDAS = [380.0 + 5.0*i for i in range(81)]
_CMF = [(xbar(l), ybar(l), zbar(l)) for l in _LAMBDAS]

def xyz_of_blackbody(T):
    X = Y = Z = 0.0
    for l, (xb, yb, zb) in zip(_LAMBDAS, _CMF):
        b = planck(l, T)
        X += b*xb; Y += b*yb; Z += b*zb
    return X, Y, Z

def xyz_to_linear_srgb(X, Y, Z):
    r =  3.2406*X - 1.5372*Y - 0.4986*Z
    g = -0.9689*X + 1.8758*Y + 0.0415*Z
    b =  0.0557*X - 0.2040*Y + 1.0570*Z
    return max(r, 0.0), max(g, 0.0), max(b, 0.0)

def planck_rgb(T):
    """Linear-sRGB colour of a blackbody at T, in the LUT's normalisation
    (Y(6500 K) = 1)."""
    _, Yref, _ = xyz_of_blackbody(6500.0)
    X, Y, Z = xyz_of_blackbody(T)
    return xyz_to_linear_srgb(X/Yref, Y/Yref, Z/Yref)

_tables = None
def tables():
    """-> (logT list, [(r,g,b,Y)] normalized so Y(6500K)=1, q list = ln(b/r))"""
    global _tables
    if _tables is not None:
        return _tables
    _, Yref, _ = xyz_of_blackbody(6500.0)
    lmin, lmax = log10(T_MIN), log10(T_MAX)
    logT = [lmin + (lmax-lmin)*i/(N-1) for i in range(N)]
    P = []
    for lt in logT:
        X, Y, Z = xyz_of_blackbody(10.0**lt)
        r, g, b = xyz_to_linear_srgb(X/Yref, Y/Yref, Z/Yref)
        P.append((r, g, b, Y/Yref))
    eps = 1e-30
    q = [log((b+eps)/(r+eps)) for (r, g, b, Y) in P]
    # enforce monotonic (blue vanishes at the cold end, flattens at the hot end)
    for i in range(1, N):
        if q[i] <= q[i-1]:
            q[i] = q[i-1] + 1e-9
    _tables = (logT, P, q)
    return _tables

def q_to_logT_table():
    """Inverse map sampled uniformly in q over [q(T_MIN), q(T_MAX)]."""
    logT, P, q = tables()
    qmin, qmax = q[0], q[-1]
    out = []
    for i in range(N):
        qq = qmin + (qmax-qmin)*i/(N-1)
        # q is monotonic increasing in T; binary search
        lo, hi = 0, N-1
        while hi - lo > 1:
            mid = (lo+hi)//2
            if q[mid] < qq: lo = mid
            else: hi = mid
        f = (qq - q[lo]) / (q[hi]-q[lo]) if q[hi] != q[lo] else 0.0
        out.append(logT[lo] + f*(logT[hi]-logT[lo]))
    return out, qmin, qmax

_gl = {}
def gl_textures():
    """-> (tex_P, tex_T, uniforms dict). Builds the GL 1D float textures once per context."""
    if _gl and glIsTexture(_gl["P"]):
        return _gl["P"], _gl["T"], _gl["u"]
    logT, P, q = tables()
    inv, qmin, qmax = q_to_logT_table()
    flat = []
    for r, g, b, Y in P:
        flat += [r, g, b, Y]
    texP = glGenTextures(1)
    glBindTexture(GL_TEXTURE_1D, texP)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, N, 0, GL_RGBA, GL_FLOAT, (GLfloat*len(flat))(*flat))
    texT = glGenTextures(1)
    glBindTexture(GL_TEXTURE_1D, texT)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    inv4 = []
    for v in inv:
        inv4 += [v, v, v, 1.0]
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, N, 0, GL_RGBA, GL_FLOAT, (GLfloat*len(inv4))(*inv4))
    glBindTexture(GL_TEXTURE_1D, 0)
    _gl["P"], _gl["T"] = texP, texT
    _gl["u"] = {"bb_logTmin": log10(T_MIN), "bb_logTmax": log10(T_MAX),
                "bb_qmin": qmin, "bb_qmax": qmax}
    return texP, texT, _gl["u"]

# GLSL (version 120).  Caller declares `uniform sampler1D bb_lutP, bb_lutT;`
# via this snippet and calls doppler_accurate(color.rgb, D, exposure).
GLSL = """
uniform sampler1D bb_lutP;
uniform sampler1D bb_lutT;
uniform float bb_logTmin, bb_logTmax, bb_qmin, bb_qmax;
uniform float exposure;
uniform float bb_logT0;   // >0: known source temperature (log10 K); 0: estimate from colour
vec3 srgb_to_linear(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
}
vec3 linear_to_srgb(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(1.0/2.2));
}
uniform float disp_max;   // display white in linear units (EDR headroom; 1 for SDR)
// Over-bright light: cones saturate, so a dazzling source is seen as white,
// not as the hue-shifted colour that per-channel clipping would give.  The
// hue is kept just above the display's limit and fades to white by ~1000x over
// (a 3000 K filament a dozen times over still reads warm, the Sun reads white).
vec3 display(vec3 lin) {
    lin = max(lin, vec3(0.0));
    float m = max(max(lin.r, lin.g), lin.b);
    if (m > disp_max) {
        vec3 c = lin * (disp_max / m);
        float s = clamp(log2(m / disp_max) * 0.1, 0.0, 1.0);
        lin = mix(c, vec3(disp_max), s);
    }
    return linear_to_srgb(lin);
}
vec4 planck_at(float logT) {
    // (r,g,b,Y) with Rayleigh-Jeans / Wien extrapolation outside the table
    float u = (logT - bb_logTmin) / (bb_logTmax - bb_logTmin);
    vec4 p = texture1D(bb_lutP, clamp(u, 0.0, 1.0));
    if (u > 1.0) p *= pow(10.0, logT - bb_logTmax);
    if (u < 0.0) p *= exp(-40.0*(-u));
    return p;
}
// ---- spectral model for textured bodies -------------------------------
// The texel is turned into a smooth reflectance spectrum (Smits 1999, 10
// bins 380-720 nm), multiplied by the illuminant's Planck spectrum at T0,
// Doppler-shifted as a whole (observed lambda <- emitted D*lambda, which
// turns Planck(T0) into Planck(D*T0) exactly) and integrated against the
// CIE 1931 observer.  Albedo features therefore slide along the spectrum.
uniform float bb_Ywhite;  // Y of a white reflector under T0 at rest (normalisation)
uniform vec3 bb_p0;       // Planck(T0) linear sRGB, Y = 1 (emitters: texel is appearance)
uniform float bb_lit;     // 1: texel is albedo under an external light; 0: emitter
float bb_lobe(float x, float mu, float s1, float s2) {
    float t = (x - mu) / (x < mu ? s1 : s2);
    return exp(-0.5*t*t);
}
vec3 bb_cmf(float l) {
    return vec3(1.056*bb_lobe(l, 599.8, 37.9, 31.0) + 0.362*bb_lobe(l, 442.0, 16.0, 26.7) - 0.065*bb_lobe(l, 501.1, 20.4, 26.2),
                0.821*bb_lobe(l, 568.8, 46.9, 40.5) + 0.286*bb_lobe(l, 530.9, 16.3, 31.1),
                1.217*bb_lobe(l, 437.0, 11.8, 36.0) + 0.681*bb_lobe(l, 459.0, 26.0, 13.8));
}
float bb_planck_um(float l, float T) {      // per-wavelength, l in micrometres
    float x = 14387.77 / (l * T);
    float d = (x < 1.0e-3) ? x * (1.0 + 0.5*x) : exp(min(x, 80.0)) - 1.0;
    return 1.0 / (pow(l, 5.0) * d);
}
vec3 doppler_spectral(vec3 srgb, float D, float expo) {
    vec3 c = srgb_to_linear(srgb);
    if (bb_lit < 0.5) c /= max(bb_p0, vec3(1e-6));
    float r = c.r, g = c.g, b = c.b;
    float cw = 0.0, cc = 0.0, cm = 0.0, cy = 0.0, cr = 0.0, cg = 0.0, cb = 0.0;
    if (r <= g && r <= b) { cw = r; if (g <= b) { cc = g - r; cb = b - g; } else { cc = b - r; cg = g - b; } }
    else if (g <= r && g <= b) { cw = g; if (r <= b) { cm = r - g; cb = b - r; } else { cm = b - g; cr = r - b; } }
    else { cw = b; if (r <= g) { cy = r - b; cg = g - r; } else { cy = g - b; cr = r - g; } }
    float R0 = cw*1.0000 + cc*0.9710 + cm*1.0000 + cy*0.0001 + cr*0.1012 + cg*0.0000 + cb*1.0000;
    float R1 = cw*1.0000 + cc*0.9426 + cm*1.0000 + cy*0.0000 + cr*0.0515 + cg*0.0000 + cb*1.0000;
    float R2 = cw*0.9999 + cc*1.0007 + cm*0.9685 + cy*0.1088 + cr*0.0000 + cg*0.0273 + cb*0.8916;
    float R3 = cw*0.9993 + cc*1.0007 + cm*0.2229 + cy*0.6651 + cr*0.0000 + cg*0.7937 + cb*0.3323;
    float R4 = cw*0.9992 + cc*1.0007 + cm*0.0000 + cy*1.0000 + cr*0.0000 + cg*1.0000 + cb*0.0000;
    float R5 = cw*0.9998 + cc*1.0007 + cm*0.0458 + cy*1.0000 + cr*0.0000 + cg*0.9418 + cb*0.0000;
    float R6 = cw*1.0000 + cc*0.1564 + cm*0.8369 + cy*0.9996 + cr*0.8325 + cg*0.1719 + cb*0.0003;
    float R7 = cw*1.0000 + cc*0.0000 + cm*1.0000 + cy*0.9586 + cr*1.0149 + cg*0.0000 + cb*0.0369;
    float R8 = cw*1.0000 + cc*0.0000 + cm*1.0000 + cy*0.9685 + cr*1.0149 + cg*0.0000 + cb*0.0483;
    float R9 = cw*1.0000 + cc*0.0000 + cm*0.9959 + cy*0.9840 + cr*1.0149 + cg*0.0025 + cb*0.0496;
    float T = pow(10.0, bb_logT0) * D;
    vec3 xyz = vec3(0.0);
    for (int i = 0; i < 24; i++) {
        float lam = 380.0 + (float(i) + 0.5) * (400.0 / 24.0);   // observed, nm
        float fb = clamp((D * lam - 380.0) / 34.0, 0.0, 9.99);    // emitted-wavelength bin
        float s1 = step(1.0, fb), s2 = step(2.0, fb), s3 = step(3.0, fb), s4 = step(4.0, fb);
        float s5 = step(5.0, fb), s6 = step(6.0, fb), s7 = step(7.0, fb), s8 = step(8.0, fb), s9 = step(9.0, fb);
        float refl = R0*(1.0 - s1) + R1*(s1 - s2) + R2*(s2 - s3) + R3*(s3 - s4) + R4*(s4 - s5)
                   + R5*(s5 - s6) + R6*(s6 - s7) + R7*(s7 - s8) + R8*(s8 - s9) + R9*s9;
        xyz += refl * bb_planck_um(lam * 0.001, T) * bb_cmf(lam);
    }
    xyz /= bb_Ywhite;
    vec3 lin = vec3( 3.2406*xyz.x - 1.5372*xyz.y - 0.4986*xyz.z,
                    -0.9689*xyz.x + 1.8758*xyz.y + 0.0415*xyz.z,
                     0.0557*xyz.x - 0.2040*xyz.y + 1.0570*xyz.z);
    return display(lin * expo);
}
vec3 doppler_accurate(vec3 srgb, float D, float expo) {
    vec3 lin = srgb_to_linear(srgb);
    float q = log(lin.b + 1e-4) - log(lin.r + 1e-4);
    float logT0 = bb_logT0 > 0.0 ? bb_logT0 : texture1D(bb_lutT, clamp((q - bb_qmin)/(bb_qmax - bb_qmin), 0.0, 1.0)).r;
    vec4 p0 = planck_at(logT0);
    vec4 p1 = planck_at(logT0 + log(D)*0.4342944819);
    vec3 emis = lin / max(p0.rgb, vec3(1e-6));
    return display(emis * p1.rgb * expo);
}
"""


def _planck_um(l_um, T):
    x = 14387.77 / (l_um * T)
    d = x * (1.0 + 0.5*x) if x < 1.0e-3 else exp(min(x, 80.0)) - 1.0
    return 1.0 / (l_um**5 * d)

_norm_cache = {}
def spectral_norm(T0):
    """Y of a white reflector under Planck(T0) at rest, sampled exactly as the
    shader does (24 bins, 380-780 nm) so a white texel at D=1 renders at Y=1."""
    v = _norm_cache.get(T0)
    if v is None:
        v = 0.0
        for i in range(24):
            lam = 380.0 + (i + 0.5) * (400.0 / 24.0)
            v += _planck_um(lam * 0.001, T0) * ybar(lam)
        _norm_cache[T0] = v
    return v

_p0_cache = {}
def planck_rgb_unitY(T0):
    v = _p0_cache.get(T0)
    if v is None:
        r, g, b = planck_rgb(T0)
        X, Y, Z = xyz_of_blackbody(T0)
        _, Yref, _ = xyz_of_blackbody(6500.0)
        y = Y / Yref
        v = (r / y, g / y, b / y)
        _p0_cache[T0] = v
    return v

def bind_luts(program_locals=None):
    """Bind the lookup textures to texture units 1 and 2 (call once per frame)."""
    texP, texT, u = gl_textures()
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_1D, texP)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_1D, texT)
    glActiveTexture(GL_TEXTURE0)
    return u

def set_uniforms(program, exposure, temperature=0.0, lit=False):
    """Set the LUT samplers and constants on a program that includes GLSL.
    temperature: known source temperature in K (0 = estimate from colour)."""
    _, _, u = gl_textures()
    from math import log10
    glUniform1f(glGetUniformLocation(program, "bb_logT0"), log10(temperature) if temperature > 0.0 else 0.0)
    if temperature > 0.0:
        glUniform1f(glGetUniformLocation(program, "bb_Ywhite"), spectral_norm(temperature))
        glUniform3f(glGetUniformLocation(program, "bb_p0"), *planck_rgb_unitY(temperature))
    glUniform1f(glGetUniformLocation(program, "bb_lit"), 1.0 if lit else 0.0)
    from program.box import BOX
    glUniform1f(glGetUniformLocation(program, "disp_max"),
                float(BOX.edr_headroom) if (BOX.HDR and getattr(BOX, "edr_enabled", False)) else 1.0)
    glUniform1i(glGetUniformLocation(program, "bb_lutP"), 1)
    glUniform1i(glGetUniformLocation(program, "bb_lutT"), 2)
    for name, val in u.items():
        glUniform1f(glGetUniformLocation(program, name), val)
    glUniform1f(glGetUniformLocation(program, "exposure"), exposure)
