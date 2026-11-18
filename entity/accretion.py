# coding: utf8
# entity/accretion.py
"""
Accretion disc around a black hole.

The disc is a thin annulus split into NRING radial rings and NSEG angular
segments.  Every segment is drawn as its own emitter with the Keplerian
4-velocity of its mid-point, so the game's per-emitter Doppler pipeline
gives the correct relativistic beaming: the approaching side is brighter
and bluer, the receding side dimmer and redder.  Each ring also carries
the gravitational redshift sqrt(1 - R_s/r) of a static emitter at its
radius.  Rings rotate at their Keplerian rate (inner ones faster) and the
texture is a hot turbulent gas sheet whose slice follows each segment
through the texture matrix.  Light bending is not modelled.
"""
from math import pi, sin, cos, sqrt, acos, atan2, radians

from OpenGL.GL import *

from go import Matrix44, Vector3, Vector4D, Quaternion
from model.polygon import Polygon
from program.const import IMG_DIR
from program.box import BOX


NSEG = 24
NRING = 3


def _pole_matrix(pole):
    px, py, pz = pole
    n = sqrt(px*px + py*py + pz*pz)
    px, py, pz = px/n, py/n, pz/n
    ax = Vector3(0.0, 1.0, 0.0).cross(Vector3(px, py, pz))
    c = max(-1.0, min(1.0, py))
    if ax.length() > 1e-12:
        ax.normalize()
        return Quaternion.from_ax(acos(c), ax).get_RotMat()
    if c > 0.0:
        return Matrix44()
    return Matrix44.z_rotation(pi)


def disc_azimuths(step=2.5, wedge=15.0, dmin=0.02, ratio=1.15):
    """The mesh azimuths (radians, rising from 0 to 2 pi inclusive): `step`
    degrees apart, but geometrically denser toward a = pi, down to `dmin`
    degrees from it.  Lensed, the far half of the disc images onto a thin
    arc over the hole, and the whole arc comes from the sources within a
    degree or so of the azimuth straight behind the hole (there the side
    the image lies on swings from one side through the top to the other);
    the mesh is turned each frame so that a = pi faces away from the ship."""
    az = set()
    a = 0.0
    while a < 360.0 - 0.5*step:
        if abs(a - 180.0) >= wedge:
            az.add(a)
        a += step
    d = dmin
    while d < wedge:
        az.add(180.0 - d); az.add(180.0 + d)
        d *= ratio
    az.add(180.0)
    return [radians(a) for a in sorted(az)] + [2.0*pi]


def write_annulus(path, r0, r1, v0, v1, nr=8, az=None):
    """Full annulus, radii r0..r1 (units of the Schwarzschild radius), in the
    plane y = 0; texcoords u = azimuth, v in [v0, v1]."""
    if az is None:
        az = disc_azimuths()
    na = len(az) - 1
    lines = ["Game Polygon Object"]
    for j in range(nr + 1):
        r = r0 + (r1 - r0)*j/nr
        v = v0 + (v1 - v0)*j/nr
        for a in az:
            lines.append("p %.6f 0.000000 %.6f %.6f %.6f" % (r*cos(a), r*sin(a), a/(2.0*pi), v))
    lines.append('m 1.0 1.0 1.0 1.0 "accretion.png"')
    idx = []
    for j in range(nr):
        for i in range(na):
            a = j*(na+1) + i; b = a + 1; c = a + na + 1; d = c + 1
            idx += [a, b, c, b, d, c, a, c, b, b, c, d]       # both windings
    lines.append("i " + " ".join(map(str, idx)))
    open(path, "w").write("\n".join(lines) + "\n")


class AccretionDisk(object):

    def __init__(self, star, inner, outer, T_in, brightness, pole, texture="accretion.png"):
        """star: the black hole (sphere_radius = Schwarzschild radius);
        inner/outer in units of that radius; T_in: temperature at the inner
        edge (T ~ r^-3/4 outward); brightness in the script's sRGB units."""
        self.star = star
        self.M = _pole_matrix(pole)
        self.inner, self.outer = inner, outer
        self.brightness, self.temperature = brightness, T_in   # at the inner edge
        self.rings = []
        Rs = star.radius
        for j in range(NRING):
            r0 = inner * (outer/inner) ** (j/float(NRING))
            r1 = inner * (outer/inner) ** ((j+1)/float(NRING))
            path = IMG_DIR + "accretion_ring%d.gpo" % j
            write_annulus(path, r0, r1, j/float(NRING), (j+1)/float(NRING))
            rm = sqrt(r0*r1)
            T = T_in * (rm/inner) ** -0.75
            b = brightness * (rm/inner) ** -1.5
            func = (lambda Rs: (lambda x, y, z: (x*Rs, y*Rs, z*Rs)))(Rs)
            poly = Polygon(IMG_DIR + "accretion_ring%d" % j, func=func, texture=False, color=(b, b, b, 1.0))
            poly.set_texture(texture)
            poly.temperature = T
            poly.disc = (Rs, 1.0)
            poly.grav_shift = sqrt(max(0.0, 1.0 - 1.0/rm))
            v = sqrt(0.5/rm)                      # Keplerian speed at rm, c = 1
            omega = v / (rm*Rs)                   # rad/s: the texture pattern co-rotates
            self.rings.append((poly, rm, omega))

    # the sense of Matrix44.y_rotation on the mesh's azimuth (x, z) = (cos a, sin a)
    _v = Matrix44.y_rotation(0.3).get_rotate_v3(Vector3(1.0, 0.0, 0.0))
    SPIN_SIGN = 1.0 if _v.z > 0.0 else -1.0

    def draw(self, Xp, L, LL, X, R, t, scale=1.0, gshift=1.0):
        """X, R: the hole's (possibly far-rescaled) centre and scale matrix."""
        # the mesh is turned so its dense wedge (a = pi) faces away from the
        # ship, and the gas pattern turned back through the texture matrix
        # so that it still rotates at the Keplerian rate
        d = Xp - X
        dm = (R * self.M).get_inverse_rot().get_rotate_v3(Vector3(d.x, d.y, d.z))
        a_obs = atan2(dm.z, dm.x)                 # the ship's azimuth in the mesh frame
        glDisable(GL_CULL_FACE)
        glMatrixMode(GL_TEXTURE)
        glPushMatrix()
        for poly, rm, omega in self.rings:
            phase = (omega*t) % (2.0*pi)
            Rk = R * self.M * Matrix44.y_rotation(self.SPIN_SIGN * a_obs)
            glLoadIdentity()
            glTranslatef((a_obs - phase) / (2.0*pi), 0.0, 0.0)
            poly.draw(Xp, L, LL, X=X, R=Rk, scale=scale, gshift=gshift)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_CULL_FACE)
