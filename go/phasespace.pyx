# coding: utf8
# cython: profile=False
# go/phasespace.pyx
cimport cython
from libc.math cimport sqrt, sinh

from go.vector4D cimport Vector4D, vec4_from_floats
from go.matrix44 cimport Matrix44, Lorentz

DEF WIGNER_SIGN = -1.0


cdef class PhaseSpace(object):

    cdef public Vector4D X, U

    def __init__(self, X, U):
        self.X = Vector4D(X)
        self.U = Vector4D(U)

    def copy(self):
        return PhaseSpace(self.X, self.U)

    def get_resist(self, double b):
        return vec4_from_floats(0.0, -self.U._x*b, -self.U._y*b, -self.U._z*b)

    @cython.cdivision(True)
    def transform(self, Vector4D acceleration, double ds):
        """
        acceleration: proper acceleration, spatial components in the
            comoving (pure-boost) rest frame.  c = 1.
        ds: proper time step.

        Exact for a proper acceleration that is constant in the comoving
        frame: the rapidity changes by |a|*ds along a.  No cap.

        Returns the Wigner (Thomas) rotation as a unit quaternion
        (t, x, y, z) with the same convention as Quaternion, such that

            Lorentz(-U_old) * Lorentz(-W)  ==  Lorentz(-U_new) * R(q)

        i.e. R maps Fermi-Walker transported body axes into the new
        pure-boost rest frame.  Computed from half-rapidity tangents, which
        stays well conditioned at any Lorentz factor (composing the boost
        matrices would lose everything above gamma ~ 1e7).
        """
        cdef double ax = acceleration._x
        cdef double ay = acceleration._y
        cdef double az = acceleration._z
        cdef double a = sqrt(ax*ax + ay*ay + az*az)
        cdef double sh, wx, wy, wz, w, gw
        if a > 0.0:
            sh = sinh(a*ds) / a
            wx = ax*sh; wy = ay*sh; wz = az*sh
            w = a*sh
        else:
            wx = wy = wz = 0.0
            w = 0.0
        gw = sqrt(1.0 + w*w)

        cdef double ux = self.U._x
        cdef double uy = self.U._y
        cdef double uz = self.U._z
        cdef double u = sqrt(ux*ux + uy*uy + uz*uz)
        cdef double g = sqrt(1.0 + u*u)
        cdef double inv = 1.0 / (g + u)          # = g - u, no cancellation

        # ---- new 4-velocity: Lorentz(-U) * (gw, w)  (no large cancellation)
        cdef double wpar = 0.0
        cdef double nx = 0.0, ny = 0.0, nz = 1.0
        if u > 0.0:
            nx = ux/u; ny = uy/u; nz = uz/u
            wpar = wx*nx + wy*ny + wz*nz
        cdef double tnew = g*gw + u*wpar
        cdef double ppar = g*wpar + u*gw          # parallel part
        cdef double vx = wx - wpar*nx + ppar*nx
        cdef double vy = wy - wpar*ny + ppar*ny
        cdef double vz = wz - wpar*nz + ppar*nz
        tnew = sqrt(1.0 + vx*vx + vy*vy + vz*vz)

        # ---- Wigner rotation from half-rapidity tangents
        # tan(theta/2) = ta tb sin(phi) / (1 + ta tb cos(phi)),
        # axis along u x w;  ta = tanh(eta_u/2) = u/(g+1),  tb = w/(gw+1)
        cdef double qt = 1.0, qx = 0.0, qy = 0.0, qz = 0.0
        cdef double ta, tb, one_m_ta, one_m_tb, cx, cy, cz, cosphi, dn, nn, s2, ex, ey, ez
        if u > 0.0 and w > 0.0:
            ta = u / (g + 1.0)
            tb = w / (gw + 1.0)
            one_m_ta = (inv + 1.0) / (g + 1.0)
            one_m_tb = (1.0/(gw + w) + 1.0) / (gw + 1.0)
            cosphi = wpar / w
            # vector part: ta tb (u x w)/(|u||w|)
            cx = (ny*wz - nz*wy) / w
            cy = (nz*wx - nx*wz) / w
            cz = (nx*wy - ny*wx) / w
            if cosphi >= 0.0:
                dn = 1.0 + ta*tb*cosphi
            else:
                ex = nx + wx/w; ey = ny + wy/w; ez = nz + wz/w
                dn = one_m_ta + ta*one_m_tb + 0.5*(ex*ex + ey*ey + ez*ez)*ta*tb
            cx *= ta*tb; cy *= ta*tb; cz *= ta*tb
            nn = sqrt(dn*dn + cx*cx + cy*cy + cz*cz)
            qt = dn / nn
            qx = WIGNER_SIGN * cx / nn
            qy = WIGNER_SIGN * cy / nn
            qz = WIGNER_SIGN * cz / nn

        # ---- trapezoidal position update
        self.X._t = self.X._t + 0.5*(self.U._t + tnew) * ds
        self.X._x = self.X._x + 0.5*(self.U._x + vx) * ds
        self.X._y = self.X._y + 0.5*(self.U._y + vy) * ds
        self.X._z = self.X._z + 0.5*(self.U._z + vz) * ds
        self.U._t = tnew
        self.U._x = vx
        self.U._y = vy
        self.U._z = vz
        return (qt, qx, qy, qz)

    @cython.cdivision(True)
    def transform_matrix_reference(self, Vector4D acceleration, double ds):
        """Old matrix-composition version, kept only for validation."""
        cdef double ax = acceleration._x
        cdef double ay = acceleration._y
        cdef double az = acceleration._z
        cdef double a = sqrt(ax*ax + ay*ay + az*az)
        cdef double sh
        cdef Vector4D W
        if a > 0.0:
            sh = sinh(a*ds) / a
            W = vec4_from_floats(0.0, ax*sh, ay*sh, az*sh)
        else:
            W = vec4_from_floats(0.0, 0.0, 0.0, 0.0)
        cdef Matrix44 Lold = Lorentz(-self.U)
        cdef Matrix44 La = Lorentz(-W)
        cdef Matrix44 M = Lold * La
        cdef double ut = M.m00
        cdef double ux = M.m10
        cdef double uy = M.m20
        cdef double uz = M.m30
        ut = sqrt(1.0 + ux*ux + uy*uy + uz*uz)
        self.X._t = self.X._t + 0.5*(self.U._t + ut) * ds
        self.X._x = self.X._x + 0.5*(self.U._x + ux) * ds
        self.X._y = self.X._y + 0.5*(self.U._y + uy) * ds
        self.X._z = self.X._z + 0.5*(self.U._z + uz) * ds
        self.U._t = ut
        self.U._x = ux
        self.U._y = uy
        self.U._z = uz
        cdef Matrix44 R = Lorentz(self.U) * M
        return R
