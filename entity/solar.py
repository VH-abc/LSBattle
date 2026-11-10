# coding: utf8
# entity/solar.py
from math import pi, sin, cos, acos

from OpenGL.GL import *

from go import Matrix44, Vector3, Vector4D, Lorentz, Quaternion, calc_repulsion
from model.polygon import Polygon
from model.flame import Flame
from entity.accretion import AccretionDisk
from program.const import IMG_DIR, c
from program.box import BOX
from program.text import drawSentence3d
from program import script
from program.utils import boost


def _high_func(sphere_radius, tilt, flattening=0.0, pole=None, spin=0.0):
    # star.gpo is modelled at radius 0.5, so scale by twice the radius to
    # draw the body at its scripted size (collisions and markers use that)
    sphere_radius = 2.0 * sphere_radius
    if pole is not None:
        # rotation taking the model's polar axis (0,1,0) onto the given direction
        px, py, pz = pole
        n = (px*px + py*py + pz*pz) ** 0.5
        px, py, pz = px/n, py/n, pz/n
        ax = Vector3(0.0, 1.0, 0.0).cross(Vector3(px, py, pz))
        c = max(-1.0, min(1.0, py))
        if ax.length() > 1e-12:
            ax.normalize()
            rot = Quaternion.from_ax(acos(c), ax).get_RotMat()
        elif c > 0.0:
            rot = Matrix44()
        else:
            rot = Matrix44.z_rotation(pi)
        mat = Matrix44.scale(sphere_radius) * rot * Matrix44.y_rotation(pi/2)
    else:
        mat = Matrix44.scale(sphere_radius) * Matrix44.z_rotation(tilt*pi/180) * Matrix44.y_rotation(pi/2 + spin*pi/180)
    k = 1.0 - flattening
    def func(x, y, z):
        return mat.get_rotate([x, y*k, z])
    return func

G_OVER_C3 = 6.67430e-11 / (2.99792458e8 ** 3)   # GM per kilogram, in seconds (c = 1)


class Star(object):

    def __init__(self, X, star_data):
        pole = (float(getattr(star_data, 'pole_x', 0.0)),
                float(getattr(star_data, 'pole_y', 0.0)),
                float(getattr(star_data, 'pole_z', 0.0)))
        if pole == (0.0, 0.0, 0.0):
            pole = None
        func = _high_func(star_data.sphere_radius, star_data.tilt,
                          float(getattr(star_data, 'flattening', 0.0)), pole,
                          spin=float(getattr(star_data, 'spin', 0.0)))
        self.name = getattr(star_data, 'display_name', star_data.name)
        self.radius = star_data.sphere_radius
        self.radius2 = self.radius**2
        # repulsion/collision radius: the short axis of a flattened body
        self.collision_radius = self.radius * (1.0 - float(getattr(star_data, 'flattening', 0.0)))
        b = float(getattr(star_data, "brightness", 1.0))
        self.brightness = b
        if star_data.texture is None:
            self.model = Polygon(IMG_DIR+star_data.model, func=func, color=(b, b, b, 1.0))
        else:
            self.model = Polygon(IMG_DIR+star_data.model, func=func, texture=False, color=(b, b, b, 1.0))
            self.model.set_texture(star_data.texture)
        self.model.temperature = float(getattr(star_data, "temperature", 0.0))
        # optional planetary ring: an annulus (resources/img/ring.gpo, radii in
        # units of the sphere radius) lying in the body's equatorial plane
        ring_tex = getattr(star_data, "ring", "") or ""
        if ring_tex and ring_tex.lower() not in ("none", "null"):
            # ring.gpo radii are in planet radii; _high_func doubles for the
            # 0.5-radius globe model, so hand it half the radius
            rfunc = _high_func(0.5*star_data.sphere_radius, star_data.tilt, 0.0, pole)
            self.ring = Polygon(IMG_DIR+"ring", func=rfunc, texture=False, color=(b, b, b, 1.0))
            self.ring.set_texture(ring_tex)
            self.ring.temperature = self.model.temperature
        else:
            self.ring = None
        self.solid = int(getattr(star_data, "solid", 1)) != 0
        lit = getattr(star_data, "lit_by", "") or ""
        self.lit_by = lit.lower() if lit.lower() not in ("", "none", "null") else None
        self.system = None          # set by SolarSystem
        self.temperature = float(getattr(star_data, "temperature", 0.0))
        self.flattening = float(getattr(star_data, "flattening", 0.0))
        self.pole = pole if pole is not None else (0.0, 1.0, 0.0)
        disk_outer = float(getattr(star_data, "disk_outer", 0.0))
        if disk_outer > 0.0:
            self.disk = AccretionDisk(self, float(getattr(star_data, "disk_inner", 3.0)), disk_outer,
                                      float(getattr(star_data, "disk_temperature", 10000.0)),
                                      float(getattr(star_data, "disk_brightness", 100.0)),
                                      self.pole)
        else:
            self.disk = None
        # largest distance of any drawn part from the centre
        self.extent = self.radius
        if self.ring is not None:
            self.extent = 2.3 * self.radius
        if self.disk is not None:
            self.extent = max(self.extent, self.disk.outer * self.radius)
        self.X = X.copy()
        self.X0 = X.copy()              # position when orbits are off
        self.U = Vector4D(1.0, 0.0, 0.0, 0.0)
        self.mass = float(getattr(star_data, "mass", 0.0))
        self.GM = G_OVER_C3 * self.mass  # light-seconds^3 / s^2
        # a body whose radius is its Schwarzschild radius: landing on it is
        # crossing the horizon
        self.is_hole = self.GM > 0.0 and 2.0*self.GM >= 0.999*self.radius
        # light from the surface climbs out of the body's own well
        self.self_shift = (max(0.0, 1.0 - 2.0*self.GM/self.radius)) ** 0.5 if self.radius > 0.0 and not self.is_hole else 1.0
        if self.is_hole:
            # the geodesic step stops just outside R_s: make sure that counts as contact
            # even when the scripted radius is rounded a little below 2GM
            self.collision_radius = max(self.collision_radius, 2.0*self.GM*(1.0 + 1.0e-8))
        self.orbit = None               # (primary name, p0, e, omega) set by SolarSystem
        self.hp = star_data.hp
        flame_data = script.world.solar.flame
        self.flame = Flame(S=flame_data.life*self.radius, v=flame_data.speed,
                           n=flame_data.num, m=flame_data.num*2,
                           color=flame_data.color,
                           psize=flame_data.size*self.radius)
        self.alive = True
        self.X_dead = None

    def depth(self, Xp):
        """Normalised distance from the centre in units of this body's own
        ellipsoid: < 1 inside, 1 on the surface."""
        d = Xp - self.X
        px, py, pz = self.pole
        par = d.x*px + d.y*py + d.z*pz
        perp2 = d.x*d.x + d.y*d.y + d.z*d.z - par*par
        R = self.radius
        k = max(1e-6, 1.0 - self.flattening)
        return (max(0.0, perp2)/(R*R) + (par*par)/((R*k)*(R*k))) ** 0.5

    GLARE_REF = 132.0       # the Sun's scripted brightness: glare sources are drawn relative to it

    def draw_glare(self, Xp, L, LL, gshift=1.0):
        """Draw this body as a glare source: its rendered radiance divided by
        the Sun's, so only what is really Sun-bright on screen glares, which
        includes a planet or the sky once they are blueshifted enough."""
        if self.hp <= 0:
            return
        if not self.solid and self.depth(Xp) < 1.0:
            return
        self._draw_globe(Xp, L, LL, ring=False, scale=1.0/self.GLARE_REF, gshift=gshift)

    def draw(self, Xp, L, LL, gshift=1.0):
        if self.hp > 0:
            if not self.solid and self.depth(Xp) < 1.0:
                return          # inside a diffuse body: nothing to draw
            self._draw_globe(Xp, L, LL, gshift=gshift)
        else:
            if not self.flame.draw(self.X_dead, Xp, L):
                self.alive = False

    def _set_lighting(self):
        """Lambert lighting from the body named in lit_by: direction and the
        irradiance falloff (star brightness * (R_star / d)^2), with the
        star's temperature as the illuminant."""
        sun = self.system.stars.get(self.lit_by) if (self.system and self.lit_by) else None
        if sun is None or sun is self:
            self.model.light = None
            if self.ring is not None:
                self.ring.light = None
            return
        d = sun.X - self.X
        n = (d.x*d.x + d.y*d.y + d.z*d.z) ** 0.5
        if n <= 0.0:
            self.model.light = None
            return
        ldir = (d.x/n, d.y/n, d.z/n)
        if BOX.COLOR_MODE == "ACCURATE":
            # a white Lambertian surface at distance n has radiance
            # L_star * (R/n)^2 in linear light; ucolor works in sRGB-encoded
            # units (the star's "brightness"), hence the 1/2.2 power
            lscale = sun.brightness * (sun.radius / n) ** (2.0 / 2.2)
        else:
            lscale = 1.0            # the game look keeps the texture brightness
        self.model.light = (ldir, 1)
        self.model.light_scale = lscale
        self.model.temperature = sun.temperature
        if self.ring is not None:
            self.ring.light = (ldir, 2)
            self.ring.light_scale = lscale
            self.ring.temperature = sun.temperature

    def _draw_globe(self, Xp, L, LL, ring=True, scale=1.0, gshift=1.0):
        gshift *= self.self_shift
        if True:
            self._set_lighting()
            Xc = self.X
            U = self.U
            if self.orbit is not None and BOX.ORBITS:
                # the light we see left the body at the retarded time; v << c
                # so one light-cone iteration is plenty
                t_ret = Xp.t - (self.X - Xp).length()
                Xc, U = self.system.state_at(self, t_ret)
            dX = Xc - Xp
            lo = self.system.player_lo
            dX.x -= lo[0]; dX.y -= lo[1]; dX.z -= lo[2]
            dX.t = -dX.length()
            dx = boost.transform_null(boost.U_of_L(L), dX)
            r = -dx.t
            if r > 0.5*BOX.far_clip:
                # distant body: shrink it towards the player so it fits inside
                # the far clip.  The mapping r -> r' is monotone (half the far
                # clip up to 95% of it), so the depth order between distant
                # bodies is kept and none of them can cover a nearer one.
                F = BOX.far_clip
                # square root spreads the planets apart in depth so the
                # 24-bit depth buffer still orders them with a close near clip
                s = (0.5*F + 0.45*F*(1.0 - (0.5*F/r) ** 0.5)) / r
                # ...but a wide object (ring, accretion disc) must still fit
                s = min(s, 0.97*F / (r + self.extent))
                X = Xp + dX*s
                R = Matrix44.scale(s)
            else:
                X = Xc
                R = Matrix44()
            self.model.draw(Xp, L, LL, X=X, U=U, R=R, scale=scale, gshift=gshift)
            if ring and self.disk is not None:
                self.disk.draw(Xp, L, LL, X, R, Xp.t, scale=scale, gshift=gshift)
            if ring and self.ring is not None:
                # translucent: drawn after the globe so it blends over it and
                # is hidden where the globe is in front
                glDisable(GL_CULL_FACE)
                self.ring.draw(Xp, L, LL, X=X, U=U, R=R, gshift=gshift)
                glEnable(GL_CULL_FACE)

    def hit_check(self, Xp, world):
        if self.hp > 0 and self.solid:
            t = Xp.t - Xp.distance_to(self.X)
            self.X.t = t
            X1 = self.X + Vector4D(0.02, 0, 0, 0)
            self.hp -= world.player.bullet_hit_check(X1, self.X, self.radius2)
            self.hp -= world.enemies.bullets.hit_check(X1, self.X, self.radius2)
            if self.hp <= 0:
                self.X_dead = self.X.copy()

class SolarSystem(object):
    star_datum = None
    GLARE_REF = Star.GLARE_REF
    def __init__(self, world, Xp, scale):
        self.player_lo = (0.0, 0.0, 0.0)   # low-order part of the player position
        self.world = world
        self.stars = {}
        self.read_data()
        if not self.star_datum:
            return
        self.calc_star_pos(Xp, scale)
        for name in self.star_pos:
            star = Star(self.star_pos[name], self.star_datum[name])
            star.system = self
            self.stars[name] = star
        self.init_orbits()
        self.time = 0.0
        self._moving = False

    # ---------------------------------------------------------------- orbits
    def init_orbits(self):
        """Circular Kepler orbits about each body's primary, in the plane
        spanned by the initial offset and its 'horizontal' perpendicular."""
        for star in self.stars.values():
            data = self.star_datum[star.name.lower()] if star.name.lower() in self.star_datum else None
            pname = data.primary_star if data is not None else None
            prim = self.stars.get(pname) if pname else None
            if prim is None or prim.GM <= 0.0:
                continue
            d = star.X0 - prim.X0
            p0 = (d.x, d.y, d.z)
            r = (d.x*d.x + d.y*d.y + d.z*d.z) ** 0.5
            if r <= 0.0:
                continue
            # e = (y_hat x p0) scaled to |p0|, falling back to x_hat x p0 near the pole
            ex, ey, ez = -d.z, 0.0, d.x
            n = (ex*ex + ez*ez) ** 0.5
            if n < 1e-9 * r:
                ex, ey, ez = 0.0, -d.z, d.y
                n = (ey*ey + ez*ez) ** 0.5
            e = (ex/n*r, ey/n*r, ez/n*r)
            omega = (prim.GM / r**3) ** 0.5
            star.orbit = (pname, p0, e, omega)

    def state_at(self, star, t, memo=None):
        """(X, U) of a body at coordinate time t: position on its orbit about
        its (moving) primary, and the 4-velocity.  Static bodies keep X0."""
        if memo is None:
            memo = {}
        key = star.name
        if key in memo:
            return memo[key]
        if star.orbit is None or not BOX.ORBITS:
            X = star.X0.copy(); X.t = t
            res = (X, Vector4D(1.0, 0.0, 0.0, 0.0))
        else:
            pname, p0, e, w = star.orbit
            Xp, Up = self.state_at(self.stars[pname], t, memo)
            c, sn = cos(w*t), sin(w*t)
            X = Vector4D(t, Xp.x + p0[0]*c + e[0]*sn, Xp.y + p0[1]*c + e[1]*sn, Xp.z + p0[2]*c + e[2]*sn)
            gp = Up.t
            vx = Up.x/gp + w*(-p0[0]*sn + e[0]*c)
            vy = Up.y/gp + w*(-p0[1]*sn + e[1]*c)
            vz = Up.z/gp + w*(-p0[2]*sn + e[2]*c)
            v2 = vx*vx + vy*vy + vz*vz
            g = 1.0 / (1.0 - v2) ** 0.5 if v2 < 1.0 else 1.0e8
            res = (X, Vector4D(g, g*vx, g*vy, g*vz))
        memo[key] = res
        return res

    def update(self, t):
        """Move the bodies to coordinate time t; with orbits off they sit at
        their scripted positions, at rest."""
        self.time = t
        if not BOX.ORBITS:
            if self._moving:
                for star in self.stars.values():
                    star.X.x, star.X.y, star.X.z = star.X0.x, star.X0.y, star.X0.z
                    star.U = Vector4D(1.0, 0.0, 0.0, 0.0)
                self._moving = False
            return
        self._moving = True
        memo = {}
        for star in self.stars.values():
            X, U = self.state_at(star, t, memo)
            star.X.x, star.X.y, star.X.z = X.x, X.y, X.z
            star.U = U

    # --------------------------------------------------------------- gravity
    # Within this many Schwarzschild radii of a body its field is integrated
    # as the exact Schwarzschild geodesic (Player._geodesic_step); the
    # weak-field expansion is used further out.  3e4 R_s is where GM/rc^2
    # reaches ~1e-5, comparable to the Sun's own surface potential.
    STRONG_FIELD = 3.0e4

    def strong_body(self, Xp, lo=None):
        """The compact body whose strong field dominates at Xp, or None."""
        best, bd = None, 0.0
        for star in self.stars.values():
            if star.GM <= 0.0 or star.hp <= 0:
                continue
            dx, dy, dz = Xp.x - star.X.x, Xp.y - star.X.y, Xp.z - star.X.z
            if lo is not None:
                dx += lo[0]; dy += lo[1]; dz += lo[2]
            r = (dx*dx + dy*dy + dz*dz) ** 0.5
            if r < self.STRONG_FIELD * 2.0 * star.GM and (best is None or r < bd):
                best, bd = star, r
        return best

    def gravity(self, Xp, lo=None):
        """Newtonian field at Xp from every massive body, in the background
        frame: returns ((gx, gy, gz), phi) with phi = sum GM/r (positive).
        lo: the low-order part of Xp (see Player.Xlo)."""
        gx = gy = gz = 0.0
        phi = 0.0
        for star in self.stars.values():
            if star.GM <= 0.0 or star.hp <= 0:
                continue
            dx, dy, dz = star.X.x - Xp.x, star.X.y - Xp.y, star.X.z - Xp.z
            if lo is not None:
                dx -= lo[0]; dy -= lo[1]; dz -= lo[2]
            r2 = dx*dx + dy*dy + dz*dz
            r = r2 ** 0.5
            if r <= 0.0:
                continue
            R = star.collision_radius
            if r < R:
                a = star.GM * r / R**3            # inside a uniform sphere
                phi += star.GM * (3.0*R*R - r2) / (2.0*R**3)
            else:
                a = star.GM / r2
                phi += star.GM / r
            gx += a * dx / r; gy += a * dy / r; gz += a * dz / r
        return (gx, gy, gz), phi

    def max_step(self, Xp):
        """Largest integration step that resolves the local gravity: a small
        fraction of the free-fall time sqrt(r^3/GM) of the dominant body.
        With the midpoint field this keeps the energy error of a close
        flyby below about 1e-4 of the binding energy per pass."""
        best = float("inf")
        for star in self.stars.values():
            if star.GM <= 0.0 or star.hp <= 0:
                continue
            r = max((star.X - Xp).length(), star.collision_radius)
            if r > 1.0e30:
                continue                      # no gravity to speak of out there
            t = 0.005 * r * (r / star.GM) ** 0.5
            if t < best:
                best = t
        return best

    def nearest(self, Xp, solid_only=True):
        best, bd = None, 0.0
        for star in self.stars.values():
            if star.hp <= 0 or (solid_only and not star.solid):
                continue
            d = (star.X - Xp).length() - star.collision_radius
            if best is None or d < bd:
                best, bd = star, d
        return best

    def land(self, P, margin, lo=None):
        """Inelastic contact: a ship that reaches a solid body's surface is
        put back on the surface and takes the body's velocity.  Returns the
        body it is resting on, or None.  lo (see Player.Xlo) is updated so
        that P.X + lo sits exactly on the surface."""
        for star in self.stars.values():
            if star.hp <= 0 or not star.solid:
                continue
            dx, dy, dz = P.X.x - star.X.x, P.X.y - star.X.y, P.X.z - star.X.z
            if lo is not None:
                dx += lo[0]; dy += lo[1]; dz += lo[2]
            r = (dx*dx + dy*dy + dz*dz) ** 0.5
            Rc = star.collision_radius + margin
            if r <= Rc*(1.0 + 1.0e-7) and r > 0.0:
                k = Rc / r
                dx *= k; dy *= k; dz *= k
                P.X.x = star.X.x + dx; P.X.y = star.X.y + dy; P.X.z = star.X.z + dz
                if lo is not None:
                    lo[0] = dx - (P.X.x - star.X.x)
                    lo[1] = dy - (P.X.y - star.X.y)
                    lo[2] = dz - (P.X.z - star.X.z)
                P.U.t, P.U.x, P.U.y, P.U.z = star.U.t, star.U.x, star.U.y, star.U.z
                return star
        return None

    @classmethod
    def read_data(cls):
        if cls.star_datum is not None:
            return
        rescale = 1.0 / c
        cls.star_datum = {}
        for star_data in script.world.solar.stars:
            star_data.sphere_radius *= rescale
            star_data.orbital_radius *= rescale
            star_data.display_name = star_data.name
            star_data.name = star_data.name.lower()
            pname = star_data.primary_star
            star_data.primary_star = pname.lower() if pname is not None else None
            cls.star_datum[star_data.name] = star_data

    def calc_pos(self, r, phi, theta=0.0):
        cos_p = cos(phi * pi / 180)
        sin_p = sin(phi * pi / 180)
        cos_t = cos(theta * pi / 180)
        sin_t = sin(theta * pi / 180)
        return Vector4D(0.0, sin_p*r*cos_t, sin_t*r, cos_p*r*cos_t)
    def calc_star_pos(self, Xp, scale):
        self.star_pos = {}
        center = script.world.solar.center.lower()
        if center not in self.star_datum:
            if "sun" not in self.star_datum:
                return
            else:
                center = "sun"
        offset = Vector4D(0.0,
                          script.world.solar.dx,
                          script.world.solar.dy,
                          script.world.solar.dz) * scale
        self.star_pos[center] = Xp + offset
        loop_count = 0
        flg = True
        while loop_count < 3 and flg:
            loop_count += 1
            flg = False
            for name in self.star_datum:
                if name in self.star_pos:
                    continue
                star_data = self.star_datum[name]
                if star_data.primary_star is not None:
                    pname = self.star_datum[star_data.primary_star].name
                    if pname in self.star_pos:
                        pos = self.calc_pos(star_data.orbital_radius,
                                            star_data.orbital_phi,
                                            getattr(star_data, "orbital_theta", 0.0))
                        self.star_pos[name] = self.star_pos[pname] + pos
                        continue
                for sname in self.star_pos:
                    sdata = self.star_datum[sname]
                    if name == sdata.primary_star:
                        pos = self.calc_pos(sdata.orbital_radius,
                                            sdata.orbital_phi,
                                            getattr(sdata, "orbital_theta", 0.0))
                        self.star_pos[name] = self.star_pos[sname] - pos
                        break
                else:
                    flg = True

    def __getitem__(self, key):
        return self.stars[key]

    def __iter__(self):
        for name in self.stars:
            yield self.stars[name]

    def well_shifts(self, Xp, lens):
        """Gravitational frequency factors from the well of the dominant
        compact body `lens` = (body, distance) or None: the observer's
        blueshift and, per body, the redshift of its light climbing out of
        that well (a body's own surface redshift is applied separately)."""
        self.obs_shift = 1.0
        self.shifts = {}
        if lens is None:
            return
        body, D = lens
        Rs = 2.0 * body.GM
        if D > Rs:
            self.obs_shift = 1.0 / (1.0 - Rs/D) ** 0.5
        for star in self.stars.values():
            if star is body:
                continue
            d = (star.X - body.X).length()
            if d > Rs:
                self.shifts[star.name] = (1.0 - Rs/d) ** 0.5

    def draw(self, Xp, L, LL):
        fo = getattr(self, "obs_shift", 1.0)
        sh = getattr(self, "shifts", {})
        for star in self.stars.values():
            star.draw(Xp, L, LL, gshift=fo * sh.get(star.name, 1.0))

    def draw_glare(self, Xp, L, LL):
        fo = getattr(self, "obs_shift", 1.0)
        sh = getattr(self, "shifts", {})
        for star in self.stars.values():
            star.draw_glare(Xp, L, LL, gshift=fo * sh.get(star.name, 1.0))

    def hit_check(self, Xp):
        for star in self.stars.values():
            if star.hp > 0:
                star.hit_check(Xp, self.world)

    def calc_repulsion(self, Xp, acceleration, collision_radius, repulsion):
        U = Vector4D(1.0, 0.0, 0.0, 0.0)
        for star in self.stars.values():
            if star.hp > 0 and star.solid:
                star.X.t = Xp.t - Xp.distance_to(star.X)
                # a ship is tiny next to a planet: contact is at the surface,
                # with only a small margin, not at the ship's game-play radius
                margin = min(collision_radius, 0.02 * star.collision_radius)
                calc_repulsion(Xp, star.X, U,
                               margin + star.collision_radius,
                               repulsion,
                               acceleration)



    def ellipsoid_depth(self, Xp, name):
        """
        Normalised distance of Xp from the centre of body `name` in units of
        its own ellipsoid: < 1 inside, 1 on the surface.  None if absent.
        """
        star = self.stars.get(name)
        if star is None:
            return None
        return star.depth(Xp)
