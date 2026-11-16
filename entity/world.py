# coding: utf8
# entity/world.py
import math
from math import atan2, pi, cos, sin
from random import random, randint

from OpenGL.GL import *

from go import Vector3, Vector4D, Matrix44, Lorentz, Quaternion
from entity import Player
from entity import Enemies
from entity import WireFrame
from entity import Sky
from entity.axes import AxisMarkers
from entity.navmarkers import NavMarkers
from entity.glare import Glare
from entity.lensing import Lensing
from program.utils import boost
from program.utils.playing import FlatScreen
from entity import SolarSystem
from entity import Item
from program.const import *
from program.const import VIEW_ANGLE
from program.box import BOX
from program import script
from program.utils import blackbody


class World(object):

    def __init__(self, level, scale, playerstate, item=None):
        self.level = level
        self.scale = scale
        self.playerstate = playerstate
        self.L = self.level.L * scale
        self.wireframe = WireFrame(scale=self.scale)
        self.sky = Sky()
        self.axes = AxisMarkers()
        self.player = Player(world=self, level=level,
                             state=playerstate,
                             pos=Vector4D(0, 0, 0, self.L))
        self.solar = SolarSystem(self, self.player.P.X, self.scale)
        self.solar.player_lo = self.player.Xlo
        self.navmarkers = NavMarkers(self.solar, arrows=level.is_travel())
        # start where the ship was last sent (Travel to), else at Earth
        if not self.start_in_orbit(BOX.SPAWN, self.spawn_factor(BOX.SPAWN)):
            self.start_in_orbit("earth", 2.0)
        self.glare = Glare()
        self.lensing = Lensing()
        self.enemies = Enemies(world=self)
        self.item = None
        self.worldline_ref = {}
        if not level.is_travel():
            self.init_set_enemy()
            self.init_set_item(item)
            self.init_set_worldline()
        self.score = 0

    def init_set_enemy(self):
        types = [randint(0, self.level.types - 1)for i in range(self.level.enemy_num-1)]
        types += [self.level.types-1]
        for typ in types:
            x = 6.0 * self.L * (2.0*random()-1.0)
            y = 6.0 * self.L * (2.0*random()-1.0)
            z = 6.0 * self.L * (1.0*random()-2.0)
            if self.level.is_planar():
                y = 0.0
            self.enemies.add(Vector4D(0, x, y, z), typ=typ, level=self.level)

    def init_set_item(self, item):
        if item is not None and item == self.playerstate.gun_num:
            pos = self.player.P.X + Vector4D(0, 0, 0.5, -5.0) * self.scale
            self.item = Item(item, pos, self.scale)

    def init_set_worldline(self):
        self.worldline_ref = {}
        lis = list(self.enemies) + [self.player]
        for a in lis:
            self.worldline_ref[id(a)] = a.worldline
            for b in lis:
                if a is b:
                    continue
                a.worldline.set_id(id(b.P.X))

    MAX_SUBSTEPS = 200

    def action(self, keys, ds):
        # Free-space motion is integrated exactly for any step, so the step
        # is set by gravity alone.  The step is chosen afresh for every
        # substep from the ship's current position: a frame that starts far
        # from a mass but plunges close to it shrinks its steps as it goes.
        # When the substep budget runs out the simulated time per frame is
        # cut short (rate_limit < 1 tells the HUD).
        self.rate_limit = 1.0
        # The pilot turns in real time: the whole frame's real time goes to
        # the first substep's turn, however the simulated time is cut or split
        self.player.turn_budget = ds / BOX.time_rate()
        self._adapt_rate(ds / BOX.time_rate())
        # Approach limiter: at a high time rate one frame can carry the ship
        # past a target, so the step is cut to the proper time in which the
        # ship closes 30% of the distance to the nearest body.  Only while
        # closing in: orbits and departures run at full rate.
        body = self.solar.nearest(self.player.P.X)
        if body is not None:
            P = self.player.P
            dx, dy, dz = body.X.x - P.X.x, body.X.y - P.X.y, body.X.z - P.X.z
            r = (dx*dx + dy*dy + dz*dz) ** 0.5
            d = r - body.collision_radius
            if r > 0.0 and d > 0.0:
                closing = (dx*P.U.x + dy*P.U.y + dz*P.U.z) / r     # coordinate distance per proper time
                if closing > 0.0:
                    # never below 2% of the body's radius per frame, or contact
                    # would only ever be approached
                    cap = 0.3 * max(d, 0.07 * body.collision_radius) / closing
                    if cap < ds:
                        self.rate_limit = cap / ds
                        ds = cap
        if BOX.GRAVITY:
            remaining = ds
            count = 0
            while remaining > 0.0 and count < self.MAX_SUBSTEPS:
                dt = min(remaining, self.solar.max_step(self.player.P.X, self.player.P.U))
                self._step(keys, dt)
                remaining -= dt
                count += 1
            self._nsub = count
            if remaining > 0.0:
                self.rate_limit *= (ds - remaining) / ds
        else:
            n = min(int(ds * 10.0) + 1, 20)
            dt = ds / n
            for _ in range(n):
                self._step(keys, dt)
        for wl in self.worldline_ref.values():
            wl.cut()

    def _step(self, keys, ds):
        self.solar.update(self.player.P.X.t)
        self.player.action(keys, ds)
        self.score += self.enemies.action(ds)
        self.solar.hit_check(self.player.P.X)
        self.item_action(ds)

    def spawn_factor(self, name):
        """Orbit radius for a start at body `name`, in body radii: 2 for a
        planet or star; 6 Schwarzschild radii for a black hole (twice the
        innermost stable orbit, outside the inner disc, with it in view)."""
        body = self.solar.stars.get(name.lower()) if name else None
        return 6.0 if body is not None and body.is_hole else 2.0

    def start_in_orbit(self, name, factor):
        """Start the ship on a circular orbit of `factor` body radii about
        body `name`, in the plane of the body's own orbit, prograde, over the
        day side so the body shows mostly lit, with the nose on the body."""
        body = self.solar.stars.get(name)
        if body is None or body.GM <= 0.0:
            return False
        P = self.player.P
        self.solar.update(P.X.t)
        nx, ny, nz = 1.0, 0.0, 0.0
        if body.orbit is not None:
            prim = self.solar.stars.get(body.orbit[0])
            if prim is not None:
                nx, ny, nz = body.X.x - prim.X.x, body.X.y - prim.X.y, body.X.z - prim.X.z
                n = (nx*nx + ny*ny + nz*nz) ** 0.5
                if n > 0.0:
                    nx, ny, nz = nx/n, ny/n, nz/n
        if self.level.is_planar():
            if abs(body.X.y - P.X.y) > 1e-9:
                # the body is off the play plane: no orbit, but at least
                # move with it instead of being left behind
                P.U.t, P.U.x, P.U.y, P.U.z = body.U.t, body.U.x, body.U.y, body.U.z
                return False
            ny = 0.0
            n = (nx*nx + nz*nz) ** 0.5
            nx, nz = nx/n, nz/n
        # tangent: (0,1,0) x n, prograde about +y like the planets
        tx, ty, tz = nz, 0.0, -nx
        t = (tx*tx + tz*tz) ** 0.5
        if t < 1e-12:
            tx, ty, tz, t = 1.0, 0.0, 0.0, 1.0
        tx /= t; tz /= t
        if body.is_hole:
            # inclined 30 degrees to the accretion disc (pole +y): the disc is
            # seen from above, never edge-on
            ci, si = cos(pi/6.0), sin(pi/6.0)
            tx, ty, tz = ci*tx, si, ci*tz
        # along the orbit to 60 degrees from the sub-solar point: the body
        # shows three-quarters lit with the terminator in view.  n was the
        # anti-sun direction, so the sun direction is -n.
        ca, sa = 0.5, 0.75 ** 0.5
        sx, sy, sz = -nx, -ny, -nz
        nx, ny, nz = ca*sx + sa*tx, ca*sy + sa*ty, ca*sz + sa*tz
        tx, ty, tz = sa*sx - ca*tx, sa*sy - ca*ty, sa*sz - ca*tz
        r = factor * body.collision_radius
        vc = (body.GM / (r - 2.0*body.GM)) ** 0.5
        P.X.x = body.X.x + nx*r; P.X.y = body.X.y + ny*r; P.X.z = body.X.z + nz*r
        ux = body.U.x + vc*tx; uy = body.U.y + vc*ty; uz = body.U.z + vc*tz
        P.U.x, P.U.y, P.U.z = ux, uy, uz
        P.U.t = (1.0 + ux*ux + uy*uy + uz*uz) ** 0.5
        self.player.Xlo[0] = self.player.Xlo[1] = self.player.Xlo[2] = 0.0
        self.player.landed = None
        # nose on the body: rotate the default nose (0,0,-1) onto -n
        ax, ay, az = -ny, nx, 0.0                    # (0,0,-1) x (-n)
        sn = (ax*ax + ay*ay) ** 0.5
        cs = nz                                      # (0,0,-1).(-n)
        if sn > 1e-12:
            self.player.quaternion = Quaternion.from_ax(-atan2(sn, cs), Vector3(ax/sn, ay/sn, 0.0))
        elif cs < 0.0:
            self.player.quaternion = Quaternion.from_ax(pi, Vector3(0.0, 1.0, 0.0))
        self.player.quaternion.normalize()
        return True

    def teleport(self, name):
        """Put the ship at rest on the surface of body `name` (a preset's
        START), on the side facing away from the body's primary."""
        body = self.solar.stars.get(name.lower())
        if body is None:
            return False
        P = self.player.P
        self.solar.update(P.X.t)
        nx, ny, nz = 0.0, 1.0, 0.0
        if body.orbit is not None:
            prim = self.solar.stars.get(body.orbit[0])
            if prim is not None:
                nx, ny, nz = body.X.x - prim.X.x, body.X.y - prim.X.y, body.X.z - prim.X.z
                n = (nx*nx + ny*ny + nz*nz) ** 0.5
                if n > 0.0:
                    nx, ny, nz = nx/n, ny/n, nz/n
        r = body.collision_radius + 1.0e-6
        P.X.x = body.X.x + nx*r; P.X.y = body.X.y + ny*r; P.X.z = body.X.z + nz*r
        P.U.t, P.U.x, P.U.y, P.U.z = body.U.t, body.U.x, body.U.y, body.U.z
        self.player.Xlo[0] = self.player.Xlo[1] = self.player.Xlo[2] = 0.0
        self.player.landed = body
        self.player._rest_offset()
        return True

    def _adapt_rate(self, dt_real):
        """Adaptive time rate: the factor for the next frame from the orbital
        period at the ship's distance from its governing body, smoothed over
        about a third of a second so a hand-over between bodies is not a jolt."""
        if not BOX.ADAPTIVE:
            BOX.rate_factor = 1.0
            return
        body = self.solar.dominant(self.player.P.X)
        if body is None or body.GM <= 0.0:
            return
        r = max((body.X - self.player.P.X).length(), body.collision_radius)
        T = 2.0*pi * (r*r*r / body.GM) ** 0.5
        target = max(1.0e-6, min(1.0e12, T / BOX.ADAPTIVE_T0))
        k = min(1.0, 3.0*max(dt_real, 0.0))
        BOX.rate_factor += (target - BOX.rate_factor) * k if BOX.rate_factor > 0.0 else target

    def _auto_exposure(self):
        """Mean brightness of the drawn frame (its top mip level), fed to
        BOX.auto_exposure_update with the real time since the last frame."""
        import sdl2
        now = sdl2.SDL_GetTicks() * 0.001
        last = getattr(self, "_auto_t", None)
        self._auto_t = now
        dt = min(0.1, now - last) if last is not None else 0.0
        W, H = int(BOX.X), int(BOX.Y)
        if getattr(self, "_auto_tex", None) is None or self._auto_size != (W, H):
            if getattr(self, "_auto_tex", None) is None:
                self._auto_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._auto_tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
            self._auto_size = (W, H)
            self._auto_levels = int(math.log(max(W, H), 2)) + 1
        else:
            glBindTexture(GL_TEXTURE_2D, self._auto_tex)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 0, 0, W, H, 0)
        top = self._auto_levels - 1
        # the top level is 1x1 when W and H are powers of two; otherwise a few
        # texels: average them
        data = glGetTexImage(GL_TEXTURE_2D, top, GL_RGB, GL_UNSIGNED_BYTE)
        glBindTexture(GL_TEXTURE_2D, 0)
        buf = bytes(data) if not isinstance(data, bytes) else data
        if buf:
            mean = sum(buf) / float(len(buf)) / 255.0
            BOX.auto_exposure_update(mean, dt)

    def draw_crosshair(self):
        """Thin reticle at the screen centre: four ticks with a gap, drawn
        in white over a dark outline so it reads on any background."""
        cx, cy = BOX.X*0.5, BOX.Y*0.5
        r_in, r_out = BOX.Y*0.012, BOX.Y*0.035
        glDisable(GL_DEPTH_TEST)
        FlatScreen.push()
        for width, color in ((3.0, (0.0, 0.0, 0.0, 0.6)), (1.0, (1.0, 1.0, 1.0, 0.9))):
            glLineWidth(width)
            glColor(*color)
            glBegin(GL_LINES)
            glVertex2f(cx - r_out, cy); glVertex2f(cx - r_in, cy)
            glVertex2f(cx + r_in, cy);  glVertex2f(cx + r_out, cy)
            glVertex2f(cx, cy - r_out); glVertex2f(cx, cy - r_in)
            glVertex2f(cx, cy + r_in);  glVertex2f(cx, cy + r_out)
            glEnd()
        glLineWidth(1.0)
        FlatScreen.pop()
        glEnable(GL_DEPTH_TEST)

    def item_action(self, ds):
        if self.item is not None:
            self.item.action(ds)
            if self.item.check_collision(self.player.P.X, self.player.collision_radius2*4):
                self.player.state.gun_get()
                self.player.gun_get_time = self.player.time
                self.item = None

    def draw(self, keys):
        # near plane: a third of the distance to the nearest surface, so a
        # landed ship still sees the ground, but never beyond the default
        body = self.solar.nearest(self.player.P.X)
        if body is not None:
            d = (body.X - self.player.P.X).length() - body.collision_radius
            BOX.set_near(max(1.0e-7, min(BOX.near_default, 0.3*d)))
        L = Lorentz(self.player.P.U)
        LL = Lorentz(-self.player.P.U)
        Xp = self.player.P.X
        matrix = self.player.quaternion.get_RotMat()
        if keys.k_look_behind:
            matrix *= Matrix44.y_rotation(pi)
        matrix_i = matrix.get_inverse_rot()

        if BOX.COLOR_MODE == "ACCURATE":
            blackbody.bind_luts()
        glDisable(GL_DEPTH_TEST)
        glLoadMatrixd(matrix_i.to_opengl())
        sun = self.solar.star_pos.get("sun") if hasattr(self.solar, "star_pos") else None
        # panorama weight: 1 inside the Milky Way disc, fading to 0 by 1.6x its size
        depth = self.solar.ellipsoid_depth(Xp, "milkyway") if hasattr(self.solar, "ellipsoid_depth") else None
        if depth is None:
            weight = 1.0
        else:
            weight = max(0.0, min(1.0, (1.6 - depth) / 0.6))
        # the dominant compact body, if any: lensing and gravitational shifts
        lens = Lensing.pick(self.solar, Xp)
        self.solar.well_shifts(Xp, lens)
        fo = self.solar.obs_shift
        if BOX.LENSING and lens is not None:
            body, D = lens
            Xc = body.X
            if body.orbit is not None and BOX.ORBITS:
                Xc, _ = self.solar.state_at(body, Xp.t - D)
            Lensing.set_current(lens, Xc, Xp, self.player.P.U)
        else:
            Lensing.set_current(None, None, None, None)
        glDepthMask(GL_FALSE)
        if not getattr(self, "_sky_checked", False):
            self._sky_checked = True
            from program.utils.shader import gl_log
            gl_log("before sky: glGetError %d; sky quad %s; program %s" % (
                glGetError(), list(self.sky.vertex)[:6], getattr(self.sky, "program", None)))
        self.sky.draw(matrix_i, self.player.P.U, (Xp - sun) if sun is not None else None, weight, gshift=fo)
        if getattr(self, "_sky_checked", False) is True:
            self._sky_checked = "done"
            from program.utils.shader import gl_log
            gl_log("after sky: glGetError %d, lens %s" % (glGetError(), Lensing.current is not None))
        glDepthMask(GL_TRUE)
        if BOX.MARKERS:
            self.axes.draw(L, matrix_i)
        if keys.k_map and BOX.GRID:
            self.wireframe.draw(Xp, L, self.player.P.U)
        glEnable(GL_DEPTH_TEST)
        self.solar.draw(Xp, L, LL)
        self.enemies.draw(Xp, L, LL, self.worldline_ref)
        if self.item is not None:
            self.item.draw(Xp, L, LL)
        
        for gun in self.player.guns:
            gun.bullets.draw(Xp, L)
        self.enemies.bullets.draw(Xp, L)

        if BOX.GLARE:
            sky_args = (matrix_i, self.player.P.U, (Xp - sun) if sun is not None else None, weight)
            def sources(k):
                if k == 1.0:
                    glDisable(GL_DEPTH_TEST)
                    self.sky.draw(*sky_args, scale=1.0/self.solar.GLARE_REF, gshift=fo)
                    glEnable(GL_DEPTH_TEST)
                self.solar.draw_glare(Xp, L, LL, scale=k)
            self.glare.apply(sources)

        if BOX.AUTOEXP and BOX.COLOR_MODE == "ACCURATE":
            self._auto_exposure()
        else:
            BOX.auto_ev = 0.0

        if not BOX.MARKERS:
            self.navmarkers.leader_of = {}
        if BOX.MARKERS:
            near = self.solar.dominant(Xp)
            self.navmarkers.draw(Xp, L, matrix, near.name.lower() if near is not None else None)
        if BOX.CROSSHAIR:
            self.draw_crosshair()

        if keys.k_map:
            glDisable(GL_DEPTH_TEST)
            self.player.draw_window(L)
            glEnable(GL_DEPTH_TEST)
            self.player.draw_hp()
            self.player.draw_gun_name()
        self.player.draw_booster(keys)
