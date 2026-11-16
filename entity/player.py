# coding: utf8
# entity/player.py
from math import pi, exp
import math

from OpenGL.GL import *

from go import Vector3, Vector4D, Matrix44, Lorentz, Quaternion
from go import PhaseSpace, WorldLine
from go import calc_repulsion
from entity import Bullets
from model.pointsprite import PointSprite
from model.lines import Lines
from program.utils.lensing import Lensing
from program.box import BOX
from program.utils import FlatScreen, fill_screen, DY_TEXTURE_EDGE
from program.text import drawSentence
from program import script


class PlayerState(object):
    def __init__(self):
        self.max_gun_num = len(script.player.guns)
        self.gun_mode = 0
        self.gun_num = 1
        self.make_gun_icon()
        self.reset_hp()

    def reset_hp(self):
        self.hp = script.player.hp
    
    def make_gun_icon(self):
        gun = script.player.guns[self.gun_mode]
        self.gun_info = script.player.gun_info.format%{
                "MODE":self.gun_mode+1,
                "NUM":self.gun_num,
                "MAX":self.max_gun_num,
                "NAME":gun.name,
                "POWER":gun.power,
                "RANGE":gun.range,
                "RELOAD_TIME":gun.reload_time,
                "ACTION":("Automatic" if gun.automatic else "Single")
            }
    
    def gun_change(self):
        self.gun_mode = (self.gun_mode+1)%self.gun_num
        self.make_gun_icon()

    def gun_get(self):
        if self.gun_num < self.max_gun_num:
            self.gun_num += 1
        self.gun_mode = self.gun_num - 1
        self.make_gun_icon()

class Gun(object):
    def __init__(self, world, gun_data):
        self.world = world
        self.name = gun_data.name
        self.bullets = Bullets(world, color=gun_data.bullet.color,
                                      psize=gun_data.bullet.size*world.scale)
        self.speed = gun_data.speed
        self.range = gun_data.range * world.scale
        self.reload_time = gun_data.reload_time
        self.power = gun_data.power
        if gun_data.automatic:
            self.gun_action = self._gun_action_automatic
        else:
            self._gun_flg = True
            self.gun_action = self._gun_action_single
        self.shoot_position = gun_data.shoot_position * self.world.scale
        self.div = gun_data.div

    def _shoot(self, ds, par_frame):
        player = self.world.player
        L = Lorentz(-player.P.U)
        LL = Lorentz(player.P.U)
        matrix = player.quaternion.get_RotMat()
        X = player.P.X.copy()
        """
        X: player's position in background frame
        """
        X += L.get_transform(Vector4D.from_tv(1, matrix.up)) * self.shoot_position
        v = -Vector3(matrix.forward)
        n = Vector4D.from_tv(1.0, v)
        L.transform(n)
        right = matrix.right
        self.bullets.add(X, player.P.U, LL, n, self.range)
        nlis = []
        for i in range(len(self.div)):
            m = Quaternion.from_ax((i+1)*pi/30, right).get_RotMat()
            d = m.get_rotate(v)
            div = self.div[i]
            for j in range(div):
                mm = Quaternion.from_ax(2*pi*j/div, v).get_RotMat()
                n = Vector4D.from_tv(1.0, mm.get_rotate(d))
                L.transform(n)
                nlis.append(n)
                self.bullets.add(X, player.P.U, LL, n, self.range)

        dt = ds / par_frame
        for i in range(1, par_frame):
            XX = X.get_linear_add(player.P.U, dt*i)
            self.bullets.add(XX, player.P.U, LL, n, self.range)
            for nn in nlis:
                self.bullets.add(XX, player.P.U, LL, nn, self.range)

    def _gun_action_single(self, keys, ds):
        if self._gun_flg:
            if keys.k_bullet > 0:
                self._shoot(ds, 1)
                self._gun_flg = False
                keys.k_bullet -= self.reload_time
        else:
            if keys.k_bullet == 0:
                self._gun_flg = True

    def _gun_action_automatic(self, keys, ds):
        if keys.k_bullet > 0:
            par_frame = int(ds*1000/self.reload_time) + 1
            self._shoot(ds, par_frame)
            keys.k_bullet -= self.reload_time * par_frame

class Player(object):

    def __init__(self, world, level, state, pos):
        self.world = world
        self.state = state
        self.level = level
        self.P = PhaseSpace(pos, Vector4D(1,0,0,0))
        self.quaternion = Quaternion()
        self.time = 0.0
        self.recovery_interval = script.player.recovery_interval
        self.recovery_time = self.time + self.recovery_interval
        start = self.P.copy()
        start.X.t = 0.0
        self.worldline = WorldLine(start)
        self.worldline.add(self.P)

        self.maxhp = script.player.hp
        self.hp_times = []
        self.bloot_time = self.time - 1.0
        self.acceleration = BOX.ACCEL if BOX.ACCEL is not None else script.player.acceleration
        self.planar = level.is_planar()
        # Contact repulsion is a game-play hack, not a thruster; keep it
        # from blowing up.  Player controls themselves are not capped.
        self.repulsion_cap = 10.0
        # Accumulated Thomas-Wigner rotation angle (radians), for the HUD.
        self.wigner_angle = 0.0
        # Position below double resolution: the true position is P.X + Xlo.
        # 1e11 light-seconds out a double resolves ~1e-5 ls, a stellar black
        # hole's radius, so the rounding residue of every step is kept.
        self.Xlo = [0.0, 0.0, 0.0]
        self.rest_offset = (0.0, 0.0, 0.0)
        self.aim_target = None       # body locked by the crosshair aim key
        self.dead = False
        self.death_message = None
        self.turn_speed_1 = 0.0
        self.turn_speed_2 = 0.0
        self.turn_acceleration = script.player.turn_acceleration
        self.turn_resistivity = script.player.turn_resistivity
        # fine turning (Caps Lock / Shift): no inertia, this fixed rate (rad/s)
        self.fine_turn_rate = 0.08
        # steady rate the inertial turn settles at: acceleration / damping
        self.turn_rate = self.turn_acceleration / max(self.turn_resistivity, 1e-9)
        self.collision_radius = script.player.collision_radius*world.scale
        self.landed = None            # body the ship is resting on
        self.inside = None            # black hole whose horizon the ship is within
        self.plunged = None           # black hole at whose singularity the ship ended
        self.collision_radius2 = (script.player.collision_radius*world.scale)**2
        self.delay = self.collision_radius
        self.repulsion = (script.player.repulsion + script.enemy.repulsion) * 0.5

        self.guns = [Gun(self.world, g) for g in script.player.guns]
        self.gun_get_time = -10.0

        self.hit_flg = False

        if script.player.window.texture is None:
            tex = DY_TEXTURE_EDGE
        else:
            tex = script.player.window.texture
        self.on_position = PointSprite(color=script.player.window.color,
                                       size=BOX.Y*script.player.window.size,
                                       size_w=False,
                                       texture=tex)
        r, g, b, a = script.player.window.pre_color
        self.lines = Lines([r, g, b, a/2])

        self.i = id(self.P.X)

    def get_acceleration(self, keys, ds):
        """
        Proper acceleration commanded by the controls, as a spatial vector
        in the comoving (pure-boost) rest frame.  Thrust is along the
        ship's body axes; "brake" is retro-thrust against the current
        velocity relative to the background frame.
        """
        accel = keys.k_accel
        brake_limit = None
        matrix = self.quaternion.get_RotMat()
        if accel&1:
            ac = -Vector4D.from_tv(0.0, matrix.forward)
        elif accel&2:
            ac = Vector4D.from_tv(0.0, matrix.forward)
        else:
            ac = Vector4D()

        if keys.k_accel_priority == 0:
            if accel&4:
                ac += Vector4D.from_tv(0.0, matrix.right)
            elif accel&8:
                ac -= Vector4D.from_tv(0.0, matrix.right)
        else:
            if accel&8:
                ac -= Vector4D.from_tv(0.0, matrix.right)
            elif accel&4:
                ac += Vector4D.from_tv(0.0, matrix.right)

        if keys.k_brake:
            # Brake towards a target rest frame: the background frame, or,
            # with orbits on, the nearest body's.  The target's 4-velocity
            # seen in the ship's comoving frame gives the thrust direction.
            target = Vector4D(1.0, 0.0, 0.0, 0.0)
            if BOX.ORBITS:
                body = self.world.solar.dominant(self.P.X)
                if body is not None:
                    target = body.U
            rel = Lorentz(self.P.U).get_transform(target)
            rel.t = 0.0
            if rel.squared_length() > 1e-24:
                ac += rel.get_normalize(1.0)
                # never overshoot rest: the rapidity still to shed, per second
                brake_limit = math.asinh(rel.length()) / ds

        if keys.k_orbit:
            # Brake towards the circular orbit around the nearest body at the
            # current radius, in the sense closest to the present motion:
            # static-frame orbital speed v = sqrt(GM/(r - R_s)) (exact for
            # Schwarzschild; sqrt(GM/r) far out), no orbit inside r = 3GM.
            body = self.world.solar.dominant(self.P.X)
            if body is not None and body.GM > 0.0:
                lo = self.Xlo
                dx = self.P.X.x - body.X.x + lo[0]; dy = self.P.X.y - body.X.y + lo[1]; dz = self.P.X.z - body.X.z + lo[2]
                r = (dx*dx + dy*dy + dz*dz) ** 0.5
                if r > 3.0*body.GM:
                    nx, ny, nz = dx/r, dy/r, dz/r
                    # velocity relative to the body, in the body's frame
                    u = Lorentz(body.U).get_transform(self.P.U)
                    ur = u.x*nx + u.y*ny + u.z*nz
                    tx, ty, tz = u.x - ur*nx, u.y - ur*ny, u.z - ur*nz
                    ut = (tx*tx + ty*ty + tz*tz) ** 0.5
                    if ut < 1e-30:
                        tx, ty, tz = (-nz, 0.0, nx) if abs(ny) < 0.9 else (0.0, nz, -ny)
                        ut = (tx*tx + ty*ty + tz*tz) ** 0.5
                    vc = (body.GM / (r - 2.0*body.GM)) ** 0.5
                    if vc < 1.0:
                        gc = 1.0 / (1.0 - vc*vc) ** 0.5
                        tb = Vector4D(gc, gc*vc*tx/ut, gc*vc*ty/ut, gc*vc*tz/ut)
                        bu = body.U
                        target = Lorentz(Vector4D(bu.t, -bu.x, -bu.y, -bu.z)).get_transform(tb)
                        rel = Lorentz(self.P.U).get_transform(target)
                        rel.t = 0.0
                        if rel.squared_length() > 1e-24:
                            ac += rel.get_normalize(1.0)
                            brake_limit = math.asinh(rel.length()) / ds

        if self.planar:
            ac.y = 0.0

        if keys.k_booster:
            ac.normalize(self.acceleration * script.player.turbo)
        else:
            ac.normalize(self.acceleration)
        if brake_limit is not None and ac.length() > brake_limit:
            ac.normalize(brake_limit)
        return ac

    def _rest_offset(self):
        """Store the ship's position relative to the body it rests on."""
        b = self.landed.X; X = self.P.X; lo = self.Xlo
        self.rest_offset = (X.x - b.x + lo[0], X.y - b.y + lo[1], X.z - b.z + lo[2])

    def _drift_lo(self, X_prev, U_prev, ds):
        """After a trapezoidal position update from U_prev to the current U,
        keep the rounding residue of the update in Xlo."""
        X, U = self.P.X, self.P.U
        h = 0.5 * ds
        lo = self.Xlo
        lo[0] += (U_prev[1] + U.x)*h - (X.x - X_prev[1])
        lo[1] += (U_prev[2] + U.y)*h - (X.y - X_prev[2])
        lo[2] += (U_prev[3] + U.z)*h - (X.z - X_prev[3])
        self._fold_lo()

    def _weak_step(self, acceleration, ds, X_prev, U_prev):
        """One step of thrust plus weak-field gravity, with gravity applied to
        the background-frame velocity rather than as a force in the ship's
        frame.  Per unit coordinate time the Newtonian field g changes the
        ship's local energy by gamma (v.g) (a falling photon blueshifts by
        the potential) and bends its path by (1+v^2) g_perp / v (twice the
        Newtonian deflection for light).  Expressed as a proper acceleration
        these are gamma and gamma^2 times g, which at gamma ~ 1e6 saturate
        the rapidity cap and throw the ship about; applied to the velocity
        they are bounded by the potential and the deflection themselves.
        The field is sampled at the midpoint of the step (second order).
        Returns (Wigner quaternion from the thrust, potential)."""
        P = self.P
        # 1. thrust: exact boost, position restored (one drift at the end)
        if acceleration.squared_length() > 0.0:
            Xt = (P.X.t, P.X.x, P.X.y, P.X.z)
            qw = P.transform(acceleration, ds)
            P.X.t, P.X.x, P.X.y, P.X.z = Xt
        else:
            qw = (1.0, 0.0, 0.0, 0.0)
        X, U = P.X, P.U
        h = 0.5 * ds
        Xm = Vector4D(X.t, X.x + U.x*h, X.y + U.y*h, X.z + U.z*h)
        (gx, gy, gz), phi = self.world.solar.gravity(Xm, self.Xlo)
        g = U.t
        dt = g * ds                                   # coordinate time of the step
        vx, vy, vz = U.x/g, U.y/g, U.z/g
        v2 = vx*vx + vy*vy + vz*vz
        # velocity update: transverse (1+v^2) g_perp dt, along the motion
        # g_par dt / gamma^2 (from d(gamma) = gamma v g_par dt); as a vector
        # sum it keeps the second-order kinetic term that a pure energy
        # update drops (which cost 1% of angular momentum per slow apoapsis)
        if v2 > 1.0e-12:
            v = v2 ** 0.5
            nx, ny, nz = vx/v, vy/v, vz/v
            gpar = gx*nx + gy*ny + gz*nz
            kt = (1.0 + v2) * dt
            kp = gpar * dt / (g*g) - gpar * kt       # so that g*kt + n*kp gives the split above
            wx = vx + gx*kt + nx*kp
            wy = vy + gy*kt + ny*kp
            wz = vz + gz*kt + nz*kp
        else:
            gpar = 0.0; v = 0.0
            wx = vx + gx*dt; wy = vy + gy*dt; wz = vz + gz*dt
        w2 = wx*wx + wy*wy + wz*wz
        if g < 1.0e4:
            if w2 >= 1.0:
                w2 = 1.0 - 1.0e-16
            gn = 1.0 / (1.0 - w2) ** 0.5
            U.x, U.y, U.z = gn*wx, gn*wy, gn*wz
            U.t = (1.0 + U.x*U.x + U.y*U.y + U.z*U.z) ** 0.5
        else:
            # ultra-relativistic: |v| is 1 to double precision, so take the
            # direction from the vector sum and gamma from the energy
            gn = g * (1.0 + gpar * v * dt)
            m = w2 ** 0.5
            U.x, U.y, U.z = gn*wx/m, gn*wy/m, gn*wz/m
            U.t = (1.0 + U.x*U.x + U.y*U.y + U.z*U.z) ** 0.5 if gn < 1.0e8 else gn
        # 2. drift, trapezoidal in the background frame, residue kept
        X.t += 0.5*(U_prev[0] + U.t) * ds
        X.x += 0.5*(U_prev[1] + U.x) * ds
        X.y += 0.5*(U_prev[2] + U.y) * ds
        X.z += 0.5*(U_prev[3] + U.z) * ds
        self._drift_lo(X_prev, U_prev, ds)
        return qw, phi

    def _fold_lo(self):
        """Move what Xlo can of itself into P.X (Kahan): Xlo stays below
        one unit in the last place of P.X."""
        X = self.P.X; lo = self.Xlo
        sx = X.x + lo[0]; lo[0] -= sx - X.x; X.x = sx
        sy = X.y + lo[1]; lo[1] -= sy - X.y; X.y = sy
        sz = X.z + lo[2]; lo[2] -= sz - X.z; X.z = sz

    def _geodesic_step(self, body, acceleration, ds):
        """One proper-time step of the exact Schwarzschild geodesic around
        body, plus the ship's own thrust.

        The game's coordinates around the body are read as Schwarzschild
        coordinates and the ship's U as its velocity relative to the local
        static observer (what the view and the Doppler shifts need).  The
        thrust is applied first as an exact boost in that local frame; the
        free fall then advances (r, dr/dtau, phi) with the conserved angular
        momentum L = r * u_tangential by a 4th-order Runge-Kutta step:
            d2r/dtau2 = -GM/r^2 + L^2/r^3 (1 - 3GM/r),   dphi/dtau = L/r^2,
            dt/dtau = gamma_static / sqrt(1 - R_s/r).
        At the horizon the static speed reaches c and land() takes over.
        Light bending is not modelled.
        """
        P = self.P; U = P.U; lo = self.Xlo
        # 1. thrust: exact boost of the local velocity (position restored)
        if acceleration.squared_length() > 0.0:
            Xt = (P.X.t, P.X.x, P.X.y, P.X.z)
            qw = P.transform(acceleration, ds)
            P.X.t, P.X.x, P.X.y, P.X.z = Xt
        else:
            qw = (1.0, 0.0, 0.0, 0.0)
        # 2. position and velocity relative to the body
        B = body.X
        dx = P.X.x - B.x + lo[0]; dy = P.X.y - B.y + lo[1]; dz = P.X.z - B.z + lo[2]
        r = (dx*dx + dy*dy + dz*dz) ** 0.5
        if r <= 0.0:
            return qw
        nx, ny, nz = dx/r, dy/r, dz/r
        ux, uy, uz = U.x, U.y, U.z
        ur = ux*nx + uy*ny + uz*nz
        tx, ty, tz = ux - ur*nx, uy - ur*ny, uz - ur*nz
        ut = (tx*tx + ty*ty + tz*tz) ** 0.5
        if ut > 1e-300:
            tx /= ut; ty /= ut; tz /= ut
        else:
            # radial motion: any perpendicular will do (nothing turns)
            if abs(ny) < 0.9:
                tx, ty, tz = -nz, 0.0, nx
            else:
                tx, ty, tz = 0.0, nz, -ny
            n2 = (tx*tx + ty*ty + tz*tz) ** 0.5
            tx /= n2; ty /= n2; tz /= n2
            ut = 0.0
        GM = body.GM; Rs = 2.0*GM
        hole = body.is_hole
        # A black hole does not stop the ship: the fall goes on through the
        # horizon to just short of the singularity, where the final state is
        # held (self.plunged).  Outside RAIN R_s the ship's U is its velocity
        # relative to the local static observer; inside it is relative to the
        # 'rain' observer (radial free fall from rest at infinity), whose
        # frame is regular across the horizon.  Either way the conserved
        # energy e, angular momentum L and dr/dtau are the same, and the
        # proper-time equations below hold everywhere.
        rain0 = hole and r < Lensing.RAIN*Rs
        rmin = Rs*0.02 if hole else Rs * (1.0 + 1.0e-9)
        rfloor = Rs*0.02
        if r < rmin:
            r = rmin
        f = 1.0 - Rs/r
        L = r*ut                          # conserved angular momentum per mass
        if rain0:
            bv = (Rs/r) ** 0.5            # the rain frame's inward speed
            rd = ur - U.t*bv              # dr/dtau
            e = U.t - bv*ur               # e = f dt/dtau
            g1 = U.t                      # d(rain time)/dtau
        else:
            rd = ur * f ** 0.5            # dr/dtau
            e = U.t * f ** 0.5
            g1 = U.t / f ** 0.5           # dt/dtau at the start
        # 3. RK4 on (r, rd, phi)
        L2 = L*L
        def deriv(r, rd):
            if r < rmin:
                r = rmin
            return rd, -GM/(r*r) + L2/(r*r*r)*(1.0 - 3.0*GM/r), L/(r*r)
        k1 = deriv(r, rd)
        k2 = deriv(r + 0.5*ds*k1[0], rd + 0.5*ds*k1[1])
        k3 = deriv(r + 0.5*ds*k2[0], rd + 0.5*ds*k2[1])
        k4 = deriv(r + ds*k3[0], rd + ds*k3[1])
        r2 = r + ds/6.0*(k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
        rd2 = rd + ds/6.0*(k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
        dph = ds/6.0*(k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])
        if hole:
            if r2 <= rfloor:
                # the end: the state at this moment is held, for looking around
                r2 = rfloor
                self.plunged = body
        elif r2 < rmin:
            r2 = rmin
            if rd2 < 0.0:
                rd2 = 0.0
        self.inside = body if (hole and r2 < Rs) else None
        # 4. back to the game's frame
        c, s = math.cos(dph), math.sin(dph)
        n2x, n2y, n2z = c*nx + s*tx, c*ny + s*ty, c*nz + s*tz
        t2x, t2y, t2z = c*tx - s*nx, c*ty - s*ny, c*tz - s*nz
        f2 = 1.0 - Rs/r2
        ut2 = L / r2
        if hole and r2 < Lensing.RAIN*Rs:
            bv2 = (Rs/r2) ** 0.5
            # gamma relative to the rain observer, in a form that has no
            # cancellation at the horizon: (1 + L^2/r^2 + rd^2)/(e - beta rd)
            gam = (1.0 + ut2*ut2 + rd2*rd2) / max(e - bv2*rd2, 1.0e-12)
            ur2 = rd2 + gam*bv2
            g2 = gam
        else:
            ur2 = rd2 / f2 ** 0.5
            g2 = None
        U.x = ur2*n2x + ut2*t2x; U.y = ur2*n2y + ut2*t2y; U.z = ur2*n2z + ut2*t2z
        u2 = U.x*U.x + U.y*U.y + U.z*U.z
        if u2 > 1.0e36:                   # at the horizon: keep gamma finite (same floor as the freeze)
            k = 1.0e18 / u2 ** 0.5
            U.x *= k; U.y *= k; U.z *= k; u2 = 1.0e36
        U.t = (1.0 + u2) ** 0.5 if u2 < 1.0e16 else u2 ** 0.5
        if g2 is None:
            g2 = U.t / f2 ** 0.5
        P.X.t += 0.5*(g1 + g2) * ds
        d2x, d2y, d2z = r2*n2x, r2*n2y, r2*n2z
        P.X.x = B.x + d2x; P.X.y = B.y + d2y; P.X.z = B.z + d2z
        lo[0] = d2x - (P.X.x - B.x); lo[1] = d2y - (P.X.y - B.y); lo[2] = d2z - (P.X.z - B.z)
        return qw

    def calc_repulsion(self, acceleration):
        for enemy in self.world.enemies:
            X1, U1 = enemy.worldline.get_XU_on_PLC(self.P.X)
            if X1 is not None:
                calc_repulsion(self.P.X, X1, U1,
                               self.collision_radius + enemy.collision_radius,
                               self.repulsion,
                               acceleration)
        # bodies: contact is inelastic (see land() in action), no repulsion

    LOOK_RATE = 3.0          # rad/s cap while a look key is held
    LOOK_GAIN = 8.0          # 1/s: exponential approach to the target

    def _body_at_crosshair(self):
        """The body whose apparent direction (retarded position, aberrated)
        is closest to the nose; when its label is merged into another
        body's, that body (the one named on screen)."""
        P = self.P
        f = self.quaternion.get_RotMat().forward
        nx, ny, nz = -f.x, -f.y, -f.z
        L = Lorentz(P.U)
        best, bc = None, -2.0
        solar = self.world.solar
        for body in solar.stars.values():
            if body.hp <= 0:
                continue
            Xc = body.X
            if body.orbit is not None and BOX.ORBITS:
                Xc, _ = solar.state_at(body, P.X.t - (body.X - P.X).length())
            dX = Xc - P.X
            n = dX.length()
            if n <= 0.0:
                continue
            dX.t = -n
            r = L.get_transform(dX)
            m = (r.x*r.x + r.y*r.y + r.z*r.z) ** 0.5
            if m <= 0.0:
                continue
            c = (r.x*nx + r.y*ny + r.z*nz) / m
            if c > bc:
                best, bc = body, c
        if best is not None:
            best = getattr(self.world.navmarkers, "leader_of", {}).get(best.name.lower(), best)
        return best

    def look_toward(self, keys, ds):
        """Held look keys drag the view (real time ds) towards a target
        direction in the ship's rest frame:
          look_body:   where the governing body appears (its retarded position,
                       aberrated: the past light cone),
          look_lead:   where to aim light to hit the nearest body if it keeps
                       moving inertially (the future light cone),
          look_motion: the ship's own motion relative to the nearest body,
          look_aim:    aim-ahead at the body nearest the crosshair (locked
                       while the key is held)."""
        if not keys.k_look_aim:
            self.aim_target = None
        if not (keys.k_look_body or keys.k_look_lead or keys.k_look_motion or keys.k_look_aim):
            return
        P = self.P
        tx = ty = tz = 0.0
        if keys.k_look_aim:
            # aim-ahead at the body nearest the crosshair, chosen when the
            # key goes down and kept while it is held
            if self.aim_target is None or self.aim_target.hp <= 0:
                self.aim_target = self._body_at_crosshair()
            if self.aim_target is None:
                return
            body = self.aim_target
            d = body.X - P.X
            Ub = body.U
            vx, vy, vz = Ub.x/Ub.t, Ub.y/Ub.t, Ub.z/Ub.t
            v2 = vx*vx + vy*vy + vz*vz
            dv = d.x*vx + d.y*vy + d.z*vz
            d2 = d.x*d.x + d.y*d.y + d.z*d.z
            t = (dv + (dv*dv + (1.0 - v2)*d2) ** 0.5) / (1.0 - v2)
            r = Lorentz(P.U).get_transform(Vector4D(t, d.x + vx*t, d.y + vy*t, d.z + vz*t))
            tx, ty, tz = r.x, r.y, r.z
        elif keys.k_look_motion:
            # own motion relative to the nearest body: in the ship's frame the
            # body moves at -v, so the target is minus the body's velocity
            body = self.world.solar.dominant(P.X)
            if body is None:
                tx, ty, tz = P.U.x, P.U.y, P.U.z
            else:
                rel = Lorentz(P.U).get_transform(body.U)
                tx, ty, tz = -rel.x, -rel.y, -rel.z
        else:
            solar = self.world.solar
            body = solar.dominant(P.X)
            if body is None:
                return
            Xp = P.X
            if keys.k_look_body:
                Xc = body.X
                if body.orbit is not None and BOX.ORBITS:
                    Xc, _ = solar.state_at(body, Xp.t - (body.X - Xp).length())
                dX = Xc - Xp
                dX.t = -dX.length()                       # past null vector
            else:
                # body worldline X_b + v t; hit when |d + v t| = t
                d = body.X - Xp
                Ub = body.U
                vx, vy, vz = Ub.x/Ub.t, Ub.y/Ub.t, Ub.z/Ub.t
                v2 = vx*vx + vy*vy + vz*vz
                dv = d.x*vx + d.y*vy + d.z*vz
                d2 = d.x*d.x + d.y*d.y + d.z*d.z
                t = (dv + (dv*dv + (1.0 - v2)*d2) ** 0.5) / (1.0 - v2)
                dX = Vector4D(t, d.x + vx*t, d.y + vy*t, d.z + vz*t)   # future null
            r = Lorentz(P.U).get_transform(dX)
            tx, ty, tz = r.x, r.y, r.z
        n = (tx*tx + ty*ty + tz*tz) ** 0.5
        if n < 1e-300:
            return
        tx /= n; ty /= n; tz /= n
        matrix = self.quaternion.get_RotMat()
        f = matrix.forward
        dx, dy, dz = -f.x, -f.y, -f.z                  # the nose: forward thrust is along -forward
        c = dx*tx + dy*ty + dz*tz
        ax, ay, az = dy*tz - dz*ty, dz*tx - dx*tz, dx*ty - dy*tx
        s_ = (ax*ax + ay*ay + az*az) ** 0.5
        if s_ < 1e-12:
            if c > 0.0:
                return
            u = matrix.up
            ax, ay, az = u.x, u.y, u.z; s_ = 1.0   # straight behind: pitch over
        angle = math.atan2(s_, max(-1.0, min(1.0, c)))
        step = min(angle * min(1.0, self.LOOK_GAIN * ds), self.LOOK_RATE * ds)
        self.turn_speed_1 = self.turn_speed_2 = 0.0    # no inertia fights the drag
        # the quaternion maps rest axes to the view, so a view rotation about
        # a rest-frame axis composes with the opposite sign
        self.quaternion *= Quaternion.from_ax(-step, Vector3(ax/s_, ay/s_, az/s_))
        self.quaternion.normalize()

    def change_direction(self, keys, ds):
        matrix = self.quaternion.get_RotMat()
        if keys.k_fine or not BOX.INERTIA:
            # fixed-rate turning, no angular momentum: slow in fine control,
            # the normal cruising rate when angular inertia is switched off
            self.turn_speed_1 = self.turn_speed_2 = 0.0
            a = (self.fine_turn_rate if keys.k_fine else self.turn_rate) * ds
            yaw = pitch = 0.0
            if keys.k_turn_right and not keys.k_turn_left:
                yaw = a
            elif keys.k_turn_left and not keys.k_turn_right:
                yaw = -a
            if keys.k_turn_up and not keys.k_turn_down:
                pitch = -a
            elif keys.k_turn_down and not keys.k_turn_up:
                pitch = a
            if yaw:
                self.quaternion *= Quaternion.from_ax(yaw, matrix.up)
            if pitch:
                self.quaternion *= Quaternion.from_ax(pitch, matrix.right)
        elif keys.k_turn_priority1 == 0:
            if keys.k_turn_right:
                self.turn_speed_1 += self.turn_acceleration * ds
            elif keys.k_turn_left:
                self.turn_speed_1 -= self.turn_acceleration * ds
        else:
            if keys.k_turn_left:
                self.turn_speed_1 -= self.turn_acceleration * ds
            elif keys.k_turn_right:
                self.turn_speed_1 += self.turn_acceleration * ds
        if keys.k_fine or not BOX.INERTIA:
            pass
        elif keys.k_turn_priority2 == 0:
            if keys.k_turn_up:
                self.turn_speed_2 -= self.turn_acceleration * ds
            elif keys.k_turn_down:
                self.turn_speed_2 += self.turn_acceleration * ds
        else:
            if keys.k_turn_down:
                self.turn_speed_2 += self.turn_acceleration * ds
            elif keys.k_turn_up:
                self.turn_speed_2 -= self.turn_acceleration * ds
        matrix = self.quaternion.get_RotMat()
        cap = 4.0                                   # rad/s
        self.turn_speed_1 = max(-cap, min(cap, self.turn_speed_1))
        self.turn_speed_2 = max(-cap, min(cap, self.turn_speed_2))
        self.quaternion *= Quaternion.from_ax(self.turn_speed_1 * ds, matrix.up)
        self.quaternion *= Quaternion.from_ax(self.turn_speed_2 * ds, matrix.right)
        # exact exponential damping: stable for any step size
        damp = exp(-self.turn_resistivity * ds)
        self.turn_speed_1 *= damp
        self.turn_speed_2 *= damp
        self.quaternion.normalize()
    
    def action(self, keys, ds):
        assert(self.i == id(self.P.X))
        # the pilot turns the ship in real time, whatever the simulation rate
        self.change_direction(keys, ds / BOX.TIME_RATE)
        self.look_toward(keys, ds / BOX.TIME_RATE)

        self.guns[self.state.gun_mode].gun_action(keys, ds)

        if self.plunged is not None:
            # at the singularity: nothing moves any more, only the view
            self.time += ds
            return

        if self.landed is not None and self.landed.hp > 0 and not (
                keys.k_accel or keys.k_impulse or keys.k_brake or keys.k_orbit):
            # At rest on a body with no thrust commanded: ride with it.  A
            # body on a tight orbit curves away faster than the landing
            # tolerance, so the ship is carried rather than integrated.
            body = self.landed; o = self.rest_offset; P = self.P
            P.X.x = body.X.x + o[0]; P.X.y = body.X.y + o[1]; P.X.z = body.X.z + o[2]
            self.Xlo[0] = o[0] - (P.X.x - body.X.x)
            self.Xlo[1] = o[1] - (P.X.y - body.X.y)
            self.Xlo[2] = o[2] - (P.X.z - body.X.z)
            P.U.t, P.U.x, P.U.y, P.U.z = body.U.t, body.U.x, body.U.y, body.U.z
            P.X.t += body.U.t * ds
            self.time += ds
            return

        acceleration = self.get_acceleration(keys, ds)
        if keys.k_impulse:
            # impulse booster: a fixed rapidity kick along the ship's axis,
            # applied as an exact boost in this substep (Wigner rotation too)
            keys.k_impulse = 0
            fwd = self.quaternion.get_RotMat().forward
            acceleration -= Vector4D.from_tv(0.0, fwd) * (BOX.IMPULSE_RAPIDITY / ds)
        repulsion = Vector4D()
        self.calc_repulsion(repulsion)
        if repulsion.squared_length() > self.repulsion_cap**2:
            repulsion.normalize(self.repulsion_cap)
        acceleration += repulsion
        if self.planar:
            acceleration.y = 0.0
        phi = 0.0
        strong = None
        if BOX.GRAVITY:
            # Near a compact body the motion is the exact Schwarzschild
            # geodesic (_geodesic_step); elsewhere the weak field is applied
            # in the background frame (_weak_step).
            strong = self.world.solar.strong_body(self.P.X, self.Xlo)
        # a huge time rate with thrust on would overflow sinh/cosh: cap the
        # rapidity change per substep (the ship still reaches gamma ~ e^30)
        amag = acceleration.length()
        if amag * ds > 30.0:
            acceleration.normalize(30.0 / ds)
        X_prev = (self.P.X.t, self.P.X.x, self.P.X.y, self.P.X.z)
        U_prev = (self.P.U.t, self.P.U.x, self.P.U.y, self.P.U.z)
        if strong is not None:
            qw = self._geodesic_step(strong, acceleration, ds)
        elif BOX.GRAVITY:
            qw, phi = self._weak_step(acceleration, ds, X_prev, U_prev)
        else:
            qw = self.P.transform(acceleration, ds)
            self._drift_lo(X_prev, U_prev, ds)
        # Beyond rapidity 45 (gamma ~ 2e19) double precision can no longer
        # place anything, so the velocity is frozen there: not a game-play
        # cap, a numerical floor of the representation.
        U = self.P.U
        if U.t > 1.75e19:
            self.P.X.t, self.P.X.x, self.P.X.y, self.P.X.z = X_prev
            self.P.U.t, self.P.U.x, self.P.U.y, self.P.U.z = U_prev
            self.P.transform(Vector4D(), ds)        # coast instead
            qw = (1.0, 0.0, 0.0, 0.0)
        if not all(math.isfinite(v) for v in (self.P.U.t, self.P.U.x, self.P.U.y, self.P.U.z,
                                               self.P.X.t, self.P.X.x, self.P.X.y, self.P.X.z)):
            self.P.X.t, self.P.X.x, self.P.X.y, self.P.X.z = X_prev
            self.P.U.t, self.P.U.x, self.P.U.y, self.P.U.z = U_prev
            qw = (1.0, 0.0, 0.0, 0.0)
        if phi > 0.0:
            # gravitational time dilation (first order): coordinate time runs
            # faster than the ship's clock by (1 + phi)
            self.P.X.t += phi * self.P.U.t * ds
        self.landed = self.world.solar.land(self.P, 1.0e-6, self.Xlo)   # 300 m above the surface
        if self.landed is not None:
            self._rest_offset()
        if strong is None or not strong.is_hole:
            self.inside = None
        # The ship carries its axes by Fermi-Walker transport; expressed in
        # the pure-boost rest frame that is a Wigner rotation per step.
        rw = Quaternion(*qw)
        self.wigner_angle += rw.get_angle()
        self.quaternion *= rw.get_conjugate()
        if self.planar:
            self.P.U.y = 0.0
            self.P.U.t = self.P.U.get_gamma()
            self.P.X.y = 0.0
        self.time += ds

        X1 = self.P.X
        X0 = self.worldline.get_last()
        if self.world.enemies.hit_check(X1, X0,
                                        self.collision_radius2,
                                        color=(0.95, 0.1, 0.1, 0.8)):
            self.hp_times.append(self.time + self.delay - 0.03)
            self.bloot_time = self.time + self.delay

        self.worldline.add(self.P)

        times = []
        for t in self.hp_times:
            if t < self.time:
                self.state.hp -= 1
                self.hit_flg = True
            else:
                times.append(t)
        self.hp_times = times

        if self.time > self.recovery_time:
            if self.state.hp < script.player.hp and not self.dead:
                self.state.hp += 1
                self.recovery_time = self.time + self.recovery_interval
            else:
                self.recovery_time = self.time + self.recovery_interval*0.1

    def draw_hp(self):
        FlatScreen.push()
        ox = BOX.X * script.player.hpbar.position_x
        oy = BOX.Y * script.player.hpbar.position_y
        lx = BOX.X * script.player.hpbar.length_x
        ly = BOX.Y * script.player.hpbar.length_y
        glBegin(GL_QUADS)
        if self.state.hp < self.maxhp:
            glColor(*script.player.hpbar.back_color)
            glVertex(ox,    oy)
            glVertex(ox+lx, oy)
            glVertex(ox+lx, oy+ly)
            glVertex(ox,    oy+ly)
        lh = lx * self.state.hp * 1.0 / self.maxhp
        dt = (self.time - self.bloot_time) / script.player.hpbar.blood_time
        r, g, b, a = script.player.hpbar.color
        if 0.0 < dt < 1.0:
            br, bg, bb, ba = script.player.hpbar.blood_color
            ba *= 1.0 - dt
            glColor(r, g, b, a*(1.0-dt))
            r = r*(1.0 - ba) + br*ba
            g = g*(1.0 - ba) + bg*ba
            b = b*(1.0 - ba) + bb*ba
        glColor(r, g, b, a)
        glVertex(ox,    oy)
        glVertex(ox+lh, oy)
        glVertex(ox+lh, oy+ly)
        glVertex(ox,    oy+ly)
        glEnd()
        FlatScreen.pop()
        glColor(0.5, 0.5, 1.0, 1.0)
        drawSentence("HP", ly, ox, oy+ly)

    def draw_gun_name(self):
        if not self.level.is_easy():
            text = self.state.gun_info
            glColor(*script.player.gun_info.color)
            drawSentence(text, BOX.Y * script.player.gun_info.height,
                               BOX.X * script.player.gun_info.position_x,
                               BOX.Y * script.player.gun_info.position_y)

    def draw_booster(self, keys):
        if keys.k_booster:
            glColor(1.0, 1.0, 1.0, 1.0)
            drawSentence("Booster on!", BOX.Y*0.08, BOX.X*0.01, BOX.Y*0.1)

    def bullet_hit_check(self, X1, X0, collision_radius2):
        damege = 0
        for i in range(self.state.gun_num):
            gun = self.guns[i]
            damege += gun.bullets.hit_check(X1, X0, collision_radius2) * gun.power
        return damege

    def draw_window(self, L):
        vertices_p = []
        vertices_f = []
        for enemy in self.world.enemies:
            X_PLC = enemy.worldline.get_X_FP(self.P.X, 0.0)
            if X_PLC is not None:
                vertices_p.extend(X_PLC.get_lis3())
                X_FLC = enemy.worldline.get_X_FP(self.P.X, 1.0)
                vertices_f.extend(X_FLC.get_lis3())
        if vertices_p:
            self.on_position.draw(self.P.X, L, vertices_p)
            self.on_position.draw(self.P.X, L, vertices_f,
                                  size=BOX.Y*script.player.window.pre_size,
                                  color=script.player.window.pre_color)
            glLineWidth(5)
            self.lines.draw(self.P.X, self.P.U, vertices_p, vertices_f)
            glLineWidth(1)
