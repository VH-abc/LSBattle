# coding: utf8
# entity/navmarkers.py
"""
Navigation markers for the solar-system bodies and the distant objects.

Every body gets a screen-space marker so the player can never lose sight
of it:
  * if its apparent angular size is below a threshold (too far to see as
    a disc) a small ring is drawn at its apparent direction, with its
    name and background-frame distance;
  * if its apparent direction is outside the field of view, an arrow is
    drawn at the edge of the screen pointing toward it.
The apparent direction is the one actually seen: the body's position on
the player's past light cone, boosted into the comoving frame, so
aberration is included.
"""
from math import pi, tan, atan2, cos, sin, hypot, sqrt

from OpenGL.GL import *

from program.utils.lensing import Lensing
from program.box import BOX
from program.const import VIEW_ANGLE
from program.text import drawSentence, MyFont
from program.utils import FlatScreen, boost

LIGHT_YEAR = 31557600.0   # in light-seconds


def format_distance(d):
    """d in light-seconds (the game's unit, c = 1)."""
    if d < 120.0:
        return "%.1f ls" % d
    if d < 7200.0:
        return "%.1f lmin" % (d/60.0)
    if d < 2*86400.0:
        return "%.1f lh" % (d/3600.0)
    if d < 0.5*LIGHT_YEAR:
        return "%.1f ld" % (d/86400.0)
    ly = d / LIGHT_YEAR
    if ly < 1000.0:
        return "%.2f ly" % ly
    if ly < 1.0e6:
        return "%.1f kly" % (ly/1000.0)
    return "%.2f Mly" % (ly/1.0e6)


def format_time(t):
    """Seconds, promoted to larger units as the value grows."""
    a = abs(t)
    if a < 120.0:
        return "%.1f s" % t
    if a < 7200.0:
        return "%.1f min" % (t/60.0)
    if a < 2*86400.0:
        return "%.2f h" % (t/3600.0)
    if a < 0.5*LIGHT_YEAR:
        return "%.2f d" % (t/86400.0)
    yr = t / LIGHT_YEAR          # a light-year of seconds is one year
    if abs(yr) < 1000.0:
        return "%.2f yr" % yr
    if abs(yr) < 1.0e6:
        return "%.2f kyr" % (yr/1000.0)
    if abs(yr) < 1.0e9:
        return "%.2f Myr" % (yr/1.0e6)
    return "%.3f Gyr" % (yr/1.0e9)


class NavMarkers(object):

    # a body smaller than this (apparent angular radius, radians) gets a ring
    ANGLE_THRESHOLD = 0.012
    CLUSTER_REL = 0.05       # merge labels of bodies closer to each other than this fraction of their distance
    MARGIN = 0.05        # screen fraction kept free at the edges for arrows
    color_ring  = (0.95, 0.85, 0.45, 0.9)
    color_arrow = (0.95, 0.85, 0.45, 0.9)
    color_text  = (1.0, 1.0, 1.0, 0.9)
    color_highlight = (0.45, 1.0, 1.0, 1.0)

    def __init__(self, solar, arrows=True):
        self.solar = solar
        self.arrows = arrows   # edge arrows for off-screen bodies (travel only)
        self.fovy = VIEW_ANGLE * pi / 180.0
        self.ty = tan(self.fovy*0.5)
        self.tx = self.ty * BOX.X / BOX.Y

    def draw(self, Xp, L, view, highlight=None):
        self.leader_of = {}
        """
        Xp: player's position (background frame)
        L: Lorentz(U_player), background -> comoving rest frame
        view: rotation rest frame -> eye (OpenGL camera looks down -z)
        """
        items = []
        Up = boost.U_of_L(L)
        for star in self.solar.stars.values():
            if star.hp <= 0:
                continue
            dX = star.X - Xp
            d_lab = dX.length()
            if not getattr(star, "solid", True) and d_lab < star.radius:
                continue
            if d_lab <= 0.0:
                continue
            dX.t = -d_lab                       # on the past light cone
            dx = boost.transform_null(Up, dX)   # comoving frame, cancellation-free
            r = -dx.t                           # rest-frame distance
            angle = star.radius / r if r > 0.0 else 1.0
            vv = [dx.x, dx.y, dx.z]
            m = hypot(hypot(vv[0], vv[1]), vv[2])
            if m > 0.0 and Lensing.current is not None:
                # bodies beyond a lens are drawn at their bent image
                w = Lensing.weight(d_lab)
                if w > 0.0:
                    vv = list(Lensing.apparent((vv[0]/m, vv[1]/m, vv[2]/m), w))
            v = view.get_rotate(vv)
            n = hypot(hypot(v[0], v[1]), v[2])
            if n == 0.0:
                continue
            ex, ey, ez = v[0]/n, v[1]/n, v[2]/n
            items.append((star, d_lab, angle, ex, ey, ez))
        if not items:
            return

        W, H = float(BOX.X), float(BOX.Y)
        th = H / 42.0                 # text height
        mx, my = W*self.MARGIN, H*self.MARGIN*1.8   # bottom/top kept clear of the HUD
        texts = []
        order = dict((name, i) for i, name in enumerate(self.solar.stars))
        for star, d_lab, angle, ex, ey, ez in items:
            key = order.get(star.name.lower(), 0)
            if star.name.lower() == highlight:
                key = (key, True)
            else:
                key = (key, False)
            label = "%s %s" % (star.name, format_distance(d_lab))
            onscreen = False
            if ez < 0.0:
                px = -ex/ez / self.tx
                py = -ey/ez / self.ty
                if abs(px) < 0.98 and abs(py) < 0.98:
                    onscreen = True
                    sx = W*0.5*(1.0 + px)
                    sy = H*0.5*(1.0 + py)
            if onscreen:
                if angle < self.ANGLE_THRESHOLD:
                    rr = th*0.6
                    texts.append((key, label, th, sx + rr*1.4, sy + th*0.5, (star, d_lab), ("ring", sx, sy, rr)))
                else:
                    # visible disc: just the label below it
                    off = angle / self.ty * H*0.5
                    texts.append((key, label, th, sx - th*MyFont.twh*len(label)*0.5, sy - off - th*0.3, None, None))
            elif self.arrows:
                # off screen: arrow on the edge pointing toward the body
                dxs, dys = ex, ey
                if ez > 0.0 and hypot(dxs, dys) < 1e-6:
                    dxs, dys = 0.0, -1.0
                m = hypot(dxs, dys)
                dxs, dys = dxs/m, dys/m
                cx, cy = W*0.5, H*0.5
                tmax = min((W*0.5 - mx) / abs(dxs) if dxs else 1e30,
                           (H*0.5 - my) / abs(dys) if dys else 1e30)
                ax, ay = cx + dxs*tmax, cy + dys*tmax
                s = th*0.8
                lw = th*0.9*MyFont.twh*len(label)
                lx = min(max(ax - dxs*(s*1.2 + lw*0.5) - lw*0.5, 2.0), W - lw - 2.0)
                ly = ay - dys*(s*1.2 + th*0.6) + th*0.5
                ly = min(max(ly, th), H - 2.0)
                texts.append((key, label, th*0.9, lx, ly, (star, d_lab), ("arrow", ax, ay, dxs, dys, s)))
        # drawSentence manages the 2D overlay itself, so text goes last.
        # Labels that would overlap are stacked downward, in a fixed per-body
        # order so the stack does not reshuffle when positions jitter.
        # Waypoints that are close to each other compared with their distance
        # from the ship (a planetary system seen from afar) show one label
        # only: the body most of the group orbits (its star), by mass on a
        # tie, with "+n" for the ones hidden.  The cyan label of the nearest
        # body (the brake's reference) is never merged away.
        ringed = [i for i, t in enumerate(texts) if t[5] is not None and not t[0][1]]
        ringed.sort(key=lambda i: (-texts[i][5][0].GM, texts[i][5][0].name))
        clusters = []                 # lists of entry indices, heaviest first
        for i in ringed:
            si, di = texts[i][5]
            for cl in clusters:
                sl, dl = texts[cl[0]][5]
                if (si.X - sl.X).length() < self.CLUSTER_REL * min(di, dl):
                    cl.append(i)
                    break
            else:
                clusters.append([i])
        stars = self.solar.stars
        def orbits(body, primary, depth=0):
            while body.orbit is not None and depth < 8:
                body = stars.get(body.orbit[0])
                if body is None:
                    return False
                if body is primary:
                    return True
                depth += 1
            return False
        hidden = set()
        extra = {}
        self.leader_of = {}           # hidden body name -> the body whose label stands for it
        for cl in clusters:
            if len(cl) < 2:
                continue
            members = [texts[i][5][0] for i in cl]
            lead = max(cl, key=lambda i: (sum(1 for m in members if orbits(m, texts[i][5][0])), texts[i][5][0].GM))
            extra[lead] = len(cl) - 1
            hidden.update(i for i in cl if i != lead)
            for i in cl:
                if i != lead:
                    self.leader_of[texts[i][5][0].name.lower()] = texts[lead][5][0]
        shown = []
        for i, t in enumerate(texts):
            if i in hidden:
                continue
            if i in extra:
                t = (t[0], t[1] + " +%d" % extra[i]) + t[2:]
            shown.append(t)
        # markers (rings, edge arrows) only for the bodies whose label shows,
        # so a merged label sits on the marker of the body it names
        FlatScreen.push()
        glLineWidth(1.5)
        for t in shown:
            mk = t[6]
            if mk is None:
                continue
            if mk[0] == "ring":
                _, sx, sy, rr = mk
                glColor(*self.color_ring)
                glBegin(GL_LINE_LOOP)
                for i in range(20):
                    a = 2.0*pi*i/20
                    glVertex(sx + rr*cos(a), sy + rr*sin(a))
                glEnd()
            else:
                _, ax, ay, dxs, dys, sz = mk
                glColor(*self.color_arrow)
                glBegin(GL_TRIANGLES)
                glVertex(ax, ay)
                glVertex(ax - dxs*sz - dys*sz*0.5, ay - dys*sz + dxs*sz*0.5)
                glVertex(ax - dxs*sz + dys*sz*0.5, ay - dys*sz - dxs*sz*0.5)
                glEnd()
        glLineWidth(1)
        FlatScreen.pop()
        placed = []
        shown.sort(key=lambda t: t[0][0])
        for key, label, h, x, y, _, _mk in shown:
            # the nearest body (the brake's reference frame) in cyan
            if key[1]:
                glColor(*self.color_highlight)
            else:
                glColor(*self.color_text)
            w = h*MyFont.twh*len(label)
            moved = True
            guard = 0
            while moved and guard < 20:
                moved = False
                guard += 1
                for (px, py, pw, ph) in placed:
                    if x < px + pw and px < x + w and y - h < py and py - ph < y:
                        y = py - ph - h*0.15
                        moved = True
            placed.append((x, y, w, h))
            drawSentence(label, h, x, y)
