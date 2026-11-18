# coding: utf8
# entity/axes.py
"""
Direction markers at infinity, fixed in the original (background) rest
frame.  Each marker is the label of a coordinate axis, "+x", "-x", ...,
drawn in the direction from which light coming along that axis reaches
the player.  Because the markers sit at infinity only aberration acts on
them: they show where the background axes appear to be from the moving,
rotated ship, and so make Thomas-Wigner rotation of the ship directly
visible.
"""
import math

from OpenGL.GL import *

from go import Vector3, Vector4D
from program.utils.lensing import Lensing
from program.box import BOX
from program.text import drawSentence3d
from program.utils import boost


class AxisMarkers(object):

    def __init__(self):
        self.markers = [
            ("+x", ( 1.0,  0.0,  0.0), (1.0, 0.45, 0.45, 1.0)),
            ("-x", (-1.0,  0.0,  0.0), (0.6, 0.25, 0.25, 1.0)),
            ("+y", ( 0.0,  1.0,  0.0), (0.45, 1.0, 0.45, 1.0)),
            ("-y", ( 0.0, -1.0,  0.0), (0.25, 0.6, 0.25, 1.0)),
            ("+z", ( 0.0,  0.0,  1.0), (0.5, 0.6, 1.0, 1.0)),
            ("-z", ( 0.0,  0.0, -1.0), (0.3, 0.35, 0.65, 1.0)),
        ]

    def draw(self, L, view_i):
        """
        L: Lorentz(U_player), the boost from the background frame into the
           player's comoving rest frame.
        view_i: inverse of the view rotation; its rows give the camera's
           right/up/forward axes in rest-frame coordinates (used to
           billboard the labels).
        """
        dist = 0.45 * BOX.far_clip
        height = dist * 0.05
        glUseProgram(0)
        glDisable(GL_CULL_FACE)
        Up = boost.U_of_L(L)
        right = Vector3(view_i.get_rotate([1.0, 0.0, 0.0]))
        up = Vector3(view_i.get_rotate([0.0, 1.0, 0.0]))
        for label, d, color in self.markers:
            # Photon arriving from direction d: 4-momentum (1, -d).
            # (1, -d) = -(past-directed null vector with spatial part d)
            kp = -boost.transform_null(Up, Vector4D(-1.0, d[0], d[1], d[2]))
            # Apparent direction of the source in the rest frame.
            n = Vector3(-kp.x, -kp.y, -kp.z)
            n.normalize()
            if Lensing.current is not None:
                # a marker at infinity is bent round the lens like the sky
                lw = Lensing.weight(1.0e30)
                if lw > 0.0:
                    n = Vector3(*Lensing.apparent((n.x, n.y, n.z), lw))
            pos = Vector4D(0.0, n.x*dist, n.y*dist, n.z*dist)

            # Doppler factor of the photon: >1 ahead (blue), <1 behind.
            D = 1.0 / kp.t
            r, g, b, a = color
            glColor(r, g, b, a)
            # ring around the marker
            glBegin(GL_LINE_LOOP)
            rr = height * 0.9
            for i in range(24):
                ang = 2.0*math.pi*i/24
                c, s = math.cos(ang), math.sin(ang)
                glVertex(pos.x + (right.x*c + up.x*s)*rr,
                         pos.y + (right.y*c + up.y*s)*rr,
                         pos.z + (right.z*c + up.z*s)*rr)
            glEnd()
            glColor(1.0, 1.0, 1.0, 1.0)
            th = height * 0.8
            # text hangs below its anchor, so lift it by half its height
            tpos = Vector4D(0.0, pos.x + up.x*th*0.5, pos.y + up.y*th*0.5, pos.z + up.z*th*0.5)
            drawSentence3d(label, th, tpos, view_i)
        glEnable(GL_CULL_FACE)
