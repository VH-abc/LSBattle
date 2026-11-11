# coding: utf8
# box.py
import math
import os, sys

from OpenGL.GL import *
from OpenGL.GLU import *
import sdl2
import sdl2.ext

from program.const import VIEW_ANGLE, GAME_NAME, CONFIG_DIR, disp_sizes
from program import script


class _Box(object):
    
    def __init__(self):
        """
        Load setting file from CONFIG_DIR/setting.ini
        if loading is failed, do nothing
        """
        self.window = None
        self.context = None
        self.X = 800
        self.Y = 600
        self.FULL_SCREEN = False
        self.MODE = "EASY"
        self.SAVE = False
        self.ACCEL = None          # ship proper acceleration (c/s); None = script default
        self.COLOR_MODE = "GAME"   # "GAME" (original look) or "ACCURATE" (blackbody Doppler)
        self.HDR = False           # request a float framebuffer + macOS EDR
        self.EXPOSURE = 1.0        # multiplier used by the accurate colour mode
        self.GRID = True           # show the background grid with the HUD
        self.BRAKE = True          # allow the brake (retro-thrust) key
        self.GLARE = False         # glare / bloom post-effect
        self.CMB = False           # cosmic microwave background in the sky
        self.CROSSHAIR = False     # aiming reticle at the screen centre
        self.LENSING = True        # gravitational light bending near a compact body
        self.INERTIA = True        # arrow keys accelerate the turn (off: fixed rate, instant stop)
        self.MARKERS = True        # navigation markers (bodies and +x/-x... axes)
        self.GRAVITY = False       # bodies attract the ship (weak-field geodesics)
        self.ORBITS = False        # bodies move on Kepler orbits
        self.TIME_RATE = 1.0       # proper seconds simulated per real second
        self.IMPULSE = False       # booster key: one fixed rapidity kick per tap
        self.IMPULSE_RAPIDITY = 0.5
        self.SPAWN = "earth"       # body the ship starts in orbit about (the last Travel-to target)
        self.edr_headroom = 1.0
        self.edr_enabled = False
        self.edr_note = ""
        try:
            for line in open(CONFIG_DIR+"setting.ini"):
                line = line.strip()
                w, v = line.split("=")
                if w == "DISPLAY_SIZE_X":
                    X = int(v)
                elif w == "DISPLAY_SIZE_Y":
                    Y = int(v)
                elif w == "FULL_SCREEN":
                    if v == "True":
                        FULL_SCREEN = True
                    else:
                        FULL_SCREEN = False
                elif w == "MODE":
                    if v in ("EASY", "NORMAL", "HARD", "TRAVEL",
                             "PLANAR", "PLANAR_TRAVEL"):
                        self.MODE = v
                elif w == "ACCEL":
                    self.ACCEL = float(v)
                elif w == "COLOR_MODE":
                    if v in ("GAME", "ACCURATE"):
                        self.COLOR_MODE = v
                elif w == "HDR":
                    self.HDR = (v == "True")
                elif w == "EXPOSURE":
                    self.EXPOSURE = float(v)
                elif w == "TIME_RATE":
                    self.TIME_RATE = float(v)
                elif w == "IMPULSE_RAPIDITY":
                    self.IMPULSE_RAPIDITY = float(v)
                elif w == "SPAWN":
                    self.SPAWN = v.strip().lower()
                elif w in ("GRID", "BRAKE", "GLARE", "CMB", "CROSSHAIR", "MARKERS", "GRAVITY", "ORBITS", "IMPULSE", "LENSING", "INERTIA"):
                    setattr(self, w, (v == "True"))
            if (X, Y) not in disp_sizes:
                X, Y = disp_sizes[0]
            self.X = X
            self.Y = Y
            self.FULL_SCREEN = FULL_SCREEN
        except:
            self.SAVE = True
            pass
        
    def game_init(self):
        if not sdl2.SDL_WasInit(sdl2.SDL_INIT_EVENTS) and self.window is None:
            # macOS: holding a key opens the accent picker, which then eats
            # the arrow keys.  Disable press-and-hold for this process.
            try:
                import Foundation
                Foundation.NSUserDefaults.standardUserDefaults().setBool_forKey_(
                    False, "ApplePressAndHoldEnabled")
            except Exception:
                pass
            sdl2.ext.init()
            w_mode = sdl2.SDL_DisplayMode()
            sdl2.SDL_GetCurrentDisplayMode(0, w_mode)
            if self.X > w_mode.w or self.Y > w_mode.h:
                self.X, self.Y = disp_sizes[0]
                self.SAVE = True
            flg = sdl2.SDL_WINDOW_OPENGL
            # LSB_HIDDEN=1: off-screen test runs (harnesses) use a hidden,
            # windowed, fixed-size window so they never take the display
            hidden = bool(os.environ.get("LSB_HIDDEN"))
            if hidden:
                flg |= sdl2.SDL_WINDOW_HIDDEN
                self.FULL_SCREEN = False
                self.X, self.Y = 1024, 768
            # SDL_GL_FLOATBUFFERS (= 27, newer than the bundled wrapper):
            # half-float colour buffer, needed for values above 1.0 (EDR).
            sdl2.SDL_GL_SetAttribute(27, 1 if self.HDR else 0)
            if self.FULL_SCREEN:
                flg |= sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP
                self.X = w_mode.w
                self.Y = w_mode.h
                self.SAVE = True
            self.window = sdl2.SDL_CreateWindow(GAME_NAME,
                                                sdl2.SDL_WINDOWPOS_UNDEFINED,
                                                sdl2.SDL_WINDOWPOS_UNDEFINED,
                                                self.X, self.Y,
                                                flg)
            if not self.window:
                print((sdl2.SDL_GetError()))
                sdl2.ext.SDL_quit()
                sys.exit(-1)
            self.context = sdl2.SDL_GL_CreateContext(self.window)
            # The game only uses raw key events; with text input on, held
            # keys are routed through the Cocoa text system (accent popup).
            sdl2.SDL_StopTextInput()
            
            self.opengl_init()
            self.resize()
            self.hdr_active = self.HDR
            self.setup_edr()

    def setup_edr(self):
        """
        Ask macOS for extended dynamic range on the window's layer so that
        framebuffer values above 1.0 are shown brighter than SDR white.
        Needs pyobjc; records what happened in edr_note.
        """
        self.edr_enabled = False
        self.edr_headroom = 1.0
        self.edr_note = ""
        if not self.HDR:
            return
        try:
            import AppKit
            import Quartz
            win = self._ns_window()
            if win is None:
                self.edr_note = "EDR: SDL window not found"
                return
            view = win.contentView()
            if view is None:
                self.edr_note = "EDR: window has no content view"
                return
            notes = []
            if view.respondsToSelector_("setWantsExtendedDynamicRangeOpenGLSurface:"):
                view.setWantsExtendedDynamicRangeOpenGLSurface_(True)
                notes.append("GL surface")
            layer = view.layer()
            if layer is None:
                self.edr_note = "EDR: view has no layer"
                return
            if not layer.respondsToSelector_("setWantsExtendedDynamicRangeContent:"):
                self.edr_note = "EDR: layer %s has no EDR flag" % layer.className()
                return
            layer.setWantsExtendedDynamicRangeContent_(True)
            cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceExtendedSRGB)
            if cs is not None and layer.respondsToSelector_("setColorspace:"):
                layer.setColorspace_(cs)
            screen = win.screen() or AppKit.NSScreen.mainScreen()
            self.edr_headroom = float(screen.maximumExtendedDynamicRangeColorComponentValue())
            self.edr_enabled = True
            notes.append("layer")
            self.edr_note = "EDR on (%s; headroom x%.2f, potential x%.0f)" % (
                "+".join(notes), self.edr_headroom,
                float(screen.maximumPotentialExtendedDynamicRangeColorComponentValue()))
        except Exception as exc:
            self.edr_note = "EDR unavailable: %s" % exc

    def _ns_window(self):
        """The NSWindow behind the SDL window: asked from SDL itself, so a
        previous window that is still closing can never be picked up."""
        import ctypes
        import AppKit
        try:
            import objc
            info = sdl2.SDL_SysWMinfo()
            sdl2.SDL_VERSION(info.version)
            if sdl2.SDL_GetWindowWMInfo(self.window, ctypes.byref(info)):
                ptr = info.info.cocoa.window
                if ptr:
                    return objc.objc_object(c_void_p=ptr)
        except Exception:
            pass
        title = GAME_NAME.decode() if isinstance(GAME_NAME, bytes) else str(GAME_NAME)
        app = AppKit.NSApplication.sharedApplication()
        for w in reversed(list(app.windows())):
            if str(w.title()) == title and w.contentView() is not None and w.isVisible():
                return w
        return None

    def sdl2_quit(self):
        sdl2.SDL_GL_DeleteContext(self.context)
        sdl2.SDL_DestroyWindow(self.window)
        sdl2.ext.quit()
        self.window = None
        self.context = None
        
    def game_quit(self):
        self.save()
        self.sdl2_quit()
        sys.exit(0)
    
    def opengl_init(self):
        """
        setting some OpenGL parameter
        """
        glEnable(GL_DEPTH_TEST)
        # <= so that bodies at the depth limit (near plane pulled in while
        # landed) are still drawn against the cleared buffer
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_CULL_FACE)
        glShadeModel(GL_SMOOTH)
        glPixelStorei(GL_PACK_ALIGNMENT, 4)
        
        # glFogfv(GL_FOG_COLOR, (1.0, 1.0, 1.0, 1.0))
        # glFogi(GL_FOG_MODE, GL_EXP)
        # glFogf(GL_FOG_START, 2)
        # glFogf(GL_FOG_END, 7)
        # glEnable(GL_FOG)

        glEnableClientState(GL_VERTEX_ARRAY)
        # glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)

        glClearColor(*script.ui.backimage.color)

        # glEnable(GL_LINE_STIPPLE)
        # glLineStipple(4 , 0xAAAA)

    def resize(self, scale=1.0):
        """
        initialize opengl's matrix
        """
        glViewport(0, 0, self.X, self.Y)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self._perspective(script.ui.near_clip*scale, script.ui.far_clip*scale)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        self.far_clip = script.ui.far_clip*scale
        self.near_clip = script.ui.near_clip*scale
        self.near_default = self.near_clip
        self.scale = scale

    def set_near(self, near):
        """Move the near clip plane (the far plane stays): used to keep a
        surface visible when the ship is standing on it."""
        if abs(near - self.near_clip) < 0.05*self.near_clip:
            return
        self.near_clip = near
        glMatrixMode(GL_PROJECTION)
        self._perspective(near, self.far_clip)
        glMatrixMode(GL_MODELVIEW)

    def set_fov(self, fov):
        """Reload the projection with vertical field of view `fov` (degrees)
        and the current near and far planes."""
        glMatrixMode(GL_PROJECTION)
        self._perspective(self.near_clip, self.far_clip, fov)
        glMatrixMode(GL_MODELVIEW)

    def _perspective(self, near, far, fov=None):
        """Load the perspective projection.  gluPerspective is not used: on
        this platform it silently keeps the previous matrix for a near plane
        below about 5e-6, which left a landed ship seeing through the
        ground.  Built in double precision, loaded with glLoadMatrixd."""
        ct = 1.0 / math.tan(math.radians(fov if fov else VIEW_ANGLE) * 0.5)
        asp = self.X * 1.0 / self.Y
        A = -(far + near) / (far - near)
        B = -2.0 * far * near / (far - near)
        m = (GLdouble * 16)(ct/asp, 0.0, 0.0, 0.0,
                            0.0, ct, 0.0, 0.0,
                            0.0, 0.0, A, -1.0,
                            0.0, 0.0, B, 0.0)
        glLoadMatrixd(m)

    def set_displaysize(self, i):
        if disp_sizes[i][0] != self.X and disp_sizes[i][1] != self.Y:
            self.X, self.Y = disp_sizes[i]
            self.FULL_SCREEN = False
            self.SAVE = True
            self.sdl2_quit()
            self.game_init()

    def set_fullscreen(self):
        if not self.FULL_SCREEN:
            self.FULL_SCREEN = True
            self.SAVE = True
            self.sdl2_quit()
            self.game_init()
    def set_mode(self, mode):
        self.MODE = mode
        self.SAVE = True

    def set_accel(self, accel):
        self.ACCEL = accel
        self.SAVE = True

    def set_color_mode(self, mode):
        self.COLOR_MODE = mode
        self.SAVE = True

    def set_exposure(self, exposure):
        self.EXPOSURE = max(1e-6, min(1e6, exposure))
        self.SAVE = True

    def set_time_rate(self, rate):
        self.TIME_RATE = max(1.0/1024.0, min(1.0e12, rate))
        self.SAVE = True

    # ---- presets: (name, {setting: value}); resources/config/presets.ini
    # can add more, one per line:  name | ACCEL=1e-11 | TIME_RATE=5e5 | GRAVITY=True
    DEFAULT_PRESETS = [
        ("Solar system continuous rocket", {"ACCEL": 1.0e-11, "TIME_RATE": 5.0e5}),
        ("Relativistic real time", {"ACCEL": 0.1, "TIME_RATE": 1.0}),
    ]

    # Systems the Settings menu can jump to: (label, body to orbit, preset name prefix)
    SYSTEMS = [("Solar system: Earth orbit", "earth", "Sun in 20 s"),
               ("Forge system: Cinder orbit", "cinder", "Cinder system"),
               ("Helios system: Ember orbit", "ember", "Helios system"),
               ("Abyss system: Ash orbit", "ash", "Abyss system")]

    def preset_index(self, prefix):
        for i, (name, vals) in enumerate(self.presets()):
            if name.startswith(prefix):
                return i
        return None

    def presets(self):
        path = os.path.join(CONFIG_DIR, "presets.ini")
        out = []
        try:
            for line in open(path):
                parts = [x.strip() for x in line.split("|")]
                if len(parts) < 2 or not parts[0] or parts[0].startswith("#"):
                    continue
                vals = {}
                for kv in parts[1:]:
                    if "=" not in kv:
                        continue
                    k, v = [x.strip() for x in kv.split("=", 1)]
                    k = k.upper()
                    if k in ("ACCEL", "TIME_RATE", "EXPOSURE", "IMPULSE_RAPIDITY"):
                        vals[k] = float(v)
                    elif k in ("GRID", "BRAKE", "GLARE", "CMB", "CROSSHAIR", "MARKERS", "GRAVITY", "ORBITS", "IMPULSE", "LENSING", "INERTIA"):
                        vals[k] = v.lower() in ("1", "true", "on", "yes")
                    elif k == "COLOR_MODE":
                        vals[k] = v.upper()
                    elif k == "START":
                        vals[k] = v          # body to stand on; honoured by the in-game keys
                out.append((parts[0], vals))
        except IOError:
            pass
        return out if out else list(self.DEFAULT_PRESETS)

    def apply_preset(self, i):
        """Apply preset i (0-based); returns its name or None."""
        plist = self.presets()
        if not 0 <= i < len(plist):
            return None
        name, vals = plist[i]
        for k, v in vals.items():
            if k == "START":
                continue
            if k == "ACCEL":
                self.set_accel(v)
            elif k == "TIME_RATE":
                self.set_time_rate(v)
            elif k == "EXPOSURE":
                self.set_exposure(v)
            elif k == "IMPULSE_RAPIDITY":
                self.set_impulse_rapidity(v)
            elif k == "COLOR_MODE":
                self.set_color_mode(v)
            else:
                self.set_flag(k, v)
        return name

    def set_spawn(self, name):
        """Remember where the ship starts: the body it was last sent to."""
        self.SPAWN = name.lower()
        self.SAVE = True

    def set_impulse_rapidity(self, eta):
        self.IMPULSE_RAPIDITY = max(1.0e-30, min(50.0, eta))   # no useful lower bound
        self.SAVE = True

    def set_flag(self, name, flag):
        """Set one of the boolean options GRID / BRAKE / GLARE / CMB / CROSSHAIR."""
        setattr(self, name, bool(flag))
        self.SAVE = True

    def set_hdr(self, flag, immediate=True):
        """Switch the float framebuffer / EDR on or off.  This needs a new
        window, which destroys every GL object, so during a game the change is
        recorded and applied by apply_hdr() once back at the title."""
        if flag != self.HDR:
            self.HDR = flag
            self.SAVE = True
            if immediate:
                self.apply_hdr()

    def apply_hdr(self):
        if self.window is not None and getattr(self, "hdr_active", None) != self.HDR:
            self.sdl2_quit()
            self.game_init()
            return True
        return False
        
    def save(self):
        if self.SAVE:
            try:
                f = open(os.path.join(CONFIG_DIR, "setting.ini"), "w")
                f.write("DISPLAY_SIZE_X=%i\n"%self.X)
                f.write("DISPLAY_SIZE_Y=%i\n"%self.Y)
                if self.FULL_SCREEN:
                    f.write("FULL_SCREEN=True\n")
                else:
                    f.write("FULL_SCREEN=False\n")
                f.write("MODE=%s\n"%self.MODE)
                if self.ACCEL is not None:
                    f.write("ACCEL=%r\n"%self.ACCEL)
                f.write("COLOR_MODE=%s\n"%self.COLOR_MODE)
                f.write("HDR=%s\n"%self.HDR)
                f.write("EXPOSURE=%r\n"%self.EXPOSURE)
                f.write("TIME_RATE=%r\n"%self.TIME_RATE)
                f.write("IMPULSE_RAPIDITY=%r\n"%self.IMPULSE_RAPIDITY)
                f.write("SPAWN=%s\n"%self.SPAWN)
                for name in ("GRID", "BRAKE", "GLARE", "CMB", "CROSSHAIR", "MARKERS", "GRAVITY", "ORBITS", "IMPULSE", "LENSING", "INERTIA"):
                    f.write("%s=%s\n"%(name, getattr(self, name)))
                f.close()
            except:
                pass

BOX = _Box()
