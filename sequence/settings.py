#coding: utf8
# Physics / display settings submenu
from OpenGL import GL
import sdl2
import sdl2.ext

from program.const import *
from program.box import BOX
from program.text import MyFont
from program import script
from sequence.locals import MenuItems, backimage

ACCELS = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


class Settings(object):
    PRESET, ACCEL, COLOR, HDR, GRID, BRAKE, GLARE, VISUAL, GLARE_SPREAD, AUTOEXP, CMB, CROSSHAIR, MARKERS, GRAVITY, ORBITS, ADAPTIVE, BOOSTER, LENSING, INERTIA, SOUND, GOTO = range(21)

    def __init__(self, in_game=False):
        self.in_game = in_game
        self.preset_i = 0
        # the Travel-to list starts on the current spawn body
        self.goto_i = 0
        for i, (label, body, prefix) in enumerate(BOX.SYSTEMS):
            if body.lower() == BOX.SPAWN:
                self.goto_i = i
        self.world = None            # set by the pause menu: the running game
        self.resume = False

    def accel(self):
        return BOX.ACCEL if BOX.ACCEL is not None else script.player.acceleration

    def texts(self):
        plist = BOX.presets()
        pname = plist[self.preset_i % len(plist)][0] if plist else "-"
        return ["Preset: %s  (<- -> choose, Enter apply; in game: keys 1-9)" % pname,
                "Acceleration: %.3g c/s  (<- ->)" % self.accel(),
                "Color: %s" % ("Accurate blackbody" if BOX.COLOR_MODE == "ACCURATE" else "Game (original)"),
                "HDR / EDR output: %s%s" % ("On" if BOX.HDR else "Off",
                                           "  (applies back at the title)" if BOX.HDR != getattr(BOX, "hdr_active", BOX.HDR)
                                           else (("  [%s]" % BOX.edr_note) if BOX.edr_note else "")),
                "Background grid (with HUD): %s" % ("On" if BOX.GRID else "Off"),
                "Brake key available: %s" % ("Yes" if BOX.BRAKE else "No"),
                "Glare effect: %s" % ("On" if BOX.GLARE else "Off"),
                "Visuals: %s  (in game: v)" % ("Realistic, the eye model throughout" if BOX.VISUAL == "REALISTIC" else "Comfortable: glare limited in spread and veil"),
                "Glare spread limit: %s" % ("unlimited (the eye's 1/angle^2 wing)" if BOX.GLARE_SPREAD <= 0.0 else "%g degrees" % BOX.GLARE_SPREAD),
                "Automatic exposure (dims a very bright view): %s" % ("On" if BOX.AUTOEXP else "Off"),
                "Cosmic microwave background: %s" % ("On" if BOX.CMB else "Off"),
                "Crosshair: %s" % ("On" if BOX.CROSSHAIR else "Off"),
                "Navigation markers: %s" % ("On" if BOX.MARKERS else "Off"),
                "Gravity: %s" % ("On" if BOX.GRAVITY else "Off"),
                "Orbiting bodies: %s" % ("On" if BOX.ORBITS else "Off"),
                "Adaptive time rate and thrust (with the local orbit): %s" % ("On" if BOX.ADAPTIVE else "Off"),
                "Booster key: %s" % (("tap = rapidity impulse %.3g  (<- ->, in game ; ')" % BOX.IMPULSE_RAPIDITY)
                                     if BOX.IMPULSE else "hold = continuous turbo"),
                "Gravitational lensing near compact bodies: %s" % ("On" if BOX.LENSING else "Off"),
                "Turning inertia (arrow keys): %s" % ("On: accelerate, coast, damp" if BOX.INERTIA else "Off: fixed rate, instant stop"),
                "Sound: %s  (in game: m; volume with <- ->)" % (("On, volume %d%%" % round(100*BOX.VOLUME)) if BOX.SOUND else "Off"),
                "Travel to: %s  (<- -> choose; Enter: go there, apply its preset, start in orbit)%s" % (
                    BOX.SYSTEMS[self.goto_i % len(BOX.SYSTEMS)][0], "" if self.world is not None else "  [in game only]")]

    def init(self, choice=0):
        texts = self.texts()
        # row height: fit every entry plus the title and the return item on screen
        self.menu = MenuItems(texts, min(BOX.Y/16, BOX.Y/(len(texts) + 4)), ret=True, title="Settings")
        self.menu.choice = choice

    def step_impulse(self, direction):
        # halve / double: the impulse is unbounded below (down to float limits)
        BOX.set_impulse_rapidity(BOX.IMPULSE_RAPIDITY * (2.0 if direction > 0 else 0.5))

    def step_accel(self, direction):
        a = self.accel()
        if direction > 0:
            cands = [x for x in ACCELS if x > a*1.001]
            a = cands[0] if cands else a
        else:
            cands = [x for x in ACCELS if x < a*0.999]
            a = cands[-1] if cands else a
        BOX.set_accel(a)

    def activate(self, choice):
        if choice == self.PRESET:
            BOX.apply_preset(self.preset_i)
        elif choice == self.ACCEL:
            self.step_accel(+1)
        elif choice == self.COLOR:
            BOX.set_color_mode("GAME" if BOX.COLOR_MODE == "ACCURATE" else "ACCURATE")
        elif choice == self.HDR:
            BOX.set_hdr(not BOX.HDR, immediate=not self.in_game)
            if not self.in_game:
                backimage.load()
                MyFont.init_font(script.ui.font.name)
        elif choice == self.GRID:
            BOX.set_flag("GRID", not BOX.GRID)
        elif choice == self.BRAKE:
            BOX.set_flag("BRAKE", not BOX.BRAKE)
        elif choice == self.GLARE:
            BOX.set_flag("GLARE", not BOX.GLARE)
        elif choice == self.VISUAL:
            BOX.toggle_visual()
        elif choice == self.GLARE_SPREAD:
            BOX.cycle_glare_spread()
        elif choice == self.AUTOEXP:
            BOX.set_flag("AUTOEXP", not BOX.AUTOEXP)
        elif choice == self.CMB:
            BOX.set_flag("CMB", not BOX.CMB)
        elif choice == self.CROSSHAIR:
            BOX.set_flag("CROSSHAIR", not BOX.CROSSHAIR)
        elif choice == self.MARKERS:
            BOX.set_flag("MARKERS", not BOX.MARKERS)
        elif choice == self.GRAVITY:
            BOX.set_flag("GRAVITY", not BOX.GRAVITY)
        elif choice == self.ORBITS:
            BOX.set_flag("ORBITS", not BOX.ORBITS)
        elif choice == self.ADAPTIVE:
            BOX.set_flag("ADAPTIVE", not BOX.ADAPTIVE)
        elif choice == self.BOOSTER:
            BOX.set_flag("IMPULSE", not BOX.IMPULSE)
        elif choice == self.LENSING:
            BOX.set_flag("LENSING", not BOX.LENSING)
        elif choice == self.INERTIA:
            BOX.set_flag("INERTIA", not BOX.INERTIA)
        elif choice == self.SOUND:
            BOX.set_flag("SOUND", not BOX.SOUND)
        elif choice == self.GOTO and self.world is not None:
            label, body, prefix = BOX.SYSTEMS[self.goto_i % len(BOX.SYSTEMS)]
            i = BOX.preset_index(prefix)
            if i is not None:
                BOX.apply_preset(i)
                self.world.player.acceleration = BOX.ACCEL
            self.world.start_in_orbit(body, self.world.spawn_factor(body))
            BOX.set_spawn(body)
            self.resume = True

    def mainloop(self):
        self.resume = False
        self.init()
        while True:
            choice = -1
            for event in sdl2.ext.get_events():
                if event.type == sdl2.SDL_QUIT:
                    BOX.game_quit()
                elif event.type == sdl2.SDL_KEYDOWN:
                    key = event.key.keysym.sym
                    if key in KS_ESC:
                        return
                    elif key == sdl2.SDLK_UP:
                        self.menu.up()
                    elif key == sdl2.SDLK_DOWN:
                        self.menu.down()
                    elif key == sdl2.SDLK_LEFT and self.menu.choice == self.PRESET:
                        self.preset_i = (self.preset_i - 1) % max(1, len(BOX.presets()))
                        self.init(self.menu.choice)
                    elif key == sdl2.SDLK_RIGHT and self.menu.choice == self.PRESET:
                        self.preset_i = (self.preset_i + 1) % max(1, len(BOX.presets()))
                        self.init(self.menu.choice)
                    elif key == sdl2.SDLK_LEFT and self.menu.choice == self.ACCEL:
                        self.step_accel(-1)
                        self.init(self.menu.choice)
                    elif key == sdl2.SDLK_RIGHT and self.menu.choice == self.ACCEL:
                        self.step_accel(+1)
                        self.init(self.menu.choice)
                    elif key == sdl2.SDLK_LEFT and self.menu.choice == self.GOTO:
                        self.goto_i = (self.goto_i - 1) % len(BOX.SYSTEMS)
                        self.init(self.menu.choice)
                    elif key == sdl2.SDLK_RIGHT and self.menu.choice == self.GOTO:
                        self.goto_i = (self.goto_i + 1) % len(BOX.SYSTEMS)
                        self.init(self.menu.choice)
                    elif key == sdl2.SDLK_LEFT and self.menu.choice == self.BOOSTER:
                        self.step_impulse(-1)
                        self.init(self.menu.choice)
                    elif key == sdl2.SDLK_RIGHT and self.menu.choice == self.BOOSTER:
                        self.step_impulse(+1)
                        self.init(self.menu.choice)
                    elif key in KS_RETURN:
                        choice = self.menu.choice
                        break
                elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                    mouse = event.button
                    if mouse.button == 1:
                        choice = self.menu.mouse_check(mouse.x, mouse.y)
                        break

            if 0 <= choice:
                if choice == self.menu.RETURN:
                    return
                self.activate(choice)
                if self.resume:
                    self.resume = False
                    return "resume"
                self.init(choice)

            GL.glClear(GL.GL_DEPTH_BUFFER_BIT|GL.GL_COLOR_BUFFER_BIT)
            self.menu.draw()
            sdl2.SDL_GL_SwapWindow(BOX.window)
            sdl2.SDL_Delay(10)
