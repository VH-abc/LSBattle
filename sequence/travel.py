#coding: utf8
from math import sqrt, pi, log

from OpenGL import GL
import sdl2
import sdl2.ext

from entity import World
from program.const import *
from program.box import BOX
from program.utils import fill_screen, FramePerSec
from program.text import Sentence, drawSentence
from program import script
from sequence.locals import Keys
from entity.navmarkers import format_time
from sequence.stopmenu import StopMenu


class Travel(object):
    
    def __init__(self, level, scale, playerstate):
        BOX.resize(scale)
        self.stopmenu = StopMenu()
        self.level = level
        self.world = World(level, scale, playerstate)

    def init(self):
        self.start_message = Sentence(self.level.stage_name, BOX.Y/10)
        self.death_sentence = None
        self.keys = Keys()
        
        self.world.action(self.keys, 0.01)
        self.world.draw(self.keys)
        sdl2.SDL_GL_SwapWindow(BOX.window)

    def mainloop(self):
        self.init()
        textHeight = BOX.Y/32
        fps = FramePerSec()
        total_time = 0
        lose_time = 0
        last_tick = sdl2.SDL_GetTicks()
        shoot = False
        while True:
            
            ret = False
            for event in sdl2.ext.get_events():
                if event.type == sdl2.SDL_QUIT:
                    BOX.game_quit()
                elif event.type == sdl2.SDL_KEYDOWN:
                    key = event.key.keysym.sym
                    if key in KS_ESC:
                        sdl2.SDL_ShowCursor(1)
                        flg = self.stopmenu.mainloop(self.world)
                        sdl2.SDL_ShowCursor(0)
                        if flg == self.stopmenu.PLAY:
                            self.keys.load_config()
                            last_tick = sdl2.SDL_GetTicks()
                            break
                        elif flg == self.stopmenu.TITLE:
                            return
                    elif key == self.keys.accel_forward and self.level.enabled("accel_forward"):
                        self.keys.k_accel |= 1
                    elif key == self.keys.accel_back and self.level.enabled("accel_back"):
                        self.keys.k_accel |= 2
                    elif key == self.keys.accel_right and self.level.enabled("accel_right"):
                        self.keys.k_accel |= 4
                        self.keys.k_accel_priority = 0
                    elif key == self.keys.accel_left and self.level.enabled("accel_left"):
                        self.keys.k_accel |= 8
                        self.keys.k_accel_priority = 1
                    elif key == self.keys.turn_right and self.level.enabled("turn_right"):
                        self.keys.k_turn_right = True
                        self.keys.k_turn_priority1 = 0
                    elif key == self.keys.turn_left and self.level.enabled("turn_left"):
                        self.keys.k_turn_left  = True
                        self.keys.k_turn_priority1 = 1
                    elif key == self.keys.turn_up and self.level.enabled("turn_up"):
                        self.keys.k_turn_up = True
                        self.keys.k_turn_priority2 = 0
                    elif key == self.keys.turn_down and self.level.enabled("turn_down"):
                        self.keys.k_turn_down  = True
                        self.keys.k_turn_priority2 = 1
                    elif key == self.keys.shoot and self.level.enabled("shoot"):
                        shoot = True
                    elif key in KS_RETURN:
                        ret = True
                    elif key == self.keys.toggle_HUD and self.level.enabled("toggle_HUD"):
                        self.keys.k_map = (self.keys.k_map+1)%2
                    elif key == self.keys.change_gun and self.level.enabled("change_gun"):
                        self.world.player.state.gun_change()
                    elif key == self.keys.brake and self.level.enabled("brake") and BOX.BRAKE:
                        self.keys.k_brake = 1
                    elif key == self.keys.orbit and self.level.enabled("brake") and BOX.BRAKE:
                        self.keys.k_orbit = 1
                    elif key == self.keys.look_body:   self.keys.k_look_body = 1
                    elif key == self.keys.look_lead:   self.keys.k_look_lead = 1
                    elif key == self.keys.look_motion: self.keys.k_look_motion = 1
                    elif key == self.keys.look_aim:    self.keys.k_look_aim = 1
                    elif key == self.keys.booster and self.level.enabled("booster"):
                        if BOX.IMPULSE:
                            self.keys.k_impulse = 1
                        else:
                            self.keys.k_booster = 1
                    elif key == sdl2.SDLK_EQUALS:
                        BOX.set_exposure(BOX.EXPOSURE * 1.4142135623730951)
                    elif key == sdl2.SDLK_MINUS:
                        BOX.set_exposure(BOX.EXPOSURE / 1.4142135623730951)
                    elif key == sdl2.SDLK_PERIOD:
                        BOX.set_accel(self.world.player.acceleration * 1.5)
                        self.world.player.acceleration = BOX.ACCEL
                    elif key == sdl2.SDLK_COMMA:
                        BOX.set_accel(self.world.player.acceleration / 1.5)
                        self.world.player.acceleration = BOX.ACCEL
                    elif key == sdl2.SDLK_g:
                        BOX.set_flag("GRID", not BOX.GRID)
                    elif key == sdl2.SDLK_v:
                        BOX.toggle_visual()
                    elif key == sdl2.SDLK_m:
                        BOX.set_flag("SOUND", not BOX.SOUND)
                    elif key == sdl2.SDLK_t:
                        from sequence.teleport import Teleport, teleport_to
                        sdl2.SDL_ShowCursor(1)
                        entry = Teleport().mainloop()
                        sdl2.SDL_ShowCursor(0)
                        if entry is not None:
                            teleport_to(self.world, entry)
                        self.keys.load_config()
                        last_tick = sdl2.SDL_GetTicks()
                    elif key == sdl2.SDLK_l:
                        BOX.set_flag("GLARE", not BOX.GLARE)
                    elif key == sdl2.SDLK_b:
                        BOX.set_flag("CMB", not BOX.CMB)
                    elif key == sdl2.SDLK_h:
                        BOX.set_flag("CROSSHAIR", not BOX.CROSSHAIR)
                    elif key == sdl2.SDLK_n:
                        BOX.set_flag("MARKERS", not BOX.MARKERS)
                    elif key == sdl2.SDLK_j:
                        BOX.set_flag("GRAVITY", not BOX.GRAVITY)
                    elif key == sdl2.SDLK_o:
                        BOX.set_flag("ORBITS", not BOX.ORBITS)
                    elif sdl2.SDLK_1 <= key <= sdl2.SDLK_9:
                        i = key - sdl2.SDLK_1
                        if BOX.apply_preset(i) is not None:
                            self.world.player.acceleration = BOX.ACCEL
                            start = BOX.presets()[i][1].get("START")
                            if start:
                                self.world.teleport(start)
                    elif key == sdl2.SDLK_QUOTE:
                        BOX.set_impulse_rapidity(BOX.IMPULSE_RAPIDITY * 2.0)
                    elif key == sdl2.SDLK_SEMICOLON:
                        BOX.set_impulse_rapidity(BOX.IMPULSE_RAPIDITY * 0.5)
                    elif key == sdl2.SDLK_RIGHTBRACKET:
                        BOX.set_time_rate(BOX.TIME_RATE * 2.0)
                    elif key == sdl2.SDLK_LEFTBRACKET:
                        BOX.set_time_rate(BOX.TIME_RATE / 2.0)
                elif event.type == sdl2.SDL_KEYUP:
                    key = event.key.keysym.sym
                    if key == self.keys.accel_forward:
                        if self.keys.k_accel&1 == 1:
                            self.keys.k_accel -= 1
                    elif key == self.keys.accel_back:
                        if self.keys.k_accel&2 == 2:
                            self.keys.k_accel -= 2
                    elif key == self.keys.booster:
                        self.keys.k_booster = 0
                    elif key == self.keys.accel_right:
                        if self.keys.k_accel&4 == 4:
                            self.keys.k_accel -= 4
                    elif key == self.keys.accel_left:
                        if self.keys.k_accel&8 == 8:
                            self.keys.k_accel -= 8
                    elif key == self.keys.brake:
                        self.keys.k_brake = 0
                    elif key == self.keys.orbit:
                        self.keys.k_orbit = 0
                    elif key == self.keys.look_body:   self.keys.k_look_body = 0
                    elif key == self.keys.look_lead:   self.keys.k_look_lead = 0
                    elif key == self.keys.look_motion: self.keys.k_look_motion = 0
                    elif key == self.keys.look_aim:    self.keys.k_look_aim = 0
                    elif key == self.keys.turn_right:    self.keys.k_turn_right = False
                    elif key == self.keys.turn_left:     self.keys.k_turn_left  = False
                    elif key == self.keys.turn_up:       self.keys.k_turn_up    = False
                    elif key == self.keys.turn_down:     self.keys.k_turn_down  = False
                    elif key == self.keys.shoot:         shoot = False

            GL.glClear(GL.GL_DEPTH_BUFFER_BIT|GL.GL_COLOR_BUFFER_BIT)
            
            tick = sdl2.SDL_GetTicks()
            dt = tick - last_tick
            if dt == 0:
                dt = 1
            ds = dt * 0.001 * BOX.time_rate()
            last_tick = tick
            fps.add(ds)
            total_time += ds
            if shoot:
                self.keys.k_bullet += dt
            else:
                if self.keys.k_bullet < 0:
                    self.keys.k_bullet += dt
                    if self.keys.k_bullet > 0:
                        self.keys.k_bullet = 0
                else:
                    self.keys.k_bullet = 0

            self.keys.k_fine = bool(sdl2.SDL_GetModState() & (sdl2.KMOD_CAPS | sdl2.KMOD_SHIFT))
            self.world.action(self.keys, ds)
            self.world.draw(self.keys)

            if self.world.player.dead:
                if lose_time == 0:
                    lose_time = total_time
                    self.death_sentence = Sentence(self.world.player.death_message or "Lost", BOX.Y/14)
                a = max(0.0, 2.0 - (total_time - lose_time) * 0.5)
                fill_screen(0.0, 0.0, 0.0, 1.0 - a/2.0)
                GL.glColor(1.0, 0.2, 0.2, 1.0)
                self.death_sentence.draw_center()
                if total_time - lose_time > 5.0:
                    return
            if total_time < 2.0:
                GL.glColor(1.0, 0.0, 0.0, 1.0-total_time*0.5)
                self.start_message.draw_center()

            if self.keys.k_map == 1:
                g = self.world.player.P.U.get_gamma()
                u = sqrt(max(0.0, 1.0 - 1.0/g**2)) if g == g and g >= 1.0 else 0.0
                v = c * u
                text = "Stage: %i\n"%self.level.stage
                text += "FPS: %i\n"%(fps.get())
                text += "Speed: {:,d}m/s\n".format(int(v))
                text += "       %.3fc\n"%(u)
                text += "Lorentz factor: %.1f\n"%(g)
                text += "Thomas rotation: %.1f deg\n"%(self.world.player.wigner_angle*180.0/pi)
                if BOX.ADAPTIVE:
                    text += "Accel: %.3g c/s adaptive  (, .)\n"%(BOX.thrust(self.world.player.acceleration))
                else:
                    text += "Accel: %.3g c/s  (, .)\n"%(self.world.player.acceleration)
                if BOX.IMPULSE:
                    text += "Booster impulse: %.3g rapidity%s  (; ')%s\n"%(BOX.impulse(), " adaptive" if BOX.ADAPTIVE else "", "; launches from a surface" if self.world.player.landed is not None else "")
                fo = getattr(self.world.solar, "obs_shift", 1.0)
                if abs(fo - 1.0) > 1e-4:
                    text += "Gravitational blueshift: x%.4g\n" % fo
                rl = getattr(self.world, "rate_limit", 1.0)
                tr = BOX.time_rate()
                if rl < 0.999:
                    text += "Time rate: x%.3g of x%.3g, limited near a body  ([ ])\n"%(tr*rl, tr)
                elif BOX.ADAPTIVE:
                    text += "Time rate: x%.3g adaptive  ([ ])\n"%(tr)
                else:
                    text += "Time rate: x%g  ([ ])\n"%(BOX.TIME_RATE)
                if self.world.player.plunged is not None:
                    text += "At the singularity of %s: the end\n" % self.world.player.plunged.name
                elif self.world.player.inside is not None:
                    text += "Inside the event horizon of %s\n" % self.world.player.inside.name
                elif self.world.player.landed is not None:
                    text += ("Inside the event horizon of %s\n" if self.world.player.landed.is_hole else "Landed on %s\n") % self.world.player.landed.name
                if BOX.COLOR_MODE == "ACCURATE":
                    text += "Visuals: %s  (v)\n" % ("realistic" if BOX.VISUAL == "REALISTIC" else "comfortable")
                    text += "Exposure: %+.1f EV  (- =)\n"%(log(BOX.EXPOSURE, 2.0))
                    if BOX.AUTOEXP and BOX.auto_ev < -0.05:
                        text += "  auto-dimmed by %.1f EV\n" % (-BOX.auto_ev)
                    if BOX.edr_enabled:
                        text += "EDR headroom: x%.1f\n"%(BOX.edr_headroom)
                GL.glColor(0.3, 0.6, 0.3, 1.0)
                drawSentence(text, textHeight, BOX.X*0.01, BOX.Y)
                nlines = text.count("\n") + 1

                text  = "Proper Time: %s\n"%(format_time(self.world.player.time))
                text += " World Time: %s\n"%(format_time(self.world.player.P.X.t))
                GL.glColor(1.0, 1.0, 1.0, 0.5)
                drawSentence(text, textHeight, BOX.X*0.01, BOX.Y-(nlines+0.5)*textHeight)

            sdl2.SDL_GL_SwapWindow(BOX.window)
