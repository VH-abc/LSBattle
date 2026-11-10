#coding: utf8
#
# keys.py
#
import sdl2

from program.const import CONFIG_DIR


class Keys(object):
    config_fname = CONFIG_DIR+"keyconfig.ini"
    keymap_fname = CONFIG_DIR+"keymap.dat"
    def __init__(self):
        self.reset()
        self.default()
        self.load_config()

    def reset(self):
        self.k_accel = 0
        self.k_accel_priority = 0
        self.k_booster = 0
        self.k_impulse = 0         # one-shot: a booster tap in impulse mode
        self.k_turn_right = False
        self.k_turn_left = False
        self.k_turn_up = False
        self.k_turn_down = False
        self.k_turn_priority1 = 0
        self.k_turn_priority2 = 0
        self.k_fine = False        # fine turning: Caps Lock on or Shift held
        self.k_bullet = 0
        self.k_look_behind = 0
        self.k_map = 1
        self.k_brake = 0
        self.k_orbit = 0           # brake towards the circular orbit
        self.k_look_body = 0       # held: look where the nearest body appears
        self.k_look_lead = 0       # held: look where to aim at it
        self.k_look_motion = 0     # held: look along own motion
        self.k_look_aim = 0        # held: aim-ahead at the body under the crosshair

    def default(self):
        self.accel_forward = sdl2.SDLK_w
        self.accel_back    = sdl2.SDLK_s
        self.accel_right   = sdl2.SDLK_d
        self.accel_left    = sdl2.SDLK_a
        self.booster       = sdl2.SDLK_f
        self.brake         = sdl2.SDLK_r
        self.orbit         = sdl2.SDLK_e
        self.look_body     = sdl2.SDLK_t
        self.look_lead     = sdl2.SDLK_y
        self.look_motion   = sdl2.SDLK_u
        self.look_aim      = sdl2.SDLK_i
        self.turn_right    = sdl2.SDLK_RIGHT
        self.turn_left     = sdl2.SDLK_LEFT
        self.turn_up       = sdl2.SDLK_UP
        self.turn_down     = sdl2.SDLK_DOWN
        self.shoot         = sdl2.SDLK_SPACE
        self.change_gun    = sdl2.SDLK_v
        self.toggle_HUD    = sdl2.SDLK_TAB
        self.names = ["accel_forward", 
                      "accel_back",
                      "accel_right",
                      "accel_left",
                      "turn_right",
                      "turn_left",
                      "turn_up",
                      "turn_down",
                      "shoot",
                      "booster",
                      "brake",
                      "orbit",
                      "look_body",
                      "look_lead",
                      "look_motion",
                      "look_aim",
                      "change_gun",
                      "toggle_HUD"]
        # self.name2 = ["Accel Forward ",
        #               "Accel Backward",
        #               "Accel Right   ",
        #               "Accel Left    ",
        #               "Booster       ",
        #               "brake         ",
        #               "Turn Right    ",
        #               "Turn Left     ",
        #               "Turn Up       ",
        #               "Turn Down     ",
        #               "Shoot         ",
        #               "Change Gun    ",
        #               "HUD ON/OFF    "]
        self.name2 = ["Accel Forward ",
                      "      Backward",
                      "      Right   ",
                      "      Left    ",
                      "Turn Right    ",
                      "     Left     ",
                      "     Up       ",
                      "     Down     ",
                      "Shoot         ",
                      "Booster       ",
                      "brake         ",
                      "Orbit brake   ",
                      "Look at body  ",
                      "Look to aim   ",
                      "Look along v  ",
                      "Aim crosshair ",
                      "Change Gun    ",
                      "HUD ON/OFF    "]

    def __iter__(self):
        for name in self.names:
            yield name, getattr(self, name)

    def load_config(self):
        try:
            f = open(self.config_fname)
            for line in f:
                name, value = line.split("=")
                if hasattr(self, name):
                    setattr(self, name, int(value))
        except IOError:
            self.save()
            return

    def load_map(self):
        self.key_map = {}
        try:
            f = open(self.keymap_fname)
            for line in f:
                name, value = line.split("=")
                self.key_map[int(value)] = name
        except IOError:
            for key in dir(sdl2):
                if key.startswith("SDLK_"):
                    self.key_map[getattr(sdl2, key)] = key[5:].upper()

    def save(self):
        f = open(self.config_fname, "w")
        for name in self.names:
            f.write("%s=%i\n"%(name, getattr(self, name)))
        f.close()


