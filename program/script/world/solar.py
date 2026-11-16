#coding: utf8
from ..common import Block, high_func_num, func_str, color_func

_default = [
    ["Earth", "earth.jpg", "star",   6378000, 149597870700,  0.0, 23.4, "Sun", 1000],
    ["Sun",   "sun.gif",   "star", 695500000,            0,  0.0,  0.0,  None, 1000],
    ["Moon",  "moon.jpg",  "star",   1738000,    384400000, 180.,  0.0, "Earth", 1000],
]

class star(Block):
    def __init__(self):
        self.name = "earth"
        self.texture = "earth.jpg"
        self.model = "star"
        self.sphere_radius = 6378000
        self.orbital_radius = 6378000
        self.orbital_phi = 0.0
        self.orbital_theta = 0.0
        self.temperature = 5778.0
        self.brightness = 1.0
        self.flattening = 0.0
        self.pole_x = 0.0
        self.pole_y = 0.0
        self.pole_z = 0.0
        self.solid = 1
        self.ring = ""
        self.lit_by = ""
        self.mass = 0.0
        self.disk_inner = 3.0
        self.disk_outer = 0.0
        self.disk_temperature = 10000.0
        self.disk_brightness = 100.0
        self.tilt = 23.4
        self.spin = 0.0           # rotation about the pole, degrees: which longitude faces +x
        self.rotation_period = 0.0  # sidereal rotation period, seconds (negative: retrograde; 0: none)
        self.hp = 1000
        self.primary_star = "sun"

        self._name_func = func_str
        self._texture_func = func_str
        self._model_func = func_str
        self._sphere_radius_func = high_func_num(float, 0.0, 10.**50)
        self._orbital_radius_func = high_func_num(float, 0.0, 10.**50)
        self._orbital_phi_func = high_func_num(float, -180, 180)
        self._orbital_theta_func = high_func_num(float, -90, 90)
        self._temperature_func = high_func_num(float, 100.0, 1.0e9)
        self._brightness_func = high_func_num(float, 0.0, 1.0e6)
        self._flattening_func = high_func_num(float, 0.0, 0.95)
        self._pole_x_func = float
        self._pole_y_func = float
        self._pole_z_func = float
        self._solid_func = high_func_num(int, 0, 1)
        self._ring_func = func_str
        self._lit_by_func = func_str
        self._mass_func = high_func_num(float, 0.0, 1.0e60)
        self._disk_inner_func = high_func_num(float, 1.0, 1000.0)
        self._disk_outer_func = high_func_num(float, 0.0, 1000.0)
        self._disk_temperature_func = high_func_num(float, 100.0, 1.0e9)
        self._disk_brightness_func = high_func_num(float, 0.0, 1.0e6)
        self._tilt_func = high_func_num(float, -180, 180)
        self._spin_func = high_func_num(float, -360, 360)
        self._rotation_period_func = high_func_num(float, -1.0e12, 1.0e12)
        self._hp_func = high_func_num(int, 1, 10**10)
        self._primary_star_func = func_str

class flame(Block):
    def __init__(self):
        self.life = 50.0
        self.speed = 0.1
        self.size = 0.2
        self.color = [1.0, 0.8, 0.8, 0.8]
        self.num = 10
        self._life_func  = high_func_num(float, 0.01, 100.0)
        self._speed_func = high_func_num(float, 0.1, 0.999)
        self._size_func  = high_func_num(float, 0.001, 1.0)
        self._color_func = color_func
        self._num_func = high_func_num(int, 4, 20)

class solar(Block):
    def __init__(self):
        self.stars = []
        self._stars_obj = star
        self.flame = flame()
        self.center = "earth"
        self.dx =  0.0
        self.dy = -0.5
        self.dz = -3.0
        self._center_func = func_str
        self._dx_func = float
        self._dy_func = float
        self._dz_func = float

    def _check(self):
        if not self.stars:
            for data in _default:
                s = star()
                s.name = data[0]
                s.texture = data[1]
                s.model = data[2]
                s.sphere_radius = data[3]
                s.orbital_radius = data[4]
                s.orbital_phi = data[5]
                s.tilt = data[6]
                s.primary_star = data[7]
                s.hp = data[8]
                self.stars.append(s)
        super(solar, self)._check()
