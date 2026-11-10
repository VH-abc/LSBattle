#coding: utf8
# gamelevel.py
from program import script

EASY   = 0
NORMAL = 1
HARD   = 2
TRAVEL = 3
PLANAR = 4         # "2D Battle": everything moves in the y = 0 plane
PLANAR_TRAVEL = 5  # "2D Travel": planar, no enemies

MODE_NAMES = {EASY: "EASY", NORMAL: "NORMAL", HARD: "HARD", TRAVEL: "TRAVEL",
              PLANAR: "PLANAR", PLANAR_TRAVEL: "PLANAR_TRAVEL"}
MODE_IDS = dict((v, k) for k, v in MODE_NAMES.items())

class GameLevel(object):
    
    def __init__(self, stage, mode):
        self.stage = stage
        self.mode = mode
        self.L = script.game.stage._colosseum_size(stage)
        self.stage_name = "STAGE %i"%stage if stage < script.game.stage_num else "LAST STAGE"
        self.enemy_num = script.game.stage._enemy_num(stage)
        self.types = script.game.stage._types_num(stage)

        T, F = True, False
        self.table = {
            #               EASY NORMAL HARD TRAVEL PLANAR PLANAR_TRAVEL
            "accel_back":  [F,   T,     T,   T,     T,     T],
            "booster":     [F,   F,     T,   T,     T,     T],
            "accel_right": [F,   T,     T,   T,     T,     T],
            "accel_left":  [F,   T,     T,   T,     T,     T],
            "change_gun":  [F,   T,     T,   F,     T,     F],
            "brake":       [F,   F,     T,   T,     T,     T],
            "toggle_HUD":  [F,   T,     T,   T,     T,     T],
            "turn_up":     [T,   T,     T,   T,     F,     F],
            "turn_down":   [T,   T,     T,   T,     F,     F],
        }
        
    def enabled(self, key):
        if key in self.table:
            return self.table[key][self.mode]
        else:
            return True

    def is_easy(self):
        return EASY == self.mode
    def is_normal(self):
        return self.mode in (NORMAL, PLANAR)
    def is_hard(self):
        return HARD == self.mode
    def is_travel(self):
        """No enemies."""
        return self.mode in (TRAVEL, PLANAR_TRAVEL)
    def is_planar(self):
        """Motion confined to the y = 0 plane."""
        return self.mode in (PLANAR, PLANAR_TRAVEL)
