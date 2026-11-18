# coding: utf8
# sequence/teleport.py
"""
The Travel-to menu: the list of systems (BOX.SYSTEMS), reached with the t
key from the title screen or during play.  Choosing one applies the
system's preset, puts the ship in orbit there and makes it the spawn.
"""
from OpenGL import GL
import sdl2
import sdl2.ext

from program.const import *
from program.box import BOX
from sequence.locals import MenuItems


def teleport_to(world, entry):
    """Apply a BOX.SYSTEMS entry (label, body, preset prefix) to a running
    world: its preset, an orbit about the body, the spawn."""
    label, body, prefix = entry
    i = BOX.preset_index(prefix)
    if i is not None:
        BOX.apply_preset(i)
        world.player.acceleration = BOX.ACCEL
    world.start_in_orbit(body, world.spawn_factor(body))
    BOX.set_spawn(body)


class Teleport(object):

    def __init__(self):
        self.menu = None

    def init(self, choice=0):
        texts = [label for label, body, prefix in BOX.SYSTEMS]
        h = min(BOX.Y/14, 0.78*BOX.Y/(1.3*max(1, len(texts))))
        self.menu = MenuItems(texts, h, ret=True, title="Travel to", ofy=-0.045*BOX.Y)
        self.menu.choice = choice

    def mainloop(self):
        """Returns the chosen BOX.SYSTEMS entry, or None."""
        start = 0
        for i, (label, body, prefix) in enumerate(BOX.SYSTEMS):
            if body.lower() == BOX.SPAWN:
                start = i
        self.init(start)
        while True:
            choice = -1
            for event in sdl2.ext.get_events():
                if event.type == sdl2.SDL_QUIT:
                    BOX.game_quit()
                elif event.type == sdl2.SDL_KEYDOWN:
                    key = event.key.keysym.sym
                    if key in KS_ESC or key == sdl2.SDLK_t:
                        return None
                    elif key == sdl2.SDLK_UP:
                        self.menu.up()
                    elif key == sdl2.SDLK_DOWN:
                        self.menu.down()
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
                    return None
                return BOX.SYSTEMS[choice]
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
            self.menu.draw()
            sdl2.SDL_GL_SwapWindow(BOX.window)
            sdl2.SDL_Delay(10)
