#coding: utf8
from OpenGL import GL
import sdl2
import sdl2.ext

from program.const import *
from program.box import BOX
from program.utils import fill_screen, Snapshot
from sequence.locals import MenuItems
from sequence.option import Option


class StopMenu(object):
    PLAY   = 0
    CONFIG = 1
    TITLE  = 2
    QUIT   = 3
    def __init__(self):
        self.option = Option(in_game=True)
        self.texts = ["Resume",
                      "Options",
                      "Title",
                      "Quit"]
        self.back_image = Snapshot()

    def init(self):
        self.menu = MenuItems(self.texts, BOX.Y/12)

    def mainloop(self, world=None):
        for g in self.option.goto:
            g.world = world
        fill_screen(0.0, 0.0, 0.0, 0.4)
        self.back_image.captcha()
        self.init()
        while True:
            choice = -1
            for event in sdl2.ext.get_events():
                if event.type == sdl2.SDL_QUIT:
                    self.back_image.delete()
                    BOX.game_quit()
                elif event.type == sdl2.SDL_KEYDOWN:
                    key = event.key.keysym.sym
                    if key in KS_ESC:
                        if self.menu.choice == self.QUIT:
                            self.back_image.delete()
                            BOX.game_quit()
                        else:
                            self.menu.choice = self.QUIT
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
                if choice == self.CONFIG:
                    if self.option.mainloop() == "resume":
                        self.back_image.delete()
                        return self.PLAY
                    self.menu.choice = 0
                elif choice == self.QUIT:
                    self.back_image.delete()
                    BOX.game_quit()
                else:
                    self.back_image.delete()
                    return choice

            GL.glClear(GL.GL_DEPTH_BUFFER_BIT|GL.GL_COLOR_BUFFER_BIT)
            self.back_image.draw()
            self.menu.draw()
            sdl2.SDL_GL_SwapWindow(BOX.window)
            sdl2.SDL_Delay(10)
