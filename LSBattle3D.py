# coding: utf8
# LSBattle.py
try:
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    # OpenGL.STORE_POINTERS = False
    # OpenGL.UNSIGNED_BYTE_IMAGES_AS_STRING = False
    import program.const
except:
    import sys, traceback, os
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if os.name == "posix":
        p = os.path.dirname(sys.argv[0]) # app/Contants/MacOS
        p = os.path.dirname(p) # app/Contants/
        p = os.path.dirname(p) # app/
        p = os.path.dirname(p)
    else:
        p = os.path.dirname(sys.argv[0])
    p = os.path.join(p, "error.log")
    traceback.print_exception(exc_type, exc_value, exc_traceback,
                              file=open(p, "w")
                              )
    sys.exit(1)

def run():
    # hang watchdog: every 20 s the Python stack of each thread goes to
    # hang.log, so a frozen game leaves a record of where it was stuck
    try:
        import faulthandler, os
        _log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hang.log"), "w")
        faulthandler.dump_traceback_later(20, repeat=True, file=_log)
    except Exception:
        pass
    from sequence.game import Game
    game = Game()
    game.mainloop()

if __name__ == "__main__":
    try:
        run()
    except Exception:
        import sys, traceback, os
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if os.name == "posix":
            p = os.path.dirname(sys.argv[0]) # app/Contants/MacOS
            p = os.path.dirname(p) # app/Contants/
            p = os.path.dirname(p) # app/
            p = os.path.dirname(p)
        else:
            p = os.path.dirname(sys.argv[0])
        p = os.path.join(p, "error.log")
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  file=open(p, "w")
                                  )
        from program.box import BOX
        BOX.game_quit()
