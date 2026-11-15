# coding: utf8
# utils/shader.py
from OpenGL.GL import *


class ProgramInfo(object):
    def __init__(self, program_id, vertex_source, fragment_source, key):
        self.program_id = program_id
        self.vertex_source = vertex_source
        self.fragment_source = fragment_source
        self.key = key

def gl_log(text):
    """Append a line to gl.log next to the game (GL diagnostics)."""
    try:
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "gl.log")
        with open(path, "a") as f:
            f.write(text.rstrip() + "\n")
    except Exception:
        pass

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    info = glGetShaderInfoLog(shader)
    if info:
        head = source.strip().split("\n")[1:3]
        gl_log("%s shader %s: %s" % ("vertex" if shader_type == GL_VERTEX_SHADER else "fragment",
                                      "FAILED" if not status else "warnings", info if isinstance(info, str) else info.decode("utf8", "replace")))
    if not status:
        s = info
        glDeleteShader(shader)
        raise ValueError("Shader compilation failed: %s"%s)
    return shader

_compiled_program = {}
_using_program_id = {}
def compile_program(vertex_source, fragment_source, cache=True):
    key = vertex_source + fragment_source
    if cache:
        program_info = _compiled_program.get(key)
        if program_info is not None and glIsProgram(program_info.program_id):
            return program_info.program_id

    vertex_shader   = compile_shader(vertex_source,   GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    program_id = glCreateProgram()
    glAttachShader(program_id, vertex_shader)
    glAttachShader(program_id, fragment_shader)
    glLinkProgram(program_id)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    status = glGetProgramiv(program_id, GL_LINK_STATUS)
    if not status:
        s = glGetProgramInfoLog(program_id)
        raise ValueError("Program Link failed: %s"%s)

    program_info = ProgramInfo(program_id, vertex_source, fragment_source, key)
    if program_id in _using_program_id:
        deleted_key = _using_program_id[program_id].key
        del _compiled_program[deleted_key]
    _compiled_program[key] = program_info
    _using_program_id[program_id] = program_info
    
    return program_id
