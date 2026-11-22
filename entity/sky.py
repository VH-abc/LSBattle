# coding: utf8
# entity/sky.py
from math import pi, tan

from OpenGL.GL import *

from go import Matrix44
from model.polygon import MqoGpoPolygon
from program.box import BOX
from program.utils import load_texture, compile_program, search_imagefile
from program.const import IMG_DIR, VIEW_ANGLE
from program import script
from program.utils import blackbody, boost
from program.utils.lensing import Lensing, GLSL as LENS_GLSL
from program.utils import kerr as kerrlib


GAME_COLOUR = """
            color *= 255.0;
            float l = max(max(color.r, color.g), color.b);
            if (l > 1) {
                color *= 255.0 / l;
                float Tr, Tg, Tb;
                Tb = 1000.0 + 904.495*exp(0.00721929 * color.b);
                Tr = 6000.0 + 8.01879e20 * pow(max(color.r, 1.0), -7.507239275877164);
                if (color.b > 254.0) {
                    Tg = 6502.86;
                } else {
                    if (color.b / max(color.r, 1.0) > 0.98084) {
                        Tg = 505.192 * exp(0.0100532 * color.g);
                    } else {
                        Tg = 6000.0 + 3.55446e34 * pow(max(color.g, 1), -13.24242861627803);
                    }
                }
                Tr *= ratio;
                Tg *= ratio;
                Tb *= ratio;
                if (1905 > Tb) {
                    color.b = 0.0;
                } else if (6700 < Tb) {
                    color.b = 255.0;
                } else {
                    color.b = -305.045 + 138.518 * log(0.01 * Tb - 10.0);
                }
                if (6689 > Tr) {
                    color.r = 255.0;
                } else {
                    color.r = 608.873 * pow(Tr - 6000.0, -0.133205);
                }
                if (506 > Tg) {
                    color.g = 0.0;
                } else if (6502.86 > Tg) {
                    color.g = -619.2 + 99.4708 * log(Tg);
                } else {
                    color.g = 406.534 * pow(Tg - 6000.0, -0.0755148);
                }
                color *= l/255;
            }
            gl_FragColor = color/255;
"""

SKY_RADIUS = 60000.0 * 31557600.0   # 60 kly in light-seconds
T_CMB = 2.725                       # K

# approximate blackbody colour (Tanner Helland fit) for the game colour mode
CMB_GAME = """
vec3 cmb_game(float T) {
    float t = clamp(T, 1000.0, 40000.0) * 0.01;
    float r = t <= 66.0 ? 1.0 : clamp(1.2929 * pow(t - 60.0, -0.1332), 0.0, 1.0);
    float g = t <= 66.0 ? clamp(0.3900 * log(t) - 0.6318, 0.0, 1.0)
                        : clamp(1.1299 * pow(t - 60.0, -0.0755), 0.0, 1.0);
    float b = t >= 66.0 ? 1.0 : (t <= 19.0 ? 0.0 : clamp(0.5432 * log(t - 10.0) - 1.1962, 0.0, 1.0));
    float bri = clamp((T - 700.0) / 3500.0, 0.0, 1.0);
    return vec3(r, g, b) * bri * bri;
}
"""

class Sky(object):
    
    def __init__(self):
        self.init_vertex()
        self.init_program()
        self.m = Matrix44.x_rotation(script.world.sky.rotation1*pi/180)
        self.m = self.m * Matrix44.y_rotation(script.world.sky.rotation0*pi/180)
        texture_name0 = search_imagefile(script.world.sky.texture0)
        self.texture_id0 = load_texture(texture_name0)
        self.texture_id_deep = load_texture(search_imagefile("deepsky.png"))
        self.texture_id_disc = load_texture(search_imagefile("accretion.png"))
        self.tex_h = (float(load_texture(texture_name0, id_only=False).height),
                      float(load_texture(search_imagefile("deepsky.png"), id_only=False).height))
        # texture_name1 = search_imagefile(script.world.sky.texture1)
        # self.texture_id1 = load_texture(texture_name1)

    def init_vertex(self, fov=None):
        self.quad_fov = fov if fov else VIEW_ANGLE
        # a quad just past the frame's corners at this (vertical) field of view
        z = -1.0
        y = 1.02 * tan(0.5*self.quad_fov*pi/180) * abs(z)
        x = y * BOX.X / BOX.Y
        vertex = [ x,  y, z,
                  -x,  y, z,
                  -x, -y, z,
                   x, -y, z]
        self.n = len(vertex)//3
        self.vertex = (GLfloat*len(vertex))(*vertex)

    exact = True      # trace the band the Kerr table cannot resolve, per pixel (the main pass)

    def init_program(self):
        self.accurate = (BOX.COLOR_MODE == "ACCURATE")
        vs = """
        #version 120
        varying vec3 pos;
        void main() {
            pos = vec3(gl_Vertex);
            gl_Position = gl_ProjectionMatrix * gl_Vertex;
        }
        """
        if self.accurate:
            # CMB: a true blackbody at T_CMB*ratio, normalised like the Sun
            colour = """
            vec3 outc = doppler_accurate(color.rgb, ratio, exposure);
            if (cmb_on > 0.5) {
                float logT = log(cmb_T * ratio) * 0.4342944819;
                vec3 cl = planck_at(logT).rgb * cmb_scale;
                outc = display(srgb_to_linear(outc) + cl * exposure);
            }
            if (lens_kerr > 0.5) {
                // the Kerr hole's disc, blended in by its share of the pixel;
                // the rest of the pixel is sky by its share, else shadow
                outc *= lens_wsky * (1.0 - lens_wdisc);
                if (lens_wdisc > 0.0) outc += lens_wdisc * doppler_T(dcolor.rgb, ratio0 * lens_gdisc, exposure, disc_logT);
            }
            gl_FragColor = vec4(outc, 1.0);
            """
            extra = blackbody.GLSL + """
            vec3 doppler_T(vec3 srgb, float D, float expo, float logT0) {
                // doppler_accurate with the source temperature known per pixel
                vec3 lin = srgb_to_linear(srgb);
                vec4 p0 = planck_at(logT0);
                vec4 p1 = planck_at(logT0 + log(D)*0.4342944819);
                return display(lin / max(p0.rgb, vec3(1e-6)) * p1.rgb * expo);
            }
            """
        else:
            colour = GAME_COLOUR + """
            if (cmb_on > 0.5) gl_FragColor.rgb += cmb_game(cmb_T * ratio);
            if (lens_kerr > 0.5) gl_FragColor.rgb = lens_wsky * (1.0 - lens_wdisc) * gl_FragColor.rgb + lens_wdisc * clamp(dcolor.rgb, 0.0, 1.0);
            """
            extra = CMB_GAME
        self.program = compile_program(vs, """
        #version 120
        #extension GL_ARB_shader_texture_lod : enable
        varying vec3 pos;
        uniform mat4 view_i;    // eye -> comoving rest frame (rotation)
        uniform mat4 skyrot;    // background frame -> sky texture frame (rotation)
        uniform vec3 sky_p;     // player position relative to the sky-sphere centre
        uniform float sky_R;    // sky-sphere radius
        uniform float g_obs;    // gravitational blueshift at the observer (1 far from any mass)
        uniform float sky_w;    // weight of the Milky Way panorama (1 inside the galaxy, 0 outside)
        uniform sampler2D texture0;
        uniform sampler2D texture_deep;
        uniform float cmb_on;   // 1: draw the cosmic microwave background
        uniform float cmb_T;    // its temperature (K)
        uniform vec2 tex_h;     // heights (texels) of the panorama and deep-sky maps
        uniform vec3 cmb_scale; // linear radiance scale: Sun-white at 5778 K
        uniform sampler2D disc_tex;  // a Kerr hole's accretion disc (drawn here, from the ray table)
        uniform float kd_rin, kd_rout;   // its edges (M)
        uniform float kd_Tin, kd_bin;    // temperature and sRGB brightness at the inner edge
        uniform float kerr_exact;        // 1: trace the band the table cannot resolve exactly, per pixel
        uniform float kerr_debug;        // 1: show (band, sky share, disc share) instead of the colour
        float disc_logT = 0.0;
        """ + boost.GLSL + LENS_GLSL + extra + kerrlib.TRACE_GLSL + """
        // The band the table cannot resolve: the four texels round the lookup
        // disagree (sky against shadow, disc against not, or the crossing
        // radius jumps, as across the thin images along the shadow's edge).
        bool kerr_band(vec2 uv) {
            vec2 f0 = floor(uv * kerr_size - 0.5);
            float sky0 = 0.0, in0 = 0.0, rmin = 1.0e9, rmax = -1.0e9;
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    vec2 c = (f0 + vec2(float(i), float(j)) + 0.5) / kerr_size;
                    c = vec2(fract(c.x), clamp(c.y, 0.0, 1.0));
                    float sky = (length(texture2D(kerr_table, c).xyz) > 0.5) ? 1.0 : 0.0;
                    float rr = texture2D(kerr_disc, c).r;
                    float ind = (kerr_rin > 0.0 && rr >= kerr_rin && rr <= kerr_rout) ? 1.0 : 0.0;
                    if (i == 0 && j == 0) { sky0 = sky; in0 = ind; }
                    else if (sky != sky0 || ind != in0) return true;
                    rmin = min(rmin, rr); rmax = max(rmax, rr);
                }
            }
            // a jump in the crossing radius that the interpolation would run
            // through the disc's range (two shadow texels, one crossing the plane
            // far out before falling in, one never: a false line of disc between)
            return kerr_rin > 0.0 && rmax > 1.3 * rmin && rmin < kerr_rout && rmax > kerr_rin;
        }
        void main() {
            gl_FragDepth = 1.0;
            vec3 d = normalize((view_i * vec4(pos, 0.0)).xyz);
            // photon direction in the background frame: Lorentz(-U) (d, -1)
            vec4 Xi = lz_neg(vec4(d, -1.0), tpv_null(d, 1.0, pb_n), pb_n, pb_u, pb_inv);
            float ratio0 = -1.0 / Xi.w;         // Doppler factor of the ship's motion, observed/emitted
            float ratio = ratio0 * g_obs;       // times the observer's gravitational blueshift
            // The background is a sphere of finite radius around the Sun, so the
            // texture direction depends on where the player is (parallax).
            vec3 dl0 = normalize(Xi.xyz);
            vec4 dcolor = vec4(0.0);
            if (lens_k > 0.0) {
                dl0 = lens_source(dl0);    // bent round the lens
                if (lens_kerr > 0.5) {
                    float in_band = (kerr_exact > 0.5 && kerr_band(lens_uv)) ? 1.0 : 0.0;
                    if (in_band > 0.5) {
                        // this pixel's own photon, where the table's texels disagree
                        vec3 xdir, xhit; float xinvE; vec4 xq;
                        int st = kerr_trace(lens_dtet, xdir, xinvE, xq, xhit);
                        lens_wsky = float(st);
                        lens_gs = (st == 1) ? xinvE : 1.0;
                        lens_disc_r = xq.r; lens_gdisc = xq.a;
                        lens_disc_u = atan(xq.b, xq.g) / 6.28318530717959;
                        lens_wdisc = (kerr_rin > 0.0 && xq.r >= kerr_rin && xq.r <= kerr_rout) ? 1.0 : 0.0;
                        lens_hit = xhit;
                        if (st == 1) dl0 = normalize(xdir.x*kerr_bx + xdir.y*kerr_by + xdir.z*kerr_bz);
                    }
                    if (lens_wdisc > 0.5 || lens_wsky <= 0.5) {
                        // the disc and the shadow are things at a distance: their
                        // depth, so that bodies behind them are hidden (the crossing
                        // point, or the hole, at its true distance in the ship's frame,
                        // as the bodies' own vertices are placed)
                        vec3 hv = (lens_hit.x*kerr_bx + lens_hit.y*kerr_by + lens_hit.z*kerr_bz) * kerr_M;
                        float dlab = max(length(hv), 1.0e-9);
                        vec3 dsh = lens_to_ship(hv / dlab);
                        float dist = dlab / max(lens_lab_dist(dsh), 1.0e-9);
                        float ze = normalize(pos).z * dist;
                        if (ze < 0.0) {
                            float zc = gl_ProjectionMatrix[2][2]*ze + gl_ProjectionMatrix[3][2];
                            gl_FragDepth = clamp(0.5*zc/(-ze) + 0.5, 0.0, 1.0);
                        }
                    }
                    if (kerr_debug > 0.5) { gl_FragColor = vec4(in_band, lens_wsky, lens_wdisc, 1.0); return; }
                    if (lens_wsky <= 0.0 && lens_wdisc <= 0.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
                    ratio = ratio0 * lens_gs;     // the table's per-direction frequency factor (the observer's included)
                    if (kd_rin > 0.0) {
                        // the disc's edges a pixel wide, wherever they fall between texels;
                        // the ramp's width is capped, so that a jump to a texel that
                        // never crossed the plane (a sentinel radius) is a hard cut,
                        // not a faint fringe of disc along the whole boundary
                        float fw = clamp(fwidth(lens_disc_r), 1e-6, 0.125 * (kd_rout - kd_rin));
                        lens_wdisc = clamp((lens_disc_r - kd_rin) / fw + 0.5, 0.0, 1.0) * clamp((kd_rout - lens_disc_r) / fw + 0.5, 0.0, 1.0);
                        lens_disc_r = clamp(lens_disc_r, kd_rin, kd_rout);
                    }
                    if (lens_wdisc > 0.0) {
                        // the hole's disc: hot gas, T ~ r^-3/4 and brightness
                        // ~ r^-3/2 from the inner edge, the pattern from the texture
                        float rr = max(lens_disc_r / kd_rin, 1.0);
                        disc_logT = log(kd_Tin * pow(rr, -0.75)) * 0.4342944819;
                        float b = kd_bin * pow(rr, -1.5);
                        float vv = log(rr) / log(kd_rout / kd_rin);
                        dcolor = vec4(texture2D(disc_tex, vec2(lens_disc_u, vv)).rgb * b, 1.0);
                    }
                } else {
                    if (dot(dl0, dl0) < 0.5) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }   // a black hole's shadow
                    if (lens_exact > 0.5) ratio *= lens_gs / g_obs;   // the table's own per-direction frequency factor
                }
            }
            vec3 dl = dl0;
            float pd = dot(sky_p, dl);
            float disc = pd*pd + sky_R*sky_R - dot(sky_p, sky_p);
            if (disc > 0.0) {
                float sdist = -pd + sqrt(disc);
                dl = normalize(sky_p + sdist*dl);
            }
            dl = (skyrot * vec4(dl, 0.0)).xyz;
            // mip level from the angle one pixel spans on the sky, the same
            // in every direction: the implicit level, taken from texture
            // coordinates, blows up at the map's poles and smears them
            float apx = max(length(dFdx(dl0)), length(dFdy(dl0)));
            float lod0 = log2(max(apx * tex_h.x * 0.31830988618379067, 1e-6));
            float lodd = log2(max(apx * tex_h.y * 0.31830988618379067, 1e-6));
            float s, t;
            s = atan(dl.y, dl.x) * 0.15915494309189533;
            if (s < 0.0) s += 1.0;
            t = 1.0 - atan(length(dl.xy), dl.z) * 0.31830988618379067;
            vec4 color = texture2DLod(texture0, vec2(s, t), lod0) * sky_w;
            // extragalactic background, truly at infinity
            vec3 dd = (skyrot * vec4(dl0, 0.0)).xyz;
            s = atan(dd.y, dd.x) * 0.15915494309189533;
            if (s < 0.0) s += 1.0;
            t = 1.0 - atan(length(dd.xy), dd.z) * 0.31830988618379067;
            color += texture2DLod(texture_deep, vec2(s, t), lodd) * (1.0 - 0.6*sky_w);
            """ + colour + """
        }
        """)
        self.view_local = glGetUniformLocation(self.program, "view_i")
        self.exact_local = glGetUniformLocation(self.program, "kerr_exact")
        self.debug_local = glGetUniformLocation(self.program, "kerr_debug")
        self.skyp_local = glGetUniformLocation(self.program, "sky_p")
        self.skyR_local = glGetUniformLocation(self.program, "sky_R")
        self.gobs_local = glGetUniformLocation(self.program, "g_obs")
        self.skyw_local = glGetUniformLocation(self.program, "sky_w")
        self.texdeep_local = glGetUniformLocation(self.program, "texture_deep")
        self.texdisc_local = glGetUniformLocation(self.program, "disc_tex")
        self.rot_local = glGetUniformLocation(self.program, "skyrot")
        self.tex0_local = glGetUniformLocation(self.program, "texture0")
        self.cmb_on_local = glGetUniformLocation(self.program, "cmb_on")
        self.cmb_T_local = glGetUniformLocation(self.program, "cmb_T")
        self.cmb_scale_local = glGetUniformLocation(self.program, "cmb_scale")
        self.texh_local = glGetUniformLocation(self.program, "tex_h")
        # Normalise so that a blackbody at the Sun's temperature renders with
        # the Sun's surface brightness (the Sun is a white texture times its
        # script brightness, in sRGB units, before the gamma curve).
        T_sun, b_sun = 5778.0, 2000.0
        for star in script.world.solar.stars:
            if star.name.lower() == "sun":
                T_sun = float(getattr(star, "temperature", T_sun))
                b_sun = float(getattr(star, "brightness", b_sun))
        pr = blackbody.planck_rgb(T_sun)
        lin = b_sun ** 2.2
        self.cmb_scale = tuple(lin / max(c, 1e-9) for c in pr)

    def draw(self, pm, U, player_rel=None, weight=1.0, scale=1.0, gshift=1.0, fov=None):
        """pm: inverse view rotation (eye -> rest frame); U: player's 4-velocity;
        player_rel: player position relative to the sky-sphere centre (Vector4D);
        weight: 1 inside the galaxy (panorama shown), 0 far outside.
        scale: output radiance factor (glare source pass); the game colour
        path has no radiometric meaning, so it draws nothing when scaled."""
        if scale != 1.0 and not self.accurate:
            return
        if (fov if fov else VIEW_ANGLE) != getattr(self, "quad_fov", VIEW_ANGLE):
            self.init_vertex(fov)
        glUseProgram(self.program)
        glUniform1f(self.skyw_local, weight)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, self.texture_id_deep)
        glUniform1i(self.texdeep_local, 3)
        glActiveTexture(GL_TEXTURE11)          # 8, 9 and 10 hold the Kerr tables
        glBindTexture(GL_TEXTURE_2D, self.texture_id_disc)
        glUniform1i(self.texdisc_local, 11)
        glUniform1f(self.exact_local, 1.0 if (scale == 1.0 and self.exact) else 0.0)
        glUniform1f(self.debug_local, 1.0 if getattr(self, 'debug', False) else 0.0)
        if player_rel is None:
            glUniform3f(self.skyp_local, 0.0, 0.0, 0.0)
        else:
            glUniform3f(self.skyp_local, player_rel.x, player_rel.y, player_rel.z)
        glUniform1f(self.skyR_local, SKY_RADIUS)
        glUniform2f(self.texh_local, *self.tex_h)
        glUniform1f(self.gobs_local, gshift)
        glUniform1f(self.cmb_on_local, 1.0 if BOX.CMB else 0.0)
        glUniform1f(self.cmb_T_local, T_CMB)
        glUniform3f(self.cmb_scale_local, *self.cmb_scale)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id0)
        glUniform1i(self.tex0_local, 0)
        glUniformMatrix4fv(self.view_local, 1, GL_FALSE, pm.to_glsl())
        glUniformMatrix4fv(self.rot_local, 1, GL_FALSE, self.m.to_glsl())
        boost.set_uniforms(self.program, U)
        Lensing.set_uniforms(self.program, table_unit=4)
        if self.accurate:
            # ucolor-style scaling acts before the gamma curve: scale^2.2 in linear
            blackbody.set_uniforms(self.program, BOX.exposure() * scale ** 2.2)
        glVertexPointer(3, GL_FLOAT, 0, self.vertex)
        glDrawArrays(GL_QUADS, 0, self.n)
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)
