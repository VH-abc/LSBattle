# coding: utf8
# entity/audio.py
"""
Sound: a small software mixer streamed through SDL's audio queue, with
sounds that carry the physics the game already computes.

* Music, as a signal from a fixed point in the background frame (the home
  star): its playback rate is the same factor the sky's colours get, the
  Doppler factor of the ship's motion toward or away from it times the
  gravitational shift, with no clamp.  Flying toward the Sun it rises and
  quickens, away it drops and drags, near a black hole it is blueshifted,
  and inside the horizon it dies away.  The track is resources/sound/
  music.wav when present, else a chiptune loop written here at start-up.
* The ship's own sound, unshifted since it is in its frame: a thrust hum
  whose pitch follows the thrust in force.
* Two clocks: a tick for every second of the ship's proper time, and a
  higher tick for every second of world time, rate-limited, so the two
  are heard to diverge when time runs slow for the ship.
* The sky: a hiss that brightens with the blueshift of the cosmic
  background ahead, and a Geiger-like crackle in the glare of an accretion
  disc.

Everything is float32 at 44.1 kHz, mono, generated a frame at a time with
numpy and kept about 120 ms ahead of the device.
"""
import os
import ctypes
import math

import numpy as np

from program.box import BOX
from program.const import IMG_DIR

RATE = 44100
AHEAD = 0.20            # seconds of audio kept queued (a frame hitch must not starve the device)
MAXBLOCK = 0.5          # never more than this per update (a frame hitch)
AUDIO_F32SYS = 0x8120   # little-endian float32 (SDL_AudioFormat)


class SDL_AudioSpec(ctypes.Structure):
    _fields_ = [("freq", ctypes.c_int), ("format", ctypes.c_uint16),
                ("channels", ctypes.c_uint8), ("silence", ctypes.c_uint8),
                ("samples", ctypes.c_uint16), ("padding", ctypes.c_uint16),
                ("size", ctypes.c_uint32), ("callback", ctypes.c_void_p),
                ("userdata", ctypes.c_void_p)]


def _sdl_audio():
    """ctypes bindings of the SDL audio functions the mixer needs."""
    import sdl2
    from sdl2.dll import _bind
    f = {}
    f["init"] = _bind("SDL_InitSubSystem", [ctypes.c_uint32], ctypes.c_int)
    f["open"] = _bind("SDL_OpenAudioDevice", [ctypes.c_char_p, ctypes.c_int,
                      ctypes.POINTER(SDL_AudioSpec), ctypes.POINTER(SDL_AudioSpec), ctypes.c_int], ctypes.c_uint32)
    f["pause"] = _bind("SDL_PauseAudioDevice", [ctypes.c_uint32, ctypes.c_int], None)
    f["queue"] = _bind("SDL_QueueAudio", [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int)
    f["queued"] = _bind("SDL_GetQueuedAudioSize", [ctypes.c_uint32], ctypes.c_uint32)
    f["clear"] = _bind("SDL_ClearQueuedAudio", [ctypes.c_uint32], None)
    f["close"] = _bind("SDL_CloseAudioDevice", [ctypes.c_uint32], None)
    f["INIT_AUDIO"] = sdl2.SDL_INIT_AUDIO
    return f


# ----------------------------------------------------------------- the music
def _lowpass(x, fc, rate=RATE, taps=48):
    """One-pole lowpass at fc as a truncated exponential kernel (vectorised)."""
    a = math.exp(-2.0 * math.pi * fc / rate)
    kern = (1.0 - a) * a ** np.arange(taps)
    return np.convolve(x, kern)[:x.size]


def _square(ph, duty=0.5):
    return np.where((ph % 1.0) < duty, 1.0, -1.0)


def _tri(ph):
    return 4.0 * np.abs((ph % 1.0) - 0.5) - 1.0


MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def chiptune(rate=RATE, bpm=150.0, bars=8, seed=7, root=9, mode="minor", style="bright"):
    """A loop in the key `root` (semitones from C) and `mode`, in a style:
    bright (square arpeggios over a triangle bass, a kick and a hat),
    driving (faster, syncopated bass, busier arpeggio), dark (slow drone,
    sparse low arpeggio, no drums), whirl (a running arpeggio that spirals
    up and down through two octaves under a slow swell, over a drone whose
    pitch wobbles and a soft pulse on the beat: the dragged frame).
    Returns float32 in [-1, 1]."""
    rng = np.random.RandomState(seed)
    beat = 60.0 / bpm
    bar = 4 * beat
    total = int(bars * bar * rate)
    t = np.arange(total) / float(rate)
    out = np.zeros(total)
    scale = MAJOR if mode == "major" else MINOR
    base = 261.6256 * 2.0 ** (root / 12.0)          # the key's root above middle C
    def deg(d, octave=0):
        o, i = divmod(d, 7)
        return base * 2.0 ** ((scale[i] + 12*(o + octave)) / 12.0)
    # chord progressions by scale degree (0-based): I V vi IV / i VI III VII
    prog = [0, 4, 5, 3, 0, 4, 5, 3] if mode == "major" else [0, 5, 2, 6, 0, 5, 2, 4]
    chords = [(d, d + 2, d + 4) for d in prog]
    if style == "dark":
        lead_amp, bass_amp, duty, lead_oct, drums, step = 0.14, 0.3, 0.5, -1, False, 2
    elif style == "calm":
        lead_amp, bass_amp, duty, lead_oct, drums, step = 0.2, 0.26, None, 0, False, 2
    elif style == "driving":
        lead_amp, bass_amp, duty, lead_oct, drums, step = 0.22, 0.3, 0.125, 0, True, 1
    elif style == "whirl":
        lead_amp, bass_amp, duty, lead_oct, drums, step = 0.2, 0.3, 0.25, 0, False, 1
    else:
        lead_amp, bass_amp, duty, lead_oct, drums, step = 0.22, 0.28, 0.25, 0, True, 1
    sixteenth = beat / 4.0
    pattern = [0, 1, 2, 1, 0, 2, 1, 2]
    spiral = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1]      # chord tone k % 3, octave k // 3
    for b in range(bars):
        ch = chords[b % len(chords)]
        for s_ in range(0, 16, step):
            i0 = int((b * bar + s_ * sixteenth) * rate)
            i1 = min(int((b * bar + (s_ + step) * sixteenth) * rate), total)
            if i1 <= i0:
                continue
            if style == "whirl":
                k = spiral[(b * 16 + s_) % len(spiral)]
                f = deg(ch[k % 3], k // 3 + lead_oct)
            else:
                k = pattern[(s_ // step) % len(pattern)]
                oct_ = (1 if (s_ // 8) % 2 else 0) + lead_oct
                f = deg(ch[k], oct_)
            tt = t[i0:i1] - t[i0]
            env = np.exp(-tt * (3.0 if style == "dark" else 9.0)) * (1.0 - np.exp(-tt * 400.0))
            out[i0:i1] += lead_amp * (_tri(f * tt) if duty is None else _square(f * tt, duty)) * env
    if style == "whirl":
        # the swell: two bars round, so the loop closes on itself
        out *= 0.65 + 0.35 * np.sin(2.0 * np.pi * t / (2.0 * bar))
    eighth = beat / 2.0
    for b in range(bars):
        root_f = deg(chords[b % len(chords)][0], -1)
        if style == "whirl":
            i0 = int(b * bar * rate); i1 = min(int((b + 1) * bar * rate), total)
            tt = t[i0:i1] - t[i0]
            wob = 0.25 * np.sin(2.0 * np.pi * tt / bar)        # one wobble a bar
            out[i0:i1] += bass_amp * _tri(root_f * 0.5 * tt + wob) * (1.0 - np.exp(-tt * 6.0))
            for q in range(4):
                j0 = int((b * bar + q * beat) * rate); j1 = min(j0 + int(0.25 * rate), total)
                tj = t[j0:j1] - t[j0]
                fk = 36.0 + 60.0 * np.exp(-tj * 20.0)
                out[j0:j1] += 0.45 * np.sin(2.0 * np.pi * np.cumsum(fk) / rate) * np.exp(-tj * 9.0)
            continue
        if style == "dark":
            i0 = int(b * bar * rate); i1 = min(int((b + 1) * bar * rate), total)
            tt = t[i0:i1] - t[i0]
            out[i0:i1] += bass_amp * _tri(root_f * 0.5 * tt) * (1.0 - np.exp(-tt * 3.0)) * np.exp(-tt * 0.4)
            continue
        for s_ in range(8):
            i0 = int((b * bar + s_ * eighth) * rate)
            i1 = min(int((b * bar + (s_ + 1) * eighth) * rate), total)
            tt = t[i0:i1] - t[i0]
            f = root_f * (2.0 if (s_ % 4 == 3 or (style == "driving" and s_ % 2 == 1)) else 1.0)
            out[i0:i1] += bass_amp * _tri(f * tt) * np.exp(-tt * 5.0)
    if drums:
        for b in range(bars):
            for q in range(4):
                i0 = int((b * bar + q * beat) * rate)
                i1 = min(i0 + int(0.18 * rate), total)
                tt = t[i0:i1] - t[i0]
                if q % 2 == 0:
                    fk = 40.0 + 120.0 * np.exp(-tt * 30.0)
                    out[i0:i1] += 0.6 * np.sin(2.0 * np.pi * np.cumsum(fk) / rate) * np.exp(-tt * 14.0)
                else:
                    out[i0:i1] += 0.25 * rng.uniform(-1, 1, i1 - i0) * np.exp(-tt * 25.0)
                j0 = int((b * bar + (q + 0.5) * beat) * rate)
                j1 = min(j0 + int(0.05 * rate), total)
                tj = t[j0:j1] - t[j0]
                out[j0:j1] += 0.12 * rng.uniform(-1, 1, j1 - j0) * np.exp(-tj * 80.0)
    y = _lowpass(out, 6000.0, rate)
    peak = np.max(np.abs(y)) or 1.0
    return (0.9 * y / peak).astype(np.float32)


# The themes.  "Bathed in the Light" (the Sun's, when its WAV is present) is
# in A major; the generated ones sit in keys sharing six or seven of its
# notes: D major, E major, F sharp minor (the relative minor), B minor.
THEMES = {
    # roots: C 0, C# 1, D 2, E 4, F# 6, A 9, B 11
    "sun":         dict(file="music.wav", root=9, mode="major", bpm=150.0, style="bright", seed=7),
    "earth":       dict(root=9, mode="major", bpm=118.0, style="calm", seed=21),
    "moon":        dict(root=6, mode="minor", bpm=60.0, style="calm", seed=22),
    "venus":       dict(root=2, mode="major", bpm=108.0, style="bright", seed=23),
    "mars":        dict(root=4, mode="major", bpm=150.0, style="driving", seed=24),
    "jupiter":     dict(root=11, mode="minor", bpm=128.0, style="driving", seed=25),
    "saturn":      dict(root=2, mode="major", bpm=92.0, style="calm", seed=26),
    "neutronstar": dict(root=1, mode="minor", bpm=180.0, style="driving", seed=27),
    "sirius":      dict(root=9, mode="major", bpm=160.0, style="bright", seed=28),
    "proxima":     dict(root=11, mode="minor", bpm=72.0, style="dark", seed=29),
    "betelgeuse":  dict(root=6, mode="minor", bpm=52.0, style="dark", seed=30),
    "helios":      dict(root=2, mode="major", bpm=140.0, style="bright", seed=11),
    "forge":       dict(root=4, mode="major", bpm=165.0, style="driving", seed=5),
    "cinder":      dict(root=4, mode="major", bpm=140.0, style="bright", seed=31),
    "abyss":       dict(root=6, mode="minor", bpm=66.0, style="dark", seed=3),
    "pyre":        dict(root=1, mode="minor", bpm=170.0, style="driving", seed=32),
    "ash":         dict(root=6, mode="minor", bpm=90.0, style="calm", seed=33),
    "blackhole":   dict(root=11, mode="minor", bpm=80.0, style="dark", seed=13),
    "maelstrom":   dict(root=1, mode="minor", bpm=132.0, style="whirl", seed=41),
    "beacon":      dict(root=4, mode="major", bpm=156.0, style="bright", seed=42),
    "eddy":        dict(root=9, mode="major", bpm=112.0, style="calm", seed=43),
    "wake":        dict(root=2, mode="major", bpm=84.0, style="calm", seed=44),
}
FADE = 2.0        # seconds for a theme to fade in or out


def load_wav(path):
    """A 16-bit PCM WAV as float32 mono at RATE, or None."""
    try:
        import wave
        w = wave.open(path, "rb")
        n, ch, sw, fr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
        w.close()
        if sw != 2:
            return None
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        if fr != RATE:
            x = np.arange(a.size) * (RATE / float(fr))
            xi = np.arange(int(a.size * RATE / float(fr)))
            a = np.interp(xi, x, a).astype(np.float32)
        return a
    except Exception:
        return None


# ----------------------------------------------------------------- the mixer
class Audio(object):

    def __init__(self, device=True):
        self.ok = False
        self.dev = 0
        self.hum_phase = 0.0
        self.hum_amp = 0.0
        self.tick_p = 0.0          # player seconds since the last proper tick
        self.tick_w = 0.0          # world seconds since the last world tick
        self.rng = np.random.RandomState(1)
        sound_dir = os.path.join(os.path.dirname(IMG_DIR.rstrip("/")), "sound")
        # a theme per body: its loop, read position, gain and target gain
        self.themes = {}
        for name, spec in THEMES.items():
            loop = None
            if "file" in spec:
                path = os.path.join(sound_dir, spec["file"])
                loop = load_wav(path) if os.path.exists(path) else None
            if loop is None or loop.size < RATE:
                loop = chiptune(root=spec["root"], mode=spec["mode"], bpm=spec["bpm"], style=spec["style"], seed=spec["seed"])
            self.themes[name] = dict(loop=loop, pos=0.0, gain=0.0, target=0.0, rate=1.0, amp=1.0, ref=None)
        if not device:
            return
        try:
            self.f = _sdl_audio()
            self.f["init"](self.f["INIT_AUDIO"])
            want = SDL_AudioSpec()
            want.freq = RATE; want.format = AUDIO_F32SYS; want.channels = 1; want.samples = 1024
            have = SDL_AudioSpec()
            self.dev = self.f["open"](None, 0, ctypes.byref(want), ctypes.byref(have), 0)
            if self.dev == 0 or have.freq != RATE or have.format != AUDIO_F32SYS or have.channels != 1:
                self.dev = 0
                return
            self.f["pause"](self.dev, 0)
            self.ok = True
        except Exception:
            self.ok = False

    def close(self):
        if self.ok:
            try:
                self.f["close"](self.dev)
            except Exception:
                pass
            self.ok = False

    @staticmethod
    def theme_of(solar, body):
        """The theme a body plays: its own, else its primary's, up the chain."""
        depth = 0
        while body is not None and depth < 8:
            if body.name.lower() in THEMES:
                return body.name.lower()
            body = solar.stars.get(body.orbit[0]) if body.orbit is not None else None
            depth += 1
        return None

    # ------------------------------------------------------------ the state
    def state_from(self, world, dt_real):
        """What the mixer needs this frame, from the world."""
        pl = world.player; P = pl.P; U = P.U
        s = dict(dt=dt_real, volume=BOX.VOLUME if BOX.SOUND else 0.0)
        # themes: gravity at the ship per theme (a body's goes to its theme,
        # bodies without one to their primary's); the top one or two play,
        # the second at a gain set by how close its pull comes to the first's
        solar = world.solar
        g = U.t
        bx, by, bz = (U.x / g, U.y / g, U.z / g) if g > 0.0 else (0.0, 0.0, 0.0)
        pull = {}
        for star in solar.stars.values():
            if star.GM <= 0.0 or star.hp <= 0:
                continue
            th = self.theme_of(solar, star)
            if th is None:
                continue
            d = (star.X - P.X).length()
            if d <= 0.0:
                continue
            pull[th] = pull.get(th, 0.0) + star.GM / (d*d)
        ranked = sorted(pull.items(), key=lambda kv: -kv[1])
        targets = {}
        if ranked:
            g1 = ranked[0][1]
            targets[ranked[0][0]] = 1.0
            if len(ranked) > 1 and ranked[1][1] > 0.0:
                ratio = ranked[1][1] / g1
                targets[ranked[1][0]] = max(0.0, min(0.7, 0.7 * (ratio / 0.02) ** 0.5)) if ratio > 1.0e-4 else 0.0
        obs = getattr(solar, "obs_shift", 1.0)
        shifts = getattr(solar, "shifts", {})
        dop = {}
        for th in self.themes:
            self.themes[th]["target"] = targets.get(th, 0.0)
            body = solar.stars.get(th)
            if body is None:
                dop[th] = 1.0
                continue
            # the source moves too: f_obs/f_em = (U_obs . k) / (U_src . k),
            # k the photon direction from the body's retarded position
            Xc, Us = body.X, body.U
            if body.orbit is not None and BOX.ORBITS:
                Xc, Us = solar.state_at(body, P.X.t - (body.X - P.X).length())
            d = P.X - Xc
            n = d.length()
            if n <= 0.0:
                dop[th] = 1.0
                continue
            kx, ky, kz = d.x / n, d.y / n, d.z / n
            num = U.t - (U.x*kx + U.y*ky + U.z*kz)
            den = Us.t - (Us.x*kx + Us.y*ky + Us.z*kz)
            D = num / den if den > 0.0 else 1.0
            D *= obs * shifts.get(th, 1.0) * getattr(body, "self_shift", 1.0)
            dop[th] = max(D, 1e-9)
            # loudness like light: power falls as the inverse square of the
            # distance (amplitude as 1/d) from a reference distance, the
            # orbit of the body's innermost planet, and rises as the fourth
            # power of the Doppler factor (amplitude as D^2)
            t = self.themes[th]
            if t["ref"] is None:
                rs = [(b.X0 - body.X0).length() for b in solar.stars.values()
                      if b.orbit is not None and b.orbit[0] == th]
                t["ref"] = min(rs) if rs else 20.0 * max(body.radius, 1e-9)
            t["amp"] = min(2.0, (t["ref"] / n) * D * D)
        s["doppler"] = dop
        # inside a horizon the outside fades away; at the singularity nothing
        fade = 1.0
        if pl.plunged is not None:
            fade = 0.0
        elif pl.inside is not None:
            Rs = 2.0 * pl.inside.GM
            r = (pl.inside.X - P.X).length()
            fade = max(0.0, min(1.0, (r / Rs - 0.2) / 0.8))
        s["fade"] = fade
        # thrust in force (c/s) and whether it is on
        keys = getattr(world, "_keys", None)
        on = bool(keys and (keys.k_accel or keys.k_booster)) and pl.plunged is None
        s["thrust_on"] = on
        s["thrust"] = BOX.thrust(pl.acceleration) * (3.0 if (keys and keys.k_booster) else 1.0)
        # clocks: world seconds per player second
        s["world_rate"] = BOX.time_rate() * g * getattr(world, "rate_limit", 1.0)
        # the blueshift ahead: cosmic background temperature factor gamma (1 + beta)
        beta = (bx*bx + by*by + bz*bz) ** 0.5 if g > 0.0 else 0.0
        s["blueshift"] = g * (1.0 + beta) * obs
        # an accretion disc's glare at the ship: clicks per second
        from program.utils.lensing import Lensing
        c = Lensing.current
        rate = 0.0
        if c is not None and c["body"].disk is not None:
            Dr = c["D"] / c["Rs"]
            rate = 60.0 * (3.0 / max(Dr, 0.5)) ** 2 * c["body"].disk.brightness / 600.0
        s["crackle"] = min(400.0, rate)
        return s

    # ------------------------------------------------------------ the sound
    def render(self, s, n):
        """n samples of everything, float32."""
        out = np.zeros(n, dtype=np.float64)
        vol = s["volume"]
        if vol <= 0.0 or n <= 0:
            return out.astype(np.float32)
        # the themes, each resampled at its own Doppler rate (unclamped),
        # gains easing to their targets over FADE seconds
        dt = s["dt"]
        k = min(1.0, dt / FADE)
        for name, th in self.themes.items():
            g0 = th["gain"]; g1 = g0 + (th["target"] - g0) * k
            th["gain"] = g1
            if g0 < 1e-4 and g1 < 1e-4:
                continue
            m = th["loop"]; L = m.size
            r = s["doppler"].get(name, 1.0)
            idx = th["pos"] + r * np.arange(n)
            th["pos"] = (th["pos"] + r * n) % L
            i0 = np.floor(idx).astype(np.int64) % L
            fr = idx - np.floor(idx)
            i1 = (i0 + 1) % L
            out += (m[i0] * (1.0 - fr) + m[i1] * fr) * (np.linspace(g0, g1, n) * s["fade"] * th["amp"])
        if not BOX.EFFECTS:
            out *= vol
            return self.finish(out)
        # thrust hum: pitch with the thrust in force
        a = s["thrust"] if s["thrust_on"] else 0.0
        target = 0.25 if s["thrust_on"] else 0.0
        f = 70.0 * 2.0 ** (max(-3.0, min(6.0, math.log10(max(a, 1e-30) / 1e-9))) / 3.0) if a > 0.0 else 70.0
        ph = self.hum_phase + 2.0 * np.pi * f * np.arange(1, n + 1) / RATE
        self.hum_phase = float(ph[-1] % (2.0 * np.pi))
        k = min(1.0, n / (0.08 * RATE))
        amp = self.hum_amp + (target - self.hum_amp) * k
        env = np.linspace(self.hum_amp, amp, n)
        self.hum_amp = amp
        out += (np.sin(ph) + 0.4 * np.sin(2.0 * ph) + 0.2 * np.sin(3.0 * ph)) * env
        # clocks
        dt = s["dt"]
        self.tick_p += dt
        if self.tick_p >= 1.0:
            self.tick_p -= 1.0
            out[:min(n, 900)] += 0.25 * np.sin(2.0 * np.pi * 1000.0 * np.arange(min(n, 900)) / RATE) * np.exp(-np.arange(min(n, 900)) / 300.0)
        wr = s["world_rate"]
        if abs(wr - 1.0) > 0.05:
            self.tick_w += dt * min(wr, 8.0)
            while self.tick_w >= 1.0:
                self.tick_w -= 1.0
                j = int(self.rng.uniform(0, max(1, n - 400)))
                k = min(400, n - j)
                out[j:j + k] += 0.18 * np.sin(2.0 * np.pi * 1500.0 * np.arange(k) / RATE) * np.exp(-np.arange(k) / 130.0)
        else:
            self.tick_w = 0.0
        # the blueshifted background: a hiss
        h = max(0.0, min(1.0, math.log10(max(s["blueshift"], 1.0)) / 4.0))
        if h > 0.0:
            noise = self.rng.uniform(-1, 1, n)
            out += 0.35 * h * _lowpass(noise, 2500.0)
        # the disc's crackle: Poisson clicks
        cr = s["crackle"]
        if cr > 0.0:
            hits = self.rng.uniform(0, 1, n) < (cr / RATE)
            if hits.any():
                imp = np.zeros(n); imp[hits] = self.rng.uniform(0.3, 1.0, int(hits.sum())) * self.rng.choice([-1.0, 1.0], int(hits.sum()))
                kern = np.exp(-np.arange(60) / 12.0)
                out += 0.5 * np.convolve(imp, kern)[:n]
        out *= vol
        return self.finish(out)

    LP_TAPS = 128

    def finish(self, out):
        """The output stage: a lowpass (12 dB/octave above 2 kHz)
        so a blueshifted track is bright, not screechy, then the clip.  The
        filter sees the previous block's tail first, so there is no step at
        the block boundary (a per-block convolution crackled at 60 Hz)."""
        hist = getattr(self, "lp_hist", None)
        if hist is None:
            hist = np.zeros(self.LP_TAPS)
        x = np.concatenate([hist, out])
        # two poles at 2 kHz: 12 dB per octave above it
        y = _lowpass(_lowpass(x, 2000.0, taps=self.LP_TAPS), 2000.0, taps=self.LP_TAPS)[self.LP_TAPS:]
        self.lp_hist = x[-self.LP_TAPS:]
        return np.tanh(y).astype(np.float32)       # a soft limiter: loud approaches saturate, never clip

    def update(self, world, dt_real):
        """Called once a frame: top the queue up to AHEAD seconds."""
        if not self.ok:
            return
        try:
            queued = self.f["queued"](self.dev) / 4.0
            n = int(AHEAD * RATE - queued)
            if n <= 0:
                return
            n = min(n, int(MAXBLOCK * RATE))
            s = self.state_from(world, dt_real)
            buf = self.render(s, n)
            self.f["queue"](self.dev, buf.ctypes.data, buf.nbytes)
        except Exception:
            self.ok = False
