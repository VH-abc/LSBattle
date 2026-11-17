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
AHEAD = 0.12            # seconds of audio kept queued
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


def chiptune(rate=RATE, bpm=150.0, bars=8, seed=7):
    """A loop in A minor: square-wave arpeggios over a triangle bass, a kick
    on 1 and 3, a hat on the offbeats.  Returns float32 in [-1, 1]."""
    rng = np.random.RandomState(seed)
    beat = 60.0 / bpm
    bar = 4 * beat
    total = int(bars * bar * rate)
    t = np.arange(total) / float(rate)
    out = np.zeros(total)
    A = 220.0
    def note(semi, octave=0):
        return A * 2.0 ** (semi / 12.0 + octave)
    # chords (semitones from A): Am, F, C, G, Am, F, C, E
    chords = [(0, 3, 7), (8, 12, 15), (3, 7, 10), (10, 14, 17), (0, 3, 7), (8, 12, 15), (3, 7, 10), (7, 11, 14)]
    # arpeggio: 16th notes cycling the chord, up then down, with an octave jump
    sixteenth = beat / 4.0
    pattern = [0, 1, 2, 1, 0, 2, 1, 2]
    for b in range(bars):
        ch = chords[b % len(chords)]
        for s in range(16):
            i0 = int((b * bar + s * sixteenth) * rate)
            i1 = min(int((b * bar + (s + 1) * sixteenth) * rate), total)
            if i1 <= i0:
                continue
            k = pattern[s % len(pattern)]
            oct_ = 1 if (s // 8) % 2 else 0
            f = note(ch[k], oct_)
            tt = t[i0:i1] - t[i0]
            env = np.exp(-tt * 9.0) * (1.0 - np.exp(-tt * 400.0))
            out[i0:i1] += 0.22 * _square(f * tt, 0.25) * env
    # bass: root in eighths, triangle
    eighth = beat / 2.0
    for b in range(bars):
        root = note(chords[b % len(chords)][0], -1)
        for s in range(8):
            i0 = int((b * bar + s * eighth) * rate)
            i1 = min(int((b * bar + (s + 1) * eighth) * rate), total)
            tt = t[i0:i1] - t[i0]
            f = root * (2.0 if s % 4 == 3 else 1.0)
            env = np.exp(-tt * 5.0)
            out[i0:i1] += 0.28 * _tri(f * tt) * env
    # drums
    for b in range(bars):
        for q in range(4):
            i0 = int((b * bar + q * beat) * rate)
            i1 = min(i0 + int(0.18 * rate), total)
            tt = t[i0:i1] - t[i0]
            if q % 2 == 0:      # kick: a sine sweep
                fk = 40.0 + 120.0 * np.exp(-tt * 30.0)
                out[i0:i1] += 0.6 * np.sin(2.0 * np.pi * np.cumsum(fk) / rate) * np.exp(-tt * 14.0)
            else:               # snare-ish: noise burst
                out[i0:i1] += 0.25 * rng.uniform(-1, 1, i1 - i0) * np.exp(-tt * 25.0)
            # hat on the offbeat
            j0 = int((b * bar + (q + 0.5) * beat) * rate)
            j1 = min(j0 + int(0.05 * rate), total)
            tj = t[j0:j1] - t[j0]
            out[j0:j1] += 0.12 * rng.uniform(-1, 1, j1 - j0) * np.exp(-tj * 80.0)
    # a lowpass to take the edge off the squares: an exponential kernel
    y = _lowpass(out, 6000.0, rate)
    peak = np.max(np.abs(y)) or 1.0
    return (0.9 * y / peak).astype(np.float32)


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
        self.music = None
        self.pos = 0.0
        self.hum_phase = 0.0
        self.hum_amp = 0.0
        self.tick_p = 0.0          # player seconds since the last proper tick
        self.tick_w = 0.0          # world seconds since the last world tick
        self.hiss_state = 0.0
        self.crackle_state = 0.0
        self.rng = np.random.RandomState(1)
        self.last = {}
        path = os.path.join(os.path.dirname(IMG_DIR.rstrip("/")), "sound", "music.wav")
        self.music = load_wav(path) if os.path.exists(path) else None
        if self.music is None or self.music.size < RATE:
            self.music = chiptune()
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

    # ------------------------------------------------------------ the state
    def state_from(self, world, dt_real):
        """What the mixer needs this frame, from the world."""
        pl = world.player; P = pl.P; U = P.U
        s = dict(dt=dt_real, volume=BOX.VOLUME if BOX.SOUND else 0.0)
        # the music's Doppler factor: source at rest at the home star
        sun = world.solar.stars.get("sun")
        g = U.t
        if sun is not None and g > 0.0:
            d = sun.X - P.X
            n = d.length()
            bx, by, bz = U.x / g, U.y / g, U.z / g
            if n > 0.0:
                D = g * (1.0 + (bx*d.x + by*d.y + bz*d.z) / n)
            else:
                D = 1.0
            sh = getattr(world.solar, "shifts", {}).get("sun", 1.0)
            D *= getattr(world.solar, "obs_shift", 1.0) * sh * getattr(sun, "self_shift", 1.0)
        else:
            D = 1.0
        s["doppler"] = max(D, 1e-9)
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
        s["blueshift"] = g * (1.0 + beta) * getattr(world.solar, "obs_shift", 1.0)
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
        # music, resampled at the Doppler rate (unclamped)
        m = self.music; L = m.size
        r = s["doppler"]
        idx = self.pos + r * np.arange(n)
        self.pos = (self.pos + r * n) % L
        i0 = np.floor(idx).astype(np.int64) % L
        fr = idx - np.floor(idx)
        i1 = (i0 + 1) % L
        out += (m[i0] * (1.0 - fr) + m[i1] * fr) * (0.6 * s["fade"])
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
        return np.clip(out, -1.0, 1.0).astype(np.float32)

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
