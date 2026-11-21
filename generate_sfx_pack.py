# -*- coding: utf-8 -*-
"""
generate_sfx_pack.py — Studio-Grade Physical-Modeling & Subtractive/FM Foley SFX Generator
===========================================================================================
Synthesizes 100% original, royalty-free (CC0 / GPL-3.0 compatible) 48 kHz 16-bit stereo WAV
sound effects for InkBlitz 3D video recording:
  1. sfx_shoot_1.wav .. sfx_shoot_3.wav : Crisp pneumatic-liquid shooter fire (AI main agents)
  2. sfx_shoot_blitzcore.wav            : Deep pressurized BlitzCore mortar launch (AI main agents)
  3. sfx_ink_dive.wav                 : Viscous Minnaert-bubble plunge into ink
  4. sfx_splat_ko.wav                   : Pressurized ink-burst elimination splash
  5. sfx_shield_pop.wav                 : Crystalline barrier fracture + sub-bass ink detonation
  6. sfx_goal_touchdown.wav             : Pedestal lock-in slam + warm harmonic victory fanfare
"""

from __future__ import annotations

import math
import os
import wave
import numpy as np

SR = 48000


def biquad_filter(x: np.ndarray, mode: str, fc: float, q: float = 0.85, sr: int = SR) -> np.ndarray:
    """Zero-dependency Robert Bristow-Johnson Audio EQ Cookbook biquad filter."""
    w0 = 2.0 * math.pi * max(20.0, min(fc, sr * 0.46)) / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * max(0.15, q))

    if mode == "lpf":
        b0 = (1.0 - cos_w0) * 0.5
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) * 0.5
    elif mode == "hpf":
        b0 = (1.0 + cos_w0) * 0.5
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) * 0.5
    elif mode == "bpf":
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
    else:
        raise ValueError(f"Unknown filter mode: {mode}")

    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    y = np.zeros_like(x, dtype=np.float64)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for i in range(len(x)):
        xi = float(x[i])
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = xi
        y2 = y1
        y1 = yi
        y[i] = yi
    return y


def minnaert_bubble_layer(
    dur_s: float,
    n_bubbles: int,
    f_range: tuple[float, float],
    t_range: tuple[float, float],
    rng: np.random.Generator,
    chirp_strength: float = 11.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Physical-modeling liquid acoustics (Minnaert 1933 / van den Doel 2005):
    Real liquid splashes, ink shots, and submerge plunges owe their organic wet timbre
    to tiny entrained air cavities oscillating with a rapid upward pitch chirp as they detach.
    """
    n = int(round(dur_s * SR))
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    for _ in range(n_bubbles):
        t0 = float(rng.uniform(t_range[0], t_range[1]))
        i0 = int(round(t0 * SR))
        if i0 >= n - 16:
            continue
        f0 = float(np.exp(rng.uniform(math.log(f_range[0]), math.log(f_range[1]))))
        # Smaller high-pitch bubbles decay faster than larger low-pitch bubbles
        tau = float(np.clip(0.052 * (650.0 / f0) ** 0.68, 0.010, 0.075))
        b_len = min(n - i0, int(round(tau * 5.5 * SR)))
        tb = np.arange(b_len, dtype=np.float64) / SR

        # Upward detachment chirp: instantaneous frequency f(t) = f0 * (1 + c * t)
        chirp = chirp_strength * (f0 / 900.0) ** 0.35
        phase = 2.0 * math.pi * f0 * (tb + 0.5 * chirp * tb * tb)
        env = (1.0 - np.exp(-tb / 0.0018)) * np.exp(-tb / tau)
        amp = float(rng.uniform(0.35, 1.0)) * (750.0 / (f0 + 250.0)) ** 0.35
        sig = amp * env * np.sin(phase)

        pan = float(rng.uniform(-0.65, 0.65))
        gl = math.cos((pan + 1.0) * math.pi * 0.25)
        gr = math.sin((pan + 1.0) * math.pi * 0.25)
        left[i0 : i0 + b_len] += sig * gl
        right[i0 : i0 + b_len] += sig * gr

    return left, right


def pitch_sweep_sine(t: np.ndarray, f_start: float, f_end: float, k_exp: float = 18.0) -> np.ndarray:
    """Exact phase integral of exponential frequency sweep f(t) = f_end + (f_start - f_end)*exp(-k*t)."""
    phase = 2.0 * math.pi * (
        f_end * t + ((f_start - f_end) / max(1e-6, k_exp)) * (1.0 - np.exp(-k_exp * t))
    )
    return np.sin(phase)


def stereo_room_tail(left: np.ndarray, right: np.ndarray, wet: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
    """Clean multi-tap early-reflection + diffused stereo room ambience."""
    taps_l = [(int(0.013 * SR), 0.42), (int(0.029 * SR), 0.28), (int(0.047 * SR), 0.18), (int(0.079 * SR), 0.10)]
    taps_r = [(int(0.017 * SR), 0.42), (int(0.031 * SR), 0.28), (int(0.053 * SR), 0.18), (int(0.083 * SR), 0.10)]
    rl = np.zeros_like(left)
    rr = np.zeros_like(right)
    n = len(left)
    for d, g in taps_l:
        if d < n:
            rl[d:] += g * right[: n - d]
    for d, g in taps_r:
        if d < n:
            rr[d:] += g * left[: n - d]
    rl = biquad_filter(rl, "lpf", 3600.0, 0.7)
    rr = biquad_filter(rr, "lpf", 3600.0, 0.7)
    return left + wet * rl, right + wet * rr


def save_stereo_wav(path: str, left: np.ndarray, right: np.ndarray, peak_dbfs: float = -2.5) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    l_s = np.tanh(left * 1.15)
    r_s = np.tanh(right * 1.15)

    # Gentle fade-in (1.5ms) & fade-out (10ms) to guarantee zero DC/boundary click
    n = len(l_s)
    fi = min(n // 4, int(0.0015 * SR))
    fo = min(n // 3, int(0.012 * SR))
    if fi > 1:
        w_in = np.linspace(0.0, 1.0, fi)
        l_s[:fi] *= w_in
        r_s[:fi] *= w_in
    if fo > 1:
        w_out = np.linspace(1.0, 0.0, fo)
        l_s[-fo:] *= w_out
        r_s[-fo:] *= w_out

    peak = max(1e-6, float(max(np.max(np.abs(l_s)), np.max(np.abs(r_s)))))
    target = 10.0 ** (peak_dbfs / 20.0)
    scale = target / peak
    pcm_l = np.clip(np.round(l_s * scale * 32767.0), -32767, 32767).astype(np.int16)
    pcm_r = np.clip(np.round(r_s * scale * 32767.0), -32767, 32767).astype(np.int16)
    interleaved = np.empty(n * 2, dtype=np.int16)
    interleaved[0::2] = pcm_l
    interleaved[1::2] = pcm_r

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(interleaved.tobytes())


def synth_ink_shoot(seed: int, pitch_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Crisp, tactile pneumatic paintball pop + wet Minnaert droplet ejection (~0.125s)."""
    rng = np.random.default_rng(seed)
    dur = 0.125
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR

    # 1. Soft punchy nozzle pop membrane (warm, non-fatiguing)
    pop_env = (1.0 - np.exp(-t / 0.0012)) * np.exp(-t / 0.022)
    pop = 0.82 * pop_env * pitch_sweep_sine(t, 235.0 * pitch_scale, 68.0 * pitch_scale, k_exp=48.0)
    pop += 0.32 * np.exp(-t / 0.009) * pitch_sweep_sine(t, 540.0 * pitch_scale, 165.0 * pitch_scale, k_exp=75.0)

    # 2. Filtered hydrodynamic liquid spritz (bandpass + lowpass, zero harsh white-noise hiss)
    nz = rng.standard_normal(n)
    spritz = biquad_filter(nz, "bpf", 1450.0 * pitch_scale, q=1.25)
    spritz = biquad_filter(spritz, "lpf", 3100.0, q=0.8)
    spritz_env = (1.0 - np.exp(-t / 0.003)) * np.exp(-t / 0.036)
    spritz *= 0.55 * spritz_env

    # 3. 5 crisp liquid droplets detaching from nozzle
    bl, br = minnaert_bubble_layer(dur, 5, (520.0 * pitch_scale, 1680.0 * pitch_scale), (0.004, 0.058), rng, chirp_strength=15.0)

    l = pop + spritz + 0.62 * bl
    r = pop + spritz * 0.96 + 0.62 * br
    return l, r


def synth_blitzcore_shot(seed: int = 109) -> tuple[np.ndarray, np.ndarray]:
    """Deep pressurized BlitzCore mortar launch (~0.30s)."""
    rng = np.random.default_rng(seed)
    dur = 0.30
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR

    thump = 0.95 * np.exp(-t / 0.068) * pitch_sweep_sine(t, 260.0, 46.0, k_exp=22.0)
    vortex = 0.42 * np.exp(-t / 0.11) * np.sin(2.0 * math.pi * (180.0 * t + 110.0 * t * t))
    whoosh = biquad_filter(rng.standard_normal(n), "bpf", 920.0, q=1.1) * np.exp(-t / 0.085) * 0.60
    bl, br = minnaert_bubble_layer(dur, 12, (360.0, 1450.0), (0.008, 0.18), rng, chirp_strength=12.0)
    return stereo_room_tail(thump + vortex + whoosh + 0.65 * bl, thump + vortex + whoosh + 0.65 * br, wet=0.14)


def synth_ink_dive(seed: int = 202) -> tuple[np.ndarray, np.ndarray]:
    """Juicy viscous plunge into friendly ink (~0.25s)."""
    rng = np.random.default_rng(seed)
    dur = 0.25
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR

    # Downward liquid cavity displacement ("bloop-plunge")
    plunge_env = (1.0 - np.exp(-t / 0.006)) * np.exp(-t / 0.065)
    plunge = 0.68 * plunge_env * pitch_sweep_sine(t, 340.0, 108.0, k_exp=19.0)

    # Soft muffled underwater low-pass swoosh
    sw = biquad_filter(rng.standard_normal(n), "lpf", 950.0, q=1.35)
    sw = biquad_filter(sw, "hpf", 180.0, q=0.7) * (1.0 - np.exp(-t / 0.012)) * np.exp(-t / 0.075) * 0.48

    # Rising cascade of 15 Minnaert ink bubbles
    bl, br = minnaert_bubble_layer(dur, 15, (410.0, 1920.0), (0.010, 0.165), rng, chirp_strength=14.0)
    return stereo_room_tail(plunge + sw + 0.85 * bl, plunge + sw + 0.85 * br, wet=0.12)


def synth_splat_ko(seed: int = 303) -> tuple[np.ndarray, np.ndarray]:
    """Satisfying pressurized ink-burst elimination splash (~0.45s)."""
    rng = np.random.default_rng(seed)
    dur = 0.45
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR

    # 1. Pressurized balloon-pop sub impact + crisp wet snap
    impact = 0.95 * np.exp(-t / 0.055) * pitch_sweep_sine(t, 185.0, 40.0, k_exp=26.0)
    snap = biquad_filter(rng.standard_normal(n), "bpf", 1850.0, q=1.0) * np.exp(-t / 0.022) * 0.85
    body_splash = biquad_filter(rng.standard_normal(n), "lpf", 2400.0, q=0.9) * (1.0 - np.exp(-t / 0.008)) * np.exp(-t / 0.115) * 0.52

    # 2. Subtle expressive deflating whistle cue so KO stands out clearly in battle
    ko_tone_env = (1.0 - np.exp(-t / 0.010)) * np.exp(-t / 0.14)
    ko_tone = 0.28 * ko_tone_env * pitch_sweep_sine(t, 760.0, 195.0, k_exp=9.5)

    # 3. Rich 28-droplet outward ink scatter
    bl, br = minnaert_bubble_layer(dur, 28, (330.0, 2350.0), (0.012, 0.33), rng, chirp_strength=13.5)

    l = impact + snap + body_splash + ko_tone + 0.88 * bl
    r = impact + snap * 0.94 + body_splash + ko_tone + 0.88 * br
    return stereo_room_tail(l, r, wet=0.20)


def synth_shield_pop(seed: int = 404) -> tuple[np.ndarray, np.ndarray]:
    """Crystalline barrier fracture + massive pressurized ink shockwave (~0.84s)."""
    rng = np.random.default_rng(seed)
    dur = 0.84
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR

    # 1. Crystalline glass/barrier fracture cracks (t = 0, 16ms, 31ms)
    shatter_l = np.zeros(n, dtype=np.float64)
    shatter_r = np.zeros(n, dtype=np.float64)
    crystal_modes = [1380.0, 2090.0, 2870.0, 3910.0, 4860.0]
    for c_idx, t_delay in enumerate((0.0, 0.015, 0.029)):
        i0 = int(round(t_delay * SR))
        ln = n - i0
        tc = np.arange(ln, dtype=np.float64) / SR
        crack_nz = biquad_filter(rng.standard_normal(ln), "hpf", 2200.0, q=1.2) * np.exp(-tc / 0.014) * 0.55
        ring = np.zeros(ln, dtype=np.float64)
        for m_i, fm in enumerate(crystal_modes):
            f_jit = fm * (1.0 + 0.04 * c_idx)
            mod = 0.65 * np.exp(-tc / 0.025) * np.sin(2.0 * math.pi * (f_jit * 1.414) * tc)
            ring += (0.24 / (1.0 + 0.25 * m_i)) * np.exp(-tc / (0.055 - 0.006 * m_i)) * np.sin(2.0 * math.pi * f_jit * tc + mod)
        pan_sign = -1.0 if (c_idx % 2 == 0) else 1.0
        shatter_l[i0:] += crack_nz + ring * (1.0 + 0.2 * pan_sign)
        shatter_r[i0:] += crack_nz + ring * (1.0 - 0.2 * pan_sign)

    # 2. Deep shockwave detonation boom (starts right as barrier bursts at t = 24ms)
    boom_l = np.zeros(n, dtype=np.float64)
    i_b = int(round(0.024 * SR))
    tb = np.arange(n - i_b, dtype=np.float64) / SR
    sub_boom = 1.15 * (1.0 - np.exp(-tb / 0.004)) * np.exp(-tb / 0.19) * pitch_sweep_sine(tb, 132.0, 31.0, k_exp=11.0)
    mid_punch = 0.58 * np.exp(-tb / 0.085) * pitch_sweep_sine(tb, 310.0, 68.0, k_exp=18.0)
    wash = biquad_filter(rng.standard_normal(n - i_b), "lpf", 1900.0, q=0.85) * (1.0 - np.exp(-tb / 0.015)) * np.exp(-tb / 0.24) * 0.55
    boom_l[i_b:] = sub_boom + mid_punch + wash

    # 3. Torrential 46-bubble ink geyser eruption
    bl, br = minnaert_bubble_layer(dur, 46, (290.0, 2400.0), (0.028, 0.66), rng, chirp_strength=13.0)

    l = shatter_l + boom_l + 0.85 * bl
    r = shatter_r + boom_l + 0.85 * br
    return stereo_room_tail(l, r, wet=0.26)


def synth_goal_touchdown(seed: int = 505) -> tuple[np.ndarray, np.ndarray]:
    """Pedestal lock-in slam + warm modern triumphant harmonic fanfare (~1.55s)."""
    rng = np.random.default_rng(seed)
    dur = 1.55
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR

    # 1. Pedestal slam anchor impact + splash at t = 0
    slam = 0.85 * np.exp(-t / 0.095) * pitch_sweep_sine(t, 155.0, 44.0, k_exp=18.0)
    bl, br = minnaert_bubble_layer(dur, 18, (420.0, 1950.0), (0.005, 0.22), rng, chirp_strength=13.0)

    # 2. Uplifting warm brass-chime ascending fanfare: C5 -> G5 -> A5 -> C6+E6+G5+D6 (C major add9!)
    # Notes: (start_sec, dur_sec, freq_hz, gain, pan)
    notes = [
        (0.02, 0.22, 523.25, 0.42, -0.25),   # C5
        (0.13, 0.22, 783.99, 0.44,  0.25),   # G5
        (0.24, 0.24, 880.00, 0.46, -0.15),   # A5
        # Triumphant final chord at t = 0.36s (C4 bass + C5 + E5 + G5 + D6 + E6 shimmer)
        (0.36, 1.12, 261.63, 0.42,  0.00),   # C4 warm root
        (0.36, 1.10, 523.25, 0.36, -0.35),   # C5
        (0.36, 1.10, 659.25, 0.36,  0.35),   # E5
        (0.36, 1.10, 783.99, 0.34, -0.20),   # G5
        (0.36, 1.08, 1174.66, 0.24, 0.22),   # D6 (9th richness)
        (0.36, 1.08, 1318.51, 0.28, 0.00),   # E6 top lead
    ]

    mus_l = np.zeros(n, dtype=np.float64)
    mus_r = np.zeros(n, dtype=np.float64)

    for t0, ndur, freq, gain, pan in notes:
        i0 = int(round(t0 * SR))
        ln = min(n - i0, int(round(ndur * SR)))
        if ln <= 16:
            continue
        tn = np.arange(ln, dtype=np.float64) / SR
        env = (1.0 - np.exp(-tn / 0.008)) * np.exp(-tn / (ndur * 0.48))
        bell_env = (1.0 - np.exp(-tn / 0.002)) * np.exp(-tn / 0.14)

        # Warm stereo-detuned analog-style harmonic stack (fundamental + 2nd + 3rd + soft 4th) + bell attack
        vib = 0.0025 * np.sin(2.0 * math.pi * 5.6 * tn) * np.clip((tn - 0.12) / 0.25, 0.0, 1.0)
        p_l = 2.0 * math.pi * freq * 0.9985 * (tn + vib / (2.0 * math.pi * 5.6))
        p_r = 2.0 * math.pi * freq * 1.0015 * (tn + vib / (2.0 * math.pi * 5.6))

        tone_l = (
            np.sin(p_l)
            + 0.42 * np.sin(2.0 * p_l)
            + 0.20 * np.sin(3.0 * p_l) * np.exp(-tn / 0.35)
            + 0.08 * np.sin(4.0 * p_l) * np.exp(-tn / 0.20)
            + 0.26 * bell_env * np.sin(2.0 * p_l + 0.55 * bell_env * np.sin(5.0 * p_l))
        ) * env * gain

        tone_r = (
            np.sin(p_r)
            + 0.42 * np.sin(2.0 * p_r)
            + 0.20 * np.sin(3.0 * p_r) * np.exp(-tn / 0.35)
            + 0.08 * np.sin(4.0 * p_r) * np.exp(-tn / 0.20)
            + 0.26 * bell_env * np.sin(2.0 * p_r + 0.55 * bell_env * np.sin(5.0 * p_r))
        ) * env * gain

        gl = math.cos((pan + 1.0) * math.pi * 0.25)
        gr = math.sin((pan + 1.0) * math.pi * 0.25)
        mus_l[i0 : i0 + ln] += tone_l * (0.65 + 0.35 * gl)
        mus_r[i0 : i0 + ln] += tone_r * (0.65 + 0.35 * gr)

    mus_l = biquad_filter(mus_l, "lpf", 5200.0, q=0.75)
    mus_r = biquad_filter(mus_r, "lpf", 5200.0, q=0.75)

    l = slam + 0.45 * bl + mus_l
    r = slam + 0.45 * br + mus_r
    return stereo_room_tail(l, r, wet=0.30)


def _resolve_unity_audio_dir(base_dir: str) -> str:
    for entry in sorted(os.listdir(base_dir)):
        cand = os.path.join(base_dir, entry, "Assets", "Resources", "Audio")
        if os.path.isdir(os.path.join(base_dir, entry, "Assets")):
            return cand
    return os.path.join(base_dir, "UnityInkBlitzLab", "Assets", "Resources", "Audio")


def main() -> None:
    out_dir = _resolve_unity_audio_dir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)

    specs = [
        ("sfx_shoot_1.wav", *synth_ink_shoot(101, 1.00), -4.0),
        ("sfx_shoot_2.wav", *synth_ink_shoot(102, 1.05), -4.0),
        ("sfx_shoot_3.wav", *synth_ink_shoot(103, 0.95), -4.0),
        ("sfx_shoot_blitzcore.wav", *synth_blitzcore_shot(109), -2.8),
        ("sfx_ink_dive.wav", *synth_ink_dive(202), -3.2),
        ("sfx_splat_ko.wav", *synth_splat_ko(303), -2.0),
        ("sfx_shield_pop.wav", *synth_shield_pop(404), -1.5),
        ("sfx_goal_touchdown.wav", *synth_goal_touchdown(505), -1.8),
    ]

    for fname, l, r, dbfs in specs:
        full_p = os.path.join(out_dir, fname)
        save_stereo_wav(full_p, l, r, peak_dbfs=dbfs)
        dur_ms = len(l) * 1000.0 / SR
        print(f"[OK] {fname:24s} ({dur_ms:6.1f} ms, 48kHz Stereo 16-bit, peak={dbfs:+.1f} dBFS)")


if __name__ == "__main__":
    main()
