"""Ambience and sound cues, synthesized in code (no sound files to license).

A low drone and a breath of wind or room tone sit under the whole story, leaning darker
in tense scenes, and dip whenever the narrator speaks so every word stays clear. Simple
cues (a knock, a heartbeat, a phone, footsteps, rain, thunder, static, a drip, a door)
play when a scene's sound list asks for them.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

TENSE_MOODS = ("tense", "terror", "dread", "fear", "panic", "horror", "menac", "urgent")


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _smooth_noise(n: int, rate: int, cutoff_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Band-limited noise via FFT shaping (fast, no per-sample loop)."""
    spectrum = np.fft.rfft(_noise(n, rng))
    freqs = np.fft.rfftfreq(n, 1 / rate)
    spectrum *= 1 / (1 + (freqs / cutoff_hz) ** 2)
    out = np.fft.irfft(spectrum, n).astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-9)


def drone(samples: int, rate: int) -> np.ndarray:
    t = np.arange(samples) / rate
    wobble = 1 + 0.004 * np.sin(2 * np.pi * 0.07 * t)
    tone = (np.sin(2 * np.pi * 55 * t * wobble) + 0.6 * np.sin(2 * np.pi * 55.7 * t)
            + 0.3 * np.sin(2 * np.pi * 82.4 * t * wobble))
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * t / 23.0) ** 2
    return (tone * swell / 2.0).astype(np.float32)


def cue(name: str, rate: int, rng: np.random.Generator) -> np.ndarray:
    """A short synthesized sound for a cue word, or an empty array."""
    def env(n: int, attack: float, decay: float) -> np.ndarray:
        t = np.arange(n) / rate
        return (1 - np.exp(-t / max(attack, 1e-4))) * np.exp(-t / decay)

    if name == "knock":
        hit = _smooth_noise(int(0.12 * rate), rate, 400, rng) * env(int(0.12 * rate), 0.001, 0.03)
        gap = np.zeros(int(0.22 * rate), dtype=np.float32)
        return np.concatenate([hit, gap, hit, gap, hit]) * 0.9
    if name == "heartbeat":
        t = np.arange(int(0.18 * rate)) / rate
        beat = np.sin(2 * np.pi * 50 * t) * env(t.size, 0.004, 0.05)
        one = np.concatenate([beat, np.zeros(int(0.12 * rate)), beat * 0.7, np.zeros(int(0.55 * rate))])
        return np.tile(one, 4).astype(np.float32) * 0.9
    if name == "ring":
        t = np.arange(int(1.6 * rate)) / rate
        tone = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * 0.25
        gate = (np.sin(2 * np.pi * 20 * t) > 0) & (t < 1.0)
        return (tone * gate).astype(np.float32)
    if name == "footsteps":
        step = _smooth_noise(int(0.09 * rate), rate, 900, rng) * env(int(0.09 * rate), 0.002, 0.025)
        gap = np.zeros(int(0.45 * rate), dtype=np.float32)
        return np.concatenate([np.concatenate([step, gap]) for _ in range(5)]) * 0.6
    if name == "rain":
        return _smooth_noise(int(6 * rate), rate, 3000, rng) * 0.18
    if name == "wind":
        n = int(7 * rate)
        gust = 0.5 + 0.5 * np.sin(np.linspace(0, np.pi, n)) ** 2
        return _smooth_noise(n, rate, 350, rng) * gust.astype(np.float32) * 0.35
    if name == "thunder":
        n = int(3.5 * rate)
        return _smooth_noise(n, rate, 120, rng) * env(n, 0.05, 1.0).astype(np.float32) * 0.9
    if name == "static":
        n = int(1.5 * rate)
        return _noise(n, rng) * 0.08 * (rng.random(n) > 0.3).astype(np.float32)
    if name == "drip":
        t = np.arange(int(0.25 * rate)) / rate
        drop = np.sin(2 * np.pi * (900 + 1200 * np.exp(-t * 40)) * t) * env(t.size, 0.001, 0.04)
        return np.concatenate([drop, np.zeros(int(0.9 * rate)), drop * 0.8]).astype(np.float32) * 0.5
    if name == "creak":
        t = np.arange(int(1.2 * rate)) / rate
        f = 140 + 60 * np.sin(2 * np.pi * 1.3 * t)
        saw = 2 * ((np.cumsum(f) / rate) % 1) - 1
        return (saw * env(t.size, 0.1, 0.6) * 0.2).astype(np.float32)
    if name == "bell":
        t = np.arange(int(2.5 * rate)) / rate
        tone = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((523, 1046, 1568, 2093), 1))
        return (tone * env(t.size, 0.002, 0.8) * 0.3).astype(np.float32)
    return np.zeros(0, dtype=np.float32)


CUE_WORDS = {
    "knock": "knock", "heartbeat": "heartbeat", "heart": "heartbeat", "phone": "ring", "ring": "ring",
    "ringtone": "ring", "footstep": "footsteps", "steps": "footsteps", "rain": "rain", "wind": "wind",
    "thunder": "thunder", "static": "static", "radio": "static", "drip": "drip", "water": "drip",
    "creak": "creak", "door": "creak", "bell": "bell", "chime": "bell",
}


def cues_for(sfx: list[Any]) -> list[str]:
    found: list[str] = []
    for item in sfx or []:
        text = str(item).lower()
        for word, name in CUE_WORDS.items():
            if re.search(rf"\b{word}", text) and name not in found:
                found.append(name)
    return found[:2]


def mix(voice: np.ndarray, rate: int, scenes: list[dict[str, Any]], times: list[tuple[float, float]],
        speech: list[tuple[float, float]], seed: int = 1) -> np.ndarray:
    """Narration over the ambience and cues, with the bed ducked under speech."""
    rng = np.random.default_rng(seed)
    n = voice.size
    seconds = n / rate
    bed = 0.55 * drone(n, rate)
    air = _smooth_noise(n, rate, 500, rng) * 0.25
    level = np.full(n, 0.07, dtype=np.float32)
    for (t0, t1), scene in zip(times, scenes):
        mood = str(scene.get("mood", "")).lower()
        a, b = int(t0 * rate), min(int(t1 * rate), n)
        if any(m in mood for m in TENSE_MOODS):
            level[a:b] = 0.11
    duck = np.ones(n, dtype=np.float32)
    for t0, t1 in speech:
        duck[int(max(t0 - 0.15, 0) * rate): int(min(t1 + 0.2, seconds) * rate)] = 0.45
    kernel = np.ones(int(0.25 * rate), dtype=np.float32) / int(0.25 * rate)
    duck = np.convolve(duck, kernel, mode="same")
    level = np.convolve(level, kernel, mode="same")
    out = voice.astype(np.float32) + (bed + air) * level * duck
    for (t0, _), scene in zip(times, scenes):
        for name in cues_for(scene.get("sfx") or []):
            sound = cue(name, rate, rng)
            a = int((t0 + 0.2) * rate)
            b = min(a + sound.size, n)
            if b > a:
                out[a:b] += sound[: b - a] * 0.35 * duck[a:b]
    fade = int(1.5 * rate)
    out[:fade] *= np.linspace(0, 1, min(fade, n))[: out[:fade].size]
    out[-fade:] *= np.linspace(1, 0, min(fade, n))[: out[-fade:].size]
    peak = float(np.max(np.abs(out))) or 1.0
    return (out * min(1.0, 0.95 / peak)).astype(np.float32)
