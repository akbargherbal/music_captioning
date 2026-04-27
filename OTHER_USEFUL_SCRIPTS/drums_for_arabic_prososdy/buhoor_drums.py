#!/usr/bin/env python3
"""
بحور الشعر — Arabic Poetic Meters Drum Generator
═══════════════════════════════════════════════════════════════
Generates drum loops reflecting the rhythmic soul of four
major Arabic poetic meters (بحور):

  الطويل  Al-Taweel  — فَعُولُن مَفَاعِيلُن  — 4/4  flowing / epic
  الكامل  Al-Kamil   — مُتَفَاعِلُن          — 3/4  ternary / waltz
  البسيط  Al-Baseet  — مُسْتَفْعِلُن فَاعِلُن — 4/4  march / declarative
  الوافر  Al-Wafir   — مُفَاعَلَتُن          — 6/8  rippling / lyrical

Each bahr produces 2-3 MP3 drum loops drawn from matching
Arabic rhythmic cycles (أوزان موسيقية).

Usage
─────
  python buhoor_drums.py                # interactive menu
  python buhoor_drums.py taweel         # one bahr by name
  python buhoor_drums.py kamil baseet   # multiple buhoor
  python buhoor_drums.py all            # everything
"""

import math
import numpy as np
import random
import os
import sys
import click
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from pydub import AudioSegment

# ─────────────────────────────────────────────────────────────
# Defaults — overridden at runtime by CLI options
SAMPLE_RATE       = 44100
OUTPUT_DIR        = "/mnt/user-data/outputs/buhoor"
TARGET_DURATION_S = 10.0

# ─────────────────────────────────────────────────────────────
#  DRUM SOUND SYNTHESIS
#  Same physics-based approach as the reference script.
#  kick = 808-style pitched sweep
#  snare = tonal body + noise burst
#  hihat = high-pass noise
#  crash = broadband noise, slow decay
# ─────────────────────────────────────────────────────────────

def _envelope(duration_s: float, attack: float = 0.002, decay_ratio: float = 0.8) -> np.ndarray:
    n        = int(SAMPLE_RATE * duration_s)
    env      = np.zeros(n)
    atk_n    = int(SAMPLE_RATE * attack)
    if atk_n:
        env[:atk_n] = np.linspace(0, 1, atk_n)
    dec_n    = n - atk_n
    env[atk_n:] = np.exp(-np.linspace(0, decay_ratio * 10, dec_n))
    return env


def synth_kick(duration: float = 0.50) -> np.ndarray:
    n   = int(SAMPLE_RATE * duration)
    t   = np.linspace(0, duration, n)
    f   = 150.0 * np.exp(np.log(50.0 / 150.0) * t / duration)
    ph  = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    w   = np.sin(ph) * _envelope(duration, 0.002, 0.6)
    return (w * 0.95).astype(np.float32)


def synth_snare(duration: float = 0.25) -> np.ndarray:
    n   = int(SAMPLE_RATE * duration)
    t   = np.linspace(0, duration, n)
    body = np.sin(2 * np.pi * 200 * t) * _envelope(duration, 0.001, 1.5) * 0.4
    noise = np.random.uniform(-1, 1, n) * _envelope(duration, 0.001, 2.5) * 0.6
    return ((body + noise) * 0.85).astype(np.float32)


def synth_hihat(duration: float = 0.08, open_hat: bool = False) -> np.ndarray:
    duration = 0.25 if open_hat else duration
    n        = int(SAMPLE_RATE * duration)
    noise    = np.random.uniform(-1, 1, n)
    sos      = butter(4, 7000, btype="high", fs=SAMPLE_RATE, output="sos")
    filt     = sosfilt(sos, noise)
    decay    = 1.2 if open_hat else 3.0
    return (filt * _envelope(duration, 0.001, decay) * 0.55).astype(np.float32)


def synth_crash(duration: float = 1.20) -> np.ndarray:
    n     = int(SAMPLE_RATE * duration)
    noise = np.random.uniform(-1, 1, n)
    sos   = butter(4, 4000, btype="high", fs=SAMPLE_RATE, output="sos")
    filt  = sosfilt(sos, noise)
    return (filt * _envelope(duration, 0.003, 0.5) * 0.45).astype(np.float32)


SYNTH_MAP = {
    "kick":  synth_kick,
    "snare": synth_snare,
    "hihat": synth_hihat,
    "crash": synth_crash,
}

# ─────────────────────────────────────────────────────────────
#  BUHOOR DEFINITIONS
#
#  Timing model
#  ────────────
#  step_duration = beats_per_bar × 60 / (bpm × steps_per_bar)
#
#  4/4  meters : beats_per_bar=4, steps_per_bar=16 (16th-note grid)
#               bpm = quarter-note tempo
#
#  3/4  meters : beats_per_bar=3, steps_per_bar=12 (16th-note grid)
#               bpm = quarter-note tempo
#
#  6/8  meters : beats_per_bar=2, steps_per_bar=12 (16th-note grid)
#               bpm = dotted-quarter tempo (the "felt" beat)
#               step = 60 / (bpm × 6)  ← 1/12 of a 6/8 bar
#
# ─────────────────────────────────────────────────────────────

BUHOOR: dict = {

    # ╔══════════════════════════════════════════════════════════╗
    # ║  الطويل — Al-Taweel                                     ║
    # ║  فَعُولُن مَفَاعِيلُن × 2                               ║
    # ║  Syllabic pattern :  ∪—— | ∪———                        ║
    # ║  Binary (iambic), 4/4, moderate — the "king of meters" ║
    # ║  ~35 % of classical Arabic corpus                       ║
    # ╚══════════════════════════════════════════════════════════╝
    "taweel": {
        "arabic"         : "الطويل",
        "taf_eela"       : "فَعُولُن مَفَاعِيلُن",
        "transliteration": "fa'oolun mafa'eelun",
        "syllable_pattern": "∪—— | ∪———",
        "time_signature" : (4, 4),
        "beats_per_bar"  : 4,
        "steps_per_bar"  : 16,
        "description": (
            "Al-Taweel is the undisputed king — the most used meter in Arabic poetry, "
            "covering roughly 35 % of the classical corpus. Its taf'eela 'fa'oolun mafa'eelun' "
            "creates an iambic (short→long) flow that breathes like a long sentence. "
            "Rhythmically it lives in 4/4 with a slight upbeat/anacrusis feel. "
            "Three Arabic cycles suit it perfectly: "
            "Maqsum (مقسوم) for storytelling, "
            "Wahda Kabeera (وحدة كبيرة) for meditation, "
            "Fallahi (فلاحي) for folk energy."
        ),
        "variants": [
            {
                "name" : "maqsum",
                "label": "Maqsum (مقسوم)",
                "bpm"  : 90,
                "mood" : "Narrative, dignified",
                "why"  : (
                    "Classic Arabic 4/4 doumbek cycle. The DUM on beat 1 anchors the "
                    "long syllable (—) of fa'oolun; the TEK fills the short (∪)."
                ),
                # 16 steps = 4 beats × 4 sixteenth notes
                # Doumbek: DUM . . . TEK . DUM TEK . . . TEK . . .
                "patterns": {
                    "kick" : [1,0,0,0,  0,0,0,0,  1,0,0,1,  0,0,0,0],
                    "snare": [0,0,0,0,  1,0,0,1,  0,0,0,0,  1,0,0,1],
                    "hihat": [1,0,1,0,  1,0,1,0,  1,0,1,0,  1,0,1,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
            {
                "name" : "wahda",
                "label": "Wahda Kabeera (وحدة كبيرة)",
                "bpm"  : 72,
                "mood" : "Meditative, spacious",
                "why"  : (
                    "Slow 4/4 with wide open space — the long vowels of mafa'eelun "
                    "need room to resonate. Perfect for elegiac or philosophical verse."
                ),
                "patterns": {
                    "kick" : [1,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,1,0],
                    "snare": [0,0,0,0,  0,0,0,0,  1,0,0,0,  0,0,0,0],
                    "hihat": [1,0,0,0,  1,0,0,0,  1,0,0,0,  1,0,0,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
            {
                "name" : "fallahy",
                "label": "Fallahi (فلاحي)",
                "bpm"  : 100,
                "mood" : "Folk, earthy, energetic",
                "why"  : (
                    "Rustic 4/4 with syncopation — used in zajal and Levantine folk "
                    "poetry recitation. The cross-rhythm hihat mirrors the iambic push."
                ),
                "patterns": {
                    "kick" : [1,0,0,0,  1,0,0,1,  0,0,0,0,  1,0,0,0],
                    "snare": [0,0,0,0,  0,0,1,0,  0,0,1,0,  0,0,1,0],
                    "hihat": [1,1,0,1,  1,0,1,1,  0,1,1,0,  1,1,0,1],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
        ],
    },

    # ╔══════════════════════════════════════════════════════════╗
    # ║  الكامل — Al-Kamil                                      ║
    # ║  مُتَفَاعِلُن × 3                                        ║
    # ║  Syllabic pattern : ∪∪— | ∪∪— | ∪∪—                   ║
    # ║  Ternary (ascending), 3/4 — "the complete"             ║
    # ║  ~18 % of corpus; favored for ghazal & love verse      ║
    # ╚══════════════════════════════════════════════════════════╝
    "kamil": {
        "arabic"          : "الكامل",
        "taf_eela"        : "مُتَفَاعِلُن",
        "transliteration" : "mutafa'ilun",
        "syllable_pattern": "∪∪— | ∪∪— | ∪∪—",
        "time_signature"  : (3, 4),
        "beats_per_bar"   : 3,
        "steps_per_bar"   : 12,    # 4 sixteenth notes × 3 beats
        "description": (
            "Al-Kamil — 'the complete' — is the second most used meter. "
            "Its taf'eela mutafa'ilun (two shorts then long: ∪∪—) is unmistakably ternary — "
            "like a waltz ascending. The cell ∪∪ rushes forward into the landing (—). "
            "It carries warmth, emotion, and a dancing quality, hence its dominance "
            "in love poetry. In music it belongs to 3/4 or 6/8. "
            "Samaai (سماعي) and Jurjina (جورجينا) are its natural cycles. "
            "[BPM here = quarter-note beat; 3 per bar]"
        ),
        "variants": [
            {
                "name" : "samaai_darij",
                "label": "Samaai Darij (سماعي دارج)",
                "bpm"  : 104,
                "mood" : "Lyrical, elegant, danceable",
                "why"  : (
                    "3/4 cycle — mutafa'ilun (∪∪—) is ascending/anapestic: two short "
                    "pickup syllables rush into the long landing. The DUM therefore "
                    "falls on step 3 (the '—'), not step 1, preserving the anacrustic "
                    "feel that classical prosodists hear in Al-Kamil."
                ),
                # 12 steps = 3 beats × 4 sixteenth notes
                # ∪∪— cell: steps 1-2 are the two shorts, step 3 is the DUM landing
                "patterns": {
                    "kick" : [0,0,1,0,  0,0,0,0,  0,0,1,0],
                    "snare": [0,0,0,0,  0,0,1,0,  0,0,0,0],
                    "hihat": [1,0,1,0,  1,0,1,0,  1,0,1,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
            {
                "name" : "andalusi_flow",
                "label": "Andalusian 6/8 Flow (أندلسي)",
                "bpm"  : 88,
                "mood" : "Flowing, Andalusian, gentle",
                "why"  : (
                    "NOTE: Previously mislabeled 'Jurjina'. Real Jurjina is a strict "
                    "10/8 cycle and cannot fit a 12-step grid. This is an original "
                    "6/8-inflected pattern inspired by Andalusian muwashshah style. "
                    "The DUM lands on step 3 (the long '—' of ∪∪—), keeping the "
                    "ascending anapestic feel of Al-Kamil intact."
                ),
                "patterns": {
                    "kick" : [0,0,1,0,  0,0,0,0,  0,0,1,0],
                    "snare": [0,0,1,0,  0,0,0,0,  1,0,0,0],
                    "hihat": [1,0,0,1,  0,0,1,0,  0,1,0,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
            {
                "name" : "muwashshah_syncopated",
                "label": "Muwashshah Syncopated (موشح متشابك)",
                "bpm"  : 116,
                "mood" : "Upbeat, festive, celebratory",
                "why"  : (
                    "NOTE: Previously mislabeled 'Dawr Hindi'. Real Dawr Hindi is a "
                    "strict 7/8 cycle (3+2+2) with no 3/4 equivalent in Arabic practice. "
                    "This is an original syncopated muwashshah-flavored 3/4 pattern. "
                    "Common in tarab gatherings."
                ),
                "patterns": {
                    "kick" : [0,0,1,1,  0,0,0,0,  1,0,0,0],
                    "snare": [0,0,1,0,  0,1,0,0,  0,1,0,0],
                    "hihat": [1,1,0,1,  1,0,1,1,  0,1,1,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
        ],
    },

    # ╔══════════════════════════════════════════════════════════╗
    # ║  البسيط — Al-Baseet                                     ║
    # ║  مُسْتَفْعِلُن فَاعِلُن × 2                              ║
    # ║  Syllabic pattern : ——∪— | —∪—                         ║
    # ║  Binary (heavy), 4/4 — authoritative, declarative      ║
    # ║  ~12 % of corpus; satire, pride, complaint              ║
    # ╚══════════════════════════════════════════════════════════╝
    "baseet": {
        "arabic"          : "البسيط",
        "taf_eela"        : "مُسْتَفْعِلُن فَاعِلُن",
        "transliteration" : "mustaf'ilun fa'ilun",
        "syllable_pattern": "——∪— | —∪—",
        "time_signature"  : (4, 4),
        "beats_per_bar"   : 4,
        "steps_per_bar"   : 16,
        "description": (
            "Al-Baseet — 'the spread out' — opens with two consecutive long syllables (——) "
            "that land like a firm step. This immediate weight gives it authority: "
            "it was the meter of satire, complaint, pride, and declaration. "
            "Unlike Al-Taweel's flowing iamb, Al-Baseet plants its feet first "
            "and then moves. Masmoudi Kabir (المصمودي الكبير) — the heaviest Arabic "
            "4/4 cycle — mirrors this perfectly. Also suited to military march and Zaffa."
        ),
        "variants": [
            {
                "name" : "masmoudi_kabir",
                "label": "Masmoudi Kabir (مصمودي كبير)",
                "bpm"  : 100,
                "mood" : "Heavy, powerful, declarative",
                "why"  : (
                    "The heaviest Arabic 4/4 cycle. Triple DUM on beats 1+2+3 (steps 1, 5, 9) "
                    "mirrors the spondaic weight of mustaf'ilun's opening ——. "
                    "NOTE: Comment previously said 'double DUM on beats 1+2' but the "
                    "pattern places kicks on beats 1, 2 AND 3 — corrected to triple DUM. "
                    "Used in classical Egyptian tarab."
                ),
                # 16 steps — triple-kick on beats 1, 2, 3; pickup on beat 4 step 15
                # Expert note: the ——  opening of Baseet is correctly captured by
                "patterns": {
                    "kick" : [1,0,0,0,  1,0,0,0,  1,0,0,0,  0,0,1,0],
                    "snare": [0,0,0,0,  0,0,1,0,  0,0,0,0,  1,0,0,0],
                    "hihat": [1,0,1,0,  1,0,1,0,  1,0,1,0,  1,0,1,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
            {
                "name" : "march",
                "label": "Askari March (مارش عسكري)",
                "bpm"  : 112,
                "mood" : "Martial, decisive, strict",
                "why"  : (
                    "The —— opening of mustaf'ilun is literally a left-right march step. "
                    "Arabs used Al-Baseet in warrior poetry (حماسة) — this cycle "
                    "brings that energy to life."
                ),
                "patterns": {
                    "kick" : [1,0,1,0,  0,0,0,0,  1,0,1,0,  0,0,0,0],
                    "snare": [0,0,0,0,  1,0,0,0,  0,0,0,0,  1,0,0,0],
                    "hihat": [1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
            {
                "name" : "zaffa",
                "label": "Zaffa (زفة) — Processional",
                "bpm"  : 96,
                "mood" : "Festive, ceremonial, proud",
                "why"  : (
                    "Wedding processional. Al-Baseet's declarative character suits "
                    "pride and celebration — the Zaffa cycle adds ceremony and swing."
                ),
                "patterns": {
                    "kick" : [1,0,0,1,  0,0,1,0,  1,0,0,0,  0,1,0,0],
                    "snare": [0,0,1,0,  1,0,0,0,  0,1,0,0,  1,0,0,1],
                    "hihat": [1,0,1,1,  0,1,1,0,  1,0,1,1,  0,1,1,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
        ],
    },

    # ╔══════════════════════════════════════════════════════════╗
    # ║  الوافر — Al-Wafir                                      ║
    # ║  مُفَاعَلَتُن × 2 + فَعُولُن                             ║
    # ║  Syllabic pattern : ∪—∪∪— | ∪—∪∪— | ∪——              ║
    # ║  Ternary-hybrid, 6/8 — rippling, abundant, lyrical     ║
    # ║  ~10 % of corpus; ghazal, love, longing                ║
    # ╚══════════════════════════════════════════════════════════╝
    "wafir": {
        "arabic"          : "الوافر",
        "taf_eela"        : "مُفَاعَلَتُن",
        "transliteration" : "mufa'alatun",
        "syllable_pattern": "∪—∪∪— | ∪—∪∪— | ∪——",
        "time_signature"  : (6, 8),
        "beats_per_bar"   : 2,    # 2 dotted-quarter pulses per bar
        "steps_per_bar"   : 12,   # 6 eighth notes × 2 sixteenth subdivisions
        "description": (
            "Al-Wafir — 'the abundant / rippling' — has an undulating wave-like quality. "
            "Mufa'alatun (∪—∪∪—) rolls forward: long crest, then two quick ripples, "
            "then another crest. It's lush and lyrical — perfect for longing and love. "
            "In 6/8 the meter fits naturally: the two short syllables (∪∪) act as "
            "a triplet pickup rushing into the next long. "
            "Jurjina (جورجينا) and Muwashshah (موشح) cycles are its natural home. "
            "[BPM here = dotted-quarter note; 2 per bar]"
        ),
        "variants": [
            {
                "name" : "muwashshah",
                "label": "Muwashshah (موشح أندلسي)",
                "bpm"  : 80,
                "mood" : "Elegant, Andalusian, flowing",
                "why"  : (
                    "Andalusian muwashshah style, slow 6/8. The hihat traces every "
                    "dotted-eighth pulse, matching the ∪—∪∪— undulation bar by bar."
                ),
                # 12 steps : strong dotted-quarter pulses on step 1 and step 7
                # ∪—∪∪— encoding: steps 1(∪), 2-3(—), 4+5 adjacent(∪∪), 6(—) per pulse
                "patterns": {
                    "kick" : [1,0,0,0,  0,0,1,0,  0,0,0,0],
                    "snare": [0,0,0,0,  1,0,0,0,  0,1,0,0],
                    "hihat": [1,0,0,1,1,0, 1,0,0,1,1,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
            {
                "name" : "wafir_ripple",
                "label": "Wafir Ripple (وافر متموج)",
                "bpm"  : 92,
                "mood" : "Forward-moving, rippling, lyrical",
                "why"  : (
                    "NOTE: Previously mislabeled 'Jurjina'. Real Jurjina is 10/8 and "
                    "cannot fit this 12-step grid. This is an original pattern. "
                    "Hihat now uses adjacent hits at steps 4+5 and 10+11 — "
                    "the only way to encode the actual ∪∪ double-short in a step grid, "
                    "as equidistant pulses are mathematically incapable of representing "
                    "the asymmetric fāṣila ṣughrā (فاصلة صغرى) of Al-Wafir."
                ),
                "patterns": {
                    "kick" : [1,0,0,0,  0,1,0,0,  0,0,0,0],
                    "snare": [0,0,1,0,  0,0,0,1,  0,0,1,0],
                    "hihat": [1,0,0,1,1,0, 1,0,0,1,1,0],
                    "crash": [1,0,0,0,  0,0,0,0,  0,0,0,0],
                },
            },
        ],
    },
}

# ─────────────────────────────────────────────────────────────
#  HUMANIZATION
#  Subtle timing jitter + velocity variance → natural feel.
#  Arabic percussion is organic; strict quantization sounds wrong.
# ─────────────────────────────────────────────────────────────

def humanize(
    pattern: list,
    timing_jitter: float = 0.007,
    velocity_variance: float = 0.12,
) -> list[tuple]:
    """Return [(step_idx, time_offset_s, velocity_mul), …] for each hit.

    Args:
        timing_jitter: Max ±seconds of per-hit timing offset (default 0.007).
        velocity_variance: Max ±fraction of per-hit velocity nudge (default 0.12).
    """
    hits = []
    for i, hit in enumerate(pattern):
        if hit:
            t_off = random.uniform(-timing_jitter, timing_jitter)
            vel   = 1.0 + random.uniform(-velocity_variance, velocity_variance)
            vel   = max(0.4, min(1.4, vel))
            hits.append((i, t_off, vel))
    return hits


# ─────────────────────────────────────────────────────────────
#  RENDER
# ─────────────────────────────────────────────────────────────

def render_variant(
    bahr_key: str,
    variant: dict,
    bars: int = 4,
    *,
    output_dir: str = OUTPUT_DIR,
    bitrate: str = "192k",
    jitter: float = 0.007,
    velocity_variance: float = 0.12,
) -> str:
    """Synthesize one rhythmic variant → MP3. Returns the output path."""
    bahr          = BUHOOR[bahr_key]
    steps_per_bar = bahr["steps_per_bar"]
    beats_per_bar = bahr["beats_per_bar"]
    bpm           = variant["bpm"]

    # Derived timing
    step_s      = beats_per_bar * 60.0 / (bpm * steps_per_bar)
    total_steps = steps_per_bar * bars
    total_n     = int(total_steps * step_s * SAMPLE_RATE) + SAMPLE_RATE  # +1 s tail

    mix = np.zeros((total_n, 2), dtype=np.float32)

    sounds = {name: fn() for name, fn in SYNTH_MAP.items()}

    # Constant-power panning law: L = cos(p·π/2), R = sin(p·π/2)
    # Ensures L²+R²=1 regardless of position, preventing the 12 dB headroom
    # imbalance that linear amplitude scaling (e.g. 0.25 / 1.00) would cause.
    def _cpan(p: float) -> tuple:
        a = p * (math.pi / 2)
        return math.cos(a), math.sin(a)

    panning = {
        "kick" : _cpan(0.50),   # centre       → L≈0.707  R≈0.707
        "snare": _cpan(0.50),   # centre       → L≈0.707  R≈0.707
        "hihat": _cpan(0.75),   # 75 % right   → L≈0.383  R≈0.924  (~7.6 dB diff)
        "crash": _cpan(0.25),   # 75 % left    → L≈0.924  R≈0.383
    }

    for inst, base_pat in variant["patterns"].items():
        sound            = sounds[inst]
        pan_l, pan_r     = panning.get(inst, (0.75, 0.75))
        full_pat         = base_pat * bars
        hits             = humanize(full_pat, timing_jitter=jitter,
                                    velocity_variance=velocity_variance)

        for step_idx, t_off, vel in hits:
            t_sec = step_idx * step_s + t_off
            start = max(0, int(t_sec * SAMPLE_RATE))
            end   = min(start + len(sound), total_n)
            n     = end - start
            if n > 0:
                mix[start:end, 0] += sound[:n] * pan_l * vel
                mix[start:end, 1] += sound[:n] * pan_r * vel

    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.9
    pcm = (mix * 32767).astype(np.int16)

    ts     = bahr["time_signature"]
    fname  = f"{bahr_key}_{variant['name']}_{bpm}bpm"
    wav_p  = f"/tmp/{fname}.wav"
    mp3_p  = os.path.join(output_dir, fname + ".mp3")

    wavfile.write(wav_p, SAMPLE_RATE, pcm)
    audio = AudioSegment.from_wav(wav_p)
    audio.export(
        mp3_p,
        format="mp3",
        bitrate=bitrate,
        tags={
            "title"  : f"{bahr['arabic']} — {variant['label']}",
            "artist" : "Buhoor Drum Generator",
            "album"  : "Arabic Poetic Meters / بحور الشعر",
            "comment": f"{bpm} BPM | {bars} bars | {ts[0]}/{ts[1]} | {variant['mood']}",
        },
    )
    os.remove(wav_p)
    return mp3_p


# ─────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 70, indent: str = "  ") -> str:
    words, line, lines = text.split(), indent, []
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = indent + w
        else:
            line += (" " if line.strip() else "") + w
    lines.append(line)
    return "\n".join(lines)


def print_header():
    print("\n╔" + "═" * 56 + "╗")
    print("║" + "   بحور الشعر — Arabic Poetic Meters Drum Gen   ".center(56) + "║")
    print("╚" + "═" * 56 + "╝\n")


def print_bahr_header(key: str):
    b  = BUHOOR[key]
    ts = b["time_signature"]
    print(f"\n{'═'*60}")
    print(f"  {b['arabic']}  —  Al-{key.title()}")
    print(f"  تفعيلة : {b['taf_eela']}")
    print(f"  Pattern: {b['syllable_pattern']}")
    print(f"  Meter  : {ts[0]}/{ts[1]}")
    print(f"{'═'*60}")
    print(_wrap(b["description"]))
    print()


def print_pattern_grid(variant: dict):
    print(f"  ▶ {variant['label']}  │  {variant['bpm']} BPM  │  {variant['mood']}")
    print(f"    Rationale: {variant['why']}")
    print(f"    {'─'*50}")
    for inst in ("kick", "snare", "hihat", "crash"):
        if inst not in variant["patterns"]:
            continue
        steps   = variant["patterns"][inst]
        grid    = "".join("█" if s else "·" for s in steps)
        n_hits  = sum(steps)
        print(f"    {inst:<8}│ {grid}  ({n_hits})")
    print()


# ─────────────────────────────────────────────────────────────
#  GENERATE ONE BAHR
# ─────────────────────────────────────────────────────────────

def generate_bahr(
    key: str,
    target_s: float = TARGET_DURATION_S,
    *,
    variant_filter: tuple[str, ...] = (),
    output_dir: str = OUTPUT_DIR,
    bitrate: str = "192k",
    jitter: float = 0.007,
    velocity_variance: float = 0.12,
    quiet: bool = False,
) -> list[str]:
    """Render all (or selected) variants for one bahr. Returns list of MP3 paths."""
    if not quiet:
        print_bahr_header(key)
    bahr  = BUHOOR[key]
    paths = []
    for variant in bahr["variants"]:
        if variant_filter and variant["name"] not in variant_filter:
            continue
        bar_s = bahr["beats_per_bar"] * 60.0 / variant["bpm"]
        bars  = math.ceil(target_s / bar_s)
        if not quiet:
            print_pattern_grid(variant)
        path    = render_variant(
            key, variant, bars,
            output_dir=output_dir,
            bitrate=bitrate,
            jitter=jitter,
            velocity_variance=velocity_variance,
        )
        size_kb = os.path.getsize(path) // 1024
        click.echo(f"    ✅  {os.path.basename(path)}  ({size_kb} KB)\n")
        paths.append(path)
    return paths


# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────

def interactive_menu() -> list[str]:
    print_header()
    click.echo("  Select a bahr to generate drum loops for:\n")
    keys = list(BUHOOR.keys())
    for i, k in enumerate(keys, 1):
        b  = BUHOOR[k]
        ts = b["time_signature"]
        nv = len(b["variants"])
        click.echo(f"  [{i}] {b['arabic']:<14}  Al-{k.title():<10}  "
                   f"{ts[0]}/{ts[1]}  —  {nv} variants")
    click.echo(f"  [{len(keys)+1}] All buhoor\n")

    while True:
        raw = click.prompt("  Your choice", prompt_suffix="").strip()
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(keys):
                return [keys[n - 1]]
            if n == len(keys) + 1:
                return keys
        elif raw.lower() in keys:
            return [raw.lower()]
        elif raw.lower() == "all":
            return keys
        click.echo("  Please enter a valid number or bahr name.")


# ─────────────────────────────────────────────────────────────
#  VALID BAHR NAMES (for click validation)
# ─────────────────────────────────────────────────────────────

BAHR_NAMES = list(BUHOOR.keys())


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "buhoor",
    nargs=-1,
    metavar="[BAHR]...",
)
@click.option(
    "--all", "select_all",
    is_flag=True,
    help="Generate all buhoor (same as passing every name).",
)
@click.option(
    "-o", "--output-dir",
    default=OUTPUT_DIR,
    show_default=True,
    envvar="BUHOOR_OUTPUT_DIR",
    help="Directory for output MP3 files.",
    type=click.Path(file_okay=False),
)
@click.option(
    "-d", "--duration",
    default=TARGET_DURATION_S,
    show_default=True,
    type=click.FloatRange(min=1.0),
    metavar="SECONDS",
    help="Target loop duration in seconds.",
)
@click.option(
    "-v", "--variant",
    "variant_filter",
    multiple=True,
    metavar="NAME",
    help=(
        "Render only the named variant(s). "
        "Repeatable: -v maqsum -v wahda. "
        "Default: all variants."
    ),
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for humanization (use -1 for a random seed).",
)
@click.option(
    "--jitter",
    default=0.007,
    show_default=True,
    type=click.FloatRange(min=0.0, max=0.05),
    metavar="SECONDS",
    help="Max ±timing jitter per hit in seconds.",
)
@click.option(
    "--velocity-variance",
    default=0.12,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Max ±velocity nudge fraction (0 = robotic, 1 = chaotic).",
)
@click.option(
    "--bitrate",
    default="192k",
    show_default=True,
    type=click.Choice(["128k", "192k", "256k", "320k"], case_sensitive=False),
    help="MP3 output bitrate.",
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Suppress descriptions and pattern grids; show only file output.",
)
@click.option(
    "-l", "--list",
    "list_buhoor",
    is_flag=True,
    help="List available buhoor (and their variants) then exit.",
)
@click.version_option("2.0.0", "-V", "--version")
def main(
    buhoor,
    select_all,
    output_dir,
    duration,
    variant_filter,
    seed,
    jitter,
    velocity_variance,
    bitrate,
    quiet,
    list_buhoor,
):
    """بحور الشعر — Arabic Poetic Meters Drum Generator

    \b
    Generate drum loops for one or more Arabic poetic meters (بحور):
      taweel   الطويل   4/4   epic / flowing
      kamil    الكامل   3/4   lyrical / waltz
      baseet   البسيط   4/4   march / declarative
      wafir    الوافر   6/8   rippling / lyrical

    \b
    Examples:
      python buhoor_drums.py                         # interactive menu
      python buhoor_drums.py taweel                  # one bahr
      python buhoor_drums.py kamil baseet            # two buhoor
      python buhoor_drums.py --all                   # everything
      python buhoor_drums.py taweel -v maqsum        # one variant only
      python buhoor_drums.py --all -d 30 -q          # 30 s, quiet
      python buhoor_drums.py --all --seed -1         # random humanization
    """
    # ── --list: just show the catalogue and exit ──────────────
    if list_buhoor:
        print_header()
        for k in BUHOOR:
            b  = BUHOOR[k]
            ts = b["time_signature"]
            click.echo(f"  {b['arabic']:<14}  al-{k:<10}  {ts[0]}/{ts[1]}")
            for v in b["variants"]:
                click.echo(f"      • {v['name']:<28}  {v['bpm']} BPM  {v['mood']}")
            click.echo()
        return

    # ── Validate BAHR arguments ───────────────────────────────
    bad = [a for a in buhoor if a not in BUHOOR]
    if bad:
        raise click.BadArgumentUsage(
            f"Unknown bahr: {', '.join(bad)}. "
            f"Valid names: {', '.join(BUHOOR)} (or use --all)."
        )

    # ── Seed RNGs ─────────────────────────────────────────────
    effective_seed = random.randint(0, 2**31) if seed == -1 else seed
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    if not quiet and seed == -1:
        click.echo(f"  Random seed: {effective_seed}\n")

    # ── Select buhoor ─────────────────────────────────────────
    if select_all or (not buhoor and not list_buhoor):
        if not buhoor:
            # No args and no --all → interactive menu
            if not select_all:
                selected = interactive_menu()
            else:
                selected = BAHR_NAMES
        else:
            selected = BAHR_NAMES
    else:
        selected = list(buhoor)

    # ── Create output dir ─────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    if not quiet:
        click.echo(f"\n  Synthesizing ~{duration:.0f} s per variant "
                   f"(seed={effective_seed}) …\n")

    # ── Render ────────────────────────────────────────────────
    all_paths: list[str] = []
    for key in selected:
        all_paths.extend(
            generate_bahr(
                key,
                target_s=duration,
                variant_filter=variant_filter,
                output_dir=output_dir,
                bitrate=bitrate,
                jitter=jitter,
                velocity_variance=velocity_variance,
                quiet=quiet,
            )
        )

    # ── Summary ───────────────────────────────────────────────
    click.echo(f"\n{'═'*60}")
    click.echo(f"  Output directory : {output_dir}")
    click.echo(f"  Total files      : {len(all_paths)}")
    for p in all_paths:
        size_kb = os.path.getsize(p) // 1024
        click.echo(f"    • {os.path.basename(p):50s}  ({size_kb} KB)")
    click.echo()


if __name__ == "__main__":
    main()
