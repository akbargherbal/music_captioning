# بحور الشعر — Arabic Poetic Meters Drum Generator

A command-line tool that synthesizes drum loops matched to the rhythmic character of the four most important Arabic poetic meters (*buhoor* — بحور). Each meter produces 2–3 MP3 files drawn from compatible Arabic percussion cycles.

---

## Background

Arabic poetry is built on a system of quantitative meters called *buhoor* (بحور, literally "seas"). Each bahr has a characteristic pattern of long (—) and short (∪) syllables that gives it a distinct rhythmic personality. This tool translates those prosodic patterns into drum grids — kick, snare, hihat, and crash — so you can hear, and produce with, the rhythmic soul of each meter.

The four meters covered represent roughly **75% of the classical Arabic poetic corpus**:

| Bahr | Arabic | Corpus share | Character |
|------|--------|-------------|-----------|
| Al-Taweel | الطويل | ~35% | Flowing, epic, iambic |
| Al-Kamil | الكامل | ~18% | Ascending, ternary, emotional |
| Al-Baseet | البسيط | ~12% | Heavy, declarative, spondaic |
| Al-Wafir | الوافر | ~10% | Rippling, lyrical, 6/8 |

---

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `pydub`
- `click`
- `ffmpeg` (required by pydub for MP3 export)

Install Python dependencies:

```bash
pip install numpy scipy pydub click
```

Install ffmpeg (if not already present):

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

---

## Usage

```bash
# Interactive menu — prompts you to choose a meter
python buhoor_drums.py

# One specific meter by name
python buhoor_drums.py taweel

# Multiple meters at once
python buhoor_drums.py kamil baseet

# All four meters
python buhoor_drums.py --all

# One meter, one variant only
python buhoor_drums.py taweel -v maqsum

# All meters, 30 s loops, quiet output
python buhoor_drums.py --all -d 30 -q

# All meters with a fresh random seed each run
python buhoor_drums.py --all --seed -1

# List all meters and variants without generating anything
python buhoor_drums.py --list
```

Valid meter names: `taweel`, `kamil`, `baseet`, `wafir`

---

## Options

```
Usage: buhoor_drums.py [OPTIONS] [BAHR]...

Options:
  --all                           Generate all buhoor
  -o, --output-dir DIRECTORY      Directory for output MP3 files
                                  [default: /mnt/user-data/outputs/buhoor]
                                  [env: BUHOOR_OUTPUT_DIR]
  -d, --duration SECONDS          Target loop duration in seconds
                                  [default: 10.0]
  -v, --variant NAME              Render only the named variant(s);
                                  repeatable: -v maqsum -v wahda
  --seed INTEGER                  Random seed for humanization;
                                  -1 picks a fresh seed each run [default: 42]
  --jitter SECONDS                Max ±timing jitter per hit [default: 0.007]
  --velocity-variance FLOAT       Max ±velocity nudge fraction
                                  (0 = robotic, 1 = chaotic) [default: 0.12]
  --bitrate [128k|192k|256k|320k] MP3 output bitrate [default: 192k]
  -q, --quiet                     Suppress descriptions and pattern grids
  -l, --list                      List available buhoor and variants, then exit
  -V, --version                   Show the version and exit
  -h, --help                      Show this message and exit
```

The output directory can also be set via the `BUHOOR_OUTPUT_DIR` environment variable, which the `--output-dir` flag overrides.

---

## Output

MP3 files are written to the output directory (default `/mnt/user-data/outputs/buhoor/`). Each file is named:

```
{meter}_{cycle}_{bpm}bpm.mp3
```

For example: `taweel_maqsum_90bpm.mp3`

Loop length is controlled by `--duration` (default 10 s); the script calculates the number of bars needed to reach that duration at each variant's BPM. ID3 tags are embedded with the meter name, BPM, time signature, and mood.

---

## The Meters and Their Cycles

### الطويل — Al-Taweel `(4/4)`
*Pattern: ∪—— | ∪———*

The king of Arabic meters. Its iambic (short→long) flow breathes like a long, unfolding sentence. Three cycles are provided:

| File | Cycle | BPM | Mood |
|------|-------|-----|------|
| `taweel_maqsum_90bpm.mp3` | Maqsum (مقسوم) | 90 | Narrative, dignified |
| `taweel_wahda_72bpm.mp3` | Wahda Kabeera (وحدة كبيرة) | 72 | Meditative, spacious |
| `taweel_fallahy_100bpm.mp3` | Fallahi (فلاحي) | 100 | Folk, earthy, energetic |

---

### الكامل — Al-Kamil `(3/4)`
*Pattern: ∪∪— | ∪∪— | ∪∪—*

"The complete." Its foot *mutafa'ilun* is ascending (anapestic): two short pickups rush forward into a long landing. The drum patterns honour this by placing the heavy kick on the **long syllable**, not the first short — preserving the anacrustic feel that defines Al-Kamil.

| File | Cycle | BPM | Mood |
|------|-------|-----|------|
| `kamil_samaai_darij_104bpm.mp3` | Samaai Darij (سماعي دارج) | 104 | Lyrical, elegant |
| `kamil_andalusi_flow_88bpm.mp3` | Andalusian 6/8 Flow (أندلسي) | 88 | Flowing, gentle |
| `kamil_muwashshah_syncopated_116bpm.mp3` | Muwashshah Syncopated (موشح متشابك) | 116 | Festive, celebratory |

> **Note on cycle names:** Two variants were previously mislabeled in an earlier version as "Jurjina" and "Dawr Hindi." Real Jurjina is a 10/8 cycle and real Dawr Hindi is a 7/8 cycle — neither fits a 12-step grid. The current names accurately describe what these patterns are.

---

### البسيط — Al-Baseet `(4/4)`
*Pattern: ——∪— | —∪—*

"The spread out." Opens with two consecutive long syllables (——) that land like a firm double-step, giving it declarative authority. It was the meter of satire, pride, and warrior poetry. The Masmoudi Kabir pattern captures this with a triple kick on beats 1, 2, and 3.

| File | Cycle | BPM | Mood |
|------|-------|-----|------|
| `baseet_masmoudi_kabir_100bpm.mp3` | Masmoudi Kabir (مصمودي كبير) | 100 | Heavy, powerful |
| `baseet_march_112bpm.mp3` | Askari March (مارش عسكري) | 112 | Martial, decisive |
| `baseet_zaffa_96bpm.mp3` | Zaffa (زفة) | 96 | Festive, ceremonial |

---

### الوافر — Al-Wafir `(6/8)`
*Pattern: ∪—∪∪— | ∪—∪∪— | ∪——*

"The abundant." Its foot *mufa'alatun* has a wave-like quality: a long crest, two quick adjacent ripples (∪∪), then another crest. The hihat patterns use **adjacent hits** to encode those two rapid syllables — the only way to represent the double-short in a step sequencer.

| File | Cycle | BPM | Mood |
|------|-------|-----|------|
| `wafir_muwashshah_80bpm.mp3` | Muwashshah (موشح أندلسي) | 80 | Elegant, Andalusian |
| `wafir_wafir_ripple_92bpm.mp3` | Wafir Ripple (وافر متموج) | 92 | Forward-moving, lyrical |

---

## Customisation

### Changing output location

```bash
# Via flag
python buhoor_drums.py --all -o ~/my_loops

# Via environment variable
export BUHOOR_OUTPUT_DIR=~/my_loops
python buhoor_drums.py --all
```

### Tweaking humanization

```bash
# Tighter timing, more velocity variation
python buhoor_drums.py --all --jitter 0.003 --velocity-variance 0.2

# Completely mechanical (no humanization)
python buhoor_drums.py --all --jitter 0 --velocity-variance 0

# Different result each run
python buhoor_drums.py --all --seed -1
```

### Adding a new variant

Append a new dictionary to the `"variants"` list of any meter in the `BUHOOR` dictionary, following the existing structure:

```python
{
    "name" : "my_variant",          # used in the filename and -v filter
    "label": "My Variant (label)",
    "bpm"  : 95,
    "mood" : "Description",
    "why"  : "Rationale for why this cycle suits the meter.",
    "patterns": {
        "kick" : [1,0,0,0, 0,0,0,0, 1,0,0,1, 0,0,0,0],  # 16 steps for 4/4
        "snare": [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,1],
        "hihat": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        "crash": [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
}
```

Step counts must match `steps_per_bar` for that meter (16 for 4/4 meters, 12 for 3/4 and 6/8).

Then render just your new variant:

```bash
python buhoor_drums.py taweel -v my_variant
```

---

## Technical Notes

**Drum synthesis** is physics-based, using no external samples:
- **Kick** — 808-style pitch sweep from 150 Hz → 50 Hz with exponential decay
- **Snare** — 200 Hz tonal body blended with filtered noise burst
- **Hihat** — White noise passed through a 7 kHz Butterworth high-pass filter
- **Crash** — Broadband noise through a 4 kHz high-pass with slow decay

**Humanisation** adds per-hit timing jitter (default ±7 ms) and velocity variance (default ±12%), both tunable via `--jitter` and `--velocity-variance`. The default seed is `42` for reproducible asset pipelines; pass `--seed -1` for a fresh seed each run.

**Panning** uses a constant-power law (`L = cos(p·π/2)`, `R = sin(p·π/2)`) to maintain equal perceived loudness across the stereo field on headphones.
