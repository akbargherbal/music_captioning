# Quick Start

This guide is for someone who has never used this project. You do not need to
know the model, the code, or Suno. Follow the steps in order.

**What you need:** a web browser and a Google account. The free Colab T4 GPU is
enough.

**Time:** about 10 minutes the first time (mostly the ~6.4 GB model download).
After setup, a 5-minute song takes about 70 seconds of captioning, plus roughly
35 seconds to load the model for each `demo.py` run.

---

## 1. What this project does

You give it one audio file. It gives you back:

1. a **prose description** of the track, and
2. three lists of **terms** found in that description:
   - `instruments`
   - `vocals`
   - `production`

```
your song ─▶ captioner model ─▶ prose caption ─▶ term lists
```

The captioner is `ACE-Step/acestep-captioner` (MIT), based on Qwen2.5-Omni-7B.
It runs in 4-bit mode so it fits on a free T4 GPU. The term lists are plain
Python string matching — no second AI model, no API key.

### What you get (example)

```json
{
  "caption": "An emotional Arabic pop ballad driven by a powerful female vocal performance. The arrangement features a prominent oud ...",
  "instruments": ["oud", "synth strings", "darbuka", "synth bass"],
  "vocals": ["female vocal", "vibrato"],
  "production": ["cinematic", "reverb", "spacious"],
  "tags_experimental": true
}
```

### What this is **not**

- **Not a music generator.** It describes music; it does not create it.
- **Not a finished Suno prompt.** The term lists are marked `tags_experimental`
  on purpose. They omit genre and mood, can miss plurals, and can mix up
  details. Read [`TAG_AUDIT.md`](TAG_AUDIT.md) before using them as a prompt.
  The old comma-joined string still exists, but only if you ask for it with
  `--experimental-suno`, and it is not supported.
- **Not a lyrics transcriber.** A separate transcription pipeline is described
  in [`TRANSCRIPTION.md`](TRANSCRIPTION.md).

---

## 2. Before you start

Check these first:

| Requirement | Why |
|---|---|
| Google Colab with **T4 GPU** | The model needs a CUDA GPU (~8 GiB VRAM peak). |
| `ffmpeg` and `ffprobe` | Used to read audio and split long tracks. Colab has them. |
| Python 3.10+ | The code uses modern type hints. |
| ~7 GB free disk | ~6.4 GB model + audio chunks. |
| A song file | Anything `ffmpeg` can read: MP3, WAV, FLAC, M4A, OGG. |

You do **not** need a Hugging Face token. Both model repositories are public.
You do **not** need to install PyTorch on Colab — it is already there.

> **CPU only?** You can run the tests on CPU, but captioning on CPU is not
> practical. Use the T4 runtime.

---

## 3. The 10-minute Colab path

### Step 1 — Open a notebook with a GPU

1. Go to <https://colab.research.google.com>.
2. Create a new notebook (or open `notebooks/colab_setup.ipynb` from the repo).
3. Menu: **Runtime → Change runtime type → T4 GPU → Save**.
4. Run this to confirm the GPU:

```python
!nvidia-smi
```

You should see `Tesla T4` and about `15360 MiB`. If you see an error, the runtime
is still CPU-only; repeat the runtime-type step.

### Step 2 — Get the code and dependencies

Paste this into a Colab cell and run it:

```python
import os
if not os.path.isdir("/content/music_captioning"):
    !git clone https://github.com/akbargherbal/music_captioning.git /content/music_captioning
%cd /content/music_captioning
!pip install -q -r requirements.txt
```

This is safe to re-run: it skips the clone if the folder exists.

### Step 3 — Download the model (once per runtime)

```python
from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(
    "Urabewe/Ace-Step-Captioner-4bit",
    local_dir="models/Ace-Step-Captioner-4bit",
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
)
hf_hub_download(
    "ACE-Step/acestep-captioner",
    "spk_dict.pt",
    local_dir="models/Ace-Step-Captioner-4bit",
)
```

This downloads about **6.4 GB** and can take a few minutes. If Colab
disconnects, run the same cell again — it resumes.

> **Colab deletes the VM disk when the runtime ends.** To avoid re-downloading,
> mount Google Drive and symlink the model folder before this step:
>
> ```python
> from google.colab import drive
> drive.mount("/content/drive")
> import os
> os.makedirs("/content/drive/MyDrive/music_captioning_models", exist_ok=True)
> if not os.path.exists("models"):
>     os.symlink("/content/drive/MyDrive/music_captioning_models", "models")
> ```

### Step 4 — Add your song

Use the **folder icon** on the left side of Colab and upload your file to
`/content`. For example: `/content/my_song.mp3`.

You can also use:

```python
from google.colab import files
uploaded = files.upload()   # choose a file from your computer
```

Then use the uploaded filename in the next step.

### Step 5 — Run the pipeline

```python
!python scripts/demo.py /content/my_song.mp3
```

Replace `my_song.mp3` with your file name. You will see the caption first, then
the experimental term lists.

### Step 6 — Save the result (optional)

```python
!python scripts/demo.py /content/my_song.mp3 --out outputs/my_song.json
```

`--out` creates parent folders and writes JSON. To print only JSON:

```python
!python scripts/demo.py /content/my_song.mp3 --json
```

### Step 7 — Check the install (optional, CPU only)

This does not use the model or the GPU:

```python
!python -m pytest tests/ -q
```

Expected: `69 passed` (or a similar count).

---

## 4. The same thing from a terminal

Use this on a Linux machine with an NVIDIA GPU and a working CUDA PyTorch
install. Colab is the supported path; a local CUDA setup varies by machine.

```bash
git clone https://github.com/akbargherbal/music_captioning.git
cd music_captioning

# PyTorch must match your CUDA version; install it first if needed.
pip install -r requirements.txt

# Download the model (same as the Colab step).
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(
    "Urabewe/Ace-Step-Captioner-4bit",
    local_dir="models/Ace-Step-Captioner-4bit",
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
)
hf_hub_download(
    "ACE-Step/acestep-captioner",
    "spk_dict.pt",
    local_dir="models/Ace-Step-Captioner-4bit",
)
PY

python scripts/demo.py /path/to/my_song.mp3 --out outputs/my_song.json
```

If `torch.cuda.is_available()` is `False`, fix the PyTorch/CUDA install before
running the pipeline.

---

## 5. Reading the output

The JSON has these fields:

| Field | Meaning |
|---|---|
| `caption` | The model's prose description. This is the most reliable part. |
| `instruments` | Instrument terms found in the caption. **Experimental.** |
| `vocals` | Vocal terms found in the caption. **Experimental.** |
| `production` | Production/mood terms found in the caption. **Experimental.** |
| `tags_experimental` | Always `true`. A warning that the three lists are not a finished prompt. |
| `suno_prompt` | Only present with `--experimental-suno`. Unsupported. |
| `suno_experimental` | `true` when the prompt was requested. |
| `chunk_captions` | Not saved by the CLI; the captioner's per-60-second text. |

The term lists only contain words the caption actually says. The extractor does
not guess. That is why it is safe to read, but also why it misses things the
caption did not mention (or said in a different form, such as `trumpets`
instead of `trumpet`).

### About the experimental prompt

If you want the old comma-joined string for an experiment:

```bash
python scripts/demo.py my_song.mp3 --experimental-suno
```

Do not treat this as a supported prompt. The audit found missing genre/mood,
plural misses, and contradictory chunk tags. See [`TAG_AUDIT.md`](TAG_AUDIT.md).

---

## 6. Common tasks

### Use a different audio format

MP3, WAV, FLAC, M4A and OGG all work if `ffmpeg` can read them. No conversion is
needed.

### Process a long track

`demo.py` splits long audio into 60-second chunks automatically. To change the
chunk size:

```bash
python scripts/demo.py long_mix.mp3 --chunk-seconds 30
```

Smaller chunks use less memory. Longer chunks give the model more context.

### Process a folder of albums

`scripts/batch.py` is the resumable runner. It expects a root folder that
contains one subfolder per album, and writes one JSON per album, updates
`outputs/index.json`, and (by default) commits and pushes so a lost Colab
session can resume.

```bash
python scripts/batch.py \
  --root "/content/MyArtist" \
  --artist-name MY_ARTIST \
  --max-track-seconds 480 \
  --min-bitrate-kbps 128 \
  --no-push
```

`--no-push` keeps everything local. Use it if you have not set up git push.
This runner is designed for the project's own discography workflow; for a
single file, use `demo.py`.

### Skip long or low-quality files

- `--max-track-seconds 480` skips tracks over 8 minutes. `0` disables the cap.
- `--min-bitrate-kbps 128` skips low-quality files. `0` disables the check.

Skipped tracks are listed under `skipped_tracks`; they are not failures.

### Re-run only failed tracks

```bash
python scripts/batch.py --retry-failed
```

### Keep a long batch alive

`batch.py` handles per-track and per-album failures, but not the whole process
being killed. Run it under the supervisor:

```bash
setsid nohup python3 -u scripts/supervise.py --log /content/batch.log -- \
  --root "/content/MyArtist" --artist-name MY_ARTIST --max-minutes 0 \
  >/content/supervisor.out 2>&1 &
```

Watch `/content/batch.log`. A restart resumes from the last committed album.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False` | Runtime is CPU-only | Runtime → Change runtime type → T4 GPU |
| `CUDA out of memory` | Something else is using VRAM, or chunks are too long | Restart the runtime; close other notebooks; try `--chunk-seconds 30` |
| `spk_dict.pt` not found | The second download step was skipped | Re-run the `hf_hub_download(... "spk_dict.pt" ...)` block |
| `ffmpeg not found` | `ffmpeg` is not on `PATH` | Colab: `!apt-get -qq install -y ffmpeg` |
| Model download stops | Network or Colab timeout | Re-run the download cell; it resumes |
| `empty caption` or nonsense | Silent, corrupt, or unsupported audio | Try a known-good MP3/WAV; check the file plays |
| `bfloat16` / dtype error | T4 has no native bf16 | Use the current `requirements.txt`; do not force bf16 |
| Batch run cannot push | No git credentials in the runtime | Use `--no-push`, or configure a token |
| Tags miss something obvious | Known extractor limitation | Use the caption; see [`TAG_AUDIT.md`](TAG_AUDIT.md) |
| `outputs/` file is missing | `demo.py` prints only by default | Add `--out outputs/name.json` |

---

## 8. Words used in this project

| Word | Simple meaning |
|---|---|
| **Captioner** | The AI model that writes the prose description. |
| **4-bit / nf4** | A compressed way to store the model so it fits on a small GPU. |
| **VRAM** | Memory on the GPU. The T4 has 15 GB; this project peaks around 8 GB. |
| **Chunk** | A 60-second piece of a longer track. |
| **Greedy decoding** | Always pick the most likely next word. Makes runs repeatable. |
| **T4** | The free GPU type on Google Colab. |
| **SDPA** | The attention implementation used here. FlashAttention is not available on T4. |
| **Manifest** | `outputs/index.json`, the list of albums already processed. |
| **Suno** | An AI music service. Its prompts are usually short lists of genre, mood and instrument tags. |

---

## 9. Where to go next

- [`../README.md`](../README.md) — project overview, all commands, full output format.
- [`RESULTS.md`](RESULTS.md) — real measured time and VRAM on a T4.
- [`TAG_AUDIT.md`](TAG_AUDIT.md) — why the term lists are experimental.
- [`../notebooks/colab_setup.ipynb`](../notebooks/colab_setup.ipynb) — the same
  setup as a notebook.
- [`TRANSCRIPTION.md`](TRANSCRIPTION.md) — the separate ABC/maqam transcription
  pipeline.
- [`../AGENTS.md`](../AGENTS.md) — engineering rules for contributors.

## 10. Getting help

When asking for help, include:

1. the exact command you ran,
2. the full error message (not a screenshot of one line),
3. whether you are on Colab or local,
4. the output of `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`,
5. the audio duration and format.

Do not upload copyrighted audio or private files to a public issue.
