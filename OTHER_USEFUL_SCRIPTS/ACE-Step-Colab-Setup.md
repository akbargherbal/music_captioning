# ACE-Step 1.5 — Google Colab (T4) Quick Setup Guide

> **Runtime:** Python 3.12 · CUDA 12.8 · Tesla T4 · pre-Ampere fp32 mode

---

## 1. Before You Start

In Colab, go to **Runtime → Change runtime type → T4 GPU** before running anything.

---

## 2. Install Dependencies

```python
import torch, sys

# Sanity check
assert torch.cuda.is_available(), "Switch to T4 GPU runtime first!"
print(f"Python {sys.version_info.major}.{sys.version_info.minor} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")

# Reinstall torch only if version doesn't match
if torch.__version__ != "2.10.0+cu128":
    %pip install -q torch==2.10.0+cu128 torchaudio==2.10.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# ACE-Step — pinned to last tested commit (not on PyPI)
COMMIT = "1d9d2d3c8a9de32011ecdbda304c7c16e5bb814c"
%pip install -q f"git+https://github.com/ACE-Step/ACE-Step-1.5.git@{COMMIT}"

# Remaining direct dependencies
%pip install -q \
    "transformers==4.57.6" \
    "diffusers==0.37.1"    \
    "accelerate==1.13.0"   \
    "soundfile==0.13.1"    \
    "huggingface-hub==0.36.2"
```

---

## 3. Download the Model

The checkpoint is downloaded fresh each session into ephemeral Colab storage.
Use `allow_patterns` to fetch only the variant you need and skip the rest.

```python
from huggingface_hub import snapshot_download

MODEL = "acestep-v15-sft"   # options: acestep-v15-turbo · sft · base · xl-*

snapshot_download(
    repo_id="ACE-Step/ACE-Step-v1.5",
    allow_patterns=[f"{MODEL}/**"],
    ignore_patterns=["*.md", "*.png", "*.jpg"],
    local_dir="/content/checkpoints",
)
print("Model ready.")
```

> **Why not Google Drive?** The checkpoint is large. Re-downloading each session
> is cheaper than spending your free Drive quota on it (~5–10 min per session).

---

## 4. Upload the Script

Upload `ace_generator.py` via the Colab file browser sidebar, or run:

```python
# If stored in a GitHub Gist or raw URL:
!wget -q <your-raw-url> -O /content/ace_generator.py
```

---

## 5. Generate Music

```bash
!python ace_generator.py \
  "Your music description here" \
  --duration 60          \
  --model acestep-v15-sft \
  --steps 60             \
  --guidance 12.0        \
  --bpm 91               \
  --time-signature "4/4" \
  --vocal-language ar    \
  --batch-size 1         \
  --seed 42
```

The output file (`generated_music.wav`) appears in `/content/` — download it from the sidebar.

---

## 6. Key Notes for T4 (pre-Ampere)

| Topic | Detail |
|---|---|
| **dtype** | Model runs in **float32** — `ace_generator.py` enforces this automatically via three patch layers. Do not pass any fp16 flags. |
| **FlashAttention** | Not supported on T4 (compute 7.5). The script disables it automatically and falls back to SDPA. |
| **Batch size** | Keep at `1`. T4 has 16 GB VRAM but fp32 mode doubles memory usage vs fp16. |
| **Steps** | `acestep-v15-turbo` caps at 8 steps. For `sft`/`base`, 50–60 steps gives best quality. |
| **Session reset** | Colab wipes `/content/` on disconnect. Re-run steps 2–3 after every new session. |

---

## Locked Versions

```text
ace-step @ git+https://github.com/ACE-Step/ACE-Step-1.5.git@1d9d2d3c8a9de32011ecdbda304c7c16e5bb814c
torch==2.10.0+cu128
torchaudio==2.10.0+cu128
transformers==4.57.6
diffusers==0.37.1
accelerate==1.13.0
soundfile==0.13.1
huggingface-hub==0.36.2
```
