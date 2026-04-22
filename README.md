# AI Music Captioning & Suno Prompt Synthesis

A pipeline for analyzing audio tracks and generating high-quality, descriptive music captions optimized for AI music generators like Suno.ai.

Instead of processing an entire track sequentially, this tool extracts **four landmark windows** (start, early, pre-mid, post-mid) and analyzes them using Nvidia's `music-flamingo-hf` vision-language-audio model. The resulting structured JSON is then passed to an LLM (using provided system prompts) to synthesize a single, cohesive style prompt.

## 🚀 Key Features

- **Landmark Chunking:** Efficiently summarizes a track by analyzing 4 specific temporal windows rather than the entire file, saving time and compute.
- **Optimized for Google Colab (T4):** Runs comfortably on a 15 GB VRAM GPU using 4-bit quantization (`bitsandbytes`) and expandable PyTorch memory segments.
- **Two Extraction Modes:**
  - `quick`: Generates one dense, comma-separated summary per chunk.
  - `detailed`: Uses four focused tag-cluster prompts per chunk (genre, instruments, vocals, tempo/key).
- **LLM Synthesis Prompts:** Includes carefully crafted System and User prompts to guide an LLM in merging the chunked JSON data into a perfect Suno-ready paragraph.

## 📋 Prerequisites & Installation

Ensure you have Python 3.8+ installed. You will need PyTorch and the latest Hugging Face `transformers` library.

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install torch torchaudio
pip install "git+https://github.com/huggingface/transformers" accelerate bitsandbytes

# Install additional requirements
pip install click librosa soundfile
```

_Note: If you are running this in Google Colab, you may also need to authenticate with Google Cloud if you are pulling audio files from a GCS bucket (as demonstrated in `Music_Captioning.md`)._

## 🛠️ Usage

The workflow is split into two steps: **1. Audio Analysis (Extraction)** and **2. Caption Synthesis**.

### Step 1: Audio Analysis

Run the `music_caption.py` script to process a directory of `.mp3` or `.wav` files.

```bash
# Quick mode (default): one dense comma-separated summary per landmark chunk
python music_caption.py --dir ./audio_files

# Detailed mode: four focused tag-cluster prompts per chunk
python music_caption.py --dir ./audio_files --mode detailed

# Custom window size (default is 40 seconds per chunk)
python music_caption.py --dir ./audio_files --window-seconds 30
```

This will generate a `.json` file for every audio track in the target directory, containing the model's analysis of the 4 landmark chunks.

### Step 2: Caption Synthesis

The Python script outputs raw, chunked data. To convert this into a flowing, Suno-ready prompt, use an LLM (like ChatGPT, Claude, or a local model) with the provided prompt templates.

1. Copy the text from `PROMPT_SUNO_SYNTHESIS/SYSTEM_PROMPT.md` and set it as the LLM's System Prompt.
2. Copy the text from `PROMPT_SUNO_SYNTHESIS/PROMPT_TEMPLATE.md`.
3. Replace `{JSON_DUMP}` with the contents of your generated `.json` file.
4. Send the prompt to the LLM.

**Example LLM Output:**

> _Middle Eastern folk fusion. A solo nylon-string acoustic guitar performs a melodic lead with frequent hammer-ons, pull-offs, and slides, accompanied by a secondary acoustic guitar providing rhythmic strumming. A male vocal performs wordless melismatic humming and chanting in a Phrygian dominant scale. The percussion consists of a darbuka playing a traditional Maqsum rhythm with sharp tek accents and resonant doum hits. The tempo is 105 BPM in 4/4 time. The arrangement features a call-and-response dynamic between the vocal lines and the guitar melodies._

## 🎛️ CLI Reference (`music_caption.py`)

| Option             | Default                               | Description                                                              |
| :----------------- | :------------------------------------ | :----------------------------------------------------------------------- |
| `--dir`            | **Required**                          | Directory containing `.mp3` / `.wav` files to process.                   |
| `--mode`           | `quick`                               | `quick` (1 summary per chunk) or `detailed` (4 focused prompts).         |
| `--window-seconds` | `40.0`                                | Duration of each landmark window in seconds.                             |
| `--max-tokens`     | `120`                                 | Max new tokens per caption generation.                                   |
| `--temperature`    | `0.3`                                 | Sampling temperature.                                                    |
| `--top-p`          | `0.9`                                 | Nucleus sampling probability.                                            |
| `--dtype`          | `auto`                                | Model data type (`auto`, `float16`, `bfloat16`, `float32`).              |
| `--quantize`       | `4bit`                                | Weight quantization (`none`, `8bit`, `4bit`). Recommended: `4bit` on T4. |
| `--flash-attn`     | `False`                               | Enable Flash Attention 2 (Requires Ampere+ GPU, NOT available on T4).    |
| `--log-file`       | `<dir>/caption_audio_<timestamp>.log` | Path for the debug log file.                                             |

## 📂 Output JSON Schema

The script outputs a structured JSON file for each track, which looks like this:

```json
{
  "metadata": {
    "model": "nvidia/music-flamingo-hf",
    "generated_at": "2026-04-22T06:28:14Z",
    "audio_file": "/absolute/path/to/track.mp3",
    "duration_seconds": 268.2,
    "window_seconds": 40.0,
    "total_chunks": 4,
    "mode": "detailed"
  },
  "chunks": [
    {
      "chunk_index": 0,
      "label": "start",
      "start_seconds": 0.0,
      "end_seconds": 40.0,
      "captions": {
        "genre_style": "Arabic folk, traditional Middle Eastern, acoustic ballad",
        "instruments": "oud with intricate plucked melodies, acoustic guitar...",
        "vocals": "male baritone, melismatic phrasing, rich vibrato...",
        "tempo_key": "85.71 BPM, D minor, 4/4 time"
      }
    }
    // ... early, pre_mid, and post_mid chunks follow
  ]
}
```

## 📁 Project Structure

- `music_caption.py`: The core PyTorch/Transformers script for audio chunking and inference.
- `Music_Captioning.md`: A Jupyter Notebook export demonstrating a full workflow (GCS download -> Inference -> GCS upload).
- `PROMPT_SUNO_SYNTHESIS/`:
  - `SYSTEM_PROMPT.md`: The system instructions for the synthesis LLM.
  - `PROMPT_TEMPLATE.md`: The user prompt template for injecting the JSON data.
