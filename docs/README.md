# Documentation

Start here if you are new to the project.

| Document | What it is for |
|---|---|
| [Quick Start](QUICKSTART.md) | Step-by-step setup on Colab, first run, output fields, troubleshooting, glossary. |
| [Results](RESULTS.md) | Real measured caption time and VRAM on a Colab T4. |
| [Chunk-length experiment](CHUNK_LENGTH_EXPERIMENT.md) | 60 s vs one-shot vs rebalanced chunks: why one-shot OOMs, the ~190 s ceiling, and why the repetition is a per-window frame. |
| [Chunk trim + merge](CHUNK_MERGE.md) | What the pipeline does now: silence trim, equal ~90 s chunks, deterministic one-framing merge, with T4 validation numbers. |
| [Tag audit](TAG_AUDIT.md) | Why the deterministic term lists are experimental, with a 5-track audit and corpus numbers. |
| [Transcription](TRANSCRIPTION.md) | The separate ABC/maqam transcription pipeline. |
| [Transcription results](TRANSCRIPTION_RESULTS.md) | Measured transcription results and known failures. |
| [Resume transcription](RESUME_TRANSCRIPTION.md) | How to resume transcription work on a fresh runtime. |

For the project overview and full command reference, see the
[README](../README.md).
