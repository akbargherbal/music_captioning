"""Subprocess wrapper around the patched llama.cpp ``llama-cli``.

Runs the ACE-Step captioner GGUF on an audio file and returns the prose
caption. Long audio is split into ~60 s chunks (the length the GGUF author
validated) and each chunk is captioned separately.

Safety notes (see AGENTS.md):
* argument lists only, never ``shell=True``;
* user paths are never interpolated into a shell string;
* every subprocess has a timeout and captured stderr.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .tags import merge_tags

PROMPT = "*Task* Describe this audio in detail"

# One-minute chunks match what the GGUF author validated.
DEFAULT_CHUNK_SECONDS = 60.0

# llama.cpp spam on stdout/stderr we do not want in the caption.
_LOG_LINE = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)?"
    r"(?:llama|build|main|system_info|load_|print_info|ggml|gguf|sampl|generate|"
    r"init|warning|error|decode|encode|clip|mtmd|srv|common|slot|chat|"
    r"llama_model_loader|llama_context|ggml_cuda|CUDA)",
    re.IGNORECASE,
)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class CaptionerError(RuntimeError):
    """Raised when the captioner cannot run or produces no usable output."""


@dataclass
class Captioner:
    """Configured wrapper around a llama.cpp binary."""

    llama_cli: Path
    model: Path
    mmproj: Path
    prompt: str = PROMPT
    n_predict: int = 400
    ctx_size: int = 8192
    n_gpu_layers: int = 999
    temperature: float = 0.0
    timeout: float = 900.0
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.llama_cli = Path(self.llama_cli)
        self.model = Path(self.model)
        self.mmproj = Path(self.mmproj)
        for label, path in (
            ("llama-cli", self.llama_cli),
            ("model", self.model),
            ("mmproj", self.mmproj),
        ):
            if not path.exists():
                raise CaptionerError(f"{label} not found: {path}")
        if not self.llama_cli.is_file():
            raise CaptionerError(f"llama-cli is not a file: {self.llama_cli}")

    def _command(self, audio_path: Path, n_gpu_layers: int, extra: list[str]) -> list[str]:
        return [
            str(self.llama_cli),
            "-m", str(self.model),
            "--mmproj", str(self.mmproj),
            "--audio", str(audio_path),
            "-p", self.prompt,
            "-n", str(self.n_predict),
            "--temp", str(self.temperature),
            "--single-turn",
            "--simple-io",
            "-ngl", str(n_gpu_layers),
            "--ctx-size", str(self.ctx_size),
            *extra,
            *self.extra_args,
        ]

    def _run(self, audio_path: Path) -> subprocess.CompletedProcess[str]:
        """Run once; on projector/OOM errors retry with documented fallbacks."""
        attempts: list[tuple[int, list[str]]] = [
            (self.n_gpu_layers, []),
            (self.n_gpu_layers, ["--no-mmproj-offload"]),
            (20, ["--no-mmproj-offload"]),
            (4, ["--no-mmproj-offload"]),
        ]
        last: subprocess.CompletedProcess[str] | None = None
        for ngl, extra in attempts:
            cmd = self._command(audio_path, ngl, extra)
            try:
                proc = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise CaptionerError(
                    f"llama-cli timed out after {self.timeout:.0f}s on {audio_path}"
                ) from exc
            last = proc
            if proc.returncode == 0:
                return proc
            blob = (proc.stderr + proc.stdout).lower()
            retryable = (
                "out of memory" in blob
                or "outofmemory" in blob
                or "mmproj" in blob
                or "projector" in blob
            )
            if not retryable:
                raise CaptionerError(
                    f"llama-cli failed (exit {proc.returncode}) on {audio_path}\n"
                    f"command: {' '.join(cmd)}\nstderr:\n{proc.stderr[-4000:]}"
                )
        assert last is not None
        raise CaptionerError(
            f"llama-cli failed after retries on {audio_path}\n"
            f"stderr:\n{last.stderr[-4000:]}"
        )

    @staticmethod
    def _clean(stdout: str) -> str:
        text = _ANSI.sub("", stdout)
        kept = [ln for ln in text.splitlines() if not _LOG_LINE.match(ln)]
        return "\n".join(kept).strip()

    def caption_chunk(self, audio_path: Path | str) -> str:
        """Caption a single audio file (kept short, ~1 min)."""
        path = Path(audio_path)
        if not path.exists():
            raise CaptionerError(f"audio not found: {path}")
        proc = self._run(path)
        caption = self._clean(proc.stdout)
        if not caption:
            raise CaptionerError(
                f"empty caption from llama-cli on {path}\n"
                f"raw stdout (last 2000 chars):\n{proc.stdout[-2000:]}\n"
                f"stderr (last 2000 chars):\n{proc.stderr[-2000:]}"
            )
        return caption

    def caption_audio(
        self, audio_path: Path | str, chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    ) -> list[str]:
        """Caption a full track, splitting into ~``chunk_seconds`` segments.

        Returns one caption per chunk (a single-element list if the track is
        already short enough).
        """
        path = Path(audio_path)
        if not path.exists():
            raise CaptionerError(f"audio not found: {path}")
        duration = audio_duration(path)
        if duration <= chunk_seconds:
            return [self.caption_chunk(path)]

        with tempfile.TemporaryDirectory(prefix="acestep_chunks_") as tmp:
            chunks = _split_audio(path, Path(tmp), chunk_seconds)
            if not chunks:
                raise CaptionerError(f"ffmpeg produced no chunks for {path}")
            return [self.caption_chunk(c) for c in chunks]


def audio_duration(path: Path | str) -> float:
    """Return duration in seconds via ffprobe."""
    if shutil.which("ffprobe") is None:
        raise CaptionerError("ffprobe not found on PATH (install ffmpeg)")
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nk=1:nw=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise CaptionerError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise CaptionerError(f"could not parse duration {proc.stdout!r}") from exc


def audio_stream_info(path: Path | str) -> tuple[float | None, float | None]:
    """Return ``(duration_seconds, audio_bitrate_kbps)`` in one ffprobe call.

    Either value may be ``None`` when ffprobe does not report it (some VBR
    files). Used by the batch runner to skip low-quality / over-long tracks.
    """
    if shutil.which("ffprobe") is None:
        raise CaptionerError("ffprobe not found on PATH (install ffmpeg)")
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=bit_rate:format=duration,bit_rate",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise CaptionerError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise CaptionerError(f"could not parse ffprobe output for {path}") from exc
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    raw_br = stream.get("bit_rate") or fmt.get("bit_rate")
    raw_dur = fmt.get("duration")
    bitrate = float(raw_br) / 1000.0 if raw_br else None
    duration = float(raw_dur) if raw_dur else None
    return duration, bitrate


def _split_audio(path: Path, out_dir: Path, chunk_seconds: float) -> list[Path]:
    """Split ``path`` into numbered chunks with ffmpeg (stream copy, no re-encode)."""
    if shutil.which("ffmpeg") is None:
        raise CaptionerError("ffmpeg not found on PATH (needed to split long audio)")
    suffix = path.suffix or ".wav"
    pattern = out_dir / f"chunk_%03d{suffix}"
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-f", "segment", "-segment_time", f"{chunk_seconds}",
            "-c", "copy", str(pattern),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise CaptionerError(f"ffmpeg split failed: {proc.stderr.strip()}")
    return sorted(out_dir.glob(f"chunk_*{suffix}"))


def caption_and_tag(
    captioner: Captioner,
    audio_path: Path | str,
    *,
    experimental_suno: bool = False,
) -> dict[str, object]:
    """Caption ``audio_path`` (chunked) and return experimental term lists.

    ``suno_prompt`` is only added when ``experimental_suno=True`` and is marked
    ``suno_experimental``; the default output is caption + experimental tags.
    """
    from .tags import extract_tags, to_suno

    captions = captioner.caption_audio(audio_path)
    per_chunk = [extract_tags(c) for c in captions]
    merged = merge_tags(per_chunk)
    full_caption = " ".join(c.strip() for c in captions)
    result: dict[str, object] = {
        "instruments": merged["instruments"],
        "vocals": merged["vocals"],
        "production": merged["production"],
        "caption": full_caption,
        "chunk_captions": captions,
        "tags_experimental": True,
    }
    if experimental_suno:
        result["suno_prompt"] = to_suno(merged)
        result["suno_experimental"] = True
    return result
