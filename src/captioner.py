"""Subprocess wrapper around the patched llama.cpp ``llama-cli``.

Runs the ACE-Step captioner GGUF on an audio file and returns the prose
caption. Long audio is trimmed of leading/trailing silence, split into equal
chunks of roughly ``chunk_seconds`` (one minute by default: the length the GGUF
author validated), and each chunk is captioned separately. The per-chunk
captions are then merged deterministically (see ``merge.py``).

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

from .merge import merge_section_captions
from .tags import merge_tags

PROMPT = "*Task* Describe this audio in detail"

# One-minute chunks match what the GGUF author validated.
DEFAULT_CHUNK_SECONDS = 60.0
DEFAULT_MAX_CHUNK_SECONDS = 90.0

# Silence trimming: a run of at least SILENCE_MIN_DURATION below
# SILENCE_NOISE_DB at either edge of the file is cut, keeping TRIM_PADDING of
# it on each side. Below MIN_TRIM_SECONDS it is not worth re-writing the file.
SILENCE_NOISE_DB = -50
SILENCE_MIN_DURATION = 0.5
TRIM_PADDING = 0.1
MIN_TRIM_SECONDS = 0.5

# A leading silence counts as "at the start of the file" within this tolerance.
_LEADING_TOLERANCE = 0.25
# A silence interval reaching this close to the end means the file ends silent.
_END_TOLERANCE = 0.25

# A trailing split fragment shorter than this is frame-snap debris, not audio.
MIN_CHUNK_SECONDS = 1.0

_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")

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
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS
    trim_silence: bool = True
    _last_run: dict[str, float] = field(default_factory=dict, repr=False, compare=False)

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
        self, audio_path: Path | str, chunk_seconds: float | None = None
    ) -> list[str]:
        """Caption a full track, splitting it into equal ``chunk_seconds`` segments.

        Leading/trailing silence is trimmed first (unless ``trim_silence`` is
        off), then the duration is divided into the fewest equal chunks that fit
        ``max_chunk_seconds`` (see :func:`plan_chunks`). Equal chunks mean no
        tiny final segment. Returns one caption per chunk (a single-element list
        when the track fits in one window).

        Details of the last run (chunk count, seconds of silence removed) are
        left in :attr:`last_run` for callers that report them.
        """
        path = Path(audio_path)
        if not path.exists():
            raise CaptionerError(f"audio not found: {path}")
        target = self.chunk_seconds if chunk_seconds is None else chunk_seconds
        original = audio_duration(path)

        with tempfile.TemporaryDirectory(prefix="acestep_chunks_") as tmp:
            workdir = Path(tmp)
            source, removed = (path, 0.0)
            if self.trim_silence:
                source, removed = trim_silence_edges(path, workdir)
            # Measure the trimmed file: a stream copy snaps to frame boundaries,
            # so it is not exactly ``original - removed``. Planning on the
            # computed value can emit a spurious trailing fragment.
            duration = audio_duration(source)
            if removed:
                removed = max(0.0, original - duration)
            count, segment = plan_chunks(duration, target, self.max_chunk_seconds)
            if count <= 1:
                captions = [self.caption_chunk(source)]
            else:
                chunks = _split_audio(source, workdir, segment)
                if not chunks:
                    raise CaptionerError(f"ffmpeg produced no chunks for {path}")
                captions = [self.caption_chunk(c) for c in chunks]

        self._last_run = {
            "original_duration_s": round(original, 3),
            "duration_s": round(duration, 3),
            "silence_trimmed_s": round(removed, 3),
            "chunk_count": len(captions),
            "segment_seconds": round(segment, 3) if count > 1 else round(duration, 3),
        }
        return captions

    @property
    def last_run(self) -> dict[str, float]:
        """Chunk count / trimmed seconds from the most recent ``caption_audio``."""
        return dict(self._last_run)


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


def plan_chunks(
    duration_s: float, target_s: float, max_chunk_s: float
) -> tuple[int, float]:
    """Return ``(count, segment_seconds)`` for equal-length chunks.

    ``count`` is the nearest whole number of ``target_s`` windows, raised until
    a segment fits ``max_chunk_s``. Because the count is rounded rather than
    ceilinged, a short remainder (e.g. 26 s after five 60 s windows) is spread
    across every segment instead of becoming a tiny final chunk.
    """
    if duration_s <= 0:
        raise CaptionerError(f"duration must be positive, got {duration_s}")
    if target_s <= 0 or max_chunk_s <= 0:
        raise CaptionerError(
            f"chunk lengths must be positive, got target={target_s}, max={max_chunk_s}"
        )
    if target_s > max_chunk_s:
        raise CaptionerError(
            f"target chunk ({target_s}s) cannot exceed max chunk ({max_chunk_s}s)"
        )
    # int(x + 0.5) rounds half up; round() would round 2.5 to 2 (banker's).
    count = max(1, int(duration_s / target_s + 0.5))
    while duration_s / count > max_chunk_s:
        count += 1
    return count, duration_s / count


@dataclass(frozen=True)
class SilenceTrim:
    """Leading/trailing silence to cut, in seconds from the start of the file."""

    start: float
    end: float
    removed: float


def parse_silence(stderr: str, duration_s: float) -> SilenceTrim | None:
    """Interpret ``ffmpeg silencedetect`` output.

    Returns the span worth keeping, or ``None`` when there is no leading or
    trailing silence of at least ``MIN_TRIM_SECONDS``. ``silence_end`` is
    optional: when a silence runs to the end of the file ffmpeg may stop after
    ``silence_start``, which still counts as trailing silence.
    """
    if duration_s <= 0:
        return None

    events: list[tuple[str, float]] = []
    for line in stderr.splitlines():
        for match in _SILENCE_START.finditer(line):
            events.append(("start", float(match.group(1))))
        for match in _SILENCE_END.finditer(line):
            events.append(("end", float(match.group(1))))
    if not events:
        return None

    intervals: list[tuple[float, float]] = []
    pending: float | None = None
    for kind, value in events:
        if kind == "start":
            pending = value
        elif pending is not None and value >= pending:
            intervals.append((pending, value))
            pending = None

    start = 0.0
    end = duration_s
    if intervals and intervals[0][0] <= _LEADING_TOLERANCE:
        start = intervals[0][1]
    if pending is not None:  # trailing silence with no silence_end line
        end = pending
    elif intervals and intervals[-1][1] >= duration_s - _END_TOLERANCE:
        end = intervals[-1][0]

    if start > 0:
        start = max(0.0, start - TRIM_PADDING)
    if end < duration_s:
        end = min(duration_s, end + TRIM_PADDING)

    removed = duration_s - (end - start)
    if end - start <= MIN_TRIM_SECONDS or removed <= MIN_TRIM_SECONDS:
        return None
    return SilenceTrim(start=start, end=end, removed=removed)


def detect_silence(path: Path | str) -> SilenceTrim | None:
    """Run ``ffmpeg silencedetect`` on ``path`` and return what to trim."""
    if shutil.which("ffmpeg") is None:
        raise CaptionerError("ffmpeg not found on PATH (needed to detect silence)")
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner",
            "-i", str(path),
            "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_DURATION}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise CaptionerError(
            f"ffmpeg silencedetect failed on {path}: {proc.stderr.strip()[-2000:]}"
        )
    return parse_silence(proc.stderr, audio_duration(path))


def trim_silence_edges(path: Path | str, out_dir: Path | str) -> tuple[Path, float]:
    """Cut leading/trailing silence, returning ``(audio_path, seconds_removed)``.

    Returns the input path unchanged with ``0.0`` when there is nothing worth
    trimming, so callers never special-case it. The trimmed file is written to
    ``out_dir`` (normally the caller's temporary directory).
    """
    source = Path(path)
    span = detect_silence(source)
    if span is None:
        return source, 0.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"trimmed{source.suffix or '.wav'}"
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{span.start:.3f}",
            "-i", str(source),
            "-t", f"{span.end - span.start:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise CaptionerError(
            f"ffmpeg silence trim failed on {source}: {proc.stderr.strip()[-2000:]}"
        )
    if not out.exists() or out.stat().st_size == 0:
        raise CaptionerError(f"ffmpeg silence trim produced no audio for {source}")
    return out, span.removed


def _split_audio(path: Path, out_dir: Path, chunk_seconds: float) -> list[Path]:
    """Split ``path`` into numbered chunks with ffmpeg (stream copy, no re-encode).

    A stream copy snaps cuts to frame boundaries, so the last segment can come
    out as a sub-second fragment (and, cut mid-frame, may not even be readable).
    Those fragments are dropped rather than captioned: they contain no music and
    the model invents text for them.
    """
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
    chunks = sorted(out_dir.glob(f"chunk_*{suffix}"))
    while len(chunks) > 1:
        try:
            tail_seconds = audio_duration(chunks[-1])
        except CaptionerError:
            tail_seconds = 0.0  # unreadable fragment
        if tail_seconds >= MIN_CHUNK_SECONDS:
            break
        chunks.pop().unlink(missing_ok=True)
    return chunks


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
    full_caption = merge_section_captions(captions)
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
