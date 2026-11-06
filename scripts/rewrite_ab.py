#!/usr/bin/env python3
"""A/B the rewrite backends on stored chunk captions (no re-captioning).

Reads album JSONs that already contain ``chunk_captions`` and runs the same
stateless rewrite through each requested backend, then reports quality and
speed side by side:

    python scripts/rewrite_ab.py outputs/fadhl_shaker_selected/*.json \
        --backends hf,llamacpp \
        --gguf models/Qwen3-4B-Instruct-2507-Q8_0.gguf

Each backend is loaded, used for every track, then unloaded before the next one
starts, so the captioner and the rewrite models never share the GPU (and the two
rewrite engines never do either).

Reported per backend: ok / fallback counts, seconds per track, average output
words, and **term coverage** — the share of instrument and vocal terms named in
the source captions that survive into the rewrite. Coverage is a proxy for
"did it drop detail", the thing quantization is most likely to damage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.rewrite import build_source_text, rewrite_caption  # noqa: E402
from src.tags import extract_tags  # noqa: E402

CATEGORIES = ("instruments", "vocals")


class MemorySampler:
    """Poll system RAM and GPU VRAM so the report shows what each backend used.

    The GPU is a T4 either way; RAM only changes load times and OOM safety, not
    tokens/second. Measuring both keeps that claim honest.
    """

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.peak_ram_gib = 0.0
        self.peak_vram_gib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _ram_used_gib() -> float:
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                values[key] = int(rest.split()[0])  # kB
            total = values.get("MemTotal", 0) / 1024**2
            available = values.get("MemAvailable", 0) / 1024**2
            return max(0.0, total - available)
        except (OSError, ValueError, IndexError):
            return 0.0

    @staticmethod
    def _vram_used_gib() -> float:
        if shutil.which("nvidia-smi") is None:
            return 0.0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            used = [float(part) for part in out.split() if part.replace(".", "", 1).isdigit()]
            return max(used) / 1024 if used else 0.0
        except (OSError, subprocess.SubprocessError, ValueError):
            return 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_ram_gib = max(self.peak_ram_gib, self._ram_used_gib())
            self.peak_vram_gib = max(self.peak_vram_gib, self._vram_used_gib())
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(target=self._run, name="mem-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


def load_albums(paths: list[Path]) -> list[dict]:
    albums = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("tracks"):
            raise SystemExit(f"{path}: not an album JSON with tracks")
        data["_path"] = str(path)
        albums.append(data)
    return albums


def make_rewriter(kind: str, args):
    if kind == "hf":
        from src.rewriter_hf import HFRewriter

        return HFRewriter(
            model_id=args.hf_model,
            quantization="none",
            max_new_tokens=args.max_new_tokens,
        )
    if kind == "llamacpp":
        if args.gguf is None:
            raise SystemExit("--gguf is required for the llamacpp backend")
        from src.rewriter_llama import LlamaCppRewriter

        return LlamaCppRewriter(
            model_path=args.gguf,
            server_bin=args.server_bin,
            max_new_tokens=args.max_new_tokens,
        )
    raise SystemExit(f"unknown backend {kind!r}")


def coverage(source: str, output: str) -> tuple[float | None, int]:
    """Share of source instrument/vocal terms that appear in the rewrite."""
    src_tags = extract_tags(source)
    out_tags = extract_tags(output)
    src_terms = [t for c in CATEGORIES for t in src_tags[c]]
    if not src_terms:
        return None, 0
    out_set = {t for c in CATEGORIES for t in out_tags[c]}
    kept = sum(1 for t in src_terms if t in out_set)
    return kept / len(src_terms), len(src_terms)


def run_backend(kind: str, albums: list[dict], args) -> dict:
    rewriter = make_rewriter(kind, args)
    print(f"\n=== backend: {kind} ({rewriter.name}) ===", flush=True)
    rows: list[dict] = []
    started = time.time()
    sampler = MemorySampler()
    with sampler:
        try:
            for album in albums:
                for track in album["tracks"]:
                    chunks = track.get("chunk_captions") or []
                    if not chunks:
                        continue
                    if args.max_tracks and len(rows) >= args.max_tracks:
                        break
                    source = build_source_text(chunks)
                    t0 = time.time()
                    result = rewrite_caption(chunks, rewriter)
                    elapsed = time.time() - t0
                    cov, n_terms = coverage(source, result.text)
                    rows.append({
                        "track": track.get("file") or track.get("title"),
                        "status": result.status,
                        "reasons": list(result.reasons),
                        "seconds": round(elapsed, 2),
                        "output_words": result.output_words,
                        "coverage": None if cov is None else round(cov, 3),
                        "source_terms": n_terms,
                    })
                    cov_s = "n/a" if cov is None else f"{cov:5.1%}"
                    print(
                        f"  {rows[-1]['track'][:44]:<44} {result.status:<8} "
                        f"{elapsed:6.1f}s  {result.output_words:4d}w  cov {cov_s}",
                        flush=True,
                    )
        finally:
            unload = getattr(rewriter, "unload", None)
            if unload is not None:
                unload()

    elapsed = time.time() - started
    ok = [r for r in rows if r["status"] == "ok"]
    covs = [r["coverage"] for r in rows if r["coverage"] is not None]
    summary = {
        "backend": kind,
        "model": rewriter.name,
        "tracks": len(rows),
        "ok": len(ok),
        "fallback": sum(1 for r in rows if r["status"] == "fallback"),
        "total_seconds": round(elapsed, 1),
        "seconds_per_track": round(elapsed / len(rows), 1) if rows else None,
        "mean_output_words": (
            round(sum(r["output_words"] for r in ok) / len(ok), 1) if ok else None
        ),
        "mean_coverage": round(sum(covs) / len(covs), 4) if covs else None,
        "peak_ram_gib": round(sampler.peak_ram_gib, 1),
        "peak_vram_gib": round(sampler.peak_vram_gib, 1),
        "rows": rows,
    }
    print(
        f"  -> {summary['ok']}/{summary['tracks']} ok, "
        f"{summary['fallback']} fallback, {summary['total_seconds']}s "
        f"({summary['seconds_per_track']}s/track), "
        f"mean coverage {summary['mean_coverage']}, "
        f"mean words {summary['mean_output_words']}, "
        f"peak RAM {summary['peak_ram_gib']} GiB, "
        f"peak VRAM {summary['peak_vram_gib']} GiB",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("albums", nargs="+", type=Path,
                        help="album JSON files containing chunk_captions")
    parser.add_argument("--backends", default="hf,llamacpp",
                        help="comma-separated: hf,llamacpp (default both)")
    parser.add_argument("--hf-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--gguf", type=Path, default=None,
                        help="GGUF weights for the llamacpp backend")
    parser.add_argument(
        "--server-bin", type=Path,
        default=REPO_ROOT / "llama.cpp-text" / "build" / "bin" / "llama-server",
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-tracks", type=int, default=0,
                        help="stop after this many tracks per backend (0 = all)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the full per-track report as JSON")
    args = parser.parse_args(argv)

    albums = load_albums(args.albums)
    n_tracks = sum(len(a["tracks"]) for a in albums)
    print(f"Albums: {len(albums)} | tracks: {n_tracks}")

    reports = []
    for kind in [k.strip() for k in args.backends.split(",") if k.strip()]:
        reports.append(run_backend(kind, albums, args))

    print("\n=== summary ===")
    header = f"{'backend':<10} {'ok':>6} {'fallback':>8} {'s/track':>8} " \
             f"{'words':>6} {'coverage':>9} {'RAM GiB':>8} {'VRAM GiB':>9}"
    print(header)
    for rep in reports:
        cov = rep["mean_coverage"]
        print(
            f"{rep['backend']:<10} {str(rep['ok']) + '/' + str(rep['tracks']):>6} "
            f"{rep['fallback']:>8} {rep['seconds_per_track']!s:>8} "
            f"{rep['mean_output_words']!s:>6} "
            f"{('n/a' if cov is None else f'{cov:.1%}'):>9} "
            f"{rep['peak_ram_gib']:>8} {rep['peak_vram_gib']:>9}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "albums": [a["_path"] for a in albums],
            "reports": reports,
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
