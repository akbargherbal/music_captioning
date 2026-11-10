#!/usr/bin/env python3
"""The only GPU process for the YT500 instrumental dataset.

Two phases, strictly serial (the captioner and SheetSage2 never share VRAM):

``caption`` (default ``--phase both``)
    Ensure every accepted local MP3 has a polished caption. It shells out to
    ``build_dataset.py caption``, which loads the 4-bit captioner, captions
    only the tracks that are new, unloads it, and then polishes only the tracks
    that are still pending. The album grows monotonically:
    ``work/captions/arabmaqamrock/YT500_INSTRUMENTAL.json``.

``transcribe``
    Run ``transcribe.sh --full --dtype auto`` on the accepted local audio that
    has no ``ok`` transcript yet, into ``work/transcripts/``.

The worker never touches GCS: the watcher restores its inputs
(``caption_album.json``, ``work/transcripts/``) and the publisher uploads its
outputs. It loads no model at all when a phase has nothing pending.

Usage::

    python3 scripts/yt500_learn.py --once --phase both --verify-schema
    python3 scripts/yt500_learn.py --watch --phase both \\
        --min-batch 12 --flush-minutes 20 --interval 60
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yt500_dataset as dataset  # noqa: E402
import yt500_gcs as gcsio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DATASET = REPO_ROOT / "scripts" / "build_dataset.py"
DEFAULT_TRANSCRIBE_SH = (REPO_ROOT.parent / "arabmaqamrock" / "transcribe.sh")
PHASES = ("caption", "transcribe")

REQUIRED_TRACK_KEYS = (
    "file", "duration_s", "chunk_captions", "caption_raw", "caption",
    "rewrite_status", "instruments", "vocals", "production",
)


# --------------------------------------------------------------------------- #
# pending counts
# --------------------------------------------------------------------------- #
def caption_pending(work_dir: Path) -> list[str]:
    return dataset.caption_state(work_dir, dataset.read_manifest(work_dir))[
        "pending"]


def transcribe_pending(work_dir: Path) -> list[str]:
    return dataset.transcript_state(work_dir, dataset.read_manifest(work_dir))[
        "pending"]


def pending_for(phase: str, work_dir: Path) -> list[str]:
    return caption_pending(work_dir) if phase == "caption" else (
        transcribe_pending(work_dir))


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def load_state(work_dir: Path) -> dict:
    data = gcsio.read_json(Path(work_dir) / "learn_state.json", {}) or {}
    return data if isinstance(data, dict) else {}


def save_state(work_dir: Path, state: dict) -> None:
    gcsio.write_json_atomic(Path(work_dir) / "learn_state.json", state)


def due(phase: str, pending: list[str], args: argparse.Namespace,
        state: dict) -> bool:
    """Batch small arrivals; flush the batch after ``--flush-minutes``."""
    if not pending:
        return False
    if args.once or args.min_batch <= 0:
        return True
    if len(pending) >= args.min_batch:
        return True
    last = float((state.get(phase) or {}).get("last_epoch") or 0)
    return (time.time() - last) >= args.flush_minutes * 60


# --------------------------------------------------------------------------- #
# subprocesses
# --------------------------------------------------------------------------- #
def run_cmd(cmd: list[str], log_path: Path, timeout: float) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=fh,
                                  stderr=subprocess.STDOUT, check=False,
                                  timeout=timeout or None)
        except subprocess.TimeoutExpired:
            fh.write(f"\nTIMEOUT after {timeout:.0f}s\n")
            return 124
    return proc.returncode


def caption_command(args: argparse.Namespace, work_dir: Path) -> list[str]:
    cmd = [
        sys.executable, str(BUILD_DATASET), "caption",
        "--audio-dir", str(dataset.audio_dir(work_dir)),
        "--run-manifest", str(dataset.manifest_path(work_dir)),
        "--out-dir", str(dataset.captions_dir(work_dir)),
        "--workdir", str(Path(work_dir) / "tmp"),
        # The watcher's manifest spans generation runs and carries no run_id,
        # so build_dataset would name the album "multiple". Name it exactly the
        # way dataset.album_path() does, or mirror_outputs() finds no album.
        "--artist", dataset.ARTIST,
        "--album", dataset.ALBUM,
        # The corpus is supplied by the watcher: no skips, so the track count
        # can always settle and an incremental pass is well defined.
        "--max-track-seconds", "0",
        "--min-bitrate-kbps", "0",
    ]
    if args.hf_model:
        cmd += ["--hf-model", str(args.hf_model)]
    if args.rewrite_gguf:
        cmd += ["--rewrite-gguf", str(args.rewrite_gguf)]
    if args.llamacpp_server:
        cmd += ["--llamacpp-server", str(args.llamacpp_server)]
    if args.repolish:
        cmd.append("--repolish")
    if args.no_rewrite:
        cmd.append("--no-rewrite")
    return cmd


def transcribe_command(args: argparse.Namespace, work_dir: Path,
                       files: list[str]) -> list[str]:
    audio = dataset.audio_dir(work_dir)
    return [
        str(args.transcribe_sh),
        *[str(audio / name) for name in files],
        "--full",
        "--dtype", args.dtype,
        "--out-dir", str(dataset.transcripts_dir(work_dir)),
        "--on-collision", "error",
        "--per-track-timeout", str(int(args.per_track_timeout)),
    ]


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def mirror_outputs(work_dir: Path) -> dict | None:
    """Publish the album view the watcher restores and the publisher uploads."""
    album = dataset.load_album(work_dir)
    if album is None:
        return None
    gcsio.write_json_atomic(dataset.caption_album_path(work_dir), album)

    runs = {r.get("file"): r.get("run_id")
            for r in dataset.accepted_rows(dataset.read_manifest(work_dir))}
    lines = []
    for track in album.get("tracks") or []:
        name = track.get("file")
        lines.append(json.dumps({
            "stem": Path(name).stem if name else None,
            "run_id": runs.get(name),
            "file": name,
            "duration_s": track.get("duration_s"),
            "rewrite_status": track.get("rewrite_status"),
            "caption": track.get("caption"),
            "instruments": track.get("instruments") or [],
            "vocals": track.get("vocals") or [],
            "production": track.get("production") or [],
        }, ensure_ascii=False))
    gcsio.write_text_atomic(dataset.captions_jsonl_path(work_dir),
                            "".join(line + "\n" for line in lines))
    return album


def verify_album(work_dir: Path) -> list[str]:
    """Structural checks for ``--verify-schema`` (pilot gate §8)."""
    from src.resume import (
        CAPTION_PIPELINE_VERSION,
        CAPTION_STAGE_POLISHED,
    )

    problems: list[str] = []
    album = dataset.load_album(work_dir)
    if album is None:
        return [f"no album at {dataset.album_path(work_dir)}"]
    if album.get("caption_pipeline_version") != CAPTION_PIPELINE_VERSION:
        problems.append(
            f"caption_pipeline_version {album.get('caption_pipeline_version')} "
            f"!= {CAPTION_PIPELINE_VERSION}")
    if album.get("caption_stage") != CAPTION_STAGE_POLISHED:
        problems.append(f"caption_stage {album.get('caption_stage')} is not "
                        f"'{CAPTION_STAGE_POLISHED}'")
    if album.get("failed_tracks"):
        problems.append(f"{len(album['failed_tracks'])} failed track(s)")
    for track in album.get("tracks") or []:
        name = track.get("file")
        missing = [k for k in REQUIRED_TRACK_KEYS if k not in track]
        if missing:
            problems.append(f"{name}: missing keys {missing}")
        if not (track.get("caption") or "").strip():
            problems.append(f"{name}: empty caption")
        if not track.get("chunk_captions"):
            problems.append(f"{name}: no chunk_captions kept")
    return problems


# --------------------------------------------------------------------------- #
# one pass
# --------------------------------------------------------------------------- #
def progress_count(phase: str, work_dir: Path) -> int:
    """Monotonic counter of work actually stored for a phase.

    ``pending`` is not usable for this: the watcher adds takes while a pass
    runs, so a pass that captioned 16 tracks looks like "no progress" (or
    worse, a regression) and would block the phase for the whole session.
    """
    if phase == "caption":
        album = dataset.load_album(work_dir) or {}
        return len(album.get("tracks") or [])
    state = dataset.transcript_state(work_dir, dataset.read_manifest(work_dir))
    return len(state["done"])


def learn_pass(args: argparse.Namespace, work_dir: Path, state: dict,
               blocked: set[str]) -> list[str]:
    """Run at most one bounded batch per due phase. Returns the phases run."""
    ran: list[str] = []
    for phase in args.phases:
        if phase in blocked:
            continue
        pending = pending_for(phase, work_dir)
        if not pending:
            print(f"  {phase}: nothing pending", flush=True)
            continue
        if not due(phase, pending, args, state):
            print(f"  {phase}: {len(pending)} pending, waiting for a batch of "
                  f"{args.min_batch} or {args.flush_minutes:g} min",
                  flush=True)
            continue

        batch = pending[: args.max_batch] if args.max_batch else pending
        if phase == "transcribe":
            print(f"  {phase}: {len(batch)} of {len(pending)} pending",
                  flush=True)
        else:
            # build_dataset drains the whole album: one captioner load for
            # every new take beats several.
            print(f"  {phase}: {len(pending)} pending", flush=True)
        started = time.time()
        before_count = progress_count(phase, work_dir)
        if phase == "caption":
            rc = run_cmd(caption_command(args, work_dir),
                         Path(work_dir) / "logs" / "learn_caption.log",
                         args.timeout_minutes * 60)
            album = mirror_outputs(work_dir)
            if album is not None:
                print(f"  caption: rc={rc} "
                      f"{album.get('track_count')} tracks, "
                      f"stage={album.get('caption_stage')}", flush=True)
        else:
            if not Path(args.transcribe_sh).is_file():
                raise SystemExit(
                    f"transcribe.sh not found: {args.transcribe_sh}\n"
                    "Pass --transcribe-sh or clone arabmaqamrock (§7.1).")
            rc = run_cmd(transcribe_command(args, work_dir, batch),
                         Path(work_dir) / "logs" / "learn_transcribe.log",
                         args.timeout_minutes * 60)
            done = len(dataset.transcript_state(
                work_dir, dataset.read_manifest(work_dir))["done"])
            print(f"  transcribe: rc={rc} {done} transcript(s) ok", flush=True)

        elapsed = round(time.time() - started, 1)
        state[phase] = {"last_epoch": time.time(),
                        "last_utc": gcsio.utcnow(),
                        "batch": len(batch), "rc": rc, "seconds": elapsed}
        save_state(work_dir, state)
        ran.append(phase)

        if progress_count(phase, work_dir) <= before_count:
            print(f"  {phase}: no progress this pass; not retrying it in this "
                  "session", file=sys.stderr, flush=True)
            blocked.add(phase)
    return ran


def write_status(work_dir: Path, phases: list[str]) -> None:
    manifest = dataset.read_manifest(work_dir)
    caps = dataset.caption_state(work_dir, manifest)
    trs = dataset.transcript_state(work_dir, manifest)
    gcsio.write_json_atomic(dataset.status_path(work_dir, "learn"), {
        "role": "learn",
        "utc": gcsio.utcnow(),
        "last_phases": phases,
        "captioned": len(caps["done"]),
        "transcribed": len(trs["done"]),
        "pending_caption": len(caps["pending"]),
        "pending_transcribe": len(trs["pending"]),
    })


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path,
                    default=REPO_ROOT / "work" / "yt500")
    ap.add_argument("--phase", choices=("caption", "transcribe", "both"),
                    default="both")
    ap.add_argument("--once", action="store_true",
                    help="drain every pending track, then exit")
    ap.add_argument("--watch", action="store_true",
                    help="keep running until stopped")
    ap.add_argument("--interval", type=float, default=60.0,
                    help="seconds between passes in --watch mode")
    ap.add_argument("--min-batch", type=int, default=12,
                    help="do not start a phase for fewer pending tracks "
                         "(0 = always run); --flush-minutes overrides")
    ap.add_argument("--flush-minutes", type=float, default=20.0)
    ap.add_argument("--max-batch", type=int, default=48,
                    help="transcription tracks per invocation (0 = all). "
                         "Captioning always drains the album: one captioner "
                         "load for every new take is cheaper than several")
    ap.add_argument("--timeout-minutes", type=float, default=0.0,
                    help="kill a phase subprocess after N minutes (0 = no cap)")
    ap.add_argument("--verify-schema", action="store_true",
                    help="after captioning, check the album structure and "
                         "exit non-zero on a problem")
    ap.add_argument("--transcribe-sh", type=Path,
                    default=DEFAULT_TRANSCRIBE_SH)
    ap.add_argument("--dtype", default="auto",
                    help="SheetSage2 precision: auto (L4/A100) or fp32 (T4)")
    ap.add_argument("--per-track-timeout", type=float, default=1800.0)
    ap.add_argument("--hf-model", type=Path, default=None)
    ap.add_argument("--rewrite-gguf", type=Path, default=None)
    ap.add_argument("--llamacpp-server", type=Path, default=None)
    ap.add_argument("--repolish", action="store_true",
                    help="debug: re-polish every track instead of the pending")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="debug: keep caption_raw as the caption")
    args = ap.parse_args(argv)

    if not args.once and not args.watch:
        args.once = True
    args.phases = PHASES if args.phase == "both" else (args.phase,)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)

    lock = gcsio.acquire_lock(work_dir / ".learn.lock")
    if lock is None:
        print(f"another GPU worker holds {work_dir / '.learn.lock'}; "
              "nothing to do")
        return 0

    try:
        manifest = dataset.read_manifest(work_dir)
        if not args.watch and not dataset.accepted_rows(manifest):
            raise SystemExit(
                f"{dataset.manifest_path(work_dir)} has no accepted takes; "
                "run scripts/yt500_sync.py first")
        if dataset.accepted_rows(manifest):
            print(f"work     : {work_dir}")
            print(f"accepted : {len(dataset.accepted_rows(manifest))}")
        else:
            print(f"work     : {work_dir}")
            print("accepted : 0 (waiting for the watcher)")
        print(f"phases   : {', '.join(args.phases)}")
        print(f"mode     : {'watch' if args.watch else 'once'}"
              f"{' verify-schema' if args.verify_schema else ''}")

        state = load_state(work_dir)
        blocked: set[str] = set()
        rounds = 0
        while True:
            rounds += 1
            # §9.2 starts the watcher and this worker back to back, so the
            # queue may not exist yet: wait instead of exiting.
            if not dataset.accepted_rows(dataset.read_manifest(work_dir)):
                if not args.watch:
                    break
                print("  waiting for the watcher to queue accepted takes",
                      flush=True)
                time.sleep(max(1.0, args.interval))
                continue
            ran = learn_pass(args, work_dir, state, blocked)
            write_status(work_dir, ran)

            if args.verify_schema:
                problems = verify_album(work_dir)
                if problems:
                    print("SCHEMA FAIL:", file=sys.stderr)
                    for problem in problems:
                        print(f"  - {problem}", file=sys.stderr)
                    return 1
                print("schema  : ok")

            if not args.watch:
                if not ran or blocked == set(args.phases):
                    break
                if rounds >= 200:
                    print("stopping after 200 passes", file=sys.stderr)
                    break
                continue
            time.sleep(max(1.0, args.interval))
        print(render_summary(work_dir))
        return 0
    finally:
        lock.close()


def render_summary(work_dir: Path) -> str:
    manifest = dataset.read_manifest(work_dir)
    caps = dataset.caption_state(work_dir, manifest)
    trs = dataset.transcript_state(work_dir, manifest)
    return (f"done     : {len(caps['done'])} captioned, "
            f"{len(trs['done'])} transcribed, "
            f"{len(caps['pending'])} caption pending, "
            f"{len(trs['pending'])} transcript pending")


if __name__ == "__main__":
    raise SystemExit(main())
