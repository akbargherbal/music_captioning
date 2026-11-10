#!/usr/bin/env python3
"""Read-only view of the YT500 instrumental dataset state.

``--status`` merges the four per-role ``status_<role>.json`` files with the
watcher manifest and prints the progress view of the plan's §11. It loads no
model, writes no state and needs no GPU, so it is safe to run while the three
workers are busy.

The ``work/`` layout and the "is this track done?" predicates live here so the
watcher, the GPU worker, the publisher and the assembler cannot disagree about
what ``work/audio/<stem>.mp3`` and ``work/transcripts/<stem>/meta.json`` mean.

Usage::

    python3 scripts/yt500_dataset.py --status
    python3 scripts/yt500_dataset.py --status --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yt500_gcs as gcsio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_PREFIX = gcsio.WORK_PREFIX

ARTIST = "arabmaqamrock"
ALBUM = "YT500_INSTRUMENTAL"
DATASET_NAME = "YT500 Instrumental 489"
DATASET_VERSION = "v0.1"

ROLES = ("sync", "learn", "publish", "assemble")


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def audio_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "audio"


def captions_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "captions"


def transcripts_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "transcripts"


def manifest_path(work_dir: Path) -> Path:
    return Path(work_dir) / "manifest.json"


def rejected_path(work_dir: Path) -> Path:
    return Path(work_dir) / "rejected.json"


def rejected_history_path(work_dir: Path) -> Path:
    return Path(work_dir) / "rejected_history.json"


def album_path(work_dir: Path) -> Path:
    from src.resume import album_output_path

    return album_output_path(captions_dir(work_dir), ARTIST, ALBUM)


def caption_album_path(work_dir: Path) -> Path:
    return Path(work_dir) / "caption_album.json"


def captions_jsonl_path(work_dir: Path) -> Path:
    return Path(work_dir) / "captions.jsonl"


def status_path(work_dir: Path, role: str) -> Path:
    return Path(work_dir) / f"status_{role}.json"


def package_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "package"


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def read_manifest(work_dir: Path) -> dict:
    data = gcsio.read_json(manifest_path(work_dir), {}) or {}
    if not isinstance(data, dict):
        return {}
    return data


def accepted_rows(manifest: dict) -> list[dict]:
    rows = manifest.get("accepted") or []
    return [r for r in rows if isinstance(r, dict) and r.get("file")]


def load_album(work_dir: Path) -> dict | None:
    data = gcsio.read_json(album_path(work_dir))
    return data if isinstance(data, dict) else None


def album_tracks(album: dict | None) -> dict[str, dict]:
    if not album:
        return {}
    return {t.get("file"): t for t in album.get("tracks") or []
            if isinstance(t, dict) and t.get("file")}


def caption_done(track: dict) -> bool:
    """Mirrors ``scripts.batch.rewrite_done`` for the whole album.

    A track is final when it carries a caption and its rewrite reached a
    terminal state. ``error`` and ``pending`` are retried; ``disabled`` means
    the debug ``--no-rewrite`` path and is re-polished on the next pass.
    """
    from scripts.batch import rewrite_done

    return rewrite_done(track)


def caption_state(work_dir: Path, manifest: dict) -> dict:
    """Which accepted takes have a final caption, and which are still pending."""
    rows = accepted_rows(manifest)
    album = load_album(work_dir)
    tracks = album_tracks(album)
    failed = {f.get("file") for f in (album or {}).get("failed_tracks") or []}
    audio = audio_dir(work_dir)

    local, done, pending = [], [], []
    for row in rows:
        name = row["file"]
        if not (audio / name).is_file():
            continue
        local.append(name)
        track = tracks.get(name)
        if (track is None or name in failed
                or not (track.get("caption") or "").strip()
                or not caption_done(track)):
            pending.append(name)
        else:
            done.append(name)
    return {"local": local, "done": done, "pending": pending,
            "album": album}


def transcript_meta(work_dir: Path, stem: str) -> dict | None:
    data = gcsio.read_json(transcripts_dir(work_dir) / stem / "meta.json")
    return data if isinstance(data, dict) else None


def transcript_ok(work_dir: Path, stem: str) -> bool:
    """A transcript is reusable when meta says ok and its ABC is present."""
    base = transcripts_dir(work_dir) / stem
    meta = transcript_meta(work_dir, stem)
    if not meta or meta.get("status") != "ok":
        return False
    score = base / "score.abc"
    if not score.is_file() or score.stat().st_size == 0:
        return False
    expected = meta.get("abc_sha256")
    if expected and gcsio.sha256_file(score) != expected:
        return False
    return True


def transcript_state(work_dir: Path, manifest: dict) -> dict:
    rows = accepted_rows(manifest)
    audio = audio_dir(work_dir)
    local = [r["file"] for r in rows if (audio / r["file"]).is_file()]
    done = [name for name in local
            if transcript_ok(work_dir, Path(name).stem)]
    return {"local": local, "done": done,
            "pending": [n for n in local if n not in set(done)]}


def rejected_names(work_dir: Path) -> list[str]:
    data = gcsio.read_json(rejected_path(work_dir), {}) or {}
    return [r.get("name") for r in data.get("rejected") or [] if r.get("name")]


def regen_stats(work_dir: Path, manifest: dict) -> dict:
    """Names ever rejected vs names a regen take has since rescued."""
    history = gcsio.read_json(rejected_history_path(work_dir), {}) or {}
    accepted = {r.get("name") for r in accepted_rows(manifest)}
    ever = set(history) if isinstance(history, dict) else set()
    recovered = sorted(ever & accepted)
    return {"ever_rejected": len(ever), "recovered": len(recovered),
            "open": len(ever - accepted)}


def summary(work_dir: Path) -> dict:
    manifest = read_manifest(work_dir)
    counts = manifest.get("counts") or {}
    caps = caption_state(work_dir, manifest)
    trs = transcript_state(work_dir, manifest)
    statuses = {}
    for role in ROLES:
        statuses[role] = gcsio.read_json(status_path(work_dir, role))
    return {
        "corpus": counts.get("corpus_total", 0),
        "accepted": counts.get("accepted", 0),
        "rejected": counts.get("rejected", 0),
        "pending_corpus": counts.get("pending", 0),
        "captioned": len(caps["done"]),
        "transcribed": len(trs["done"]),
        "pending_caption": len(caps["pending"]),
        "pending_transcribe": len(trs["pending"]),
        "regen": regen_stats(work_dir, manifest),
        "runs": len(manifest.get("source_runs") or []),
        "status": statuses,
    }


def _when(status: dict | None) -> str:
    if not status:
        return "never"
    return str(status.get("utc") or "never")


def render_status(work_dir: Path) -> str:
    data = summary(work_dir)
    learn = data["status"].get("learn") or {}
    last_learn = _when(learn)
    phases = learn.get("last_phases")
    if learn and phases:
        last_learn += f" ({', '.join(phases)})"
    regen = data["regen"]
    lines = [
        f"corpus         {data['corpus']}",
        f"accepted       {data['accepted']}",
        f"rejected       {data['rejected']}",
        f"captioned      {data['captioned']}",
        f"transcribed    {data['transcribed']}",
        f"pending cap    {data['pending_caption']}",
        f"pending trs    {data['pending_transcribe']}",
        f"last sync      {_when(data['status'].get('sync'))}",
        f"last learn     {last_learn}",
        f"last publish   {_when(data['status'].get('publish'))}",
        f"last assemble  {_when(data['status'].get('assemble'))}",
        f"regen          {regen['recovered']} / {regen['ever_rejected']} "
        f"recovered ({regen['open']} open)",
    ]
    for role in ROLES:
        status = data["status"].get(role) or {}
        for error in status.get("errors") or []:
            lines.append(f"! {role}: {error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true",
                    help="print the merged progress view (default action)")
    ap.add_argument("--json", action="store_true",
                    help="print the summary as JSON instead of text")
    ap.add_argument("--work-dir", type=Path,
                    default=REPO_ROOT / "work" / "yt500")
    args = ap.parse_args(argv)

    work_dir = Path(args.work_dir)
    if args.json:
        print(json.dumps(summary(work_dir), indent=2, ensure_ascii=False))
    else:
        print(render_status(work_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
