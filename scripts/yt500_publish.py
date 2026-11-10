#!/usr/bin/env python3
"""Upload the YT500 instrumental work state to GCS and verify it.

This is the only process that writes to GCS. It never captions, transcribes or
downloads audio, so a slow upload cannot stall the GPU and a GPU crash cannot
lose a track that was already captioned.

One pass uploads, in order:

1. the watcher's ``manifest.json``, ``rejected.json``, ``rejected_history.json``;
2. ``caption_album.json`` and ``captions.jsonl`` (only once the album file has
   been quiet for ``--settle-seconds``, so a mid-write album is never copied);
3. ``work/transcripts/<stem>/`` for stems whose ``meta.json.status`` is ``ok``
   and whose ``score.abc`` matches ``meta.abc_sha256`` -- a broken or partial
   transcript is never uploaded;
4. the final package (``work/package``) once ``_SUCCESS`` exists;
5. its own ``status_publish.json``.

Every object is verified by size + MD5 after upload and recorded in
``work/publish/ledger.jsonl``; a second pass uploads nothing. **Nothing is ever
deleted**: there is no ``--delete`` and no remote ``rm`` anywhere in this file.

Usage::

    python3 scripts/yt500_publish.py --once
    python3 scripts/yt500_publish.py --watch --interval 60
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yt500_dataset as dataset  # noqa: E402
import yt500_gcs as gcsio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILES = ("manifest.json", "rejected.json", "rejected_history.json")
TRANSCRIPT_SUFFIXES = {".abc", ".json", ".lab", ".mid", ".txt"}


def ledger_path(work_dir: Path) -> Path:
    return Path(work_dir) / "publish" / "ledger.jsonl"


def load_ledger(work_dir: Path) -> dict[str, dict]:
    rows = {}
    for row in gcsio.load_jsonl(ledger_path(work_dir)):
        remote = row.get("remote")
        if remote:
            rows[remote] = row
    return rows


def record(work_dir: Path, entry: dict) -> None:
    gcsio.append_jsonl(ledger_path(work_dir), {"utc": gcsio.utcnow(), **entry})


def upload_file(client: gcsio.Gcs, work_dir: Path, local: Path, remote: str,
                ledger: dict[str, dict], *, dry_run: bool,
                reverify: bool) -> str:
    """Upload one file if its content changed. Returns uploaded/skipped.

    The bytes are staged into a private copy first: ``manifest.json``,
    ``status_*.json`` and the album are rewritten by their own worker while this
    runs, and verifying a hash of a file that changed mid-upload aborts the
    whole pass ("md5 mismatch", 2026-11-10) for a difference that is not
    corruption. Staging makes hash, upload and verification agree on one
    snapshot; the next pass picks up whatever changed meanwhile.
    """
    digest = gcsio.sha256_file(local)
    prior = ledger.get(remote)
    if prior and prior.get("sha256") == digest and not reverify:
        return "skipped"
    if dry_run:
        print(f"  WOULD upload {remote}")
        return "dry-run"
    stage_dir = work_dir / "publish" / "staging"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / local.name
    with local.open("rb") as src, staged.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    digest = gcsio.sha256_file(staged)
    client.upload(staged, remote, verify=True)
    entry = {"kind": "file", "remote": remote, "sha256": digest,
             "bytes": local.stat().st_size, "action": "uploaded"}
    record(work_dir, entry)
    ledger[remote] = entry
    return "uploaded"


def transcript_files(work_dir: Path) -> list[tuple[Path, str]]:
    """Files of every reusable transcript, as (local, relative) pairs."""
    base = dataset.transcripts_dir(work_dir)
    if not base.is_dir():
        return []
    pairs: list[tuple[Path, str]] = []
    for stem_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if not dataset.transcript_ok(work_dir, stem_dir.name):
            continue
        for path in sorted(stem_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TRANSCRIPT_SUFFIXES:
                continue
            pairs.append((path, f"transcripts/{stem_dir.name}/"
                                f"{path.relative_to(stem_dir)}"))
    return pairs


def album_is_settled(work_dir: Path, settle_seconds: float) -> bool:
    """True once ``caption_album.json`` has been quiet for a while.

    ``learn`` mirrors the album only after its caption subprocess returns, and
    writes it atomically, so this is belt-and-braces against copying a
    half-written album.
    """
    album = dataset.caption_album_path(work_dir)
    if not album.is_file():
        return False
    return (time.time() - album.stat().st_mtime) >= settle_seconds


def publish_once(client: gcsio.Gcs, *, work_dir: Path, gcs_work: str,
                 gcs_dataset: str, ledger: dict[str, dict], args) -> dict:
    base = gcs_work.rstrip("/")
    counts = {"uploaded": 0, "skipped": 0, "dry-run": 0}
    errors: list[str] = []
    package = dataset.package_dir(work_dir)
    package_uploaded = False

    def push(local: Path, remote: str) -> None:
        action = upload_file(client, work_dir, local, remote, ledger,
                             dry_run=args.dry_run, reverify=args.reverify)
        counts[action] = counts.get(action, 0) + 1

    for name in STATE_FILES:
        local = Path(work_dir) / name
        if local.is_file():
            push(local, f"{base}/{name}")

    # Each role's status file, so the work prefix is a complete progress view.
    for local in sorted(Path(work_dir).glob("status_*.json")):
        push(local, f"{base}/{local.name}")

    if album_is_settled(work_dir, args.settle_seconds):
        for name in ("caption_album.json", "captions.jsonl"):
            local = Path(work_dir) / name
            if local.is_file():
                push(local, f"{base}/{name}")
    else:
        print("  album: still changing, not uploaded this pass")

    for local, rel in transcript_files(work_dir):
        push(local, f"{base}/{rel}")

    if (package / "_SUCCESS").is_file():
        if args.dry_run:
            print(f"  WOULD rsync {package} -> {gcs_dataset}")
        else:
            client.rsync_up(package, gcs_dataset)
            comparison = client.compare_dir(package, gcs_dataset)
            counts["package"] = 1
            counts["compare"] = {k: len(v) for k, v in comparison.items()}
            for key in ("missing", "mismatched", "extra"):
                if comparison[key]:
                    errors.append(f"package {key}: {comparison[key][:5]}")
            record(work_dir, {"kind": "package", "remote": gcs_dataset,
                              "compare": counts["compare"],
                              "action": "uploaded"})
        package_uploaded = True

    status = {
        "role": "publish",
        "utc": gcsio.utcnow(),
        "gcs_work": base,
        "gcs_dataset": gcs_dataset,
        "counts": counts,
        "package_uploaded": package_uploaded,
        "ledger_entries": len(ledger),
        "errors": errors,
        **({"dry_run": True} if args.dry_run else {}),
    }
    if not args.dry_run:
        gcsio.write_json_atomic(dataset.status_path(work_dir, "publish"),
                                status)
    print(f"  [{status['utc']}] uploaded={counts.get('uploaded', 0)} "
          f"skipped={counts.get('skipped', 0)} "
          f"package={'yes' if package_uploaded else 'no'}")
    for error in errors:
        print(f"  ERROR: {error}", file=sys.stderr)
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path,
                    default=REPO_ROOT / "work" / "yt500")
    ap.add_argument("--gcs-work", default=gcsio.WORK_PREFIX)
    ap.add_argument("--gcs-dataset", default=gcsio.DATASET_PREFIX)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--settle-seconds", type=float, default=30.0,
                    help="wait this long after the album's last write before "
                         "uploading it")
    ap.add_argument("--reverify", action="store_true",
                    help="re-check every remote object instead of trusting "
                         "the ledger")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.once and not args.watch:
        args.once = True
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    lock = None
    if not args.dry_run:
        lock = gcsio.acquire_lock(work_dir / ".publish.lock")
        if lock is None:
            print(f"another publisher holds {work_dir / '.publish.lock'}; "
                  "nothing to do")
            return 0

    try:
        client = gcsio.Gcs(dry_run=args.dry_run,
                           log=work_dir / "publish" / "gcloud.log")
        ledger = load_ledger(work_dir)
        print(f"work     : {work_dir}")
        print(f"dest     : {args.gcs_work.rstrip('/')}")
        print(f"mode     : {'watch' if args.watch else 'once'}"
              f"{' dry-run' if args.dry_run else ''}")

        started = time.monotonic()
        exit_code = 0
        while True:
            status = publish_once(client, work_dir=work_dir,
                                  gcs_work=args.gcs_work,
                                  gcs_dataset=args.gcs_dataset,
                                  ledger=ledger, args=args)
            if status["errors"]:
                exit_code = 2
            if not args.watch or args.dry_run:
                break
            if args.max_minutes and (
                    time.monotonic() - started) >= args.max_minutes * 60:
                print("time budget reached; exiting")
                break
            time.sleep(max(1.0, args.interval))
        return exit_code
    except gcsio.GcsError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        # A verification failure must not be retried blindly: the remote
        # object is the durable copy.
        return 2
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
