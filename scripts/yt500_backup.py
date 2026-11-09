#!/usr/bin/env python3
"""Independent GCS backup worker for the ``yt_top_500`` transcription run.

Transcription and backup are deliberately **two processes**:

    GPU    : scripts/abc_batch.py --skip-backup
             download -> transcribe -> QC -> pack locally

    backup : scripts/yt500_backup.py --watch
             archive complete tracks -> upload -> verify -> index

The worker only reads local artefacts that are already complete
(``meta.json.status == "ok"`` and a matching ``abc_sha256``).  It never touches
audio and never launches the model, so a slow upload cannot stall the GPU and a
GPU crash cannot lose a track that was already transcribed.

Remote layout (``--gcs-dest`` defaults to ``TRANSCRIPTS/YT500``)::

    <dest>/_manifest.jsonl          frozen manifest
    <dest>/_manifest.sha256
    <dest>/duplicates.json
    <dest>/shards.json
    <dest>/pilot.json
    <dest>/_index.json              backup run index (written last)
    <dest>/_index-<utc>.json        versioned copy of the same index
    <dest>/YT500/<shard>/<stem>.tar.gz
    <dest>/YT500/abc/<shard>.json

The policy is append-only.  A remote transcript whose ``abc_sha256`` differs
from the local track is a hard stop, never an overwrite.  A ledger at
``<work-dir>/backup/ledger.jsonl`` makes repeat passes idempotent and cheap.

Usage
-----
One pass (CI / debugging)::

    python3 scripts/yt500_backup.py --once

Daemon, in parallel with the GPU process::

    setsid nohup python3 -u scripts/yt500_backup.py --watch \\
      --stop-when-complete > logs/yt500_backup.out 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import abc_batch as batch  # noqa: E402

DEFAULT_DEST = "gs://akbar-december-2024-backup/TRANSCRIPTS/YT500"
BUNDLE_FILES = (
    "manifest.jsonl",
    "manifest.sha256",
    "duplicates.json",
    "shards.json",
    "pilot.json",
)
REMOTE_BUNDLE_NAMES = {
    "manifest.jsonl": "_manifest.jsonl",
    "manifest.sha256": "_manifest.sha256",
    "duplicates.json": "duplicates.json",
    "shards.json": "shards.json",
    "pilot.json": "pilot.json",
}
TERMINAL_SHARD_STATES = {"done", "partial"}


def utc_stamp() -> str:
    return batch.utcnow().replace(":", "")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def append_ledger(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_ledger(path: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    if not path.is_file():
        return entries
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            remote = row.get("remote")
            if remote:
                entries[remote] = row
    return entries


def group_tracks(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['artist']}/{row['album']}"
        groups.setdefault(key, []).append(row)
    for members in groups.values():
        members.sort(key=lambda r: (r.get("source_rank") or 0, r["file"]))
    return dict(sorted(groups.items()))


def track_dir_for(work_dir: Path, row: dict) -> Path:
    return (work_dir / "out" / row["artist"] / row["album"]
            / Path(row["file"]).stem)


def remote_track_uri(dest: str, row: dict) -> str:
    stem = Path(row["file"]).stem
    return f"{dest.rstrip('/')}/{row['artist']}/{row['album']}/{stem}.tar.gz"


def remote_packed_uri(dest: str, row: dict) -> str:
    return f"{dest.rstrip('/')}/{row['artist']}/abc/{row['album']}.json"


def remote_stat(uri: str) -> tuple[int, str | None]:
    """Return (size, md5_b64) for one remote object, or raise if missing."""
    listing = batch.run(["gcloud", "storage", "ls", "-l", uri])
    if listing.returncode != 0:
        raise batch.BatchError(f"remote object not found: {uri}")
    try:
        size = int(listing.stdout.split(None, 1)[0])
    except (ValueError, IndexError):
        raise batch.BatchError(f"cannot read remote size: {uri}")
    hashed = batch.run(["gcloud", "storage", "hash", uri])
    md5 = None
    if hashed.returncode == 0:
        for line in hashed.stdout.splitlines():
            if line.startswith("md5_hash:"):
                md5 = line.split(":", 1)[1].strip()
    return size, md5


def object_exists(uri: str) -> bool:
    return batch.run(["gcloud", "storage", "ls", "-l", uri]).returncode == 0


def remote_meta_abc_sha(uri: str, log: Path) -> str | None:
    """Read ``meta.json`` out of a remote track archive without extracting it."""
    with tempfile.TemporaryDirectory(prefix="yt500-backup-") as tmp:
        archive = Path(tmp) / "track.tar.gz"
        proc = batch.run(["gcloud", "storage", "cp", uri, str(archive)], log=log)
        if proc.returncode != 0:
            raise batch.BatchError(f"cannot download remote archive: {uri}")
        top = None
        # stdout is parsed, so these calls must not be redirected to the log.
        listing = batch.run(["tar", "-tzf", str(archive)])
        if listing.returncode != 0:
            raise batch.BatchError(f"remote archive is corrupt: {uri}")
        for name in listing.stdout.splitlines():
            parts = name.split("/")
            if len(parts) == 2 and parts[1] == "meta.json":
                top = parts[0]
                break
        if top is None:
            raise batch.BatchError(f"remote archive has no meta.json: {uri}")
        proc = batch.run(["tar", "-xzOf", str(archive), f"{top}/meta.json"])
        if proc.returncode != 0:
            raise batch.BatchError(f"cannot read meta.json from {uri}")
        try:
            meta = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise batch.BatchError(f"remote meta.json is invalid: {uri}")
        return meta.get("abc_sha256")


def ensure_bundle(outputs: Path, manifest: Path, dest: str, log: Path,
                  reverify: bool) -> list[str]:
    """Upload the frozen manifest bundle; refuse to change an uploaded object."""
    uploaded: list[str] = []
    source_dir = manifest.parent
    for name in BUNDLE_FILES:
        local = outputs / "YT500" / name
        if name == "manifest.jsonl":
            local = manifest
        elif not local.is_file():
            local = source_dir / name
        if not local.is_file():
            continue
        remote = f"{dest.rstrip('/')}/{REMOTE_BUNDLE_NAMES[name]}"
        if not reverify and object_exists(remote):
            remote_size, remote_md5 = remote_stat(remote)
            if remote_size != local.stat().st_size:
                raise batch.BatchError(
                    f"append-only violation: {remote} exists with a different "
                    f"size ({remote_size} != {local.stat().st_size})")
            local_md5 = batch.md5_b64(local)
            if remote_md5 and remote_md5 != local_md5:
                raise batch.BatchError(
                    f"append-only violation: {remote} exists with a different "
                    f"content (md5 {remote_md5} != {local_md5})")
            continue
        proc = batch.run(["gcloud", "storage", "cp", str(local), remote], log=log)
        if proc.returncode != 0:
            raise batch.BatchError(f"bundle upload failed: {remote}")
        batch.verify_archive(local, remote)
        uploaded.append(remote)
    return uploaded


def ensure_track(row: dict, track_dir: Path, dest: str, archive_root: Path,
                 ledger: dict[str, dict], ledger_path: Path, log: Path,
                 reverify: bool, dry_run: bool) -> str:
    """Upload one complete track if needed; return one of uploaded/skipped."""
    if not batch.track_complete(track_dir):
        raise batch.BatchError(f"refusing to back up incomplete track: {track_dir}")
    meta = read_json(track_dir / "meta.json") or {}
    abc_sha = meta.get("abc_sha256")
    if not abc_sha:
        raise batch.BatchError(f"{track_dir}: meta.json has no abc_sha256")
    remote = remote_track_uri(dest, row)
    prior = ledger.get(remote)

    if prior and prior.get("abc_sha256") == abc_sha and not reverify:
        return "skipped"
    if prior and prior.get("abc_sha256") not in (None, abc_sha):
        raise batch.BatchError(
            f"append-only violation: {remote} already holds abc_sha256="
            f"{prior['abc_sha256']}, local is {abc_sha}; use a new prefix")

    if object_exists(remote):
        remote_abc = remote_meta_abc_sha(remote, log)
        if remote_abc != abc_sha:
            raise batch.BatchError(
                f"append-only violation: {remote} holds abc_sha256="
                f"{remote_abc}, local is {abc_sha}")
        remote_size, _ = remote_stat(remote)
        append_ledger(ledger_path, {
            "utc": batch.utcnow(), "kind": "track", "remote": remote,
            "abc_sha256": abc_sha, "bytes": remote_size,
            "source_wav_sha256": meta.get("source_wav_sha256"),
            "action": "adopted",
        })
        ledger[remote] = {"abc_sha256": abc_sha, "bytes": remote_size}
        return "skipped"

    if dry_run:
        print(f"  WOULD upload {remote}")
        return "dry-run"

    archive = archive_root / row["artist"] / row["album"] / f"{Path(row['file']).stem}.tar.gz"
    batch.make_track_archive(track_dir.parent, track_dir, archive, log)
    proc = batch.run(["gcloud", "storage", "cp", str(archive), remote], log=log)
    if proc.returncode != 0:
        raise batch.BatchError(f"track archive upload failed: {remote}")
    batch.verify_archive(archive, remote)
    append_ledger(ledger_path, {
        "utc": batch.utcnow(), "kind": "track", "remote": remote,
        "abc_sha256": abc_sha, "bytes": archive.stat().st_size,
        "source_wav_sha256": meta.get("source_wav_sha256"),
        "action": "uploaded",
    })
    ledger[remote] = {"abc_sha256": abc_sha, "bytes": archive.stat().st_size}
    return "uploaded"


def ensure_packed(row: dict, packed: Path, dest: str, ledger: dict[str, dict],
                  ledger_path: Path, log: Path, reverify: bool,
                  dry_run: bool) -> str:
    remote = remote_packed_uri(dest, row)
    digest = batch.sha256_file(packed)
    prior = ledger.get(remote)
    if prior and prior.get("sha256") == digest and not reverify:
        return "skipped"
    if prior and prior.get("sha256") not in (None, digest):
        raise batch.BatchError(
            f"append-only violation: {remote} already holds sha256="
            f"{prior['sha256']}, local is {digest}")
    if object_exists(remote):
        remote_size, remote_md5 = remote_stat(remote)
        local_md5 = batch.md5_b64(packed)
        if remote_size != packed.stat().st_size or remote_md5 != local_md5:
            raise batch.BatchError(
                f"append-only violation: {remote} differs from {packed} "
                f"(size {remote_size} vs {packed.stat().st_size}, "
                f"md5 {remote_md5} vs {local_md5})")
        append_ledger(ledger_path, {
            "utc": batch.utcnow(), "kind": "packed", "remote": remote,
            "sha256": digest, "bytes": remote_size, "action": "adopted"})
        ledger[remote] = {"sha256": digest, "bytes": remote_size}
        return "skipped"
    if dry_run:
        print(f"  WOULD upload {remote}")
        return "dry-run"
    proc = batch.run(["gcloud", "storage", "cp", str(packed), remote], log=log)
    if proc.returncode != 0:
        raise batch.BatchError(f"packed JSON upload failed: {remote}")
    batch.verify_archive(packed, remote)
    append_ledger(ledger_path, {
        "utc": batch.utcnow(), "kind": "packed", "remote": remote,
        "sha256": digest, "bytes": packed.stat().st_size, "action": "uploaded"})
    ledger[remote] = {"sha256": digest, "bytes": packed.stat().st_size}
    return "uploaded"


def merge_remote_index(index_path: Path, dest: str, log: Path) -> None:
    """Seed the backup index from GCS once, so a fresh VM never clobbers it."""
    if index_path.is_file():
        return
    remote = f"{dest.rstrip('/')}/_index.json"
    proc = batch.run(["gcloud", "storage", "cat", remote])
    if proc.returncode != 0 or not proc.stdout.strip():
        return
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return
    payload.setdefault("version", 1)
    payload.setdefault("albums", {})
    write_json_atomic(index_path, payload)
    batch.append_log(log, f"seeded backup index from {remote}\n")


def sync_once(rows: list[dict], outputs: Path, index_path: Path, work_dir: Path,
              dest: str, log: Path, *, reverify: bool, dry_run: bool,
              backup_index: Path, ledger_path: Path,
              ledger: dict[str, dict]) -> dict:
    groups = group_tracks(rows)
    local_index = read_json(index_path) or {"version": 1, "albums": {}}
    remote_index = read_json(backup_index) or {"version": 1, "albums": {}}
    archive_root = work_dir / "archives"
    counts = {"uploaded": 0, "skipped": 0, "pending": 0, "shards": 0}
    pending_shards: list[str] = []

    for key, tracks in groups.items():
        first = tracks[0]
        for row in tracks:
            track_dir = track_dir_for(work_dir, row)
            if not batch.track_complete(track_dir):
                counts["pending"] += 1
                continue
            action = ensure_track(row, track_dir, dest, archive_root, ledger,
                                  ledger_path, log, reverify, dry_run)
            if action in counts:
                counts[action] += 1

        packed = outputs / first["artist"] / "abc" / f"{first['album']}.json"
        entry = local_index.get("albums", {}).get(key)
        if packed.is_file() and entry is not None:
            action = ensure_packed(first, packed, dest, ledger, ledger_path,
                                   log, reverify, dry_run)
            if action in counts:
                counts[action] += 1
            remote_index["albums"][key] = entry
            counts["shards"] += 1
        elif packed.is_file() or entry is not None:
            pending_shards.append(key)
        else:
            pending_shards.append(key)

    changed = (remote_index.get("albums", {})
               != (read_json(backup_index) or {}).get("albums", {}))
    if changed:
        remote_index["updated_at"] = batch.utcnow()
        if not dry_run:
            write_json_atomic(backup_index, remote_index)
    counts["pending_shards"] = len(pending_shards)
    counts["index_changed"] = changed
    return counts


def acquire_lock(path: Path) -> object | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(f"pid={os.getpid()} utc={batch.utcnow()}\n")
    fh.flush()
    return fh


def complete(rows: list[dict], outputs: Path, index_path: Path,
             backup_index: Path) -> bool:
    groups = group_tracks(rows)
    local_index = read_json(index_path) or {"albums": {}}
    backup = read_json(backup_index) or {"albums": {}}
    for key, tracks in groups.items():
        entry = local_index.get("albums", {}).get(key)
        if entry is None or entry.get("status") not in TERMINAL_SHARD_STATES:
            return False
        if key not in backup.get("albums", {}):
            return False
        packed = outputs / tracks[0]["artist"] / "abc" / f"{tracks[0]['album']}.json"
        if not packed.is_file():
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--manifest-sha256", default=None)
    ap.add_argument("--outputs", type=Path, default=batch.REPO_ROOT / "outputs")
    ap.add_argument("--index", type=Path, default=None,
                    help="local packed index (default <outputs>/YT500/abc_index.json)")
    ap.add_argument("--work-dir", type=Path, default=Path("/content/work/yt500"))
    ap.add_argument("--gcs-dest", default=DEFAULT_DEST)
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--watch", action="store_true",
                    help="keep running, rescanning until stopped")
    ap.add_argument("--stop-when-complete", action="store_true",
                    help="with --watch, exit once every shard is backed up")
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between passes in --watch mode")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop the watch loop after N minutes (0 = no limit)")
    ap.add_argument("--reverify", action="store_true",
                    help="re-check every remote object instead of trusting the ledger")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        raise batch.BatchError(f"manifest not found: {args.manifest}")
    batch.verify_manifest_sha256(args.manifest, args.manifest_sha256)
    if shutil.which("gcloud") is None:
        raise batch.BatchError("gcloud is not installed or not on PATH")

    outputs = args.outputs
    index_path = args.index or (outputs / "YT500" / "abc_index.json")
    work_dir = args.work_dir
    backup_dir = work_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    log = backup_dir / "backup.log"
    ledger_path = backup_dir / "ledger.jsonl"
    backup_index = backup_dir / "abc_index.json"
    status_path = backup_dir / "status.json"

    lock = acquire_lock(backup_dir / "backup.lock")
    if lock is None:
        print("another backup worker holds the lock; nothing to do")
        return 0

    rows = batch.load_manifest(args.manifest)
    merge_remote_index(backup_index, args.gcs_dest, log)
    ledger = load_ledger(ledger_path)
    started = time.monotonic()

    print(f"manifest : {args.manifest} ({len(rows)} tracks)")
    print(f"dest     : {args.gcs_dest}")
    print(f"work     : {work_dir}")
    print(f"mode     : {'watch' if args.watch else 'once'}"
          f"{' dry-run' if args.dry_run else ''}")

    bundles = ensure_bundle(outputs, args.manifest, args.gcs_dest, log,
                            args.reverify) if not args.dry_run else []
    if bundles:
        print(f"  bundle: {len(bundles)} object(s) uploaded")

    loops = 0
    while True:
        loops += 1
        counts = sync_once(rows, outputs, index_path, work_dir, args.gcs_dest,
                           log, reverify=args.reverify, dry_run=args.dry_run,
                           backup_index=backup_index, ledger_path=ledger_path,
                           ledger=ledger)
        index_changed = counts.pop("index_changed", False)
        if index_changed and not args.dry_run:
            batch.upload_index(backup_index, args.gcs_dest, log)
            print("  index uploaded")

        summary = {
            "utc": batch.utcnow(), "loop": loops, **counts,
            "tracks_total": len(rows),
            "ledger_entries": len(ledger),
        }
        if not args.dry_run:
            write_json_atomic(status_path, summary)
        print(f"  loop {loops}: uploaded={counts['uploaded']} "
              f"skipped={counts['skipped']} pending_tracks={counts['pending']} "
              f"pending_shards={counts['pending_shards']} "
              f"shards_done={counts['shards']}")

        if args.dry_run or not args.watch:
            break
        if args.stop_when_complete and complete(rows, outputs, index_path,
                                                backup_index):
            print("all shards backed up; exiting (--stop-when-complete)")
            break
        if args.max_minutes and (time.monotonic() - started) >= args.max_minutes * 60:
            print("time budget reached; exiting")
            break
        time.sleep(max(1.0, args.interval))

    if ledger:
        total = sum(int(v.get("bytes") or 0) for v in ledger.values())
        print(f"ledger   : {len(ledger)} object(s), {total/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except batch.BatchError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        # Exit 2 is a hard stop: a verification failure must not be retried
        # blindly, because the remote object is the durable copy.
        raise SystemExit(2)
