#!/usr/bin/env python3
"""Per-album ABC transcription driver for ephemeral Colab runtimes (§8.1).

One album is the checkpoint. For each pending album:

1. download its audio from GCS (skip-if-present, verify the manifest SHA-256).
   A single bad track is logged, recorded, and skipped; the rest of the album
   still runs and the album is packed as ``partial``,
2. restore missing transcripts from per-track remote archives (optional),
3. run ``transcribe.sh --device cuda --full`` into ``work/out/<artist>/<album>``,
4. run the §9 QC gate per track (ok / flagged / failed),
5. pack ``outputs/<artist>/abc/<album>.json`` + the local index.

**Two modes.**

*Split (recommended for the 495-track run):* pass ``--skip-backup``. This
process only downloads, transcribes, QCs and packs locally. It never writes to
GCS and never touches git. A separate ``scripts/yt500_backup.py --watch`` worker
archives complete tracks, uploads them, verifies size+md5 and maintains the
remote index — concurrently.

*Combined (legacy, small runs):* without ``--skip-backup`` this process also
archives, uploads, verifies, updates the GCS index and optionally commits. A
failed upload aborts before the next album.

Download failures only abort when they look systemic (auth/permission/disk),
when no track in the album is usable, or when more than
``--max-download-failures`` tracks fail. ABC text is never ``git add``-ed.

Usage:
    python3 scripts/abc_batch.py --skip-backup --manifest ... --device cuda
    python3 scripts/abc_batch.py --artist ELISSA --max-minutes 170
    python3 scripts/abc_batch.py --list
    python3 scripts/abc_batch.py --dry-run --artist FADL_SHAKER
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from abcscore import qc_track  # noqa: E402

DEFAULT_ROOT = "gs://akbar-december-2024-backup/DISCOGRAPHY"
DEFAULT_TRANSCRIPTS = "gs://akbar-december-2024-backup/TRANSCRIPTS"
FALLBACK_GIT_NAME = "akbargherbal"
FALLBACK_GIT_EMAIL = "akbargherbal@users.noreply.github.com"

TRANSCRIBE = "transcribe.py"


class BatchError(RuntimeError):
    """A session-stopping failure (bad upload, unpacked commit, no manifest)."""


def run(cmd: list[str], *, cwd: Path | None = None,
        log: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    if log is None:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              check=False, timeout=timeout)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                              text=True, check=False, timeout=timeout)


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise BatchError(
            f"manifest hash mismatch: {path} is {actual}, expected {expected}")


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise BatchError(f"manifest not found: {path} (run scripts/yt500_manifest.py)")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record") == "header":
                continue
            rows.append(row)
    return rows


def group_albums(rows: list[dict]) -> dict[str, list[dict]]:
    albums: dict[str, list[dict]] = {}
    for row in rows:
        albums.setdefault(f"{row['artist']}/{row['album']}", []).append(row)
    for tracks in albums.values():
        tracks.sort(key=lambda r: (r.get("source_rank") or r.get("track_no") or 0,
                                   r["file"]))
    return albums


def load_index(path: Path) -> dict:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("version", 1)
        data.setdefault("albums", {})
        return data
    return {"version": 1, "albums": {}}


def album_pending(key: str, index: dict, outputs: Path,
                  force: bool, retry_failed: bool) -> bool:
    entry = index["albums"].get(key)
    if force or not entry:
        return True
    status = entry.get("status")
    if status == "done":
        # Durable in GCS/git even if the local packed file is gone.
        return False
    if status in ("partial", "failed"):
        # Skipped by default: one poison track must not cause a retry every
        # session. --retry-failed re-runs the album to fill the gaps.
        return retry_failed
    return True


def ensure_git_identity(repo: Path) -> None:
    for field, value in (("user.name", FALLBACK_GIT_NAME),
                         ("user.email", FALLBACK_GIT_EMAIL)):
        current = run(["git", "-C", str(repo), "config", field])
        if current.returncode != 0 or not current.stdout.strip():
            run(["git", "-C", str(repo), "config", field, value])


GCS_WILDCARDS = set("*?[]")


def download_object_api(uri: str, dest: Path) -> None:
    """Download one object by exact name via the GCS JSON API.

    The gcloud CLI treats ``[``/``]``/``*``/``?`` in *both* source and
    destination as URL wildcards, so ``13.Kermalak [Remix].mp3`` cannot be
    copied with it at all. The API has no wildcard language.
    """
    if not uri.startswith("gs://"):
        raise BatchError(f"not a GCS URI: {uri}")
    bucket, _, obj = uri[5:].partition("/")
    if not bucket or not obj:
        raise BatchError(f"cannot parse GCS URI: {uri}")
    token = run(["gcloud", "auth", "print-access-token"])
    if token.returncode != 0 or not token.stdout.strip():
        raise BatchError("cannot get a gcloud access token for the GCS API")
    url = (f"https://storage.googleapis.com/storage/v1/b/"
           f"{urllib.parse.quote(bucket, safe='')}/o/"
           f"{urllib.parse.quote(obj, safe='')}?alt=media")
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token.stdout.strip()}"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=300) as response, \
                tmp.open("wb") as fh:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                fh.write(block)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise BatchError(f"GCS API download failed for {uri}: {exc}") from exc


# A download error containing one of these is not about one track: every
# subsequent download will fail the same way, so the session must stop.
SYSTEMIC_DOWNLOAD_MARKERS = (
    "401", "403", "unauthorized", "forbidden", "permission denied",
    "invalid_grant", "reauthentication", "no space left", "disk quota",
    "quota exceeded", "could not resolve host", "name or service not known",
    "connection refused", "network is unreachable",
    "cannot get a gcloud access token",
)


def is_systemic_download_error(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in SYSTEMIC_DOWNLOAD_MARKERS)


def append_log(log: Path, text: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if text and not text.endswith("\n"):
            fh.write("\n")


def _expected_source_sha(row: dict) -> str | None:
    value = row.get("source_sha256") or row.get("sha256")
    return str(value) if value else None


def _download_once(row: dict, dest: Path, log: Path) -> None:
    if GCS_WILDCARDS & set(row["source_uri"] + str(dest)):
        download_object_api(row["source_uri"], dest)
        return
    cmd = ["gcloud", "storage", "cp", row["source_uri"], str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    append_log(log, f"\n$ {' '.join(cmd)}\n{proc.stdout}{proc.stderr}")
    if proc.returncode != 0:
        lines = [line.strip() for line in
                 (proc.stderr or proc.stdout or "").splitlines() if line.strip()]
        detail = " | ".join(lines[-2:]) if lines else f"exit {proc.returncode}"
        raise BatchError(f"download failed: {row['source_uri']}: {detail}")


def download_track(row: dict, album_dir: Path, log: Path) -> None:
    """Download one track, verify its SHA-256, and retry once if corrupt.

    A local file is reused only when its hash matches the frozen manifest. A
    mismatch deletes the cache file and re-downloads; a second mismatch is a
    track failure, not a silent acceptance.
    """
    dest = album_dir / row["file"]
    expected = _expected_source_sha(row)
    if dest.is_file() and dest.stat().st_size > 0:
        if not expected or sha256_file(dest) == expected:
            return
        append_log(log, f"CORRUPT cache {dest}: sha256 mismatch; re-downloading\n")
        dest.unlink(missing_ok=True)

    last_error = ""
    for attempt in (1, 2):
        try:
            _download_once(row, dest, log)
        except BatchError as exc:
            last_error = str(exc)
            if attempt == 1:
                append_log(log, f"RETRY {row['track_id']}: {last_error}\n")
                time.sleep(1)
                continue
            raise
        if not expected:
            return
        actual = sha256_file(dest)
        if actual == expected:
            return
        dest.unlink(missing_ok=True)
        last_error = (f"sha256 mismatch for {row['file']}: "
                      f"got {actual}, expected {expected}")
        append_log(log, f"{last_error}; attempt {attempt}\n")
    raise BatchError(f"download failed after retry: {row['source_uri']}: {last_error}")


def download_album(rows: list[dict], album_dir: Path, log: Path,
                   *, strict: bool = False,
                   max_failures: int = 3) -> tuple[list[dict], list[dict]]:
    """Download what is available; return ``(available_rows, failures)``.

    A single missing/corrupt object is a data problem, not a session problem:
    log it, keep going, and let the album be packed as ``partial``. Auth,
    permission, disk and network failures abort immediately. ``strict``
    restores the old all-or-nothing behaviour.
    """
    album_dir.mkdir(parents=True, exist_ok=True)
    available: list[dict] = []
    failures: list[dict] = []
    for row in rows:
        try:
            download_track(row, album_dir, log)
            available.append(row)
        except BatchError as exc:
            message = str(exc)
            # Do not let a track name like "403.mp3" look like an HTTP error.
            probe = message.replace(row["source_uri"], "")
            if is_systemic_download_error(probe):
                raise BatchError(
                    f"systemic download failure; aborting album: {message}"
                ) from exc
            failures.append({
                "track_id": row["track_id"],
                "file": row["file"],
                "source_uri": row["source_uri"],
                "error": message,
            })
            print(f"  download failed, skipping: {row['file']}: {message}")
            append_log(log, f"SKIP {row['track_id']}: {message}\n")

    if strict and failures:
        raise BatchError(
            f"{len(failures)} download(s) failed (--strict-downloads): "
            f"{[f['file'] for f in failures][:3]}")
    if not available:
        raise BatchError(
            f"no track could be downloaded; {len(failures)} failed "
            f"(first: {failures[0]['error'] if failures else 'unknown'})")
    if len(failures) > max_failures:
        raise BatchError(
            f"{len(failures)} downloads failed (> {max_failures}); "
            f"treating as systemic")
    return available, failures


def gcs_inventory(uri: str) -> dict[str, int]:
    """``{relative_path: size}`` for everything under a GCS prefix."""
    prefix = uri.rstrip("/") + "/"
    proc = run(["gcloud", "storage", "ls", "-l", "-r", prefix + "**"])
    if proc.returncode != 0:
        raise BatchError(f"gcloud ls failed for {uri}: {proc.stderr.strip()}")
    out: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or not parts[2].startswith(prefix):
            continue
        try:
            out[parts[2][len(prefix):]] = int(parts[0])
        except ValueError:
            continue
    return out


def local_inventory(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_size
            for p in root.rglob("*") if p.is_file()}


def verify_upload(local: Path, remote_uri: str) -> None:
    expected = local_inventory(local)
    actual = gcs_inventory(remote_uri)
    missing = sorted(set(expected) - set(actual))
    wrong_size = sorted(k for k in expected if k in actual and actual[k] != expected[k])
    if missing or wrong_size:
        raise BatchError(
            f"upload verification failed for {remote_uri}: "
            f"{len(missing)} missing, {len(wrong_size)} size mismatch; "
            f"first missing={missing[:3]}")


def make_archive(out_dir: Path, archive: Path, log: Path) -> Path:
    """One deterministic-ish tar.gz per album (raw artifacts only)."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    proc = run(["tar", "-czf", str(archive), "-C", str(out_dir), "."], log=log)
    if proc.returncode != 0 or not archive.is_file():
        raise BatchError(f"archive creation failed: {archive}")
    return archive


def md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def verify_archive(local: Path, remote_uri: str) -> None:
    """Verify the one archive object by size and GCS md5 (hard stop on mismatch)."""
    listing = run(["gcloud", "storage", "ls", "-l", remote_uri])
    if listing.returncode != 0:
        raise BatchError(f"archive not visible in GCS: {remote_uri}")
    try:
        remote_size = int(listing.stdout.split(None, 1)[0])
    except (ValueError, IndexError):
        raise BatchError(f"cannot read remote size for {remote_uri}")
    if remote_size != local.stat().st_size:
        raise BatchError(
            f"archive size mismatch: local={local.stat().st_size} "
            f"remote={remote_size} ({remote_uri})")

    remote = run(["gcloud", "storage", "hash", remote_uri])
    remote_md5 = ""
    for line in remote.stdout.splitlines():
        if line.startswith("md5_hash:"):
            remote_md5 = line.split(":", 1)[1].strip()
    local_md5 = md5_b64(local)
    if remote.returncode != 0 or remote_md5 != local_md5:
        raise BatchError(
            f"archive md5 mismatch for {remote_uri}: "
            f"local={local_md5} remote={remote_md5 or 'unreadable'}")


def track_complete(track_dir: Path) -> bool:
    """A restored/cached track is reusable only when its own hashes agree."""
    meta_path = track_dir / "meta.json"
    abc_path = track_dir / "score.abc"
    if not meta_path.is_file() or not abc_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if meta.get("status") != "ok":
        return False
    expected = meta.get("abc_sha256")
    if not expected:
        return False
    return sha256_file(abc_path) == expected


def make_track_archive(out_dir: Path, track_dir: Path, archive: Path,
                       log: Path) -> Path:
    """Archive one complete track as ``<stem>.tar.gz`` (top-level ``<stem>/``)."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_name(archive.name + ".tmp")
    proc = run(["tar", "-czf", str(tmp), "-C", str(out_dir), track_dir.name],
               log=log)
    if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise BatchError(f"track archive creation failed: {archive}")
    tmp.replace(archive)
    return archive


def remote_track_archive(remote_base: str, artist: str, album: str,
                         stem: str) -> str:
    return f"{remote_base.rstrip('/')}/{artist}/{album}/{stem}.tar.gz"


def archive_track(out_dir: Path, track_dir: Path, remote_base: str,
                  artist: str, album: str, archive_root: Path,
                  log: Path) -> str:
    """Archive, upload and verify one complete track; return its remote URI."""
    if not track_complete(track_dir):
        raise BatchError(f"refusing to archive incomplete track: {track_dir}")
    archive = archive_root / artist / album / f"{track_dir.name}.tar.gz"
    make_track_archive(out_dir, track_dir, archive, log)
    remote = remote_track_archive(remote_base, artist, album, track_dir.name)
    proc = run(["gcloud", "storage", "cp", str(archive), remote], log=log)
    if proc.returncode != 0:
        raise BatchError(f"track archive upload failed: {remote}")
    verify_archive(archive, remote)
    return remote


def restore_track(out_dir: Path, stem: str, remote_uri: str,
                  log: Path) -> bool:
    """Restore one track archive into ``out_dir``; return True when usable."""
    track_dir = out_dir / stem
    if track_complete(track_dir):
        return True
    if track_dir.exists():
        # A previous partial restore must not block a fresh transcription.
        bad = out_dir / f".{stem}.restore-bad-{utcnow().replace(':', '')}"
        track_dir.replace(bad)
        log_line = f"moved incomplete restore aside: {track_dir} -> {bad}\n"
        append_log(log, log_line)

    listing = run(["gcloud", "storage", "ls", "-l", remote_uri])
    if listing.returncode != 0:
        return False
    tmp_dir = out_dir.parent / ".restore"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive = tmp_dir / f"{stem}.tar.gz"
    proc = run(["gcloud", "storage", "cp", remote_uri, str(archive)], log=log)
    if proc.returncode != 0:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = run(["tar", "-xzf", str(archive), "-C", str(out_dir)], log=log)
    archive.unlink(missing_ok=True)
    if proc.returncode != 0 or not track_complete(track_dir):
        if track_dir.exists():
            bad = out_dir / f".{stem}.restore-bad-{utcnow().replace(':', '')}"
            track_dir.replace(bad)
        return False
    return True


def restore_shard(out_dir: Path, rows: list[dict], remote_base: str,
                  artist: str, album: str, log: Path) -> int:
    restored = 0
    for row in rows:
        stem = Path(row["file"]).stem
        if track_complete(out_dir / stem):
            continue
        remote = remote_track_archive(remote_base, artist, album, stem)
        if restore_track(out_dir, stem, remote, log):
            restored += 1
    if restored:
        append_log(log, f"restored {restored} track(s) from {remote_base}\n")
    return restored


def run_transcribe(album_dir: Path, out_dir: Path, remaining_s: float,
                   amr_dir: Path, log: Path, *, device: str = "cuda",
                   on_collision: str = "error",
                   per_track_timeout: float = 0.0) -> int:
    cmd = [str(amr_dir / "transcribe.sh"), str(album_dir),
           "--device", device, "--full", "--on-collision", on_collision,
           "--out-dir", str(out_dir)]
    if per_track_timeout > 0:
        cmd += ["--per-track-timeout", str(int(per_track_timeout))]
    if remaining_s > 0:
        cmd += ["--time-budget", str(int(remaining_s))]
    proc = run(cmd, cwd=amr_dir, log=log)
    return proc.returncode


def qc_album(out_dir: Path, rows: list[dict]) -> dict[str, str]:
    """Map stem/file -> ok|flagged|failed using the §9 gate."""
    if not out_dir.is_dir():
        raise BatchError(
            f"no output directory {out_dir}; transcribe.sh produced nothing "
            f"(check the per-album log)")
    by_stem = {Path(r["file"]).stem: r for r in rows}
    result: dict[str, str] = {}
    for track_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        row = by_stem.get(track_dir.name)
        if row is None:
            # --on-collision suffix appends _2, _3 to the stem.
            candidates = [r for r in rows
                          if track_dir.name.startswith(Path(r["file"]).stem)]
            row = candidates[0] if len(candidates) == 1 else None
        if row is None:
            continue
        meta_path = track_dir / "meta.json"
        if not meta_path.is_file():
            result[row["file"]] = "failed"
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "ok":
            result[row["file"]] = "failed"
            continue
        qc = qc_track(track_dir, meta.get("duration_s"), file=row["file"])
        result[row["file"]] = "ok" if not qc["flags"] else "flagged"
    return result


def pack_album(repo: Path, outputs: Path, manifest: Path, index: Path,
               artist: str, album: str, log: Path,
               raw_gcs: str | None = None,
               raw_archive: bool = False,
               raw_root: Path | None = None) -> None:
    cmd = [sys.executable, str(repo / "scripts" / "abc_pack.py"),
           "--outputs", str(outputs), "--manifest", str(manifest),
           "--artist", artist, "--album", album, "--index", str(index)]
    if raw_root is not None:
        cmd += ["--raw-root", str(raw_root)]
    if raw_gcs:
        cmd += ["--raw-gcs", raw_gcs]
    if raw_archive:
        cmd += ["--raw-archive"]
    proc = run(cmd, log=log)
    if proc.returncode != 0:
        raise BatchError(f"abc_pack failed for {artist}/{album}")
    packed = outputs / artist / "abc" / f"{album}.json"
    if not packed.is_file():
        raise BatchError(f"packed JSON not written: {packed}")


def upload_packed(packed: Path, artist: str, album: str,
                  transcripts_root: str, log: Path) -> str:
    dest = f"{transcripts_root.rstrip('/')}/{artist}/abc/{album}.json"
    proc = run(["gcloud", "storage", "cp", str(packed), dest], log=log)
    if proc.returncode != 0:
        raise BatchError(f"packed JSON upload failed: {dest}")
    listing = run(["gcloud", "storage", "ls", "-l", dest])
    if listing.returncode != 0:
        raise BatchError(f"packed JSON not visible in GCS: {dest}")
    try:
        remote_size = int(listing.stdout.split(None, 1)[0])
    except (ValueError, IndexError):
        raise BatchError(f"cannot read remote size for {dest}")
    if remote_size != packed.stat().st_size:
        raise BatchError(f"packed JSON size mismatch at {dest}")
    return dest


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def restore_index(index_path: Path, transcripts_root: str, log: Path) -> bool:
    """Merge the GCS run index into the local index; return True if found."""
    remote = f"{transcripts_root.rstrip('/')}/_index.json"
    proc = run(["gcloud", "storage", "cat", remote])
    if proc.returncode != 0 or not proc.stdout.strip():
        append_log(log, f"no remote index at {remote}\n")
        return False
    try:
        remote_index = json.loads(proc.stdout)
    except json.JSONDecodeError:
        append_log(log, f"remote index is not JSON: {remote}\n")
        return False
    local_index = load_index(index_path)
    merged = remote_index
    merged.setdefault("version", 1)
    merged.setdefault("albums", {})
    for key, entry in local_index.get("albums", {}).items():
        old = merged["albums"].get(key)
        if old is None or entry.get("status") == "done":
            merged["albums"][key] = entry
    _write_json_atomic(index_path, merged)
    append_log(log, f"restored index {remote} "
                    f"({len(merged['albums'])} shard(s))\n")
    return True


def upload_index(index_path: Path, transcripts_root: str, log: Path) -> None:
    """Upload the run index as versioned + current, then verify size."""
    if not index_path.is_file():
        raise BatchError(f"index missing, refusing to upload: {index_path}")
    base = transcripts_root.rstrip("/")
    stamp = utcnow().replace(":", "")
    dests = [f"{base}/_index-{stamp}.json", f"{base}/_index.json"]
    for dest in dests:
        proc = run(["gcloud", "storage", "cp", str(index_path), dest], log=log)
        if proc.returncode != 0:
            raise BatchError(f"index upload failed: {dest}")
        listing = run(["gcloud", "storage", "ls", "-l", dest])
        if listing.returncode != 0:
            raise BatchError(f"index not visible in GCS: {dest}")
        try:
            remote_size = int(listing.stdout.split(None, 1)[0])
        except (ValueError, IndexError):
            raise BatchError(f"cannot read remote index size: {dest}")
        if remote_size != index_path.stat().st_size:
            raise BatchError(f"index size mismatch at {dest}")


def commit_checkpoint(repo: Path, message: str, no_push: bool,
                      log: Path, retries: int = 3,
                      index_path: Path | None = None) -> None:
    ensure_git_identity(repo)
    index_rel = "outputs/abc_index.json"
    if index_path is not None:
        try:
            index_rel = str(index_path.resolve().relative_to(repo.resolve()))
        except ValueError:
            pass
    # IMPORTANT: never `git add -A` -- it would commit the ABC text (§15.1).
    proc = run(["git", "-C", str(repo), "add", "--",
                index_rel, "scripts", "docs"], log=log)
    if proc.returncode != 0:
        raise BatchError("git add failed")
    staged = run(["git", "-C", str(repo), "diff", "--cached", "--quiet"])
    if staged.returncode != 0:
        proc = run(["git", "-C", str(repo), "commit", "-q", "-m", message],
                   log=log)
        if proc.returncode != 0:
            raise BatchError("git commit failed")
    tracked = run(["git", "-C", str(repo), "ls-files", "--error-unmatch",
                   index_rel])
    if tracked.returncode != 0:
        raise BatchError(f"{index_rel} is not tracked")
    if no_push:
        print("  (--no-push: commit kept local, not verified against origin)")
        return
    for attempt in range(1, retries + 1):
        proc = run(["git", "-C", str(repo), "push", "-u", "origin", "HEAD"],
                   log=log)
        if proc.returncode == 0:
            break
        print(f"  push failed ({attempt}/{retries})")
        time.sleep(2 * attempt)
    else:
        raise BatchError("git push failed; aborting before the next album")

    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()
    upstream = run(["git", "-C", str(repo), "rev-parse", "@{u}"])
    if upstream.returncode != 0:
        raise BatchError("no upstream after push; commit did not land")
    if head != upstream.stdout.strip():
        raise BatchError("HEAD is ahead of origin after push; commit did not land")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artist", default=None)
    ap.add_argument("--album", default=None)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop before an album that would overrun; 0 = no limit")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-failed", action="store_true",
                    help="retry albums left partial/failed (for example after "
                         "a skipped download); partial albums are skipped by "
                         "default so one poison track cannot loop every session")
    ap.add_argument("--strict-downloads", action="store_true",
                    help="abort the album if any single track fails to download")
    ap.add_argument("--max-download-failures", type=int, default=3,
                    help="treat more than N per-album download failures as "
                         "systemic (default 3)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--amr-dir", type=Path, default=Path("/content/amr"))
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--work-dir", type=Path, default=Path("/content/work"))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--manifest-sha256", default=None,
                    help="refuse to run if the manifest file hash differs")
    ap.add_argument("--outputs", type=Path, default=None)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda",
                    help="passed to transcribe.sh (default cuda)")
    ap.add_argument("--per-track-timeout", type=float, default=1800.0,
                    help="best-effort per-track alarm in seconds (0 = off)")
    ap.add_argument("--on-collision", choices=["error", "suffix"],
                    default="suffix", help="passed to transcribe.sh")
    ap.add_argument("--archive-every-track", action="store_true",
                    help="archive/upload each complete track instead of one "
                         "per-shard tar.gz")
    ap.add_argument("--restore-from-gcs", action="store_true",
                    help="restore missing local transcripts from per-track "
                         "archives before transcoding (needs "
                         "--archive-every-track)")
    ap.add_argument("--gcs-index", action="store_true",
                    help="restore/upload the run index at "
                         "<gcs-transcripts>/_index.json")
    ap.add_argument("--no-commit", action="store_true",
                    help="do not touch git; use GCS as the durable index")
    ap.add_argument("--skip-backup", action="store_true",
                    help="transcribe + QC + pack locally only: no archive, "
                         "no GCS upload, no git commit. Run "
                         "scripts/yt500_backup.py alongside this process; it "
                         "owns every write.")
    ap.add_argument("--gcs-root", default=DEFAULT_ROOT)
    ap.add_argument("--gcs-transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--no-archive", action="store_true",
                    help="upload the raw tree file-by-file instead of one "
                         "per-album tar.gz (debugging only)")
    ap.add_argument("--assumed-rtf", type=float, default=2.5,
                    help="conservative audio-seconds per GPU-second for the "
                         "pre-album budget check")
    args = ap.parse_args(argv)

    # transcribe.sh runs with cwd=amr_dir, so any work-dir path handed to it
    # (and any output path compared afterwards) must be absolute. Resolve the
    # roots once here instead of trusting the caller's relative --work-dir.
    args.repo = args.repo.resolve()
    args.amr_dir = args.amr_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.outputs is not None:
        args.outputs = args.outputs.resolve()
    if args.index is not None:
        args.index = args.index.resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()

    outputs = args.outputs or (args.repo / "outputs")
    index_path = args.index or (outputs / "abc_index.json")
    manifest_path = args.manifest
    if manifest_path is None:
        for candidate in (args.repo / "work" / "manifest.jsonl",
                          args.work_dir / "manifest.jsonl",
                          Path("/content/manifest.jsonl")):
            if candidate.is_file():
                manifest_path = candidate
                break
    if manifest_path is None:
        raise BatchError("no manifest; pass --manifest")
    manifest_path = manifest_path.resolve()
    if not (args.amr_dir / "transcribe.sh").is_file():
        raise BatchError(f"transcribe.sh not found in {args.amr_dir}")
    if args.restore_from_gcs and not (args.archive_every_track
                                      or args.skip_backup):
        raise BatchError(
            "--restore-from-gcs needs per-track archives; pair it with "
            "--archive-every-track, or with --skip-backup when a separate "
            "backup worker owns the archives")
    if args.no_archive and args.archive_every_track:
        raise BatchError("--no-archive and --archive-every-track conflict")
    if args.skip_backup and (args.archive_every_track or args.no_archive):
        raise BatchError(
            "--skip-backup conflicts with --archive-every-track/--no-archive; "
            "the backup worker archives, this process transcribes")

    verify_manifest_sha256(manifest_path, args.manifest_sha256)
    log_dir = args.work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.gcs_index:
        restore_index(index_path, args.gcs_transcripts,
                      log_dir / "index_sync.log")

    rows = load_manifest(manifest_path)
    albums = group_albums(rows)
    index = load_index(index_path)
    pending = [k for k in sorted(albums)
               if album_pending(k, index, outputs, args.force, args.retry_failed)]
    if args.artist:
        pending = [k for k in pending if k.split("/", 1)[0] == args.artist]
    if args.album:
        pending = [k for k in pending
                   if k.split("/", 1)[1] == args.album]

    if args.list or args.dry_run:
        print(f"manifest: {manifest_path} ({len(rows)} tracks, {len(albums)} albums)")
        print(f"index:    {index_path} ({len(index['albums'])} packed)")
        for key in pending:
            audio_s = sum(r.get("duration_s") or 0 for r in albums[key])
            print(f"  {'PLAN' if args.dry_run else 'PENDING'}  {key}  "
                  f"{len(albums[key])} tracks  {audio_s/3600:.2f} h")
        if not pending:
            print("  (nothing pending)")
        return 0

    if shutil.which("abc2midi") is None:
        raise BatchError(
            "abc2midi is not installed; run `apt-get install -y abcmidi` "
            "before the QC gate")

    started = time.monotonic()
    budget_s = args.max_minutes * 60 if args.max_minutes else 0.0

    for key in pending:
        artist, album = key.split("/", 1)
        tracks = albums[key]
        audio_s = sum(r.get("duration_s") or 0 for r in tracks)
        elapsed = time.monotonic() - started
        remaining = budget_s - elapsed if budget_s else 0.0
        estimate = audio_s / max(args.assumed_rtf, 0.1) + 60
        if budget_s and remaining < estimate:
            print(f"stopping before {key}: {remaining/60:.1f} min left, "
                  f"estimated {estimate/60:.1f} min (never start mid-album)")
            break

        print(f"\n== {key} ({len(tracks)} tracks, {audio_s/3600:.2f} h) ==")
        log = log_dir / f"abc_{artist}_{album}.log".replace(" ", "_")
        album_dir = args.work_dir / "audio" / artist / album
        out_dir = args.work_dir / "out" / artist / album

        available, download_failures = download_album(
            tracks, album_dir, log,
            strict=args.strict_downloads,
            max_failures=args.max_download_failures)
        if download_failures:
            print(f"  {len(download_failures)} track(s) skipped; continuing "
                  f"with {len(available)}/{len(tracks)}")

        if args.restore_from_gcs:
            restored = restore_shard(out_dir, tracks, args.gcs_transcripts,
                                     artist, album, log)
            if restored:
                print(f"  restored {restored} transcript(s) from GCS")

        t0 = time.monotonic()
        rc = run_transcribe(album_dir, out_dir, remaining, args.amr_dir, log,
                            device=args.device,
                            on_collision=args.on_collision,
                            per_track_timeout=args.per_track_timeout)
        gpu_s = time.monotonic() - t0
        statuses = qc_album(out_dir, tracks)
        n_ok = sum(v == "ok" for v in statuses.values())
        n_flagged = sum(v == "flagged" for v in statuses.values())
        n_failed = sum(v == "failed" for v in statuses.values())
        print(f"  transcribe exit={rc}  wall={gpu_s/60:.1f} min  "
              f"qc: {n_ok} ok / {n_flagged} flagged / {n_failed} failed")
        if not statuses:
            raise BatchError(f"{key}: no track output; see {log}")

        # 1) Raw artifacts land in GCS. Per-track archives bound the loss
        # window to one track. In --skip-backup mode the separate backup
        # worker owns this step; this process only records the intended URI.
        raw_gcs = f"{args.gcs_transcripts.rstrip('/')}/{artist}/{album}"
        raw_archive = True
        if args.skip_backup:
            print(f"  backup deferred to yt500_backup.py: {raw_gcs}/<stem>.tar.gz")
        elif args.no_archive:
            proc = run(["gcloud", "storage", "rsync", "-r", str(out_dir), raw_gcs],
                       log=log)
            if proc.returncode != 0:
                raise BatchError(f"raw rsync failed for {key}; aborting session")
            verify_upload(out_dir, raw_gcs)
            raw_archive = False
            print(f"  raw tree verified: {raw_gcs}")
        elif args.archive_every_track:
            archived = 0
            for row in tracks:
                stem = Path(row["file"]).stem
                track_dir = out_dir / stem
                if not track_complete(track_dir):
                    continue
                archive_track(out_dir, track_dir, args.gcs_transcripts,
                              artist, album, args.work_dir / "archives", log)
                archived += 1
            print(f"  {archived} track archive(s) verified under {raw_gcs}")
            if archived == 0:
                raise BatchError(f"{key}: no complete track could be archived")
        else:
            archive = args.work_dir / "archives" / artist / f"{album}.tar.gz"
            make_archive(out_dir, archive, log)
            raw_uri = (f"{args.gcs_transcripts.rstrip('/')}/"
                       f"{artist}/{album}.tar.gz")
            proc = run(["gcloud", "storage", "cp", str(archive), raw_uri],
                       log=log)
            if proc.returncode != 0:
                raise BatchError(f"raw archive upload failed for {key}; "
                                 f"aborting session")
            verify_archive(archive, raw_uri)
            raw_gcs = raw_uri
            raw_archive = False
            print(f"  raw archive verified ({archive.stat().st_size/1e6:.1f} MB): "
                  f"{raw_uri}")

        # 2) pack + index locally; the backup worker uploads the packed JSON
        pack_album(args.repo, outputs, manifest_path, index_path,
                   artist, album, log, raw_gcs, raw_archive=raw_archive,
                   raw_root=args.work_dir / "out")

        if args.skip_backup:
            counts = f"{n_ok} ok, {n_flagged} flagged, {n_failed} failed"
            if download_failures:
                counts += f", {len(download_failures)} skipped"
            print(f"  packed locally only: abc({artist}/{album}) {counts}")
        else:
            packed = outputs / artist / "abc" / f"{album}.json"
            upload_packed(packed, artist, album, args.gcs_transcripts, log)
            print("  packed JSON verified in GCS")

            # 3) durable index: GCS first, git optional.
            if args.gcs_index:
                upload_index(index_path, args.gcs_transcripts, log)
                print("  index uploaded to GCS")
            counts = f"{n_ok} ok, {n_flagged} flagged, {n_failed} failed"
            if download_failures:
                counts += f", {len(download_failures)} skipped"
            message = (f"abc({artist}/{album}): {n_ok + n_flagged} tracks "
                       f"({counts})")
            if args.no_commit:
                print(f"  commit skipped (--no-commit): {message}")
                if not args.gcs_index:
                    print("  WARNING: index is local only; pass --gcs-index")
            else:
                commit_checkpoint(args.repo, message, args.no_push, log,
                                  index_path=index_path)
                print(f"  committed: {message}")

        if budget_s and time.monotonic() - started >= budget_s:
            print("time budget reached; stopping cleanly after a complete album")
            break

    print(f"\ndone in {(time.monotonic()-started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        # Exit 2 is a hard stop: the supervisor must NOT restart into the next
        # album after a failed upload or an unlanded commit.
        raise SystemExit(2)
