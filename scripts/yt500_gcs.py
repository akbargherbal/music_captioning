#!/usr/bin/env python3
"""GCS access for the YT500 instrumental dataset workers.

Only two roles speak to GCS: the watcher (``yt500_sync.py``, read + restore)
and the publisher (``yt500_publish.py``, write + verify). The GPU worker never
does. This module keeps that contract in one place:

* object listing that returns ``size``/``updated`` without downloading bytes
  (so a duration gate can run on metadata alone);
* small-object reads (``cat``) and atomic downloads (``<name>.part``, fsync,
  rename);
* uploads verified by size + MD5, and a directory comparison for the final
  package;
* a per-role ``flock`` so a second instance exits instead of double-writing.

Everything shells out to ``gcloud`` -- the same tool the rest of the repo uses
-- except a download whose path contains a wildcard character, which goes
through the JSON API because the CLI cannot address such an object at all.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BUCKET = "gs://akbar-december-2024-backup"
RUNS_PREFIX = f"{DEFAULT_BUCKET}/arabmaqamrock/runs"
WORK_PREFIX = f"{DEFAULT_BUCKET}/GENERATION/YT500_INSTRUMENTAL/work"
DATASET_PREFIX = f"{DEFAULT_BUCKET}/DATASET/YT500_INSTRUMENTAL"

GCS_WILDCARDS = set("*?[]")
SYSTEMIC_MARKERS = (
    "401", "403", "unauthorized", "forbidden", "permission denied",
    "invalid_grant", "reauth", "could not resolve host",
    "name or service not known", "connection refused",
    "network is unreachable", "no space left", "disk quota",
)


class GcsError(RuntimeError):
    """A network, credential or remote-state problem the operator must see."""


def utcnow() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5_b64(path: Path) -> str:
    import base64

    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return base64.b64encode(h.digest()).decode("ascii")


def is_systemic(message: str) -> bool:
    """A failure that will repeat for every object, so a loop must stop."""
    low = (message or "").lower()
    return any(marker in low for marker in SYSTEMIC_MARKERS)


def _run(cmd: list[str], timeout: float | None = None
         ) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False,
                          timeout=timeout)


def _fsync(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# local state files
# --------------------------------------------------------------------------- #
def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json_atomic(path: Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_text_atomic(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def append_jsonl(path: Path, entry: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def acquire_lock(path: Path):
    """Take an exclusive non-blocking lock; return the handle or ``None``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(f"pid={os.getpid()} utc={utcnow()}\n")
    fh.flush()
    return fh


# --------------------------------------------------------------------------- #
# remote objects
# --------------------------------------------------------------------------- #
def parse_ls_line(line: str) -> dict | None:
    """Parse one ``gcloud storage ls -l`` row into uri/size/updated."""
    parts = line.split()
    if len(parts) < 3 or not parts[-1].startswith("gs://"):
        return None
    try:
        size = int(parts[0])
    except ValueError:
        return None
    return {"uri": parts[-1], "size": size, "updated": parts[1]}


class Gcs:
    """Thin, testable wrapper around the ``gcloud storage`` CLI."""

    def __init__(self, *, dry_run: bool = False, log: Path | None = None,
                 timeout: float = 600.0):
        self.dry_run = dry_run
        self.log = Path(log) if log else None
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------- #
    def _exec(self, cmd: list[str], timeout: float | None = None
              ) -> subprocess.CompletedProcess:
        if shutil.which("gcloud") is None:
            raise GcsError("gcloud is not installed or not on PATH")
        proc = _run(cmd, timeout=timeout or self.timeout)
        if self.log is not None:
            self._log(f"$ {' '.join(cmd)}\nexit={proc.returncode}")
        return proc

    def _log(self, text: str) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n")

    # -- read -------------------------------------------------------------- #
    def list_objects(self, pattern: str) -> dict[str, dict]:
        """``{uri: {size, updated}}`` for every object under a prefix/glob."""
        proc = self._exec(
            ["gcloud", "storage", "ls", "-l", "-r", pattern])
        if proc.returncode != 0:
            text = f"{proc.stdout}\n{proc.stderr}".lower()
            if "matched no objects" in text:
                return {}
            raise GcsError(f"cannot list {pattern}: {proc.stderr.strip()}")
        objects: dict[str, dict] = {}
        for line in proc.stdout.splitlines():
            info = parse_ls_line(line)
            if info and not info["uri"].endswith("/"):
                objects[info["uri"]] = {"size": info["size"],
                                        "updated": info["updated"]}
        return objects

    def stat(self, uri: str) -> dict | None:
        proc = self._exec(["gcloud", "storage", "ls", "-l", uri])
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            info = parse_ls_line(line)
            if info and info["uri"] == uri:
                return {"size": info["size"], "updated": info["updated"]}
        return None

    def exists(self, uri: str) -> bool:
        return self.stat(uri) is not None

    def cat(self, uri: str, *, missing_ok: bool = False) -> str | None:
        proc = self._exec(["gcloud", "storage", "cat", uri])
        if proc.returncode != 0:
            if missing_ok:
                return None
            raise GcsError(f"cannot read {uri}: {proc.stderr.strip()}")
        return proc.stdout

    def cat_json(self, uri: str, *, missing_ok: bool = True) -> dict | None:
        text = self.cat(uri, missing_ok=missing_ok)
        if text is None:
            return None
        try:
            data = json.loads(text)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def md5(self, uri: str) -> str | None:
        proc = self._exec(["gcloud", "storage", "hash", uri])
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            if line.startswith("md5_hash:"):
                return line.split(":", 1)[1].strip() or None
        return None

    # -- transfer ---------------------------------------------------------- #
    def download(self, uri: str, dest: Path) -> Path:
        """Download to ``<dest>.part``, fsync, then rename. Never partial."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            return dest
        tmp = dest.with_name(dest.name + ".part")
        try:
            if GCS_WILDCARDS & set(uri + str(tmp)):
                self._download_api(uri, tmp)
            else:
                proc = self._exec(["gcloud", "storage", "cp", uri, str(tmp)])
                if proc.returncode != 0:
                    raise GcsError(
                        f"download failed: {uri}: {proc.stderr.strip()}")
            _fsync(tmp)
            os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return dest

    def _download_api(self, uri: str, dest: Path) -> None:
        bucket, _, obj = uri[5:].partition("/")
        if not bucket or not obj:
            raise GcsError(f"cannot parse GCS URI: {uri}")
        token = self._exec(["gcloud", "auth", "print-access-token"])
        if token.returncode != 0 or not token.stdout.strip():
            raise GcsError("cannot get a gcloud access token for the GCS API")
        url = (f"https://storage.googleapis.com/storage/v1/b/"
               f"{urllib.parse.quote(bucket, safe='')}/o/"
               f"{urllib.parse.quote(obj, safe='')}?alt=media")
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token.stdout.strip()}"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response, \
                    dest.open("wb") as fh:
                shutil.copyfileobj(response, fh, 1 << 20)
        except (urllib.error.URLError, OSError) as exc:
            raise GcsError(f"GCS API download failed for {uri}: {exc}") from exc

    def upload(self, local: Path, uri: str, *, verify: bool = True) -> dict:
        local = Path(local)
        digest = md5_b64(local)
        size = local.stat().st_size
        if self.dry_run:
            return {"uri": uri, "bytes": size, "md5": digest,
                    "action": "dry-run"}
        proc = self._exec(["gcloud", "storage", "cp", str(local), uri])
        if proc.returncode != 0:
            raise GcsError(f"upload failed: {uri}: {proc.stderr.strip()}")
        if verify:
            self.verify_upload(local, uri, md5=digest)
        return {"uri": uri, "bytes": size, "md5": digest, "action": "uploaded"}

    def verify_upload(self, local: Path, uri: str,
                      md5: str | None = None) -> dict:
        remote = self.stat(uri)
        if remote is None:
            raise GcsError(f"uploaded object is missing: {uri}")
        size = Path(local).stat().st_size
        if remote["size"] != size:
            raise GcsError(
                f"upload size mismatch: {uri} is {remote['size']}, "
                f"local is {size}")
        remote_md5 = self.md5(uri)
        want = md5 or md5_b64(Path(local))
        if remote_md5 and remote_md5 != want:
            raise GcsError(
                f"upload md5 mismatch: {uri} is {remote_md5}, local is {want}")
        return {"uri": uri, "bytes": size, "md5": want}

    def rsync_up(self, local_dir: Path, uri: str) -> None:
        """Append-only upload: no ``--delete``, checksum comparison."""
        local_dir = Path(local_dir)
        if self.dry_run:
            return
        proc = self._exec(
            ["gcloud", "storage", "rsync", "-r", "--checksums-only",
             str(local_dir), uri])
        if proc.returncode != 0:
            raise GcsError(f"rsync failed: {uri}: {proc.stderr.strip()}")

    def copy_glob(self, patterns, dest_dir: Path) -> bool:
        """Copy every object matching each pattern into a local directory.

        Used for the dozens of tiny metadata objects of one run in a single
        call (a per-object ``cat`` costs one CLI start-up each). Returns True
        when at least one pattern matched something.
        """
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        proc = self._exec(
            ["gcloud", "storage", "cp", *patterns, str(dest_dir) + "/"])
        if proc.returncode == 0:
            return True
        text = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
        if "matched no objects" in text.lower():
            return False
        if is_systemic(text):
            raise GcsError(f"cannot read {patterns}: {text}")
        # A transient failure leaves whatever did copy; a missing file only
        # makes a take look pending, which is the safe direction.
        return False

    def compare_dir(self, local_dir: Path, uri: str) -> dict:
        """Size + MD5 comparison against a remote prefix (0/0/0 when equal)."""
        local_dir = Path(local_dir)
        remote = self.list_objects(uri.rstrip("/") + "/**")
        base = uri.rstrip("/") + "/"
        seen: set[str] = set()
        missing: list[str] = []
        mismatched: list[dict] = []
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(local_dir))
            remote_uri = base + rel
            info = remote.get(remote_uri)
            seen.add(remote_uri)
            if info is None:
                missing.append(rel)
                continue
            if info["size"] != path.stat().st_size:
                mismatched.append({"path": rel, "why": "size",
                                   "remote": info["size"],
                                   "local": path.stat().st_size})
                continue
            remote_md5 = self.md5(remote_uri)
            if remote_md5 and remote_md5 != md5_b64(path):
                mismatched.append({"path": rel, "why": "md5"})
        extra = sorted(set(remote) - seen)
        return {"missing": missing, "mismatched": mismatched, "extra": extra}


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(__doc__)
    sys.exit(0)
