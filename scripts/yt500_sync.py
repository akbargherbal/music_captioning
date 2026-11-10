#!/usr/bin/env python3
"""Watch the YT500 instrumental generation bucket and queue accepted takes.

This is the **only** process that reads the generation bucket, and the only one
that downloads audio. It never loads a model and never captions anything.

One pass:

1. list every object in the ``*_yt500inst*`` runs (metadata only -- the
   duration gate needs the sidecar's ``wav_duration_s``, not the MP3 bytes);
2. apply the acceptance gate of ``docs/YT500_INSTRUMENTAL_DATASET.md`` §3 to
   every take: ABC hash, ``Exit status: 0`` marker, MP3, sidecar, and
   ``min_ratio <= wav_duration_s / nominal_seconds <= max_ratio``;
3. group by corpus name and keep the accepted take with the latest ``end_utc``
   (so a regenerated take with a new seed replaces a rejected one, whenever it
   appears -- the watcher never waits for the full corpus);
4. download the accepted MP3s atomically, restoring ``work/audio`` after a VM
   loss;
5. restore ``caption_album.json`` and ``work/transcripts/`` from GCS when they
   are missing locally;
6. write ``manifest.json``, ``rejected.json`` and ``status_sync.json``.

It is deliberately indifferent to whether generation has finished: names with
no accepted take stay pending (or land in ``rejected.json`` once their run is
final), and a later regen take is picked up on the next poll.

Usage::

    python3 scripts/yt500_sync.py --once --limit 4      # pilot
    python3 scripts/yt500_sync.py --watch --interval 60 # bulk
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yt500_dataset as dataset  # noqa: E402
import yt500_gcs as gcsio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_GLOB = "*_yt500inst*"
MARKER_OK = "Exit status: 0"

#: Reasons that make a take unusable. ``pending`` is *not* here: an unfinished
#: run must not fill ``rejected.json`` with a name that is simply not ready.
REASONS = (
    "abc_mismatch",
    "missing_mp3",
    "no_marker",
    "missing_sidecar",
    "no_nominal",
    "short_take",
    "long_take",
)

CORPUS_COLUMNS = (
    "title", "source_artist", "country", "tier", "source_rank",
    "duration_s", "source_abc_sha256", "generation_status",
)
#: The corpus index calls the source length ``duration_s``; in a dataset row
#: that name is taken by the rendered audio, so the source keeps a prefix.
CORPUS_RENAMES = {"duration_s": "source_duration_s"}


# --------------------------------------------------------------------------- #
# corpus index
# --------------------------------------------------------------------------- #
def corpus_candidates(explicit: Path | None, work_dir: Path) -> list[Path]:
    """Where ``generation/index.json`` may live, most specific first."""
    candidates = [work_dir / "corpus" / "index.json"]
    if explicit is not None:
        return [Path(explicit)] + candidates
    candidates.append(REPO_ROOT.parent / "arabmaqamrock"
                      / "outputs" / "YT500" / "generation" / "index.json")
    candidates.append(Path("/content/arabmaqamrock/outputs/YT500")
                      / "generation" / "index.json")
    candidates.append(Path("outputs/YT500/generation/index.json"))
    return candidates


def load_corpus(path: Path) -> dict:
    data = gcsio.read_json(path)
    if not isinstance(data, dict) or not data.get("tracks"):
        raise SystemExit(
            f"{path}: not a YT500 generation corpus index (no tracks). "
            "Fetch it in arabmaqamrock with "
            "`python3 plumbing/yt500_generation_corpus.py --fetch`, or pass "
            "--corpus-index.")
    by_name = {t["name"]: t for t in data["tracks"] if t.get("name")}
    names = [name for name, track in by_name.items()
             if track.get("generation_status") == "ready"]
    return {
        "path": str(path),
        "sha256": gcsio.sha256_file(path),
        "version": data.get("version"),
        "generated_at": data.get("generated_at"),
        "counts": data.get("counts") or {},
        "total": len(names),
        "names": names,
        "by_name": by_name,
    }


def find_corpus(explicit: Path | None, work_dir: Path) -> Path:
    tried = corpus_candidates(explicit, work_dir)
    for path in tried:
        if Path(path).is_file():
            return Path(path)
    raise SystemExit(
        "no corpus index found; tried:\n  "
        + "\n  ".join(str(p) for p in tried)
        + "\nFetch it in arabmaqamrock (`plumbing/yt500_generation_corpus.py "
          "--fetch`) or pass --corpus-index.")


# --------------------------------------------------------------------------- #
# the acceptance gate (§3.1)
# --------------------------------------------------------------------------- #
def ratio_of(wav_duration_s, nominal_seconds) -> float | None:
    try:
        wav = float(wav_duration_s)
        nominal = float(nominal_seconds)
    except (TypeError, ValueError):
        return None
    if nominal <= 0 or wav <= 0:
        return None
    return round(wav / nominal, 4)


def evaluate_take(
    *,
    name: str,
    run_id: str,
    seed,
    take,
    file: str,
    mp3_uri: str,
    expected_abc_sha256: str | None,
    abc_sha256: str | None,
    marker_text: str | None,
    has_mp3: bool,
    sidecar: dict | None,
    nominal_seconds,
    run_final: bool,
    min_ratio: float = 0.85,
    max_ratio: float = 1.25,
) -> dict:
    """Apply the §3.1 gate to one take.

    ``pending`` means "not decided yet" (the run has not finished, so a missing
    marker or MP3 is normal). Only a finalised run can produce a rejection.
    """
    ratio = ratio_of((sidecar or {}).get("wav_duration_s"), nominal_seconds)
    result = {
        "name": name,
        "run_id": run_id,
        "seed": seed,
        "take": take,
        "file": file,
        "mp3_uri": mp3_uri,
        "abc_sha256": abc_sha256,
        "wav_duration_s": (sidecar or {}).get("wav_duration_s"),
        "nominal_seconds": nominal_seconds,
        "ratio": ratio,
        "end_utc": (sidecar or {}).get("end_utc") or "",
        "accepted": False,
        "pending": False,
        "reason": None,
    }

    def reject(reason: str) -> dict:
        result["reason"] = reason
        return result

    def undecided(reason: str) -> dict:
        result["pending"] = not run_final
        result["reason"] = reason if run_final else None
        return result

    if expected_abc_sha256 and abc_sha256 != expected_abc_sha256:
        # A score/run mismatch is a corpus error, never a usable take.
        return reject("abc_mismatch")
    if not has_mp3:
        return undecided("missing_mp3")
    if not marker_text or MARKER_OK not in marker_text:
        return undecided("no_marker")
    if not sidecar or sidecar.get("wav_duration_s") is None:
        return undecided("missing_sidecar")
    if nominal_seconds is None or float(nominal_seconds) <= 0:
        return reject("no_nominal")
    if ratio is None:
        return reject("missing_sidecar")
    if ratio < min_ratio:
        return reject("short_take")
    if ratio > max_ratio:
        return reject("long_take")
    result["accepted"] = True
    return result


def select_takes(takes: list[dict]) -> dict:
    """One accepted take per name (latest ``end_utc``) + rejected + pending."""
    accepted: dict[str, dict] = {}
    rejected: dict[str, list[dict]] = {}
    pending: dict[str, list[dict]] = {}
    replaced = 0

    for take in takes:
        name = take["name"]
        if take["accepted"]:
            prior = accepted.get(name)
            if prior is None:
                accepted[name] = take
                continue
            replaced += 1
            keep, drop = ((take, prior)
                          if (take["end_utc"], take["seed"] or 0)
                          >= (prior["end_utc"], prior["seed"] or 0)
                          else (prior, take))
            accepted[name] = keep
            continue
        if take["pending"]:
            pending.setdefault(name, []).append(take)
        else:
            rejected.setdefault(name, []).append(take)

    for name in accepted:
        rejected.pop(name, None)
        pending.pop(name, None)

    rejected_rows = []
    for name in sorted(rejected):
        takes = sorted(rejected[name],
                       key=lambda t: (t["run_id"], t["seed"] or 0))
        rejected_rows.append({
            "name": name,
            "reasons": sorted({t["reason"] for t in takes if t["reason"]}),
            "takes": [{"run_id": t["run_id"], "seed": t["seed"],
                       "take": t["take"], "reason": t["reason"],
                       "ratio": t["ratio"],
                       "wav_duration_s": t["wav_duration_s"],
                       "nominal_seconds": t["nominal_seconds"]}
                      for t in takes],
        })

    return {
        "accepted": [accepted[name] for name in sorted(accepted)],
        "rejected": rejected_rows,
        "pending": sorted(pending),
        "superseded_takes": replaced,
    }


def pick_spread(accepted: list[dict], limit: int) -> list[dict]:
    """Pick ``limit`` takes spread over length, plus the newest take.

    §8's pilot asks for a short, a median, a long and one from the active run.
    """
    if limit <= 0 or len(accepted) <= limit:
        return list(accepted)
    ordered = sorted(accepted, key=lambda t: (t["nominal_seconds"] or 0,
                                              t["name"]))
    slots = max(1, min(limit, 3))
    chosen: list[dict] = []
    for i in range(slots):
        pos = 0 if slots == 1 else round(i * (len(ordered) - 1) / (slots - 1))
        chosen.append(ordered[pos])
    seen = {t["name"] for t in chosen}
    newest = sorted((t for t in accepted if t["name"] not in seen),
                    key=lambda t: t["end_utc"], reverse=True)
    chosen.extend(newest[: max(0, limit - len(chosen))])
    if len(chosen) < limit:
        rest = sorted((t for t in accepted
                       if t["name"] not in {c["name"] for c in chosen}),
                      key=lambda t: t["name"])
        chosen.extend(rest[: limit - len(chosen)])
    return sorted(chosen, key=lambda t: t["name"])


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #
def split_run_uri(uri: str, runs_prefix: str) -> tuple[str, str] | None:
    base = runs_prefix.rstrip("/") + "/"
    if not uri.startswith(base):
        return None
    rest = uri[len(base):]
    run_id, _, rel = rest.partition("/")
    if not run_id or not rel:
        return None
    return run_id, rel


def scan_runs(client, *, runs_prefix: str, run_glob: str, corpus: dict,
              work_dir: Path, min_ratio: float, max_ratio: float) -> dict:
    """List + gate every take in every matching run. No audio download.

    The duration gate only needs the sidecar's ``wav_duration_s``, so each run
    contributes its ``*.json`` and ``*_time.txt`` files and nothing else.
    """
    pattern = f"{runs_prefix.rstrip('/')}/{run_glob}/"
    objects = client.list_objects(pattern)

    per_run: dict[str, dict[str, dict]] = {}
    for uri, info in objects.items():
        parsed = split_run_uri(uri, runs_prefix)
        if parsed is None:
            continue
        run_id, rel = parsed
        per_run.setdefault(run_id, {})[rel] = {"uri": uri, **info}

    meta_root = Path(work_dir) / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    fingerprints = gcsio.read_json(meta_root / "runs.json", {}) or {}
    takes: list[dict] = []
    source_runs: list[dict] = []
    mismatches: list[dict] = []
    warnings: list[str] = []

    for run_id in sorted(per_run):
        run_dir = f"{runs_prefix.rstrip('/')}/{run_id}"
        run_meta = meta_root / run_id
        rels = per_run[run_id]
        fingerprint = {
            "objects": len(rels),
            "updated": max((info["updated"] for info in rels.values()),
                           default=""),
        }
        # A finalised run is append-only, so its metadata never changes again:
        # one copy per run instead of one per poll.
        if not (fingerprints.get(run_id) == fingerprint
                and (run_meta / "batch_manifest.json").is_file()):
            client.copy_glob([f"{run_dir}/*.json", f"{run_dir}/*_time.txt"],
                             run_meta)
            if (run_meta / "batch_manifest.json").is_file():
                fingerprints[run_id] = fingerprint
            else:
                warnings.append(f"{run_id}: no readable batch_manifest.json")
                continue
        manifest_path = run_meta / "batch_manifest.json"
        manifest_obj = gcsio.read_json(manifest_path)
        if not isinstance(manifest_obj, dict):
            warnings.append(f"{run_id}: no readable batch_manifest.json")
            continue
        run_final = (run_meta / "batch_run_summary.json").is_file()
        run_count = 0

        for track in manifest_obj.get("tracks") or []:
            name, seed = track.get("name"), track.get("seed")
            if not name or seed is None:
                continue
            entry = corpus["by_name"].get(name)
            stem = f"{name}_{seed}"
            sidecar = gcsio.read_json(run_meta / f"{stem}.json")
            marker_path = run_meta / f"{stem}_time.txt"
            marker_text = (marker_path.read_text(encoding="utf-8",
                                                 errors="replace")
                           if marker_path.is_file() else None)
            take = evaluate_take(
                name=name,
                run_id=run_id,
                seed=seed,
                take=track.get("take"),
                file=f"{stem}.mp3",
                mp3_uri=f"{run_dir}/{stem}.mp3",
                expected_abc_sha256=(entry or {}).get(
                    "instrumental_abc_sha256"),
                abc_sha256=track.get("abc_sha256"),
                marker_text=marker_text,
                has_mp3=f"{stem}.mp3" in per_run[run_id],
                sidecar=sidecar if isinstance(sidecar, dict) else None,
                nominal_seconds=(entry or {}).get("nominal_seconds"),
                run_final=run_final,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )
            if entry is not None:
                take["corpus"] = {key: entry.get(key)
                                  for key in CORPUS_COLUMNS}
            takes.append(take)
            run_count += 1
            if take["reason"] == "abc_mismatch":
                mismatches.append({"run_id": run_id, "name": name,
                                   "seed": seed,
                                   "got": track.get("abc_sha256"),
                                   "want": (entry or {}).get(
                                       "instrumental_abc_sha256")})
        source_runs.append({
            "run_id": run_id,
            "final": run_final,
            "takes": run_count,
            "accepted": sum(1 for t in takes if t["run_id"] == run_id
                            and t["accepted"]),
        })

    gcsio.write_json_atomic(meta_root / "runs.json", fingerprints)
    return {"takes": takes, "source_runs": source_runs,
            "abc_mismatches": mismatches, "warnings": warnings}

# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def public_take(take: dict) -> dict:
    row = {
        "name": take["name"],
        "run_id": take["run_id"],
        "seed": take["seed"],
        "take": take["take"],
        "file": take["file"],
        "stem": Path(take["file"]).stem,
        "gs_uri": take["mp3_uri"],
        "abc_sha256": take["abc_sha256"],
        "wav_duration_s": take["wav_duration_s"],
        "nominal_seconds": take["nominal_seconds"],
        "take_ratio": take["ratio"],
        "end_utc": take["end_utc"],
    }
    corpus = take.get("corpus") or {}
    for key in CORPUS_COLUMNS:
        value = corpus.get(key)
        if value is not None:
            row[CORPUS_RENAMES.get(key, key)] = value
    return row


def build_manifest(*, corpus: dict, scan: dict, selection: dict,
                   accepted: list[dict], limit: int,
                   runs_prefix: str) -> dict:
    total = corpus["total"]
    rejected_names = {r["name"] for r in selection["rejected"]}
    pending_names = [n for n in corpus["names"]
                     if n not in {t["name"] for t in selection["accepted"]}
                     and n not in rejected_names]
    takes_accepted = sum(1 for t in scan["takes"] if t["accepted"])
    takes_pending = sum(1 for t in scan["takes"] if t["pending"])
    counts = {
        "corpus_total": total,
        "accepted": len(selection["accepted"]),
        "accepted_in_manifest": len(accepted),
        "rejected": len(selection["rejected"]),
        "pending": len(pending_names),
        "takes_scanned": len(scan["takes"]),
        "takes_accepted": takes_accepted,
        "takes_rejected": len(scan["takes"]) - takes_accepted - takes_pending,
        "takes_pending": takes_pending,
        "superseded_takes": selection["superseded_takes"],
        "runs": len(scan["source_runs"]),
    }
    return {
        "version": 1,
        "generated_at": gcsio.utcnow(),
        "runs_prefix": runs_prefix.rstrip("/"),
        "run_glob": RUN_GLOB,
        "corpus_total": total,
        "counts": counts,
        "take_selection": {
            "rule": "one accepted take per name, latest end_utc",
            "takes_scanned": len(scan["takes"]),
            "takes_accepted": takes_accepted,
            "takes_rejected": counts["takes_rejected"],
            "takes_pending": takes_pending,
            "superseded_takes": selection["superseded_takes"],
            "limit": limit or None,
        },
        "corpus": {key: corpus[key] for key in
                   ("path", "sha256", "version", "generated_at", "counts",
                    "total")},
        "source_runs": scan["source_runs"],
        "abc_mismatches": scan["abc_mismatches"],
        "accepted": [public_take(t) for t in accepted],
        "rejected": selection["rejected"],
        "pending": pending_names,
    }


# --------------------------------------------------------------------------- #
# downloads + restore
# --------------------------------------------------------------------------- #
def record_rejections(work_dir: Path, rejected: list[dict]) -> None:
    """Remember every name that was ever rejected, to report regen recovery."""
    path = dataset.rejected_history_path(work_dir)
    history = gcsio.read_json(path, {}) or {}
    if not isinstance(history, dict):
        history = {}
    changed = False
    for row in rejected:
        name = row.get("name")
        if name and name not in history:
            history[name] = {"first_rejected_utc": gcsio.utcnow(),
                             "reasons": row.get("reasons") or []}
            changed = True
    if changed:
        gcsio.write_json_atomic(path, history)


def download_accepted(client, *, work_dir: Path, accepted: list[dict],
                      limit: int) -> dict:
    """Fetch accepted MP3s that are not local yet. Atomic, skip-if-present."""
    downloaded, present, failed = [], 0, []
    for row in accepted:
        dest = dataset.audio_dir(work_dir) / row["file"]
        if dest.is_file() and dest.stat().st_size > 0:
            present += 1
            continue
        if limit and len(downloaded) >= limit:
            continue
        try:
            client.download(row["gs_uri"], dest)
            got = dest.stat().st_size
            if got <= 0:
                raise gcsio.GcsError(f"downloaded 0 bytes: {row['gs_uri']}")
            downloaded.append(row["stem"])
        except gcsio.GcsError as exc:
            failed.append({"stem": row["stem"], "error": str(exc)})
            if gcsio.is_systemic(str(exc)):
                raise
    return {"downloaded": downloaded, "present": present,
            "failed": failed}


def restore_state(client, *, gcs_work: str, work_dir: Path) -> dict:
    """Restore the GPU worker's inputs from GCS when they are missing.

    Only ever writes a path that does not exist locally: after the restore the
    GPU worker is the single writer of the album and the transcripts.
    """
    restored = {"album": False, "transcripts": [], "skipped": 0}
    if not gcs_work:
        return restored

    base = gcs_work.rstrip("/")
    album_local = dataset.album_path(work_dir)
    if not album_local.is_file():
        remote = f"{base}/caption_album.json"
        if client.exists(remote):
            client.download(remote, album_local)
            restored["album"] = True

    transcripts_local = Path(work_dir) / "transcripts"
    objects = client.list_objects(f"{base}/transcripts/**")
    for uri, info in sorted(objects.items()):
        rel = uri.split("/transcripts/", 1)[-1]
        if not rel:
            continue
        dest = transcripts_local / rel
        if dest.is_file() and dest.stat().st_size == info["size"]:
            restored["skipped"] += 1
            continue
        client.download(uri, dest)
        restored["transcripts"].append(rel)
    return restored


def restore_corpus_index(source: Path, work_dir: Path) -> Path:
    """Keep a byte-identical copy of the index that gated this queue."""
    dest = Path(work_dir) / "corpus" / "index.json"
    if dest.is_file() and gcsio.sha256_file(dest) == gcsio.sha256_file(source):
        return dest
    return gcsio.write_text_atomic(
        dest, Path(source).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# one pass
# --------------------------------------------------------------------------- #
def sync_once(client, *, args, corpus: dict, loop: int) -> dict:
    work_dir = Path(args.work_dir)
    scan = scan_runs(client, runs_prefix=args.runs_prefix,
                     run_glob=args.run_glob, corpus=corpus,
                     work_dir=work_dir, min_ratio=args.min_ratio,
                     max_ratio=args.max_ratio)
    for warning in scan["warnings"]:
        print(f"  WARNING: {warning}", file=sys.stderr)
    selection = select_takes(scan["takes"])
    accepted = pick_spread(selection["accepted"], args.limit)
    manifest = build_manifest(corpus=corpus, scan=scan, selection=selection,
                              accepted=accepted, limit=args.limit,
                              runs_prefix=args.runs_prefix)

    if args.dry_run:
        print(f"  [{gcsio.utcnow()}] loop {loop}: "
              f"{manifest['counts']['accepted']}/"
              f"{manifest['counts']['corpus_total']} accepted, "
              f"{manifest['counts']['rejected']} rejected, "
              f"{manifest['counts']['pending']} pending, "
              f"{manifest['counts']['takes_scanned']} take(s) scanned "
              f"(dry run, nothing written)")
        return {"counts": manifest["counts"], "downloads": {},
                "restored": {}, "manifest": manifest, "scan": scan}

    if not args.allow_abc_mismatch:
        check_systemic_abc(scan)

    # sync is the single writer of these three files.
    gcsio.write_json_atomic(dataset.manifest_path(work_dir), manifest)
    gcsio.write_json_atomic(dataset.rejected_path(work_dir), {
        "version": 1,
        "generated_at": gcsio.utcnow(),
        "count": len(selection["rejected"]),
        "rejected": selection["rejected"],
    })
    record_rejections(work_dir, selection["rejected"])

    restored = restore_state(client, gcs_work=args.gcs_work,
                             work_dir=work_dir)
    downloads = download_accepted(client, work_dir=work_dir,
                                  accepted=manifest["accepted"],
                                  limit=args.download_limit)

    status = {
        "role": "sync",
        "utc": gcsio.utcnow(),
        "loop": loop,
        "work_dir": str(work_dir),
        "runs_prefix": args.runs_prefix,
        "run_glob": args.run_glob,
        "band": [args.min_ratio, args.max_ratio],
        "counts": manifest["counts"],
        "source_runs": scan["source_runs"],
        "abc_mismatches": scan["abc_mismatches"],
        "downloaded": downloads["downloaded"],
        "download_failures": downloads["failed"],
        "restored": restored,
    }
    gcsio.write_json_atomic(dataset.status_path(work_dir, "sync"), status)

    counts = manifest["counts"]
    print(f"  [{status['utc']}] loop {loop}: "
          f"accepted={counts['accepted']}/{counts['corpus_total']} "
          f"rejected={counts['rejected']} pending={counts['pending']} "
          f"downloaded={len(downloads['downloaded'])} "
          f"local={downloads['present']} "
          f"restored={int(restored['album'])}album/"
          f"{len(restored['transcripts'])}trs", flush=True)
    for item in downloads["failed"]:
        print(f"  WARNING: download failed for {item['stem']}: "
              f"{item['error']}", file=sys.stderr)
    for item in scan["abc_mismatches"]:
        print(f"  WARNING: abc_sha256 mismatch in {item['run_id']}: "
              f"{item['name']} got {item['got']} want {item['want']} "
              "(take not used)", file=sys.stderr)
    for row in selection["rejected"]:
        print(f"  rejected {row['name']}: {', '.join(row['reasons'])} "
              f"({len(row['takes'])} take(s))")
    return {"counts": counts, "downloads": downloads,
            "restored": restored, "manifest": manifest}


def check_systemic_abc(scan: dict) -> None:
    """A corpus/run mismatch is a hard stop, not a per-take rejection."""
    if not scan["abc_mismatches"]:
        return
    names = {item["name"] for item in scan["abc_mismatches"]}
    covered = {t["name"] for t in scan["takes"] if t["accepted"]}
    if names - covered:
        sample = sorted(names - covered)[:5]
        raise gcsio.GcsError(
            f"{len(names - covered)} name(s) have no take whose abc_sha256 "
            f"matches the frozen corpus ({sample}). The run and the corpus "
            "index disagree; fix the corpus index before captioning. Use "
            "--allow-abc-mismatch to keep polling anyway.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def complete(manifest: dict) -> bool:
    counts = manifest.get("counts") or {}
    return (counts.get("accepted", 0) > 0
            and counts.get("accepted", 0) == counts.get("corpus_total"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-prefix", default=gcsio.RUNS_PREFIX,
                    help="GCS prefix that holds the generation runs")
    ap.add_argument("--run-glob", default=RUN_GLOB,
                    help=f"run directory glob (default: {RUN_GLOB})")
    ap.add_argument("--gcs-work", default=gcsio.WORK_PREFIX,
                    help="GCS prefix holding the shared work state to restore")
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "work" / "yt500")
    ap.add_argument("--corpus-index", type=Path, default=None)
    ap.add_argument("--min-ratio", type=float, default=0.85)
    ap.add_argument("--max-ratio", type=float, default=1.25)
    ap.add_argument("--limit", type=int, default=0,
                    help="queue only N accepted takes (0 = all); the pilot "
                         "uses 4 and gets a short/median/long spread")
    ap.add_argument("--download-limit", type=int, default=0,
                    help="download at most N new MP3s per pass (0 = all)")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--watch", action="store_true",
                    help="keep scanning until stopped")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop the watch loop after N minutes (0 = no limit)")
    ap.add_argument("--idle-stop-minutes", type=float, default=0.0,
                    help="with --watch, exit after N idle minutes. Only use "
                         "once every name is accepted; a rejected name is "
                         "waiting for a regen take, not dead.")
    ap.add_argument("--stop-when-complete", action="store_true",
                    help="with --watch, exit when accepted == corpus_total")
    ap.add_argument("--allow-abc-mismatch", action="store_true",
                    help="keep polling when a run's score hash disagrees with "
                         "the corpus index (takes are still never captioned)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.once and not args.watch:
        args.once = True
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    lock = None
    if not args.dry_run:
        lock = gcsio.acquire_lock(work_dir / ".sync.lock")
        if lock is None:
            print(f"another watcher holds {work_dir / '.sync.lock'}; "
                  "nothing to do")
            return 0

    try:
        source = find_corpus(args.corpus_index, work_dir)
        if not args.dry_run:
            source = restore_corpus_index(source, work_dir)
        corpus = load_corpus(source)
        if corpus["total"] == 0:
            raise SystemExit(f"{source}: no tracks with "
                             "generation_status == 'ready'")
        client = gcsio.Gcs(dry_run=args.dry_run,
                           log=work_dir / "sync_gcloud.log")

        print(f"corpus   : {source} ({corpus['total']} ready names)")
        print(f"runs     : {args.runs_prefix.rstrip('/')}/{args.run_glob}/")
        print(f"work     : {work_dir}")
        print(f"band     : {args.min_ratio} <= ratio <= {args.max_ratio}")
        print(f"mode     : {'watch' if args.watch else 'once'}"
              f"{' dry-run' if args.dry_run else ''}"
              f"{f' limit={args.limit}' if args.limit else ''}")

        started = time.monotonic()
        last_progress = started
        prior_accepted = None
        loop = 0
        while True:
            loop += 1
            pass_started = time.monotonic()
            try:
                result = sync_once(client, args=args, corpus=corpus, loop=loop)
            except gcsio.GcsError as exc:
                if gcsio.is_systemic(str(exc)):
                    print(f"STOP: {exc}", file=sys.stderr)
                    return 2
                print(f"WARNING: {exc}", file=sys.stderr)
                if not args.watch:
                    return 2
                time.sleep(max(1.0, args.interval))
                continue

            accepted = result["counts"].get("accepted", 0)
            if accepted != prior_accepted:
                prior_accepted = accepted
                last_progress = time.monotonic()

            if args.dry_run or not args.watch:
                break
            if args.stop_when_complete and complete(result["manifest"]):
                print("all names accepted; exiting (--stop-when-complete)")
                break
            if args.max_minutes and (
                    time.monotonic() - started) >= args.max_minutes * 60:
                print("time budget reached; exiting")
                break
            if args.idle_stop_minutes and (
                    time.monotonic() - last_progress
            ) >= args.idle_stop_minutes * 60:
                print(f"no new accepted take for {args.idle_stop_minutes:g} "
                      "min; exiting")
                break
            elapsed = time.monotonic() - pass_started
            time.sleep(max(1.0, args.interval - elapsed))
        return 0
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
