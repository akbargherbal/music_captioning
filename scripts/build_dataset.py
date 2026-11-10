#!/usr/bin/env python3
"""Build a caption + ABC dataset for one generated-audio run.

Two strictly ordered phases, mirroring ``scripts/batch.py`` but without git,
without a manifest and without any GCS write:

``caption`` (GPU)
    Caption every track of a local audio directory with the 4-bit ACE-Step
    captioner, then unload it and polish the multi-chunk captions with the small
    rewrite model. The result is one album JSON in ``--out-dir`` (the same
    schema as ``outputs/<artist>/<album>.json``), resumable and atomic.

``assemble`` (CPU only)
    Join that album JSON with (a) the reference ABC score that generated each
    track, taken from the run manifest and verified by SHA-256, and (b) the
    SheetSage2 ABC transcription of the rendered audio (with its QC block), then
    write a self-contained dataset package (JSONL + Parquet + ABC files +
    per-track caption JSON + manifest + dataset card).

Nothing is uploaded here: inspect the package, then ``gcloud storage rsync`` it.

Usage:
    python3 scripts/build_dataset.py caption --audio-dir work/dataset/audio \\
        --run-manifest work/dataset/run_manifest.json

    python3 scripts/build_dataset.py assemble \\
        --run-manifest work/dataset/run_manifest.json \\
        --audio-dir work/dataset/audio --captions work/dataset/captions \\
        --abc-ref work/dataset/abc_ref --transcripts work/dataset/transcripts \\
        --out work/dataset/package
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
DEFAULT_HF_MODEL = REPO_ROOT / "models" / "Ace-Step-Captioner-4bit"
DEFAULT_REWRITE_GGUF = REPO_ROOT / "models" / "Qwen3-4B-Instruct-2507-Q8_0.gguf"
DEFAULT_LLAMACPP_SERVER = (
    REPO_ROOT / "llama.cpp-text" / "build" / "bin" / "llama-server"
)

#: Optional per-track columns carried by a corpus index / take-selection
#: manifest. A single-run manifest has none of them, so the reference
#: ``fadhl_shaker_instrumental_18`` schema stays byte-compatible.
CORPUS_COLUMNS = (
    "title", "source_artist", "country", "tier", "source_rank",
    "source_duration_s", "source_abc_sha256", "generation_status",
    "take_ratio", "take",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_json(path: Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_run_manifest(path: Path) -> dict:
    """Load a run manifest, accepting the watcher's ``accepted`` list too.

    A single-run manifest carries ``tracks`` and a ``run_dir``; the multi-run
    queue written by ``yt500_sync.py`` carries ``accepted`` (one take per
    corpus name) and each entry names its own ``run_id``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tracks = data.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        tracks = data.get("accepted")
    if not isinstance(tracks, list) or not tracks:
        raise SystemExit(f"{path}: no tracks/accepted in run manifest")
    data["tracks"] = tracks
    return data


def run_id_from_manifest(manifest: dict) -> str:
    run_dir = manifest.get("run_dir") or ""
    name = Path(run_dir).name
    if name:
        return name
    explicit = manifest.get("run_id")
    if explicit:
        return str(explicit)
    if manifest.get("accepted"):
        # One queue can span runs; the per-track run_id is the real one.
        return "multiple"
    raise SystemExit("run manifest has no run_dir; pass --run-id")


def audio_files(audio_dir: Path) -> list[Path]:
    files = sorted(p for p in Path(audio_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise SystemExit(f"no audio files under {audio_dir}")
    return files


def stem_for(track: dict) -> str:
    """``al_hal_1486538140.mp3`` -> ``al_hal_1486538140``."""
    return Path(track["file"]).stem if "file" in track else Path(track["audio"]).stem


# --------------------------------------------------------------------------- #
# phase 1: caption + polish
# --------------------------------------------------------------------------- #
def already_polished(data: dict, track_count: int) -> bool:
    """True when a stored album can be reused as-is.

    A failed track must force a retry: the caption phase isolates failures, so
    an album with ``failed_tracks`` can have the current stage and the right
    number of tracks while still missing a caption.
    """
    from src.resume import CAPTION_STAGE_POLISHED, is_current_caption_output

    return (is_current_caption_output(data)
            and data.get("caption_stage") == CAPTION_STAGE_POLISHED
            and len(data.get("tracks") or []) == track_count
            and not data.get("failed_tracks"))


def caption_phase(args: argparse.Namespace) -> int:
    from src.captioner_hf import HFCaptioner
    from src.resume import (
        album_output_path,
        atomic_write_json as resume_atomic_write,
        is_current_caption_output,
    )
    from scripts.batch import process_album  # noqa: E402  (import after sys.path)

    manifest = load_run_manifest(args.run_manifest)
    run_id = args.run_id or run_id_from_manifest(manifest)
    artist = args.artist or "arabmaqamrock"
    album = args.album or run_id.split("_", 1)[-1]
    entry = {
        "artist": artist,
        "album": album,
        "source_dir": str(Path(args.audio_dir).resolve()),
        "tracks": [{"filename": p.name, "uri": str(p.resolve())}
                   for p in audio_files(args.audio_dir)],
    }
    out_dir = Path(args.out_dir)
    out_path = album_output_path(out_dir, artist, album)

    existing = None
    if out_path.is_file() and not args.force:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        if already_polished(data, len(entry["tracks"])):
            print(f"already polished: {out_path} ({len(data['tracks'])} tracks)")
            return 0
        if is_current_caption_output(data) and data.get("tracks"):
            print(f"resuming {out_path}: reusing stored captions")
            existing = data

    if args.max_tracks and args.max_tracks > 0:
        # Bound the pass: the album is written and polished per pass, so a
        # 279-track drain leaves hours with no final caption and no upload.
        stored = {t.get("file") for t in (existing or {}).get("tracks") or []}
        todo = [t for t in entry["tracks"] if t["filename"] not in stored]
        entry["tracks"] = todo[: args.max_tracks]
        print(f"  bounded pass: {len(entry['tracks'])} of {len(todo)} pending "
              f"tracks (--max-tracks {args.max_tracks})", flush=True)

    captioner = None
    prior_files = {t.get("file") for t in (existing or {}).get("tracks") or []}
    failed_files = {f.get("file") for f in
                    (existing or {}).get("failed_tracks") or []}
    pending = [t for t in entry["tracks"]
               if t["filename"] not in prior_files
               or t["filename"] in failed_files]
    if pending:
        captioner = HFCaptioner(model_dir=Path(args.hf_model))
    else:
        # Everything is already captioned: do not load 6.4 GB of weights just
        # to polish, and never touch the GPU on a no-op pass.
        print(f"  no track needs captioning ({len(entry['tracks'])} stored); "
              "skipping the captioner load")
    rewriter = None
    if not args.no_rewrite:
        from src.rewriter_llama import LlamaCppRewriter

        if not Path(args.rewrite_gguf).is_file():
            raise SystemExit(f"rewrite GGUF not found: {args.rewrite_gguf}")
        if not Path(args.llamacpp_server).is_file():
            raise SystemExit(f"llama-server not found: {args.llamacpp_server}")
        rewriter = LlamaCppRewriter(
            model_path=Path(args.rewrite_gguf),
            server_bin=Path(args.llamacpp_server),
        )

    def checkpoint(obj: dict, stage: str) -> None:
        resume_atomic_write(out_path, obj)
        print(f"  checkpoint [{stage}]: {out_path}", flush=True)

    obj = process_album(
        captioner, entry, Path(args.workdir), out_dir,
        existing=existing,
        retry_only_failed=bool(existing),
        max_track_seconds=args.max_track_seconds,
        min_bitrate_kbps=args.min_bitrate_kbps,
        rewriter=rewriter,
        rewrite=not args.no_rewrite,
        only_pending_polish=bool(existing) and not args.repolish,
        checkpoint=checkpoint,
    )
    resume_atomic_write(out_path, obj)
    print(f"wrote {out_path}: {obj['track_count']} tracks, "
          f"status={obj['status']}, stage={obj['caption_stage']}")
    if obj.get("failed_tracks"):
        print(f"WARNING: {len(obj['failed_tracks'])} track(s) failed", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# phase 2: assemble
# --------------------------------------------------------------------------- #
def find_album_json(captions_dir: Path) -> Path:
    hits = sorted(Path(captions_dir).glob("*/*.json"))
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one album JSON under {captions_dir}, got {hits}")
    return hits[0]


def copy_verified(src: Path, dest: Path) -> dict:
    text = Path(src).read_text(encoding="utf-8")
    atomic_write_text(dest, text)
    return {"path": dest.name, "sha256": sha256_file(dest),
            "bytes": dest.stat().st_size}


def assemble_phase(args: argparse.Namespace) -> int:
    from abcscore import qc_track

    manifest = load_run_manifest(args.run_manifest)
    run_id = args.run_id or run_id_from_manifest(manifest)
    album_path = find_album_json(args.captions)
    album = json.loads(album_path.read_text(encoding="utf-8"))
    track_captions = {t["file"]: t for t in album.get("tracks") or []}
    if not track_captions:
        raise SystemExit(f"{album_path}: no captioned tracks")

    out = Path(args.out)
    for sub in ("abc", "abc_transcription", "captions"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    def instrumental_for(run: str) -> bool:
        # An explicit flag wins; otherwise a run label that says so decides
        # (`fadhl_shaker_instrumental_18`, `20261110-083615_yt500inst`).
        if args.instrumental is not None:
            return bool(args.instrumental)
        label = str(run).lower()
        return "instrumental" in label or "inst" in label

    rows: list[dict] = []
    files: list[dict] = []
    missing: list[dict] = []

    available = {p.name: p for p in audio_files(args.audio_dir)}

    def resolve_audio(track: dict) -> str | None:
        name = track.get("file") or track.get("audio")
        if name and name in available:
            return name
        expected = f"{track['name']}_{track.get('seed')}"
        hits = [n for n in available if Path(n).stem == expected]
        if not hits:
            hits = [n for n in available
                    if Path(n).stem.startswith(f"{track['name']}_")]
        return hits[0] if len(hits) == 1 else None

    for track in manifest["tracks"]:
        audio_name = resolve_audio(track)
        if audio_name is None:
            missing.append({"track": track["name"],
                            "reason": "no matching audio file"})
            continue
        stem = Path(audio_name).stem
        audio_path = Path(args.audio_dir) / audio_name
        cap = track_captions.get(audio_name)
        if cap is None:
            missing.append({"track": stem, "reason": "no caption"})
            continue
        if not audio_path.is_file():
            missing.append({"track": stem, "reason": "no local audio"})
            continue

        abc_name = track.get("score_alias") or track["name"]
        abc_src = Path(args.abc_ref) / f"{abc_name}.abc"
        if not abc_src.is_file():
            missing.append({"track": stem, "reason": "no reference abc"})
            continue
        abc_dest = out / "abc" / f"{stem}.abc"
        abc_info = copy_verified(abc_src, abc_dest)
        expected_abc = track.get("abc_sha256")
        if expected_abc and abc_info["sha256"] != expected_abc:
            raise SystemExit(
                f"reference ABC hash mismatch for {abc_name}: "
                f"{abc_info['sha256']} != run manifest {expected_abc}")

        transcript_dir = Path(args.transcripts) / stem
        abc_tr = None
        if (transcript_dir / "score.abc").is_file():
            tr_dest = out / "abc_transcription" / f"{stem}.abc"
            tr_info = copy_verified(transcript_dir / "score.abc", tr_dest)
            meta = {}
            meta_path = transcript_dir / "meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            duration = cap.get("duration_s") or meta.get("duration_s")
            abc_tr = {
                **tr_info,
                "qc": qc_track(transcript_dir, duration, file=audio_name),
                "model": meta.get("model"),
                "model_revision": meta.get("model_revision"),
                "seconds": meta.get("seconds"),
                "status": meta.get("status"),
            }

        audio_sha = sha256_file(audio_path)
        # A queue can span runs: the take knows which run produced it.
        row_run_id = track.get("run_id") or run_id
        gs_uri = (track.get("gs_uri")
                  or f"{args.source_uri.rstrip('/')}/{audio_name}")
        caption_json = {
            "track_id": f"{row_run_id}/{stem}",
            "run_id": row_run_id,
            "name": track["name"],
            "seed": track.get("seed"),
            "audio": {
                "file": audio_name,
                "gs_uri": gs_uri,
                "sha256": audio_sha,
                "bytes": audio_path.stat().st_size,
                "duration_s": cap.get("duration_s"),
                "bitrate_kbps": cap.get("bitrate_kbps"),
            },
            "generation": {
                key: track.get(key)
                for key in ("idx", "take", "seed", "cap", "quantile",
                            "cot", "lora_alias", "lora_scale", "score_s",
                            "abc_file", "abc_sha256")
                if key in track
            },
            "abc_reference": {**abc_info,
                              "path": f"abc/{stem}.abc",
                              "matches_run_manifest": expected_abc is None
                              or abc_info["sha256"] == expected_abc},
            "abc_transcription": abc_tr,
            "caption": {
                "text": cap.get("caption", ""),
                "caption_raw": cap.get("caption_raw", ""),
                "chunk_captions": cap.get("chunk_captions", []),
                "instruments": cap.get("instruments", []),
                "vocals": cap.get("vocals", []),
                "production": cap.get("production", []),
                "tags_experimental": cap.get("tags_experimental", True),
                "rewrite_status": cap.get("rewrite_status"),
                "rewrite_model": cap.get("rewrite_model"),
                "caption_seconds": cap.get("caption_seconds"),
                "chunk_count": cap.get("chunk_count"),
                "silence_trimmed_s": cap.get("silence_trimmed_s"),
            },
        }
        cap_dest = out / "captions" / f"{stem}.json"
        atomic_write_json(cap_dest, caption_json)
        files.append({"path": str(cap_dest.relative_to(out)),
                      "sha256": sha256_file(cap_dest),
                      "bytes": cap_dest.stat().st_size})
        row = {
            "track_id": caption_json["track_id"],
            "run_id": row_run_id,
            "name": track["name"],
            "seed": track.get("seed"),
            "gs_uri": caption_json["audio"]["gs_uri"],
            "audio_file": audio_name,
            "audio_sha256": audio_sha,
            "audio_bytes": audio_path.stat().st_size,
            "duration_s": cap.get("duration_s"),
            "bitrate_kbps": cap.get("bitrate_kbps"),
            "abc_reference_path": f"abc/{stem}.abc",
            "abc_reference_sha256": abc_info["sha256"],
            "abc_transcription_path": (f"abc_transcription/{stem}.abc"
                                       if abc_tr else None),
            "abc_transcription_sha256": abc_tr["sha256"] if abc_tr else None,
            "abc_transcription_parse": (abc_tr["qc"]["parse"] if abc_tr else None),
            "abc_transcription_coverage": (abc_tr["qc"]["coverage"]
                                           if abc_tr else None),
            "abc_transcription_flags": (abc_tr["qc"]["flags"] if abc_tr else []),
            "instrumental": instrumental_for(row_run_id),
            "caption": caption_json["caption"]["text"],
            "caption_raw": caption_json["caption"]["caption_raw"],
            "caption_chunk_count": caption_json["caption"]["chunk_count"],
            "caption_seconds": caption_json["caption"]["caption_seconds"],
            "instruments": caption_json["caption"]["instruments"],
            "vocals": caption_json["caption"]["vocals"],
            "production": caption_json["caption"]["production"],
            "tags_experimental": caption_json["caption"]["tags_experimental"],
            "rewrite_status": caption_json["caption"]["rewrite_status"],
            "rewrite_model": caption_json["caption"]["rewrite_model"],
        }
        # Corpus columns are optional: a single-run manifest does not carry
        # them, and the reference package must keep its exact schema.
        for key in CORPUS_COLUMNS:
            value = track.get(key)
            if value is not None:
                row[key] = value
        rows.append(row)

    if not rows:
        raise SystemExit("no complete track rows; nothing to write")

    jsonl_path = out / "tracks.jsonl"
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)
    files.append({"path": "tracks.jsonl", "sha256": sha256_file(jsonl_path),
                  "bytes": jsonl_path.stat().st_size})

    write_parquet(out / "tracks.parquet", rows)
    files.append({"path": "tracks.parquet",
                  "sha256": sha256_file(out / "tracks.parquet"),
                  "bytes": (out / "tracks.parquet").stat().st_size})

    for sub in ("abc", "abc_transcription"):
        for path in sorted((out / sub).glob("*.abc")):
            files.append({"path": str(path.relative_to(out)),
                          "sha256": sha256_file(path),
                          "bytes": path.stat().st_size})

    from src.resume import CAPTION_PIPELINE_VERSION
    from src.rewrite import REWRITE_PROMPT_VERSION

    instrumental_run = any(r["instrumental"] for r in rows)
    statuses = {}
    for row in rows:
        statuses[row["rewrite_status"]] = statuses.get(row["rewrite_status"], 0) + 1
    tr_parse = {}
    for row in rows:
        tr_parse[row["abc_transcription_parse"]] = (
            tr_parse.get(row["abc_transcription_parse"], 0) + 1)

    manifest_obj = {
        "dataset": args.dataset_name,
        "dataset_version": args.dataset_version,
        "generated_at": utcnow(),
        "run_id": run_id,
        "source_run_uri": args.source_uri.rstrip("/"),
        "instrumental_run": instrumental_run,
        "source_run_manifest": str(Path(args.run_manifest).name),
        "source_runs": manifest.get("source_runs") or [run_id],
        "run_ids": sorted({str(r["run_id"]) for r in rows}),
        "rejected": manifest.get("rejected") or [],
        "take_selection": manifest.get("take_selection") or {},
        "corpus": manifest.get("corpus") or {},
        "counts": {
            "tracks": len(rows),
            "manifest_tracks": len(manifest["tracks"]),
            "with_caption": sum(bool(r["caption"]) for r in rows),
            "with_abc_reference": sum(bool(r["abc_reference_sha256"]) for r in rows),
            "with_abc_transcription": sum(bool(r["abc_transcription_sha256"])
                                          for r in rows),
        },
        "rewrite_status": statuses,
        "abc_transcription_parse": tr_parse,
        "provenance": {
            "caption_model": "ACE-Step/acestep-captioner (4-bit nf4, fp16, SDPA)",
            "caption_weights": "Urabewe/Ace-Step-Captioner-4bit",
            "caption_prompt": "*Task* Describe this audio in detail",
            "caption_pipeline_version": CAPTION_PIPELINE_VERSION,
            "rewrite_model": "Qwen3-4B-Instruct-2507-Q8_0 (llama.cpp)",
            "rewrite_prompt_version": REWRITE_PROMPT_VERSION,
            "abc_transcriber": "m-a-p/SheetSage2 (full: melody + chords)",
            "tags": "deterministic lexicon matching (experimental)",
        },
        "missing": missing,
        "files": sorted(files, key=lambda f: f["path"]),
    }
    atomic_write_json(out / "manifest.json", manifest_obj)
    atomic_write_text(out / "DATASET_CARD.md", render_card(manifest_obj, rows))
    atomic_write_text(out / "_SUCCESS", utcnow() + "\n")
    print(f"wrote {out}: {len(rows)} tracks, "
          f"{manifest_obj['counts']['with_abc_transcription']} transcribed")
    if missing:
        print(f"WARNING: {len(missing)} track(s) omitted: {missing[:3]}",
              file=sys.stderr)
    return 0


def write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Optional corpus columns are not on every row; keep the union in
    # first-seen order so a mixed manifest still writes every column.
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    arrays = {c: pa.array([r.get(c) for r in rows]) for c in columns}
    pq.write_table(pa.table(arrays), path, compression="zstd")


def render_card(manifest: dict, rows: list[dict]) -> str:
    counts = manifest["counts"]
    flags = sorted({f for r in rows for f in (r["abc_transcription_flags"] or [])})
    example = next((r for r in rows if r["caption"]), rows[0])
    lines = [
        f"# {manifest['dataset']} — {manifest['dataset_version']}",
        "",
        f"**Generated:** {manifest['generated_at']}  ",
        f"**Runs:** `{'`, `'.join(manifest.get('run_ids') or [manifest['run_id']])}`  ",
        f"**Source:** `{manifest['source_run_uri']}`",
        "",
        "## What this is",
        "",
        "Captions and symbolic scores for one or more generated-instrumental",
        "runs. Every track is a rendered MP3, paired with (a) a prose caption",
        "of the rendered audio and (b) two ABC files: the reference score used to",
        "generate the track, and an independent SheetSage2 transcription of the",
        "audio. Audio is referenced by GCS URI + SHA-256, not shipped.",
        "",
        "## Counts",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| Tracks | {counts['tracks']} |",
        f"| Tracks in source run(s) | {counts['manifest_tracks']} |",
        f"| With caption | {counts['with_caption']} |",
        f"| With reference ABC | {counts['with_abc_reference']} |",
        f"| With ABC transcription | {counts['with_abc_transcription']} |",
        "",
        "## Files",
        "",
        "| Path | Contents |",
        "|---|---|",
        "| `tracks.jsonl` | One row per track (caption, tags, hashes, score paths) |",
        "| `tracks.parquet` | Same rows, flat columns |",
        "| `captions/<track>.json` | Full caption record incl. per-chunk text |",
        "| `abc/<track>.abc` | Reference score used to generate the track |",
        "| `abc_transcription/<track>.abc` | SheetSage2 transcription of the MP3 |",
        "| `manifest.json` | Counts, provenance, SHA-256 of every file |",
        "",
        "## Example",
        "",
        f"`{example['name']}` — {example['duration_s']}s",
        "",
        "> " + " ".join(str(example["caption"]).split())[:600],
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(manifest["provenance"], indent=2),
        "```",
        "",
        "## Caveats",
        "",
        "* The caption prose is machine-generated; the term lists are",
        "  deterministic lexicon matches marked `tags_experimental: true`",
        "  (see `docs/TAG_AUDIT.md` in the music_captioning repo).",
        "* `rewrite_status` says whether the polish model was used (`ok`),",
        "  rejected (`fallback`) or skipped (`single_chunk`/`disabled`).",
        "* The reference ABC is the generation input, not a measurement of the",
        "  audio; only `abc_transcription/*.abc` describes what was rendered.",
        "* Transcription QC flags travel with each row: "
        + (", ".join(f"`{f}`" for f in flags) if flags else "none") + ".",
    ]
    if manifest.get("instrumental_run"):
        lines.append(
            "* This run is instrumental, so a `vocal_sparse` flag is expected: "
            "it fires whenever the vocal line has fewer than 50 notes and is "
            "not a transcription failure here.")
    lines.append("")
    if manifest.get("missing"):
        lines += ["## Omitted tracks", "",
                  "```json", json.dumps(manifest["missing"], indent=2), "```", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("caption", help="caption + polish one audio directory")
    cap.add_argument("--audio-dir", type=Path, required=True)
    cap.add_argument("--run-manifest", type=Path, required=True)
    cap.add_argument("--out-dir", type=Path,
                     default=REPO_ROOT / "work" / "dataset" / "captions")
    cap.add_argument("--workdir", type=Path,
                     default=REPO_ROOT / "work" / "dataset" / "tmp")
    cap.add_argument("--run-id", default=None)
    cap.add_argument("--artist", default=None)
    cap.add_argument("--album", default=None)
    cap.add_argument("--hf-model", type=Path, default=DEFAULT_HF_MODEL)
    cap.add_argument("--rewrite-gguf", type=Path, default=DEFAULT_REWRITE_GGUF)
    cap.add_argument("--llamacpp-server", type=Path,
                     default=DEFAULT_LLAMACPP_SERVER)
    cap.add_argument("--no-rewrite", action="store_true")
    cap.add_argument("--repolish", action="store_true",
                     help="re-run the rewrite model for every track; the "
                          "default on an incremental pass is to polish only the "
                          "tracks that are still pending")
    cap.add_argument("--force", action="store_true")
    cap.add_argument("--max-tracks", type=int, default=0,
                     help="caption at most N not-yet-stored tracks this pass "
                          "(0 = drain the album). Bounds one pass so polish and "
                          "the GCS checkpoint land every N tracks instead of "
                          "hours apart.")
    cap.add_argument("--max-track-seconds", type=float, default=0.0,
                     help="skip longer tracks (0 = no cap; the default, so an "
                          "incremental corpus never leaves a track un-counted)")
    cap.add_argument("--min-bitrate-kbps", type=float, default=0.0)

    asm = sub.add_parser("assemble", help="join captions + ABC into a package")
    asm.add_argument("--run-manifest", type=Path, required=True)
    asm.add_argument("--audio-dir", type=Path, required=True)
    asm.add_argument("--captions", type=Path, required=True)
    asm.add_argument("--abc-ref", type=Path, required=True)
    asm.add_argument("--transcripts", type=Path, required=True)
    asm.add_argument("--out", type=Path, required=True)
    asm.add_argument("--run-id", default=None)
    asm.add_argument("--source-uri", required=True,
                     help="GCS prefix that holds the run's audio files")
    asm.add_argument("--dataset-name", default="Fadl Shaker Instrumental 18")
    asm.add_argument("--dataset-version", default="v0.1")
    asm.add_argument("--instrumental", action=argparse.BooleanOptionalAction,
                     default=None,
                     help="whether the rows are instrumentals; default: infer "
                          "from the run id (as the card's QC wording depends "
                          "on it)")

    args = ap.parse_args(argv)
    if args.command == "caption":
        return caption_phase(args)
    return assemble_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
