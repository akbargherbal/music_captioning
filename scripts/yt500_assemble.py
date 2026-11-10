#!/usr/bin/env python3
"""Assemble the final YT500 instrumental package once the corpus is complete.

Refuses to build a final package for an incomplete corpus: every one of the
``corpus_total`` names needs an accepted take, a polished caption and an ``ok``
transcription, and ``rejected.json`` must be empty. That gate is the whole
point of the dataset, so ``--force`` is the only way past it and it is not used
in the runbook.

Then it runs ``build_dataset.py assemble`` over the whole queue and writes
``work/package`` (``tracks.jsonl``, ``tracks.parquet``, per-track caption JSON,
reference ABC, transcribed ABC, ``manifest.json``, ``DATASET_CARD.md``,
``_SUCCESS``). The publisher uploads the package; this script never does.

Usage::

    python3 scripts/yt500_assemble.py --dry-run     # readiness report only
    python3 scripts/yt500_assemble.py              # build the package
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yt500_dataset as dataset  # noqa: E402
import yt500_gcs as gcsio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DATASET = REPO_ROOT / "scripts" / "build_dataset.py"
DEFAULT_ABC_REF = (REPO_ROOT.parent / "arabmaqamrock" / "scores"
                   / "yt500_instrumental")
DEFAULT_SOURCE_URI = "gs://akbar-december-2024-backup/arabmaqamrock/runs"


def readiness(work_dir: Path) -> dict:
    """The §12 gate, as data."""
    manifest = dataset.read_manifest(work_dir)
    counts = manifest.get("counts") or {}
    caps = dataset.caption_state(work_dir, manifest)
    trs = dataset.transcript_state(work_dir, manifest)
    accepted = int(counts.get("accepted") or 0)
    total = int(counts.get("corpus_total") or 0)
    rejected = dataset.rejected_names(work_dir)
    problems: list[str] = []
    if not total:
        problems.append("manifest has no corpus_total (run yt500_sync.py)")
    if accepted != total:
        problems.append(f"accepted {accepted} != corpus_total {total}")
    if rejected:
        problems.append(f"{len(rejected)} name(s) still rejected: "
                        f"{rejected[:5]}")
    if len(caps["done"]) != accepted:
        problems.append(f"captioned {len(caps['done'])} != accepted {accepted}")
    if len(trs["done"]) != accepted:
        problems.append(
            f"transcribed {len(trs['done'])} != accepted {accepted}")
    return {"corpus_total": total, "accepted": accepted,
            "captioned": len(caps["done"]), "transcribed": len(trs["done"]),
            "rejected": len(rejected), "problems": problems,
            "ready": not problems}


def render(report: dict) -> str:
    lines = [
        f"corpus       {report['corpus_total']}",
        f"accepted     {report['accepted']}",
        f"captioned    {report['captioned']}",
        f"transcribed  {report['transcribed']}",
        f"rejected     {report['rejected']}",
        f"ready        {'yes' if report['ready'] else 'no'}",
    ]
    for problem in report["problems"]:
        lines.append(f"  - {problem}")
    return "\n".join(lines)


def assemble_command(args: argparse.Namespace, work_dir: Path) -> list[str]:
    return [
        sys.executable, str(BUILD_DATASET), "assemble",
        "--run-manifest", str(dataset.manifest_path(work_dir)),
        "--audio-dir", str(dataset.audio_dir(work_dir)),
        "--captions", str(dataset.captions_dir(work_dir)),
        "--abc-ref", str(args.abc_ref),
        "--transcripts", str(dataset.transcripts_dir(work_dir)),
        "--out", str(dataset.package_dir(work_dir)),
        "--run-id", "YT500_INSTRUMENTAL",
        "--source-uri", args.source_uri,
        "--dataset-name", args.dataset_name,
        "--dataset-version", args.dataset_version,
        "--instrumental",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path,
                    default=REPO_ROOT / "work" / "yt500")
    ap.add_argument("--abc-ref", type=Path, default=DEFAULT_ABC_REF,
                    help="frozen instrumental scores (from arabmaqamrock)")
    ap.add_argument("--source-uri", default=DEFAULT_SOURCE_URI)
    ap.add_argument("--dataset-name", default=dataset.DATASET_NAME)
    ap.add_argument("--dataset-version", default=dataset.DATASET_VERSION)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the readiness report and stop")
    ap.add_argument("--force", action="store_true",
                    help="build even when the corpus is incomplete (never use "
                         "this for the published package)")
    ap.add_argument("--timeout-minutes", type=float, default=120.0)
    args = ap.parse_args(argv)

    work_dir = Path(args.work_dir)
    report = readiness(work_dir)
    print(render(report))

    if args.dry_run:
        if not args.force and not report["ready"]:
            print("(dry run: the package would not be built yet)")
        return 0
    if not report["ready"] and not args.force:
        print("refusing to build an incomplete package "
              "(use --force only for a debug build)", file=sys.stderr)
        return 1
    if not Path(args.abc_ref).is_dir():
        raise SystemExit(
            f"reference scores not found: {args.abc_ref}\n"
            "Fetch them in arabmaqamrock "
            "(`plumbing/yt500_generation_corpus.py --fetch`).")

    status = {"role": "assemble", "utc": gcsio.utcnow(),
              "report": report, "package": str(dataset.package_dir(work_dir))}
    cmd = assemble_command(args, work_dir)
    print(f"$ {' '.join(cmd)}", flush=True)
    log = work_dir / "logs" / "assemble.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=fh,
                                  stderr=subprocess.STDOUT, check=False,
                                  timeout=args.timeout_minutes * 60 or None)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\nTIMEOUT after {args.timeout_minutes:g} min\n")
            rc = 124

    status["returncode"] = rc
    status["success"] = (dataset.package_dir(work_dir) / "_SUCCESS").is_file()
    gcsio.write_json_atomic(dataset.status_path(work_dir, "assemble"), status)
    if rc != 0:
        print(f"assemble failed (rc={rc}); see {log}", file=sys.stderr)
        return rc
    if not status["success"]:
        print(f"assemble wrote no _SUCCESS; see {log}", file=sys.stderr)
        return 1
    print(f"package  : {dataset.package_dir(work_dir)}")
    print("next     : python3 scripts/yt500_publish.py --once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
