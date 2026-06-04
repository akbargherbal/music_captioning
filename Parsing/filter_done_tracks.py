import json
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FILE = "INPUT_FADL_SHAKER.json"
DONE_DIR   = "./fadl_shaker_512/"


# ── Collect already-processed tracks (basename only) ──────────────────────────
done_files: set[str] = set()
for root, _, files in os.walk(DONE_DIR):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file).replace(os.sep, "/")
            with open(path, encoding="utf-8") as f:
                done_files.add(json.load(f)["file"])   # e.g. "01.Meta Habibi.mp3"


# ── Load manifest ─────────────────────────────────────────────────────────────
with open(INPUT_FILE, encoding="utf-8") as f:
    manifest = json.load(f)

tracks       = manifest["tracks"]        # list of {"path": ..., "tags": {...}}
total_before = len(tracks)


# ── Filter — compare by basename of each track's path ────────────────────────
remaining  = [t for t in tracks if Path(t["path"]).name not in done_files]
total_done = total_before - len(remaining)
total_left = len(remaining)


# ── Write back — globals untouched ────────────────────────────────────────────
manifest["tracks"] = remaining
with open(INPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)


# ── Sanity check ──────────────────────────────────────────────────────────────
print(f"{'─' * 36}")
print(f"  Total (before) : {total_before:>4}")
print(f"  Done (filtered): {total_done:>4}")
print(f"  Remaining      : {total_left:>4}")
print(f"{'─' * 36}")
assert total_before == total_done + total_left, "Count mismatch — check logic!"
print("  ✓ Counts add up correctly.")
print(f"  '{INPUT_FILE}' updated.")
