import json
import os

INPUT_FILE = "./INPUT_FADL_SHAKER.json"
DONE_DIR = "./fadl_shaker_512/"

# ── Inspect INPUT JSON ────────────────────────────────────────────────────────
with open(INPUT_FILE, encoding="utf-8") as f:
    original = json.load(f)

print(f"\n── INPUT_FADL_SHAKER.json ───────────────────────")
print(f"  Type           : {type(original).__name__}")
print(f"  len()          : {len(original)}")

if isinstance(original, dict):
    print(f"  Keys           : {list(original.keys())}")
    for k, v in original.items():
        print(
            f"  [{k!r}] → type={type(v).__name__}, len={len(v) if hasattr(v,'__len__') else 'n/a'}"
        )
        if isinstance(v, list) and v:
            print(f"    first item : {v[0]}")
elif isinstance(original, list) and original:
    print(f"  First item     : {original[0]}")

# ── Inspect DONE dir ──────────────────────────────────────────────────────────
print(f"\n── {DONE_DIR} ────────────────────────────────────")
json_files = []
for root, _, files in os.walk(DONE_DIR):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file).replace(os.sep, "/"))

print(f"  .json files found : {len(json_files)}")
if json_files:
    sample = json_files[0]
    with open(sample, encoding="utf-8") as f:
        sample_data = json.load(f)
    print(f"  Sample file    : {sample}")
    print(f"  Sample keys    : {list(sample_data.keys())}")
    print(f"  Sample 'file'  : {sample_data.get('file', '⚠ key not found')}")
