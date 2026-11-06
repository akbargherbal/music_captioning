# Resume plan — Elissa + Fadl Shaker full ABC transcription

_Last updated: 2026-10-28 09:10 UTC._
_**Status: COMPLETE.** All **23 / 23 albums** are transcribed and committed.
Nothing is left to transcribe._

> **بالعربي:** اكتمل النسخ لكل الألبومات (23 ألبوم / 236 تراك). لا يوجد شيء
> ناقص للنسخ. المتبقي اختياري: بناء الـ dataset، وإعادة محاولة 9 تراكات
> فشلت (`no-abc`). أي شيء داخل `/content` يُمسح عند فصل كولاب؛ الحالة
> الدائمة في GitHub وGCS. بعد إعادة التشغيل اتبع §2–§4 ثم §6.

## 0. Final status

| Metric | Value |
|---|---:|
| Albums | **23** (8 ELISSA + 15 FADL) |
| Tracks | **236** |
| `ok` | **224** |
| `flagged` (`octave_suspect`) | **3** |
| `failed` (`no-abc`) | **9** |
| Audio | **19.188 h** |
| GPU | 212.3 min |
| RTF | **5.423×** |
| GCS objects verified | **46 / 46** (23 archives + 23 JSONs, size + md5) |

Last checkpoint commit: `67ff7a8` (`FADL_SHAKER/15.The Best Of`).

## 1. What is durable

| Item | Durable location |
|---|---|
| `outputs/abc_index.json` | **git** `music_captioning` |
| `scripts/`, `docs/` | **git** |
| Packed album JSON (`outputs/<ARTIST>/abc/<album>.json`) | **GCS** `TRANSCRIPTS/<ARTIST>/abc/<album>.json` — **not in git** |
| Raw SheetSage2 artifacts | **GCS** `TRANSCRIPTS/<ARTIST>/<album>.tar.gz` |
| Caption JSONs (`outputs/<ARTIST>/<album>.json`) | git |

Lost with `/content`: `work/manifest.jsonl`, `work/audio`, `work/out`,
`work/archives`, venv, model weights, local logs.

## 2. Rebuild the runtime (only needed for dataset/retry)

```bash
git clone https://github.com/akbargherbal/arabmaqamrock.git /content/amr
cd /content/amr && pip install -q uv
./plumbing/setup_transcribe.sh --cuda --warm
apt-get -qq install -y abcmidi
./transcribe.sh --doctor
source /content/amr/.venv-sheetsage2/bin/activate
```

Auth checks:

```bash
gcloud storage ls gs://akbar-december-2024-backup/ >/dev/null && echo GCS_OK
gh auth status
```

If git pushes fail, run `gh auth setup-git`.

## 3. Clone the data repo and rebuild the manifest

```bash
git clone https://github.com/akbargherbal/music_captioning.git /content/mc
cd /content/mc && mkdir -p work

python3 scripts/abc_manifest.py \
  --artists ELISSA FADL_SHAKER \
  --root gs://akbar-december-2024-backup/DISCOGRAPHY \
  --outputs outputs --out work/manifest.jsonl
```

Expect **236 rows / 23 albums**.

## 4. Restore the packed ABC JSONs from GCS — mandatory for the dataset

```bash
cd /content/mc
mkdir -p outputs/ELISSA/abc outputs/FADL_SHAKER/abc
gcloud storage rsync -r "gs://akbar-december-2024-backup/TRANSCRIPTS/ELISSA/abc" outputs/ELISSA/abc
gcloud storage rsync -r "gs://akbar-december-2024-backup/TRANSCRIPTS/FADL_SHAKER/abc" outputs/FADL_SHAKER/abc
```

Verify every index entry has a local file:

```bash
python3 - <<'PY'
import json, pathlib
idx = json.load(open('outputs/abc_index.json'))['albums']
missing = [v['path'] for v in idx.values()
           if not (pathlib.Path('outputs') / v['path']).is_file()]
print('albums in index:', len(idx))
print('missing local JSONs:', missing or 'none')
PY
```

## 5. Confirm nothing is pending

```bash
cd /content/mc
python3 scripts/abc_batch.py --list
```

It must print `(nothing pending)`. If it lists albums, the index was not pulled
or the artist filter is wrong — do not rerun them blindly.

## 6. Build the dataset (the remaining step)

```bash
cd /content/mc
python3 scripts/build_abc_dataset.py \
  --outputs outputs --manifest work/manifest.jsonl \
  --out datasets/elissa_fadl
wc -l datasets/elissa_fadl/*.jsonl
```

Expected: **227 rows** across `train` + `val` and **9 rows** in `quarantine`
(the `no-abc` tracks). Album-level split; duplicate groups are kept whole, so
val is ~31 % (open decision in §8), not ~10 %.

Do **not** commit `datasets/` (it contains ABC text; §15.1 unresolved). If a
backup is wanted, upload it to GCS under `TRANSCRIPTS/_datasets/`.

## 7. Optional: retry the 9 `no-abc` tracks

`transcribe.py` re-runs a `no-abc` track on retry and skips the `ok` ones.

| Track | Duration (s) |
|---|---:|
| `ELISSA/Elissa - 2012  As3ad Wa7da/05-te3ebt-minnak` | 303.8 |
| `FADL_SHAKER/01.Walla Zaman.1998/03-adee-wala-tesalemsh` | 282.3 |
| `FADL_SHAKER/01.Walla Zaman.1998/08-nazra-wahda` | 303.5 |
| `FADL_SHAKER/02.Bayya'a El Qeloub.1999/04-zaid-el-malam` | 416.4 |
| `FADL_SHAKER/05.Hobak Khayal.2001/03-hazzak-ya-albi` | 365.8 |
| `FADL_SHAKER/08.Hafla 2004/05-hobbak-khayal` | 304.2 |
| `FADL_SHAKER/08.Hafla 2004/09-la-telomuni` | 244.5 |
| `FADL_SHAKER/10.Saharni El Shoq.2005/04-tadri-lieh` | 263.3 |
| `FADL_SHAKER/15.The Best Of (Al Rawae3).2010/01-rouh` | 324.9 |

Retry one album:

```bash
python3 scripts/abc_batch.py --artist ELISSA \
  --album "Elissa - 2012  As3ad Wa7da" --retry-failed --max-minutes 30
```

Retry all partial FADL albums:

```bash
python3 scripts/abc_batch.py --artist FADL_SHAKER --retry-failed --max-minutes 60
```

If a track returns `no-abc` again, record it as a model/audio limitation. Do
not hand-edit an ABC.

Three `octave_suspect` flags are expected and never "fixed":
`07.Fadl Shaker.2003/02-el-oyoun-el-soud`,
`09.Arabian Night.2004/02-oulou-laeyn-el-shames`,
`15.The Best Of (Al Rawae3).2010/14-ahawel`.

## 8. Known issues / decisions

- **ABC text in git (§15.1): unresolved.** Local + GCS only.
- **Val split (§10.2): open.** Duplicate groups kept whole projects ~31 % val.
  Either accept it or make `audio_sha256` authoritative for grouping. Do not
  let duplicates straddle.
- **`no-abc` rate:** 9 / 236 (3.8 %); 1/96 ELISSA, 8/140 FADL. All isolated.
- **Wildcard filenames:** `gcloud storage cp` cannot copy `[ ] * ?`; the batch
  uses the GCS JSON API for those. Do not rename objects in GCS.
- **Storage:** one tar.gz per album, verified by size + md5 before commit.

## 9. Don'ts

- No `git add -A`.
- No `--force` on `abc_batch.py`.
- Do not write to `outputs/<ARTIST>/<album>.json` (caption files).
- Do not hand-edit an ABC.
- Do not run two batches concurrently.
