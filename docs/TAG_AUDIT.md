# Deterministic Suno-tag extraction — audit

Date: 2026-11-06

Scope: the **caption → tags → Suno prompt** stage (`src/tags.py` + `src/lexicon.json`).
Audio is not stored in the repo, so this audit does **not** judge whether the
captioner described the audio correctly. It judges:

1. whether every emitted tag is actually supported by the caption,
2. which salient caption content is lost,
3. whether the resulting string is a good Suno prompt.

Sample: 5 tracks drawn with `random.seed(20261106)` from all 407 captioned
tracks in `outputs/` (32 albums).

## Sample results

| Track | Tags | Prompt chars | What worked | What was lost / wrong |
|---|---|---|---|---|
| ARAB_MAQAM_ROCK / GRP 1 — `...الفارس-المدجج...` | 28 | 278 | piano, orchestral strings, electric guitar, drums, male vocal, cinematic/reverb | **bass** (3× "bassline"), **synth pads** ("strings and pads"); 5 overlapping string tags |
| ELISSA / Ahla Donya — `14.Music.mp3` | 18 | 175 | drum machine, brass, congas, piano, four-on-the-floor, syncopated | **synth bass**, **trumpets**, **saxophones**, **timbales**; no vocal tags (correct) |
| ELISSA / As3ad Wa7da — `08. Rohtelo.mp3` | 24 | 232 | strings, woodwinds, oud, darbuka, bass guitar, synth bass, cymbal, female vocal, melisma | **synth/string pads**, **electronic beat**; `harmony` from "anchors the harmony" mis-tagged as **vocals** |
| ELISSA / Baddy Doub — `06. Sho Ma Sar.mp3` | 31 | 296 | oud, darbuka, synth bass, claps, piano, vocal chops | **female vocal** missed (says "female lead vocal" twice) while "male vocal" is kept; Arabic + Latin chunk descriptions both unioned |
| FADL_SHAKER / Allah Alam — `03.Allah Alam.mp3` | 29 | 259 | ney/flute, oud, darbuka, synth strings, kick, mandolin, bouzouki | **clap** ("kick and clap"), **synth lead**; keeps *both* mandolin and bouzouki from "mandolin **or** bouzouki" |

### Fidelity

- **Lexical precision: 100%.** Every emitted tag occurs literally in the caption.
  The extractor never invents a term. Multi-word preference and overlap
  suppression work as tested.
- **Salient recall: ~80–90%.** 11 central caption items were missed across the 5
  tracks (~2.2 per track), almost all for mechanical reasons below.
- **Prompt utility: weak.** 0/5 prompts contains the track's genre, although
  5/5 captions name it (symphonic rock/power metal, Latin house, Arabic pop,
  Arabic dance-pop, Khaleeji pop). No mood, language, tempo or key either.

## Corpus-level numbers (407 tracks)

| Metric | Value |
|---|---|
| Tags per prompt | mean **29.7**, median 30, max 44 |
| Prompts with >20 tags | **396 / 407 (97%)** |
| Share of tags from a generic set (`vocal`, `melodic`, `reverb`, `bassline`, `layered`, `cinematic`, `clean`, `mix`, …) | **43%** |
| Prompts containing any real genre token | **30 / 407 (7%)** (mostly `ambient`/`electronic`) |
| Prompts with any bass instrument tag | 167 / 407 (41%), though 362 captions say "bassline" |
| Captions with "pads" that yield no pad tag | 92 / 332 (28%) |
| Captions with "clap"/"claps" that yield no clap tag | 29 / 37 (78%) |
| "trumpets" / "saxophones" with no matching tag | 7/8, 3/4 |
| "timbales" with no tag | 6/6 (not in lexicon) |

## Failure modes (deterministic, reproducible)

1. **No genre/mood vocabulary.** `lexicon.json` has only instruments, vocals,
   production. The most useful Suno controls are absent.
2. **Plural/inflection misses.** `\bterm\b` rejects `trumpets`, `saxophones`,
   `clap`, `timbales`; `pad` does not match `pads`. The `basses`/`guitarist`
   guard in `tests/test_tags.py` is correct, but there is no plural handling.
3. **Contiguous-only phrases.** `female lead vocal`, `synthesized string pads`,
   `synth lead` do not contain the lexicon entries, so the gender/instrument is
   dropped. `female vocal` + one inserted adjective fails.
4. **No negation / uncertainty.** `extract_tags("instrumental track with no
   vocals")` returns `vocals`; `extract_tags("without drums or guitar")` returns
   both. "mandolin or bouzouki" emits both.
5. **Chunk union without voting.** Per-chunk captions are concatenated and tags
   unioned (`merge_tags`). Contradictory chunks (male/female, Arabic/Latin,
   vocal/instrumental) all survive. `chunk_captions` is popped before output, so
   the disagreement is not even visible.
6. **Synonym spam.** `strings` / `string section` / `synth strings` /
   `orchestral` / `synthesizer`, `drums` / `drum kit`, `vocal` / `vocals` /
   `singing` / `lead vocals` all coexist. Exact-cluster redundancy is 3–11%,
   but conceptual duplication is much higher.
7. **Category leakage.** `harmony` (harmonic foundation) lands in `vocals`;
   `orchestral` in `instruments`; `bassline` in `production` while the bass
   instrument is untagged; `mix` matches "through the mix".

## Verdict

The deterministic extractor is **excellent as an auditable, cheap,
non-hallucinating keyword extractor** and **mediocre as a Suno prompt
generator**. It faithfully reports what the caption says, but it emits a
30-tag bag dominated by generic adjectives, drops genre and key instruments
through plural/phrase gaps, and cannot resolve contradictory chunks.

## Status (2026-11-06)

Implemented response: `suno_prompt` was removed from all existing outputs and
from every default code path. The term lists are marked `tags_experimental`, and
the string is opt-in only (`experimental_suno=True` / `--experimental-suno`),
marked `suno_experimental`. The extractor remains available for experiments.

The fixes below are **not implemented yet**; they are the bar for calling this a
supported prompt feature again.

## Low-risk improvements (no model or prompt change)

1. Add `genre` and `mood` categories to `lexicon.json`; emit
   `genre → mood → instruments → vocals → production` in `to_suno()`.
2. Add explicit plural/phrase variants (`trumpets`, `saxophones`, `clap`,
   `timbales`, `pads`, `female lead vocal`, `synthesized string pads`,
   `synth lead`) or optional-adjective patterns for vocal-gender phrases.
3. Collapse synonym clusters to one canonical tag before joining.
4. Prune to a budget (~15–25 tags), dropping low-information production terms.
5. Vote across chunks: keep a tag only if it appears in ≥2 chunks (or a
   majority); retain only a consistent vocal gender. This requires keeping
   `chunk_captions` for the merge step.
6. Skip matches inside "no/without X" spans; for "A or B", keep the first only.
7. Add regression tests for all of the above.
