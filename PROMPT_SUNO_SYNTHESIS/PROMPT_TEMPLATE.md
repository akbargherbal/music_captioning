## Your Task
Synthesize the four-chunk music caption JSON below into a single Suno-ready caption paragraph.

## Rules (quick reminder)
- **Genre:** consensus wins; single-chunk outliers are discarded
- **Instruments:** union weighted by consistency; 2+ chunks = include, 1 chunk = discard unless genre-plausible
- **Vocals:** anchor to the dominant/target voice type; treat isolated contradictions as captioning artifacts
- **Tempo:** median of consistent values; report a range if values diverge by more than ~5 BPM
- **Key:** use the most reliable chunk (stable BPM + on-target genre); omit if genuinely unresolvable
- **Contradictions:** 3-vs-1 → discard the outlier; 2-vs-2 → compromise or report both
- **Output:** one flowing paragraph, 60–120 words, no bullets, no hedging, no commentary

## Input JSON
```json
{JSON_DUMP}
```
