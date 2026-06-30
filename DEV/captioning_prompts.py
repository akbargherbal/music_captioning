"""
Captioning prompt snippets for building a labeled dataset of AI-generated music tracks.

Each snippet is meant to be sent independently to an audio-captioning model (one
attribute per call) so you get a clean, single-purpose label rather than one
model trying to juggle five tasks in a single response. Wording leans on
recording-engineer / music-production vocabulary rather than vague adjectives,
since vague output ("sounds cool", "energetic") is useless for a dataset you
intend to mine for patterns later.
"""

CAPTIONING_PROMPTS = {

    "genre": (
        "Listen to this track and identify its genre with precision. Name the primary "
        "genre, then any subgenres or fusion elements present. If it blends multiple "
        "styles, name each one and estimate their relative weight (e.g., 70% orchestral "
        "rock, 30% folk). Note any specific era or scene the production style evokes "
        "(e.g., 80s synthwave, 90s alternative, 2020s bedroom pop). Avoid single-word "
        "genre labels like 'pop' or 'rock' alone; always qualify with at least one "
        "modifier. Estimate tempo feel (slow / mid-tempo / driving / fast) and approximate "
        "BPM if discernible. Do not describe instrumentation, vocals, or production here, "
        "stay strictly on genre classification."
    ),

    "vocals": (
        "Listen to the vocal performance in this track and describe it technically. "
        "Identify vocal gender, register (e.g., baritone, alto, tenor, contralto), and "
        "vocal technique used (chest voice, head voice, mixed voice, falsetto, belting, "
        "melismatic runs, vibrato, rasp, growl). Describe the emotional delivery in "
        "concrete terms (controlled, strained, breathy, commanding, detached) rather than "
        "vague mood words. Note whether the lead vocal sounds forced or naturally "
        "supported by breath. Identify language and, if discernible, dialect or accent. "
        "List any backing vocals, harmonies, or vocal layering separately from the lead. "
        "If there are no vocals, state that explicitly."
    ),

    "instrumentation": (
        "Identify every instrument audible in this track, ordered from most to least "
        "prominent in the mix. For each instrument, describe the playing technique or "
        "articulation where identifiable (e.g., palm-muted electric guitar, fingerpicked "
        "acoustic guitar, four-on-the-floor kick pattern, arco strings versus pizzicato). "
        "Distinguish acoustic instruments from electronic or synthesized ones, and note "
        "the synthesis type if it's clearly synthetic (e.g., FM bass, wavetable lead, "
        "sawtooth pad). Flag any instrument that sounds unusual or absent for the genre "
        "identified. Do not comment on mixing or mastering here, only what instruments "
        "are present and how they're played."
    ),

    "production": (
        "Describe the production and mixing characteristics of this track using "
        "recording-engineer language, not subjective impressions. Cover: stereo image "
        "(narrow/mono, moderate, wide), vocal placement in the mix (forward/centered "
        "versus recessed/buried), reverb type and amount (dry, short room, hall, plate, "
        "washed-out), perceived recording environment (small room, large hall, studio-"
        "polished, lo-fi/bedroom), dynamic behavior (compressed and flat versus dynamic "
        "with audible rises and falls), and how instruments and vocals interact rhythmically "
        "(do instruments recede during sustained vocal lines, or stay constant). Note any "
        "audible production artifacts (tape saturation, vinyl crackle, sidechain pumping, "
        "harsh digital edges)."
    ),

    "mastering": (
        "Evaluate the final mastering characteristics of this track. Describe overall "
        "loudness and dynamic range (heavily compressed/loud versus dynamic with real "
        "peaks and valleys), tonal balance (bass-heavy, bright/treble-forward, scooped "
        "mids, balanced), and stereo width at the master bus level (narrow/mono-compatible "
        "versus wide/expansive). Note transient character (punchy and sharp versus soft "
        "and rounded) and any audible distortion, clipping, or saturation introduced at "
        "the mastering stage. State whether the overall sound reads as vintage/analog-"
        "mastered or modern/digital-mastered, and whether it sounds professionally "
        "polished or intentionally raw/unmastered."
    ),

}


if __name__ == "__main__":
    for attribute, snippet in CAPTIONING_PROMPTS.items():
        print(f"--- {attribute} ---")
        print(snippet)
        print()
