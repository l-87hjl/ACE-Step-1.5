"""Command-line options for continue_song."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description="Extend a short song with ACE-Step repaint-outpainting.")
    p.add_argument("input")
    p.add_argument("--target-duration", type=float, required=True)
    p.add_argument("--segment", type=float, default=60.0, help="new seconds per pass")
    p.add_argument("--overlap", type=float, default=6.0, help="seconds regenerated before the end")
    p.add_argument("--context", type=float, default=60.0, help="max seconds fed as context (0=all)")
    p.add_argument("--fade-ms", type=float, default=250.0)
    p.add_argument("--lookahead", type=float, default=10.0,
                   help="extra seconds generated then discarded on intermediate passes")
    p.add_argument("--no-trim-fade", dest="trim_fade", action="store_false",
                   help="splice at the literal end even if the clip fades out")
    p.add_argument("--caption", default="")
    p.add_argument("--lyrics", default=None)
    p.add_argument("--lyrics-file", default=None)
    p.add_argument("--instrumental", action="store_true")
    p.add_argument("--bpm", type=int, default=None, help="BPM hint passed to the model")
    p.add_argument("--grid-bpm", type=float, default=None,
                   help="BPM for the splice beat grid (default: --bpm); not sent to the model")
    p.add_argument("--key", default="")
    p.add_argument("--timesig", default="")
    p.add_argument("--language", default="unknown")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--model", default="acestep-v15-turbo")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--repaint-mode", default="balanced")
    p.add_argument("--repaint-strength", type=float, default=0.5)
    p.add_argument("--reference-self", action="store_true",
                   help="also pass input as reference_audio")
    p.add_argument("--mlx-dit", action="store_true", help="use MLX DiT (more memory)")
    p.add_argument("--mlx-cache-gb", type=float, default=2.0, help="cap on MLX buffer cache")
    p.add_argument("--lyrics-plan", default=None,
                   help="JSON per-pass plan (segment + lyrics); overrides --segment/--lyrics")
    p.add_argument("--first-overlap", type=float, default=None,
                   help="overlap for the first join only (default: --overlap)")
    p.add_argument("--beat-offset", type=float, default=None,
                   help="time of one beat (s); with --bpm, cuts are snapped to the beat grid")
    p.add_argument("--words-json", default=None,
                   help='source word timings {"words": [{"s":..,"e":..}]}; first cut avoids them')
    p.add_argument("--word-margin", type=float, default=0.35,
                   help="seconds kept clear around each sung word when choosing the first cut")
    p.add_argument("--gain-match-db", type=float, default=3.0,
                   help="max gain (dB) applied to new audio to match level at the join; 0=off")
    p.add_argument("--max-passes", type=int, default=0, help="stop after N passes (0=no limit)")
    p.add_argument("--out-dir", default="acestep_output/continue_song",
                   help="default lies under the upstream-ignored acestep_output/")
    return p
