# continue_song (fork-local contrib, experimental)

Extends an existing short song clip (e.g. 30 s) into a longer song while keeping the
original audio as the beginning of the result.

**Status:** exploratory proof of concept on the `claude/continuation` branch of the
`l-87hjl/ACE-Step-1.5` fork. It is not part of upstream ACE-Step and is not intended
for an upstream PR in this form.

## Why this exists (divergence from upstream)

Upstream ACE-Step 1.5 already contains the core mechanism: a **repaint** task whose
`repainting_end` lies beyond the source length right-pads the source and generates the
padded region *in context* (outpainting). See
`acestep/core/generation/handler/padding_utils.py` and
`acestep/core/generation/handler/conditioning_masks.py`. What upstream does not provide
is the workflow around it:

| Gap in upstream | What this tool does |
|---|---|
| `duration` is ignored for repaint (the documented "start=30, end=-1, duration=60" recipe yields a 30 s file) | passes an explicit `repainting_end` |
| `GenerationParams.instruction` defaults to the text2music prompt; the Python API does not derive it from `task_type` | passes the repaint instruction explicitly |
| output peak-normalization rescales the preserved original | disables normalization so original and continuation share gain |
| AI-generated clips often end in a fade; extending from the literal end leaves a gap | finds the clip's effective end and regenerates an overlap window that covers the fade |
| the model writes an outro at the end of every generated window | intermediate passes generate `--lookahead` extra seconds and discard them |
| no repeated-extension loop, splice choice, or candidate handling | multi-pass loop, spectral-similarity splice point inside the overlap, equal-power fade, N candidates per pass, JSON manifest |

**No upstream file is modified.** The tool only imports `acestep.inference`,
`acestep.handler` and `acestep.constants`, so `git merge upstream/main` stays clean.

## Usage

```bash
uv run python contrib/continue_song/continue_song.py input.wav --target-duration 240 \
  --caption "indie pop, female vocals, jangly guitar" --lyrics-file lyrics.txt \
  --bpm 112 --key "G major" --candidates 2
```

Key options: `--segment` (new seconds per pass, default 60), `--overlap` (seconds
regenerated before the clip's effective end, default 6), `--context` (max seconds fed
as context, default 60; 0 = everything), `--lookahead`, `--fade-ms`, `--model`
(`acestep-v15-turbo` default; `acestep-v15-base`/`-sft` use 50 steps, CFG, DCW off),
`--repaint-mode`, `--reference-self`, `--seed`, `--candidates`, `--mlx-dit`.

Added for the first real-song test:

- `--lyrics-plan plan.json`: per-pass `segment` and `lyrics`. Each pass's sheet covers
  its context window plus the new material, so sections advance instead of every pass
  seeing the whole song.
- `--first-overlap`: a separate overlap for the first join.
- `--grid-bpm` + `--beat-offset`: snap cuts to the beat grid. The grid BPM is not sent
  to the model.
- `--words-json` + `--word-margin`: word timings of the source (for example from
  Whisper). The first cut avoids sung words.
- `--gain-match-db`: gain-match the new audio to the existing audio over the overlap
  (clipped to +/- N dB).
- `--max-passes`: stop early, for first-join studies.

Outputs: every candidate of every pass (float WAV), `continued_48k.wav`,
`continued_native.wav`, and `manifest.json` (splice and fade times, gains, levels at each
join, chosen candidates). If the input is not 48 kHz, `continued_native.wav` is written at
the input's own rate and is bit-identical to the input up to the first crossfade. The default `--out-dir` is
`acestep_output/continue_song/`, which upstream's `.gitignore` already excludes. Keep
source songs outside the repository or in an ignored path; upstream ignores `*.wav`,
`*.mp3`, `data/`, `checkpoints/` but **not** `*.flac`, `*.m4a`, `*.ogg` or `*.aiff`.

On a 24 GB Apple Silicon machine these settings kept peak memory around 20 GB:
`ACESTEP_MLX_VAE_FP16=1 ACESTEP_MLX_VAE_CHUNK=192` and the default PyTorch-MPS DiT
(`--mlx-dit` off). With the default MLX DiT path, a float32 PyTorch copy and an MLX copy
of the DiT are both resident.

## Tests

```bash
python -m unittest contrib/continue_song/splice_test.py contrib/continue_song/plan_test.py
```

## Known limitations

- Candidate choice uses an objective proxy (spectral distance in the overlap); it is not
  a musical judgement. Listen to the candidates.
- Without `--lyrics-plan`, lyrics are passed whole for every pass. In a 4-pass test the
  lyrics advanced in order, but the bridge and chorus came back in the last pass.
- Phrase protection applies to the first join only; later joins are between generated
  passes, for which no word timings exist.
- The original is bit-exact only up to the start of the fade at the chosen splice.
