# Investigation: temporal continuation with ACE-Step 1.5

Question: can ACE-Step 1.5 condition on an existing clip and generate new audio
*temporally after it*, keeping the original as the opening of the finished song?

Short answer: **yes, natively.** Repaint with `repainting_end` beyond the source length is
an outpainting operation. The model sees the whole source (as in-context latents) and
generates the padded region. Every other mode either regenerates inside the source's
duration or adds layers. What upstream lacks is a workflow (see `README.md`).

Tested against upstream `main` at `ca1e85fe9430179831e6bc6be790c332190a3866`
(`v0.1.8-21-gca1e85f`, 2026-08-29), Apple M4 / 24 GB / macOS 27.0.1.

## What each mode does (from source, confirmed by runs)

| Mode | Mechanism | Length | Temporal continuation? |
|---|---|---|---|
| text2music | DiT from noise; optional LM audio codes as `src_latents` (then internally `cover`) | `duration` | No source |
| reference audio | 3 x 10 s of the reference become a 30 s timbre/style embedding | any | No: global style, no timeline |
| cover / remix | source latents (FSQ-quantised "melodic skeleton") condition the whole track | locked to source | No |
| repaint, end <= source | masked span replaced by silence latent in `src_latents`; outside span re-injected for `injection_ratio` of steps; latent and waveform boundary blends | source | No: in-place edit |
| **repaint, end > source** | `padding_utils.prepare_padding_info` right-pads the source with zeros; padded span is the repaint mask | `repainting_end` | **Yes** |
| lego (base) | generates one named stem over the source context | source (padding also applies) | No: adds a layer |
| complete (base) | "Complete the input track with <instruments>": arranges around the input | source | No: accompaniment |
| extract (base) | isolates a named stem | source | No |
| audio codes | `convert_src_audio_to_codes` -> 5 Hz FSQ tokens; `understand_music` -> caption, BPM, key, lyrics | n/a | Analysis only |

Repaint modes: `balanced` (strength 0.5 -> injection 0.5, ~0.5 s latent and 25 ms
waveform crossfades), `conservative` (full injection), `aggressive` (no injection, no
splice, so the "preserved" part is regenerated too and the original is **not** kept).

## Pitfalls found in the public API

1. `duration` is ignored for repaint. The recipe "start=30, end=-1, duration=60"
   (maintainer reply on upstream issue #413) returns a 30 s file. Pass `repainting_end`.
2. `GenerationParams.instruction` defaults to the text2music prompt and is **not** derived
   from `task_type`. Python callers must pass `TASK_INSTRUCTIONS["repaint"]` (the Gradio
   UI and `cli.py` do this; the bare API does not).
3. Output peak normalization (on by default) rescales the preserved original.
4. AI-generated short clips often end in a decay. Extending from the literal end leaves
   a gap; regenerating an overlap that covers the decay removes it.

## Objective results (turbo, 8 steps)

Two inputs: a synthetic clip (D major, 96 BPM) and a 30 s ACE-Step vocal pop clip
(G major, ~115 BPM, ends in a ~3 s decay).

| 30 s -> 60 s | key new | tempo | RMS step at clip's effective end | timbre cos (10 s either side) | chroma cos |
|---|---|---|---|---|---|
| vocal, start=30 (balanced) | G major | 114.8 = orig | -16 -> -33 dB (gap) | 0.96 | 0.98 |
| vocal, start=24 (6 s overlap) | G major | 114.8 = orig | -12.4 -> -13.0 dB | 0.994 | 0.998 |
| vocal, aggressive | G major | 114.8 | -18.6 -> -15.1 dB | 0.995 | 0.975 (original not kept) |
| synth, 6 s overlap | B minor (relative) | ~97 | -16.8 -> -16.6 dB | 0.986 | 0.990 |

Chain proof: the vocal clip run through `continue_song` to 240 s (4 passes, 2 candidates
each, 60 s context, 6 s overlap, 10 s lookahead) completed in about 10 minutes. The
original is bit-exact up to the first fade. Tempo is 114.8 in every 30 s window. Key is
G major in 6 of 8 windows (one reads A minor, one G minor). LUFS ranges -11.4 to -14.6
(original -13.3). RMS steps at the four splices are 1.2 to 2.6 dB. A Whisper
transcript shows the lyrics advancing verse 1 -> chorus -> verse 2 -> chorus -> bridge
-> final chorus -> outro. The bridge and chorus recur in the last pass because every
pass receives the full lyric sheet.

None of these numbers replace listening.

## Base model

`acestep-v15-base` runs on this machine with `dcw_enabled=False, shift=1.0`, 50 steps:
about 3 min per 60 s repaint-extension job. The key was kept (D major) and the output
was not noisy by a spectral-flatness check. Base is not required for continuation;
repaint is available on turbo/sft/base.

## Memory on a 24 GB Mac

The default MLX DiT path keeps a float32 PyTorch copy *and* an MLX copy of the DiT
(about 20 GB at idle in Gradio). VAE encode and decode in float32 peak at 27-29 GB.
Settings that worked with no code change: `use_mlx_dit=False` (PyTorch-MPS DiT, which
was also faster here), `ACESTEP_MLX_VAE_FP16=1`, `ACESTEP_MLX_VAE_CHUNK=192`, and clearing
the MLX/MPS caches between passes (peak about 20 GB).
