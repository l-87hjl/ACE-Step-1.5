# Plan: move continue_song out of the ACE-Step fork

Status: **proposal**. Nothing is moved or deleted until Geoff has reviewed the
real-song listening test. Creating the new repository is Geoff's decision.

## Why move it

- `contrib/continue_song` (about 600 lines including tests) is an application. Upstream
  ACE-Step does not need it, and keeping it in the fork makes the fork diverge for no
  benefit to upstream merges.
- Only one engine-specific function touches ACE-Step: `generate_window`, together with
  the handler init and memory hygiene. Everything else (planning, splicing, loudness
  matching, phrase protection, the multi-pass loop, the native-rate master, the
  manifest) is engine-neutral already.

## Proposed repository

`l-87hjl/continue-song` (name open). Private is recommended at first because run plans
and lyrics are song-specific. The code itself contains nothing private.

```
continue_song/
  core/
    plan.py        # lyric plans, beat grid, phrase protection, gain match   (from plan.py)
    splice.py      # effective end, splice choice, equal-power fade, metrics (from splice.py)
    audio_io.py    # load/resample, native-rate master                       (from runtime.py)
    pipeline.py    # pass loop: context window, overlap, lookahead, candidates, manifest
    manifest.py    # manifest schema + writer (versioned)
  engines/
    base.py        # ContinuationEngine protocol + EngineCapabilities
    acestep.py     # ACE-Step adapter: repaint-outpaint call, instruction, normalization off,
                   # PyTorch-MPS / MLX-VAE settings, cache release
    fake.py        # deterministic test engine (tone continuation) for model-free tests
  cli.py           # `continue-song INPUT --target-duration 240 --engine acestep ...`
review/            # optional extra: metrics, Whisper word timings, MP3 export, LISTEN.md
tests/             # unittest; full pipeline runs against engines/fake.py
```

Engine contract (sketch):

```python
class ContinuationEngine(Protocol):
    capabilities: EngineCapabilities   # native_sr, max_window_s, lyrics, reference_audio
    def generate(self, context: np.ndarray, sr: int, regen_start_s: float,
                 regen_end_s: float, seed: int, lyrics: str, conditioning: dict) -> np.ndarray:
        """Return audio on the context's timeline, regenerated from regen_start_s to
        regen_end_s (which may extend past the context's end)."""
```

The pipeline relies only on that timeline contract. Any backend that can "regenerate
from t to T with the material before t as context" fits: ACE-Step repaint now; another
outpainting or continuation model later.

Dependencies: `numpy`, `scipy`, `soundfile` for the core. Extras: `[acestep]` pins
ACE-Step to an upstream tag or commit (or this fork) by git URL. `[review]` adds
`librosa`, `pyloudnorm`, `mlx-whisper`, `ffmpeg` for MP3s. The checkpoint location stays
`ACESTEP_CHECKPOINTS_DIR`.

## Migration steps

1. Geoff approves the repo name and visibility (Ask-Geoff). If it is a MetaOps
   ecosystem member, register it by PR.
2. Extract with history:
   `git subtree split -P contrib/continue_song -b continue-song-export` from
   `claude/continuation`, then push that branch to the new repository.
3. Restructure into `core/`, `engines/`, `cli.py`. Move `generate_window`,
   `init_handler` and `release_cached_memory` into `engines/acestep.py`.
4. Add `engines/fake.py` and a full-pipeline test. Keep the existing unit tests.
5. Parity check: re-run the recommended real-song command with the same seeds and
   compare the manifests (cut points, gains). Allow small numeric drift; MPS is not
   bit-deterministic.
6. **Only after the listening review:** delete `contrib/continue_song` from the
   `claude/continuation` branch (or leave the branch as a frozen record), and remove the
   worktree with `worktree.sh done`. The fork's `main` stays identical to upstream.
7. Optional small upstream PRs from the fork, separate from the tool: honour `duration`
   for repaint when `repainting_end < 0`, and derive `instruction` from `task_type` in
   `generate_music`.

## Not in scope

GUI, batch services, automatic lyric writing, and any change to ACE-Step internals.
