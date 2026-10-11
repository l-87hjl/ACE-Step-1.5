"""continue_song: extend an existing clip to a target length with ACE-Step repaint-outpainting.

Uses only the public ``acestep.inference.generate_music`` API (task_type="repaint" with
``repainting_end`` beyond the source length, which right-pads the source and generates
the padded region in context). The original audio is kept bit-exact up to a short
equal-power fade inside a configurable overlap window.

Example:
    python contrib/continue_song/continue_song.py input.wav --target-duration 240 \
        --caption "indie pop, female vocals" --lyrics-file lyrics.txt --candidates 3
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime import SR, load_48k_stereo, release_cached_memory  # noqa: E402
from splice import (choose_splice_sample, effective_end_s, equal_power_splice,  # noqa: E402
                    frame_log_energy_bands)


def overlap_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean log-band distance between two equal-timeline excerpts (lower = closer)."""
    n = min(len(a), len(b))
    fa = frame_log_energy_bands(a[:n].mean(1), 960)
    fb = frame_log_energy_bands(b[:n].mean(1), 960)
    return float(np.linalg.norm(fa - fb, axis=1).mean())


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
    p.add_argument("--bpm", type=int, default=None)
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
    p.add_argument("--out-dir", default="acestep_output/continue_song",
                   help="default lies under the upstream-ignored acestep_output/")
    return p


def init_handler(model: str, use_mlx_dit: bool):
    """Initialize the ACE-Step DiT handler from the repository root."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from acestep.handler import AceStepHandler
    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=root, config_path=model, device="auto",
                                   use_mlx_dit=use_mlx_dit)
    if not ok:
        raise RuntimeError(msg)
    return h


def generate_window(h, a, ctx_path: str, rs: float, re_: float, seed: int, lyrics: str,
                    ref: str, tmp: str) -> np.ndarray:
    """Run one repaint-outpaint pass on the context window; return 48 kHz audio."""
    from acestep.constants import TASK_INSTRUCTIONS
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    kw = {} if a.steps is None else {"inference_steps": a.steps}
    # GenerationParams.instruction defaults to the text2music prompt; the public API does
    # not derive it from task_type, so pass the repaint instruction explicitly.
    kw["instruction"] = TASK_INSTRUCTIONS["repaint"]
    if "turbo" not in a.model:
        kw.update(dcw_enabled=False, shift=1.0, inference_steps=a.steps or 50)
    params = GenerationParams(
        task_type="repaint", src_audio=ctx_path, reference_audio=ref,
        repainting_start=rs, repainting_end=re_, repaint_mode=a.repaint_mode,
        repaint_strength=a.repaint_strength, caption=a.caption, lyrics=lyrics,
        instrumental=a.instrumental, bpm=a.bpm, keyscale=a.key, timesignature=a.timesig,
        vocal_language=a.language, seed=seed, thinking=False, use_cot_metas=False,
        use_cot_caption=False, use_cot_language=False, enable_normalization=False, **kw)
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    res = generate_music(h, None, params, cfg, save_dir=tmp)
    if not res.success:
        raise RuntimeError(res.error)
    audio = load_48k_stereo(res.audios[0]["path"])
    release_cached_memory()
    return audio


def run_pass(h, a, cur: np.ndarray, n_pass: int, lyrics: str, tmp: str) -> dict:
    """Generate candidates for one extension pass and return the pass record.

    The splice window is ``[end - overlap, end]`` where ``end`` ignores a trailing
    fade (see ``effective_end_s``). Intermediate passes generate ``lookahead`` extra
    seconds and discard them, so an outro the model writes at the window end is not kept.
    """
    d = effective_end_s(cur, SR) if a.trim_fade else len(cur) / SR
    ctx0 = 0.0 if a.context <= 0 else max(0.0, d - a.context)
    i0 = int(ctx0 * SR)
    ctx_path = os.path.join(tmp, f"ctx{n_pass}.wav")
    sf.write(ctx_path, cur[i0:], SR, subtype="FLOAT")
    keep_end = min(a.target_duration, d + a.segment)
    gen_end = keep_end if keep_end >= a.target_duration else keep_end + a.lookahead
    rs, re_ = max(0.0, d - a.overlap) - ctx0, gen_end - ctx0
    ref = a.input if a.reference_self else None
    orig = cur[i0:]
    cands = []
    for k in range(a.candidates):
        seed = a.seed + 1000 * n_pass + k
        gen = generate_window(h, a, ctx_path, rs, re_, seed, lyrics, ref, tmp)
        gen = gen[: int((keep_end - ctx0) * SR)]
        cut = choose_splice_sample(orig, gen, SR, rs, d - ctx0)
        fade = int(a.fade_ms * SR / 1000)
        full = np.concatenate([cur[:i0], equal_power_splice(orig, gen, cut, fade)])
        w0, w1 = int(rs * SR), int((d - ctx0) * SR)
        dist = overlap_distance(orig[w0:w1], gen[w0:w1]) if w1 > w0 else 0.0
        path = os.path.join(a.out_dir, f"pass{n_pass}_cand{k}_seed{seed}.wav")
        sf.write(path, full, SR, subtype="PCM_24")
        cands.append({"seed": seed, "path": path, "cut_s": (i0 + cut) / SR,
                      "overlap_dist": dist, "audio": full})
    best = min(cands, key=lambda c: c["overlap_dist"])
    return {"pass": n_pass, "context_start_s": ctx0, "src_effective_end_s": d,
            "repaint_window_s": [rs + ctx0, gen_end], "kept_end_s": keep_end,
            "chosen": best["path"], "audio": best["audio"],
            "candidates": [{k: v for k, v in c.items() if k != "audio"} for c in cands]}


def main() -> None:
    """Extend the input pass by pass and write candidates, the result and a manifest."""
    a = build_parser().parse_args()
    lyrics = a.lyrics or (open(a.lyrics_file).read() if a.lyrics_file else "")
    if a.instrumental and not lyrics:
        lyrics = "[Instrumental]"
    os.makedirs(a.out_dir, exist_ok=True)
    cur = load_48k_stereo(a.input)
    h = init_handler(a.model, a.mlx_dit)
    release_cached_memory(a.mlx_cache_gb)
    manifest = {"input": a.input, "args": vars(a), "passes": []}
    tmp = tempfile.mkdtemp(prefix="continue_song_")
    while len(cur) / SR < a.target_duration - 0.5:
        rec = run_pass(h, a, cur, len(manifest["passes"]) + 1, lyrics, tmp)
        cur = rec.pop("audio")
        manifest["passes"].append(rec)
        print(json.dumps({k: rec[k] for k in ("pass", "kept_end_s", "chosen")}), flush=True)
    out = os.path.join(a.out_dir, "continued.wav")
    sf.write(out, cur, SR, subtype="PCM_24")
    manifest["output"] = out
    json.dump(manifest, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps({"output": out, "duration_s": len(cur) / SR}))


if __name__ == "__main__":
    main()
