"""continue_song: extend an existing clip to a target length with ACE-Step repaint-outpainting.

Uses only the public ``acestep.inference.generate_music`` API (task_type="repaint" with
``repainting_end`` beyond the source length, which right-pads the source and generates
the padded region in context). The original audio is kept bit-exact up to a short
equal-power fade inside a configurable overlap window.

Example:
    python contrib/continue_song/continue_song.py input.wav --target-duration 240 \
        --caption "indie pop, female vocals" --lyrics-file lyrics.txt --candidates 3
"""

import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime import SR, load_48k_stereo, release_cached_memory  # noqa: E402
from cli_args import build_parser  # noqa: E402
from plan import beat_grid, load_lyrics_plan, match_gain, unsung_times  # noqa: E402
from runtime import write_native_master  # noqa: E402
from splice import (choose_splice_sample, effective_end_s, equal_power_splice,  # noqa: E402
                    join_record, overlap_distance)


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


def splice_candidates(a, n_pass: int, rs: float, d: float, ctx0: float, words):
    """Allowed cut times (window coordinates) for this join, or ``None`` = any frame."""
    bpm = a.grid_bpm or a.bpm
    if bpm is None or a.beat_offset is None:
        return None
    beats = beat_grid(bpm, a.beat_offset, rs + ctx0, d)
    if n_pass == 1:
        free = unsung_times(beats, words, a.word_margin)
        beats = free if len(free) else beats
    return beats - ctx0


def run_pass(h, a, cur: np.ndarray, n_pass: int, plan: dict, words, tmp: str) -> dict:
    """Generate candidates for one extension pass and return the pass record.

    The splice window is ``[end - overlap, end]`` where ``end`` ignores a trailing
    fade. Intermediate passes generate ``lookahead`` extra seconds and discard them, so
    an outro the model writes at the window end is not kept. New audio is gain-matched
    to the existing audio over the overlap before the equal-power splice.
    """
    d = effective_end_s(cur, SR) if a.trim_fade else len(cur) / SR
    ctx0 = 0.0 if a.context <= 0 else max(0.0, d - a.context)
    i0 = int(ctx0 * SR)
    ctx_path = os.path.join(tmp, f"ctx{n_pass}.wav")
    sf.write(ctx_path, cur[i0:], SR, subtype="FLOAT")
    keep_end = min(a.target_duration, d + plan["segment"])
    gen_end = keep_end if keep_end >= a.target_duration else keep_end + a.lookahead
    overlap = a.first_overlap if (n_pass == 1 and a.first_overlap is not None) else a.overlap
    rs, re_ = max(0.0, d - overlap) - ctx0, gen_end - ctx0
    ref = a.input if a.reference_self else None
    orig = cur[i0:]
    w0, w1 = int(rs * SR), int((d - ctx0) * SR)
    allowed = splice_candidates(a, n_pass, rs, d, ctx0, words)
    fade = int(a.fade_ms * SR / 1000)
    cands = []
    for k in range(a.candidates):
        seed = a.seed + 1000 * n_pass + k
        gen = generate_window(h, a, ctx_path, rs, re_, seed, plan["lyrics"], ref, tmp)
        gen = gen[: int((keep_end - ctx0) * SR)]
        g = match_gain(orig[w0:w1], gen[w0:w1], a.gain_match_db)
        gen = gen * g
        cut = choose_splice_sample(orig, gen, SR, rs, d - ctx0, beat_times_s=allowed)
        full = np.concatenate([cur[:i0], equal_power_splice(orig, gen, cut, fade)])
        dist = overlap_distance(orig[w0:w1], gen[w0:w1]) if w1 > w0 else 0.0
        path = os.path.join(a.out_dir, f"pass{n_pass}_cand{k}_seed{seed}.wav")
        sf.write(path, full, SR, subtype="FLOAT")
        cands.append({"seed": seed, "path": path, "cut_s": round((i0 + cut) / SR, 4),
                      "fade_s": [round((i0 + cut - fade // 2) / SR, 4),
                                 round((i0 + cut + fade // 2) / SR, 4)],
                      "gain_db": round(float(20 * np.log10(g)), 2), "overlap_dist": dist,
                      **join_record(full, i0 + cut, fade, SR), "audio": full})
    best = min(cands, key=lambda c: c["overlap_dist"])
    return {"pass": n_pass, "label": plan.get("label", ""), "context_start_s": ctx0,
            "src_effective_end_s": d, "overlap_s": overlap,
            "repaint_window_s": [rs + ctx0, gen_end], "kept_end_s": keep_end,
            "phrase_protected_cut": bool(n_pass == 1 and words and allowed is not None),
            "chosen": best["path"], "audio": best["audio"],
            "candidates": [{k: v for k, v in c.items() if k != "audio"} for c in cands]}


def pass_plans(a) -> list:
    """Per-pass segment/lyrics: from --lyrics-plan, else one repeating default pass."""
    if a.lyrics_plan:
        return load_lyrics_plan(a.lyrics_plan)
    lyrics = a.lyrics or (open(a.lyrics_file).read() if a.lyrics_file else "")
    if a.instrumental and not lyrics:
        lyrics = "[Instrumental]"
    return [{"segment": a.segment, "lyrics": lyrics}]


def main() -> None:
    """Extend the input pass by pass and write candidates, the result and a manifest."""
    a = build_parser().parse_args()
    plans = pass_plans(a)
    words = json.load(open(a.words_json))["words"] if a.words_json else None
    os.makedirs(a.out_dir, exist_ok=True)
    cur = load_48k_stereo(a.input)
    h = init_handler(a.model, a.mlx_dit)
    release_cached_memory(a.mlx_cache_gb)
    manifest = {"input": a.input, "args": vars(a), "passes": []}
    tmp = tempfile.mkdtemp(prefix="continue_song_")
    while len(cur) / SR < a.target_duration - 0.5:
        n = len(manifest["passes"]) + 1
        if a.max_passes and n > a.max_passes:
            break
        rec = run_pass(h, a, cur, n, plans[min(n, len(plans)) - 1], words, tmp)
        cur = rec.pop("audio")
        manifest["passes"].append(rec)
        print(json.dumps({k: rec[k] for k in ("pass", "kept_end_s", "chosen")}), flush=True)
    out = os.path.join(a.out_dir, "continued_48k.wav")
    sf.write(out, cur, SR, subtype="FLOAT")
    manifest["output_48k"] = out
    first = manifest["passes"][0]["candidates"] if manifest["passes"] else []
    if first:
        chosen = [c for c in first if c["path"] == manifest["passes"][0]["chosen"]][0]
        manifest["native_master"] = write_native_master(
            a.input, cur, chosen["fade_s"][0], os.path.join(a.out_dir, "continued_native.wav"))
    json.dump(manifest, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps({"output": out, "duration_s": len(cur) / SR}))


if __name__ == "__main__":
    main()
