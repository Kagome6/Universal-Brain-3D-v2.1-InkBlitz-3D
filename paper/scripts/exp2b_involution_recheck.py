# -*- coding: utf-8 -*-
"""
exp2b_involution_recheck.py — Matched-RNG side-swap check (12 seeds, 24 games).
With identical policies AND identical random salts, the two games of a side-swapped pair are effectively
the same match, so pair ties are expected by construction. The informative outputs are the raw Game-1
South record (spawn-side bias) and the maximum mirrored progress asymmetry.
Output: paper_arxiv/experiments/results/exp2b_involution_recheck.json
"""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)
from exp2_behavioral_ablations import (
    resolve_primary_ckpt, make_base_cfg, run_batched_side_swapped_condition, WholeBrainNeuralEngineWrapper,
)


def main() -> None:
    ckpt = resolve_primary_ckpt()
    seeds = [2000 + 97 * k for k in range(24)][:12]
    eng = WholeBrainNeuralEngineWrapper(make_base_cfg(ckpt, "turf_objective", 2026))
    res, _, _ = run_batched_side_swapped_condition(ckpt, seeds, "turf_objective", eng, eng, True, {}, identical_rng_salt=True)
    pw = res["paired_win_test"]
    out = {
        "n_pos": pw["n_pos"], "n_neg": pw["n_neg"], "n_tie": pw["n_tie"],
        "full_win_rate_mean": res["full_win_rate"]["mean"], "full_win_rate_sd": res["full_win_rate"]["sd"],
        "raw_game1_south_w_l_t": res["raw_game1_south_w_l_t"],
        "max_involution_prog_asymmetry": res["max_involution_prog_asymmetry"],
    }
    print(json.dumps(out, indent=2))
    out_path = os.path.join(ROOT, "paper_arxiv", "experiments", "results", "exp2b_involution_recheck.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    eng.release_memory_and_vram()


if __name__ == "__main__":
    main()
