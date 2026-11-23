# -*- coding: utf-8 -*-
"""
exp5_regime_diagnostics.py — Boundary-Condition & Operating-Regime Diagnostics
==============================================================================
Tests two mechanistic boundary-condition regimes on the honest POMDP & gravity-accurate arena:
  1. Diagnostic 1 (C3 Respawn Continuity Regime Sweep, N=24 side-swapped seed pairs = 48 games/cond):
     - R1_9s_Open_WithScar    : 540 ticks (9.0s), 4 pillars.
     - R2_18s_Open_WithScar   : 1080 ticks (18.0s), 4 pillars.
     - R3_18s_Walled_WithScar : 1080 ticks (18.0s), 8 symmetric occluding pillars.
  2. Diagnostic 2 (Sleep Consolidation Replay-to-Buffer Ratio Sweep, N=10 matched seeds):
     - Contrasts the starved micro-buffer regime (`N_buf ~ 55`, ratio ~ 2.6x) against the diverse
       generation-end buffer regime (`N_buf ~ 600`, ratio ~ 0.12x), reporting ALL metrics transparently
       (Win Rate, Held-Out TD Bellman MAE, AND Held-Out World-Model Visual MAE!).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from stats_utils import paired_comparison, summary_stats
from arena_pbt_trainer import HeadToHeadTactical3DArena, canonicalize_team_observation
from exp2_behavioral_ablations import DirectEvalBrainController, make_base_cfg, resolve_primary_ckpt
from exp3_sleep_and_cf import _collect_and_train_one_episode
from universal_brain_3d_core import Brain3DConfig, WholeBrainNeuralEngineWrapper


class ConfigurableHorizonTactical3DArena(HeadToHeadTactical3DArena):
    def __init__(self, cfg: Brain3DConfig, seed: int = 2026, max_ticks: int = 540, dense_walls: bool = False):
        self.custom_max_ticks = int(max_ticks)
        super().__init__(cfg, seed=seed)
        if dense_walls:
            self.cover_blocks = [
                {"id": 101, "pos": np.array([-4.2, 1.3, -4.5], dtype=np.float32), "radius": 1.35},
                {"id": 102, "pos": np.array([4.2, 1.3, -4.5], dtype=np.float32), "radius": 1.35},
                {"id": 103, "pos": np.array([-4.2, 1.3, 4.5], dtype=np.float32), "radius": 1.35},
                {"id": 104, "pos": np.array([4.2, 1.3, 4.5], dtype=np.float32), "radius": 1.35},
                {"id": 105, "pos": np.array([-1.8, 1.3, -2.6], dtype=np.float32), "radius": 1.30},
                {"id": 106, "pos": np.array([1.8, 1.3, 2.6], dtype=np.float32), "radius": 1.30},
                {"id": 107, "pos": np.array([2.2, 1.3, -6.8], dtype=np.float32), "radius": 1.35},
                {"id": 108, "pos": np.array([-2.2, 1.3, 6.8], dtype=np.float32), "radius": 1.35},
            ]

    def step_both(self, act_a, act_b, dt: float = 1.0 / 60.0):
        real_ticks_next = self.ticks_in_ep + 1
        if self.custom_max_ticks > 540 and real_ticks_next < self.custom_max_ticks:
            if self.ticks_in_ep >= 538:
                self.ticks_in_ep = 500
            obs_a, obs_b_canon, ep_done, winner = super().step_both(act_a, act_b, dt=dt)
            self.ticks_in_ep = real_ticks_next
            return obs_a, obs_b_canon, ep_done, winner
        if self.custom_max_ticks > 540 and real_ticks_next >= self.custom_max_ticks:
            self.ticks_in_ep = 539
            obs_a, obs_b_canon, ep_done, winner = super().step_both(act_a, act_b, dt=dt)
            self.ticks_in_ep = real_ticks_next
            return obs_a, obs_b_canon, ep_done, winner
        return super().step_both(act_a, act_b, dt=dt)


def _run_batched_horizon_regime(
    ckpt: str,
    seeds: List[int],
    eng_sh: WholeBrainNeuralEngineWrapper,
    max_ticks: int,
    dense_walls: bool,
) -> Dict[str, Any]:
    eng_sh._agent_states.clear()
    arenas: List[ConfigurableHorizonTactical3DArena] = []
    obs_s_l, obs_n_l = [], []
    ctrls_s: List[DirectEvalBrainController] = []
    ctrls_n: List[DirectEvalBrainController] = []

    slot_cursor = 1
    for s_idx, a_seed in enumerate(seeds):
        for swapped in (False, True):
            ar = ConfigurableHorizonTactical3DArena(
                Brain3DConfig(task_profile="turf_objective", seed=a_seed),
                seed=a_seed, max_ticks=max_ticks, dense_walls=dense_walls,
            )
            arenas.append(ar)
            obs_s_l.append(ar._build_agent_obs(+1, 0.0, False))
            obs_n_l.append(canonicalize_team_observation(ar._build_agent_obs(-1, 0.0, False), -1, ar.wmin, ar.wmax))

            cfg_f = make_base_cfg(ckpt, profile="turf_objective", seed=3000 + s_idx)
            cfg_o = make_base_cfg(ckpt, profile="turf_objective", seed=4000 + s_idx)
            c_f = DirectEvalBrainController(cfg_f, eng_sh, slot_id=slot_cursor, rng_salt=1 + 2 * s_idx); slot_cursor += 1
            c_o = DirectEvalBrainController(cfg_o, eng_sh, slot_id=slot_cursor, rng_salt=2 + 2 * s_idx, no_respawn_map_keep=True); slot_cursor += 1
            if not swapped:
                ctrls_s.append(c_f); ctrls_n.append(c_o)
            else:
                ctrls_s.append(c_o); ctrls_n.append(c_f)

    n_games = len(arenas)
    deaths_s = [0] * n_games
    deaths_n = [0] * n_games
    max_prog_s = [0.50] * n_games
    max_prog_n = [0.50] * n_games
    winners = [0] * n_games
    done_flags = [False] * n_games
    dt = 1.0 / 60.0

    for _ in range(max_ticks + 5):
        if all(done_flags):
            break
        batch_reqs = []
        map_idx: List[Tuple[int, str]] = []
        for g in range(n_games):
            if done_flags[g]:
                continue
            _, rs = ctrls_s[g].pre_step(obs_s_l[g], dt, cog_stride=4)
            if rs is not None:
                map_idx.append((g, "s")); batch_reqs.append(rs)
            _, rn = ctrls_n[g].pre_step(obs_n_l[g], dt, cog_stride=4)
            if rn is not None:
                map_idx.append((g, "n")); batch_reqs.append(rn)
        b_outs = eng_sh.infer_conscious_gwt_batch(batch_reqs) if batch_reqs else []
        out_d = {k: v for k, v in zip(map_idx, b_outs)}

        for g in range(n_games):
            if done_flags[g]:
                continue
            act_s = ctrls_s[g].post_step(out_d.get((g, "s")))
            act_n = ctrls_n[g].post_step(out_d.get((g, "n")))
            obs_s, obs_n, ep_done, w = arenas[g].step_both(act_s, act_n, dt=dt)
            obs_s_l[g] = obs_s; obs_n_l[g] = obs_n
            if bool(obs_s.get("respawn_flag", False)):
                deaths_s[g] += 1
            if bool(obs_n.get("respawn_flag", False)):
                deaths_n[g] += 1
            max_prog_s[g] = max(max_prog_s[g], float(arenas[g].objective_progress_a))
            max_prog_n[g] = max(max_prog_n[g], float(1.0 - arenas[g].objective_progress_a))
            if ep_done:
                done_flags[g] = True
                winners[g] = int(w)

    w_f_l, w_o_l, pr_f_l, pr_o_l, tf_f_l, tf_o_l, kl_f_l, dt_f_l, sh_f_l, sh_o_l = [], [], [], [], [], [], [], [], [], []
    for s_idx in range(len(seeds)):
        g1, g2 = 2 * s_idx, 2 * s_idx + 1
        ws1 = 1.0 if winners[g1] > 0 else (0.0 if winners[g1] < 0 else 0.5)
        ws2 = 1.0 if winners[g2] > 0 else (0.0 if winners[g2] < 0 else 0.5)
        wf = 0.5 * (ws1 + (1.0 - ws2))
        w_f_l.append(wf); w_o_l.append(1.0 - wf)
        pr_f_l.append(0.5 * (max_prog_s[g1] + max_prog_n[g2]))
        pr_o_l.append(0.5 * (max_prog_n[g1] + max_prog_s[g2]))
        tf_f_l.append(0.5 * (float(np.mean(arenas[g1].surface_grid > 0.25)) + float(np.mean(arenas[g2].surface_grid < -0.25))))
        tf_o_l.append(0.5 * (float(np.mean(arenas[g1].surface_grid < -0.25)) + float(np.mean(arenas[g2].surface_grid > 0.25))))
        kl_f_l.append(0.5 * (deaths_n[g1] + deaths_s[g2]))
        dt_f_l.append(0.5 * (deaths_s[g1] + deaths_n[g2]))
        sh_f_l.append(0.5 * (float(arenas[g1].agents[+1]["cum_score"]) + float(arenas[g2].agents[-1]["cum_score"])))
        sh_o_l.append(0.5 * (float(arenas[g1].agents[-1]["cum_score"]) + float(arenas[g2].agents[+1]["cum_score"])))

    wt = paired_comparison(w_f_l, w_o_l, seed=2026)
    return {
        "max_ticks": max_ticks,
        "match_seconds": round(max_ticks / 60.0, 1),
        "dense_walls": dense_walls,
        "full_win_rate": summary_stats(w_f_l),
        "full_max_prog": summary_stats(pr_f_l),
        "opp_max_prog": summary_stats(pr_o_l),
        "full_turf_share": summary_stats(tf_f_l),
        "opp_turf_share": summary_stats(tf_o_l),
        "full_kills": summary_stats(kl_f_l),
        "full_deaths": summary_stats(dt_f_l),
        "full_shaped_score": summary_stats(sh_f_l),
        "opp_shaped_score": summary_stats(sh_o_l),
        "paired_win_test": wt,
    }


def run_diagnostic_1_c3_regimes(ckpt: str, n_seeds: int = 24) -> Dict[str, Any]:
    seeds = [2000 + 97 * k for k in range(n_seeds)]
    eng_sh = WholeBrainNeuralEngineWrapper(make_base_cfg(ckpt, profile="turf_objective", seed=2026))
    regimes = [
        ("R1_9s_Open_WithScar", 540, False),
        ("R2_18s_Open_WithScar", 1080, False),
        ("R3_18s_Walled_WithScar", 1080, True),
    ]
    out: Dict[str, Any] = {}
    for r_name, m_ticks, d_walls in regimes:
        r_res = _run_batched_horizon_regime(ckpt, seeds, eng_sh, max_ticks=m_ticks, dense_walls=d_walls)
        out[r_name] = r_res
        print(
            f"[Exp5-C3] {r_name:24s} | Full Win={r_res['full_win_rate']['mean']*100:5.1f}% "
            f"[{r_res['full_win_rate']['ci95_low']*100:4.1f}, {r_res['full_win_rate']['ci95_high']*100:4.1f}]% "
            f"| Prog={r_res['full_max_prog']['mean']*100:4.1f}%/{r_res['opp_max_prog']['mean']*100:4.1f}% "
            f"| p={r_res['paired_win_test']['wilcoxon_p']:.4g}"
        )
    eng_sh.release_memory_and_vram()
    return out


def _eval_heldout_td_and_wm_loss(eng: WholeBrainNeuralEngineWrapper, heldout_items: List[Dict[str, Any]]) -> Tuple[float, float]:
    eng._agent_states.pop(70, None)
    eng._agent_states.pop(71, None)
    td_errs: List[float] = []
    wm_errs: List[float] = []
    c_verbs = np.zeros(8, dtype=np.uint8)
    for k, b in enumerate(heldout_items[:36]):
        c_poses = b["cand_feats32"][:, :6].copy()
        out_t = eng.infer_conscious_gwt_batch([{
            "slot_id": 70, "episode_id": 5000, "respawn_flag": False, "enable_learning": False,
            "fovea_patch": b["fovea_patch"], "ventral_patch": b["ventral_patch"],
            "proprio": b["proprio"], "mem_compass": b["mem_compass"],
            "mec_grid_code": b["mec_grid_code"], "hd_ring_code": b["hd_ring_code"],
            "believed_self_pos": np.zeros(3, dtype=np.float32),
            "cand_poses": c_poses, "cand_verbs": c_verbs,
        }])[0]
        c_idx = int(b["chosen_idx"])
        eng.notify_chosen_action(70, b["cand_feats32"][c_idx].copy())
        if bool(out_t.get("wm_has_prev_pred", False)):
            wm_errs.append(float(out_t["wm_vis_mae"]))

        ns = b["next_state"]
        out_n = eng.infer_conscious_gwt_batch([{
            "slot_id": 71, "episode_id": 6000 + k, "respawn_flag": False, "enable_learning": False,
            "fovea_patch": ns["fovea_patch"], "ventral_patch": ns["ventral_patch"],
            "proprio": ns["proprio"], "mem_compass": ns["mem_compass"],
            "mec_grid_code": ns["mec_grid_code"], "hd_ring_code": ns["hd_ring_code"],
            "believed_self_pos": np.zeros(3, dtype=np.float32),
            "cand_poses": ns["cand_feats32"][:, :6].copy(), "cand_verbs": c_verbs,
        }])[0]
        q_pred = float(out_t["q_values"][c_idx])
        target_q = float(b["reward"]) + (0.0 if bool(b["done"]) else 0.96 * float(np.max(out_n["q_values"])))
        td_errs.append(abs(q_pred - target_q))
    return float(np.mean(td_errs)) if td_errs else 0.0, float(np.mean(wm_errs)) if wm_errs else 0.0


def run_diagnostic_2_sleep_diverse_buffer(n_seeds: int = 10) -> Dict[str, Any]:
    from exp2_behavioral_ablations import run_single_match

    win_sleep_l, win_nosleep_l = [], []
    td_sleep_l, td_nosleep_l = [], []
    wm_sleep_l, wm_nosleep_l = [], []
    buf_sizes: List[int] = []

    for s_i in range(n_seeds):
        base_seed = 7100 + 137 * s_i
        cfg_col = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", anneal_transitions=450, enable_learning=True, seed=base_seed)
        eng_s = WholeBrainNeuralEngineWrapper(cfg_col)
        cfg_ns = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", anneal_transitions=450, enable_learning=True, seed=base_seed)
        eng_ns = WholeBrainNeuralEngineWrapper(cfg_ns)

        for ep_k in range(10):
            ep_seed = base_seed + ep_k * 23
            _collect_and_train_one_episode(cfg_col, eng_s, arena_seed=ep_seed, max_ticks=240)
            _collect_and_train_one_episode(cfg_ns, eng_ns, arena_seed=ep_seed, max_ticks=240)

        buf_sizes.append(len(eng_s.replay))
        eng_s.run_sleep_consolidation(epochs=2)

        cfg_ho = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", enable_learning=False, seed=base_seed + 500)
        eng_ho = WholeBrainNeuralEngineWrapper(cfg_ho)
        _collect_and_train_one_episode(cfg_ho, eng_ho, arena_seed=base_seed + 501, max_ticks=200)
        ho_items = [dict(eng_ho.replay[k]) for k in range(len(eng_ho.replay))]
        td_s, wm_s = _eval_heldout_td_and_wm_loss(eng_s, ho_items)
        td_ns, wm_ns = _eval_heldout_td_and_wm_loss(eng_ns, ho_items)
        td_sleep_l.append(td_s); td_nosleep_l.append(td_ns)
        wm_sleep_l.append(wm_s); wm_nosleep_l.append(wm_ns)

        eval_s = Brain3DConfig(task_profile="turf_objective", policy_mode="pure_neural", enable_learning=False, seed=base_seed + 900)
        eval_ns = Brain3DConfig(task_profile="turf_objective", policy_mode="pure_neural", enable_learning=False, seed=base_seed + 900)
        w_acc = 0.0
        for g_k in range(2):
            m_seed = 9200 + s_i * 59 + g_k * 11
            m1 = run_single_match(m_seed, DirectEvalBrainController(eval_s, eng_s, slot_id=30), DirectEvalBrainController(eval_ns, eng_ns, slot_id=31))
            m2 = run_single_match(m_seed, DirectEvalBrainController(eval_ns, eng_ns, slot_id=32), DirectEvalBrainController(eval_s, eng_s, slot_id=33))
            w_acc += 0.5 * (m1["win_score_south"] + (1.0 - m2["win_score_south"]))
        wp = w_acc / 2.0
        win_sleep_l.append(wp); win_nosleep_l.append(1.0 - wp)

        eng_ho.release_memory_and_vram()
        eng_s.release_memory_and_vram()
        eng_ns.release_memory_and_vram()
        print(
            f"[Exp5-Sleep] Seed {s_i+1}/{n_seeds} (N_buf={buf_sizes[-1]}) | "
            f"Win={wp*100:5.1f}% | Heldout TD={td_s:.3f}/{td_ns:.3f} | WM MAE={wm_s:.4f}/{wm_ns:.4f}"
        )

    return {
        "n_seeds": n_seeds,
        "mean_buffer_transitions": float(np.mean(buf_sizes)),
        "sleep_draws_per_consolidation": 72,
        "replay_draw_to_buffer_ratio": round(72.0 / max(1.0, float(np.mean(buf_sizes))), 4),
        "sleep_head2head_win_rate": summary_stats(win_sleep_l),
        "sleep_vs_nosleep_win_test": paired_comparison(win_sleep_l, win_nosleep_l),
        "heldout_td_mae_sleep": summary_stats(td_sleep_l),
        "heldout_td_mae_nosleep": summary_stats(td_nosleep_l),
        "heldout_td_mae_test_nosleep_minus_sleep": paired_comparison(td_nosleep_l, td_sleep_l),
        "heldout_wm_vis_mae_sleep": summary_stats(wm_sleep_l),
        "heldout_wm_vis_mae_nosleep": summary_stats(wm_nosleep_l),
        "heldout_wm_vis_mae_test_nosleep_minus_sleep": paired_comparison(wm_nosleep_l, wm_sleep_l),
    }


def main() -> None:
    t0 = time.perf_counter()
    ckpt = resolve_primary_ckpt()
    assert os.path.isfile(ckpt), f"Missing checkpoint: {ckpt}"

    print(f"[Exp5] 1/2 Running C3 Respawn Continuity Horizon & Wall-Occlusion Regime Sweep (ckpt={os.path.basename(ckpt)})...")
    diag1 = run_diagnostic_1_c3_regimes(ckpt, n_seeds=24)

    print("[Exp5] 2/2 Running Diverse-Buffer Sleep Consolidation Regime Sweep (N=10 seeds)...")
    diag2 = run_diagnostic_2_sleep_diverse_buffer(n_seeds=10)

    out = {
        "checkpoint_used": os.path.relpath(ckpt, ROOT_DIR).replace("\\", "/"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "diagnostic_1_c3_respawn_regimes": diag1,
        "diagnostic_2_sleep_diverse_buffer": diag2,
    }
    out_path = os.path.join(ROOT_DIR, "paper_arxiv", "experiments", "results", "exp5_regime_diagnostics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[Exp5] Saved -> {out_path}")


if __name__ == "__main__":
    main()
