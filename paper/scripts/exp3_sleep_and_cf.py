# -*- coding: utf-8 -*-
"""
exp3_sleep_and_cf.py — Controlled Multi-Seed Training Ablations (Held-Out & Interventional Calibration):
  Part A: Inter-Episode 4-Phase Sleep Consolidation (`S-1..S-4`) ON vs. OFF (`N = 10` matched seeds)
  Part B: Counterfactual Open-Space Exposure Loss (`cf_loss`, Eq. 7) ON vs. OFF (`N = 10` matched seeds),
          evaluated strictly on UNSEEN HELD-OUT natural exposed states AND calibrated against true
          interventional simulator rollouts (`M4` active escape vs. `M7` open-space idle)!
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from stats_utils import paired_comparison, summary_stats
from exp2_behavioral_ablations import DirectEvalBrainController, run_single_match
from universal_brain_3d_core import (
    ActionStagnationHomeostasis3D,
    Brain3DConfig,
    CAND_IDX_EVADE_REFUGE,
    CAND_IDX_IDLE_OBSERVE,
    Cerebellar3DKinematicPredictorV2,
    EntorhinalHippocampal3DEngineV2,
    StandaloneTactical3DArena,
    VERB_IDLE_OBSERVE,
    WholeBrainNeuralEngineWrapper,
    compute_beta_teacher,
    compute_gene_modulated_mip_and_fr,
    compute_task_scaffold_targets,
    compute_universal_memory_compass,
    derive_ventral_fallback_from_dorsal_16,
    is_in_stealth_or_cover_refuge,
    project_pomdp_proprio,
    propose_universal_3d_candidates,
    select_policy_candidate,
)


def _collect_and_train_one_episode(
    cfg: Brain3DConfig,
    eng: WholeBrainNeuralEngineWrapper,
    arena_seed: int,
    max_ticks: int = 240,
    capture_exposed_rollouts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    acfg = Brain3DConfig(task_profile=cfg.task_profile, seed=arena_seed)
    arena = StandaloneTactical3DArena(acfg, opponent_skill=0.65)
    hippo = EntorhinalHippocampal3DEngineV2(cfg)
    hippo.set_learning_context(True)
    cereb = Cerebellar3DKinematicPredictorV2(cfg)
    homeo = ActionStagnationHomeostasis3D(cfg.world_min, cfg.world_max)
    vent_prev: Dict[str, Any] = {}

    target_pose = np.zeros(6, dtype=np.float32)
    gaze_ypd = np.array([0.0, -0.15, 6.0], dtype=np.float32)
    verb = 1
    pending_tr = None
    prev_pi = None
    dt = 1.0 / 60.0

    for tick in range(1, max_ticks + 1):
        obs = arena.step(target_pose, verb, True, gaze_ypd, dt=dt)
        ep_done = bool(obs["episode_done"]) or (tick == max_ticks)
        vent_p = derive_ventral_fallback_from_dorsal_16(
            obs["fovea_patch"], yaw_pitch=obs["self_yaw_pitch"], prev_state=vent_prev, episode_id=int(obs["episode_id"])
        )
        sp_out = hippo.step(obs, dt, ventral_override=vent_p)
        lead_xyz, _, _, _ = cereb.predict_lead_and_crossfire(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"]
        )
        prop = project_pomdp_proprio(obs["proprio"], sp_out["believed_self_pos"], cfg.world_min, cfg.world_max, cfg)
        mc24 = compute_universal_memory_compass(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop, hippo.field_map, eng.profile
        )
        c_poses, c_verbs = propose_universal_3d_candidates(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop,
            mc24, hippo.field_map, lead_xyz, sp_out["active_gaze_ypd"], cfg.world_min, cfg.world_max, eng.profile
        )

        if tick % 4 == 1 or ep_done or bool(sp_out["respawn_event_flag"]):
            # If requested, capture genuine natural exposed states and compute TRUE interventional survival of M4 vs M7!
            cur_haz = max(float(obs["hazard_level"]), float(sp_out["believed_hazard"]))
            if (
                capture_exposed_rollouts is not None
                and len(capture_exposed_rollouts) < 28
                and (not is_in_stealth_or_cover_refuge(prop))
                and cur_haz >= 0.25
            ):
                def _rollout_true_surv(cand_i: int) -> float:
                    ar_sim = copy.deepcopy(arena)
                    pose_i = c_poses[cand_i].copy()
                    verb_i = int(c_verbs[cand_i])
                    gz_i = sp_out["active_gaze_ypd"].copy()
                    splatted = False
                    peak_h = float(obs["hazard_level"])
                    # Roll out 3 cognitive intervals (12 physics ticks = 0.20s, plus projectile travel to 20 ticks)
                    for _ in range(20):
                        o_sim = ar_sim.step(pose_i, verb_i, True, gz_i, dt=dt)
                        if bool(o_sim.get("respawn_flag", False)):
                            splatted = True
                            break
                        peak_h = max(peak_h * 0.92, float(o_sim["hazard_level"]))
                        if cand_i == CAND_IDX_IDLE_OBSERVE:
                            pose_i[0:3] = (ar_sim.self_pos - 0.5 * (ar_sim.wmin + ar_sim.wmax)) / (0.5 * ar_sim.span)
                            verb_i = VERB_IDLE_OBSERVE
                    if splatted:
                        return 0.05
                    return float(np.clip(1.0 - 0.85 * peak_h, 0.05, 0.99))

                y_m4 = _rollout_true_surv(CAND_IDX_EVADE_REFUGE)
                y_m7 = _rollout_true_surv(CAND_IDX_IDLE_OBSERVE)
                capture_exposed_rollouts.append({
                    "fovea_patch": obs["fovea_patch"].copy(),
                    "ventral_patch": vent_p.copy(),
                    "proprio": prop.copy(),
                    "mem_compass": mc24.copy(),
                    "mec_grid_code": sp_out["mec_grid_code"].copy(),
                    "hd_ring_code": sp_out["hd_ring_code"].copy(),
                    "believed_self_pos": sp_out["believed_self_pos"].copy(),
                    "cand_poses": c_poses.copy(),
                    "cand_verbs": c_verbs.copy(),
                    "true_surv_m4": y_m4,
                    "true_surv_m7": y_m7,
                })

            gwt_out = eng.infer_conscious_gwt_batch([{
                "slot_id": 0,
                "episode_id": int(obs["episode_id"]),
                "respawn_flag": bool(sp_out["respawn_event_flag"]),
                "enable_learning": True,
                "fovea_patch": obs["fovea_patch"],
                "ventral_patch": vent_p,
                "proprio": prop,
                "mem_compass": mc24,
                "mec_grid_code": sp_out["mec_grid_code"],
                "hd_ring_code": sp_out["hd_ring_code"],
                "believed_self_pos": sp_out["believed_self_pos"],
                "cand_poses": c_poses,
                "cand_verbs": c_verbs,
            }])[0]

            q_star, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                gwt_out["q_values"], gwt_out["survival_probs"], gwt_out["pred_adv_probs"],
                gwt_out["gwt_workspace"], c_poses, c_verbs, prop, mc24, prev_pi, cfg,
                habituation_trace=homeo.habituation_trace,
                stagnation_level=homeo.stagnation_level,
                q_raw_values=gwt_out.get("q_raw_values"),
            )
            prev_pi = pi.copy()
            beta_t = compute_beta_teacher(cfg.policy_mode, eng.transitions_recorded, cfg.anneal_transitions)
            sp_dict = {
                "believed_self_pos": sp_out["believed_self_pos"],
                "believed_hazard": sp_out["believed_hazard"],
                "mem_compass": mc24,
                "believed_entities": sp_out["believed_entities"],
                "believed_count": sp_out["believed_count"],
            }
            tq, _, tidx = compute_task_scaffold_targets(sp_dict, obs, c_poses, c_verbs, cfg.world_min, cfg.world_max, eng.profile)
            if beta_t > 0.01 and ((tick * 1664525 + 1013904223) & 0xFFFF) < int(beta_t * 65535.0):
                c_idx = tidx
            else:
                c_idx = select_policy_candidate(q_star, pi, trainable=True, rng_seed_val=tick + arena_seed)

            eng.notify_chosen_action(0, gwt_out["cand_feats32"][c_idx].copy())
            r_int = homeo.update_and_compute_intrinsic(
                episode_id=int(obs["episode_id"]),
                pos_xz=sp_out["believed_self_pos"][[0, 2]],
                proprio=prop,
                chosen_idx=c_idx,
                step_extrinsic_reward=float(obs["last_step_reward"]),
                world_surprise=float(gwt_out["world_surprise"]),
                self_surprise=float(gwt_out["self_surprise"]),
                respawn_flag=bool(sp_out["respawn_event_flag"]),
            )
            cur_state_snap = {
                "fovea_patch": obs["fovea_patch"].copy(),
                "ventral_patch": vent_p.copy(),
                "proprio": prop.copy(),
                "mem_compass": mc24.copy(),
                "mec_grid_code": sp_out["mec_grid_code"].copy(),
                "hd_ring_code": sp_out["hd_ring_code"].copy(),
                "cand_feats32": gwt_out["cand_feats32"].copy(),
            }
            if pending_tr is not None:
                rew_acc = float(obs["cumulative_score"]) - float(pending_tr["cum_score"])
                pending_tr["reward"] = float(np.clip(rew_acc + r_int, -25.0, 30.0))
                pending_tr["next_hazard"] = cur_haz
                pending_tr["obj_adv_delta"] = float(prop[15]) - float(pending_tr["obj_prog"])
                pending_tr["done"] = ep_done
                pending_tr["respawn"] = bool(sp_out["respawn_event_flag"])
                pending_tr["episode_score"] = float(obs["cumulative_score"])
                pending_tr["next_state"] = cur_state_snap
                eng.record_transition_and_maybe_train(pending_tr)

            pending_tr = {
                "slot_id": 0,
                "episode_id": int(obs["episode_id"]),
                "enable_learning": bool(cfg.enable_learning),
                "cum_score": float(obs["cumulative_score"]),
                "obj_prog": float(prop[15]),
                "hazard": cur_haz,
                "fovea_patch": obs["fovea_patch"].copy(),
                "ventral_patch": vent_p.copy(),
                "proprio": prop.copy(),
                "mem_compass": mc24.copy(),
                "mec_grid_code": sp_out["mec_grid_code"].copy(),
                "hd_ring_code": sp_out["hd_ring_code"].copy(),
                "gwt_workspace": gwt_out["gwt_workspace"].copy(),
                "rssm_h": gwt_out["rssm_h"].copy(),
                "cand_feats32": gwt_out["cand_feats32"].copy(),
                "chosen_cand_feat32": gwt_out["cand_feats32"][c_idx].copy(),
                "chosen_idx": c_idx,
                "chosen_verb": int(c_verbs[c_idx]),
                "proto_sym_id": int(gwt_out["proto_sym_id"]),
                "world_surprise": float(gwt_out["world_surprise"]),
                "self_surprise": float(gwt_out["self_surprise"]),
                "teacher_q": tq.copy(),
                "teacher_idx": int(tidx),
                "beta_dagger": float(beta_t),
            }
            target_pose = c_poses[c_idx].copy()
            target_pose[5] = float(gwt_out["speed_mod"][c_idx])
            verb = int(c_verbs[c_idx])
            gaze_ypd = sp_out["active_gaze_ypd"].copy()

        if ep_done:
            break

    return {"cum_score": float(arena.cumulative_score)}


def run_part_a_sleep_ablation(n_seeds: int = 10, n_episodes: int = 5) -> Dict[str, Any]:
    sleep_win_rates: List[float] = []
    nosleep_win_rates: List[float] = []
    sleep_q_abs_max: List[float] = []
    nosleep_q_abs_max: List[float] = []
    sleep_curve = np.zeros(n_episodes, dtype=np.float64)
    nosleep_curve = np.zeros(n_episodes, dtype=np.float64)

    for s_i in range(n_seeds):
        base_seed = 5000 + 131 * s_i
        cfg_s = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", anneal_transitions=320, enable_learning=True, seed=base_seed)
        cfg_ns = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", anneal_transitions=320, enable_learning=True, seed=base_seed)

        eng_s = WholeBrainNeuralEngineWrapper(cfg_s)
        eng_ns = WholeBrainNeuralEngineWrapper(cfg_ns)

        for ep_k in range(n_episodes):
            ep_seed = base_seed + ep_k * 19
            r_s = _collect_and_train_one_episode(cfg_s, eng_s, arena_seed=ep_seed, max_ticks=220)
            r_ns = _collect_and_train_one_episode(cfg_ns, eng_ns, arena_seed=ep_seed, max_ticks=220)
            sleep_curve[ep_k] += r_s["cum_score"] / float(n_seeds)
            nosleep_curve[ep_k] += r_ns["cum_score"] / float(n_seeds)
            eng_s.run_sleep_consolidation(epochs=6)

        eval_cfg_s = Brain3DConfig(task_profile="turf_objective", policy_mode="pure_neural", enable_learning=False, seed=base_seed + 900)
        eval_cfg_ns = Brain3DConfig(task_profile="turf_objective", policy_mode="pure_neural", enable_learning=False, seed=base_seed + 900)

        w_acc = 0.0
        for g_k in range(2):
            m_seed = 8000 + s_i * 53 + g_k * 7
            m1 = run_single_match(
                m_seed,
                DirectEvalBrainController(eval_cfg_s, eng_s, slot_id=30),
                DirectEvalBrainController(eval_cfg_ns, eng_ns, slot_id=31),
            )
            m2 = run_single_match(
                m_seed,
                DirectEvalBrainController(eval_cfg_ns, eng_ns, slot_id=32),
                DirectEvalBrainController(eval_cfg_s, eng_s, slot_id=33),
            )
            w_acc += 0.5 * (m1["win_score_south"] + (1.0 - m2["win_score_south"]))
        w_pair = w_acc / 2.0
        sleep_win_rates.append(w_pair)
        nosleep_win_rates.append(1.0 - w_pair)

        q_s_max, q_ns_max = 0.0, 0.0
        if len(eng_s.replay) >= 8:
            for b in [eng_s.replay[k] for k in range(min(16, len(eng_s.replay)))]:
                o_s = eng_s.infer_s1_reflex_batch([{
                    "slot_id": 99, "fovea_patch": b["fovea_patch"], "ventral_patch": b["ventral_patch"],
                    "proprio": b["proprio"], "mem_compass": b["mem_compass"], "mec_grid_code": b["mec_grid_code"],
                    "hd_ring_code": b["hd_ring_code"], "gwt_workspace": b["gwt_workspace"], "rssm_h": b["rssm_h"],
                    "z_topdown_delta": np.zeros(32, dtype=np.float32),
                    "cand_poses": b["cand_feats32"][:, :6], "cand_verbs": np.zeros(8, dtype=np.uint8),
                }])[0]
                o_ns = eng_ns.infer_s1_reflex_batch([{
                    "slot_id": 98, "fovea_patch": b["fovea_patch"], "ventral_patch": b["ventral_patch"],
                    "proprio": b["proprio"], "mem_compass": b["mem_compass"], "mec_grid_code": b["mec_grid_code"],
                    "hd_ring_code": b["hd_ring_code"], "gwt_workspace": b["gwt_workspace"], "rssm_h": b["rssm_h"],
                    "z_topdown_delta": np.zeros(32, dtype=np.float32),
                    "cand_poses": b["cand_feats32"][:, :6], "cand_verbs": np.zeros(8, dtype=np.uint8),
                }])[0]
                q_s_max = max(q_s_max, float(np.max(np.abs(o_s["q_values"]))))
                q_ns_max = max(q_ns_max, float(np.max(np.abs(o_ns["q_values"]))))
        sleep_q_abs_max.append(q_s_max)
        nosleep_q_abs_max.append(q_ns_max)

        eng_s.release_memory_and_vram()
        eng_ns.release_memory_and_vram()
        print(f"[Exp3A] Seed {s_i+1}/{n_seeds} | Sleep vs No-Sleep WinRate={w_pair*100:5.1f}%")

    return {
        "n_seeds": n_seeds,
        "sleep_head2head_win_rate": summary_stats(sleep_win_rates),
        "sleep_vs_nosleep_test": paired_comparison(sleep_win_rates, nosleep_win_rates),
        "sleep_q_abs_max": summary_stats(sleep_q_abs_max),
        "nosleep_q_abs_max": summary_stats(nosleep_q_abs_max),
        "mean_episode_return_curve_sleep": [round(float(x), 3) for x in sleep_curve],
        "mean_episode_return_curve_nosleep": [round(float(x), 3) for x in nosleep_curve],
    }


def run_part_b_counterfactual_loss_ablation(n_seeds: int = 10) -> Dict[str, Any]:
    """
    Non-circular Held-Out & True Interventional Rollout Calibration of Eq. (7):
      - Trains `CF_ON` vs `CF_OFF` on identical training episodes (`seed_v + 1..3`).
      - Evaluates strictly on UNSEEN HELD-OUT episodes (`seed_v + 501..503`) at naturally occurring
        exposed states (`not in_refuge` and `hazard >= 0.25`, zero artificial proprio mutation!),
        comparing predicted survival against TRUE interventional simulator rollouts (`M4` vs `M7`).
    """
    import brain3d_neural_engine as bne

    cf_on_s_m4, cf_on_s_m7, cf_on_gap = [], [], []
    cf_off_s_m4, cf_off_s_m7, cf_off_gap = [], [], []
    true_s_m4_all, true_s_m7_all = [], []
    brier_m7_on, brier_m7_off = [], []
    heldout_states_counts: List[int] = []

    orig_mask = float(bne.SHIELD_HAZARD_CF_MASK)

    try:
        for s_i in range(n_seeds):
            seed_v = 6200 + 89 * s_i
            cfg_on = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", enable_learning=True, seed=seed_v)
            cfg_off = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", enable_learning=True, seed=seed_v)

            eng_on = WholeBrainNeuralEngineWrapper(cfg_on)
            eng_off = WholeBrainNeuralEngineWrapper(cfg_off)

            bne.SHIELD_HAZARD_CF_MASK = 999.0
            cfg_col = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", enable_learning=False, seed=seed_v)
            eng_col = WholeBrainNeuralEngineWrapper(cfg_col)
            for ep_off in (1, 2, 3):
                _collect_and_train_one_episode(cfg_col, eng_col, arena_seed=seed_v + ep_off, max_ticks=260)

            for k in range(len(eng_col.replay)):
                item = dict(eng_col.replay[k])
                prio = float(eng_col.replay_priorities[k])
                eng_on.replay.push(item, prio)
                eng_off.replay.push(dict(item), prio)
            for ch_tup in eng_col.chunk4_ring:
                eng_on.chunk4_ring.append(ch_tup)
                eng_off.chunk4_ring.append(ch_tup)

            # Collect strictly unseen HELD-OUT natural exposed states + true interventional M4/M7 rollouts
            heldout_exposed: List[Dict[str, Any]] = []
            cfg_ho = Brain3DConfig(task_profile="turf_objective", policy_mode="dagger", enable_learning=False, seed=seed_v + 500)
            eng_ho = WholeBrainNeuralEngineWrapper(cfg_ho)
            for ho_ep in (501, 502, 503):
                _collect_and_train_one_episode(
                    cfg_ho, eng_ho, arena_seed=seed_v + ho_ep, max_ticks=280, capture_exposed_rollouts=heldout_exposed
                )
                if len(heldout_exposed) >= 24:
                    break

            bne.SHIELD_HAZARD_CF_MASK = orig_mask
            np.random.seed(seed_v)
            for _ in range(110):
                eng_on._train_minibatch(batch_size=16)

            bne.SHIELD_HAZARD_CF_MASK = 999.0
            np.random.seed(seed_v)
            for _ in range(110):
                eng_off._train_minibatch(batch_size=16)

            bne.SHIELD_HAZARD_CF_MASK = orig_mask

            s4_on_l, s7_on_l, s4_off_l, s7_off_l = [], [], [], []
            t4_l, t7_l, br_on_l, br_off_l = [], [], [], []
            for k, st_item in enumerate(heldout_exposed):
                r_on = eng_on.infer_conscious_gwt_batch([{
                    "slot_id": 50, "episode_id": 2000 + k, "respawn_flag": False, "enable_learning": False,
                    "fovea_patch": st_item["fovea_patch"], "ventral_patch": st_item["ventral_patch"],
                    "proprio": st_item["proprio"], "mem_compass": st_item["mem_compass"],
                    "mec_grid_code": st_item["mec_grid_code"], "hd_ring_code": st_item["hd_ring_code"],
                    "believed_self_pos": st_item["believed_self_pos"],
                    "cand_poses": st_item["cand_poses"], "cand_verbs": st_item["cand_verbs"],
                }])[0]
                r_off = eng_off.infer_conscious_gwt_batch([{
                    "slot_id": 51, "episode_id": 2000 + k, "respawn_flag": False, "enable_learning": False,
                    "fovea_patch": st_item["fovea_patch"], "ventral_patch": st_item["ventral_patch"],
                    "proprio": st_item["proprio"], "mem_compass": st_item["mem_compass"],
                    "mec_grid_code": st_item["mec_grid_code"], "hd_ring_code": st_item["hd_ring_code"],
                    "believed_self_pos": st_item["believed_self_pos"],
                    "cand_poses": st_item["cand_poses"], "cand_verbs": st_item["cand_verbs"],
                }])[0]
                p4_on = float(r_on["survival_probs"][CAND_IDX_EVADE_REFUGE])
                p7_on = float(r_on["survival_probs"][CAND_IDX_IDLE_OBSERVE])
                p4_off = float(r_off["survival_probs"][CAND_IDX_EVADE_REFUGE])
                p7_off = float(r_off["survival_probs"][CAND_IDX_IDLE_OBSERVE])
                y4 = float(st_item["true_surv_m4"])
                y7 = float(st_item["true_surv_m7"])

                s4_on_l.append(p4_on); s7_on_l.append(p7_on)
                s4_off_l.append(p4_off); s7_off_l.append(p7_off)
                t4_l.append(y4); t7_l.append(y7)
                br_on_l.append((p7_on - y7) ** 2)
                br_off_l.append((p7_off - y7) ** 2)

            heldout_states_counts.append(len(s4_on_l))
            m4_on, m7_on = float(np.mean(s4_on_l)), float(np.mean(s7_on_l))
            m4_off, m7_off = float(np.mean(s4_off_l)), float(np.mean(s7_off_l))
            cf_on_s_m4.append(m4_on); cf_on_s_m7.append(m7_on); cf_on_gap.append(m4_on - m7_on)
            cf_off_s_m4.append(m4_off); cf_off_s_m7.append(m7_off); cf_off_gap.append(m4_off - m7_off)
            true_s_m4_all.append(float(np.mean(t4_l)))
            true_s_m7_all.append(float(np.mean(t7_l)))
            brier_m7_on.append(float(np.mean(br_on_l)))
            brier_m7_off.append(float(np.mean(br_off_l)))

            eng_ho.release_memory_and_vram()
            eng_col.release_memory_and_vram()
            eng_on.release_memory_and_vram()
            eng_off.release_memory_and_vram()
            print(
                f"[Exp3B] Seed {s_i+1}/{n_seeds} (N_heldout={heldout_states_counts[-1]}) | "
                f"True S(M4/M7)=({true_s_m4_all[-1]:.3f}/{true_s_m7_all[-1]:.3f}) | "
                f"CF_ON=({m4_on:.3f}/{m7_on:.3f}, Brier={brier_m7_on[-1]:.4f}) | "
                f"CF_OFF=({m4_off:.3f}/{m7_off:.3f}, Brier={brier_m7_off[-1]:.4f})"
            )
    finally:
        bne.SHIELD_HAZARD_CF_MASK = orig_mask

    return {
        "n_seeds": n_seeds,
        "evaluation_protocol": "Strictly unseen held-out episodes; unmodified natural exposed states; ground-truth interventional simulator rollouts",
        "mean_heldout_exposed_states_per_seed": float(np.mean(heldout_states_counts)),
        "true_interventional_surv_m4_escape": summary_stats(true_s_m4_all),
        "true_interventional_surv_m7_exposed_idle": summary_stats(true_s_m7_all),
        "cf_on_surv_m4_escape": summary_stats(cf_on_s_m4),
        "cf_on_surv_m7_exposed_idle": summary_stats(cf_on_s_m7),
        "cf_on_escape_minus_idle_gap": summary_stats(cf_on_gap),
        "cf_off_surv_m4_escape": summary_stats(cf_off_s_m4),
        "cf_off_surv_m7_exposed_idle": summary_stats(cf_off_s_m7),
        "cf_off_escape_minus_idle_gap": summary_stats(cf_off_gap),
        "brier_m7_cf_on": summary_stats(brier_m7_on),
        "brier_m7_cf_off": summary_stats(brier_m7_off),
        "paired_brier_improvement_test": paired_comparison(brier_m7_off, brier_m7_on),
        "paired_gap_test_cf_on_vs_off": paired_comparison(cf_on_gap, cf_off_gap),
        "paired_m7_optimism_reduction_test": paired_comparison(cf_off_s_m7, cf_on_s_m7),
    }


def main() -> None:
    t0 = time.perf_counter()
    part_a = run_part_a_sleep_ablation(n_seeds=10, n_episodes=5)
    part_b = run_part_b_counterfactual_loss_ablation(n_seeds=10)
    out = {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "part_a_sleep_consolidation": part_a,
        "part_b_counterfactual_deconfounding": part_b,
    }
    out_path = os.path.join(ROOT_DIR, "paper_arxiv", "experiments", "results", "exp3_sleep_and_cf.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[Exp3] Saved -> {out_path}")


if __name__ == "__main__":
    main()
