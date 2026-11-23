# -*- coding: utf-8 -*-
"""
exp4_consciousness_and_wm_metrics.py — Pillar C: Functional Consciousness Indicators, Surrogates & World Model
=============================================================================================================
Rigorous Statistical & De-Confounded Design:
  1. Proposition 3 Extended Monte Carlo (`N = 50` trials): reports BOTH raw un-debiased and Wilks-debiased
     Phi_G across Off-Diagonal Ridge (`lam=0.14`), matched Isotropic Ridge (`lam=0.14` & `0.55`), OLS,
     AR(1) autocorrelated common input (`rho=0.9`), AND the true cut `{0,1}|{2,3,4}` of the 2+3 block system!
  2. Matched All-Step Paired Comparison (`Trained` vs `Untrained`) + Temporal-Permutation Surrogate Control
     (proving Phi_G > 0 comes from lagged temporal dynamics rather than marginal variance gating!).
  3. Within-Episode Paired Comparison (`Crisis` vs `Cruise`) on identical episodes.
  4. Surgical GNWT Habituation Ablation (`hab_trace = zeros` with `novelty_5` & phasic ignition branch fixed!).
"""

from __future__ import annotations

from collections import deque
import json
import math
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

from stats_utils import paired_comparison, summary_stats, unpaired_comparison
from arena_pbt_trainer import HeadToHeadTactical3DArena, canonicalize_team_observation
from exp2_behavioral_ablations import resolve_primary_ckpt
from universal_brain_3d_core import (
    ActionStagnationHomeostasis3D,
    Brain3DConfig,
    Cerebellar3DKinematicPredictorV2,
    EntorhinalHippocampal3DEngineV2,
    PROPRIO_IDX_RESOURCE_CARRIER,
    RSSM_STOCH_CATS,
    WholeBrainNeuralEngineWrapper,
    _FIVE_MODULE_BIPARTITIONS,
    apply_checkpoint_genes_to_cfg,
    compute_gene_modulated_mip_and_fr,
    compute_oizumi_phi_g_2016,
    compute_universal_memory_compass,
    derive_ventral_fallback_from_dorsal_16,
    extract_5module_activity_vec,
    load_checkpoint_contract_meta,
    project_pomdp_proprio,
    propose_universal_3d_candidates,
    select_policy_candidate,
)


def _eval_var_phi_g_detailed(
    hist_mat_tx5: np.ndarray,
    mode: str = "offdiag",
    lam: float = 0.14,
    debias: bool = True,
    fixed_cut: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
) -> float:
    mat_x = np.asarray(hist_mat_tx5, dtype=np.float64).T
    k_pairs = mat_x.shape[1] - 1
    w_t = np.power(0.98, np.arange(k_pairs - 1, -1, -1, dtype=np.float64))
    w_t /= np.sum(w_t)
    n_eff = 1.0 / np.sum(w_t * w_t)
    x_c = mat_x - np.mean(mat_x, axis=1, keepdims=True)
    x_std = np.std(x_c, axis=1, keepdims=True)
    z_5 = np.where(x_std > 1e-5, (x_c / np.maximum(x_std, 1e-5)) * np.clip(x_std / 0.08, 0.0, 1.0), 0.0)
    x_past, y_curr = z_5[:, :-1], z_5[:, 1:]
    eye5 = np.eye(5, dtype=np.float64)
    sig_xx_raw = (x_past * w_t[None, :]) @ x_past.T
    sig_yx_raw = (y_curr * w_t[None, :]) @ x_past.T

    if mode == "offdiag":
        a_full = np.zeros((5, 5), dtype=np.float64)
        for r_i in range(5):
            off_p = eye5.copy()
            off_p[r_i, r_i] = 0.0
            a_full[r_i, :] = np.linalg.solve(sig_xx_raw + lam * off_p + 1e-3 * eye5, sig_yx_raw[r_i, :])
    elif mode == "isotropic":
        a_full = sig_yx_raw @ np.linalg.inv(sig_xx_raw + lam * eye5)
    else:
        a_full = sig_yx_raw @ np.linalg.inv(sig_xx_raw + 1e-6 * eye5)

    res_full = y_curr - (a_full @ x_past)
    sig_cond = ((res_full * w_t[None, :]) @ res_full.T) + 0.05 * eye5
    sig_cond = 0.5 * (sig_cond + sig_cond.T)
    eig_c, vec_c = np.linalg.eigh(sig_cond)
    sig_cond = (vec_c * np.maximum(eig_c, 0.03)) @ vec_c.T
    _, ldet_full = np.linalg.slogdet(sig_cond)
    sig_xx = sig_xx_raw + 0.05 * eye5
    sig_yx = a_full @ sig_xx
    ln2 = math.log(2.0)

    cuts = [fixed_cut] if fixed_cut is not None else list(_FIVE_MODULE_BIPARTITIONS)
    best_norm = float("inf")
    best_bits = 0.0
    for p_a, p_b in cuts:
        ia, ib, iab, iba = np.ix_(p_a, p_a), np.ix_(p_b, p_b), np.ix_(p_a, p_b), np.ix_(p_b, p_a)
        inv_aa, inv_bb = np.linalg.inv(sig_xx[ia]), np.linalg.inv(sig_xx[ib])
        a_pa, a_pb = sig_yx[ia] @ inv_aa, sig_yx[ib] @ inv_bb
        a_disc = np.zeros((5, 5), dtype=np.float64)
        for _ in range(4):
            a_disc.fill(0.0)
            a_disc[ia], a_disc[ib] = a_pa, a_pb
            da = a_full - a_disc
            sig_e = 0.5 * ((sig_cond + da @ sig_xx @ da.T) + (sig_cond + da @ sig_xx @ da.T).T)
            prec_e = np.linalg.inv(sig_e)
            m_pyx = prec_e @ sig_yx
            a_pa = np.linalg.solve(prec_e[ia], m_pyx[ia] - prec_e[iab] @ a_pb @ sig_xx[iba]) @ inv_aa
            a_pb = np.linalg.solve(prec_e[ib], m_pyx[ib] - prec_e[iba] @ a_pa @ sig_xx[iab]) @ inv_bb
        a_disc.fill(0.0)
        a_disc[ia], a_disc[ib] = a_pa, a_pb
        da = a_full - a_disc
        _, ldet_e = np.linalg.slogdet(0.5 * ((sig_cond + da @ sig_xx @ da.T) + (sig_cond + da @ sig_xx @ da.T).T))
        raw_nats = 0.5 * max(0.0, float(ldet_e - ldet_full))
        d_cut = 2.0 * float(len(p_a) * len(p_b))
        bias_nats = (0.28 * d_cut / max(10.0, n_eff)) if debias else 0.0
        val_bits = max(0.0, raw_nats - bias_nats) / ln2
        k_n = float(min(len(p_a), len(p_b)))
        if val_bits / k_n < best_norm:
            best_norm = val_bits / k_n
            best_bits = val_bits
    return best_bits


def run_part_a_proposition3_benchmark(n_trials: int = 50, t_len: int = 48) -> Dict[str, Any]:
    rng = np.random.default_rng(2026)
    null_od_deb, null_od_raw = [], []
    null_iso14_deb, null_iso14_raw = [], []
    null_iso55_deb, null_ols_raw = [], []
    null_ar1_od_deb, null_ar1_od_raw = [], []
    split_mip_deb, split_truecut_deb, split_truecut_raw = [], [], []
    split_exact_mask_hits = 0
    ring_phi = []

    a_diag = np.diag([0.82, 0.78, 0.75, 0.80, 0.76])
    a_split = np.array([
        [0.55, 0.36, 0.00, 0.00, 0.00],
        [0.36, 0.55, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.50, 0.32, 0.00],
        [0.00, 0.00, 0.00, 0.50, 0.32],
        [0.00, 0.00, 0.32, 0.00, 0.50],
    ], dtype=np.float64)
    a_ring = np.zeros((5, 5), dtype=np.float64)
    for i in range(5):
        a_ring[i, i] = 0.42
        a_ring[(i + 1) % 5, i] = 0.48

    for _ in range(n_trials):
        # 1. White Gaussian Common-Input Flash
        x = np.zeros((t_len, 5), dtype=np.float64)
        for t in range(1, t_len):
            shared_flash = rng.normal(0.0, 1.35)
            x[t] = a_diag @ x[t - 1] + shared_flash + rng.normal(0.0, 0.25, size=5)
        phi_od, _ = compute_oizumi_phi_g_2016(x)
        null_od_deb.append(phi_od)
        null_od_raw.append(_eval_var_phi_g_detailed(x, "offdiag", lam=0.14, debias=False))
        null_iso14_deb.append(_eval_var_phi_g_detailed(x, "isotropic", lam=0.14, debias=True))
        null_iso14_raw.append(_eval_var_phi_g_detailed(x, "isotropic", lam=0.14, debias=False))
        null_iso55_deb.append(_eval_var_phi_g_detailed(x, "isotropic", lam=0.55, debias=True))
        null_ols_raw.append(_eval_var_phi_g_detailed(x, "ols", lam=0.0, debias=False))

        # 1b. Autocorrelated AR(1) Common-Input Flash (`rho = 0.9`)
        x_ar = np.zeros((t_len, 5), dtype=np.float64)
        u_prev = 0.0
        for t in range(1, t_len):
            u_prev = 0.90 * u_prev + math.sqrt(1.0 - 0.81) * rng.normal(0.0, 1.35)
            x_ar[t] = a_diag @ x_ar[t - 1] + u_prev + rng.normal(0.0, 0.25, size=5)
        p_ar, _ = compute_oizumi_phi_g_2016(x_ar)
        null_ar1_od_deb.append(p_ar)
        null_ar1_od_raw.append(_eval_var_phi_g_detailed(x_ar, "offdiag", lam=0.14, debias=False))

        # 2. Independent 2+3 Block Split (`{0,1} | {2,3,4}`)
        xs = np.zeros((t_len, 5), dtype=np.float64)
        for t in range(1, t_len):
            xs[t] = a_split @ xs[t - 1] + rng.normal(0.0, 0.35, size=5)
        ps, m_s = compute_oizumi_phi_g_2016(xs)
        split_mip_deb.append(ps)
        split_truecut_deb.append(_eval_var_phi_g_detailed(xs, "offdiag", lam=0.14, debias=True, fixed_cut=((0, 1), (2, 3, 4))))
        split_truecut_raw.append(_eval_var_phi_g_detailed(xs, "offdiag", lam=0.14, debias=False, fixed_cut=((0, 1), (2, 3, 4))))
        if m_s in (0b00011, 0b11100):
            split_exact_mask_hits += 1

        # 3. 5-Node Directed Ring
        xr = np.zeros((t_len, 5), dtype=np.float64)
        for t in range(1, t_len):
            xr[t] = a_ring @ xr[t - 1] + rng.normal(0.0, 0.35, size=5)
        pr, _ = compute_oizumi_phi_g_2016(xr)
        ring_phi.append(pr)

    return {
        "n_trials": n_trials,
        "common_input_null_offdiag_ridge_bits": summary_stats(null_od_deb),
        "common_input_null_offdiag_raw_undebiased_bits": summary_stats(null_od_raw),
        "common_input_null_isotropic_matched_lam014_bits": summary_stats(null_iso14_deb),
        "common_input_null_isotropic_matched_lam014_raw_bits": summary_stats(null_iso14_raw),
        "common_input_null_isotropic_ridge_bits": summary_stats(null_iso55_deb),
        "common_input_null_ols_raw_bits": summary_stats(null_ols_raw),
        "autocorr_ar1_null_offdiag_debiased_bits": summary_stats(null_ar1_od_deb),
        "autocorr_ar1_null_offdiag_raw_bits": summary_stats(null_ar1_od_raw),
        "block_split_2v3_phi_bits": summary_stats(split_mip_deb),
        "block_split_2v3_true_cut_debiased_bits": summary_stats(split_truecut_deb),
        "block_split_2v3_true_cut_raw_bits": summary_stats(split_truecut_raw),
        "block_split_exact_mip_recovery_rate": split_exact_mask_hits / float(n_trials),
        "directed_ring_phi_bits": summary_stats(ring_phi),
    }


def _run_arena_cognitive_sweep(ckpt_path: str, n_episodes: int = 24) -> Dict[str, Any]:
    cfg_tr = Brain3DConfig(
        task_profile="turf_objective", enable_learning=False, policy_mode="pure_neural",
        load_weights_path=ckpt_path, seed=2026,
    )
    meta = load_checkpoint_contract_meta(ckpt_path)
    if "genes" in meta:
        apply_checkpoint_genes_to_cfg(cfg_tr, meta["genes"])

    cfg_un = Brain3DConfig(
        task_profile="turf_objective", enable_learning=False, policy_mode="pure_neural",
        load_weights_path="", seed=7777,
    )
    if "genes" in meta:
        apply_checkpoint_genes_to_cfg(cfg_un, meta["genes"])

    eng_tr = WholeBrainNeuralEngineWrapper(cfg_tr)
    eng_opp = WholeBrainNeuralEngineWrapper(cfg_tr)
    eng_un = WholeBrainNeuralEngineWrapper(cfg_un)

    ep_un_phi_g, ep_un_phi_act, ep_un_vis_mae, ep_un_dep_mae = [], [], [], []
    ep_tr_all_phi_g, ep_tr_all_phi_act, ep_tr_shuf_phi_g = [], [], []
    ep_cruise_phi_g, ep_crisis_phi_g = [], []
    ep_cruise_phi_act, ep_crisis_phi_act = [], []
    ep_cruise_ign, ep_crisis_ign = [], []
    ep_cruise_ssurp, ep_crisis_ssurp = [], []
    paired_both_cruise_pg, paired_both_crisis_pg = [], []
    paired_both_cruise_pa, paired_both_crisis_pa = [], []
    paired_both_cruise_ss, paired_both_crisis_ss = [], []
    ep_tr_vis_mae, ep_tr_dep_mae, ep_tr_cf_spread = [], [], []

    # Surgical habituation ablation (same phasic ignition formula & novelty_5 intact, zeroing ONLY hab_trace!)
    ep_hab_on_ign, ep_hab_off_surg_ign, ep_hab_off_legacy_ign = [], [], []
    ep_hab_on_handoffs_per_s, ep_hab_off_surg_handoffs_per_s, ep_hab_off_legacy_handoffs_per_s = [], [], []
    ep_hab_on_ent, ep_hab_off_surg_ent, ep_hab_off_legacy_ent = [], [], []

    max_plan_a_q_diff = 0.0
    plan_a_states_checked = 0
    best_trace: Dict[str, Any] = {}
    shuf_rng = np.random.default_rng(2026)

    for ep_i in range(n_episodes):
        a_seed = 4100 + 83 * ep_i

        for mode_label, main_eng in (("trained", eng_tr), ("untrained", eng_un)):
            main_eng._agent_states.clear()
            eng_opp._agent_states.clear()
            arena = HeadToHeadTactical3DArena(Brain3DConfig(task_profile="turf_objective", seed=a_seed), seed=a_seed)
            obs_a = arena._build_agent_obs(+1, 0.0, False)
            obs_b = canonicalize_team_observation(arena._build_agent_obs(-1, 0.0, False), -1, arena.wmin, arena.wmax)

            hippo_a = EntorhinalHippocampal3DEngineV2(cfg_tr)
            hippo_b = EntorhinalHippocampal3DEngineV2(cfg_tr)
            cereb_a = Cerebellar3DKinematicPredictorV2(cfg_tr)
            cereb_b = Cerebellar3DKinematicPredictorV2(cfg_tr)
            homeo_a = ActionStagnationHomeostasis3D(cfg_tr.world_min, cfg_tr.world_max)
            homeo_b = ActionStagnationHomeostasis3D(cfg_tr.world_min, cfg_tr.world_max)
            mod_hist_a: deque = deque(maxlen=48)
            vent_a, vent_b = {}, {}
            prev_pi_a, prev_pi_b = None, None
            pose_a, verb_a, gz_a = np.zeros(6, dtype=np.float32), 1, np.array([0.0, -0.15, 6.0], dtype=np.float32)
            pose_b, verb_b, gz_b = np.zeros(6, dtype=np.float32), 1, np.array([0.0, -0.15, 6.0], dtype=np.float32)
            resp_seq_a, resp_seq_b = 0, 0

            cruise_pg, crisis_pg = [], []
            cruise_pa, crisis_pa = [], []
            cruise_ig, crisis_ig = [], []
            cruise_ss, crisis_ss = [], []
            all_pg, all_pa, shuf_pg, vis_maes, dep_maes, cf_spreads = [], [], [], [], [], []
            hab_on_winners, hab_surg_winners, hab_leg_winners = [], [], []
            hab_on_igs, hab_surg_igs, hab_leg_igs = [], [], []
            hab_on_ents, hab_surg_ents, hab_leg_ents = [], [], []

            trace_t, trace_haz, trace_ign, trace_phig, trace_phiact, trace_veto, trace_resp, trace_alpha = [], [], [], [], [], [], [], []

            for tick in range(1, 541):
                if obs_a.get("respawn_flag"):
                    resp_seq_a += 1
                if obs_b.get("respawn_flag"):
                    resp_seq_b += 1
                vp_a = derive_ventral_fallback_from_dorsal_16(obs_a["fovea_patch"], yaw_pitch=obs_a["self_yaw_pitch"], prev_state=vent_a, episode_id=ep_i + 1)
                vp_b = derive_ventral_fallback_from_dorsal_16(obs_b["fovea_patch"], yaw_pitch=obs_b["self_yaw_pitch"], prev_state=vent_b, episode_id=ep_i + 1)
                sp_a = hippo_a.step(obs_a, 1.0 / 60.0, ventral_override=vp_a, respawn_atom=resp_seq_a)
                sp_b = hippo_b.step(obs_b, 1.0 / 60.0, ventral_override=vp_b, respawn_atom=resp_seq_b)
                la_a, _, _, _ = cereb_a.predict_lead_and_crossfire(sp_a["believed_self_pos"], sp_a["believed_entities"], sp_a["believed_count"])
                la_b, _, _, _ = cereb_b.predict_lead_and_crossfire(sp_b["believed_self_pos"], sp_b["believed_entities"], sp_b["believed_count"])
                pr_a = project_pomdp_proprio(obs_a["proprio"], sp_a["believed_self_pos"], cfg_tr.world_min, cfg_tr.world_max, cfg_tr)
                pr_b = project_pomdp_proprio(obs_b["proprio"], sp_b["believed_self_pos"], cfg_tr.world_min, cfg_tr.world_max, cfg_tr)
                mc_a = compute_universal_memory_compass(sp_a["believed_self_pos"], sp_a["believed_entities"], sp_a["believed_count"], pr_a, hippo_a.field_map, main_eng.profile)
                mc_b = compute_universal_memory_compass(sp_b["believed_self_pos"], sp_b["believed_entities"], sp_b["believed_count"], pr_b, hippo_b.field_map, eng_opp.profile)
                cp_a, cv_a = propose_universal_3d_candidates(sp_a["believed_self_pos"], sp_a["believed_entities"], sp_a["believed_count"], pr_a, mc_a, hippo_a.field_map, la_a, sp_a["active_gaze_ypd"], cfg_tr.world_min, cfg_tr.world_max, main_eng.profile)
                cp_b, cv_b = propose_universal_3d_candidates(sp_b["believed_self_pos"], sp_b["believed_entities"], sp_b["believed_count"], pr_b, mc_b, hippo_b.field_map, la_b, sp_b["active_gaze_ypd"], cfg_tr.world_min, cfg_tr.world_max, eng_opp.profile)

                if tick % 4 == 1 or bool(sp_a["respawn_event_flag"]):
                    req_a = {
                        "slot_id": 1, "episode_id": ep_i + 1, "respawn_flag": bool(sp_a["respawn_event_flag"]),
                        "enable_learning": False, "fovea_patch": obs_a["fovea_patch"], "ventral_patch": vp_a,
                        "proprio": pr_a, "mem_compass": mc_a, "mec_grid_code": sp_a["mec_grid_code"],
                        "hd_ring_code": sp_a["hd_ring_code"], "believed_self_pos": sp_a["believed_self_pos"],
                        "cand_poses": cp_a, "cand_verbs": cv_a,
                    }
                    g_a = main_eng.infer_conscious_gwt_batch([req_a])[0]

                    if mode_label == "trained" and plan_a_states_checked < 500:
                        st_snap = main_eng._agent_states[1]["ctx_ring"][-1][1]
                        for sid_chk in (3, 4):
                            main_eng._agent_states[sid_chk] = {
                                "ep_id": ep_i + 1,
                                "rssm_h": st_snap["prev_h"].copy(),
                                "rssm_stoch": st_snap["prev_st"].copy(),
                                "gwt_w": st_snap["prev_gwt"].copy(),
                                "prev_alpha": st_snap["prev_alpha"].copy(),
                                "hab_alpha": st_snap["prev_hab"].copy(),
                                "prev_compass": st_snap["prev_comp"].copy(),
                                "self_s": st_snap["prev_self"].copy(),
                                "prev_self_err": st_snap["prev_self_err"].copy(),
                                "prev_self_surp": st_snap["prev_self_surp"],
                                "z_td": st_snap["prev_z_td"].copy(),
                                "prev_cand_feat32": st_snap["prev_act"].copy(),
                                "prev_slots": None,
                                "prev_pred_vis8": None,
                                "prev_topdown_gain_16": st_snap["prev_spot_16"].copy(),
                                "_last_vis_step": None,
                                "strat_hist": deque([row.copy() for row in st_snap["strat_seq"]], maxlen=8),
                                "ctx_ring": deque(maxlen=16),
                            }
                        req_b_on = dict(req_a); req_b_on["slot_id"] = 3
                        req_b_on["lang_cand_bias"] = np.linspace(-0.25, 0.25, 8, dtype=np.float32)
                        req_b_off = dict(req_a); req_b_off["slot_id"] = 4
                        g_b_on = main_eng.infer_conscious_gwt_batch([req_b_on])[0]
                        g_b_off = main_eng.infer_conscious_gwt_batch([req_b_off])[0]
                        max_plan_a_q_diff = max(max_plan_a_q_diff, float(np.max(np.abs(g_b_on["q_values"] - g_b_off["q_values"]))))
                        plan_a_states_checked += 1

                    if mode_label == "trained":
                        with main_eng.torch.no_grad():
                            t = main_eng._t
                            tf = main_eng.torch.float32
                            st_snap = main_eng._agent_states[1]["ctx_ring"][-1][1]
                            z_t = t(g_a["latent_z"][None, :], tf)
                            sctx_t = t(g_a["slot_ctx64"][None, :], tf)
                            mec_t = t(sp_a["mec_grid_code"][None, :], tf)
                            hd_t = t(sp_a["hd_ring_code"][None, :], tf)
                            comp_t = t(mc_a[None, :], tf)
                            h_t = t(g_a["rssm_h"][None, :], tf)
                            st_t = t(main_eng._agent_states[1]["rssm_stoch"][None, :], tf)
                            pr_t = t(pr_a[None, :], tf)
                            haz_t = main_eng.models["hazard_net"](z_t, pr_t, comp_t)
                            sseq_t = t(st_snap["strat_seq"][None, ...], tf)
                            semb_t, _, _ = main_eng.models["strat_tf"](sseq_t)
                            prev_gwt_t = t(st_snap["prev_gwt"][None, :], tf)
                            ssurp_t = t([float(st_snap["prev_self_surp"])], tf)

                            # Reconstruct exact novelty_5 so surgical Hab-OFF uses the IDENTICAL phasic ignition branch!
                            perr_m = float(np.mean(st_snap["prev_perr_16"])) * 2.4
                            cdiff = float(np.linalg.norm(mc_a - st_snap["prev_comp"])) * 0.95
                            nov5_np = np.array([[
                                perr_m,
                                min(2.0, cdiff),
                                float(g_a["world_surprise"]) * 1.35 + 0.65 * float(g_a["wm_cf_spread"]),
                                float(haz_t[0].item()) * 1.15 + 0.45 * max(0.0, 0.35 - float(pr_a[PROPRIO_IDX_RESOURCE_CARRIER])),
                                float(st_snap["prev_self_surp"]) * 1.80,
                            ]], dtype=np.float32)
                            nov5_t = t(nov5_np, tf)
                            zero_hab_t = main_eng.torch.zeros((1, 5), device=z_t.device, dtype=tf)

                            # Surgical Hab-OFF: ONLY fatigue depression disabled (`hab_trace = 0`), novelty_5 & formula IDENTICAL!
                            _, alp_surg, ign_surg, _ = main_eng.models["gwt"](
                                z_t, sctx_t, mec_t, hd_t, comp_t, h_t, st_t, pr_t, haz_t, semb_t,
                                prev_gwt_t, meta_salience=ssurp_t, hab_trace=zero_hab_t, novelty_5=nov5_t,
                            )
                            # Unadapted static branch (`hab_trace=None, novelty_5=None`)
                            _, alp_leg, ign_leg, _ = main_eng.models["gwt"](
                                z_t, sctx_t, mec_t, hd_t, comp_t, h_t, st_t, pr_t, haz_t, semb_t,
                                prev_gwt_t, meta_salience=None, hab_trace=None, novelty_5=None,
                            )
                            as_np = alp_surg[0].cpu().numpy()
                            al_np = alp_leg[0].cpu().numpy()
                            is_v = float(ign_surg[0].item())
                            il_v = float(ign_leg[0].item())

                        alp_on = g_a["gwt_alpha"]
                        hab_on_igs.append(float(g_a["ignition"]))
                        hab_surg_igs.append(is_v)
                        hab_leg_igs.append(il_v)
                        hab_on_winners.append(int(np.argmax(alp_on)))
                        hab_surg_winners.append(int(np.argmax(as_np)))
                        hab_leg_winners.append(int(np.argmax(al_np)))
                        hab_on_ents.append(float(-np.sum(alp_on * np.log(np.clip(alp_on, 1e-8, 1.0))) / math.log(5.0)))
                        hab_surg_ents.append(float(-np.sum(as_np * np.log(np.clip(as_np, 1e-8, 1.0))) / math.log(5.0)))
                        hab_leg_ents.append(float(-np.sum(al_np * np.log(np.clip(al_np, 1e-8, 1.0))) / math.log(5.0)))

                    mod_hist_a.append(extract_5module_activity_vec(g_a, sp_a["mec_grid_code"], pr_a, mc_a))
                    mip_info: Dict[str, Any] = {}
                    q_star, pi_a, _, _, _, veto_a = compute_gene_modulated_mip_and_fr(
                        g_a["q_values"], g_a["survival_probs"], g_a["pred_adv_probs"],
                        g_a["gwt_workspace"], cp_a, cv_a, pr_a, mc_a, prev_pi_a, cfg_tr,
                        habituation_trace=homeo_a.habituation_trace,
                        stagnation_level=homeo_a.stagnation_level,
                        q_raw_values=g_a.get("q_raw_values"),
                        cortical_mod_history=mod_hist_a,
                        self_surprise=float(g_a["self_surprise"]),
                        mip_info_out=mip_info,
                    )
                    prev_pi_a = pi_a.copy()
                    c_a = select_policy_candidate(q_star, pi_a, trainable=False, rng_seed_val=tick + a_seed)
                    main_eng.notify_chosen_action(1, g_a["cand_feats32"][c_a].copy())
                    pose_a = cp_a[c_a].copy()
                    pose_a[5] = float(g_a["speed_mod"][c_a])
                    verb_a = int(cv_a[c_a])
                    gz_a = sp_a["active_gaze_ypd"].copy()

                    if len(mod_hist_a) >= 10:
                        pg = float(mip_info.get("phi_causal_bits", 0.0))
                        pa = float(mip_info.get("phi_act_bits", 0.0))
                        all_pg.append(pg)
                        all_pa.append(pa)
                        if mode_label == "trained" and len(mod_hist_a) >= 24 and (len(all_pg) % 4 == 0):
                            arr_m = np.asarray(list(mod_hist_a), dtype=np.float64)
                            perm_idx = shuf_rng.permutation(arr_m.shape[0])
                            pg_sh, _ = compute_oizumi_phi_g_2016(arr_m[perm_idx])
                            shuf_pg.append(pg_sh)

                        haz_cur = max(float(obs_a["hazard_level"]), float(sp_a["believed_hazard"]))
                        if veto_a or haz_cur >= 0.30:
                            crisis_pg.append(pg); crisis_pa.append(pa)
                            crisis_ig.append(float(g_a["ignition"])); crisis_ss.append(float(g_a["self_surprise"]))
                        else:
                            cruise_pg.append(pg); cruise_pa.append(pa)
                            cruise_ig.append(float(g_a["ignition"])); cruise_ss.append(float(g_a["self_surprise"]))

                    if bool(g_a["wm_has_prev_pred"]):
                        vis_maes.append(float(g_a["wm_vis_mae"]))
                        dep_maes.append(float(g_a["wm_depth_mae_m"]))
                    cf_spreads.append(float(g_a["wm_cf_spread"]))

                    if mode_label == "trained":
                        trace_t.append(round(tick / 60.0, 3))
                        trace_haz.append(round(max(float(obs_a["hazard_level"]), float(sp_a["believed_hazard"])), 4))
                        trace_ign.append(round(float(g_a["ignition"]), 4))
                        trace_phig.append(round(float(mip_info.get("phi_causal_bits", 0.0)), 4))
                        trace_phiact.append(round(float(mip_info.get("phi_act_bits", 0.0)), 4))
                        trace_veto.append(1 if veto_a else 0)
                        trace_resp.append(1 if bool(sp_a["respawn_event_flag"]) else 0)
                        trace_alpha.append([round(float(x), 4) for x in g_a["gwt_alpha"]])

                    g_b = eng_opp.infer_conscious_gwt_batch([{
                        "slot_id": 10, "episode_id": ep_i + 1, "respawn_flag": bool(sp_b["respawn_event_flag"]),
                        "enable_learning": False, "fovea_patch": obs_b["fovea_patch"], "ventral_patch": vp_b,
                        "proprio": pr_b, "mem_compass": mc_b, "mec_grid_code": sp_b["mec_grid_code"],
                        "hd_ring_code": sp_b["hd_ring_code"], "believed_self_pos": sp_b["believed_self_pos"],
                        "cand_poses": cp_b, "cand_verbs": cv_b,
                    }])[0]
                    qs_b, pi_b, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                        g_b["q_values"], g_b["survival_probs"], g_b["pred_adv_probs"],
                        g_b["gwt_workspace"], cp_b, cv_b, pr_b, mc_b, prev_pi_b, cfg_tr,
                        habituation_trace=homeo_b.habituation_trace, stagnation_level=homeo_b.stagnation_level,
                        q_raw_values=g_b.get("q_raw_values"),
                    )
                    prev_pi_b = pi_b.copy()
                    c_b = select_policy_candidate(qs_b, pi_b, trainable=False, rng_seed_val=tick + a_seed + 17)
                    eng_opp.notify_chosen_action(10, g_b["cand_feats32"][c_b].copy())
                    pose_b = cp_b[c_b].copy()
                    pose_b[5] = float(g_b["speed_mod"][c_b])
                    verb_b = int(cv_b[c_b])
                    gz_b = sp_b["active_gaze_ypd"].copy()

                obs_a, obs_b, done, _ = arena.step_both(
                    (pose_a, verb_a, True, gz_a),
                    (pose_b, verb_b, True, gz_b),
                    dt=1.0 / 60.0,
                )
                if done:
                    break

            ep_dur_s = max(1.0, tick / 60.0)
            if mode_label == "untrained":
                ep_un_phi_g.append(float(np.mean(all_pg)) if all_pg else 0.0)
                ep_un_phi_act.append(float(np.mean(all_pa)) if all_pa else 0.0)
                ep_un_vis_mae.append(float(np.mean(vis_maes)) if vis_maes else 0.0)
                ep_un_dep_mae.append(float(np.mean(dep_maes)) if dep_maes else 0.0)
            else:
                ep_tr_all_phi_g.append(float(np.mean(all_pg)) if all_pg else 0.0)
                ep_tr_all_phi_act.append(float(np.mean(all_pa)) if all_pa else 0.0)
                ep_tr_shuf_phi_g.append(float(np.mean(shuf_pg)) if shuf_pg else 0.0)
                if cruise_pg:
                    ep_cruise_phi_g.append(float(np.mean(cruise_pg)))
                    ep_cruise_phi_act.append(float(np.mean(cruise_pa)))
                    ep_cruise_ign.append(float(np.mean(cruise_ig)))
                    ep_cruise_ssurp.append(float(np.mean(cruise_ss)))
                if crisis_pg:
                    ep_crisis_phi_g.append(float(np.mean(crisis_pg)))
                    ep_crisis_phi_act.append(float(np.mean(crisis_pa)))
                    ep_crisis_ign.append(float(np.mean(crisis_ig)))
                    ep_crisis_ssurp.append(float(np.mean(crisis_ss)))
                if cruise_pg and crisis_pg:
                    paired_both_cruise_pg.append(float(np.mean(cruise_pg)))
                    paired_both_crisis_pg.append(float(np.mean(crisis_pg)))
                    paired_both_cruise_pa.append(float(np.mean(cruise_pa)))
                    paired_both_crisis_pa.append(float(np.mean(crisis_pa)))
                    paired_both_cruise_ss.append(float(np.mean(cruise_ss)))
                    paired_both_crisis_ss.append(float(np.mean(crisis_ss)))

                ep_tr_vis_mae.append(float(np.mean(vis_maes)) if vis_maes else 0.0)
                ep_tr_dep_mae.append(float(np.mean(dep_maes)) if dep_maes else 0.0)
                ep_tr_cf_spread.append(float(np.mean(cf_spreads)) if cf_spreads else 0.0)

                sw_on = sum(1 for k in range(1, len(hab_on_winners)) if hab_on_winners[k] != hab_on_winners[k - 1])
                sw_surg = sum(1 for k in range(1, len(hab_surg_winners)) if hab_surg_winners[k] != hab_surg_winners[k - 1])
                sw_leg = sum(1 for k in range(1, len(hab_leg_winners)) if hab_leg_winners[k] != hab_leg_winners[k - 1])
                ep_hab_on_ign.append(float(np.mean(hab_on_igs)) if hab_on_igs else 0.0)
                ep_hab_off_surg_ign.append(float(np.mean(hab_surg_igs)) if hab_surg_igs else 0.0)
                ep_hab_off_legacy_ign.append(float(np.mean(hab_leg_igs)) if hab_leg_igs else 0.0)
                ep_hab_on_handoffs_per_s.append(sw_on / ep_dur_s)
                ep_hab_off_surg_handoffs_per_s.append(sw_surg / ep_dur_s)
                ep_hab_off_legacy_handoffs_per_s.append(sw_leg / ep_dur_s)
                ep_hab_on_ent.append(float(np.mean(hab_on_ents)) if hab_on_ents else 0.0)
                ep_hab_off_surg_ent.append(float(np.mean(hab_surg_ents)) if hab_surg_ents else 0.0)
                ep_hab_off_legacy_ent.append(float(np.mean(hab_leg_ents)) if hab_leg_ents else 0.0)

                score_tr = sum(trace_veto) + 5 * sum(trace_resp) + (10 if len(trace_t) >= 80 else 0)
                if score_tr > best_trace.get("score", -1):
                    best_trace = {
                        "score": score_tr,
                        "episode_seed": a_seed,
                        "t_sec": trace_t,
                        "hazard": trace_haz,
                        "ignition": trace_ign,
                        "phi_causal_bits": trace_phig,
                        "phi_act_bits": trace_phiact,
                        "veto": trace_veto,
                        "respawn": trace_resp,
                        "gwt_alpha_5": trace_alpha,
                    }

    eng_tr.release_memory_and_vram()
    eng_opp.release_memory_and_vram()
    eng_un.release_memory_and_vram()

    return {
        "n_independent_episodes": n_episodes,
        "phi_causal_bits": {
            "untrained": summary_stats(ep_un_phi_g),
            "trained_all_steps": summary_stats(ep_tr_all_phi_g),
            "trained_time_shuffled_surrogate": summary_stats(ep_tr_shuf_phi_g),
            "trained_cruise": summary_stats(ep_cruise_phi_g),
            "trained_crisis_veto": summary_stats(ep_crisis_phi_g),
            "paired_trained_all_vs_untrained": paired_comparison(ep_tr_all_phi_g, ep_un_phi_g),
            "paired_trained_vs_time_shuffled": paired_comparison(ep_tr_all_phi_g, ep_tr_shuf_phi_g),
            "test_trained_cruise_vs_untrained": unpaired_comparison(ep_cruise_phi_g, ep_un_phi_g),
            "paired_within_ep_crisis_vs_cruise": paired_comparison(paired_both_crisis_pg, paired_both_cruise_pg) if paired_both_crisis_pg else {},
        },
        "phi_act_override_bits": {
            "untrained": summary_stats(ep_un_phi_act),
            "trained_all_steps": summary_stats(ep_tr_all_phi_act),
            "trained_cruise": summary_stats(ep_cruise_phi_act),
            "trained_crisis_veto": summary_stats(ep_crisis_phi_act),
            "paired_within_ep_crisis_vs_cruise": paired_comparison(paired_both_crisis_pa, paired_both_cruise_pa) if paired_both_crisis_pa else {},
            "test_crisis_vs_cruise": unpaired_comparison(ep_crisis_phi_act, ep_cruise_phi_act),
        },
        "gnwt_metastability": {
            "hab_on_ignition": summary_stats(ep_hab_on_ign),
            "hab_off_surgical_fatigue_only_ignition": summary_stats(ep_hab_off_surg_ign),
            "hab_off_ignition": summary_stats(ep_hab_off_legacy_ign),
            "hab_on_handoffs_per_sec": summary_stats(ep_hab_on_handoffs_per_s),
            "hab_off_surgical_fatigue_only_handoffs_per_sec": summary_stats(ep_hab_off_surg_handoffs_per_s),
            "hab_off_handoffs_per_sec": summary_stats(ep_hab_off_legacy_handoffs_per_s),
            "hab_on_specialist_entropy": summary_stats(ep_hab_on_ent),
            "hab_off_surgical_specialist_entropy": summary_stats(ep_hab_off_surg_ent),
            "hab_off_specialist_entropy": summary_stats(ep_hab_off_legacy_ent),
            "paired_surgical_handoff_test": paired_comparison(ep_hab_on_handoffs_per_s, ep_hab_off_surg_handoffs_per_s),
            "paired_surgical_entropy_test": paired_comparison(ep_hab_on_ent, ep_hab_off_surg_ent),
            "paired_ignition_test": paired_comparison(ep_hab_off_legacy_ign, ep_hab_on_ign),
            "paired_handoff_test": paired_comparison(ep_hab_on_handoffs_per_s, ep_hab_off_legacy_handoffs_per_s),
        },
        "world_model_ubwm": {
            "untrained_vis_mae": summary_stats(ep_un_vis_mae),
            "trained_vis_mae": summary_stats(ep_tr_vis_mae),
            "untrained_depth_mae_m": summary_stats(ep_un_dep_mae),
            "trained_depth_mae_m": summary_stats(ep_tr_dep_mae),
            "trained_cf_spread": summary_stats(ep_tr_cf_spread),
            "paired_vis_mae_test": paired_comparison(ep_un_vis_mae, ep_tr_vis_mae),
            "paired_depth_mae_test": paired_comparison(ep_un_dep_mae, ep_tr_dep_mae),
        },
        "self_surprise": {
            "trained_cruise": summary_stats(ep_cruise_ssurp),
            "trained_crisis": summary_stats(ep_crisis_ssurp),
            "paired_within_ep_crisis_vs_cruise": paired_comparison(paired_both_crisis_ss, paired_both_cruise_ss) if paired_both_crisis_ss else {},
            "test_crisis_vs_cruise": unpaired_comparison(ep_crisis_ssurp, ep_cruise_ssurp),
        },
        "plan_a_motor_isolation": {
            "states_checked": plan_a_states_checked,
            "max_abs_q_diff_broca_on_vs_off": float(max_plan_a_q_diff),
        },
        "representative_trace": best_trace,
    }


def main() -> None:
    t0 = time.perf_counter()
    ckpt = resolve_primary_ckpt()
    assert os.path.isfile(ckpt), f"Missing checkpoint: {ckpt}"

    print("[Exp4] 1/2 Running Extended Proposition 3 Monte Carlo benchmark (50 trials)...")
    prop3 = run_part_a_proposition3_benchmark(n_trials=50, t_len=48)
    print(
        f"[Exp4] Prop3 Null Flash -> OffDiag(deb/raw)=({prop3['common_input_null_offdiag_ridge_bits']['mean']:.4f}/"
        f"{prop3['common_input_null_offdiag_raw_undebiased_bits']['mean']:.4f}b) | "
        f"Iso014={prop3['common_input_null_isotropic_matched_lam014_bits']['mean']:.4f}b | "
        f"2+3 TrueCut={prop3['block_split_2v3_true_cut_debiased_bits']['mean']:.4f}b | "
        f"MIP Exact={prop3['block_split_exact_mip_recovery_rate']*100:.1f}%"
    )

    print(f"[Exp4] 2/2 Running N=24 independent-episode live cognitive & world-model sweep (ckpt={os.path.basename(ckpt)})...")
    arena_cog = _run_arena_cognitive_sweep(ckpt, n_episodes=24)

    out = {
        "checkpoint_used": os.path.relpath(ckpt, ROOT_DIR).replace("\\", "/"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "proposition3_monte_carlo": prop3,
        "live_arena_episode_metrics": arena_cog,
    }
    out_path = os.path.join(ROOT_DIR, "paper_arxiv", "experiments", "results", "exp4_consciousness_wm.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[Exp4] Saved -> {out_path}")


if __name__ == "__main__":
    main()
