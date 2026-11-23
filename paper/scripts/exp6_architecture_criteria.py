# -*- coding: utf-8 -*-
"""
exp6_architecture_criteria.py — Pillar D: Whole-Brain Architectural Integration Audit
=====================================================================================
Evaluates three system-level cognitive architecture criteria on the remediated v2.1 system:
  1. Replaceability (Plug-Compatibility): 5 cross-layer organ swaps (`S1..S5`) verifying zero contract
     errors (`NaN`, bounds, shape) and quantifying behavioral shifts.
  2. Integration vs. Interference:
     - (2A) Exact 9-Loss Multi-Objective Pairwise Gradient Cosine & Norm Matrix (verifying encoder
            gradient orthogonality after counterfactual & value trunk stop-gradient protections).
     - (2B) 10-Link Cross-Module Mean-Clamp Sensitivity Scan ("Silent Passenger" audit).
  3. Experimentability (Trade-Off Ladder): 6-Rung Cumulative Organ Ladder (`L0 -> L1 -> L2 -> L3 -> L4 -> L5`)
     against the fixed non-neural `L2_homeo_scaffold` yardstick, simultaneously measuring improvements
     AND degradations across Win%, Progress%, Kills, Deaths, Turf%, Shots/Game, and Stagnation.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from stats_utils import paired_comparison, summary_stats
from arena_pbt_trainer import HeadToHeadTactical3DArena, canonicalize_team_observation
from exp2_behavioral_ablations import make_base_cfg, resolve_primary_ckpt
from brain3d_neural_engine import _extract_candidate_compass_features_np, compose_vis8_patch
from universal_brain_3d_core import (
    ActionStagnationHomeostasis3D,
    BasalGanglia3DTimingEngineV2,
    Brain3DConfig,
    CAND_IDX_EVADE_REFUGE,
    CAND_IDX_FLANK_DETOUR,
    CAND_IDX_FRONTIER_TURF,
    CAND_IDX_IDLE_OBSERVE,
    CAND_IDX_OBJECTIVE_ADVANCE,
    CAND_IDX_RALLY_OR_PICKUP,
    CAND_IDX_SHIELD_OR_BURST,
    CAND_IDX_STRAFE_RIGHT_ENGAGE,
    CFEAT_DIM,
    CFEAT_DIR_CLEARANCE,
    CFEAT_DIR_FRIENDLY_SURF,
    CFEAT_HAZARD,
    CFEAT_IN_REFUGE,
    CFEAT_SPEED_SCALE,
    Cerebellar3DKinematicPredictorV2,
    EntorhinalHippocampal3DEngineV2,
    GOAL_MODE_BALANCED_EXPLORE,
    LATENT_Z_DIM,
    MC_OBJ_DX,
    MC_OBJ_DZ,
    MC_THR_DIST,
    MC_THR_VALID,
    NUM_OBJECT_SLOTS,
    PROPRIO_IDX_CAN_COMMIT,
    PROPRIO_IDX_OBJ_PROGRESS,
    PROPRIO_IDX_RESOURCE_CARRIER,
    PROPRIO_IDX_SURFACE_POSTURE,
    PROTO_EMBED_DIM,
    RSSM_STOCH_CATS,
    SHIELD_HAZARD_CF_MASK,
    SPATIAL_MODE_DRIFT_ONLY,
    SPATIAL_MODE_TUNNEL_RAW,
    TIMING_MODE_CAUTIOUS,
    TIMING_MODE_INSTANT,
    VERB_APPROACH_HOVER,
    WholeBrainNeuralEngineWrapper,
    compute_gene_modulated_mip_and_fr,
    compute_universal_memory_compass,
    derive_ventral_fallback_from_dorsal_16,
    get_resource_tank_level,
    get_task_profile,
    is_in_stealth_or_cover_refuge,
    project_pomdp_proprio,
    propose_universal_3d_candidates,
    select_policy_candidate,
)


def _onehot_st0(_: Any = None) -> np.ndarray:
    arr = np.zeros(64, dtype=np.float32)
    arr[0] = 1.0
    return arr


def run_gradient_conflict_audit(ckpt: str) -> Dict[str, Any]:
    cfg = make_base_cfg(ckpt, profile="turf_objective", seed=2026)
    cfg.enable_learning = True
    eng = WholeBrainNeuralEngineWrapper(cfg)
    opp_cfg = make_base_cfg(ckpt, profile="turf_objective", seed=2027)
    eng_opp = WholeBrainNeuralEngineWrapper(opp_cfg)

    for ep_i in range(2):
        aseed = 3100 + ep_i * 71
        arena = HeadToHeadTactical3DArena(Brain3DConfig(task_profile="turf_objective", seed=aseed), seed=aseed)
        obs_s = arena._build_agent_obs(+1, 0.0, False)
        obs_n = canonicalize_team_observation(arena._build_agent_obs(-1, 0.0, False), -1, arena.wmin, arena.wmax)
        hip_s, hip_n = EntorhinalHippocampal3DEngineV2(cfg), EntorhinalHippocampal3DEngineV2(opp_cfg)
        cer_s, cer_n = Cerebellar3DKinematicPredictorV2(cfg), Cerebellar3DKinematicPredictorV2(opp_cfg)
        hom_s = ActionStagnationHomeostasis3D(cfg.world_min, cfg.world_max)
        vp_s, vp_n = {}, {}
        rseq_s, rseq_n = 0, 0
        pose_s, verb_s, gz_s = np.zeros(6, dtype=np.float32), 1, np.array([0.0, -0.15, 6.0], dtype=np.float32)
        pose_n, verb_n, gz_n = np.zeros(6, dtype=np.float32), 1, np.array([0.0, -0.15, 6.0], dtype=np.float32)
        pending = None

        for tick in range(1, 380):
            if obs_s.get("respawn_flag"):
                rseq_s += 1
            if obs_n.get("respawn_flag"):
                rseq_n += 1
            vs = derive_ventral_fallback_from_dorsal_16(obs_s["fovea_patch"], yaw_pitch=obs_s["self_yaw_pitch"], prev_state=vp_s, episode_id=ep_i + 1)
            vn = derive_ventral_fallback_from_dorsal_16(obs_n["fovea_patch"], yaw_pitch=obs_n["self_yaw_pitch"], prev_state=vp_n, episode_id=ep_i + 1)
            sps = hip_s.step(obs_s, 1.0 / 60.0, ventral_override=vs, respawn_atom=rseq_s)
            spn = hip_n.step(obs_n, 1.0 / 60.0, ventral_override=vn, respawn_atom=rseq_n)
            las, _, _, _ = cer_s.predict_lead_and_crossfire(sps["believed_self_pos"], sps["believed_entities"], sps["believed_count"])
            lan, _, _, _ = cer_n.predict_lead_and_crossfire(spn["believed_self_pos"], spn["believed_entities"], spn["believed_count"])
            prs = project_pomdp_proprio(obs_s["proprio"], sps["believed_self_pos"], cfg.world_min, cfg.world_max, cfg)
            prn = project_pomdp_proprio(obs_n["proprio"], spn["believed_self_pos"], opp_cfg.world_min, opp_cfg.world_max, opp_cfg)
            mcs = compute_universal_memory_compass(sps["believed_self_pos"], sps["believed_entities"], sps["believed_count"], prs, hip_s.field_map, eng.profile)
            mcn = compute_universal_memory_compass(spn["believed_self_pos"], spn["believed_entities"], spn["believed_count"], prn, hip_n.field_map, eng_opp.profile)
            cps, cvs = propose_universal_3d_candidates(sps["believed_self_pos"], sps["believed_entities"], sps["believed_count"], prs, mcs, hip_s.field_map, las, sps["active_gaze_ypd"], cfg.world_min, cfg.world_max, eng.profile)
            cpn, cvn = propose_universal_3d_candidates(spn["believed_self_pos"], spn["believed_entities"], spn["believed_count"], prn, mcn, hip_n.field_map, lan, spn["active_gaze_ypd"], opp_cfg.world_min, opp_cfg.world_max, eng_opp.profile)

            if tick % 4 == 1:
                gs = eng.infer_conscious_gwt_batch([{
                    "slot_id": 0, "episode_id": ep_i + 1, "respawn_flag": bool(sps["respawn_event_flag"]),
                    "enable_learning": True, "fovea_patch": obs_s["fovea_patch"], "ventral_patch": vs,
                    "proprio": prs, "mem_compass": mcs, "mec_grid_code": sps["mec_grid_code"],
                    "hd_ring_code": sps["hd_ring_code"], "believed_self_pos": sps["believed_self_pos"],
                    "cand_poses": cps, "cand_verbs": cvs,
                }])[0]
                gn = eng_opp.infer_conscious_gwt_batch([{
                    "slot_id": 0, "episode_id": ep_i + 1, "respawn_flag": bool(spn["respawn_event_flag"]),
                    "enable_learning": False, "fovea_patch": obs_n["fovea_patch"], "ventral_patch": vn,
                    "proprio": prn, "mem_compass": mcn, "mec_grid_code": spn["mec_grid_code"],
                    "hd_ring_code": spn["hd_ring_code"], "believed_self_pos": spn["believed_self_pos"],
                    "cand_poses": cpn, "cand_verbs": cvn,
                }])[0]
                qs, pis, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                    gs["q_values"], gs["survival_probs"], gs["pred_adv_probs"], gs["gwt_workspace"],
                    cps, cvs, prs, mcs, None, cfg, habituation_trace=hom_s.habituation_trace,
                    stagnation_level=hom_s.stagnation_level, q_raw_values=gs["q_raw_values"],
                )
                cs_i = select_policy_candidate(qs, pis, trainable=True, rng_seed_val=tick + aseed)
                eng.notify_chosen_action(0, gs["cand_feats32"][cs_i].copy())
                _ = hom_s.update_and_compute_intrinsic(ep_i + 1, sps["believed_self_pos"][[0, 2]], prs, cs_i, float(obs_s["last_step_reward"]))
                pose_s = cps[cs_i].copy(); pose_s[5] = float(gs["speed_mod"][cs_i]); verb_s = int(cvs[cs_i]); gz_s = sps["active_gaze_ypd"].copy()

                cn_i = int(np.argmax(gn["q_values"]))
                pose_n = cpn[cn_i].copy(); pose_n[5] = float(gn["speed_mod"][cn_i]); verb_n = int(cvn[cn_i]); gz_n = spn["active_gaze_ypd"].copy()

                cur_haz = max(float(obs_s["hazard_level"]), float(sps["believed_hazard"]))
                if pending is not None:
                    pending["reward"] = float(obs_s["cumulative_score"]) - float(pending["cum_score"])
                    pending["next_hazard"] = cur_haz
                    pending["obj_adv_delta"] = float(prs[PROPRIO_IDX_OBJ_PROGRESS]) - float(pending["obj_prog"])
                    pending["done"] = False
                    pending["respawn"] = bool(sps["respawn_event_flag"])
                    pending["enable_learning"] = False
                    pending["next_state"] = {
                        "fovea_patch": obs_s["fovea_patch"].copy(), "ventral_patch": vs.copy(),
                        "proprio": prs.copy(), "mem_compass": mcs.copy(),
                        "mec_grid_code": sps["mec_grid_code"].copy(), "hd_ring_code": sps["hd_ring_code"].copy(),
                        "cand_feats32": gs["cand_feats32"].copy(),
                    }
                    eng.record_transition_and_maybe_train(pending)
                pending = {
                    "slot_id": 0, "episode_id": ep_i + 1, "action_epoch": int(obs_s["action_epoch"]),
                    "tick": tick, "cum_score": float(obs_s["cumulative_score"]),
                    "hazard": cur_haz if tick % 12 != 1 else 0.55,
                    "obj_prog": float(prs[PROPRIO_IDX_OBJ_PROGRESS]),
                    "fovea_patch": obs_s["fovea_patch"].copy(), "ventral_patch": vs.copy(),
                    "proprio": prs.copy(), "mem_compass": mcs.copy(),
                    "mec_grid_code": sps["mec_grid_code"].copy(), "hd_ring_code": sps["hd_ring_code"].copy(),
                    "gwt_workspace": gs["gwt_workspace"].copy(), "rssm_h": gs["rssm_h"].copy(),
                    "cand_feats32": gs["cand_feats32"].copy(), "chosen_cand_feat32": gs["cand_feats32"][cs_i].copy(),
                    "chosen_idx": cs_i, "chosen_verb": verb_s,
                    "proto_sym_id": int(gs["proto_sym_id"]), "proto_embed": gs["proto_embed"].copy(),
                    "world_surprise": float(gs["world_surprise"]), "self_surprise": float(gs["self_surprise"]),
                    "beta_teacher": 0.0,
                }
                if tick % 12 == 1:
                    pending["cand_feats32"][CAND_IDX_IDLE_OBSERVE, CFEAT_IN_REFUGE] = 0.0
                    pending["cand_feats32"][CAND_IDX_IDLE_OBSERVE, CFEAT_HAZARD] = 0.55

            obs_s, obs_n, done, _ = arena.step_both((pose_s, verb_s, True, gz_s), (pose_n, verb_n, True, gz_n), 1.0 / 60.0)
            if done:
                break

    loss_names = ["L_td", "L_speed", "L_rssm", "L_cf", "L_vis", "L_slot", "L_strat", "L_self", "L_lang"]
    shared_mods = ["foveal_enc", "slot_cortex", "topo_adapter", "hazard_net", "rssm", "gwt", "strat_tf", "self_model", "d3qn", "broca", "concept_grounder"]

    cos_accum = np.zeros((len(loss_names), len(loss_names)), dtype=np.float64)
    cos_counts = np.zeros((len(loss_names), len(loss_names)), dtype=np.int64)
    ov_cos_accum = np.zeros((len(loss_names), len(loss_names)), dtype=np.float64)
    ov_cos_counts = np.zeros((len(loss_names), len(loss_names)), dtype=np.int64)
    norm_accum = {ln: [] for ln in loss_names}
    mod_touch = {ln: set() for ln in loss_names}
    mod_norm_accum = {ln: {mn: [] for mn in shared_mods} for ln in loss_names}

    all_params, param_mod_names = [], []
    for mn in shared_mods:
        for _, p in eng.models[mn].named_parameters():
            if p.requires_grad:
                all_params.append(p)
                param_mod_names.append(mn)

    for _ in range(8):
        _, batch, is_w = eng.replay.sample_prioritized(16)
        bsz = len(batch)
        fov = eng._t(np.stack([b["fovea_patch"] for b in batch]), torch.float32)
        vis8 = eng._t(np.stack([compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch")) for b in batch]), torch.float32)
        prop = eng._t(np.stack([b["proprio"] for b in batch]), torch.float32)
        comp = eng._t(np.stack([b["mem_compass"] for b in batch]), torch.float32)
        mec = eng._t(np.stack([b["mec_grid_code"] for b in batch]), torch.float32)
        hd = eng._t(np.stack([b["hd_ring_code"] for b in batch]), torch.float32)
        c_feats = eng._t(np.stack([b["cand_feats32"] for b in batch]), torch.float32)
        chosen_idx = eng._t([b["chosen_idx"] for b in batch], torch.long)
        is_w_t = eng._t(is_w, torch.float32)

        rewards_nstep = eng._t([b.get("nstep_return", b["reward"]) for b in batch], torch.float32)
        gammas_nstep = eng._t([b.get("nstep_gamma", 0.96) for b in batch], torch.float32)
        dones_nstep = eng._t([1.0 if b.get("nstep_done", b["done"]) else 0.0 for b in batch], torch.float32)
        rewards_1step = eng._t([b.get("reward_1step", b["reward"]) for b in batch], torch.float32)
        dones_1step = eng._t([1.0 if (b["done"] or b.get("respawn", False)) else 0.0 for b in batch], torch.float32)
        haz_tgt = eng._t([float(np.clip(b.get("next_hazard", b["hazard"]), 0.0, 1.0)) for b in batch], torch.float32)
        surv_tgt = eng._t([0.05 if b.get("respawn", False) else float(np.clip(1.0 - 0.85 * b.get("next_hazard", 0.0), 0.05, 0.99)) for b in batch], torch.float32)
        adv_tgt = eng._t([1.0 if b.get("obj_adv_delta", 0.0) > 0.002 or b.get("reward", 0.0) > 0.05 else 0.35 for b in batch], torch.float32)

        ns1_list = [b.get("next_state_1step", b.get("next_state")) for b in batch]
        nsn_list = [b.get("next_state_nstep", b.get("next_state")) for b in batch]
        nvis8_1 = eng._t(np.stack([compose_vis8_patch(ns["fovea_patch"], ns.get("ventral_patch")) for ns in ns1_list]), torch.float32)
        nfov_n = eng._t(np.stack([ns["fovea_patch"] for ns in nsn_list]), torch.float32)
        nvis8_n = eng._t(np.stack([compose_vis8_patch(ns["fovea_patch"], ns.get("ventral_patch")) for ns in nsn_list]), torch.float32)
        nprop_n = eng._t(np.stack([ns["proprio"] for ns in nsn_list]), torch.float32)
        ncomp_n = eng._t(np.stack([ns["mem_compass"] for ns in nsn_list]), torch.float32)
        nmec_n = eng._t(np.stack([ns["mec_grid_code"] for ns in nsn_list]), torch.float32)
        nhd_n = eng._t(np.stack([ns["hd_ring_code"] for ns in nsn_list]), torch.float32)
        ncf_n = eng._t(np.stack([ns["cand_feats32"] for ns in nsn_list]), torch.float32)

        prev_h_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_h", b["rssm_h"]) for b in batch]), torch.float32)
        prev_st_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_st", _onehot_st0(b)) for b in batch]), torch.float32)
        prev_act_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_act", np.zeros(CFEAT_DIM, dtype=np.float32)) for b in batch]), torch.float32)
        prev_gwt_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_gwt", b["gwt_workspace"]) for b in batch]), torch.float32)
        prev_alpha_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_alpha", np.full(5, 0.2, dtype=np.float32)) for b in batch]), torch.float32)
        prev_hab_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_hab", np.full(5, 0.2, dtype=np.float32)) for b in batch]), torch.float32)
        prev_z_td_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_z_td", np.zeros(LATENT_Z_DIM, dtype=np.float32)) for b in batch]), torch.float32)
        prev_comp_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_comp", b["mem_compass"]) for b in batch]), torch.float32)
        prev_self_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_self", np.zeros(8, dtype=np.float32)) for b in batch]), torch.float32)
        prev_self_err_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_self_err", np.zeros(8, dtype=np.float32)) for b in batch]), torch.float32)
        prev_ssurp_t = eng._t([float(b.get("_gpu_ctx", {}).get("prev_self_surp", 0.0)) for b in batch], torch.float32)
        meta_err9_t = torch.cat([prev_self_err_t, prev_ssurp_t.unsqueeze(-1)], dim=-1)
        prev_perr_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_perr_16", np.zeros((1, 16, 16), dtype=np.float32)) for b in batch]), torch.float32)
        prev_spot_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_spot_16", np.ones((1, 16, 16), dtype=np.float32)) for b in batch]), torch.float32)
        strat_seq_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("strat_seq", np.zeros((8, 160), dtype=np.float32)) for b in batch]), torch.float32)

        # Exact remediated training graph matching brain3d_neural_engine.py
        z = eng.models["foveal_enc"](fov, prop, comp)
        slots, s_coords, s_val, slot_aux = eng.models["slot_cortex"](
            vis8, pred_err_16=prev_perr_t, topdown_gain_16=prev_spot_t,
            td_context=torch.cat([prev_gwt_t, prev_h_t], dim=-1), return_aux=True,
        )
        z_perc = torch.tanh(z + slot_aux["delta_z"] + 0.30 * prev_z_td_t)
        haz_pred = eng.models["hazard_net"](z_perc, prop, comp)
        slot_ctx64 = eng.models["topo_adapter"](slots, mec, hd, comp)
        chosen_cf = c_feats.gather(1, chosen_idx.view(bsz, 1, 1).expand(-1, 1, CFEAT_DIM)).squeeze(1)
        h_post, post_st, p_post, p_prior = eng.models["rssm"].observe_step(prev_h_t, prev_st_t, prev_act_t, z_perc, slot_ctx64, prop, comp)
        p_rew, p_surv, p_adv = eng.models["rssm"].imagine_candidates(h_post, post_st, c_feats, horizon=3)

        kl_surp_tr = (p_post.detach() * (torch.log(p_post.detach() + 1e-7) - torch.log(p_prior.detach() + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
        cf_sp_tr = 0.35 * p_rew.detach() + 0.40 * p_surv.detach() + 0.25 * p_adv.detach()
        nov5_tr = torch.stack([
            prev_perr_t.mean(dim=(1, 2, 3)) * 2.4,
            torch.clamp((comp - prev_comp_t).norm(dim=-1) * 0.95, 0.0, 2.0),
            kl_surp_tr * 1.35 + 0.65 * (cf_sp_tr.max(dim=-1).values - cf_sp_tr.min(dim=-1).values),
            haz_pred.detach() * 1.15 + 0.45 * torch.clamp(0.35 - prop[:, PROPRIO_IDX_RESOURCE_CARRIER], min=0.0),
            prev_ssurp_t * 1.80,
        ], dim=-1)
        strat_emb, mode_logits, focus_off_xz = eng.models["strat_tf"](strat_seq_t, meta_err9=meta_err9_t)
        gwt_base, alp_live, ign_live, _ = eng.models["gwt"](
            z_perc, slot_ctx64, mec, hd, comp, h_post.detach(), post_st.detach(), prop, haz_pred.detach(), strat_emb,
            prev_gwt_t, meta_salience=prev_ssurp_t, hab_trace=prev_hab_t, novelty_5=nov5_tr,
        )
        delta_gwt, topdown_gain_16 = eng.models["topo_adapter"].compute_topdown_spotlight(gwt_base, slot_aux["feat16"], slots, s_coords)
        gwt_live = gwt_base + delta_gwt
        z_td = (0.08 + 0.24 * ign_live).unsqueeze(-1) * eng.models["gwt"].topdown_z(gwt_live)
        self_cur, self_surp_live, self_err_live, self_pred = eng.models["self_model"](
            gwt_live.detach(), prop, prev_self_t, gwt_alpha=alp_live.detach(), prev_act=prev_act_t, prev_alpha=prev_alpha_t,
        )
        z_perc_for_q = 0.25 * z_perc + 0.75 * z_perc.detach()
        z_eff = torch.tanh(z_perc_for_q + 0.70 * z_td)
        q_all, spd_all = eng.models["d3qn"](z_eff, gwt_live, h_post.detach(), mec, hd, prop, comp, c_feats)
        q_chosen = q_all.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            n_pred_v8_1, _ = eng.models["slot_cortex"].predict_next_visual_field(slots, slot_aux["pixel_stats"], s_coords, chosen_cf, rssm_h=h_post)
            n_perr_16_1 = (nvis8_1 - n_pred_v8_1).abs().mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
            nslots_1, _, _, _ = eng.models["slot_cortex"](nvis8_1, pred_err_16=n_perr_16_1, topdown_gain_16=topdown_gain_16, return_aux=True)
            nz_n = eng.models["foveal_enc"](nfov_n, nprop_n, ncomp_n)
            nslots_n, ns_coords_n, _, nslot_aux_n = eng.models["slot_cortex"](nvis8_n, topdown_gain_16=topdown_gain_16, return_aux=True)
            nz_perc_n = torch.tanh(nz_n + nslot_aux_n["delta_z"])
            nhaz_n = eng.models["hazard_net"](nz_perc_n, nprop_n, ncomp_n)
            nslot_ctx64_n = eng.models["topo_adapter"](nslots_n, nmec_n, nhd_n, ncomp_n)
            nh_next_n, npost_st_n, _, _ = eng.models["rssm"].observe_step(h_post, post_st, chosen_cf, nz_perc_n, nslot_ctx64_n, nprop_n, ncomp_n)
            n_snap = torch.cat([gwt_live, self_cur, comp], dim=-1).unsqueeze(1)
            n_strat_seq = torch.cat([strat_seq_t[:, 1:, :], n_snap], dim=1)
            n_meta_err9 = torch.cat([self_err_live, self_surp_live.unsqueeze(-1)], dim=-1)
            n_strat_emb, _, _ = eng.models["strat_tf"](n_strat_seq, meta_err9=n_meta_err9)
            ngwt_base, _, nign, _ = eng.models["gwt"](nz_perc_n, nslot_ctx64_n, nmec_n, nhd_n, ncomp_n, nh_next_n, npost_st_n, nprop_n, nhaz_n, n_strat_emb, gwt_live, meta_salience=self_surp_live)
            ndelta_gwt, _ = eng.models["topo_adapter"].compute_topdown_spotlight(ngwt_base, nslot_aux_n["feat16"], nslots_n, ns_coords_n)
            ngwt = ngwt_base + ndelta_gwt
            nz_td = (0.08 + 0.24 * nign).unsqueeze(-1) * eng.models["gwt"].topdown_z(ngwt)
            nz_eff = torch.tanh(nz_perc_n + nz_td)
            nq_online, _ = eng.models["d3qn"](nz_eff, ngwt, nh_next_n, nmec_n, nhd_n, nprop_n, ncomp_n, ncf_n)
            best_next = torch.argmax(nq_online, dim=-1, keepdim=True)
            nq_targ, _ = eng.models["d3qn_target"](nz_eff, ngwt, nh_next_n, nmec_n, nhd_n, nprop_n, ncomp_n, ncf_n)
            v_next = nq_targ.gather(1, best_next).squeeze(1)
            td_target = torch.clamp(rewards_nstep + gammas_nstep * (1.0 - dones_nstep) * v_next, -32.0, 38.0)

        loss_td = (F.smooth_l1_loss(q_chosen, td_target, reduction="none") * is_w_t).mean()
        spd_base = c_feats[:, :, CFEAT_SPEED_SCALE] * (0.85 + 0.15 * c_feats[:, :, CFEAT_DIR_CLEARANCE] + 0.10 * (c_feats[:, :, CFEAT_DIR_FRIENDLY_SURF] - 0.5))
        spd_bonus = (0.10 * (adv_tgt - 0.5) - 0.12 * dones_1step * (1.0 - surv_tgt)).unsqueeze(1)
        spd_tgt = torch.clamp(spd_base.scatter_add(1, chosen_idx.unsqueeze(1), spd_bonus), 0.25, 1.0)
        loss_speed = 0.20 * F.smooth_l1_loss(spd_all, spd_tgt)

        rew_ch = p_rew.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
        surv_ch = p_surv.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
        adv_ch = p_adv.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
        kl_dyn = (p_post.detach() * (torch.log(p_post.detach() + 1e-7) - torch.log(p_prior + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
        kl_rep = (p_post * (torch.log(p_post + 1e-7) - torch.log(p_prior.detach() + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
        kl_loss = 0.80 * torch.clamp(kl_dyn - 0.08, min=0.0).mean() + 0.20 * torch.clamp(kl_rep - 0.08, min=0.0).mean()
        loss_rssm = 0.5 * F.smooth_l1_loss(rew_ch, rewards_1step) + F.binary_cross_entropy(surv_ch.clamp(1e-4, 1.0 - 1e-4), surv_tgt) + 0.4 * F.binary_cross_entropy(adv_ch.clamp(1e-4, 1.0 - 1e-4), adv_tgt) + 0.15 * kl_loss

        exposed_mask = (c_feats[:, CAND_IDX_IDLE_OBSERVE, CFEAT_IN_REFUGE] < 0.5) & (c_feats[:, CAND_IDX_IDLE_OBSERVE, CFEAT_HAZARD] >= SHIELD_HAZARD_CF_MASK)
        if exposed_mask.any():
            _, p_surv_cf, _ = eng.models["rssm"].imagine_candidates(h_post[exposed_mask].detach(), post_st[exposed_mask].detach(), c_feats[exposed_mask], horizon=3)
            cf_surv_m7 = p_surv_cf[:, CAND_IDX_IDLE_OBSERVE].clamp(1e-4, 1.0 - 1e-4)
            q_cf_exp, _ = eng.models["d3qn"](
                z_eff[exposed_mask].detach(), gwt_live[exposed_mask].detach(), h_post[exposed_mask].detach(),
                mec[exposed_mask], hd[exposed_mask], prop[exposed_mask], comp[exposed_mask], c_feats[exposed_mask],
            )
            loss_cf = 0.45 * F.binary_cross_entropy(cf_surv_m7, torch.full_like(cf_surv_m7, 0.12)) + 0.25 * F.relu(q_cf_exp[:, CAND_IDX_IDLE_OBSERVE] - q_chosen[exposed_mask].detach() + 0.6).mean()
        else:
            loss_cf = z.sum() * 0.0

        recon_d = eng.models["foveal_enc"].decode_depth(z_perc)
        loss_vis = 0.25 * F.mse_loss(recon_d, fov[:, :, :, 0].clamp(0.0, 1.0)) + 0.40 * F.mse_loss(haz_pred, haz_tgt)

        slot_attn = slot_aux["attn"]
        vis8_8x8 = F.avg_pool2d(vis8.permute(0, 3, 1, 2), kernel_size=2, stride=2).flatten(2).permute(0, 2, 1)
        attn_d = slot_attn.detach()
        slot_uv = torch.einsum("bsn,bnd->bsd", attn_d, eng.models["slot_cortex"].grid_xy.expand(bsz, -1, -1))
        slot_vis = torch.einsum("bsn,bnd->bsd", attn_d, vis8_8x8)
        depth_s = slot_vis[:, :, 0].clamp(0.0, 1.0)
        ny_s = slot_vis[:, :, 2]
        mot_s = slot_vis[:, :, 7].clamp(0.0, 1.0)
        sharpness = (attn_d * 64.0).max(dim=-1).values
        tgt_coords = torch.cat([slot_uv, 2.0 * depth_s.unsqueeze(-1) - 1.0], dim=-1)
        threat_prox = comp[:, MC_THR_VALID].clamp(0.0, 1.0) * (1.0 - comp[:, MC_THR_DIST]).clamp(0.0, 1.0)
        emp_hostile = torch.clamp(torch.maximum(torch.maximum(haz_tgt, dones_1step * (1.0 - surv_tgt)), 0.75 * threat_prox), 0.0, 1.0)
        emp_objective = torch.clamp(0.55 * adv_tgt + 0.45 * torch.sigmoid(rewards_1step * 0.75), 0.0, 1.0)
        emp_cover = torch.clamp(0.50 * surv_tgt + 0.50 * prop[:, PROPRIO_IDX_SURFACE_POSTURE].clamp(0.0, 1.0), 0.0, 1.0)
        w_hostile = F.softmax(1.20 * mot_s + 0.80 * depth_s + 0.20 * sharpness - 0.80 * F.relu(ny_s - 0.60), dim=-1)
        w_obj = F.softmax(0.90 * depth_s + 0.20 * sharpness - 0.60 * F.relu(ny_s - 0.60) + 0.40 * (1.0 - slot_uv[:, :, 1].abs()), dim=-1)
        w_cover = F.softmax(1.20 * F.relu(ny_s - 0.35) + 0.50 * (1.0 - mot_s) - 0.10 * sharpness, dim=-1)
        pred_hostile = (w_hostile * s_val[:, :, 0]).sum(dim=-1).clamp(1e-4, 1.0 - 1e-4)
        pred_obj = (w_obj * s_val[:, :, 1]).sum(dim=-1).clamp(1e-4, 1.0 - 1e-4)
        pred_cover = (w_cover * s_val[:, :, 2]).sum(dim=-1).clamp(1e-4, 1.0 - 1e-4)
        loss_val_pool = (F.binary_cross_entropy(pred_hostile, emp_hostile) + F.binary_cross_entropy(pred_obj, emp_objective) + F.binary_cross_entropy(pred_cover, emp_cover)) / 3.0
        soft_slot_tgt = torch.stack([
            (emp_hostile.unsqueeze(1) * (float(NUM_OBJECT_SLOTS) * w_hostile).clamp(0.0, 1.25)).clamp(0.0, 1.0),
            (emp_objective.unsqueeze(1) * (float(NUM_OBJECT_SLOTS) * w_obj).clamp(0.0, 1.25)).clamp(0.0, 1.0),
            (emp_cover.unsqueeze(1) * (float(NUM_OBJECT_SLOTS) * w_cover).clamp(0.0, 1.25)).clamp(0.0, 1.0),
        ], dim=-1)
        pred_nvis8, pred_nslots = eng.models["slot_cortex"].predict_next_visual_field(slots.detach(), slot_aux["pixel_stats"].detach(), s_coords.detach(), chosen_cf, rssm_h=h_post.detach())
        fg_focus_16 = torch.clamp(0.65 * slot_aux["sal_16"].detach().clamp(0.0, 1.0) + 0.35 * ((nvis8_1 - vis8).permute(0, 3, 1, 2).abs().mean(dim=1, keepdim=True) * 2.5).clamp(0.0, 1.0), 0.0, 1.0)
        loss_td_prior = 0.12 * F.mse_loss(torch.tanh(slot_aux["td_prior_raw"]), torch.tanh(slots.detach())) if slot_aux.get("td_prior_raw") is not None else z.new_tensor(0.0)
        loss_slot = (
            0.20 * F.mse_loss(s_coords, tgt_coords) + 0.14 * loss_val_pool + 0.08 * F.mse_loss(s_val, soft_slot_tgt)
            + 0.20 * F.mse_loss(pred_nslots, nslots_1.detach()) + 0.35 * F.smooth_l1_loss(pred_nvis8, nvis8_1, beta=0.05)
            + 0.20 * F.mse_loss(topdown_gain_16, 0.85 + 0.55 * fg_focus_16) + loss_td_prior
        )

        mode_q = torch.stack([
            q_all[:, CAND_IDX_FRONTIER_TURF], q_all[:, CAND_IDX_OBJECTIVE_ADVANCE],
            torch.maximum(q_all[:, CAND_IDX_STRAFE_RIGHT_ENGAGE], q_all[:, CAND_IDX_SHIELD_OR_BURST]),
            torch.maximum(q_all[:, CAND_IDX_EVADE_REFUGE], torch.maximum(q_all[:, CAND_IDX_RALLY_OR_PICKUP], q_all[:, CAND_IDX_IDLE_OBSERVE])),
            q_all[:, CAND_IDX_FLANK_DETOUR],
        ], dim=-1).detach()
        mode_tgt_probs = F.softmax(mode_q / 0.45, dim=-1)
        mode_log_probs = F.log_softmax(mode_logits, dim=-1)
        focus_tgt = 4.0 * comp[:, MC_OBJ_DX : MC_OBJ_DZ + 1]
        loss_strat = -(mode_tgt_probs * mode_log_probs).sum(dim=-1).mean() + 0.15 * F.smooth_l1_loss(focus_off_xz, focus_tgt)

        self_tgt8 = torch.clamp(torch.stack([
            2.0 * surv_tgt - 1.0, 1.0 - 2.0 * haz_tgt, 2.0 * prop[:, PROPRIO_IDX_RESOURCE_CARRIER] - 1.0,
            2.0 * prop[:, PROPRIO_IDX_SURFACE_POSTURE] - 1.0, 2.0 * prop[:, PROPRIO_IDX_CAN_COMMIT] - 1.0,
            2.0 * prop[:, PROPRIO_IDX_OBJ_PROGRESS] - 1.0, torch.tanh(q_chosen.detach() / 8.0), torch.tanh(kl_rep.detach() - 0.5),
        ], dim=-1), -1.0, 1.0)
        loss_self = 0.25 * F.smooth_l1_loss(self_cur, self_tgt8) + 0.22 * F.mse_loss(self_pred, self_cur.detach())

        tok_np = np.stack([eng._synthesize_endogenous_tokens_np(b) for b in batch])
        tok_t = eng._t(tok_np, torch.long)
        proto_ids_t = eng._t([int(b.get("proto_sym_id", 0)) for b in batch], torch.long)
        p_emb_t = eng._t(np.stack([b.get("proto_embed", np.zeros(PROTO_EMBED_DIM, dtype=np.float32)) for b in batch]), torch.float32)
        tf_l, broca_h = eng.models["broca"].teacher_forcing_loss(gwt_live.detach(), p_emb_t, slot_ctx64.detach(), tok_t)
        wern_h = eng.models["broca"].encode_tokens(tok_t)
        align_l = eng.models["concept_grounder"].alignment_loss(slot_ctx64.detach(), wern_h, token_ids=tok_t, slot_pstats=slot_aux["pixel_stats"].detach(), slot_coords=s_coords.detach(), proto_ids=proto_ids_t)
        q_center = q_all.detach() - q_all.detach().mean(dim=-1, keepdim=True)
        bias_pred = 0.25 * eng.models["broca"].cand_bias_head(broca_h.detach())
        loss_lang = 0.20 * tf_l + 0.15 * align_l + 0.05 * F.mse_loss(bias_pred, 0.25 * torch.tanh(q_center / 2.0))

        losses = [loss_td, loss_speed, loss_rssm, loss_cf, loss_vis, loss_slot, loss_strat, loss_self, loss_lang]
        grad_vecs, per_param_grads = [], []
        for li, lv in enumerate(losses):
            grads = torch.autograd.grad(lv, all_params, retain_graph=True, allow_unused=True)
            p_list = []
            mod_sq = {mn: 0.0 for mn in shared_mods}
            for g, p, mn in zip(grads, all_params, param_mod_names):
                if g is not None and float(g.abs().max().item()) > 1e-12:
                    mod_touch[loss_names[li]].add(mn)
                    gf = g.detach().reshape(-1)
                    p_list.append(gf)
                    mod_sq[mn] += float(gf.pow(2).sum().item())
                else:
                    p_list.append(None)
            for mn in shared_mods:
                mod_norm_accum[loss_names[li]][mn].append(math.sqrt(mod_sq[mn]))
            full_g = torch.cat([gf if gf is not None else torch.zeros(p.numel(), device=z.device, dtype=z.dtype) for gf, p in zip(p_list, all_params)])
            norm_accum[loss_names[li]].append(float(full_g.norm().item()))
            grad_vecs.append(full_g)
            per_param_grads.append(p_list)

        for i in range(len(loss_names)):
            for j in range(len(loss_names)):
                ni, nj = float(grad_vecs[i].norm().item()), float(grad_vecs[j].norm().item())
                if ni > 1e-9 and nj > 1e-9:
                    cos_accum[i, j] += float(torch.dot(grad_vecs[i], grad_vecs[j]).item() / (ni * nj))
                    cos_counts[i, j] += 1
                ov_i = [gi for gi, gj in zip(per_param_grads[i], per_param_grads[j]) if gi is not None and gj is not None]
                ov_j = [gj for gi, gj in zip(per_param_grads[i], per_param_grads[j]) if gi is not None and gj is not None]
                if ov_i:
                    vi, vj = torch.cat(ov_i), torch.cat(ov_j)
                    oni, onj = float(vi.norm().item()), float(vj.norm().item())
                    if oni > 1e-9 and onj > 1e-9:
                        ov_cos_accum[i, j] += float(torch.dot(vi, vj).item() / (oni * onj))
                        ov_cos_counts[i, j] += 1

    cos_mat = np.divide(cos_accum, np.maximum(1, cos_counts))
    ov_cos_mat = np.divide(ov_cos_accum, np.maximum(1, ov_cos_counts))
    saved_snapshots = [dict(eng.replay[k]) for k in range(len(eng.replay))]
    eng.release_memory_and_vram()
    eng_opp.release_memory_and_vram()

    return {
        "loss_names": loss_names,
        "mean_grad_l2_norm": {k: round(float(np.mean(v)), 5) for k, v in norm_accum.items()},
        "per_module_grad_norm": {
            ln: {mn: round(float(np.mean(vals)), 5) for mn, vals in md.items() if float(np.mean(vals)) > 1e-9}
            for ln, md in mod_norm_accum.items()
        },
        "modules_touched_by_loss": {k: sorted(list(v)) for k, v in mod_touch.items()},
        "pairwise_grad_cosine_full": [[round(float(cos_mat[i, j]), 4) for j in range(len(loss_names))] for i in range(len(loss_names))],
        "pairwise_grad_cosine_shared_overlap": [
            [round(float(ov_cos_mat[i, j]), 4) if ov_cos_counts[i, j] > 0 else None for j in range(len(loss_names))]
            for i in range(len(loss_names))
        ],
        "_snapshots": saved_snapshots,
    }


def run_silent_passenger_scan(ckpt: str, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = make_base_cfg(ckpt, profile="turf_objective", seed=2026)
    eng = WholeBrainNeuralEngineWrapper(cfg)
    torch_m = eng.torch

    bsz = len(snapshots)
    fov = eng._t(np.stack([b["fovea_patch"] for b in snapshots]), torch_m.float32)
    vis8 = eng._t(np.stack([compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch")) for b in snapshots]), torch_m.float32)
    prop = eng._t(np.stack([b["proprio"] for b in snapshots]), torch_m.float32)
    comp = eng._t(np.stack([b["mem_compass"] for b in snapshots]), torch_m.float32)
    mec = eng._t(np.stack([b["mec_grid_code"] for b in snapshots]), torch_m.float32)
    hd = eng._t(np.stack([b["hd_ring_code"] for b in snapshots]), torch_m.float32)
    c_feats = eng._t(np.stack([b["cand_feats32"] for b in snapshots]), torch_m.float32)

    prev_h = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_h", b["rssm_h"]) for b in snapshots]), torch_m.float32)
    prev_st = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_st", _onehot_st0(b)) for b in snapshots]), torch_m.float32)
    prev_act = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_act", np.zeros(CFEAT_DIM, dtype=np.float32)) for b in snapshots]), torch_m.float32)
    prev_gwt = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_gwt", b["gwt_workspace"]) for b in snapshots]), torch_m.float32)
    prev_hab = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_hab", np.full(5, 0.2, dtype=np.float32)) for b in snapshots]), torch_m.float32)
    prev_z_td = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_z_td", np.zeros(LATENT_Z_DIM, dtype=np.float32)) for b in snapshots]), torch_m.float32)
    prev_comp = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_comp", b["mem_compass"]) for b in snapshots]), torch_m.float32)
    prev_self_err = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_self_err", np.zeros(8, dtype=np.float32)) for b in snapshots]), torch_m.float32)
    prev_ssurp = eng._t([float(b.get("_gpu_ctx", {}).get("prev_self_surp", 0.0)) for b in snapshots], torch_m.float32)
    meta_err9 = torch_m.cat([prev_self_err, prev_ssurp.unsqueeze(-1)], dim=-1)
    prev_perr = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_perr_16", np.zeros((1, 16, 16), dtype=np.float32)) for b in snapshots]), torch_m.float32)
    prev_spot = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_spot_16", np.ones((1, 16, 16), dtype=np.float32)) for b in snapshots]), torch_m.float32)
    strat_seq = eng._t(np.stack([b.get("_gpu_ctx", {}).get("strat_seq", np.zeros((8, 160), dtype=np.float32)) for b in snapshots]), torch_m.float32)

    def _mean_clamp(x):
        return x.mean(dim=0, keepdim=True).expand_as(x)

    def _eval_with_clamp(clamp_key: str):
        with torch_m.no_grad():
            z = eng.models["foveal_enc"](fov, prop, comp)
            eff_spot = _mean_clamp(prev_spot) if clamp_key == "L3_topdown_t1_to_t" else prev_spot
            eff_ztd_prev = _mean_clamp(prev_z_td) if clamp_key == "L3_topdown_t1_to_t" else prev_z_td
            slots, s_coords, _, slot_aux = eng.models["slot_cortex"](
                vis8, pred_err_16=prev_perr, topdown_gain_16=eff_spot,
                td_context=torch_m.cat([prev_gwt, prev_h], dim=-1), return_aux=True,
            )
            dz = _mean_clamp(slot_aux["delta_z"]) if clamp_key == "L2_slot_cortex" else slot_aux["delta_z"]
            z_perc = torch_m.tanh((_mean_clamp(z) if clamp_key == "L1_foveal_z" else z) + dz + 0.30 * eff_ztd_prev)

            eff_mec = _mean_clamp(mec) if clamp_key == "L4_mec_hd_3d" else mec
            eff_hd = _mean_clamp(hd) if clamp_key == "L4_mec_hd_3d" else hd
            eff_comp = _mean_clamp(comp) if clamp_key == "L5_mem_compass_24d" else comp
            eff_cf = c_feats.clone()
            if clamp_key == "L5_mem_compass_24d":
                eff_cf[:, :, 14:] = _mean_clamp(eff_cf[:, :, 14:])

            haz_p = eng.models["hazard_net"](z_perc, prop, eff_comp)
            sctx = eng.models["topo_adapter"](slots, eff_mec, eff_hd, eff_comp)
            if clamp_key == "L2_slot_cortex":
                sctx = _mean_clamp(sctx)

            h_next, post_st, p_post, p_prior = eng.models["rssm"].observe_step(prev_h, prev_st, prev_act, z_perc, sctx, prop, eff_comp)
            p_rew, p_surv, p_adv = eng.models["rssm"].imagine_candidates(h_next, post_st, eff_cf, horizon=3)

            eff_h = _mean_clamp(h_next) if clamp_key == "L6_rssm_recurrent_state" else h_next
            eff_pst = _mean_clamp(post_st) if clamp_key == "L6_rssm_recurrent_state" else post_st

            kl_surp = (p_post * (torch_m.log(p_post + 1e-7) - torch_m.log(p_prior + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
            cf_sp = 0.35 * p_rew + 0.40 * p_surv + 0.25 * p_adv
            nov5 = torch_m.stack([
                prev_perr.mean(dim=(1, 2, 3)) * 2.4,
                torch_m.clamp((eff_comp - prev_comp).norm(dim=-1) * 0.95, 0.0, 2.0),
                kl_surp * 1.35 + 0.65 * (cf_sp.max(dim=-1).values - cf_sp.min(dim=-1).values),
                haz_p * 1.15 + 0.45 * torch_m.clamp(0.35 - prop[:, PROPRIO_IDX_RESOURCE_CARRIER], min=0.0),
                prev_ssurp * 1.80,
            ], dim=-1)

            eff_meta = _mean_clamp(meta_err9) if clamp_key == "L10_self_and_rvq" else meta_err9
            eff_ssurp = _mean_clamp(prev_ssurp) if clamp_key == "L10_self_and_rvq" else prev_ssurp
            strat_emb, _, _ = eng.models["strat_tf"](strat_seq, meta_err9=eff_meta)
            if clamp_key == "L9_strategic_tf":
                strat_emb = _mean_clamp(strat_emb)

            gwt_base, _, ign, _ = eng.models["gwt"](
                z_perc, sctx, eff_mec, eff_hd, eff_comp, eff_h, eff_pst, prop, haz_p, strat_emb,
                prev_gwt, meta_salience=eff_ssurp, hab_trace=prev_hab, novelty_5=nov5,
            )
            delta_gwt, _ = eng.models["topo_adapter"].compute_topdown_spotlight(gwt_base, slot_aux["feat16"], slots, s_coords)
            gwt_w = gwt_base + delta_gwt
            if clamp_key == "L8_gwt_broadcast":
                gwt_w = _mean_clamp(gwt_w)

            z_td = (0.08 + 0.24 * ign).unsqueeze(-1) * eng.models["gwt"].topdown_z(gwt_w)
            if clamp_key == "L3_topdown_t1_to_t":
                z_td = _mean_clamp(z_td)
            z_eff = torch_m.tanh(z_perc + 0.70 * z_td)

            q_raw, _ = eng.models["d3qn"](z_eff, gwt_w, eff_h, eff_mec, eff_hd, prop, eff_comp, eff_cf)
            q_np, rew_np, surv_np, adv_np = q_raw.cpu().numpy(), p_rew.cpu().numpy(), p_surv.cpu().numpy(), p_adv.cpu().numpy()
            gwt_np, h_np = gwt_w.cpu().numpy(), eff_h.cpu().numpy()

            if clamp_key == "L7_rssm_3head_imagination":
                rew_np = np.broadcast_to(rew_np.mean(axis=0, keepdims=True), rew_np.shape).copy()
                surv_np = np.broadcast_to(surv_np.mean(axis=0, keepdims=True), surv_np.shape).copy()
                adv_np = np.broadcast_to(adv_np.mean(axis=0, keepdims=True), adv_np.shape).copy()

            q_stars, pis = [], []
            for k in range(bsz):
                _, _, _, pb = eng.rvq.quantize(gwt_np[k], np.zeros(8, dtype=np.float32), h_np[k], np.array([1, 4, 4, 5, 5, 4, 1, 0], dtype=np.uint8), update_ema=False)
                if clamp_key == "L10_self_and_rvq":
                    pb = np.zeros_like(pb)
                qf = q_np[k] + 0.30 * rew_np[k] + 0.25 * (adv_np[k] - 0.5) + 0.15 * pb
                c_poses = snapshots[k]["cand_feats32"][:, :6]
                c_verbs = np.array([1, 4, 4, 5, 5, 4, 1, 0], dtype=np.uint8)
                ss = 0.03 if clamp_key == "L10_self_and_rvq" else float(snapshots[k].get("self_surprise", 0.04))
                qs, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                    qf, surv_np[k], adv_np[k], gwt_np[k], c_poses, c_verbs, snapshots[k]["proprio"],
                    eff_comp[k].cpu().numpy(), None, cfg, q_raw_values=q_np[k], self_surprise=ss,
                )
                q_stars.append(qs); pis.append(pi)
            return np.stack(q_stars), np.stack(pis)

    q_full, pi_full = _eval_with_clamp("NONE")
    top_full = np.argmax(q_full, axis=-1)

    links = [
        "L1_foveal_z", "L2_slot_cortex", "L3_topdown_t1_to_t", "L4_mec_hd_3d", "L5_mem_compass_24d",
        "L6_rssm_recurrent_state", "L7_rssm_3head_imagination", "L8_gwt_broadcast", "L9_strategic_tf", "L10_self_and_rvq",
    ]
    out: Dict[str, Any] = {"n_states": bsz, "links": {}}
    ln2 = math.log(2.0)
    for lk in links:
        qc, pic = _eval_with_clamp(lk)
        top_c = np.argmax(qc, axis=-1)
        flip_pct = float(np.mean(top_c != top_full) * 100.0)
        max_dq = float(np.mean(np.max(np.abs(q_full - qc), axis=-1)))
        p1, p2 = np.clip(pi_full, 1e-7, 1.0), np.clip(pic, 1e-7, 1.0)
        kl_bits = float(np.mean(np.sum(p1 * (np.log(p1) - np.log(p2)), axis=-1)) / ln2)
        out["links"][lk] = {
            "action_flip_pct": round(flip_pct, 2),
            "mean_max_abs_dq": round(max_dq, 4),
            "policy_kl_bits": round(kl_bits, 5),
        }
        print(f"  [Link {lk:26s}] Flip={flip_pct:5.2f}% | mean max|dQ|={max_dq:.4f} | KL={kl_bits:.5f}b")

    eng.release_memory_and_vram()
    return out


class ModularAgentSlot:
    def __init__(self, cfg: Brain3DConfig, slot_id: int, variant: str):
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self.slot_id = int(slot_id)
        self.variant = variant
        if variant == "S4_slam_drift_only":
            self.cfg.spatial_mode = SPATIAL_MODE_DRIFT_ONLY
        elif variant == "L0_reactive":
            self.cfg.spatial_mode = SPATIAL_MODE_TUNNEL_RAW
        if variant in ("L0_reactive", "L1_slam_map", "L2_homeo_scaffold", "L3_neural_d3qn", "L4_plus_rssm"):
            self.cfg.timing_mode = TIMING_MODE_INSTANT
        elif variant == "S5_bg_cautious":
            self.cfg.timing_mode = TIMING_MODE_CAUTIOUS

        self.hippo = EntorhinalHippocampal3DEngineV2(self.cfg)
        self.hippo.set_learning_context(False)
        self.cereb = Cerebellar3DKinematicPredictorV2(self.cfg)
        self.bg = BasalGanglia3DTimingEngineV2(self.cfg)
        self.homeostasis = ActionStagnationHomeostasis3D(self.cfg.world_min, self.cfg.world_max)
        self.ventral_prev: Dict[str, Any] = {}
        self.last_lead_xyz = np.zeros(3, dtype=np.float32)
        self.last_goal_mode = GOAL_MODE_BALANCED_EXPLORE
        self.last_focus_xyz = np.zeros(3, dtype=np.float32)
        self.last_slot_coords = None
        self.last_slot_valence = None
        self.last_vis_surp = 0.0
        self.last_w_surp = 0.0
        self.last_ign = 0.0
        self.last_s_surp = 0.0
        self.prev_bel_xz = np.zeros(2, dtype=np.float32)
        self.respawn_seq = 0
        self.cached_idx = 0
        self.cached_verb = VERB_APPROACH_HOVER
        self.cached_pose = np.zeros(6, dtype=np.float32)
        self.cached_q = 0.5
        self.cached_surv = 0.85
        self.s1_fast_idx = 0
        self.stag_samples: List[float] = []
        self.nan_or_contract_errors = 0

    @property
    def needs_gpu(self) -> bool:
        return self.variant not in ("L0_reactive", "L1_slam_map", "L2_homeo_scaffold")

    def pre_step(self, obs: Dict[str, Any], dt: float, cog_stride: int = 4) -> Optional[Dict[str, Any]]:
        if bool(obs.get("respawn_flag", False)):
            self.respawn_seq += 1
        if self.variant == "L0_reactive":
            self.hippo.field_map.reset_full_episode()
            self.hippo.landmark_anchors.clear()
            self.hippo.memory_slots.clear()

        vent_p = derive_ventral_fallback_from_dorsal_16(obs["fovea_patch"], yaw_pitch=obs["self_yaw_pitch"], prev_state=self.ventral_prev, episode_id=1)
        sp_out = self.hippo.step(
            obs, dt, ventral_override=vent_p, respawn_atom=self.respawn_seq,
            topdown_goal_mode=self.last_goal_mode, topdown_focus_xyz=self.last_focus_xyz, lead_aim_xyz=self.last_lead_xyz,
            vis_slot_coords=self.last_slot_coords, vis_slot_valence=self.last_slot_valence,
            vis_pred_surprise=self.last_vis_surp, world_surprise=self.last_w_surp, gwt_ignition=self.last_ign,
        )
        if self.variant == "L0_reactive":
            self.hippo.field_map.reset_full_episode()
        elif bool(sp_out.get("respawn_event_flag", False)):
            self.hippo.field_map.register_strategic_hazard_scar(float(self.prev_bel_xz[0]), float(self.prev_bel_xz[1]))
        self.prev_bel_xz = sp_out["believed_self_pos"][[0, 2]].copy()

        lead_xyz, opp_prob, kin_risk, sc_spd = self.cereb.predict_lead_and_crossfire(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"]
        )
        self.last_lead_xyz = lead_xyz.copy()
        prop = project_pomdp_proprio(obs["proprio"], sp_out["believed_self_pos"], self.cfg.world_min, self.cfg.world_max, self.cfg)
        mc24 = compute_universal_memory_compass(sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop, self.hippo.field_map, self.profile)
        c_poses, c_verbs = propose_universal_3d_candidates(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop,
            mc24, self.hippo.field_map, lead_xyz, sp_out["active_gaze_ypd"],
            self.cfg.world_min, self.cfg.world_max, self.profile, goal_mode=self.last_goal_mode,
        )
        self._ctx = (obs, dt, vent_p, sp_out, opp_prob, kin_risk, sc_spd, prop, mc24, c_poses, c_verbs)
        tick = int(obs["physics_tick"])
        want_cog = (tick % cog_stride == 0) or bool(obs["episode_done"]) or bool(sp_out["respawn_event_flag"]) or (tick <= 1)
        self._want_cog = want_cog
        if want_cog and self.needs_gpu:
            return {
                "slot_id": self.slot_id, "episode_id": 1, "respawn_flag": bool(sp_out["respawn_event_flag"]),
                "enable_learning": False, "fovea_patch": obs["fovea_patch"], "ventral_patch": vent_p,
                "proprio": prop, "mem_compass": mc24, "mec_grid_code": sp_out["mec_grid_code"],
                "hd_ring_code": sp_out["hd_ring_code"], "believed_self_pos": sp_out["believed_self_pos"],
                "cand_poses": c_poses, "cand_verbs": c_verbs,
            }
        return None

    def post_step(self, gwt_out: Optional[Dict[str, Any]], engine: WholeBrainNeuralEngineWrapper, a_seed: int) -> Tuple[np.ndarray, int, bool, np.ndarray]:
        obs, dt, _, sp_out, opp_prob, kin_risk, sc_spd, prop, mc24, c_poses, c_verbs = self._ctx
        tick = int(obs["physics_tick"])

        if self._want_cog:
            rng_seed = tick + self.slot_id * 7919 + a_seed * 31
            if not self.needs_gpu:
                zeros8 = np.zeros(8, dtype=np.float32)
                surv8 = np.full(8, 0.85, dtype=np.float32)
                adv8 = np.full(8, 0.50, dtype=np.float32)
                use_homeo = (self.variant == "L2_homeo_scaffold")
                if self.variant == "L0_reactive":
                    c_idx = int(((rng_seed * 1664525 + 1013904223) >> 8) % 6)
                else:
                    q_star, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                        zeros8, surv8, adv8, np.zeros(128, dtype=np.float32), c_poses, c_verbs, prop, mc24, None, self.cfg,
                        habituation_trace=(self.homeostasis.habituation_trace if use_homeo else None),
                        stagnation_level=(self.homeostasis.stagnation_level if use_homeo else 0.0),
                    )
                    c_idx = select_policy_candidate(q_star, pi, trainable=False, rng_seed_val=rng_seed)
                _ = self.homeostasis.update_and_compute_intrinsic(1, sp_out["believed_self_pos"][[0, 2]], prop, c_idx, float(obs["last_step_reward"]))
                self.stag_samples.append(float(self.homeostasis.stagnation_level))
                self.cached_idx = c_idx
                self.cached_verb = int(c_verbs[c_idx])
                self.cached_pose = c_poses[c_idx].copy()
                self.cached_pose[5] = 0.88
                self.cached_q = 0.5
                self.cached_surv = 0.85
                self.s1_fast_idx = c_idx
            else:
                assert gwt_out is not None
                if self.variant in ("S2_gwt_uniform_pool", "S3_gwt_hard_wta"):
                    with engine.torch.no_grad():
                        t = engine._t; tf = engine.torch.float32
                        st_s = engine._agent_states[self.slot_id]
                        z_t = t(gwt_out["latent_z"][None, :], tf)
                        sctx_t = t(gwt_out["slot_ctx64"][None, :], tf)
                        mec_t = t(sp_out["mec_grid_code"][None, :], tf)
                        hd_t = t(sp_out["hd_ring_code"][None, :], tf)
                        comp_t = t(mc24[None, :], tf)
                        h_t = t(gwt_out["rssm_h"][None, :], tf)
                        st_t = t(st_s["rssm_stoch"][None, :], tf)
                        pr_t = t(prop[None, :], tf)
                        haz_t = engine.models["hazard_net"](z_t, pr_t, comp_t)
                        sh_list = list(st_s["strat_hist"])
                        while len(sh_list) < 8:
                            sh_list.insert(0, sh_list[0] if sh_list else np.zeros(160, dtype=np.float32))
                        sseq_t = t(np.stack(sh_list[-8:], axis=0)[None, ...], tf)
                        semb_t, _, _ = engine.models["strat_tf"](sseq_t)
                        gm = engine.models["gwt"]
                        sps = engine.torch.stack([
                            F.gelu(gm.s0_vis(engine.torch.cat([z_t, sctx_t], dim=-1))),
                            F.gelu(gm.s1_spa(engine.torch.cat([mec_t, hd_t, comp_t], dim=-1))),
                            F.gelu(gm.s2_rssm(engine.torch.cat([h_t, st_t], dim=-1))),
                            F.gelu(gm.s3_prop(engine.torch.cat([pr_t, haz_t.unsqueeze(-1)], dim=-1))),
                            F.gelu(gm.s4_strat(semb_t)),
                        ], dim=1)
                        if self.variant == "S2_gwt_uniform_pool":
                            alp_sw = engine.torch.full((1, 5), 0.20, device=z_t.device)
                        else:
                            sc_raw = gm.salience_scorer(sps).squeeze(-1)
                            alp_sw = F.one_hot(sc_raw.argmax(dim=-1), 5).float()
                        pooled_sw = engine.torch.einsum("bs,bsd->bd", alp_sw, sps)
                        gw_sw = gm.broadcast_norm(0.50 * t(st_s["gwt_w"][None, :], tf) + 0.50 * pooled_sw)
                        cf_t = t(_extract_candidate_compass_features_np(c_poses, c_verbs, prop, mc24)[None, ...], tf)
                        q_sw, spd_sw = engine.models["d3qn"](z_t, gw_sw, h_t, mec_t, hd_t, pr_t, comp_t, cf_t)
                        q_raw_sw = q_sw[0].cpu().numpy()
                        gwt_out = dict(gwt_out)
                        gwt_out["q_raw_values"] = q_raw_sw
                        gwt_out["q_values"] = q_raw_sw + 0.30 * gwt_out["pred_rew"] + 0.25 * (gwt_out["pred_adv_probs"] - 0.5) + 0.15 * gwt_out["proto_prior_bias"]
                        gwt_out["speed_mod"] = spd_sw[0].cpu().numpy()
                        gwt_out["gwt_workspace"] = gw_sw[0].cpu().numpy()
                        st_s["gwt_w"] = gwt_out["gwt_workspace"].copy()

                self.last_goal_mode = int(gwt_out["topdown_goal_mode"])
                self.last_focus_xyz = gwt_out["topdown_focus_xyz"].copy()
                self.last_vis_surp = float(gwt_out.get("vis_pred_surprise", 0.0))
                self.last_w_surp = float(gwt_out.get("world_surprise", 0.0))
                self.last_ign = float(gwt_out.get("ignition", 0.0))
                self.last_s_surp = float(gwt_out.get("self_surprise", 0.0))
                self.last_slot_coords = gwt_out["vis_slot_coords"].copy()
                self.last_slot_valence = gwt_out["vis_slot_valence"].copy()

                if self.variant == "L3_neural_d3qn":
                    q_use = gwt_out["q_raw_values"] + 0.15 * gwt_out["proto_prior_bias"]
                    surv_use = np.full(8, 0.88, dtype=np.float32)
                    adv_use = np.full(8, 0.50, dtype=np.float32)
                elif self.variant == "S1_rssm_mpc_no_d3qn":
                    q_use = 1.25 * gwt_out["pred_rew"] + 0.85 * (gwt_out["pred_adv_probs"] - 0.5) + 0.65 * (gwt_out["survival_probs"] - 0.5) + 0.15 * gwt_out["proto_prior_bias"]
                    surv_use = gwt_out["survival_probs"]
                    adv_use = gwt_out["pred_adv_probs"]
                else:
                    q_use = gwt_out["q_values"]
                    surv_use = gwt_out["survival_probs"]
                    adv_use = gwt_out["pred_adv_probs"]

                if (not np.all(np.isfinite(q_use))) or float(np.max(np.abs(q_use))) > 50.0:
                    self.nan_or_contract_errors += 1

                q_star, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                    q_use, surv_use, adv_use, gwt_out["gwt_workspace"], c_poses, c_verbs, prop, mc24, None, self.cfg,
                    habituation_trace=self.homeostasis.habituation_trace, stagnation_level=self.homeostasis.stagnation_level,
                    q_raw_values=gwt_out.get("q_raw_values"), self_surprise=self.last_s_surp,
                )
                c_idx = select_policy_candidate(q_star, pi, trainable=False, rng_seed_val=rng_seed)
                if c_idx != int(gwt_out.get("prov_chosen_idx", -1)):
                    engine.notify_chosen_action(self.slot_id, gwt_out["cand_feats32"][c_idx].copy())
                _ = self.homeostasis.update_and_compute_intrinsic(
                    1, sp_out["believed_self_pos"][[0, 2]], prop, c_idx, float(obs["last_step_reward"]),
                    world_surprise=self.last_w_surp, self_surprise=self.last_s_surp, respawn_flag=bool(sp_out["respawn_event_flag"]),
                )
                self.stag_samples.append(float(self.homeostasis.stagnation_level))
                self.cached_idx = c_idx
                self.cached_verb = int(c_verbs[c_idx])
                self.cached_pose = c_poses[c_idx].copy()
                self.cached_pose[5] = float(gwt_out["speed_mod"][c_idx])
                self.cached_q = float(q_use[c_idx])
                self.cached_surv = float(surv_use[c_idx])
                self.s1_fast_idx = int(np.argmax(gwt_out["q_raw_values"]))

        bg_res = self.bg.step(
            episode_id=1, action_epoch=int(obs["action_epoch"]), can_commit_now=bool(obs["can_commit_now"]),
            chosen_cand_idx=self.cached_idx, s1_fast_cand_idx=self.s1_fast_idx, chosen_verb=self.cached_verb,
            q_chosen=self.cached_q, survival_prob=self.cached_surv, scene_opportunity_prob=opp_prob,
            local_kinetic_risk=kin_risk, believed_scene_speed=sc_spd, mean_fog=float(sp_out["mean_fog"]),
            saccade_hold_active=bool(sp_out["saccade_hold_active"]), respawn_event_flag=bool(sp_out["respawn_event_flag"]),
            coach_brake=False, in_refuge=is_in_stealth_or_cover_refuge(prop), tank_level=get_resource_tank_level(prop),
            hazard_level=float(obs["hazard_level"]), step_reward=float(obs["last_step_reward"]), dt=dt, self_surprise=self.last_s_surp,
        )
        return self.cached_pose.copy(), self.cached_verb, bool(bg_res["commit_gate_open"]), sp_out["active_gaze_ypd"].copy()


def evaluate_variant_vs_yardstick(
    ckpt: str,
    variant_name: str,
    yardstick_name: str,
    eng_shared: WholeBrainNeuralEngineWrapper,
    n_seeds: int = 24,
) -> Dict[str, Any]:
    seeds = [2000 + 97 * k for k in range(n_seeds)]
    eng_shared._agent_states.clear()
    dt = 1.0 / 60.0

    arenas, obs_s_list, obs_n_list = [], [], []
    slots_s: List[ModularAgentSlot] = []
    slots_n: List[ModularAgentSlot] = []
    game_meta = []

    slot_cursor = 0
    for s_i, a_seed in enumerate(seeds):
        for swapped in (False, True):
            acfg = Brain3DConfig(task_profile="turf_objective", seed=a_seed)
            ar = HeadToHeadTactical3DArena(acfg, seed=a_seed)
            arenas.append(ar)
            obs_s_list.append(ar._build_agent_obs(+1, 0.0, False))
            obs_n_list.append(canonicalize_team_observation(ar._build_agent_obs(-1, 0.0, False), -1, ar.wmin, ar.wmax))

            v_south = yardstick_name if swapped else variant_name
            v_north = variant_name if swapped else yardstick_name
            cs = make_base_cfg(ckpt, profile="turf_objective", seed=3000 + s_i)
            cn = make_base_cfg(ckpt, profile="turf_objective", seed=4000 + s_i)
            slots_s.append(ModularAgentSlot(cs, slot_cursor, v_south)); slot_cursor += 1
            slots_n.append(ModularAgentSlot(cn, slot_cursor, v_north)); slot_cursor += 1
            game_meta.append((s_i, swapped, a_seed))

    n_games = len(arenas)
    deaths_s, deaths_n = [0] * n_games, [0] * n_games
    shots_s, shots_n = [0] * n_games, [0] * n_games
    max_p_s, max_p_n = [0.50] * n_games, [0.50] * n_games
    winners = [0] * n_games
    done_flags = [False] * n_games

    t_t0 = time.perf_counter()
    for _ in range(545):
        if all(done_flags):
            break
        batch_reqs = []
        req_map_s = [None] * n_games
        req_map_n = [None] * n_games
        for g in range(n_games):
            if done_flags[g]:
                continue
            rs = slots_s[g].pre_step(obs_s_list[g], dt, cog_stride=4)
            if rs is not None:
                req_map_s[g] = len(batch_reqs); batch_reqs.append(rs)
            rn = slots_n[g].pre_step(obs_n_list[g], dt, cog_stride=4)
            if rn is not None:
                req_map_n[g] = len(batch_reqs); batch_reqs.append(rn)

        batch_outs = eng_shared.infer_conscious_gwt_batch(batch_reqs) if batch_reqs else []

        for g in range(n_games):
            if done_flags[g]:
                continue
            _, _, a_seed = game_meta[g]
            gs_out = batch_outs[req_map_s[g]] if req_map_s[g] is not None else None
            gn_out = batch_outs[req_map_n[g]] if req_map_n[g] is not None else None
            prev_cd_s = int(arenas[g].agents[+1]["cd"])
            prev_cd_n = int(arenas[g].agents[-1]["cd"])
            act_s = slots_s[g].post_step(gs_out, eng_shared, a_seed)
            act_n = slots_n[g].post_step(gn_out, eng_shared, a_seed)
            obs_s, obs_n, ep_done, w = arenas[g].step_both(act_s, act_n, dt=dt)
            obs_s_list[g] = obs_s; obs_n_list[g] = obs_n
            if int(arenas[g].agents[+1]["cd"]) > prev_cd_s:
                shots_s[g] += 1
            if int(arenas[g].agents[-1]["cd"]) > prev_cd_n:
                shots_n[g] += 1
            if bool(obs_s.get("respawn_flag", False)):
                deaths_s[g] += 1
            if bool(obs_n.get("respawn_flag", False)):
                deaths_n[g] += 1
            max_p_s[g] = max(max_p_s[g], float(arenas[g].objective_progress_a))
            max_p_n[g] = max(max_p_n[g], float(1.0 - arenas[g].objective_progress_a))
            if ep_done:
                done_flags[g] = True
                winners[g] = int(w)

    wall_s = time.perf_counter() - t_t0
    pair_win, pair_prog, pair_kills, pair_deaths, pair_turf, pair_shots, pair_stag = [], [], [], [], [], [], []
    contract_errs = 0

    for s_i in range(n_seeds):
        g1, g2 = 2 * s_i, 2 * s_i + 1
        w1 = 1.0 if winners[g1] > 0 else (0.0 if winners[g1] < 0 else 0.5)
        w2 = 1.0 if winners[g2] < 0 else (0.0 if winners[g2] > 0 else 0.5)
        t1 = float(np.mean(arenas[g1].surface_grid > 0.25))
        t2 = float(np.mean(arenas[g2].surface_grid < -0.25))
        pair_win.append(0.5 * (w1 + w2))
        pair_prog.append(0.5 * (max_p_s[g1] + max_p_n[g2]) * 100.0)
        pair_kills.append(0.5 * (deaths_n[g1] + deaths_s[g2]))
        pair_deaths.append(0.5 * (deaths_s[g1] + deaths_n[g2]))
        pair_turf.append(0.5 * (t1 + t2) * 100.0)
        pair_shots.append(0.5 * (shots_s[g1] + shots_n[g2]))
        st1 = float(np.mean(slots_s[g1].stag_samples)) if slots_s[g1].stag_samples else 0.0
        st2 = float(np.mean(slots_n[g2].stag_samples)) if slots_n[g2].stag_samples else 0.0
        pair_stag.append(0.5 * (st1 + st2))
        contract_errs += slots_s[g1].nan_or_contract_errors + slots_n[g2].nan_or_contract_errors

    pc = paired_comparison(pair_win, [1.0 - x for x in pair_win], seed=2026)
    res = {
        "win_rate_pct": round(float(np.mean(pair_win)) * 100.0, 2),
        "ci95_pct": [round(summary_stats(pair_win)["ci95_low"] * 100.0, 1), round(summary_stats(pair_win)["ci95_high"] * 100.0, 1)],
        "w_l_t": f"{pc['n_pos']}-{pc['n_neg']}-{pc['n_tie']}",
        "sign_p": round(float(pc["binom_sign_p"]), 5),
        "wilcoxon_p": round(float(pc["wilcoxon_p"]), 5),
        "dz": round(float(pc["cohens_dz"]), 3),
        "max_prog_pct": round(float(np.mean(pair_prog)), 2),
        "kills_per_game": round(float(np.mean(pair_kills)), 2),
        "deaths_per_game": round(float(np.mean(pair_deaths)), 2),
        "turf_share_pct": round(float(np.mean(pair_turf)), 2),
        "shots_per_game": round(float(np.mean(pair_shots)), 1),
        "mean_stagnation": round(float(np.mean(pair_stag)), 3),
        "contract_errors": int(contract_errs),
        "eval_wall_sec_48games": round(wall_s, 2),
    }
    print(
        f"  [{variant_name:22s} vs {yardstick_name}] WR={res['win_rate_pct']:5.1f}% ({res['w_l_t']}) "
        f"prog={res['max_prog_pct']:5.1f}% K/D={res['kills_per_game']:.2f}/{res['deaths_per_game']:.2f} "
        f"turf={res['turf_share_pct']:4.1f}% shots={res['shots_per_game']:4.1f} stag={res['mean_stagnation']:.3f} "
        f"errs={res['contract_errors']} ({wall_s:.1f}s)"
    )
    return res


def main() -> None:
    t0 = time.perf_counter()
    ckpt = resolve_primary_ckpt()
    assert os.path.isfile(ckpt), f"Missing checkpoint: {ckpt}"
    print(f"[Exp6] Using checkpoint: {ckpt}")

    print("=== [Exp6-2A] Running 9-Loss Multi-Objective Gradient Conflict Audit ===")
    grad_audit = run_gradient_conflict_audit(ckpt)
    snapshots = grad_audit.pop("_snapshots")

    print("=== [Exp6-2B] Running 10-Link Cross-Module 'Silent Passenger' Scan ===")
    passenger_audit = run_silent_passenger_scan(ckpt, snapshots)

    eng_sh = WholeBrainNeuralEngineWrapper(make_base_cfg(ckpt, profile="turf_objective", seed=2026))

    print("=== [Exp6-3] Running 6-Rung Cumulative Organ Ladder (vs Fixed Yardstick L2) ===")
    ladder_res = {
        v: evaluate_variant_vs_yardstick(ckpt, v, "L2_homeo_scaffold", eng_sh, n_seeds=24)
        for v in ("L0_reactive", "L1_slam_map", "L2_homeo_scaffold", "L3_neural_d3qn", "L4_plus_rssm", "L5_full_v21")
    }

    print("=== [Exp6-1] Running 5 Organ-Swap Replaceability Tests (vs Fixed Yardstick L2) ===")
    swap_res = {
        v: evaluate_variant_vs_yardstick(ckpt, v, "L2_homeo_scaffold", eng_sh, n_seeds=24)
        for v in ("S1_rssm_mpc_no_d3qn", "S2_gwt_uniform_pool", "S3_gwt_hard_wta", "S4_slam_drift_only", "S5_bg_cautious")
    }

    eng_sh.release_memory_and_vram()

    payload = {
        "checkpoint_used": os.path.relpath(ckpt, ROOT_DIR).replace("\\", "/"),
        "criterion_1_replaceability_swaps": swap_res,
        "criterion_2a_gradient_conflict": grad_audit,
        "criterion_2b_silent_passenger_links": passenger_audit,
        "criterion_3_cumulative_ladder": ladder_res,
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    out_path = os.path.join(ROOT_DIR, "paper_arxiv", "experiments", "results", "exp6_architecture_criteria.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[Exp6] Saved -> {out_path}")


if __name__ == "__main__":
    main()
