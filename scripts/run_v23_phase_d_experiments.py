# -*- coding: utf-8 -*-
"""
scripts/run_v23_phase_d_experiments.py — Universal Brain 3D v2.3 Phase D Full Study
===================================================================================
Fast Multi-Core Zero-PyTorch Arena Pool + Single-GPU Server (`Barrier 1` Enforced):
  1. Multi-Seed Progressive `Scaffold Annealing` Training (`seeds = [2026, 2027, 2028]`):
     - Mixed-Stage Curriculum (`Stage 1: Classic Citadel` + `Stage 2: Occluded Twin-Flank`).
     - Progressive `Scaffold Annealing` (`scaffold_prior_scale`: `1.00 -> 0.65 -> 0.35 -> 0.12 -> 0.00`),
       removing the hand-crafted `L2` compass prior crutch so the trained v2.3 brain operates 100%
       autonomously on `D3QN + 3-Head RSSM + Active GWT + Strategic Transformer + RVQ + BG`.
     - High-Update Regime (`~3,200 - 3,800` gradient updates & `12` Sleep `S-1..S-4` cycles per seed).
  2. Falsifiable Integration Audits (Criterion 1 & 2):
     - Exact 9-Loss Shared-Overlap Pairwise Gradient Cosine Matrix (verifying Phase B world-model
       encoder ownership eliminates the `L_td x L_rssm` `-0.329` conflict).
     - 10-Link `Silent-Passenger` Scan under BOTH legacy `scaf = 1.0` AND `Autonomous (scaf = 0.0)`.
     - Cross-Layer Organ Swaps (`S1..S5`, verifying `0` contract/NaN errors).
  3. Stage 1 & Stage 2 Side-Swapped Behavioral Benchmarks (`N = 24` pairs = `48` games/seed, `144` pooled):
     - Autonomous Head-to-Head (`scaf = 0.0`): `v2.3` vs `Untrained` & `v2.3` vs `v2.2 Champion` (Stage 1 & 2).
     - Autonomous Neural (`scaf = 0.0`) vs Full Hand-Crafted `L2` Heuristic (`scaf = 1.0`) (Stage 1 & 2).
     - Hybrid Head-to-Head (`scaf = 1.0`): `v2.3` vs `v2.2 Champion` on `Stage 2: Occluded Twin-Flank`.
     - Cumulative Organ Ladder (`L0 -> L2 -> L3 -> L4 -> L5`) on `Stage 2: Occluded Twin-Flank`.

All checkpoints save to `checkpoints/v23_full_study/` and structured JSON saves to
`docs/v23_phase_d_benchmark_results.json`, keeping `user_runs/`, `remediated_v22/`, and `paper_arxiv/` untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
EXP_SCRIPTS_DIR = os.path.join(ROOT_DIR, "paper_arxiv", "experiments", "scripts")
if EXP_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, EXP_SCRIPTS_DIR)

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
    CAND_TO_GOAL_MODE_TUPLE,
    CFEAT_DIM,
    CONT_POSE_DIM,
    Cerebellar3DKinematicPredictorV2,
    EntorhinalHippocampal3DEngineV2,
    FOVEA_H,
    FOVEA_W,
    GOAL_MODE_BALANCED_EXPLORE,
    LATENT_Z_DIM,
    MC_FRONT_DX,
    MC_FRONT_DZ,
    MC_OBJ_DX,
    MC_OBJ_DZ,
    MC_OBJ_VALID,
    MC_THR_DIST,
    MC_THR_DX,
    MC_THR_DZ,
    MC_THR_VALID,
    NUM_CANDIDATES,
    NUM_OBJECT_SLOTS,
    PROPRIO_IDX_CAN_COMMIT,
    PROPRIO_IDX_HAZARD,
    PROPRIO_IDX_OBJ_PROGRESS,
    PROPRIO_IDX_RESOURCE_CARRIER,
    PROPRIO_IDX_SURFACE_POSTURE,
    PROTO_EMBED_DIM,
    RSSM_STOCH_CATS,
    RSSM_STOCH_FLAT,
    SELF_MODEL_DIM,
    SPATIAL_MODE_DRIFT_ONLY,
    SPATIAL_MODE_TUNNEL_RAW,
    TIMING_MODE_CAUTIOUS,
    TIMING_MODE_INSTANT,
    VERB_APPROACH_HOVER,
    VERB_IDLE_OBSERVE,
    VERB_IMPULSE_PUSH,
    VERB_RELEASE_PLACE,
    _ignore_sigint_in_worker,
    apply_checkpoint_genes_to_cfg,
    compose_vis8_patch,
    compute_gene_modulated_mip_and_fr,
    compute_task_scaffold_targets,
    compute_universal_memory_compass,
    derive_ventral_fallback_from_dorsal_16,
    get_resource_tank_level,
    get_task_profile,
    is_in_stealth_or_cover_refuge,
    load_checkpoint_contract_meta,
    project_pomdp_proprio,
    propose_universal_3d_candidates,
    select_policy_candidate,
)
from arena_pbt_trainer import (
    HeadToHeadTactical3DArena,
    canonicalize_team_observation,
    uncanonicalize_team_motor,
)
from stats_utils import paired_comparison, summary_stats


def make_eval_cfg(
    ckpt_path: str,
    *,
    stage_id: int = 1,
    scaffold_scale: float = 0.0,
    seed: int = 2026,
) -> Brain3DConfig:
    cfg = Brain3DConfig(
        task_profile="turf_objective",
        policy_mode="pure_neural",
        enable_learning=False,
        load_weights_path=ckpt_path,
        default_stage_id=int(stage_id),
        scaffold_prior_scale=float(np.clip(scaffold_scale, 0.0, 1.0)),
        seed=int(seed),
    )
    if ckpt_path:
        meta = load_checkpoint_contract_meta(ckpt_path)
        genes = meta.get("genes") if isinstance(meta.get("genes"), dict) else None
        apply_checkpoint_genes_to_cfg(cfg, genes)
    # Explicit experiment override for scaffold_prior_scale & stage_id
    cfg.scaffold_prior_scale = float(np.clip(scaffold_scale, 0.0, 1.0))
    cfg.default_stage_id = int(stage_id)
    return cfg


# ==============================================================================
# 1. Fast Zero-PyTorch Embodied Agent Slot (4x Fewer Non-Cognitive A* Calls)
# ==============================================================================
class FastEmbodiedSlot:
    """
    Zero-PyTorch CPU embodied controller supporting:
      - Full neural variants (`L5_full`, `L4_plus_rssm`, `L3_neural_d3qn`)
      - Organ swaps (`S1_rssm_mpc_no_d3qn`, `S2_gwt_uniform_pool`, `S4_slam_drift_only`, `S5_bg_cautious`)
      - Non-neural baselines (`L0_reactive`, `L1_slam_map`, `L2_homeo_scaffold`)
      - Trainable self-play transition recording (`trainable=True`)
    Strictly preserves cognitive-tick (`want_cog == True`) inputs while caching `mc24/c_poses` across
    the 3 intermediate 60Hz motor-execution ticks (`50ms`), matching the 15Hz System-1/2 architecture.
    """

    def __init__(
        self,
        cfg: Brain3DConfig,
        slot_id: int,
        variant: str = "L5_full",
        trainable: bool = False,
    ):
        self.cfg = Brain3DConfig(**asdict(cfg))
        self.profile = get_task_profile(self.cfg.task_profile)
        self.slot_id = int(slot_id)
        self.variant = str(variant)
        self.trainable = bool(trainable)

        if self.variant == "S4_slam_drift_only":
            self.cfg.spatial_mode = SPATIAL_MODE_DRIFT_ONLY
        elif self.variant == "L0_reactive":
            self.cfg.spatial_mode = SPATIAL_MODE_TUNNEL_RAW

        if self.variant in ("L0_reactive", "L1_slam_map", "L2_homeo_scaffold", "L3_neural_d3qn", "L4_plus_rssm"):
            self.cfg.timing_mode = TIMING_MODE_INSTANT
        elif self.variant == "S5_bg_cautious":
            self.cfg.timing_mode = TIMING_MODE_CAUTIOUS

        if self.variant in ("L1_slam_map", "L2_homeo_scaffold"):
            self.cfg.scaffold_prior_scale = 1.0

        self.hippo = EntorhinalHippocampal3DEngineV2(self.cfg)
        self.hippo.set_learning_context(self.trainable)
        self.cereb = Cerebellar3DKinematicPredictorV2(self.cfg)
        self.bg = BasalGanglia3DTimingEngineV2(self.cfg)
        self.homeostasis = ActionStagnationHomeostasis3D(self.cfg.world_min, self.cfg.world_max)

        self.ventral_prev: Dict[str, Any] = {}
        self.prev_pi: Optional[np.ndarray] = None
        self.last_lead_xyz = np.zeros(3, dtype=np.float32)
        self.last_goal_mode = GOAL_MODE_BALANCED_EXPLORE
        self.last_focus_xyz = np.zeros(3, dtype=np.float32)
        self.last_slot_coords: Optional[np.ndarray] = None
        self.last_slot_valence: Optional[np.ndarray] = None
        self.last_vis_surp = 0.0
        self.last_w_surp = 0.0
        self.last_ign = 0.0
        self.last_s_surp = 0.0
        self.prev_bel_xz = np.zeros(2, dtype=np.float32)
        self.respawn_seq = 0
        self.last_ep_seen = -1

        self.cached_idx = 0
        self.cached_verb = VERB_APPROACH_HOVER
        self.cached_pose = np.zeros(CONT_POSE_DIM, dtype=np.float32)
        self.cached_speed = 0.88
        self.cached_q = 0.5
        self.cached_surv = 0.85
        self.s1_fast_idx = 0
        self._cached_mc24: Optional[np.ndarray] = None
        self._cached_c_poses: Optional[np.ndarray] = None
        self._cached_c_verbs: Optional[np.ndarray] = None
        self.pending_tr: Optional[Dict[str, Any]] = None
        self.stag_samples: List[float] = []
        self.nan_or_contract_errors = 0
        self._ctx: Any = None
        self._want_cog = False

    @property
    def needs_gpu(self) -> bool:
        return self.variant not in ("L0_reactive", "L1_slam_map", "L2_homeo_scaffold")

    def pre_step(self, obs: Dict[str, Any], dt: float, cog_stride: int = 4) -> Optional[Dict[str, Any]]:
        ep_id = int(obs["episode_id"])
        if ep_id != self.last_ep_seen:
            self.last_ep_seen = ep_id
            self.pending_tr = None
            self.prev_pi = None
            self.last_slot_coords = None
            self.last_slot_valence = None
            self.last_vis_surp = 0.0
            self.last_w_surp = 0.0
            self.last_ign = 0.0
            self.last_s_surp = 0.0
            self._cached_mc24 = None
            self._cached_c_poses = None
            self._cached_c_verbs = None
        if bool(obs.get("respawn_flag", False)):
            self.respawn_seq += 1
        if self.variant == "L0_reactive":
            self.hippo.field_map.reset_full_episode()
            self.hippo.landmark_anchors.clear()
            self.hippo.memory_slots.clear()

        vent_p = derive_ventral_fallback_from_dorsal_16(
            obs["fovea_patch"], yaw_pitch=obs["self_yaw_pitch"], prev_state=self.ventral_prev, episode_id=ep_id
        )
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

        tick = int(obs["physics_tick"])
        ep_done = bool(obs["episode_done"])
        respawn_ev = bool(sp_out["respawn_event_flag"])
        want_cog = (tick % cog_stride == 0) or ep_done or respawn_ev or (tick <= 1) or (self.trainable and self.pending_tr is None)
        self._want_cog = want_cog

        if want_cog or self._cached_mc24 is None or self._cached_c_poses is None or self._cached_c_verbs is None:
            mc24 = compute_universal_memory_compass(
                sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop, self.hippo.field_map, self.profile
            )
            c_poses, c_verbs = propose_universal_3d_candidates(
                sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop,
                mc24, self.hippo.field_map, lead_xyz, sp_out["active_gaze_ypd"],
                self.cfg.world_min, self.cfg.world_max, self.profile, goal_mode=self.last_goal_mode,
            )
            self._cached_mc24 = mc24
            self._cached_c_poses = c_poses
            self._cached_c_verbs = c_verbs
        else:
            mc24 = self._cached_mc24
            c_poses = self._cached_c_poses
            c_verbs = self._cached_c_verbs

        self._ctx = (obs, dt, vent_p, sp_out, opp_prob, kin_risk, sc_spd, prop, mc24, c_poses, c_verbs, tick, ep_done, respawn_ev)
        if want_cog and self.needs_gpu:
            req: Dict[str, Any] = {
                "slot_id": self.slot_id,
                "episode_id": ep_id,
                "respawn_flag": respawn_ev,
                "enable_learning": self.trainable,
                "fovea_patch": obs["fovea_patch"],
                "ventral_patch": vent_p,
                "proprio": prop,
                "mem_compass": mc24,
                "mec_grid_code": sp_out["mec_grid_code"],
                "hd_ring_code": sp_out["hd_ring_code"],
                "believed_self_pos": sp_out["believed_self_pos"],
                "cand_poses": c_poses,
                "cand_verbs": c_verbs,
            }
            if self.variant == "S2_gwt_uniform_pool":
                req["gwt_pool_override"] = "uniform"
            return req
        return None

    def post_step(
        self,
        gwt_out: Optional[Dict[str, Any]],
        *,
        beta_teacher: float = 0.0,
        arena_seed: int = 0,
        chosen_notify_list: Optional[List[Tuple[int, np.ndarray]]] = None,
        transition_out_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int, bool, np.ndarray]:
        obs, dt, vent_p, sp_out, opp_prob, kin_risk, sc_spd, prop, mc24, c_poses, c_verbs, tick, ep_done, respawn_ev = self._ctx
        ep_id = int(obs["episode_id"])
        rng_seed = tick + self.slot_id * 7919 + ep_id * 131 + arena_seed * 31
        just_switched = False

        if self._want_cog:
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
                _ = self.homeostasis.update_and_compute_intrinsic(ep_id, sp_out["believed_self_pos"][[0, 2]], prop, c_idx, float(obs["last_step_reward"]))
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
                    q_use = (
                        1.25 * gwt_out["pred_rew"]
                        + 0.85 * (gwt_out["pred_adv_probs"] - 0.5)
                        + 0.65 * (gwt_out["survival_probs"] - 0.5)
                        + 0.15 * gwt_out["proto_prior_bias"]
                    )
                    surv_use = gwt_out["survival_probs"]
                    adv_use = gwt_out["pred_adv_probs"]
                else:
                    q_use = gwt_out["q_values"]
                    surv_use = gwt_out["survival_probs"]
                    adv_use = gwt_out["pred_adv_probs"]

                if (not np.all(np.isfinite(q_use))) or float(np.max(np.abs(q_use))) > 50.0:
                    self.nan_or_contract_errors += 1

                q_star, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                    q_use, surv_use, adv_use, gwt_out["gwt_workspace"], c_poses, c_verbs, prop, mc24,
                    (self.prev_pi if self.trainable else None), self.cfg,
                    habituation_trace=self.homeostasis.habituation_trace,
                    stagnation_level=self.homeostasis.stagnation_level,
                    q_raw_values=gwt_out.get("q_raw_values"),
                    self_surprise=self.last_s_surp,
                )
                self.prev_pi = pi.copy()
                self.s1_fast_idx = int(np.argmax(gwt_out["q_raw_values"]))

                tq, tsurv, tidx = None, None, 0
                if beta_teacher > 0.01:
                    sp_dict = {
                        "believed_self_pos": sp_out["believed_self_pos"],
                        "believed_hazard": sp_out["believed_hazard"],
                        "mem_compass": mc24,
                        "believed_entities": sp_out["believed_entities"],
                        "believed_count": sp_out["believed_count"],
                    }
                    tq, tsurv, tidx = compute_task_scaffold_targets(
                        sp_dict, obs, c_poses, c_verbs, self.cfg.world_min, self.cfg.world_max, self.profile
                    )
                if beta_teacher >= 0.999 or (beta_teacher > 0.01 and (((rng_seed * 1664525 + 1013904223) & 0xFFFF) < int(beta_teacher * 65535.0))):
                    c_idx = int(tidx)
                else:
                    c_idx = select_policy_candidate(q_star, pi, trainable=self.trainable, rng_seed_val=rng_seed)

                if c_idx != int(gwt_out.get("prov_chosen_idx", -1)) and chosen_notify_list is not None:
                    chosen_notify_list.append((self.slot_id, gwt_out["cand_feats32"][c_idx].copy()))

                r_int = self.homeostasis.update_and_compute_intrinsic(
                    episode_id=ep_id,
                    pos_xz=sp_out["believed_self_pos"][[0, 2]],
                    proprio=prop,
                    chosen_idx=c_idx,
                    step_extrinsic_reward=float(obs["last_step_reward"]),
                    world_surprise=self.last_w_surp,
                    self_surprise=self.last_s_surp,
                    respawn_flag=respawn_ev,
                )
                self.stag_samples.append(float(self.homeostasis.stagnation_level))

                just_switched = (c_idx != self.cached_idx)
                self.cached_idx = c_idx
                self.cached_speed = float(gwt_out["speed_mod"][c_idx])
                self.cached_verb = int(c_verbs[c_idx])
                self.cached_pose = c_poses[c_idx].copy()
                self.cached_pose[5] = self.cached_speed
                self.cached_q = float(q_use[c_idx])
                self.cached_surv = float(surv_use[c_idx])

                cur_haz = max(float(obs["hazard_level"]), float(sp_out["believed_hazard"]))
                if self.trainable and self.pending_tr is not None and transition_out_list is not None:
                    same_ep = int(self.pending_tr["episode_id"]) == ep_id
                    if (not same_ep) or ep_done or respawn_ev or (tick - int(self.pending_tr["tick"]) >= 8):
                        rew_acc = float(obs["cumulative_score"]) - float(self.pending_tr["cum_score"]) if same_ep else float(obs["last_step_reward"])
                        haz_delta = cur_haz - float(self.pending_tr["hazard"])
                        self.pending_tr["reward"] = float(np.clip(rew_acc + r_int - 0.15 * max(0.0, haz_delta), -28.0, 35.0))
                        self.pending_tr["next_hazard"] = cur_haz
                        self.pending_tr["obj_adv_delta"] = float(prop[PROPRIO_IDX_OBJ_PROGRESS]) - float(self.pending_tr["obj_prog"])
                        self.pending_tr["done"] = bool(ep_done or (not same_ep))
                        self.pending_tr["respawn"] = bool(respawn_ev)
                        self.pending_tr["episode_score"] = float(obs["cumulative_score"])
                        self.pending_tr["enable_learning"] = True
                        self.pending_tr["next_state"] = {
                            "fovea_patch": obs["fovea_patch"].copy(),
                            "ventral_patch": vent_p.copy(),
                            "proprio": prop.copy(),
                            "mem_compass": mc24.copy(),
                            "mec_grid_code": sp_out["mec_grid_code"].copy(),
                            "hd_ring_code": sp_out["hd_ring_code"].copy(),
                            "cand_feats32": gwt_out["cand_feats32"].copy(),
                        }
                        transition_out_list.append(self.pending_tr)
                        self.pending_tr = None

                if self.trainable and self.pending_tr is None and (not ep_done):
                    tr_d: Dict[str, Any] = {
                        "slot_id": self.slot_id,
                        "episode_id": ep_id,
                        "action_epoch": int(obs["action_epoch"]),
                        "tick": tick,
                        "cum_score": float(obs["cumulative_score"]),
                        "hazard": cur_haz,
                        "obj_prog": float(prop[PROPRIO_IDX_OBJ_PROGRESS]),
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
                        "proto_embed": gwt_out["proto_embed"].copy(),
                        "world_surprise": float(gwt_out["world_surprise"]),
                        "self_surprise": self.last_s_surp,
                        "beta_teacher": float(beta_teacher),
                    }
                    if beta_teacher > 0.02 and tq is not None and tsurv is not None:
                        tr_d["teacher_q"] = tq
                        tr_d["teacher_surv"] = tsurv
                    self.pending_tr = tr_d
        else:
            c_live = int(np.clip(self.cached_idx, 0, NUM_CANDIDATES - 1))
            self.cached_verb = int(c_verbs[c_live])
            self.cached_pose = c_poses[c_live].copy()
            self.cached_pose[5] = self.cached_speed

        bg_res = self.bg.step(
            episode_id=ep_id, action_epoch=int(obs["action_epoch"]), can_commit_now=bool(obs["can_commit_now"]),
            chosen_cand_idx=self.cached_idx, s1_fast_cand_idx=self.s1_fast_idx, chosen_verb=self.cached_verb,
            q_chosen=self.cached_q, survival_prob=self.cached_surv, scene_opportunity_prob=opp_prob,
            local_kinetic_risk=kin_risk, believed_scene_speed=sc_spd, mean_fog=float(sp_out["mean_fog"]),
            saccade_hold_active=bool(sp_out["saccade_hold_active"]), respawn_event_flag=respawn_ev,
            coach_brake=False, in_refuge=is_in_stealth_or_cover_refuge(prop), tank_level=get_resource_tank_level(prop),
            hazard_level=max(float(obs["hazard_level"]), float(sp_out["believed_hazard"])),
            step_reward=float(obs["last_step_reward"]), dt=dt, self_surprise=self.last_s_surp,
        )
        gate_open = bool(bg_res["commit_gate_open"])
        if just_switched and self.cached_verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH):
            gate_open = False
        return self.cached_pose.copy(), self.cached_verb, gate_open, sp_out["active_gaze_ypd"].copy()


# ==============================================================================
# 2. Multi-Core Zero-PyTorch Arena Worker + Single-GPU Server (`Barrier 1` Safe)
# ==============================================================================
def _cpu_arena_batch_worker(
    worker_id: int,
    task_q: mp.Queue,
    gpu_req_q: mp.Queue,
    gpu_resp_q: mp.Queue,
    done_q: mp.Queue,
) -> None:
    """
    Strictly Zero-PyTorch (`assert "torch" not in sys.modules`) CPU worker!
    Executes assigned slices of 1v1 matches in parallel across CPU cores while sending batched
    `infer_gwt`, `notify_chosen`, and `record_tr` RPCs to the single GPU server process.
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violated inside _cpu_arena_batch_worker!")
    _ignore_sigint_in_worker()
    dt = 1.0 / 60.0

    while True:
        cmd = task_q.get()
        if cmd is None or cmd.get("op") == "stop":
            break

        specs: List[Dict[str, Any]] = cmd["specs"]
        n_games = len(specs)
        arenas: List[HeadToHeadTactical3DArena] = []
        slots_s: List[FastEmbodiedSlot] = []
        slots_n: List[FastEmbodiedSlot] = []
        obs_s_list, obs_n_list = [], []
        eng_key_s: List[str] = []
        eng_key_n: List[str] = []
        beta_s: List[float] = []
        max_ticks_all = 545

        for sp in specs:
            st_id = int(sp["stage_id"])
            a_seed = int(sp["arena_seed"])
            ar = HeadToHeadTactical3DArena(
                Brain3DConfig(task_profile="turf_objective", default_stage_id=st_id, seed=a_seed),
                seed=a_seed,
            )
            ar.episode_id = int(sp.get("episode_id", 1))
            arenas.append(ar)
            obs_s_list.append(ar._build_agent_obs(+1, 0.0, False))
            obs_n_list.append(canonicalize_team_observation(ar._build_agent_obs(-1, 0.0, False), -1, ar.wmin, ar.wmax))
            if st_id == 2:
                max_ticks_all = max(max_ticks_all, 785)

            cs = Brain3DConfig(**sp["cfg_south"])
            cn = Brain3DConfig(**sp["cfg_north"])
            slots_s.append(FastEmbodiedSlot(cs, int(sp["slot_id_south"]), sp["variant_south"], bool(sp.get("trainable_south", False))))
            slots_n.append(FastEmbodiedSlot(cn, int(sp["slot_id_north"]), sp["variant_north"], False))
            eng_key_s.append(str(sp["eng_south"]))  # "A" | "B" | "NONE"
            eng_key_n.append(str(sp["eng_north"]))
            beta_s.append(float(sp.get("beta_south", 0.0)))

        deaths_s, deaths_n = [0] * n_games, [0] * n_games
        shots_s, shots_n = [0] * n_games, [0] * n_games
        max_p_s, max_p_n = [0.50] * n_games, [0.50] * n_games
        winners = [0] * n_games
        done_flags = [False] * n_games
        pending_notifies_a: List[Tuple[int, np.ndarray]] = []
        pending_notifies_b: List[Tuple[int, np.ndarray]] = []
        pending_trs_a: List[Dict[str, Any]] = []

        for _ in range(max_ticks_all):
            if all(done_flags):
                break
            reqs_a, reqs_b = [], []
            map_s: List[Optional[Tuple[str, int]]] = [None] * n_games
            map_n: List[Optional[Tuple[str, int]]] = [None] * n_games

            for g in range(n_games):
                if done_flags[g]:
                    continue
                rs = slots_s[g].pre_step(obs_s_list[g], dt, cog_stride=4)
                if rs is not None:
                    if eng_key_s[g] == "A":
                        map_s[g] = ("A", len(reqs_a)); reqs_a.append(rs)
                    elif eng_key_s[g] == "B":
                        map_s[g] = ("B", len(reqs_b)); reqs_b.append(rs)
                rn = slots_n[g].pre_step(obs_n_list[g], dt, cog_stride=4)
                if rn is not None:
                    if eng_key_n[g] == "A":
                        map_n[g] = ("A", len(reqs_a)); reqs_a.append(rn)
                    elif eng_key_n[g] == "B":
                        map_n[g] = ("B", len(reqs_b)); reqs_b.append(rn)

            outs_a, outs_b = [], []
            if reqs_a or reqs_b or pending_notifies_a or pending_notifies_b or pending_trs_a:
                gpu_req_q.put({
                    "wid": worker_id,
                    "reqs_a": reqs_a,
                    "reqs_b": reqs_b,
                    "notifies_a": pending_notifies_a,
                    "notifies_b": pending_notifies_b,
                    "trs_a": pending_trs_a,
                })
                pending_notifies_a = []
                pending_notifies_b = []
                pending_trs_a = []
                resp = gpu_resp_q.get()
                outs_a = resp["outs_a"]
                outs_b = resp["outs_b"]

            for g in range(n_games):
                if done_flags[g]:
                    continue
                gs = (outs_a[map_s[g][1]] if map_s[g][0] == "A" else outs_b[map_s[g][1]]) if map_s[g] is not None else None
                gn = (outs_a[map_n[g][1]] if map_n[g][0] == "A" else outs_b[map_n[g][1]]) if map_n[g] is not None else None

                not_s = pending_notifies_a if eng_key_s[g] == "A" else (pending_notifies_b if eng_key_s[g] == "B" else None)
                not_n = pending_notifies_a if eng_key_n[g] == "A" else (pending_notifies_b if eng_key_n[g] == "B" else None)
                tr_s = pending_trs_a if (slots_s[g].trainable and eng_key_s[g] == "A") else None

                a_seed = int(specs[g]["arena_seed"])
                prev_cd_s = int(arenas[g].agents[+1]["cd"])
                prev_cd_n = int(arenas[g].agents[-1]["cd"])

                act_s = slots_s[g].post_step(gs, beta_teacher=beta_s[g], arena_seed=a_seed, chosen_notify_list=not_s, transition_out_list=tr_s)
                act_n = slots_n[g].post_step(gn, beta_teacher=0.0, arena_seed=a_seed, chosen_notify_list=not_n, transition_out_list=None)
                obs_s, obs_n, ep_done, w = arenas[g].step_both(act_s, act_n, dt=dt)
                obs_s_list[g] = obs_s
                obs_n_list[g] = obs_n

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
                    if slots_s[g].trainable and eng_key_s[g] == "A":
                        rs_fin = slots_s[g].pre_step(obs_s, dt, cog_stride=4)
                        if rs_fin is not None:
                            gpu_req_q.put({
                                "wid": worker_id, "reqs_a": [rs_fin], "reqs_b": [],
                                "notifies_a": pending_notifies_a, "notifies_b": pending_notifies_b, "trs_a": pending_trs_a,
                            })
                            pending_notifies_a, pending_notifies_b, pending_trs_a = [], [], []
                            fin_resp = gpu_resp_q.get()
                            _ = slots_s[g].post_step(
                                fin_resp["outs_a"][0], beta_teacher=beta_s[g], arena_seed=a_seed,
                                chosen_notify_list=pending_notifies_a, transition_out_list=pending_trs_a,
                            )
                    done_flags[g] = True
                    winners[g] = int(w)

        if pending_notifies_a or pending_notifies_b or pending_trs_a:
            gpu_req_q.put({
                "wid": worker_id, "reqs_a": [], "reqs_b": [],
                "notifies_a": pending_notifies_a, "notifies_b": pending_notifies_b, "trs_a": pending_trs_a,
            })
            _ = gpu_resp_q.get()

        results = []
        for g in range(n_games):
            results.append({
                "game_idx": int(specs[g]["game_idx"]),
                "winner": int(winners[g]),
                "deaths_s": int(deaths_s[g]),
                "deaths_n": int(deaths_n[g]),
                "shots_s": int(shots_s[g]),
                "shots_n": int(shots_n[g]),
                "max_p_s": float(max_p_s[g]),
                "max_p_n": float(max_p_n[g]),
                "turf_s": float(np.mean(arenas[g].surface_grid > 0.25)),
                "turf_n": float(np.mean(arenas[g].surface_grid < -0.25)),
                "stag_s": float(np.mean(slots_s[g].stag_samples)) if slots_s[g].stag_samples else 0.0,
                "stag_n": float(np.mean(slots_n[g].stag_samples)) if slots_n[g].stag_samples else 0.0,
                "contract_errors": int(slots_s[g].nan_or_contract_errors + slots_n[g].nan_or_contract_errors),
                "zero_torch_worker": ("torch" not in sys.modules),
            })
        done_q.put({"wid": worker_id, "results": results})


def _apply_s2_uniform_pool_override(eng: Any, req: Dict[str, Any], gwt_out: Dict[str, Any]) -> Dict[str, Any]:
    """Exact S2 uniform GWT salience override on GPU when `req['gwt_pool_override'] == 'uniform'`."""
    from brain3d_neural_engine import _extract_candidate_compass_features_np

    F = eng.F
    with eng.torch.no_grad():
        t = eng._t
        tf = eng.torch.float32
        sid = int(req["slot_id"])
        st_s = eng._agent_states[sid]
        z_t = t(gwt_out["latent_z"][None, :], tf)
        sctx_t = t(gwt_out["slot_ctx64"][None, :], tf)
        mec_t = t(req["mec_grid_code"][None, :], tf)
        hd_t = t(req["hd_ring_code"][None, :], tf)
        comp_t = t(req["mem_compass"][None, :], tf)
        h_t = t(gwt_out["rssm_h"][None, :], tf)
        st_t = t(st_s["rssm_stoch"][None, :], tf)
        pr_t = t(req["proprio"][None, :], tf)
        haz_t = eng.models["hazard_net"](z_t, pr_t, comp_t)
        sh_list = list(st_s["strat_hist"])
        while len(sh_list) < 8:
            sh_list.insert(0, sh_list[0] if sh_list else np.zeros(160, dtype=np.float32))
        sseq_t = t(np.stack(sh_list[-8:], axis=0)[None, ...], tf)
        semb_t, _, _ = eng.models["strat_tf"](sseq_t)
        gm = eng.models["gwt"]
        sps = eng.torch.stack([
            F.gelu(gm.s0_vis(eng.torch.cat([z_t, sctx_t], dim=-1))),
            F.gelu(gm.s1_spa(eng.torch.cat([mec_t, hd_t, comp_t], dim=-1))),
            F.gelu(gm.s2_rssm(eng.torch.cat([h_t, st_t], dim=-1))),
            F.gelu(gm.s3_prop(eng.torch.cat([pr_t, haz_t.unsqueeze(-1)], dim=-1))),
            F.gelu(gm.s4_strat(semb_t)),
        ], dim=1)
        alp_sw = eng.torch.full((1, 5), 0.20, device=z_t.device)
        pooled_sw = eng.torch.einsum("bs,bsd->bd", alp_sw, sps)
        gw_sw = gm.broadcast_norm(0.50 * t(st_s["gwt_w"][None, :], tf) + 0.50 * pooled_sw)
        cf_t = t(_extract_candidate_compass_features_np(req["cand_poses"], req["cand_verbs"], req["proprio"], req["mem_compass"])[None, ...], tf)
        q_sw, spd_sw = eng.models["d3qn"](z_t, gw_sw, h_t, mec_t, hd_t, pr_t, comp_t, cf_t)
        q_raw_sw = q_sw[0].cpu().numpy()
        out = dict(gwt_out)
        out["q_raw_values"] = q_raw_sw
        out["q_values"] = 0.78 * q_raw_sw + 0.62 * out["pred_rew"] + 0.42 * (out["pred_adv_probs"] - 0.5) + 0.20 * out["proto_prior_bias"]
        out["speed_mod"] = spd_sw[0].cpu().numpy()
        out["gwt_workspace"] = gw_sw[0].cpu().numpy()
        st_s["gwt_w"] = out["gwt_workspace"].copy()
        return out


def _pump_gpu_for_workers(
    eng_a: Any,
    eng_b: Optional[Any],
    gpu_req_q: mp.Queue,
    gpu_resp_queues: Dict[int, mp.Queue],
    done_q: mp.Queue,
    expected_workers: int,
) -> List[Dict[str, Any]]:
    """
    Micro-batch GPU `infer_conscious_gwt_batch` across all active CPU arena workers on 1 GPU
    until all `expected_workers` report completion on `done_q`!
    """
    collected: List[Dict[str, Any]] = []
    finished_workers = 0

    while finished_workers < expected_workers:
        # Drain any completed worker batches
        while True:
            try:
                d_msg = done_q.get_nowait()
                collected.extend(d_msg["results"])
                finished_workers += 1
            except queue.Empty:
                break
        if finished_workers >= expected_workers:
            break

        # Gather 1..W pending GPU requests into a coalesced micro-batch (1.5ms window cuts CUDA launches by ~6x!)
        batch_msgs = []
        try:
            first = gpu_req_q.get(timeout=0.015)
            batch_msgs.append(first)
            coal_deadline = time.perf_counter() + 0.0018
            while len(batch_msgs) < expected_workers and time.perf_counter() < coal_deadline:
                rem = max(0.0002, coal_deadline - time.perf_counter())
                try:
                    batch_msgs.append(gpu_req_q.get(timeout=rem))
                except queue.Empty:
                    break
        except queue.Empty:
            continue

        # 1. Apply pending chosen-action notifications & transitions first
        for m in batch_msgs:
            for sid, cf in m["notifies_a"]:
                eng_a.notify_chosen_action(sid, cf)
            if eng_b is not None:
                for sid, cf in m["notifies_b"]:
                    eng_b.notify_chosen_action(sid, cf)
            for tr in m["trs_a"]:
                eng_a.record_transition_and_maybe_train(tr)

        # 2. Combined batched forward pass on `eng_a` and `eng_b`
        flat_a, slices_a = [], []
        flat_b, slices_b = [], []
        for m in batch_msgs:
            s0_a = len(flat_a); flat_a.extend(m["reqs_a"]); slices_a.append((s0_a, len(flat_a)))
            s0_b = len(flat_b); flat_b.extend(m["reqs_b"]); slices_b.append((s0_b, len(flat_b)))

        all_outs_a = eng_a.infer_conscious_gwt_batch(flat_a) if flat_a else []
        for idx_r, r_a in enumerate(flat_a):
            if r_a.get("gwt_pool_override") == "uniform":
                all_outs_a[idx_r] = _apply_s2_uniform_pool_override(eng_a, r_a, all_outs_a[idx_r])

        all_outs_b = eng_b.infer_conscious_gwt_batch(flat_b) if (flat_b and eng_b is not None) else []
        if eng_b is not None:
            for idx_r, r_b in enumerate(flat_b):
                if r_b.get("gwt_pool_override") == "uniform":
                    all_outs_b[idx_r] = _apply_s2_uniform_pool_override(eng_b, r_b, all_outs_b[idx_r])

        for idx_m, m in enumerate(batch_msgs):
            wid = int(m["wid"])
            sa0, sa1 = slices_a[idx_m]
            sb0, sb1 = slices_b[idx_m]
            gpu_resp_queues[wid].put({
                "outs_a": all_outs_a[sa0:sa1],
                "outs_b": all_outs_b[sb0:sb1],
            })

    collected.sort(key=lambda r: r["game_idx"])
    return collected


class ParallelArenaPool:
    """Reusable Zero-PyTorch CPU worker pool bound to the current single-GPU worker process."""

    def __init__(self, n_workers: int = 6):
        self.n_workers = max(1, int(n_workers))
        self.ctx = mp.get_context("spawn")
        self.task_queues = [self.ctx.Queue() for _ in range(self.n_workers)]
        self.gpu_req_q = self.ctx.Queue()
        self.gpu_resp_queues = {w: self.ctx.Queue() for w in range(self.n_workers)}
        self.done_q = self.ctx.Queue()
        self.procs = []
        for w in range(self.n_workers):
            p = self.ctx.Process(
                target=_cpu_arena_batch_worker,
                args=(w, self.task_queues[w], self.gpu_req_q, self.gpu_resp_queues[w], self.done_q),
                name=f"PhaseDCpuArena_{w}",
            )
            p.start()
            self.procs.append(p)

    def run_specs(self, specs: List[Dict[str, Any]], eng_a: Any, eng_b: Optional[Any] = None) -> List[Dict[str, Any]]:
        shards: List[List[Dict[str, Any]]] = [[] for _ in range(self.n_workers)]
        for i, sp in enumerate(specs):
            shards[i % self.n_workers].append(sp)
        active = 0
        for w in range(self.n_workers):
            if shards[w]:
                self.task_queues[w].put({"op": "run", "specs": shards[w]})
                active += 1
        return _pump_gpu_for_workers(eng_a, eng_b, self.gpu_req_q, self.gpu_resp_queues, self.done_q, active)

    def close(self) -> None:
        for q in self.task_queues:
            try:
                q.put({"op": "stop"})
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


def _extract_gate_telemetry(eng: Any) -> Dict[str, float]:
    t = eng.torch
    with t.no_grad():
        sc = eng.models["slot_cortex"]
        ta = eng.models["topo_adapter"]
        sm = eng.models["self_model"]
        st = eng.models["strat_tf"]
        return {
            "tanh_td_slot_gate": round(float(t.tanh(sc.td_slot_gate).item()), 5),
            "tanh_z_res_gate": round(float(t.tanh(sc.z_res_gate).item()), 5),
            "tanh_rssm_slot_gate": round(float(t.tanh(sc.rssm_slot_gate).item()), 5),
            "tanh_gwt_res_gate": round(float(t.tanh(ta.gwt_res_gate).item()), 5),
            "tanh_spot_gate": round(float(t.tanh(ta.spot_gate).item()), 5),
            "tanh_schema_gate": round(float(t.tanh(sm.schema_gate).item()), 5),
            "tanh_meta_err_gate": round(float(t.tanh(st.meta_err_gate).item()), 5),
            "rvq_prior_abs_max": round(float(np.max(np.abs(eng.rvq.sym_verb_prior))), 5),
            "rvq_prior_std": round(float(np.std(eng.rvq.sym_verb_prior)), 5),
        }


def _evaluate_matchup_battery(
    pool: ParallelArenaPool,
    *,
    label: str,
    eng_a: Any,
    cfg_a_template: Brain3DConfig,
    variant_a: str,
    eng_b: Optional[Any],
    cfg_b_template: Brain3DConfig,
    variant_b: str,
    stage_id: int = 1,
    n_seeds: int = 24,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run `2 * n_seeds` (`48` games for `n_seeds=24`) exact side-swapped matches across the CPU worker pool!
    """
    eng_a._agent_states.clear()
    if eng_b is not None:
        eng_b._agent_states.clear()

    seeds = [2000 + 97 * k for k in range(n_seeds)]
    specs: List[Dict[str, Any]] = []
    slot_a_id = 0
    slot_b_id = 500

    for s_i, a_seed in enumerate(seeds):
        for swapped in (False, True):
            g_idx = len(specs)
            ca = asdict(cfg_a_template)
            cb = asdict(cfg_b_template)
            ca["default_stage_id"] = int(stage_id)
            cb["default_stage_id"] = int(stage_id)
            ca["seed"] = 3000 + s_i
            cb["seed"] = 4000 + s_i

            eb_key = "B" if eng_b is not None else ("A" if variant_b not in ("L0_reactive", "L1_slam_map", "L2_homeo_scaffold") else "NONE")
            ea_key = "A" if variant_a not in ("L0_reactive", "L1_slam_map", "L2_homeo_scaffold") else "NONE"

            if not swapped:
                specs.append({
                    "game_idx": g_idx, "stage_id": int(stage_id), "arena_seed": a_seed,
                    "cfg_south": ca, "variant_south": variant_a, "eng_south": ea_key, "slot_id_south": slot_a_id,
                    "cfg_north": cb, "variant_north": variant_b, "eng_north": eb_key, "slot_id_north": slot_b_id,
                })
            else:
                specs.append({
                    "game_idx": g_idx, "stage_id": int(stage_id), "arena_seed": a_seed,
                    "cfg_south": cb, "variant_south": variant_b, "eng_south": eb_key, "slot_id_south": slot_b_id,
                    "cfg_north": ca, "variant_north": variant_a, "eng_north": ea_key, "slot_id_north": slot_a_id,
                })
            slot_a_id += 1
            slot_b_id += 1

    t0 = time.perf_counter()
    raw = pool.run_specs(specs, eng_a, eng_b)
    wall_s = time.perf_counter() - t0

    pair_win, pair_prog, pair_kills, pair_deaths, pair_turf, pair_shots, pair_stag = [], [], [], [], [], [], []
    contract_errs = 0
    all_workers_zero_torch = True

    for s_i in range(n_seeds):
        r1, r2 = raw[2 * s_i], raw[2 * s_i + 1]
        w1 = 1.0 if r1["winner"] > 0 else (0.0 if r1["winner"] < 0 else 0.5)
        w2 = 1.0 if r2["winner"] < 0 else (0.0 if r2["winner"] > 0 else 0.5)
        pair_win.append(0.5 * (w1 + w2))
        pair_prog.append(0.5 * (r1["max_p_s"] + r2["max_p_n"]) * 100.0)
        pair_kills.append(0.5 * (r1["deaths_n"] + r2["deaths_s"]))
        pair_deaths.append(0.5 * (r1["deaths_s"] + r2["deaths_n"]))
        pair_turf.append(0.5 * (r1["turf_s"] + r2["turf_n"]) * 100.0)
        pair_shots.append(0.5 * (r1["shots_s"] + r2["shots_n"]))
        pair_stag.append(0.5 * (r1["stag_s"] + r2["stag_n"]))
        contract_errs += int(r1["contract_errors"] + r2["contract_errors"])
        all_workers_zero_torch = all_workers_zero_torch and bool(r1["zero_torch_worker"]) and bool(r2["zero_torch_worker"])

    pc = paired_comparison(pair_win, [1.0 - x for x in pair_win], seed=2026)
    res = {
        "label": label,
        "stage_id": int(stage_id),
        "n_pairs": int(n_seeds),
        "n_games": int(2 * n_seeds),
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
        "stagnation_mean": round(float(np.mean(pair_stag)), 3),
        "contract_errors": int(contract_errs),
        "barrier1_workers_zero_torch": bool(all_workers_zero_torch),
        "wall_sec": round(wall_s, 2),
        "_pair_win": [round(float(x), 4) for x in pair_win],
    }
    if verbose:
        print(
            f"  [S{stage_id} | {label:38s}] WR={res['win_rate_pct']:5.1f}% ({res['w_l_t']}, p={res['wilcoxon_p']:.4f}) "
            f"prog={res['max_prog_pct']:5.1f}% K/D={res['kills_per_game']:.2f}/{res['deaths_per_game']:.2f} "
            f"turf={res['turf_share_pct']:4.1f}% errs={res['contract_errors']} ({wall_s:.1f}s)",
            flush=True,
        )
    return res


# ==============================================================================
# 3. Phase B Exact 9x9 Gradient Cosine Audit & Dual-Mode 10-Link Passenger Scan
# ==============================================================================
def _run_phase_b_gradient_and_passenger_audit(ckpt_path: str) -> Dict[str, Any]:
    from universal_brain_3d_core import WholeBrainNeuralEngineWrapper, CFEAT_HAZARD, CFEAT_IN_REFUGE
    from exp6_architecture_criteria import _onehot_st0

    cfg = make_eval_cfg(ckpt_path, stage_id=2, scaffold_scale=0.0, seed=2026)
    cfg.enable_learning = True
    eng = WholeBrainNeuralEngineWrapper(cfg)
    opp_cfg = make_eval_cfg(ckpt_path, stage_id=2, scaffold_scale=0.0, seed=2027)
    eng_opp = WholeBrainNeuralEngineWrapper(opp_cfg)

    # Collect fresh replay across Stage 1 (ep 0) and Stage 2 (ep 1) in Autonomous mode (scaf=0.0)!
    for ep_i, st_id in enumerate((1, 2)):
        aseed = 3100 + ep_i * 71
        arena = HeadToHeadTactical3DArena(Brain3DConfig(task_profile="turf_objective", default_stage_id=st_id, seed=aseed), seed=aseed)
        obs_s = arena._build_agent_obs(+1, 0.0, False)
        obs_n = canonicalize_team_observation(arena._build_agent_obs(-1, 0.0, False), -1, arena.wmin, arena.wmax)
        slot_s = FastEmbodiedSlot(cfg, slot_id=0, variant="L5_full", trainable=True)
        slot_n = FastEmbodiedSlot(opp_cfg, slot_id=1, variant="L5_full", trainable=False)

        for tick in range(1, 380):
            rs = slot_s.pre_step(obs_s, 1.0 / 60.0, cog_stride=4)
            rn = slot_n.pre_step(obs_n, 1.0 / 60.0, cog_stride=4)
            gs = eng.infer_conscious_gwt_batch([rs])[0] if rs is not None else None
            gn = eng_opp.infer_conscious_gwt_batch([rn])[0] if rn is not None else None
            not_s: List[Tuple[int, np.ndarray]] = []
            trs_s: List[Dict[str, Any]] = []
            act_s = slot_s.post_step(gs, beta_teacher=0.0, arena_seed=aseed, chosen_notify_list=not_s, transition_out_list=trs_s)
            act_n = slot_n.post_step(gn, beta_teacher=0.0, arena_seed=aseed)
            for sid, cf in not_s:
                eng.notify_chosen_action(sid, cf)
            for tr in trs_s:
                tr["enable_learning"] = False  # Record into replay without mutating weights
                if tick % 16 == 0:
                    tr["hazard"] = 0.55
                    tr["cand_feats32"][CAND_IDX_IDLE_OBSERVE, CFEAT_IN_REFUGE] = 0.0
                    tr["cand_feats32"][CAND_IDX_IDLE_OBSERVE, CFEAT_HAZARD] = 0.55
                eng.record_transition_and_maybe_train(tr)
            obs_s, obs_n, done, _ = arena.step_both(act_s, act_n, 1.0 / 60.0)
            if done:
                break

    torch = eng.torch
    F = eng.F
    shared_mods = ["foveal_enc", "slot_cortex", "topo_adapter", "hazard_net", "rssm", "gwt", "strat_tf", "self_model", "d3qn", "broca", "concept_grounder"]
    for mn in shared_mods:
        eng.models[mn].train()

    all_params, param_mod_names = [], []
    for mn in shared_mods:
        for p in eng.models[mn].parameters():
            if p.requires_grad:
                all_params.append(p)
                param_mod_names.append(mn)

    loss_names = [
        "L_td_d3qn", "L_speed", "L_rssm_world", "L_cf_shield", "L_vis_depth_haz",
        "L_slot_pred", "L_strat_tf", "L_self_agency", "L_lang_ground",
    ]
    n_batches = 8
    cos_accum = np.zeros((len(loss_names), len(loss_names)), dtype=np.float64)
    cos_counts = np.zeros((len(loss_names), len(loss_names)), dtype=np.float64)
    ov_cos_accum = np.zeros((len(loss_names), len(loss_names)), dtype=np.float64)
    ov_cos_counts = np.zeros((len(loss_names), len(loss_names)), dtype=np.float64)
    norm_accum = {k: [] for k in loss_names}
    mod_touch = {k: set() for k in loss_names}

    np.random.seed(2026)
    torch.manual_seed(2026)

    for _ in range(n_batches):
        _, batch, is_w = eng.replay.sample_prioritized(min(24, len(eng.replay)), alpha=0.60, beta=0.60)
        bsz = len(batch)
        is_w_t = eng._t(is_w, torch.float32)
        fov = eng._t(np.stack([b["fovea_patch"] for b in batch]), torch.float32)
        vis8 = eng._t(np.stack([compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch")) for b in batch]), torch.float32)
        prop = eng._t(np.stack([b["proprio"] for b in batch]), torch.float32)
        comp = eng._t(np.stack([b["mem_compass"] for b in batch]), torch.float32)
        mec = eng._t(np.stack([b["mec_grid_code"] for b in batch]), torch.float32)
        hd = eng._t(np.stack([b["hd_ring_code"] for b in batch]), torch.float32)
        c_feats = eng._t(np.stack([b["cand_feats32"] for b in batch]), torch.float32)

        prev_h_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_h", b["rssm_h"]) for b in batch]), torch.float32)
        prev_st_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_st", _onehot_st0(b)) for b in batch]), torch.float32)
        prev_act_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_act", np.zeros(CFEAT_DIM, dtype=np.float32)) for b in batch]), torch.float32)
        prev_gwt_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_gwt", b["gwt_workspace"]) for b in batch]), torch.float32)
        prev_alpha_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_alpha", np.full(5, 0.2, dtype=np.float32)) for b in batch]), torch.float32)
        prev_hab_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_hab", np.full(5, 0.2, dtype=np.float32)) for b in batch]), torch.float32)
        prev_z_td_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_z_td", np.zeros(LATENT_Z_DIM, dtype=np.float32)) for b in batch]), torch.float32)
        prev_comp_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_comp", b["mem_compass"]) for b in batch]), torch.float32)
        prev_self_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_self", np.zeros(SELF_MODEL_DIM, dtype=np.float32)) for b in batch]), torch.float32)
        prev_self_err_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_self_err", np.zeros(SELF_MODEL_DIM, dtype=np.float32)) for b in batch]), torch.float32)
        prev_ssurp_t = eng._t([float(b.get("_gpu_ctx", {}).get("prev_self_surp", b.get("self_surprise", 0.0))) for b in batch], torch.float32)
        prev_perr_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_perr_16", np.zeros((1, 16, 16), dtype=np.float32)) for b in batch]), torch.float32)
        prev_spot_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_spot_16", np.ones((1, 16, 16), dtype=np.float32)) for b in batch]), torch.float32)
        meta_err9_t = torch.cat([prev_self_err_t, prev_ssurp_t.unsqueeze(-1)], dim=-1)
        strat_seq_t = eng._t(np.stack([b.get("_gpu_ctx", {}).get("strat_seq", np.zeros((8, 160), dtype=np.float32)) for b in batch]), torch.float32)

        chosen_idx = eng._t([int(b["chosen_idx"]) for b in batch], torch.long)
        rewards_nstep = eng._t([float(b.get("nstep_return", b["reward"])) for b in batch], torch.float32)
        gammas_nstep = eng._t([float(b.get("nstep_gamma", 0.96)) for b in batch], torch.float32)
        dones_nstep = eng._t([1.0 if b.get("nstep_done", b.get("done") or b.get("respawn")) else 0.0 for b in batch], torch.float32)
        rewards_1step = eng._t([float(b.get("reward_1step", b.get("reward", 0.0))) for b in batch], torch.float32)
        rssm_rew_tgt = eng._t([float(b.get("rssm_rew_target", b.get("reward_1step", b.get("reward", 0.0)))) for b in batch], torch.float32)
        dones_1step = eng._t([1.0 if b.get("done_1step", b.get("done") or b.get("respawn")) else 0.0 for b in batch], torch.float32)
        haz_tgt = eng._t([float(b.get("next_hazard", 0.0)) for b in batch], torch.float32)
        surv_tgt = eng._t([0.05 if b.get("respawn") else float(np.clip(1.0 - 0.42 * float(b.get("next_hazard", 0.0)), 0.25, 0.99)) for b in batch], torch.float32)
        adv_tgt = eng._t([1.0 if float(b.get("obj_adv_delta", 0.0)) > 0.002 or float(b.get("reward_1step", b["reward"])) > 0.05 else 0.35 for b in batch], torch.float32)

        ns1_list = [b.get("next_state_1step", b.get("next_state")) for b in batch]
        nvis8_1 = eng._t(np.stack([compose_vis8_patch(ns["fovea_patch"], ns.get("ventral_patch")) if ns else compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch")) for ns, b in zip(ns1_list, batch)]), torch.float32)
        nsn_list = [b.get("next_state_nstep", b.get("next_state")) for b in batch]
        nfov_n = eng._t(np.stack([ns["fovea_patch"] if ns else b["fovea_patch"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        nvis8_n = eng._t(np.stack([compose_vis8_patch(ns["fovea_patch"], ns.get("ventral_patch")) if ns else compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch")) for ns, b in zip(nsn_list, batch)]), torch.float32)
        nprop_n = eng._t(np.stack([ns["proprio"] if ns else b["proprio"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        ncomp_n = eng._t(np.stack([ns["mem_compass"] if ns else b["mem_compass"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        nmec_n = eng._t(np.stack([ns["mec_grid_code"] if ns else b["mec_grid_code"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        nhd_n = eng._t(np.stack([ns["hd_ring_code"] if ns else b["hd_ring_code"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        ncf_n = eng._t(np.stack([ns["cand_feats32"] if ns else b["cand_feats32"] for ns, b in zip(nsn_list, batch)]), torch.float32)

        eng.optimizer.zero_grad()
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
        # Exact Phase B World-Model Encoder Ownership (`z_perc.detach()`, `slot_ctx64.detach()`):
        gwt_base, alp_live, ign_live, _ = eng.models["gwt"](
            z_perc.detach(), slot_ctx64.detach(), mec, hd, comp, h_post.detach(), post_st.detach(), prop, haz_pred.detach(), strat_emb,
            prev_gwt_t, meta_salience=prev_ssurp_t, hab_trace=prev_hab_t, novelty_5=nov5_tr,
        )
        delta_gwt, topdown_gain_16 = eng.models["topo_adapter"].compute_topdown_spotlight(
            gwt_base, slot_aux["feat16"].detach(), slots.detach(), s_coords.detach()
        )
        gwt_live = gwt_base + delta_gwt
        z_td = (0.08 + 0.24 * ign_live).unsqueeze(-1) * eng.models["gwt"].topdown_z(gwt_live)
        self_cur, _, _, self_pred = eng.models["self_model"](gwt_live.detach(), prop, prev_self_t, gwt_alpha=alp_live.detach(), prev_act=prev_act_t, prev_alpha=prev_alpha_t)

        z_eff = torch.tanh(z_perc.detach() + 0.70 * z_td)
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
            nstrat_emb_n, _, _ = eng.models["strat_tf"](n_strat_seq)
            ngwt_base_n, _, nign_n, _ = eng.models["gwt"](nz_perc_n, nslot_ctx64_n, nmec_n, nhd_n, ncomp_n, nh_next_n, npost_st_n, nprop_n, nhaz_n, nstrat_emb_n, gwt_live)
            ndelta_gwt_n, _ = eng.models["topo_adapter"].compute_topdown_spotlight(ngwt_base_n, nslot_aux_n["feat16"], nslots_n, ns_coords_n)
            ngwt_n = ngwt_base_n + ndelta_gwt_n
            nz_td_n = (0.08 + 0.24 * nign_n).unsqueeze(-1) * eng.models["gwt"].topdown_z(ngwt_n)
            nz_eff_n = torch.tanh(nz_perc_n + 0.70 * nz_td_n)
            nq_online, _ = eng.models["d3qn"](nz_eff_n, ngwt_n, nh_next_n, nmec_n, nhd_n, nprop_n, ncomp_n, ncf_n)
            best_next = torch.argmax(nq_online, dim=1, keepdim=True)
            nq_target, _ = eng.models["d3qn_target"](nz_eff_n, ngwt_n, nh_next_n, nmec_n, nhd_n, nprop_n, ncomp_n, ncf_n)
            v_next = nq_target.gather(1, best_next).squeeze(1)
            td_target = (rewards_nstep + gammas_nstep * (1.0 - dones_nstep) * v_next).clamp(-35.0, 35.0)

        td_elem = F.smooth_l1_loss(q_chosen, td_target, reduction="none")
        loss_td = (td_elem * is_w_t).mean()
        spd_chosen = spd_all.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
        spd_tgt = torch.where(haz_tgt > 0.45, torch.full_like(spd_chosen, 0.92), torch.full_like(spd_chosen, 0.80))
        loss_speed = 0.08 * F.mse_loss(spd_chosen, spd_tgt)

        pr_c = p_rew.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
        ps_c = p_surv.gather(1, chosen_idx.unsqueeze(1)).squeeze(1).clamp(1e-4, 1.0 - 1e-4)
        pa_c = p_adv.gather(1, chosen_idx.unsqueeze(1)).squeeze(1).clamp(1e-4, 1.0 - 1e-4)
        kl_d = (p_post.detach() * (torch.log(p_post.detach() + 1e-7) - torch.log(p_prior + 1e-7))).sum(dim=(-1, -2))
        kl_r = (p_post * (torch.log(p_post + 1e-7) - torch.log(p_prior.detach() + 1e-7))).sum(dim=(-1, -2))
        loss_rssm = 0.5 * F.smooth_l1_loss(pr_c, rssm_rew_tgt) + 0.6 * F.binary_cross_entropy(ps_c, surv_tgt) + 0.4 * F.binary_cross_entropy(pa_c, adv_tgt) + 0.10 * (0.8 * torch.clamp(kl_d, min=1.0).mean() + 0.2 * torch.clamp(kl_r, min=1.0).mean())

        exposed_mask = (prop[:, PROPRIO_IDX_HAZARD] >= 0.35) & (prop[:, PROPRIO_IDX_SURFACE_POSTURE] < 0.85)
        if bool(exposed_mask.any().item()):
            _, p_surv_cf, _ = eng.models["rssm"].imagine_candidates(h_post[exposed_mask].detach(), post_st[exposed_mask].detach(), c_feats[exposed_mask], horizon=3)
            cf_surv_m7 = p_surv_cf[:, CAND_IDX_IDLE_OBSERVE].clamp(1e-4, 1.0 - 1e-4)
            q_cf_exp, _ = eng.models["d3qn"](z_eff[exposed_mask].detach(), gwt_live[exposed_mask].detach(), h_post[exposed_mask].detach(), mec[exposed_mask], hd[exposed_mask], prop[exposed_mask], comp[exposed_mask], c_feats[exposed_mask])
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

        slots_vis_in = 0.25 * slots + 0.75 * slots.detach()
        pred_nvis8, _ = eng.models["slot_cortex"].predict_next_visual_field(slots_vis_in, slot_aux["pixel_stats"].detach(), s_coords.detach(), chosen_cf, rssm_h=h_post.detach())
        pred_nslots = eng.models["slot_cortex"].predict_next_slots(slots.detach(), chosen_cf, rssm_h=h_post.detach())
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
        cand_to_mode = eng._t(CAND_TO_GOAL_MODE_TUPLE, torch.long)
        chosen_mode = cand_to_mode[chosen_idx]
        td_adv_pos = F.relu(td_target.detach() - q_chosen.detach()).clamp(0.0, 3.0).unsqueeze(1)
        mode_q = mode_q.scatter_add(1, chosen_mode.unsqueeze(1), td_adv_pos)
        mode_tgt_probs = F.softmax(mode_q / 0.45, dim=-1)
        mode_log_probs = F.log_softmax(mode_logits, dim=-1)
        sp4_live = F.gelu(eng.models["gwt"].s4_strat(strat_emb))
        ws_sp4_logp = F.log_softmax(eng.models["gwt"].ws_mode_head(sp4_live), dim=-1)
        ws_gwt_logp = F.log_softmax(eng.models["gwt"].ws_mode_head(gwt_live), dim=-1)
        focus_tgt = torch.where(
            comp[:, MC_THR_VALID : MC_THR_VALID + 1] > 0.5,
            4.0 * comp[:, MC_THR_DX : MC_THR_DZ + 1],
            torch.where(comp[:, MC_OBJ_VALID : MC_OBJ_VALID + 1] > 0.5, 4.0 * comp[:, MC_OBJ_DX : MC_OBJ_DZ + 1], 3.5 * comp[:, MC_FRONT_DX : MC_FRONT_DZ + 1]),
        )
        loss_strat = (
            -(mode_tgt_probs * mode_log_probs).sum(dim=-1).mean()
            - 0.22 * (mode_tgt_probs * ws_sp4_logp).sum(dim=-1).mean()
            - 0.18 * (mode_tgt_probs * ws_gwt_logp).sum(dim=-1).mean()
            + 0.15 * F.smooth_l1_loss(focus_off_xz, focus_tgt)
        )

        self_tgt8 = torch.clamp(torch.stack([
            2.0 * surv_tgt - 1.0, 1.0 - 2.0 * haz_tgt, 2.0 * prop[:, PROPRIO_IDX_RESOURCE_CARRIER] - 1.0,
            2.0 * prop[:, PROPRIO_IDX_SURFACE_POSTURE] - 1.0, 2.0 * prop[:, PROPRIO_IDX_CAN_COMMIT] - 1.0,
            2.0 * prop[:, PROPRIO_IDX_OBJ_PROGRESS] - 1.0, torch.tanh(q_chosen.detach() / 8.0), torch.tanh(kl_r.detach() - 0.5),
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
            for g, p, mn in zip(grads, all_params, param_mod_names):
                if g is not None and float(g.abs().max().item()) > 1e-12:
                    mod_touch[loss_names[li]].add(mn)
                    p_list.append(g.detach().reshape(-1))
                else:
                    p_list.append(None)
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
    off_diag_ov = [
        float(ov_cos_mat[i, j])
        for i in range(len(loss_names))
        for j in range(len(loss_names))
        if i != j and ov_cos_counts[i, j] > 0
    ]

    snapshots = [dict(eng.replay[k]) for k in range(len(eng.replay))]
    for mn in shared_mods:
        eng.models[mn].eval()

    # Run 10-Link Silent-Passenger Scan under BOTH `scaf = 1.0` AND `Autonomous (scaf = 0.0)` using exact v2.3 fusion!
    def _scan_10_links(scaf_scale: float) -> Dict[str, Any]:
        cfg_scan = make_eval_cfg(ckpt_path, stage_id=2, scaffold_scale=scaf_scale, seed=2026)
        bsz_s = len(snapshots)
        fov_s = eng._t(np.stack([b["fovea_patch"] for b in snapshots]), torch.float32)
        vis8_s = eng._t(np.stack([compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch")) for b in snapshots]), torch.float32)
        prop_s = eng._t(np.stack([b["proprio"] for b in snapshots]), torch.float32)
        comp_s = eng._t(np.stack([b["mem_compass"] for b in snapshots]), torch.float32)
        mec_s = eng._t(np.stack([b["mec_grid_code"] for b in snapshots]), torch.float32)
        hd_s = eng._t(np.stack([b["hd_ring_code"] for b in snapshots]), torch.float32)
        cf_s = eng._t(np.stack([b["cand_feats32"] for b in snapshots]), torch.float32)
        prev_h_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_h", b["rssm_h"]) for b in snapshots]), torch.float32)
        prev_st_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_st", _onehot_st0(b)) for b in snapshots]), torch.float32)
        prev_act_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_act", np.zeros(CFEAT_DIM, dtype=np.float32)) for b in snapshots]), torch.float32)
        prev_gwt_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_gwt", b["gwt_workspace"]) for b in snapshots]), torch.float32)
        prev_hab_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_hab", np.full(5, 0.2, dtype=np.float32)) for b in snapshots]), torch.float32)
        prev_ztd_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_z_td", np.zeros(LATENT_Z_DIM, dtype=np.float32)) for b in snapshots]), torch.float32)
        prev_comp_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_comp", b["mem_compass"]) for b in snapshots]), torch.float32)
        prev_serr_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_self_err", np.zeros(8, dtype=np.float32)) for b in snapshots]), torch.float32)
        prev_ssurp_s = eng._t([float(b.get("_gpu_ctx", {}).get("prev_self_surp", 0.0)) for b in snapshots], torch.float32)
        meta_err9_s = torch.cat([prev_serr_s, prev_ssurp_s.unsqueeze(-1)], dim=-1)
        prev_perr_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_perr_16", np.zeros((1, 16, 16), dtype=np.float32)) for b in snapshots]), torch.float32)
        prev_spot_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("prev_spot_16", np.ones((1, 16, 16), dtype=np.float32)) for b in snapshots]), torch.float32)
        strat_seq_s = eng._t(np.stack([b.get("_gpu_ctx", {}).get("strat_seq", np.zeros((8, 160), dtype=np.float32)) for b in snapshots]), torch.float32)

        def _mc(x):
            return x.mean(dim=0, keepdim=True).expand_as(x)

        def _eval_clamp(ck: str):
            with torch.no_grad():
                z = eng.models["foveal_enc"](fov_s, prop_s, comp_s)
                eff_spot = _mc(prev_spot_s) if ck == "L3_topdown_t1_to_t" else prev_spot_s
                eff_ztd = _mc(prev_ztd_s) if ck == "L3_topdown_t1_to_t" else prev_ztd_s
                sl, sco, _, saux = eng.models["slot_cortex"](
                    vis8_s, pred_err_16=prev_perr_s, topdown_gain_16=eff_spot,
                    td_context=torch.cat([prev_gwt_s, prev_h_s], dim=-1), return_aux=True,
                )
                dz = _mc(saux["delta_z"]) if ck == "L2_slot_cortex" else saux["delta_z"]
                zp = torch.tanh((_mc(z) if ck == "L1_foveal_z" else z) + dz + 0.30 * eff_ztd)
                eff_mec = _mc(mec_s) if ck == "L4_mec_hd_3d" else mec_s
                eff_hd = _mc(hd_s) if ck == "L4_mec_hd_3d" else hd_s
                eff_comp = _mc(comp_s) if ck == "L5_mem_compass_24d" else comp_s
                eff_cf = cf_s.clone()
                if ck == "L5_mem_compass_24d":
                    eff_cf[:, :, 14:] = _mc(eff_cf[:, :, 14:])
                haz_p = eng.models["hazard_net"](zp, prop_s, eff_comp)
                sctx = eng.models["topo_adapter"](sl, eff_mec, eff_hd, eff_comp)
                if ck == "L2_slot_cortex":
                    sctx = _mc(sctx)
                hn, pst, pp, ppr = eng.models["rssm"].observe_step(prev_h_s, prev_st_s, prev_act_s, zp, sctx, prop_s, eff_comp)
                prw, psv, pad = eng.models["rssm"].imagine_candidates(hn, pst, eff_cf, horizon=3)
                eff_h = _mc(hn) if ck == "L6_rssm_recurrent_state" else hn
                eff_pst = _mc(pst) if ck == "L6_rssm_recurrent_state" else pst
                kls = (pp * (torch.log(pp + 1e-7) - torch.log(ppr + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
                cfs = 0.35 * prw + 0.40 * psv + 0.25 * pad
                nov5 = torch.stack([
                    prev_perr_s.mean(dim=(1, 2, 3)) * 2.4,
                    torch.clamp((eff_comp - prev_comp_s).norm(dim=-1) * 0.95, 0.0, 2.0),
                    kls * 1.35 + 0.65 * (cfs.max(dim=-1).values - cfs.min(dim=-1).values),
                    haz_p * 1.15 + 0.45 * torch.clamp(0.35 - prop_s[:, PROPRIO_IDX_RESOURCE_CARRIER], min=0.0),
                    prev_ssurp_s * 1.80,
                ], dim=-1)
                eff_meta = _mc(meta_err9_s) if ck == "L10_self_and_rvq" else meta_err9_s
                eff_ss = _mc(prev_ssurp_s) if ck == "L10_self_and_rvq" else prev_ssurp_s
                semb, _, _ = eng.models["strat_tf"](strat_seq_s, meta_err9=eff_meta)
                if ck == "L9_strategic_tf":
                    semb = _mc(semb)
                gb, _, ign, _ = eng.models["gwt"](
                    zp, sctx, eff_mec, eff_hd, eff_comp, eff_h, eff_pst, prop_s, haz_p, semb,
                    prev_gwt_s, meta_salience=eff_ss, hab_trace=prev_hab_s, novelty_5=nov5,
                )
                dgw, _ = eng.models["topo_adapter"].compute_topdown_spotlight(gb, saux["feat16"], sl, sco)
                gw = gb + dgw
                if ck == "L8_gwt_broadcast":
                    gw = _mc(gw)
                ztd = (0.08 + 0.24 * ign).unsqueeze(-1) * eng.models["gwt"].topdown_z(gw)
                if ck == "L3_topdown_t1_to_t":
                    ztd = _mc(ztd)
                ze = torch.tanh(zp + 0.70 * ztd)
                qr, _ = eng.models["d3qn"](ze, gw, eff_h, eff_mec, eff_hd, prop_s, eff_comp, eff_cf)
                q_np, r_np, s_np, a_np = qr.cpu().numpy(), prw.cpu().numpy(), psv.cpu().numpy(), pad.cpu().numpy()
                gw_np, h_np = gw.cpu().numpy(), eff_h.cpu().numpy()
                if ck == "L7_rssm_3head_imagination":
                    r_np = np.broadcast_to(r_np.mean(axis=0, keepdims=True), r_np.shape).copy()
                    s_np = np.broadcast_to(s_np.mean(axis=0, keepdims=True), s_np.shape).copy()
                    a_np = np.broadcast_to(a_np.mean(axis=0, keepdims=True), a_np.shape).copy()
                q_stars, pis = [], []
                c_verbs = np.array([1, 4, 4, 5, 5, 4, 1, 0], dtype=np.uint8)
                for k in range(bsz_s):
                    _, _, _, pb = eng.rvq.quantize(gw_np[k], np.zeros(8, dtype=np.float32), h_np[k], c_verbs, update_ema=False)
                    if ck == "L10_self_and_rvq":
                        pb = np.zeros_like(pb)
                    qf = 0.78 * q_np[k] + 0.62 * r_np[k] + 0.42 * (a_np[k] - 0.5) + 0.20 * pb
                    ss_k = 0.03 if ck == "L10_self_and_rvq" else float(snapshots[k].get("self_surprise", 0.04))
                    qs, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                        qf, s_np[k], a_np[k], gw_np[k], snapshots[k]["cand_feats32"][:, :6], c_verbs,
                        snapshots[k]["proprio"], eff_comp[k].cpu().numpy(), None, cfg_scan,
                        q_raw_values=q_np[k], self_surprise=ss_k,
                    )
                    q_stars.append(qs); pis.append(pi)
                return np.stack(q_stars), np.stack(pis)

        q_full, pi_full = _eval_clamp("NONE")
        top_full = np.argmax(q_full, axis=-1)
        links = [
            "L1_foveal_z", "L2_slot_cortex", "L3_topdown_t1_to_t", "L4_mec_hd_3d", "L5_mem_compass_24d",
            "L6_rssm_recurrent_state", "L7_rssm_3head_imagination", "L8_gwt_broadcast", "L9_strategic_tf", "L10_self_and_rvq",
        ]
        out_links = {}
        silent_cnt = 0
        ln2 = math.log(2.0)
        for lk in links:
            qc, pic = _eval_clamp(lk)
            flip_pct = float(np.mean(np.argmax(qc, axis=-1) != top_full) * 100.0)
            max_dq = float(np.mean(np.max(np.abs(q_full - qc), axis=-1)))
            p1, p2 = np.clip(pi_full, 1e-7, 1.0), np.clip(pic, 1e-7, 1.0)
            kl_b = float(np.mean(np.sum(p1 * (np.log(p1) - np.log(p2)), axis=-1)) / ln2)
            if flip_pct < 1.5 and kl_b < 0.01:
                silent_cnt += 1
            out_links[lk] = {
                "action_flip_pct": round(flip_pct, 2),
                "mean_max_abs_dq": round(max_dq, 4),
                "policy_kl_bits": round(kl_b, 5),
            }
        return {"n_states": bsz_s, "silent_count": silent_cnt, "links": out_links}

    pass_scaf1 = _scan_10_links(1.0)
    pass_auto0 = _scan_10_links(0.0)
    eng.release_memory_and_vram()
    eng_opp.release_memory_and_vram()

    td_rssm_ov = ov_cos_mat[0, 2] if ov_cos_counts[0, 2] > 0 else None
    return {
        "gradient_audit": {
            "loss_names": loss_names,
            "mean_grad_l2_norm": {k: round(float(np.mean(v)), 5) for k, v in norm_accum.items()},
            "modules_touched_by_loss": {k: sorted(list(v)) for k, v in mod_touch.items()},
            "min_shared_overlap_offdiag_cos": round(min(off_diag_ov) if off_diag_ov else 0.0, 4),
            "max_shared_overlap_offdiag_cos": round(max(off_diag_ov) if off_diag_ov else 0.0, 4),
            "td_vs_rssm_shared_overlap_cos": round(float(td_rssm_ov), 4) if td_rssm_ov is not None else None,
            "pairwise_grad_cosine_full": [[round(float(cos_mat[i, j]), 4) for j in range(len(loss_names))] for i in range(len(loss_names))],
            "pairwise_grad_cosine_shared_overlap": [
                [round(float(ov_cos_mat[i, j]), 4) if ov_cos_counts[i, j] > 0 else None for j in range(len(loss_names))]
                for i in range(len(loss_names))
            ],
        },
        "silent_passenger_scaf_1_0": pass_scaf1,
        "silent_passenger_autonomous_0_0": pass_auto0,
    }


# ==============================================================================
# 4. Single-GPU Worker: Progressive Scaffold Annealing Training & Full Benchmark
# ==============================================================================
def _gpu_phase_d_worker(
    seeds: List[int],
    generations: int,
    arenas_per_gen: int,
    sleep_epochs: int,
    online_interval: int,
    n_eval_seeds: int,
    n_cpu_workers: int,
    v22_ckpt: str,
    study_dir: str,
    out_json_path: str,
    skip_train: bool,
    res_q: mp.Queue,
) -> None:
    from universal_brain_3d_core import WholeBrainNeuralEngineWrapper, PrioritizedReplayRingBuffer

    t_start = time.perf_counter()
    partial_path = out_json_path + ".partial.json"
    saved_state: Dict[str, Any] = {}
    if os.path.isfile(partial_path):
        try:
            with open(partial_path, "r", encoding="utf-8") as fh:
                saved_state = json.load(fh)
        except Exception:
            saved_state = {}

    def _flush_partial() -> None:
        os.makedirs(os.path.dirname(partial_path), exist_ok=True)
        with open(partial_path, "w", encoding="utf-8") as fh:
            json.dump(saved_state, fh, indent=2, ensure_ascii=False)

    pool = ParallelArenaPool(n_workers=n_cpu_workers)
    try:
        per_seed_train: Dict[str, Any] = dict(saved_state.get("per_seed_training", {}))
        trained_ckpts: Dict[int, str] = {}

        for sd in seeds:
            sd_dir = os.path.join(study_dir, f"seed_{sd}")
            os.makedirs(sd_dir, exist_ok=True)
            best_ckpt = os.path.join(sd_dir, "pbt_best_champion.pt")
            latest_ckpt = os.path.join(sd_dir, "pbt_latest_champion.pt")

            if (skip_train or str(sd) in per_seed_train) and os.path.isfile(best_ckpt):
                trained_ckpts[sd] = best_ckpt
                if str(sd) not in per_seed_train:
                    meta_b = load_checkpoint_contract_meta(best_ckpt)
                    per_seed_train[str(sd)] = {"best_ckpt": os.path.relpath(best_ckpt, ROOT_DIR).replace("\\", "/"), "resumed_existing": True, "meta": meta_b}
                print(f"[Resume] Reusing completed checkpoint for seed {sd}: {best_ckpt}", flush=True)
                continue

            print(f"\n{'=' * 84}\n  [Train Seed {sd}] Mixed-Stage (S1+S2) + Progressive Scaffold Annealing (1.0 -> 0.0)\n{'=' * 84}", flush=True)
            t_sd = time.perf_counter()
            cfg_learn = Brain3DConfig(
                task_profile="turf_objective",
                policy_mode="pure_neural",
                online_update_interval=max(1, int(online_interval)),
                enable_overfit_guard=False,
                continual_anchor_l2=0.0,
                seed=int(sd),
            )
            learner = WholeBrainNeuralEngineWrapper(cfg_learn)
            learner.max_replay = 8192
            learner.replay = PrioritizedReplayRingBuffer(capacity=8192, on_clear_cb=learner._clear_temporal_queues)

            opp_cfg = Brain3DConfig(task_profile="turf_objective", policy_mode="pure_neural", enable_learning=False, seed=int(sd) + 500)
            opp_eng = WholeBrainNeuralEngineWrapper(opp_cfg)

            league_pool: List[Tuple[str, Dict[str, Any], float]] = [
                ("snap_gen0_untrained", learner.get_weights_state_dict(as_numpy=True), 1.0)
            ]
            if v22_ckpt and os.path.isfile(v22_ckpt):
                opp_eng.load_checkpoint(v22_ckpt)
                league_pool.append(("v22_champion_ref", opp_eng.get_weights_state_dict(as_numpy=True), 0.5))

            rng = np.random.default_rng(int(sd) + 99)
            best_val_metric = -1e9
            gen_logs = []

            for gen in range(1, int(generations) + 1):
                gt0 = time.perf_counter()
                # Smooth DAgger warm-start -> Pure Neural + Progressive Scaffold Annealing (1.00 -> 0.00)!
                # Generations 6..generations operate at 100% Autonomous `scaffold_prior_scale == 0.00`!
                if gen == 1:
                    beta_t, scaf_s = 0.55, 1.00
                elif gen == 2:
                    beta_t, scaf_s = 0.18, 0.80
                elif gen == 3:
                    beta_t, scaf_s = 0.00, 0.55
                elif gen == 4:
                    beta_t, scaf_s = 0.00, 0.35
                elif gen == 5:
                    beta_t, scaf_s = 0.00, 0.15
                else:
                    beta_t, scaf_s = 0.00, 0.00

                cfg_learn.scaffold_prior_scale = scaf_s
                pick_idx = int(rng.integers(0, len(league_pool)))
                opp_name, opp_state, opp_scaf = league_pool[pick_idx]
                opp_eng.load_weights_state_dict(opp_state)
                opp_eng._agent_states.clear()
                learner._agent_states.clear()

                specs = []
                for k in range(int(arenas_per_gen)):
                    st_id = 1 if (k % 2 == 0) else 2
                    a_seed = int(sd) * 13 + gen * 1009 + k * 97
                    cs_d = asdict(cfg_learn)
                    cs_d["default_stage_id"] = st_id
                    cs_d["scaffold_prior_scale"] = scaf_s

                    cn_d = asdict(opp_cfg)
                    cn_d["default_stage_id"] = st_id

                    if k < max(1, int(arenas_per_gen) // 2):
                        cn_d["scaffold_prior_scale"] = opp_scaf
                        v_n, e_n, sid_n = "L5_full", "B", 100 + k
                    elif k < int(arenas_per_gen) - 2:
                        cn_d["scaffold_prior_scale"] = scaf_s
                        v_n, e_n, sid_n = "L5_full", "A", 200 + k
                    else:
                        cn_d["scaffold_prior_scale"] = 1.0
                        v_n, e_n, sid_n = "L2_homeo_scaffold", "NONE", 300 + k

                    specs.append({
                        "game_idx": k, "episode_id": gen * 100 + k + 1,
                        "stage_id": st_id, "arena_seed": a_seed,
                        "cfg_south": cs_d, "variant_south": "L5_full", "eng_south": "A", "slot_id_south": k,
                        "trainable_south": True, "beta_south": beta_t,
                        "cfg_north": cn_d, "variant_north": v_n, "eng_north": e_n, "slot_id_north": sid_n,
                    })

                roll_res = pool.run_specs(specs, learner, opp_eng)
                sleep_res = learner.run_sleep_consolidation(epochs=max(4, int(sleep_epochs)))
                league_pool.append((f"snap_gen{gen}", learner.get_weights_state_dict(as_numpy=True), scaf_s))
                if len(league_pool) > 8:
                    league_pool = league_pool[:2] + league_pool[-6:]

                gen_wr = float(np.mean([1.0 if r["winner"] > 0 else (0.0 if r["winner"] < 0 else 0.5) for r in roll_res])) * 100.0
                gates_now = _extract_gate_telemetry(learner)
                learner.save_checkpoint(latest_ckpt, extra_meta={"v23_gen": gen, "scaffold_prior_scale": scaf_s})

                val_auto_s1, val_auto_s2 = None, None
                if gen >= 6 and (gen % 2 == 0 or gen == int(generations)):
                    c_auto = make_eval_cfg(latest_ckpt, stage_id=1, scaffold_scale=0.0, seed=sd)
                    c_l2 = make_eval_cfg("", stage_id=1, scaffold_scale=1.0, seed=sd + 1)
                    v1 = _evaluate_matchup_battery(
                        pool, label="val_auto_s1", eng_a=learner, cfg_a_template=c_auto, variant_a="L5_full",
                        eng_b=None, cfg_b_template=c_l2, variant_b="L2_homeo_scaffold", stage_id=1, n_seeds=8, verbose=False,
                    )
                    v2 = _evaluate_matchup_battery(
                        pool, label="val_auto_s2", eng_a=learner, cfg_a_template=c_auto, variant_a="L5_full",
                        eng_b=None, cfg_b_template=c_l2, variant_b="L2_homeo_scaffold", stage_id=2, n_seeds=8, verbose=False,
                    )
                    val_auto_s1 = v1["win_rate_pct"]
                    val_auto_s2 = v2["win_rate_pct"]
                    val_metric = 0.45 * val_auto_s1 + 0.55 * val_auto_s2 + 0.15 * (v1["max_prog_pct"] + v2["max_prog_pct"]) * 0.5
                    if scaf_s <= 0.05:
                        val_metric += 5.0
                    if val_metric >= best_val_metric:
                        best_val_metric = val_metric
                        learner.save_checkpoint(
                            best_ckpt,
                            extra_meta={
                                "v23_gen": gen, "scaffold_prior_scale": scaf_s,
                                "val_auto_s1_wr_pct": val_auto_s1, "val_auto_s2_wr_pct": val_auto_s2,
                            },
                        )

                g_sec = time.perf_counter() - gt0
                gen_logs.append({
                    "gen": gen, "beta_teacher": beta_t, "scaffold_prior_scale": scaf_s,
                    "train_steps_done": int(learner.train_steps_done),
                    "transitions_recorded": int(learner.transitions_recorded),
                    "nan_rollbacks": int(learner.nan_rollback_count),
                    "gen_train_wr_pct": round(gen_wr, 1),
                    "val_auto_s1_wr_pct": val_auto_s1,
                    "val_auto_s2_wr_pct": val_auto_s2,
                    "mean_sleep_loss": round(float(sleep_res["mean_loss"]), 4),
                    "gates": gates_now,
                    "sec": round(g_sec, 2),
                })
                v_str = f"S1={val_auto_s1:.1f}%/S2={val_auto_s2:.1f}%" if val_auto_s1 is not None else "       -       "
                print(
                    f"[Seed {sd} Gen {gen:02d}/{generations:02d}] scaf={scaf_s:.2f} steps={learner.train_steps_done:4d} "
                    f"trWR={gen_wr:5.1f}% AutoVal({v_str}) "
                    f"gates(td={gates_now['tanh_td_slot_gate']:+.3f}, z={gates_now['tanh_z_res_gate']:+.3f}, "
                    f"gwt={gates_now['tanh_gwt_res_gate']:+.3f}, meta={gates_now['tanh_meta_err_gate']:+.3f}, "
                    f"rvq={gates_now['rvq_prior_abs_max']:.3f}) ({g_sec:.1f}s)",
                    flush=True,
                )

            if not os.path.isfile(best_ckpt):
                learner.save_checkpoint(best_ckpt, extra_meta={"v23_gen": int(generations), "scaffold_prior_scale": 0.0})
            vram_mb = round(float(learner.torch.cuda.max_memory_allocated(learner.device)) / (1024.0 * 1024.0), 2) if learner.device.type == "cuda" else 0.0
            per_seed_train[str(sd)] = {
                "best_ckpt": os.path.relpath(best_ckpt, ROOT_DIR).replace("\\", "/"),
                "train_steps_done": int(learner.train_steps_done),
                "transitions_recorded": int(learner.transitions_recorded),
                "sleep_cycles_done": int(learner.sleep_cycles_done),
                "nan_rollback_count": int(learner.nan_rollback_count),
                "vram_peak_mb": vram_mb,
                "final_gates": _extract_gate_telemetry(learner),
                "wall_clock_sec": round(time.perf_counter() - t_sd, 2),
                "generation_logs": gen_logs,
            }
            trained_ckpts[sd] = best_ckpt
            saved_state["per_seed_training"] = per_seed_train
            _flush_partial()
            learner.release_memory_and_vram()
            opp_eng.release_memory_and_vram()

        # ======================================================================
        # Phase D Falsifiable Integration Audit + Multi-Seed Stage 1 & 2 Battery
        # ======================================================================
        primary_ckpt = trained_ckpts[seeds[0]]
        if "integration_audit_primary_seed" in saved_state:
            integ_audit = saved_state["integration_audit_primary_seed"]
            print("[Resume] Loaded completed integration_audit_primary_seed from partial JSON.", flush=True)
        else:
            print(f"\n{'=' * 84}\n  [Audit] Phase B 9x9 Gradient Cosine & Dual-Mode 10-Link Silent-Passenger Scan\n{'=' * 84}", flush=True)
            integ_audit = _run_phase_b_gradient_and_passenger_audit(primary_ckpt)
            saved_state["integration_audit_primary_seed"] = integ_audit
            _flush_partial()
        ga = integ_audit["gradient_audit"]
        print(
            f"  [Gradient Overlap] min_offdiag_cos={ga['min_shared_overlap_offdiag_cos']:+.4f} | "
            f"L_td x L_rssm overlap={ga['td_vs_rssm_shared_overlap_cos']} | "
            f"SilentLinks(scaf=1.0)={integ_audit['silent_passenger_scaf_1_0']['silent_count']}/10 | "
            f"SilentLinks(Autonomous=0.0)={integ_audit['silent_passenger_autonomous_0_0']['silent_count']}/10",
            flush=True,
        )

        # Evaluate every trained seed across core Stage 1 & Stage 2 matchups!
        per_seed_eval: Dict[str, Any] = dict(saved_state.get("per_seed_matchups", {}))
        pooled_pairs: Dict[str, List[float]] = dict(saved_state.get("_pooled_pairs", {}))

        def _rec(cond_key: str, res_d: Dict[str, Any], store: Dict[str, Any]) -> None:
            pw = res_d.pop("_pair_win")
            pooled_pairs.setdefault(cond_key, []).extend(pw)
            store[cond_key] = res_d
            saved_state["per_seed_matchups"] = per_seed_eval
            saved_state["_pooled_pairs"] = pooled_pairs
            _flush_partial()

        for sd in seeds:
            if str(sd) in per_seed_eval and len(per_seed_eval[str(sd)]) >= 7:
                print(f"[Resume] Skipping completed benchmark suite for seed {sd}.", flush=True)
                continue
            ckpt_sd = trained_ckpts[sd]
            print(f"\n{'=' * 84}\n  [Benchmark Seed {sd}] Stage 1 & Stage 2 Autonomous & Head-to-Head Suite (N={n_eval_seeds} pairs)\n{'=' * 84}", flush=True)
            c_v23_auto = make_eval_cfg(ckpt_sd, stage_id=1, scaffold_scale=0.0, seed=sd)
            c_v23_hyb = make_eval_cfg(ckpt_sd, stage_id=1, scaffold_scale=1.0, seed=sd)
            c_v22_auto = make_eval_cfg(v22_ckpt, stage_id=1, scaffold_scale=0.0, seed=sd + 100)
            c_v22_hyb = make_eval_cfg(v22_ckpt, stage_id=1, scaffold_scale=1.0, seed=sd + 100)
            c_untr_auto = Brain3DConfig(task_profile="turf_objective", policy_mode="pure_neural", enable_learning=False, scaffold_prior_scale=0.0, seed=sd + 200)
            c_l2_full = make_eval_cfg("", stage_id=1, scaffold_scale=1.0, seed=sd + 300)

            eng_v23 = WholeBrainNeuralEngineWrapper(c_v23_auto)
            eng_v22 = WholeBrainNeuralEngineWrapper(c_v22_auto)
            eng_untr = WholeBrainNeuralEngineWrapper(c_untr_auto)

            sd_store: Dict[str, Any] = {}
            per_seed_eval[str(sd)] = sd_store
            # 1. Autonomous v2.3 (scaf=0.0) vs Untrained (scaf=0.0)
            _rec("S1_v23Auto_vs_UntrainedAuto", _evaluate_matchup_battery(pool, label="v2.3(Auto,0.0) vs Untrained(0.0)", eng_a=eng_v23, cfg_a_template=c_v23_auto, variant_a="L5_full", eng_b=eng_untr, cfg_b_template=c_untr_auto, variant_b="L5_full", stage_id=1, n_seeds=n_eval_seeds), sd_store)
            _rec("S2_v23Auto_vs_UntrainedAuto", _evaluate_matchup_battery(pool, label="v2.3(Auto,0.0) vs Untrained(0.0)", eng_a=eng_v23, cfg_a_template=c_v23_auto, variant_a="L5_full", eng_b=eng_untr, cfg_b_template=c_untr_auto, variant_b="L5_full", stage_id=2, n_seeds=n_eval_seeds), sd_store)

            # 2. Autonomous v2.3 (scaf=0.0) vs Autonomous v2.2 Champion (scaf=0.0) — Direct Neural-to-Neural Test!
            _rec("S1_v23Auto_vs_v22Auto", _evaluate_matchup_battery(pool, label="v2.3(Auto,0.0) vs v2.2(Auto,0.0)", eng_a=eng_v23, cfg_a_template=c_v23_auto, variant_a="L5_full", eng_b=eng_v22, cfg_b_template=c_v22_auto, variant_b="L5_full", stage_id=1, n_seeds=n_eval_seeds), sd_store)
            _rec("S2_v23Auto_vs_v22Auto", _evaluate_matchup_battery(pool, label="v2.3(Auto,0.0) vs v2.2(Auto,0.0)", eng_a=eng_v23, cfg_a_template=c_v23_auto, variant_a="L5_full", eng_b=eng_v22, cfg_b_template=c_v22_auto, variant_b="L5_full", stage_id=2, n_seeds=n_eval_seeds), sd_store)

            # 3. Autonomous v2.3 (scaf=0.0) vs Full Hand-Crafted L2 Heuristic (scaf=1.0) on Stage 1 & Stage 2!
            _rec("S1_v23Auto_vs_L2FullHeuristic", _evaluate_matchup_battery(pool, label="v2.3(Auto,0.0) vs L2_Heuristic(1.0)", eng_a=eng_v23, cfg_a_template=c_v23_auto, variant_a="L5_full", eng_b=None, cfg_b_template=c_l2_full, variant_b="L2_homeo_scaffold", stage_id=1, n_seeds=n_eval_seeds), sd_store)
            _rec("S2_v23Auto_vs_L2FullHeuristic", _evaluate_matchup_battery(pool, label="v2.3(Auto,0.0) vs L2_Heuristic(1.0)", eng_a=eng_v23, cfg_a_template=c_v23_auto, variant_a="L5_full", eng_b=None, cfg_b_template=c_l2_full, variant_b="L2_homeo_scaffold", stage_id=2, n_seeds=n_eval_seeds), sd_store)

            # 4. Hybrid v2.3 (scaf=1.0) vs Hybrid v2.2 (scaf=1.0) on Stage 2!
            _rec("S2_v23Hybrid_vs_v22Hybrid", _evaluate_matchup_battery(pool, label="v2.3(Hyb,1.0) vs v2.2(Hyb,1.0)", eng_a=eng_v23, cfg_a_template=c_v23_hyb, variant_a="L5_full", eng_b=eng_v22, cfg_b_template=c_v22_hyb, variant_b="L5_full", stage_id=2, n_seeds=n_eval_seeds), sd_store)

            eng_v23.release_memory_and_vram()
            eng_v22.release_memory_and_vram()
            eng_untr.release_memory_and_vram()

        # Fixed Reference Check: Frozen v2.2 Autonomous (scaf=0.0) vs Full L2 Heuristic (scaf=1.0) on Stage 1 & Stage 2
        v22_ref_auto = dict(saved_state.get("v22_autonomous_vs_l2_reference", {}))
        if len(v22_ref_auto) < 2:
            print(f"\n{'=' * 84}\n  [Reference] Frozen v2.2 Champion under Autonomous Mode (scaf=0.0) vs L2 Heuristic (1.0)\n{'=' * 84}", flush=True)
            c_v22_a0 = make_eval_cfg(v22_ckpt, stage_id=1, scaffold_scale=0.0, seed=2026)
            c_l2_r = make_eval_cfg("", stage_id=1, scaffold_scale=1.0, seed=2526)
            eng_v22_r = WholeBrainNeuralEngineWrapper(c_v22_a0)
            r_v22_s1 = _evaluate_matchup_battery(pool, label="v2.2(Auto,0.0) vs L2_Heuristic(1.0)", eng_a=eng_v22_r, cfg_a_template=c_v22_a0, variant_a="L5_full", eng_b=None, cfg_b_template=c_l2_r, variant_b="L2_homeo_scaffold", stage_id=1, n_seeds=n_eval_seeds)
            r_v22_s2 = _evaluate_matchup_battery(pool, label="v2.2(Auto,0.0) vs L2_Heuristic(1.0)", eng_a=eng_v22_r, cfg_a_template=c_v22_a0, variant_a="L5_full", eng_b=None, cfg_b_template=c_l2_r, variant_b="L2_homeo_scaffold", stage_id=2, n_seeds=n_eval_seeds)
            r_v22_s1.pop("_pair_win", None)
            r_v22_s2.pop("_pair_win", None)
            v22_ref_auto = {"Stage1": r_v22_s1, "Stage2": r_v22_s2}
            saved_state["v22_autonomous_vs_l2_reference"] = v22_ref_auto
            _flush_partial()
            eng_v22_r.release_memory_and_vram()

        # Primary Seed Full Stage-2 Cumulative Organ Ladder & Organ Swaps (`S1..S5`)
        print(f"\n{'=' * 84}\n  [Stage 2 Organ Ladder & Swaps] Primary Seed {seeds[0]} (Autonomous scaf=0.0 vs L2 Yardstick)\n{'=' * 84}", flush=True)
        c_prim_auto = make_eval_cfg(primary_ckpt, stage_id=2, scaffold_scale=0.0, seed=seeds[0])
        c_l2_yard = make_eval_cfg("", stage_id=2, scaffold_scale=1.0, seed=seeds[0] + 500)
        eng_prim = WholeBrainNeuralEngineWrapper(c_prim_auto)

        stage2_ladder = dict(saved_state.get("stage2_autonomous_organ_ladder", {}))
        for v_rung in ("L0_reactive", "L2_homeo_scaffold", "L3_neural_d3qn", "L4_plus_rssm", "L5_full"):
            if v_rung in stage2_ladder:
                continue
            r_rung = _evaluate_matchup_battery(
                pool, label=f"Ladder_{v_rung}(Auto) vs L2", eng_a=eng_prim, cfg_a_template=c_prim_auto,
                variant_a=v_rung, eng_b=None, cfg_b_template=c_l2_yard, variant_b="L2_homeo_scaffold",
                stage_id=2, n_seeds=n_eval_seeds,
            )
            r_rung.pop("_pair_win", None)
            stage2_ladder[v_rung] = r_rung
            saved_state["stage2_autonomous_organ_ladder"] = stage2_ladder
            _flush_partial()

        stage2_swaps = dict(saved_state.get("stage2_autonomous_organ_swaps", {}))
        for v_sw in ("S1_rssm_mpc_no_d3qn", "S2_gwt_uniform_pool", "S4_slam_drift_only", "S5_bg_cautious"):
            if v_sw in stage2_swaps:
                continue
            r_sw = _evaluate_matchup_battery(
                pool, label=f"Swap_{v_sw}(Auto) vs L2", eng_a=eng_prim, cfg_a_template=c_prim_auto,
                variant_a=v_sw, eng_b=None, cfg_b_template=c_l2_yard, variant_b="L2_homeo_scaffold",
                stage_id=2, n_seeds=n_eval_seeds,
            )
            r_sw.pop("_pair_win", None)
            stage2_swaps[v_sw] = r_sw
            saved_state["stage2_autonomous_organ_swaps"] = stage2_swaps
            _flush_partial()

        eng_prim.release_memory_and_vram()

        # Pooled Multi-Seed Statistical Summary across all K seeds (`K * 24` pairs = `K * 48` games!)
        pooled_summary = {}
        for ck, vals in pooled_pairs.items():
            pc_p = paired_comparison(vals, [1.0 - x for x in vals], seed=2026)
            st_p = summary_stats(vals)
            seed_wrs = [per_seed_eval[str(sd)][ck]["win_rate_pct"] for sd in seeds if str(sd) in per_seed_eval and ck in per_seed_eval[str(sd)]]
            pooled_summary[ck] = {
                "n_seeds": len(seed_wrs),
                "total_pairs": len(vals),
                "total_games": 2 * len(vals),
                "mean_win_rate_pct": round(float(np.mean(vals)) * 100.0, 2),
                "seed_std_pct": round(float(np.std(seed_wrs)), 2),
                "per_seed_win_rates_pct": seed_wrs,
                "ci95_pct": [round(st_p["ci95_low"] * 100.0, 1), round(st_p["ci95_high"] * 100.0, 1)],
                "w_l_t": f"{pc_p['n_pos']}-{pc_p['n_neg']}-{pc_p['n_tie']}",
                "sign_p": round(float(pc_p["binom_sign_p"]), 6),
                "wilcoxon_p": round(float(pc_p["wilcoxon_p"]), 6),
                "dz": round(float(pc_p["cohens_dz"]), 3),
            }

        final_doc = {
            "schema": "universal_brain_3d_v23_phase_d_study",
            "ok": True,
            "seeds": seeds,
            "per_seed_training": per_seed_train,
            "integration_audit_primary_seed": integ_audit,
            "pooled_multi_seed_matchups": pooled_summary,
            "v22_autonomous_vs_l2_reference": v22_ref_auto,
            "per_seed_matchups": per_seed_eval,
            "stage2_autonomous_organ_ladder": stage2_ladder,
            "stage2_autonomous_organ_swaps": stage2_swaps,
            "worker_wall_sec": round(time.perf_counter() - t_start, 2),
        }
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as fh:
            json.dump(final_doc, fh, indent=2, ensure_ascii=False)
        res_q.put({"ok": True, "out_json_path": out_json_path, "worker_wall_sec": final_doc["worker_wall_sec"]})
    finally:
        pool.close()


def main() -> int:
    assert "torch" not in sys.modules, "Barrier 1 violated in parent coordinator!"
    ap = argparse.ArgumentParser(description="Universal Brain 3D v2.3 Phase D Full Study Runner")
    ap.add_argument("--seeds", type=str, default="2026,2027,2028")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--arenas-per-gen", type=int, default=8)
    ap.add_argument("--sleep-epochs", type=int, default=18)
    ap.add_argument("--online-interval", type=int, default=2)
    ap.add_argument("--eval-pairs", type=int, default=24)
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--smoke-test", action="store_true", default=False)
    args = ap.parse_args()

    v22_ckpt = os.path.join(ROOT_DIR, "checkpoints", "remediated_v22", "seed_2026", "pbt_best_champion.pt")
    assert os.path.isfile(v22_ckpt), f"Missing v2.2 reference checkpoint: {v22_ckpt}"

    if args.smoke_test:
        import tempfile

        print("=== [Smoke Test] Verifying Multi-Core Barrier-1 Pool, Phase B Gradient Cosines & Speed ===")
        with tempfile.TemporaryDirectory(prefix="ub3d_phased_smoke_") as tmp_dir:
            smoke_json = os.path.join(tmp_dir, "smoke_out.json")
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(
                target=_gpu_phase_d_worker,
                args=([2026], 1, 4, 4, 2, 2, 4, v22_ckpt, tmp_dir, smoke_json, False, q),
            )
            try:
                p.start()
                out = q.get(timeout=600.0)
                p.join(timeout=15.0)
                assert p.exitcode == 0 and out["ok"] is True
                assert "torch" not in sys.modules
                doc = json.load(open(smoke_json, encoding="utf-8"))
                ga = doc["integration_audit_primary_seed"]["gradient_audit"]
                print(
                    f"[PASS] Smoke Test OK in {out['worker_wall_sec']}s! "
                    f"min_overlap_cos={ga['min_shared_overlap_offdiag_cos']:+.4f}, "
                    f"td_vs_rssm={ga['td_vs_rssm_shared_overlap_cos']}, "
                    f"ParentZeroTorch={('torch' not in sys.modules)}"
                )
            finally:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5.0)
        return 0

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    study_dir = os.path.join(ROOT_DIR, "checkpoints", "v23_full_study")
    os.makedirs(study_dir, exist_ok=True)
    out_json = os.path.join(ROOT_DIR, "docs", "v23_phase_d_benchmark_results.json")

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_gpu_phase_d_worker,
        args=(
            seeds, int(args.generations), int(args.arenas_per_gen), int(args.sleep_epochs),
            int(args.online_interval), int(args.eval_pairs), int(args.cpu_workers),
            v22_ckpt, study_dir, out_json, bool(args.skip_train), q,
        ),
    )
    try:
        p.start()
        study_meta = q.get(timeout=14400.0)
        p.join(timeout=20.0)
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
    assert p.exitcode == 0 and study_meta.get("ok") is True, f"Phase D worker exited with code {p.exitcode}"
    assert "torch" not in sys.modules, "Barrier 1 violated after Phase D run!"

    with open(out_json, "r", encoding="utf-8") as fh:
        full_doc = json.load(fh)
    full_doc["barrier1_parent_zero_torch"] = ("torch" not in sys.modules)
    full_doc["total_wall_sec"] = round(time.perf_counter() - t0, 2)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(full_doc, fh, indent=2, ensure_ascii=False)

    print(f"\n[COMPLETE] Saved Phase D Full Multi-Seed Study JSON -> {out_json} ({full_doc['total_wall_sec']}s)")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
