# -*- coding: utf-8 -*-
"""
scripts/train_and_audit_v23_short_term.py — Universal Brain 3D v2.3 Short-Term Upgrade
High-Update Single-GPU League Trainer (~3,000+ Gradient Updates) & Falsifiable Exp6 Audit
=========================================================================================
Barrier 1 Guarantee:
  The parent coordinator NEVER imports `torch` (`assert "torch" not in sys.modules`).
  Training and auditing execute inside isolated spawned GPU worker processes.
  All outputs are saved under `checkpoints/v23_experiments/` and `docs/v23_short_term_audit_results.json`,
  leaving `checkpoints/user_runs/`, `checkpoints/remediated_v22/`, and `paper_arxiv/` 100% untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import multiprocessing as mp
import os
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
    MULTIMODAL_EMBED_DIM,
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
    VERB_APPROACH_HOVER,
    VERB_IDLE_OBSERVE,
    VERB_IMPULSE_PUSH,
    VERB_RELEASE_PLACE,
    ActionStagnationHomeostasis3D,
    BasalGanglia3DTimingEngineV2,
    Cerebellar3DKinematicPredictorV2,
    EntorhinalHippocampal3DEngineV2,
    _extract_candidate_compass_features_np,
    compose_vis8_patch,
    compute_gene_modulated_mip_and_fr,
    compute_task_scaffold_targets,
    compute_universal_memory_compass,
    derive_ventral_fallback_from_dorsal_16,
    extract_5module_activity_vec,
    get_resource_tank_level,
    get_task_profile,
    is_in_stealth_or_cover_refuge,
    project_pomdp_proprio,
    propose_universal_3d_candidates,
    select_policy_candidate,
)
from arena_pbt_trainer import (
    HeadToHeadTactical3DArena,
    canonicalize_team_observation,
    uncanonicalize_team_motor,
)


class BatchedLearnerSlotController:
    """
    Embodied 1st-Person POMDP Controller for Batched Single-GPU Self-Play Training (`v2.3`).
    Uses identical SLAM, 4-Layer Field Map, Cerebellar Lead-Aim, Basal Ganglia Gate, and Homeostasis
    as `LocalCpuBrainAgentDriver`, while batching GPU forward passes across `N` simultaneous arenas.
    """

    def __init__(self, cfg: Brain3DConfig, slot_id: int, trainable: bool = True, scaffold_only: bool = False):
        self.cfg = Brain3DConfig(**asdict(cfg))
        self.profile = get_task_profile(self.cfg.task_profile)
        self.slot_id = int(slot_id)
        self.trainable = bool(trainable)
        self.scaffold_only = bool(scaffold_only)

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
        self.respawn_seq = 0
        self.last_ep_seen = -1

        self.cached_pose = np.zeros(CONT_POSE_DIM, dtype=np.float32)
        self.cached_verb = VERB_APPROACH_HOVER
        self.cached_idx = 0
        self.cached_speed = 0.88
        self.cached_q = 0.5
        self.cached_surv = 0.85
        self.s1_fast_idx = 0
        self.pending_tr: Optional[Dict[str, Any]] = None
        self._ctx: Any = None
        self._want_cog = False

    def reset_for_new_episode(self, ep_id: int) -> None:
        self.last_ep_seen = int(ep_id)
        self.pending_tr = None
        self.prev_pi = None
        self.last_slot_coords = None
        self.last_slot_valence = None
        self.last_vis_surp = 0.0
        self.last_w_surp = 0.0
        self.last_ign = 0.0
        self.last_s_surp = 0.0

    def pre_step(self, obs: Dict[str, Any], dt: float, cog_stride: int = 4) -> Optional[Dict[str, Any]]:
        ep_id = int(obs["episode_id"])
        if ep_id != self.last_ep_seen:
            self.reset_for_new_episode(ep_id)
        if bool(obs.get("respawn_flag", False)):
            self.respawn_seq += 1

        vent_p = derive_ventral_fallback_from_dorsal_16(
            obs["fovea_patch"], yaw_pitch=obs["self_yaw_pitch"], prev_state=self.ventral_prev, episode_id=ep_id
        )
        sp_out = self.hippo.step(
            obs, dt, ventral_override=vent_p, respawn_atom=self.respawn_seq,
            topdown_goal_mode=self.last_goal_mode, topdown_focus_xyz=self.last_focus_xyz, lead_aim_xyz=self.last_lead_xyz,
            vis_slot_coords=self.last_slot_coords, vis_slot_valence=self.last_slot_valence,
            vis_pred_surprise=self.last_vis_surp, world_surprise=self.last_w_surp, gwt_ignition=self.last_ign,
        )
        lead_xyz, opp_prob, kin_risk, sc_spd = self.cereb.predict_lead_and_crossfire(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"]
        )
        self.last_lead_xyz = lead_xyz.copy()
        prop = project_pomdp_proprio(obs["proprio"], sp_out["believed_self_pos"], self.cfg.world_min, self.cfg.world_max, self.cfg)
        mc24 = compute_universal_memory_compass(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop, self.hippo.field_map, self.profile
        )
        c_poses, c_verbs = propose_universal_3d_candidates(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop,
            mc24, self.hippo.field_map, lead_xyz, sp_out["active_gaze_ypd"],
            self.cfg.world_min, self.cfg.world_max, self.profile, goal_mode=self.last_goal_mode,
        )
        tick = int(obs["physics_tick"])
        ep_done = bool(obs["episode_done"])
        respawn_ev = bool(sp_out["respawn_event_flag"])
        want_cog = (tick % cog_stride == 0) or ep_done or respawn_ev or (tick <= 1) or (self.trainable and self.pending_tr is None)
        self._want_cog = want_cog
        self._ctx = (obs, dt, vent_p, sp_out, opp_prob, kin_risk, sc_spd, prop, mc24, c_poses, c_verbs, tick, ep_done, respawn_ev)

        if want_cog and (not self.scaffold_only):
            return {
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
        return None

    def post_step(
        self,
        gwt_out: Optional[Dict[str, Any]],
        engine: Any,
        beta_teacher: float = 0.0,
        arena_seed: int = 0,
    ) -> Tuple[np.ndarray, int, bool, np.ndarray]:
        obs, dt, vent_p, sp_out, opp_prob, kin_risk, sc_spd, prop, mc24, c_poses, c_verbs, tick, ep_done, respawn_ev = self._ctx
        ep_id = int(obs["episode_id"])
        rng_seed = tick + self.slot_id * 7919 + ep_id * 131 + arena_seed * 31
        just_switched = False

        if self._want_cog:
            if self.scaffold_only:
                zeros8 = np.zeros(8, dtype=np.float32)
                surv8 = np.full(8, 0.85, dtype=np.float32)
                adv8 = np.full(8, 0.50, dtype=np.float32)
                q_star, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                    zeros8, surv8, adv8, np.zeros(128, dtype=np.float32), c_poses, c_verbs, prop, mc24, None, self.cfg,
                    habituation_trace=self.homeostasis.habituation_trace, stagnation_level=self.homeostasis.stagnation_level,
                )
                c_idx = select_policy_candidate(q_star, pi, trainable=False, rng_seed_val=rng_seed)
                _ = self.homeostasis.update_and_compute_intrinsic(ep_id, sp_out["believed_self_pos"][[0, 2]], prop, c_idx, float(obs["last_step_reward"]))
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

                q_star, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                    gwt_out["q_values"], gwt_out["survival_probs"], gwt_out["pred_adv_probs"],
                    gwt_out["gwt_workspace"], c_poses, c_verbs, prop, mc24, self.prev_pi, self.cfg,
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

                if c_idx != int(gwt_out.get("prov_chosen_idx", -1)):
                    engine.notify_chosen_action(self.slot_id, gwt_out["cand_feats32"][c_idx].copy())

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

                just_switched = (c_idx != self.cached_idx)
                self.cached_idx = c_idx
                self.cached_speed = float(gwt_out["speed_mod"][c_idx])
                self.cached_verb = int(c_verbs[c_idx])
                self.cached_pose = c_poses[c_idx].copy()
                self.cached_pose[5] = self.cached_speed
                self.cached_q = float(gwt_out["q_values"][c_idx])
                self.cached_surv = float(gwt_out["survival_probs"][c_idx])

                cur_haz = max(float(obs["hazard_level"]), float(sp_out["believed_hazard"]))
                if self.trainable and self.pending_tr is not None:
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
                        engine.record_transition_and_maybe_train(self.pending_tr)
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


def _gpu_train_v23_worker(
    seed: int,
    generations: int,
    arenas_per_gen: int,
    sleep_epochs: int,
    online_interval: int,
    out_dir: str,
    v22_ref_ckpt: str,
    res_q: mp.Queue,
) -> None:
    from universal_brain_3d_core import WholeBrainNeuralEngineWrapper, PrioritizedReplayRingBuffer
    from exp6_architecture_criteria import evaluate_variant_vs_yardstick

    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    cfg_learn = Brain3DConfig(
        task_profile="turf_objective",
        policy_mode="pure_neural",
        online_update_interval=max(1, int(online_interval)),
        enable_overfit_guard=False,  # Use validation-league selection instead of noisy 3-ep rolling score blend
        continual_anchor_l2=0.0,     # Allow full multi-thousand-step plastic representation learning from scratch
        seed=int(seed),
    )
    learner = WholeBrainNeuralEngineWrapper(cfg_learn)
    # Expand replay capacity to 8192 for rich multi-generation PFSP retention
    learner.max_replay = 8192
    learner.replay = PrioritizedReplayRingBuffer(capacity=8192, on_clear_cb=learner._clear_temporal_queues)

    opp_cfg = Brain3DConfig(task_profile="turf_objective", policy_mode="pure_neural", enable_learning=False, seed=int(seed) + 500)
    opp_engine = WholeBrainNeuralEngineWrapper(opp_cfg)

    league_pool: List[Tuple[str, Dict[str, Any]]] = [
        ("snap_gen0_untrained", learner.get_weights_state_dict(as_numpy=True))
    ]
    if v22_ref_ckpt and os.path.isfile(v22_ref_ckpt):
        opp_engine.load_checkpoint(v22_ref_ckpt)
        league_pool.append(("v22_champion_ref", opp_engine.get_weights_state_dict(as_numpy=True)))

    rng = np.random.default_rng(int(seed) + 99)
    dt = 1.0 / 60.0
    best_val_metric = -1e9
    best_ckpt_path = os.path.join(out_dir, "pbt_best_champion.pt")
    latest_ckpt_path = os.path.join(out_dir, "pbt_latest_champion.pt")
    gen_logs: List[Dict[str, Any]] = []

    for gen in range(1, int(generations) + 1):
        gt0 = time.perf_counter()
        # Smooth DAgger -> Pure Neural schedule: Gen 1..3 warm-starts, Gen 4+ is 100% pure neural self-play!
        if gen == 1:
            beta_t = 0.60
        elif gen == 2:
            beta_t = 0.28
        elif gen == 3:
            beta_t = 0.08
        else:
            beta_t = 0.0

        # Pick a league opponent snapshot for the neural sparring slots
        pick_idx = int(rng.integers(0, len(league_pool)))
        opp_name, opp_state = league_pool[pick_idx]
        opp_engine.load_weights_state_dict(opp_state)
        opp_engine._agent_states.clear()
        learner._agent_states.clear()

        arenas: List[HeadToHeadTactical3DArena] = []
        learners_s: List[BatchedLearnerSlotController] = []
        opps_n: List[BatchedLearnerSlotController] = []
        opp_modes: List[str] = []  # "league" | "mirror" | "l2_scaffold"
        obs_s_list, obs_n_list = [], []
        ep_seeds = []

        for k in range(int(arenas_per_gen)):
            ep_seed = int(seed) * 13 + gen * 1009 + k * 97
            ep_seeds.append(ep_seed)
            ar = HeadToHeadTactical3DArena(Brain3DConfig(task_profile="turf_objective", seed=ep_seed), seed=ep_seed)
            ar.episode_id = gen * 100 + k + 1
            arenas.append(ar)
            obs_s_list.append(ar._build_agent_obs(+1, 0.0, False))
            obs_n_list.append(canonicalize_team_observation(ar._build_agent_obs(-1, 0.0, False), -1, ar.wmin, ar.wmax))

            l_slot = BatchedLearnerSlotController(cfg_learn, slot_id=k, trainable=True, scaffold_only=False)
            learners_s.append(l_slot)

            if k < max(1, int(arenas_per_gen) // 2):
                opp_modes.append("league")
                opps_n.append(BatchedLearnerSlotController(opp_cfg, slot_id=100 + k, trainable=False, scaffold_only=False))
            elif k < int(arenas_per_gen) - 2:
                opp_modes.append("mirror")
                opps_n.append(BatchedLearnerSlotController(opp_cfg, slot_id=200 + k, trainable=False, scaffold_only=False))
            else:
                opp_modes.append("l2_scaffold")
                opps_n.append(BatchedLearnerSlotController(opp_cfg, slot_id=300 + k, trainable=False, scaffold_only=True))

        done_flags = [False] * len(arenas)
        winners = [0] * len(arenas)

        for _ in range(545):
            if all(done_flags):
                break
            learn_reqs = []
            learn_map_s: List[Optional[int]] = [None] * len(arenas)
            learn_map_n_mirror: List[Optional[int]] = [None] * len(arenas)
            opp_reqs = []
            opp_map_n: List[Optional[int]] = [None] * len(arenas)

            for k in range(len(arenas)):
                if done_flags[k]:
                    continue
                rs = learners_s[k].pre_step(obs_s_list[k], dt, cog_stride=4)
                if rs is not None:
                    learn_map_s[k] = len(learn_reqs)
                    learn_reqs.append(rs)
                rn = opps_n[k].pre_step(obs_n_list[k], dt, cog_stride=4)
                if rn is not None:
                    if opp_modes[k] == "mirror":
                        learn_map_n_mirror[k] = len(learn_reqs)
                        learn_reqs.append(rn)
                    elif opp_modes[k] == "league":
                        opp_map_n[k] = len(opp_reqs)
                        opp_reqs.append(rn)

            learn_outs = learner.infer_conscious_gwt_batch(learn_reqs) if learn_reqs else []
            opp_outs = opp_engine.infer_conscious_gwt_batch(opp_reqs) if opp_reqs else []

            for k in range(len(arenas)):
                if done_flags[k]:
                    continue
                gs = learn_outs[learn_map_s[k]] if learn_map_s[k] is not None else None
                if opp_modes[k] == "mirror":
                    gn = learn_outs[learn_map_n_mirror[k]] if learn_map_n_mirror[k] is not None else None
                    eng_n = learner
                elif opp_modes[k] == "league":
                    gn = opp_outs[opp_map_n[k]] if opp_map_n[k] is not None else None
                    eng_n = opp_engine
                else:
                    gn = None
                    eng_n = None

                act_s = learners_s[k].post_step(gs, learner, beta_teacher=beta_t, arena_seed=ep_seeds[k])
                tp_nc, vb_n, gt_n, gz_nc = opps_n[k].post_step(gn, eng_n, beta_teacher=0.0, arena_seed=ep_seeds[k])
                tp_nw, gz_nw = uncanonicalize_team_motor(tp_nc, gz_nc, -1)
                obs_s, obs_n_raw, ep_done, w = arenas[k].step_both(act_s, (tp_nw, vb_n, gt_n, gz_nw), dt=dt)
                obs_s_list[k] = obs_s
                obs_n_list[k] = canonicalize_team_observation(obs_n_raw, -1, arenas[k].wmin, arenas[k].wmax)
                if ep_done:
                    # Flush final terminal transition on the learner slot!
                    rs_fin = learners_s[k].pre_step(obs_s, dt, cog_stride=4)
                    if rs_fin is not None:
                        fin_out = learner.infer_conscious_gwt_batch([rs_fin])[0]
                        _ = learners_s[k].post_step(fin_out, learner, beta_teacher=beta_t, arena_seed=ep_seeds[k])
                    done_flags[k] = True
                    winners[k] = int(w)

        # Inter-generation Offline Sleep Consolidation (`S-1 .. S-4`)
        sleep_res = learner.run_sleep_consolidation(epochs=max(4, int(sleep_epochs)))
        league_pool.append((f"snap_gen{gen}", learner.get_weights_state_dict(as_numpy=True)))
        if len(league_pool) > 8:
            # Keep snap_gen0_untrained + v22_ref + latest 6 snapshots
            league_pool = league_pool[:2] + league_pool[-6:]

        gen_wr = float(np.mean([1.0 if w > 0 else (0.0 if w < 0 else 0.5) for w in winners])) * 100.0
        gates_now = _extract_gate_telemetry(learner)
        learner.save_checkpoint(latest_ckpt_path, extra_meta={"v23_gen": gen, "beta_teacher": beta_t})

        val_wr_l2 = None
        if gen >= 3 and (gen % 3 == 0 or gen == int(generations)):
            val_res = evaluate_variant_vs_yardstick(latest_ckpt_path, "L5_full_v21", "L2_homeo_scaffold", learner, n_seeds=8)
            val_wr_l2 = float(val_res["win_rate_pct"])
            val_metric = val_wr_l2 + 0.20 * float(val_res["max_prog_pct"])
            if val_metric >= best_val_metric:
                best_val_metric = val_metric
                learner.save_checkpoint(
                    best_ckpt_path,
                    extra_meta={"v23_gen": gen, "val_wr_vs_l2_pct": val_wr_l2, "val_max_prog_pct": val_res["max_prog_pct"]},
                )

        g_elapsed = time.perf_counter() - gt0
        log_item = {
            "gen": gen,
            "beta_teacher": beta_t,
            "opp_sampled": opp_name,
            "train_steps_done": int(learner.train_steps_done),
            "transitions_recorded": int(learner.transitions_recorded),
            "sleep_cycles_done": int(learner.sleep_cycles_done),
            "nan_rollbacks": int(learner.nan_rollback_count),
            "gen_train_wr_pct": round(gen_wr, 1),
            "val_wr_vs_l2_pct": val_wr_l2,
            "mean_sleep_loss": round(float(sleep_res["mean_loss"]), 4),
            "gates": gates_now,
            "sec": round(g_elapsed, 2),
        }
        gen_logs.append(log_item)
        print(
            f"[v2.3 Train Gen {gen:02d}/{generations:02d}] steps={learner.train_steps_done:4d} "
            f"trans={learner.transitions_recorded:5d} beta={beta_t:.2f} trainWR={gen_wr:5.1f}% "
            f"valL2={str(val_wr_l2) + '%' if val_wr_l2 is not None else '  -  '} "
            f"gates(td={gates_now['tanh_td_slot_gate']:+.3f}, z={gates_now['tanh_z_res_gate']:+.3f}, "
            f"gwt={gates_now['tanh_gwt_res_gate']:+.3f}, spot={gates_now['tanh_spot_gate']:+.3f}, "
            f"meta={gates_now['tanh_meta_err_gate']:+.3f}, rvq={gates_now['rvq_prior_abs_max']:.3f}) "
            f"({g_elapsed:.1f}s)",
            flush=True,
        )

    if not os.path.isfile(best_ckpt_path):
        learner.save_checkpoint(best_ckpt_path, extra_meta={"v23_gen": int(generations)})

    vram_peak_mb = 0.0
    if learner.device.type == "cuda":
        vram_peak_mb = round(float(learner.torch.cuda.max_memory_allocated(learner.device)) / (1024.0 * 1024.0), 2)

    out_summary = {
        "best_ckpt_path": best_ckpt_path,
        "latest_ckpt_path": latest_ckpt_path,
        "train_steps_done": int(learner.train_steps_done),
        "transitions_recorded": int(learner.transitions_recorded),
        "sleep_cycles_done": int(learner.sleep_cycles_done),
        "nan_rollback_count": int(learner.nan_rollback_count),
        "vram_peak_mb": vram_peak_mb,
        "final_gates": _extract_gate_telemetry(learner),
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "generation_logs": gen_logs,
    }
    learner.release_memory_and_vram()
    opp_engine.release_memory_and_vram()
    res_q.put(out_summary)


def _collect_audit_replay_local(ckpt: str) -> Tuple[Any, Any]:
    from universal_brain_3d_core import WholeBrainNeuralEngineWrapper, CFEAT_HAZARD, CFEAT_IN_REFUGE
    from exp6_architecture_criteria import make_base_cfg

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
        vp_s: Dict[str, Any] = {}
        vp_n: Dict[str, Any] = {}
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
    return eng, eng_opp


def _run_v23_gradient_conflict_audit(ckpt: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Exact 9-Loss Multi-Objective Gradient Cosine Audit on the full v2.3 training graph
    (including `apply_zero_gate`, `slots_vis_in`, and `ws_mode_head` inside `loss_strat`).
    """
    from exp6_architecture_criteria import _onehot_st0

    eng, eng_opp = _collect_audit_replay_local(ckpt)
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
        _, batch, is_w = eng.replay.sample_prioritized(24, alpha=0.60, beta=0.60)
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
        rewards_nstep = eng._t([float(b["nstep_return"]) for b in batch], torch.float32)
        gammas_nstep = eng._t([float(b["nstep_gamma"]) for b in batch], torch.float32)
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
        gwt_base, alp_live, ign_live, _ = eng.models["gwt"](
            z_perc, slot_ctx64, mec, hd, comp, h_post.detach(), post_st.detach(), prop, haz_pred.detach(), strat_emb,
            prev_gwt_t, meta_salience=prev_ssurp_t, hab_trace=prev_hab_t, novelty_5=nov5_tr,
        )
        delta_gwt, topdown_gain_16 = eng.models["topo_adapter"].compute_topdown_spotlight(gwt_base, slot_aux["feat16"], slots, s_coords)
        gwt_live = gwt_base + delta_gwt
        z_td = (0.08 + 0.24 * ign_live).unsqueeze(-1) * eng.models["gwt"].topdown_z(gwt_live)
        self_cur, _, _, self_pred = eng.models["self_model"](gwt_live.detach(), prop, prev_self_t, gwt_alpha=alp_live.detach(), prev_act=prev_act_t, prev_alpha=prev_alpha_t)

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
    saved_snapshots = [dict(eng.replay[k]) for k in range(len(eng.replay))]
    eng.release_memory_and_vram()
    eng_opp.release_memory_and_vram()

    return {
        "loss_names": loss_names,
        "mean_grad_l2_norm": {k: round(float(np.mean(v)), 5) for k, v in norm_accum.items()},
        "modules_touched_by_loss": {k: sorted(list(v)) for k, v in mod_touch.items()},
        "min_shared_overlap_offdiag_cos": round(min(off_diag_ov) if off_diag_ov else 0.0, 4),
        "max_shared_overlap_offdiag_cos": round(max(off_diag_ov) if off_diag_ov else 0.0, 4),
        "pairwise_grad_cosine_full": [[round(float(cos_mat[i, j]), 4) for j in range(len(loss_names))] for i in range(len(loss_names))],
        "pairwise_grad_cosine_shared_overlap": [
            [round(float(ov_cos_mat[i, j]), 4) if ov_cos_counts[i, j] > 0 else None for j in range(len(loss_names))]
            for i in range(len(loss_names))
        ],
    }, saved_snapshots


def _evaluate_neural_head_to_head(
    ckpt_a: str,
    label_a: str,
    ckpt_b: str,
    label_b: str,
    n_seeds: int = 24,
) -> Dict[str, Any]:
    """
    Exact side-swapped (`2 * n_seeds` = 48 games) Head-to-Head evaluation between two neural checkpoints
    (`ckpt_a` vs `ckpt_b`; if `ckpt_b == ""`, `b` is freshly initialized untrained weights).
    """
    from universal_brain_3d_core import WholeBrainNeuralEngineWrapper
    from exp6_architecture_criteria import ModularAgentSlot, make_base_cfg
    from stats_utils import paired_comparison, summary_stats

    eng_a = WholeBrainNeuralEngineWrapper(make_base_cfg(ckpt_a, profile="turf_objective", seed=2026))
    cfg_b = make_base_cfg(ckpt_b, profile="turf_objective", seed=3026) if ckpt_b else Brain3DConfig(
        task_profile="turf_objective", policy_mode="pure_neural", enable_learning=False, seed=3026
    )
    eng_b = WholeBrainNeuralEngineWrapper(cfg_b)

    seeds = [2000 + 97 * k for k in range(n_seeds)]
    dt = 1.0 / 60.0
    arenas, obs_s_list, obs_n_list = [], [], []
    slots_s: List[ModularAgentSlot] = []
    slots_n: List[ModularAgentSlot] = []
    s_uses_a: List[bool] = []
    game_meta = []

    slot_a_cur, slot_b_cur = 0, 0
    for s_i, a_seed in enumerate(seeds):
        for swapped in (False, True):
            ar = HeadToHeadTactical3DArena(Brain3DConfig(task_profile="turf_objective", seed=a_seed), seed=a_seed)
            arenas.append(ar)
            obs_s_list.append(ar._build_agent_obs(+1, 0.0, False))
            obs_n_list.append(canonicalize_team_observation(ar._build_agent_obs(-1, 0.0, False), -1, ar.wmin, ar.wmax))

            south_is_a = (not swapped)
            s_uses_a.append(south_is_a)
            cs = make_base_cfg(ckpt_a if south_is_a else ckpt_b, profile="turf_objective", seed=3000 + s_i)
            cn = make_base_cfg(ckpt_b if south_is_a else ckpt_a, profile="turf_objective", seed=4000 + s_i)
            if south_is_a:
                slots_s.append(ModularAgentSlot(cs, slot_a_cur, "L5_full_v21")); slot_a_cur += 1
                slots_n.append(ModularAgentSlot(cn, slot_b_cur, "L5_full_v21")); slot_b_cur += 1
            else:
                slots_s.append(ModularAgentSlot(cs, slot_b_cur, "L5_full_v21")); slot_b_cur += 1
                slots_n.append(ModularAgentSlot(cn, slot_a_cur, "L5_full_v21")); slot_a_cur += 1
            game_meta.append((s_i, swapped, a_seed))

    n_games = len(arenas)
    deaths_s, deaths_n = [0] * n_games, [0] * n_games
    shots_s, shots_n = [0] * n_games, [0] * n_games
    max_p_s, max_p_n = [0.50] * n_games, [0.50] * n_games
    winners = [0] * n_games
    done_flags = [False] * n_games

    t0 = time.perf_counter()
    for _ in range(545):
        if all(done_flags):
            break
        reqs_a, reqs_b = [], []
        map_s, map_n = [None] * n_games, [None] * n_games
        for g in range(n_games):
            if done_flags[g]:
                continue
            rs = slots_s[g].pre_step(obs_s_list[g], dt, cog_stride=4)
            rn = slots_n[g].pre_step(obs_n_list[g], dt, cog_stride=4)
            if s_uses_a[g]:
                if rs is not None:
                    map_s[g] = len(reqs_a); reqs_a.append(rs)
                if rn is not None:
                    map_n[g] = len(reqs_b); reqs_b.append(rn)
            else:
                if rs is not None:
                    map_s[g] = len(reqs_b); reqs_b.append(rs)
                if rn is not None:
                    map_n[g] = len(reqs_a); reqs_a.append(rn)

        outs_a = eng_a.infer_conscious_gwt_batch(reqs_a) if reqs_a else []
        outs_b = eng_b.infer_conscious_gwt_batch(reqs_b) if reqs_b else []

        for g in range(n_games):
            if done_flags[g]:
                continue
            _, _, a_seed = game_meta[g]
            if s_uses_a[g]:
                gs = outs_a[map_s[g]] if map_s[g] is not None else None
                gn = outs_b[map_n[g]] if map_n[g] is not None else None
                es, en = eng_a, eng_b
            else:
                gs = outs_b[map_s[g]] if map_s[g] is not None else None
                gn = outs_a[map_n[g]] if map_n[g] is not None else None
                es, en = eng_b, eng_a
            prev_cd_s = int(arenas[g].agents[+1]["cd"])
            prev_cd_n = int(arenas[g].agents[-1]["cd"])
            act_s = slots_s[g].post_step(gs, es, a_seed)
            act_n = slots_n[g].post_step(gn, en, a_seed)
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

    wall_s = time.perf_counter() - t0
    pair_win, pair_prog, pair_kills, pair_deaths, pair_turf, pair_shots = [], [], [], [], [], []
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
        contract_errs += slots_s[g1].nan_or_contract_errors + slots_n[g1].nan_or_contract_errors + slots_s[g2].nan_or_contract_errors + slots_n[g2].nan_or_contract_errors

    pc = paired_comparison(pair_win, [1.0 - x for x in pair_win], seed=2026)
    eng_a.release_memory_and_vram()
    eng_b.release_memory_and_vram()
    res = {
        "matchup": f"{label_a} vs {label_b}",
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
        "contract_errors": int(contract_errs),
        "eval_wall_sec_48games": round(wall_s, 2),
    }
    print(
        f"  [{label_a:16s} vs {label_b:16s}] WR={res['win_rate_pct']:5.1f}% ({res['w_l_t']}, p={res['wilcoxon_p']:.4f}) "
        f"prog={res['max_prog_pct']:5.1f}% K/D={res['kills_per_game']:.2f}/{res['deaths_per_game']:.2f} "
        f"turf={res['turf_share_pct']:4.1f}% errs={res['contract_errors']} ({wall_s:.1f}s)",
        flush=True,
    )
    return res


def _gpu_audit_v23_worker(v23_ckpt: str, v22_ckpt: str, res_q: mp.Queue) -> None:
    from universal_brain_3d_core import WholeBrainNeuralEngineWrapper
    from exp6_architecture_criteria import evaluate_variant_vs_yardstick, make_base_cfg, run_silent_passenger_scan

    print("\n=== [v2.3 Audit 1/4] 9-Loss Multi-Objective Gradient Cosine Audit ===")
    grad_audit_v23, snapshots_v23 = _run_v23_gradient_conflict_audit(v23_ckpt)
    print(
        f"  [Gradient Overlap Cosines] min_offdiag_cos={grad_audit_v23['min_shared_overlap_offdiag_cos']:+.4f} | "
        f"max_offdiag_cos={grad_audit_v23['max_shared_overlap_offdiag_cos']:+.4f}",
        flush=True,
    )

    print("\n=== [v2.3 Audit 2/4] Exact Exp6 10-Link Silent-Passenger Scan (Unmodified Inference Clamp) ===")
    passenger_v23 = run_silent_passenger_scan(v23_ckpt, snapshots_v23)

    eng_v23 = WholeBrainNeuralEngineWrapper(make_base_cfg(v23_ckpt, profile="turf_objective", seed=2026))
    gates_v23 = _extract_gate_telemetry(eng_v23)

    print("\n=== [v2.3 Audit 3/4] Cumulative Organ Ladder vs Fixed Yardstick L2 (N=24 pairs = 48 games each) ===")
    ladder_v23 = {
        v: evaluate_variant_vs_yardstick(v23_ckpt, v, "L2_homeo_scaffold", eng_v23, n_seeds=24)
        for v in ("L2_homeo_scaffold", "L3_neural_d3qn", "L4_plus_rssm", "L5_full_v21")
    }
    swap_v23 = {
        v: evaluate_variant_vs_yardstick(v23_ckpt, v, "L2_homeo_scaffold", eng_v23, n_seeds=24)
        for v in ("S1_rssm_mpc_no_d3qn", "S2_gwt_uniform_pool", "S4_slam_drift_only", "S5_bg_cautious")
    }
    eng_v23.release_memory_and_vram()

    print("\n=== [v2.3 Audit 4/4] Side-Swapped Head-to-Head Matchups (N=24 pairs = 48 games each) ===")
    h2h_vs_untrained = _evaluate_neural_head_to_head(v23_ckpt, "v2.3_Champion", "", "Untrained_Init", n_seeds=24)
    h2h_vs_v22 = _evaluate_neural_head_to_head(v23_ckpt, "v2.3_Champion", v22_ckpt, "v2.2_Champion", n_seeds=24)

    eng_v22 = WholeBrainNeuralEngineWrapper(make_base_cfg(v22_ckpt, profile="turf_objective", seed=2026))
    gates_v22 = _extract_gate_telemetry(eng_v22)
    v22_steps = int(eng_v22.train_steps_done)
    eng_v22.release_memory_and_vram()

    res_q.put({
        "v22_reference": {
            "checkpoint": os.path.relpath(v22_ckpt, ROOT_DIR).replace("\\", "/"),
            "train_steps_done": v22_steps,
            "gates": gates_v22,
        },
        "v23_gates": gates_v23,
        "criterion_2a_gradient_conflict_v23": grad_audit_v23,
        "criterion_2b_silent_passenger_links_v23": passenger_v23,
        "criterion_3_cumulative_ladder_v23": ladder_v23,
        "criterion_1_replaceability_swaps_v23": swap_v23,
        "head_to_head_matchups": {
            "v23_vs_untrained": h2h_vs_untrained,
            "v23_vs_v22_champion": h2h_vs_v22,
            "v23_vs_l2_scaffold": ladder_v23["L5_full_v21"],
        },
    })


def main() -> int:
    assert "torch" not in sys.modules, "Barrier 1 violation in parent coordinator!"
    ap = argparse.ArgumentParser(description="Universal Brain 3D v2.3 High-Update League Trainer & Exp6 Audit")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--arenas-per-gen", type=int, default=8)
    ap.add_argument("--sleep-epochs", type=int, default=18)
    ap.add_argument("--online-interval", type=int, default=2)
    ap.add_argument("--skip-train", action="store_true", default=False)
    args = ap.parse_args()

    out_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "v23_experiments", f"seed_{args.seed}")
    v22_ckpt = os.path.join(ROOT_DIR, "checkpoints", "remediated_v22", "seed_2026", "pbt_best_champion.pt")
    assert os.path.isfile(v22_ckpt), f"Missing v2.2 reference checkpoint: {v22_ckpt}"

    ctx = mp.get_context("spawn")
    t_all = time.perf_counter()

    train_summary: Dict[str, Any] = {}
    if not args.skip_train:
        print("=" * 82)
        print("  [Phase 1] Universal Brain 3D v2.3 — Single-GPU Batched League Training")
        print("=" * 82)
        q_tr = ctx.Queue()
        p_tr = ctx.Process(
            target=_gpu_train_v23_worker,
            args=(args.seed, args.generations, args.arenas_per_gen, args.sleep_epochs, args.online_interval, out_ckpt_dir, v22_ckpt, q_tr),
        )
        p_tr.start()
        train_summary = q_tr.get(timeout=1800.0)
        p_tr.join(timeout=15.0)
        assert p_tr.exitcode == 0, f"Training worker failed with exitcode {p_tr.exitcode}"

    v23_best = os.path.join(out_ckpt_dir, "pbt_best_champion.pt")
    v23_latest = os.path.join(out_ckpt_dir, "pbt_latest_champion.pt")
    assert os.path.isfile(v23_best), f"Missing trained v2.3 checkpoint: {v23_best}"
    if args.skip_train:
        best_meta = json.load(open(v23_best + ".meta.json", encoding="utf-8")) if os.path.isfile(v23_best + ".meta.json") else {}
        latest_meta = json.load(open(v23_latest + ".meta.json", encoding="utf-8")) if os.path.isfile(v23_latest + ".meta.json") else {}
        train_summary = {
            "best_ckpt_path": v23_best,
            "latest_ckpt_path": v23_latest,
            "best_checkpoint_meta": best_meta,
            "latest_checkpoint_meta": latest_meta,
        }

    print("\n" + "=" * 82)
    print("  [Phase 2] Universal Brain 3D v2.3 — Falsifiable Exp6 Integration & Win-Rate Audit")
    print("=" * 82)
    q_au = ctx.Queue()
    p_au = ctx.Process(target=_gpu_audit_v23_worker, args=(v23_best, v22_ckpt, q_au))
    p_au.start()
    audit_summary = q_au.get(timeout=4200.0)
    p_au.join(timeout=15.0)
    assert p_au.exitcode == 0, f"Audit worker failed with exitcode {p_au.exitcode}"

    assert "torch" not in sys.modules, "Barrier 1 violation after spawned runs!"
    full_report = {
        "schema": "universal_brain_3d_v23_short_term_audit",
        "barrier1_parent_zero_torch": ("torch" not in sys.modules),
        "training_summary": train_summary,
        "audit_summary": audit_summary,
        "total_wall_sec": round(time.perf_counter() - t_all, 2),
    }
    report_path = os.path.join(ROOT_DIR, "docs", "v23_short_term_audit_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(full_report, fh, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved complete v2.3 Short-Term Audit JSON -> {report_path}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
