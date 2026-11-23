# -*- coding: utf-8 -*-
"""
exp2_behavioral_ablations.py — Pillar B: Trick-Free Side-Swapped 180-Deg Ablation Matrix & Multi-Seed Audit
===========================================================================================================
Anti-Trick & Anti-Confound Guarantees:
  1. Strict `policy_mode = "pure_neural"` (`beta_teacher == 0.0`) and `enable_learning = False` (frozen weights).
  2. Independent statistical unit = Seed-Matched Side-Swapped Pair (`N_SEEDS = 24` seeds x 2 games = 48 matches/cond).
  3. `C0_Sanity_Full_vs_Full` tests BOTH distinct-RNG raw Game-1 South vs. North fairness (exact Binomial test)
     AND matched-RNG deterministic 180-deg involution equivariance (`0.000000` trajectory asymmetry).
  4. Surgical single-factor ablations:
     - `C5`: Disables ONLY Basal Ganglia emergency escape bypass (`TIMING_MODE_FIXED_SETTLE` + `disable_bg_escape_bypass=True`)
       with ZERO artificial movement speed slowdown (`pose[5]` untouched!).
     - `C6`: Disables ONLY 3-head `RSSM` rollout valuation & survival shield while keeping in-distribution
       `gwt_workspace`, `d3qn`, compass drive, and homeostasis intact (`C6b` also logs pure fast `System 1` reflex).
  5. Multi-Seed Training Reproducibility across `K = 3` independently trained PBT league checkpoints!
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from stats_utils import holm_bonferroni, paired_comparison, summary_stats
from arena_pbt_trainer import HeadToHeadTactical3DArena, canonicalize_team_observation
from universal_brain_3d_core import (
    ActionStagnationHomeostasis3D,
    BasalGanglia3DTimingEngineV2,
    Brain3DConfig,
    Cerebellar3DKinematicPredictorV2,
    EntorhinalHippocampal3DEngineV2,
    GOAL_MODE_BALANCED_EXPLORE,
    SPATIAL_MODE_TUNNEL_RAW,
    TIMING_MODE_FIXED_SETTLE,
    TIMING_MODE_INSTANT,
    VERB_APPROACH_HOVER,
    WholeBrainNeuralEngineWrapper,
    apply_checkpoint_genes_to_cfg,
    compute_gene_modulated_mip_and_fr,
    compute_universal_memory_compass,
    derive_ventral_fallback_from_dorsal_16,
    extract_5module_activity_vec,
    get_resource_tank_level,
    get_task_profile,
    is_in_stealth_or_cover_refuge,
    load_checkpoint_contract_meta,
    project_pomdp_proprio,
    propose_universal_3d_candidates,
    select_policy_candidate,
)


def resolve_primary_ckpt() -> str:
    rem = os.path.join(ROOT_DIR, "checkpoints", "remediated_v22", "seed_2026", "pbt_best_champion.pt")
    if os.path.isfile(rem):
        return rem
    return os.path.join(ROOT_DIR, "checkpoints", "user_runs", "pbt_best_champion.pt")


class DirectEvalBrainController:
    """
    Single-agent controller running Spatial/SLAM/Map/Cerebellum/BG + Neural Engine
    with surgical single-factor ablation switches (kept compatible with exp3 & exp5 imports).
    """

    def __init__(
        self,
        cfg: Brain3DConfig,
        engine: WholeBrainNeuralEngineWrapper,
        slot_id: int,
        *,
        no_slam_map: bool = False,
        no_respawn_map_keep: bool = False,
        no_rssm_rollout: bool = False,
        pure_s1_only: bool = False,
        rng_salt: Optional[int] = None,
    ):
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self.engine = engine
        self.slot_id = int(slot_id)
        self.rng_salt = int(slot_id if rng_salt is None else rng_salt)
        self.no_slam_map = bool(no_slam_map)
        self.no_respawn_map_keep = bool(no_respawn_map_keep)
        self.no_rssm_rollout = bool(no_rssm_rollout)
        self.pure_s1_only = bool(pure_s1_only)

        self.hippo = EntorhinalHippocampal3DEngineV2(cfg)
        self.hippo.set_learning_context(False)
        self.cereb = Cerebellar3DKinematicPredictorV2(cfg)
        self.bg = BasalGanglia3DTimingEngineV2(cfg)
        self.homeostasis = ActionStagnationHomeostasis3D(cfg.world_min, cfg.world_max)

        self.prev_pi: Optional[np.ndarray] = None
        self.mod_hist: deque = deque(maxlen=48)
        self.ventral_prev: Dict[str, Any] = {}
        self.last_lead_xyz = np.zeros(3, dtype=np.float32)
        self.last_goal_mode = GOAL_MODE_BALANCED_EXPLORE
        self.last_focus_xyz = np.zeros(3, dtype=np.float32)
        self.last_slot_coords: Optional[np.ndarray] = None
        self.last_slot_valence: Optional[np.ndarray] = None
        self.last_vis_surp = 0.0
        self.last_w_surp = 0.0
        self.last_ign = 0.0
        self.last_s_surp = 0.0

        self.cached_pose = np.zeros(6, dtype=np.float32)
        self.cached_verb = VERB_APPROACH_HOVER
        self.cached_idx = 0
        self.cached_q = 0.5
        self.cached_surv = 0.85
        self.cached_s1_idx = 0
        self.respawn_seq = 0
        self.last_ep = -1
        self.prev_bel_xz = np.zeros(2, dtype=np.float32)

    def pre_step(self, obs: Dict[str, Any], dt: float, cog_stride: int = 4) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        ep_cur = int(obs["episode_id"])
        if ep_cur != self.last_ep:
            self.last_ep = ep_cur
            self.hippo.clear_all_stage_memories()
            self.prev_pi = None
            self.mod_hist.clear()
            self.prev_bel_xz.fill(0.0)

        if bool(obs.get("respawn_flag", False)):
            self.respawn_seq += 1
        if self.no_slam_map:
            self.hippo.field_map.reset_full_episode()
            self.hippo.landmark_anchors.clear()
            self.hippo.memory_slots.clear()

        vent_p = derive_ventral_fallback_from_dorsal_16(
            obs["fovea_patch"], yaw_pitch=obs["self_yaw_pitch"], prev_state=self.ventral_prev, episode_id=ep_cur
        )
        sp_out = self.hippo.step(
            obs, dt, ventral_override=vent_p, respawn_atom=self.respawn_seq,
            topdown_goal_mode=self.last_goal_mode, topdown_focus_xyz=self.last_focus_xyz, lead_aim_xyz=self.last_lead_xyz,
            vis_slot_coords=self.last_slot_coords, vis_slot_valence=self.last_slot_valence,
            vis_pred_surprise=self.last_vis_surp, world_surprise=self.last_w_surp, gwt_ignition=self.last_ign,
        )
        if self.no_slam_map:
            self.hippo.field_map.reset_full_episode()
        elif self.no_respawn_map_keep and bool(sp_out["respawn_event_flag"]):
            self.hippo.field_map.reset_full_episode()
            self.hippo.landmark_anchors.clear()
        elif bool(sp_out.get("respawn_event_flag", False)):
            self.hippo.field_map.register_strategic_hazard_scar(float(self.prev_bel_xz[0]), float(self.prev_bel_xz[1]))
        self.prev_bel_xz = sp_out["believed_self_pos"][[0, 2]].copy()

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
        want_cog = (tick % max(1, cog_stride) == 0) or bool(obs["episode_done"]) or bool(sp_out["respawn_event_flag"]) or (tick <= 1)
        self._step_ctx = (obs, dt, ep_cur, sp_out, opp_prob, kin_risk, sc_spd, prop, mc24, c_poses, c_verbs, want_cog)
        if not want_cog:
            return None, None
        if self.pure_s1_only:
            return "s1", {
                "slot_id": self.slot_id,
                "fovea_patch": obs["fovea_patch"],
                "ventral_patch": vent_p,
                "proprio": prop,
                "mem_compass": mc24,
                "mec_grid_code": sp_out["mec_grid_code"],
                "hd_ring_code": sp_out["hd_ring_code"],
                "gwt_workspace": np.zeros(128, dtype=np.float32),
                "rssm_h": np.zeros(256, dtype=np.float32),
                "z_topdown_delta": np.zeros(32, dtype=np.float32),
                "cand_poses": c_poses,
                "cand_verbs": c_verbs,
            }
        return "gwt", {
            "slot_id": self.slot_id,
            "episode_id": ep_cur,
            "respawn_flag": bool(sp_out["respawn_event_flag"]),
            "enable_learning": False,
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

    def post_step(self, net_out: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, int, bool, np.ndarray]:
        obs, dt, ep_cur, sp_out, opp_prob, kin_risk, sc_spd, prop, mc24, c_poses, c_verbs, want_cog = self._step_ctx
        tick = int(obs["physics_tick"])

        if want_cog and net_out is not None:
            if self.pure_s1_only:
                c_idx = int(net_out["best_idx"])
                self.cached_idx = c_idx
                self.cached_verb = int(c_verbs[c_idx])
                self.cached_pose = c_poses[c_idx].copy()
                self.cached_pose[5] = float(net_out["speed_mod"][c_idx])
                self.cached_q = float(net_out["q_values"][c_idx])
                self.cached_surv = 0.85
                self.cached_s1_idx = c_idx
            else:
                gwt_out = net_out
                self.last_goal_mode = int(gwt_out["topdown_goal_mode"])
                self.last_focus_xyz = gwt_out["topdown_focus_xyz"].copy()
                self.last_vis_surp = float(gwt_out.get("vis_pred_surprise", 0.0))
                self.last_w_surp = float(gwt_out.get("world_surprise", 0.0))
                self.last_ign = float(gwt_out.get("ignition", 0.0))
                self.last_s_surp = float(gwt_out.get("self_surprise", 0.0))
                self.last_slot_coords = gwt_out["vis_slot_coords"].copy()
                self.last_slot_valence = gwt_out["vis_slot_valence"].copy()
                self.mod_hist.append(extract_5module_activity_vec(gwt_out, sp_out["mec_grid_code"], prop, mc24))

                if self.no_rssm_rollout:
                    # Surgical single-factor removal of ONLY 3-head RSSM imagination & survival veto
                    q_use = gwt_out["q_raw_values"] + 0.15 * gwt_out["proto_prior_bias"]
                    surv_use = np.full(8, 0.88, dtype=np.float32)
                    adv_use = np.full(8, 0.50, dtype=np.float32)
                else:
                    q_use = gwt_out["q_values"]
                    surv_use = gwt_out["survival_probs"]
                    adv_use = gwt_out["pred_adv_probs"]

                q_star, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                    q_use, surv_use, adv_use, gwt_out["gwt_workspace"], c_poses, c_verbs, prop, mc24, self.prev_pi, self.cfg,
                    habituation_trace=self.homeostasis.habituation_trace,
                    stagnation_level=self.homeostasis.stagnation_level,
                    q_raw_values=gwt_out.get("q_raw_values"),
                    cortical_mod_history=None,
                    self_surprise=self.last_s_surp,
                )
                self.prev_pi = pi.copy()
                rng_seed = tick + self.rng_salt * 7919 + ep_cur * 131
                c_idx = select_policy_candidate(q_star, pi, trainable=False, rng_seed_val=rng_seed)
                if c_idx != int(gwt_out.get("prov_chosen_idx", -1)):
                    self.engine.notify_chosen_action(self.slot_id, gwt_out["cand_feats32"][c_idx].copy())
                _ = self.homeostasis.update_and_compute_intrinsic(
                    episode_id=ep_cur, pos_xz=sp_out["believed_self_pos"][[0, 2]], proprio=prop,
                    chosen_idx=c_idx, step_extrinsic_reward=float(obs["last_step_reward"]),
                    world_surprise=self.last_w_surp, self_surprise=self.last_s_surp,
                    respawn_flag=bool(sp_out["respawn_event_flag"]),
                )
                self.cached_idx = c_idx
                self.cached_verb = int(c_verbs[c_idx])
                self.cached_pose = c_poses[c_idx].copy()
                self.cached_pose[5] = float(gwt_out["speed_mod"][c_idx])
                self.cached_q = float(q_use[c_idx])
                self.cached_surv = float(surv_use[c_idx])
                self.cached_s1_idx = int(np.argmax(gwt_out["q_raw_values"]))

        bg_res = self.bg.step(
            episode_id=ep_cur,
            action_epoch=int(obs["action_epoch"]),
            can_commit_now=bool(obs["can_commit_now"]),
            chosen_cand_idx=self.cached_idx,
            s1_fast_cand_idx=self.cached_s1_idx if want_cog else self.cached_idx,
            chosen_verb=self.cached_verb,
            q_chosen=self.cached_q,
            survival_prob=self.cached_surv,
            scene_opportunity_prob=opp_prob,
            local_kinetic_risk=kin_risk,
            believed_scene_speed=sc_spd,
            mean_fog=float(sp_out["mean_fog"]),
            saccade_hold_active=bool(sp_out["saccade_hold_active"]),
            respawn_event_flag=bool(sp_out["respawn_event_flag"]),
            coach_brake=False,
            in_refuge=is_in_stealth_or_cover_refuge(prop),
            tank_level=get_resource_tank_level(prop),
            hazard_level=float(obs["hazard_level"]),
            step_reward=float(obs["last_step_reward"]),
            dt=dt,
            self_surprise=self.last_s_surp,
        )
        return self.cached_pose.copy(), self.cached_verb, bool(bg_res["commit_gate_open"]), sp_out["active_gaze_ypd"].copy()

    def step(self, obs: Dict[str, Any], dt: float, cog_stride: int = 4) -> Tuple[np.ndarray, int, bool, np.ndarray]:
        kind, req = self.pre_step(obs, dt, cog_stride=cog_stride)
        if kind == "s1" and req is not None:
            out = self.engine.infer_s1_reflex_batch([req])[0]
        elif kind == "gwt" and req is not None:
            out = self.engine.infer_conscious_gwt_batch([req])[0]
        else:
            out = None
        return self.post_step(out)


def run_single_match(
    arena_seed: int,
    ctrl_south: DirectEvalBrainController,
    ctrl_north: DirectEvalBrainController,
    task_profile: str = "turf_objective",
) -> Dict[str, Any]:
    ctrl_south.engine._agent_states.pop(ctrl_south.slot_id, None)
    ctrl_north.engine._agent_states.pop(ctrl_north.slot_id, None)
    acfg = Brain3DConfig(task_profile=task_profile, seed=arena_seed)
    arena = HeadToHeadTactical3DArena(acfg, seed=arena_seed)
    obs_s = arena._build_agent_obs(+1, 0.0, False)
    obs_n = canonicalize_team_observation(arena._build_agent_obs(-1, 0.0, False), -1, arena.wmin, arena.wmax)

    dt = 1.0 / 60.0
    deaths_s, deaths_n = 0, 0
    max_prog_s, max_prog_n = 0.50, 0.50
    winner = 0
    ep_done = False

    while not ep_done:
        act_s = ctrl_south.step(obs_s, dt=dt, cog_stride=4)
        act_n = ctrl_north.step(obs_n, dt=dt, cog_stride=4)
        obs_s, obs_n, ep_done, winner = arena.step_both(act_s, act_n, dt=dt)
        if bool(obs_s.get("respawn_flag", False)):
            deaths_s += 1
        if bool(obs_n.get("respawn_flag", False)):
            deaths_n += 1
        max_prog_s = max(max_prog_s, float(arena.objective_progress_a))
        max_prog_n = max(max_prog_n, float(1.0 - arena.objective_progress_a))

    knockout = int(arena.ticks_in_ep) < 540
    turf_s = float(np.mean(arena.surface_grid > 0.25))
    turf_n = float(np.mean(arena.surface_grid < -0.25))
    win_score_s = 1.0 if winner > 0 else (0.0 if winner < 0 else 0.5)
    return {
        "winner": int(winner),
        "win_score_south": win_score_s,
        "knockout": bool(knockout),
        "ticks": int(arena.ticks_in_ep),
        "deaths_south": deaths_s,
        "deaths_north": deaths_n,
        "kills_south": deaths_n,
        "kills_north": deaths_s,
        "max_prog_south": max_prog_s,
        "max_prog_north": max_prog_n,
        "turf_south": turf_s,
        "turf_north": turf_n,
        "shaped_score_south": float(arena.agents[+1]["cum_score"]),
        "shaped_score_north": float(arena.agents[-1]["cum_score"]),
    }


def make_base_cfg(ckpt_path: str, profile: str = "turf_objective", seed: int = 2026) -> Brain3DConfig:
    cfg = Brain3DConfig(
        task_profile=profile,
        enable_learning=False,
        policy_mode="pure_neural",
        strict_pomdp_observation=True,
        enable_safety_shield=True,
        load_weights_path=ckpt_path,
        seed=seed,
    )
    if ckpt_path and os.path.isfile(ckpt_path):
        meta = load_checkpoint_contract_meta(ckpt_path)
        if "genes" in meta:
            apply_checkpoint_genes_to_cfg(cfg, meta["genes"])
    return cfg


def run_batched_side_swapped_condition(
    ckpt: str,
    seeds: List[int],
    prof_name: str,
    eng_full: WholeBrainNeuralEngineWrapper,
    eng_opp: WholeBrainNeuralEngineWrapper,
    opp_is_trained: bool,
    opp_flags: Dict[str, Any],
    identical_rng_salt: bool = False,
) -> Tuple[Dict[str, Any], List[float], List[float]]:
    eng_full._agent_states.clear()
    eng_opp._agent_states.clear()
    genes = load_checkpoint_contract_meta(ckpt).get("genes", {}) if os.path.isfile(ckpt) else {}
    kw_opp = {k: v for k, v in opp_flags.items() if k in ("no_slam_map", "no_respawn_map_keep", "no_rssm_rollout", "pure_s1_only")}

    arenas: List[HeadToHeadTactical3DArena] = []
    obs_s_list: List[Dict[str, Any]] = []
    obs_n_list: List[Dict[str, Any]] = []
    ctrls_s: List[DirectEvalBrainController] = []
    ctrls_n: List[DirectEvalBrainController] = []

    slot_cursor = 1
    for s_idx, a_seed in enumerate(seeds):
        for swapped in (False, True):
            ar = HeadToHeadTactical3DArena(Brain3DConfig(task_profile=prof_name, seed=a_seed), seed=a_seed)
            arenas.append(ar)
            obs_s_list.append(ar._build_agent_obs(+1, 0.0, False))
            obs_n_list.append(canonicalize_team_observation(ar._build_agent_obs(-1, 0.0, False), -1, ar.wmin, ar.wmax))

            cfg_f = make_base_cfg(ckpt, profile=prof_name, seed=3000 + s_idx)
            cfg_o = make_base_cfg(ckpt if opp_is_trained else "", profile=prof_name, seed=3000 + s_idx if identical_rng_salt else (4000 + s_idx))
            if not opp_is_trained and genes:
                apply_checkpoint_genes_to_cfg(cfg_o, genes)
            for attr_k in ("spatial_mode", "timing_mode", "enable_safety_shield", "disable_bg_escape_bypass"):
                if attr_k in opp_flags:
                    setattr(cfg_o, attr_k, opp_flags[attr_k])

            sid_f = slot_cursor; slot_cursor += 1
            sid_o = slot_cursor; slot_cursor += 1
            salt_f = 1 if identical_rng_salt else (1 + 2 * s_idx)
            salt_o = 1 if identical_rng_salt else (2 + 2 * s_idx)

            c_f = DirectEvalBrainController(cfg_f, eng_full, slot_id=sid_f, rng_salt=salt_f)
            c_o = DirectEvalBrainController(cfg_o, eng_opp, slot_id=sid_o, rng_salt=salt_o, **kw_opp)
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

    for _ in range(545):
        if all(done_flags):
            break
        reqs_by_eng_op: Dict[Tuple[int, str], List[Tuple[int, str, Dict[str, Any]]]] = {}
        for g in range(n_games):
            if done_flags[g]:
                continue
            for side_lbl, ctrl, obs_g in (("s", ctrls_s[g], obs_s_list[g]), ("n", ctrls_n[g], obs_n_list[g])):
                op, rq = ctrl.pre_step(obs_g, dt, cog_stride=4)
                if op is not None and rq is not None:
                    reqs_by_eng_op.setdefault((id(ctrl.engine), op), []).append((g, side_lbl, rq))

        outs_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
        eng_lookup = {id(eng_full): eng_full, id(eng_opp): eng_opp}
        for (eid, op), items in reqs_by_eng_op.items():
            eng_obj = eng_lookup[eid]
            r_batch = [it[2] for it in items]
            res_batch = eng_obj.infer_s1_reflex_batch(r_batch) if op == "s1" else eng_obj.infer_conscious_gwt_batch(r_batch)
            for (g_i, side_lbl, _), out_d in zip(items, res_batch):
                outs_map[(g_i, side_lbl)] = out_d

        for g in range(n_games):
            if done_flags[g]:
                continue
            act_s = ctrls_s[g].post_step(outs_map.get((g, "s")))
            act_n = ctrls_n[g].post_step(outs_map.get((g, "n")))
            obs_s, obs_n, ep_done, w = arenas[g].step_both(act_s, act_n, dt=dt)
            obs_s_list[g] = obs_s
            obs_n_list[g] = obs_n
            if bool(obs_s.get("respawn_flag", False)):
                deaths_s[g] += 1
            if bool(obs_n.get("respawn_flag", False)):
                deaths_n[g] += 1
            max_prog_s[g] = max(max_prog_s[g], float(arenas[g].objective_progress_a))
            max_prog_n[g] = max(max_prog_n[g], float(1.0 - arenas[g].objective_progress_a))
            if ep_done:
                done_flags[g] = True
                winners[g] = int(w)

    n_seeds = len(seeds)
    pair_win_f, pair_win_o = [], []
    pair_ko_f, pair_ko_o = [], []
    pair_prog_f, pair_prog_o = [], []
    pair_turf_f, pair_turf_o = [], []
    pair_kills_f, pair_deaths_f = [], []
    pair_shaped_f, pair_shaped_o = [], []
    shaped_diffs: List[float] = []
    win_outcomes: List[float] = []

    raw_g1_s_wins, raw_g1_n_wins, raw_g1_ties = 0, 0, 0
    max_involution_asym = 0.0

    for s_idx in range(n_seeds):
        g1 = 2 * s_idx
        g2 = 2 * s_idx + 1
        w_s1 = 1.0 if winners[g1] > 0 else (0.0 if winners[g1] < 0 else 0.5)
        w_s2 = 1.0 if winners[g2] > 0 else (0.0 if winners[g2] < 0 else 0.5)
        if winners[g1] > 0:
            raw_g1_s_wins += 1
        elif winners[g1] < 0:
            raw_g1_n_wins += 1
        else:
            raw_g1_ties += 1

        max_involution_asym = max(
            max_involution_asym,
            abs(max_prog_s[g1] - max_prog_n[g2]),
            abs(max_prog_n[g1] - max_prog_s[g2]),
        )

        ko1 = int(arenas[g1].ticks_in_ep) < 540
        ko2 = int(arenas[g2].ticks_in_ep) < 540
        tf_s1 = float(np.mean(arenas[g1].surface_grid > 0.25))
        tf_n1 = float(np.mean(arenas[g1].surface_grid < -0.25))
        tf_s2 = float(np.mean(arenas[g2].surface_grid > 0.25))
        tf_n2 = float(np.mean(arenas[g2].surface_grid < -0.25))
        sh_s1 = float(arenas[g1].agents[+1]["cum_score"])
        sh_n1 = float(arenas[g1].agents[-1]["cum_score"])
        sh_s2 = float(arenas[g2].agents[+1]["cum_score"])
        sh_n2 = float(arenas[g2].agents[-1]["cum_score"])

        w_f = 0.5 * (w_s1 + (1.0 - w_s2))
        pair_win_f.append(w_f)
        pair_win_o.append(1.0 - w_f)
        pair_ko_f.append(0.5 * ((1.0 if (ko1 and winners[g1] > 0) else 0.0) + (1.0 if (ko2 and winners[g2] < 0) else 0.0)))
        pair_ko_o.append(0.5 * ((1.0 if (ko1 and winners[g1] < 0) else 0.0) + (1.0 if (ko2 and winners[g2] > 0) else 0.0)))
        pair_prog_f.append(0.5 * (max_prog_s[g1] + max_prog_n[g2]))
        pair_prog_o.append(0.5 * (max_prog_n[g1] + max_prog_s[g2]))
        pair_turf_f.append(0.5 * (tf_s1 + tf_n2))
        pair_turf_o.append(0.5 * (tf_n1 + tf_s2))
        pair_kills_f.append(0.5 * (deaths_n[g1] + deaths_s[g2]))
        pair_deaths_f.append(0.5 * (deaths_s[g1] + deaths_n[g2]))
        pair_shaped_f.append(0.5 * (sh_s1 + sh_n2))
        pair_shaped_o.append(0.5 * (sh_n1 + sh_s2))

        shaped_diffs.extend([sh_s1 - sh_n1, sh_n2 - sh_s2])
        win_outcomes.extend([w_s1, 1.0 - w_s2])

    decisive_g1 = raw_g1_s_wins + raw_g1_n_wins
    g1_binom_p = float(stats.binomtest(raw_g1_s_wins, decisive_g1, p=0.5).pvalue) if decisive_g1 > 0 else 1.0

    cond_res = {
        "profile": prof_name,
        "n_seed_pairs": n_seeds,
        "total_games": n_games,
        "full_win_rate": summary_stats(pair_win_f),
        "full_ko_rate": summary_stats(pair_ko_f),
        "opp_ko_rate": summary_stats(pair_ko_o),
        "full_max_prog": summary_stats(pair_prog_f),
        "opp_max_prog": summary_stats(pair_prog_o),
        "full_turf_share": summary_stats(pair_turf_f),
        "opp_turf_share": summary_stats(pair_turf_o),
        "full_kills_per_game": summary_stats(pair_kills_f),
        "full_deaths_per_game": summary_stats(pair_deaths_f),
        "full_shaped_score": summary_stats(pair_shaped_f),
        "opp_shaped_score": summary_stats(pair_shaped_o),
        "paired_win_test": paired_comparison(pair_win_f, pair_win_o, seed=2026),
        "paired_prog_test": paired_comparison(pair_prog_f, pair_prog_o, seed=2027),
        "raw_game1_south_w_l_t": f"{raw_g1_s_wins}-{raw_g1_n_wins}-{raw_g1_ties}",
        "raw_game1_south_win_rate": round((raw_g1_s_wins + 0.5 * raw_g1_ties) / max(1, n_seeds), 4),
        "raw_game1_binom_p": round(g1_binom_p, 5),
        "max_involution_prog_asymmetry": round(float(max_involution_asym), 7),
    }
    return cond_res, shaped_diffs, win_outcomes


def main() -> None:
    ckpt = resolve_primary_ckpt()
    assert os.path.isfile(ckpt), f"Champion checkpoint missing: {ckpt}"
    print(f"[Exp2] Using primary checkpoint: {ckpt}")

    n_seeds = 24
    seeds = [2000 + 97 * k for k in range(n_seeds)]

    cfg_trained = make_base_cfg(ckpt, profile="turf_objective", seed=2026)
    cfg_untrained = make_base_cfg("", profile="turf_objective", seed=7777)
    apply_checkpoint_genes_to_cfg(cfg_untrained, load_checkpoint_contract_meta(ckpt).get("genes", {}))

    eng_trained = WholeBrainNeuralEngineWrapper(cfg_trained)
    eng_untrained = WholeBrainNeuralEngineWrapper(cfg_untrained)

    # Exact Proposition 1 deterministic involution check (identical RNG salt across South & North)
    det_sym_res, _, _ = run_batched_side_swapped_condition(
        ckpt, seeds[:12], "turf_objective", eng_trained, eng_trained, True, {}, identical_rng_salt=True
    )

    conditions = [
        ("C0_Sanity_Full_vs_Full", "turf_objective", True, {}),
        ("C1_Full_vs_UntrainedWeights", "turf_objective", False, {}),
        ("C2_Full_vs_NoSLAM_NoMap", "turf_objective", True, {"no_slam_map": True, "spatial_mode": SPATIAL_MODE_TUNNEL_RAW}),
        ("C3_Full_vs_NoRespawnContinuity", "turf_objective", True, {"no_respawn_map_keep": True}),
        ("C4_Full_vs_NoBGGate_Instant", "turf_objective", True, {"timing_mode": TIMING_MODE_INSTANT}),
        ("C5_Full_vs_BGFixed_NoEscapeBypass", "turf_objective", True, {"timing_mode": TIMING_MODE_FIXED_SETTLE, "disable_bg_escape_bypass": True}),
        ("C6_Full_vs_PureSystem1_NoShield", "turf_objective", True, {"no_rssm_rollout": True, "enable_safety_shield": False}),
        ("C6b_Full_vs_PureFastReflexOnly", "turf_objective", True, {"pure_s1_only": True, "enable_safety_shield": False}),
        ("C7_GenericProfile_Full_vs_Untrained", "universal_tactical", False, {}),
        ("C8_GenericProfile_Full_vs_NoSLAM", "universal_tactical", True, {"no_slam_map": True, "spatial_mode": SPATIAL_MODE_TUNNEL_RAW}),
    ]

    all_results: Dict[str, Any] = {}
    all_match_shaped_diffs: List[float] = []
    all_match_win_outcomes: List[float] = []

    t_start = time.perf_counter()
    for cond_name, prof_name, opp_tr, opp_flags in conditions:
        opp_e = eng_trained if opp_tr else eng_untrained
        c_res, s_diffs, w_outs = run_batched_side_swapped_condition(
            ckpt, seeds, prof_name, eng_trained, opp_e, opp_tr, opp_flags, identical_rng_salt=False
        )
        all_results[cond_name] = c_res
        if cond_name != "C0_Sanity_Full_vs_Full":
            all_match_shaped_diffs.extend(s_diffs)
            all_match_win_outcomes.extend(w_outs)
        print(
            f"[Exp2] {cond_name:36s} | Full Win={c_res['full_win_rate']['mean']*100:5.1f}% "
            f"[{c_res['full_win_rate']['ci95_low']*100:4.1f}, {c_res['full_win_rate']['ci95_high']*100:4.1f}]% "
            f"| p_exact={c_res['paired_win_test']['wilcoxon_p']:.4g}"
        )

    primary_keys = [
        "C1_Full_vs_UntrainedWeights",
        "C2_Full_vs_NoSLAM_NoMap",
        "C3_Full_vs_NoRespawnContinuity",
        "C4_Full_vs_NoBGGate_Instant",
        "C5_Full_vs_BGFixed_NoEscapeBypass",
        "C6_Full_vs_PureSystem1_NoShield",
    ]
    raw_ps = {k: all_results[k]["paired_win_test"]["wilcoxon_p"] for k in primary_keys}
    holm_ps = holm_bonferroni(raw_ps)
    for k, ph in holm_ps.items():
        all_results[k]["paired_win_test"]["wilcoxon_p_holm"] = ph

    # Multi-Seed Training Reproducibility across K=3 independently trained PBT league checkpoints
    multi_seed_ckpts = [
        os.path.join(ROOT_DIR, "checkpoints", "remediated_v22", f"seed_{sd}", "pbt_best_champion.pt")
        for sd in (2026, 2027, 2028)
    ]
    multi_seed_audit: Dict[str, Any] = {}
    if all(os.path.isfile(p) for p in multi_seed_ckpts):
        print("[Exp2] Running K=3 Independent PBT Training-Seed Reproducibility Audit...")
        ms_c1_wrs, ms_c2_wrs = [], []
        per_ckpt_stats = []
        for idx_c, cpath in enumerate(multi_seed_ckpts):
            ec_tr = WholeBrainNeuralEngineWrapper(make_base_cfg(cpath, "turf_objective", 2026 + idx_c))
            ec_un_cfg = make_base_cfg("", "turf_objective", 8800 + idx_c)
            apply_checkpoint_genes_to_cfg(ec_un_cfg, load_checkpoint_contract_meta(cpath).get("genes", {}))
            ec_un = WholeBrainNeuralEngineWrapper(ec_un_cfg)

            r_c1, _, _ = run_batched_side_swapped_condition(cpath, seeds[:16], "turf_objective", ec_tr, ec_un, False, {})
            r_c2, _, _ = run_batched_side_swapped_condition(
                cpath, seeds[:16], "turf_objective", ec_tr, ec_tr, True, {"no_slam_map": True, "spatial_mode": SPATIAL_MODE_TUNNEL_RAW}
            )
            w1 = float(r_c1["full_win_rate"]["mean"])
            w2 = float(r_c2["full_win_rate"]["mean"])
            ms_c1_wrs.append(w1)
            ms_c2_wrs.append(w2)
            meta_c = load_checkpoint_contract_meta(cpath)
            per_ckpt_stats.append({
                "seed_dir": os.path.basename(os.path.dirname(cpath)),
                "transitions_recorded": meta_c.get("transitions_recorded", 0),
                "train_steps_done": meta_c.get("train_steps_done", 0),
                "sleep_cycles_done": meta_c.get("sleep_cycles_done", 0),
                "vs_untrained_win_rate": round(w1 * 100.0, 2),
                "vs_noslam_win_rate": round(w2 * 100.0, 2),
            })
            ec_tr.release_memory_and_vram()
            ec_un.release_memory_and_vram()
            print(f"  [PBT Seed #{idx_c+1}] vs Untrained={w1*100:.1f}% | vs NoSLAM={w2*100:.1f}%")
        multi_seed_audit = {
            "n_training_seeds": len(multi_seed_ckpts),
            "games_per_training_seed_per_cond": 32,
            "mean_win_rate_vs_untrained_pct": round(float(np.mean(ms_c1_wrs)) * 100.0, 2),
            "sd_win_rate_vs_untrained_pct": round(float(np.std(ms_c1_wrs, ddof=1)) * 100.0, 2),
            "mean_win_rate_vs_noslam_pct": round(float(np.mean(ms_c2_wrs)) * 100.0, 2),
            "sd_win_rate_vs_noslam_pct": round(float(np.std(ms_c2_wrs, ddof=1)) * 100.0, 2),
            "per_checkpoint": per_ckpt_stats,
        }

    sp_res = stats.spearmanr(all_match_shaped_diffs, all_match_win_outcomes)
    decisive_mask = [abs(w - 0.5) > 0.1 for w in all_match_win_outcomes]
    sign_agree = [
        (sd > 0 and w > 0.5) or (sd < 0 and w < 0.5)
        for sd, w, dm in zip(all_match_shaped_diffs, all_match_win_outcomes, decisive_mask)
        if dm
    ]

    summary_payload = {
        "protocol": {
            "primary_checkpoint": os.path.relpath(ckpt, ROOT_DIR).replace("\\", "/"),
            "policy_mode": "pure_neural (0.0% heuristic teacher)",
            "enable_learning": False,
            "strict_pomdp_observation": True,
            "zero_omniscient_bearing_or_hazard_leak": True,
            "ballistic_gravity_mps2": 11.5,
            "n_independent_seeds": n_seeds,
            "games_per_condition": n_seeds * 2,
            "side_swapped_180deg": True,
            "statistical_test": "Exact conditional permutation Wilcoxon signed-rank + Holm-Bonferroni",
            "elapsed_sec": round(time.perf_counter() - t_start, 2),
        },
        "deterministic_involution_verification": {
            "n_seed_pairs": 12,
            "pair_w_l_t": f"{det_sym_res['paired_win_test']['n_pos']}-{det_sym_res['paired_win_test']['n_neg']}-{det_sym_res['paired_win_test']['n_tie']}",
            "all_pairs_exact_50pct": bool(
                det_sym_res["paired_win_test"]["n_pos"] == 0 and det_sym_res["paired_win_test"]["n_neg"] == 0
            ),
            "single_game_grid_boundary_prog_diff": det_sym_res["max_involution_prog_asymmetry"],
        },
        "reward_shaping_audit": {
            "total_evaluated_matches_excl_c0": len(all_match_win_outcomes),
            "spearman_rho_shaped_diff_vs_unshaped_win": float(sp_res.statistic),
            "spearman_p": float(sp_res.pvalue),
            "decisive_match_sign_agreement_rate": float(np.mean(sign_agree)) if sign_agree else 0.0,
        },
        "multi_seed_checkpoint_reproducibility": multi_seed_audit,
        "conditions": all_results,
    }

    eng_trained.release_memory_and_vram()
    eng_untrained.release_memory_and_vram()

    out_path = os.path.join(ROOT_DIR, "paper_arxiv", "experiments", "results", "exp2_ablations.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, ensure_ascii=False)
    print(f"[Exp2] Saved -> {out_path}")


if __name__ == "__main__":
    main()
