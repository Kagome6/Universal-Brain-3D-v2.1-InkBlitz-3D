# -*- coding: utf-8 -*-
"""
exp1_systems_and_vram.py — Pillar A: Real-Time 7-Process Microkernel, SeqLock, Parity & VRAM Benchmark
======================================================================================================
Respects Barrier 1 strictly:
  - Phase 1 runs the live 7-process `run_universal_brain_3d_orchestrator` from a Zero-PyTorch parent
    across multiple episodes (verifying live SLAM loop closures, inter-episode Sleep S-1..S-4, and 0 CRC errors).
  - Phase 2 compares the live 7-process asynchronous runtime (`60/24/12 Hz` SeqLock) against the
    deterministic synchronous lockstep evaluator across `N = 8` matched arena seeds.
  - Phase 3 spawns an isolated GPU benchmark subprocess to time P1/P3/P4/P5 step latencies (`p50/p95/p99`),
    count parameters per neural module, and measure CUDA VRAM scaling (`1 / 4 / 8` brains).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import sys
import time
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from stats_utils import paired_comparison, summary_stats


def _resolve_default_ckpt() -> str:
    rem = os.path.join(ROOT_DIR, "checkpoints", "remediated_v22", "seed_2026", "pbt_best_champion.pt")
    if os.path.isfile(rem):
        return rem
    return os.path.join(ROOT_DIR, "checkpoints", "user_runs", "pbt_best_champion.pt")


def _sync_lockstep_worker(out_q: mp.Queue, ckpt_path: str, seeds: List[int]) -> None:
    from universal_brain_3d_core import (
        ActionStagnationHomeostasis3D,
        BasalGanglia3DTimingEngineV2,
        Brain3DConfig,
        Cerebellar3DKinematicPredictorV2,
        EntorhinalHippocampal3DEngineV2,
        StandaloneTactical3DArena,
        WholeBrainNeuralEngineWrapper,
        apply_checkpoint_genes_to_cfg,
        compute_gene_modulated_mip_and_fr,
        compute_universal_memory_compass,
        derive_ventral_fallback_from_dorsal_16,
        get_resource_tank_level,
        is_in_stealth_or_cover_refuge,
        load_checkpoint_contract_meta,
        project_pomdp_proprio,
        propose_universal_3d_candidates,
        select_policy_candidate,
    )

    cfg_base = Brain3DConfig(
        task_profile="turf_objective",
        enable_learning=False,
        policy_mode="pure_neural",
        load_weights_path=ckpt_path if os.path.isfile(ckpt_path) else "",
        seed=2026,
    )
    if os.path.isfile(ckpt_path):
        meta = load_checkpoint_contract_meta(ckpt_path)
        if "genes" in meta:
            apply_checkpoint_genes_to_cfg(cfg_base, meta["genes"])
    eng = WholeBrainNeuralEngineWrapper(cfg_base)

    scores: List[float] = []
    turfs: List[float] = []
    for sd in seeds:
        eng._agent_states.clear()
        scfg = Brain3DConfig(
            task_profile="turf_objective",
            enable_learning=False,
            policy_mode="pure_neural",
            load_weights_path=ckpt_path if os.path.isfile(ckpt_path) else "",
            seed=sd,
        )
        if os.path.isfile(ckpt_path):
            meta = load_checkpoint_contract_meta(ckpt_path)
            if "genes" in meta:
                apply_checkpoint_genes_to_cfg(scfg, meta["genes"])
        arena = StandaloneTactical3DArena(scfg)
        hippo = EntorhinalHippocampal3DEngineV2(scfg)
        hippo.set_learning_context(False)
        cereb = Cerebellar3DKinematicPredictorV2(scfg)
        bg = BasalGanglia3DTimingEngineV2(scfg)
        homeo = ActionStagnationHomeostasis3D(scfg.world_min, scfg.world_max)
        vent_p_st: Dict[str, Any] = {}
        prev_pi = None
        pose = np.zeros(6, dtype=np.float32)
        verb = 1
        commit_ok = True
        gaze = np.array([0.0, -0.15, 6.0], dtype=np.float32)
        c_idx = 0
        s1_idx = 0
        c_q = 0.5
        c_surv = 0.85
        last_goal = 0
        last_foc = np.zeros(3, dtype=np.float32)
        last_lead = np.zeros(3, dtype=np.float32)
        peak_turf = 0.0
        peak_score = -1e9
        rseq = 0

        for tick in range(1, 725):
            obs = arena.step(pose, verb, commit_ok, gaze, dt=1.0 / 60.0)
            peak_score = max(peak_score, float(obs["cumulative_score"]))
            if obs.get("respawn_flag"):
                rseq += 1
            vp = derive_ventral_fallback_from_dorsal_16(obs["fovea_patch"], yaw_pitch=obs["self_yaw_pitch"], prev_state=vent_p_st, episode_id=1)
            sp_out = hippo.step(obs, 1.0 / 60.0, ventral_override=vp, respawn_atom=rseq, topdown_goal_mode=last_goal, topdown_focus_xyz=last_foc, lead_aim_xyz=last_lead)
            peak_turf = max(peak_turf, float(sp_out["friendly_turf_ratio"]))
            lead_xyz, opp_prob, kin_risk, sc_spd = cereb.predict_lead_and_crossfire(sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"])
            last_lead = lead_xyz.copy()
            prop = project_pomdp_proprio(obs["proprio"], sp_out["believed_self_pos"], scfg.world_min, scfg.world_max, scfg)
            mc24 = compute_universal_memory_compass(sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop, hippo.field_map, eng.profile)
            c_poses, c_verbs = propose_universal_3d_candidates(sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop, mc24, hippo.field_map, lead_xyz, sp_out["active_gaze_ypd"], scfg.world_min, scfg.world_max, eng.profile, goal_mode=last_goal)
            if tick % 4 == 1 or bool(obs["episode_done"]) or bool(sp_out["respawn_event_flag"]):
                gw = eng.infer_conscious_gwt_batch([{
                    "slot_id": 0, "episode_id": 1, "respawn_flag": bool(sp_out["respawn_event_flag"]),
                    "enable_learning": False, "fovea_patch": obs["fovea_patch"], "ventral_patch": vp,
                    "proprio": prop, "mem_compass": mc24, "mec_grid_code": sp_out["mec_grid_code"],
                    "hd_ring_code": sp_out["hd_ring_code"], "believed_self_pos": sp_out["believed_self_pos"],
                    "cand_poses": c_poses, "cand_verbs": c_verbs,
                }])[0]
                last_goal = int(gw["topdown_goal_mode"])
                last_foc = gw["topdown_focus_xyz"].copy()
                q_star, pi, _, _, _, _ = compute_gene_modulated_mip_and_fr(
                    gw["q_values"], gw["survival_probs"], gw["pred_adv_probs"], gw["gwt_workspace"],
                    c_poses, c_verbs, prop, mc24, prev_pi, scfg,
                    habituation_trace=homeo.habituation_trace, stagnation_level=homeo.stagnation_level,
                    q_raw_values=gw.get("q_raw_values"), self_surprise=float(gw["self_surprise"]),
                )
                prev_pi = pi.copy()
                c_idx = select_policy_candidate(q_star, pi, trainable=False, rng_seed_val=tick + sd)
                eng.notify_chosen_action(0, gw["cand_feats32"][c_idx].copy())
                _ = homeo.update_and_compute_intrinsic(1, sp_out["believed_self_pos"][[0, 2]], prop, c_idx, float(obs["last_step_reward"]))
                pose = c_poses[c_idx].copy()
                pose[5] = float(gw["speed_mod"][c_idx])
                verb = int(c_verbs[c_idx])
                c_q = float(gw["q_values"][c_idx])
                c_surv = float(gw["survival_probs"][c_idx])
                s1_idx = int(np.argmax(gw["q_raw_values"]))
            bg_res = bg.step(
                episode_id=1, action_epoch=int(obs["action_epoch"]), can_commit_now=bool(obs["can_commit_now"]),
                chosen_cand_idx=c_idx, s1_fast_cand_idx=s1_idx, chosen_verb=verb, q_chosen=c_q, survival_prob=c_surv,
                scene_opportunity_prob=opp_prob, local_kinetic_risk=kin_risk, believed_scene_speed=sc_spd,
                mean_fog=float(sp_out["mean_fog"]), saccade_hold_active=bool(sp_out["saccade_hold_active"]),
                respawn_event_flag=bool(sp_out["respawn_event_flag"]), coach_brake=False,
                in_refuge=is_in_stealth_or_cover_refuge(prop), tank_level=get_resource_tank_level(prop),
                hazard_level=float(obs["hazard_level"]), step_reward=float(obs["last_step_reward"]), dt=1.0 / 60.0,
            )
            commit_ok = bool(bg_res["commit_gate_open"])
            gaze = sp_out["active_gaze_ypd"].copy()
            if bool(obs["episode_done"]):
                break
        scores.append(float(peak_score if peak_score > -1e8 else arena.cumulative_score))
        turfs.append(float(peak_turf))

    eng.release_memory_and_vram()
    out_q.put({"scores": scores, "peak_turfs": turfs})


def _gpu_microbenchmark_worker(out_q: mp.Queue, ckpt_path: str) -> None:
    import torch
    from universal_brain_3d_core import (
        BasalGanglia3DTimingEngineV2,
        Brain3DConfig,
        Cerebellar3DKinematicPredictorV2,
        EntorhinalHippocampal3DEngineV2,
        SharedSingleRegionBus,
        StandaloneTactical3DArena,
        WORLD3D_DTYPE,
        WholeBrainNeuralEngineWrapper,
        compute_oizumi_phi_g_2016,
        compute_universal_memory_compass,
        derive_ventral_fallback_from_dorsal_16,
        extract_5module_activity_vec,
        pack_ub3d_sensory_frame,
        pack_ubmc_motor_command,
        project_pomdp_proprio,
        propose_universal_3d_candidates,
        unpack_ub3d_sensory_frame,
        unpack_ubmc_motor_command,
    )
    from collections import deque

    hw_info = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_logical_cores": os.cpu_count(),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0.0,
    }

    cfg = Brain3DConfig(
        task_profile="turf_objective",
        enable_learning=False,
        policy_mode="pure_neural",
        load_weights_path=ckpt_path if os.path.isfile(ckpt_path) else "",
        seed=2026,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    eng1 = WholeBrainNeuralEngineWrapper(cfg)
    param_counts: Dict[str, int] = {}
    total_trainable_params = 0
    for k, m in eng1.models.items():
        cnt = sum(int(p.numel()) for p in m.parameters())
        param_counts[k] = cnt
        if k != "d3qn_target":
            total_trainable_params += cnt

    arena = StandaloneTactical3DArena(cfg)
    hippo = EntorhinalHippocampal3DEngineV2(cfg)
    cereb = Cerebellar3DKinematicPredictorV2(cfg)
    bg = BasalGanglia3DTimingEngineV2(cfg)
    uid = f"EXP1_{os.getpid()}"
    w_bus = SharedSingleRegionBus(uid, WORLD3D_DTYPE, create=True)
    mod_hist: deque = deque(maxlen=48)
    vent_state: Dict[str, Any] = {}

    p1_ms: List[float] = []
    p3_ms: List[float] = []
    p4_b1_ms: List[float] = []
    p4_b8_ms: List[float] = []
    p5_b1_ms: List[float] = []
    p5_phi_only_ms: List[float] = []

    target_pose = np.zeros(6, dtype=np.float32)
    gaze_ypd = np.array([0.0, -0.15, 6.0], dtype=np.float32)
    verb = 1
    commit_ok = True

    try:
        for step_i in range(550):
            t0 = time.perf_counter_ns()
            obs = arena.step(target_pose, verb, commit_ok, gaze_ypd, dt=1.0 / 60.0)
            pkt = pack_ub3d_sensory_frame(
                frame_seq=step_i + 1,
                timestamp_ns=t0,
                episode_id=int(obs["episode_id"]),
                action_epoch=int(obs["action_epoch"]),
                flags=16 if obs.get("respawn_flag") else 0,
                self_pos=obs["self_pos"],
                self_vel=obs["self_vel"],
                self_yaw_pitch=obs["self_yaw_pitch"],
                step_reward=float(obs["last_step_reward"]),
                hazard_level=float(obs["hazard_level"]),
                proprio=obs["proprio"],
                visible_entities=obs["visible_entities"],
                visible_count=int(obs["visible_count"]),
                fovea_patch=obs["fovea_patch"],
            )
            dec_obs = unpack_ub3d_sensory_frame(pkt)
            assert dec_obs is not None
            w_bus.publish_dict({
                "timestamp_ns": t0,
                "physics_tick": int(obs["physics_tick"]),
                "episode_id": int(obs["episode_id"]),
                "action_epoch": int(obs["action_epoch"]),
                "stage_id": 1,
                "can_commit_now": int(obs["can_commit_now"]),
                "episode_done": int(obs["episode_done"]),
                "is_sleeping": 0,
                "visible_count": int(obs["visible_count"]),
                "measured_fps": 60.0,
                "cumulative_score": float(obs["cumulative_score"]),
                "last_step_reward": float(obs["last_step_reward"]),
                "hazard_level": float(obs["hazard_level"]),
                "self_pos": obs["self_pos"],
                "self_vel": obs["self_vel"],
                "self_yaw_pitch": obs["self_yaw_pitch"],
                "proprio": obs["proprio"],
                "visible_entities": obs["visible_entities"],
                "fovea_patch": obs["fovea_patch"],
                "caregiver_utf8": b"",
            })
            w_rec, _ = w_bus.read_latest()
            mpkt = pack_ubmc_motor_command(
                cmd_seq=step_i + 1,
                episode_id=int(obs["episode_id"]),
                action_epoch=int(obs["action_epoch"]),
                motor_verb=verb,
                commit_gate_open=commit_ok,
                veto_active=False,
                hold_reason_id=0,
                target_pose=target_pose,
                gaze_ypd=gaze_ypd,
                telemetry6=np.zeros(6, dtype=np.float32),
            )
            _ = unpack_ubmc_motor_command(mpkt)
            t1 = time.perf_counter_ns()

            vent_p = derive_ventral_fallback_from_dorsal_16(
                w_rec["fovea_patch"], yaw_pitch=w_rec["self_yaw_pitch"], prev_state=vent_state, episode_id=int(w_rec["episode_id"])
            )
            sp_out = hippo.step(w_rec, 1.0 / 60.0, ventral_override=vent_p)
            lead_xyz, opp_prob, kin_risk, sc_spd = cereb.predict_lead_and_crossfire(
                sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"]
            )
            prop = project_pomdp_proprio(w_rec["proprio"], sp_out["believed_self_pos"], cfg.world_min, cfg.world_max, cfg)
            mc24 = compute_universal_memory_compass(
                sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop, hippo.field_map, eng1.profile
            )
            c_poses, c_verbs = propose_universal_3d_candidates(
                sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"], prop,
                mc24, hippo.field_map, lead_xyz, sp_out["active_gaze_ypd"], cfg.world_min, cfg.world_max, eng1.profile
            )
            bg_res = bg.step(
                episode_id=int(w_rec["episode_id"]),
                action_epoch=int(w_rec["action_epoch"]),
                can_commit_now=bool(w_rec["can_commit_now"]),
                chosen_cand_idx=0,
                s1_fast_cand_idx=0,
                chosen_verb=int(c_verbs[0]),
                q_chosen=0.5,
                survival_prob=0.8,
                scene_opportunity_prob=opp_prob,
                local_kinetic_risk=kin_risk,
                believed_scene_speed=sc_spd,
                mean_fog=float(sp_out["mean_fog"]),
                saccade_hold_active=bool(sp_out["saccade_hold_active"]),
                respawn_event_flag=bool(sp_out["respawn_event_flag"]),
                coach_brake=False,
                in_refuge=False,
                tank_level=0.8,
                hazard_level=float(w_rec["hazard_level"]),
                step_reward=float(w_rec["last_step_reward"]),
                dt=1.0 / 60.0,
            )
            t2 = time.perf_counter_ns()

            gwt_req = {
                "slot_id": 0,
                "episode_id": int(w_rec["episode_id"]),
                "respawn_flag": bool(sp_out["respawn_event_flag"]),
                "enable_learning": False,
                "fovea_patch": w_rec["fovea_patch"].copy(),
                "ventral_patch": vent_p,
                "proprio": prop,
                "mem_compass": mc24,
                "mec_grid_code": sp_out["mec_grid_code"],
                "hd_ring_code": sp_out["hd_ring_code"],
                "believed_self_pos": sp_out["believed_self_pos"],
                "cand_poses": c_poses,
                "cand_verbs": c_verbs,
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter_ns()
            gwt_out = eng1.infer_conscious_gwt_batch([gwt_req])[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t4 = time.perf_counter_ns()
            mod_hist.append(extract_5module_activity_vec(gwt_out, sp_out["mec_grid_code"], prop, mc24))
            _, _ = compute_oizumi_phi_g_2016(mod_hist)
            t5 = time.perf_counter_ns()

            s1_req = {
                "slot_id": 0,
                "fovea_patch": w_rec["fovea_patch"].copy(),
                "ventral_patch": vent_p,
                "proprio": prop,
                "mem_compass": mc24,
                "mec_grid_code": sp_out["mec_grid_code"],
                "hd_ring_code": sp_out["hd_ring_code"],
                "gwt_workspace": gwt_out["gwt_workspace"],
                "rssm_h": gwt_out["rssm_h"],
                "z_topdown_delta": gwt_out["z_topdown_delta"],
                "cand_poses": c_poses,
                "cand_verbs": c_verbs,
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t6 = time.perf_counter_ns()
            s1_out = eng1.infer_s1_reflex_batch([s1_req])[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t7 = time.perf_counter_ns()
            _ = eng1.infer_s1_reflex_batch([s1_req for _ in range(8)])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t8 = time.perf_counter_ns()

            b_idx = int(s1_out["best_idx"])
            target_pose = c_poses[b_idx].copy()
            verb = int(c_verbs[b_idx])
            commit_ok = bool(bg_res["commit_gate_open"])
            gaze_ypd = sp_out["active_gaze_ypd"].copy()
            if bool(obs["episode_done"]):
                arena.reset_episode()

            if step_i >= 50:
                p1_ms.append((t1 - t0) / 1e6)
                p3_ms.append((t2 - t1) / 1e6)
                p5_b1_ms.append((t5 - t3) / 1e6)
                p5_phi_only_ms.append((t5 - t4) / 1e6)
                p4_b1_ms.append((t7 - t6) / 1e6)
                p4_b8_ms.append((t8 - t7) / 1e6)

        vram_1_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
        vram_1_res_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

        extra_engines = []
        vram_4_alloc_mb, vram_8_alloc_mb = 0.0, 0.0
        vram_4_res_mb, vram_8_res_mb = 0.0, 0.0
        for idx_e in range(1, 8):
            ecfg = Brain3DConfig(task_profile="turf_objective", enable_learning=(idx_e < 4), seed=2026 + idx_e)
            e_i = WholeBrainNeuralEngineWrapper(ecfg)
            _ = e_i.infer_conscious_gwt_batch([gwt_req])
            extra_engines.append(e_i)
            if idx_e == 3 and torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_4_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                vram_4_res_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
            if idx_e == 7 and torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_8_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                vram_8_res_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)

        crc_err = int(w_bus.crc_mismatches)
        for e_i in extra_engines:
            e_i.release_memory_and_vram()
        eng1.release_memory_and_vram()
    finally:
        w_bus.close(unlink=True)

    out_q.put({
        "hardware": hw_info,
        "parameters": {
            "per_model": param_counts,
            "total_trainable_params": total_trainable_params,
        },
        "step_latency_ms": {
            "P1_WorldBridge_UB3D_SeqLock": summary_stats(p1_ms),
            "P3_SpatialTiming_SLAM_Map_AStar_BG": summary_stats(p3_ms),
            "P4_System1Reflex_Infer_B1": summary_stats(p4_b1_ms),
            "P4_System1Reflex_Infer_B8": summary_stats(p4_b8_ms),
            "P5_ConsciousGWT_RSSM_And_PhiG": summary_stats(p5_b1_ms),
            "P5_PhiG_15Cut_MM_Only": summary_stats(p5_phi_only_ms),
        },
        "vram_scaling_mb": {
            "1_brain_alloc_mb": vram_1_alloc_mb,
            "1_brain_reserved_mb": vram_1_res_mb,
            "4_brains_alloc_mb": vram_4_alloc_mb,
            "4_brains_reserved_mb": vram_4_res_mb,
            "8_brains_alloc_mb": vram_8_alloc_mb,
            "8_brains_reserved_mb": vram_8_res_mb,
        },
        "seqlock_crc_mismatches_550_steps": crc_err,
    })


def main() -> None:
    assert "torch" not in sys.modules, "Barrier 1: Parent must not import torch!"
    from universal_brain_3d_core import Brain3DConfig, run_universal_brain_3d_orchestrator

    ckpt = _resolve_default_ckpt()
    print(f"[Exp1] 1/3 Running live 7-process orchestrator benchmark (18.0s wall-clock, ckpt={os.path.basename(ckpt)})...")
    orch_cfg = Brain3DConfig(
        task_profile="turf_objective",
        enable_learning=True,
        sleep_epochs_per_episode=2,
        autosave_per_episode=False,
        save_on_shutdown=False,
        policy_mode="pure_neural",
        load_weights_path=ckpt if os.path.isfile(ckpt) else "",
        seed=2026,
    )
    orch_summary = run_universal_brain_3d_orchestrator(
        orch_cfg, env_backend="builtin_3d", duration_sec=18.0
    )
    assert "torch" not in sys.modules, "Barrier 1 violated after orchestrator run!"

    print("[Exp1] 2/3 Running N=8 seed Async 7-Process (60/24/12 Hz) vs. Sync Lockstep (15 Hz) Parity Audit...")
    parity_seeds = [3100 + 73 * k for k in range(8)]
    async_scores: List[float] = []
    async_turfs: List[float] = []
    for sd in parity_seeds:
        acfg = Brain3DConfig(
            task_profile="turf_objective",
            enable_learning=False,
            save_on_shutdown=False,
            policy_mode="pure_neural",
            load_weights_path=ckpt if os.path.isfile(ckpt) else "",
            seed=sd,
        )
        asum = run_universal_brain_3d_orchestrator(acfg, env_backend="builtin_3d", duration_sec=16.0, max_episodes=1)
        async_scores.append(float(asum.get("peak_score", 0.0)))
        async_turfs.append(float(asum.get("peak_friendly_turf_ratio", 0.0)))
    assert "torch" not in sys.modules, "Barrier 1 violated after async parity runs!"

    ctx = mp.get_context("spawn")
    q_sync: mp.Queue = ctx.Queue()
    p_sync = ctx.Process(target=_sync_lockstep_worker, args=(q_sync, ckpt, parity_seeds))
    p_sync.start()
    sync_res = q_sync.get(timeout=180.0)
    p_sync.join(timeout=15.0)

    parity_audit = {
        "n_matched_seeds": len(parity_seeds),
        "async_7proc_score": summary_stats(async_scores),
        "sync_lockstep_score": summary_stats(sync_res["scores"]),
        "paired_score_test": paired_comparison(async_scores, sync_res["scores"]),
        "async_7proc_peak_turf": summary_stats(async_turfs),
        "sync_lockstep_peak_turf": summary_stats(sync_res["peak_turfs"]),
        "paired_turf_test": paired_comparison(async_turfs, sync_res["peak_turfs"]),
    }

    print("[Exp1] 3/3 Running isolated GPU component timing & VRAM scaling benchmark...")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_gpu_microbenchmark_worker, args=(q, ckpt))
    p.start()
    micro_res = q.get(timeout=180.0)
    p.join(timeout=15.0)

    full_out = {
        "checkpoint_used": os.path.relpath(ckpt, ROOT_DIR).replace("\\", "/"),
        "orchestrator_7proc_live": orch_summary,
        "async_7proc_vs_sync_lockstep_parity": parity_audit,
        "parent_torch_imported": ("torch" in sys.modules),
        **micro_res,
    }
    out_path = os.path.join(ROOT_DIR, "paper_arxiv", "experiments", "results", "exp1_systems.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(full_out, fh, indent=2, ensure_ascii=False)
    print(f"[Exp1] Saved -> {out_path}")


if __name__ == "__main__":
    main()
