# -*- coding: utf-8 -*-
"""
train_and_verify_v231_emergent_radio.py — v2.3.1 Vygotskian 2v2 Emergent Protolanguage Bootstrap & Audit
========================================================================================================
Barrier 1 Guarantee: Coordinator strictly maintains `"torch" not in sys.modules`.
All PyTorch/CUDA execution occurs inside a single spawned GPU worker / `PopGPUServer`.

Workflow:
  1. Verifies exact `0.000000` step-0 forward compatibility on `v2.3.0` checkpoints (`apply_zero_gate == 0`).
  2. Runs autonomous (`scaffold_prior_scale = 0.0`) 2v2 self-play continual adaptation on
     `Stage 2: Occluded Twin-Flank` starting from `v2.3.0` (`seed_2026/pbt_best_champion.pt`).
  3. Saves `checkpoints/user_runs/v231_vygotsky_2v2_champion.pt` (+ `.meta.json`).
  4. Runs frozen causal evaluation measuring:
       - Empirical Mutual Information `I(Msg_10bit ; TacticalReferent)` via `LiveRosettaStoneProbe`
       - Discovered Rosetta Stone Lexicon (`◈R`, `◆S`, `▶I` -> highest-PMI empirical meanings)
       - Exact Counterfactual Positive-Listening KL (`Phi_comm` & `Phi_comm^ally` in bits)
       - Symmetric 2v2 `Radio-ON` vs `Radio-Muted` head-to-head evaluation on `Stage 2`.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from arena_pbt_trainer import (
    LocalCpuBrainAgentDriver,
    PBT_ARCHETYPES,
    apply_checkpoint_genes_to_cfg,
    run_population_gpu_server_process,
    step_observation_quad,
)
from brain3d_config_ipc import Brain3DConfig, CONT_POSE_DIM
from task_profiles_3d import VERB_APPROACH_HOVER
from vygotsky_rosetta_2v2 import (
    ALL_2V2_KEYS,
    AgentKey2v2,
    HeadToHead2v2Tactical3DArena,
    LiveRosettaStoneProbe,
)

if "torch" in sys.modules:
    raise RuntimeError("Barrier 1 violation: train_and_verify_v231_emergent_radio.py must NEVER import torch!")


def _isolated_step0_compat_worker(ckpt_path: str, out_q: mp.Queue) -> None:
    """Spawned single-GPU check verifying exact 0.000000 step-0 diff on v2.3.0 checkpoints."""
    from brain3d_neural_engine import WholeBrainNeuralEngineWrapper

    cfg_on = Brain3DConfig(
        task_profile="turf_objective",
        policy_mode="pure_neural",
        scaffold_prior_scale=0.0,
        enable_learning=False,
        enable_vygotsky_protocortex=True,
        load_weights_path=ckpt_path,
        seed=2026,
    )
    cfg_off = Brain3DConfig(
        task_profile="turf_objective",
        policy_mode="pure_neural",
        scaffold_prior_scale=0.0,
        enable_learning=False,
        enable_vygotsky_protocortex=False,
        load_weights_path=ckpt_path,
        seed=2026,
    )
    eng_on = WholeBrainNeuralEngineWrapper(cfg_on)
    eng_off = WholeBrainNeuralEngineWrapper(cfg_off)

    rng = np.random.default_rng(2026)
    req = {
        "slot_id": 0,
        "episode_id": 1,
        "physics_tick": 65,
        "ally_radio_pkt": (3, 11, 5, 60),
        "respawn_flag": False,
        "enable_learning": False,
        "fovea_patch": rng.uniform(0.0, 1.0, size=(16, 16, 4)).astype(np.float32),
        "ventral_patch": rng.uniform(-0.5, 0.5, size=(16, 16, 4)).astype(np.float32),
        "proprio": rng.uniform(0.0, 1.0, size=(16,)).astype(np.float32),
        "mem_compass": rng.uniform(-0.5, 0.5, size=(24,)).astype(np.float32),
        "mec_grid_code": rng.uniform(0.0, 1.0, size=(256,)).astype(np.float32),
        "hd_ring_code": rng.uniform(0.0, 1.0, size=(32,)).astype(np.float32),
        "believed_self_pos": np.array([1.5, 1.5, -4.0], dtype=np.float32),
        "cand_poses": rng.uniform(-0.8, 0.8, size=(8, 6)).astype(np.float32),
        "cand_verbs": np.array([1, 1, 4, 6, 5, 2, 0, 1], dtype=np.int32),
    }
    out_on = eng_on.infer_conscious_gwt_batch([dict(req)])[0]
    out_off = eng_off.infer_conscious_gwt_batch([dict(req)])[0]
    max_q_diff = float(np.max(np.abs(out_on["q_values"] - out_off["q_values"])))
    max_gwt_diff = float(np.max(np.abs(out_on["gwt_workspace"] - out_off["gwt_workspace"])))
    eng_on.release_memory_and_vram()
    eng_off.release_memory_and_vram()
    out_q.put({
        "ok": True,
        "max_q_diff": max_q_diff,
        "max_gwt_diff": max_gwt_diff,
        "emitted_pkt_rsi": list(out_on["emitted_pkt_rsi"]),
        "plan_a_broca_isolated": (not bool(cfg_on.allow_broca_motor_bias)),
    })


def _resolve_seed_checkpoint() -> str:
    candidates = [
        os.path.join(_ROOT_DIR, "checkpoints", "v23_full_study", "seed_2026", "pbt_best_champion.pt"),
        os.path.join(_ROOT_DIR, "checkpoints", "user_runs", "v23_autonomous_champion.pt"),
        os.path.join(_ROOT_DIR, "checkpoints", "user_runs", "pbt_best_champion.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


def run_v231_bootstrap_and_audit(train_episodes: int = 24, eval_pairs: int = 8) -> Dict[str, Any]:
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation in parent!")

    seed_ckpt = _resolve_seed_checkpoint()
    print("=" * 82)
    print("  Universal Brain 3D v2.3.1 — Vygotskian 2v2 Protolanguage Bootstrap & Audit")
    print(f"  Base Checkpoint: {seed_ckpt or '(fresh init)'}")
    print("=" * 82, flush=True)

    ctx = mp.get_context("spawn")

    # 1. Step-0 Exact Backward Compatibility Check
    q_chk = ctx.Queue()
    p_chk = ctx.Process(target=_isolated_step0_compat_worker, args=(seed_ckpt, q_chk))
    p_chk.start()
    step0_res = q_chk.get(timeout=45.0)
    p_chk.join(timeout=10.0)
    assert step0_res["max_q_diff"] <= 1e-6, f"Step-0 Q perturbation too large: {step0_res['max_q_diff']}"
    assert step0_res["max_gwt_diff"] <= 1e-6, f"Step-0 GWT perturbation too large: {step0_res['max_gwt_diff']}"
    print(
        f"[PASS] Phase 1: Step-0 v2.3.0 Backward Compatibility Verified "
        f"(max_q_diff={step0_res['max_q_diff']:.7f}, max_gwt_diff={step0_res['max_gwt_diff']:.7f}, "
        f"Plan-A={step0_res['plan_a_broca_isolated']}).",
        flush=True,
    )

    # 2. Launch Single-GPU PopGPUServer (Slot 0 = Team A Learner, Slot 1 = Team B Sparring Snapshot)
    base_cfg = Brain3DConfig(
        task_profile="turf_objective",
        policy_mode="pure_neural",
        scaffold_prior_scale=0.0,
        default_stage_id=2,  # Stage 2: Occluded Twin-Flank!
        online_update_interval=6,
        continual_anchor_l2=6.0e-5,
        enable_vygotsky_protocortex=True,
        enable_2v2_teammate_radio=True,
        strat_hist_bottleneck_dropout=0.30,
        enable_strategy_kb=False,
        lr_scale=1.25,
        seed=2026,
    )

    stop_ev = ctx.Event()
    abort_ev = ctx.Event()
    gpu_ready = ctx.Event()
    req_q = ctx.Queue(maxsize=128)
    resp_queues = {0: ctx.Queue(maxsize=128)}
    ctrl_q = ctx.Queue(maxsize=32)
    ctrl_resp_q = ctx.Queue(maxsize=32)

    def _gpu_ctrl(cmd: str, pay: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
        ctrl_q.put((cmd, pay))
        return ctrl_resp_q.get(timeout=timeout)

    gpu_proc = ctx.Process(
        target=run_population_gpu_server_process,
        args=(base_cfg, 1, 1, req_q, resp_queues, ctrl_q, ctrl_resp_q, stop_ev, gpu_ready, abort_ev),
        name="PopGPUServerV231",
    )

    out_ckpt = os.path.join(_ROOT_DIR, "checkpoints", "user_runs", "v231_vygotsky_2v2_champion.pt")
    report_json = os.path.join(_ROOT_DIR, "checkpoints", "user_runs", "v231_rosetta_emergence_report.json")
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)

    try:
        gpu_proc.start()
        if not gpu_ready.wait(timeout=35.0):
            raise RuntimeError("PopGPUServerV231 failed to start within 35s!")
        barrier_info = _gpu_ctrl("check_barrier", {})

        if seed_ckpt:
            _gpu_ctrl("load_slot_file", {"slot_id": 0, "path": seed_ckpt})
            _gpu_ctrl("load_slot_file", {"slot_id": 1, "path": seed_ckpt})

        # Instantiate 4 Embodied Zero-GPU Drivers (A1, A2 -> brain_id=0; B1, B2 -> brain_id=1)
        drivers: Dict[AgentKey2v2, LocalCpuBrainAgentDriver] = {}
        for idx_k, k in enumerate(ALL_2V2_KEYS):
            t_sign, _ = k
            d_cfg = Brain3DConfig(**asdict(base_cfg))
            apply_checkpoint_genes_to_cfg(d_cfg, PBT_ARCHETYPES[0])
            d_cfg.scaffold_prior_scale = 0.0
            d_cfg.lr_scale = 1.25
            drivers[k] = LocalCpuBrainAgentDriver(
                d_cfg,
                brain_id=(0 if t_sign > 0 else 1),
                worker_id=0,
                req_q=req_q,
                resp_q=resp_queues[0],
                trainable=(t_sign > 0),
                agent_state_id=idx_k,
                stop_ev=stop_ev,
            )

        arena = HeadToHead2v2Tactical3DArena(base_cfg, seed=2026, mute_radio=False)
        train_probe = LiveRosettaStoneProbe(ema_decay=0.9985)
        dt = 1.0 / 60.0
        wins_a = 0
        wins_b = 0
        draws = 0
        rolling_wins: List[float] = []
        t0_train = time.perf_counter()

        for ep in range(1, int(train_episodes) + 1):
            obs_map = arena.reset_episode()
            acts_map: Dict[AgentKey2v2, Tuple[np.ndarray, int, bool, np.ndarray]] = {
                k: (
                    np.zeros(CONT_POSE_DIM, dtype=np.float32),
                    VERB_APPROACH_HOVER,
                    False,
                    np.array([0.0, -0.15, 6.0], dtype=np.float32),
                )
                for k in ALL_2V2_KEYS
            }
            while True:
                drv_seq = [drivers[k] for k in ALL_2V2_KEYS]
                obs_seq = [obs_map[k] for k in ALL_2V2_KEYS]
                act_list = step_observation_quad(drv_seq, obs_seq, dt, cog_stride=4)
                for k, act_k in zip(ALL_2V2_KEYS, act_list):
                    acts_map[k] = act_k
                    drv_k = drivers[k]
                    tx_ok = arena.submit_radio_packet(
                        k, drv_k.latest_emitted_rsi, sender_ego_ref=drv_k.latest_ego_ref_targets
                    )
                    if (tx_ok or (arena.physics_tick > 0 and arena.physics_tick % 16 == 0)) and k[0] > 0:
                        train_probe.observe_packet(
                            drv_k.latest_emitted_rsi,
                            drv_k.latest_ego_ref_targets,
                            canonical_pos_xz=obs_map[k]["self_pos"][[0, 2]],
                            comm_kl_bits=drv_k.latest_comm_kl_bits,
                            ally_comm_kl_bits=drv_k.latest_ally_comm_kl_bits,
                        )
                obs_map, done, winner = arena.step_quad(acts_map, dt)
                if done:
                    # Flush terminal transitions on Team A learners
                    term_seq = [obs_map[k] for k in ALL_2V2_KEYS]
                    step_observation_quad(drv_seq, term_seq, dt, cog_stride=1)
                    if winner > 0:
                        wins_a += 1
                        rolling_wins.append(1.0)
                    elif winner < 0:
                        wins_b += 1
                        rolling_wins.append(0.0)
                    else:
                        draws += 1
                        rolling_wins.append(0.5)
                    break

            if ep % 4 == 0:
                slp = _gpu_ctrl("sleep_all_learners", {"epochs": 4})
                recent_wr = float(np.mean(rolling_wins[-4:])) if len(rolling_wins) >= 4 else 0.5
                if recent_wr >= 0.55:
                    st0 = _gpu_ctrl("get_slot_state", {"slot_id": 0})
                    _gpu_ctrl("set_slot_state", {"slot_id": 1, "state": st0["state"]})
                pm = train_probe.compute_metrics()
                print(
                    f"  [2v2 Ep {ep:02d}/{train_episodes:02d}] W-L-D={wins_a}-{wins_b}-{draws} "
                    f"| SleepLoss={slp.get('mean_sleep_loss', 0.0):.4f} "
                    f"| JointMI={pm['joint_msg_mi_bits']:.3f}b SlotMI={pm['sum_slot_mi_bits']:.3f}b "
                    f"| Vocab=({pm['active_vocab_r']},{pm['active_vocab_s']},{pm['active_vocab_i']}) "
                    f"| Phi_comm={pm['mean_comm_kl_bits']:.4f}b (Ally={pm['mean_ally_comm_kl_bits']:.4f}b)",
                    flush=True,
                )

        train_sec = time.perf_counter() - t0_train

        # Sync latest trained Slot 0 onto Slot 1 for symmetric frozen evaluation
        st_final = _gpu_ctrl("get_slot_state", {"slot_id": 0})
        _gpu_ctrl("set_slot_state", {"slot_id": 1, "state": st_final["state"]})
        for k in ALL_2V2_KEYS:
            drivers[k].rebind_brain(0 if k[0] > 0 else 1, "pure_neural", False)

        # 3. Frozen Evaluation: Rosetta Lexicon MI + Radio-ON (Team A) vs Radio-Muted (Team B) on Stage 2
        eval_probe = LiveRosettaStoneProbe(ema_decay=1.0)
        radio_on_wins = 0
        radio_muted_wins = 0
        eval_draws = 0
        score_diffs: List[float] = []

        for p_idx in range(int(eval_pairs)):
            ep_seed = 6000 + p_idx * 37
            for swap_side in (0, 1):
                # swap_side == 0: Team A (+1) has Radio ON, Team B (-1) has Radio Muted
                # swap_side == 1: Team A (+1) has Radio Muted, Team B (-1) has Radio ON (eliminates North/South bias!)
                ev_arena = HeadToHead2v2Tactical3DArena(base_cfg, seed=ep_seed, mute_radio=False)
                obs_map = ev_arena.reset_episode()
                radio_team = +1 if swap_side == 0 else -1

                while True:
                    for k in ALL_2V2_KEYS:
                        if k[0] != radio_team:
                            obs_map[k]["ally_radio_pkt"] = None
                            obs_map[k]["caregiver_text"] = ""
                            obs_map[k]["mute_ally_radio"] = True
                    drv_seq = [drivers[k] for k in ALL_2V2_KEYS]
                    obs_seq = [obs_map[k] for k in ALL_2V2_KEYS]
                    act_list = step_observation_quad(drv_seq, obs_seq, dt, cog_stride=4)
                    acts_map = {}
                    for k, act_k in zip(ALL_2V2_KEYS, act_list):
                        acts_map[k] = act_k
                        drv_k = drivers[k]
                        if k[0] == radio_team:
                            ev_arena.submit_radio_packet(k, drv_k.latest_emitted_rsi)
                            if ev_arena.physics_tick % 4 == 0:
                                eval_probe.observe_packet(
                                    drv_k.latest_emitted_rsi,
                                    drv_k.latest_ego_ref_targets,
                                    canonical_pos_xz=obs_map[k]["self_pos"][[0, 2]],
                                    comm_kl_bits=drv_k.latest_comm_kl_bits,
                                    ally_comm_kl_bits=drv_k.latest_ally_comm_kl_bits,
                                )
                    obs_map, done, winner = ev_arena.step_quad(acts_map, dt)
                    if done:
                        sc_on = 0.5 * (
                            float(ev_arena.agents[(radio_team, 0)]["cum_score"])
                            + float(ev_arena.agents[(radio_team, 1)]["cum_score"])
                        )
                        sc_mut = 0.5 * (
                            float(ev_arena.agents[(-radio_team, 0)]["cum_score"])
                            + float(ev_arena.agents[(-radio_team, 1)]["cum_score"])
                        )
                        score_diffs.append(sc_on - sc_mut)
                        if winner == radio_team:
                            radio_on_wins += 1
                        elif winner == -radio_team:
                            radio_muted_wins += 1
                        else:
                            eval_draws += 1
                        break

        eval_metrics = eval_probe.compute_metrics()
        tot_eval_games = max(1, radio_on_wins + radio_muted_wins + eval_draws)
        radio_win_rate = (radio_on_wins + 0.5 * eval_draws) / float(tot_eval_games)

        # Inspect learned zero-gate magnitudes directly from numpy state_dict (Barrier 1 clean!)
        pl_sd = st_final["state"]["models"].get("protolang", {})
        g_strat = float(np.tanh(np.asarray(pl_sd.get("strat_comm_gate", [0.0]), dtype=np.float64).ravel()[0]))
        g_gwt = float(np.tanh(np.asarray(pl_sd.get("gwt_comm_gate", [0.0]), dtype=np.float64).ravel()[0]))

        ckpt_meta = {
            "version": "2.3.1",
            "architecture": "Vygotskian Dual-Loop 10-bit Protolanguage + 2v2 Narrowband Radio",
            "base_checkpoint": os.path.basename(seed_ckpt) if seed_ckpt else "fresh",
            "stage_id": 2,
            "scaffold_prior_scale": 0.0,
            "train_episodes_2v2": int(train_episodes),
            "rosetta_metrics": eval_metrics,
            "tanh_strat_comm_gate": round(g_strat, 5),
            "tanh_gwt_comm_gate": round(g_gwt, 5),
            "radio_on_vs_muted_wr": round(radio_win_rate, 4),
            "genes": dict(PBT_ARCHETYPES[0], scaffold_prior_scale=0.0),
        }
        _gpu_ctrl("save_slot", {"slot_id": 0, "path": out_ckpt, "meta": ckpt_meta})

        report = {
            "ok": True,
            "barrier1_parent_zero_torch": ("torch" not in sys.modules),
            "gpu_server_info": barrier_info,
            "step0_compat": step0_res,
            "train_episodes": int(train_episodes),
            "train_wall_sec": round(train_sec, 2),
            "saved_checkpoint": out_ckpt,
            "learned_gates": {
                "tanh_strat_comm_gate": round(g_strat, 5),
                "tanh_gwt_comm_gate": round(g_gwt, 5),
            },
            "frozen_eval_rosetta": eval_metrics,
            "stage2_2v2_radio_ablation": {
                "games": tot_eval_games,
                "radio_on_wins": radio_on_wins,
                "radio_muted_wins": radio_muted_wins,
                "draws": eval_draws,
                "radio_on_win_rate": round(radio_win_rate, 4),
                "mean_team_score_delta": round(float(np.mean(score_diffs)) if score_diffs else 0.0, 3),
            },
        }
        with open(report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

        print("\n" + "=" * 82)
        print("  v2.3.1 Emergent Protolanguage & 2v2 Radio Audit Summary")
        print("=" * 82)
        print(f"  Saved v2.3.1 Checkpoint : {out_ckpt}")
        print(f"  Learned Epistemic Gates : tanh(g_strat)={g_strat:+.4f}, tanh(g_gwt)={g_gwt:+.4f}")
        print(
            f"  Frozen Rosetta MI       : Joint={eval_metrics['joint_msg_mi_bits']:.3f} bits | "
            f"SlotSum={eval_metrics['sum_slot_mi_bits']:.3f} bits "
            f"(R={eval_metrics['mi_r_bits']:.3f}, S={eval_metrics['mi_s_bits']:.3f}, I={eval_metrics['mi_i_bits']:.3f})"
        )
        print(
            f"  Active Vocabulary       : R={eval_metrics['active_vocab_r']}/8, "
            f"S={eval_metrics['active_vocab_s']}/16, I={eval_metrics['active_vocab_i']}/8 "
            f"(UniquePackets={eval_metrics['unique_packets']})"
        )
        print(
            f"  Positive Listening KL   : Phi_comm={eval_metrics['mean_comm_kl_bits']:.4f} bits | "
            f"Phi_comm^ally={eval_metrics['mean_ally_comm_kl_bits']:.4f} bits"
        )
        print(
            f"  Stage 2 2v2 Radio ON/OFF: {radio_on_wins}-{radio_muted_wins}-{eval_draws} "
            f"(WR={radio_win_rate * 100.0:.1f}%, ScoreDelta={np.mean(score_diffs):+.2f})"
        )
        print("  Top Emerged Rosetta Stone Symbols:")
        for item in eval_metrics["top_lexicon"][:6]:
            print(
                f"    {item['token']}  ➔  {item['meaning_ja']} ({item['meaning_en']}) "
                f"| PMI={item['pmi_bits']:.2f}b, P(ref|tok)={item['confidence'] * 100.0:.0f}%, n={item['count']}"
            )
        print("=" * 82, flush=True)

        for d in drivers.values():
            d.close()
        return report
    finally:
        stop_ev.set()
        gpu_proc.join(timeout=6.0)
        if gpu_proc.is_alive():
            gpu_proc.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    res = run_v231_bootstrap_and_audit(train_episodes=24, eval_pairs=8)
    assert res["barrier1_parent_zero_torch"] is True
    assert res["frozen_eval_rosetta"]["unique_packets"] >= 3, "Emergent codebook collapsed!"
    assert (
        abs(res["learned_gates"]["tanh_strat_comm_gate"]) > 1e-4
        or abs(res["learned_gates"]["tanh_gwt_comm_gate"]) > 1e-4
    ), "Epistemic communication gates failed to open!"
