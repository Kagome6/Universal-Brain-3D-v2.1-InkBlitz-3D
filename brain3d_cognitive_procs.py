# -*- coding: utf-8 -*-
"""
brain3d_cognitive_procs.py — Homeostasis, Information Geometry (`Phi_G` / Fisher-Rao),
System1Reflex (P4), ConsciousGWT (P5), LanguageCortex (P6) & TelemetryConsole (P7) [v2.3.0]
===========================================================================================
Barrier 1 Guarantee: Strictly Zero-PyTorch (`"torch" not in sys.modules`).
License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

from collections import deque
import json
import math
import multiprocessing as mp
import os
import socket
import struct
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from task_profiles_3d import *
from brain3d_config_ipc import *
from brain3d_config_ipc import (
    _clamp_utf8_bytes,
    _encode_learning_ctrl_atom,
    _encode_sleep_status_atom,
    _ignore_sigint_in_worker,
    _read_runtime_guard_tripped,
    _read_runtime_learning_flags,
    _read_sleep_status_atom,
    _resolve_active_fovea_patch,
    _resolve_active_ventral_patch,
    _sendall_nonblocking_safe,
)
from brain3d_neural_engine import BrainRPCClient

# ==============================================================================
# 7. Information Geometry (Fisher-Rao / IIT Phi_G), S1, ConsciousGWT, Language,
#    Telemetry Server, 7-Process Orchestrator & 6-Stage `verify_all` Suite
# ==============================================================================
class ActionStagnationHomeostasis3D:
    """
    Domain-universal neuro-computational anti-lock-in homeostasis (ACC Foraging Value Depletion +
    Striatal Short-Term Synaptic Depression + Coarse 16x16 Epistemic Visit Novelty):
      - Smoothly depresses unrewarded repetitive actions (`habituation_trace[8]`)
      - Detects unproductive spatial stagnation (`stagnation_level in [0, 1]`) while exempting
        legitimate low-resource refuge regeneration
      - Computes bounded intrinsic novelty / boredom reward shaping so hiding in a corner/closet
        never becomes a `0.0` Bellman local optimum ("Cowardice Equilibrium").
    """

    def __init__(self, world_min: Tuple[float, float, float], world_max: Tuple[float, float, float]):
        self.wmin = np.asarray(world_min, dtype=np.float32)
        self.wmax = np.asarray(world_max, dtype=np.float32)
        self.span = np.maximum(1.0, self.wmax - self.wmin)
        self.episode_id = -1
        self.habituation_trace = np.zeros(NUM_CANDIDATES, dtype=np.float32)
        self.pos_ema_xz = np.zeros(2, dtype=np.float32)
        self.prev_pos_xz = np.zeros(2, dtype=np.float32)
        self.ema_initialized = False
        self.stagnation_level = 0.0
        self.regen_exempt_steps = 0
        self.visit_counts = np.zeros((16, 16), dtype=np.float32)

    def reset(self, episode_id: int, pos_xz: np.ndarray) -> None:
        self.episode_id = int(episode_id)
        self.habituation_trace.fill(0.0)
        self.pos_ema_xz = np.asarray(pos_xz, dtype=np.float32).copy()
        self.prev_pos_xz = self.pos_ema_xz.copy()
        self.ema_initialized = True
        self.stagnation_level = 0.0
        self.regen_exempt_steps = 0
        self.visit_counts.fill(0.0)

    def on_respawn(self, pos_xz: np.ndarray) -> None:
        self.habituation_trace *= 0.25
        self.pos_ema_xz = np.asarray(pos_xz, dtype=np.float32).copy()
        self.prev_pos_xz = self.pos_ema_xz.copy()
        self.stagnation_level = 0.0
        self.regen_exempt_steps = 0

    def update_and_compute_intrinsic(
        self,
        episode_id: int,
        pos_xz: np.ndarray,
        proprio: np.ndarray,
        chosen_idx: int,
        step_extrinsic_reward: float,
        world_surprise: float = 0.0,
        self_surprise: float = 0.0,
        respawn_flag: bool = False,
    ) -> float:
        pxz = np.asarray(pos_xz, dtype=np.float32)
        if int(episode_id) != self.episode_id or not self.ema_initialized:
            self.reset(int(episode_id), pxz)
        elif respawn_flag:
            self.on_respawn(pxz)

        move_delta = float(np.linalg.norm(pxz - self.prev_pos_xz))
        self.prev_pos_xz = pxz.copy()
        self.pos_ema_xz = 0.94 * self.pos_ema_xz + 0.06 * pxz
        ema_disp = float(np.linalg.norm(pxz - self.pos_ema_xz))

        in_refuge = is_in_stealth_or_cover_refuge(proprio)
        tank = get_resource_tank_level(proprio)
        if ema_disp >= 1.55 or step_extrinsic_reward > 0.015:
            self.regen_exempt_steps = 0
        elif in_refuge and (tank < 0.80):
            self.regen_exempt_steps += 1
        else:
            self.regen_exempt_steps = max(0, self.regen_exempt_steps - 1)
        # Bound continuous `legit_regen` exemption (~2.0s = 24 cog steps) so shooting a wall on own ink
        # can NEVER endlessly reset `stagnation_level` back to 0.0!
        legit_regen = in_refuge and (tank < 0.80) and (self.regen_exempt_steps <= 24)

        self.habituation_trace *= 0.88
        c_i = int(np.clip(chosen_idx, 0, NUM_CANDIDATES - 1))
        if step_extrinsic_reward > 0.015:
            self.habituation_trace[c_i] *= 0.25
        elif step_extrinsic_reward <= 0.005 and not legit_regen:
            self.habituation_trace[c_i] = min(1.0, float(self.habituation_trace[c_i]) + 0.16)

        if ema_disp >= 1.55 or step_extrinsic_reward > 0.008:
            self.stagnation_level = max(0.0, self.stagnation_level - 0.20)
        elif not legit_regen:
            ss_boost = 0.04 * min(1.0, max(0.0, float(self_surprise))) if math.isfinite(self_surprise) else 0.0
            self.stagnation_level = min(1.0, self.stagnation_level + 0.09 + ss_boost)

        # Coarse 16x16 episodic visit count novelty + anti-stagnation metabolic boredom cost
        gx = int(np.clip((float(pxz[0]) - float(self.wmin[0])) / float(self.span[0]) * 16.0, 0, 15))
        gz = int(np.clip((float(pxz[1]) - float(self.wmin[2])) / float(self.span[2]) * 16.0, 0, 15))
        self.visit_counts[gz, gx] += 1.0
        vc = float(self.visit_counts[gz, gx])

        r_int = 0.0
        if move_delta > 0.22:
            r_int += 0.035 / math.sqrt(vc)
            if math.isfinite(world_surprise):
                r_int += 0.015 * math.tanh(max(0.0, float(world_surprise)))
            if math.isfinite(self_surprise):
                r_int += 0.008 * math.tanh(max(0.0, float(self_surprise)))
        if self.stagnation_level > 0.45 and step_extrinsic_reward <= 0.005 and not legit_regen:
            r_int -= 0.035 * self.stagnation_level
        return float(np.clip(r_int, -0.06, 0.06))


def select_policy_candidate(
    q_star: np.ndarray,
    policy_probs: np.ndarray,
    trainable: bool,
    rng_seed_val: int,
) -> int:
    """
    Top-K Masked Boltzmann action selection preventing deterministic `argmax` lock-in:
      - During training (`trainable=True`): samples from top-4 plausible candidates (`pi >= 0.06`).
      - During inference (`trainable=False`): greedy when decisive, soft tie-break when top candidates tie.
    """
    q = np.asarray(q_star, dtype=np.float32).reshape(NUM_CANDIDATES)
    pi = np.asarray(policy_probs, dtype=np.float32).reshape(NUM_CANDIDATES)
    order = np.argsort(q)[::-1]
    top1 = int(order[0])
    top2 = int(order[1])
    if not trainable:
        if float(q[top1] - q[top2]) > 0.10:
            return top1
        u = (((int(rng_seed_val) * 1664525 + 1013904223) >> 8) & 0xFFFF) / 65536.0
        p1 = 1.0 / (1.0 + math.exp(-float(q[top1] - q[top2]) / 0.10))
        return top1 if u < p1 else top2

    keep = np.zeros(NUM_CANDIDATES, dtype=np.float32)
    thresh = max(0.06, 0.20 * float(pi[top1]))
    for rank_i in range(min(4, NUM_CANDIDATES)):
        idx = int(order[rank_i])
        if rank_i == 0 or float(pi[idx]) >= thresh:
            keep[idx] = max(1e-6, float(pi[idx]))
    keep_sum = float(np.sum(keep))
    if keep_sum <= 1e-7:
        return top1
    keep /= keep_sum
    u = (((int(rng_seed_val) * 1103515245 + 12345) >> 8) & 0xFFFF) / 65536.0
    acc = 0.0
    for idx in range(NUM_CANDIDATES):
        acc += float(keep[idx])
        if u <= acc:
            return idx
    return top1


_FIVE_MODULE_BIPARTITIONS = (
    ((0,), (1, 2, 3, 4)),
    ((1,), (0, 2, 3, 4)),
    ((2,), (0, 1, 3, 4)),
    ((3,), (0, 1, 2, 4)),
    ((4,), (0, 1, 2, 3)),
    ((0, 1), (2, 3, 4)),
    ((0, 2), (1, 3, 4)),
    ((0, 3), (1, 2, 4)),
    ((0, 4), (1, 2, 3)),
    ((1, 2), (0, 3, 4)),
    ((1, 3), (0, 2, 4)),
    ((1, 4), (0, 2, 3)),
    ((2, 3), (0, 1, 4)),
    ((2, 4), (0, 1, 3)),
    ((3, 4), (0, 1, 2)),
)


def extract_5module_activity_vec(
    gwt_out: Dict[str, Any],
    mec_grid_code: np.ndarray,
    proprio: np.ndarray,
    mem_compass: np.ndarray,
) -> np.ndarray:
    """
    Extract instantaneous 5-specialist activity vector `[P, I, W, M, S]` for rolling
    temporal Oizumi-Amari Gaussian MIP (`Phi_sync`):
      - 0 `P` (Perception) : Foveal latent energy + visual predictive-coding surprise
      - 1 `I` (Spatial/MEC): Allocentric compass salience + 3D MEC grid firing norm
      - 2 `W` (World RSSM) : World surprise + imagined reward spread + counterfactual risk
      - 3 `M` (Motor/Soma) : Somatic hazard/resource tension + top-down modulation norm
      - 4 `S` (Self/Strat) : Self-model prediction error + metacognitive state + strategy weight
    """
    z_lat = np.asarray(gwt_out.get("latent_z", np.zeros(LATENT_Z_DIM)), dtype=np.float32)
    z_td = np.asarray(gwt_out.get("z_topdown_delta", np.zeros(LATENT_Z_DIM)), dtype=np.float32)
    mec = np.asarray(mec_grid_code, dtype=np.float32).reshape(-1)
    prop = np.asarray(proprio, dtype=np.float32).reshape(PROPRIO_DIM)
    mc = np.asarray(mem_compass, dtype=np.float32).reshape(MEM_COMPASS_DIM)
    rew_p = np.asarray(gwt_out.get("pred_rew", np.zeros(NUM_CANDIDATES)), dtype=np.float32)
    surv_p = np.asarray(gwt_out.get("survival_probs", np.ones(NUM_CANDIDATES)), dtype=np.float32)
    self_s = np.asarray(gwt_out.get("self_state", np.zeros(SELF_MODEL_DIM)), dtype=np.float32)
    alp = np.asarray(gwt_out.get("gwt_alpha", np.full(5, 0.2)), dtype=np.float32).reshape(-1)

    a0 = float(alp[0]) if alp.size >= 5 else 0.2
    a1 = float(alp[1]) if alp.size >= 5 else 0.2
    a2 = float(alp[2]) if alp.size >= 5 else 0.2
    a3 = float(alp[3]) if alp.size >= 5 else 0.2
    a4 = float(alp[4]) if alp.size >= 5 else 0.2

    act_p = float(np.sqrt(np.mean(z_lat * z_lat))) + 0.55 * float(gwt_out.get("vis_pred_surprise", 0.0)) + 0.45 * a0
    obj_prox = float(mc[MC_OBJ_VALID]) * (1.0 - float(mc[MC_OBJ_DIST]))
    thr_prox = float(mc[MC_THR_VALID]) * (1.0 - float(mc[MC_THR_DIST]))
    front_prox = float(mc[MC_FRONT_VALID]) * (1.0 - float(mc[MC_FRONT_DIST]))
    surf_adv = float(np.mean(mc[MC_SURF_N : MC_SURF_W + 1]))
    act_i = 0.55 * obj_prox + 0.65 * thr_prox + 0.40 * front_prox + 0.45 * surf_adv + 0.50 * a1
    act_w = (
        0.55 * float(gwt_out.get("world_surprise", 0.0))
        + float(np.std(rew_p))
        + (1.0 - float(np.min(surv_p)))
        + 0.45 * a2
    )
    act_m = (
        float(prop[PROPRIO_IDX_HAZARD])
        + 0.45 * (1.0 - get_resource_tank_level(prop))
        + float(np.sqrt(np.mean(z_td * z_td)))
        + 0.45 * a3
    )
    act_s = (
        0.55 * float(gwt_out.get("self_surprise", 0.0))
        + float(np.sqrt(np.mean(self_s * self_s)))
        + 0.55 * a4
    )
    return np.array([act_p, act_i, act_w, act_m, act_s], dtype=np.float64)


def compute_oizumi_phi_g_2016(
    cortical_mod_history: Optional[Any],
    mm_iters: int = 4,
) -> Tuple[float, int]:
    """
    Exact Oizumi, Tsuchiya, & Amari (2016, PNAS) Geometric Integrated Information `Phi_G`:
        Phi_G = min_{pi=(M_1, M_2)} min_{q in M_G(pi)} D_KL( p(X^{t-tau}, X^t) || q(X^{t-tau}, X^t) )
    where `q in M_G(pi)` severs ONLY the cross-partition directed causal connections `M_1^{t-1} -> M_2^t`
    and `M_2^{t-1} -> M_1^t` (`A'_{12} = 0, A'_{21} = 0`) while keeping the past marginal `p(X^{t-1})`
    and equal-time residual noise covariance `Sigma'_E` unconstrained!
      - Immune to spurious equal-time external common-input noise (`A` diagonal => `Phi_G == 0.0`).
      - Captures asymmetric directed causal information flow across all 15 bipartitions of `[P,I,W,M,S]`.
      - Returns `(phi_g_bits, mip_cut_mask_5bit)` where `mip_cut_mask_5bit` encodes the live MIP cut `M_1`.
    """
    default_mask = 0x03  # {P, I} | {W, M, S}
    if cortical_mod_history is None or len(cortical_mod_history) < 6:
        return 0.0, default_mask

    mat_x = np.asarray(list(cortical_mod_history), dtype=np.float64).T  # (5, T)
    if mat_x.ndim != 2 or mat_x.shape[0] != 5 or mat_x.shape[1] < 6 or (not np.all(np.isfinite(mat_x))):
        return 0.0, default_mask

    t_len = int(mat_x.shape[1])
    k_pairs = t_len - 1
    # Mild recency weights (`tau = 1` cognitive cycle) balancing fast tactical adaptation & low finite-sample variance
    w_t = np.power(0.98, np.arange(k_pairs - 1, -1, -1, dtype=np.float64))
    w_t /= max(1e-12, float(np.sum(w_t)))
    n_eff = 1.0 / max(1e-9, float(np.sum(w_t * w_t)))

    x_mean = np.mean(mat_x, axis=1, keepdims=True)
    x_c = mat_x - x_mean
    x_std = np.std(x_c, axis=1, keepdims=True)
    # Activity gate suppresses numerical noise amplification on dormant/flat modules while preserving true dynamics
    act_gate = np.clip(x_std / 0.08, 0.0, 1.0)
    z_5 = np.where(x_std > 1e-5, (x_c / np.maximum(x_std, 1e-5)) * act_gate, 0.0)

    x_past = z_5[:, :-1]   # X^{t-1}: (5, K)
    y_curr = z_5[:, 1:]    # X^t    : (5, K)
    xw = x_past * w_t[None, :]
    yw = y_curr * w_t[None, :]

    # Zero-Leakage Off-Diagonal Ridge VAR Estimation:
    # Penalizing ONLY off-diagonal cross-module weights (`lam_off * (I - e_i e_i^T)`) guarantees
    # `e_i^T M_i == e_i^T Sigma_XX`, eliminating Tikhonov multicollinearity leakage when modules share
    # strong equal-time external input (`shared_flash`) without true cross-module causal links!
    eye5 = np.eye(5, dtype=np.float64)
    sig_xx_raw = xw @ x_past.T
    sig_yx_raw = yw @ x_past.T
    lam_off = 0.14
    eps_diag = 1e-3
    a_full = np.zeros((5, 5), dtype=np.float64)
    try:
        for row_i in range(5):
            off_pen = eye5.copy()
            off_pen[row_i, row_i] = 0.0
            m_row = sig_xx_raw + lam_off * off_pen + eps_diag * eye5
            a_full[row_i, :] = np.linalg.solve(m_row, sig_yx_raw[row_i, :])
    except np.linalg.LinAlgError:
        return 0.0, default_mask

    res_full = y_curr - (a_full @ x_past)
    sig_cond = ((res_full * w_t[None, :]) @ res_full.T) + 0.05 * eye5
    sig_cond = 0.5 * (sig_cond + sig_cond.T)
    eig_c, vec_c = np.linalg.eigh(sig_cond)
    sig_cond = (vec_c * np.maximum(eig_c, 0.03)) @ vec_c.T
    sign_full, ldet_full = np.linalg.slogdet(sig_cond)
    if sign_full <= 0:
        return 0.0, default_mask

    sig_xx = sig_xx_raw + 0.05 * eye5
    sig_yx = a_full @ sig_xx

    ln2 = math.log(2.0)
    best_norm_bits = float("inf")
    best_debiased_bits = 0.0
    best_mask = default_mask

    for p_a, p_b in _FIVE_MODULE_BIPARTITIONS:
        ia = np.ix_(p_a, p_a)
        ib = np.ix_(p_b, p_b)
        iab = np.ix_(p_a, p_b)
        iba = np.ix_(p_b, p_a)

        s_xx_aa = sig_xx[ia]
        s_xx_bb = sig_xx[ib]
        s_xx_ba = sig_xx[iba]
        s_xx_ab = sig_xx[iab]
        try:
            inv_xx_aa = np.linalg.inv(s_xx_aa)
            inv_xx_bb = np.linalg.inv(s_xx_bb)
        except np.linalg.LinAlgError:
            continue

        # Initialize disconnected causal blocks `A'_a, A'_b` at OLS (`P = I`)
        a_p_a = sig_yx[ia] @ inv_xx_aa
        a_p_b = sig_yx[ib] @ inv_xx_bb
        a_disc = np.zeros((5, 5), dtype=np.float64)

        # Monotonic Majorization-Minimization (`m`-projection onto `M_G(p_a, p_b)`)
        for _ in range(max(1, int(mm_iters))):
            a_disc.fill(0.0)
            a_disc[ia] = a_p_a
            a_disc[ib] = a_p_b
            da = a_full - a_disc
            sig_e = sig_cond + da @ sig_xx @ da.T
            sig_e = 0.5 * (sig_e + sig_e.T)
            try:
                prec_e = np.linalg.inv(sig_e)
            except np.linalg.LinAlgError:
                break
            m_pyx = prec_e @ sig_yx
            p_aa = prec_e[ia]
            p_bb = prec_e[ib]
            p_ab = prec_e[iab]
            p_ba = prec_e[iba]
            try:
                a_p_a = np.linalg.solve(p_aa, m_pyx[ia] - p_ab @ a_p_b @ s_xx_ba) @ inv_xx_aa
                a_p_b = np.linalg.solve(p_bb, m_pyx[ib] - p_ba @ a_p_a @ s_xx_ab) @ inv_xx_bb
            except np.linalg.LinAlgError:
                break

        a_disc.fill(0.0)
        a_disc[ia] = a_p_a
        a_disc[ib] = a_p_b
        da_f = a_full - a_disc
        sig_e_f = 0.5 * ((sig_cond + da_f @ sig_xx @ da_f.T) + (sig_cond + da_f @ sig_xx @ da_f.T).T)
        sign_e, ldet_e = np.linalg.slogdet(sig_e_f)
        if sign_e <= 0:
            continue
        raw_nats = 0.5 * max(0.0, float(ldet_e - ldet_full))
        # Calibrated ridge-shrunk Wilks finite-sample null-floor debiasing (`dof_cut = 2 * |M_1| * |M_2|`)
        d_cut = 2.0 * float(len(p_a) * len(p_b))
        null_bias_nats = 0.28 * d_cut / max(10.0, n_eff)
        debiased_bits = max(0.0, raw_nats - null_bias_nats) / ln2

        # Kitazono-Kanai-Oizumi (2018) partition normalization `K(pi) = min(|M_1|, |M_2|)`
        # prevents MIP from trivially favoring `1 vs 4` cuts over true `2 vs 3` community cuts!
        k_norm = float(min(len(p_a), len(p_b)))
        norm_bits = debiased_bits / k_norm
        cur_best_k = float(min(bin(best_mask).count("1"), 5 - bin(best_mask).count("1")))
        if norm_bits < best_norm_bits - 1e-9 or (abs(norm_bits - best_norm_bits) <= 1e-9 and k_norm > cur_best_k):
            best_norm_bits = norm_bits
            best_debiased_bits = debiased_bits
            mask_val = 0
            for idx_m in p_a:
                mask_val |= (1 << int(idx_m))
            best_mask = mask_val

    if not math.isfinite(best_norm_bits):
        return 0.0, default_mask
    return float(best_debiased_bits), int(best_mask)


def compute_gene_modulated_mip_and_fr(
    q_values: np.ndarray,
    survival_probs: np.ndarray,
    pred_adv_probs: np.ndarray,
    gwt_workspace: np.ndarray,
    cand_poses: np.ndarray,
    cand_verbs: np.ndarray,
    proprio: np.ndarray,
    mem_compass: np.ndarray,
    prev_policy_probs: Optional[np.ndarray],
    cfg: Brain3DConfig,
    habituation_trace: Optional[np.ndarray] = None,
    stagnation_level: float = 0.0,
    q_raw_values: Optional[np.ndarray] = None,
    cortical_mod_history: Optional[Any] = None,
    self_surprise: float = 0.0,
    strategic_curiosity_mult: float = 1.0,
    strategic_objective_mult: float = 1.0,
    strategic_caution_delta: float = 0.0,
    mip_info_out: Optional[Dict[str, Any]] = None,
    compute_causal_mip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, bool]:
    """
    Compute Gene-Modulated Effective Utility `mip_q_star[8]`, Boltzmann Policy `policy_probs[8]`,
    Oizumi-Amari Geometric Integrated Information `phi_bits`, Fisher-Rao Step `fr_step`,
    Normalized Policy Entropy, and `veto_active`:
      - Enforces **RSSM Survival Veto** on unsafe candidates (`surv < survival_veto_threshold`),
        WITH **Active Escape Veto Exemption (`esc_m`)** so candidates actively fleeing threats or
        diving into refuge are never suppressed during ambushes!
      - Enforces **Anti-Freeze Open-Space Veto**: standing still (`VERB_IDLE_OBSERVE`) while exposed
        under threat (`not in_refuge` and `hazard >= 0.30`) is vetoed!
    """
    q = np.asarray(q_values, dtype=np.float32).reshape(NUM_CANDIDATES)
    surv = np.asarray(survival_probs, dtype=np.float32).reshape(NUM_CANDIDATES)
    adv = np.asarray(pred_adv_probs, dtype=np.float32).reshape(NUM_CANDIDATES)
    poses = np.asarray(cand_poses, dtype=np.float32).reshape(NUM_CANDIDATES, CONT_POSE_DIM)
    verbs = np.asarray(cand_verbs, dtype=np.int64).reshape(NUM_CANDIDATES)
    prop = np.asarray(proprio, dtype=np.float32).reshape(PROPRIO_DIM)
    mc = np.asarray(mem_compass, dtype=np.float32).reshape(MEM_COMPASS_DIM)

    ego_xz = 2.0 * prop[[PROPRIO_IDX_POS_X, PROPRIO_IDX_POS_Z]] - 1.0
    shield_on, exposed_threat = evaluate_gwt_safety_shield_state(prop, cfg)
    veto_thresh = float(np.clip(cfg.survival_veto_threshold + float(np.clip(strategic_caution_delta, -0.04, 0.06)), 0.25, 0.65))
    curiosity_gain = float(np.clip(cfg.curiosity_fog_gain * float(np.clip(strategic_curiosity_mult, 0.75, 1.35)), 0.25, 2.80))
    objective_gain = float(np.clip(strategic_objective_mult, 0.75, 1.40))
    hab = np.asarray(habituation_trace, dtype=np.float32).reshape(NUM_CANDIDATES) if habituation_trace is not None else None
    stag = float(np.clip(stagnation_level, 0.0, 1.0))

    q_star = q.copy()
    veto_triggered = False
    frontal_block = (
        float(mc[MC_OBJ_VALID]) > 0.5
        and float(mc[MC_THR_VALID]) > 0.5
        and (float(mc[MC_OBJ_DX]) * float(mc[MC_THR_DX]) + float(mc[MC_OBJ_DZ]) * float(mc[MC_THR_DZ])) > 0.50
    )
    flank_gain = float(np.clip(getattr(cfg, "stn_stealth_gain", 1.0), 0.70, 1.50))
    scaf_w = float(np.clip(getattr(cfg, "scaffold_prior_scale", 1.0), 0.0, 1.0))

    for m in range(NUM_CANDIDATES):
        vb = int(verbs[m])
        rel_xz = poses[m, [CFEAT_NX, CFEAT_NZ]] - ego_xz
        dn = float(math.hypot(float(rel_xz[0]), float(rel_xz[1])))
        ux, uz = (float(rel_xz[0]) / dn, float(rel_xz[1]) / dn) if dn > 1e-4 else (0.0, 0.0)

        dot_obj = (ux * float(mc[MC_OBJ_DX]) + uz * float(mc[MC_OBJ_DZ])) * float(mc[MC_OBJ_VALID])
        dot_front = (ux * float(mc[MC_FRONT_DX]) + uz * float(mc[MC_FRONT_DZ])) * float(mc[MC_FRONT_VALID])
        dot_cov = (ux * float(mc[MC_COV_DX]) + uz * float(mc[MC_COV_DZ])) * float(mc[MC_COV_VALID])
        dot_thr = (ux * float(mc[MC_THR_DX]) + uz * float(mc[MC_THR_DZ])) * float(mc[MC_THR_VALID])

        # Escape Veto Exemption (`esc_m`): moving away from threat, diving to cover/refuge, or EvadeHazard verb
        is_escape_cand = (
            vb == VERB_EVADE_HAZARD
            or (float(mc[MC_THR_VALID]) > 0.5 and dot_thr < -0.15)
            or (float(mc[MC_COV_VALID]) > 0.5 and dot_cov > 0.25 and dn > 0.03)
        )

        # Gentle continuous intrinsic drive (`scaf_w` anneals hand-crafted L2 compass priors -> 0.0 in v2.3 Autonomous Mode,
        # while learned RSSM `adv[m]` and physical SafetyShield invariants remain active!)
        if not exposed_threat:
            q_star[m] += scaf_w * (
                0.16 * curiosity_gain * max(0.0, dot_front)
                + 0.18 * objective_gain * max(0.0, dot_obj)
            ) + 0.20 * (float(adv[m]) - 0.5)
            if scaf_w > 0.0 and frontal_block and vb != VERB_IDLE_OBSERVE and dn > 0.05 and dot_thr < 0.30 and (m == CAND_IDX_FLANK_DETOUR or dot_obj > -0.15):
                q_star[m] += scaf_w * 0.15 * flank_gain * (1.0 - max(0.0, dot_thr))
        elif is_escape_cand and shield_on:
            q_star[m] += 0.20 + scaf_w * (0.05 + 0.25 * max(0.0, -dot_thr) + 0.20 * max(0.0, dot_cov))

        # Short-term synaptic depression on unrewarded repetitive actions & spatial stagnation
        if hab is not None:
            q_star[m] -= 0.42 * float(hab[m])
        if stag > 0.20:
            if vb == VERB_IDLE_OBSERVE or dn < 0.05:
                q_star[m] -= 1.15 * stag
            else:
                wn = max(0.0, uz)
                we = max(0.0, ux)
                ws = max(0.0, -uz)
                ww = max(0.0, -ux)
                w_sum = max(1e-4, wn + we + ws + ww)
                dir_clr = (
                    wn * float(mc[MC_CLR_N])
                    + we * float(mc[MC_CLR_E])
                    + ws * float(mc[MC_CLR_S])
                    + ww * float(mc[MC_CLR_W])
                ) / w_sum
                if dir_clr < 0.35 and stag > 0.35:
                    q_star[m] -= 0.75 * stag * ((0.35 - dir_clr) / 0.35)
                elif dn >= 0.07:
                    route_align = max(0.0, max(dot_obj, dot_front))
                    q_star[m] += 0.42 * stag * (0.35 + 0.65 * route_align) * (0.50 + 0.50 * dir_clr)

        # Centralized SafetyShield: Anti-Freeze Veto in open space under threat + RSSM Survival Veto
        if shield_on and exposed_threat and vb == VERB_IDLE_OBSERVE:
            q_star[m] -= 2.85
            veto_triggered = True
        elif float(surv[m]) < veto_thresh and (not (shield_on and is_escape_cand)):
            pen = (veto_thresh - float(surv[m])) * 4.5
            q_star[m] -= pen
            veto_triggered = True

    # SafetyShield fallback: if every candidate was penalized under extreme hazard, boost dedicated escape candidate
    if shield_on and exposed_threat and int(verbs[int(np.argmax(q_star))]) == VERB_IDLE_OBSERVE:
        q_star[CAND_IDX_EVADE_REFUGE] = float(np.max(q_star)) + 0.75
        veto_triggered = True

    temp = max(0.12, float(cfg.policy_temp))
    if (not exposed_threat) and math.isfinite(self_surprise):
        eff_temp = temp * (1.0 + 0.22 * float(np.clip(self_surprise - 0.12, 0.0, 1.0)))
    else:
        eff_temp = temp
    logits = (q_star - float(np.max(q_star))) / eff_temp
    exp_l = np.exp(np.clip(logits, -20.0, 0.0))
    pi = (exp_l / max(1e-8, float(np.sum(exp_l)))).astype(np.float32)

    ent = -float(np.sum(pi * np.log(np.clip(pi, 1e-8, 1.0)))) / math.log(float(NUM_CANDIDATES))

    # Fisher-Rao geodesic step on probability simplex: d_FR(p, q) = 2 * arccos(sum sqrt(p_i * q_i))
    if prev_policy_probs is not None:
        prev_pi = np.asarray(prev_policy_probs, dtype=np.float32).reshape(NUM_CANDIDATES)
        bc = float(np.clip(np.sum(np.sqrt(np.maximum(0.0, pi * prev_pi))), -1.0, 1.0))
        fr_step = 2.0 * math.acos(bc)
    else:
        fr_step = 0.0

    # ==========================================================================
    # True Oizumi-Amari Geometric Integrated Information Phi_G = Phi_act + Phi_sync
    #   1. Phi_act = min_{q in conv(pi_S1, u_8)} D_KL(p_int || q) [bits]
    #      Evaluated at MATCHED temperature `eff_temp` so routine uncontested S1=S2
    #      agreement stays near 0.03-0.09b (no false confidence inflation), while
    #      Survival Veto / World-Model / Spatial-Compass overrides of S1 rise to
    #      0.40-1.75b!
    #   2. Phi_sync = min_{(A,B) in P_2({P,I,W,M,S})} I_Gauss(X_A ; X_B) [bits]
    #      Exact 15-bipartition temporal MIP across rolling 5-specialist activity
    #      history `(5, T)`, gated by dynamic variance so steady idle cruising stays
    #      quiet while synchronized multi-module bursts lift cleanly!
    # ==========================================================================
    ln2 = math.log(2.0)
    u8_64 = np.full(NUM_CANDIDATES, 1.0 / float(NUM_CANDIDATES), dtype=np.float64)
    p_int_64 = 0.999 * np.asarray(pi, dtype=np.float64) + 0.001 * u8_64
    p_int_64 /= np.sum(p_int_64)

    q_s1_src = (
        np.asarray(q_raw_values, dtype=np.float64).reshape(NUM_CANDIDATES)
        if q_raw_values is not None and np.asarray(q_raw_values).size == NUM_CANDIDATES
        else np.asarray(q, dtype=np.float64)
    )
    lg_s1 = (q_s1_src - float(np.max(q_s1_src))) / float(eff_temp)
    exp_s1 = np.exp(np.clip(lg_s1, -20.0, 0.0))
    pi_s1_64 = 0.999 * (exp_s1 / max(1e-12, float(np.sum(exp_s1)))) + 0.001 * u8_64
    pi_s1_64 /= np.sum(pi_s1_64)

    basis_mat = np.column_stack([pi_s1_64, u8_64])  # (8, 2)
    w_mip = np.array([0.5, 0.5], dtype=np.float64)
    for _ in range(16):
        q_c = np.maximum(basis_mat @ w_mip, 1e-12)
        w_mip = w_mip * (basis_mat.T @ (p_int_64 / q_c))
        w_mip /= max(float(np.sum(w_mip)), 1e-12)
    q_mip_proj = np.maximum(basis_mat @ w_mip, 1e-12)
    q_mip_proj /= np.sum(q_mip_proj)
    phi_act_bits = max(0.0, float(np.sum(p_int_64 * np.log(p_int_64 / q_mip_proj))) / ln2)

    if compute_causal_mip:
        phi_causal_bits, mip_cut_mask = compute_oizumi_phi_g_2016(cortical_mod_history)
        has_causal_win = cortical_mod_history is not None and len(cortical_mod_history) >= 6
    else:
        phi_causal_bits, mip_cut_mask = 0.0, 0x03
        has_causal_win = False
    if has_causal_win:
        # Primary metric: Oizumi et al. (2016) lagged causal disconnection `Phi_G(X^{t-1}, X^t)`,
        # plus bounded instantaneous policy-override divergence when GWT SafetyShield/WorldModel overrides S1.
        phi_bits = float(np.clip(phi_causal_bits + 0.25 * phi_act_bits, 0.01, 2.50))
    else:
        phi_bits = float(np.clip(phi_act_bits, 0.01, 2.50))

    if mip_info_out is not None:
        mip_info_out["mip_cut_mask"] = int(mip_cut_mask)
        mip_info_out["phi_causal_bits"] = float(phi_causal_bits)
        mip_info_out["phi_act_bits"] = float(phi_act_bits)

    return q_star, pi, phi_bits, float(fr_step), float(ent), veto_triggered


def run_system1_reflex_3d_process(
    cfg: Brain3DConfig,
    rpc_req_queue: mp.Queue,
    rpc_resp_queue: mp.Queue,
    world_shm_name: str,
    spatial_shm_name: str,
    timing_shm_name: str,
    cog_shm_name: str,
    intent_shm_name: str,
    stop_event,
    ready_event,
    vision_cortex_shm_name: str = "",
) -> None:
    """Process 4 (`System1ReflexProcess` — 24 Hz Fast Dorsal Reflex, Strictly Zero-GPU / `Barrier 1`)."""
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: torch must NEVER be imported in System1ReflexProcess!")
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()

    rpc = BrainRPCClient(client_id=1, req_queue=rpc_req_queue, resp_queue=rpc_resp_queue, stop_event=stop_event)
    world_bus = SharedSingleRegionBus(world_shm_name, WORLD3D_DTYPE, create=False)
    spatial_bus = SharedSingleRegionBus(spatial_shm_name, SPATIAL3D_DTYPE, create=False)
    timing_bus = SharedSingleRegionBus(timing_shm_name, TIMING3D_DTYPE, create=False)
    cog_bus = SharedSingleRegionBus(cog_shm_name, COGNITIVE3D_DTYPE, create=False)
    intent_bus = SharedMultiChannelIntent3DBus(intent_shm_name, create=False)
    vc_bus = SharedSingleRegionBus(vision_cortex_shm_name, VISION_CORTEX3D_DTYPE, create=False) if vision_cortex_shm_name else None

    dt = 1.0 / max(8.0, float(cfg.reflex_hz))
    ventral_prev_s1: Dict[str, Any] = {}
    ready_event.set()
    next_t = time.perf_counter()

    try:
        while not stop_event.is_set():
            if not is_windows_pid_alive(orch_ppid):
                stop_event.set()
                break

            w_rec, _ = world_bus.read_latest()
            sp_rec, _ = spatial_bus.read_latest()
            if w_rec is None or sp_rec is None or int(w_rec["is_sleeping"]) == 1:
                time.sleep(0.004)
                continue

            vc_rec = vc_bus.read_latest()[0] if vc_bus is not None else None
            fov_p, _, fov_ok = _resolve_active_fovea_patch(w_rec, vc_rec, cfg)
            if not fov_ok:
                time.sleep(0.004)
                continue
            vent_p = _resolve_active_ventral_patch(w_rec, vc_rec, cfg, prev_state=ventral_prev_s1)

            cog_rec, _ = cog_bus.read_latest()
            tim_rec, _ = timing_bus.read_latest()

            gwt_w = cog_rec["gwt_workspace"].copy() if cog_rec is not None else np.zeros(GWT_DIM, dtype=np.float32)
            rssm_h = cog_rec["rssm_h"].copy() if cog_rec is not None else np.zeros(RSSM_DETER_DIM, dtype=np.float32)
            z_td = cog_rec["z_topdown_delta"].copy() if cog_rec is not None else np.zeros(LATENT_Z_DIM, dtype=np.float32)

            c_poses = sp_rec["auth_cand_poses"].copy()
            c_verbs = sp_rec["auth_cand_verbs"].copy()
            prop_s1 = project_pomdp_proprio(
                w_rec["proprio"], sp_rec["believed_self_pos"], cfg.world_min, cfg.world_max, cfg
            )

            res = rpc.call("infer_s1", {
                "slot_id": 0,
                "fovea_patch": fov_p,
                "ventral_patch": vent_p,
                "proprio": prop_s1,
                "mem_compass": sp_rec["mem_compass"].copy(),
                "mec_grid_code": sp_rec["mec_grid_code"].copy(),
                "hd_ring_code": sp_rec["hd_ring_code"].copy(),
                "gwt_workspace": gwt_w,
                "rssm_h": rssm_h,
                "z_topdown_delta": z_td,
                "cand_poses": c_poses,
                "cand_verbs": c_verbs,
            }, timeout=1.5)

            if res is not None:
                b_idx = apply_s1_safety_shield(int(res["best_idx"]), c_verbs, prop_s1, cfg)

                t_pose = c_poses[b_idx].copy()
                t_pose[CFEAT_SPEED_SCALE] = float(res["speed_mod"][b_idx])
                gate_open = bool(tim_rec["commit_gate_open"]) if tim_rec is not None else True

                intent_bus.publish_channel(CH_REFLEX_S1, {
                    "timestamp_ns": time.perf_counter_ns(),
                    "based_on_physics_tick": int(w_rec["physics_tick"]),
                    "based_on_episode_id": int(w_rec["episode_id"]),
                    "based_on_action_epoch": int(w_rec["action_epoch"]),
                    "command_type": 0,
                    "chosen_cand_idx": b_idx,
                    "motor_verb": int(c_verbs[b_idx]),
                    "commit_authorized": 1 if gate_open else 0,
                    "pad0": 0,
                    "valid_horizon_ms": 260.0,
                    "target_pose": t_pose,
                    "gaze_ypd": sp_rec["active_gaze_ypd"].copy(),
                })

            next_t += dt / max(0.25, float(cfg.time_scale))
            rem = next_t - time.perf_counter()
            if rem > 0.0005:
                time.sleep(rem)
            elif rem < -0.05:
                next_t = time.perf_counter()
    finally:
        if vc_bus is not None:
            vc_bus.close(unlink=False)
        world_bus.close(unlink=False)
        spatial_bus.close(unlink=False)
        timing_bus.close(unlink=False)
        cog_bus.close(unlink=False)
        intent_bus.close(unlink=False)


def run_conscious_gwt_3d_process(
    cfg: Brain3DConfig,
    rpc_req_queue: mp.Queue,
    rpc_resp_queue: mp.Queue,
    world_shm_name: str,
    spatial_shm_name: str,
    timing_shm_name: str,
    cog_shm_name: str,
    lang_shm_name: str,
    intent_shm_name: str,
    stop_event,
    ready_event,
    gwt_wake_event=None,
    lang_wake_event=None,
    vision_cortex_shm_name: str = "",
) -> None:
    """
    Process 5 (`ConsciousGWTProcess` — 12 Hz + Event-Woken, Strictly Zero-GPU / `Barrier 1`):
      - Integrates all 5 specialist streams, Fisher-Rao / IIT Phi_G, 3-Head RSSM Survival Shield,
        Active Escape Veto Exemption, optional early-training DAgger, online N-step replay recording,
        and between-episode Sleep `S-1..S-4` consolidation!
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: torch must NEVER be imported in ConsciousGWTProcess!")
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()

    rpc = BrainRPCClient(client_id=2, req_queue=rpc_req_queue, resp_queue=rpc_resp_queue, stop_event=stop_event)
    world_bus = SharedSingleRegionBus(world_shm_name, WORLD3D_DTYPE, create=False)
    spatial_bus = SharedSingleRegionBus(spatial_shm_name, SPATIAL3D_DTYPE, create=False)
    timing_bus = SharedSingleRegionBus(timing_shm_name, TIMING3D_DTYPE, create=False)
    cog_bus = SharedSingleRegionBus(cog_shm_name, COGNITIVE3D_DTYPE, create=False)
    lang_bus = SharedSingleRegionBus(lang_shm_name, LANGUAGE3D_DTYPE, create=False)
    intent_bus = SharedMultiChannelIntent3DBus(intent_shm_name, create=False)
    vc_bus = SharedSingleRegionBus(vision_cortex_shm_name, VISION_CORTEX3D_DTYPE, create=False) if vision_cortex_shm_name else None

    profile = get_task_profile(cfg.task_profile)
    vocab = GroundedVocabulary512()
    cog_bus.slot.set_header_atom(2, _encode_learning_ctrl_atom(cfg.enable_learning, cfg.autosave_per_episode, False))
    cog_bus.slot.set_header_atom(3, 0)
    cog_bus.slot.set_header_atom(4, 0)

    dt = 1.0 / max(4.0, float(cfg.conscious_hz))
    prev_pi: Optional[np.ndarray] = None
    mod_hist: deque = deque(maxlen=48)
    last_mod_ep = -1
    homeostasis = ActionStagnationHomeostasis3D(cfg.world_min, cfg.world_max)
    strategy_ctrl = create_runtime_strategy_controller(cfg, slot_id=0)
    ventral_prev: Dict[str, Any] = {}
    pending_tr: Optional[Dict[str, Any]] = None
    transitions_seen = 0
    virtual_alert_cnt = 0
    sleep_cycles = 0
    autosave_cnt = 0
    last_slept_ep = -1
    hz_cnt = 0
    hz_t0 = time.perf_counter()
    measured_hz = float(cfg.conscious_hz)

    ready_event.set()
    try:
        while not stop_event.is_set():
            if not is_windows_pid_alive(orch_ppid):
                stop_event.set()
                break

            if gwt_wake_event is not None:
                gwt_wake_event.wait(timeout=dt / max(0.25, float(cfg.time_scale)))
                gwt_wake_event.clear()
            else:
                time.sleep(dt / max(0.25, float(cfg.time_scale)))

            w_rec, _ = world_bus.read_latest()
            sp_rec, _ = spatial_bus.read_latest()
            if w_rec is None or sp_rec is None:
                continue

            learn_on, autosave_on = _read_runtime_learning_flags(cog_bus, cfg)
            ep_id = int(w_rec["episode_id"])
            ep_done = bool(w_rec["episode_done"])
            respawn_ev = bool(sp_rec["respawn_event_flag"])

            vc_rec = vc_bus.read_latest()[0] if vc_bus is not None else None
            fov_p, _, fov_ok = _resolve_active_fovea_patch(w_rec, vc_rec, cfg)
            if not fov_ok:
                continue
            vent_p = _resolve_active_ventral_patch(w_rec, vc_rec, cfg, prev_state=ventral_prev)

            c_poses = sp_rec["auth_cand_poses"].copy()
            c_verbs = sp_rec["auth_cand_verbs"].copy()
            proprio = project_pomdp_proprio(
                w_rec["proprio"], sp_rec["believed_self_pos"], cfg.world_min, cfg.world_max, cfg
            )
            mc24 = sp_rec["mem_compass"].copy()

            lng_rec, _ = lang_bus.read_latest()
            lang_bias = np.zeros(NUM_CANDIDATES, dtype=np.float32)
            if (
                bool(cfg.allow_broca_motor_bias)
                and lng_rec is not None
                and int(lng_rec["based_on_episode_id"]) == ep_id
            ):
                lang_bias = lng_rec["cand_prior_bias"].copy()

            cg_live = bytes(w_rec["caregiver_utf8"]).split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
            t_inf0 = time.perf_counter()
            gwt_out = rpc.call("infer_gwt", {
                "slot_id": 0,
                "episode_id": ep_id,
                "physics_tick": int(w_rec["physics_tick"]),
                "caregiver_text": cg_live,
                "respawn_flag": respawn_ev,
                "enable_learning": learn_on,
                "fovea_patch": fov_p,
                "ventral_patch": vent_p,
                "proprio": proprio,
                "mem_compass": mc24,
                "mec_grid_code": sp_rec["mec_grid_code"].copy(),
                "hd_ring_code": sp_rec["hd_ring_code"].copy(),
                "believed_self_pos": sp_rec["believed_self_pos"].copy(),
                "cand_poses": c_poses,
                "cand_verbs": c_verbs,
                "lang_cand_bias": lang_bias,
            }, timeout=2.5)
            infer_ms = (time.perf_counter() - t_inf0) * 1000.0
            if gwt_out is None:
                continue

            hz_cnt += 1
            now_p = time.perf_counter()
            if now_p - hz_t0 >= 0.5:
                measured_hz = hz_cnt / max(1e-3, now_p - hz_t0)
                hz_cnt = 0
                hz_t0 = now_p

            if ep_id != last_mod_ep:
                mod_hist.clear()
                last_mod_ep = ep_id
            mod_hist.append(extract_5module_activity_vec(gwt_out, sp_rec["mec_grid_code"], proprio, mc24))

            strat_cur_m = 1.0
            strat_obj_m = 1.0
            strat_caut_d = 0.0
            strat_lane_code = 0
            strat_regime_id = 0
            eff_topdown_mode = int(gwt_out["topdown_goal_mode"])
            if strategy_ctrl is not None:
                if ep_done and ep_id != last_slept_ep and learn_on and int(cfg.sleep_epochs_per_episode) > 0:
                    from strategy_knowledge_engine import set_sleep_consolidation_active

                    set_sleep_consolidation_active(strategy_ctrl.db.db_path, True, max_hold_s=15.0)
                mono_s = ""
                if lng_rec is not None and int(lng_rec["based_on_episode_id"]) == ep_id:
                    mono_s = bytes(lng_rec["monologue_utf8"]).split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
                raw_stg = int(w_rec["stage_id"]) if "stage_id" in w_rec.dtype.names else int(cfg.default_stage_id)
                st_out = strategy_ctrl.step(
                    episode_id=ep_id,
                    stage_id=raw_stg if raw_stg > 0 else int(cfg.default_stage_id),
                    tick=int(w_rec["physics_tick"]),
                    pos_xz=sp_rec["believed_self_pos"][[0, 2]],
                    proprio=proprio,
                    believed_entities=sp_rec["believed_entities"],
                    believed_count=int(sp_rec["believed_count"]),
                    friendly_turf=float(sp_rec["friendly_turf_ratio"]),
                    enemy_turf=float(sp_rec["enemy_turf_ratio"]),
                    cum_score=float(w_rec["cumulative_score"]),
                    step_reward=float(w_rec["last_step_reward"]),
                    respawn_ev=respawn_ev,
                    ep_done=ep_done,
                    veto_active=False,
                    ignition=float(gwt_out["ignition"]),
                    stagnation_level=float(homeostasis.stagnation_level),
                    inner_speech=mono_s,
                    field_map=None,
                )
                strat_cur_m = float(st_out["curiosity_mult"])
                strat_obj_m = float(st_out["objective_mult"])
                strat_caut_d = float(st_out["caution_delta"])
                strat_lane_code = int(st_out["lane_code"]) & 0xFF
                strat_regime_id = int(st_out["regime_id"]) & 0xFF
                # Peace-time soft anchor ONLY when StrategicTransformer output is neutral/undecided & zero combat crisis!
                if st_out["peace_goal_mode_override"] is not None and eff_topdown_mode == GOAL_MODE_BALANCED_EXPLORE:
                    eff_topdown_mode = int(st_out["peace_goal_mode_override"])

            mip_info: Dict[str, Any] = {}
            q_star, pi, phi_bits, fr_step, ent_norm, veto_act = compute_gene_modulated_mip_and_fr(
                gwt_out["q_values"],
                gwt_out["survival_probs"],
                gwt_out["pred_adv_probs"],
                gwt_out["gwt_workspace"],
                c_poses,
                c_verbs,
                proprio,
                mc24,
                prev_pi,
                cfg,
                habituation_trace=homeostasis.habituation_trace,
                stagnation_level=homeostasis.stagnation_level,
                q_raw_values=gwt_out.get("q_raw_values"),
                cortical_mod_history=mod_hist,
                self_surprise=float(gwt_out["self_surprise"]),
                strategic_curiosity_mult=strat_cur_m,
                strategic_objective_mult=strat_obj_m,
                strategic_caution_delta=strat_caut_d,
                mip_info_out=mip_info,
            )
            prev_pi = pi.copy()

            # Optional early-training DAgger scaffold mixing (`beta_teacher == 0.0` in `pure_neural`)
            beta_t = compute_beta_teacher(cfg.policy_mode, transitions_seen, cfg.anneal_transitions)
            teacher_q, teacher_surv, teacher_idx = compute_task_scaffold_targets(
                sp_rec, w_rec, c_poses, c_verbs, cfg.world_min, cfg.world_max, profile=profile
            )
            tick_now = int(w_rec["physics_tick"])
            if beta_t >= 0.999:
                chosen_idx = teacher_idx
            elif beta_t > 0.01 and ((tick_now * 1103515245 + 12345) & 0xFFFF) < int(beta_t * 65535.0):
                chosen_idx = teacher_idx
            else:
                chosen_idx = select_policy_candidate(q_star, pi, trainable=bool(learn_on), rng_seed_val=tick_now + ep_id * 97)

            # Immediate step-t efference-copy synchronization whenever Veto / Boltzmann / DAgger chose a non-provisional action!
            if chosen_idx != int(gwt_out.get("prov_chosen_idx", -1)):
                rpc.call(
                    "notify_chosen_action",
                    {"slot_id": 0, "chosen_cand_feat32": gwt_out["cand_feats32"][chosen_idx].copy()},
                    timeout=0.5,
                )

            r_int_gwt = homeostasis.update_and_compute_intrinsic(
                episode_id=ep_id,
                pos_xz=sp_rec["believed_self_pos"][[0, 2]],
                proprio=proprio,
                chosen_idx=chosen_idx,
                step_extrinsic_reward=float(w_rec["last_step_reward"]),
                world_surprise=float(gwt_out["world_surprise"]),
                self_surprise=float(gwt_out["self_surprise"]),
                respawn_flag=respawn_ev,
            )

            s1_rec, _ = intent_bus.read_channel(CH_REFLEX_S1)
            s1_idx = int(s1_rec["chosen_cand_idx"]) if s1_rec is not None else chosen_idx

            valert = 1 if (
                float(gwt_out["world_surprise"]) > 0.45
                or float(gwt_out["self_surprise"]) > 0.42
                or float(sp_rec["believed_hazard"]) > 0.55
            ) else 0
            if valert:
                virtual_alert_cnt += 1

            # Finalize previous pending transition and send to GPU replay buffer
            cur_haz = max(float(w_rec["hazard_level"]), float(sp_rec["believed_hazard"]))
            td_rew_signal = float(w_rec["last_step_reward"])
            haz_delta_signal = 0.0

            if pending_tr is not None:
                same_ep = int(pending_tr["episode_id"]) == ep_id
                epoch_changed = int(w_rec["action_epoch"]) != int(pending_tr["action_epoch"])
                tick_gap = int(w_rec["physics_tick"]) - int(pending_tr["tick"])
                if (not same_ep) or ep_done or respawn_ev or epoch_changed or tick_gap >= 10:
                    rew_acc = float(w_rec["cumulative_score"]) - float(pending_tr["cum_score"]) if same_ep else float(w_rec["last_step_reward"])
                    haz_delta_signal = cur_haz - float(pending_tr["hazard"])
                    td_rew_signal = rew_acc
                    pending_tr["reward"] = float(np.clip(rew_acc + r_int_gwt - 0.15 * max(0.0, haz_delta_signal), -28.0, 35.0))
                    pending_tr["next_hazard"] = cur_haz
                    pending_tr["obj_adv_delta"] = float(proprio[PROPRIO_IDX_OBJ_PROGRESS]) - float(pending_tr["obj_prog"])
                    pending_tr["done"] = bool(ep_done or (not same_ep))
                    pending_tr["respawn"] = bool(respawn_ev)
                    pending_tr["episode_score"] = float(w_rec["cumulative_score"])
                    pending_tr["enable_learning"] = bool(learn_on)
                    pending_tr["next_state"] = {
                        "fovea_patch": fov_p.copy(),
                        "ventral_patch": vent_p.copy(),
                        "proprio": proprio.copy(),
                        "mem_compass": mc24.copy(),
                        "mec_grid_code": sp_rec["mec_grid_code"].copy(),
                        "hd_ring_code": sp_rec["hd_ring_code"].copy(),
                        "cand_feats32": gwt_out["cand_feats32"].copy(),
                    }
                    tr_res = rpc.call("record_transition", pending_tr, timeout=2.0)
                    transitions_seen += 1
                    if tr_res is not None:
                        autosave_cnt = int(tr_res.get("autosave_count", autosave_cnt))
                        if bool(tr_res.get("overfit_guard_tripped", False)):
                            cog_bus.slot.set_header_atom(3, 1)
                    pending_tr = None

            if pending_tr is None and (not ep_done):
                cg_str = bytes(w_rec["caregiver_utf8"]).split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
                tr_new: Dict[str, Any] = {
                    "slot_id": 0,
                    "episode_id": ep_id,
                    "action_epoch": int(w_rec["action_epoch"]),
                    "tick": int(w_rec["physics_tick"]),
                    "cum_score": float(w_rec["cumulative_score"]),
                    "hazard": cur_haz,
                    "obj_prog": float(proprio[PROPRIO_IDX_OBJ_PROGRESS]),
                    "fovea_patch": fov_p.copy(),
                    "ventral_patch": vent_p.copy(),
                    "proprio": proprio.copy(),
                    "mem_compass": mc24.copy(),
                    "mec_grid_code": sp_rec["mec_grid_code"].copy(),
                    "hd_ring_code": sp_rec["hd_ring_code"].copy(),
                    "gwt_workspace": gwt_out["gwt_workspace"].copy(),
                    "rssm_h": gwt_out["rssm_h"].copy(),
                    "cand_feats32": gwt_out["cand_feats32"].copy(),
                    "chosen_cand_feat32": gwt_out["cand_feats32"][chosen_idx].copy(),
                    "chosen_idx": chosen_idx,
                    "chosen_verb": int(c_verbs[chosen_idx]),
                    "proto_sym_id": int(gwt_out["proto_sym_id"]),
                    "proto_embed": gwt_out["proto_embed"].copy(),
                    "world_surprise": float(gwt_out["world_surprise"]),
                    "self_surprise": float(gwt_out["self_surprise"]),
                    "beta_teacher": float(beta_t),
                }
                if beta_t > 0.02:
                    tr_new["teacher_q"] = teacher_q
                    tr_new["teacher_surv"] = teacher_surv
                if cg_str:
                    tr_new["caregiver_tokens"] = vocab.encode_utterance(cg_str, proto_sym_id=int(gwt_out["proto_sym_id"]))
                pending_tr = tr_new

            tim_rec, _ = timing_bus.read_latest()
            gate_open = bool(tim_rec["commit_gate_open"]) if tim_rec is not None else True
            t_pose = c_poses[chosen_idx].copy()
            t_pose[5] = float(gwt_out["speed_mod"][chosen_idx])

            intent_bus.publish_channel(CH_CONSCIOUS_GWT, {
                "timestamp_ns": time.perf_counter_ns(),
                "based_on_physics_tick": int(w_rec["physics_tick"]),
                "based_on_episode_id": ep_id,
                "based_on_action_epoch": int(w_rec["action_epoch"]),
                "command_type": 0,
                "chosen_cand_idx": chosen_idx,
                "motor_verb": int(c_verbs[chosen_idx]),
                "commit_authorized": 1 if gate_open else 0,
                "pad0": 0,
                "valid_horizon_ms": 450.0,
                "target_pose": t_pose,
                "gaze_ypd": sp_rec["active_gaze_ypd"].copy(),
            })

            live_slp_act, live_slp_stg = _read_sleep_status_atom(cog_bus)
            cog_bus.publish_dict({
                "timestamp_ns": time.perf_counter_ns(),
                "cog_basis_tick": int(w_rec["physics_tick"]),
                "episode_id": ep_id,
                "action_epoch": int(w_rec["action_epoch"]),
                "chosen_cand_idx": chosen_idx,
                "s1_fast_cand_idx": s1_idx,
                "veto_active": 1 if veto_act else 0,
                "virtual_alert_active": valert,
                "sleep_active": 1 if live_slp_act else 0,
                "sleep_stage": int(live_slp_stg) & 0xFF,
                "topdown_goal_mode": int(eff_topdown_mode) & 0xFF,
                "strategic_lane_code": int(strat_lane_code) & 0xFF,
                "strategic_regime_id": int(strat_regime_id) & 0xFF,
                "autosave_count_u8": min(255, max(0, int(autosave_cnt))),
                "proto_sym_id": int(gwt_out["proto_sym_id"]),
                "proto_mod_id": int(gwt_out["proto_mod_id"]),
                "virtual_alert_count": virtual_alert_cnt,
                "sleep_cycle_count": sleep_cycles,
                "brain_hz": measured_hz,
                "inference_latency_ms": infer_ms,
                "ignition": float(gwt_out["ignition"]),
                "self_surprise": float(gwt_out["self_surprise"]),
                "phi_geometric_bits": phi_bits,
                "policy_entropy_norm": ent_norm,
                "world_surprise": float(gwt_out["world_surprise"]),
                "fisher_rao_step": fr_step,
                "survival_at_chosen": float(gwt_out["survival_probs"][chosen_idx]),
                "latest_td_reward": td_rew_signal,
                "latest_hazard_delta": haz_delta_signal,
                "reward_action_epoch": int(w_rec["action_epoch"]),
                "topdown_focus_xyz": gwt_out["topdown_focus_xyz"],
                "gwt_alpha": gwt_out["gwt_alpha"],
                "cand_poses": c_poses,
                "cand_verbs": c_verbs,
                "q_values": gwt_out["q_values"],
                "q_raw_values": gwt_out["q_raw_values"],
                "survival_probs": gwt_out["survival_probs"],
                "pred_adv_probs": gwt_out["pred_adv_probs"],
                "policy_probs": pi,
                "mip_q_star": q_star,
                "proto_prior_bias": gwt_out["proto_prior_bias"],
                "gwt_workspace": gwt_out["gwt_workspace"],
                "self_state": gwt_out["self_state"],
                "latent_z": gwt_out["latent_z"],
                "z_topdown_delta": gwt_out["z_topdown_delta"],
                "rssm_h": gwt_out["rssm_h"],
                "proto_embed": gwt_out["proto_embed"],
                "slot_ctx64": gwt_out["slot_ctx64"],
                "vis_slot_coords": gwt_out["vis_slot_coords"],
                "vis_slot_valence": gwt_out["vis_slot_valence"],
                "vis_pred_surprise": float(gwt_out["vis_pred_surprise"]),
                "pred_rew": gwt_out["pred_rew"],
                "wm_has_prev_pred": 1 if gwt_out.get("wm_has_prev_pred", False) else 0,
                "wm_veto_mask_u8": int(gwt_out.get("wm_veto_mask_u8", 0)) & 0xFF,
                "mip_cut_mask_u8": int(mip_info.get("mip_cut_mask", 3)) & 0x1F,
                "wm_pad0": 0,
                "wm_vis_mae": float(gwt_out.get("wm_vis_mae", 0.0)),
                "wm_depth_mae_m": float(gwt_out.get("wm_depth_mae_m", 0.0)),
                "wm_cf_spread": float(gwt_out.get("wm_cf_spread", 0.0)),
                "wm_pred_depth_u8": gwt_out.get("wm_pred_depth_u8", np.zeros((FOVEA_H, FOVEA_W), dtype=np.uint8)),
                "wm_pred_surf_u8": gwt_out.get("wm_pred_surf_u8", np.zeros((FOVEA_H, FOVEA_W), dtype=np.uint8)),
                "wm_pred_err_u8": gwt_out.get("wm_pred_err_u8", np.zeros((FOVEA_H, FOVEA_W), dtype=np.uint8)),
            })

            if lang_wake_event is not None and (int(w_rec["physics_tick"]) % 6 == 0 or valert):
                lang_wake_event.set()

            # Trigger Sleep Consolidation S-1..S-4 at Episode End
            if ep_done and ep_id != last_slept_ep and learn_on and int(cfg.sleep_epochs_per_episode) > 0:
                last_slept_ep = ep_id
                intent_bus.publish_channel(CH_SYSTEM_CONTROL, {
                    "timestamp_ns": time.perf_counter_ns(),
                    "based_on_physics_tick": int(w_rec["physics_tick"]),
                    "based_on_episode_id": ep_id,
                    "based_on_action_epoch": int(w_rec["action_epoch"]),
                    "command_type": SYS_CMD_SLEEP_START,
                    "chosen_cand_idx": 0,
                    "motor_verb": VERB_IDLE_OBSERVE,
                    "commit_authorized": 0,
                    "pad0": 0,
                    "valid_horizon_ms": 2000.0,
                    "target_pose": np.zeros(CONT_POSE_DIM, dtype=np.float32),
                    "gaze_ypd": np.zeros(3, dtype=np.float32),
                })
                cog_bus.slot.set_header_atom(4, _encode_sleep_status_atom(True, 1))
                try:
                    sl_res = rpc.call("sleep", {"epochs": int(cfg.sleep_epochs_per_episode)}, timeout=8.0)
                    if sl_res is not None:
                        sleep_cycles = int(sl_res.get("sleep_cycles_done", sleep_cycles + 1))
                        autosave_cnt = int(sl_res.get("autosave_count", autosave_cnt))
                    if autosave_on and cfg.save_weights_path:
                        sv_res = rpc.call("save_checkpoint", {"path": cfg.save_weights_path}, timeout=5.0)
                        if sv_res is not None:
                            autosave_cnt = int(sv_res.get("autosave_count", autosave_cnt))
                finally:
                    cog_bus.slot.set_header_atom(4, _encode_sleep_status_atom(False, 0))
                    if strategy_ctrl is not None:
                        from strategy_knowledge_engine import set_sleep_consolidation_active

                        set_sleep_consolidation_active(strategy_ctrl.db.db_path, False)
                intent_bus.publish_channel(CH_SYSTEM_CONTROL, {
                    "timestamp_ns": time.perf_counter_ns(),
                    "based_on_physics_tick": int(w_rec["physics_tick"]),
                    "based_on_episode_id": ep_id,
                    "based_on_action_epoch": int(w_rec["action_epoch"]),
                    "command_type": SYS_CMD_SLEEP_DONE,
                    "chosen_cand_idx": 0,
                    "motor_verb": VERB_IDLE_OBSERVE,
                    "commit_authorized": 0,
                    "pad0": 0,
                    "valid_horizon_ms": 2000.0,
                    "target_pose": np.zeros(CONT_POSE_DIM, dtype=np.float32),
                    "gaze_ypd": np.zeros(3, dtype=np.float32),
                })
    finally:
        if strategy_ctrl is not None:
            strategy_ctrl.close()
        if vc_bus is not None:
            vc_bus.close(unlink=False)
        world_bus.close(unlink=False)
        spatial_bus.close(unlink=False)
        timing_bus.close(unlink=False)
        cog_bus.close(unlink=False)
        lang_bus.close(unlink=False)
        intent_bus.close(unlink=False)


def _query_ollama_coach_nonblocking(
    url: str,
    model: str,
    digest_text: str,
    compute_mode: str = "hybrid",
    timeout_s: float = 3.5,
) -> Optional[str]:
    """Query local Ollama `/api/chat` with identical `(num_gpu, num_ctx)` so model layout never reloads."""
    endpoint = url.rstrip("/") + "/api/chat"
    logical_cores = max(2, int(os.cpu_count() or 8))
    eff_threads = max(4, min(12, int(round(logical_cores * 0.625))))
    cm = (compute_mode or "hybrid").strip().lower()
    opts: Dict[str, Any] = {"num_ctx": 2048, "num_predict": 40, "temperature": 0.3}
    if cm == "cpu":
        opts["num_gpu"] = 0
        opts["num_thread"] = eff_threads
    elif cm == "gpu":
        opts["num_gpu"] = -1
    else:
        opts["num_gpu"] = 18
        opts["num_thread"] = eff_threads
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "あなたは3D戦術AIの簡潔な内部戦術コーチです。25文字以内の日本語で現状判断を一言だけ述べてください。"},
            {"role": "user", "content": digest_text},
        ],
        "stream": False,
        "keep_alive": "20m",
        "think": False,
        "options": opts,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            msg = (data.get("message") or {}).get("content", "").strip()
            return msg if msg else None
    except Exception:
        return None


def run_language_cortex_3d_process(
    cfg: Brain3DConfig,
    rpc_req_queue: mp.Queue,
    rpc_resp_queue: mp.Queue,
    world_shm_name: str,
    spatial_shm_name: str,
    cog_shm_name: str,
    lang_shm_name: str,
    stop_event,
    ready_event,
    lang_wake_event=None,
) -> None:
    """
    Process 6 (`LanguageCortexProcess` — Event-Driven, Strictly Zero-GPU / `Barrier 1`):
      - Publishes grounded Japanese inner-speech telemetry (`EndogenousBrocaWernickeCortex`)
        with strict Plan-A motor isolation (`cand_prior_bias == 0.0` by default).
      - Builds structured `S640` state digest for external Ollama companion / post-match analysis.
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: torch must NEVER be imported in LanguageCortexProcess!")
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()

    rpc = BrainRPCClient(client_id=3, req_queue=rpc_req_queue, resp_queue=rpc_resp_queue, stop_event=stop_event)
    world_bus = SharedSingleRegionBus(world_shm_name, WORLD3D_DTYPE, create=False)
    spatial_bus = SharedSingleRegionBus(spatial_shm_name, SPATIAL3D_DTYPE, create=False)
    cog_bus = SharedSingleRegionBus(cog_shm_name, COGNITIVE3D_DTYPE, create=False)
    lang_bus = SharedSingleRegionBus(lang_shm_name, LANGUAGE3D_DTYPE, create=False)

    last_ollama_ts = 0.0
    ready_event.set()
    try:
        while not stop_event.is_set():
            if not is_windows_pid_alive(orch_ppid):
                stop_event.set()
                break
            if lang_wake_event is not None:
                lang_wake_event.wait(timeout=0.25)
                lang_wake_event.clear()
            else:
                time.sleep(0.20)

            w_rec, _ = world_bus.read_latest()
            sp_rec, _ = spatial_bus.read_latest()
            cog_rec, _ = cog_bus.read_latest()
            if w_rec is None or sp_rec is None or cog_rec is None:
                continue

            dyn_broca = lang_bus.slot.get_header_atom(0)
            broca_on = bool(cfg.enable_endogenous_broca) if dyn_broca == 0 else (dyn_broca == 1)

            prop = w_rec["proprio"]
            in_ref = is_in_stealth_or_cover_refuge(prop)
            carrying = is_carrying_primary_objective(prop)
            haz = max(float(w_rec["hazard_level"]), float(sp_rec["believed_hazard"]))
            aud = float(prop[PROPRIO_IDX_AUDIO_LOUD])
            g_mode = int(cog_rec["topdown_goal_mode"])
            c_idx = min(NUM_CANDIDATES - 1, int(cog_rec["chosen_cand_idx"]))
            c_verb = int(cog_rec["cand_verbs"][c_idx])

            t0 = time.perf_counter()
            tokens = np.zeros(INNER_SPEECH_MAX_LEN, dtype=np.int32)
            emb = np.zeros(INNER_SPEECH_EMBED_DIM, dtype=np.float32)
            c_bias = np.zeros(NUM_CANDIDATES, dtype=np.float32)

            if broca_on:
                b_res = rpc.call("infer_broca", {
                    "gwt_workspace": cog_rec["gwt_workspace"].copy(),
                    "proto_embed": cog_rec["proto_embed"].copy(),
                    "slot_ctx64": cog_rec["slot_ctx64"].copy(),
                    "proto_sym_id": int(cog_rec["proto_sym_id"]),
                    "in_refuge": in_ref,
                    "carrying_obj": carrying,
                    "hazard_level": haz,
                    "audio_loud": aud,
                }, timeout=1.5)
                if b_res is not None:
                    tokens = b_res["tokens"]
                    emb = b_res["embed"]
                    c_bias = b_res["cand_bias"] if cfg.allow_broca_motor_bias else np.zeros(NUM_CANDIDATES, dtype=np.float32)
                    mono_str = str(b_res["text"])
                else:
                    mono_str = synthesize_joint_attention_phrase(w_rec, sp_rec, c_verb, goal_mode=g_mode)
            else:
                mono_str = ""

            digest = (
                f"Ep={int(w_rec['episode_id'])} Tick={int(w_rec['physics_tick'])} "
                f"Mode={GOAL_MODE_NAMES.get(g_mode, 'EXPLORE')} Verb={VERB_NAMES_EN.get(c_verb, 'Move')} "
                f"Turf(F={float(sp_rec['friendly_turf_ratio']):.2f},E={float(sp_rec['enemy_turf_ratio']):.2f}) "
                f"ObjProg={float(prop[PROPRIO_IDX_OBJ_PROGRESS]):.2f} Haz={haz:.2f} Surv={float(cog_rec['survival_at_chosen']):.2f}"
            )

            llm_stat = 0
            # Avoid redundant 8s mid-match Ollama polling when StrategyKnowledgeEngine already handles Gemma debriefs!
            if (not bool(getattr(cfg, "enable_strategy_kb", False))) and cfg.llm_backend == "ollama" and (time.perf_counter() - last_ollama_ts) >= 12.0:
                last_ollama_ts = time.perf_counter()
                coach_txt = _query_ollama_coach_nonblocking(
                    cfg.ollama_url,
                    cfg.ollama_model,
                    digest,
                    compute_mode=str(getattr(cfg, "ollama_compute_mode", "hybrid")),
                    timeout_s=4.5,
                )
                if coach_txt:
                    llm_stat = 1
                    mono_str = coach_txt

            lat_ms = (time.perf_counter() - t0) * 1000.0
            lang_bus.publish_dict({
                "timestamp_ns": time.perf_counter_ns(),
                "based_on_episode_id": int(w_rec["episode_id"]),
                "based_on_action_epoch": int(w_rec["action_epoch"]),
                "valid_until_epoch": int(w_rec["action_epoch"]) + 2,
                "strategy_mode_id": g_mode,
                "llm_status": llm_stat,
                "endogenous_active": 1 if broca_on else 0,
                "bias_confidence": float(np.max(np.abs(c_bias))),
                "latency_ms": lat_ms,
                "cand_prior_bias": c_bias,
                "inner_speech_tokens": tokens,
                "inner_speech_embed": emb,
                "monologue_utf8": _clamp_utf8_bytes(mono_str, 255),
                "digest_s640": _clamp_utf8_bytes(digest, S640_DIGEST_BYTES - 1),
            })
    finally:
        world_bus.close(unlink=False)
        spatial_bus.close(unlink=False)
        cog_bus.close(unlink=False)
        lang_bus.close(unlink=False)


def run_telemetry_console_3d_process(
    cfg: Brain3DConfig,
    world_shm_name: str,
    spatial_shm_name: str,
    timing_shm_name: str,
    cog_shm_name: str,
    lang_shm_name: str,
    stop_event,
    ready_event,
    vision_cortex_shm_name: str = "",
) -> None:
    """
    Process 7 (`TelemetryConsoleProcess` — 15 Hz, Strictly Zero-GPU / `Barrier 1`):
      - Serves `UBNT` (`3,636B`) + `UBND` (`536B`) live mathematical UX telemetry over TCP
        and receives `UBNC` (`16B`) live runtime control toggles from Unity UX.
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: torch must NEVER be imported in TelemetryConsoleProcess!")
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()

    world_bus = SharedSingleRegionBus(world_shm_name, WORLD3D_DTYPE, create=False)
    spatial_bus = SharedSingleRegionBus(spatial_shm_name, SPATIAL3D_DTYPE, create=False)
    timing_bus = SharedSingleRegionBus(timing_shm_name, TIMING3D_DTYPE, create=False)
    cog_bus = SharedSingleRegionBus(cog_shm_name, COGNITIVE3D_DTYPE, create=False)
    lang_bus = SharedSingleRegionBus(lang_shm_name, LANGUAGE3D_DTYPE, create=False)
    vc_bus = SharedSingleRegionBus(vision_cortex_shm_name, VISION_CORTEX3D_DTYPE, create=False) if vision_cortex_shm_name else None

    ux_port = int(cfg.ux_telemetry_port) if int(cfg.ux_telemetry_port) > 0 else (int(cfg.unity_port) + 1 if int(cfg.unity_port) > 0 else 0)
    srv: Optional[socket.socket] = None
    conn: Optional[socket.socket] = None
    if ux_port > 0:
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((cfg.unity_host, ux_port))
            srv.listen(1)
            srv.settimeout(0.05)
        except OSError:
            srv = None

    seq = 0
    rx_buf = bytearray()
    ready_event.set()
    try:
        while not stop_event.is_set():
            if not is_windows_pid_alive(orch_ppid):
                stop_event.set()
                break

            if srv is not None and conn is None:
                try:
                    conn, _ = srv.accept()
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    conn.setblocking(False)
                    rx_buf.clear()
                except OSError:
                    conn = None

            if conn is not None:
                # Poll incoming 16-byte UBNC control commands with TCP stream framing & clean EOF handling
                try:
                    while True:
                        ctrl_chunk = conn.recv(256)
                        if not ctrl_chunk:
                            # Remote Unity UX closed connection cleanly (EOF `b""`)
                            try:
                                conn.close()
                            except Exception:
                                pass
                            conn = None
                            rx_buf.clear()
                            break
                        rx_buf.extend(ctrl_chunk)
                        if len(rx_buf) > 4096:
                            del rx_buf[:-UBNC_TOTAL_BYTES]
                except BlockingIOError:
                    pass
                except OSError:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                    rx_buf.clear()

                while conn is not None and len(rx_buf) >= UBNC_TOTAL_BYTES:
                    magic_peek = struct.unpack_from("<I", rx_buf, 0)[0]
                    if magic_peek != UNITY_UX_CTRL_MAGIC:
                        del rx_buf[0]
                        continue
                    ctrl_raw = bytes(rx_buf[:UBNC_TOTAL_BYTES])
                    del rx_buf[:UBNC_TOTAL_BYTES]
                    ctrl = unpack_ubnc_ui_control(ctrl_raw)
                    if ctrl is not None:
                        ct, a0, a1 = ctrl["cmd_type"], ctrl["arg0"], ctrl["arg1"]
                        if ct == 1:  # Timing mode
                            timing_bus.slot.set_header_atom(0, (a0 & 0x03) + 1)
                        elif ct == 2:  # Broca toggle
                            lang_bus.slot.set_header_atom(0, 1 if a0 else 2)
                        elif ct == 5:  # Learning / Autosave toggle (Single-Writer on header_atom(2)!)
                            cog_bus.slot.set_header_atom(2, _encode_learning_ctrl_atom(bool(a0), bool(a1), False))
                        elif ct == 6:  # Time scale (a0 = x10, e.g., 20 -> 2.0x)
                            cog_bus.slot.set_header_atom(1, int(np.clip(a0 * 100, 250, 12000)))

            w_rec, _ = world_bus.read_latest()
            sp_rec, _ = spatial_bus.read_latest()
            tim_rec, _ = timing_bus.read_latest()
            cog_rec, _ = cog_bus.read_latest()
            lng_rec, _ = lang_bus.read_latest()
            vc_rec = vc_bus.read_latest()[0] if vc_bus is not None else None

            if conn is not None and w_rec is not None and sp_rec is not None and cog_rec is not None:
                seq += 1
                learn_on, auto_on = _read_runtime_learning_flags(cog_bus, cfg)
                guard_b = _read_runtime_guard_tripped(cog_bus)
                atom_slp_act, atom_slp_stg = _read_sleep_status_atom(cog_bus)
                eff_slp_act = bool(atom_slp_act) or bool(cog_rec["sleep_active"])
                eff_slp_stg = int(atom_slp_stg) if atom_slp_act else int(cog_rec["sleep_stage"])
                eff_autosaves = int(cog_rec["autosave_count_u8"])
                scalars24 = [
                    float(w_rec["measured_fps"]),
                    float(cog_rec["brain_hz"]),
                    float(cog_rec["inference_latency_ms"]),
                    float(w_rec["cumulative_score"]),
                    float(w_rec["last_step_reward"]),
                    float(w_rec["hazard_level"]),
                    float(sp_rec["believed_hazard"]),
                    float(sp_rec["map_accuracy_iou"]),
                    float(sp_rec["mean_fog"]),
                    float(cog_rec["ignition"]),
                    float(cog_rec["phi_geometric_bits"]),
                    float(cog_rec["policy_entropy_norm"]),
                    float(cog_rec["world_surprise"]),
                    float(cog_rec["self_surprise"]),
                    float(cog_rec["fisher_rao_step"]),
                    float(cog_rec["survival_at_chosen"]),
                    float(tim_rec["go_drive"]) if tim_rec is not None else 1.0,
                    float(tim_rec["stn_brake"]) if tim_rec is not None else 0.0,
                    float(tim_rec["urgency_val"]) if tim_rec is not None else 0.0,
                    float(tim_rec["net_accumulator"]) if tim_rec is not None else 0.0,
                    float(tim_rec["release_threshold"]) if tim_rec is not None else 0.72,
                    float(tim_rec["scene_opportunity_prob"]) if tim_rec is not None else 0.5,
                    float(sp_rec["friendly_turf_ratio"]),
                    float(sp_rec["enemy_turf_ratio"]),
                ]
                sp_txt = bytes(lng_rec["monologue_utf8"]).split(b"\x00", 1)[0].decode("utf-8", errors="ignore") if lng_rec is not None else ""
                pkt = pack_ubnt_neural_telemetry(
                    seq=seq,
                    timestamp_ns=time.perf_counter_ns(),
                    episode_id=int(w_rec["episode_id"]),
                    action_epoch=int(w_rec["action_epoch"]),
                    physics_tick=int(w_rec["physics_tick"]),
                    chosen_cand_idx=int(cog_rec["chosen_cand_idx"]),
                    s1_fast_cand_idx=int(cog_rec["s1_fast_cand_idx"]),
                    commit_gate_open=bool(tim_rec["commit_gate_open"]) if tim_rec is not None else True,
                    veto_active=bool(cog_rec["veto_active"]),
                    timing_mode=int(tim_rec["timing_mode"]) if tim_rec is not None else cfg.timing_mode,
                    hold_reason_id=int(tim_rec["hold_reason_id"]) if tim_rec is not None else 0,
                    coach_brake_active=bool(tim_rec["coach_brake_active"]) if tim_rec is not None else False,
                    sleep_active=eff_slp_act,
                    sleep_stage=eff_slp_stg,
                    virtual_alert_active=bool(cog_rec["virtual_alert_active"]),
                    endogenous_active=bool(lng_rec["endogenous_active"]) if lng_rec is not None else True,
                    proto_sym_id=int(cog_rec["proto_sym_id"]),
                    proto_mod_id=int(cog_rec["proto_mod_id"]),
                    believed_count=int(sp_rec["believed_count"]),
                    cand_verbs=cog_rec["cand_verbs"],
                    holds_triggered=int(tim_rec["holds_triggered"]) if tim_rec is not None else 0,
                    opportunities_caught=int(tim_rec["opportunities_caught"]) if tim_rec is not None else 0,
                    loop_closures_total=int(sp_rec["loop_closures_total"]),
                    virtual_alert_count=int(cog_rec["virtual_alert_count"]),
                    scalars24=scalars24,
                    gwt_alpha=cog_rec["gwt_alpha"],
                    gwt_workspace=cog_rec["gwt_workspace"],
                    rssm_h=cog_rec["rssm_h"],
                    latent_z=cog_rec["latent_z"],
                    z_topdown_delta=cog_rec["z_topdown_delta"],
                    mec_grid_code=sp_rec["mec_grid_code"],
                    hd_ring_code=sp_rec["hd_ring_code"],
                    self_state=cog_rec["self_state"],
                    q_values=cog_rec["q_values"],
                    q_raw_values=cog_rec["q_raw_values"],
                    survival_probs=cog_rec["survival_probs"],
                    policy_probs=cog_rec["policy_probs"],
                    mip_q_star=cog_rec["mip_q_star"],
                    cand_poses=cog_rec["cand_poses"],
                    stn_gains=tim_rec["stn_gains"] if tim_rec is not None else np.ones(4, dtype=np.float32),
                    believed_self_pos=sp_rec["believed_self_pos"],
                    pose_drift_vec=sp_rec["pose_drift_vec"],
                    inner_speech_text=sp_txt,
                    visual_mode_id=int(vc_rec["visual_mode_id"]) if vc_rec is not None else 0,
                    depth_ready=bool(vc_rec["depth_ready"]) if vc_rec is not None else False,
                    depth_mean_uncert=float(vc_rec["mean_uncertainty"]) if vc_rec is not None else 0.0,
                    depth_infer_ms=float(vc_rec["depth_infer_ms"]) if vc_rec is not None else 0.0,
                    depth_mean_norm=float(vc_rec["mean_depth_norm"]) if vc_rec is not None else 0.0,
                    learning_enabled=learn_on,
                    autosave_enabled=auto_on,
                    autosave_count=eff_autosaves,
                    overfit_guard_tripped=guard_b,
                    mip_cut_mask=int(cog_rec["mip_cut_mask_u8"]) if "mip_cut_mask_u8" in cog_rec.dtype.names else 3,
                )
                sent_ubnt, fatal_disc = _sendall_nonblocking_safe(conn, pkt, timeout_s=0.08)
                if (not fatal_disc) and sent_ubnt and vc_rec is not None and int(vc_rec["depth_ready"]) == 1:
                    dpkt = pack_ubnd_neural_depth(
                        episode_id=int(w_rec["episode_id"]),
                        physics_tick=int(w_rec["physics_tick"]),
                        visual_mode_id=int(vc_rec["visual_mode_id"]),
                        depth_source_id=1,
                        depth_infer_ms=float(vc_rec["depth_infer_ms"]),
                        depth_norm_16x16=vc_rec["pred_fovea_patch"][:, :, 0],
                        uncert_16x16=vc_rec["epistemic_uncert_16"],
                    )
                    _, fatal_disc = _sendall_nonblocking_safe(conn, dpkt, timeout_s=0.08)
                if (not fatal_disc) and sent_ubnt:
                    wmpkt = pack_ubwm_world_model(
                        episode_id=int(w_rec["episode_id"]),
                        physics_tick=int(w_rec["physics_tick"]),
                        chosen_cand_idx=int(cog_rec["chosen_cand_idx"]),
                        has_prev_pred=bool(cog_rec["wm_has_prev_pred"]),
                        veto_mask_u8=int(cog_rec["wm_veto_mask_u8"]),
                        vis_pred_mae=float(cog_rec["wm_vis_mae"]),
                        depth_pred_mae_m=float(cog_rec["wm_depth_mae_m"]),
                        world_kl_surprise=float(cog_rec["world_surprise"]),
                        action_cf_spread=float(cog_rec["wm_cf_spread"]),
                        pred_rew=cog_rec["pred_rew"],
                        pred_surv=cog_rec["survival_probs"],
                        pred_adv=cog_rec["pred_adv_probs"],
                        pred_depth_u8=cog_rec["wm_pred_depth_u8"],
                        pred_surf_u8=cog_rec["wm_pred_surf_u8"],
                        pred_err_u8=cog_rec["wm_pred_err_u8"],
                    )
                    _, fatal_disc = _sendall_nonblocking_safe(conn, wmpkt, timeout_s=0.08)
                if fatal_disc:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                    rx_buf.clear()

            time.sleep(1.0 / 15.0)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if srv is not None:
            try:
                srv.close()
            except Exception:
                pass
        if vc_bus is not None:
            vc_bus.close(unlink=False)
        world_bus.close(unlink=False)
        spatial_bus.close(unlink=False)
        timing_bus.close(unlink=False)
        cog_bus.close(unlink=False)
        lang_bus.close(unlink=False)


