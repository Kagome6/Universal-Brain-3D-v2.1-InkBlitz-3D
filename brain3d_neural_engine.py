# -*- coding: utf-8 -*-
"""
brain3d_neural_engine.py — Emergent RVQ Symbols, PER RingBuffer, Whole-Brain Continual
Learner (`WholeBrainNeuralEngineWrapper`), RPC Client & `GPUInferenceProcess` (P2, v2.2)
========================================================================================
Barrier 1 Guarantee:
  Zero top-level `torch` imports! `torch` and `brain3d_torch_models` are imported strictly
  inside `WholeBrainNeuralEngineWrapper.__init__` (Process #2 GPU boundary).

License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

from collections import deque
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from task_profiles_3d import *
from brain3d_config_ipc import *
from brain3d_config_ipc import (
    _encode_sleep_status_atom,
    _ignore_sigint_in_worker,
)

# ==============================================================================
# 6. Emergent RVQ Proto-Symbol Book & Whole-Brain PyTorch Neural Suite (Process #2)
# ==============================================================================
class OnlineProtoSymbolRVQ3D:
    """
    2-Stage Residual Vector Quantizer (`32` coarse symbols x `8` residual modifiers)
    grounding continuous Global Workspace states into discrete tactical proto-symbols (`◆00..◆31`).
    """

    def __init__(self, dim: int = 64, n_coarse: int = NUM_PROTO_SYMBOLS, n_mod: int = NUM_PROTO_MODS, seed: int = 2026):
        rng = np.random.default_rng(seed + 77)
        self.dim = int(dim)
        self.n_coarse = int(n_coarse)
        self.n_mod = int(n_mod)
        self.codebook_c = (rng.normal(0.0, 0.25, size=(self.n_coarse, self.dim))).astype(np.float32)
        self.codebook_m = (rng.normal(0.0, 0.12, size=(self.n_mod, self.dim))).astype(np.float32)
        self.sym_verb_prior = np.zeros((self.n_coarse, NUM_MOTOR_VERBS), dtype=np.float32)

    def _project_vec(self, gwt_w: np.ndarray, self_s: np.ndarray, rssm_h: np.ndarray) -> np.ndarray:
        v = np.concatenate([
            np.asarray(gwt_w, dtype=np.float32).ravel()[:32],
            np.asarray(self_s, dtype=np.float32).ravel()[:8],
            np.asarray(rssm_h, dtype=np.float32).ravel()[:24],
        ])
        nrm = float(np.linalg.norm(v))
        return (v / max(1e-5, nrm)).astype(np.float32)

    def quantize(
        self,
        gwt_w: np.ndarray,
        self_s: np.ndarray,
        rssm_h: np.ndarray,
        cand_verbs: np.ndarray,
        update_ema: bool = True,
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
        v = self._project_vec(gwt_w, self_s, rssm_h)
        d_c = np.sum((self.codebook_c - v[None, :]) ** 2, axis=-1)
        sym_id = int(np.argmin(d_c))
        res = v - self.codebook_c[sym_id]
        d_m = np.sum((self.codebook_m - res[None, :]) ** 2, axis=-1)
        mod_id = int(np.argmin(d_m))

        if update_ema:
            self.codebook_c[sym_id] = 0.96 * self.codebook_c[sym_id] + 0.04 * v
            self.codebook_m[mod_id] = 0.96 * self.codebook_m[mod_id] + 0.04 * res

        q_vec = (self.codebook_c[sym_id] + self.codebook_m[mod_id])[:PROTO_EMBED_DIM].copy()
        prior_bias = np.zeros(NUM_CANDIDATES, dtype=np.float32)
        for m in range(NUM_CANDIDATES):
            vb = int(cand_verbs[m]) % NUM_MOTOR_VERBS
            prior_bias[m] = float(np.clip(self.sym_verb_prior[sym_id, vb], -0.35, 0.35))
        return sym_id, mod_id, q_vec, prior_bias

    def reinforce_symbol_verb(self, sym_id: int, verb: int, td_adv: float) -> None:
        if 0 <= sym_id < self.n_coarse and 0 <= verb < NUM_MOTOR_VERBS:
            self.sym_verb_prior[sym_id, verb] = float(
                np.clip(0.98 * self.sym_verb_prior[sym_id, verb] + 0.03 * float(np.tanh(td_adv)), -0.5, 0.5)
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "codebook_c": self.codebook_c.copy(),
            "codebook_m": self.codebook_m.copy(),
            "sym_verb_prior": self.sym_verb_prior.copy(),
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if "codebook_c" in sd and np.asarray(sd["codebook_c"]).shape == self.codebook_c.shape:
            self.codebook_c = np.asarray(sd["codebook_c"], dtype=np.float32).copy()
        if "codebook_m" in sd and np.asarray(sd["codebook_m"]).shape == self.codebook_m.shape:
            self.codebook_m = np.asarray(sd["codebook_m"], dtype=np.float32).copy()
        if "sym_verb_prior" in sd and np.asarray(sd["sym_verb_prior"]).shape == self.sym_verb_prior.shape:
            self.sym_verb_prior = np.asarray(sd["sym_verb_prior"], dtype=np.float32).copy()


def _extract_candidate_compass_features_np(
    cand_poses: np.ndarray,
    cand_verbs: np.ndarray,
    proprio: np.ndarray,
    mem_compass: np.ndarray,
) -> np.ndarray:
    """
    Build the rich `(M=8, 32)` candidate-to-compass alignment feature matrix:
      - `[0:6]`  : raw `target_pose` (`[nx, ny, nz, aim_yaw, aim_pitch, speed_scale]`)
      - `[6:14]` : motor verb one-hot (`8`)
      - `[14:18]`: `[rel_dx, rel_dz, step_norm, strafe_cos_move_vs_aim]`
      - `[18:22]`: directional dot alignments `[dot_goal, dot_frontier, dot_refuge, dot_threat]`
      - `[22:26]`: `[dir_clearance, dir_friendly_surface, in_refuge, carrying_or_ammo]`
      - `[26:32]`: `[sin(aim_yaw), cos(aim_yaw), aim_pitch, hazard, audio_loud, obj_progress]`
    """
    poses = np.asarray(cand_poses, dtype=np.float32).reshape(NUM_CANDIDATES, CONT_POSE_DIM)
    verbs = np.asarray(cand_verbs, dtype=np.int64).reshape(NUM_CANDIDATES)
    prop = np.asarray(proprio, dtype=np.float32).reshape(PROPRIO_DIM)
    mc = np.asarray(mem_compass, dtype=np.float32).reshape(MEM_COMPASS_DIM)

    ego_xz = 2.0 * prop[[PROPRIO_IDX_POS_X, PROPRIO_IDX_POS_Z]] - 1.0
    in_ref = 1.0 if is_in_stealth_or_cover_refuge(prop) else 0.0
    carry_or_tank = float(prop[PROPRIO_IDX_RESOURCE_CARRIER])
    haz = float(prop[PROPRIO_IDX_HAZARD])
    aud = float(prop[PROPRIO_IDX_AUDIO_LOUD])
    obj_p = float(prop[PROPRIO_IDX_OBJ_PROGRESS])

    feats = np.zeros((NUM_CANDIDATES, CFEAT_DIM), dtype=np.float32)
    feats[:, CFEAT_NX : CFEAT_SPEED_SCALE + 1] = poses
    for m in range(NUM_CANDIDATES):
        vb = int(verbs[m]) % NUM_MOTOR_VERBS
        feats[m, CFEAT_VERB_ONEHOT_BASE + vb] = 1.0
        rel_xz = poses[m, [CFEAT_NX, CFEAT_NZ]] - ego_xz
        dn = float(math.hypot(float(rel_xz[0]), float(rel_xz[1])))
        ux, uz = (float(rel_xz[0]) / dn, float(rel_xz[1]) / dn) if dn > 1e-4 else (0.0, 0.0)
        ay = float(poses[m, CFEAT_AIM_YAW])
        ap = float(poses[m, CFEAT_AIM_PITCH])
        asx, acz = math.sin(ay), math.cos(ay)
        strafe_cos = ux * asx + uz * acz if dn > 1e-4 else 1.0

        feats[m, CFEAT_REL_DX : CFEAT_STRAFE_COS + 1] = [rel_xz[0], rel_xz[1], min(1.5, dn * 4.0), strafe_cos]
        feats[m, CFEAT_DOT_GOAL] = (ux * float(mc[MC_OBJ_DX]) + uz * float(mc[MC_OBJ_DZ])) * float(mc[MC_OBJ_VALID])
        feats[m, CFEAT_DOT_FRONTIER] = (ux * float(mc[MC_FRONT_DX]) + uz * float(mc[MC_FRONT_DZ])) * float(mc[MC_FRONT_VALID])
        feats[m, CFEAT_DOT_REFUGE] = (ux * float(mc[MC_COV_DX]) + uz * float(mc[MC_COV_DZ])) * float(mc[MC_COV_VALID])
        feats[m, CFEAT_DOT_THREAT] = (ux * float(mc[MC_THR_DX]) + uz * float(mc[MC_THR_DZ])) * float(mc[MC_THR_VALID])

        wn, we, ws, ww = max(0.0, uz), max(0.0, ux), max(0.0, -uz), max(0.0, -ux)
        wsum = max(1e-4, wn + we + ws + ww)
        d_clr = (
            (wn * float(mc[MC_CLR_N]) + we * float(mc[MC_CLR_E]) + ws * float(mc[MC_CLR_S]) + ww * float(mc[MC_CLR_W])) / wsum
            if dn > 1e-4
            else 1.0
        )
        d_srf = (
            (wn * float(mc[MC_SURF_N]) + we * float(mc[MC_SURF_E]) + ws * float(mc[MC_SURF_S]) + ww * float(mc[MC_SURF_W])) / wsum
            if dn > 1e-4
            else 0.5
        )

        feats[m, CFEAT_DIR_CLEARANCE : CFEAT_CARRY_OR_TANK + 1] = [d_clr, d_srf, in_ref, carry_or_tank]
        feats[m, CFEAT_SIN_AIM_YAW : CFEAT_OBJ_PROGRESS + 1] = [asx, acz, ap, haz, aud, obj_p]
    return feats


class _ReplayPriorityView:
    """Backward-compatible priority view wrapping `PrioritizedReplayRingBuffer`."""

    def __init__(self, ring: "PrioritizedReplayRingBuffer"):
        self._ring = ring

    def __len__(self) -> int:
        return len(self._ring)

    def __getitem__(self, idx: int) -> float:
        if idx < 0:
            idx += len(self._ring)
        return float(self._ring._priorities[int(idx)])

    def __setitem__(self, idx: int, val: float) -> None:
        self._ring.update_priority(int(idx), float(val))

    def clear(self) -> None:
        pass


class PrioritizedReplayRingBuffer:
    """O(1) Fixed-Capacity Ring Buffer with Prioritized Experience Replay (PER) & Importance Sampling."""

    def __init__(self, capacity: int = 4096, on_clear_cb: Optional[Callable[[], None]] = None):
        self.capacity = max(16, int(capacity))
        self._entries: List[Optional[Dict[str, Any]]] = [None] * self.capacity
        self._priorities = np.zeros(self.capacity, dtype=np.float64)
        self._write_idx = 0
        self._size = 0
        self._on_clear_cb = on_clear_cb

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += self._size
        if not (0 <= idx < self._size):
            raise IndexError(idx)
        item = self._entries[idx]
        assert item is not None
        return item

    def push(self, item: Dict[str, Any], priority: float) -> int:
        pos = self._write_idx
        self._entries[pos] = item
        self._priorities[pos] = max(1e-4, float(priority))
        self._write_idx = (pos + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        return pos

    def update_priority(self, phys_idx: int, priority: float) -> None:
        if 0 <= int(phys_idx) < self._size:
            self._priorities[int(phys_idx)] = max(1e-4, float(priority))

    def sample_prioritized(
        self,
        batch_size: int,
        alpha: float = 0.60,
        beta: float = 0.45,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        n = self._size
        bsz = min(int(batch_size), n)
        p_slice = np.maximum(1e-6, self._priorities[:n])
        p_pow = p_slice ** float(alpha)
        probs = p_pow / max(1e-12, float(np.sum(p_pow)))
        idxs = np.random.choice(n, size=bsz, replace=False, p=probs)
        batch = [self._entries[int(k)] for k in idxs]  # type: ignore[misc]
        raw_w = (float(n) * probs[idxs]) ** (-float(beta))
        is_weights = (raw_w / max(1e-8, float(np.max(raw_w)))).astype(np.float32)
        return idxs, batch, is_weights  # type: ignore[return-value]

    def clear(self) -> None:
        for i in range(self._size):
            self._entries[i] = None
        self._priorities.fill(0.0)
        self._write_idx = 0
        self._size = 0
        if self._on_clear_cb is not None:
            self._on_clear_cb()


class WholeBrainNeuralEngineWrapper:
    """
    Unified GPU/CPU Whole-Brain Inference, Online Continual Learner, Counterfactual De-Confounder &
    4-Stage Sleep Consolidation Engine.
    Can be instantiated standalone inside `GPUInferenceProcess` OR inside `arena_pbt_trainer.py`
    to serve batched multi-agent populations on a single GPU!
    """

    def __init__(self, cfg: Brain3DConfig, device_str: str = "auto"):
        import torch
        import torch.nn.functional as F
        from brain3d_torch_models import build_pytorch_brain_models

        self.torch = torch
        self.F = F
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self.vocab = GroundedVocabulary512()
        self.rvq = OnlineProtoSymbolRVQ3D(seed=cfg.seed)

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        torch.manual_seed(int(cfg.seed))
        self.models = build_pytorch_brain_models(torch, cfg)
        for m in self.models.values():
            m.to(self.device)
        self.models["d3qn_target"].load_state_dict(self.models["d3qn"].state_dict())
        self.models["d3qn_target"].eval()

        core_params: List[Any] = []
        for k, m in self.models.items():
            if k not in ("d3qn_target", "depth_net"):
                core_params.extend(list(m.parameters()))
        self.base_lr = 2.8e-4 * float(np.clip(cfg.lr_scale, 0.30, 2.50))
        self.optimizer = torch.optim.AdamW(core_params, lr=self.base_lr, weight_decay=1e-4)
        self.depth_optimizer = torch.optim.AdamW(self.models["depth_net"].parameters(), lr=6.0e-4, weight_decay=1e-5)

        # Replay (O(1) Ring Buffer + PER Importance Sampling) & Per-Slot N-Step / 4-Step Chunk Queues
        self.max_replay = 4096
        self._nstep_queues: Dict[int, deque] = {}
        self._slot_chunk4_win: Dict[int, deque] = {}
        self.chunk4_ring: deque = deque(maxlen=1024)
        self.replay = PrioritizedReplayRingBuffer(
            capacity=self.max_replay,
            on_clear_cb=self._clear_temporal_queues,
        )
        self.replay_priorities = _ReplayPriorityView(self.replay)
        self.transitions_recorded = 0
        self.train_steps_done = 0
        self.sleep_cycles_done = 0
        self.autosave_count = 0
        self.overfit_guard_tripped = False
        self.nan_rollback_count = 0
        self._anchor_weights: Optional[Dict[str, Any]] = None
        self._loaded_ckpt_param_keys: Optional[Set[str]] = None
        self._best_rolling_score = -1e9
        self._ep_scores: deque = deque(maxlen=12)

        # Depth calibration tracking
        self.calib_steps_done = 0
        self.calib_val_mae = 0.25
        self.calib_val_cos = 0.85

        # Per-agent recurrent hidden states (`agent_slot_id -> state dict`) so 1 engine can serve N agents!
        self._agent_states: Dict[int, Dict[str, Any]] = {}

        if cfg.load_weights_path and os.path.isfile(cfg.load_weights_path):
            self.load_checkpoint(cfg.load_weights_path)
        if cfg.depth_weights_path and os.path.isfile(cfg.depth_weights_path):
            self.load_depth_checkpoint(cfg.depth_weights_path)
        if self._anchor_weights is None:
            self._snapshot_anchor_weights()

    def _get_agent_state(self, slot_id: int, ep_id: int, respawn_flag: bool = False) -> Dict[str, Any]:
        st = self._agent_states.get(slot_id)
        if st is None or st["ep_id"] != int(ep_id):
            st = {
                "ep_id": int(ep_id),
                "rssm_h": np.zeros(RSSM_DETER_DIM, dtype=np.float32),
                "rssm_stoch": np.zeros(RSSM_STOCH_FLAT, dtype=np.float32),
                "gwt_w": np.zeros(GWT_DIM, dtype=np.float32),
                "prev_alpha": np.full(5, 0.2, dtype=np.float32),
                "hab_alpha": np.full(5, 0.2, dtype=np.float32),
                "prev_compass": np.zeros(MEM_COMPASS_DIM, dtype=np.float32),
                "self_s": np.zeros(SELF_MODEL_DIM, dtype=np.float32),
                "prev_self_err": np.zeros(SELF_MODEL_DIM, dtype=np.float32),
                "prev_self_surp": 0.0,
                "z_td": np.zeros(LATENT_Z_DIM, dtype=np.float32),
                "prev_cand_feat32": np.zeros(32, dtype=np.float32),
                "prev_slots": None,
                "prev_pred_vis8": None,
                "prev_topdown_gain_16": None,
                "_last_vis_step": None,
                "strat_hist": deque([np.zeros(GWT_DIM + SELF_MODEL_DIM + MEM_COMPASS_DIM, dtype=np.float32) for _ in range(8)], maxlen=8),
                "ctx_ring": deque(maxlen=16),
            }
            st["rssm_stoch"][0] = 1.0
            self._agent_states[slot_id] = st
        elif respawn_flag:
            # On mid-episode respawn: keep strategic history & self-schema, softly damp short-term RSSM shock & specialist fatigue
            st["rssm_h"] *= 0.50
            st["hab_alpha"] = (0.50 * st["hab_alpha"] + 0.50 * 0.20).astype(np.float32)
            st["prev_slots"] = None
            st["prev_pred_vis8"] = None
            st["prev_topdown_gain_16"] = None
            st["_last_vis_step"] = None
            st["prev_cand_feat32"].fill(0.0)
        return st

    def _t(self, arr: Any, dtype=None):
        t = self.torch.as_tensor(arr, device=self.device)
        return t.to(dtype) if dtype is not None else t

    def infer_s1_reflex_batch(self, batch_reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batched fast System-1 Reflex evaluation (`TT-Foveal + Zero-Gated Slot Residual + TT-D3QN + SpeedMod`)."""
        if not batch_reqs:
            return []
        torch = self.torch
        bsz = len(batch_reqs)
        foveas = np.stack([r["fovea_patch"] for r in batch_reqs], axis=0).astype(np.float32)
        ventrals = np.stack([
            r.get("ventral_patch", derive_ventral_fallback_from_dorsal_16(r["fovea_patch"]))
            for r in batch_reqs
        ], axis=0).astype(np.float32)
        vis8 = np.concatenate([foveas, ventrals], axis=-1).astype(np.float32)
        proprios = np.stack([r["proprio"] for r in batch_reqs], axis=0).astype(np.float32)
        compasses = np.stack([r["mem_compass"] for r in batch_reqs], axis=0).astype(np.float32)
        mecs = np.stack([r["mec_grid_code"] for r in batch_reqs], axis=0).astype(np.float32)
        hds = np.stack([r["hd_ring_code"] for r in batch_reqs], axis=0).astype(np.float32)
        gwts = np.stack([r["gwt_workspace"] for r in batch_reqs], axis=0).astype(np.float32)
        rssm_hs = np.stack([r["rssm_h"] for r in batch_reqs], axis=0).astype(np.float32)
        z_tds = np.stack([r["z_topdown_delta"] for r in batch_reqs], axis=0).astype(np.float32)
        c_feats = np.stack([
            _extract_candidate_compass_features_np(r["cand_poses"], r["cand_verbs"], r["proprio"], r["mem_compass"])
            for r in batch_reqs
        ], axis=0).astype(np.float32)

        perr_list = []
        spot_list = []
        for i, r in enumerate(batch_reqs):
            st_i = self._agent_states.get(int(r.get("slot_id", 0)))
            if st_i is not None and st_i.get("prev_pred_vis8") is not None:
                pe = np.mean(np.abs(vis8[i] - st_i["prev_pred_vis8"]), axis=-1, keepdims=True).transpose(2, 0, 1)
                perr_list.append(pe.astype(np.float32))
            else:
                perr_list.append(np.zeros((1, FOVEA_H, FOVEA_W), dtype=np.float32))
            if st_i is not None and st_i.get("prev_topdown_gain_16") is not None:
                spot_list.append(st_i["prev_topdown_gain_16"].astype(np.float32))
            else:
                spot_list.append(np.ones((1, FOVEA_H, FOVEA_W), dtype=np.float32))

        with torch.no_grad():
            z = self.models["foveal_enc"](
                self._t(foveas, torch.float32),
                self._t(proprios, torch.float32),
                self._t(compasses, torch.float32),
            )
            t_gwts = self._t(gwts, torch.float32)
            t_rssm_hs = self._t(rssm_hs, torch.float32)
            t_z_tds = self._t(z_tds, torch.float32)
            _, _, _, s1_aux = self.models["slot_cortex"](
                self._t(vis8, torch.float32),
                pred_err_16=self._t(np.stack(perr_list, axis=0), torch.float32),
                topdown_gain_16=self._t(np.stack(spot_list, axis=0), torch.float32),
                td_context=torch.cat([t_gwts, t_rssm_hs], dim=-1),
                return_aux=True,
            )
            z_eff = torch.tanh(z + s1_aux["delta_z"] + t_z_tds)
            q_vals, spd_mod = self.models["d3qn"](
                z_eff,
                t_gwts,
                t_rssm_hs,
                self._t(mecs, torch.float32),
                self._t(hds, torch.float32),
                self._t(proprios, torch.float32),
                self._t(compasses, torch.float32),
                self._t(c_feats, torch.float32),
            )
            q_np = q_vals.cpu().numpy()
            spd_np = spd_mod.cpu().numpy()

        out: List[Dict[str, Any]] = []
        for i in range(bsz):
            best_i = int(np.argmax(q_np[i]))
            out.append({
                "best_idx": best_i,
                "q_values": q_np[i].astype(np.float32),
                "speed_mod": spd_np[i].astype(np.float32),
            })
        return out

    def infer_conscious_gwt_batch(self, batch_reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batched Whole-Brain Conscious Global Workspace Ignition (`5 Specialists + Nonlinear Ignition Gate +
        Multi-Step 3-Head RSSM + Predictive Slot Cortex & 16x16 Spotlight + HOT/AST Self-Surprise Loop`).
        """
        if not batch_reqs:
            return []
        torch = self.torch
        bsz = len(batch_reqs)

        states = [
            self._get_agent_state(int(r.get("slot_id", 0)), int(r["episode_id"]), bool(r.get("respawn_flag", False)))
            for r in batch_reqs
        ]
        foveas = np.stack([r["fovea_patch"] for r in batch_reqs], axis=0).astype(np.float32)
        ventrals = np.stack([r.get("ventral_patch", derive_ventral_fallback_from_dorsal_16(r["fovea_patch"])) for r in batch_reqs], axis=0).astype(np.float32)
        vis8 = np.concatenate([foveas, ventrals], axis=-1).astype(np.float32)
        proprios = np.stack([r["proprio"] for r in batch_reqs], axis=0).astype(np.float32)
        compasses = np.stack([r["mem_compass"] for r in batch_reqs], axis=0).astype(np.float32)
        mecs = np.stack([r["mec_grid_code"] for r in batch_reqs], axis=0).astype(np.float32)
        hds = np.stack([r["hd_ring_code"] for r in batch_reqs], axis=0).astype(np.float32)
        c_feats = np.stack([
            _extract_candidate_compass_features_np(r["cand_poses"], r["cand_verbs"], r["proprio"], r["mem_compass"])
            for r in batch_reqs
        ], axis=0).astype(np.float32)

        prev_h = np.stack([s["rssm_h"] for s in states], axis=0)
        prev_st = np.stack([s["rssm_stoch"] for s in states], axis=0)
        prev_act = np.stack([s["prev_cand_feat32"] for s in states], axis=0)
        prev_gwt = np.stack([s["gwt_w"] for s in states], axis=0)
        prev_alpha = np.stack([s["prev_alpha"] for s in states], axis=0)
        prev_hab = np.stack([s["hab_alpha"] for s in states], axis=0)
        prev_z_td = np.stack([s["z_td"] for s in states], axis=0)
        prev_comp = np.stack([s["prev_compass"] for s in states], axis=0)
        prev_self = np.stack([s["self_s"] for s in states], axis=0)
        prev_self_err = np.stack([s["prev_self_err"] for s in states], axis=0)
        prev_self_surp = np.array([float(s["prev_self_surp"]) for s in states], dtype=np.float32)
        meta_err9 = np.concatenate([prev_self_err, prev_self_surp[:, None]], axis=-1).astype(np.float32)
        strat_seq = np.stack([np.stack(list(s["strat_hist"]), axis=0) for s in states], axis=0)

        perr_16_list = []
        spot_16_list = []
        has_prev_vis = []
        depth_mae_m_list = []
        for i, s in enumerate(states):
            if s["prev_pred_vis8"] is not None:
                pe = np.mean(np.abs(vis8[i] - s["prev_pred_vis8"]), axis=-1, keepdims=True).transpose(2, 0, 1)
                perr_16_list.append(pe.astype(np.float32))
                has_prev_vis.append(True)
                d_mae = float(np.mean(np.abs(vis8[i, :, :, 0] - np.clip(s["prev_pred_vis8"][:, :, 0], 0.0, 1.0)))) * float(self.cfg.fovea_max_range_m)
                depth_mae_m_list.append(d_mae)
            else:
                perr_16_list.append(np.zeros((1, FOVEA_H, FOVEA_W), dtype=np.float32))
                has_prev_vis.append(False)
                depth_mae_m_list.append(0.0)
            if s["prev_topdown_gain_16"] is not None:
                spot_16_list.append(s["prev_topdown_gain_16"].astype(np.float32))
            else:
                spot_16_list.append(np.ones((1, FOVEA_H, FOVEA_W), dtype=np.float32))
        perr_16_np = np.stack(perr_16_list, axis=0)
        spot_16_np = np.stack(spot_16_list, axis=0)

        with torch.no_grad():
            t_fov = self._t(foveas, torch.float32)
            t_vis8 = self._t(vis8, torch.float32)
            t_perr = self._t(perr_16_np, torch.float32)
            t_spot = self._t(spot_16_np, torch.float32)
            t_prop = self._t(proprios, torch.float32)
            t_comp = self._t(compasses, torch.float32)
            t_mec = self._t(mecs, torch.float32)
            t_hd = self._t(hds, torch.float32)
            t_cf = self._t(c_feats, torch.float32)
            t_prev_act = self._t(prev_act, torch.float32)
            t_prev_alpha = self._t(prev_alpha, torch.float32)
            t_prev_hab = self._t(prev_hab, torch.float32)
            t_prev_z_td = self._t(prev_z_td, torch.float32)
            t_prev_gwt = self._t(prev_gwt, torch.float32)
            t_prev_h = self._t(prev_h, torch.float32)
            t_prev_ssurp = self._t(prev_self_surp, torch.float32)
            t_meta_err9 = self._t(meta_err9, torch.float32)

            z = self.models["foveal_enc"](t_fov, t_prop, t_comp)
            slots, s_coords, s_val, slot_aux = self.models["slot_cortex"](
                t_vis8,
                pred_err_16=t_perr,
                topdown_gain_16=t_spot,
                td_context=torch.cat([t_prev_gwt, t_prev_h], dim=-1),
                return_aux=True,
            )
            # Closed-loop thalamocortical recurrence: `0.30 * prev_z_td` enters `z_perc(t)` BEFORE bottom-up GWT/RSSM,
            # while `0.70 * z_td(t)` completes `z_eff(t)` (`0.30 + 0.70 == 1.00` steady-state gain preservation!).
            z_perc = torch.tanh(z + slot_aux["delta_z"] + 0.30 * t_prev_z_td)
            haz_p = self.models["hazard_net"](z_perc, t_prop, t_comp)
            slot_ctx64 = self.models["topo_adapter"](slots, t_mec, t_hd, t_comp)

            h_next, post_stoch, p_post, p_prior = self.models["rssm"].observe_step(
                t_prev_h,
                self._t(prev_st, torch.float32),
                t_prev_act,
                z_perc,
                slot_ctx64,
                t_prop,
                t_comp,
            )
            p_rew, p_surv, p_adv = self.models["rssm"].imagine_candidates(h_next, post_stoch, t_cf, horizon=3)

            # World surprise = KL(post || prior)
            kl_surp = (p_post * (torch.log(p_post + 1e-7) - torch.log(p_prior + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)

            # 5-Specialist Phasic Novelty `[P:VisErr, I:SpaShift, W:WorldKL+Spread, M:Hazard, S:SelfErr]`
            nov_p = t_perr.mean(dim=(1, 2, 3)) * 2.4
            comp_diff = self._t(np.linalg.norm(compasses - prev_comp, axis=-1), torch.float32)
            nov_i = torch.clamp(comp_diff * 0.95, 0.0, 2.0)
            cf_sp_t = (0.35 * p_rew + 0.40 * p_surv + 0.25 * p_adv)
            nov_w = kl_surp * 1.35 + 0.65 * (cf_sp_t.max(dim=-1).values - cf_sp_t.min(dim=-1).values)
            nov_m = haz_p * 1.15 + 0.45 * torch.clamp(0.35 - t_prop[:, PROPRIO_IDX_RESOURCE_CARRIER], min=0.0)
            nov_s = t_prev_ssurp * 1.80
            t_nov5 = torch.stack([nov_p, nov_i, nov_w, nov_m, nov_s], dim=-1)

            strat_emb, mode_logits, focus_off_xz = self.models["strat_tf"](
                self._t(strat_seq, torch.float32), meta_err9=t_meta_err9
            )
            gwt_base, gwt_alpha, ignition, _ = self.models["gwt"](
                z_perc, slot_ctx64, t_mec, t_hd, t_comp, h_next, post_stoch, t_prop, haz_p, strat_emb,
                t_prev_gwt, meta_salience=t_prev_ssurp, hab_trace=t_prev_hab, novelty_5=t_nov5,
            )
            delta_gwt, next_spot_16 = self.models["topo_adapter"].compute_topdown_spotlight(
                gwt_base, slot_aux["feat16"], slots, s_coords
            )
            gwt_w = gwt_base + delta_gwt
            z_td = (0.08 + 0.24 * ignition).unsqueeze(-1) * self.models["gwt"].topdown_z(gwt_w)

            self_s, self_surp, self_err_vec, _ = self.models["self_model"](
                gwt_w, t_prop, self._t(prev_self, torch.float32),
                gwt_alpha=gwt_alpha, prev_act=t_prev_act, prev_alpha=t_prev_alpha,
            )

            z_eff = torch.tanh(z_perc + 0.70 * z_td)
            nov5_np = t_nov5.cpu().numpy()
            q_raw, spd_mod = self.models["d3qn"](z_eff, gwt_w, h_next, t_mec, t_hd, t_prop, t_comp, t_cf)

            slots_np = slots.cpu().numpy()
            pstats_np = slot_aux["pixel_stats"].cpu().numpy()
            next_spot_np = next_spot_16.cpu().numpy()
            z_np = z_eff.cpu().numpy()
            z_td_np = z_td.cpu().numpy()
            h_np = h_next.cpu().numpy()
            st_np = post_stoch.cpu().numpy()
            gwt_np = gwt_w.cpu().numpy()
            alp_np = gwt_alpha.cpu().numpy()
            ign_np = ignition.cpu().numpy()
            self_np = self_s.cpu().numpy()
            ssurp_np = self_surp.cpu().numpy()
            serr_np = self_err_vec.cpu().numpy()
            wsurp_np = kl_surp.cpu().numpy()
            q_np = q_raw.cpu().numpy()
            rew_np = p_rew.cpu().numpy()
            surv_np = p_surv.cpu().numpy()
            adv_np = p_adv.cpu().numpy()
            spd_np = spd_mod.cpu().numpy()
            mode_np = torch.argmax(mode_logits, dim=-1).cpu().numpy()
            foff_np = focus_off_xz.cpu().numpy()
            sc_np = s_coords.cpu().numpy()
            sv_np = s_val.cpu().numpy()
            sctx_np = slot_ctx64.cpu().numpy()

        results: List[Dict[str, Any]] = []
        for i, req in enumerate(batch_reqs):
            st = states[i]
            v_surp = 0.0
            if has_prev_vis[i]:
                pe_flat = perr_16_np[i].ravel()
                top16_pe = float(np.mean(np.partition(pe_flat, -16)[-16:])) if pe_flat.size >= 16 else float(np.mean(pe_flat))
                pix_surp = 0.45 * float(np.mean(pe_flat)) + 0.55 * top16_pe
                slot_surp = float(np.mean((slots_np[i] - st["prev_slots"]) ** 2)) if st["prev_slots"] is not None else 0.0
                v_surp = float(0.65 * pix_surp + 0.35 * slot_surp)
            elif st["prev_slots"] is not None:
                v_surp = float(np.mean((slots_np[i] - st["prev_slots"]) ** 2))

            if "ctx_ring" in st:
                st["ctx_ring"].append((
                    gwt_np[i].tobytes(),
                    {
                        "prev_h": prev_h[i].copy(),
                        "prev_st": prev_st[i].copy(),
                        "prev_act": prev_act[i].copy(),
                        "prev_gwt": prev_gwt[i].copy(),
                        "prev_alpha": prev_alpha[i].copy(),
                        "prev_hab": prev_hab[i].copy(),
                        "prev_z_td": prev_z_td[i].copy(),
                        "prev_comp": prev_comp[i].copy(),
                        "prev_self": prev_self[i].copy(),
                        "prev_self_err": prev_self_err[i].copy(),
                        "prev_self_surp": float(prev_self_surp[i]),
                        "prev_perr_16": perr_16_np[i].copy(),
                        "prev_spot_16": spot_16_np[i].copy(),
                        "strat_seq": strat_seq[i].copy(),
                    },
                ))

            sym_id, mod_id, proto_emb, proto_bias = self.rvq.quantize(
                gwt_np[i], self_np[i], h_np[i], req["cand_verbs"], update_ema=bool(req.get("enable_learning", True))
            )

            # Plan A Strict Motor Isolation: Broca inner-speech bias is ZERO unless explicitly unlocked!
            broca_bias = np.zeros(NUM_CANDIDATES, dtype=np.float32)
            if bool(getattr(self.cfg, "allow_broca_motor_bias", False)) and ("lang_cand_bias" in req):
                broca_bias = np.asarray(req["lang_cand_bias"], dtype=np.float32).reshape(NUM_CANDIDATES) * float(self.cfg.broca_prior_gain)

            q_fused = (
                q_np[i]
                + 0.30 * rew_np[i]
                + 0.25 * (adv_np[i] - 0.5)
                + 0.15 * proto_bias
                + broca_bias
            ).astype(np.float32)

            # Layer-1 Temporal Alignment: predict step `t+1` visual field & slots conditioned on step `t` action!
            prov_idx = int(np.argmax(q_fused))
            prov_cf32 = c_feats[i, prov_idx].copy()
            with torch.no_grad():
                pred_v8_i, pred_sl_i = self.models["slot_cortex"].predict_next_visual_field(
                    self._t(slots_np[i : i + 1], torch.float32),
                    self._t(pstats_np[i : i + 1], torch.float32),
                    self._t(sc_np[i : i + 1], torch.float32),
                    self._t(prov_cf32[None, :], torch.float32),
                    rssm_h=self._t(h_np[i : i + 1], torch.float32),
                )
            st["_last_vis_step"] = (
                slots_np[i].copy(),
                pstats_np[i].copy(),
                sc_np[i].copy(),
                h_np[i].copy(),
                prov_cf32.copy(),
            )
            pred_v8_np = pred_v8_i[0].cpu().numpy().astype(np.float32)
            st["prev_pred_vis8"] = pred_v8_np
            st["prev_slots"] = pred_sl_i[0].cpu().numpy().astype(np.float32)
            st["prev_topdown_gain_16"] = next_spot_np[i].copy()
            st["prev_cand_feat32"] = prov_cf32
            st["rssm_h"] = h_np[i].copy()
            st["rssm_stoch"] = st_np[i].copy()
            st["gwt_w"] = gwt_np[i].copy()
            st["prev_alpha"] = alp_np[i].copy()
            dishab = np.exp(-0.90 * np.clip(nov5_np[i], 0.0, 2.0))
            st["hab_alpha"] = np.clip(0.80 * (st["hab_alpha"] * dishab + 0.20 * (1.0 - dishab)) + 0.20 * alp_np[i], 0.02, 0.98).astype(np.float32)
            st["prev_compass"] = compasses[i].copy()
            st["self_s"] = self_np[i].copy()
            st["prev_self_err"] = serr_np[i].copy()
            st["prev_self_surp"] = float(ssurp_np[i])
            st["z_td"] = z_td_np[i].copy()
            st["strat_hist"].append(np.concatenate([gwt_np[i], self_np[i], compasses[i]], axis=0).astype(np.float32))

            wm_pred_depth_u8 = np.clip(np.rint(np.clip(pred_v8_np[:, :, 0], 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)
            wm_pred_surf_u8 = np.clip(np.rint(np.clip(pred_v8_np[:, :, 3], 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)
            if has_prev_vis[i]:
                wm_pred_err_u8 = np.clip(np.rint(np.clip(perr_16_np[i, 0] * 2.5, 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)
                wm_vis_mae = float(np.mean(perr_16_np[i]))
            else:
                wm_pred_err_u8 = np.zeros((FOVEA_H, FOVEA_W), dtype=np.uint8)
                wm_vis_mae = 0.0
            cf_util = 0.35 * rew_np[i] + 0.40 * surv_np[i] + 0.25 * adv_np[i]
            wm_cf_spread = float(np.max(cf_util) - np.min(cf_util))
            surv_thr = float(self.cfg.survival_veto_threshold)
            wm_veto_mask = 0
            for c_k in range(NUM_CANDIDATES):
                if float(surv_np[i, c_k]) < surv_thr:
                    wm_veto_mask |= (1 << c_k)

            bel_pos = np.asarray(req.get("believed_self_pos", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            focus_xyz = np.array([
                float(bel_pos[0] + foff_np[i, 0]),
                float(bel_pos[1]),
                float(bel_pos[2] + foff_np[i, 1]),
            ], dtype=np.float32)

            results.append({
                "latent_z": z_np[i],
                "z_topdown_delta": z_td_np[i],
                "rssm_h": h_np[i],
                "gwt_workspace": gwt_np[i],
                "gwt_alpha": alp_np[i],
                "ignition": float(ign_np[i]),
                "self_state": self_np[i],
                "self_surprise": float(ssurp_np[i]),
                "world_surprise": float(wsurp_np[i]),
                "q_raw_values": q_np[i].astype(np.float32),
                "q_values": q_fused,
                "pred_rew": rew_np[i].astype(np.float32),
                "survival_probs": surv_np[i].astype(np.float32),
                "pred_adv_probs": adv_np[i].astype(np.float32),
                "speed_mod": spd_np[i].astype(np.float32),
                "proto_sym_id": int(sym_id),
                "proto_mod_id": int(mod_id),
                "proto_embed": proto_emb,
                "proto_prior_bias": proto_bias,
                "topdown_goal_mode": int(mode_np[i]),
                "topdown_focus_xyz": focus_xyz,
                "vis_slot_coords": sc_np[i].astype(np.float32),
                "vis_slot_valence": sv_np[i].astype(np.float32),
                "vis_pred_surprise": float(v_surp),
                "wm_has_prev_pred": bool(has_prev_vis[i]),
                "wm_veto_mask_u8": int(wm_veto_mask),
                "wm_vis_mae": float(wm_vis_mae),
                "wm_depth_mae_m": float(depth_mae_m_list[i]),
                "wm_cf_spread": float(wm_cf_spread),
                "wm_pred_depth_u8": wm_pred_depth_u8,
                "wm_pred_surf_u8": wm_pred_surf_u8,
                "wm_pred_err_u8": wm_pred_err_u8,
                "slot_ctx64": sctx_np[i].astype(np.float32),
                "cand_feats32": c_feats[i].astype(np.float32),
                "prov_chosen_idx": int(prov_idx),
            })
        return results

    def notify_chosen_action(self, slot_id: int, chosen_cand_feat32: np.ndarray) -> None:
        st = self._agent_states.get(int(slot_id))
        if st is not None:
            cf32 = np.asarray(chosen_cand_feat32, dtype=np.float32).reshape(32).copy()
            st["prev_cand_feat32"] = cf32
            last_vs = st.get("_last_vis_step")
            if last_vs is not None:
                sl_np, ps_np, sc_np, h_np, prov_cf = last_vs
                if float(np.max(np.abs(cf32 - prov_cf))) > 1e-5:
                    with self.torch.no_grad():
                        pred_v8, pred_sl = self.models["slot_cortex"].predict_next_visual_field(
                            self._t(sl_np[None, ...], self.torch.float32),
                            self._t(ps_np[None, ...], self.torch.float32),
                            self._t(sc_np[None, ...], self.torch.float32),
                            self._t(cf32[None, :], self.torch.float32),
                            rssm_h=self._t(h_np[None, :], self.torch.float32),
                        )
                    st["prev_pred_vis8"] = pred_v8[0].cpu().numpy().astype(np.float32)
                    st["prev_slots"] = pred_sl[0].cpu().numpy().astype(np.float32)
                    st["_last_vis_step"] = (sl_np, ps_np, sc_np, h_np, cf32.copy())

    def infer_broca_utterance(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visually grounded inner-speech via Broca-Wernicke-ConceptGrounder arcuate self-monitoring."""
        torch = self.torch
        F = torch.nn.functional
        gwt_w = self._t(np.asarray(req["gwt_workspace"], dtype=np.float32)[None, :], torch.float32)
        p_emb = self._t(np.asarray(req["proto_embed"], dtype=np.float32)[None, :], torch.float32)
        s_ctx = self._t(np.asarray(req.get("slot_ctx64", np.zeros(MULTIMODAL_EMBED_DIM, dtype=np.float32)), dtype=np.float32)[None, :], torch.float32)
        sym_id = int(req.get("proto_sym_id", 0))

        with torch.no_grad():
            seq0, h0, bias0, lp0 = self.models["broca"].generate_inner_speech(gwt_w, p_emb, s_ctx, proto_sym_id=sym_id, alt_first_word=False)
            seq1, h1, bias1, lp1 = self.models["broca"].generate_inner_speech(gwt_w, p_emb, s_ctx, proto_sym_id=sym_id, alt_first_word=True)

            # Wernicke + SlotConceptLanguageGrounder3D arcuate-fasciculus multimodal alignment check
            seq_pair = torch.cat([seq0, seq1], dim=0)
            wern_pair = self.models["broca"].encode_tokens(seq_pair)
            v_norm = F.normalize(self.models["concept_grounder"].vis_proj(s_ctx), dim=-1, eps=1e-6)
            l_norm = F.normalize(self.models["concept_grounder"].lang_proj(wern_pair), dim=-1, eps=1e-6)
            cos_pair = (v_norm * l_norm).sum(dim=-1)
            lp_pair = torch.cat([lp0, lp1], dim=0)
            has_vis_ctx = float(s_ctx.abs().max().item()) > 1e-6
            scores = lp_pair + (0.35 * cos_pair if has_vis_ctx else 0.0)
            pick_alt = bool(scores[1].item() > scores[0].item())

            seq = seq1 if pick_alt else seq0
            h_emb = h1 if pick_alt else h0
            raw_bias = bias1 if pick_alt else bias0
            align_cos = float(cos_pair[1 if pick_alt else 0].item())

            toks = seq[0].cpu().numpy().astype(np.int32)
            emb_np = h_emb[0].cpu().numpy().astype(np.float32)
            bias_np = raw_bias[0].cpu().numpy().astype(np.float32)

        if not bool(getattr(self.cfg, "allow_broca_motor_bias", False)):
            bias_np.fill(0.0)

        text = self.vocab.decode_grounded_tokens(
            toks,
            in_refuge=bool(req.get("in_refuge", False)),
            carrying_obj=bool(req.get("carrying_obj", False)),
            hazard_level=float(req.get("hazard_level", 0.0)),
            audio_loud=float(req.get("audio_loud", 0.0)),
        )
        return {
            "tokens": toks,
            "embed": emb_np,
            "cand_bias": bias_np,
            "text": text,
            "concept_align_cos": align_cos,
        }

    def _clear_temporal_queues(self) -> None:
        self._nstep_queues.clear()
        self._slot_chunk4_win.clear()
        self.chunk4_ring.clear()

    def _commit_nstep_front(self, slot_id: int, q: deque, force_terminal_boundary: bool = False) -> None:
        """
        Pop and commit the oldest transition in `q` with strict separation between:
          1. Immediate 1-step outcome (`reward_1step`, `next_state_1step`, `done_1step`) for
             Visual Predictive Coding (`predict_next_visual_field`) and 1-step RSSM supervision.
          2. True N-step Bellman return (`nstep_return`, `nstep_gamma`, `nstep_done`, `next_state_nstep`)
             for Double-DQN TD targets, plus convex `(0.70, 0.20, 0.10)` `rssm_rew_target` matching
             `CategoricalRSSM3D.imagine_candidates` scale!
        """
        if not q:
            return
        items = list(q)[:3]
        committed_tr = dict(q.popleft())
        ep0 = int(committed_tr["episode_id"])

        g_ret = 0.0
        gam = 1.0
        nstep_done = bool(force_terminal_boundary)
        last_valid_ns = committed_tr.get("next_state_1step", committed_tr.get("next_state"))

        horizon_w = (0.70, 0.20, 0.10)
        rssm_w_acc = 0.0
        rssm_w_sum = 0.0

        for k, item in enumerate(items):
            if k > 0 and int(item["episode_id"]) != ep0:
                nstep_done = True
                break
            r_k = float(item.get("reward_1step", item.get("reward", 0.0)))
            if not math.isfinite(r_k):
                r_k = 0.0
            g_ret += gam * r_k
            gam *= 0.96
            wk = horizon_w[k] if k < len(horizon_w) else 0.10
            rssm_w_acc += wk * r_k
            rssm_w_sum += wk
            ns_k = item.get("next_state_1step", item.get("next_state"))
            if ns_k is not None:
                last_valid_ns = ns_k
            if bool(item.get("done", False)) or bool(item.get("respawn", False)):
                nstep_done = True
                break

        r_1 = float(committed_tr.get("reward_1step", committed_tr.get("reward", 0.0)))
        if not math.isfinite(r_1):
            r_1 = 0.0
        committed_tr["reward_1step"] = r_1
        committed_tr["done_1step"] = bool(
            committed_tr.get("done", False) or committed_tr.get("respawn", False) or force_terminal_boundary
        )
        committed_tr["next_state_1step"] = committed_tr.get("next_state_1step", committed_tr.get("next_state"))
        committed_tr["rssm_rew_target"] = float(rssm_w_acc / max(1e-6, rssm_w_sum)) if rssm_w_sum > 0.0 else r_1
        committed_tr["nstep_return"] = float(g_ret) if math.isfinite(g_ret) else 0.0
        committed_tr["nstep_gamma"] = float(gam)
        committed_tr["nstep_done"] = bool(nstep_done)
        committed_tr["next_state_nstep"] = last_valid_ns

        wsurp = float(committed_tr.get("world_surprise", 0.2))
        if not math.isfinite(wsurp):
            wsurp = 0.2
        ssurp = float(committed_tr.get("self_surprise", 0.0))
        if not math.isfinite(ssurp):
            ssurp = 0.0
        prio = max(0.15, abs(float(committed_tr["nstep_return"])) + 0.50 * wsurp + 0.45 * ssurp)
        self.replay.push(committed_tr, prio)

        # Build true per-slot consecutive 4-step trajectory chunks in O(1) (immune to multi-agent interleaving!)
        cwin = self._slot_chunk4_win.setdefault(int(slot_id), deque(maxlen=4))
        if cwin and (
            int(cwin[-1]["episode_id"]) != ep0
            or bool(cwin[-1].get("done_1step", False))
        ):
            cwin.clear()
        cwin.append(committed_tr)
        if len(cwin) == 4:
            self.chunk4_ring.append(tuple(cwin))

    def record_transition_and_maybe_train(self, tr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store experience transition into Prioritized Replay (with per-slot 3-step return accumulation
        and strict 1-step vs N-step target separation) and run online Continual Learning.
        Note: `notify_chosen_action` is invoked synchronously at step `t` decision time (never here
        from delayed `t-k` transitions, preventing temporal efference-copy overwrite!).
        """
        self.transitions_recorded += 1
        slot_id = int(tr.get("slot_id", 0))
        tr_clean = dict(tr)
        if "_gpu_ctx" not in tr_clean and "gwt_workspace" in tr_clean:
            st = self._agent_states.get(slot_id)
            if st is not None and "ctx_ring" in st:
                key_bytes = np.asarray(tr_clean["gwt_workspace"], dtype=np.float32).tobytes()
                for g_bytes, ctx_dict in reversed(st["ctx_ring"]):
                    if g_bytes == key_bytes:
                        tr_clean["_gpu_ctx"] = ctx_dict
                        break

        raw_rew = float(tr_clean.get("reward", 0.0))
        if not math.isfinite(raw_rew):
            raw_rew = 0.0
        tr_clean["reward"] = raw_rew
        tr_clean["reward_1step"] = raw_rew
        tr_clean["next_state_1step"] = tr_clean.get("next_state")

        self.rvq.reinforce_symbol_verb(
            int(tr_clean.get("proto_sym_id", 0)),
            int(tr_clean.get("chosen_verb", 0)),
            raw_rew - 0.8 * max(0.0, float(tr_clean.get("next_hazard", 0.0)) - float(tr_clean.get("hazard", 0.0))),
        )

        # Per-slot 3-step rolling window (zero duplicate insertion, full terminal/respawn flush!)
        q = self._nstep_queues.setdefault(slot_id, deque())
        if q and int(q[-1]["episode_id"]) != int(tr_clean["episode_id"]):
            while q:
                self._commit_nstep_front(slot_id, q, force_terminal_boundary=True)

        q.append(tr_clean)
        is_terminal_or_respawn = bool(tr_clean.get("done", False)) or bool(tr_clean.get("respawn", False))
        if is_terminal_or_respawn:
            while q:
                self._commit_nstep_front(slot_id, q, force_terminal_boundary=False)
        elif len(q) >= 3:
            self._commit_nstep_front(slot_id, q, force_terminal_boundary=False)

        if bool(tr_clean.get("done", False)) and "episode_score" in tr_clean:
            ep_sc = float(tr_clean["episode_score"])
            if math.isfinite(ep_sc):
                self._ep_scores.append(ep_sc)
                if len(self._ep_scores) >= 3:
                    mean_sc = float(np.mean(self._ep_scores))
                    if mean_sc > self._best_rolling_score:
                        if self._snapshot_anchor_weights():
                            self._best_rolling_score = mean_sc
                    elif (
                        self.cfg.enable_overfit_guard
                        and self._anchor_weights is not None
                        and mean_sc < self._best_rolling_score - 8.5
                    ):
                        self._blend_toward_anchor(alpha=0.30)
                        self.overfit_guard_tripped = True

        loss_val = 0.0
        if (
            bool(tr_clean.get("enable_learning", self.cfg.enable_learning))
            and len(self.replay) >= 16
            and (self.transitions_recorded % max(1, int(self.cfg.online_update_interval)) == 0)
        ):
            loss_val = self._train_minibatch(batch_size=16)

        return {
            "transitions_recorded": self.transitions_recorded,
            "train_steps_done": self.train_steps_done,
            "loss": float(loss_val),
            "overfit_guard_tripped": bool(self.overfit_guard_tripped),
            "nan_rollback_count": int(self.nan_rollback_count),
        }

    def _snapshot_anchor_weights(self) -> bool:
        """Snapshot 100% clean core weights only if all tensors are finite and numerically bounded."""
        for k, m in self.models.items():
            if k in ("d3qn_target", "depth_net"):
                continue
            for p in m.parameters():
                if (not bool(self.torch.isfinite(p).all().item())) or float(p.detach().abs().max().item()) > 1e3:
                    return False
        self._anchor_weights = {
            k: {pn: p.detach().clone() for pn, p in m.named_parameters()}
            for k, m in self.models.items()
            if k not in ("d3qn_target", "depth_net")
        }
        return True

    def _emergency_rollback_to_anchor(self) -> None:
        """Immediately restore 100% clean anchor weights, sync target network, and purge polluted AdamW state."""
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()
        with self.torch.no_grad():
            if self._anchor_weights is not None:
                for k, m in self.models.items():
                    if k in self._anchor_weights:
                        for pn, p in m.named_parameters():
                            if pn in self._anchor_weights[k]:
                                p.data.copy_(self._anchor_weights[k][pn])
            else:
                for k, m in self.models.items():
                    if k not in ("d3qn_target", "depth_net"):
                        for p in m.parameters():
                            p.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).clamp_(-5.0, 5.0)
            self.models["d3qn_target"].load_state_dict(self.models["d3qn"].state_dict())
        self.overfit_guard_tripped = True
        self.nan_rollback_count += 1

    def _blend_toward_anchor(self, alpha: float = 0.25) -> None:
        if self._anchor_weights is None:
            return
        with self.torch.no_grad():
            for k, m in self.models.items():
                if k in self._anchor_weights:
                    for pn, p in m.named_parameters():
                        if pn in self._anchor_weights[k]:
                            p.data.mul_(1.0 - alpha).add_(self._anchor_weights[k][pn] * alpha)

    def _synthesize_endogenous_tokens_np(self, b: Dict[str, Any]) -> np.ndarray:
        """Build deterministic grounded token sequence within CORE_WORDS (`36..83`) when no external teacher text is supplied."""
        prop = np.asarray(b["proprio"], dtype=np.float32)
        verb = int(b.get("chosen_verb", VERB_APPROACH_HOVER))
        sym_id = int(b.get("proto_sym_id", 0)) % 32
        haz = max(float(b.get("next_hazard", 0.0)), float(prop[PROPRIO_IDX_HAZARD]) if prop.shape[0] > PROPRIO_IDX_HAZARD else 0.0)
        obj_p = float(prop[PROPRIO_IDX_OBJ_PROGRESS]) if prop.shape[0] > PROPRIO_IDX_OBJ_PROGRESS else 0.5

        if is_carrying_primary_objective(prop):
            w1 = 49  # 目標確保
        elif verb == VERB_EVADE_HAZARD:
            w1 = 64  # 緊急離脱
        elif is_in_stealth_or_cover_refuge(prop):
            w1 = 57  # 潜伏回復
        elif verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH):
            w1 = 62  # 偏差射撃
        elif verb in (VERB_COMMIT_INTERACT, VERB_GRASP_PICK):
            w1 = 48  # 目標発見
        else:
            w1 = 36  # 前進

        if haz >= 0.45:
            w2 = 60  # 敵接近
        elif obj_p >= 0.56:
            w2 = 72  # 優勢
        elif obj_p <= 0.44:
            w2 = 73  # 劣勢
        else:
            w2 = 46  # 塗拡

        seq = np.full(INNER_SPEECH_MAX_LEN, GroundedVocabulary512.PAD_ID, dtype=np.int32)
        seq[0] = GroundedVocabulary512.BOS_ID
        seq[1] = GroundedVocabulary512.PROTO_OFFSET + sym_id
        seq[2] = w1
        seq[3] = w2
        seq[4] = GroundedVocabulary512.EOS_ID
        return seq

    def _train_minibatch(self, batch_size: int = 16, dreaming_noise: float = 0.0) -> float:
        torch = self.torch
        F = self.F
        n_buf = len(self.replay)
        if n_buf < 8:
            return 0.0

        beta_per = min(1.0, 0.45 + 0.55 * float(self.train_steps_done) / 2000.0)
        idxs, batch, is_weights_np = self.replay.sample_prioritized(batch_size, alpha=0.60, beta=beta_per)
        bsz = len(batch)
        is_w_t = self._t(is_weights_np, torch.float32)

        fov = self._t(np.stack([b["fovea_patch"] for b in batch]), torch.float32)
        vis8 = self._t(np.stack([compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch")) for b in batch]), torch.float32)
        prop = self._t(np.stack([b["proprio"] for b in batch]), torch.float32)
        comp = self._t(np.stack([b["mem_compass"] for b in batch]), torch.float32)
        mec = self._t(np.stack([b["mec_grid_code"] for b in batch]), torch.float32)
        hd = self._t(np.stack([b["hd_ring_code"] for b in batch]), torch.float32)
        c_feats = self._t(np.stack([b["cand_feats32"] for b in batch]), torch.float32)

        def _ctx_stack(key: str, fallback_fn: Callable[[Dict[str, Any]], np.ndarray]):
            arrs = [
                b["_gpu_ctx"][key] if ("_gpu_ctx" in b and key in b["_gpu_ctx"]) else fallback_fn(b)
                for b in batch
            ]
            return self._t(np.stack(arrs, axis=0), torch.float32)

        def _onehot_st0(_: Dict[str, Any]) -> np.ndarray:
            arr = np.zeros(RSSM_STOCH_FLAT, dtype=np.float32)
            arr[0] = 1.0
            return arr

        prev_h_t = _ctx_stack("prev_h", lambda b: b["rssm_h"])
        prev_st_t = _ctx_stack("prev_st", _onehot_st0)
        prev_act_t = _ctx_stack("prev_act", lambda _: np.zeros(CFEAT_DIM, dtype=np.float32))
        prev_gwt_t = _ctx_stack("prev_gwt", lambda b: b["gwt_workspace"])
        prev_alpha_t = _ctx_stack("prev_alpha", lambda _: np.full(5, 0.2, dtype=np.float32))
        prev_hab_t = _ctx_stack("prev_hab", lambda _: np.full(5, 0.2, dtype=np.float32))
        prev_z_td_t = _ctx_stack("prev_z_td", lambda _: np.zeros(LATENT_Z_DIM, dtype=np.float32))
        prev_comp_t = _ctx_stack("prev_comp", lambda b: b["mem_compass"])
        prev_self_t = _ctx_stack("prev_self", lambda _: np.zeros(SELF_MODEL_DIM, dtype=np.float32))
        prev_self_err_t = _ctx_stack("prev_self_err", lambda _: np.zeros(SELF_MODEL_DIM, dtype=np.float32))
        prev_ssurp_t = self._t(
            [
                float(b["_gpu_ctx"]["prev_self_surp"])
                if ("_gpu_ctx" in b and "prev_self_surp" in b["_gpu_ctx"])
                else float(b.get("self_surprise", 0.0))
                for b in batch
            ],
            torch.float32,
        )
        prev_perr_t = _ctx_stack("prev_perr_16", lambda _: np.zeros((1, FOVEA_H, FOVEA_W), dtype=np.float32))
        prev_spot_t = _ctx_stack("prev_spot_16", lambda _: np.ones((1, FOVEA_H, FOVEA_W), dtype=np.float32))
        meta_err9_t = torch.cat([prev_self_err_t, prev_ssurp_t.unsqueeze(-1)], dim=-1)
        strat_seq_t = _ctx_stack(
            "strat_seq",
            lambda b: np.repeat(
                np.concatenate([b["gwt_workspace"], np.zeros(SELF_MODEL_DIM, dtype=np.float32), b["mem_compass"]], axis=0)[None, :],
                8,
                axis=0,
            ).astype(np.float32),
        )

        chosen_idx = self._t([int(b["chosen_idx"]) for b in batch], torch.long)
        rewards_nstep = self._t([float(b["nstep_return"]) for b in batch], torch.float32)
        gammas_nstep = self._t([float(b["nstep_gamma"]) for b in batch], torch.float32)
        dones_nstep = self._t([1.0 if b.get("nstep_done", b.get("done") or b.get("respawn")) else 0.0 for b in batch], torch.float32)

        rewards_1step = self._t([float(b.get("reward_1step", b.get("reward", 0.0))) for b in batch], torch.float32)
        rssm_rew_tgt = self._t([float(b.get("rssm_rew_target", b.get("reward_1step", b.get("reward", 0.0)))) for b in batch], torch.float32)
        dones_1step = self._t([1.0 if b.get("done_1step", b.get("done") or b.get("respawn")) else 0.0 for b in batch], torch.float32)

        haz_tgt = self._t([float(b.get("next_hazard", 0.0)) for b in batch], torch.float32)
        surv_tgt = self._t([0.05 if b.get("respawn") else float(np.clip(1.0 - 0.85 * float(b.get("next_hazard", 0.0)), 0.05, 0.99)) for b in batch], torch.float32)
        adv_tgt = self._t([1.0 if float(b.get("obj_adv_delta", 0.0)) > 0.002 or float(b.get("reward_1step", b["reward"])) > 0.05 else 0.35 for b in batch], torch.float32)

        # Strictly separate 1-step next_state (`s_{t+1}` for visual predictive coding) vs N-step next_state (`s_{t+n}` for Double-DQN)
        ns1_list = [b.get("next_state_1step", b.get("next_state")) for b in batch]
        nvis8_1 = self._t(np.stack([
            compose_vis8_patch(ns["fovea_patch"], ns.get("ventral_patch"))
            if ns
            else compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch"))
            for ns, b in zip(ns1_list, batch)
        ]), torch.float32)

        nsn_list = [b.get("next_state_nstep", b.get("next_state")) for b in batch]
        nfov_n = self._t(np.stack([ns["fovea_patch"] if ns else b["fovea_patch"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        nvis8_n = self._t(np.stack([
            compose_vis8_patch(ns["fovea_patch"], ns.get("ventral_patch"))
            if ns
            else compose_vis8_patch(b["fovea_patch"], b.get("ventral_patch"))
            for ns, b in zip(nsn_list, batch)
        ]), torch.float32)
        nprop_n = self._t(np.stack([ns["proprio"] if ns else b["proprio"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        ncomp_n = self._t(np.stack([ns["mem_compass"] if ns else b["mem_compass"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        nmec_n = self._t(np.stack([ns["mec_grid_code"] if ns else b["mec_grid_code"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        nhd_n = self._t(np.stack([ns["hd_ring_code"] if ns else b["hd_ring_code"] for ns, b in zip(nsn_list, batch)]), torch.float32)
        ncf_n = self._t(np.stack([ns["cand_feats32"] if ns else b["cand_feats32"] for ns, b in zip(nsn_list, batch)]), torch.float32)

        self.optimizer.zero_grad()

        z = self.models["foveal_enc"](fov, prop, comp)
        if dreaming_noise > 0.0:
            z = torch.tanh(z + torch.randn_like(z) * dreaming_noise)
        slots, s_coords, s_val, slot_aux = self.models["slot_cortex"](
            vis8,
            pred_err_16=prev_perr_t,
            topdown_gain_16=prev_spot_t,
            td_context=torch.cat([prev_gwt_t, prev_h_t], dim=-1),
            return_aux=True,
        )
        slot_attn = slot_aux["attn"]
        z_perc = torch.tanh(z + slot_aux["delta_z"] + 0.30 * prev_z_td_t)
        haz_pred = self.models["hazard_net"](z_perc, prop, comp)
        slot_ctx64 = self.models["topo_adapter"](slots, mec, hd, comp)

        # Whole-Brain Forward Graph aligned 1-to-1 with `infer_conscious_gwt_batch`
        chosen_cf = c_feats.gather(1, chosen_idx.view(bsz, 1, 1).expand(-1, 1, CFEAT_DIM)).squeeze(1)
        h_post, post_st, p_post, p_prior = self.models["rssm"].observe_step(
            prev_h_t, prev_st_t, prev_act_t, z_perc, slot_ctx64, prop, comp
        )
        p_rew, p_surv, p_adv = self.models["rssm"].imagine_candidates(h_post, post_st, c_feats, horizon=3)

        kl_surp_tr = (p_post.detach() * (torch.log(p_post.detach() + 1e-7) - torch.log(p_prior.detach() + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
        cf_sp_tr = (0.35 * p_rew.detach() + 0.40 * p_surv.detach() + 0.25 * p_adv.detach())
        nov5_tr = torch.stack([
            prev_perr_t.mean(dim=(1, 2, 3)) * 2.4,
            torch.clamp((comp - prev_comp_t).norm(dim=-1) * 0.95, 0.0, 2.0),
            kl_surp_tr * 1.35 + 0.65 * (cf_sp_tr.max(dim=-1).values - cf_sp_tr.min(dim=-1).values),
            haz_pred.detach() * 1.15 + 0.45 * torch.clamp(0.35 - prop[:, PROPRIO_IDX_RESOURCE_CARRIER], min=0.0),
            prev_ssurp_t * 1.80,
        ], dim=-1)

        strat_emb, mode_logits, focus_off_xz = self.models["strat_tf"](strat_seq_t, meta_err9=meta_err9_t)
        gwt_base, alp_live, ign_live, _ = self.models["gwt"](
            z_perc, slot_ctx64, mec, hd, comp, h_post.detach(), post_st.detach(), prop, haz_pred.detach(), strat_emb,
            prev_gwt_t, meta_salience=prev_ssurp_t, hab_trace=prev_hab_t, novelty_5=nov5_tr,
        )
        delta_gwt, topdown_gain_16 = self.models["topo_adapter"].compute_topdown_spotlight(
            gwt_base, slot_aux["feat16"], slots, s_coords
        )
        gwt_live = gwt_base + delta_gwt
        z_td = (0.08 + 0.24 * ign_live).unsqueeze(-1) * self.models["gwt"].topdown_z(gwt_live)

        self_cur, self_surp_live, self_err_live, self_pred = self.models["self_model"](
            gwt_live.detach(), prop, prev_self_t,
            gwt_alpha=alp_live.detach(), prev_act=prev_act_t, prev_alpha=prev_alpha_t,
        )
        # Straight-through 0.25 gradient scaling protects foveal_enc depth/hazard representations from 9x TD dominance
        z_perc_for_q = 0.25 * z_perc + 0.75 * z_perc.detach()
        z_eff = torch.tanh(z_perc_for_q + 0.70 * z_td)

        # 1. Double-DQN N-Step TD Loss (with PER Importance Sampling weights) & Speed Modulation Head
        q_all, spd_all = self.models["d3qn"](z_eff, gwt_live, h_post.detach(), mec, hd, prop, comp, c_feats)
        q_chosen = q_all.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 1-step next-slot teacher target strictly from `s_{t+1}` (`nvis8_1`)
            n_pred_v8_1, _ = self.models["slot_cortex"].predict_next_visual_field(
                slots, slot_aux["pixel_stats"], s_coords, chosen_cf, rssm_h=h_post
            )
            n_perr_16_1 = (nvis8_1 - n_pred_v8_1).abs().mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
            nslots_1, _, _, _ = self.models["slot_cortex"](
                nvis8_1, pred_err_16=n_perr_16_1, topdown_gain_16=topdown_gain_16, return_aux=True
            )

            # N-step bootstrap value strictly from `s_{t+n}` (`nfov_n`, `nvis8_n`, `nprop_n`, `ncomp_n`, ...)
            nz_n = self.models["foveal_enc"](nfov_n, nprop_n, ncomp_n)
            nslots_n, ns_coords_n, _, nslot_aux_n = self.models["slot_cortex"](
                nvis8_n, topdown_gain_16=topdown_gain_16, return_aux=True
            )
            nz_perc_n = torch.tanh(nz_n + nslot_aux_n["delta_z"])
            nhaz_n = self.models["hazard_net"](nz_perc_n, nprop_n, ncomp_n)
            nslot_ctx64_n = self.models["topo_adapter"](nslots_n, nmec_n, nhd_n, ncomp_n)
            nh_next_n, npost_st_n, _, _ = self.models["rssm"].observe_step(
                h_post, post_st, chosen_cf, nz_perc_n, nslot_ctx64_n, nprop_n, ncomp_n
            )
            n_snap = torch.cat([gwt_live, self_cur, comp], dim=-1).unsqueeze(1)
            n_strat_seq = torch.cat([strat_seq_t[:, 1:, :], n_snap], dim=1)
            n_meta_err9 = torch.cat([self_err_live, self_surp_live.unsqueeze(-1)], dim=-1)
            n_strat_emb, _, _ = self.models["strat_tf"](n_strat_seq, meta_err9=n_meta_err9)
            ngwt_base, _, nign, _ = self.models["gwt"](
                nz_perc_n, nslot_ctx64_n, nmec_n, nhd_n, ncomp_n, nh_next_n, npost_st_n, nprop_n, nhaz_n, n_strat_emb,
                gwt_live, meta_salience=self_surp_live,
            )
            ndelta_gwt, _ = self.models["topo_adapter"].compute_topdown_spotlight(
                ngwt_base, nslot_aux_n["feat16"], nslots_n, ns_coords_n
            )
            ngwt = ngwt_base + ndelta_gwt
            nz_td = (0.08 + 0.24 * nign).unsqueeze(-1) * self.models["gwt"].topdown_z(ngwt)
            nz_eff = torch.tanh(nz_perc_n + nz_td)
            nq_online, _ = self.models["d3qn"](nz_eff, ngwt, nh_next_n, nmec_n, nhd_n, nprop_n, ncomp_n, ncf_n)
            best_next = torch.argmax(nq_online, dim=-1, keepdim=True)
            nq_targ, _ = self.models["d3qn_target"](nz_eff, ngwt, nh_next_n, nmec_n, nhd_n, nprop_n, ncomp_n, ncf_n)
            v_next = nq_targ.gather(1, best_next).squeeze(1)
            td_target = torch.clamp(rewards_nstep + gammas_nstep * (1.0 - dones_nstep) * v_next, -32.0, 38.0)

        loss_td = (F.smooth_l1_loss(q_chosen, td_target, reduction="none") * is_w_t).mean()

        spd_base = c_feats[:, :, CFEAT_SPEED_SCALE] * (
            0.85 + 0.15 * c_feats[:, :, CFEAT_DIR_CLEARANCE] + 0.10 * (c_feats[:, :, CFEAT_DIR_FRIENDLY_SURF] - 0.5)
        )
        spd_bonus = (0.10 * (adv_tgt - 0.5) - 0.12 * dones_1step * (1.0 - surv_tgt)).unsqueeze(1)
        spd_tgt = torch.clamp(spd_base.scatter_add(1, chosen_idx.unsqueeze(1), spd_bonus), 0.25, 1.0)
        loss_speed = 0.20 * F.smooth_l1_loss(spd_all, spd_tgt)

        # 2. 3-Head Multi-Step RSSM World Model (DreamerV3 Asymmetric KL + O(1) Contiguous 4-Step Chunk BPTT + Counterfactual De-Confounding)
        rew_ch = p_rew.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
        surv_ch = p_surv.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
        adv_ch = p_adv.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)

        kl_dyn = (p_post.detach() * (torch.log(p_post.detach() + 1e-7) - torch.log(p_prior + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
        kl_rep = (p_post * (torch.log(p_post + 1e-7) - torch.log(p_prior.detach() + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
        kl_vec = kl_rep.detach()
        kl_loss = (
            0.80 * torch.clamp(kl_dyn - 0.08, min=0.0).mean()
            + 0.20 * torch.clamp(kl_rep - 0.08, min=0.0).mean()
            + 0.01 * (0.80 * kl_dyn.mean() + 0.20 * kl_rep.mean())
        )

        # O(1) Contiguous same-slot, same-episode 4-step temporal chunk BPTT + 2-step open-loop rollout supervision
        loss_rssm_seq = z.new_tensor(0.0)
        if len(self.chunk4_ring) >= 1:
            n_ch = min(4, len(self.chunk4_ring))
            ch_sel = np.random.choice(len(self.chunk4_ring), size=n_ch, replace=False)
            c_tuples = [self.chunk4_ring[int(ci)] for ci in ch_sel]

            ch_fovs = np.stack([[ck[k]["fovea_patch"] for k in range(4)] for ck in c_tuples]).reshape(n_ch * 4, FOVEA_H, FOVEA_W, FOVEA_C)
            ch_vis8 = np.stack([
                [compose_vis8_patch(ck[k]["fovea_patch"], ck[k].get("ventral_patch")) for k in range(4)]
                for ck in c_tuples
            ]).reshape(n_ch * 4, FOVEA_H, FOVEA_W, VIS8_C)
            ch_props = np.stack([[ck[k]["proprio"] for k in range(4)] for ck in c_tuples]).reshape(n_ch * 4, PROPRIO_DIM)
            ch_comps = np.stack([[ck[k]["mem_compass"] for k in range(4)] for ck in c_tuples]).reshape(n_ch * 4, MEM_COMPASS_DIM)
            ch_mecs = np.stack([[ck[k]["mec_grid_code"] for k in range(4)] for ck in c_tuples]).reshape(n_ch * 4, MEC_GRID_DIM)
            ch_hds = np.stack([[ck[k]["hd_ring_code"] for k in range(4)] for ck in c_tuples]).reshape(n_ch * 4, HD_RING_DIM)

            with torch.no_grad():
                tf_c = self._t(ch_fovs, torch.float32)
                tv_c = self._t(ch_vis8, torch.float32)
                tp_c = self._t(ch_props, torch.float32)
                tc_c = self._t(ch_comps, torch.float32)
                tm_c = self._t(ch_mecs, torch.float32)
                th_c = self._t(ch_hds, torch.float32)
                z_c0 = self.models["foveal_enc"](tf_c, tp_c, tc_c)
                sl_c, _, _, aux_c = self.models["slot_cortex"](tv_c, return_aux=True)
                zp_c = torch.tanh(z_c0 + aux_c["delta_z"]).view(n_ch, 4, LATENT_Z_DIM)
                sctx_c = self.models["topo_adapter"](sl_c, tm_c, th_c, tc_c).view(n_ch, 4, MULTIMODAL_EMBED_DIM)
                tp_seq = tp_c.view(n_ch, 4, PROPRIO_DIM)
                tc_seq = tc_c.view(n_ch, 4, MEM_COMPASS_DIM)

            h_seq = self._t(
                np.stack([
                    ck[0]["_gpu_ctx"]["prev_h"] if ("_gpu_ctx" in ck[0] and "prev_h" in ck[0]["_gpu_ctx"]) else ck[0]["rssm_h"]
                    for ck in c_tuples
                ]),
                torch.float32,
            )
            st_seq = self._t(
                np.stack([
                    ck[0]["_gpu_ctx"]["prev_st"] if ("_gpu_ctx" in ck[0] and "prev_st" in ck[0]["_gpu_ctx"]) else _onehot_st0(ck[0])
                    for ck in c_tuples
                ]),
                torch.float32,
            )
            act0_prev = self._t(
                np.stack([
                    ck[0]["_gpu_ctx"]["prev_act"] if ("_gpu_ctx" in ck[0] and "prev_act" in ck[0]["_gpu_ctx"]) else np.zeros(CFEAT_DIM, dtype=np.float32)
                    for ck in c_tuples
                ]),
                torch.float32,
            )
            act_chosen_seq = self._t(
                np.stack([[ck[k]["chosen_cand_feat32"] for k in range(4)] for ck in c_tuples]),
                torch.float32,
            )  # (n_ch, 4, 32)
            rew_seq_tgt = self._t(
                [[float(np.clip(ck[k].get("reward_1step", ck[k]["reward"]), -10.0, 15.0)) for k in range(4)] for ck in c_tuples],
                torch.float32,
            )
            surv_seq_tgt = self._t(
                [[
                    0.05 if ck[k].get("respawn") else float(np.clip(1.0 - 0.85 * float(ck[k].get("next_hazard", 0.0)), 0.05, 0.99))
                    for k in range(4)
                ] for ck in c_tuples],
                torch.float32,
            )
            adv_seq_tgt = self._t(
                [[
                    1.0 if float(ck[k].get("obj_adv_delta", 0.0)) > 0.002 or float(ck[k].get("reward_1step", ck[k]["reward"])) > 0.05 else 0.35
                    for k in range(4)
                ] for ck in c_tuples],
                torch.float32,
            )

            post_h_list = []
            post_st_list = []
            post_p_list = []
            seq_acc = z.new_tensor(0.0)
            for k in range(4):
                prev_a_k = act0_prev if k == 0 else act_chosen_seq[:, k - 1]
                h_seq, st_seq, pp_k, pr_k = self.models["rssm"].observe_step(
                    h_seq, st_seq, prev_a_k, zp_c[:, k], sctx_c[:, k], tp_seq[:, k], tc_seq[:, k]
                )
                post_h_list.append(h_seq)
                post_st_list.append(st_seq)
                post_p_list.append(pp_k)

                im_k = self.models["rssm"].imag_step(
                    torch.cat([h_seq, st_seq, self.models["rssm"].act_proj(act_chosen_seq[:, k])], dim=-1)
                )
                rk_p = self.models["rssm"].rew_head(im_k).squeeze(-1)
                sk_p = torch.sigmoid(self.models["rssm"].surv_head(im_k).squeeze(-1)).clamp(1e-4, 1.0 - 1e-4)
                ak_p = torch.sigmoid(self.models["rssm"].adv_head(im_k).squeeze(-1)).clamp(1e-4, 1.0 - 1e-4)
                kld_k = (pp_k.detach() * (torch.log(pp_k.detach() + 1e-7) - torch.log(pr_k + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
                klr_k = (pp_k * (torch.log(pp_k + 1e-7) - torch.log(pr_k.detach() + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
                kl_b_k = 0.80 * torch.clamp(kld_k - 0.08, min=0.0).mean() + 0.20 * torch.clamp(klr_k - 0.08, min=0.0).mean()
                seq_acc = seq_acc + 0.25 * (
                    0.40 * F.smooth_l1_loss(rk_p, rew_seq_tgt[:, k])
                    + 0.65 * F.binary_cross_entropy(sk_p, surv_seq_tgt[:, k])
                    + 0.30 * F.binary_cross_entropy(ak_p, adv_seq_tgt[:, k])
                    + 0.12 * kl_b_k
                )

            # 2-Step Open-Loop Multi-Horizon Imagination Consistency from k = 0 and k = 1
            ol_acc = z.new_tensor(0.0)
            for k_start in (0, 1):
                h_ol = post_h_list[k_start]
                st_ol = post_st_list[k_start]
                for m_step in (1, 2):
                    t_idx = k_start + m_step
                    h_ol, st_ol, r_ol, s_ol, a_ol, pr_ol = self.models["rssm"].imagine_step_open_loop(
                        h_ol, st_ol, act_chosen_seq[:, t_idx - 1], eval_cand_feat32=act_chosen_seq[:, t_idx]
                    )
                    pp_t = post_p_list[t_idx].detach()
                    kl_ol = (pp_t * (torch.log(pp_t + 1e-7) - torch.log(pr_ol + 1e-7))).sum(dim=(-1, -2)) / float(RSSM_STOCH_CATS)
                    w_m = 0.60 if m_step == 1 else 0.40
                    ol_acc = ol_acc + 0.50 * w_m * (
                        0.40 * F.smooth_l1_loss(r_ol, rew_seq_tgt[:, t_idx])
                        + 0.65 * F.binary_cross_entropy(s_ol.clamp(1e-4, 1.0 - 1e-4), surv_seq_tgt[:, t_idx])
                        + 0.30 * F.binary_cross_entropy(a_ol.clamp(1e-4, 1.0 - 1e-4), adv_seq_tgt[:, t_idx])
                        + 0.15 * torch.clamp(kl_ol - 0.08, min=0.0).mean()
                    )
            loss_rssm_seq = 0.35 * (seq_acc + ol_acc)

        loss_rssm = (
            0.5 * F.smooth_l1_loss(rew_ch, torch.clamp(rssm_rew_tgt, -10.0, 15.0))
            + F.binary_cross_entropy(surv_ch.clamp(1e-4, 1.0 - 1e-4), surv_tgt)
            + 0.4 * F.binary_cross_entropy(adv_ch.clamp(1e-4, 1.0 - 1e-4), adv_tgt)
            + 0.15 * kl_loss
            + loss_rssm_seq
        )

        # Counterfactual Exposure De-confounding:
        # When exposed under active danger (`not in_refuge` and `hazard >= SHIELD_HAZARD_CF_MASK`), standing still (`CAND_IDX_IDLE_OBSERVE`)
        # MUST NOT inherit optimistic survival or Q-values from successful escapes!
        cf_loss = z.new_tensor(0.0)
        exposed_mask = (
            (c_feats[:, CAND_IDX_IDLE_OBSERVE, CFEAT_IN_REFUGE] < 0.5)
            & (c_feats[:, CAND_IDX_IDLE_OBSERVE, CFEAT_HAZARD] >= SHIELD_HAZARD_CF_MASK)
        )
        if exposed_mask.any():
            # Detach factual posterior/visual latents so counterfactual M7 regularization updates ONLY
            # action-conditioned imagination & candidate-advantage heads (zero encoder gradient conflict!)
            _, p_surv_cf, _ = self.models["rssm"].imagine_candidates(
                h_post[exposed_mask].detach(), post_st[exposed_mask].detach(), c_feats[exposed_mask], horizon=3
            )
            cf_surv_m7 = p_surv_cf[:, CAND_IDX_IDLE_OBSERVE].clamp(1e-4, 1.0 - 1e-4)
            cf_tgt = torch.full_like(cf_surv_m7, 0.12)
            q_cf_exp, _ = self.models["d3qn"](
                z_eff[exposed_mask].detach(),
                gwt_live[exposed_mask].detach(),
                h_post[exposed_mask].detach(),
                mec[exposed_mask],
                hd[exposed_mask],
                prop[exposed_mask],
                comp[exposed_mask],
                c_feats[exposed_mask],
            )
            cf_loss = (
                0.45 * F.binary_cross_entropy(cf_surv_m7, cf_tgt)
                + 0.25 * F.relu(q_cf_exp[:, CAND_IDX_IDLE_OBSERVE] - q_chosen[exposed_mask].detach() + 0.6).mean()
            )

        # 3. Depth Auto-Encoding, Amygdala Hazard, Predictive Visual Field & Spotlight, Strategic & Self Losses
        recon_d = self.models["foveal_enc"].decode_depth(z_perc)
        loss_vis = 0.25 * F.mse_loss(recon_d, fov[:, :, :, 0].clamp(0.0, 1.0)) + 0.40 * F.mse_loss(haz_pred, haz_tgt)

        vis8_8x8 = F.avg_pool2d(vis8.permute(0, 3, 1, 2), kernel_size=2, stride=2).flatten(2).permute(0, 2, 1)
        attn_d = slot_attn.detach()
        slot_uv = torch.einsum("bsn,bnd->bsd", attn_d, self.models["slot_cortex"].grid_xy.expand(bsz, -1, -1))
        slot_vis = torch.einsum("bsn,bnd->bsd", attn_d, vis8_8x8)
        depth_s = slot_vis[:, :, 0].clamp(0.0, 1.0)
        ny_s = slot_vis[:, :, 2]
        mot_s = slot_vis[:, :, 7].clamp(0.0, 1.0)
        sharpness = (attn_d * 64.0).max(dim=-1).values
        tgt_coords = torch.cat([slot_uv, 2.0 * depth_s.unsqueeze(-1) - 1.0], dim=-1)

        # Domain-agnostic empirical outcome + geometric credit assignment (strictly 1-step outcomes + named compass indices!)
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
        loss_val_pool = (
            F.binary_cross_entropy(pred_hostile, emp_hostile)
            + F.binary_cross_entropy(pred_obj, emp_objective)
            + F.binary_cross_entropy(pred_cover, emp_cover)
        ) / 3.0
        soft_slot_tgt = torch.stack([
            (emp_hostile.unsqueeze(1) * (float(NUM_OBJECT_SLOTS) * w_hostile).clamp(0.0, 1.25)).clamp(0.0, 1.0),
            (emp_objective.unsqueeze(1) * (float(NUM_OBJECT_SLOTS) * w_obj).clamp(0.0, 1.25)).clamp(0.0, 1.0),
            (emp_cover.unsqueeze(1) * (float(NUM_OBJECT_SLOTS) * w_cover).clamp(0.0, 1.25)).clamp(0.0, 1.0),
        ], dim=-1)

        pred_nvis8, pred_nslots = self.models["slot_cortex"].predict_next_visual_field(
            slots.detach(), slot_aux["pixel_stats"].detach(), s_coords.detach(), chosen_cf, rssm_h=h_post.detach()
        )
        fg_pix_mask = (nvis8_1[:, :, :, 3:4] > 0.22) | (nvis8_1[:, :, :, 4:6].abs().sum(dim=-1, keepdim=True) > 0.30) | (nvis8_1[:, :, :, 7:8] > 0.20)
        pix_w = torch.where(fg_pix_mask, torch.full_like(nvis8_1, 3.0), torch.ones_like(nvis8_1))
        cont_w = (1.0 - dones_1step).view(bsz, 1, 1, 1).clamp(min=0.25)
        loss_vis_pred = 0.35 * (F.smooth_l1_loss(pred_nvis8, nvis8_1, beta=0.05, reduction="none") * pix_w * cont_w).mean()

        fg_focus_16 = torch.clamp(
            0.65 * slot_aux["sal_16"].detach().clamp(0.0, 1.0)
            + 0.35 * ((nvis8_1 - vis8).permute(0, 3, 1, 2).abs().mean(dim=1, keepdim=True) * 2.5).clamp(0.0, 1.0),
            0.0,
            1.0,
        )
        spot_tgt_16 = 0.85 + 0.55 * fg_focus_16
        loss_spot = 0.20 * F.mse_loss(topdown_gain_16, spot_tgt_16)

        loss_td_prior = (
            0.12 * F.mse_loss(torch.tanh(slot_aux["td_prior_raw"]), torch.tanh(slots.detach()))
            if slot_aux.get("td_prior_raw") is not None
            else z.new_tensor(0.0)
        )
        loss_slot = (
            0.20 * F.mse_loss(s_coords, tgt_coords)
            + 0.14 * loss_val_pool
            + 0.08 * F.mse_loss(s_val, soft_slot_tgt)
            + 0.20 * F.mse_loss(pred_nslots, nslots_1.detach())
            + loss_vis_pred
            + loss_spot
            + loss_td_prior
        )

        mode_q = torch.stack([
            q_all[:, CAND_IDX_FRONTIER_TURF],
            q_all[:, CAND_IDX_OBJECTIVE_ADVANCE],
            torch.maximum(q_all[:, CAND_IDX_STRAFE_RIGHT_ENGAGE], q_all[:, CAND_IDX_SHIELD_OR_BURST]),
            torch.maximum(
                q_all[:, CAND_IDX_EVADE_REFUGE],
                torch.maximum(q_all[:, CAND_IDX_RALLY_OR_PICKUP], q_all[:, CAND_IDX_IDLE_OBSERVE]),
            ),
            q_all[:, CAND_IDX_FLANK_DETOUR],
        ], dim=-1).detach()
        cand_to_mode = self._t(CAND_TO_GOAL_MODE_TUPLE, torch.long)
        chosen_mode = cand_to_mode[chosen_idx]
        td_adv_pos = F.relu(td_target.detach() - q_chosen.detach()).clamp(0.0, 3.0).unsqueeze(1)
        mode_q = mode_q.scatter_add(1, chosen_mode.unsqueeze(1), td_adv_pos)
        mode_tgt_probs = F.softmax(mode_q / 0.45, dim=-1)
        mode_log_probs = F.log_softmax(mode_logits, dim=-1)
        mode_ent = -(torch.exp(mode_log_probs) * mode_log_probs).sum(dim=-1).mean()
        focus_tgt = torch.where(
            comp[:, MC_THR_VALID : MC_THR_VALID + 1] > 0.5,
            4.0 * comp[:, MC_THR_DX : MC_THR_DZ + 1],
            torch.where(
                comp[:, MC_OBJ_VALID : MC_OBJ_VALID + 1] > 0.5,
                4.0 * comp[:, MC_OBJ_DX : MC_OBJ_DZ + 1],
                3.5 * comp[:, MC_FRONT_DX : MC_FRONT_DZ + 1],
            ),
        )
        loss_strat = -(mode_tgt_probs * mode_log_probs).sum(dim=-1).mean() - 0.02 * mode_ent + 0.15 * F.smooth_l1_loss(focus_off_xz, focus_tgt)

        self_tgt8 = torch.clamp(torch.stack([
            2.0 * surv_tgt - 1.0,
            1.0 - 2.0 * haz_tgt,
            2.0 * prop[:, PROPRIO_IDX_RESOURCE_CARRIER] - 1.0,
            2.0 * prop[:, PROPRIO_IDX_SURFACE_POSTURE] - 1.0,
            2.0 * prop[:, PROPRIO_IDX_CAN_COMMIT] - 1.0,
            2.0 * prop[:, PROPRIO_IDX_OBJ_PROGRESS] - 1.0,
            torch.tanh(q_chosen.detach() / 8.0),
            torch.tanh(kl_vec.detach() - 0.5),
        ], dim=-1), -1.0, 1.0)
        loss_self = 0.25 * F.smooth_l1_loss(self_cur, self_tgt8) + 0.22 * F.mse_loss(self_pred, self_cur.detach())

        loss_dagger = z.new_tensor(0.0)
        has_teacher = [i for i, b in enumerate(batch) if float(b.get("beta_teacher", 0.0)) > 0.02 and "teacher_q" in b]
        if has_teacher:
            ti = self._t(has_teacher, torch.long)
            tq = self._t(np.stack([batch[i]["teacher_q"] for i in has_teacher]), torch.float32)
            ts = self._t(np.stack([batch[i]["teacher_surv"] for i in has_teacher]), torch.float32)
            w_b = float(np.mean([batch[i]["beta_teacher"] for i in has_teacher]))
            loss_dagger = w_b * (0.35 * F.smooth_l1_loss(q_all[ti], tq) + 0.30 * F.mse_loss(p_surv[ti], ts))

        # 4. Caregiver / Endogenous Grounded Language + Foreground-Slot MIL-InfoNCE Alignment Loss (isolated from D3QN!)
        tok_np = np.stack([
            b["caregiver_tokens"] if "caregiver_tokens" in b else self._synthesize_endogenous_tokens_np(b)
            for b in batch
        ])
        tok_t = self._t(tok_np, torch.long)
        proto_ids_t = self._t([int(b.get("proto_sym_id", 0)) for b in batch], torch.long)
        p_emb_t = self._t(np.stack([b.get("proto_embed", np.zeros(PROTO_EMBED_DIM, dtype=np.float32)) for b in batch]), torch.float32)
        tf_l, broca_h = self.models["broca"].teacher_forcing_loss(
            gwt_live.detach(), p_emb_t, slot_ctx64.detach(), tok_t
        )
        wern_h = self.models["broca"].encode_tokens(tok_t)
        align_l = self.models["concept_grounder"].alignment_loss(
            slot_ctx64.detach(),
            wern_h,
            token_ids=tok_t,
            slot_pstats=slot_aux["pixel_stats"].detach(),
            slot_coords=s_coords.detach(),
            proto_ids=proto_ids_t,
        )
        q_center = q_all.detach() - q_all.detach().mean(dim=-1, keepdim=True)
        bias_pred = 0.25 * self.models["broca"].cand_bias_head(broca_h.detach())
        bias_loss = F.mse_loss(bias_pred, 0.25 * torch.tanh(q_center / 2.0))
        loss_lang = 0.20 * tf_l + 0.15 * align_l + 0.05 * bias_loss

        # 5. L2-SP Continual Anchor Regularization against Catastrophic Forgetting (exempting newly added parameters on legacy ckpts)
        loss_l2sp = z.new_tensor(0.0)
        if self._anchor_weights is not None and float(self.cfg.continual_anchor_l2) > 0.0:
            reg = z.new_tensor(0.0)
            for k in ("foveal_enc", "d3qn", "rssm", "gwt", "strat_tf"):
                if k in self._anchor_weights:
                    for pn, p in self.models[k].named_parameters():
                        if pn in self._anchor_weights[k]:
                            if self._loaded_ckpt_param_keys is not None and f"{k}.{pn}" not in self._loaded_ckpt_param_keys:
                                continue
                            reg = reg + torch.sum((p - self._anchor_weights[k][pn]) ** 2)
            loss_l2sp = float(self.cfg.continual_anchor_l2) * reg

        total_loss = (
            loss_td + loss_rssm + cf_loss + loss_vis + loss_speed + loss_slot
            + loss_strat + loss_self + loss_dagger + loss_lang + loss_l2sp
        )

        # 6. Immediate Numerical Divergence / Q-Explosion Defense (100% Anchor Restore + Optimizer Purge)
        q_finite = bool(torch.isfinite(q_all).all().item())
        td_finite = bool(torch.isfinite(td_target).all().item())
        loss_finite = bool(torch.isfinite(total_loss).all().item())
        q_max_abs = float(q_all.detach().abs().max().item()) if q_finite else float("inf")
        td_err_max = float((q_chosen.detach() - td_target.detach()).abs().max().item()) if (q_finite and td_finite) else float("inf")
        if (not loss_finite) or (not q_finite) or (not td_finite) or (q_max_abs > 50.0) or (td_err_max > 65.0):
            self._emergency_rollback_to_anchor()
            return 0.0

        total_loss.backward()
        all_params = [p for k, m in self.models.items() if k not in ("d3qn_target", "depth_net") for p in m.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.5)
        grad_norm_val = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
        if not math.isfinite(grad_norm_val):
            self._emergency_rollback_to_anchor()
            return 0.0

        self.optimizer.step()
        param_maxes = torch.stack([p.detach().abs().max() for p in all_params])
        if (not bool(torch.isfinite(param_maxes).all().item())) or float(param_maxes.max().item()) > 1e3:
            self._emergency_rollback_to_anchor()
            return 0.0

        # Soft Polyak target network update
        with torch.no_grad():
            for p, pt in zip(self.models["d3qn"].parameters(), self.models["d3qn_target"].parameters()):
                pt.data.mul_(0.985).add_(p.data * 0.015)

        # Refresh sampled priorities in O(1)
        with torch.no_grad():
            td_errs = torch.abs(q_chosen - td_target).cpu().numpy()
            for k_pos, buf_idx in enumerate(idxs):
                self.replay.update_priority(int(buf_idx), float(max(0.10, td_errs[k_pos] + 0.15)))

        self.train_steps_done += 1
        return float(total_loss.item())

    def run_sleep_consolidation(
        self,
        epochs: int = 6,
        progress_cb: Optional[Callable[[int], None]] = None,
        should_abort_cb: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        4-Stage Biologically Inspired Offline Sleep Consolidation (`S-1 .. S-4`):
          - `S-1` (NREM Slow-Wave Sharp-Wave Ripple Replay): Prioritized N-step D3QN + RSSM consolidation
          - `S-2` (Cortical-Semantic Consolidation)        : Strategic Goal-Mode + Broca/Wernicke grounding
          - `S-3` (REM Counterfactual Dreaming)            : Noisy latent imagination & exposure de-confounding
          - `S-4` (Synaptic Homeostasis & Target Sync)     : Grad-guarded 2D+ norm rescaling + Polyak target sync
        """
        n_ep = max(1, int(epochs))
        losses: List[float] = []
        if progress_cb:
            progress_cb(1)
        for _ in range(max(1, n_ep // 2)):
            if should_abort_cb is not None and should_abort_cb():
                return {"sleep_cycles_done": self.sleep_cycles_done, "mean_loss": float(np.mean(losses)) if losses else 0.0, "aborted": True}
            losses.append(self._train_minibatch(batch_size=24, dreaming_noise=0.0))

        if should_abort_cb is not None and should_abort_cb():
            return {"sleep_cycles_done": self.sleep_cycles_done, "mean_loss": float(np.mean(losses)) if losses else 0.0, "aborted": True}
        if progress_cb:
            progress_cb(2)
        losses.append(self._train_minibatch(batch_size=24, dreaming_noise=0.02))

        if progress_cb:
            progress_cb(3)
        for _ in range(max(1, n_ep - max(1, n_ep // 2) - 1)):
            if should_abort_cb is not None and should_abort_cb():
                return {"sleep_cycles_done": self.sleep_cycles_done, "mean_loss": float(np.mean(losses)) if losses else 0.0, "aborted": True}
            losses.append(self._train_minibatch(batch_size=24, dreaming_noise=0.08))

        if should_abort_cb is not None and should_abort_cb():
            return {"sleep_cycles_done": self.sleep_cycles_done, "mean_loss": float(np.mean(losses)) if losses else 0.0, "aborted": True}

        if progress_cb:
            progress_cb(4)
        if len(self.replay) >= 8:
            with self.torch.no_grad():
                for k, m in self.models.items():
                    if k not in ("d3qn_target", "depth_net"):
                        for p in m.parameters():
                            if p.ndim >= 2 and p.grad is not None:
                                p.data.mul_(0.9995)
                for p, pt in zip(self.models["d3qn"].parameters(), self.models["d3qn_target"].parameters()):
                    pt.data.mul_(0.85).add_(p.data * 0.15)

        self.sleep_cycles_done += 1
        if self.device.type == "cuda":
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
        return {
            "sleep_cycles_done": self.sleep_cycles_done,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        }

    def calibrate_and_infer_depth(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stereo Retinotopic Depth inference (and online teacher calibration when `train=True`)."""
        torch = self.torch
        F = self.F
        t0 = time.perf_counter()
        st_u8 = self._t(np.asarray(req["stereo_rgb"], dtype=np.uint8)[None, ...], torch.uint8)
        do_train = bool(req.get("train", False)) and ("teacher_fovea" in req)

        if do_train:
            self.models["depth_net"].train()
            self.depth_optimizer.zero_grad()
            pred_p, uncert, logvar = self.models["depth_net"](st_u8)
            tgt = self._t(np.asarray(req["teacher_fovea"], dtype=np.float32)[None, ...], torch.float32)
            sq_err = (pred_p[:, :, :, 0] - tgt[:, :, :, 0].clamp(0.0, 1.0)) ** 2
            hetero_loss = (0.5 * torch.exp(-logvar) * sq_err + 0.5 * logvar).mean()
            norm_loss = F.smooth_l1_loss(pred_p[:, :, :, 1:4], tgt[:, :, :, 1:4])
            loss = hetero_loss + 1.2 * norm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models["depth_net"].parameters(), 1.5)
            self.depth_optimizer.step()
            self.calib_steps_done += 1
            with torch.no_grad():
                mae = float(torch.abs(pred_p[:, :, :, 0] - tgt[:, :, :, 0]).mean().item()) * float(self.cfg.fovea_max_range_m)
                n_p = F.normalize(pred_p[:, :, :, 1:3], dim=-1)
                n_t = F.normalize(tgt[:, :, :, 1:3] + 1e-5, dim=-1)
                cos_sim = float((n_p * n_t).sum(dim=-1).mean().item())
                self.calib_val_mae = 0.90 * self.calib_val_mae + 0.10 * mae
                self.calib_val_cos = 0.90 * self.calib_val_cos + 0.10 * cos_sim
        else:
            self.models["depth_net"].eval()
            with torch.no_grad():
                pred_p, uncert, _ = self.models["depth_net"](st_u8)

        p_np = pred_p[0].detach().cpu().numpy().astype(np.float32)
        u_np = uncert[0].detach().cpu().numpy().astype(np.float32)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pred_fovea_patch": p_np,
            "epistemic_uncert_16": u_np,
            "depth_infer_ms": float(infer_ms),
            "calib_steps_done": int(self.calib_steps_done),
            "calib_val_mae": float(self.calib_val_mae),
            "calib_val_cos": float(self.calib_val_cos),
        }

    def get_weights_state_dict(self, as_numpy: bool = False) -> Dict[str, Any]:
        if as_numpy:
            models_out = {
                k: {pn: p.detach().cpu().numpy().copy() for pn, p in m.state_dict().items()}
                for k, m in self.models.items()
            }
        else:
            models_out = {
                k: {pn: p.detach().cpu().clone() for pn, p in m.state_dict().items()}
                for k, m in self.models.items()
            }
        return {
            "models": models_out,
            "rvq": self.rvq.state_dict(),
            "transitions_recorded": int(self.transitions_recorded),
            "train_steps_done": int(self.train_steps_done),
            "sleep_cycles_done": int(self.sleep_cycles_done),
        }

    def load_weights_state_dict(self, state: Dict[str, Any], lr_scale: Optional[float] = None) -> None:
        m_dict = state.get("models", {})
        loaded_keys: Set[str] = set()
        for k, m in self.models.items():
            if k in m_dict:
                cur_sd = m.state_dict()
                conv = {}
                for pn, v in m_dict[k].items():
                    t_v = self.torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v.to(self.device)
                    if pn in cur_sd and tuple(cur_sd[pn].shape) == tuple(t_v.shape):
                        conv[pn] = t_v
                        loaded_keys.add(f"{k}.{pn}")
                m.load_state_dict(conv, strict=False)
        self._loaded_ckpt_param_keys = loaded_keys if loaded_keys else None
        if "d3qn" in m_dict and "d3qn_target" not in m_dict:
            self.models["d3qn_target"].load_state_dict(self.models["d3qn"].state_dict())
        if "rvq" in state:
            self.rvq.load_state_dict(state["rvq"])
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()
        if lr_scale is not None:
            new_lr = 2.8e-4 * float(np.clip(lr_scale, 0.30, 2.50))
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
        self._snapshot_anchor_weights()

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        genes = extract_brain_genes_from_cfg(self.cfg)
        meta: Dict[str, Any] = {
            "schema": "universal_brain_3d_v2",
            "task_profile": str(self.cfg.task_profile),
            "genes": genes,
            "transitions_recorded": int(self.transitions_recorded),
            "train_steps_done": int(self.train_steps_done),
            "sleep_cycles_done": int(self.sleep_cycles_done),
            "saved_at_unix": time.time(),
        }
        if extra_meta:
            meta.update(extra_meta)
        payload = self.get_weights_state_dict()
        payload["meta"] = meta
        tmp_p = path + ".tmp"
        self.torch.save(payload, tmp_p)
        os.replace(tmp_p, path)
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        self.autosave_count += 1

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        if not path or not os.path.isfile(path):
            return {}
        ckpt = self.torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "models" in ckpt:
            self.load_weights_state_dict(ckpt)
            self.transitions_recorded = int(ckpt.get("transitions_recorded", self.transitions_recorded))
            self.train_steps_done = int(ckpt.get("train_steps_done", self.train_steps_done))
            self.sleep_cycles_done = int(ckpt.get("sleep_cycles_done", self.sleep_cycles_done))
            self._snapshot_anchor_weights()
            return ckpt.get("meta", {})
        return {}

    def save_depth_checkpoint(self, path: str) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.torch.save({
            "depth_net": self.models["depth_net"].state_dict(),
            "calib_steps_done": self.calib_steps_done,
            "calib_val_mae": self.calib_val_mae,
            "calib_val_cos": self.calib_val_cos,
        }, path)

    def load_depth_checkpoint(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            return
        d = self.torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(d, dict) and "depth_net" in d:
            self.models["depth_net"].load_state_dict(d["depth_net"], strict=False)
            self.calib_steps_done = int(d.get("calib_steps_done", 0))
            self.calib_val_mae = float(d.get("calib_val_mae", 0.20))
            self.calib_val_cos = float(d.get("calib_val_cos", 0.90))

    def release_memory_and_vram(self) -> None:
        """
        Ordered cooperative teardown (~<1s):
          1. All PyTorch model gradients (`zero_grad(set_to_none=True)`)
          2. Optimizer moment/variance states (`optimizer.state.clear()`)
          3. Experience replay, N-step queues, recurrent agent rings & anchor snapshots
          4. Model parameter offload + Python GC + CUDA cache release (`torch.cuda.empty_cache()`)
        """
        torch_mod = getattr(self, "torch", None)
        try:
            # 1. Clear gradients across all PyTorch brain models first
            for m in list(self.models.values()):
                try:
                    m.zero_grad(set_to_none=True)
                except Exception:
                    pass
            # 2. Clear AdamW optimizer states (1st/2nd moment CUDA tensors)
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer.state.clear()
            if hasattr(self, "depth_optimizer") and self.depth_optimizer is not None:
                self.depth_optimizer.zero_grad(set_to_none=True)
                self.depth_optimizer.state.clear()
            # 3. Clear Replay RingBuffer, per-slot N-step/chunk4 queues, recurrent state rings & anchor weights
            self.replay.clear()
            self.replay_priorities.clear()
            self._clear_temporal_queues()
            if hasattr(self, "_agent_states") and isinstance(self._agent_states, dict):
                self._agent_states.clear()
            self._anchor_weights = None
            for m in list(self.models.values()):
                try:
                    m.cpu()
                except Exception:
                    pass
            self.models.clear()
        except Exception:
            pass
        try:
            # 4. Run GC & flush CUDA caching allocator blocks back to the GPU driver
            import gc
            gc.collect()
            if torch_mod is not None and torch_mod.cuda.is_available():
                torch_mod.cuda.synchronize()
                torch_mod.cuda.empty_cache()
        except Exception:
            pass


class BrainRPCClient:
    """Zero-PyTorch RPC client used by CPU processes (`S1`, `GWT`, `Language`) to query `GPUInferenceProcess`."""

    def __init__(self, client_id: int, req_queue: mp.Queue, resp_queue: mp.Queue, stop_event=None):
        self.client_id = int(client_id)
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.stop_event = stop_event
        self._seq = 0

    def call(self, op: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        if self.stop_event is not None and self.stop_event.is_set():
            return None
        self._seq += 1
        req_id = (self.client_id << 24) | (self._seq & 0xFFFFFF)
        self.req_queue.put((self.client_id, req_id, op, payload))
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.stop_event is not None and self.stop_event.is_set():
                return None
            try:
                rid, res = self.resp_queue.get(timeout=min(0.05, max(0.005, deadline - time.perf_counter())))
                if rid == req_id:
                    return res
            except queue.Empty:
                continue
        return None


def run_gpu_inference_3d_process(
    cfg: Brain3DConfig,
    req_queue: mp.Queue,
    resp_queues: Dict[int, mp.Queue],
    stop_event,
    ready_event,
    retinal_shm_name: str = "",
    vision_cortex_shm_name: str = "",
    abort_save_event=None,
    cog_shm_name: str = "",
) -> None:
    """
    Process 2 (`GPUInferenceProcess`):
      - ONLY process in the 7-process brain allowed to `import torch` (`Barrier 1` strict enforcement!).
      - Dynamically micro-batches `infer_s1` and `infer_gwt` requests within `cfg.batch_window_ms`.
      - Drives `VISION_CORTEX3D` stereo depth inference/calibration when `retinal_shm_name` is active.
    """
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()
    engine = WholeBrainNeuralEngineWrapper(cfg)

    ret_bus = SharedSingleRegionBus(retinal_shm_name, RETINAL3D_DTYPE, create=False) if retinal_shm_name else None
    vc_bus = SharedSingleRegionBus(vision_cortex_shm_name, VISION_CORTEX3D_DTYPE, create=False) if vision_cortex_shm_name else None
    cog_bus = SharedSingleRegionBus(cog_shm_name, COGNITIVE3D_DTYPE, create=False) if cog_shm_name else None
    last_ret_tick = -1
    ventral_state: Dict[str, Any] = {}

    ready_event.set()
    win_sec = max(0.0005, float(cfg.batch_window_ms) / 1000.0)

    try:
        while not stop_event.is_set():
            if not is_windows_pid_alive(orch_ppid):
                stop_event.set()
                break

            # Service Stereo Retinotopic Cortex if a fresh stereo frame arrived
            if ret_bus is not None and vc_bus is not None:
                ret_rec, _ = ret_bus.read_latest()
                if ret_rec is not None and int(ret_rec["retina_valid"]) == 1 and int(ret_rec["physics_tick"]) != last_ret_tick:
                    last_ret_tick = int(ret_rec["physics_tick"])
                    vmode = int(ret_rec["visual_mode_id"])
                    if vmode in (1, 2):
                        d_res = engine.calibrate_and_infer_depth({
                            "stereo_rgb": ret_rec["stereo_rgb"],
                            "teacher_fovea": ret_rec["teacher_fovea"],
                            "train": (vmode == 2 and int(ret_rec["teacher_valid"]) == 1),
                        })
                        low_ap, fl_pit = compute_depth_geometry_cues_3d(d_res["pred_fovea_patch"], cfg.fovea_max_range_m)
                        vent_p = compute_ventral_appearance_patch_16(
                            ret_rec["stereo_rgb"], ret_rec["self_yaw_pitch"], prev_state=ventral_state, episode_id=int(ret_rec["episode_id"])
                        )
                        vc_bus.publish_dict({
                            "timestamp_ns": time.perf_counter_ns(),
                            "physics_tick": last_ret_tick,
                            "episode_id": int(ret_rec["episode_id"]),
                            "action_epoch": int(ret_rec["action_epoch"]),
                            "visual_mode_id": vmode,
                            "depth_ready": 1,
                            "pad0": 0,
                            "depth_infer_ms": d_res["depth_infer_ms"],
                            "mean_uncertainty": float(np.mean(d_res["epistemic_uncert_16"])),
                            "mean_depth_norm": float(np.mean(d_res["pred_fovea_patch"][:, :, 0])),
                            "calib_val_mae": d_res["calib_val_mae"],
                            "calib_val_cos": d_res["calib_val_cos"],
                            "calib_steps_done": d_res["calib_steps_done"],
                            "pred_fovea_patch": d_res["pred_fovea_patch"],
                            "epistemic_uncert_16": d_res["epistemic_uncert_16"],
                            "low_aperture_16": low_ap,
                            "floor_pit_16": fl_pit,
                            "ventral_fovea_patch": vent_p,
                        })

            # Drain micro-batch window from RPC queue
            pending: List[Tuple[int, int, str, Dict[str, Any]]] = []
            try:
                first_item = req_queue.get(timeout=0.004)
                pending.append(first_item)
                deadline = time.perf_counter() + win_sec
                while time.perf_counter() < deadline and len(pending) < 16:
                    try:
                        pending.append(req_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            s1_group: List[Tuple[int, int, Dict[str, Any]]] = []
            gwt_group: List[Tuple[int, int, Dict[str, Any]]] = []
            other_group: List[Tuple[int, int, str, Dict[str, Any]]] = []

            for cid, rid, op, pay in pending:
                if op == "infer_s1":
                    s1_group.append((cid, rid, pay))
                elif op == "infer_gwt":
                    gwt_group.append((cid, rid, pay))
                else:
                    other_group.append((cid, rid, op, pay))

            if s1_group:
                outs = engine.infer_s1_reflex_batch([x[2] for x in s1_group])
                for (cid, rid, _), out in zip(s1_group, outs):
                    if cid in resp_queues:
                        resp_queues[cid].put((rid, out))

            if gwt_group:
                outs = engine.infer_conscious_gwt_batch([x[2] for x in gwt_group])
                for (cid, rid, _), out in zip(gwt_group, outs):
                    if cid in resp_queues:
                        resp_queues[cid].put((rid, out))

            for cid, rid, op, pay in other_group:
                res: Dict[str, Any] = {"ok": True}
                if op == "infer_broca":
                    res = engine.infer_broca_utterance(pay)
                elif op == "notify_chosen_action":
                    engine.notify_chosen_action(int(pay.get("slot_id", 0)), pay["chosen_cand_feat32"])
                    res = {"ok": True}
                elif op == "record_transition":
                    res = engine.record_transition_and_maybe_train(pay)
                    res["autosave_count"] = int(engine.autosave_count)
                elif op == "sleep":
                    def _on_sleep_stage(stg: int) -> None:
                        if cog_bus is not None:
                            cog_bus.slot.set_header_atom(4, _encode_sleep_status_atom(True, int(stg)))
                    try:
                        res = engine.run_sleep_consolidation(
                            int(pay.get("epochs", cfg.sleep_epochs_per_episode)),
                            progress_cb=_on_sleep_stage,
                            should_abort_cb=stop_event.is_set,
                        )
                    finally:
                        if cog_bus is not None:
                            cog_bus.slot.set_header_atom(4, _encode_sleep_status_atom(False, 0))
                    res["autosave_count"] = int(engine.autosave_count)
                elif op == "save_checkpoint":
                    if abort_save_event is not None and abort_save_event.is_set():
                        res = {"saved": False, "skipped_emergency_abort": True, "autosave_count": engine.autosave_count}
                    else:
                        engine.save_checkpoint(str(pay.get("path", cfg.save_weights_path)), pay.get("meta"))
                        res = {"saved": True, "autosave_count": engine.autosave_count}
                elif op == "check_barrier":
                    res = {"torch_in_gpu_only": ("torch" in sys.modules), "device": str(engine.device)}
                if cid in resp_queues:
                    resp_queues[cid].put((rid, res))
    finally:
        skip_unverified_save = (abort_save_event is not None and abort_save_event.is_set())
        if (not skip_unverified_save) and cfg.save_on_shutdown and cfg.save_weights_path:
            try:
                engine.save_checkpoint(cfg.save_weights_path)
            except Exception:
                pass
        if (not skip_unverified_save) and cfg.save_depth_weights_path and engine.calib_steps_done > 0:
            try:
                engine.save_depth_checkpoint(cfg.save_depth_weights_path)
            except Exception:
                pass
        # Steps 1..4: Gradients -> Optimizer state -> Replay buffer -> CUDA empty_cache()
        engine.release_memory_and_vram()
        # Step 5 (worker view): Detach SeqLock SharedMemory views before parent unlinks
        if cog_bus is not None:
            cog_bus.close(unlink=False)
        if ret_bus is not None:
            ret_bus.close(unlink=False)
        if vc_bus is not None:
            vc_bus.close(unlink=False)



