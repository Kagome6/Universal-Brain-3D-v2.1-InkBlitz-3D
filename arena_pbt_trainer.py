# -*- coding: utf-8 -*-
"""
arena_pbt_trainer.py — 1-GPU Parallel Multi-Arena PBT & Competitive Self-Play League Trainer (v2.3.0)
=====================================================================================================
Encapsulates simultaneous multi-environment execution (`builtin_3d` parallel worker processes OR
`unity_tcp` multi-port simultaneous arenas inside a single Unity instance) + Population-Based
Training (PBT) + Prioritized Fictitious Self-Play (PFSP) Elo League on a SINGLE CUDA GPU.

Key Capabilities:
  1. Strict `Barrier 1` Single-GPU Multiplexing:
     - `N` active learner slots + `M` frozen Hall-of-Fame opponent slots share ONE `GPUServerProcess`.
     - All `World + Spatial + Cerebellum + BasalGanglia` arena workers run in parallel Zero-PyTorch
       CPU processes, dynamically micro-batching `infer_gwt` and `record_transition` calls on 1 GPU.
  2. Simultaneous Unity Multi-Arena TCP Support (`--env unity_tcp`):
     - Opens `ports = base_port .. base_port + pop_size - 1` simultaneously (e.g., `50055..50058` for
       4 parallel arenas running at 4x-8x `Time.timeScale` inside one Unity scene).
     - In `--matchmaking self_play --env unity_tcp`, pairs `2` consecutive ports per match arena
       (`(50055, 50056)`, `(50057, 50058)`, ...) with automatic `180 deg` team-perspective canonicalization
       so any checkpoint can play either Team A (South->North) or Team B (North->South)!
  3. Competitive Self-Play League (`OpponentLeaguePool` + Elo Ratings):
     - Tracks Elo ratings (`K = 24`) across learners and frozen Hall-of-Fame champions.
     - Prevents strategy cycling / forgetting via 70% live peer + 30% Hall-of-Fame PFSP matchmaking.
  4. Generation-Tagged Queue Draining:
     - Explicitly tags all worker telemetry with `gen_idx` and drains stale packets between generations
       so generations NEVER cut short prematurely.

Quick Start:
  python arena_pbt_trainer.py --mode verify_pbt
  python arena_pbt_trainer.py --mode train --matchmaking self_play --pop-size 4 --generations 8
  python arena_pbt_trainer.py --mode train --env unity_tcp --base-port 50055 --pop-size 4 --generations 12

License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
import os
import queue
import select
import socket
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PARENT_WS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isfile(os.path.join(_PARENT_WS_DIR, "universal_brain_3d_core.py")) and _PARENT_WS_DIR not in sys.path:
    sys.path.insert(0, _PARENT_WS_DIR)

try:
    from native_accel.fast_kernels_bridge import (
        NATIVE_KERNELS_AVAILABLE as _ARENA_NATIVE_OK,
        f32_ptr as _arena_f32_ptr,
        get_native_lib as _arena_get_native_lib,
    )
    _ARENA_NATIVE_LIB = _arena_get_native_lib() if _ARENA_NATIVE_OK else None
except Exception:
    _ARENA_NATIVE_OK = False
    _ARENA_NATIVE_LIB = None

from task_profiles_3d import (
    CONT_POSE_DIM,
    GOAL_MODE_BALANCED_EXPLORE,
    MEM_COMPASS_DIM,
    NUM_CANDIDATES,
    PROPRIO_DIM,
    PROPRIO_IDX_AUDIO_LOUD,
    PROPRIO_IDX_HAZARD,
    PROPRIO_IDX_OBJ_PROGRESS,
    PROPRIO_IDX_RESOURCE_CARRIER,
    PROPRIO_IDX_SURFACE_POSTURE,
    PROPRIO_IDX_TRANSIT_RESPAWN,
    SemanticTag,
    VERB_APPROACH_HOVER,
    VERB_COMMIT_INTERACT,
    VERB_EVADE_HAZARD,
    VERB_GRASP_PICK,
    VERB_IDLE_OBSERVE,
    VERB_IMPULSE_PUSH,
    VERB_RELEASE_PLACE,
    compute_beta_teacher,
    compute_task_scaffold_targets,
    get_resource_tank_level,
    get_task_profile,
    is_carrying_primary_objective,
    is_in_stealth_or_cover_refuge,
)
from universal_brain_3d_core import (
    ActionStagnationHomeostasis3D,
    BasalGanglia3DTimingEngineV2,
    Brain3DConfig,
    Cerebellar3DKinematicPredictorV2,
    ENTITY_DIM,
    EntorhinalHippocampal3DEngineV2,
    FOVEA_C,
    FOVEA_H,
    FOVEA_W,
    HIPPO_MEM_ADAPTIVE,
    HIPPO_MEM_MODES,
    HOLD_GO_CLEAR,
    MAX_ENTITIES,
    StandaloneTactical3DArena,
    _ignore_sigint_in_worker,
    _recv_ub3d_packet,
    _sendall_nonblocking_safe,
    apply_checkpoint_genes_to_cfg,
    compute_gene_modulated_mip_and_fr,
    compute_universal_memory_compass,
    create_runtime_strategy_controller,
    derive_ventral_fallback_from_dorsal_16,
    extract_5module_activity_vec,
    extract_brain_genes_from_cfg,
    is_windows_pid_alive,
    pack_ubmc_motor_command,
    pack_ubnt_neural_telemetry,
    pack_ubwm_world_model,
    project_pomdp_proprio,
    propose_universal_3d_candidates,
    select_policy_candidate,
    start_parent_and_stdin_watchdog,
    synthesize_joint_attention_phrase,
    unpack_ub3d_sensory_frame,
)

HOLD_INTER_GEN_WAIT = 250


# ==============================================================================
# 1. PBT Evolvable Brain Archetypes & Elo Rating League Manager
# ==============================================================================
PBT_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "archetype": "ObjectiveStriker",
        "lr_scale": 1.15,
        "curiosity_fog_gain": 0.90,
        "stn_stealth_gain": 0.85,
        "broca_prior_gain": 1.00,
        "policy_temp": 0.30,
        "survival_veto_threshold": 0.38,
    },
    {
        "archetype": "TurfController",
        "lr_scale": 1.00,
        "curiosity_fog_gain": 1.45,
        "stn_stealth_gain": 0.95,
        "broca_prior_gain": 1.00,
        "policy_temp": 0.36,
        "survival_veto_threshold": 0.42,
    },
    {
        "archetype": "StealthInterceptor",
        "lr_scale": 0.95,
        "curiosity_fog_gain": 1.05,
        "stn_stealth_gain": 1.35,
        "broca_prior_gain": 1.00,
        "policy_temp": 0.28,
        "survival_veto_threshold": 0.46,
    },
    {
        "archetype": "BalancedTactician",
        "lr_scale": 1.05,
        "curiosity_fog_gain": 1.15,
        "stn_stealth_gain": 1.05,
        "broca_prior_gain": 1.00,
        "policy_temp": 0.33,
        "survival_veto_threshold": 0.41,
    },
]


def compute_elo_update(elo_a: float, elo_b: float, score_a: float, k_factor: float = 24.0) -> Tuple[float, float]:
    """
    Standard Elo update where `score_a` is `1.0` (A win), `0.5` (draw), or `0.0` (B win).
    """
    exp_a = 1.0 / (1.0 + math.pow(10.0, (float(elo_b) - float(elo_a)) / 400.0))
    exp_b = 1.0 - exp_a
    new_a = float(elo_a) + float(k_factor) * (float(score_a) - exp_a)
    new_b = float(elo_b) + float(k_factor) * ((1.0 - float(score_a)) - exp_b)
    return new_a, new_b


@dataclass
class LeagueSnapshotEntry:
    snapshot_id: str
    generation: int
    archetype: str
    elo: float
    genes: Dict[str, Any]
    weights_state: Dict[str, Any]


class OpponentLeaguePool:
    """
    Prioritized Fictitious Self-Play (PFSP) Hall-of-Fame Pool:
    Retains diverse historical checkpoints so competitive learners cannot collapse into cyclic
    rock-paper-scissors strategies or forget basic objective defense/pushing.
    """

    def __init__(self, max_snapshots: int = 8, seed: int = 2026):
        self.max_snapshots = max(2, int(max_snapshots))
        self.rng = np.random.default_rng(seed + 501)
        self.snapshots: List[LeagueSnapshotEntry] = []

    def add_champion(self, snap_id: str, gen: int, archetype: str, elo: float, genes: Dict[str, Any], weights_state: Dict[str, Any]) -> None:
        entry = LeagueSnapshotEntry(
            snapshot_id=str(snap_id),
            generation=int(gen),
            archetype=str(archetype),
            elo=float(elo),
            genes=dict(genes),
            weights_state=weights_state,
        )
        if len(self.snapshots) >= self.max_snapshots:
            # Keep the highest-Elo champion AND diverse generations; evict lowest-Elo older snapshot
            worst_idx = int(np.argmin([s.elo for s in self.snapshots[:-1]]))
            self.snapshots.pop(worst_idx)
        self.snapshots.append(entry)

    def sample_pfsp_opponent(self, learner_elo: float) -> Optional[LeagueSnapshotEntry]:
        if not self.snapshots:
            return None
        # Prioritize opponents with similar Elo (`win_prob ~ 0.5` maximizes learning signal!)
        weights = []
        for s in self.snapshots:
            p_win = 1.0 / (1.0 + math.pow(10.0, (s.elo - float(learner_elo)) / 400.0))
            pfsp_w = max(0.05, 4.0 * p_win * (1.0 - p_win))
            weights.append(pfsp_w)
        w_arr = np.asarray(weights, dtype=np.float64)
        w_arr /= np.sum(w_arr)
        idx = int(self.rng.choice(len(self.snapshots), p=w_arr))
        return self.snapshots[idx]


# ==============================================================================
# 2. Canonical Team-Perspective Transform (`180 deg` Symmetry for Self-Play)
# ==============================================================================
def canonicalize_team_observation(obs: Dict[str, Any], team_sign: int, wmin: np.ndarray, wmax: np.ndarray) -> Dict[str, Any]:
    """
    Transform a Team-B (`team_sign == -1`, North-base defending `+Z` and attacking `-Z`) sensory frame
    into the canonical South-to-North (`-Z -> +Z`) coordinate frame used by all brains:
      - `x' = mid_x - (x - mid_x)`, `z' = mid_z - (z - mid_z)`
      - `vx' = -vx`, `vz' = -vz`
      - `yaw' = wrap_pi(yaw + pi)`
      - Fovea `nx' = -nx`; Proprio `[0, 2]` inverted around `0.5`, `[3, 5]` negated
      - Attack/Defend pedestal tags swapped ONLY if raw environment did not already assign them per-team.
    For `team_sign == +1`, returns `obs` unchanged.
    """
    if int(team_sign) >= 0:
        return obs

    mid = 0.5 * (wmin + wmax)
    out = dict(obs)

    sp = np.asarray(obs["self_pos"], dtype=np.float32).copy()
    sp[0] = 2.0 * float(mid[0]) - sp[0]
    sp[2] = 2.0 * float(mid[2]) - sp[2]
    out["self_pos"] = sp

    sv = np.asarray(obs["self_vel"], dtype=np.float32).copy()
    sv[0] = -sv[0]
    sv[2] = -sv[2]
    out["self_vel"] = sv

    yp = np.asarray(obs["self_yaw_pitch"], dtype=np.float32).copy()
    yp[0] = math.atan2(math.sin(float(yp[0]) + math.pi), math.cos(float(yp[0]) + math.pi))
    out["self_yaw_pitch"] = yp

    prop = np.asarray(obs["proprio"], dtype=np.float32).copy()
    prop[0] = 1.0 - prop[0]
    prop[2] = 1.0 - prop[2]
    prop[3] = -prop[3]
    prop[5] = -prop[5]
    out["proprio"] = prop

    ents = np.asarray(obs["visible_entities"], dtype=np.float32).copy()
    vcnt = min(MAX_ENTITIES, int(obs["visible_count"]))
    for i in range(vcnt):
        ents[i, 1] = 2.0 * float(mid[0]) - ents[i, 1]
        ents[i, 3] = 2.0 * float(mid[2]) - ents[i, 3]
        ents[i, 4] = -ents[i, 4]
        ents[i, 6] = -ents[i, 6]
    out["visible_entities"] = ents

    fov = np.asarray(obs["fovea_patch"], dtype=np.float32).copy()
    fov[:, :, 1] = -fov[:, :, 1]
    out["fovea_patch"] = fov
    return out


def uncanonicalize_team_motor(
    target_pose: np.ndarray,
    gaze_ypd: np.ndarray,
    team_sign: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map canonical South-to-North motor + aim commands back to world coordinates for Team B (`team_sign == -1`)."""
    if int(team_sign) >= 0:
        return target_pose, gaze_ypd
    tp = np.asarray(target_pose, dtype=np.float32).copy()
    tp[0] = -tp[0]
    tp[2] = -tp[2]
    tp[3] = math.atan2(math.sin(float(tp[3]) + math.pi), math.cos(float(tp[3]) + math.pi))

    gz = np.asarray(gaze_ypd, dtype=np.float32).copy()
    gz[0] = math.atan2(math.sin(float(gz[0]) + math.pi), math.cos(float(gz[0]) + math.pi))
    return tp, gz


# ==============================================================================
# 3. Head-to-Head 2-Brain Competitive Tactical 3D Arena (Symmetric Self-Play)
# ==============================================================================
class HeadToHeadTactical3DArena:
    """
    Pure-NumPy Symmetric 1v1 Competitive Turf & BlitzCore Objective Arena:
      - Agent A (`+1`, South spawn `-Z`, attacks North pedestal `+Z`)
      - Agent B (`-1`, North spawn `+Z`, attacks South pedestal `-Z`)
      - Shared `48x48` dynamic floor surface grid (`+1.0` = A turf, `-1.0` = B turf)
      - Shared central Primary Objective (`BlitzCore`), cover blocks, and ballistic projectiles!
      - Both agents receive strictly 1st-person canonicalized observations (zero omniscient wallhacks).
    """

    def __init__(self, cfg: Brain3DConfig, seed: int = 2026):
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self.rng = np.random.default_rng(seed)
        self.wmin = np.array(cfg.world_min, dtype=np.float32)
        self.wmax = np.array(cfg.world_max, dtype=np.float32)
        self.span = np.maximum(1.0, self.wmax - self.wmin)
        self.fovea_cos = float(math.cos(math.radians(cfg.fovea_half_angle_deg)))
        self.grid_n = 48
        self.surface_grid = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)

        self.episode_id = 0
        self.physics_tick = 0
        self.ticks_in_ep = 0
        self.dropped_idle_ticks = 0
        self.stage_id = int(getattr(cfg, "default_stage_id", 1))
        self.cover_blocks: List[Dict[str, Any]] = []
        self.north_pedestal = np.array([0.0, 1.2, float(self.wmax[2] - 3.0)], dtype=np.float32)
        self.south_pedestal = np.array([0.0, 1.2, float(self.wmin[2] + 3.0)], dtype=np.float32)
        self.objective_pos = np.array([0.0, 1.1, 0.0], dtype=np.float32)
        self.carrier_team = 0  # 0 neutral, +1 Agent A, -1 Agent B
        self.objective_progress_a = 0.50
        self.projectiles: List[Dict[str, Any]] = []
        self.agents: Dict[int, Dict[str, Any]] = {}
        self._apply_stage_geometry(self.stage_id)
        self.reset_episode()

    def _apply_stage_geometry(self, stage_id: int) -> None:
        self.stage_id = 2 if int(stage_id) == 2 else 1
        if self.stage_id == 2:
            # Stage 2: Occluded Twin-Flank — 180-deg involution-symmetric central S-baffle chicane + twin flank gates
            self.south_pedestal = np.array([6.8, 1.28, float(self.wmin[2] + 3.2)], dtype=np.float32)
            self.north_pedestal = np.array([-6.8, 1.28, float(self.wmax[2] - 3.2)], dtype=np.float32)
            self.cover_blocks = [
                {"id": 101, "pos": np.array([-1.2, 1.3, -3.85], dtype=np.float32), "radius": 1.35},
                {"id": 102, "pos": np.array([1.2, 1.3, 3.85], dtype=np.float32), "radius": 1.35},
                {"id": 103, "pos": np.array([1.1, 1.3, -3.55], dtype=np.float32), "radius": 1.40},
                {"id": 104, "pos": np.array([-1.1, 1.3, 3.55], dtype=np.float32), "radius": 1.40},
                {"id": 105, "pos": np.array([3.3, 1.3, -3.25], dtype=np.float32), "radius": 1.30},
                {"id": 106, "pos": np.array([-3.3, 1.3, 3.25], dtype=np.float32), "radius": 1.30},
                {"id": 107, "pos": np.array([-6.2, 1.3, -1.8], dtype=np.float32), "radius": 1.20},
                {"id": 108, "pos": np.array([6.2, 1.3, 1.8], dtype=np.float32), "radius": 1.20},
                {"id": 109, "pos": np.array([3.0, 1.3, -8.8], dtype=np.float32), "radius": 1.20},
                {"id": 110, "pos": np.array([-3.0, 1.3, 8.8], dtype=np.float32), "radius": 1.20},
            ]
        else:
            self.north_pedestal = np.array([0.0, 1.2, float(self.wmax[2] - 3.0)], dtype=np.float32)
            self.south_pedestal = np.array([0.0, 1.2, float(self.wmin[2] + 3.0)], dtype=np.float32)
            self.cover_blocks = [
                {"id": 101, "pos": np.array([-4.2, 1.3, -4.5], dtype=np.float32), "radius": 1.25},
                {"id": 102, "pos": np.array([4.2, 1.3, -4.5], dtype=np.float32), "radius": 1.25},
                {"id": 103, "pos": np.array([-4.2, 1.3, 4.5], dtype=np.float32), "radius": 1.25},
                {"id": 104, "pos": np.array([4.2, 1.3, 4.5], dtype=np.float32), "radius": 1.25},
            ]

    def _world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        ix = int(np.clip(math.floor((float(x) - float(self.wmin[0])) / float(self.span[0]) * self.grid_n), 0, self.grid_n - 1))
        iz = int(np.clip(math.floor((float(z) - float(self.wmin[2])) / float(self.span[2]) * self.grid_n), 0, self.grid_n - 1))
        return iz, ix

    def _paint_disc(self, x: float, z: float, radius_m: float, team_sign: int) -> int:
        flipped = 0
        cz, cx = self._world_to_cell(x, z)
        rc = max(1, int(math.ceil(radius_m / max(0.35, float(self.span[0]) / self.grid_n))))
        for dz in range(-rc, rc + 1):
            iz = cz + dz
            if not (0 <= iz < self.grid_n):
                continue
            for dx in range(-rc, rc + 1):
                ix = cx + dx
                if not (0 <= ix < self.grid_n) or (dz * dz + dx * dx > rc * rc + 1):
                    continue
                prev = float(self.surface_grid[iz, ix])
                if prev * float(team_sign) < 0.30:
                    flipped += 1
                self.surface_grid[iz, ix] = float(np.clip(0.30 * prev + 0.85 * float(team_sign), -1.0, 1.0))
        return flipped

    def _is_occluded(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        seg = p1[[0, 2]] - p0[[0, 2]]
        l2 = float(np.dot(seg, seg))
        if l2 < 1e-5:
            return False
        for blk in self.cover_blocks:
            t = float(np.clip(np.dot(blk["pos"][[0, 2]] - p0[[0, 2]], seg) / l2, 0.0, 1.0))
            if float(np.linalg.norm(p0[[0, 2]] + t * seg - blk["pos"][[0, 2]])) < float(blk["radius"]) * 0.90:
                return True
        return False

    def reset_episode(self) -> None:
        cfg_st = int(getattr(self.cfg, "default_stage_id", 1))
        if cfg_st != self.stage_id:
            self._apply_stage_geometry(cfg_st)
        self.episode_id += 1
        self.ticks_in_ep = 0
        self.dropped_idle_ticks = 0
        self.surface_grid.fill(0.0)
        self.projectiles.clear()
        self.objective_pos = np.array([0.0, 1.1, 0.0], dtype=np.float32)
        self.carrier_team = 0
        self.objective_progress_a = 0.50

        # Exact 180-deg involution-symmetric spawn offset: (+x_a, z_south) <-> (-x_a, -z_south)
        x0 = float(self.rng.uniform(-1.5, 1.5)) if self.stage_id == 2 else float(self.rng.uniform(-2.0, 2.0))
        base_x_a = (-7.0 + x0) if self.stage_id == 2 else x0
        z_margin = 2.8 if self.stage_id == 2 else 3.5
        for t_sign in (+1, -1):
            z_sp = float(self.wmin[2] + z_margin) if t_sign > 0 else float(self.wmax[2] - z_margin)
            x_sp = base_x_a if t_sign > 0 else -base_x_a
            yaw0 = 0.0 if t_sign > 0 else math.pi
            self.agents[t_sign] = {
                "pos": np.array([x_sp, 1.5, z_sp], dtype=np.float32),
                "vel": np.zeros(3, dtype=np.float32),
                "gaze_ypd": np.array([yaw0, -0.15, 6.0], dtype=np.float32),
                "aim_yp": np.array([yaw0, -0.15], dtype=np.float32),
                "hp": 1.0,
                "regen_delay": 0.0,
                "tank": 1.0,
                "submerge_stam": 4.0,
                "submerge_lock": 0.0,
                "submerged": False,
                "in_refuge": False,
                "just_fired": False,
                "cd": 0,
                "invuln": 0,
                "respawn_flag": False,
                "epoch": 1,
                "cum_score": 0.0,
            }
            self._paint_disc(float(self.agents[t_sign]["pos"][0]), z_sp, 2.2, t_sign)

    def step_both(
        self,
        act_a: Tuple[np.ndarray, int, bool, np.ndarray],
        act_b_canon: Tuple[np.ndarray, int, bool, np.ndarray],
        dt: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], bool, int]:
        """
        Advance both agents 1 tick (`act_b_canon` is automatically uncanonicalized from South->North to North->South!).
        Returns `(obs_a_canon, obs_b_canon, episode_done, winner_team)`.
        """
        self.physics_tick += 1
        self.ticks_in_ep += 1
        tp_b_world, gz_b_world = uncanonicalize_team_motor(act_b_canon[0], act_b_canon[3], -1)
        cmds = {
            +1: act_a,
            -1: (tp_b_world, act_b_canon[1], act_b_canon[2], gz_b_world),
        }
        rews = {+1: 0.0, -1: 0.0}
        ep_done = False
        winner = 0

        mid = 0.5 * (self.wmin + self.wmax)
        half = 0.5 * self.span

        for t_sign in (+1, -1):
            ag = self.agents[t_sign]
            ag["respawn_flag"] = False
            ag["just_fired"] = False
            if ag["cd"] > 0:
                ag["cd"] -= 1
            if ag["invuln"] > 0:
                ag["invuln"] -= 1

            tp, verb, commit_ok, gz = cmds[t_sign]
            if np.linalg.norm(gz) > 1e-5:
                ag["gaze_ypd"] = np.array([float(gz[0]), float(np.clip(gz[1], -1.0, 1.0)), 6.0], dtype=np.float32)
            ag["aim_yp"] = np.array([float(tp[3]), float(np.clip(tp[4], -0.9, 0.9))], dtype=np.float32)

            iz, ix = self._world_to_cell(float(ag["pos"][0]), float(ag["pos"][2]))
            my_surf = float(self.surface_grid[iz, ix]) * float(t_sign)
            near_cov = any(float(np.linalg.norm(ag["pos"][[0, 2]] - b["pos"][[0, 2]])) <= float(b["radius"]) + 1.3 for b in self.cover_blocks)
            if ag["submerge_lock"] > 0.0:
                ag["submerge_lock"] = max(0.0, float(ag["submerge_lock"]) - dt)
            if ag["regen_delay"] > 0.0:
                ag["regen_delay"] = max(0.0, float(ag["regen_delay"]) - dt)
            can_sub = (ag["submerge_lock"] <= 0.0) and (ag["submerge_stam"] > 1e-5)
            want_sub = can_sub and (my_surf > 0.28) and (verb in (VERB_IDLE_OBSERVE, VERB_EVADE_HAZARD))
            if want_sub:
                ag["submerge_stam"] = max(0.0, float(ag["submerge_stam"]) - dt)
                if ag["submerge_stam"] <= 1e-4:
                    ag["submerge_stam"] = 0.0
                    want_sub = False
                    ag["submerge_lock"] = 1.2
            else:
                ag["submerge_stam"] = min(4.0, float(ag["submerge_stam"]) + 2.0 * dt)
            ag["submerged"] = bool(want_sub)
            ag["in_refuge"] = bool((want_sub or near_cov) and (verb == VERB_IDLE_OBSERVE) and (self.carrier_team != t_sign))
            ag["tank"] = float(min(1.0, ag["tank"] + (0.48 if want_sub else 0.09) * dt))
            if ag["regen_delay"] <= 0.0:
                ag["hp"] = float(min(1.0, ag["hp"] + (0.28 if want_sub else 0.05) * dt))

            des = np.clip(mid + np.clip(tp[0:3], -1.0, 1.0) * half, self.wmin + 0.8, self.wmax - 0.8)
            des[1] = 1.5
            spd = 6.36 if want_sub else (1.55 if my_surf < -0.28 else 4.30)
            if self.carrier_team == t_sign:
                spd *= 0.72
            if ag["in_refuge"]:
                spd *= 0.20
            spd *= float(np.clip(tp[5], 0.25, 1.0)) if abs(float(tp[5])) > 1e-3 else 0.85

            err = des - ag["pos"]
            err[1] = 0.0
            de = float(np.linalg.norm(err))
            ag["vel"] = (err / de) * spd if de > 0.08 else ag["vel"] * 0.6
            nxt = np.clip(ag["pos"] + ag["vel"] * dt, self.wmin + 0.7, self.wmax - 0.7)
            nxt[1] = 1.5
            for blk in self.cover_blocks:
                dxz = nxt[[0, 2]] - blk["pos"][[0, 2]]
                db = float(np.linalg.norm(dxz))
                mr = float(blk["radius"]) + 0.45
                if db < mr:
                    pdir = dxz / max(1e-4, db)
                    nxt[0], nxt[2] = float(blk["pos"][0] + pdir[0] * mr), float(blk["pos"][2] + pdir[1] * mr)
            ag["pos"] = nxt

            if commit_ok and ag["cd"] == 0:
                if verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH) and ag["tank"] >= self.profile.min_resource_to_fire and self.carrier_team != t_sign:
                    ag["tank"] = max(0.0, ag["tank"] - (0.14 if verb == VERB_RELEASE_PLACE else 0.24))
                    ag["just_fired"] = True
                    sy, cy = math.sin(float(ag["aim_yp"][0])), math.cos(float(ag["aim_yp"][0]))
                    cp, sp = max(0.25, math.cos(float(ag["aim_yp"][1]))), math.sin(float(ag["aim_yp"][1]))
                    dv = np.array([sy * cp, sp, cy * cp], dtype=np.float32)
                    dv /= max(1e-5, float(np.linalg.norm(dv)))
                    self.projectiles.append({
                        "pos": ag["pos"].copy() + dv * 0.8,
                        "vel": dv * float(self.profile.projectile_speed_mps),
                        "team": t_sign,
                        "ttl": 32,
                        "splash": 1.65 if verb == VERB_RELEASE_PLACE else 2.25,
                    })
                    fl = self._paint_disc(float(ag["pos"][0] + dv[0] * 2.0), float(ag["pos"][2] + dv[2] * 2.0), 1.3, t_sign)
                    rews[t_sign] += 0.04 * float(min(5, fl))
                    ag["cd"] = 8
                    ag["epoch"] += 1

        # Order-independent simultaneous neutral objective pickup (preserves exact 180-deg symmetry!)
        if self.carrier_team == 0:
            da_obj = float(np.linalg.norm(self.agents[+1]["pos"][[0, 2]] - self.objective_pos[[0, 2]]))
            db_obj = float(np.linalg.norm(self.agents[-1]["pos"][[0, 2]] - self.objective_pos[[0, 2]]))
            if da_obj <= 1.45 and (db_obj > 1.45 or da_obj < db_obj - 1e-6):
                self.carrier_team = +1
                rews[+1] += 3.0
                rews[-1] -= 0.80
                self.agents[+1]["epoch"] += 1
            elif db_obj <= 1.45 and (da_obj > 1.45 or db_obj < da_obj - 1e-6):
                self.carrier_team = -1
                rews[-1] += 3.0
                rews[+1] -= 0.80
                self.agents[-1]["epoch"] += 1

        # Update carried objective & knockout check (Stage 2 also decays unescorted dropped cores back toward neutral)
        if self.carrier_team != 0:
            self.dropped_idle_ticks = 0
            c_ag = self.agents[self.carrier_team]
            self.objective_pos = c_ag["pos"].copy()
            prev_p = self.objective_progress_a
            ped_vec = self.north_pedestal[[0, 2]] - self.south_pedestal[[0, 2]]
            ped_l2 = max(1.0, float(np.dot(ped_vec, ped_vec)))
            zf = float(np.dot(self.objective_pos[[0, 2]] - self.south_pedestal[[0, 2]], ped_vec)) / ped_l2
            self.objective_progress_a = float(np.clip(zf, 0.0, 1.0))
            delta_a = self.objective_progress_a - prev_p
            rews[+1] += delta_a * 8.0
            rews[-1] -= delta_a * 8.0

            goal_pos = self.north_pedestal if self.carrier_team > 0 else self.south_pedestal
            if float(np.linalg.norm(c_ag["pos"][[0, 2]] - goal_pos[[0, 2]])) <= 2.1:
                rews[self.carrier_team] += 15.0
                rews[-self.carrier_team] -= 10.0
                ep_done = True
                winner = self.carrier_team
        elif self.stage_id == 2 and float(np.linalg.norm(self.objective_pos[[0, 2]])) > 0.15:
            self.dropped_idle_ticks += 1
            if self.dropped_idle_ticks > 120:
                self.objective_pos[0] *= 0.965
                self.objective_pos[2] *= 0.965
                if float(np.linalg.norm(self.objective_pos[[0, 2]])) < 0.20:
                    self.objective_pos[:] = [0.0, 1.1, 0.0]
                    self.dropped_idle_ticks = 0
                ped_vec = self.north_pedestal[[0, 2]] - self.south_pedestal[[0, 2]]
                ped_l2 = max(1.0, float(np.dot(ped_vec, ped_vec)))
                self.objective_progress_a = float(
                    np.clip(float(np.dot(self.objective_pos[[0, 2]] - self.south_pedestal[[0, 2]], ped_vec)) / ped_l2, 0.0, 1.0)
                )

        # Advance ballistic projectiles with exact parabolic gravity (`profile.projectile_gravity_mps2`, matching Unity 6!)
        g_proj = float(getattr(self.profile, "projectile_gravity_mps2", 0.0))
        alive_p = []
        pending_splats = []
        for pr in self.projectiles:
            pr["pos"] = pr["pos"] + pr["vel"] * dt + np.array([0.0, -0.5 * g_proj * dt * dt, 0.0], dtype=np.float32)
            pr["vel"][1] = float(pr["vel"][1]) - g_proj * dt
            pr["ttl"] -= 1
            hit = False
            if pr["pos"][1] <= float(self.wmin[1]) + 0.15 or pr["ttl"] <= 0:
                self._paint_disc(float(pr["pos"][0]), float(pr["pos"][2]), float(pr["splash"]), int(pr["team"]))
                hit = True
            else:
                for blk in self.cover_blocks:
                    if float(np.linalg.norm(pr["pos"][[0, 2]] - blk["pos"][[0, 2]])) <= float(blk["radius"]):
                        hit = True
                        break
            if not hit:
                opp_sign = -int(pr["team"])
                opp_ag = self.agents[opp_sign]
                hit_rad = 0.82 if bool(opp_ag.get("submerged", False)) else 1.10
                if opp_ag["invuln"] == 0 and float(np.linalg.norm(pr["pos"] - opp_ag["pos"])) <= hit_rad:
                    opp_ag["hp"] -= 0.55
                    opp_ag["regen_delay"] = 1.0
                    opp_ag["submerged"] = False
                    rews[int(pr["team"])] += 1.25
                    rews[opp_sign] -= 0.85
                    self._paint_disc(float(opp_ag["pos"][0]), float(opp_ag["pos"][2]), 1.6, int(pr["team"]))
                    hit = True
                    if opp_ag["hp"] <= 0.0 and opp_sign not in pending_splats:
                        pending_splats.append(opp_sign)
            if not hit:
                alive_p.append(pr)
        for opp_sign in pending_splats:
            opp_ag = self.agents[opp_sign]
            rews[-opp_sign] += 3.5
            rews[opp_sign] -= 3.0
            if self.carrier_team == opp_sign:
                self.carrier_team = 0
                self.dropped_idle_ticks = 0
                self.objective_pos = np.array([float(opp_ag["pos"][0]), 1.1, float(opp_ag["pos"][2])], dtype=np.float32)
            z_margin_r = 2.8 if self.stage_id == 2 else 3.2
            z_sp = float(self.wmin[2] + z_margin_r) if opp_sign > 0 else float(self.wmax[2] - z_margin_r)
            rx_raw = float(self.rng.uniform(-1.5, 1.5)) if self.stage_id == 2 else float(self.rng.uniform(-2.0, 2.0))
            rx_a = (-7.0 + rx_raw) if self.stage_id == 2 else rx_raw
            opp_ag["pos"] = np.array([rx_a if opp_sign > 0 else -rx_a, 1.5, z_sp], dtype=np.float32)
            opp_ag["vel"].fill(0.0)
            opp_ag["hp"] = 1.0
            opp_ag["regen_delay"] = 0.0
            opp_ag["tank"] = 1.0
            opp_ag["invuln"] = 18
            opp_ag["respawn_flag"] = True
            opp_ag["epoch"] += 1
            self._paint_disc(float(opp_ag["pos"][0]), z_sp, 2.0, opp_sign)
        self.projectiles = alive_p[-24:]

        max_ep_ticks = 780 if self.stage_id == 2 else 540
        if self.ticks_in_ep >= max_ep_ticks and not ep_done:
            ep_done = True
            if abs(self.objective_progress_a - 0.50) > 0.04:
                winner = +1 if self.objective_progress_a > 0.50 else -1
            else:
                turf_diff = float(np.sum(self.surface_grid > 0.25)) - float(np.sum(self.surface_grid < -0.25))
                winner = +1 if turf_diff > 2 else (-1 if turf_diff < -2 else 0)

        for t_sign in (+1, -1):
            self.agents[t_sign]["cum_score"] += rews[t_sign]

        obs_a = self._build_agent_obs(+1, rews[+1], ep_done)
        obs_b_raw = self._build_agent_obs(-1, rews[-1], ep_done)
        obs_b_canon = canonicalize_team_observation(obs_b_raw, -1, self.wmin, self.wmax)
        return obs_a, obs_b_canon, ep_done, winner

    def _build_agent_obs(self, t_sign: int, step_rew: float, ep_done: bool) -> Dict[str, Any]:
        me = self.agents[t_sign]
        en = self.agents[-t_sign]
        cy, sy = math.cos(float(me["gaze_ypd"][0])), math.sin(float(me["gaze_ypd"][0]))
        cp, sp = math.cos(float(me["gaze_ypd"][1])), math.sin(float(me["gaze_ypd"][1]))
        fwd = np.array([sy * cp, sp, cy * cp], dtype=np.float32)
        right = np.array([cy, 0.0, -sy], dtype=np.float32)
        up = np.cross(fwd, right).astype(np.float32)

        vis_list: List[np.ndarray] = []

        def _add(eid: int, p3: np.ndarray, v3: np.ndarray, rad: float, tag: float, occ_chk: bool = True) -> None:
            rel = p3 - me["pos"]
            d = float(np.linalg.norm(rel))
            if d < 1e-4 or d > self.cfg.fovea_max_range_m:
                return
            ca = float(np.dot(rel / d, fwd))
            if ca < self.fovea_cos and d > 2.3:
                return
            if occ_chk and self._is_occluded(me["pos"], p3):
                return
            vis_list.append(np.array([float(eid), p3[0], p3[1], p3[2], v3[0], v3[1], v3[2], rad, tag, max(0.2, ca)], dtype=np.float32))

        atk_ped = self.north_pedestal if t_sign > 0 else self.south_pedestal
        def_ped = self.south_pedestal if t_sign > 0 else self.north_pedestal
        _add(11, atk_ped, np.zeros(3, dtype=np.float32), 1.3, SemanticTag.ATTACK_GOAL_PEDESTAL, False)
        _add(12, def_ped, np.zeros(3, dtype=np.float32), 1.3, SemanticTag.DEFEND_BASE_PEDESTAL, False)
        if self.carrier_team != t_sign:
            ov = en["vel"] if self.carrier_team == -t_sign else np.zeros(3, dtype=np.float32)
            _add(21, self.objective_pos, ov, 0.85, SemanticTag.PRIMARY_OBJECTIVE, True)
        for b in self.cover_blocks:
            _add(int(b["id"]), b["pos"], np.zeros(3, dtype=np.float32), float(b["radius"]), SemanticTag.COVER_OR_FRIENDLY_TURF, False)

        d_en = float(np.linalg.norm(en["pos"] - me["pos"]))
        if (not en["in_refuge"]) or d_en < 3.6:
            etag = SemanticTag.ENEMY_CARRIER_OR_SHOT if self.carrier_team == -t_sign else SemanticTag.OPPONENT_AGENT
            _add(201, en["pos"], en["vel"], 0.75, etag, True)
        for pidx, pr in enumerate(self.projectiles[:6]):
            if int(pr["team"]) == -t_sign:
                _add(500 + pidx, pr["pos"], pr["vel"], 0.35, SemanticTag.ENEMY_CARRIER_OR_SHOT, True)

        vis_arr = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
        vc = min(MAX_ENTITIES, len(vis_list))
        if vc > 0:
            vis_arr[:vc] = np.stack(vis_list[:vc], axis=0)

        # Render 16x16x4 fovea with team-relative surface encoding (`>0.25` friendly -> `0.78`, `<-0.25` enemy -> `0.25`)
        th = math.tan(math.radians(self.cfg.fovea_half_angle_deg))
        max_r = float(self.cfg.fovea_max_range_m)
        if _ARENA_NATIVE_LIB is not None and self.surface_grid.flags["C_CONTIGUOUS"]:
            fovea = np.empty((FOVEA_H, FOVEA_W, 4), dtype=np.float32)
            n_vis_r = min(12, vc)
            ent_ptr = _arena_f32_ptr(vis_arr) if n_vis_r > 0 else None
            _ARENA_NATIVE_LIB.ub3d_render_fovea_16x16(
                float(me["pos"][0]), float(me["pos"][1]), float(me["pos"][2]),
                float(me["gaze_ypd"][0]), float(me["gaze_ypd"][1]),
                float(th), float(max_r),
                float(self.wmin[0]), float(self.wmin[1]), float(self.wmin[2]),
                float(self.span[0]), float(self.span[2]),
                _arena_f32_ptr(self.surface_grid), int(self.grid_n), float(t_sign),
                ent_ptr, int(n_vis_r),
                _arena_f32_ptr(fovea),
            )
        else:
            u = np.linspace(-th, th, FOVEA_W, dtype=np.float32)
            v = np.linspace(th, -th, FOVEA_H, dtype=np.float32)
            uu, vv = np.meshgrid(u, v)
            dirs = fwd[None, None, :] + uu[:, :, None] * right[None, None, :] + vv[:, :, None] * up[None, None, :]
            dirs /= np.maximum(1e-6, np.linalg.norm(dirs, axis=-1, keepdims=True))
            best_t = np.full((FOVEA_H, FOVEA_W), max_r, dtype=np.float32)
            nx = np.zeros((FOVEA_H, FOVEA_W), dtype=np.float32)
            ny = np.zeros((FOVEA_H, FOVEA_W), dtype=np.float32)
            sal = np.full((FOVEA_H, FOVEA_W), 0.10, dtype=np.float32)

            dy = dirs[:, :, 1]
            m_fl = dy < -1e-4
            t_fl = (float(self.wmin[1]) - float(me["pos"][1])) / np.where(m_fl, dy, -1.0)
            h_fl = m_fl & (t_fl > 0.05) & (t_fl < best_t)
            if np.any(h_fl):
                best_t = np.where(h_fl, t_fl, best_t)
                ny = np.where(h_fl, 1.0, ny)
                hx = me["pos"][0] + best_t * dirs[:, :, 0]
                hz = me["pos"][2] + best_t * dirs[:, :, 2]
                ix = np.clip(np.floor((hx - self.wmin[0]) / self.span[0] * self.grid_n).astype(np.int32), 0, self.grid_n - 1)
                iz = np.clip(np.floor((hz - self.wmin[2]) / self.span[2] * self.grid_n).astype(np.int32), 0, self.grid_n - 1)
                sv = self.surface_grid[iz, ix] * float(t_sign)
                sal = np.where(h_fl, np.where(sv > 0.25, 0.78, np.where(sv < -0.25, 0.25, 0.50)), sal)

            for ent in vis_list[:12]:
                oc = ent[1:4] - me["pos"]
                rad = float(ent[7])
                bp = np.einsum("hwc,c->hw", dirs, oc)
                disc = bp * bp - float(np.dot(oc, oc) - rad * rad)
                ts = bp - np.sqrt(np.maximum(0.0, disc))
                cl = (disc > 0.0) & (ts > 0.05) & (ts < best_t)
                if np.any(cl):
                    best_t = np.where(cl, ts, best_t)
                    n_vec = (me["pos"][None, None, :] + best_t[:, :, None] * dirs - ent[None, None, 1:4]) / max(1e-3, rad)
                    nx = np.where(cl, n_vec[:, :, 0], nx)
                    ny = np.where(cl, n_vec[:, :, 1], ny)
                    sal = np.where(cl, 0.95 if ent[8] < 0 else 0.60, sal)

            fovea = np.stack([1.0 - np.clip(best_t / max_r, 0.0, 1.0), nx, ny, sal], axis=-1).astype(np.float32)

        iz_m, ix_m = self._world_to_cell(float(me["pos"][0]), float(me["pos"][2]))
        my_s = float(self.surface_grid[iz_m, ix_m]) * float(t_sign)

        # Strict 1st-person event-gated acoustic & somatic/visual hazard model (matching Unity 6 ArenaAudioSystem.cs — ZERO wallhack/bearing leak!)
        en_spd = float(np.linalg.norm(en["vel"][[0, 2]]))
        occ_atten = 0.52 if self._is_occluded(me["pos"], en["pos"]) else 1.0
        raw_loud = 0.0
        if bool(en.get("just_fired", False)) or int(en.get("cd", 0)) >= 6:
            raw_loud = max(raw_loud, 0.92 * max(0.0, 1.0 - d_en / 15.0))
        elif (not en["in_refuge"]) and (not bool(en.get("submerged", False))) and en_spd > 0.55:
            raw_loud = max(raw_loud, 0.62 * max(0.0, 1.0 - d_en / 11.0))
        elif bool(en.get("submerged", False)) and en_spd > 0.80:
            raw_loud = max(raw_loud, 0.38 * max(0.0, 1.0 - d_en / 4.2))
        loud = float(np.clip(raw_loud * occ_atten, 0.0, 1.0))
        if loud >= 0.05 and d_en > 1e-4:
            rel_en = en["pos"][[0, 2]] - me["pos"][[0, 2]]
            ang_en = math.atan2(float(rel_en[0]), float(rel_en[1])) - float(me["gaze_ypd"][0])
            aud_sin, aud_cos = math.sin(ang_en), math.cos(ang_en)
        else:
            loud = 0.0
            aud_sin, aud_cos = 0.0, 1.0

        # Somatic + 1st-person visual/acoustic hazard only (never uses unseen/silent true distance d_en!)
        haz = 0.85 * float(np.clip(1.0 - float(me["hp"]), 0.0, 1.0))
        if float(me.get("regen_delay", 0.0)) > 0.40:
            haz = max(haz, 0.35)
        if my_s < -0.30:
            haz = max(haz, 0.25)
        for ent in vis_list:
            if float(ent[8]) < -0.25:
                dv_h = float(np.linalg.norm(ent[1:4] - me["pos"]))
                if int(round(float(ent[0]))) >= 500:
                    if dv_h < 3.6:
                        haz = max(haz, float(np.clip((3.6 - dv_h) / 3.6, 0.0, 1.0)))
                elif dv_h < 8.5:
                    haz = max(haz, float(np.clip((8.5 - dv_h) / 8.5 * 0.80 * float(ent[9]), 0.0, 1.0)))
        if (bool(en.get("just_fired", False)) or int(en.get("cd", 0)) >= 6) and loud >= 0.30:
            haz = max(haz, float(np.clip((loud - 0.25) * 0.75, 0.0, 0.55)))
        haz = float(np.clip(haz, 0.0, 1.0))

        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[0:3] = (me["pos"] - self.wmin) / self.span
        proprio[3:6] = np.clip(me["vel"] / 8.0, -1.0, 1.0)
        proprio[6] = 1.0 if me["cd"] == 0 else 0.0
        proprio[7] = 1.0 if me["respawn_flag"] else 0.0
        proprio[8] = haz
        proprio[9] = float(vc) / float(MAX_ENTITIES)
        proprio[10] = aud_sin
        proprio[11] = aud_cos
        proprio[12] = loud
        proprio[13] = 0.95 if me["in_refuge"] else (0.75 if my_s > 0.28 else (0.10 if my_s < -0.28 else 0.45))
        proprio[14] = 0.96 if self.carrier_team == t_sign else float(np.clip(me["tank"] * 0.88, 0.0, 0.88))
        proprio[15] = self.objective_progress_a if t_sign > 0 else (1.0 - self.objective_progress_a)

        return {
            "episode_id": self.episode_id,
            "action_epoch": me["epoch"],
            "stage_id": int(getattr(self.cfg, "default_stage_id", 1)),
            "physics_tick": self.physics_tick,
            "can_commit_now": 1 if me["cd"] == 0 else 0,
            "episode_done": 1 if ep_done else 0,
            "respawn_flag": bool(me["respawn_flag"]),
            "visible_count": vc,
            "cumulative_score": float(me["cum_score"]),
            "last_step_reward": float(step_rew),
            "hazard_level": haz,
            "self_pos": me["pos"].copy(),
            "self_vel": me["vel"].copy(),
            "self_yaw_pitch": me["gaze_ypd"][:2].copy(),
            "proprio": proprio,
            "visible_entities": vis_arr,
            "fovea_patch": fovea,
        }


# ==============================================================================
# 4. Single-GPU Batched Population & Opponent Server (`Barrier 1` Enforced)
# ==============================================================================
def run_population_gpu_server_process(
    base_cfg: Brain3DConfig,
    pop_size: int,
    max_opp_slots: int,
    req_q: mp.Queue,
    resp_queues: Dict[int, mp.Queue],
    ctrl_q: mp.Queue,
    ctrl_resp_q: mp.Queue,
    stop_ev,
    ready_ev,
    abort_save_ev=None,
) -> None:
    """
    Exclusive PyTorch/CUDA Server Process hosting:
      - Learner Brains `0 .. pop_size - 1` (trainable online + Sleep S-1..S-4)
      - Opponent Shadow Slots `pop_size .. pop_size + max_opp_slots - 1` (inference-only snapshots for Self-Play!)
    """
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()

    from universal_brain_3d_core import WholeBrainNeuralEngineWrapper

    engines: Dict[int, WholeBrainNeuralEngineWrapper] = {}
    for s_id in range(pop_size + max_opp_slots):
        scfg = Brain3DConfig(**asdict(base_cfg))
        scfg.seed = int(base_cfg.seed) + s_id * 17
        if s_id < pop_size:
            apply_checkpoint_genes_to_cfg(scfg, PBT_ARCHETYPES[s_id % len(PBT_ARCHETYPES)])
        engines[s_id] = WholeBrainNeuralEngineWrapper(scfg)

    # Warm up CUDA/cuDNN kernels before signaling ready so Tick 1 never stalls TCP bridges
    warmup_req = {
        "episode_id": 0,
        "respawn_flag": False,
        "enable_learning": False,
        "fovea_patch": np.zeros((FOVEA_H, FOVEA_W, FOVEA_C), dtype=np.float32),
        "ventral_patch": np.zeros((FOVEA_H, FOVEA_W, 4), dtype=np.float32),
        "proprio": np.zeros(PROPRIO_DIM, dtype=np.float32),
        "mem_compass": np.zeros(24, dtype=np.float32),
        "mec_grid_code": np.zeros(256, dtype=np.float32),
        "hd_ring_code": np.zeros(32, dtype=np.float32),
        "believed_self_pos": np.zeros(3, dtype=np.float32),
        "cand_poses": np.zeros((NUM_CANDIDATES, CONT_POSE_DIM), dtype=np.float32),
        "cand_verbs": np.zeros(NUM_CANDIDATES, dtype=np.int32),
    }
    engines[0].infer_conscious_gwt_batch([warmup_req])

    ready_ev.set()

    try:
        while not stop_ev.is_set():
            if not is_windows_pid_alive(orch_ppid):
                stop_ev.set()
                break

            # 1. Service coordinator control commands (Exploit/Explore, Load Opponent, Sleep, Save Champion)
            try:
                while True:
                    cmd, pay = ctrl_q.get_nowait()
                    if cmd == "get_slot_state":
                        sid = int(pay["slot_id"])
                        ctrl_resp_q.put({"ok": True, "state": engines[sid].get_weights_state_dict(as_numpy=True), "genes": extract_brain_genes_from_cfg(engines[sid].cfg)})
                    elif cmd == "set_slot_state":
                        sid = int(pay["slot_id"])
                        if "genes" in pay:
                            apply_checkpoint_genes_to_cfg(engines[sid].cfg, pay["genes"])
                        engines[sid].load_weights_state_dict(pay["state"], lr_scale=engines[sid].cfg.lr_scale)
                        ctrl_resp_q.put({"ok": True})
                    elif cmd == "sleep_all_learners":
                        ep_cnt = int(pay.get("epochs", 4))
                        losses = [
                            engines[i].run_sleep_consolidation(ep_cnt, should_abort_cb=stop_ev.is_set)["mean_loss"]
                            for i in range(pop_size)
                            if not stop_ev.is_set()
                        ]
                        ctrl_resp_q.put({"ok": True, "mean_sleep_loss": float(np.mean(losses)) if losses else 0.0})
                    elif cmd == "save_slot":
                        if stop_ev.is_set() or (abort_save_ev is not None and abort_save_ev.is_set()):
                            ctrl_resp_q.put({"ok": True, "skipped_emergency_abort": True})
                        else:
                            sid = int(pay["slot_id"])
                            engines[sid].save_checkpoint(str(pay["path"]), extra_meta=pay.get("meta"))
                            ctrl_resp_q.put({"ok": True})
                    elif cmd == "load_slot_file":
                        sid = int(pay["slot_id"])
                        meta = engines[sid].load_checkpoint(str(pay["path"]))
                        ctrl_resp_q.put({"ok": True, "meta": meta})
                    elif cmd == "check_barrier":
                        ctrl_resp_q.put({"ok": True, "torch_loaded": ("torch" in sys.modules), "device": str(engines[0].device)})
            except queue.Empty:
                pass

            # 2. Drain worker inference & transition requests in a micro-batch
            batch_items: List[Tuple[int, int, str, Dict[str, Any]]] = []
            try:
                batch_items.append(req_q.get(timeout=0.004))
                t_end = time.perf_counter() + (0.0005 if pop_size == 1 else 0.0010)
                target_batch = max(2, pop_size * (2 if max_opp_slots > 0 else 1))
                while time.perf_counter() < t_end and len(batch_items) < target_batch:
                    try:
                        batch_items.append(req_q.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            # Group by brain_id so identical-weight requests batch together cleanly
            by_brain_gwt: Dict[int, List[Tuple[int, int, Dict[str, Any]]]] = {}
            for worker_id, req_id, op, pay in batch_items:
                bid = int(pay.get("brain_id", worker_id))
                eng = engines.get(bid, engines[0])
                if op == "infer_gwt":
                    by_brain_gwt.setdefault(bid, []).append((worker_id, req_id, pay))
                elif op == "notify_chosen_action":
                    eng.notify_chosen_action(int(pay.get("slot_id", 0)), pay["chosen_cand_feat32"])
                    if worker_id in resp_queues:
                        resp_queues[worker_id].put((req_id, {"ok": True}))
                elif op == "record_transition":
                    res = eng.record_transition_and_maybe_train(pay) if bid < pop_size else {"ok": True}
                    if worker_id in resp_queues:
                        resp_queues[worker_id].put((req_id, res))

            for bid, req_list in by_brain_gwt.items():
                outs = engines[bid].infer_conscious_gwt_batch([x[2] for x in req_list])
                for (wid, rid, _), out in zip(req_list, outs):
                    if wid in resp_queues:
                        resp_queues[wid].put((rid, out))
    finally:
        for eng in list(engines.values()):
            try:
                eng.release_memory_and_vram()
            except Exception:
                pass
        engines.clear()
        for q_obj in [req_q, ctrl_q, ctrl_resp_q, *resp_queues.values()]:
            try:
                q_obj.cancel_join_thread()
            except Exception:
                pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass


# ==============================================================================
# 5. Local Zero-GPU Embodied Brain Controller for Arena Workers
# ==============================================================================
class LocalCpuBrainAgentDriver:
    """
    Lightweight Zero-GPU (`Barrier 1` safe) per-agent embodied controller running Spatial Cognition v2,
    4-Layer Field Map, Ballistic Cerebellum, Basal Ganglia Gate, and querying the shared GPU server.
    """

    def __init__(
        self,
        cfg: Brain3DConfig,
        brain_id: int,
        worker_id: int,
        req_q: mp.Queue,
        resp_q: mp.Queue,
        trainable: bool = True,
        agent_state_id: Optional[int] = None,
        stop_ev=None,
    ):
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self.brain_id = int(brain_id)
        self.worker_id = int(worker_id)
        self.agent_state_id = int(agent_state_id) if agent_state_id is not None else (self.worker_id * 2 + (0 if trainable else 1))
        self.req_q = req_q
        self.resp_q = resp_q
        self.trainable = bool(trainable)
        self.stop_ev = stop_ev
        self._rpc_seq = 0

        self.hippo = EntorhinalHippocampal3DEngineV2(cfg)
        self.hippo.set_learning_context(self.trainable)
        self.cereb = Cerebellar3DKinematicPredictorV2(cfg)
        self.bg = BasalGanglia3DTimingEngineV2(cfg)
        self.homeostasis = ActionStagnationHomeostasis3D(cfg.world_min, cfg.world_max)
        self.strategy_ctrl = create_runtime_strategy_controller(cfg, slot_id=self.worker_id)
        self.latest_strategy_tag: str = ""

        self.prev_pi: Optional[np.ndarray] = None
        self.mod_hist: deque = deque(maxlen=48)
        self.ventral_prev: Dict[str, Any] = {}
        self.last_lead_xyz = np.zeros(3, dtype=np.float32)
        self.last_goal_mode = GOAL_MODE_BALANCED_EXPLORE
        self.last_focus_xyz = np.zeros(3, dtype=np.float32)
        self.last_slot_coords: Optional[np.ndarray] = None
        self.last_slot_valence: Optional[np.ndarray] = None
        self.last_vis_pred_surprise: float = 0.0
        self.last_world_surprise: float = 0.0
        self.last_ignition: float = 0.0
        self.last_self_surprise: float = 0.0
        self.cached_pose = np.zeros(CONT_POSE_DIM, dtype=np.float32)
        self.cached_verb = VERB_APPROACH_HOVER
        self.cached_chosen_idx = 0
        self.cached_speed_mod = 0.85
        self.cached_q = 0.5
        self.cached_surv = 0.85
        self.pending_tr: Optional[Dict[str, Any]] = None
        self.transitions_sent = 0
        self.cog_steps_done = 0
        self.ep_cog_steps_done = 0
        self.last_ep_seen = -1
        self.respawn_seq = 0
        self._inflight_rid: Optional[int] = None
        self._inflight_ep: int = -1
        self.latest_veto: bool = False
        self.latest_hold_reason: int = HOLD_GO_CLEAR
        self._last_gwt_snap: Optional[Dict[str, Any]] = None
        self.latest_ubnt_ctx: Optional[Dict[str, Any]] = None

    def _enqueue_rpc(self, op: str, payload: Dict[str, Any]) -> int:
        self._rpc_seq += 1
        rid = (self.agent_state_id << 22) | (self._rpc_seq & 0x3FFFFF)
        payload["brain_id"] = self.brain_id
        payload["slot_id"] = self.agent_state_id
        self.req_q.put((self.worker_id, rid, op, payload))
        return rid

    def _rpc(self, op: str, payload: Dict[str, Any], timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        if self.stop_ev is not None and self.stop_ev.is_set():
            return None
        rid = self._enqueue_rpc(op, payload)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.stop_ev is not None and self.stop_ev.is_set():
                return None
            try:
                got_id, res = self.resp_q.get(timeout=min(0.05, max(0.005, deadline - time.perf_counter())))
                if got_id == rid:
                    return res
            except queue.Empty:
                continue
        return None

    def rebind_brain(self, brain_id: int, policy_mode: str, trainable: bool) -> None:
        new_bid = int(brain_id)
        if new_bid != self.brain_id:
            self.brain_id = new_bid
            self.hippo.clear_all_stage_memories()
            self.pending_tr = None
            self.prev_pi = None
            self._last_gwt_snap = None
        self.trainable = bool(trainable)
        self.cfg.policy_mode = str(policy_mode)
        self.hippo.set_learning_context(self.trainable)

    def close(self) -> None:
        if self.strategy_ctrl is not None:
            self.strategy_ctrl.close()

    def begin_step_observation(
        self,
        obs: Dict[str, Any],
        dt: float,
        cog_stride: int = 4,
        async_gwt: bool = False,
    ) -> Dict[str, Any]:
        ep_cur = int(obs["episode_id"])
        self.hippo.set_learning_context(self.trainable)
        if ep_cur != self.last_ep_seen:
            self.last_ep_seen = ep_cur
            self.ep_cog_steps_done = 0
            self._inflight_rid = None
            self._inflight_ep = ep_cur
            self.last_slot_coords = None
            self.last_slot_valence = None
            self.last_vis_pred_surprise = 0.0
            self.last_world_surprise = 0.0
            self.last_ignition = 0.0
            self.last_self_surprise = 0.0
            self.mod_hist.clear()
        if obs.get("respawn_flag", False):
            self.respawn_seq += 1
        vent_p = derive_ventral_fallback_from_dorsal_16(
            obs["fovea_patch"], yaw_pitch=obs["self_yaw_pitch"], prev_state=self.ventral_prev, episode_id=int(obs["episode_id"])
        )
        sp_out = self.hippo.step(
            obs, dt, ventral_override=vent_p, respawn_atom=self.respawn_seq,
            topdown_goal_mode=self.last_goal_mode, topdown_focus_xyz=self.last_focus_xyz, lead_aim_xyz=self.last_lead_xyz,
            vis_slot_coords=self.last_slot_coords, vis_slot_valence=self.last_slot_valence,
            vis_pred_surprise=self.last_vis_pred_surprise,
            world_surprise=self.last_world_surprise,
            gwt_ignition=self.last_ignition,
        )
        lead_xyz, opp_prob, kin_risk, sc_spd = self.cereb.predict_lead_and_crossfire(
            sp_out["believed_self_pos"], sp_out["believed_entities"], sp_out["believed_count"]
        )
        self.last_lead_xyz = lead_xyz.copy()

        prop = project_pomdp_proprio(
            obs["proprio"], sp_out["believed_self_pos"], self.cfg.world_min, self.cfg.world_max, self.cfg
        )
        tick = int(obs["physics_tick"])
        ep_done = bool(obs["episode_done"])
        respawn_ev = bool(sp_out["respawn_event_flag"])

        rid_gwt: Optional[int] = None
        need_first_tr = self.ep_cog_steps_done == 0 or (self.trainable and self.pending_tr is None)
        want_cog = (tick % max(1, cog_stride) == 0) or ep_done or respawn_ev or need_first_tr

        cached_mc = getattr(self, "_cached_mc24", None)
        cached_cp = getattr(self, "_cached_c_poses", None)
        cached_cv = getattr(self, "_cached_c_verbs", None)
        if want_cog or cached_mc is None or cached_cp is None or cached_cv is None:
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
            mc24 = cached_mc
            c_poses = cached_cp
            c_verbs = cached_cv

        if want_cog:
            gwt_pay = {
                "episode_id": ep_cur,
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
            if async_gwt and (not self.trainable) and self.ep_cog_steps_done > 0 and (not ep_done):
                if self._inflight_rid is None:
                    self._inflight_rid = self._enqueue_rpc("infer_gwt", gwt_pay)
                    self._inflight_ep = ep_cur
            else:
                rid_gwt = self._enqueue_rpc("infer_gwt", gwt_pay)

        return {
            "obs": obs,
            "dt": dt,
            "vent_p": vent_p,
            "sp_out": sp_out,
            "opp_prob": opp_prob,
            "kin_risk": kin_risk,
            "sc_spd": sc_spd,
            "prop": prop,
            "mc24": mc24,
            "c_poses": c_poses,
            "c_verbs": c_verbs,
            "tick": tick,
            "ep_done": ep_done,
            "respawn_ev": respawn_ev,
            "rid_gwt": rid_gwt,
        }

    def finish_step_observation(
        self,
        step_ctx: Dict[str, Any],
        gwt_out: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, int, bool, np.ndarray]:
        obs = step_ctx["obs"]
        dt = float(step_ctx["dt"])
        vent_p = step_ctx["vent_p"]
        sp_out = step_ctx["sp_out"]
        opp_prob = float(step_ctx["opp_prob"])
        kin_risk = float(step_ctx["kin_risk"])
        sc_spd = float(step_ctx["sc_spd"])
        prop = step_ctx["prop"]
        mc24 = step_ctx["mc24"]
        c_poses = step_ctx["c_poses"]
        c_verbs = step_ctx["c_verbs"]
        tick = int(step_ctx["tick"])
        ep_done = bool(step_ctx["ep_done"])
        respawn_ev = bool(step_ctx["respawn_ev"])

        strat_cur_m = 1.0
        strat_obj_m = 1.0
        strat_caut_d = 0.0
        strat_peace_mode: Optional[int] = None
        if self.strategy_ctrl is not None and (gwt_out is not None or ep_done or respawn_ev):
            ign_v = float(gwt_out["ignition"]) if gwt_out is not None else 0.0
            synth_sp = ""
            if gwt_out is not None and (ign_v >= 0.50 or respawn_ev or self.latest_veto):
                synth_sp = synthesize_joint_attention_phrase(obs, sp_out, int(self.cached_verb), goal_mode=int(self.last_goal_mode))
            st_out = self.strategy_ctrl.step(
                episode_id=int(obs["episode_id"]),
                stage_id=int(obs.get("stage_id", self.cfg.default_stage_id)),
                tick=tick,
                pos_xz=sp_out["believed_self_pos"][[0, 2]],
                proprio=prop,
                believed_entities=sp_out["believed_entities"],
                believed_count=int(sp_out["believed_count"]),
                friendly_turf=float(sp_out["friendly_turf_ratio"]),
                enemy_turf=float(sp_out["enemy_turf_ratio"]),
                cum_score=float(obs["cumulative_score"]),
                step_reward=float(obs["last_step_reward"]),
                respawn_ev=respawn_ev,
                ep_done=ep_done,
                veto_active=bool(self.latest_veto),
                ignition=ign_v,
                stagnation_level=float(self.homeostasis.stagnation_level),
                inner_speech=synth_sp,
                field_map=self.hippo.field_map,
            )
            strat_cur_m = float(st_out["curiosity_mult"])
            strat_obj_m = float(st_out["objective_mult"])
            strat_caut_d = float(st_out["caution_delta"])
            strat_peace_mode = st_out["peace_goal_mode_override"]
            self.latest_strategy_tag = self.strategy_ctrl.format_hud_status_tag()

        if gwt_out is not None:
            self.cog_steps_done += 1
            self.ep_cog_steps_done += 1
            raw_gmode = int(gwt_out["topdown_goal_mode"])
            if strat_peace_mode is not None and raw_gmode == GOAL_MODE_BALANCED_EXPLORE:
                self.last_goal_mode = int(strat_peace_mode)
            else:
                self.last_goal_mode = raw_gmode
            self.last_focus_xyz = gwt_out["topdown_focus_xyz"].copy()
            self.last_vis_pred_surprise = float(gwt_out.get("vis_pred_surprise", 0.0))
            self.last_world_surprise = float(gwt_out.get("world_surprise", 0.0))
            self.last_ignition = float(gwt_out.get("ignition", 0.0))
            self.last_self_surprise = float(gwt_out.get("self_surprise", 0.0))
            if "vis_slot_coords" in gwt_out and "vis_slot_valence" in gwt_out:
                self.last_slot_coords = gwt_out["vis_slot_coords"].copy()
                self.last_slot_valence = gwt_out["vis_slot_valence"].copy()
            self.mod_hist.append(extract_5module_activity_vec(gwt_out, sp_out["mec_grid_code"], prop, mc24))
            mip_info: Dict[str, Any] = {}
            q_star, pi, phi_bits, fr_step, ent_norm, veto_act = compute_gene_modulated_mip_and_fr(
                gwt_out["q_values"], gwt_out["survival_probs"], gwt_out["pred_adv_probs"],
                gwt_out["gwt_workspace"], c_poses, c_verbs, prop, mc24, self.prev_pi, self.cfg,
                habituation_trace=self.homeostasis.habituation_trace,
                stagnation_level=self.homeostasis.stagnation_level,
                q_raw_values=gwt_out.get("q_raw_values"),
                cortical_mod_history=self.mod_hist,
                self_surprise=self.last_self_surprise,
                strategic_curiosity_mult=strat_cur_m,
                strategic_objective_mult=strat_obj_m,
                strategic_caution_delta=strat_caut_d,
                mip_info_out=mip_info,
            )
            self.prev_pi = pi.copy()
            self.latest_veto = bool(veto_act)
            s1_fast_idx = int(np.argmax(gwt_out["q_raw_values"]))
            self._last_gwt_snap = {
                "gwt_out": gwt_out,
                "q_star": q_star.copy(),
                "pi": pi.copy(),
                "phi_bits": float(phi_bits),
                "mip_cut_mask": int(mip_info.get("mip_cut_mask", 3)),
                "fr_step": float(fr_step),
                "ent_norm": float(ent_norm),
                "veto_act": bool(veto_act),
                "s1_fast_idx": s1_fast_idx,
            }

            sp_dict = {
                "believed_self_pos": sp_out["believed_self_pos"],
                "believed_hazard": sp_out["believed_hazard"],
                "mem_compass": mc24,
                "believed_entities": sp_out["believed_entities"],
                "believed_count": sp_out["believed_count"],
            }
            if self.trainable:
                beta_t = compute_beta_teacher(self.cfg.policy_mode, self.transitions_sent, self.cfg.anneal_transitions)
            elif self.cfg.policy_mode == "dagger":
                beta_t = 0.35 * compute_beta_teacher("dagger", self.cog_steps_done // 2, self.cfg.anneal_transitions)
            else:
                beta_t = 0.0
            tq, tsurv, tidx = compute_task_scaffold_targets(sp_dict, obs, c_poses, c_verbs, self.cfg.world_min, self.cfg.world_max, self.profile)
            rng_seed = tick + self.agent_state_id * 7919 + int(obs["episode_id"]) * 131
            if beta_t >= 0.999 or (beta_t > 0.01 and ((rng_seed * 1664525 + 1013904223) & 0xFFFF) < int(beta_t * 65535.0)):
                c_idx = tidx
            else:
                c_idx = select_policy_candidate(q_star, pi, trainable=self.trainable, rng_seed_val=rng_seed)

            if c_idx != int(gwt_out.get("prov_chosen_idx", -1)):
                self._rpc(
                    "notify_chosen_action",
                    {"chosen_cand_feat32": gwt_out["cand_feats32"][c_idx].copy()},
                    timeout=0.5,
                )

            r_int = self.homeostasis.update_and_compute_intrinsic(
                episode_id=int(obs["episode_id"]),
                pos_xz=sp_out["believed_self_pos"][[0, 2]],
                proprio=prop,
                chosen_idx=c_idx,
                step_extrinsic_reward=float(obs["last_step_reward"]),
                world_surprise=float(gwt_out["world_surprise"]),
                self_surprise=self.last_self_surprise,
                respawn_flag=respawn_ev,
            )

            prev_cand_idx = self.cached_chosen_idx
            self.cached_chosen_idx = c_idx
            self.cached_speed_mod = float(gwt_out["speed_mod"][c_idx])
            self.cached_verb = int(c_verbs[c_idx])
            self.cached_pose = c_poses[c_idx].copy()
            self.cached_pose[5] = self.cached_speed_mod
            self.cached_q = float(gwt_out["q_values"][c_idx])
            self.cached_surv = float(gwt_out["survival_probs"][c_idx])
            just_switched_cand = (c_idx != prev_cand_idx)

            cur_haz = max(float(obs["hazard_level"]), float(sp_out["believed_hazard"]))
            if self.trainable and self.pending_tr is not None:
                same_ep = int(self.pending_tr["episode_id"]) == int(obs["episode_id"])
                if (not same_ep) or ep_done or respawn_ev or (tick - int(self.pending_tr["tick"]) >= 8):
                    rew_acc = float(obs["cumulative_score"]) - float(self.pending_tr["cum_score"]) if same_ep else float(obs["last_step_reward"])
                    haz_delta_sig = cur_haz - float(self.pending_tr["hazard"])
                    self.pending_tr["reward"] = float(np.clip(rew_acc + r_int - 0.15 * max(0.0, haz_delta_sig), -28.0, 35.0))
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
                    self._rpc("record_transition", self.pending_tr)
                    self.transitions_sent += 1
                    self.pending_tr = None

            if self.trainable and self.pending_tr is None and (not ep_done):
                tr_d: Dict[str, Any] = {
                    "episode_id": int(obs["episode_id"]),
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
                    "self_surprise": self.last_self_surprise,
                    "beta_teacher": float(beta_t),
                }
                if beta_t > 0.02:
                    tr_d["teacher_q"] = tq
                    tr_d["teacher_surv"] = tsurv
                self.pending_tr = tr_d
        else:
            just_switched_cand = False
            c_live = int(np.clip(self.cached_chosen_idx, 0, NUM_CANDIDATES - 1))
            self.cached_verb = int(c_verbs[c_live])
            self.cached_pose = c_poses[c_live].copy()
            self.cached_pose[5] = self.cached_speed_mod

        s1_idx_for_bg = self._last_gwt_snap["s1_fast_idx"] if self._last_gwt_snap is not None else self.cached_chosen_idx
        bg_res = self.bg.step(
            episode_id=int(obs["episode_id"]),
            action_epoch=int(obs["action_epoch"]),
            can_commit_now=bool(obs["can_commit_now"]),
            chosen_cand_idx=self.cached_chosen_idx,
            s1_fast_cand_idx=s1_idx_for_bg,
            chosen_verb=self.cached_verb,
            q_chosen=self.cached_q,
            survival_prob=self.cached_surv,
            scene_opportunity_prob=opp_prob,
            local_kinetic_risk=kin_risk,
            believed_scene_speed=sc_spd,
            mean_fog=sp_out["mean_fog"],
            saccade_hold_active=bool(sp_out["saccade_hold_active"]),
            respawn_event_flag=bool(sp_out["respawn_event_flag"]),
            coach_brake=False,
            in_refuge=is_in_stealth_or_cover_refuge(prop),
            tank_level=get_resource_tank_level(prop),
            hazard_level=max(float(obs["hazard_level"]), sp_out["believed_hazard"]),
            step_reward=float(obs["last_step_reward"]),
            dt=dt,
            self_surprise=self.last_self_surprise,
        )
        gate_open = bool(bg_res["commit_gate_open"])
        self.latest_hold_reason = int(bg_res.get("hold_reason_id", HOLD_GO_CLEAR))
        if just_switched_cand and self.cached_verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH):
            gate_open = False
        if self._last_gwt_snap is not None:
            self.latest_ubnt_ctx = {
                "obs": obs,
                "sp_out": sp_out,
                "bg_res": bg_res,
                "c_poses": c_poses,
                "c_verbs": c_verbs,
                "opp_prob": opp_prob,
                "gate_open": gate_open,
            }
        return self.cached_pose.copy(), self.cached_verb, gate_open, sp_out["active_gaze_ypd"].copy()

    def build_ubnt_packet(self, seq: int) -> Optional[bytes]:
        if self._last_gwt_snap is None or self.latest_ubnt_ctx is None:
            return None
        gs = self._last_gwt_snap
        gw = gs["gwt_out"]
        ctx_u = self.latest_ubnt_ctx
        obs = ctx_u["obs"]
        sp = ctx_u["sp_out"]
        bg = ctx_u["bg_res"]
        scalars24 = [
            60.0,
            15.0,
            4.5,
            float(obs.get("cumulative_score", 0.0)),
            float(obs.get("last_step_reward", 0.0)),
            float(obs.get("hazard_level", 0.0)),
            float(sp.get("believed_hazard", 0.0)),
            float(sp.get("map_accuracy_iou", 1.0)),
            float(sp.get("mean_fog", 0.5)),
            float(gw.get("ignition", 0.0)),
            float(gs["phi_bits"]),
            float(gs["ent_norm"]),
            float(gw.get("world_surprise", 0.0)),
            float(gw.get("self_surprise", 0.0)),
            float(gs["fr_step"]),
            float(self.cached_surv),
            float(bg.get("go_drive", 1.0)),
            float(bg.get("stn_brake", 0.0)),
            float(bg.get("urgency_val", 0.0)),
            float(bg.get("net_accumulator", 0.0)),
            float(bg.get("release_threshold", 0.72)),
            float(ctx_u["opp_prob"]),
            float(sp.get("friendly_turf_ratio", 0.0)),
            float(sp.get("enemy_turf_ratio", 0.0)),
        ]
        return pack_ubnt_neural_telemetry(
            seq=int(seq),
            timestamp_ns=time.perf_counter_ns(),
            episode_id=int(obs["episode_id"]),
            action_epoch=int(obs["action_epoch"]),
            physics_tick=int(obs["physics_tick"]),
            chosen_cand_idx=int(self.cached_chosen_idx),
            s1_fast_cand_idx=int(gs["s1_fast_idx"]),
            commit_gate_open=bool(ctx_u["gate_open"]),
            veto_active=bool(gs["veto_act"]),
            timing_mode=int(bg.get("timing_mode", self.cfg.timing_mode)),
            hold_reason_id=int(bg.get("hold_reason_id", HOLD_GO_CLEAR)),
            coach_brake_active=False,
            sleep_active=False,
            sleep_stage=0,
            virtual_alert_active=bool(gs["veto_act"]),
            endogenous_active=False,
            proto_sym_id=int(gw.get("proto_sym_id", 0)),
            proto_mod_id=int(gw.get("proto_mod_id", 0)),
            believed_count=int(sp.get("believed_count", 0)),
            cand_verbs=ctx_u["c_verbs"],
            holds_triggered=int(bg.get("holds_triggered", 0)),
            opportunities_caught=int(bg.get("opportunities_caught", 0)),
            loop_closures_total=int(sp.get("loop_closures_total", 0)),
            virtual_alert_count=0,
            scalars24=scalars24,
            gwt_alpha=gw["gwt_alpha"],
            gwt_workspace=gw["gwt_workspace"],
            rssm_h=gw["rssm_h"],
            latent_z=gw["latent_z"],
            z_topdown_delta=gw["z_topdown_delta"],
            mec_grid_code=sp["mec_grid_code"],
            hd_ring_code=sp["hd_ring_code"],
            self_state=gw["self_state"],
            q_values=gw["q_values"],
            q_raw_values=gw["q_raw_values"],
            survival_probs=gw["survival_probs"],
            policy_probs=gs["pi"],
            mip_q_star=gs["q_star"],
            cand_poses=ctx_u["c_poses"],
            stn_gains=bg.get("stn_gains", np.ones(4, dtype=np.float32)),
            believed_self_pos=sp["believed_self_pos"],
            pose_drift_vec=sp.get("pose_drift_vec", np.zeros(3, dtype=np.float32)),
            inner_speech_text=self.latest_strategy_tag,
            learning_enabled=bool(self.trainable),
            mip_cut_mask=int(gs.get("mip_cut_mask", 3)),
        )

    def build_ubwm_packet(self) -> Optional[bytes]:
        if self._last_gwt_snap is None or self.latest_ubnt_ctx is None:
            return None
        gw = self._last_gwt_snap["gwt_out"]
        obs = self.latest_ubnt_ctx["obs"]
        z16 = np.zeros((FOVEA_H, FOVEA_W), dtype=np.uint8)
        return pack_ubwm_world_model(
            episode_id=int(obs["episode_id"]),
            physics_tick=int(obs["physics_tick"]),
            chosen_cand_idx=int(self.cached_chosen_idx),
            has_prev_pred=bool(gw.get("wm_has_prev_pred", False)),
            veto_mask_u8=int(gw.get("wm_veto_mask_u8", 0)),
            vis_pred_mae=float(gw.get("wm_vis_mae", 0.0)),
            depth_pred_mae_m=float(gw.get("wm_depth_mae_m", 0.0)),
            world_kl_surprise=float(gw.get("world_surprise", 0.0)),
            action_cf_spread=float(gw.get("wm_cf_spread", 0.0)),
            pred_rew=gw.get("pred_rew", np.zeros(NUM_CANDIDATES, dtype=np.float32)),
            pred_surv=gw.get("survival_probs", np.ones(NUM_CANDIDATES, dtype=np.float32)),
            pred_adv=gw.get("pred_adv_probs", np.full(NUM_CANDIDATES, 0.5, dtype=np.float32)),
            pred_depth_u8=gw.get("wm_pred_depth_u8", z16),
            pred_surf_u8=gw.get("wm_pred_surf_u8", z16),
            pred_err_u8=gw.get("wm_pred_err_u8", z16),
        )

    def step_observation(self, obs: Dict[str, Any], dt: float, cog_stride: int = 4) -> Tuple[np.ndarray, int, bool, np.ndarray]:
        use_async = not self.trainable
        ctx_step = self.begin_step_observation(obs, dt, cog_stride=cog_stride, async_gwt=use_async)
        rid = ctx_step["rid_gwt"]
        gwt_out = None
        if use_async and self._inflight_rid is not None and rid is None:
            while True:
                try:
                    got_id, res = self.resp_q.get_nowait()
                    if got_id == self._inflight_rid:
                        self._inflight_rid = None
                        if self._inflight_ep == int(obs["episode_id"]):
                            gwt_out = res
                        break
                except queue.Empty:
                    break
        elif rid is not None:
            deadline = time.perf_counter() + 3.0
            while time.perf_counter() < deadline:
                if self.stop_ev is not None and self.stop_ev.is_set():
                    break
                try:
                    got_id, res = self.resp_q.get(timeout=min(0.05, max(0.005, deadline - time.perf_counter())))
                    if got_id == rid:
                        gwt_out = res
                        break
                except queue.Empty:
                    continue
        return self.finish_step_observation(ctx_step, gwt_out)


def step_observation_pair(
    drv_a: LocalCpuBrainAgentDriver,
    obs_a: Dict[str, Any],
    drv_b: LocalCpuBrainAgentDriver,
    obs_b: Dict[str, Any],
    dt: float,
    cog_stride: int = 4,
) -> Tuple[Tuple[np.ndarray, int, bool, np.ndarray], Tuple[np.ndarray, int, bool, np.ndarray]]:
    """
    Dispatch BOTH Team A and Team B GPU `infer_gwt` requests concurrently BEFORE waiting on `resp_q`.
    When both drivers are in pure-inference Spectate mode (`not trainable`), cortical `infer_gwt` runs
    asynchronously (pipelined over 1 physics tick, matching the 7-process GWT architecture!) so Unity's
    60Hz main thread NEVER stalls on inter-process GPU RPCs!
    """
    async_spectate = (not drv_a.trainable) and (not drv_b.trainable)
    ctx_a = drv_a.begin_step_observation(obs_a, dt, cog_stride=cog_stride, async_gwt=async_spectate)
    ctx_b = drv_b.begin_step_observation(obs_b, dt, cog_stride=cog_stride, async_gwt=async_spectate)
    rid_a = ctx_a["rid_gwt"]
    rid_b = ctx_b["rid_gwt"]
    out_a: Optional[Dict[str, Any]] = None
    out_b: Optional[Dict[str, Any]] = None

    # 1. Drain any already-completed async GWT packets sitting in resp_q (zero blocking!)
    if async_spectate:
        while True:
            try:
                got_id, res = drv_a.resp_q.get_nowait()
            except queue.Empty:
                break
            if rid_a is not None and got_id == rid_a:
                out_a = res
                rid_a = None
            elif rid_b is not None and got_id == rid_b:
                out_b = res
                rid_b = None
            elif drv_a._inflight_rid is not None and got_id == drv_a._inflight_rid:
                drv_a._inflight_rid = None
                if drv_a._inflight_ep == int(obs_a["episode_id"]):
                    out_a = res
            elif drv_b._inflight_rid is not None and got_id == drv_b._inflight_rid:
                drv_b._inflight_rid = None
                if drv_b._inflight_ep == int(obs_b["episode_id"]):
                    out_b = res

    # 2. Synchronous wait when required (all Training ticks + tick 0 bootstrap of Spectate)
    needed: Dict[int, Optional[Dict[str, Any]]] = {}
    if rid_a is not None:
        needed[rid_a] = None
    if rid_b is not None:
        needed[rid_b] = None

    if needed:
        deadline = time.perf_counter() + 3.0
        remaining = set(needed.keys())
        while remaining and time.perf_counter() < deadline:
            if drv_a.stop_ev is not None and drv_a.stop_ev.is_set():
                break
            try:
                got_id, res = drv_a.resp_q.get(timeout=min(0.05, max(0.004, deadline - time.perf_counter())))
                if got_id in remaining:
                    needed[got_id] = res
                    remaining.remove(got_id)
                elif drv_a._inflight_rid is not None and got_id == drv_a._inflight_rid:
                    drv_a._inflight_rid = None
                    if drv_a._inflight_ep == int(obs_a["episode_id"]):
                        out_a = res
                elif drv_b._inflight_rid is not None and got_id == drv_b._inflight_rid:
                    drv_b._inflight_rid = None
                    if drv_b._inflight_ep == int(obs_b["episode_id"]):
                        out_b = res
            except queue.Empty:
                continue
        if rid_a is not None:
            out_a = needed.get(rid_a)
        if rid_b is not None:
            out_b = needed.get(rid_b)

    act_a = drv_a.finish_step_observation(ctx_a, out_a)
    act_b = drv_b.finish_step_observation(ctx_b, out_b)
    return act_a, act_b


# ==============================================================================
# 6. Parallel Arena Worker Process (`builtin_3d` OR Simultaneous `unity_tcp`)
# ==============================================================================
def _pump_inter_gen_unity_sockets(
    srv_a: Optional[socket.socket],
    srv_b: Optional[socket.socket],
    conn_a: Optional[socket.socket],
    conn_b: Optional[socket.socket],
    cfg: Brain3DConfig,
    matchmaking: str,
) -> Tuple[Optional[socket.socket], Optional[socket.socket]]:
    """Keep Unity TCP lockstep alive with HOLD_INTER_GEN_WAIT while waiting between PBT generations."""
    watch = []
    if srv_a is not None:
        watch.append(srv_a)
    if conn_a is not None:
        watch.append(conn_a)
    if matchmaking == "self_play":
        if srv_b is not None:
            watch.append(srv_b)
        if conn_b is not None:
            watch.append(conn_b)
    if not watch:
        time.sleep(0.02)
        return conn_a, conn_b

    readable, _, _ = select.select(watch, [], [], 0.02)
    idle_gaze = np.array([0.0, -0.15, 6.0], dtype=np.float32)
    zero_pose = np.zeros(CONT_POSE_DIM, dtype=np.float32)
    zero_telem = np.zeros(6, dtype=np.float32)

    for sock in readable:
        if sock is srv_a:
            if conn_a is not None:
                try:
                    conn_a.close()
                except Exception:
                    pass
                conn_a = None
            try:
                conn_a, _ = srv_a.accept()
                conn_a.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn_a.settimeout(5.0)
            except OSError:
                conn_a = None
        elif sock is srv_b:
            if conn_b is not None:
                try:
                    conn_b.close()
                except Exception:
                    pass
                conn_b = None
            try:
                conn_b, _ = srv_b.accept()
                conn_b.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn_b.settimeout(5.0)
            except OSError:
                conn_b = None
        elif sock is conn_a or sock is conn_b:
            pkt = _recv_ub3d_packet(sock)
            if pkt is None:
                try:
                    sock.close()
                except Exception:
                    pass
                if sock is conn_a:
                    conn_a = None
                else:
                    conn_b = None
                continue
            fr = unpack_ub3d_sensory_frame(pkt)
            if fr is not None:
                try:
                    sock.sendall(pack_ubmc_motor_command(
                        cmd_seq=0,
                        episode_id=fr["episode_id"],
                        action_epoch=fr["action_epoch"],
                        motor_verb=VERB_IDLE_OBSERVE,
                        commit_gate_open=False,
                        veto_active=False,
                        hold_reason_id=HOLD_INTER_GEN_WAIT,
                        target_pose=zero_pose,
                        gaze_ypd=idle_gaze,
                        telemetry6=zero_telem,
                        time_scale=cfg.time_scale,
                    ))
                except OSError:
                    try:
                        sock.close()
                    except Exception:
                        pass
                    if sock is conn_a:
                        conn_a = None
                    else:
                        conn_b = None
    return conn_a, conn_b


def run_arena_slot_worker_process(
    slot_id: int,
    base_cfg: Brain3DConfig,
    env_backend: str,
    matchmaking: str,
    unity_port_a: int,
    unity_port_b: int,
    req_q: mp.Queue,
    resp_q: mp.Queue,
    gen_cmd_q: mp.Queue,
    telem_q: mp.Queue,
    stop_ev,
    ready_ev,
    ux_telemetry_port: int = 0,
) -> None:
    """
    Zero-GPU (`Barrier 1` verified) parallel worker driving 1 Arena Slot:
      - In `pbt_solo`: runs `StandaloneTactical3DArena` (or single `unity_tcp` port) for Learner `slot_id`.
      - In `self_play`: runs `HeadToHeadTactical3DArena` (or dual `unity_tcp` ports `port_a, port_b`)
        pitting Learner `slot_id` against its assigned `opp_brain_id`!
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation in ArenaSlotWorker!")
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()

    cfg = Brain3DConfig(**asdict(base_cfg))
    cfg.seed = int(base_cfg.seed) + slot_id * 101
    dt = 1.0 / 60.0

    srv_a: Optional[socket.socket] = None
    srv_b: Optional[socket.socket] = None
    srv_ux: Optional[socket.socket] = None
    conn_a: Optional[socket.socket] = None
    conn_b: Optional[socket.socket] = None
    conn_ux: Optional[socket.socket] = None
    ux_seq = 0
    last_ux_push_ts = 0.0

    if env_backend == "unity_tcp":
        srv_a = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_a.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_a.bind((cfg.unity_host, int(unity_port_a)))
        srv_a.listen(1)
        srv_a.settimeout(0.25)
        if matchmaking == "self_play" and unity_port_b > 0:
            srv_b = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv_b.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv_b.bind((cfg.unity_host, int(unity_port_b)))
            srv_b.listen(1)
            srv_b.settimeout(0.25)
        if ux_telemetry_port > 0:
            try:
                srv_ux = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv_ux.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv_ux.bind((cfg.unity_host, int(ux_telemetry_port)))
                srv_ux.listen(1)
                srv_ux.setblocking(False)
            except OSError:
                srv_ux = None

    solo_arena = StandaloneTactical3DArena(cfg) if (env_backend != "unity_tcp" and matchmaking != "self_play") else None
    h2h_arena = HeadToHeadTactical3DArena(cfg, seed=cfg.seed) if (env_backend != "unity_tcp" and matchmaking == "self_play") else None

    learner_driver = LocalCpuBrainAgentDriver(
        cfg, brain_id=slot_id, worker_id=slot_id, req_q=req_q, resp_q=resp_q,
        trainable=True, agent_state_id=slot_id * 2, stop_ev=stop_ev,
    )
    opp_cfg = Brain3DConfig(**asdict(cfg))
    opp_cfg.enable_strategy_kb = False  # Keep League sparring opponent as a fixed measurement benchmark!
    opp_driver = LocalCpuBrainAgentDriver(
        opp_cfg, brain_id=slot_id, worker_id=slot_id, req_q=req_q, resp_q=resp_q,
        trainable=False, agent_state_id=slot_id * 2 + 1, stop_ev=stop_ev,
    )
    ready_ev.set()

    try:
        while not stop_ev.is_set():
            if not is_windows_pid_alive(orch_ppid):
                stop_ev.set()
                break
            g_cmd = None
            if env_backend == "unity_tcp":
                try:
                    g_cmd = gen_cmd_q.get_nowait()
                except queue.Empty:
                    conn_a, conn_b = _pump_inter_gen_unity_sockets(
                        srv_a, srv_b, conn_a, conn_b, cfg, matchmaking
                    )
                    continue
            else:
                try:
                    g_cmd = gen_cmd_q.get(timeout=0.15)
                except queue.Empty:
                    continue
            if g_cmd.get("cmd") == "stop":
                break

            gen_idx = int(g_cmd["gen_idx"])
            episodes_target = max(1, int(g_cmd.get("episodes_per_gen", 2)))
            opp_brain_id = int(g_cmd.get("opp_brain_id", slot_id))
            opp_id_label = str(g_cmd.get("opp_label", f"slot_{opp_brain_id}"))
            is_spectate = bool(g_cmd.get("spectate_endless", False))
            will_sleep_after = bool(g_cmd.get("sleep_after_ep", False))
            apply_checkpoint_genes_to_cfg(learner_driver.cfg, g_cmd.get("genes"))
            eff_l_train = bool(g_cmd.get("trainable", False)) if is_spectate else bool(g_cmd.get("trainable", True))
            eff_l_pol = "pure_neural" if is_spectate else str(g_cmd.get("policy_mode", cfg.policy_mode))
            learner_driver.rebind_brain(slot_id, eff_l_pol, eff_l_train)

            eff_opp_pol = "pure_neural" if is_spectate else str(g_cmd.get("opp_policy_mode", cfg.policy_mode))
            opp_driver.rebind_brain(opp_brain_id, eff_opp_pol, False)

            ep_scores: List[float] = []
            wins = 0
            losses = 0
            draws = 0

            if env_backend == "builtin_3d":
                for _ in range(episodes_target):
                    if stop_ev.is_set():
                        break
                    if matchmaking == "self_play" and h2h_arena is not None:
                        h2h_arena.reset_episode()
                        act_a = (np.zeros(CONT_POSE_DIM, dtype=np.float32), VERB_APPROACH_HOVER, False, np.array([0.0, -0.15, 6.0], dtype=np.float32))
                        act_b = (np.zeros(CONT_POSE_DIM, dtype=np.float32), VERB_APPROACH_HOVER, False, np.array([0.0, -0.15, 6.0], dtype=np.float32))
                        while not stop_ev.is_set():
                            obs_a, obs_b, done, winner = h2h_arena.step_both(act_a, act_b, dt)
                            act_a, act_b = step_observation_pair(learner_driver, obs_a, opp_driver, obs_b, dt, cog_stride=4)
                            if done:
                                sc_a = float(obs_a["cumulative_score"])
                                ep_scores.append(sc_a)
                                if winner > 0:
                                    wins += 1
                                elif winner < 0:
                                    losses += 1
                                else:
                                    draws += 1
                                if is_spectate:
                                    if will_sleep_after and learner_driver.strategy_ctrl is not None:
                                        from strategy_knowledge_engine import set_sleep_consolidation_active

                                        set_sleep_consolidation_active(learner_driver.strategy_ctrl.db.db_path, True, max_hold_s=15.0)
                                    telem_q.put({
                                        "evt": "spectate_match",
                                        "match": len(ep_scores),
                                        "score_a": round(sc_a, 2),
                                        "wins_a": int(wins),
                                        "wins_b": int(losses),
                                        "draws": int(draws),
                                    })
                                break
                    elif solo_arena is not None:
                        solo_arena.reset_episode()
                        act_a = (np.zeros(CONT_POSE_DIM, dtype=np.float32), VERB_APPROACH_HOVER, False, np.array([0.0, -0.15, 6.0], dtype=np.float32))
                        while not stop_ev.is_set():
                            obs_a = solo_arena.step(*act_a, dt)
                            act_a = learner_driver.step_observation(obs_a, dt, cog_stride=4)
                            if obs_a["episode_done"]:
                                sc = float(obs_a["cumulative_score"])
                                ep_scores.append(sc)
                                if sc > 2.0:
                                    wins += 1
                                break
            else:
                # Unity TCP Lockstep Arena Loop (order-independent select pump: zero reconnect livelock!)
                eps_done = 0
                cmd_seq = 0
                wmin = np.array(cfg.world_min, dtype=np.float32)
                wmax = np.array(cfg.world_max, dtype=np.float32)
                cum_a = 0.0
                last_ep_a = -1

                def _unpack_obs(fr: Dict[str, Any], cum_score: float) -> Dict[str, Any]:
                    return {
                        "episode_id": fr["episode_id"],
                        "action_epoch": fr["action_epoch"],
                        "stage_id": int(fr.get("stage_id", 0)) or int(getattr(cfg, "default_stage_id", 1)),
                        "physics_tick": fr["frame_seq"],
                        "can_commit_now": 1 if (fr["flags"] & 1) else 0,
                        "episode_done": 1 if (fr["flags"] & 2) else 0,
                        "respawn_flag": fr["respawn_flag"],
                        "visible_count": fr["visible_count"],
                        "cumulative_score": cum_score,
                        "last_step_reward": fr["step_reward"],
                        "hazard_level": fr["hazard_level"],
                        "self_pos": fr["self_pos"],
                        "self_vel": fr["self_vel"],
                        "self_yaw_pitch": fr["self_yaw_pitch"],
                        "proprio": fr["proprio"],
                        "visible_entities": fr["visible_entities"],
                        "fovea_patch": fr["fovea_patch"],
                    }

                def _record_ep_done_a(fr_a: Dict[str, Any], obs_a: Dict[str, Any]) -> None:
                    nonlocal eps_done, wins, losses, draws
                    if not obs_a["episode_done"]:
                        return
                    eps_done += 1
                    ep_scores.append(cum_a)
                    if matchmaking == "self_play":
                        term_rew = float(fr_a["step_reward"])
                        if term_rew > 5.0:
                            wins += 1
                        elif term_rew < -5.0:
                            losses += 1
                        elif float(obs_a["proprio"][PROPRIO_IDX_OBJ_PROGRESS]) > 0.52:
                            wins += 1
                        elif float(obs_a["proprio"][PROPRIO_IDX_OBJ_PROGRESS]) < 0.48:
                            losses += 1
                        else:
                            draws += 1
                    else:
                        if float(obs_a["proprio"][PROPRIO_IDX_OBJ_PROGRESS]) > 0.52 or cum_a > 2.0:
                            wins += 1
                        elif float(obs_a["proprio"][PROPRIO_IDX_OBJ_PROGRESS]) < 0.48:
                            losses += 1
                        else:
                            draws += 1
                    if is_spectate:
                        if will_sleep_after and learner_driver.strategy_ctrl is not None:
                            from strategy_knowledge_engine import set_sleep_consolidation_active

                            set_sleep_consolidation_active(learner_driver.strategy_ctrl.db.db_path, True, max_hold_s=15.0)
                        telem_q.put({
                            "evt": "spectate_match",
                            "match": int(eps_done),
                            "score_a": round(float(cum_a), 2),
                            "wins_a": int(wins),
                            "wins_b": int(losses),
                            "draws": int(draws),
                        })
                    elif will_sleep_after and eps_done >= episodes_target and learner_driver.strategy_ctrl is not None:
                        from strategy_knowledge_engine import set_sleep_consolidation_active

                        set_sleep_consolidation_active(learner_driver.strategy_ctrl.db.db_path, True, max_hold_s=20.0)

                def _service_conn_a_single() -> None:
                    nonlocal conn_a, cum_a, last_ep_a, cmd_seq
                    if conn_a is None:
                        return
                    pkt_a = _recv_ub3d_packet(conn_a)
                    if pkt_a is None:
                        try:
                            conn_a.close()
                        except Exception:
                            pass
                        conn_a = None
                        return
                    fr_a = unpack_ub3d_sensory_frame(pkt_a)
                    if fr_a is None:
                        return
                    if fr_a["episode_id"] != last_ep_a:
                        cum_a = 0.0
                        last_ep_a = fr_a["episode_id"]
                    cum_a += fr_a["step_reward"]
                    obs_a = _unpack_obs(fr_a, cum_a)
                    tp_a, vb_a, gate_a, gz_a = learner_driver.step_observation(obs_a, dt, cog_stride=4)
                    cmd_seq += 1
                    try:
                        conn_a.sendall(pack_ubmc_motor_command(
                            cmd_seq=cmd_seq, episode_id=fr_a["episode_id"], action_epoch=fr_a["action_epoch"],
                            motor_verb=vb_a, commit_gate_open=gate_a, veto_active=False, hold_reason_id=HOLD_GO_CLEAR,
                            target_pose=tp_a, gaze_ypd=gz_a, telemetry6=np.zeros(6, dtype=np.float32),
                            time_scale=cfg.time_scale,
                        ))
                    except OSError:
                        try:
                            conn_a.close()
                        except Exception:
                            pass
                        conn_a = None
                        return
                    if is_spectate:
                        _maybe_push_spectate_ubnt()
                    _record_ep_done_a(fr_a, obs_a)

                def _service_conn_b() -> None:
                    nonlocal conn_b
                    if conn_b is None:
                        return
                    pkt_b = _recv_ub3d_packet(conn_b)
                    if pkt_b is None:
                        try:
                            conn_b.close()
                        except Exception:
                            pass
                        conn_b = None
                        return
                    fr_b = unpack_ub3d_sensory_frame(pkt_b)
                    if fr_b is None:
                        return
                    obs_b_raw = _unpack_obs(fr_b, 0.0)
                    obs_b_can = canonicalize_team_observation(obs_b_raw, -1, wmin, wmax)
                    tp_bc, vb_b, gate_b, gz_bc = opp_driver.step_observation(obs_b_can, dt, cog_stride=4)
                    tp_bw, gz_bw = uncanonicalize_team_motor(tp_bc, gz_bc, -1)
                    try:
                        conn_b.sendall(pack_ubmc_motor_command(
                            cmd_seq=cmd_seq, episode_id=fr_b["episode_id"], action_epoch=fr_b["action_epoch"],
                            motor_verb=vb_b, commit_gate_open=gate_b, veto_active=False, hold_reason_id=HOLD_GO_CLEAR,
                            target_pose=tp_bw, gaze_ypd=gz_bw, telemetry6=np.zeros(6, dtype=np.float32),
                            time_scale=cfg.time_scale,
                        ))
                    except OSError:
                        try:
                            conn_b.close()
                        except Exception:
                            pass
                        conn_b = None

                def _maybe_push_spectate_ubnt() -> None:
                    nonlocal conn_ux, ux_seq, last_ux_push_ts
                    if srv_ux is None:
                        return
                    if conn_ux is None:
                        try:
                            c_ux, _ = srv_ux.accept()
                            c_ux.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            c_ux.setblocking(False)
                            conn_ux = c_ux
                        except OSError:
                            pass
                    if conn_ux is None:
                        return
                    now_u = time.perf_counter()
                    if (now_u - last_ux_push_ts) < 0.055:
                        return
                    ubnt_pkt = learner_driver.build_ubnt_packet(ux_seq + 1)
                    if ubnt_pkt is None:
                        return
                    ubwm_pkt = learner_driver.build_ubwm_packet()
                    payload = ubnt_pkt + ubwm_pkt if ubwm_pkt is not None else ubnt_pkt
                    sent_ok, fatal_d = _sendall_nonblocking_safe(conn_ux, payload, timeout_s=0.06)
                    if sent_ok:
                        ux_seq += 1
                        last_ux_push_ts = now_u
                    elif fatal_d:
                        try:
                            conn_ux.close()
                        except Exception:
                            pass
                        conn_ux = None

                while (is_spectate or eps_done < episodes_target) and not stop_ev.is_set():
                    watch = []
                    if srv_a is not None:
                        watch.append(srv_a)
                    if conn_a is not None:
                        watch.append(conn_a)
                    if matchmaking == "self_play":
                        if srv_b is not None:
                            watch.append(srv_b)
                        if conn_b is not None:
                            watch.append(conn_b)
                    if not watch:
                        time.sleep(0.02)
                        continue

                    readable, _, _ = select.select(watch, [], [], 0.25)
                    serviced_b = False

                    # 1. Accept incoming sockets first
                    if srv_a is not None and srv_a in readable:
                        if conn_a is not None:
                            try:
                                conn_a.close()
                            except Exception:
                                pass
                            conn_a = None
                        try:
                            conn_a, _ = srv_a.accept()
                            conn_a.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            conn_a.settimeout(5.0)
                        except OSError:
                            conn_a = None
                    if srv_b is not None and srv_b in readable:
                        if conn_b is not None:
                            try:
                                conn_b.close()
                            except Exception:
                                pass
                            conn_b = None
                        try:
                            conn_b, _ = srv_b.accept()
                            conn_b.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            conn_b.settimeout(5.0)
                        except OSError:
                            conn_b = None

                    # 2. Fast-path: Paired lockstep GPU batching when both Team A & Team B sockets are active!
                    a_triggered = conn_a is not None and conn_a in readable
                    b_triggered = conn_b is not None and conn_b in readable
                    if matchmaking == "self_play" and conn_a is not None and conn_b is not None and (a_triggered or b_triggered):
                        both_ready = a_triggered and b_triggered
                        if not both_ready:
                            other_sock = conn_b if a_triggered else conn_a
                            r_other, _, _ = select.select([other_sock], [], [], 0.02)
                            both_ready = bool(r_other)

                        if both_ready:
                            pkt_a = _recv_ub3d_packet(conn_a)
                            pkt_b = _recv_ub3d_packet(conn_b)
                            if pkt_a is None:
                                try:
                                    conn_a.close()
                                except Exception:
                                    pass
                                conn_a = None
                            if pkt_b is None:
                                try:
                                    conn_b.close()
                                except Exception:
                                    pass
                                conn_b = None

                            fr_a = unpack_ub3d_sensory_frame(pkt_a) if pkt_a is not None else None
                            fr_b = unpack_ub3d_sensory_frame(pkt_b) if pkt_b is not None else None

                            if fr_a is not None and fr_b is not None:
                                if fr_a["episode_id"] != last_ep_a:
                                    cum_a = 0.0
                                    last_ep_a = fr_a["episode_id"]
                                cum_a += fr_a["step_reward"]
                                obs_a = _unpack_obs(fr_a, cum_a)
                                obs_b_can = canonicalize_team_observation(_unpack_obs(fr_b, 0.0), -1, wmin, wmax)

                                (tp_a, vb_a, gate_a, gz_a), (tp_bc, vb_b, gate_b, gz_bc) = step_observation_pair(
                                    learner_driver, obs_a, opp_driver, obs_b_can, dt, cog_stride=4
                                )
                                tp_bw, gz_bw = uncanonicalize_team_motor(tp_bc, gz_bc, -1)
                                cmd_seq += 1

                                if conn_a is not None:
                                    try:
                                        conn_a.sendall(pack_ubmc_motor_command(
                                            cmd_seq=cmd_seq, episode_id=fr_a["episode_id"], action_epoch=fr_a["action_epoch"],
                                            motor_verb=vb_a, commit_gate_open=gate_a,
                                            veto_active=learner_driver.latest_veto, hold_reason_id=learner_driver.latest_hold_reason,
                                            target_pose=tp_a, gaze_ypd=gz_a, telemetry6=np.zeros(6, dtype=np.float32),
                                            time_scale=cfg.time_scale,
                                        ))
                                    except OSError:
                                        try:
                                            conn_a.close()
                                        except Exception:
                                            pass
                                        conn_a = None

                                if conn_b is not None:
                                    try:
                                        conn_b.sendall(pack_ubmc_motor_command(
                                            cmd_seq=cmd_seq, episode_id=fr_b["episode_id"], action_epoch=fr_b["action_epoch"],
                                            motor_verb=vb_b, commit_gate_open=gate_b, veto_active=False, hold_reason_id=HOLD_GO_CLEAR,
                                            target_pose=tp_bw, gaze_ypd=gz_bw, telemetry6=np.zeros(6, dtype=np.float32),
                                            time_scale=cfg.time_scale,
                                        ))
                                    except OSError:
                                        try:
                                            conn_b.close()
                                        except Exception:
                                            pass
                                        conn_b = None
                                serviced_b = True
                                if is_spectate:
                                    _maybe_push_spectate_ubnt()
                                _record_ep_done_a(fr_a, obs_a)
                            elif fr_a is not None and conn_a is not None:
                                if fr_a["episode_id"] != last_ep_a:
                                    cum_a = 0.0
                                    last_ep_a = fr_a["episode_id"]
                                cum_a += fr_a["step_reward"]
                                obs_a = _unpack_obs(fr_a, cum_a)
                                tp_a, vb_a, gate_a, gz_a = learner_driver.step_observation(obs_a, dt, cog_stride=4)
                                cmd_seq += 1
                                try:
                                    conn_a.sendall(pack_ubmc_motor_command(
                                        cmd_seq=cmd_seq, episode_id=fr_a["episode_id"], action_epoch=fr_a["action_epoch"],
                                        motor_verb=vb_a, commit_gate_open=gate_a, veto_active=False, hold_reason_id=HOLD_GO_CLEAR,
                                        target_pose=tp_a, gaze_ypd=gz_a, telemetry6=np.zeros(6, dtype=np.float32),
                                        time_scale=cfg.time_scale,
                                    ))
                                except OSError:
                                    try:
                                        conn_a.close()
                                    except Exception:
                                        pass
                                    conn_a = None
                                if is_spectate:
                                    _maybe_push_spectate_ubnt()
                                _record_ep_done_a(fr_a, obs_a)
                            elif fr_b is not None and conn_b is not None:
                                obs_b_can = canonicalize_team_observation(_unpack_obs(fr_b, 0.0), -1, wmin, wmax)
                                tp_bc, vb_b, gate_b, gz_bc = opp_driver.step_observation(obs_b_can, dt, cog_stride=4)
                                tp_bw, gz_bw = uncanonicalize_team_motor(tp_bc, gz_bc, -1)
                                try:
                                    conn_b.sendall(pack_ubmc_motor_command(
                                        cmd_seq=cmd_seq, episode_id=fr_b["episode_id"], action_epoch=fr_b["action_epoch"],
                                        motor_verb=vb_b, commit_gate_open=gate_b, veto_active=False, hold_reason_id=HOLD_GO_CLEAR,
                                        target_pose=tp_bw, gaze_ypd=gz_bw, telemetry6=np.zeros(6, dtype=np.float32),
                                        time_scale=cfg.time_scale,
                                    ))
                                except OSError:
                                    try:
                                        conn_b.close()
                                    except Exception:
                                        pass
                                    conn_b = None
                                serviced_b = True
                        else:
                            if a_triggered:
                                _service_conn_a_single()
                            if b_triggered:
                                _service_conn_b()
                                serviced_b = True
                    else:
                        if a_triggered:
                            _service_conn_a_single()
                        if b_triggered:
                            _service_conn_b()
                            serviced_b = True

                    if (
                        (not is_spectate)
                        and eps_done >= episodes_target
                        and matchmaking == "self_play"
                        and conn_b is not None
                        and not serviced_b
                    ):
                        rb, _, _ = select.select([conn_b], [], [], 0.03)
                        if rb:
                            _service_conn_b()

            if stop_ev.is_set():
                break

            # Generation-tagged completion packet (prevents stale packet race across generations!)
            telem_q.put({
                "gen_idx": gen_idx,
                "slot_id": slot_id,
                "opp_label": opp_id_label,
                "mean_score": float(np.mean(ep_scores)) if ep_scores else 0.0,
                "wins": int(wins),
                "losses": int(losses),
                "draws": int(draws),
                "episodes": len(ep_scores),
                "transitions_sent": int(learner_driver.transitions_sent),
            })
    finally:
        learner_driver.close()
        opp_driver.close()
        for q_obj in (req_q, resp_q, gen_cmd_q, telem_q):
            try:
                q_obj.cancel_join_thread()
            except Exception:
                pass
        for c in (conn_a, conn_b, conn_ux, srv_a, srv_b, srv_ux):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass


# ==============================================================================
# 7. Top-Level PBT + Self-Play League Coordinator & `verify_pbt` Suite
# ==============================================================================
def _emit_pbt_stat(payload: Dict[str, Any]) -> None:
    print("[PBT_STAT] " + json.dumps(payload, ensure_ascii=True, separators=(",", ":")), flush=True)


def run_arena_pbt_coordinator(
    base_cfg: Brain3DConfig,
    *,
    env_backend: str = "builtin_3d",
    matchmaking: str = "self_play",
    pop_size: int = 4,
    generations: int = 4,
    episodes_per_gen: int = 2,
    base_port: int = 50055,
    checkpoint_dir: str = "",
    init_weights_path: str = "",
    parent_pid: int = 0,
    enable_sleep: bool = True,
    sleep_epochs: int = 6,
    scaffold_anneal_end: Optional[float] = None,
    watch_stdin: bool = False,
) -> Dict[str, Any]:
    """
    Coordinate `pop_size` simultaneous arenas + 1 shared GPU inference/learning server +
    PBT Exploit/Explore + Competitive Elo Self-Play League.
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: PBT Coordinator must NEVER import torch!")

    pop_n = max(2, int(pop_size))
    max_opp_slots = pop_n
    ctx = mp.get_context("spawn")
    stop_ev = ctx.Event()
    abort_save_ev = ctx.Event()
    start_parent_and_stdin_watchdog(
        parent_pid,
        lambda: stop_ev.set(),
        watch_stdin=watch_stdin,
        on_emergency_abort=lambda: abort_save_ev.set(),
    )

    req_q = ctx.Queue(maxsize=256)
    resp_queues = {i: ctx.Queue(maxsize=32) for i in range(pop_n)}
    ctrl_q = ctx.Queue(maxsize=64)
    ctrl_resp_q = ctx.Queue(maxsize=64)
    telem_q = ctx.Queue(maxsize=128)
    gen_cmd_queues = [ctx.Queue(maxsize=8) for _ in range(pop_n)]

    def _gpu_ctrl(cmd: str, payload: Dict[str, Any], timeout: float = 15.0) -> Dict[str, Any]:
        ctrl_q.put((cmd, payload))
        return ctrl_resp_q.get(timeout=timeout)

    gpu_ready = ctx.Event()
    gpu_proc = ctx.Process(
        target=run_population_gpu_server_process,
        args=(base_cfg, pop_n, max_opp_slots, req_q, resp_queues, ctrl_q, ctrl_resp_q, stop_ev, gpu_ready, abort_save_ev),
        name="PopGPUServer",
    )

    workers: List[Tuple[int, Any, Any]] = []
    league = OpponentLeaguePool(max_snapshots=8, seed=base_cfg.seed)
    slot_elos = [1200.0 for _ in range(pop_n)]
    slot_genes = [dict(PBT_ARCHETYPES[i % len(PBT_ARCHETYPES)]) for i in range(pop_n)]
    rng = np.random.default_rng(base_cfg.seed + 999)
    history: List[Dict[str, Any]] = []
    best_champ_elo = -1e9

    try:
        gpu_proc.start()
        if not gpu_ready.wait(timeout=30.0):
            raise RuntimeError("PopGPUServer failed to initialize within 30s!")

        barrier_stat = _gpu_ctrl("check_barrier", {})

        resumed_from = ""
        if init_weights_path and os.path.isfile(init_weights_path):
            for i in range(pop_n):
                res_ld = _gpu_ctrl("load_slot_file", {"slot_id": i, "path": init_weights_path})
                meta_ld = res_ld.get("meta") if isinstance(res_ld.get("meta"), dict) else {}
                slot_elos[i] = float(meta_ld.get("elo", 1200.0))
            resumed_from = os.path.basename(init_weights_path)
            print(f"[PBT] Resumed all {pop_n} learner slots from {init_weights_path} (base_elo={slot_elos[0]:.1f})", flush=True)

        for i in range(pop_n):
            w_ready = ctx.Event()
            if matchmaking == "self_play" and env_backend == "unity_tcp":
                pa, pb = base_port + 2 * i, base_port + 2 * i + 1
            else:
                pa, pb = base_port + i, 0
            wp = ctx.Process(
                target=run_arena_slot_worker_process,
                args=(i, base_cfg, env_backend, matchmaking, pa, pb, req_q, resp_queues[i], gen_cmd_queues[i], telem_q, stop_ev, w_ready),
                name=f"ArenaWorker_{i}",
            )
            wp.start()
            workers.append((i, wp, w_ready))

        for i, wp, w_ready in workers:
            if not w_ready.wait(timeout=20.0):
                raise RuntimeError(f"ArenaWorker_{i} failed to become ready!")

        print(
            f"[PBT] Ready: pop={pop_n} mode={matchmaking} env={env_backend} base_port={base_port}",
            flush=True,
        )

        # Seed initial Hall-of-Fame snapshot from Slot 0
        init_st = _gpu_ctrl("get_slot_state", {"slot_id": 0})
        league.add_champion("gen0_seed", 0, slot_genes[0]["archetype"], slot_elos[0], slot_genes[0], init_st["state"])

        for g in range(1, int(generations) + 1):
            if stop_ev.is_set():
                break

            # Drain any stale telemetry from prior generations BEFORE dispatching generation `g`
            while True:
                try:
                    telem_q.get_nowait()
                except queue.Empty:
                    break

            # Schedule Policy Mode (early DAgger -> Pure Neural autonomy) & optional Scaffold Annealing (lambda_scaf: start -> end)
            gen_policy = "dagger" if (base_cfg.policy_mode == "dagger" and g <= max(1, generations // 2)) else (
                "pure_neural" if base_cfg.policy_mode == "dagger" else base_cfg.policy_mode
            )
            scaf_start = float(np.clip(getattr(base_cfg, "scaffold_prior_scale", 1.0), 0.0, 1.0))
            if scaffold_anneal_end is not None and int(generations) > 1:
                frac_g = float(g - 1) / float(max(1, int(generations) - 1))
                gen_scaf_scale = float(np.clip(scaf_start + (float(scaffold_anneal_end) - scaf_start) * frac_g, 0.0, 1.0))
            elif scaffold_anneal_end is not None:
                gen_scaf_scale = float(np.clip(float(scaffold_anneal_end), 0.0, 1.0))
            else:
                gen_scaf_scale = scaf_start
            for sg in slot_genes:
                sg["scaffold_prior_scale"] = gen_scaf_scale

            opp_meta_for_slot: Dict[int, Tuple[str, float, Optional[LeagueSnapshotEntry], int]] = {}
            for i in range(pop_n):
                opp_shadow_slot = pop_n + i
                use_hof = (matchmaking == "self_play") and (len(league.snapshots) > 0) and (rng.random() < 0.40 or pop_n == 1)
                if use_hof:
                    hof_entry = league.sample_pfsp_opponent(slot_elos[i])
                    assert hof_entry is not None
                    _gpu_ctrl("set_slot_state", {"slot_id": opp_shadow_slot, "state": hof_entry.weights_state, "genes": hof_entry.genes})
                    opp_meta_for_slot[i] = ("hof", hof_entry.elo, hof_entry, -1)
                    opp_bid = opp_shadow_slot
                    opp_lbl = hof_entry.snapshot_id
                else:
                    peer_idx = (i + 1 + int(rng.integers(0, max(1, pop_n - 1)))) % pop_n
                    peer_st = _gpu_ctrl("get_slot_state", {"slot_id": peer_idx})
                    _gpu_ctrl("set_slot_state", {"slot_id": opp_shadow_slot, "state": peer_st["state"], "genes": slot_genes[peer_idx]})
                    opp_meta_for_slot[i] = ("peer", slot_elos[peer_idx], None, peer_idx)
                    opp_bid = opp_shadow_slot
                    opp_lbl = f"peer_slot_{peer_idx}"

                opp_pol = "dagger" if (g == 1 and not resumed_from and gen_policy == "dagger") else "pure_neural"
                gen_cmd_queues[i].put({
                    "cmd": "run_gen",
                    "gen_idx": g,
                    "episodes_per_gen": episodes_per_gen,
                    "opp_brain_id": opp_bid,
                    "opp_label": opp_lbl,
                    "genes": slot_genes[i],
                    "policy_mode": gen_policy,
                    "opp_policy_mode": opp_pol,
                    "sleep_after_ep": bool(enable_sleep and int(sleep_epochs) > 0),
                })

            _emit_pbt_stat({
                "evt": "gen_start",
                "gen": g,
                "total_gens": int(generations),
                "eps_per_gen": int(episodes_per_gen),
                "done_slots": 0,
                "pop_size": pop_n,
                "policy": gen_policy,
                "league_size": len(league.snapshots),
                "resumed_from": resumed_from,
                "slots": [
                    {
                        "slot": i,
                        "archetype": slot_genes[i]["archetype"],
                        "elo": round(slot_elos[i], 1),
                        "score": 0.0,
                        "w": 0,
                        "l": 0,
                        "d": 0,
                        "opp": opp_meta_for_slot[i][0],
                    }
                    for i in range(pop_n)
                ],
            })

            # Collect strictly `gen_idx == g` results from all `pop_n` workers
            gen_results: Dict[int, Dict[str, Any]] = {}
            deadline = time.perf_counter() + max(540.0, float(episodes_per_gen) * 500.0)
            while len(gen_results) < pop_n and time.perf_counter() < deadline and not stop_ev.is_set():
                try:
                    pkt = telem_q.get(timeout=0.5)
                    if int(pkt.get("gen_idx", -1)) == g:
                        sid = int(pkt["slot_id"])
                        gen_results[sid] = pkt
                        _emit_pbt_stat({
                            "evt": "slot_done",
                            "gen": g,
                            "total_gens": int(generations),
                            "eps_per_gen": int(episodes_per_gen),
                            "done_slots": len(gen_results),
                            "pop_size": pop_n,
                            "champ_slot": sid,
                            "champ_score": round(float(pkt["mean_score"]), 2),
                        })
                except queue.Empty:
                    if any(not wp.is_alive() for _, wp, _ in workers):
                        break

            if stop_ev.is_set():
                break

            # Update Elo ratings & compute composite fitness
            fitness = np.zeros(pop_n, dtype=np.float32)
            for i in range(pop_n):
                r = gen_results.get(i, {"mean_score": 0.0, "wins": 0, "losses": 0, "draws": 1})
                total_g = max(1, int(r["wins"]) + int(r["losses"]) + int(r["draws"]))
                win_score = (float(r["wins"]) + 0.5 * float(r["draws"])) / float(total_g)
                kind, opp_elo, hof_e, peer_idx = opp_meta_for_slot[i]
                if matchmaking == "self_play":
                    new_a, new_b = compute_elo_update(slot_elos[i], opp_elo, win_score, k_factor=24.0)
                    slot_elos[i] = new_a
                    if kind == "hof" and hof_e is not None:
                        hof_e.elo = new_b
                fitness[i] = float(r["mean_score"]) + 0.04 * (slot_elos[i] - 1200.0)

            # Offline Sleep S-1..S-4 across all learners (serialized BEFORE Ollama 12B via sleep gate!)
            eff_sleep_ep = max(0, int(sleep_epochs)) if enable_sleep else 0
            strat_db_for_gate = str(getattr(base_cfg, "strategy_db_path", "") or "").strip() if bool(getattr(base_cfg, "enable_strategy_kb", False)) else ""
            if eff_sleep_ep > 0:
                if strat_db_for_gate:
                    from strategy_knowledge_engine import set_sleep_consolidation_active

                    set_sleep_consolidation_active(strat_db_for_gate, True, max_hold_s=25.0)
                try:
                    _emit_pbt_stat({
                        "evt": "sleep_start",
                        "gen": g,
                        "total_gens": int(generations),
                        "sleep_epochs": eff_sleep_ep,
                    })
                    sleep_stat = _gpu_ctrl("sleep_all_learners", {"epochs": eff_sleep_ep})
                finally:
                    if strat_db_for_gate:
                        from strategy_knowledge_engine import set_sleep_consolidation_active

                        set_sleep_consolidation_active(strat_db_for_gate, False)
            else:
                if strat_db_for_gate:
                    from strategy_knowledge_engine import set_sleep_consolidation_active

                    set_sleep_consolidation_active(strat_db_for_gate, False)
                sleep_stat = {"mean_sleep_loss": 0.0}

            ranked = list(np.argsort(fitness)[::-1])  # Best -> Worst
            champ_slot = int(ranked[0])
            champ_st = _gpu_ctrl("get_slot_state", {"slot_id": champ_slot})
            league.add_champion(
                f"gen{g}_slot{champ_slot}", g, slot_genes[champ_slot]["archetype"],
                slot_elos[champ_slot], slot_genes[champ_slot], champ_st["state"]
            )

            if stop_ev.is_set() or abort_save_ev.is_set():
                break

            if checkpoint_dir and (not stop_ev.is_set()) and (not abort_save_ev.is_set()):
                champ_meta = {
                    "generation": g,
                    "elo": round(slot_elos[champ_slot], 2),
                    "mean_score": float(gen_results.get(champ_slot, {}).get("mean_score", 0.0)),
                    "archetype": slot_genes[champ_slot]["archetype"],
                    "matchmaking": matchmaking,
                }
                champ_path = os.path.join(checkpoint_dir, "pbt_latest_champion.pt")
                _gpu_ctrl("save_slot", {"slot_id": champ_slot, "path": champ_path, "meta": champ_meta})
                if slot_elos[champ_slot] >= best_champ_elo:
                    best_champ_elo = float(slot_elos[champ_slot])
                    best_path = os.path.join(checkpoint_dir, "pbt_best_champion.pt")
                    _gpu_ctrl("save_slot", {"slot_id": champ_slot, "path": best_path, "meta": champ_meta})

            # PBT Exploit & Explore: bottom slot copies top champion weights + perturbs genes
            if g < int(generations) and pop_n >= 2 and (not stop_ev.is_set()) and (not abort_save_ev.is_set()):
                loser_slot = int(ranked[-1])
                mut_genes = dict(slot_genes[champ_slot])
                for gk, lo, hi in (
                    ("lr_scale", 0.35, 2.20),
                    ("curiosity_fog_gain", 0.40, 2.40),
                    ("stn_stealth_gain", 0.45, 2.30),
                    ("policy_temp", 0.18, 0.70),
                    ("survival_veto_threshold", 0.28, 0.58),
                ):
                    mut_genes[gk] = float(np.clip(float(mut_genes[gk]) * float(rng.uniform(0.85, 1.18)), lo, hi))
                slot_genes[loser_slot] = mut_genes
                slot_elos[loser_slot] = 0.5 * (slot_elos[loser_slot] + slot_elos[champ_slot])
                _gpu_ctrl("set_slot_state", {"slot_id": loser_slot, "state": champ_st["state"], "genes": mut_genes})

            gen_rec = {
                "generation": g,
                "champion_slot": champ_slot,
                "champion_archetype": slot_genes[champ_slot]["archetype"],
                "champion_elo": round(slot_elos[champ_slot], 2),
                "champion_score": round(float(gen_results.get(champ_slot, {}).get("mean_score", 0.0)), 3),
                "mean_sleep_loss": round(float(sleep_stat.get("mean_sleep_loss", 0.0)), 5),
                "league_size": len(league.snapshots),
                "slots": [
                    {
                        "slot": i,
                        "archetype": slot_genes[i]["archetype"],
                        "elo": round(slot_elos[i], 1),
                        "score": round(float(gen_results.get(i, {}).get("mean_score", 0.0)), 2),
                        "w": int(gen_results.get(i, {}).get("wins", 0)),
                        "l": int(gen_results.get(i, {}).get("losses", 0)),
                        "d": int(gen_results.get(i, {}).get("draws", 0)),
                        "opp": opp_meta_for_slot[i][0],
                    }
                    for i in range(pop_n)
                ],
            }
            history.append(gen_rec)
            if stop_ev.is_set() or abort_save_ev.is_set():
                break

            if checkpoint_dir and (not stop_ev.is_set()) and (not abort_save_ev.is_set()):
                summary_path = os.path.join(checkpoint_dir, "pbt_training_summary.json")
                with open(summary_path, "w", encoding="utf-8") as fh:
                    json.dump({"generations_completed": len(history), "history": history}, fh, ensure_ascii=False, indent=2)

            _emit_pbt_stat({
                "evt": "gen_done",
                "gen": g,
                "total_gens": int(generations),
                "eps_per_gen": int(episodes_per_gen),
                "done_slots": pop_n,
                "pop_size": pop_n,
                "policy": gen_policy,
                "champ_slot": champ_slot,
                "champ_arch": gen_rec["champion_archetype"],
                "champ_elo": gen_rec["champion_elo"],
                "champ_score": gen_rec["champion_score"],
                "sleep_loss": gen_rec["mean_sleep_loss"],
                "league_size": gen_rec["league_size"],
                "saved_ckpt": bool(checkpoint_dir),
                "slots": gen_rec["slots"],
            })
            print(
                f"[Gen {g:02d}/{generations:02d}] Champ=Slot#{champ_slot}({gen_rec['champion_archetype']}) "
                f"Score={gen_rec['champion_score']:+.2f} Elo={gen_rec['champion_elo']:.1f} "
                f"SleepLoss={gen_rec['mean_sleep_loss']:.4f} League={gen_rec['league_size']}",
                flush=True,
            )

        if checkpoint_dir and history and (not stop_ev.is_set()) and (not abort_save_ev.is_set()):
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            sess_path = os.path.join(checkpoint_dir, f"pbt_session_{ts_str}.pt")
            _gpu_ctrl("save_slot", {
                "slot_id": int(history[-1]["champion_slot"]),
                "path": sess_path,
                "meta": {
                    "generation": int(history[-1]["generation"]),
                    "elo": history[-1]["champion_elo"],
                    "mean_score": history[-1]["champion_score"],
                    "archetype": history[-1]["champion_archetype"],
                    "generations": len(history),
                },
            })

        if history and (not stop_ev.is_set()) and (not abort_save_ev.is_set()):
            _emit_pbt_stat({
                "evt": "train_complete",
                "gen": len(history),
                "total_gens": int(generations),
                "champ_slot": int(history[-1]["champion_slot"]),
                "champ_arch": history[-1]["champion_archetype"],
                "champ_elo": history[-1]["champion_elo"],
                "champ_score": history[-1]["champion_score"],
                "sleep_loss": history[-1]["mean_sleep_loss"],
                "league_size": history[-1]["league_size"],
                "saved_ckpt": bool(checkpoint_dir),
                "slots": history[-1]["slots"],
            })

        return {
            "ok": (len(history) > 0) or bool(stop_ev.is_set()),
            "aborted_unverified_save": bool(abort_save_ev.is_set()),
            "barrier1_parent_zero_torch": ("torch" not in sys.modules),
            "gpu_server_info": barrier_stat,
            "generations_completed": len(history),
            "final_elos": [round(e, 2) for e in slot_elos],
            "league_snapshots": len(league.snapshots),
            "history": history,
        }
    finally:
        stop_ev.set()
        for q in gen_cmd_queues:
            try:
                q.put_nowait({"cmd": "stop"})
            except Exception:
                pass
        for _, wp, _ in workers:
            wp.join(timeout=5.0)
            if wp.is_alive():
                wp.terminate()
        gpu_proc.join(timeout=6.0)
        if gpu_proc.is_alive():
            gpu_proc.terminate()
        league.snapshots.clear()


def run_arena_spectate_coordinator(
    base_cfg: Brain3DConfig,
    *,
    env_backend: str = "unity_tcp",
    base_port: int = 50055,
    ux_port: int = 0,
    load_weights: str = "",
    opp_weights: str = "",
    continual_learning: bool = False,
    continual_save_path: str = "",
    continual_sleep_epochs: int = 2,
    parent_pid: int = 0,
    watch_stdin: bool = False,
) -> Dict[str, Any]:
    """
    Endless AI vs AI Self-Play Spectate coordinator:
      - `continual_learning=False`: 100% pure-inference frozen weights (`trainable=False`).
      - `continual_learning=True`: Agent A (`Slot 0`) learns online from live trajectories, runs
        2-epoch S-1..S-4 mini-sleep consolidation between matches (serialized BEFORE Ollama 12B!),
        autosaves to `continual_save_path`, and refreshes sparring opponent `Slot 1` on wins!
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: Spectate Coordinator must NEVER import torch!")

    ctx = mp.get_context("spawn")
    stop_ev = ctx.Event()
    abort_save_ev = ctx.Event()
    start_parent_and_stdin_watchdog(
        parent_pid,
        lambda: stop_ev.set(),
        watch_stdin=watch_stdin,
        on_emergency_abort=lambda: (abort_save_ev.set(), stop_ev.set()),
    )

    req_q = ctx.Queue(maxsize=64)
    resp_queues = {0: ctx.Queue(maxsize=32)}
    ctrl_q = ctx.Queue(maxsize=32)
    ctrl_resp_q = ctx.Queue(maxsize=32)
    telem_q = ctx.Queue(maxsize=64)
    gen_cmd_q = ctx.Queue(maxsize=8)

    def _gpu_ctrl(cmd: str, payload: Dict[str, Any], timeout: float = 15.0) -> Dict[str, Any]:
        ctrl_q.put((cmd, payload))
        return ctrl_resp_q.get(timeout=timeout)

    gpu_ready = ctx.Event()
    gpu_proc = ctx.Process(
        target=run_population_gpu_server_process,
        args=(base_cfg, 1, 1, req_q, resp_queues, ctrl_q, ctrl_resp_q, stop_ev, gpu_ready, abort_save_ev),
        name="PopGPUServerSpectate",
    )
    eff_ux_port = int(ux_port) if int(ux_port) > 0 else (int(base_port) + 100 if env_backend == "unity_tcp" else 0)
    w_ready = ctx.Event()
    wp = ctx.Process(
        target=run_arena_slot_worker_process,
        args=(0, base_cfg, env_backend, "self_play", base_port, base_port + 1, req_q, resp_queues[0], gen_cmd_q, telem_q, stop_ev, w_ready, eff_ux_port),
        name="ArenaSpectateWorker",
    )

    matches_observed = 0
    prev_wins_a = 0
    strat_db_for_gate = str(getattr(base_cfg, "strategy_db_path", "") or "").strip() if bool(getattr(base_cfg, "enable_strategy_kb", False)) else ""
    try:
        gpu_proc.start()
        if not gpu_ready.wait(timeout=30.0):
            raise RuntimeError("PopGPUServerSpectate failed to initialize within 30s!")
        barrier_stat = _gpu_ctrl("check_barrier", {})

        meta_a: Dict[str, Any] = {}
        meta_b: Dict[str, Any] = {}
        if load_weights and os.path.isfile(load_weights):
            res_a = _gpu_ctrl("load_slot_file", {"slot_id": 0, "path": load_weights})
            meta_a = res_a.get("meta") if isinstance(res_a.get("meta"), dict) else {}
        eff_opp = opp_weights if (opp_weights and os.path.isfile(opp_weights)) else load_weights
        has_distinct_opp = bool(
            opp_weights
            and os.path.isfile(opp_weights)
            and (not (load_weights and os.path.isfile(load_weights)) or os.path.abspath(opp_weights) != os.path.abspath(load_weights))
        )
        # In continual learning mode, always isolate Opponent onto Slot 1 so Learner Slot 0 updates cleanly!
        if (has_distinct_opp or continual_learning) and eff_opp and os.path.isfile(eff_opp):
            res_b = _gpu_ctrl("load_slot_file", {"slot_id": 1, "path": eff_opp})
            meta_b = res_b.get("meta") if isinstance(res_b.get("meta"), dict) else {}
        spectate_opp_brain_id = 1 if (has_distinct_opp or continual_learning) else 0

        wp.start()
        if not w_ready.wait(timeout=20.0):
            raise RuntimeError("ArenaSpectateWorker failed to become ready!")

        eff_sleep_ep = max(0, int(continual_sleep_epochs)) if continual_learning else 0
        print(
            f"[PBT] Ready: pop=1 mode={'spectate_continual_learn' if continual_learning else 'spectate_self_play'} "
            f"env={env_backend} base_port={base_port} batch_shared={int(spectate_opp_brain_id == 0)} "
            f"ckptA={os.path.basename(load_weights) if load_weights else 'init'}(gen={meta_a.get('generation', '-')},elo={meta_a.get('elo', '-')}) "
            f"ckptB={os.path.basename(eff_opp) if eff_opp else 'init'}",
            flush=True,
        )

        init_genes = dict(PBT_ARCHETYPES[0])
        if isinstance(meta_a.get("genes"), dict):
            init_genes.update(meta_a["genes"])

        gen_cmd_q.put({
            "cmd": "run_gen",
            "gen_idx": 1,
            "episodes_per_gen": 1000000000,
            "opp_brain_id": spectate_opp_brain_id,
            "opp_label": "spectate_opp",
            "genes": init_genes,
            "trainable": bool(continual_learning),
            "sleep_after_ep": bool(continual_learning and eff_sleep_ep > 0),
            "spectate_endless": True,
            "policy_mode": "pure_neural",
            "opp_policy_mode": "pure_neural",
        })

        base_elo = float(meta_a.get("elo", 1200.0)) if isinstance(meta_a.get("elo"), (int, float)) else 1200.0
        base_gen = int(meta_a.get("generation", 0)) if isinstance(meta_a.get("generation"), int) else 0

        while not stop_ev.is_set():
            if not wp.is_alive() or not gpu_proc.is_alive():
                break
            try:
                pkt = telem_q.get(timeout=0.5)
                if pkt.get("evt") == "spectate_match":
                    matches_observed = int(pkt.get("match", matches_observed + 1))
                    cur_wins_a = int(pkt.get("wins_a", 0))
                    won_this = cur_wins_a > prev_wins_a
                    prev_wins_a = cur_wins_a
                    slp_loss = 0.0

                    if continual_learning and (not stop_ev.is_set()) and (not abort_save_ev.is_set()):
                        if strat_db_for_gate and eff_sleep_ep > 0:
                            from strategy_knowledge_engine import set_sleep_consolidation_active

                            set_sleep_consolidation_active(strat_db_for_gate, True, max_hold_s=15.0)
                        try:
                            if eff_sleep_ep > 0:
                                slp_res = _gpu_ctrl("sleep_all_learners", {"epochs": eff_sleep_ep})
                                slp_loss = float(slp_res.get("mean_sleep_loss", 0.0))
                            if continual_save_path and (not abort_save_ev.is_set()):
                                save_dir = os.path.dirname(os.path.abspath(continual_save_path))
                                if save_dir:
                                    os.makedirs(save_dir, exist_ok=True)
                                _gpu_ctrl("save_slot", {
                                    "slot_id": 0,
                                    "path": continual_save_path,
                                    "meta": {
                                        "generation": base_gen + matches_observed,
                                        "elo": round(base_elo + 2.0 * (2 * cur_wins_a - matches_observed), 1),
                                        "mean_score": float(pkt.get("score_a", 0.0)),
                                        "archetype": str(init_genes.get("archetype", "ContinualSpectate")),
                                        "matchmaking": "spectate_continual",
                                    },
                                })
                                pkt["saved_ckpt"] = True
                            # When Learner A wins (and no custom fixed opponent file was forced), evolve sparring partner Slot 1!
                            if won_this and (not has_distinct_opp):
                                st0 = _gpu_ctrl("get_slot_state", {"slot_id": 0})
                                _gpu_ctrl("set_slot_state", {"slot_id": 1, "state": st0["state"]})
                        finally:
                            if strat_db_for_gate:
                                from strategy_knowledge_engine import set_sleep_consolidation_active

                                set_sleep_consolidation_active(strat_db_for_gate, False)

                    pkt["sleep_loss"] = round(slp_loss, 5)
                    _emit_pbt_stat(pkt)
                    print(
                        f"[Spectate] Match #{matches_observed} Complete | "
                        f"ScoreA={pkt.get('score_a', 0.0):+.2f} | Record={pkt.get('wins_a', 0)}-{pkt.get('wins_b', 0)}-{pkt.get('draws', 0)} "
                        f"| ContinualLearn={int(continual_learning)} SleepLoss={slp_loss:.4f}",
                        flush=True,
                    )
            except queue.Empty:
                continue

        return {
            "ok": True,
            "barrier1_parent_zero_torch": ("torch" not in sys.modules),
            "gpu_server_info": barrier_stat,
            "matches_observed": matches_observed,
            "continual_learning": bool(continual_learning),
        }
    finally:
        stop_ev.set()
        try:
            gen_cmd_q.put_nowait({"cmd": "stop"})
        except Exception:
            pass
        if wp.is_alive():
            wp.join(timeout=4.0)
            if wp.is_alive():
                wp.terminate()
        if gpu_proc.is_alive():
            gpu_proc.join(timeout=5.0)
            if gpu_proc.is_alive():
                gpu_proc.terminate()


def run_pbt_self_verification() -> bool:
    """
    End-to-end verification of:
      1. Canonical 180-deg Team Symmetry (`canonicalize_team_observation` / `uncanonicalize_team_motor` involution)
      2. 1-GPU Parallel Multi-Arena Competitive Self-Play League + PBT Exploit/Explore + Elo updates
      3. Strict `Barrier 1` (`torch` never imported in Coordinator or Arena Workers) & Zero Residual Files!
    """
    print("=" * 78)
    print("  Arena PBT & Competitive Self-Play League v2.3.0 — Verification Suite")
    print("=" * 78)

    # 1. Verify Team-Perspective 180-degree Canonicalization Involution
    wmin = np.array([-12.0, 0.0, -16.0], dtype=np.float32)
    wmax = np.array([12.0, 6.0, 16.0], dtype=np.float32)
    dummy_obs = {
        "self_pos": np.array([3.5, 1.5, 10.0], dtype=np.float32),
        "self_vel": np.array([-1.2, 0.0, -3.0], dtype=np.float32),
        "self_yaw_pitch": np.array([math.pi, -0.15], dtype=np.float32),
        "proprio": np.linspace(0.1, 0.9, PROPRIO_DIM, dtype=np.float32),
        "visible_count": 1,
        "visible_entities": np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32),
        "fovea_patch": np.ones((FOVEA_H, FOVEA_W, FOVEA_C), dtype=np.float32) * 0.4,
    }
    c1 = canonicalize_team_observation(dummy_obs, -1, wmin, wmax)
    c2 = canonicalize_team_observation(c1, -1, wmin, wmax)
    assert np.allclose(c2["self_pos"], dummy_obs["self_pos"], atol=1e-5)
    assert abs(float(c1["self_pos"][2]) - (-10.0)) < 1e-5, "North (+10) did not canonicalize to South (-10)!"

    tp0 = np.array([0.3, 0.0, 0.6, 0.25, -0.1, 0.9], dtype=np.float32)
    gz0 = np.array([0.25, -0.1, 6.0], dtype=np.float32)
    tp_w, gz_w = uncanonicalize_team_motor(tp0, gz0, -1)
    tp_back, _ = uncanonicalize_team_motor(tp_w, gz_w, -1)
    assert np.allclose(tp_back[:3], tp0[:3], atol=1e-5)
    print("[PASS] 1/2: 180-Degree Team-Symmetry Canonicalization & Motor Involution Verified.")

    # 2. Run 2-Generation Competitive Self-Play PBT League inside TemporaryDirectory
    with tempfile.TemporaryDirectory() as tmp_ckpt:
        cfg = Brain3DConfig(
            task_profile="turf_objective",
            policy_mode="dagger",
            anneal_transitions=80,
            online_update_interval=6,
        )
        res = run_arena_pbt_coordinator(
            cfg,
            env_backend="builtin_3d",
            matchmaking="self_play",
            pop_size=2,
            generations=2,
            episodes_per_gen=1,
            checkpoint_dir=tmp_ckpt,
        )
        assert res["ok"] is True
        assert res["barrier1_parent_zero_torch"] is True
        assert res["generations_completed"] == 2
        assert res["league_snapshots"] >= 2
        assert os.path.isfile(os.path.join(tmp_ckpt, "pbt_latest_champion.pt"))
        assert os.path.isfile(os.path.join(tmp_ckpt, "pbt_latest_champion.pt.meta.json"))
    print(f"[PASS] 2/2: 1-GPU Self-Play PBT League Verified (Elos={res['final_elos']}, League={res['league_snapshots']}).")
    print("=" * 78)
    print("  ALL PBT & SELF-PLAY VERIFICATION CHECKS PASSED!")
    print("=" * 78)
    return True


def main() -> int:
    mp.freeze_support()
    ap = argparse.ArgumentParser(description="1-GPU Parallel Multi-Arena PBT & Competitive Self-Play Trainer")
    ap.add_argument("--mode", type=str, default="verify_pbt", choices=["verify_pbt", "train", "spectate"])
    ap.add_argument("--env", type=str, default="builtin_3d", choices=["builtin_3d", "unity_tcp"])
    ap.add_argument("--matchmaking", type=str, default="self_play", choices=["self_play", "pbt_solo"])
    ap.add_argument("--task-profile", type=str, default="turf_objective", choices=["universal_tactical", "turf_objective"])
    ap.add_argument("--pop-size", type=int, default=4)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--episodes-per-gen", type=int, default=3)
    ap.add_argument("--policy-mode", type=str, default="dagger", choices=["scaffold", "anneal", "dagger", "pure_neural"])
    ap.add_argument("--hippo-memory-mode", type=str, default=HIPPO_MEM_ADAPTIVE, choices=list(HIPPO_MEM_MODES))
    ap.add_argument("--stage-id", type=int, default=1)
    ap.add_argument("--time-scale", type=float, default=4.0)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=50055)
    ap.add_argument("--ux-port", type=int, default=0)
    ap.add_argument("--sleep-consolidation", type=int, default=1)
    ap.add_argument("--sleep-epochs", type=int, default=6)
    ap.add_argument("--online-update-interval", type=int, default=8)
    ap.add_argument("--anneal-transitions", type=int, default=1600)
    ap.add_argument("--scaffold-prior-scale", type=float, default=1.0)
    ap.add_argument("--scaffold-anneal-end", type=float, default=-1.0, help="If >= 0, linearly anneal scaffold_prior_scale to this value across generations")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints/user_runs")
    ap.add_argument("--init-weights", type=str, default="")
    ap.add_argument("--load-weights", type=str, default="")
    ap.add_argument("--opp-weights", type=str, default="")
    ap.add_argument("--continual-learn", action="store_true", default=False)
    ap.add_argument("--save-continual-weights", type=str, default="")
    ap.add_argument("--ollama-model", type=str, default="gemma4:12b")
    ap.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-mode", type=str, default="hybrid", choices=["hybrid", "cpu", "gpu"])
    ap.add_argument("--disable-strategy-kb", action="store_true", default=False)
    ap.add_argument("--strategy-db-path", type=str, default="")
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--watch-stdin", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if args.mode == "verify_pbt":
        return 0 if run_pbt_self_verification() else 1

    ckpt_root = args.checkpoint_dir.strip() if args.checkpoint_dir and args.checkpoint_dir.strip() else "checkpoints/user_runs"
    resolved_strat_db = args.strategy_db_path.strip() or os.path.join(ckpt_root, "strategy_knowledge.sqlite3")

    if args.mode == "spectate":
        cont_on = bool(args.continual_learn)
        cont_save = args.save_continual_weights.strip() or (
            os.path.join(ckpt_root, "spectate_continual_latest.pt") if cont_on else ""
        )
        cfg = Brain3DConfig(
            task_profile=args.task_profile,
            policy_mode="pure_neural",
            scaffold_prior_scale=float(np.clip(args.scaffold_prior_scale, 0.0, 1.0)),
            default_stage_id=args.stage_id,
            hippo_memory_mode=args.hippo_memory_mode,
            time_scale=args.time_scale,
            unity_host=args.host,
            seed=args.seed,
            enable_learning=cont_on,
            llm_backend="auto",
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            ollama_compute_mode=args.ollama_mode,
            enable_strategy_kb=(not bool(args.disable_strategy_kb)),
            strategy_db_path=resolved_strat_db,
        )
        summary = run_arena_spectate_coordinator(
            cfg,
            env_backend=args.env,
            base_port=args.base_port,
            ux_port=args.ux_port,
            load_weights=args.load_weights,
            opp_weights=args.opp_weights,
            continual_learning=cont_on,
            continual_save_path=cont_save,
            continual_sleep_epochs=max(0, int(args.sleep_epochs)) if int(args.sleep_consolidation) != 0 else 0,
            parent_pid=args.parent_pid,
            watch_stdin=args.watch_stdin,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if summary.get("ok") else 1

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    cfg = Brain3DConfig(
        task_profile=args.task_profile,
        policy_mode=args.policy_mode,
        anneal_transitions=max(16, int(args.anneal_transitions)),
        online_update_interval=max(1, int(args.online_update_interval)),
        scaffold_prior_scale=float(np.clip(args.scaffold_prior_scale, 0.0, 1.0)),
        default_stage_id=args.stage_id,
        hippo_memory_mode=args.hippo_memory_mode,
        time_scale=args.time_scale,
        unity_host=args.host,
        seed=args.seed,
        llm_backend="auto",
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        ollama_compute_mode=args.ollama_mode,
        enable_strategy_kb=(not bool(args.disable_strategy_kb)),
        strategy_db_path=resolved_strat_db,
    )
    summary = run_arena_pbt_coordinator(
        cfg,
        env_backend=args.env,
        matchmaking=args.matchmaking,
        pop_size=args.pop_size,
        generations=args.generations,
        episodes_per_gen=args.episodes_per_gen,
        base_port=args.base_port,
        checkpoint_dir=args.checkpoint_dir,
        init_weights_path=args.init_weights,
        parent_pid=args.parent_pid,
        enable_sleep=bool(int(args.sleep_consolidation) != 0),
        sleep_epochs=int(args.sleep_epochs),
        scaffold_anneal_end=(float(np.clip(args.scaffold_anneal_end, 0.0, 1.0)) if float(args.scaffold_anneal_end) >= 0.0 else None),
        watch_stdin=args.watch_stdin,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
