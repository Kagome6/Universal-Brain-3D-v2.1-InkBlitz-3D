# -*- coding: utf-8 -*-
"""
brain3d_spatial_timing.py — 4-Layer Allocentric Field Map, Landmark EKF-SLAM,
Ballistic Cerebellum, Authoritative 8-Candidate Proposal & Basal Ganglia Gate (P3, v2.3.0)
==========================================================================================
Barrier 1 Guarantee: Strictly Zero-PyTorch (`"torch" not in sys.modules`).
License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

from collections import OrderedDict, deque
import heapq
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from task_profiles_3d import *
from brain3d_config_ipc import *
from brain3d_config_ipc import (
    _ignore_sigint_in_worker,
    _read_runtime_learning_flags,
    _resolve_active_fovea_patch,
    _resolve_active_ventral_patch,
)
import ctypes as _ctypes
try:
    from native_accel.fast_kernels_bridge import (
        NATIVE_KERNELS_AVAILABLE as _NATIVE_OK,
        f32_ptr as _f32_ptr,
        get_native_lib as _get_native_lib,
        u8_ptr as _u8_ptr,
    )
    _NATIVE_LIB = _get_native_lib() if _NATIVE_OK else None
except Exception:
    _NATIVE_OK = False
    _NATIVE_LIB = None

# ==============================================================================
# 5. Upgraded Spatial Cognition v2: 4-Layer Dynamic Field Map, Landmark EKF-SLAM,
#    Respawn Discontinuity Detector, Ballistic Cerebellum & Basal Ganglia Gate
# ==============================================================================
class MultiLayerCognitiveFieldMap3D:
    """
    48x48 4-Layer Allocentric Cognitive Field Map built strictly from 1st-person experience
    (plus optional low-rate HUD tactical overview sync when enabled by game rules):
      - Layer 0 (`occ_state` / `wall_score`): Traversable floor (`1`) vs. structural obstacle (`2`) vs. unknown (`0`).
      - Layer 1 (`surface_map`)             : Dynamic surface/turf control in `[-1.0 .. +1.0]`
                                              (`-1.0` Enemy Turf, `0.0` Neutral/Uncertain, `+1.0` Friendly Turf),
                                              with exponential epistemic staleness decay when unobserved!
      - Layer 2 (`elev_map`)                : Relative local 3D surface/ledge height in `[-4.0m .. +4.0m]`.
      - Layer 3 (`visit_freshness`)         : Observation freshness in `[0.0 .. 1.0]` driving frontier exploration.
    """

    UNKNOWN = 0
    FREE = 1
    WALL = 2

    def __init__(
        self,
        world_min: np.ndarray,
        world_max: np.ndarray,
        grid_n: int = FIELD_MAP_MAX_N,
        surface_decay_per_sec: float = 0.05,
    ):
        self.wmin = np.asarray(world_min, dtype=np.float32).copy()
        self.wmax = np.asarray(world_max, dtype=np.float32).copy()
        self.span = np.maximum(1.0, self.wmax - self.wmin)
        self.grid_n = int(min(FIELD_MAP_MAX_N, max(16, grid_n)))
        self.cell_w = float(self.span[0]) / float(self.grid_n)
        self.cell_d = float(self.span[2]) / float(self.grid_n)
        self.surface_decay_per_sec = float(max(0.0, surface_decay_per_sec))

        self.occ_state = np.zeros((self.grid_n, self.grid_n), dtype=np.uint8)
        self.wall_score = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.surface_map = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.surface_obs_age = np.full((self.grid_n, self.grid_n), 99.0, dtype=np.float32)
        self.elev_map = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.visit_freshness = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.dwell_saturation = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.strategic_hazard_scar = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self._hazard_scar_tail = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.frac_memory_alpha: float = 0.52
        self.strategic_lane_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._strategic_lane_mult_grid = np.ones((self.grid_n, self.grid_n), dtype=np.float32)
        self._committed_frontier_cell: Optional[Tuple[int, int]] = None
        self._committed_frontier_ticks: int = 0
        self._committed_flank_side: int = 0
        self._committed_flank_ticks: int = 0
        self._map_step_counter: int = 0
        self._route_cache: Dict[Tuple[int, int, int, int, int, int], Tuple[int, Optional[np.ndarray], float]] = {}
        self._zz_grid, self._xx_grid = np.indices((self.grid_n, self.grid_n), dtype=np.float32)
        self._col_world_x = (float(self.wmin[0]) + (np.arange(self.grid_n, dtype=np.float32) + 0.5) * self.cell_w)

        # Precomputed ray tangent grid for fast vectorized 16x16 frustum projection
        self._cached_tan_deg = -1.0
        self._uu: Optional[np.ndarray] = None
        self._vv: Optional[np.ndarray] = None

    def reset_full_episode(self) -> None:
        self.occ_state.fill(self.UNKNOWN)
        self.wall_score.fill(0.0)
        self.surface_map.fill(0.0)
        self.surface_obs_age.fill(99.0)
        self.elev_map.fill(0.0)
        self.visit_freshness.fill(0.0)
        self.dwell_saturation.fill(0.0)
        self.strategic_hazard_scar.fill(0.0)
        self._hazard_scar_tail.fill(0.0)
        self._committed_frontier_cell = None
        self._committed_frontier_ticks = 0
        self._committed_flank_side = 0
        self._committed_flank_ticks = 0
        self._map_step_counter = 0
        self._route_cache.clear()

    def set_strategic_lane_weights(
        self,
        w_left: float,
        w_center: float,
        w_right: float,
        clear_commitment: bool = False,
    ) -> None:
        wl = float(np.clip(w_left, 0.55, 1.50))
        wc = float(np.clip(w_center, 0.55, 1.50))
        wr = float(np.clip(w_right, 0.55, 1.50))
        prev = self.strategic_lane_weights
        if abs(wl - prev[0]) < 1e-3 and abs(wc - prev[1]) < 1e-3 and abs(wr - prev[2]) < 1e-3:
            if clear_commitment:
                self._committed_frontier_cell = None
                self._committed_frontier_ticks = 0
                self._committed_flank_side = 0
                self._committed_flank_ticks = 0
            return
        self.strategic_lane_weights = (wl, wc, wr)
        wx = self._col_world_x
        row_m = np.where(wx < -3.6, wl, np.where(wx > 3.6, wr, wc)).astype(np.float32)
        self._strategic_lane_mult_grid[:, :] = row_m[None, :]
        if clear_commitment:
            self._committed_frontier_cell = None
            self._committed_frontier_ticks = 0
            self._committed_flank_side = 0
            self._committed_flank_ticks = 0
            self._route_cache.clear()

    def set_strategic_lane_code(self, lane_code: int) -> None:
        presets = {
            0: (1.00, 1.00, 1.00),
            1: (1.32, 0.86, 0.94),
            2: (0.92, 1.26, 0.92),
            3: (0.94, 0.86, 1.32),
            4: (1.26, 0.76, 1.26),
        }
        w_tup = presets.get(int(lane_code), (1.00, 1.00, 1.00))
        changed = any(abs(w_tup[i] - self.strategic_lane_weights[i]) > 0.02 for i in range(3))
        self.set_strategic_lane_weights(w_tup[0], w_tup[1], w_tup[2], clear_commitment=changed)

    def register_strategic_hazard_scar(self, dead_x: float, dead_z: float) -> None:
        """
        Deposit a two-pole fractional (Mittag-Leffler-like) danger memory around a splat site [v2.3.2]:
        fast transient sheds quickly so M0 will not swing to stadium rims, while `_hazard_scar_tail`
        preserves a subtle power-law caution trace at repeated ambush sites.
        """
        iz, ix = self.world_to_cell(float(dead_x), float(dead_z))
        tail_w = float(np.clip(1.0 - float(self.frac_memory_alpha), 0.14, 0.65))
        for dz in range(-2, 3):
            for dx in range(-2, 3):
                jz, jx = iz + dz, ix + dx
                if 0 <= jz < self.grid_n and 0 <= jx < self.grid_n:
                    d2 = float(dz * dz + dx * dx)
                    imp = 0.30 / (1.0 + 0.55 * d2)
                    self._hazard_scar_tail[jz, jx] = min(0.24, float(self._hazard_scar_tail[jz, jx]) + tail_w * 0.42 * imp)
                    self.strategic_hazard_scar[jz, jx] = min(0.52, float(self.strategic_hazard_scar[jz, jx]) + imp)
        self._route_cache.clear()

    def export_structural_schema(self) -> Dict[str, Any]:
        """
        Export ONLY stage-invariant structural geometry (`occ_state`, `wall_score`, `elev_map`).
        Transient match-specific ink/turf (`surface_map`) and visit freshness are intentionally excluded!
        """
        explored = int(np.sum(self.occ_state != self.UNKNOWN))
        return {
            "occ_state": self.occ_state.copy(),
            "wall_score": self.wall_score.copy(),
            "elev_map": self.elev_map.copy(),
            "explored_cells": explored,
        }

    def restore_structural_schema_for_new_episode(
        self,
        schema: Dict[str, Any],
        spawn_pos: np.ndarray,
        retention: float = 0.88,
    ) -> None:
        """
        Start a new match on a known stage:
          - Wipes 100% of dynamic match-specific turf (`surface_map = 0.0`), freshness, and route caches
            to prevent proactive ink hallucination from the previous match.
          - Restores structural traversability (`occ_state`) & 3D elevation (`elev_map`) with plastic
            `wall_score * retention` so dynamic stage variants remain adaptable.
        """
        self.reset_full_episode()
        occ = schema.get("occ_state")
        wsc = schema.get("wall_score")
        elv = schema.get("elev_map")
        if occ is not None and occ.shape == self.occ_state.shape:
            self.occ_state[:, :] = occ
        if wsc is not None and wsc.shape == self.wall_score.shape:
            self.wall_score[:, :] = np.asarray(wsc, dtype=np.float32) * float(np.clip(retention, 0.50, 0.98))
        if elv is not None and elv.shape == self.elev_map.shape:
            self.elev_map[:, :] = np.asarray(elv, dtype=np.float32)

        iz, ix = self.world_to_cell(float(spawn_pos[0]), float(spawn_pos[2]))
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                jz = int(np.clip(iz + dz, 0, self.grid_n - 1))
                jx = int(np.clip(ix + dx, 0, self.grid_n - 1))
                self.occ_state[jz, jx] = self.FREE
                self.wall_score[jz, jx] = 0.0
                self.surface_map[jz, jx] = 0.70
                self.surface_obs_age[jz, jx] = 0.0
                self.visit_freshness[jz, jx] = 0.75

    def on_mid_episode_respawn(self, spawn_pos: np.ndarray) -> None:
        """
        Preserve structural wall map & explored surface beliefs across mid-match respawns,
        while aging unobserved surface cells slightly (time passed during respawn) and
        marking the spawn pad as traversable friendly turf.
        """
        self.surface_map *= 0.85
        self.surface_obs_age += 2.5
        self.dwell_saturation *= 0.35
        self._committed_frontier_cell = None
        self._committed_frontier_ticks = 0
        self._committed_flank_side = 0
        self._committed_flank_ticks = 0
        self._route_cache.clear()
        iz, ix = self.world_to_cell(float(spawn_pos[0]), float(spawn_pos[2]))
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                jz = int(np.clip(iz + dz, 0, self.grid_n - 1))
                jx = int(np.clip(ix + dx, 0, self.grid_n - 1))
                self.occ_state[jz, jx] = self.FREE
                self.wall_score[jz, jx] = 0.0
                self.surface_map[jz, jx] = max(0.65, float(self.surface_map[jz, jx]))
                self.surface_obs_age[jz, jx] = 0.0
                self.visit_freshness[jz, jx] = 1.0

    def world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        ix = int(np.clip(math.floor((float(x) - float(self.wmin[0])) / self.cell_w), 0, self.grid_n - 1))
        iz = int(np.clip(math.floor((float(z) - float(self.wmin[2])) / self.cell_d), 0, self.grid_n - 1))
        return iz, ix

    def cell_to_world_xz(self, iz: int, ix: int) -> np.ndarray:
        wx = float(self.wmin[0]) + (float(ix) + 0.5) * self.cell_w
        wz = float(self.wmin[2]) + (float(iz) + 0.5) * self.cell_d
        return np.array([wx, wz], dtype=np.float32)

    def sync_tactical_overview(self, overview_surface_grid: np.ndarray, blend: float = 0.65) -> None:
        """
        Optional low-rate tactical HUD map sync (e.g., tactical overhead turf map):
        Updates ONLY floor surface beliefs (`surface_map`), never leaking hidden enemy positions.
        """
        g = np.asarray(overview_surface_grid, dtype=np.float32)
        if g.shape != (self.grid_n, self.grid_n):
            return
        b = float(np.clip(blend, 0.05, 1.0))
        self.surface_map = np.clip((1.0 - b) * self.surface_map + b * g, -1.0, 1.0)
        self.surface_obs_age = np.minimum(self.surface_obs_age, 0.5)

    def integrate_first_person_frame(
        self,
        believed_pos: np.ndarray,
        gaze_ypd: np.ndarray,
        fovea_patch: np.ndarray,
        proprio: np.ndarray,
        fovea_half_angle_deg: float,
        max_range_m: float,
        dt: float,
        ventral_patch: Optional[np.ndarray] = None,
        epistemic_uncert_16: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Vectorized 60Hz integration of 1st-person dorsal+ventral rays & footstep proprioception.
        Returns `(friendly_turf_ratio, enemy_turf_ratio)` over explored traversable cells.
        """
        # 1. Temporal staleness decay on unobserved surface & visit freshness (+ v2.3.2 heavy-tail scar memory)
        self._map_step_counter += 1
        if np.any(self._hazard_scar_tail > 1e-4):
            self._hazard_scar_tail *= math.exp(-0.012 * float(dt))
        if _NATIVE_LIB is not None and self.occ_state.flags["C_CONTIGUOUS"]:
            surf_posture = float(proprio[PROPRIO_IDX_SURFACE_POSTURE]) if proprio.shape[0] > PROPRIO_IDX_SURFACE_POSTURE else 0.45
            fov_c = np.ascontiguousarray(fovea_patch, dtype=np.float32)
            vent_ptr = _f32_ptr(np.ascontiguousarray(ventral_patch, dtype=np.float32)) if ventral_patch is not None else None
            unc_ptr = _f32_ptr(np.ascontiguousarray(epistemic_uncert_16, dtype=np.float32)) if epistemic_uncert_16 is not None else None
            out_r2 = np.zeros(2, dtype=np.float32)
            _NATIVE_LIB.ub3d_integrate_first_person_frame(
                _u8_ptr(self.occ_state),
                _f32_ptr(self.wall_score),
                _f32_ptr(self.surface_map),
                _f32_ptr(self.surface_obs_age),
                _f32_ptr(self.elev_map),
                _f32_ptr(self.visit_freshness),
                _f32_ptr(self.dwell_saturation),
                _f32_ptr(self.strategic_hazard_scar),
                int(self.grid_n),
                float(self.wmin[0]), float(self.wmin[1]), float(self.wmin[2]),
                float(self.cell_w), float(self.cell_d),
                float(self.surface_decay_per_sec),
                float(believed_pos[0]), float(believed_pos[1]), float(believed_pos[2]),
                float(gaze_ypd[0]), float(gaze_ypd[1]),
                float(surf_posture),
                float(math.tan(math.radians(float(fovea_half_angle_deg)))),
                float(max_range_m), float(dt),
                _f32_ptr(fov_c), vent_ptr, unc_ptr,
                _f32_ptr(out_r2),
            )
            if np.any(self._hazard_scar_tail > 1e-4):
                np.maximum(self.strategic_hazard_scar, self._hazard_scar_tail, out=self.strategic_hazard_scar)
            return float(out_r2[0]), float(out_r2[1])

        self.surface_obs_age = np.minimum(99.0, self.surface_obs_age + float(dt))
        if self.surface_decay_per_sec > 0.0:
            stale_mask = self.surface_obs_age > 1.0
            if np.any(stale_mask):
                decay_k = math.exp(-self.surface_decay_per_sec * float(dt))
                self.surface_map = np.where(stale_mask, self.surface_map * decay_k, self.surface_map)
        self.visit_freshness *= math.exp(-0.025 * float(dt))
        self.dwell_saturation *= math.exp(-0.085 * float(dt))
        self.strategic_hazard_scar = np.maximum(
            self.strategic_hazard_scar * math.exp(-0.095 * float(dt)),
            self._hazard_scar_tail,
        )

        # 2. Mark ego footprint & proprioceptive surface contact (including immediate 4-connected touch radius)
        ego_iz, ego_ix = self.world_to_cell(float(believed_pos[0]), float(believed_pos[2]))
        self.occ_state[ego_iz, ego_ix] = self.FREE
        self.wall_score[ego_iz, ego_ix] = max(0.0, float(self.wall_score[ego_iz, ego_ix]) - 0.5)
        self.visit_freshness[ego_iz, ego_ix] = 1.0
        self.dwell_saturation[ego_iz, ego_ix] = min(2.5, float(self.dwell_saturation[ego_iz, ego_ix]) + 0.42 * float(dt))
        for ndz, ndx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            jz, jx = ego_iz + ndz, ego_ix + ndx
            if 0 <= jz < self.grid_n and 0 <= jx < self.grid_n:
                if int(self.occ_state[jz, jx]) == self.UNKNOWN:
                    self.occ_state[jz, jx] = self.FREE
                self.visit_freshness[jz, jx] = max(float(self.visit_freshness[jz, jx]), 0.88)
                self.dwell_saturation[jz, jx] = min(2.5, float(self.dwell_saturation[jz, jx]) + 0.24 * float(dt))

        surf_posture = float(proprio[PROPRIO_IDX_SURFACE_POSTURE]) if proprio.shape[0] > PROPRIO_IDX_SURFACE_POSTURE else 0.45
        if surf_posture >= 0.62:
            self.surface_map[ego_iz, ego_ix] = float(np.clip(0.35 * self.surface_map[ego_iz, ego_ix] + 0.65 * 0.90, -1.0, 1.0))
            self.surface_obs_age[ego_iz, ego_ix] = 0.0
        elif surf_posture <= 0.20:
            self.surface_map[ego_iz, ego_ix] = float(np.clip(0.35 * self.surface_map[ego_iz, ego_ix] - 0.65 * 0.90, -1.0, 1.0))
            self.surface_obs_age[ego_iz, ego_ix] = 0.0

        # 3. Project 16x16 foveal rays into 3D world coordinates
        if abs(self._cached_tan_deg - float(fovea_half_angle_deg)) > 1e-3 or self._uu is None:
            th = math.tan(math.radians(float(fovea_half_angle_deg)))
            u = np.linspace(-th, th, FOVEA_W, dtype=np.float32)
            v = np.linspace(th, -th, FOVEA_H, dtype=np.float32)
            self._uu, self._vv = np.meshgrid(u, v)
            self._cached_tan_deg = float(fovea_half_angle_deg)

        cy, sy = math.cos(float(gaze_ypd[0])), math.sin(float(gaze_ypd[0]))
        cp, sp = math.cos(float(gaze_ypd[1])), math.sin(float(gaze_ypd[1]))
        fwd = np.array([sy * cp, sp, cy * cp], dtype=np.float32)
        right = np.array([cy, 0.0, -sy], dtype=np.float32)
        up = np.cross(fwd, right).astype(np.float32)

        dirs = fwd[None, None, :] + self._uu[:, :, None] * right[None, None, :] + self._vv[:, :, None] * up[None, None, :]
        dirs = dirs / np.maximum(1e-6, np.linalg.norm(dirs, axis=-1, keepdims=True))

        p = np.asarray(fovea_patch, dtype=np.float32).reshape(FOVEA_H, FOVEA_W, FOVEA_C)
        norm_d = np.clip(p[:, :, 0], 0.0, 1.0)
        ny = p[:, :, 2]
        sal = p[:, :, 3]
        dist_m = (1.0 - norm_d) * float(max_range_m)

        conf_w = np.ones((FOVEA_H, FOVEA_W), dtype=np.float32)
        if epistemic_uncert_16 is not None:
            conf_w = np.clip(1.0 - np.asarray(epistemic_uncert_16, dtype=np.float32).reshape(FOVEA_H, FOVEA_W), 0.15, 1.0)

        hit_pts = believed_pos[None, None, :] + dist_m[:, :, None] * dirs
        valid_hit = (norm_d > 0.03) & (dist_m > 0.25) & (conf_w > 0.30)

        # Mark intermediate ray samples along horizontal rays as FREE (clears dynamic obstacles)
        mid_pts = believed_pos[None, None, :] + (0.55 * dist_m)[:, :, None] * dirs
        mid_mask = (dist_m > 1.6) & (np.abs(dirs[:, :, 1]) < 0.45) & (conf_w > 0.35)
        if np.any(mid_mask):
            m_ix = np.clip(np.floor((mid_pts[:, :, 0][mid_mask] - self.wmin[0]) / self.cell_w).astype(np.int32), 0, self.grid_n - 1)
            m_iz = np.clip(np.floor((mid_pts[:, :, 2][mid_mask] - self.wmin[2]) / self.cell_d).astype(np.int32), 0, self.grid_n - 1)
            self.wall_score[m_iz, m_ix] = np.maximum(0.0, self.wall_score[m_iz, m_ix] - 0.18)
            self.occ_state[m_iz, m_ix] = np.where(self.wall_score[m_iz, m_ix] < 0.45, self.FREE, self.occ_state[m_iz, m_ix])
            self.visit_freshness[m_iz, m_ix] = np.maximum(self.visit_freshness[m_iz, m_ix], 0.55)

        # Floor / elevated walkable platform hits (`ny > 0.58`)
        floor_mask = valid_hit & (ny > 0.58)
        if np.any(floor_mask):
            fx = hit_pts[:, :, 0][floor_mask]
            fy = hit_pts[:, :, 1][floor_mask]
            fz = hit_pts[:, :, 2][floor_mask]
            fsal = sal[floor_mask]
            fw = conf_w[floor_mask]

            f_ix = np.clip(np.floor((fx - self.wmin[0]) / self.cell_w).astype(np.int32), 0, self.grid_n - 1)
            f_iz = np.clip(np.floor((fz - self.wmin[2]) / self.cell_d).astype(np.int32), 0, self.grid_n - 1)

            self.wall_score[f_iz, f_ix] = np.maximum(0.0, self.wall_score[f_iz, f_ix] - 0.25 * fw)
            self.occ_state[f_iz, f_ix] = np.where(self.wall_score[f_iz, f_ix] < 0.50, self.FREE, self.occ_state[f_iz, f_ix])
            rel_elev = np.clip(fy - float(self.wmin[1]), -4.0, 4.0)
            self.elev_map[f_iz, f_ix] = 0.70 * self.elev_map[f_iz, f_ix] + 0.30 * rel_elev
            self.visit_freshness[f_iz, f_ix] = np.maximum(self.visit_freshness[f_iz, f_ix], 0.85 * fw)

            # Decode surface turf valence from dorsal salience channel (and ventral color when neutral)
            surf_obs = np.where(fsal > 0.66, 1.0, np.where(fsal < 0.36, -1.0, 0.0)).astype(np.float32)
            if ventral_patch is not None:
                vp = np.asarray(ventral_patch, dtype=np.float32).reshape(FOVEA_H, FOVEA_W, VENTRAL_C)
                by_fl = vp[:, :, 1][floor_mask]
                rg_fl = vp[:, :, 0][floor_mask]
                # Subtle ventral tie-breaker only when dorsal code is near-ambiguous
                ambig = (fsal >= 0.36) & (fsal <= 0.66) & (np.abs(by_fl - rg_fl) > 0.45)
                surf_obs = np.where(ambig, np.clip(by_fl - rg_fl, -0.75, 0.75), surf_obs)

            alpha_s = np.clip(0.45 * fw, 0.10, 0.55)
            self.surface_map[f_iz, f_ix] = np.clip(
                (1.0 - alpha_s) * self.surface_map[f_iz, f_ix] + alpha_s * surf_obs,
                -1.0,
                1.0,
            )
            self.surface_obs_age[f_iz, f_ix] = 0.0

        # Vertical structural obstacle / cover wall hits (`|ny| < 0.42`, excluding interactive goals/objectives/actors `sal >= 0.70`)
        wall_mask = valid_hit & (np.abs(ny) < 0.42) & (sal < 0.70) & (hit_pts[:, :, 1] < float(believed_pos[1]) + 1.6)
        if np.any(wall_mask):
            wx = hit_pts[:, :, 0][wall_mask]
            wy = hit_pts[:, :, 1][wall_mask]
            wz = hit_pts[:, :, 2][wall_mask]
            ww = conf_w[wall_mask]
            w_ix = np.clip(np.floor((wx - self.wmin[0]) / self.cell_w).astype(np.int32), 0, self.grid_n - 1)
            w_iz = np.clip(np.floor((wz - self.wmin[2]) / self.cell_d).astype(np.int32), 0, self.grid_n - 1)
            self.wall_score[w_iz, w_ix] = np.minimum(3.0, self.wall_score[w_iz, w_ix] + 0.38 * ww)
            self.occ_state[w_iz, w_ix] = np.where(self.wall_score[w_iz, w_ix] >= 0.75, self.WALL, self.occ_state[w_iz, w_ix])
            self.elev_map[w_iz, w_ix] = np.maximum(self.elev_map[w_iz, w_ix], np.clip(wy - float(self.wmin[1]), 0.0, 4.0))
            self.visit_freshness[w_iz, w_ix] = np.maximum(self.visit_freshness[w_iz, w_ix], 0.70 * ww)

        free_cells = self.occ_state == self.FREE
        n_free = int(np.sum(free_cells))
        if n_free <= 0:
            return 0.0, 0.0
        friendly_ratio = float(np.sum(free_cells & (self.surface_map > 0.25))) / float(n_free)
        enemy_ratio = float(np.sum(free_cells & (self.surface_map < -0.25))) / float(n_free)
        return friendly_ratio, enemy_ratio

    def find_best_frontier_or_turf_target(
        self,
        ego_xz: np.ndarray,
        unknown_weight: float = 2.2,
        turf_weight: float = 1.8,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Select the highest-salience reachable frontier cell balancing:
          1. Epistemic exploration (`UNKNOWN` cells adjacent to `FREE` cells, or low `visit_freshness`)
          2. Territorial surface expansion (`FREE` cells currently Enemy `< 0` or Neutral `~ 0`)
        Returns `(unit_dir_xz, norm_dist, valid)`.
        """
        ego_iz, ego_ix = self.world_to_cell(float(ego_xz[0]), float(ego_xz[1]))
        if _NATIVE_LIB is not None and self.occ_state.flags["C_CONTIGUOUS"]:
            ccz, ccx = self._committed_frontier_cell if self._committed_frontier_cell is not None else (-1, -1)
            out_bz = _ctypes.c_int(-1)
            out_bx = _ctypes.c_int(-1)
            out_nt = _ctypes.c_int(0)
            rc = _NATIVE_LIB.ub3d_select_frontier_cell(
                _u8_ptr(self.occ_state),
                _f32_ptr(self.surface_map),
                _f32_ptr(self.visit_freshness),
                _f32_ptr(self._strategic_lane_mult_grid),
                _f32_ptr(self.dwell_saturation),
                _f32_ptr(self.strategic_hazard_scar),
                int(self.grid_n),
                int(ego_iz), int(ego_ix),
                float(unknown_weight), float(turf_weight),
                int(ccz), int(ccx), int(self._committed_frontier_ticks),
                _ctypes.byref(out_bz), _ctypes.byref(out_bx), _ctypes.byref(out_nt),
            )
            if rc == 0:
                return np.array([0.0, 1.0], dtype=np.float32), 0.35, True
            if rc == 1:
                return np.array([0.0, 1.0], dtype=np.float32), 0.25, True
            bz, bx = int(out_bz.value), int(out_bx.value)
            self._committed_frontier_cell = (bz, bx)
            self._committed_frontier_ticks = int(out_nt.value)
            tgt_xz = self.cell_to_world_xz(bz, bx)
            return self.find_tactical_next_step(ego_xz, tgt_xz)

        free_mask = self.occ_state == self.FREE
        if not np.any(free_mask):
            return np.array([0.0, 1.0], dtype=np.float32), 0.35, True

        pad_unk = np.pad((self.occ_state == self.UNKNOWN).astype(np.float32), 1, mode="constant", constant_values=0.0)
        unk_adj = pad_unk[:-2, 1:-1] + pad_unk[2:, 1:-1] + pad_unk[1:-1, :-2] + pad_unk[1:-1, 2:]

        unpainted_gain = np.clip(0.65 - self.surface_map, 0.0, 1.65)
        stale_gain = np.clip(1.0 - self.visit_freshness, 0.0, 1.0)

        dist_cells = np.hypot(self._xx_grid - float(ego_ix), self._zz_grid - float(ego_iz))
        valid_cand = free_mask & (dist_cells >= 1.8)
        if not np.any(valid_cand):
            return np.array([0.0, 1.0], dtype=np.float32), 0.25, True

        score = (
            (
                float(unknown_weight) * (0.45 * unk_adj + 0.55 * stale_gain)
                + float(turf_weight) * unpainted_gain
            ) * self._strategic_lane_mult_grid
            - 1.35 * self.dwell_saturation
            - 1.25 * self.strategic_hazard_scar
            - 0.14 * dist_cells
        )

        # Spatial target commitment hysteresis: keep current target until reached (< 1.15 cells) or saturated/stale
        bz, bx = -1, -1
        if self._committed_frontier_cell is not None:
            cz, cx = self._committed_frontier_cell
            c_dist = math.hypot(float(cx - ego_ix), float(cz - ego_iz))
            if (
                0 <= cz < self.grid_n
                and 0 <= cx < self.grid_n
                and bool(free_mask[cz, cx])
                and c_dist >= 1.15
                and self._committed_frontier_ticks < 42
                and float(score[cz, cx]) > 0.15
            ):
                bz, bx = cz, cx
                self._committed_frontier_ticks += 1

        if bz < 0:
            masked_score = np.where(valid_cand, score, -1e6)
            flat_idx = int(np.argmax(masked_score))
            bz, bx = divmod(flat_idx, self.grid_n)
            self._committed_frontier_cell = (bz, bx)
            self._committed_frontier_ticks = 1

        tgt_xz = self.cell_to_world_xz(bz, bx)
        return self.find_tactical_next_step(ego_xz, tgt_xz)

    def find_nearest_friendly_refuge(self, ego_xz: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Locate the nearest traversable cell with Friendly Turf (`surface > 0.30`) or adjacent cover wall."""
        ego_iz, ego_ix = self.world_to_cell(float(ego_xz[0]), float(ego_xz[1]))
        free_mask = self.occ_state == self.FREE
        pad_wall = np.pad((self.occ_state == self.WALL).astype(np.float32), 1, mode="constant", constant_values=0.0)
        wall_adj = (pad_wall[:-2, 1:-1] + pad_wall[2:, 1:-1] + pad_wall[1:-1, :-2] + pad_wall[1:-1, 2:]) > 0.5
        refuge_mask = free_mask & ((self.surface_map > 0.30) | wall_adj)
        if not np.any(refuge_mask):
            return np.zeros(2, dtype=np.float32), 1.0, False
        if bool(refuge_mask[ego_iz, ego_ix]):
            return np.zeros(2, dtype=np.float32), 0.0, True

        dist_cells = np.hypot(self._xx_grid - float(ego_ix), self._zz_grid - float(ego_iz))
        quality_cost = dist_cells - 1.2 * np.maximum(0.0, self.surface_map)
        quality_cost = np.where(refuge_mask, quality_cost, 1e6)
        bz, bx = divmod(int(np.argmin(quality_cost)), self.grid_n)
        tgt_xz = self.cell_to_world_xz(bz, bx)
        delta = tgt_xz - ego_xz
        d_m = float(np.linalg.norm(delta))
        max_diag = float(math.hypot(float(self.span[0]), float(self.span[2])))
        if d_m < 0.25:
            return np.zeros(2, dtype=np.float32), 0.0, True
        return (delta / d_m).astype(np.float32), float(np.clip(d_m / max_diag, 0.0, 1.0)), True

    def find_tactical_next_step(
        self,
        start_xz: np.ndarray,
        goal_xz: np.ndarray,
        threat_xz: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Surface-and-Threat-Aware Dijkstra path planner on the `48x48` field map:
          - Avoids `WALL` cells
          - Prefers fast Friendly Turf (`surface > 0`) and penalizes sticky Enemy Turf (`surface < 0`)
          - Adds a soft repulsion halo around `threat_xz` so routes naturally flank around danger!
        Returns `(unit_step_dir_xz, normalized_route_dist, valid)`.
        """
        direct = goal_xz - start_xz
        direct_d = float(np.linalg.norm(direct))
        max_diag = float(math.hypot(float(self.span[0]), float(self.span[2])))
        if direct_d < 0.45:
            u = (direct / max(1e-4, direct_d)).astype(np.float32) if direct_d > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)
            return u, float(np.clip(direct_d / max_diag, 0.0, 1.0)), True

        sz, sx = self.world_to_cell(float(start_xz[0]), float(start_xz[1]))
        gz, gx = self.world_to_cell(float(goal_xz[0]), float(goal_xz[1]))
        if (sz, sx) == (gz, gx):
            return (direct / direct_d).astype(np.float32), float(np.clip(direct_d / max_diag, 0.0, 1.0)), True

        tz, tx = (-999, -999)
        if threat_xz is not None:
            tz, tx = self.world_to_cell(float(threat_xz[0]), float(threat_xz[1]))

        ckey = (sz, sx, gz, gx, tz, tx)
        cached = self._route_cache.get(ckey)
        if cached is not None and (self._map_step_counter - cached[0]) <= 3:
            wp_cached, norm_cached = cached[1], cached[2]
            if wp_cached is None:
                return (direct / direct_d).astype(np.float32), float(np.clip(direct_d / max_diag, 0.0, 1.0)), True
            step_v = wp_cached - start_xz
            sd = float(np.linalg.norm(step_v))
            if sd < 1e-4:
                step_v = direct
                sd = direct_d
            return (step_v / max(1e-4, sd)).astype(np.float32), norm_cached, True

        if _NATIVE_LIB is not None and self.occ_state.flags["C_CONTIGUOUS"]:
            out_dir = np.zeros(2, dtype=np.float32)
            out_wp = np.zeros(2, dtype=np.float32)
            out_nr = _ctypes.c_float(0.0)
            out_has_wp = _ctypes.c_int(0)
            ok_c = _NATIVE_LIB.ub3d_astar_tactical_step(
                _u8_ptr(self.occ_state),
                _f32_ptr(self.elev_map),
                _f32_ptr(self.surface_map),
                _f32_ptr(self.strategic_hazard_scar),
                int(self.grid_n),
                int(sz), int(sx),
                int(gz), int(gx),
                int(tz), int(tx),
                float(self.cell_w), float(self.cell_d),
                float(self.wmin[0]), float(self.wmin[2]),
                float(start_xz[0]), float(start_xz[1]),
                float(direct[0]), float(direct[1]), float(direct_d), float(max_diag),
                _f32_ptr(out_dir),
                _ctypes.byref(out_nr),
                _f32_ptr(out_wp),
                _ctypes.byref(out_has_wp),
            )
            if ok_c:
                if len(self._route_cache) >= 32:
                    self._route_cache.clear()
                nr_val = float(out_nr.value)
                self._route_cache[ckey] = (self._map_step_counter, out_wp.copy() if out_has_wp.value else None, nr_val)
                return out_dir, nr_val, True

        n = self.grid_n
        nn = n * n
        occ_l = self.occ_state.ravel().tolist()
        elev_l = self.elev_map.ravel().tolist()
        surf_l = self.surface_map.ravel().tolist()
        scar_l = self.strategic_hazard_scar.ravel().tolist()

        inf_c = 1e9
        dist_l = [inf_c] * nn
        parent_l = [-1] * nn
        s_flat = sz * n + sx
        g_flat = gz * n + gx
        dist_l[s_flat] = 0.0

        ad0 = abs(sz - gz)
        bd0 = abs(sx - gx)
        mn0 = ad0 if ad0 < bd0 else bd0
        h0 = 0.749 * (float(abs(ad0 - bd0)) + 1.414 * float(mn0))
        pq: List[Tuple[float, float, int]] = [(h0, 0.0, s_flat)]
        nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414))
        wall_code = int(self.WALL)
        unk_code = int(self.UNKNOWN)

        lof_ux, lof_uz, lof_active = 0.0, 0.0, False
        if tz >= 0:
            dz_ts = float(sz - tz)
            dx_ts = float(sx - tx)
            d_ts = math.hypot(dx_ts, dz_ts)
            if 1.5 <= d_ts <= 22.0:
                lof_ux = dx_ts / d_ts
                lof_uz = dz_ts / d_ts
                lof_active = True

        reached = False
        while pq:
            _, c_cur, c_flat = heapq.heappop(pq)
            if c_cur > dist_l[c_flat] + 1e-5:
                continue
            if c_flat == g_flat:
                reached = True
                break
            cz = c_flat // n
            cx = c_flat - cz * n
            elev_c = elev_l[c_flat]
            for dz, dx, base_step in nbrs:
                nz = cz + dz
                nx_i = cx + dx
                if nz < 0 or nz >= n or nx_i < 0 or nx_i >= n:
                    continue
                n_flat = nz * n + nx_i
                occ = occ_l[n_flat]
                if occ == wall_code and n_flat != g_flat:
                    continue
                dh = elev_l[n_flat] - elev_c
                if dh > 0.62 and n_flat != g_flat:
                    continue
                surf = surf_l[n_flat]
                s_pos = surf if surf > 0.0 else 0.0
                s_neg = -surf if surf < 0.0 else 0.0
                dh_pos = dh if dh > 0.0 else 0.0
                step_cost = base_step * (1.0 - 0.25 * s_pos + 1.15 * s_neg + 0.65 * dh_pos + 1.35 * scar_l[n_flat])
                if occ == unk_code:
                    step_cost += 0.22 * base_step
                if tz >= 0:
                    rdz = float(nz - tz)
                    rdx = float(nx_i - tx)
                    d2_thr = rdz * rdz + rdx * rdx
                    if d2_thr < 49.0:
                        step_cost += 2.6 / (1.0 + 0.28 * d2_thr)
                    if lof_active and 2.0 < d2_thr < 196.0:
                        proj_lof = rdx * lof_ux + rdz * lof_uz
                        if proj_lof > 0.5:
                            perp2 = max(0.0, d2_thr - proj_lof * proj_lof)
                            if perp2 < 6.25:
                                step_cost += 0.85 * base_step / (1.0 + 0.45 * perp2)
                nc = c_cur + step_cost
                if nc + 1e-5 < dist_l[n_flat]:
                    dist_l[n_flat] = nc
                    parent_l[n_flat] = c_flat
                    ad = nz - gz if nz >= gz else gz - nz
                    bd = nx_i - gx if nx_i >= gx else gx - nx_i
                    mn = ad if ad < bd else bd
                    h_val = 0.749 * (float(ad - bd if ad >= bd else bd - ad) + 1.414 * float(mn))
                    heapq.heappush(pq, (nc + h_val, nc, n_flat))

        if len(self._route_cache) >= 32:
            self._route_cache.clear()

        if not reached:
            self._route_cache[ckey] = (self._map_step_counter, None, float(np.clip(direct_d / max_diag, 0.0, 1.0)))
            return (direct / direct_d).astype(np.float32), float(np.clip(direct_d / max_diag, 0.0, 1.0)), True

        # Backtrack to lookahead waypoint ~2-3 cells ahead of start
        path_flats: List[int] = []
        cur_f = g_flat
        while cur_f != s_flat and len(path_flats) < n * 2:
            path_flats.append(cur_f)
            pf = parent_l[cur_f]
            if pf < 0:
                break
            cur_f = pf
        path_flats.reverse()
        look_idx = min(len(path_flats) - 1, 2)
        wp_flat = path_flats[look_idx]
        wp_xz = self.cell_to_world_xz(wp_flat // n, wp_flat % n)
        step_v = wp_xz - start_xz
        sd = float(np.linalg.norm(step_v))
        if sd < 1e-4:
            step_v = direct
            sd = direct_d
        route_len_m = float(len(path_flats)) * 0.5 * (self.cell_w + self.cell_d)
        norm_route = float(np.clip(route_len_m / max_diag, 0.0, 1.0))
        self._route_cache[ckey] = (self._map_step_counter, wp_xz, norm_route)
        return (step_v / max(1e-4, sd)).astype(np.float32), norm_route, True

    def find_flank_detour_next_step(
        self,
        start_xz: np.ndarray,
        goal_xz: np.ndarray,
        direct_dir_xz: np.ndarray,
        threat_xz: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Compute a genuine off-axis Wing / Flank Detour step (`M3` candidate) distinct from `M0`:
          - Within `4.0m` of `goal_xz`, smoothly converges into the goal at a shallow flank angle
            so the agent never circles or runs past an open objective/pedestal.
          - Beyond `4.0m`, selects between Left-Wing and Right-Wing tactical waypoints based on
            threat separation, death-scar avoidance, friendly ink share, and strategic lane weights,
            with short-term hysteresis (`_committed_flank_side`) preventing left-right oscillation.
        Returns `(unit_flank_dir_xz, norm_dist, already_on_wing)`.
        """
        to_goal = goal_xz - start_xz
        d_goal = float(np.linalg.norm(to_goal))
        max_diag = float(math.hypot(float(self.span[0]), float(self.span[2])))
        u_dir = (direct_dir_xz / max(1e-4, float(np.linalg.norm(direct_dir_xz)))).astype(np.float32)

        if d_goal <= 3.6:
            side_sign = float(self._committed_flank_side) if self._committed_flank_side != 0 else -1.0
            perp = np.array([-u_dir[1] * side_sign, u_dir[0] * side_sign], dtype=np.float32)
            conv = 0.86 * (to_goal / max(1e-4, d_goal)) + 0.14 * perp
            cn = float(np.linalg.norm(conv))
            return (conv / max(1e-4, cn)).astype(np.float32), float(np.clip(d_goal / max_diag, 0.0, 1.0)), True

        # Proportional target/obstacle-relative flank offset (2.8m..4.5m around ego->goal line, NEVER hardcoded stadium rim ±8.16m!)
        u_goal = (to_goal / max(1e-4, d_goal)).astype(np.float32)
        perp_g = np.array([-u_goal[1], u_goal[0]], dtype=np.float32)  # Left of ego->goal axis
        lat_off = float(np.clip(0.34 * d_goal, 2.8, 4.5))

        if threat_xz is not None:
            rel_thr = threat_xz - start_xz
            proj_t = float(np.dot(rel_thr, u_goal))
            if 0.8 <= proj_t <= d_goal + 1.5 and abs(float(np.dot(rel_thr, perp_g))) <= 4.2:
                anchor_xz = np.asarray(threat_xz, dtype=np.float32)
            else:
                anchor_xz = 0.52 * start_xz + 0.48 * goal_xz
        else:
            anchor_xz = 0.52 * start_xz + 0.48 * goal_xz

        raw_l = anchor_xz + perp_g * lat_off
        raw_r = anchor_xz - perp_g * lat_off
        margin_x = 1.35
        margin_z = 1.80

        def _snap_open_wp(wx: float, wz: float, push_perp: np.ndarray) -> Tuple[np.ndarray, float, float]:
            cx_w = float(np.clip(wx, float(self.wmin[0]) + margin_x, float(self.wmax[0]) - margin_x))
            cz_w = float(np.clip(wz, float(self.wmin[2]) + margin_z, float(self.wmax[2]) - margin_z))
            iz0, ix0 = self.world_to_cell(cx_w, cz_w)
            best_iz, best_ix = iz0, ix0
            if int(self.occ_state[iz0, ix0]) == self.WALL:
                # First try stepping further laterally past wide cover/baffle edges before radial fallback
                found = False
                for step_m in (1.0, 2.0, 3.0):
                    tx = float(np.clip(cx_w + float(push_perp[0]) * step_m, float(self.wmin[0]) + margin_x, float(self.wmax[0]) - margin_x))
                    tz = float(np.clip(cz_w + float(push_perp[1]) * step_m, float(self.wmin[2]) + margin_z, float(self.wmax[2]) - margin_z))
                    jz, jx = self.world_to_cell(tx, tz)
                    if int(self.occ_state[jz, jx]) != self.WALL:
                        best_iz, best_ix = jz, jx
                        found = True
                        break
                for r_c in range(1, 5):
                    if found:
                        break
                    for oz in range(-r_c, r_c + 1):
                        for ox in range(-r_c, r_c + 1):
                            jz, jx = iz0 + oz, ix0 + ox
                            if 0 <= jz < self.grid_n and 0 <= jx < self.grid_n and int(self.occ_state[jz, jx]) != self.WALL:
                                best_iz, best_ix = jz, jx
                                found = True
                                break
                        if found:
                            break
            z0 = max(0, best_iz - 2)
            z1 = min(self.grid_n, best_iz + 3)
            x0 = max(0, best_ix - 2)
            x1 = min(self.grid_n, best_ix + 3)
            loc_surf = float(np.mean(self.surface_map[z0:z1, x0:x1]))
            loc_scar = float(np.mean(self.strategic_hazard_scar[z0:z1, x0:x1]))
            return self.cell_to_world_xz(best_iz, best_ix), loc_surf, loc_scar

        wp_l, surf_l, scar_l = _snap_open_wp(float(raw_l[0]), float(raw_l[1]), perp_g)
        wp_r, surf_r, scar_r = _snap_open_wp(float(raw_r[0]), float(raw_r[1]), -perp_g)
        wl_strat, _, wr_strat = self.strategic_lane_weights

        def _score_wing(wp: np.ndarray, surf_v: float, scar_v: float, strat_w: float, side_id: int) -> float:
            rel_w = wp - start_xz
            dw = float(np.linalg.norm(rel_w))
            uw = (rel_w / dw) if dw > 1e-4 else u_dir
            distinct = 1.0 - float(np.dot(uw, u_dir))
            thr_sep = 0.0
            if threat_xz is not None:
                d_thr_wp = float(np.linalg.norm(wp - threat_xz))
                thr_sep = min(1.6, d_thr_wp / max(3.5, 0.28 * float(self.span[0])))
            hyst = 0.12 if (self._committed_flank_side == side_id and self._committed_flank_ticks < 10) else 0.0
            return (
                0.85 * float(strat_w)
                + 0.65 * surf_v
                + 0.85 * thr_sep
                + 0.28 * distinct
                - 0.85 * scar_v
                - 0.06 * dw
                + hyst
            )

        sc_l = _score_wing(wp_l, surf_l, scar_l, wl_strat, -1)
        sc_r = _score_wing(wp_r, surf_r, scar_r, wr_strat, +1)
        chosen_side = -1 if sc_l >= sc_r else +1
        if chosen_side == self._committed_flank_side:
            self._committed_flank_ticks += 1
        else:
            self._committed_flank_side = chosen_side
            self._committed_flank_ticks = 1

        wp_best = wp_l if chosen_side < 0 else wp_r
        already_on_wing = float(np.linalg.norm(wp_best - start_xz)) <= 3.0
        nav_tgt = goal_xz if already_on_wing else wp_best
        step_dir, norm_d, _ = self.find_tactical_next_step(start_xz, nav_tgt, threat_xz=threat_xz)
        return step_dir, norm_d, already_on_wing

    def directional_clearance_and_surface(self, ego_xz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute `[N(+Z), E(+X), S(-Z), W(-X)]` structural clearance in `[0,1]` and medium-range
        corridor surface advantage in `[0,1]` (blending immediate foot turf + 7m lane lookahead so
        `D3QN` / `RSSM` perceive enemy-ink blockades along each route without tensor shape changes).
        """
        cz, cx = self.world_to_cell(float(ego_xz[0]), float(ego_xz[1]))
        if _NATIVE_LIB is not None and self.occ_state.flags["C_CONTIGUOUS"]:
            clr = np.ones(4, dtype=np.float32)
            surf_adv = np.full(4, 0.5, dtype=np.float32)
            _NATIVE_LIB.ub3d_directional_clearance_and_surface(
                _u8_ptr(self.occ_state), _f32_ptr(self.surface_map),
                int(self.grid_n), int(cz), int(cx), _f32_ptr(clr), _f32_ptr(surf_adv),
            )
            return clr, surf_adv

        dirs_4 = ((1, 0), (0, 1), (-1, 0), (0, -1))  # +Z(N), +X(E), -Z(S), -X(W)
        clr = np.ones(4, dtype=np.float32)
        surf_adv = np.full(4, 0.5, dtype=np.float32)
        near_steps = 6
        corridor_steps = 11
        for di, (dz, dx) in enumerate(dirs_4):
            pz, px = -dx, dz
            free_steps = 0
            near_acc: List[float] = []
            far_acc: List[float] = []
            blocked_center = False
            for k in range(1, corridor_steps + 1):
                nz, nx_i = cz + dz * k, cx + dx * k
                if not (0 <= nz < self.grid_n and 0 <= nx_i < self.grid_n):
                    break
                if int(self.occ_state[nz, nx_i]) == self.WALL:
                    blocked_center = True
                    break
                if k <= near_steps:
                    free_steps += 1
                    near_acc.append(0.5 * (float(self.surface_map[nz, nx_i]) + 1.0))
                else:
                    # 3-cell wide corridor sample at medium range (~4m..7.5m)
                    s_sum = 0.5 * (float(self.surface_map[nz, nx_i]) + 1.0)
                    s_cnt = 1
                    for lat in (-1, 1):
                        lz_i, lx_i = nz + pz * lat, nx_i + px * lat
                        if 0 <= lz_i < self.grid_n and 0 <= lx_i < self.grid_n and int(self.occ_state[lz_i, lx_i]) != self.WALL:
                            s_sum += 0.5 * (float(self.surface_map[lz_i, lx_i]) + 1.0)
                            s_cnt += 1
                    far_acc.append(s_sum / float(s_cnt))
            clr[di] = float(free_steps) / float(near_steps)
            if near_acc and far_acc:
                surf_adv[di] = float(np.clip(0.56 * float(np.mean(near_acc)) + 0.44 * float(np.mean(far_acc)), 0.0, 1.0))
            elif near_acc:
                surf_adv[di] = float(np.clip(np.mean(near_acc) * (0.92 if blocked_center else 1.0), 0.0, 1.0))
        return clr, surf_adv

    def has_clear_line_of_fire(
        self,
        start_xz: np.ndarray,
        target_xz: np.ndarray,
        target_clearance_m: float = 0.95,
    ) -> bool:
        """
        Strictly 1st-person POMDP check on remembered `occ_state`: returns `False` if any observed
        `WALL` cell lies along the line of fire between `start_xz` and `target_xz` (excluding the
        target's own bounding radius `target_clearance_m`). Zero ground-truth wallhacks!
        """
        dx = float(target_xz[0] - start_xz[0])
        dz = float(target_xz[1] - start_xz[1])
        dist = math.hypot(dx, dz)
        check_d = dist - max(0.35, float(target_clearance_m))
        if check_d <= 0.30 or dist <= 1e-4:
            return True
        ux = dx / dist
        uz = dz / dist
        n_steps = max(1, int(math.ceil(check_d / 0.38)))
        step_m = check_d / float(n_steps)
        for k in range(1, n_steps + 1):
            s = min(check_d, float(k) * step_m)
            if s < 0.28:
                continue
            iz, ix = self.world_to_cell(float(start_xz[0]) + ux * s, float(start_xz[1]) + uz * s)
            if int(self.occ_state[iz, ix]) == self.WALL:
                return False
        return True

    def _slide_step_around_observed_wall(self, ego_xz: np.ndarray, target_xz: np.ndarray) -> np.ndarray:
        """
        If `ego_xz -> target_xz` crosses an observed `WALL` cell in `occ_state` (e.g. a 1.1m-thick
        baffle wall where `target_xz` lands in open space on the far side), deflect the step laterally
        along the clearer wall tangent so `CharacterController` never rams head-on into solid cover.
        """
        dx = float(target_xz[0] - ego_xz[0])
        dz = float(target_xz[1] - ego_xz[1])
        dist = math.hypot(dx, dz)
        if dist <= 0.22:
            return target_xz
        ux = dx / dist
        uz = dz / dist
        hit_wall = False
        n_steps = max(1, int(math.ceil(dist / 0.35)))
        for k in range(1, n_steps + 1):
            s = float(k) * (dist / float(n_steps))
            iz, ix = self.world_to_cell(float(ego_xz[0]) + ux * s, float(ego_xz[1]) + uz * s)
            if int(self.occ_state[iz, ix]) == self.WALL:
                hit_wall = True
                break
        if not hit_wall:
            return target_xz

        # Evaluate left `(-uz, ux)` vs right `(uz, -ux)` lateral slide clearance on observed `occ_state`
        slide_len = max(1.15, dist)
        tang_l = np.array([-uz, ux], dtype=np.float32)
        tang_r = np.array([uz, -ux], dtype=np.float32)

        def _free_score(t_vec: np.ndarray) -> float:
            sc = 0.0
            for frac in (0.45, 0.85, 1.25):
                px = float(ego_xz[0]) + float(t_vec[0]) * slide_len * frac
                pz = float(ego_xz[1]) + float(t_vec[1]) * slide_len * frac
                if not (float(self.wmin[0]) + 0.7 <= px <= float(self.wmax[0]) - 0.7 and float(self.wmin[2]) + 0.7 <= pz <= float(self.wmax[2]) - 0.7):
                    sc -= 1.5
                    continue
                jz, jx = self.world_to_cell(px, pz)
                occ_v = int(self.occ_state[jz, jx])
                if occ_v == self.WALL:
                    sc -= 2.0
                else:
                    sc += 1.0 + 0.25 * float(self.surface_map[jz, jx])
            return sc

        sc_l = _free_score(tang_l) + (0.15 if self._committed_flank_side < 0 else 0.0)
        sc_r = _free_score(tang_r) + (0.15 if self._committed_flank_side > 0 else 0.0)
        best_tang = tang_l if sc_l >= sc_r else tang_r
        slid_dir = 0.88 * best_tang - 0.12 * np.array([ux, uz], dtype=np.float32)
        slid_dir /= max(1e-4, float(np.linalg.norm(slid_dir)))
        return np.clip(
            np.asarray(ego_xz, dtype=np.float32) + slid_dir * slide_len,
            self.wmin[[0, 2]] + 0.65,
            self.wmax[[0, 2]] - 0.65,
        )

    def smooth_local_step(self, ego_xz: np.ndarray, desired_xz: np.ndarray) -> np.ndarray:
        """Apply soft wall-clearance repulsion so proposed candidate targets never jam into obstacle corners."""
        if _NATIVE_LIB is not None and self.occ_state.flags["C_CONTIGUOUS"]:
            out_xz = np.zeros(2, dtype=np.float32)
            _NATIVE_LIB.ub3d_smooth_local_step(
                _u8_ptr(self.occ_state), int(self.grid_n),
                float(self.cell_w), float(self.cell_d),
                float(self.wmin[0]), float(self.wmin[2]),
                float(self.wmax[0]), float(self.wmax[2]),
                float(ego_xz[0]), float(ego_xz[1]),
                float(desired_xz[0]), float(desired_xz[1]),
                _f32_ptr(out_xz),
            )
            return self._slide_step_around_observed_wall(ego_xz, out_xz)

        out_xz = np.clip(
            np.asarray(desired_xz, dtype=np.float32).copy(),
            self.wmin[[0, 2]] + 0.65,
            self.wmax[[0, 2]] - 0.65,
        )
        iz, ix = self.world_to_cell(float(out_xz[0]), float(out_xz[1]))
        repulse = np.zeros(2, dtype=np.float32)
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                jz, jx = iz + dz, ix + dx
                if 0 <= jz < self.grid_n and 0 <= jx < self.grid_n and int(self.occ_state[jz, jx]) == self.WALL:
                    wc = self.cell_to_world_xz(jz, jx)
                    diff = out_xz - wc
                    d = float(np.linalg.norm(diff))
                    min_clear = 0.75 * max(self.cell_w, self.cell_d) + 0.40
                    if d < min_clear:
                        push = (diff / max(1e-4, d)) if d > 1e-4 else (ego_xz - wc) / max(1e-4, float(np.linalg.norm(ego_xz - wc)))
                        repulse += push * (min_clear - d)
        out_xz = np.clip(out_xz + repulse, self.wmin[[0, 2]] + 0.65, self.wmax[[0, 2]] - 0.65)
        return self._slide_step_around_observed_wall(ego_xz, out_xz)

    def export_bus_arrays(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        occ_u8 = np.zeros((FIELD_MAP_MAX_N, FIELD_MAP_MAX_N), dtype=np.uint8)
        surf_i8 = np.zeros((FIELD_MAP_MAX_N, FIELD_MAP_MAX_N), dtype=np.int8)
        elev_i8 = np.zeros((FIELD_MAP_MAX_N, FIELD_MAP_MAX_N), dtype=np.int8)
        vis_f16 = np.zeros((FIELD_MAP_MAX_N, FIELD_MAP_MAX_N), dtype=np.float16)

        n = self.grid_n
        occ_u8[:n, :n] = self.occ_state
        surf_i8[:n, :n] = np.clip(np.rint(self.surface_map * 100.0), -100, 100).astype(np.int8)
        elev_i8[:n, :n] = np.clip(np.rint(self.elev_map * 25.0), -100, 100).astype(np.int8)
        vis_f16[:n, :n] = self.visit_freshness.astype(np.float16)
        return n, occ_u8, surf_i8, elev_i8, vis_f16


class EntorhinalHippocampal3DEngineV2:
    """
    Upgraded 3D Spatial Cognition Engine (`EntorhinalHippocampal3DEngineV2`):
      1. 256D 3D Face-Centered-Cubic (FCC) Multi-Scale MEC Grid Cells (`4` scales x `4x4x4` phases).
      2. 32D Fibonacci-Sphere 3D Head-Direction Ring (`hd_ring_code`).
      3. Stationary-Landmark-Filtered EKF-SLAM (`SemanticTag.is_static_landmark_for_slam` prevents
         moving opponents, allies, and carried objectives from corrupting self-pose!).
      4. Mid-Episode Respawn Discontinuity Detector: preserves `field_map` & landmark anchors across
         respawns while cleanly resetting EKF drift and clearing transient threat ghosts.
      5. Multi-Timescale Object Working Memory + Directional Acoustic Phantom Threat Slot (`9991`).
      6. Top-Down Modulated Active Saccade / Aim Controller (`topdown_goal_mode` + `lead_aim_xyz`).
    """

    ACOUSTIC_PHANTOM_ID = 9991

    def __init__(self, cfg: Brain3DConfig):
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self.rng = np.random.default_rng(cfg.seed + 31)
        self.wmin = np.array(cfg.world_min, dtype=np.float32)
        self.wmax = np.array(cfg.world_max, dtype=np.float32)
        self.span = np.maximum(1.0, self.wmax - self.wmin)

        self.field_map = MultiLayerCognitiveFieldMap3D(
            self.wmin,
            self.wmax,
            grid_n=FIELD_MAP_MAX_N,
            surface_decay_per_sec=self.profile.surface_staleness_decay_per_sec,
        )
        self.field_map.frac_memory_alpha = float(getattr(cfg, "frac_memory_alpha", 0.52))
        self._gaze_yaw_frac_diff = CaputoFractionalDiffO1(dim=2, alpha=0.68, tau_poles=(0.05, 0.22, 0.90))

        # Precompute 32 Fibonacci-sphere 3D Head-Direction preferred unit vectors
        idx = np.arange(HD_RING_DIM, dtype=np.float32)
        phi_g = (1.0 + math.sqrt(5.0)) * 0.5
        theta = 2.0 * math.pi * idx / phi_g
        z_hd = 1.0 - (2.0 * idx + 1.0) / float(HD_RING_DIM)
        r_xy = np.sqrt(np.maximum(0.0, 1.0 - z_hd * z_hd))
        self.hd_dirs = np.stack([r_xy * np.cos(theta), z_hd, r_xy * np.sin(theta)], axis=-1).astype(np.float32)

        # Precompute 3D FCC lattice basis & 4x4x4 phase offsets
        self.fcc_basis = np.array([
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ], dtype=np.float32) / math.sqrt(3.0)
        p1d = np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False, dtype=np.float32)
        px, py, pz = np.meshgrid(p1d, p1d, p1d, indexing="ij")
        self.mec_phases = np.stack([px.ravel(), py.ravel(), pz.ravel()], axis=-1).astype(np.float32)  # (64, 3)
        self.mec_wavelengths = np.array([1.6, 3.4, 7.2, 15.0], dtype=np.float32)

        self.current_episode_id = -1
        self.current_stage_id = 0
        self.is_learning_context = bool(getattr(cfg, "enable_learning", False))
        self.stage_schema_bank: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self.carried_over_this_episode = False
        self.episodes_carried_over = 0
        self.last_respawn_atom = 0
        self.believed_pos = np.zeros(3, dtype=np.float32)
        self.last_true_pos = np.zeros(3, dtype=np.float32)
        self.pose_cov = 0.05
        self.loop_closures_total = 0
        self.volumetric_fog = np.ones((FOG_GRID_X, FOG_GRID_Y, FOG_GRID_Z), dtype=np.float32)
        self.landmark_anchors: Dict[int, np.ndarray] = {}
        self.memory_slots: Dict[int, Dict[str, Any]] = {}
        self.active_gaze_ypd = np.array([0.0, -0.18, 6.0], dtype=np.float32)
        self.saccade_hold_ticks = 0
        self.scan_hold_ticks = 0
        self.scan_target_yaw = 0.0
        self.slot_foveate_ticks = 0
        self.slot_foveate_cooldown = 0

    def set_learning_context(self, is_learning: bool) -> None:
        self.is_learning_context = bool(is_learning)

    def clear_all_stage_memories(self) -> None:
        self.stage_schema_bank.clear()
        self.current_stage_id = 0
        self.current_episode_id = -1
        self.carried_over_this_episode = False

    def _archive_active_stage_schema(self) -> None:
        if self.current_stage_id <= 0 or self.current_episode_id < 0:
            return
        schema = self.field_map.export_structural_schema()
        if int(schema.get("explored_cells", 0)) < 4 and not self.landmark_anchors:
            return

        # Archive static landmark anchors (in strict POMDP mode, never access ground-truth `last_true_pos`!)
        if bool(getattr(self.cfg, "strict_pomdp_observation", True)):
            drift_offset = np.zeros(3, dtype=np.float32)
        else:
            drift_offset = (self.believed_pos - self.last_true_pos).astype(np.float32)
        prev_entry = self.stage_schema_bank.get(self.current_stage_id)
        merged_anchors: Dict[int, np.ndarray] = {}
        if prev_entry is not None and isinstance(prev_entry.get("landmark_anchors"), dict):
            for eid, apos in prev_entry["landmark_anchors"].items():
                merged_anchors[int(eid)] = np.asarray(apos, dtype=np.float32).copy()
        for eid, apos in self.landmark_anchors.items():
            clean_pos = (np.asarray(apos, dtype=np.float32) - drift_offset).astype(np.float32)
            if int(eid) in merged_anchors:
                merged_anchors[int(eid)] = (0.50 * merged_anchors[int(eid)] + 0.50 * clean_pos).astype(np.float32)
            else:
                merged_anchors[int(eid)] = clean_pos

        if prev_entry is not None and int(prev_entry.get("explored_cells", 0)) > int(schema.get("explored_cells", 0)):
            # Union newly discovered FREE/WALL cells into the existing bank entry without losing prior coverage
            old_occ = prev_entry["occ_state"]
            new_occ = schema["occ_state"]
            keep_old = (new_occ == MultiLayerCognitiveFieldMap3D.UNKNOWN) & (old_occ != MultiLayerCognitiveFieldMap3D.UNKNOWN)
            schema["occ_state"] = np.where(keep_old, old_occ, new_occ)
            schema["wall_score"] = np.where(keep_old, prev_entry["wall_score"], schema["wall_score"])
            schema["elev_map"] = np.where(keep_old, prev_entry["elev_map"], schema["elev_map"])
            schema["explored_cells"] = int(np.sum(schema["occ_state"] != MultiLayerCognitiveFieldMap3D.UNKNOWN))

        schema["landmark_anchors"] = merged_anchors
        if self.current_stage_id in self.stage_schema_bank:
            self.stage_schema_bank.move_to_end(self.current_stage_id)
        self.stage_schema_bank[self.current_stage_id] = schema
        while len(self.stage_schema_bank) > 8:
            self.stage_schema_bank.popitem(last=False)

    def reset_for_episode(self, episode_id: int, init_pos: np.ndarray, init_yp: np.ndarray, stage_id: int = 0) -> None:
        self._archive_active_stage_schema()

        eff_stage = int(stage_id) if int(stage_id) > 0 else int(getattr(self.cfg, "default_stage_id", 1))
        mem_mode = str(getattr(self.cfg, "hippo_memory_mode", HIPPO_MEM_ADAPTIVE)).strip().lower()
        bank_entry = self.stage_schema_bank.get(eff_stage) if eff_stage > 0 else None

        do_carry = False
        if bank_entry is not None and mem_mode != HIPPO_MEM_RESET_ALWAYS:
            if mem_mode == HIPPO_MEM_CARRY_SAME_STAGE:
                do_carry = True
            else:
                # HIPPO_MEM_ADAPTIVE: 100% carry in inference/spectate; stochastic mix in training for unseen-stage generalization!
                if not self.is_learning_context:
                    do_carry = True
                else:
                    p_carry = float(np.clip(getattr(self.cfg, "hippo_train_carry_prob", 0.50), 0.0, 1.0))
                    do_carry = bool(self.rng.random() < p_carry)

        self.current_episode_id = int(episode_id)
        self.current_stage_id = eff_stage
        self.believed_pos = np.asarray(init_pos, dtype=np.float32).copy()
        self.last_true_pos = self.believed_pos.copy()
        self.pose_cov = 0.04
        self.memory_slots.clear()

        if do_carry and bank_entry is not None:
            self.carried_over_this_episode = True
            self.episodes_carried_over += 1
            self.field_map.restore_structural_schema_for_new_episode(bank_entry, self.believed_pos)
            self.landmark_anchors = {
                int(eid): np.asarray(apos, dtype=np.float32).copy()
                for eid, apos in bank_entry.get("landmark_anchors", {}).items()
            }
            # Structural skeleton known, but dynamic enemy/ink positions still require active gaze scanning!
            self.volumetric_fog.fill(0.48)
        else:
            self.carried_over_this_episode = False
            self.volumetric_fog.fill(0.85)
            self.landmark_anchors.clear()
            self.field_map.reset_full_episode()

        self.active_gaze_ypd = np.array([float(init_yp[0]), float(init_yp[1]), 6.0], dtype=np.float32)
        self.saccade_hold_ticks = 0
        self.scan_hold_ticks = 0
        self.scan_target_yaw = float(init_yp[0])
        self.slot_foveate_ticks = 0
        self.slot_foveate_cooldown = 0

    def _encode_mec_grid_256(self, pos3: np.ndarray) -> np.ndarray:
        if self.cfg.spatial_mode == SPATIAL_MODE_TUNNEL_RAW:
            return np.zeros(MEC_GRID_DIM, dtype=np.float32)
        proj = pos3 @ self.fcc_basis.T  # (3,)
        out = np.empty(MEC_GRID_DIM, dtype=np.float32)
        for s_idx, wl in enumerate(self.mec_wavelengths):
            k = (2.0 * math.pi / float(wl)) * proj[None, :]  # (1, 3)
            act = (np.cos(k[:, 0:1] - self.mec_phases[:, 0:1])
                   + np.cos(k[:, 1:2] - self.mec_phases[:, 1:2])
                   + np.cos(k[:, 2:3] - self.mec_phases[:, 2:3])) / 3.0
            out[s_idx * 64 : (s_idx + 1) * 64] = np.maximum(0.0, act.ravel())
        return out

    def _encode_hd_ring_32(self, yaw: float, pitch: float) -> np.ndarray:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        g_vec = np.array([sy * cp, sp, cy * cp], dtype=np.float32)
        dots = self.hd_dirs @ g_vec
        return np.exp(3.5 * (dots - 1.0)).astype(np.float32)

    def step(
        self,
        world_rec: Any,
        dt: float,
        fovea_override: Optional[np.ndarray] = None,
        ventral_override: Optional[np.ndarray] = None,
        epistemic_uncert_16: Optional[np.ndarray] = None,
        respawn_atom: int = 0,
        topdown_goal_mode: int = GOAL_MODE_BALANCED_EXPLORE,
        topdown_focus_xyz: Optional[np.ndarray] = None,
        lead_aim_xyz: Optional[np.ndarray] = None,
        vis_slot_coords: Optional[np.ndarray] = None,
        vis_slot_valence: Optional[np.ndarray] = None,
        vis_pred_surprise: float = 0.0,
        world_surprise: float = 0.0,
        gwt_ignition: float = 0.0,
    ) -> Dict[str, Any]:
        ep_id = int(world_rec["episode_id"])
        raw_stage_id = 0
        if isinstance(world_rec, dict):
            raw_stage_id = int(world_rec.get("stage_id", 0))
        elif hasattr(world_rec, "dtype") and getattr(world_rec.dtype, "names", None) and "stage_id" in world_rec.dtype.names:
            raw_stage_id = int(world_rec["stage_id"])
        eff_stage_id = raw_stage_id if raw_stage_id > 0 else int(getattr(self.cfg, "default_stage_id", 1))

        true_pos = np.asarray(world_rec["self_pos"], dtype=np.float32)
        true_vel = np.asarray(world_rec["self_vel"], dtype=np.float32)
        true_yp = np.asarray(world_rec["self_yaw_pitch"], dtype=np.float32)
        proprio = np.asarray(world_rec["proprio"], dtype=np.float32)

        if ep_id != self.current_episode_id or (self.current_stage_id > 0 and eff_stage_id != self.current_stage_id):
            self.reset_for_episode(ep_id, true_pos, true_yp, stage_id=eff_stage_id)
            self.last_respawn_atom = int(respawn_atom)

        # 1. Mid-Episode Respawn Discontinuity Detection (preserves stage/turf map!)
        pos_jump = float(np.linalg.norm(true_pos - self.last_true_pos))
        respawn_prop = float(proprio[PROPRIO_IDX_TRANSIT_RESPAWN]) >= 0.90 if proprio.shape[0] > PROPRIO_IDX_TRANSIT_RESPAWN else False
        atom_changed = int(respawn_atom) != int(self.last_respawn_atom)
        if bool(getattr(self.cfg, "strict_pomdp_observation", True)):
            is_respawn = bool(atom_changed or respawn_prop)
        else:
            is_respawn = bool((atom_changed or respawn_prop or pos_jump > 5.5))
        self.last_respawn_atom = int(respawn_atom)
        self.last_true_pos = true_pos.copy()

        if is_respawn:
            self.believed_pos = true_pos.copy()
            self.pose_cov = 0.04
            self.slot_foveate_ticks = 0
            self.slot_foveate_cooldown = 0
            self.field_map.on_mid_episode_respawn(true_pos)
            # Clear transient short-lived hostile slots while keeping landmarks & objective
            stale_ids = [
                eid for eid, slot in self.memory_slots.items()
                if SemanticTag.is_hostile(slot["tag"]) or eid == self.ACOUSTIC_PHANTOM_ID
            ]
            for eid in stale_ids:
                self.memory_slots.pop(eid, None)
        else:
            # 2. Noisy Dead-Reckoning Path Integration + Proprioceptive Anchor Bound
            if self.cfg.spatial_mode in (SPATIAL_MODE_TUNNEL_RAW, SPATIAL_MODE_OMNI_DEBUG):
                self.believed_pos = true_pos.copy()
                self.pose_cov = 0.01
            else:
                odom_sigma = 0.035 * float(self.cfg.odom_noise_scale) * math.sqrt(max(1e-4, dt))
                drift_noise = self.rng.normal(0.0, odom_sigma, size=3).astype(np.float32)
                drift_noise[1] *= 0.25
                self.believed_pos = np.clip(self.believed_pos + true_vel * float(dt) + drift_noise, self.wmin, self.wmax)
                self.pose_cov = min(2.5, self.pose_cov + 0.08 * float(dt))

        # 3. Stationary-Landmark-Filtered EKF-SLAM Loop Closure + Multi-Timescale Object Working Memory
        vis_cnt = min(MAX_ENTITIES, int(world_rec["visible_count"]))
        vis_ents = np.asarray(world_rec["visible_entities"], dtype=np.float32)
        pose_offset = self.believed_pos - true_pos

        for i in range(vis_cnt):
            row = vis_ents[i]
            eid = int(round(float(row[0])))
            if eid <= 0:
                continue
            rel_obs = row[1:4] - true_pos
            obs_bel_pos = self.believed_pos + rel_obs
            vel3 = row[4:7].copy()
            spd = float(np.linalg.norm(vel3))
            rad = float(row[7])
            tag = float(row[8])
            conf = float(np.clip(row[9], 0.15, 1.0))

            # EKF-SLAM loop closure ONLY on genuinely stationary landmarks!
            if (
                self.cfg.spatial_mode == SPATIAL_MODE_FULL_MEC_3D
                and SemanticTag.is_static_landmark_for_slam(tag, spd)
            ):
                if eid not in self.landmark_anchors:
                    self.landmark_anchors[eid] = obs_bel_pos.copy()
                else:
                    anchor_pos = self.landmark_anchors[eid]
                    innov = anchor_pos - obs_bel_pos
                    innov_norm = float(np.linalg.norm(innov))
                    if innov_norm < 3.5:
                        k_gain = float(self.pose_cov / (self.pose_cov + 0.18 / max(0.2, conf)))
                        if innov_norm > 0.12 and k_gain > 0.15:
                            self.loop_closures_total += 1
                        self.believed_pos = self.believed_pos + k_gain * innov
                        self.pose_cov = max(0.02, (1.0 - k_gain) * self.pose_cov)
                        self.landmark_anchors[eid] = 0.96 * anchor_pos + 0.04 * (self.believed_pos + rel_obs)
                obs_bel_pos = self.believed_pos + rel_obs

            self.memory_slots[eid] = {
                "id": eid,
                "pos": obs_bel_pos.copy(),
                "vel": vel3,
                "radius": rad,
                "tag": tag,
                "conf": conf,
                "init_conf": conf,
                "disconfirmed": False,
                "unseen_sec": 0.0,
            }

        # Update unseen slots with v2.3.2 Mittag-Leffler fractional persistence, 1st-person visual disconfirmation & extrapolation
        frac_alpha = float(getattr(self.cfg, "frac_memory_alpha", 0.52))
        self.field_map.frac_memory_alpha = frac_alpha
        active_fovea = fovea_override if fovea_override is not None else np.asarray(world_rec["fovea_patch"], dtype=np.float32)
        cy_cur, sy_cur = math.cos(float(true_yp[0])), math.sin(float(true_yp[0]))
        cp_cur, sp_cur = math.cos(float(true_yp[1])), math.sin(float(true_yp[1]))
        g_fwd_cur = np.array([sy_cur * cp_cur, sp_cur, cy_cur * cp_cur], dtype=np.float32)
        cam_right_cur = np.array([cy_cur, 0.0, -sy_cur], dtype=np.float32)
        cam_up_cur = np.cross(g_fwd_cur, cam_right_cur).astype(np.float32)
        th_fov_cur = max(1e-3, math.tan(math.radians(float(self.cfg.fovea_half_angle_deg))))
        cos_fov_inner = math.cos(math.radians(float(self.cfg.fovea_half_angle_deg) * 0.82))
        max_fov_rng = float(self.cfg.fovea_max_range_m)

        to_prune: List[int] = []
        for eid, slot in self.memory_slots.items():
            if slot["unseen_sec"] > 0.0:
                tag = float(slot["tag"])
                if SemanticTag.is_hostile(tag):
                    decay = max(1.85, self.profile.opponent_memory_decay_per_sec) if eid == self.ACOUSTIC_PHANTOM_ID else self.profile.opponent_memory_decay_per_sec
                    visually_disconfirmed = bool(slot.get("disconfirmed", False))
                    rel_s = slot["pos"] - self.believed_pos
                    dist_s = float(np.linalg.norm(rel_s))
                    if 0.60 <= dist_s <= max_fov_rng - 1.0:
                        cos_s = float(np.dot(rel_s / dist_s, g_fwd_cur))
                        if cos_s >= cos_fov_inner:
                            denom_s = max(0.25, float(np.dot(rel_s, g_fwd_cur)))
                            u_n = float(np.dot(rel_s, cam_right_cur)) / (denom_s * th_fov_cur)
                            v_n = float(np.dot(rel_s, cam_up_cur)) / (denom_s * th_fov_cur)
                            if abs(u_n) <= 0.88 and abs(v_n) <= 0.88:
                                c_px = int(np.clip(round(0.5 * (u_n + 1.0) * (FOVEA_W - 1)), 0, FOVEA_W - 1))
                                r_px = int(np.clip(round(0.5 * (1.0 - v_n) * (FOVEA_H - 1)), 0, FOVEA_H - 1))
                                ray_d_m = (1.0 - float(np.clip(active_fovea[r_px, c_px, 0], 0.0, 1.0))) * max_fov_rng
                                if ray_d_m > dist_s + 0.75:
                                    visually_disconfirmed = True
                                    slot["disconfirmed"] = True
                                    decay = max(decay, 5.50)
                    if (not visually_disconfirmed) and eid != self.ACOUSTIC_PHANTOM_ID and slot["unseen_sec"] <= 0.65:
                        slot["pos"] = np.clip(slot["pos"] + slot["vel"] * float(dt), self.wmin + 0.5, self.wmax - 0.5)
                    if (not visually_disconfirmed) and eid != self.ACOUSTIC_PHANTOM_ID and slot["unseen_sec"] <= 12.0:
                        init_c = float(slot.get("init_conf", slot["conf"]))
                        slot["conf"] = init_c * eval_mittag_leffler_survival(float(slot["unseen_sec"]), decay, alpha=frac_alpha)
                    else:
                        slot["conf"] = float(slot["conf"]) * math.exp(-max(decay, 5.50 if visually_disconfirmed else decay) * float(dt))
                elif SemanticTag.is_primary_objective(tag):
                    slot["conf"] = float(slot["conf"]) * math.exp(-self.profile.objective_memory_decay_per_sec * float(dt))
                else:
                    slot["conf"] = float(slot["conf"]) * math.exp(-0.012 * float(dt))
                if slot["conf"] < 0.08:
                    to_prune.append(eid)
            slot["unseen_sec"] = float(slot["unseen_sec"]) + float(dt)
        for eid in to_prune:
            self.memory_slots.pop(eid, None)

        # Synthesize Acoustic Phantom Threat (`9991`) when unseen directional sound is loud
        audio_loud = float(proprio[PROPRIO_IDX_AUDIO_LOUD]) if proprio.shape[0] > PROPRIO_IDX_AUDIO_LOUD else 0.0
        has_visible_hostile = any(
            SemanticTag.is_hostile(s["tag"]) and s["unseen_sec"] <= float(dt) * 1.5
            for s in self.memory_slots.values()
        )
        if audio_loud >= 0.18 and (not has_visible_hostile):
            a_sin = float(proprio[PROPRIO_IDX_AUDIO_SIN])
            a_cos = float(proprio[PROPRIO_IDX_AUDIO_COS])
            world_ang = float(true_yp[0]) + math.atan2(a_sin, a_cos)
            est_dist = float(np.clip((1.0 - audio_loud) * 12.0 + 2.2, 2.0, 13.0))
            ph_pos = np.array([
                float(np.clip(self.believed_pos[0] + math.sin(world_ang) * est_dist, self.wmin[0] + 0.8, self.wmax[0] - 0.8)),
                float(self.believed_pos[1]),
                float(np.clip(self.believed_pos[2] + math.cos(world_ang) * est_dist, self.wmin[2] + 0.8, self.wmax[2] - 0.8)),
            ], dtype=np.float32)
            self.memory_slots[self.ACOUSTIC_PHANTOM_ID] = {
                "id": self.ACOUSTIC_PHANTOM_ID,
                "pos": ph_pos,
                "vel": np.zeros(3, dtype=np.float32),
                "radius": 0.85,
                "tag": SemanticTag.OPPONENT_AGENT,
                "conf": float(np.clip(0.35 + 0.55 * audio_loud, 0.25, 0.88)),
                "unseen_sec": float(dt),
            }

        # 4. Integrate 1st-Person 4-Layer Field Map & 3D Volumetric Fog
        friendly_ratio, enemy_ratio = self.field_map.integrate_first_person_frame(
            believed_pos=self.believed_pos,
            gaze_ypd=np.array([float(true_yp[0]), float(true_yp[1]), float(self.active_gaze_ypd[2])], dtype=np.float32),
            fovea_patch=active_fovea,
            proprio=proprio,
            fovea_half_angle_deg=self.cfg.fovea_half_angle_deg,
            max_range_m=self.cfg.fovea_max_range_m,
            dt=dt,
            ventral_patch=ventral_override,
            epistemic_uncert_16=epistemic_uncert_16,
        )

        # Update 8x4x8 volumetric fog + Top-Down GWT/RSSM Epistemic Vigilance Projection (never touches occ/surface/SLAM!)
        self.volumetric_fog = np.minimum(1.0, self.volumetric_fog + 0.018 * float(dt))
        cy, sy = math.cos(float(true_yp[0])), math.sin(float(true_yp[0]))
        cp, sp = math.cos(float(true_yp[1])), math.sin(float(true_yp[1]))
        g_fwd = np.array([sy * cp, sp, cy * cp], dtype=np.float32)

        w_surp_c = float(np.clip(world_surprise, 0.0, 1.5)) if math.isfinite(float(world_surprise)) else 0.0
        v_surp_f = float(np.clip(vis_pred_surprise, 0.0, 1.5)) if math.isfinite(float(vis_pred_surprise)) else 0.0
        ign_c = float(np.clip(gwt_ignition, 0.0, 1.0)) if math.isfinite(float(gwt_ignition)) else 0.0
        td_vigilance = float(np.clip(0.55 * w_surp_c + 0.45 * v_surp_f, 0.0, 1.2)) * (0.35 + 0.65 * ign_c)
        if td_vigilance > 0.04 and topdown_focus_xyz is not None and float(np.linalg.norm(topdown_focus_xyz)) > 1e-4:
            foc_p = np.asarray(topdown_focus_xyz, dtype=np.float32)
            rel_foc = foc_p - self.believed_pos
            d_foc = float(np.linalg.norm(rel_foc))
            in_clear_center = (d_foc > 0.35 and float(np.dot(rel_foc / d_foc, g_fwd)) > 0.92 and d_foc < 8.5)
            if not in_clear_center:
                fnorm = (foc_p - self.wmin) / self.span
                if 0.0 <= float(fnorm[0]) <= 1.0 and 0.0 <= float(fnorm[2]) <= 1.0:
                    fvx = min(FOG_GRID_X - 1, max(0, int(float(fnorm[0]) * FOG_GRID_X)))
                    fvz = min(FOG_GRID_Z - 1, max(0, int(float(fnorm[2]) * FOG_GRID_Z)))
                    fog_boost = float(0.28 * td_vigilance * min(1.0, float(dt) * 15.0))
                    for dx_f in (-1, 0, 1):
                        for dz_f in (-1, 0, 1):
                            jx = fvx + dx_f
                            jz = fvz + dz_f
                            if 0 <= jx < FOG_GRID_X and 0 <= jz < FOG_GRID_Z:
                                w_ker = 1.0 if (dx_f == 0 and dz_f == 0) else 0.55
                                self.volumetric_fog[jx, :, jz] = np.minimum(1.0, self.volumetric_fog[jx, :, jz] + fog_boost * w_ker)
                    if topdown_goal_mode in (GOAL_MODE_EVADE_REGROUP, GOAL_MODE_STEALTH_FLANK) and float(world_rec["hazard_level"]) >= 0.25:
                        iz_c, ix_c = self.field_map.world_to_cell(float(foc_p[0]), float(foc_p[2]))
                        self.field_map.strategic_hazard_scar[iz_c, ix_c] = min(
                            1.4, float(self.field_map.strategic_hazard_scar[iz_c, ix_c]) + 0.15 * td_vigilance * min(1.0, float(dt) * 15.0)
                        )

        for step_r in (1.5, 3.5, 6.0, 9.0):
            sample_p = self.believed_pos + g_fwd * step_r
            norm_v = (sample_p - self.wmin) / self.span
            if np.all((norm_v >= 0.0) & (norm_v <= 1.0)):
                vx = min(FOG_GRID_X - 1, int(norm_v[0] * FOG_GRID_X))
                vy = min(FOG_GRID_Y - 1, int(norm_v[1] * FOG_GRID_Y))
                vz = min(FOG_GRID_Z - 1, int(norm_v[2] * FOG_GRID_Z))
                self.volumetric_fog[vx, vy, vz] = max(0.0, float(self.volumetric_fog[vx, vy, vz]) - 0.35)

        # 5. Export sorted believed_entities & believed_hazard
        sorted_slots = sorted(
            self.memory_slots.values(),
            key=lambda s: (1 if SemanticTag.is_hostile(s["tag"]) else 0, float(s["conf"])),
            reverse=True,
        )[:MAX_ENTITIES]
        bel_arr = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
        believed_hazard = float(world_rec["hazard_level"])
        nearest_hostile_pos: Optional[np.ndarray] = None
        nearest_hostile_d = 999.0

        for idx_s, s in enumerate(sorted_slots):
            bel_arr[idx_s] = np.array([
                float(s["id"]), s["pos"][0], s["pos"][1], s["pos"][2],
                s["vel"][0], s["vel"][1], s["vel"][2], float(s["radius"]), float(s["tag"]), float(s["conf"]),
            ], dtype=np.float32)
            if SemanticTag.is_hostile(s["tag"]):
                d_h = float(np.linalg.norm(s["pos"] - self.believed_pos))
                if d_h < nearest_hostile_d:
                    nearest_hostile_d = d_h
                    nearest_hostile_pos = s["pos"].copy()
                if SemanticTag.is_area_hazard(s["tag"]) and d_h < 4.5:
                    believed_hazard = max(
                        believed_hazard,
                        float(np.clip((0.58 + 0.42 * (4.5 - d_h) / 4.5) * float(s["conf"]), 0.0, 1.0)),
                    )
                elif d_h < 9.0:
                    believed_hazard = max(believed_hazard, float(np.clip((9.0 - d_h) / 9.0 * float(s["conf"]), 0.0, 1.0)))

        # 5.5 Top-Down Retinotopic Object-Slot Unprojection (`SlotObjectCortex3D` -> Hippocampus / Superior Colliculus)
        slot_dirs_xz: List[Tuple[float, float, float]] = []
        best_slot_foveate_xyz: Optional[np.ndarray] = None
        best_slot_contrast = -1.0
        vis_surp_c = float(np.clip(vis_pred_surprise, 0.0, 1.5)) if math.isfinite(float(vis_pred_surprise)) else 0.0
        if self.slot_foveate_cooldown > 0:
            self.slot_foveate_cooldown = max(0, self.slot_foveate_cooldown - (2 if vis_surp_c >= 0.18 else 1))

        if vis_slot_coords is not None and vis_slot_valence is not None:
            sc_arr = np.asarray(vis_slot_coords, dtype=np.float32)
            sv_arr = np.asarray(vis_slot_valence, dtype=np.float32)
            if sc_arr.shape == (NUM_OBJECT_SLOTS, 3) and sv_arr.shape == (NUM_OBJECT_SLOTS, 3):
                if topdown_goal_mode == GOAL_MODE_PUSH_OBJECTIVE:
                    s_sal = 0.75 * sv_arr[:, 0] + 1.15 * sv_arr[:, 1]
                elif topdown_goal_mode in (GOAL_MODE_EVADE_REGROUP, GOAL_MODE_STEALTH_FLANK):
                    s_sal = 0.85 * sv_arr[:, 0] + 0.55 * sv_arr[:, 1] + 0.45 * sv_arr[:, 2]
                else:
                    s_sal = 1.15 * sv_arr[:, 0] + 0.80 * sv_arr[:, 1]
                mean_slot_sal = float(np.mean(s_sal))

                th_fov = math.tan(math.radians(float(self.cfg.fovea_half_angle_deg)))
                cam_right = np.array([cy, 0.0, -sy], dtype=np.float32)
                cam_up = np.cross(g_fwd, cam_right).astype(np.float32)
                max_rng = float(self.cfg.fovea_max_range_m)
                fov_contrast_thr = 0.08 if vis_surp_c >= 0.16 else 0.12
                fov_val_thr = 0.62 if vis_surp_c >= 0.16 else 0.70

                for k_sl in range(NUM_OBJECT_SLOTS):
                    contrast_k = float(s_sal[k_sl]) - mean_slot_sal
                    norm_d_k = float(np.clip(0.5 * (float(sc_arr[k_sl, 2]) + 1.0), 0.04, 0.96))
                    if contrast_k < 0.06 or norm_d_k < 0.06:
                        continue
                    u_k = float(np.clip(sc_arr[k_sl, 0], -1.0, 1.0))
                    v_k = float(np.clip(sc_arr[k_sl, 1], -1.0, 1.0))
                    dist_m_k = float(np.clip((1.0 - norm_d_k) * max_rng, 0.8, max_rng))
                    ray_k = g_fwd + (u_k * th_fov) * cam_right + (-v_k * th_fov) * cam_up
                    rn_k = float(np.linalg.norm(ray_k))
                    if rn_k <= 1e-6:
                        continue
                    ray_k = ray_k / rn_k
                    s_xyz = np.clip(self.believed_pos + ray_k * dist_m_k, self.wmin + 0.4, self.wmax - 0.4).astype(np.float32)

                    rel_xz = s_xyz[[0, 2]] - self.believed_pos[[0, 2]]
                    rel_n = float(np.linalg.norm(rel_xz))
                    if rel_n > 0.25:
                        surp_gain = 1.0 + 0.80 * min(1.0, vis_surp_c)
                        slot_dirs_xz.append((float(rel_xz[0] / rel_n), float(rel_xz[1] / rel_n), min(0.85, contrast_k * surp_gain)))

                    v_host = float(sv_arr[k_sl, 0])
                    v_obj = float(sv_arr[k_sl, 1])
                    if v_host >= 0.72 and contrast_k >= 0.12 and dist_m_k < 8.0:
                        believed_hazard = max(believed_hazard, float(np.clip(0.35 * v_host * (8.0 - dist_m_k) / 8.0, 0.0, 0.55)))

                    if max(v_host, v_obj) >= fov_val_thr and contrast_k >= fov_contrast_thr and contrast_k > best_slot_contrast:
                        best_slot_contrast = contrast_k
                        best_slot_foveate_xyz = s_xyz

        # 6. Top-Down Modulated Active Saccade / Inhibition-of-Return Epistemic Gaze Search
        mean_fog = float(np.mean(self.volumetric_fog))
        focus_target: Optional[np.ndarray] = None
        if lead_aim_xyz is not None and float(np.linalg.norm(lead_aim_xyz)) > 1e-4 and (
            topdown_goal_mode == GOAL_MODE_ENGAGE_INTERCEPT or nearest_hostile_d <= self.profile.ranged_engage_max_m * 1.15
        ):
            focus_target = np.asarray(lead_aim_xyz, dtype=np.float32)
            self.slot_foveate_ticks = 0
        elif nearest_hostile_pos is not None and nearest_hostile_d <= self.profile.ranged_engage_max_m:
            focus_target = nearest_hostile_pos
            self.slot_foveate_ticks = 0
        elif best_slot_foveate_xyz is not None and mean_fog <= (0.82 if vis_surp_c >= 0.16 else 0.72) and (self.slot_foveate_cooldown <= 0 or vis_surp_c >= 0.24):
            if self.slot_foveate_ticks < 5:
                focus_target = best_slot_foveate_xyz
                self.slot_foveate_ticks += 1
            else:
                self.slot_foveate_ticks = 0
                self.slot_foveate_cooldown = 16
        else:
            self.slot_foveate_ticks = 0

        saccade_hold = 0
        if focus_target is not None:
            rel_f = focus_target - self.believed_pos
            d_f = max(0.5, float(np.linalg.norm(rel_f)))
            des_yaw = math.atan2(float(rel_f[0]), float(rel_f[2]))
            des_pitch = float(np.clip(math.asin(np.clip(float(rel_f[1]) / d_f, -0.85, 0.85)), -0.75, 0.65))
            if abs(des_pitch) < 0.05 and d_f > 3.0:
                des_pitch = -0.10
            self.active_gaze_ypd = np.array([des_yaw, des_pitch, min(d_f, self.cfg.fovea_max_range_m)], dtype=np.float32)
            self.scan_target_yaw = des_yaw
        else:
            # Superior-Colliculus Inhibition-of-Return (IOR): scan uninspected 3D volumetric-fog sectors!
            if mean_fog > 0.68 and self.saccade_hold_ticks <= 0:
                self.saccade_hold_ticks = 4
            if self.saccade_hold_ticks > 0:
                self.saccade_hold_ticks -= 1
                saccade_hold = 1

            if self.scan_hold_ticks <= 0:
                fr_dir, _, _ = self.field_map.find_best_frontier_or_turf_target(self.believed_pos[[0, 2]])
                td_off_xz = np.zeros(2, dtype=np.float32)
                if topdown_focus_xyz is not None and float(np.linalg.norm(topdown_focus_xyz)) > 1e-4:
                    raw_off = np.asarray(topdown_focus_xyz[[0, 2]], dtype=np.float32) - self.believed_pos[[0, 2]]
                    n_off = float(np.linalg.norm(raw_off))
                    if n_off > 0.25:
                        td_off_xz = raw_off / n_off

                best_sec_score = -1e9
                best_yaw = float(self.active_gaze_ypd[0])
                for k_sec in range(8):
                    ang = -math.pi + (float(k_sec) + 0.5) * (math.pi / 4.0)
                    dx_s, dz_s = math.sin(ang), math.cos(ang)
                    fog_acc = 0.0
                    valid_samples = 0
                    for r_s in (3.2, 6.8):
                        sp_x = float(self.believed_pos[0]) + dx_s * r_s
                        sp_z = float(self.believed_pos[2]) + dz_s * r_s
                        nx_f = (sp_x - float(self.wmin[0])) / float(self.span[0])
                        nz_f = (sp_z - float(self.wmin[2])) / float(self.span[2])
                        if 0.02 <= nx_f <= 0.98 and 0.02 <= nz_f <= 0.98:
                            vx = min(FOG_GRID_X - 1, int(nx_f * FOG_GRID_X))
                            vz = min(FOG_GRID_Z - 1, int(nz_f * FOG_GRID_Z))
                            fog_acc += float(np.mean(self.volumetric_fog[vx, :, vz]))
                            valid_samples += 1
                    if valid_samples == 0:
                        continue
                    fog_mean_s = fog_acc / float(valid_samples)
                    align_fr = dx_s * float(fr_dir[0]) + dz_s * float(fr_dir[1])
                    align_td = dx_s * float(td_off_xz[0]) + dz_s * float(td_off_xz[1])
                    slot_sec_bonus = 0.0
                    for sdx, sdz, sw in slot_dirs_xz:
                        slot_sec_bonus = max(slot_sec_bonus, max(0.0, dx_s * sdx + dz_s * sdz) * sw)
                    cur_diff = abs(math.atan2(math.sin(ang - float(self.active_gaze_ypd[0])), math.cos(ang - float(self.active_gaze_ypd[0]))))
                    sec_score = 1.25 * fog_mean_s + 0.35 * align_fr + (0.25 + 0.55 * td_vigilance) * align_td + 0.35 * slot_sec_bonus - 0.12 * cur_diff
                    if sec_score > best_sec_score:
                        best_sec_score = sec_score
                        best_yaw = ang
                self.scan_target_yaw = best_yaw
                self.scan_hold_ticks = 18
            else:
                self.scan_hold_ticks -= 1

            cur_y = float(self.active_gaze_ypd[0])
            yaw_cs = np.array([math.cos(cur_y), math.sin(cur_y)], dtype=np.float64)
            d_cs = self._gaze_yaw_frac_diff.step(yaw_cs, dt)
            frac_yaw_rate = float(-yaw_cs[1] * float(d_cs[0]) + yaw_cs[0] * float(d_cs[1]))
            dyaw = math.atan2(math.sin(self.scan_target_yaw - cur_y), math.cos(self.scan_target_yaw - cur_y))
            pd_step = float(np.clip(dyaw - 0.035 * frac_yaw_rate, -0.22, 0.22))
            next_yaw = math.atan2(math.sin(cur_y + pd_step), math.cos(cur_y + pd_step))
            self.active_gaze_ypd = np.array([next_yaw, -0.20, 7.0], dtype=np.float32)

        body_yaw = math.atan2(float(true_vel[0]), float(true_vel[2])) if float(np.linalg.norm(true_vel[[0, 2]])) > 0.25 else float(self.active_gaze_ypd[0])
        ego_aim_off = np.array([
            math.atan2(math.sin(float(self.active_gaze_ypd[0]) - body_yaw), math.cos(float(self.active_gaze_ypd[0]) - body_yaw)),
            float(self.active_gaze_ypd[1]),
        ], dtype=np.float32)

        drift_vec = self.believed_pos - true_pos
        map_iou = float(np.clip(1.0 - float(np.linalg.norm(drift_vec)) / 4.0, 0.0, 1.0))

        return {
            "believed_self_pos": self.believed_pos.copy(),
            "pose_drift_vec": drift_vec.astype(np.float32),
            "active_gaze_ypd": self.active_gaze_ypd.copy(),
            "ego_aim_offset_yp": ego_aim_off,
            "believed_hazard": float(np.clip(believed_hazard, 0.0, 1.0)),
            "map_accuracy_iou": map_iou,
            "mean_fog": mean_fog,
            "friendly_turf_ratio": friendly_ratio,
            "enemy_turf_ratio": enemy_ratio,
            "believed_count": len(sorted_slots),
            "believed_entities": bel_arr,
            "loop_closures_total": int(self.loop_closures_total),
            "saccade_hold_active": int(saccade_hold),
            "respawn_event_flag": 1 if is_respawn else 0,
            "mec_grid_code": self._encode_mec_grid_256(self.believed_pos),
            "hd_ring_code": self._encode_hd_ring_32(float(self.active_gaze_ypd[0]), float(self.active_gaze_ypd[1])),
            "volumetric_fog": self.volumetric_fog.ravel().copy(),
        }


def compute_universal_memory_compass(
    believed_pos: np.ndarray,
    believed_entities: np.ndarray,
    believed_count: int,
    proprio: np.ndarray,
    field_map: MultiLayerCognitiveFieldMap3D,
    profile: TaskProfile3D,
) -> np.ndarray:
    """
    Compute the 24D Allocentric-to-Egocentric Memory Compass (`mem_compass[24]`):
      - `[0:4]`  : `[dx, dz, norm_d, valid]` Dijkstra tactical step toward Active Primary Goal
                   (Attack Goal Pedestal if carrying objective; Enemy Carrier if enemy holds objective;
                    else Neutral Primary Objective if known!)
      - `[4:8]`  : `[dx, dz, norm_d, valid]` Frontier / Unpainted Turf expansion direction
      - `[8:12]` : `[dx, dz, norm_d, valid]` Nearest Cover Node OR Friendly Turf Refuge
      - `[12:16]`: `[dx, dz, norm_d, valid]` Highest-priority Believed Threat (visual or acoustic)
      - `[16:20]`: `[clr_N, clr_E, clr_S, clr_W]` local structural clearance in `[0, 1]`
      - `[20:24]`: `[surf_N, surf_E, surf_S, surf_W]` directional friendly surface share in `[0, 1]`
    """
    mc = np.zeros(MEM_COMPASS_DIM, dtype=np.float32)
    ego_xz = np.asarray(believed_pos[[0, 2]], dtype=np.float32)
    carrying = is_carrying_primary_objective(proprio)
    max_diag = float(math.hypot(float(field_map.span[0]), float(field_map.span[2])))

    best_threat_xz: Optional[np.ndarray] = None
    best_threat_score = -1.0
    best_cover_xz: Optional[np.ndarray] = None
    best_cover_d = 999.0
    obj_xz: Optional[np.ndarray] = None
    atk_goal_xz: Optional[np.ndarray] = None
    enemy_carrier_xz: Optional[np.ndarray] = None

    n_ent = min(MAX_ENTITIES, int(believed_count))
    for i in range(n_ent):
        ent = believed_entities[i]
        tag = float(ent[8])
        conf = float(ent[9])
        if conf <= 0.05:
            continue
        exz = ent[[1, 3]]
        d_m = float(np.linalg.norm(exz - ego_xz))

        if SemanticTag.is_hostile(tag):
            prio_mult = 2.4 if SemanticTag.is_area_hazard(tag) else (1.6 if SemanticTag.is_high_priority_threat(tag) else 1.0)
            prio = prio_mult * conf / (1.0 + 0.18 * d_m)
            if prio > best_threat_score:
                best_threat_score = prio
                best_threat_xz = exz.copy()
            if SemanticTag.is_enemy_carrier_or_shot(tag) and ent[7] >= 0.60:
                enemy_carrier_xz = exz.copy()
        elif SemanticTag.is_cover_or_refuge(tag):
            if d_m < best_cover_d:
                best_cover_d = d_m
                best_cover_xz = exz.copy()
        elif SemanticTag.is_primary_objective(tag):
            obj_xz = exz.copy()
        elif SemanticTag.is_attack_goal(tag):
            atk_goal_xz = exz.copy()

    # [12:16] Threat vector
    if best_threat_xz is not None:
        tv = best_threat_xz - ego_xz
        td = float(np.linalg.norm(tv))
        if td > 1e-4:
            mc[MC_THR_DX : MC_THR_DZ + 1] = tv / td
        mc[MC_THR_DIST] = float(np.clip(td / max_diag, 0.0, 1.0))
        mc[MC_THR_VALID] = 1.0

    # [0:4] Active Primary Goal Dijkstra route vector
    active_goal_xz = atk_goal_xz if carrying else (enemy_carrier_xz if enemy_carrier_xz is not None else obj_xz)
    if active_goal_xz is not None:
        g_dir, g_nd, g_ok = field_map.find_tactical_next_step(ego_xz, active_goal_xz, threat_xz=best_threat_xz)
        if g_ok:
            mc[MC_OBJ_DX : MC_OBJ_DZ + 1] = g_dir
            mc[MC_OBJ_DIST] = g_nd
            mc[MC_OBJ_VALID] = 1.0

    # [4:8] Frontier / Turf expansion vector
    f_dir, f_nd, f_ok = field_map.find_best_frontier_or_turf_target(
        ego_xz,
        unknown_weight=profile.frontier_unknown_weight,
        turf_weight=profile.frontier_turf_weight,
    )
    if f_ok:
        mc[MC_FRONT_DX : MC_FRONT_DZ + 1] = f_dir
        mc[MC_FRONT_DIST] = f_nd
        mc[MC_FRONT_VALID] = 1.0

    # [8:12] Cover / Friendly Turf Refuge vector
    ref_dir, ref_nd, ref_ok = field_map.find_nearest_friendly_refuge(ego_xz)
    if best_cover_xz is not None and (not ref_ok or (best_cover_d / max_diag) < ref_nd):
        cv = best_cover_xz - ego_xz
        cd = float(np.linalg.norm(cv))
        if cd > 1e-4:
            mc[MC_COV_DX : MC_COV_DZ + 1] = cv / cd
        mc[MC_COV_DIST] = float(np.clip(cd / max_diag, 0.0, 1.0))
        mc[MC_COV_VALID] = 1.0
    elif ref_ok:
        mc[MC_COV_DX : MC_COV_DZ + 1] = ref_dir
        mc[MC_COV_DIST] = ref_nd
        mc[MC_COV_VALID] = 1.0

    # [16:24] 4-way clearance & surface advantage
    clr4, surf4 = field_map.directional_clearance_and_surface(ego_xz)
    mc[MC_CLR_N : MC_CLR_W + 1] = clr4
    mc[MC_SURF_N : MC_SURF_W + 1] = surf4
    return mc


class Cerebellar3DKinematicPredictorV2:
    """
    Forward Cerebellar Internal Model (`Cerebellar3DKinematicPredictorV2` [v2.3.2 Fractional FO-PD]):
      1. Computes analytic 3D ballistic projectile lead-interception `lead_aim_xyz` for moving targets:
         solves `||p_target + v_eff * t_hit - p_self|| = v_projectile * t_hit` where `v_eff` blends
         instantaneous velocity with a `K=3` Caputo fractional position derivative (`mu = 0.72`)
         to suppress high-frequency strafe-juking / feint overshoot while preserving phase lead!
      2. Simulates a `2.0s` forward kinematic rollout of believed dynamic entities to quantify:
         - `scene_opportunity_prob` (clean line-of-fire / objective window alignment)
         - `local_kinetic_risk` (incoming projectile / opponent collision crossfire risk)
    """

    def __init__(self, cfg: Brain3DConfig):
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self._ent_frac_filters: Dict[int, Tuple[CaputoFractionalDiffO1, int, int]] = {}
        self._step_seq: int = 0

    def reset_filters(self) -> None:
        self._ent_frac_filters.clear()
        self._step_seq = 0

    def predict_lead_and_crossfire(
        self,
        believed_self_pos: np.ndarray,
        believed_entities: np.ndarray,
        believed_count: int,
        dt: float = 1.0 / 60.0,
    ) -> Tuple[np.ndarray, float, float, float]:
        """Returns `(lead_aim_xyz, scene_opportunity_prob, local_kinetic_risk, max_scene_speed)`."""
        self._step_seq += 1
        dt_c = float(np.clip(dt, 1.0 / 240.0, 0.20))
        frac_mu = float(np.clip(0.55 + 0.30 * float(getattr(self.cfg, "frac_memory_alpha", 0.52)), 0.60, 0.84))
        v_proj = max(4.0, float(self.profile.projectile_speed_mps))
        lead_xyz = np.zeros(3, dtype=np.float32)
        best_target_score = -1.0
        local_risk = 0.0
        opp_prob = 0.15
        max_spd = 0.0

        n = min(MAX_ENTITIES, int(believed_count))
        for i in range(n):
            ent = believed_entities[i]
            conf = float(ent[9])
            if conf <= 0.08:
                continue
            eid = int(round(float(ent[0])))
            pos = ent[1:4]
            vel = ent[4:7]
            spd = float(np.linalg.norm(vel))
            max_spd = max(max_spd, spd)
            tag = float(ent[8])

            if SemanticTag.is_hostile(tag):
                rel = pos - believed_self_pos
                dist = float(np.linalg.norm(rel))
                if SemanticTag.is_area_hazard(tag):
                    if dist < 4.2:
                        local_risk = max(local_risk, float(np.clip((4.2 - dist) / 2.6 * conf, 0.0, 1.0)))
                    continue

                # Fractional-Order FO-PD^mu anti-juke velocity estimation (exact `vel` on 1st observation)
                eff_vel = vel
                if eid > 0 and eid != EntorhinalHippocampal3DEngineV2.ACOUSTIC_PHANTOM_ID:
                    entry = self._ent_frac_filters.get(eid)
                    if entry is None or (self._step_seq - entry[2]) > 18:
                        flt = CaputoFractionalDiffO1(dim=3, alpha=frac_mu, tau_poles=(0.12, 0.42, 1.50))
                        flt.reset(pos)
                        self._ent_frac_filters[eid] = (flt, 1, self._step_seq)
                    else:
                        flt, obs_cnt, _ = entry
                        flt.set_alpha(frac_mu)
                        d_mu_pos = flt.step(pos, dt_c)
                        # Reset on teleport/respawn jump (`> 3.5m` in one step)
                        if float(np.linalg.norm(d_mu_pos)) > 35.0:
                            flt.reset(pos)
                            self._ent_frac_filters[eid] = (flt, 1, self._step_seq)
                        else:
                            obs_cnt_next = min(64, obs_cnt + 1)
                            self._ent_frac_filters[eid] = (flt, obs_cnt_next, self._step_seq)
                            if obs_cnt_next >= 3:
                                w_frac = float(np.clip((obs_cnt_next - 2) * 0.24, 0.0, 0.72))
                                v_frac = np.clip(d_mu_pos * math.gamma(2.0 - frac_mu), -14.0, 14.0)
                                eff_vel = ((1.0 - w_frac) * vel + w_frac * v_frac).astype(np.float32)

                # Iterative ballistic interception time `t_hit` + parabolic gravity compensation
                t_hit = dist / v_proj
                for _ in range(2):
                    pred_p = pos + eff_vel * t_hit
                    t_hit = float(np.linalg.norm(pred_p - believed_self_pos)) / v_proj
                t_hit = float(np.clip(t_hit, 0.0, 1.15))
                cand_lead = (pos + eff_vel * t_hit).copy()
                g_proj = float(getattr(self.profile, "projectile_gravity_mps2", 0.0))
                if g_proj > 0.0:
                    cand_lead[1] += min(1.05, 0.5 * g_proj * t_hit * t_hit)

                # Evaluate closest approach over next 1.2s for crossfire risk
                v_sq = float(np.dot(vel, vel))
                t_cpa = float(np.clip(-np.dot(rel, vel) / max(1e-4, v_sq), 0.0, 1.2)) if v_sq > 0.05 else 0.0
                d_cpa = float(np.linalg.norm(rel + vel * t_cpa))
                if d_cpa < 4.5:
                    local_risk = max(local_risk, float(np.clip((4.5 - d_cpa) / 4.5 * conf, 0.0, 1.0)))

                if dist <= self.profile.ranged_engage_max_m * 1.15:
                    t_score = (1.5 if SemanticTag.is_high_priority_threat(tag) else 1.0) * conf / (1.0 + 0.12 * dist)
                    if t_score > best_target_score:
                        best_target_score = t_score
                        lead_xyz = cand_lead.astype(np.float32)
                    opt_delta = abs(dist - self.profile.ranged_optimal_m)
                    opp_prob = max(opp_prob, float(np.clip((1.0 - opt_delta / 8.0) * conf, 0.15, 0.95)))
            elif SemanticTag.is_primary_objective(tag) and float(ent[7]) >= 1.05:
                d_obj = float(np.linalg.norm(pos - believed_self_pos))
                if d_obj <= self.profile.ranged_engage_max_m:
                    opp_prob = max(
                        opp_prob,
                        float(np.clip((self.profile.ranged_engage_max_m + 1.0 - d_obj) / (self.profile.ranged_engage_max_m + 1.0) * conf, 0.25, 0.94)),
                    )

        if len(self._ent_frac_filters) > 64:
            stale_eids = [k for k, v in self._ent_frac_filters.items() if (self._step_seq - v[2]) > 30]
            for k in stale_eids:
                self._ent_frac_filters.pop(k, None)

        return lead_xyz, float(opp_prob), float(local_risk), float(max_spd)


def propose_universal_3d_candidates(
    believed_pos: np.ndarray,
    believed_entities: np.ndarray,
    believed_count: int,
    proprio: np.ndarray,
    mem_compass: np.ndarray,
    field_map: MultiLayerCognitiveFieldMap3D,
    lead_aim_xyz: np.ndarray,
    active_gaze_ypd: np.ndarray,
    world_min: Tuple[float, float, float],
    world_max: Tuple[float, float, float],
    profile: TaskProfile3D,
    goal_mode: int = GOAL_MODE_BALANCED_EXPLORE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate `M = 8` authoritative, domain-agnostic 3D tactical action candidates with
    DECOUPLED body movement (`target_pose[0:3]`) vs. weapon/camera aim (`target_pose[3:5]`):
      - `M0`: Objective / Tactical Dijkstra Route Advance (`CommitInteract` if in reach, else `ApproachHover`)
      - `M1`: Frontier / Surface Turf Expansion (`ReleasePlace` paint-ahead or `ApproachHover`)
      - `M2`: Strafe-Right + Ballistic Lead-Aim Engage (`ReleasePlace` / `ApproachHover`)
      - `M3`: Strafe-Left / Flank + Ballistic Lead-Aim (`ReleasePlace` / `ApproachHover`)
      - `M4`: Tactical Retreat / Friendly-Turf or Cover Dive (`EvadeHazard`)
      - `M5`: High-Impact Burst / Pickup / Countermeasure (`ImpulsePush` or `GraspPick`)
      - `M6`: Team Rally / Defend Base Pedestal (`CommunicatePing` or `ApproachHover`)
      - `M7`: Submerge-in-Place / Refuge Regen & Observe (`IdleObserve`)
    """
    wmin = np.asarray(world_min, dtype=np.float32)
    wmax = np.asarray(world_max, dtype=np.float32)
    mid = 0.5 * (wmin + wmax)
    half = np.maximum(0.5, 0.5 * (wmax - wmin))

    ego_xz = np.asarray(believed_pos[[0, 2]], dtype=np.float32)
    ego_y = float(believed_pos[1])
    max_diag = float(math.hypot(float(wmax[0] - wmin[0]), float(wmax[2] - wmin[2])))
    tank = get_resource_tank_level(proprio)
    carrying = is_carrying_primary_objective(proprio)
    can_commit_prop = float(proprio[PROPRIO_IDX_CAN_COMMIT]) > 0.5 if proprio.shape[0] > PROPRIO_IDX_CAN_COMMIT else True
    has_ammo = can_commit_prop if carrying else (tank >= profile.min_resource_to_fire)
    on_friendly_surf = proprio.shape[0] > PROPRIO_IDX_SURFACE_POSTURE and float(proprio[PROPRIO_IDX_SURFACE_POSTURE]) >= 0.68
    has_lead = float(np.linalg.norm(lead_aim_xyz)) > 1e-4

    def _to_norm_pos3(xz: np.ndarray, y_val: float = ego_y) -> np.ndarray:
        sm_xz = field_map.smooth_local_step(ego_xz, xz)
        p3 = np.array([sm_xz[0], y_val, sm_xz[1]], dtype=np.float32)
        return np.clip((p3 - mid) / half, -1.0, 1.0)

    def _aim_yp_toward(target_xyz: Optional[np.ndarray], fallback_dir_xz: np.ndarray) -> Tuple[float, float]:
        if target_xyz is not None and float(np.linalg.norm(target_xyz)) > 1e-4:
            rel = target_xyz - believed_pos
            d = max(0.35, float(np.linalg.norm(rel)))
            return math.atan2(float(rel[0]), float(rel[2])), float(np.clip(math.asin(np.clip(float(rel[1]) / d, -0.85, 0.85)), -0.75, 0.65))
        if float(np.linalg.norm(fallback_dir_xz)) > 1e-4:
            return math.atan2(float(fallback_dir_xz[0]), float(fallback_dir_xz[1])), -0.14
        return float(active_gaze_ypd[0]), float(active_gaze_ypd[1])

    poses = np.zeros((NUM_CANDIDATES, CONT_POSE_DIM), dtype=np.float32)
    verbs = np.zeros((NUM_CANDIDATES,), dtype=np.uint8)

    # Scan believed entities for primary objective, attack/defend pedestals, threats, pickups, and allies
    pickup_xz: Optional[np.ndarray] = None
    defend_xz: Optional[np.ndarray] = None
    atk_goal_xz: Optional[np.ndarray] = None
    enemy_carrier_xz: Optional[np.ndarray] = None
    thr_pos_xz: Optional[np.ndarray] = None
    best_thr_prio = -1.0
    ally_xz: Optional[np.ndarray] = None
    obj_xyz: Optional[np.ndarray] = None
    obj_radius = 0.50
    for i in range(min(MAX_ENTITIES, int(believed_count))):
        ent = believed_entities[i]
        conf_i = float(ent[9])
        if conf_i <= 0.08:
            continue
        t = float(ent[8])
        if SemanticTag.is_hostile(t):
            exz = ent[[1, 3]]
            dm_i = float(np.linalg.norm(exz - ego_xz))
            prio_mult_i = 2.4 if SemanticTag.is_area_hazard(t) else (1.6 if SemanticTag.is_high_priority_threat(t) else 1.0)
            prio_i = prio_mult_i * conf_i / (1.0 + 0.18 * dm_i)
            if prio_i > best_thr_prio:
                best_thr_prio = prio_i
                thr_pos_xz = exz.copy()
            if SemanticTag.is_enemy_carrier_or_shot(t) and float(ent[7]) >= 0.60 and enemy_carrier_xz is None:
                enemy_carrier_xz = exz.copy()
        elif SemanticTag.is_primary_objective(t) and obj_xyz is None:
            obj_xyz = ent[1:4].copy()
            obj_radius = float(ent[7])
        elif SemanticTag.is_attack_goal(t) and atk_goal_xz is None:
            atk_goal_xz = ent[[1, 3]].copy()
        elif SemanticTag.is_resource_pickup(t) and pickup_xz is None:
            pickup_xz = ent[[1, 3]].copy()
        elif SemanticTag.is_defend_pedestal(t) and defend_xz is None:
            defend_xz = ent[[1, 3]].copy()
        elif SemanticTag.is_teammate(t) and ally_xz is None:
            ally_xz = ent[[1, 3]].copy()

    # Extract compass directions (named MC_* constants — zero raw index magic numbers!)
    obj_dir = mem_compass[MC_OBJ_DX : MC_OBJ_DZ + 1] if float(mem_compass[MC_OBJ_VALID]) > 0.5 else mem_compass[MC_FRONT_DX : MC_FRONT_DZ + 1]
    if float(np.linalg.norm(obj_dir)) < 1e-4:
        obj_dir = np.array([0.0, 1.0], dtype=np.float32)
    front_dir = mem_compass[MC_FRONT_DX : MC_FRONT_DZ + 1] if float(mem_compass[MC_FRONT_VALID]) > 0.5 else obj_dir
    cov_dir = mem_compass[MC_COV_DX : MC_COV_DZ + 1] if float(mem_compass[MC_COV_VALID]) > 0.5 else -obj_dir
    thr_valid = float(mem_compass[MC_THR_VALID]) > 0.5
    thr_dir = mem_compass[MC_THR_DX : MC_THR_DZ + 1] if thr_valid else obj_dir
    d_thr_m = float(mem_compass[MC_THR_DIST]) * max_diag if thr_valid else 999.0
    d_obj_m = float(np.linalg.norm(obj_xyz[[0, 2]] - ego_xz)) if obj_xyz is not None else 999.0

    # M0: Direct Objective / Pedestal Route Advance — keeps safe ~4.25m standoff ONLY when line-of-fire is clear!
    # (If a Stage-2 baffle wall occludes `obj_xyz` in `field_map.occ_state`, follow Dijkstra `obj_dir` around the wall!)
    d_goal_m = float(mem_compass[MC_OBJ_DIST]) * max_diag if float(mem_compass[MC_OBJ_VALID]) > 0.5 else profile.step_move_m
    obj_los_clear = (
        field_map.has_clear_line_of_fire(ego_xz, obj_xyz[[0, 2]], target_clearance_m=max(0.85, obj_radius + 0.25))
        if obj_xyz is not None
        else True
    )
    shielded_obj = (not carrying) and (enemy_carrier_xz is None) and (obj_xyz is not None) and (obj_radius >= 1.05)
    if shielded_obj and obj_los_clear:
        to_sh_xz = obj_xyz[[0, 2]] - ego_xz
        d_sh = float(np.linalg.norm(to_sh_xz))
        u_sh = (to_sh_xz / d_sh) if d_sh > 1e-4 else obj_dir
        standoff_delta0 = d_sh - 4.25
        step0 = float(np.clip(standoff_delta0, -1.80, profile.step_move_m))
        m0_xz = ego_xz + (obj_dir if standoff_delta0 > 1.20 else u_sh) * step0
        m0_yaw, m0_pitch = _aim_yp_toward(obj_xyz, u_sh)
        poses[CAND_IDX_OBJECTIVE_ADVANCE, 0:3] = _to_norm_pos3(m0_xz)
        poses[CAND_IDX_OBJECTIVE_ADVANCE, 3:6] = [m0_yaw, m0_pitch, 0.90]
        if has_ammo and d_sh <= profile.ranged_engage_max_m:
            verbs[CAND_IDX_OBJECTIVE_ADVANCE] = VERB_RELEASE_PLACE
        elif on_friendly_surf and standoff_delta0 > 1.80 and not thr_valid:
            verbs[CAND_IDX_OBJECTIVE_ADVANCE] = VERB_EVADE_HAZARD
        else:
            verbs[CAND_IDX_OBJECTIVE_ADVANCE] = VERB_APPROACH_HOVER
    else:
        m0_step = float(np.clip(max(d_goal_m, 1.35 if (shielded_obj and not obj_los_clear) else 0.0), 0.25, profile.step_move_m))
        m0_xz = ego_xz + obj_dir * m0_step
        m0_yaw, m0_pitch = _aim_yp_toward(lead_aim_xyz if has_lead else (obj_xyz if obj_los_clear else None), obj_dir)
        poses[CAND_IDX_OBJECTIVE_ADVANCE, 0:3] = _to_norm_pos3(m0_xz)
        poses[CAND_IDX_OBJECTIVE_ADVANCE, 3:6] = [m0_yaw, m0_pitch, 0.94]
        if on_friendly_surf and m0_step > 1.20 and not thr_valid:
            verbs[CAND_IDX_OBJECTIVE_ADVANCE] = VERB_EVADE_HAZARD
        elif shielded_obj and (not obj_los_clear) and has_ammo and (not on_friendly_surf):
            poses[CAND_IDX_OBJECTIVE_ADVANCE, 4] = -0.22
            verbs[CAND_IDX_OBJECTIVE_ADVANCE] = VERB_RELEASE_PLACE
        elif float(mem_compass[MC_OBJ_VALID]) > 0.5 and d_goal_m <= 1.15 and not on_friendly_surf and not shielded_obj:
            verbs[CAND_IDX_OBJECTIVE_ADVANCE] = VERB_COMMIT_INTERACT
        else:
            verbs[CAND_IDX_OBJECTIVE_ADVANCE] = VERB_APPROACH_HOVER

    # M1: Turf / Epistemic Frontier Expansion + Forward Paint
    d_front_m = float(mem_compass[MC_FRONT_DIST]) * max_diag if float(mem_compass[MC_FRONT_VALID]) > 0.5 else profile.step_move_m
    m1_step = float(np.clip(d_front_m, 0.45, profile.step_move_m))
    m1_xz = ego_xz + front_dir * m1_step
    m1_yaw, m1_pitch = _aim_yp_toward(None, front_dir)
    poses[CAND_IDX_FRONTIER_TURF, 0:3] = _to_norm_pos3(m1_xz)
    poses[CAND_IDX_FRONTIER_TURF, 3:6] = [m1_yaw, -0.22 if has_ammo else m1_pitch, 0.85]
    verbs[CAND_IDX_FRONTIER_TURF] = VERB_RELEASE_PLACE if (has_ammo and float(np.min(mem_compass[MC_SURF_N : MC_SURF_W + 1])) < 0.65) else VERB_APPROACH_HOVER

    # M2: Strafe-Right + Independent Ballistic Lead-Aim Shot!
    perp_r = np.array([thr_dir[1], -thr_dir[0]], dtype=np.float32)
    if float(np.linalg.norm(perp_r)) < 1e-4:
        perp_r = np.array([1.0, 0.0], dtype=np.float32)
    m2_xz = ego_xz + perp_r * (0.85 * profile.step_move_m)
    m2_yaw, m2_pitch = _aim_yp_toward(lead_aim_xyz if has_lead else obj_xyz, thr_dir)
    poses[CAND_IDX_STRAFE_RIGHT_ENGAGE, 0:3] = _to_norm_pos3(m2_xz)
    poses[CAND_IDX_STRAFE_RIGHT_ENGAGE, 3:6] = [m2_yaw, m2_pitch, 0.88]
    verbs[CAND_IDX_STRAFE_RIGHT_ENGAGE] = VERB_RELEASE_PLACE if (has_ammo and (has_lead or thr_valid or obj_xyz is not None)) else VERB_APPROACH_HOVER

    # M3: Wing / Off-Axis Flank Detour + Fast Stealth Swim or Ballistic Lead-Aim!
    flank_goal_xz = (
        atk_goal_xz
        if (carrying and atk_goal_xz is not None)
        else (enemy_carrier_xz if enemy_carrier_xz is not None else (obj_xyz[[0, 2]] if obj_xyz is not None else (ego_xz + obj_dir * 8.0)))
    )
    if thr_valid and d_thr_m <= 5.2:
        perp_l = -perp_r
        flank_dir = 0.72 * perp_l + 0.28 * obj_dir
        flank_dir /= max(1e-4, float(np.linalg.norm(flank_dir)))
    else:
        flank_dir, _, _ = field_map.find_flank_detour_next_step(ego_xz, flank_goal_xz, obj_dir, threat_xz=thr_pos_xz)
    step_l = profile.evade_move_m if on_friendly_surf else (profile.stealth_move_m if goal_mode == GOAL_MODE_STEALTH_FLANK else profile.step_move_m)
    m3_xz = ego_xz + flank_dir * step_l
    wn3, we3, ws3, ww3 = max(0.0, float(flank_dir[1])), max(0.0, float(flank_dir[0])), max(0.0, -float(flank_dir[1])), max(0.0, -float(flank_dir[0]))
    ws3_sum = max(1e-4, wn3 + we3 + ws3 + ww3)
    flank_surf_adv = (
        wn3 * float(mem_compass[MC_SURF_N])
        + we3 * float(mem_compass[MC_SURF_E])
        + ws3 * float(mem_compass[MC_SURF_S])
        + ww3 * float(mem_compass[MC_SURF_W])
    ) / ws3_sum
    if has_ammo and has_lead and (goal_mode == GOAL_MODE_ENGAGE_INTERCEPT or d_thr_m <= 7.2):
        m3_yaw, m3_pitch = _aim_yp_toward(lead_aim_xyz, thr_dir)
        m3_spd = 0.86
        m3_verb = VERB_RELEASE_PLACE
    elif on_friendly_surf:
        m3_yaw, m3_pitch = _aim_yp_toward(lead_aim_xyz if has_lead else None, thr_dir if thr_valid else flank_dir)
        m3_spd = 0.96
        m3_verb = VERB_EVADE_HAZARD
    elif has_ammo and (not carrying) and d_thr_m > 8.2 and flank_surf_adv < 0.52:
        m3_yaw, _ = _aim_yp_toward(None, flank_dir)
        m3_pitch = -0.22
        m3_spd = 0.86
        m3_verb = VERB_RELEASE_PLACE
    else:
        m3_yaw, m3_pitch = _aim_yp_toward(lead_aim_xyz if has_lead else None, thr_dir if thr_valid else flank_dir)
        m3_spd = 0.85
        m3_verb = VERB_APPROACH_HOVER
    poses[CAND_IDX_FLANK_DETOUR, 0:3] = _to_norm_pos3(m3_xz)
    poses[CAND_IDX_FLANK_DETOUR, 3:6] = [m3_yaw, m3_pitch, m3_spd]
    verbs[CAND_IDX_FLANK_DETOUR] = m3_verb

    # M4: Cover / Friendly-Turf Evade & Retreat (clamped to refuge distance when safe so it never oscillates!)
    d_ref_m = float(mem_compass[MC_COV_DIST]) * max_diag if float(mem_compass[MC_COV_VALID]) > 0.5 else profile.evade_move_m
    if (not thr_valid) and float(mem_compass[MC_COV_VALID]) > 0.5 and d_ref_m < 0.30:
        m4_xz = ego_xz.copy()
        escape_dir = -obj_dir
    else:
        escape_dir = cov_dir if (float(mem_compass[MC_COV_VALID]) > 0.5 and float(np.linalg.norm(cov_dir)) > 1e-4) else (-thr_dir if thr_valid else -obj_dir)
        if thr_valid and float(np.dot(escape_dir, thr_dir)) > 0.35:
            escape_dir = -thr_dir
        escape_dir = escape_dir / max(1e-4, float(np.linalg.norm(escape_dir)))
        ev_step = profile.evade_move_m if thr_valid else min(profile.evade_move_m, max(0.35, d_ref_m))
        m4_xz = ego_xz + escape_dir * ev_step
    m4_yaw, m4_pitch = _aim_yp_toward(lead_aim_xyz if has_lead else None, escape_dir)
    poses[CAND_IDX_EVADE_REFUGE, 0:3] = _to_norm_pos3(m4_xz)
    poses[CAND_IDX_EVADE_REFUGE, 3:6] = [m4_yaw, m4_pitch, 1.00]
    verbs[CAND_IDX_EVADE_REFUGE] = VERB_EVADE_HAZARD

    # M5: Dedicated Objective Shield-Breaker / Bare-Objective Grasp Rush OR Close-Combat Burst Countermeasure
    close_cqc_threat = thr_valid and (d_thr_m < 5.0) and (d_thr_m + 0.5 < d_obj_m)
    prioritize_obj_m5 = (
        (obj_xyz is not None)
        and (not carrying)
        and (not close_cqc_threat)
        and (
            (obj_radius < 1.05 and d_obj_m <= 14.0)
            or (obj_radius >= 1.05 and has_ammo and d_obj_m <= profile.ranged_engage_max_m)
        )
    )
    if prioritize_obj_m5:
        to_obj = obj_xyz[[0, 2]] - ego_xz
        d_obj = float(np.linalg.norm(to_obj))
        u_obj = (to_obj / d_obj) if d_obj > 1e-4 else obj_dir
        if obj_radius >= 1.05 and has_ammo and obj_los_clear:
            standoff_delta = d_obj - 4.50
            lat_weave = np.array([u_obj[1], -u_obj[0]], dtype=np.float32) * 0.75
            m5_xz = ego_xz + u_obj * float(np.clip(standoff_delta, -1.60, profile.step_move_m)) + lat_weave
            m5_yaw, m5_pitch = _aim_yp_toward(obj_xyz, u_obj)
            poses[CAND_IDX_SHIELD_OR_BURST, 0:3] = _to_norm_pos3(m5_xz)
            poses[CAND_IDX_SHIELD_OR_BURST, 3:6] = [m5_yaw, m5_pitch, 0.88]
            verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_RELEASE_PLACE
        else:
            r_step_dir = obj_dir if float(mem_compass[MC_OBJ_VALID]) > 0.5 else flank_dir
            m5_step = float(np.clip(max(d_obj, 1.50 if not obj_los_clear else 0.0), 0.25, profile.step_move_m))
            m5_xz = ego_xz + r_step_dir * m5_step
            m5_yaw, m5_pitch = _aim_yp_toward(obj_xyz if obj_los_clear else None, r_step_dir)
            poses[CAND_IDX_SHIELD_OR_BURST, 0:3] = _to_norm_pos3(m5_xz)
            poses[CAND_IDX_SHIELD_OR_BURST, 3:6] = [m5_yaw, m5_pitch, 1.00]
            if d_obj <= 1.65 and obj_radius < 1.05:
                verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_GRASP_PICK
            elif on_friendly_surf:
                verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_EVADE_HAZARD
            elif has_ammo and not obj_los_clear:
                poses[CAND_IDX_SHIELD_OR_BURST, 4] = -0.22
                verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_RELEASE_PLACE
            else:
                verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_APPROACH_HOVER
    elif has_ammo and (has_lead or thr_valid):
        m5_xz = ego_xz - 0.35 * thr_dir * profile.step_move_m
        m5_yaw, m5_pitch = _aim_yp_toward(lead_aim_xyz if has_lead else None, thr_dir)
        poses[CAND_IDX_SHIELD_OR_BURST, 0:3] = _to_norm_pos3(m5_xz)
        poses[CAND_IDX_SHIELD_OR_BURST, 3:6] = [m5_yaw, m5_pitch, 0.85]
        verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_IMPULSE_PUSH if tank >= 0.32 else VERB_RELEASE_PLACE
    elif obj_xyz is not None and (not carrying):
        to_obj = obj_xyz[[0, 2]] - ego_xz
        d_obj = float(np.linalg.norm(to_obj))
        u_obj = (to_obj / d_obj) if d_obj > 1e-4 else obj_dir
        if obj_radius >= 1.05 and has_ammo and obj_los_clear:
            standoff_delta = d_obj - 4.80
            lat_weave = np.array([u_obj[1], -u_obj[0]], dtype=np.float32) * 0.75
            m5_xz = ego_xz + u_obj * float(np.clip(standoff_delta, -1.60, profile.step_move_m)) + lat_weave
            m5_yaw, m5_pitch = _aim_yp_toward(obj_xyz, u_obj)
            poses[CAND_IDX_SHIELD_OR_BURST, 0:3] = _to_norm_pos3(m5_xz)
            poses[CAND_IDX_SHIELD_OR_BURST, 3:6] = [m5_yaw, m5_pitch, 0.88]
            verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_RELEASE_PLACE
        else:
            r_step_dir = obj_dir if float(mem_compass[MC_OBJ_VALID]) > 0.5 else flank_dir
            m5_step = float(np.clip(max(d_obj, 1.50 if not obj_los_clear else 0.0), 0.25, profile.step_move_m))
            m5_xz = ego_xz + r_step_dir * m5_step
            m5_yaw, m5_pitch = _aim_yp_toward(obj_xyz if obj_los_clear else None, r_step_dir)
            poses[CAND_IDX_SHIELD_OR_BURST, 0:3] = _to_norm_pos3(m5_xz)
            poses[CAND_IDX_SHIELD_OR_BURST, 3:6] = [m5_yaw, m5_pitch, 1.00]
            if d_obj <= 1.65 and obj_radius < 1.05:
                verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_GRASP_PICK
            elif on_friendly_surf:
                verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_EVADE_HAZARD
            elif has_ammo and not obj_los_clear:
                poses[CAND_IDX_SHIELD_OR_BURST, 4] = -0.22
                verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_RELEASE_PLACE
            else:
                verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_APPROACH_HOVER
    elif has_ammo and carrying:
        m5_xz = ego_xz + obj_dir * (0.75 * profile.step_move_m)
        m5_yaw, m5_pitch = _aim_yp_toward(None, obj_dir)
        poses[CAND_IDX_SHIELD_OR_BURST, 0:3] = _to_norm_pos3(m5_xz)
        poses[CAND_IDX_SHIELD_OR_BURST, 3:6] = [m5_yaw, -0.15, 0.90]
        verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_RELEASE_PLACE
    else:
        p_dir = (pickup_xz - ego_xz) if pickup_xz is not None else front_dir
        p_dir = p_dir / max(1e-4, float(np.linalg.norm(p_dir)))
        poses[CAND_IDX_SHIELD_OR_BURST, 0:3] = _to_norm_pos3(ego_xz + p_dir * profile.step_move_m)
        poses[CAND_IDX_SHIELD_OR_BURST, 3:6] = [math.atan2(float(p_dir[0]), float(p_dir[1])), -0.18, 0.85]
        verbs[CAND_IDX_SHIELD_OR_BURST] = VERB_GRASP_PICK if pickup_xz is not None else VERB_APPROACH_HOVER

    # M6: Tactical Resource/Pickup Route (when visible & base safe) OR Team Rally / Defend Base Pedestal
    if (
        pickup_xz is not None
        and (not carrying)
        and float(proprio[PROPRIO_IDX_OBJ_PROGRESS]) >= 0.45
    ):
        d_pick = float(np.linalg.norm(pickup_xz - ego_xz))
        r_dir, _, _ = field_map.find_tactical_next_step(ego_xz, pickup_xz, threat_xz=None)
        m6_step = float(np.clip(d_pick, 0.25, profile.step_move_m))
        m6_yaw, m6_pitch = _aim_yp_toward(lead_aim_xyz if has_lead else None, thr_dir if thr_valid else r_dir)
        poses[CAND_IDX_RALLY_OR_PICKUP, 0:3] = _to_norm_pos3(ego_xz + r_dir * m6_step)
        poses[CAND_IDX_RALLY_OR_PICKUP, 3:6] = [m6_yaw, m6_pitch, 0.92]
        if d_pick <= 1.50:
            verbs[CAND_IDX_RALLY_OR_PICKUP] = VERB_GRASP_PICK
        elif on_friendly_surf:
            verbs[CAND_IDX_RALLY_OR_PICKUP] = VERB_EVADE_HAZARD
        else:
            verbs[CAND_IDX_RALLY_OR_PICKUP] = VERB_APPROACH_HOVER
    else:
        rally_tgt = ally_xz if ally_xz is not None else defend_xz
        if rally_tgt is not None:
            r_dir, _, _ = field_map.find_tactical_next_step(ego_xz, rally_tgt, threat_xz=None)
        else:
            r_dir = -obj_dir
        poses[CAND_IDX_RALLY_OR_PICKUP, 0:3] = _to_norm_pos3(ego_xz + r_dir * profile.step_move_m)
        poses[CAND_IDX_RALLY_OR_PICKUP, 3:6] = [math.atan2(float(r_dir[0]), float(r_dir[1])), -0.12, 0.85]
        verbs[CAND_IDX_RALLY_OR_PICKUP] = VERB_COMMUNICATE_PING if (ally_xz is not None and float(proprio[PROPRIO_IDX_OBJ_PROGRESS]) < 0.45) else VERB_APPROACH_HOVER

    # M7: Hold / Submerge-in-Place Regen & Observe
    m7_yaw, m7_pitch = _aim_yp_toward(lead_aim_xyz if has_lead else None, thr_dir if thr_valid else front_dir)
    poses[CAND_IDX_IDLE_OBSERVE, 0:3] = np.clip((believed_pos - mid) / half, -1.0, 1.0)
    poses[CAND_IDX_IDLE_OBSERVE, 3:6] = [m7_yaw, m7_pitch, 0.20]
    verbs[CAND_IDX_IDLE_OBSERVE] = VERB_IDLE_OBSERVE

    return poses, verbs


class DopamineSNcTimingPlasticity3D:
    """Eligibility-trace Dopamine SNc plasticity adapting the 4 STN Hyperdirect brake gains online."""

    def __init__(self, stn_stealth_gain: float = 1.0):
        g = float(np.clip(stn_stealth_gain, 0.35, 2.60))
        self.gains = np.array([1.05 * g, 0.90, 0.75, 0.65 * g], dtype=np.float32)
        self.eligibility = np.zeros(4, dtype=np.float32)

    def record_brake_factors(self, crossfire_f: float, lead_sync_f: float, conflict_f: float, fog_f: float, dt: float) -> None:
        factors = np.array([crossfire_f, lead_sync_f, conflict_f, fog_f], dtype=np.float32)
        self.eligibility = 0.92 * self.eligibility + 0.08 * factors * min(1.0, dt * 60.0)

    def apply_td_feedback(self, td_reward: float, hazard_delta: float, learning_enabled: bool = True) -> None:
        if not learning_enabled:
            return
        da_signal = float(np.clip(0.35 * td_reward - 1.25 * hazard_delta, -1.5, 1.5))
        # Positive outcome after cautious hold reinforces active brake factors; unnecessary delay trims them
        self.gains = np.clip(self.gains + 0.015 * da_signal * self.eligibility, 0.25, 2.80)


class BasalGanglia3DTimingEngineV2:
    """
    Basal Ganglia Direct (`Go`) vs. Hyperdirect (`STN-SNr Brake`) Bounded-Urgency Race Model [v2.3.2]:
      - Feeds a `K=4` Caputo Fractional Danger Derivative `D_t^alpha(Hazard)` into `STN-SNr` and
        Emergency Escape so rising multi-scale threat pressure triggers proactive open-space escape
        or deeper cover hold, while subsiding danger (`D_t^alpha < 0`) releases counterattack windows!
      - Guarantees **Instant Escape Exemption**: if `VERB_EVADE_HAZARD` is selected OR the agent is
        exposed in open space under threat (`eff_hazard >= 0.42`), the gate opens immediately (`0 ms` delay)!
      - Supports **Cover/Swim Regen Hold (`HOLD_COVER_STEALTH_REGEN`)**: maintains refuge while tank
        regenerates or heavy crossfire passes outside cover.
      - Synchronizes ballistic ranged shots (`HOLD_OPPORTUNITY_SYNC`) with cerebellar lead windows.
    """

    def __init__(self, cfg: Brain3DConfig):
        self.cfg = cfg
        self.da_plasticity = DopamineSNcTimingPlasticity3D(cfg.stn_stealth_gain)
        self.hazard_frac_diff = CaputoFractionalDiffO1(
            dim=1,
            alpha=float(getattr(cfg, "frac_memory_alpha", 0.52)),
            tau_poles=(0.08, 0.40, 2.00, 9.00),
        )
        self.last_d_alpha_hazard: float = 0.0
        self.current_ep = -1
        self.current_epoch = -1
        self.turn_wait_ms = 0.0
        self.net_accum = 0.0
        self.holds_triggered = 0
        self.opportunities_caught = 0
        self.wait_history_ms: deque = deque(maxlen=64)
        self.reward_while_holding = 0.0
        self.was_holding_this_epoch = False

    def step(
        self,
        *,
        episode_id: int,
        action_epoch: int,
        can_commit_now: bool,
        chosen_cand_idx: int,
        s1_fast_cand_idx: int,
        chosen_verb: int,
        q_chosen: float,
        survival_prob: float,
        scene_opportunity_prob: float,
        local_kinetic_risk: float,
        believed_scene_speed: float,
        mean_fog: float,
        saccade_hold_active: bool,
        respawn_event_flag: bool,
        coach_brake: bool,
        in_refuge: bool,
        tank_level: float,
        hazard_level: float,
        step_reward: float,
        dt: float,
        self_surprise: float = 0.0,
    ) -> Dict[str, Any]:
        composite_threat = max(float(hazard_level), 0.85 * float(local_kinetic_risk))
        if episode_id != self.current_ep:
            self.hazard_frac_diff.set_alpha(float(getattr(self.cfg, "frac_memory_alpha", 0.52)))
            self.hazard_frac_diff.reset(composite_threat)
            self.last_d_alpha_hazard = 0.0
        elif respawn_event_flag:
            self.hazard_frac_diff.reset(composite_threat)
            self.last_d_alpha_hazard = 0.0

        if episode_id != self.current_ep or action_epoch != self.current_epoch:
            if self.current_ep == episode_id and self.turn_wait_ms > 1.0:
                self.wait_history_ms.append(self.turn_wait_ms)
            self.current_ep = int(episode_id)
            self.current_epoch = int(action_epoch)
            self.turn_wait_ms = 0.0
            self.net_accum = 0.0
            self.was_holding_this_epoch = False

        self.hazard_frac_diff.set_alpha(float(getattr(self.cfg, "frac_memory_alpha", 0.52)))
        d_alpha_haz = float(np.clip(self.hazard_frac_diff.step_scalar(composite_threat, dt), -1.5, 1.5))
        self.last_d_alpha_hazard = d_alpha_haz

        t_mode = int(self.cfg.timing_mode)
        if not can_commit_now:
            return self._pack_result(
                episode_id, action_epoch, t_mode, False, HOLD_COOLDOWN_PREP, coach_brake,
                chosen_cand_idx, 0.0, 0.0, 0.0, 1.0, scene_opportunity_prob, local_kinetic_risk, believed_scene_speed,
            )

        # Mode overrides for controlled ablation
        if t_mode == TIMING_MODE_INSTANT:
            return self._pack_result(
                episode_id, action_epoch, t_mode, True, HOLD_GO_CLEAR, coach_brake,
                chosen_cand_idx, 1.5, 0.0, 0.0, 0.5, scene_opportunity_prob, local_kinetic_risk, believed_scene_speed,
            )
        if t_mode == TIMING_MODE_FIXED_SETTLE:
            self.turn_wait_ms += dt * 1000.0
            open_g = self.turn_wait_ms >= 500.0
            return self._pack_result(
                episode_id, action_epoch, t_mode, open_g, HOLD_GO_CLEAR if open_g else HOLD_OPPORTUNITY_SYNC, coach_brake,
                chosen_cand_idx, 0.8, 0.5, self.turn_wait_ms / 500.0, 1.0, scene_opportunity_prob, local_kinetic_risk, believed_scene_speed,
            )

        # Hyperdirect Emergency Escape Exemption (augmented by rising fractional danger `d_alpha_haz > 0` in open space!):
        eff_escape_hazard = float(hazard_level) + (0.28 * max(0.0, d_alpha_haz) if not in_refuge else 0.0)
        if is_bg_emergency_escape_exempt(chosen_verb, in_refuge, eff_escape_hazard, local_kinetic_risk, self.cfg):
            return self._pack_result(
                episode_id, action_epoch, t_mode, True, HOLD_GO_CLEAR, False,
                chosen_cand_idx, 2.2, 0.0, 1.0, 0.45, scene_opportunity_prob, local_kinetic_risk, believed_scene_speed,
            )

        # Compute Direct `Go` drive (+ subsiding-danger counterattack release) vs. 4-factor Hyperdirect `STN` brake
        counter_window_boost = float(np.clip(-0.30 * min(0.0, d_alpha_haz + 0.04), 0.0, 0.24))
        go_drive = float(
            np.clip(
                0.55 + 0.25 * math.tanh(0.35 * q_chosen) + 0.45 * survival_prob + 0.35 * scene_opportunity_prob + counter_window_boost,
                0.15,
                2.25,
            )
        )

        rising_pressure_f = float(np.clip(0.55 * max(0.0, d_alpha_haz), 0.0, 0.45)) if in_refuge else 0.0
        crossfire_f = float(local_kinetic_risk) * (1.25 if in_refuge else 0.40) + rising_pressure_f
        has_ballistic_target = (int(chosen_cand_idx) != int(CAND_IDX_FRONTIER_TURF)) and (
            float(scene_opportunity_prob) > 0.16 or float(local_kinetic_risk) > 0.05
        )
        lead_sync_f = (
            float(np.clip(0.65 - scene_opportunity_prob, 0.0, 0.65))
            if (chosen_verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH) and has_ballistic_target)
            else 0.0
        )
        s1_gwt_conf = 0.45 if (chosen_cand_idx != s1_fast_cand_idx and self.turn_wait_ms < 120.0) else 0.0
        meta_surp_conf = (
            0.35 * float(np.clip(float(self_surprise) - 0.16, 0.0, 0.85))
            if (math.isfinite(float(self_surprise)) and self.turn_wait_ms < 120.0)
            else 0.0
        )
        conflict_f = max(s1_gwt_conf, meta_surp_conf)
        fog_f = float(np.clip(mean_fog - 0.65, 0.0, 0.40)) if saccade_hold_active else 0.0

        self.da_plasticity.record_brake_factors(crossfire_f, lead_sync_f, conflict_f, fog_f, dt)
        g = self.da_plasticity.gains
        stn_brake = float(g[0] * crossfire_f + g[1] * lead_sync_f + g[2] * conflict_f + g[3] * fog_f)

        # Refuge tank recovery hold when submerged/behind cover with low tank
        refuge_hold = in_refuge and (tank_level < 0.55 or hazard_level >= 0.35) and (self.turn_wait_ms < 650.0)
        if refuge_hold:
            stn_brake += 0.65
        if respawn_event_flag and self.turn_wait_ms < 85.0:
            stn_brake += 0.55
        if coach_brake and self.turn_wait_ms < 900.0:
            stn_brake += 0.90
        if t_mode == TIMING_MODE_CAUTIOUS:
            stn_brake *= 1.35

        # Bounded Cisek urgency signal guarantees finite upper bound on deliberation time
        max_wait_ms = 750.0 if in_refuge else 260.0
        urgency = float(np.clip(self.turn_wait_ms / max_wait_ms, 0.0, 1.6))
        release_thresh = 0.72
        integ_rate = 14.0 if scene_opportunity_prob >= 0.50 else 7.5
        self.net_accum = float(np.clip(self.net_accum + (go_drive + 1.15 * urgency - stn_brake) * (dt * integ_rate), 0.0, 1.5))

        gate_open = self.net_accum >= release_thresh or self.turn_wait_ms >= max_wait_ms
        if not gate_open:
            self.turn_wait_ms += dt * 1000.0
            self.reward_while_holding += float(step_reward)
            if not self.was_holding_this_epoch:
                self.holds_triggered += 1
                self.was_holding_this_epoch = True
            if coach_brake:
                h_reason = HOLD_EXTERNAL_COACH_BRAKE
            elif respawn_event_flag and self.turn_wait_ms < 90.0:
                h_reason = HOLD_POST_RESPAWN_REANCHOR
            elif refuge_hold:
                h_reason = HOLD_COVER_STEALTH_REGEN
            elif crossfire_f >= 0.35:
                h_reason = HOLD_KINETIC_CROSSFIRE
            elif lead_sync_f >= 0.25:
                h_reason = HOLD_OPPORTUNITY_SYNC
            elif conflict_f > 0.0:
                h_reason = HOLD_CONFLICT_ARBITRATION
            else:
                h_reason = HOLD_EPISTEMIC_FOG_SACCADE
        else:
            h_reason = HOLD_GO_CLEAR
            if self.was_holding_this_epoch:
                if scene_opportunity_prob >= 0.52:
                    self.opportunities_caught += 1
                self.was_holding_this_epoch = False

        return self._pack_result(
            episode_id, action_epoch, t_mode, gate_open, h_reason, coach_brake,
            chosen_cand_idx, go_drive, stn_brake, urgency, release_thresh,
            scene_opportunity_prob, local_kinetic_risk, believed_scene_speed,
        )

    def _pack_result(
        self,
        ep_id: int,
        epoch: int,
        t_mode: int,
        gate_open: bool,
        h_reason: int,
        coach_brake: bool,
        cand_idx: int,
        go_d: float,
        stn_b: float,
        urg: float,
        thresh: float,
        opp_p: float,
        kin_r: float,
        sc_spd: float,
    ) -> Dict[str, Any]:
        mean_w = float(np.mean(self.wait_history_ms)) if self.wait_history_ms else float(self.turn_wait_ms)
        return {
            "timestamp_ns": time.perf_counter_ns(),
            "episode_id": int(ep_id),
            "action_epoch": int(epoch),
            "timing_mode": int(t_mode),
            "commit_gate_open": 1 if gate_open else 0,
            "hold_reason_id": int(h_reason),
            "coach_brake_active": 1 if coach_brake else 0,
            "target_cand_idx": int(cand_idx),
            "holds_triggered": int(self.holds_triggered),
            "opportunities_caught": int(self.opportunities_caught),
            "go_drive": float(go_d),
            "stn_brake": float(stn_b),
            "urgency_val": float(urg),
            "net_accumulator": float(self.net_accum),
            "release_threshold": float(thresh),
            "scene_opportunity_prob": float(opp_p),
            "local_kinetic_risk": float(kin_r),
            "believed_scene_speed": float(sc_spd),
            "turn_wait_ms": float(self.turn_wait_ms),
            "mean_wait_ms": mean_w,
            "reward_gained_while_holding": float(self.reward_while_holding),
            "stn_gains": self.da_plasticity.gains.copy(),
        }


def run_spatial_timing_3d_process(
    cfg: Brain3DConfig,
    world_shm_name: str,
    spatial_shm_name: str,
    timing_shm_name: str,
    cog_shm_name: str,
    lang_shm_name: str,
    intent_shm_name: str,
    stop_event,
    ready_event,
    vision_cortex_shm_name: str = "",
    field_map_shm_name: str = "",
) -> None:
    """
    Process 3 (`SpatialTimingProcess` — 60 Hz, Strictly Zero-GPU / `Barrier 1`):
      - Runs `EntorhinalHippocampal3DEngineV2` + `MultiLayerCognitiveFieldMap3D`
      - Runs `Cerebellar3DKinematicPredictorV2` (analytic ballistic lead-aim + crossfire risk)
      - Publishes authoritative `mem_compass[24]` and `auth_cand_poses[8, 6]` / `auth_cand_verbs[8]`
        directly on `SPATIAL3D_DTYPE` so `System1Reflex` and `ConsciousGWT` evaluate identical candidates!
      - Runs `BasalGanglia3DTimingEngineV2` (`STN-SNr` Gate) + active gaze saccade publication.
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: torch must NEVER be imported in SpatialTimingProcess!")
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()

    world_bus = SharedSingleRegionBus(world_shm_name, WORLD3D_DTYPE, create=False)
    spatial_bus = SharedSingleRegionBus(spatial_shm_name, SPATIAL3D_DTYPE, create=False)
    timing_bus = SharedSingleRegionBus(timing_shm_name, TIMING3D_DTYPE, create=False)
    cog_bus = SharedSingleRegionBus(cog_shm_name, COGNITIVE3D_DTYPE, create=False)
    lang_bus = SharedSingleRegionBus(lang_shm_name, LANGUAGE3D_DTYPE, create=False)
    intent_bus = SharedMultiChannelIntent3DBus(intent_shm_name, create=False)
    vc_bus = SharedSingleRegionBus(vision_cortex_shm_name, VISION_CORTEX3D_DTYPE, create=False) if vision_cortex_shm_name else None
    fmap_bus = SharedSingleRegionBus(field_map_shm_name, FIELD_MAP3D_DTYPE, create=False) if field_map_shm_name else None

    profile = get_task_profile(cfg.task_profile)
    hippo = EntorhinalHippocampal3DEngineV2(cfg)
    cerebellum = Cerebellar3DKinematicPredictorV2(cfg)
    bg_engine = BasalGanglia3DTimingEngineV2(cfg)

    dt = 1.0 / max(15.0, float(cfg.spatial_hz))
    ventral_prev: Dict[str, Any] = {}
    last_reward_epoch = -1
    last_lead_xyz = np.zeros(3, dtype=np.float32)

    ready_event.set()
    next_t = time.perf_counter()
    try:
        while not stop_event.is_set():
            if not is_windows_pid_alive(orch_ppid):
                stop_event.set()
                break

            w_rec, _ = world_bus.read_latest()
            if w_rec is None:
                time.sleep(0.002)
                continue

            vc_rec = vc_bus.read_latest()[0] if vc_bus is not None else None
            cog_rec, _ = cog_bus.read_latest()
            lng_rec, _ = lang_bus.read_latest()

            learn_on, _ = _read_runtime_learning_flags(cog_bus, cfg)
            hippo.set_learning_context(learn_on)

            fovea_p, uncert_16, _ = _resolve_active_fovea_patch(w_rec, vc_rec, cfg)
            ventral_p = _resolve_active_ventral_patch(w_rec, vc_rec, cfg, prev_state=ventral_prev)
            respawn_atom = world_bus.slot.get_header_atom(3)

            td_goal_mode = int(cog_rec["topdown_goal_mode"]) if cog_rec is not None else GOAL_MODE_BALANCED_EXPLORE
            if cog_rec is not None and int(cog_rec["episode_id"]) == int(w_rec["episode_id"]):
                hippo.field_map.set_strategic_lane_code(int(cog_rec["strategic_lane_code"]))
            prev_bel_xz = hippo.believed_pos[[0, 2]].copy()
            td_focus_xyz = cog_rec["topdown_focus_xyz"].copy() if cog_rec is not None else None
            td_slot_coords = cog_rec["vis_slot_coords"].copy() if cog_rec is not None else None
            td_slot_valence = cog_rec["vis_slot_valence"].copy() if cog_rec is not None else None
            cog_same_ep = cog_rec is not None and int(cog_rec["episode_id"]) == int(w_rec["episode_id"])
            td_vis_surp = float(cog_rec["vis_pred_surprise"]) if cog_same_ep else 0.0
            td_w_surp = float(cog_rec["world_surprise"]) if cog_same_ep else 0.0
            td_ign = float(cog_rec["ignition"]) if cog_same_ep else 0.0

            sp_out = hippo.step(
                w_rec,
                dt,
                fovea_override=fovea_p,
                ventral_override=ventral_p,
                epistemic_uncert_16=uncert_16,
                respawn_atom=respawn_atom,
                topdown_goal_mode=td_goal_mode,
                topdown_focus_xyz=td_focus_xyz,
                lead_aim_xyz=last_lead_xyz,
                vis_slot_coords=td_slot_coords,
                vis_slot_valence=td_slot_valence,
                vis_pred_surprise=td_vis_surp,
                world_surprise=td_w_surp,
                gwt_ignition=td_ign,
            )
            if bool(sp_out.get("respawn_event_flag", False)):
                hippo.field_map.register_strategic_hazard_scar(float(prev_bel_xz[0]), float(prev_bel_xz[1]))

            lead_xyz, opp_prob, kin_risk, scene_spd = cerebellum.predict_lead_and_crossfire(
                sp_out["believed_self_pos"],
                sp_out["believed_entities"],
                sp_out["believed_count"],
                dt=dt,
            )
            last_lead_xyz = lead_xyz.copy()

            proprio = project_pomdp_proprio(
                w_rec["proprio"], sp_out["believed_self_pos"], cfg.world_min, cfg.world_max, cfg
            )
            mc24 = compute_universal_memory_compass(
                sp_out["believed_self_pos"],
                sp_out["believed_entities"],
                sp_out["believed_count"],
                proprio,
                hippo.field_map,
                profile,
            )
            auth_poses, auth_verbs = propose_universal_3d_candidates(
                sp_out["believed_self_pos"],
                sp_out["believed_entities"],
                sp_out["believed_count"],
                proprio,
                mc24,
                hippo.field_map,
                lead_xyz,
                sp_out["active_gaze_ypd"],
                cfg.world_min,
                cfg.world_max,
                profile,
                goal_mode=td_goal_mode,
            )

            spatial_bus.publish_dict({
                "timestamp_ns": time.perf_counter_ns(),
                "physics_tick": int(w_rec["physics_tick"]),
                "episode_id": int(w_rec["episode_id"]),
                "action_epoch": int(w_rec["action_epoch"]),
                "spatial_mode": int(cfg.spatial_mode),
                "believed_count": sp_out["believed_count"],
                "loop_closures_total": sp_out["loop_closures_total"],
                "saccade_hold_active": sp_out["saccade_hold_active"],
                "respawn_event_flag": sp_out["respawn_event_flag"],
                "active_goal_mode": int(td_goal_mode) & 0xFF,
                "pad_sp": 0,
                "believed_self_pos": sp_out["believed_self_pos"],
                "pose_drift_vec": sp_out["pose_drift_vec"],
                "active_gaze_ypd": sp_out["active_gaze_ypd"],
                "ego_aim_offset_yp": sp_out["ego_aim_offset_yp"],
                "lead_aim_xyz": lead_xyz,
                "believed_hazard": sp_out["believed_hazard"],
                "map_accuracy_iou": sp_out["map_accuracy_iou"],
                "mean_fog": sp_out["mean_fog"],
                "friendly_turf_ratio": sp_out["friendly_turf_ratio"],
                "enemy_turf_ratio": sp_out["enemy_turf_ratio"],
                "mec_grid_code": sp_out["mec_grid_code"],
                "hd_ring_code": sp_out["hd_ring_code"],
                "volumetric_fog": sp_out["volumetric_fog"],
                "mem_compass": mc24,
                "believed_entities": sp_out["believed_entities"],
                "auth_cand_poses": auth_poses,
                "auth_cand_verbs": auth_verbs,
            })

            if fmap_bus is not None:
                fn, f_occ, f_surf, f_elev, f_vis = hippo.field_map.export_bus_arrays()
                fmap_bus.publish_dict({
                    "timestamp_ns": time.perf_counter_ns(),
                    "physics_tick": int(w_rec["physics_tick"]),
                    "episode_id": int(w_rec["episode_id"]),
                    "field_grid_n": fn,
                    "field_occ_u8": f_occ,
                    "field_surface_i8": f_surf,
                    "field_elev_i8": f_elev,
                    "field_visit_f16": f_vis,
                })

            # Publish active gaze saccade command
            intent_bus.publish_channel(CH_GAZE_SACCADE, {
                "timestamp_ns": time.perf_counter_ns(),
                "based_on_physics_tick": int(w_rec["physics_tick"]),
                "based_on_episode_id": int(w_rec["episode_id"]),
                "based_on_action_epoch": int(w_rec["action_epoch"]),
                "command_type": 0,
                "chosen_cand_idx": 0,
                "motor_verb": VERB_IDLE_OBSERVE,
                "commit_authorized": 0,
                "pad0": 0,
                "valid_horizon_ms": 250.0,
                "target_pose": np.zeros(CONT_POSE_DIM, dtype=np.float32),
                "gaze_ypd": sp_out["active_gaze_ypd"],
            })

            # Basal Ganglia timing step + Dopamine SNc plasticity
            learn_on, _ = _read_runtime_learning_flags(cog_bus, cfg)
            if cog_rec is not None and int(cog_rec["reward_action_epoch"]) != last_reward_epoch:
                last_reward_epoch = int(cog_rec["reward_action_epoch"])
                bg_engine.da_plasticity.apply_td_feedback(
                    float(cog_rec["latest_td_reward"]),
                    float(cog_rec["latest_hazard_delta"]),
                    learning_enabled=learn_on,
                )

            cog_fresh = (
                cog_rec is not None
                and int(cog_rec["episode_id"]) == int(w_rec["episode_id"])
                and int(cog_rec["action_epoch"]) == int(w_rec["action_epoch"])
            )
            s1_live, _ = intent_bus.read_channel(CH_REFLEX_S1)
            s1_fresh = (
                s1_live is not None
                and int(s1_live["based_on_episode_id"]) == int(w_rec["episode_id"])
                and int(s1_live["based_on_action_epoch"]) == int(w_rec["action_epoch"])
            )
            if cog_fresh:
                c_idx = min(NUM_CANDIDATES - 1, int(cog_rec["chosen_cand_idx"]))
                s1_idx = min(NUM_CANDIDATES - 1, int(cog_rec["s1_fast_cand_idx"]))
            elif s1_fresh:
                c_idx = min(NUM_CANDIDATES - 1, int(s1_live["chosen_cand_idx"]))
                s1_idx = c_idx
            else:
                c_idx = min(NUM_CANDIDATES - 1, int(cog_rec["chosen_cand_idx"])) if cog_rec is not None else 0
                s1_idx = min(NUM_CANDIDATES - 1, int(cog_rec["s1_fast_cand_idx"])) if cog_rec is not None else 0
            c_verb = int(auth_verbs[c_idx])
            q_val = float(cog_rec["q_values"][c_idx]) if cog_rec is not None else 0.5
            surv_p = float(cog_rec["survival_probs"][c_idx]) if cog_rec is not None else 0.85

            coach_brake = False
            if lng_rec is not None and int(lng_rec["based_on_episode_id"]) == int(w_rec["episode_id"]):
                if int(lng_rec["strategy_mode_id"]) == 99 and int(w_rec["action_epoch"]) <= int(lng_rec["valid_until_epoch"]):
                    coach_brake = True

            dyn_tmode = timing_bus.slot.get_header_atom(0)
            if 1 <= dyn_tmode <= 4:
                cfg.timing_mode = dyn_tmode - 1

            bg_res = bg_engine.step(
                episode_id=int(w_rec["episode_id"]),
                action_epoch=int(w_rec["action_epoch"]),
                can_commit_now=bool(w_rec["can_commit_now"]),
                chosen_cand_idx=c_idx,
                s1_fast_cand_idx=s1_idx,
                chosen_verb=c_verb,
                q_chosen=q_val,
                survival_prob=surv_p,
                scene_opportunity_prob=opp_prob,
                local_kinetic_risk=kin_risk,
                believed_scene_speed=scene_spd,
                mean_fog=sp_out["mean_fog"],
                saccade_hold_active=bool(sp_out["saccade_hold_active"]),
                respawn_event_flag=bool(sp_out["respawn_event_flag"]),
                coach_brake=coach_brake,
                in_refuge=is_in_stealth_or_cover_refuge(proprio),
                tank_level=get_resource_tank_level(proprio),
                hazard_level=max(float(w_rec["hazard_level"]), sp_out["believed_hazard"]),
                step_reward=float(w_rec["last_step_reward"]),
                dt=dt,
                self_surprise=float(cog_rec["self_surprise"]) if cog_fresh else 0.0,
            )
            bg_res["physics_tick"] = int(w_rec["physics_tick"])
            timing_bus.publish_dict(bg_res)

            next_t += dt / max(0.25, float(cfg.time_scale))
            rem = next_t - time.perf_counter()
            if rem > 0.0004:
                time.sleep(rem)
            elif rem < -0.05:
                next_t = time.perf_counter()
    finally:
        if fmap_bus is not None:
            fmap_bus.close(unlink=False)
        if vc_bus is not None:
            vc_bus.close(unlink=False)
        world_bus.close(unlink=False)
        spatial_bus.close(unlink=False)
        timing_bus.close(unlink=False)
        cog_bus.close(unlink=False)
        lang_bus.close(unlink=False)
        intent_bus.close(unlink=False)



