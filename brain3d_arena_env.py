# -*- coding: utf-8 -*-
"""
brain3d_arena_env.py — Built-in 60Hz 3D Tactical Arena Sandbox & WorldBridgeProcess (P1, v2.2)
==============================================================================================
Barrier 1 Guarantee: Strictly Zero-PyTorch (`"torch" not in sys.modules`).
License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

import math
import os
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from task_profiles_3d import *
from brain3d_config_ipc import *
from brain3d_config_ipc import (
    _clamp_utf8_bytes,
    _ignore_sigint_in_worker,
    _recv_ub3d_packet,
)

class StandaloneTactical3DArena:
    """
    Built-in Pure-NumPy 60Hz 3D Tactical Turf & Objective Sandbox (`[-12..12] x [0..6] x [-16..16]`).
    Designed for zero-Unity standalone verification AND headless multi-agent / PBT pretraining:
      - Dynamic `48x48` paintable floor surface grid (`-1.0` Enemy Turf .. `0.0` Neutral .. `+1.0` Friendly Turf).
      - Coupled Swim/Cover Refuge (`proprio[13] >= 0.90`): submerging on friendly turf or behind cover blocks
        regenerates the resource tank (`proprio[14]`) and hides the agent from distant opponents.
      - Ballistic Projectiles (`VERB_RELEASE_PLACE` / `VERB_IMPULSE_PUSH`): fired along independent aim
        yaw/pitch (`target_pose[3:5]` or `gaze_ypd[0:2]`), painting friendly turf on impact and tagging opponents.
      - Central Primary Objective (`tag = 3.0`, e.g., BlitzCore) + North Attack Goal Pedestal (`tag = 4.0`)
        + South Defend Base Pedestal (`tag = 6.0`).
      - Mid-Episode Respawn (`respawn_event_flag`): splatted agents respawn at South base with `proprio[7] = 1.0`
        while preserving match turf & objective state until knockout or match timeout (`720` ticks).
    """

    def __init__(self, cfg: Brain3DConfig, opponent_skill: float = 0.55):
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self.opponent_skill = float(np.clip(opponent_skill, 0.10, 1.0))
        self.rng = np.random.default_rng(cfg.seed)
        self.wmin = np.array(cfg.world_min, dtype=np.float32)
        self.wmax = np.array(cfg.world_max, dtype=np.float32)
        self.span = np.maximum(1.0, self.wmax - self.wmin)
        self.fovea_cos = float(math.cos(math.radians(cfg.fovea_half_angle_deg)))

        self.grid_n = FIELD_MAP_MAX_N
        self.surface_grid = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)

        self.episode_id = 0
        self.action_epoch = 0
        self.physics_tick = 0
        self.cumulative_score = 0.0
        self.cooldown_ticks = 0
        self.respawn_invuln_ticks = 0
        self.step_ticks_in_ep = 0
        self.last_respawn_flag = False

        self.self_pos = np.array([0.0, 1.5, -11.5], dtype=np.float32)
        self.self_vel = np.zeros(3, dtype=np.float32)
        self.aim_yp = np.array([0.0, -0.15], dtype=np.float32)
        self.gaze_ypd = np.array([0.0, -0.15, 6.0], dtype=np.float32)
        self.hp = 1.0
        self.resource_tank = 1.0
        self.in_refuge = False
        self.carrying_objective = False
        self.objective_progress = 0.50

        self.cover_blocks: List[Dict[str, Any]] = []
        self.opponents: List[Dict[str, Any]] = []
        self.projectiles: List[Dict[str, Any]] = []
        self.objective_pos = np.zeros(3, dtype=np.float32)
        self.objective_carrier_enemy_idx = -1
        self.attack_pedestal_pos = np.array([0.0, 1.2, 12.5], dtype=np.float32)
        self.defend_pedestal_pos = np.array([0.0, 1.2, -12.5], dtype=np.float32)
        self.reset_episode()

    def _world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        u = (float(x) - float(self.wmin[0])) / float(self.span[0])
        v = (float(z) - float(self.wmin[2])) / float(self.span[2])
        ix = int(np.clip(math.floor(u * self.grid_n), 0, self.grid_n - 1))
        iz = int(np.clip(math.floor(v * self.grid_n), 0, self.grid_n - 1))
        return iz, ix

    def _paint_disc(self, x: float, z: float, radius_m: float, team_sign: float) -> int:
        newly_flipped = 0
        cz, cx = self._world_to_cell(x, z)
        rad_cells = max(1, int(math.ceil(radius_m / max(0.35, float(self.span[0]) / self.grid_n))))
        for dz in range(-rad_cells, rad_cells + 1):
            iz = cz + dz
            if not (0 <= iz < self.grid_n):
                continue
            for dx in range(-rad_cells, rad_cells + 1):
                ix = cx + dx
                if not (0 <= ix < self.grid_n):
                    continue
                if dz * dz + dx * dx <= rad_cells * rad_cells + 1:
                    prev = float(self.surface_grid[iz, ix])
                    if team_sign > 0.0 and prev < 0.35:
                        newly_flipped += 1
                    self.surface_grid[iz, ix] = float(np.clip(0.35 * prev + 0.85 * team_sign, -1.0, 1.0))
        return newly_flipped

    def _sample_surface(self, x: float, z: float) -> float:
        iz, ix = self._world_to_cell(x, z)
        return float(self.surface_grid[iz, ix])

    def _is_occluded_by_cover(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        seg = p1[[0, 2]] - p0[[0, 2]]
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 < 1e-5:
            return False
        for blk in self.cover_blocks:
            c_xz = blk["pos"][[0, 2]]
            t = float(np.clip(np.dot(c_xz - p0[[0, 2]], seg) / seg_len2, 0.0, 1.0))
            proj = p0[[0, 2]] + t * seg
            if float(np.linalg.norm(proj - c_xz)) < float(blk["radius"]) * 0.92:
                return True
        return False

    def reset_episode(self) -> None:
        self.episode_id += 1
        self.action_epoch = 1
        self.cumulative_score = 0.0
        self.cooldown_ticks = 0
        self.respawn_invuln_ticks = 0
        self.step_ticks_in_ep = 0
        self.last_respawn_flag = False

        self.surface_grid.fill(0.0)
        self.self_pos = np.array(
            [float(self.rng.uniform(-2.2, 2.2)), 1.5, float(self.wmin[2] + 3.5)],
            dtype=np.float32,
        )
        self.self_vel = np.zeros(3, dtype=np.float32)
        self.aim_yp = np.array([0.0, -0.12], dtype=np.float32)
        self.gaze_ypd = np.array([0.0, -0.12, 6.0], dtype=np.float32)
        self.hp = 1.0
        self.resource_tank = 1.0
        self.in_refuge = False
        self.carrying_objective = False
        self.objective_progress = 0.50
        self.objective_carrier_enemy_idx = -1
        self.projectiles.clear()

        self.attack_pedestal_pos = np.array([0.0, 1.2, float(self.wmax[2] - 3.0)], dtype=np.float32)
        self.defend_pedestal_pos = np.array([0.0, 1.2, float(self.wmin[2] + 3.0)], dtype=np.float32)
        self.objective_pos = np.array([0.0, 1.1, 0.0], dtype=np.float32)

        # Spawn initial base ink pads
        self._paint_disc(float(self.self_pos[0]), float(self.self_pos[2]), 2.4, +1.0)
        self._paint_disc(0.0, float(self.wmax[2] - 3.5), 2.4, -1.0)

        # Symmetric tactical cover blocks + resource packs
        self.cover_blocks = [
            {"id": 101, "pos": np.array([-4.2, 1.3, -4.5], dtype=np.float32), "radius": 1.25, "tag": SemanticTag.COVER_OR_FRIENDLY_TURF},
            {"id": 102, "pos": np.array([4.2, 1.3, -4.5], dtype=np.float32), "radius": 1.25, "tag": SemanticTag.COVER_OR_FRIENDLY_TURF},
            {"id": 103, "pos": np.array([-4.2, 1.3, 4.5], dtype=np.float32), "radius": 1.25, "tag": SemanticTag.COVER_OR_FRIENDLY_TURF},
            {"id": 104, "pos": np.array([4.2, 1.3, 4.5], dtype=np.float32), "radius": 1.25, "tag": SemanticTag.COVER_OR_FRIENDLY_TURF},
            {"id": 105, "pos": np.array([-7.5, 1.6, 0.0], dtype=np.float32), "radius": 1.40, "tag": SemanticTag.NEUTRAL_PROP},
            {"id": 106, "pos": np.array([7.5, 1.6, 0.0], dtype=np.float32), "radius": 1.40, "tag": SemanticTag.NEUTRAL_PROP},
        ]

        self.opponents = []
        for idx in range(2):
            ox = -3.2 if idx == 0 else 3.2
            oz = float(self.wmax[2] - 4.2)
            self.opponents.append({
                "id": 201 + idx,
                "pos": np.array([ox, 1.5, oz], dtype=np.float32),
                "vel": np.array([0.0, 0.0, -2.2], dtype=np.float32),
                "hp": 1.0,
                "respawn_ticks": 0,
                "fire_cd": 18 + idx * 12,
                "phase": float(self.rng.uniform(0.0, 6.28)),
            })

    def _trigger_self_respawn(self) -> None:
        if self.carrying_objective:
            self.carrying_objective = False
            self.objective_pos = np.array([float(self.self_pos[0]), 1.1, float(self.self_pos[2])], dtype=np.float32)
        self.self_pos = np.array(
            [float(self.rng.uniform(-2.0, 2.0)), 1.5, float(self.wmin[2] + 3.2)],
            dtype=np.float32,
        )
        self.self_vel.fill(0.0)
        self.hp = 1.0
        self.resource_tank = 1.0
        self.in_refuge = False
        self.respawn_invuln_ticks = 18
        self.last_respawn_flag = True
        self._paint_disc(float(self.self_pos[0]), float(self.self_pos[2]), 2.0, +1.0)

    def step(
        self,
        target_pose: np.ndarray,
        motor_verb: int,
        commit_authorized: bool,
        gaze_ypd: np.ndarray,
        dt: float,
    ) -> Dict[str, Any]:
        self.physics_tick += 1
        self.step_ticks_in_ep += 1
        step_reward = 0.0
        episode_done = False
        self.last_respawn_flag = False

        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
        if self.respawn_invuln_ticks > 0:
            self.respawn_invuln_ticks -= 1

        # 1. Update independent gaze & aim orientation
        if np.linalg.norm(gaze_ypd) > 1e-5:
            self.gaze_ypd[0] = float(gaze_ypd[0])
            self.gaze_ypd[1] = float(np.clip(gaze_ypd[1], -1.15, 1.15))
            self.gaze_ypd[2] = float(np.clip(gaze_ypd[2], 0.5, self.cfg.fovea_max_range_m))

        if abs(float(target_pose[3])) > 1e-4 or abs(float(target_pose[4])) > 1e-4:
            self.aim_yp[0] = float(target_pose[3])
            self.aim_yp[1] = float(np.clip(target_pose[4], -0.95, 0.95))
        else:
            self.aim_yp[0] = float(self.gaze_ypd[0])
            self.aim_yp[1] = float(self.gaze_ypd[1])

        # 2. Check current surface & cover proximity
        surf_here = self._sample_surface(float(self.self_pos[0]), float(self.self_pos[2]))
        near_cover = any(
            float(np.linalg.norm(self.self_pos[[0, 2]] - blk["pos"][[0, 2]])) <= float(blk["radius"]) + 1.35
            for blk in self.cover_blocks
            if SemanticTag.is_cover_or_refuge(blk["tag"])
        )

        # Enter/maintain swim-or-cover refuge when on friendly turf or near cover and not actively shooting
        wants_refuge = (motor_verb == VERB_IDLE_OBSERVE) or (
            motor_verb == VERB_EVADE_HAZARD and surf_here > 0.25
        )
        self.in_refuge = bool((surf_here > 0.28 or near_cover) and wants_refuge and (not self.carrying_objective))
        if self.in_refuge:
            self.resource_tank = float(min(1.0, self.resource_tank + 0.55 * dt))
            self.hp = float(min(1.0, self.hp + 0.30 * dt))
        else:
            self.resource_tank = float(min(1.0, self.resource_tank + 0.08 * dt))
            self.hp = float(min(1.0, self.hp + 0.06 * dt))

        # 3. Locomotion toward denormalized `target_pose[0:3]` (decoupled from aim `[3:5]`!)
        mid = 0.5 * (self.wmin + self.wmax)
        half = 0.5 * self.span
        desired_pos = np.clip(mid + np.clip(target_pose[0:3], -1.0, 1.0) * half, self.wmin + 0.8, self.wmax - 0.8)
        desired_pos[1] = 1.5
        speed_scale = float(np.clip(target_pose[5], 0.20, 1.0)) if abs(float(target_pose[5])) > 1e-3 else 0.80

        base_speed = 5.4
        if surf_here > 0.30:
            base_speed = 6.8 if motor_verb == VERB_EVADE_HAZARD else 6.1
        elif surf_here < -0.30:
            base_speed = 2.6
        if self.carrying_objective:
            base_speed *= 0.86
        if motor_verb == VERB_IDLE_OBSERVE and self.in_refuge:
            base_speed *= 0.18

        pos_err = desired_pos - self.self_pos
        pos_err[1] = 0.0
        dist_err = float(np.linalg.norm(pos_err))
        if dist_err > 0.08:
            self.self_vel = (pos_err / dist_err) * (base_speed * speed_scale)
        else:
            self.self_vel *= 0.65

        next_pos = np.clip(self.self_pos + self.self_vel * dt, self.wmin + 0.7, self.wmax - 0.7)
        next_pos[1] = 1.5
        # Push out of solid cover obstacles
        for blk in self.cover_blocks:
            delta_xz = next_pos[[0, 2]] - blk["pos"][[0, 2]]
            d_blk = float(np.linalg.norm(delta_xz))
            min_r = float(blk["radius"]) + 0.45
            if d_blk < min_r:
                push_dir = (delta_xz / max(1e-4, d_blk)) if d_blk > 1e-4 else np.array([0.0, -1.0], dtype=np.float32)
                next_pos[0] = float(blk["pos"][0] + push_dir[0] * min_r)
                next_pos[2] = float(blk["pos"][2] + push_dir[1] * min_r)
        self.self_pos = next_pos

        # 4. Execute committed action (shoot / paint / pick objective / score)
        if commit_authorized and self.cooldown_ticks == 0:
            if motor_verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH) and self.resource_tank >= self.profile.min_resource_to_fire:
                cost = 0.14 if motor_verb == VERB_RELEASE_PLACE else 0.24
                self.resource_tank = max(0.0, self.resource_tank - cost)
                self.in_refuge = False
                sy, cy = math.sin(float(self.aim_yp[0])), math.cos(float(self.aim_yp[0]))
                cp = max(0.25, math.cos(float(self.aim_yp[1])))
                sp = math.sin(float(self.aim_yp[1]))
                dir_v = np.array([sy * cp, sp, cy * cp], dtype=np.float32)
                dir_v /= max(1e-5, float(np.linalg.norm(dir_v)))
                proj_spd = float(self.profile.projectile_speed_mps)
                self.projectiles.append({
                    "pos": self.self_pos.copy() + dir_v * 0.75,
                    "vel": dir_v * proj_spd,
                    "team": 1,
                    "ttl": 32,
                    "splash": 1.65 if motor_verb == VERB_RELEASE_PLACE else 2.35,
                })
                flipped = self._paint_disc(
                    float(self.self_pos[0] + dir_v[0] * 2.2),
                    float(self.self_pos[2] + dir_v[2] * 2.2),
                    1.35,
                    +1.0,
                )
                step_reward += 0.04 * float(min(6, flipped))
                self.cooldown_ticks = 8 if motor_verb == VERB_RELEASE_PLACE else 14
                self.action_epoch += 1
            elif motor_verb in (VERB_COMMIT_INTERACT, VERB_GRASP_PICK):
                if (not self.carrying_objective) and self.objective_carrier_enemy_idx < 0:
                    if float(np.linalg.norm(self.self_pos[[0, 2]] - self.objective_pos[[0, 2]])) <= self.profile.interact_reach_m:
                        self.carrying_objective = True
                        step_reward += 3.5
                        self.cooldown_ticks = 8
                        self.action_epoch += 1

        # Automatic objective pickup & pedestal progress shaping
        if (not self.carrying_objective) and self.objective_carrier_enemy_idx < 0:
            if float(np.linalg.norm(self.self_pos[[0, 2]] - self.objective_pos[[0, 2]])) <= 1.35:
                self.carrying_objective = True
                step_reward += 3.0
                self.action_epoch += 1

        if self.carrying_objective:
            self.objective_pos = self.self_pos.copy()
            prev_prog = self.objective_progress
            z_frac = (float(self.self_pos[2]) - float(self.defend_pedestal_pos[2])) / max(
                1.0, float(self.attack_pedestal_pos[2] - self.defend_pedestal_pos[2])
            )
            self.objective_progress = float(np.clip(z_frac, 0.0, 1.0))
            if self.objective_progress > prev_prog:
                step_reward += (self.objective_progress - prev_prog) * 8.0
            if float(np.linalg.norm(self.self_pos[[0, 2]] - self.attack_pedestal_pos[[0, 2]])) <= 2.1:
                step_reward += 15.0
                episode_done = True

        # 5. Update opponent bots & ballistic projectiles
        loudest_audio = 0.0
        loudest_vec_xz = np.zeros(2, dtype=np.float32)
        hazard_accum = max(
            0.25 if surf_here < -0.35 else 0.0,
            0.85 * float(np.clip(1.0 - float(self.hp), 0.0, 1.0)),
        )

        for idx, opp in enumerate(self.opponents):
            if opp["respawn_ticks"] > 0:
                opp["respawn_ticks"] -= 1
                if opp["respawn_ticks"] == 0:
                    opp["pos"] = np.array([-3.0 + 6.0 * idx, 1.5, float(self.wmax[2] - 3.5)], dtype=np.float32)
                    opp["hp"] = 1.0
                continue

            # Opponent pursues carrier or patrols central objective zone
            can_sense_player = (not self.in_refuge) or (
                float(np.linalg.norm(opp["pos"][[0, 2]] - self.self_pos[[0, 2]])) < 3.6
            )
            if can_sense_player and not self._is_occluded_by_cover(opp["pos"], self.self_pos):
                tgt_xz = self.self_pos[[0, 2]]
            elif (not self.carrying_objective) and self.objective_carrier_enemy_idx < 0:
                tgt_xz = self.objective_pos[[0, 2]]
            else:
                opp["phase"] += dt * 0.9
                tgt_xz = np.array([math.sin(opp["phase"]) * 5.0, math.cos(opp["phase"] * 0.7) * 3.5], dtype=np.float32)

            if self.objective_carrier_enemy_idx == idx:
                tgt_xz = self.defend_pedestal_pos[[0, 2]]
                self.objective_pos = opp["pos"].copy()
                z_frac = (float(opp["pos"][2]) - float(self.defend_pedestal_pos[2])) / max(
                    1.0, float(self.attack_pedestal_pos[2] - self.defend_pedestal_pos[2])
                )
                self.objective_progress = float(np.clip(z_frac, 0.0, 1.0))

            diff_xz = tgt_xz - opp["pos"][[0, 2]]
            d_opp = float(np.linalg.norm(diff_xz))
            opp_spd = 3.2 + 1.8 * self.opponent_skill
            if d_opp > 0.25:
                opp["vel"][0] = float(diff_xz[0] / d_opp) * opp_spd
                opp["vel"][2] = float(diff_xz[1] / d_opp) * opp_spd
            else:
                opp["vel"][0] *= 0.5
                opp["vel"][2] *= 0.5
            opp["pos"][0] = float(np.clip(opp["pos"][0] + opp["vel"][0] * dt, self.wmin[0] + 1.0, self.wmax[0] - 1.0))
            opp["pos"][2] = float(np.clip(opp["pos"][2] + opp["vel"][2] * dt, self.wmin[2] + 1.0, self.wmax[2] - 1.0))

            if (not self.carrying_objective) and self.objective_carrier_enemy_idx < 0:
                if float(np.linalg.norm(opp["pos"][[0, 2]] - self.objective_pos[[0, 2]])) <= 1.25:
                    self.objective_carrier_enemy_idx = idx

            opp_just_fired = False
            vec_to_opp = opp["pos"][[0, 2]] - self.self_pos[[0, 2]]
            dist_to_self = float(np.linalg.norm(vec_to_opp))
            if dist_to_self < 8.5 and can_sense_player and not self._is_occluded_by_cover(opp["pos"], self.self_pos):
                if opp["fire_cd"] <= 0:
                    dir3 = self.self_pos - opp["pos"]
                    v_opp_proj = 12.5 + 3.0 * self.opponent_skill
                    g_opp_proj = float(getattr(self.profile, "projectile_gravity_mps2", 0.0))
                    t_est = float(np.linalg.norm(dir3)) / max(4.0, v_opp_proj)
                    dir3[1] += min(0.95, 0.5 * g_opp_proj * t_est * t_est)
                    dir3 /= max(1e-4, float(np.linalg.norm(dir3)))
                    self.projectiles.append({
                        "pos": opp["pos"].copy() + dir3 * 0.8,
                        "vel": dir3 * v_opp_proj,
                        "team": -1,
                        "ttl": 30,
                        "splash": 1.45,
                    })
                    opp["fire_cd"] = max(14, int(round(32 - 14 * self.opponent_skill)))
                    opp_just_fired = True
            if opp["fire_cd"] > 0:
                opp["fire_cd"] -= 1

            # Event-gated acoustic emission (gunfire or active un-submerged footsteps only!)
            occ_atten = 0.52 if self._is_occluded_by_cover(self.self_pos, opp["pos"]) else 1.0
            opp_cur_spd = float(math.hypot(float(opp["vel"][0]), float(opp["vel"][2])))
            raw_l = 0.0
            if opp_just_fired:
                raw_l = 0.92 * max(0.0, 1.0 - dist_to_self / 15.0)
            elif opp_cur_spd > 0.55:
                raw_l = 0.62 * max(0.0, 1.0 - dist_to_self / 11.0)
            loud = float(np.clip(raw_l * occ_atten, 0.0, 1.0))
            if loud >= 0.05 and loud > loudest_audio and dist_to_self > 1e-3:
                loudest_audio = loud
                loudest_vec_xz = vec_to_opp / dist_to_self
            if opp_just_fired and loud >= 0.30:
                hazard_accum = max(hazard_accum, float(np.clip((loud - 0.25) * 0.75, 0.0, 0.55)))

        # Advance projectiles with parabolic gravity matching profile.projectile_gravity_mps2
        g_proj = float(getattr(self.profile, "projectile_gravity_mps2", 0.0))
        alive_proj: List[Dict[str, Any]] = []
        for pr in self.projectiles:
            pr["pos"] = pr["pos"] + pr["vel"] * dt + np.array([0.0, -0.5 * g_proj * dt * dt, 0.0], dtype=np.float32)
            pr["vel"][1] = float(pr["vel"][1]) - g_proj * dt
            pr["ttl"] -= 1
            hit_something = False

            if pr["pos"][1] <= float(self.wmin[1]) + 0.15 or pr["ttl"] <= 0:
                self._paint_disc(float(pr["pos"][0]), float(pr["pos"][2]), float(pr["splash"]), float(pr["team"]))
                hit_something = True
            else:
                for blk in self.cover_blocks:
                    if float(np.linalg.norm(pr["pos"][[0, 2]] - blk["pos"][[0, 2]])) <= float(blk["radius"]):
                        hit_something = True
                        break

            if not hit_something:
                if pr["team"] > 0:
                    for idx, opp in enumerate(self.opponents):
                        if opp["respawn_ticks"] == 0 and float(np.linalg.norm(pr["pos"] - opp["pos"])) <= 1.15:
                            opp["hp"] -= 0.55
                            step_reward += 1.25
                            self._paint_disc(float(opp["pos"][0]), float(opp["pos"][2]), 1.8, +1.0)
                            if opp["hp"] <= 0.0:
                                opp["respawn_ticks"] = 75
                                step_reward += 3.5
                                if self.objective_carrier_enemy_idx == idx:
                                    self.objective_carrier_enemy_idx = -1
                                    self.objective_pos = np.array([float(opp["pos"][0]), 1.1, float(opp["pos"][2])], dtype=np.float32)
                            hit_something = True
                            break
                else:
                    d_me = float(np.linalg.norm(pr["pos"] - self.self_pos))
                    if d_me <= 1.05 and self.respawn_invuln_ticks == 0:
                        self.hp -= 0.52
                        self._paint_disc(float(self.self_pos[0]), float(self.self_pos[2]), 1.5, -1.0)
                        step_reward -= 1.2
                        hit_something = True
                        if self.hp <= 0.0:
                            step_reward -= 3.5
                            self._trigger_self_respawn()
                            self.action_epoch += 1
            if not hit_something:
                alive_proj.append(pr)
        self.projectiles = alive_proj[-20:]

        if self.step_ticks_in_ep >= 720:
            episode_done = True

        self.cumulative_score += step_reward

        # 6. Build 1st-person frustum-culled visible entities + 16x16x4 fovea
        cy, sy = math.cos(float(self.gaze_ypd[0])), math.sin(float(self.gaze_ypd[0]))
        cp, sp = math.cos(float(self.gaze_ypd[1])), math.sin(float(self.gaze_ypd[1]))
        fwd = np.array([sy * cp, sp, cy * cp], dtype=np.float32)
        right = np.array([cy, 0.0, -sy], dtype=np.float32)
        up = np.cross(fwd, right).astype(np.float32)

        vis_list: List[np.ndarray] = []

        def _try_add_entity(eid: int, pos3: np.ndarray, vel3: np.ndarray, rad: float, tag: float, check_occ: bool = True) -> None:
            rel = pos3 - self.self_pos
            dist = float(np.linalg.norm(rel))
            if dist < 1e-4 or dist > self.cfg.fovea_max_range_m:
                return
            cos_ang = float(np.dot(rel / dist, fwd))
            if cos_ang < self.fovea_cos and dist > 2.3:
                return
            if check_occ and self._is_occluded_by_cover(self.self_pos, pos3):
                return
            conf = float(np.clip(cos_ang * (1.0 - 0.35 * (dist / self.cfg.fovea_max_range_m)), 0.20, 1.0))
            vis_list.append(np.array([
                float(eid), pos3[0], pos3[1], pos3[2],
                vel3[0], vel3[1], vel3[2], float(rad), float(tag), conf,
            ], dtype=np.float32))

        # Landmarks & pedestals
        _try_add_entity(11, self.attack_pedestal_pos, np.zeros(3, dtype=np.float32), 1.3, SemanticTag.ATTACK_GOAL_PEDESTAL, check_occ=False)
        _try_add_entity(12, self.defend_pedestal_pos, np.zeros(3, dtype=np.float32), 1.3, SemanticTag.DEFEND_BASE_PEDESTAL, check_occ=False)
        if not self.carrying_objective:
            obj_vel = self.opponents[self.objective_carrier_enemy_idx]["vel"] if self.objective_carrier_enemy_idx >= 0 else np.zeros(3, dtype=np.float32)
            _try_add_entity(21, self.objective_pos, obj_vel, 0.85, SemanticTag.PRIMARY_OBJECTIVE, check_occ=True)

        for blk in self.cover_blocks:
            _try_add_entity(int(blk["id"]), blk["pos"], np.zeros(3, dtype=np.float32), float(blk["radius"]), float(blk["tag"]), check_occ=False)

        for idx, opp in enumerate(self.opponents):
            if opp["respawn_ticks"] == 0:
                opp_tag = SemanticTag.ENEMY_CARRIER_OR_SHOT if self.objective_carrier_enemy_idx == idx else SemanticTag.OPPONENT_AGENT
                _try_add_entity(int(opp["id"]), opp["pos"], opp["vel"], 0.75, opp_tag, check_occ=True)

        for pidx, pr in enumerate(self.projectiles[:6]):
            if pr["team"] < 0:
                _try_add_entity(500 + pidx, pr["pos"], pr["vel"], 0.35, SemanticTag.ENEMY_CARRIER_OR_SHOT, check_occ=True)

        vis_arr = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
        vcnt = min(MAX_ENTITIES, len(vis_list))
        if vcnt > 0:
            vis_arr[:vcnt] = np.stack(vis_list[:vcnt], axis=0)

        # Somatic + strictly 1st-person visible hostile hazard (no behind-back invisible proximity leak!)
        for ent in vis_list:
            if float(ent[8]) < -0.25:
                dv_h = float(np.linalg.norm(ent[1:4] - self.self_pos))
                if int(round(float(ent[0]))) >= 500:
                    if dv_h < 3.6:
                        hazard_accum = max(hazard_accum, float(np.clip((3.6 - dv_h) / 3.6, 0.0, 1.0)))
                elif dv_h < 8.5:
                    hazard_accum = max(
                        hazard_accum,
                        float(np.clip((8.5 - dv_h) / 8.5 * float(ent[9]) * (0.85 if self.respawn_invuln_ticks == 0 else 0.20), 0.0, 1.0)),
                    )

        fovea = self._render_foveal_patch(fwd, right, up, vis_list)

        # Ego-relative directional audio sin/cos (zeroed to ego-forward when below audible threshold)
        audio_sin = 0.0
        audio_cos = 1.0
        if loudest_audio >= 0.05:
            world_ang = math.atan2(float(loudest_vec_xz[0]), float(loudest_vec_xz[1]))
            rel_ang = world_ang - float(self.gaze_ypd[0])
            audio_sin = float(math.sin(rel_ang))
            audio_cos = float(math.cos(rel_ang))
        else:
            loudest_audio = 0.0

        if self.in_refuge:
            posture_val = 0.95
        elif surf_here > 0.28:
            posture_val = 0.75
        elif surf_here < -0.28:
            posture_val = 0.10
        else:
            posture_val = 0.45

        res_carrier_val = 0.96 if self.carrying_objective else float(np.clip(self.resource_tank * 0.88, 0.0, 0.88))

        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[0:3] = (self.self_pos - self.wmin) / self.span
        proprio[3:6] = np.clip(self.self_vel / 8.0, -1.0, 1.0)
        proprio[PROPRIO_IDX_CAN_COMMIT] = 1.0 if self.cooldown_ticks == 0 else 0.0
        proprio[PROPRIO_IDX_TRANSIT_RESPAWN] = 1.0 if (self.last_respawn_flag or self.respawn_invuln_ticks > 12) else 0.0
        proprio[PROPRIO_IDX_HAZARD] = float(np.clip(hazard_accum, 0.0, 1.0))
        proprio[PROPRIO_IDX_VIS_RATIO] = float(vcnt) / float(MAX_ENTITIES)
        proprio[PROPRIO_IDX_AUDIO_SIN] = audio_sin
        proprio[PROPRIO_IDX_AUDIO_COS] = audio_cos
        proprio[PROPRIO_IDX_AUDIO_LOUD] = loudest_audio
        proprio[PROPRIO_IDX_SURFACE_POSTURE] = posture_val
        proprio[PROPRIO_IDX_RESOURCE_CARRIER] = res_carrier_val
        proprio[PROPRIO_IDX_OBJ_PROGRESS] = float(self.objective_progress)

        out: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "action_epoch": self.action_epoch,
            "stage_id": int(getattr(self.cfg, "default_stage_id", 1)),
            "physics_tick": self.physics_tick,
            "can_commit_now": 1 if self.cooldown_ticks == 0 else 0,
            "episode_done": 1 if episode_done else 0,
            "respawn_flag": bool(self.last_respawn_flag),
            "visible_count": vcnt,
            "cumulative_score": float(self.cumulative_score),
            "last_step_reward": float(step_reward),
            "hazard_level": float(np.clip(hazard_accum, 0.0, 1.0)),
            "self_pos": self.self_pos.copy(),
            "self_vel": self.self_vel.copy(),
            "self_yaw_pitch": self.gaze_ypd[:2].copy(),
            "proprio": proprio,
            "visible_entities": vis_arr,
            "fovea_patch": fovea,
        }
        if str(getattr(self.cfg, "visual_mode", VISUAL_MODE_RAYCAST)) != VISUAL_MODE_RAYCAST:
            out["stereo_rgb"] = self._render_synthetic_stereo_rgb(fwd, right, up, vis_list)
            out["teacher_fovea"] = fovea.copy()
        return out

    def _render_foveal_patch(
        self,
        fwd: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        vis_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        Vectorized `16x16x4` 3D raycast sensor returning `[norm_depth, nx, ny, surface_or_salience]`.
        Floor hits encode local surface state directly into channel 3:
          - `0.25` = Enemy surface (`< -0.25`)
          - `0.50` = Neutral floor
          - `0.78` = Friendly turf (`> +0.25`)
        """
        tan_half = math.tan(math.radians(self.cfg.fovea_half_angle_deg))
        u = np.linspace(-tan_half, tan_half, FOVEA_W, dtype=np.float32)
        v = np.linspace(tan_half, -tan_half, FOVEA_H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        dirs = fwd[None, None, :] + uu[:, :, None] * right[None, None, :] + vv[:, :, None] * up[None, None, :]
        dirs = dirs / np.maximum(1e-6, np.linalg.norm(dirs, axis=-1, keepdims=True))

        max_r = float(self.cfg.fovea_max_range_m)
        best_t = np.full((FOVEA_H, FOVEA_W), max_r, dtype=np.float32)
        nx = np.zeros((FOVEA_H, FOVEA_W), dtype=np.float32)
        ny = np.zeros((FOVEA_H, FOVEA_W), dtype=np.float32)
        sal = np.full((FOVEA_H, FOVEA_W), 0.10, dtype=np.float32)

        # Floor intersection y = wmin[1] + surface turf color encoding
        dy = dirs[:, :, 1]
        mask_floor = dy < -1e-4
        t_floor = (float(self.wmin[1]) - float(self.self_pos[1])) / np.where(mask_floor, dy, -1.0)
        hit_floor = mask_floor & (t_floor > 0.05) & (t_floor < best_t)
        if np.any(hit_floor):
            best_t = np.where(hit_floor, t_floor, best_t)
            ny = np.where(hit_floor, 1.0, ny)
            hit_x = self.self_pos[0] + best_t * dirs[:, :, 0]
            hit_z = self.self_pos[2] + best_t * dirs[:, :, 2]
            ix = np.clip(np.floor((hit_x - self.wmin[0]) / self.span[0] * self.grid_n).astype(np.int32), 0, self.grid_n - 1)
            iz = np.clip(np.floor((hit_z - self.wmin[2]) / self.span[2] * self.grid_n).astype(np.int32), 0, self.grid_n - 1)
            surf_vals = self.surface_grid[iz, ix]
            floor_code = np.where(surf_vals > 0.25, 0.78, np.where(surf_vals < -0.25, 0.25, 0.50)).astype(np.float32)
            sal = np.where(hit_floor, floor_code, sal)

        # Visible spheres / blocks intersection
        for ent in vis_list[:16]:
            oc = ent[1:4] - self.self_pos
            rad = float(ent[7])
            b_proj = np.einsum("hwc,c->hw", dirs, oc)
            c_val = float(np.dot(oc, oc) - rad * rad)
            disc = b_proj * b_proj - c_val
            hit = disc > 0.0
            t_sph = b_proj - np.sqrt(np.maximum(0.0, disc))
            closer = hit & (t_sph > 0.05) & (t_sph < best_t)
            if np.any(closer):
                best_t = np.where(closer, t_sph, best_t)
                hit_pts = self.self_pos[None, None, :] + best_t[:, :, None] * dirs
                n_vec = (hit_pts - ent[None, None, 1:4]) / max(1e-3, rad)
                nx = np.where(closer, n_vec[:, :, 0], nx)
                ny = np.where(closer, n_vec[:, :, 1], ny)
                tag_sal = 0.95 if ent[8] < 0 else float(np.clip(0.40 + 0.12 * ent[8], 0.30, 0.92))
                sal = np.where(closer, tag_sal, sal)

        return np.stack([1.0 - np.clip(best_t / max_r, 0.0, 1.0), nx, ny, sal], axis=-1).astype(np.float32)

    def _render_synthetic_stereo_rgb(
        self,
        fwd: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        vis_list: List[np.ndarray],
    ) -> np.ndarray:
        """Render lightweight `2x64x64x3` binocular stereo RGB24 pair for standalone stereo calibration tests."""
        tan_half = math.tan(math.radians(self.cfg.stereo_fov_deg * 0.5))
        u = np.linspace(-tan_half, tan_half, RETINA_W, dtype=np.float32)
        v = np.linspace(tan_half, -tan_half, RETINA_H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        dirs = fwd[None, None, :] + uu[:, :, None] * right[None, None, :] + vv[:, :, None] * up[None, None, :]
        dirs = dirs / np.maximum(1e-6, np.linalg.norm(dirs, axis=-1, keepdims=True))
        max_r = float(self.cfg.fovea_max_range_m)
        half_base = 0.5 * float(self.cfg.stereo_baseline_m)

        eyes = np.zeros((2, RETINA_H, RETINA_W, RETINA_C), dtype=np.uint8)
        for eye_idx, sign in enumerate((-1.0, +1.0)):
            eye_pos = self.self_pos + sign * half_base * right
            best_t = np.full((RETINA_H, RETINA_W), max_r, dtype=np.float32)
            rgb = np.zeros((RETINA_H, RETINA_W, 3), dtype=np.float32)
            rgb[:, :, 0] = 0.12
            rgb[:, :, 1] = 0.15
            rgb[:, :, 2] = 0.22

            dy = dirs[:, :, 1]
            m_fl = dy < -1e-4
            t_fl = (float(self.wmin[1]) - float(eye_pos[1])) / np.where(m_fl, dy, -1.0)
            hit_fl = m_fl & (t_fl > 0.05) & (t_fl < best_t)
            if np.any(hit_fl):
                best_t = np.where(hit_fl, t_fl, best_t)
                hx = eye_pos[0] + best_t * dirs[:, :, 0]
                hz = eye_pos[2] + best_t * dirs[:, :, 2]
                ix = np.clip(np.floor((hx - self.wmin[0]) / self.span[0] * self.grid_n).astype(np.int32), 0, self.grid_n - 1)
                iz = np.clip(np.floor((hz - self.wmin[2]) / self.span[2] * self.grid_n).astype(np.int32), 0, self.grid_n - 1)
                sv = self.surface_grid[iz, ix]
                checker = (((ix + iz) & 1).astype(np.float32) * 0.08)
                fl_r = np.where(sv < -0.25, 0.82, np.where(sv > 0.25, 0.12, 0.34 + checker))
                fl_g = np.where(sv > 0.25, 0.72, np.where(sv < -0.25, 0.18, 0.36 + checker))
                fl_b = np.where(sv > 0.25, 0.88, np.where(sv < -0.25, 0.45, 0.38 + checker))
                rgb[:, :, 0] = np.where(hit_fl, fl_r, rgb[:, :, 0])
                rgb[:, :, 1] = np.where(hit_fl, fl_g, rgb[:, :, 1])
                rgb[:, :, 2] = np.where(hit_fl, fl_b, rgb[:, :, 2])

            for ent in vis_list[:12]:
                oc = ent[1:4] - eye_pos
                rad = float(ent[7])
                b_proj = np.einsum("hwc,c->hw", dirs, oc)
                disc = b_proj * b_proj - float(np.dot(oc, oc) - rad * rad)
                t_s = b_proj - np.sqrt(np.maximum(0.0, disc))
                closer = (disc > 0.0) & (t_s > 0.05) & (t_s < best_t)
                if np.any(closer):
                    best_t = np.where(closer, t_s, best_t)
                    shade = np.clip(1.0 - best_t / max_r, 0.25, 1.0)
                    if ent[8] < 0:
                        c_rgb = (0.95, 0.22, 0.30)
                    elif SemanticTag.is_primary_objective(ent[8]):
                        c_rgb = (0.98, 0.86, 0.18)
                    else:
                        c_rgb = (0.30, 0.75, 0.55)
                    for ch_i in range(3):
                        rgb[:, :, ch_i] = np.where(closer, c_rgb[ch_i] * shade, rgb[:, :, ch_i])

            eyes[eye_idx] = np.clip(np.rint(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
        return eyes


def run_world_bridge_process(
    cfg: Brain3DConfig,
    env_backend: str,
    world_shm_name: str,
    spatial_shm_name: str,
    timing_shm_name: str,
    cog_shm_name: str,
    lang_shm_name: str,
    intent_shm_name: str,
    stop_event,
    ready_event,
    gwt_wake_event=None,
    retinal_shm_name: str = "",
) -> None:
    """
    Process 1 (`WorldBridgeProcess`):
      - Strictly Zero-GPU / Zero-PyTorch (`Barrier 1` enforced).
      - Runs either `StandaloneTactical3DArena` (`--env builtin_3d`) OR accepts a live
        Unity C# client over `TCP_NODELAY` (`--env unity_tcp`) using `UB3D` / `UBMC`.
    """
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: torch must NEVER be imported in WorldBridgeProcess!")
    _ignore_sigint_in_worker()
    orch_ppid = os.getppid()

    world_bus = SharedSingleRegionBus(world_shm_name, WORLD3D_DTYPE, create=False)
    spatial_bus = SharedSingleRegionBus(spatial_shm_name, SPATIAL3D_DTYPE, create=False)
    timing_bus = SharedSingleRegionBus(timing_shm_name, TIMING3D_DTYPE, create=False)
    cog_bus = SharedSingleRegionBus(cog_shm_name, COGNITIVE3D_DTYPE, create=False)
    lang_bus = SharedSingleRegionBus(lang_shm_name, LANGUAGE3D_DTYPE, create=False)
    intent_bus = SharedMultiChannelIntent3DBus(intent_shm_name, create=False)
    retinal_bus = SharedSingleRegionBus(retinal_shm_name, RETINAL3D_DTYPE, create=False) if retinal_shm_name else None

    dt = 1.0 / max(10.0, float(cfg.world_hz))
    sim_ms_per_tick = 1000.0 / max(1.0, float(cfg.world_hz))
    dt_wall = dt / max(0.25, float(cfg.time_scale))
    is_sleeping = False
    cmd_seq = 0
    respawn_counter = 0
    fps_counter = 0
    fps_t0 = time.perf_counter()
    measured_fps = float(cfg.world_hz)

    def _resolve_latest_motor_and_gaze(curr_ep: int, curr_epoch: int, curr_tick: int) -> Tuple[np.ndarray, int, bool, np.ndarray]:
        nonlocal is_sleeping
        sys_cmd, _ = intent_bus.read_channel(CH_SYSTEM_CONTROL)
        if sys_cmd is not None:
            ct = int(sys_cmd["command_type"])
            if ct == SYS_CMD_SLEEP_START:
                is_sleeping = True
            elif ct == SYS_CMD_SLEEP_DONE:
                is_sleeping = False
            elif ct == SYS_CMD_SHUTDOWN:
                stop_event.set()

        gaze_rec, _ = intent_bus.read_channel(CH_GAZE_SACCADE)
        gaze_ypd = gaze_rec["gaze_ypd"].copy() if gaze_rec is not None else np.array([0.0, -0.15, 6.0], dtype=np.float32)

        chosen_rec = None
        for ch in (CH_CONSCIOUS_GWT, CH_REFLEX_S1):
            rec, _ = intent_bus.read_channel(ch)
            if rec is None:
                continue
            tick_age = max(0, int(curr_tick) - int(rec["based_on_physics_tick"]))
            sim_age_ms = tick_age * sim_ms_per_tick
            if (
                int(rec["based_on_episode_id"]) == curr_ep
                and int(rec["based_on_action_epoch"]) == curr_epoch
                and sim_age_ms <= float(rec["valid_horizon_ms"])
            ):
                chosen_rec = rec
                break

        if chosen_rec is None:
            return np.zeros(CONT_POSE_DIM, dtype=np.float32), VERB_IDLE_OBSERVE, False, gaze_ypd

        m_verb = int(chosen_rec["motor_verb"])
        c_idx = int(chosen_rec["chosen_cand_idx"])
        tim_live, _ = timing_bus.read_latest()
        if m_verb == VERB_EVADE_HAZARD:
            c_auth = True
        elif (
            tim_live is not None
            and int(tim_live["episode_id"]) == int(curr_ep)
            and int(tim_live["action_epoch"]) == int(curr_epoch)
        ):
            if int(tim_live["target_cand_idx"]) == c_idx:
                c_auth = bool(tim_live["commit_gate_open"])
            elif m_verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH):
                c_auth = False
            else:
                c_auth = bool(tim_live["commit_gate_open"])
        else:
            c_auth = bool(chosen_rec["commit_authorized"])

        return (
            chosen_rec["target_pose"].copy(),
            m_verb,
            c_auth,
            gaze_ypd,
        )

    try:
        if env_backend == "unity_tcp":
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((cfg.unity_host, int(cfg.unity_port)))
            srv.listen(1)
            srv.settimeout(0.25)
            ready_event.set()

            conn: Optional[socket.socket] = None
            phys_tick = 0
            cum_score = 0.0
            last_ep = 0
            while not stop_event.is_set():
                if not is_windows_pid_alive(orch_ppid):
                    stop_event.set()
                    break
                if conn is None:
                    try:
                        conn, _ = srv.accept()
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        conn.settimeout(max(0.5, float(getattr(cfg, "unity_socket_timeout_sec", 2.5))))
                    except socket.timeout:
                        continue
                    except OSError:
                        break

                pkt = _recv_ub3d_packet(conn)
                if pkt is not None and len(pkt) == 0:
                    # Clean read timeout before header arrival (e.g., Unity paused / loading / Sleep consolidation): keep socket alive!
                    _resolve_latest_motor_and_gaze(last_ep, 1, phys_tick)
                    continue
                if pkt is None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                    continue

                frame = unpack_ub3d_sensory_frame(pkt)
                if frame is None:
                    continue

                phys_tick += 1
                fps_counter += 1
                now_p = time.perf_counter()
                if now_p - fps_t0 >= 0.5:
                    measured_fps = fps_counter / max(1e-3, now_p - fps_t0)
                    fps_counter = 0
                    fps_t0 = now_p

                if frame["episode_id"] != last_ep:
                    cum_score = 0.0
                    last_ep = frame["episode_id"]
                cum_score += frame["step_reward"]

                flags = frame["flags"]
                can_commit = 1 if (flags & 1) else 0
                ep_done = 1 if (flags & 2) else 0
                respawn_ev = bool(frame.get("respawn_flag", False))
                proprio_arr = frame["proprio"].copy()
                if respawn_ev:
                    respawn_counter += 1
                    world_bus.slot.set_header_atom(3, respawn_counter)
                    proprio_arr[PROPRIO_IDX_TRANSIT_RESPAWN] = max(1.0, float(proprio_arr[PROPRIO_IDX_TRANSIT_RESPAWN]))

                vis_entities = frame["visible_entities"]
                vis_cnt = int(frame["visible_count"])
                if float(cfg.visual_entity_dropout) > 0.0 and vis_cnt > 0:
                    keep_p = float(np.clip(1.0 - float(cfg.visual_entity_dropout), 0.0, 1.0))
                    vis_entities = vis_entities.copy()
                    kept: List[np.ndarray] = []
                    for ei in range(vis_cnt):
                        if keep_p > 0.0 and ((phys_tick * 2654435761 + ei * 40503) & 0xFFFF) < int(keep_p * 65535.0):
                            kept.append(vis_entities[ei].copy())
                    vis_entities.fill(0.0)
                    vis_cnt = len(kept)
                    for ki, row in enumerate(kept):
                        vis_entities[ki] = row

                eff_stage_id = int(frame.get("stage_id", 0)) or int(getattr(cfg, "default_stage_id", 1))
                world_bus.publish_dict({
                    "timestamp_ns": time.perf_counter_ns(),
                    "physics_tick": phys_tick,
                    "episode_id": frame["episode_id"],
                    "action_epoch": frame["action_epoch"],
                    "stage_id": eff_stage_id,
                    "can_commit_now": can_commit,
                    "episode_done": ep_done,
                    "is_sleeping": 1 if is_sleeping else 0,
                    "visible_count": vis_cnt,
                    "measured_fps": measured_fps,
                    "cumulative_score": cum_score,
                    "last_step_reward": frame["step_reward"],
                    "hazard_level": frame["hazard_level"],
                    "self_pos": frame["self_pos"],
                    "self_vel": frame["self_vel"],
                    "self_yaw_pitch": frame["self_yaw_pitch"],
                    "proprio": proprio_arr,
                    "visible_entities": vis_entities,
                    "fovea_patch": frame["fovea_patch"],
                    "caregiver_utf8": frame["caregiver_utf8"],
                })

                if retinal_bus is not None and int(frame.get("proto_version", 1)) == UNITY_PROTO_VERSION_V2:
                    sp_r, _ = spatial_bus.read_latest()
                    g_ypd = sp_r["active_gaze_ypd"].copy() if sp_r is not None else np.array([frame["self_yaw_pitch"][0], frame["self_yaw_pitch"][1], 6.0], dtype=np.float32)
                    retinal_bus.publish_dict({
                        "timestamp_ns": time.perf_counter_ns(),
                        "physics_tick": phys_tick,
                        "episode_id": frame["episode_id"],
                        "action_epoch": frame["action_epoch"],
                        "visual_mode_id": int(frame.get("visual_mode_id", 0)),
                        "teacher_valid": 1 if frame.get("teacher_valid", False) else 0,
                        "retina_valid": 1 if frame.get("retina_valid", False) else 0,
                        "pad0": 0,
                        "self_pos": frame["self_pos"],
                        "self_yaw_pitch": frame["self_yaw_pitch"],
                        "gaze_ypd": g_ypd,
                        "stereo_rgb": frame["stereo_rgb"],
                        "teacher_fovea": frame["teacher_fovea"],
                    })

                if gwt_wake_event is not None and (can_commit or ep_done or respawn_ev or (phys_tick % 4 == 0)):
                    gwt_wake_event.set()

                t_pose, m_verb, c_auth, g_ypd = _resolve_latest_motor_and_gaze(frame["episode_id"], frame["action_epoch"], phys_tick)
                cog_rec, _ = cog_bus.read_latest()
                tim_rec, _ = timing_bus.read_latest()
                lng_rec, _ = lang_bus.read_latest()

                veto_b = bool(cog_rec["veto_active"]) if cog_rec is not None else False
                h_reason = int(tim_rec["hold_reason_id"]) if tim_rec is not None else HOLD_GO_CLEAR
                telem6 = np.zeros(6, dtype=np.float32)
                if cog_rec is not None:
                    c_idx = min(NUM_CANDIDATES - 1, int(cog_rec["chosen_cand_idx"]))
                    telem6[0] = float(cog_rec["ignition"])
                    telem6[1] = float(cog_rec["phi_geometric_bits"])
                    telem6[2] = float(cog_rec["survival_at_chosen"])
                    telem6[4] = float(cog_rec["q_values"][c_idx])
                if tim_rec is not None:
                    telem6[3] = float(tim_rec["stn_brake"])
                    telem6[5] = float(tim_rec["mean_wait_ms"])
                sp_tag = ""
                if lng_rec is not None:
                    sp_tag = bytes(lng_rec["monologue_utf8"]).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

                dyn_ts_milli = cog_bus.slot.get_header_atom(1)
                eff_time_scale = (float(dyn_ts_milli) / 1000.0) if (250 <= dyn_ts_milli <= 12000) else float(cfg.time_scale)

                cmd_seq += 1
                reply = pack_ubmc_motor_command(
                    cmd_seq=cmd_seq,
                    episode_id=frame["episode_id"],
                    action_epoch=frame["action_epoch"],
                    motor_verb=m_verb,
                    commit_gate_open=c_auth,
                    veto_active=veto_b,
                    hold_reason_id=h_reason,
                    target_pose=t_pose,
                    gaze_ypd=g_ypd,
                    telemetry6=telem6,
                    inner_speech_tag=sp_tag,
                    time_scale=eff_time_scale,
                )
                try:
                    conn.sendall(reply)
                except OSError:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
            if conn is not None:
                conn.close()
            srv.close()
        else:
            arena = StandaloneTactical3DArena(cfg)
            ready_event.set()
            next_t = time.perf_counter()
            while not stop_event.is_set():
                if not is_windows_pid_alive(orch_ppid):
                    stop_event.set()
                    break
                t_pose, m_verb, c_auth, g_ypd = _resolve_latest_motor_and_gaze(
                    arena.episode_id, arena.action_epoch, arena.physics_tick
                )
                if not is_sleeping:
                    st = arena.step(t_pose, m_verb, c_auth, g_ypd, dt)
                    fps_counter += 1
                    now_p = time.perf_counter()
                    if now_p - fps_t0 >= 0.4:
                        measured_fps = fps_counter / max(1e-3, now_p - fps_t0)
                        fps_counter = 0
                        fps_t0 = now_p

                    if st.get("respawn_flag", False):
                        respawn_counter += 1
                        world_bus.slot.set_header_atom(3, respawn_counter)

                    cg_phrase = ""
                    if cfg.caregiver_curriculum == "developmental" and (st["physics_tick"] % 20 == 0):
                        sp_latest, _ = spatial_bus.read_latest()
                        if sp_latest is not None:
                            cg_phrase = synthesize_joint_attention_phrase(st, sp_latest, m_verb)

                    world_bus.publish_dict({
                        "timestamp_ns": time.perf_counter_ns(),
                        "physics_tick": st["physics_tick"],
                        "episode_id": st["episode_id"],
                        "action_epoch": st["action_epoch"],
                        "stage_id": int(st.get("stage_id", getattr(cfg, "default_stage_id", 1))),
                        "can_commit_now": st["can_commit_now"],
                        "episode_done": st["episode_done"],
                        "is_sleeping": 0,
                        "visible_count": st["visible_count"],
                        "measured_fps": measured_fps,
                        "cumulative_score": st["cumulative_score"],
                        "last_step_reward": st["last_step_reward"],
                        "hazard_level": st["hazard_level"],
                        "self_pos": st["self_pos"],
                        "self_vel": st["self_vel"],
                        "self_yaw_pitch": st["self_yaw_pitch"],
                        "proprio": st["proprio"],
                        "visible_entities": st["visible_entities"],
                        "fovea_patch": st["fovea_patch"],
                        "caregiver_utf8": _clamp_utf8_bytes(cg_phrase, 127),
                    })

                    if retinal_bus is not None and "stereo_rgb" in st:
                        retinal_bus.publish_dict({
                            "timestamp_ns": time.perf_counter_ns(),
                            "physics_tick": st["physics_tick"],
                            "episode_id": st["episode_id"],
                            "action_epoch": st["action_epoch"],
                            "visual_mode_id": VISUAL_MODE_TO_ID.get(cfg.visual_mode, 0),
                            "teacher_valid": 1,
                            "retina_valid": 1,
                            "pad0": 0,
                            "self_pos": st["self_pos"],
                            "self_yaw_pitch": st["self_yaw_pitch"],
                            "gaze_ypd": g_ypd,
                            "stereo_rgb": st["stereo_rgb"],
                            "teacher_fovea": st["teacher_fovea"],
                        })

                    if gwt_wake_event is not None and (st["can_commit_now"] or st["episode_done"] or (st["physics_tick"] % 4 == 0)):
                        gwt_wake_event.set()
                    if st["episode_done"]:
                        time.sleep(0.06)
                        arena.reset_episode()
                else:
                    w_prev, _ = world_bus.read_latest()
                    if w_prev is not None:
                        d_copy = {name: w_prev[name] for name in WORLD3D_DTYPE.names}
                        d_copy["timestamp_ns"] = time.perf_counter_ns()
                        d_copy["is_sleeping"] = 1
                        d_copy["episode_done"] = 0
                        world_bus.publish_dict(d_copy)

                next_t += dt_wall
                rem = next_t - time.perf_counter()
                if rem > 0.0005:
                    time.sleep(rem)
                elif rem < -0.05:
                    next_t = time.perf_counter()
    finally:
        if retinal_bus is not None:
            retinal_bus.close(unlink=False)
        world_bus.close(unlink=False)
        spatial_bus.close(unlink=False)
        timing_bus.close(unlink=False)
        cog_bus.close(unlink=False)
        lang_bus.close(unlink=False)
        intent_bus.close(unlink=False)



