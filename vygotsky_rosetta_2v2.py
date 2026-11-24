# -*- coding: utf-8 -*-
"""
vygotsky_rosetta_2v2.py — 2v2 Occluded Narrowband Radio Arena & Live Rosetta Stone MI Probe (v2.3.1)
====================================================================================================
Barrier 1 Guarantee: Strictly Zero-PyTorch (`"torch" not in sys.modules`).
Provides:
  1. `LiveRosettaStoneProbe`: Online contingency-table, Pointwise Mutual Information (PMI), and
     Miller-Madow bias-corrected Mutual Information analyzer that empirically decodes emergent
     10-bit `[◈R(8) · ◆S(16) · ▶I(8)]` packets without any hardcoded human dictionary.
  2. `HeadToHead2v2Tactical3DArena`: Symmetric 2v2 (`Team A: (+1,0),(+1,1)` vs `Team B: (-1,0),(+1->-1,1)`)
     3D InkBlitz tactical arena supporting `Stage 1` and `Stage 2: Occluded Twin-Flank`, enforcing
     a strict `1.0 Hz` (`60`-tick) `10-bit` `[r, s, i]` radio channel between same-team allies.

License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

from collections import deque
import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from brain3d_config_ipc import (
    Brain3DConfig,
    ENTITY_DIM,
    FOVEA_H,
    FOVEA_W,
    MAX_ENTITIES,
    PROPRIO_DIM,
    PROTOLANG_I_DIM,
    PROTOLANG_R_DIM,
    PROTOLANG_S_DIM,
    RADIO_COOLDOWN_TICKS,
    derive_ego_referent_targets_np,
    format_protolang_glyph,
    pack_protolang_wire_tag,
)
from task_profiles_3d import (
    SemanticTag,
    VERB_EVADE_HAZARD,
    VERB_IDLE_OBSERVE,
    VERB_IMPULSE_PUSH,
    VERB_RELEASE_PLACE,
    get_task_profile,
)

if "torch" in sys.modules:
    raise RuntimeError("Barrier 1 violation: vygotsky_rosetta_2v2.py must NEVER import torch!")


# Empirical Referent Labels for Live Rosetta Stone Discovery (discovered purely via PMI!)
_REF_AXIS_LABELS_JA: Tuple[Tuple[str, ...], ...] = (
    ("視界クリア", "左翼脅威", "中央脅威", "右翼脅威"),            # Axis 0: Threat Sector (4)
    ("索敵巡航", "コア捕捉", "被弾交戦", "コア運搬"),              # Axis 1: Tactical State (4)
    ("塗拡探索", "コア前進", "迎撃制圧", "退避潜伏", "隠密裏取"),  # Axis 2: Macro Intent (5)
    ("自陣防衛", "左レーン", "中央広場", "右レーン"),              # Axis 3: Team-Canonical Zone (4)
)

_REF_AXIS_LABELS_EN: Tuple[Tuple[str, ...], ...] = (
    ("Clear", "Threat-L", "Threat-C", "Threat-R"),
    ("Patrol", "CoreSpot", "Combat", "CarryCore"),
    ("Explore", "PushCore", "Intercept", "Evade", "Flank"),
    ("RearBase", "LeftLane", "MidHub", "RightLane"),
)


def classify_canonical_arena_zone(pos_xz: Optional[np.ndarray], team_sign: int = +1) -> int:
    """Classify canonical team-frame position into 4 coarse zones: 0=RearBase, 1=LeftLane, 2=MidHub, 3=RightLane."""
    if pos_xz is None:
        return 2
    arr = np.asarray(pos_xz, dtype=np.float32).ravel()
    if arr.size < 2:
        return 2
    cx = float(arr[0]) * (1.0 if int(team_sign) >= 0 else -1.0)
    cz = float(arr[1]) * (1.0 if int(team_sign) >= 0 else -1.0)
    if cz < -5.5:
        return 0
    if cx < -3.2:
        return 1
    if cx > 3.2:
        return 3
    return 2


def _compute_discrete_mi_bits(counts_2d: np.ndarray) -> float:
    """Compute Miller-Madow bias-corrected empirical Mutual Information `I(X; Y)` in bits."""
    c = np.asarray(counts_2d, dtype=np.float64)
    n_tot = float(np.sum(c))
    if n_tot < 8.0:
        return 0.0
    pxy = c / n_tot
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    denom = px * py
    mask = (pxy > 1e-12) & (denom > 1e-12)
    raw_mi = float(np.sum(pxy[mask] * np.log2(pxy[mask] / denom[mask])))
    # Miller-Madow finite-sample bias correction: (K_xy - K_x - K_y + 1) / (2 * N * ln 2)
    k_xy = int(np.sum(c > 0.5))
    k_x = int(np.sum(np.sum(c, axis=1) > 0.5))
    k_y = int(np.sum(np.sum(c, axis=0) > 0.5))
    mm_bias = max(0.0, float(k_xy - k_x - k_y + 1)) / (2.0 * n_tot * math.log(2.0))
    return float(max(0.0, raw_mi - mm_bias))


class LiveRosettaStoneProbe:
    """
    Real-Time Empirical Mutual-Information & PMI Lexicon Decoder (`LiveRosettaStoneProbe` v2.3.1):
      - Observes live emitted `[R(8), S(16), I(8)]` packets alongside 1st-person POMDP referents.
      - Never forces any slot to represent any pre-assigned human axis; instead tracks all `3 x 4`
        slot-by-axis contingency tables + joint packet distribution, discovering whichever meanings
        crystallize with the highest empirical Pointwise Mutual Information (PMI).
    """

    SLOT_DIMS = (PROTOLANG_R_DIM, PROTOLANG_S_DIM, PROTOLANG_I_DIM)
    SLOT_PREFIXES = ("◈", "◆", "▶")
    AXIS_DIMS = (4, 4, 5, 4)

    def __init__(self, ema_decay: float = 0.9992):
        self.ema_decay = float(np.clip(ema_decay, 0.98, 1.0))
        self.tables: List[List[np.ndarray]] = [
            [np.zeros((s_dim, a_dim), dtype=np.float64) for a_dim in self.AXIS_DIMS]
            for s_dim in self.SLOT_DIMS
        ]
        self.joint_counts: Dict[Tuple[int, int], float] = {}
        self.pkt_counts: Dict[int, float] = {}
        self.ref_counts: Dict[int, float] = {}
        self.total_samples: int = 0
        self.comm_kl_hist: deque = deque(maxlen=512)
        self.ally_kl_hist: deque = deque(maxlen=512)

    def observe_packet(
        self,
        rsi: Tuple[int, int, int],
        ego_ref_targets: Tuple[int, int, int],
        canonical_pos_xz: Optional[np.ndarray] = None,
        comm_kl_bits: float = 0.0,
        ally_comm_kl_bits: float = 0.0,
    ) -> None:
        r_id = int(rsi[0]) % PROTOLANG_R_DIM
        s_id = int(rsi[1]) % PROTOLANG_S_DIM
        i_id = int(rsi[2]) % PROTOLANG_I_DIM
        thr_c = int(np.clip(ego_ref_targets[0], 0, 3))
        tac_c = int(np.clip(ego_ref_targets[1], 0, 3))
        int_c = int(np.clip(ego_ref_targets[2], 0, 4))
        zon_c = classify_canonical_arena_zone(canonical_pos_xz, team_sign=+1)
        axes_vals = (thr_c, tac_c, int_c, zon_c)
        slot_vals = (r_id, s_id, i_id)

        if self.ema_decay < 0.99999 and (self.total_samples % 64 == 63):
            dec = self.ema_decay ** 64
            for s_i in range(3):
                for a_i in range(4):
                    self.tables[s_i][a_i] *= dec

        for s_i, sv in enumerate(slot_vals):
            for a_i, av in enumerate(axes_vals):
                self.tables[s_i][a_i][sv, av] += 1.0

        pkt_code = (r_id << 7) | (s_id << 3) | i_id  # 0..1023
        ref_code = ((thr_c * 4 + tac_c) * 5 + int_c)  # 0..79
        self.pkt_counts[pkt_code] = self.pkt_counts.get(pkt_code, 0.0) + 1.0
        self.ref_counts[ref_code] = self.ref_counts.get(ref_code, 0.0) + 1.0
        self.joint_counts[(pkt_code, ref_code)] = self.joint_counts.get((pkt_code, ref_code), 0.0) + 1.0
        self.total_samples += 1

        if math.isfinite(comm_kl_bits):
            self.comm_kl_hist.append(float(max(0.0, comm_kl_bits)))
        if math.isfinite(ally_comm_kl_bits):
            self.ally_kl_hist.append(float(max(0.0, ally_comm_kl_bits)))

    def decode_slot_token(
        self,
        slot_idx: int,
        tok_id: int,
        exclude_axes: Sequence[int] = (),
        lang: str = "ja",
    ) -> Tuple[str, int, float, float]:
        """Return `(gloss_label, best_axis_idx, pmi_bits, p_cond)` for a single discrete slot symbol."""
        s_i = int(slot_idx) % 3
        t_i = int(tok_id) % self.SLOT_DIMS[s_i]
        labels_bank = _REF_AXIS_LABELS_JA if lang == "ja" else _REF_AXIS_LABELS_EN

        best_lbl = "?"
        best_axis = -1
        best_pmi = -1e9
        best_pcond = 0.0

        for a_i in range(4):
            if a_i in exclude_axes:
                continue
            tbl = self.tables[s_i][a_i]
            n_tot = float(np.sum(tbl))
            row_sum = float(np.sum(tbl[t_i]))
            if n_tot < 10.0 or row_sum < 2.5:
                continue
            col_sums = np.sum(tbl, axis=0)
            for c_j in range(tbl.shape[1]):
                cnt = float(tbl[t_i, c_j])
                if cnt < 1.5 or col_sums[c_j] < 1.0:
                    continue
                p_cond = (cnt + 0.25) / (row_sum + 0.25 * tbl.shape[1])
                if p_cond < 0.08:
                    continue
                p_marg = (float(col_sums[c_j]) + 0.25) / (n_tot + 0.25 * tbl.shape[1])
                pmi = math.log2(max(1e-6, p_cond / max(1e-6, p_marg)))
                score = pmi * math.pow(p_cond, 0.35) * math.sqrt(min(1.0, row_sum / 8.0))
                if score > best_pmi:
                    best_pmi = score
                    best_lbl = labels_bank[a_i][c_j]
                    best_axis = a_i
                    best_pcond = p_cond

        return best_lbl, best_axis, float(max(0.0, best_pmi)), float(best_pcond)

    def decode_packet_gloss(self, r_id: int, s_id: int, i_id: int, lang: str = "ja") -> str:
        """ empirically decode `[r, s, i]` into distinct non-redundant tactical concepts."""
        used_axes: List[int] = []
        parts: List[str] = []
        for s_i, tid in enumerate((r_id, s_id, i_id)):
            lbl, ax_i, pmi, _ = self.decode_slot_token(s_i, tid, exclude_axes=used_axes, lang=lang)
            if ax_i >= 0 and pmi >= 0.06:
                used_axes.append(ax_i)
                parts.append(lbl)
            else:
                parts.append("未分化" if lang == "ja" else "ungrounded")
        return "·".join(parts)

    def format_radio_hud_line(self, sender_tag: str, rsi: Tuple[int, int, int], kl_bits: float = 0.0) -> str:
        glyph = format_protolang_glyph(rsi[0], rsi[1], rsi[2])
        gloss = self.decode_packet_gloss(rsi[0], rsi[1], rsi[2], lang="ja")
        return f"{sender_tag} {glyph} [{gloss}] Φc={kl_bits:.2f}b"

    def compute_metrics(self) -> Dict[str, Any]:
        slot_mis: List[float] = []
        active_vocabs: List[int] = []
        for s_i in range(3):
            mis_for_slot = [_compute_discrete_mi_bits(self.tables[s_i][a_i]) for a_i in range(4)]
            slot_mis.append(float(max(mis_for_slot)) if mis_for_slot else 0.0)
            row_totals = np.sum(self.tables[s_i][0], axis=1)
            tot = max(1.0, float(np.sum(row_totals)))
            active_vocabs.append(int(np.sum((row_totals / tot) >= (0.25 / self.SLOT_DIMS[s_i]))))

        # Joint empirical MI `I(Packet_10bit ; Referent_80)` with Miller-Madow correction
        joint_mi = 0.0
        n_j = float(sum(self.pkt_counts.values()))
        if n_j >= 16.0:
            raw_j = 0.0
            for (pk, rf), c_val in self.joint_counts.items():
                if c_val > 0.0:
                    p_xy = c_val / n_j
                    p_x = self.pkt_counts[pk] / n_j
                    p_y = self.ref_counts[rf] / n_j
                    raw_j += p_xy * math.log2(max(1e-12, p_xy / max(1e-12, p_x * p_y)))
            mm_j = max(0.0, float(len(self.joint_counts) - len(self.pkt_counts) - len(self.ref_counts) + 1)) / (
                2.0 * n_j * math.log(2.0)
            )
            joint_mi = float(max(0.0, raw_j - mm_j))

        top_lex: List[Dict[str, Any]] = []
        for s_i in range(3):
            row_t = np.sum(self.tables[s_i][0], axis=1)
            top_ids = np.argsort(row_t)[::-1][:3]
            for tid in top_ids:
                if row_t[tid] < 3.0:
                    continue
                lbl_ja, _, pmi_b, p_c = self.decode_slot_token(s_i, int(tid), lang="ja")
                lbl_en, _, _, _ = self.decode_slot_token(s_i, int(tid), lang="en")
                if pmi_b >= 0.05:
                    top_lex.append({
                        "token": f"{self.SLOT_PREFIXES[s_i]}{int(tid):X}",
                        "meaning_ja": lbl_ja,
                        "meaning_en": lbl_en,
                        "pmi_bits": round(pmi_b, 3),
                        "confidence": round(p_c, 3),
                        "count": int(round(row_t[tid])),
                    })
        top_lex.sort(key=lambda d: float(d["pmi_bits"]), reverse=True)

        return {
            "samples": int(self.total_samples),
            "mi_r_bits": round(slot_mis[0], 4),
            "mi_s_bits": round(slot_mis[1], 4),
            "mi_i_bits": round(slot_mis[2], 4),
            "sum_slot_mi_bits": round(float(sum(slot_mis)), 4),
            "joint_msg_mi_bits": round(joint_mi, 4),
            "active_vocab_r": int(active_vocabs[0]),
            "active_vocab_s": int(active_vocabs[1]),
            "active_vocab_i": int(active_vocabs[2]),
            "unique_packets": int(len(self.pkt_counts)),
            "mean_comm_kl_bits": round(float(np.mean(self.comm_kl_hist)) if self.comm_kl_hist else 0.0, 4),
            "mean_ally_comm_kl_bits": round(float(np.mean(self.ally_kl_hist)) if self.ally_kl_hist else 0.0, 4),
            "top_lexicon": top_lex[:8],
        }


AgentKey2v2 = Tuple[int, int]  # (team_sign in {+1, -1}, wing_idx in {0, 1})
ALL_2V2_KEYS: Tuple[AgentKey2v2, ...] = ((+1, 0), (+1, 1), (-1, 0), (-1, 1))


class HeadToHead2v2Tactical3DArena:
    """
    Symmetric 2v2 Occluded Narrowband Radio Arena (`InkBlitz 3D` 2v2):
      - `Stage 2: Occluded Twin-Flank` spawns teammates on opposite Left (`wing 0`) and Right (`wing 1`)
        corridors separated by central twin baffles, creating genuine cross-flank partial observability.
      - Teammates communicate strictly via `1.0 Hz` (`60`-tick cooldown) `10-bit` `[r, s, i]` packets
        (`zero` raw coordinate or hidden-state leakage!).
    """

    def __init__(self, cfg: Brain3DConfig, seed: int = 2026, mute_radio: bool = False):
        # Lazy import from arena_pbt_trainer to reuse exact stage geometry & native C++ fovea renderer
        from arena_pbt_trainer import (
            HeadToHeadTactical3DArena,
            _ARENA_NATIVE_LIB,
            _arena_f32_ptr,
            canonicalize_team_observation,
            uncanonicalize_team_motor,
        )

        self._base = HeadToHeadTactical3DArena(cfg, seed=seed)
        self._canon_obs = canonicalize_team_observation
        self._uncanon_mot = uncanonicalize_team_motor
        self._native_lib = _ARENA_NATIVE_LIB
        self._f32_ptr = _arena_f32_ptr
        self.cfg = cfg
        self.profile = get_task_profile(cfg.task_profile)
        self.rng = np.random.default_rng(seed)
        self.mute_radio = bool(mute_radio)

        self.wmin = self._base.wmin
        self.wmax = self._base.wmax
        self.span = self._base.span
        self.grid_n = self._base.grid_n
        self.stage_id = self._base.stage_id
        self.cover_blocks = self._base.cover_blocks
        self.north_pedestal = self._base.north_pedestal
        self.south_pedestal = self._base.south_pedestal
        self.fovea_cos = self._base.fovea_cos

        self.episode_id = 0
        self.physics_tick = 0
        self.ticks_in_ep = 0
        self.surface_grid = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.objective_pos = np.array([0.0, 1.1, 0.0], dtype=np.float32)
        self.objective_progress_a = 0.50
        self.carrier_key: Optional[AgentKey2v2] = None
        self.dropped_idle_ticks = 0
        self.projectiles: List[Dict[str, Any]] = []
        self.agents: Dict[AgentKey2v2, Dict[str, Any]] = {}
        self.radio_slots: Dict[AgentKey2v2, Dict[str, Any]] = {}
        self.reset_episode()

    def _world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        return self._base._world_to_cell(x, z)

    def _paint_disc(self, cx: float, cz: float, radius: float, team_sign: int) -> int:
        self._base.surface_grid = self.surface_grid
        return self._base._paint_disc(cx, cz, radius, team_sign)

    def _is_occluded(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return self._base._is_occluded(p0, p1)

    def submit_radio_packet(
        self,
        sender_key: AgentKey2v2,
        rsi: Tuple[int, int, int],
        sender_ego_ref: Optional[Tuple[int, int, int]] = None,
    ) -> bool:
        """Transmit a `10-bit` `[r, s, i]` packet on the narrowband team radio (rate-limited to `1.0 Hz` / 60 ticks)."""
        if self.mute_radio or (not bool(getattr(self.cfg, "enable_2v2_teammate_radio", True))):
            return False
        slot = self.radio_slots.get(sender_key)
        if slot is None:
            return False
        if int(slot["emit_tick"]) >= 0 and (self.physics_tick - int(slot["emit_tick"])) < RADIO_COOLDOWN_TICKS:
            return False
        r_c = int(rsi[0]) % PROTOLANG_R_DIM
        s_c = int(rsi[1]) % PROTOLANG_S_DIM
        i_c = int(rsi[2]) % PROTOLANG_I_DIM
        slot["rsi"] = (r_c, s_c, i_c)
        slot["emit_tick"] = int(self.physics_tick)
        slot["sender_ego_ref"] = tuple(sender_ego_ref) if sender_ego_ref is not None else None
        return True

    def reset_episode(self) -> Dict[AgentKey2v2, Dict[str, Any]]:
        self.episode_id += 1
        self.ticks_in_ep = 0
        self.surface_grid.fill(0.0)
        self.projectiles.clear()

        # On Stage 2 (Occluded Twin-Flank), occasionally nudge initial BlitzCore or enemy stack onto Left or Right
        # corridor symmetrically across episodes so cross-flank teammate callouts resolve true hidden asymmetry!
        if self.stage_id == 2 and float(self.rng.random()) < 0.50:
            flank_sign = -1.0 if (self.episode_id % 2 == 0) else 1.0
            self.objective_pos = np.array([5.8 * flank_sign, 1.1, 0.0], dtype=np.float32)
        else:
            self.objective_pos = np.array([0.0, 1.1, 0.0], dtype=np.float32)
        self.objective_progress_a = 0.50
        self.carrier_key = None
        self.dropped_idle_ticks = 0

        z_margin = 2.8 if self.stage_id == 2 else 3.2
        stack_enemy_flank = (self.stage_id == 2) and (float(self.rng.random()) < 0.45)
        stacked_x_b = -6.8 if float(self.rng.random()) < 0.5 else 6.8

        self.agents.clear()
        self.radio_slots.clear()
        for t_sign in (+1, -1):
            z_sp = float(self.wmin[2] + z_margin) if t_sign > 0 else float(self.wmax[2] - z_margin)
            yaw0 = 0.0 if t_sign > 0 else math.pi
            for w_idx in (0, 1):
                # Canonical left wing (w_idx=0) vs right wing (w_idx=1)
                canon_x = -6.8 if w_idx == 0 else 6.8
                if self.stage_id != 2:
                    canon_x = -3.8 if w_idx == 0 else 3.8
                world_x = canon_x if t_sign > 0 else -canon_x
                if t_sign < 0 and stack_enemy_flank:
                    world_x = stacked_x_b + (-0.9 if w_idx == 0 else 0.9)
                world_x += float(self.rng.uniform(-0.45, 0.45))

                k = (t_sign, w_idx)
                self.agents[k] = {
                    "pos": np.array([world_x, 1.5, z_sp], dtype=np.float32),
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
                self.radio_slots[k] = {
                    "rsi": (-1, -1, -1),
                    "emit_tick": -999,
                    "sender_ego_ref": None,
                }
                self._paint_disc(float(self.agents[k]["pos"][0]), z_sp, 2.0, t_sign)

        return self._collect_all_canonical_obs({k: 0.0 for k in ALL_2V2_KEYS}, False)

    def _collect_all_canonical_obs(
        self,
        rews: Dict[AgentKey2v2, float],
        ep_done: bool,
    ) -> Dict[AgentKey2v2, Dict[str, Any]]:
        out: Dict[AgentKey2v2, Dict[str, Any]] = {}
        for k in ALL_2V2_KEYS:
            t_sign, w_idx = k
            raw_obs = self._build_agent_obs_2v2(k, float(rews[k]), ep_done)
            obs_c = self._canon_obs(raw_obs, t_sign, self.wmin, self.wmax) if t_sign < 0 else raw_obs
            ally_key = (t_sign, 1 - w_idx)
            ally_slot = self.radio_slots[ally_key]
            if (
                (not self.mute_radio)
                and int(ally_slot["rsi"][0]) >= 0
                and (self.physics_tick - int(ally_slot["emit_tick"])) <= 180
            ):
                r_a, s_a, i_a = ally_slot["rsi"]
                obs_c["ally_radio_pkt"] = (r_a, s_a, i_a, int(ally_slot["emit_tick"]))
                obs_c["caregiver_text"] = pack_protolang_wire_tag(r_a, s_a, i_a)
                obs_c["ally_ref_targets"] = ally_slot["sender_ego_ref"]
                obs_c["mute_ally_radio"] = False
            else:
                obs_c["ally_radio_pkt"] = None
                obs_c["caregiver_text"] = ""
                obs_c["ally_ref_targets"] = None
                obs_c["mute_ally_radio"] = bool(self.mute_radio)
            out[k] = obs_c
        return out

    def step_quad(
        self,
        acts_canon: Dict[AgentKey2v2, Tuple[np.ndarray, int, bool, np.ndarray]],
        dt: float,
    ) -> Tuple[Dict[AgentKey2v2, Dict[str, Any]], bool, int]:
        """Advance all 4 agents 1 physics tick in symmetric order."""
        self.physics_tick += 1
        self.ticks_in_ep += 1

        cmds: Dict[AgentKey2v2, Tuple[np.ndarray, int, bool, np.ndarray]] = {}
        for k in ALL_2V2_KEYS:
            t_sign, _ = k
            ac = acts_canon[k]
            if t_sign < 0:
                tp_w, gz_w = self._uncanon_mot(ac[0], ac[3], -1)
                cmds[k] = (tp_w, ac[1], ac[2], gz_w)
            else:
                cmds[k] = ac

        rews: Dict[AgentKey2v2, float] = {k: 0.0 for k in ALL_2V2_KEYS}
        ep_done = False
        winner = 0
        mid = 0.5 * (self.wmin + self.wmax)
        half = 0.5 * self.span

        for k in ALL_2V2_KEYS:
            t_sign, _ = k
            ag = self.agents[k]
            is_carrier = (self.carrier_key == k)
            ag["respawn_flag"] = False
            ag["just_fired"] = False
            if ag["cd"] > 0:
                ag["cd"] -= 1
            if ag["invuln"] > 0:
                ag["invuln"] -= 1

            tp, verb, commit_ok, gz = cmds[k]
            if np.linalg.norm(gz) > 1e-5:
                ag["gaze_ypd"] = np.array([float(gz[0]), float(np.clip(gz[1], -1.0, 1.0)), 6.0], dtype=np.float32)
            ag["aim_yp"] = np.array([float(tp[3]), float(np.clip(tp[4], -0.9, 0.9))], dtype=np.float32)

            iz, ix = self._world_to_cell(float(ag["pos"][0]), float(ag["pos"][2]))
            my_surf = float(self.surface_grid[iz, ix]) * float(t_sign)
            near_cov = any(
                float(np.linalg.norm(ag["pos"][[0, 2]] - b["pos"][[0, 2]])) <= float(b["radius"]) + 1.3
                for b in self.cover_blocks
            )
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
            ag["in_refuge"] = bool((want_sub or near_cov) and (verb == VERB_IDLE_OBSERVE) and (not is_carrier))
            ag["tank"] = float(min(1.0, ag["tank"] + (0.48 if want_sub else 0.09) * dt))
            if ag["regen_delay"] <= 0.0:
                ag["hp"] = float(min(1.0, ag["hp"] + (0.28 if want_sub else 0.05) * dt))

            des = np.clip(mid + np.clip(tp[0:3], -1.0, 1.0) * half, self.wmin + 0.8, self.wmax - 0.8)
            des[1] = 1.5
            spd = 6.36 if want_sub else (1.55 if my_surf < -0.28 else 4.30)
            if is_carrier:
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
                if verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH) and ag["tank"] >= self.profile.min_resource_to_fire and (not is_carrier):
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
                        "owner": k,
                        "ttl": 32,
                        "splash": 1.65 if verb == VERB_RELEASE_PLACE else 2.25,
                    })
                    fl = self._paint_disc(float(ag["pos"][0] + dv[0] * 2.0), float(ag["pos"][2] + dv[2] * 2.0), 1.3, t_sign)
                    rews[k] += 0.04 * float(min(5, fl))
                    ag["cd"] = 8
                    ag["epoch"] += 1

        # Symmetric BlitzCore pickup across all 4 agents (team-shared + individual claim bonus!)
        if self.carrier_key is None:
            dists = [(float(np.linalg.norm(self.agents[k]["pos"][[0, 2]] - self.objective_pos[[0, 2]])), k) for k in ALL_2V2_KEYS]
            dists.sort(key=lambda item: item[0])
            if dists[0][0] <= 1.45 and (dists[1][0] > 1.45 or dists[0][0] < dists[1][0] - 1e-6):
                best_k = dists[0][1]
                self.carrier_key = best_k
                c_team = best_k[0]
                rews[best_k] += 2.6
                rews[(c_team, 1 - best_k[1])] += 1.2  # Team cooperative objective reward!
                rews[(-c_team, 0)] -= 0.70
                rews[(-c_team, 1)] -= 0.70
                self.agents[best_k]["epoch"] += 1

        if self.carrier_key is not None:
            self.dropped_idle_ticks = 0
            c_team, c_wing = self.carrier_key
            c_ag = self.agents[self.carrier_key]
            self.objective_pos = c_ag["pos"].copy()
            prev_p = self.objective_progress_a
            ped_vec = self.north_pedestal[[0, 2]] - self.south_pedestal[[0, 2]]
            ped_l2 = max(1.0, float(np.dot(ped_vec, ped_vec)))
            zf = float(np.dot(self.objective_pos[[0, 2]] - self.south_pedestal[[0, 2]], ped_vec)) / ped_l2
            self.objective_progress_a = float(np.clip(zf, 0.0, 1.0))
            delta_a = self.objective_progress_a - prev_p
            # Team-shared objective progress credit so escorting wingman & radio spotter share victory return!
            for w_i in (0, 1):
                rews[(+1, w_i)] += delta_a * 7.5
                rews[(-1, w_i)] -= delta_a * 7.5

            goal_pos = self.north_pedestal if c_team > 0 else self.south_pedestal
            if float(np.linalg.norm(c_ag["pos"][[0, 2]] - goal_pos[[0, 2]])) <= 2.1:
                for w_i in (0, 1):
                    rews[(c_team, w_i)] += 14.0
                    rews[(-c_team, w_i)] -= 9.5
                ep_done = True
                winner = c_team
        elif self.stage_id == 2 and float(np.linalg.norm(self.objective_pos[[0, 2]])) > 0.15:
            self.dropped_idle_ticks += 1
            if self.dropped_idle_ticks > 120:
                self.objective_pos[0] *= 0.965
                self.objective_pos[2] *= 0.965
                if float(np.linalg.norm(self.objective_pos[[0, 2]])) < 0.20:
                    self.objective_pos[:] = [0.0, 1.1, 0.0]
                    self.dropped_idle_ticks = 0

        g_proj = float(getattr(self.profile, "projectile_gravity_mps2", 0.0))
        alive_p = []
        pending_splats: List[Tuple[AgentKey2v2, AgentKey2v2]] = []
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
                for opp_w in (0, 1):
                    opp_k = (opp_sign, opp_w)
                    opp_ag = self.agents[opp_k]
                    hit_rad = 0.82 if bool(opp_ag.get("submerged", False)) else 1.10
                    if opp_ag["invuln"] == 0 and float(np.linalg.norm(pr["pos"] - opp_ag["pos"])) <= hit_rad:
                        opp_ag["hp"] -= 0.55
                        opp_ag["regen_delay"] = 1.0
                        opp_ag["submerged"] = False
                        atk_k = pr["owner"]
                        rews[atk_k] += 1.15
                        rews[(atk_k[0], 1 - atk_k[1])] += 0.45  # Crossfire assist credit!
                        rews[opp_k] -= 0.85
                        self._paint_disc(float(opp_ag["pos"][0]), float(opp_ag["pos"][2]), 1.6, int(pr["team"]))
                        hit = True
                        if opp_ag["hp"] <= 0.0 and all(sk != opp_k for sk, _ in pending_splats):
                            pending_splats.append((opp_k, atk_k))
                        break
            if not hit:
                alive_p.append(pr)

        for opp_k, atk_k in pending_splats:
            opp_sign, opp_w = opp_k
            opp_ag = self.agents[opp_k]
            rews[atk_k] += 2.8
            rews[(atk_k[0], 1 - atk_k[1])] += 1.1  # Team intercept / callout-assist reward!
            rews[opp_k] -= 2.8
            if self.carrier_key == opp_k:
                self.carrier_key = None
                self.dropped_idle_ticks = 0
                self.objective_pos = np.array([float(opp_ag["pos"][0]), 1.1, float(opp_ag["pos"][2])], dtype=np.float32)
            z_margin_r = 2.8 if self.stage_id == 2 else 3.2
            z_sp = float(self.wmin[2] + z_margin_r) if opp_sign > 0 else float(self.wmax[2] - z_margin_r)
            canon_x = -6.8 if opp_w == 0 else 6.8
            if self.stage_id != 2:
                canon_x = -3.8 if opp_w == 0 else 3.8
            world_x = (canon_x if opp_sign > 0 else -canon_x) + float(self.rng.uniform(-0.8, 0.8))
            opp_ag["pos"] = np.array([world_x, 1.5, z_sp], dtype=np.float32)
            opp_ag["vel"].fill(0.0)
            opp_ag["hp"] = 1.0
            opp_ag["regen_delay"] = 0.0
            opp_ag["tank"] = 1.0
            opp_ag["invuln"] = 18
            opp_ag["respawn_flag"] = True
            opp_ag["epoch"] += 1
            self._paint_disc(float(opp_ag["pos"][0]), z_sp, 1.9, opp_sign)
        self.projectiles = alive_p[-36:]

        max_ep_ticks = 660 if self.stage_id == 2 else 540
        if self.ticks_in_ep >= max_ep_ticks and not ep_done:
            ep_done = True
            if abs(self.objective_progress_a - 0.50) > 0.04:
                winner = +1 if self.objective_progress_a > 0.50 else -1
            else:
                turf_diff = float(np.sum(self.surface_grid > 0.25)) - float(np.sum(self.surface_grid < -0.25))
                winner = +1 if turf_diff > 2 else (-1 if turf_diff < -2 else 0)

        for k in ALL_2V2_KEYS:
            self.agents[k]["cum_score"] += rews[k]

        return self._collect_all_canonical_obs(rews, ep_done), ep_done, winner

    def _build_agent_obs_2v2(self, me_key: AgentKey2v2, step_rew: float, ep_done: bool) -> Dict[str, Any]:
        t_sign, w_idx = me_key
        me = self.agents[me_key]
        ally = self.agents[(t_sign, 1 - w_idx)]
        enemies = [self.agents[(-t_sign, 0)], self.agents[(-t_sign, 1)]]
        enemy_keys = [(-t_sign, 0), (-t_sign, 1)]

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
            vis_list.append(
                np.array([float(eid), p3[0], p3[1], p3[2], v3[0], v3[1], v3[2], rad, tag, max(0.2, ca)], dtype=np.float32)
            )

        atk_ped = self.north_pedestal if t_sign > 0 else self.south_pedestal
        def_ped = self.south_pedestal if t_sign > 0 else self.north_pedestal
        _add(11, atk_ped, np.zeros(3, dtype=np.float32), 1.3, SemanticTag.ATTACK_GOAL_PEDESTAL, False)
        _add(12, def_ped, np.zeros(3, dtype=np.float32), 1.3, SemanticTag.DEFEND_BASE_PEDESTAL, False)

        if self.carrier_key != me_key:
            ov = self.agents[self.carrier_key]["vel"] if self.carrier_key is not None else np.zeros(3, dtype=np.float32)
            _add(21, self.objective_pos, ov, 0.85, SemanticTag.PRIMARY_OBJECTIVE, True)
        for b in self.cover_blocks:
            _add(int(b["id"]), b["pos"], np.zeros(3, dtype=np.float32), float(b["radius"]), SemanticTag.COVER_OR_FRIENDLY_TURF, False)

        # Teammate is visible only within line-of-sight (NO wallhack — cross-wall coordination MUST use 10-bit radio!)
        _add(101, ally["pos"], ally["vel"], 0.72, SemanticTag.TEAMMATE_ALLY, True)

        for e_i, (ek, en) in enumerate(zip(enemy_keys, enemies)):
            d_en = float(np.linalg.norm(en["pos"] - me["pos"]))
            if (not en["in_refuge"]) or d_en < 3.6:
                etag = SemanticTag.ENEMY_CARRIER_OR_SHOT if self.carrier_key == ek else SemanticTag.OPPONENT_AGENT
                _add(201 + e_i, en["pos"], en["vel"], 0.75, etag, True)

        for pidx, pr in enumerate(self.projectiles[:8]):
            if int(pr["team"]) == -t_sign:
                _add(500 + pidx, pr["pos"], pr["vel"], 0.35, SemanticTag.ENEMY_CARRIER_OR_SHOT, True)

        vis_arr = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
        vc = min(MAX_ENTITIES, len(vis_list))
        if vc > 0:
            vis_arr[:vc] = np.stack(vis_list[:vc], axis=0)

        th = math.tan(math.radians(self.cfg.fovea_half_angle_deg))
        max_r = float(self.cfg.fovea_max_range_m)
        if self._native_lib is not None and self.surface_grid.flags["C_CONTIGUOUS"]:
            fovea = np.empty((FOVEA_H, FOVEA_W, 4), dtype=np.float32)
            n_vis_r = min(14, vc)
            ent_ptr = self._f32_ptr(vis_arr) if n_vis_r > 0 else None
            self._native_lib.ub3d_render_fovea_16x16(
                float(me["pos"][0]), float(me["pos"][1]), float(me["pos"][2]),
                float(me["gaze_ypd"][0]), float(me["gaze_ypd"][1]),
                float(th), float(max_r),
                float(self.wmin[0]), float(self.wmin[1]), float(self.wmin[2]),
                float(self.span[0]), float(self.span[2]),
                self._f32_ptr(self.surface_grid), int(self.grid_n), float(t_sign),
                ent_ptr, int(n_vis_r),
                self._f32_ptr(fovea),
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

            for ent in vis_list[:14]:
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

        # Strongest 1st-person enemy acoustic cue (attenuated by walls, zero omniscient bearing!)
        loud = 0.0
        aud_sin, aud_cos = 0.0, 1.0
        for en in enemies:
            d_en = float(np.linalg.norm(en["pos"] - me["pos"]))
            en_spd = float(np.linalg.norm(en["vel"][[0, 2]]))
            occ_atten = 0.45 if self._is_occluded(me["pos"], en["pos"]) else 1.0
            r_l = 0.0
            if bool(en.get("just_fired", False)) or int(en.get("cd", 0)) >= 6:
                r_l = 0.92 * max(0.0, 1.0 - d_en / 14.0)
            elif (not en["in_refuge"]) and (not bool(en.get("submerged", False))) and en_spd > 0.55:
                r_l = 0.60 * max(0.0, 1.0 - d_en / 10.0)
            cand_l = float(np.clip(r_l * occ_atten, 0.0, 1.0))
            if cand_l > loud and cand_l >= 0.05 and d_en > 1e-4:
                loud = cand_l
                rel_en = en["pos"][[0, 2]] - me["pos"][[0, 2]]
                ang_en = math.atan2(float(rel_en[0]), float(rel_en[1])) - float(me["gaze_ypd"][0])
                aud_sin, aud_cos = math.sin(ang_en), math.cos(ang_en)

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
        proprio[14] = 0.96 if self.carrier_key == me_key else float(np.clip(me["tank"] * 0.88, 0.0, 0.88))
        proprio[15] = self.objective_progress_a if t_sign > 0 else (1.0 - self.objective_progress_a)

        return {
            "episode_id": self.episode_id,
            "action_epoch": me["epoch"],
            "stage_id": int(getattr(self.cfg, "default_stage_id", 2)),
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
