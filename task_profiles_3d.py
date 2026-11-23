# -*- coding: utf-8 -*-
"""
task_profiles_3d.py — Pluggable Task Semantic Ontology, Proprioception Contract & Optional DAgger Profiles (v2.3.0)
===================================================================================================================
Separates game-specific affordance semantics, proprioception channel conventions, grounded Japanese
telemetry vocabulary, and optional early-training DAgger scaffolds from the domain-agnostic brain core
(`universal_brain_3d_core.py`).

To adapt the brain to a new 3D Unity environment (e.g., InkBlitz 3D BlitzCore / Turf Battle,
capture-the-flag, 3D stealth-action, or multi-agent robotics), engineers only need to configure or
subclass `TaskProfile3D` in this file — without modifying the 7-process shared-memory buses,
4-layer spatial cognitive map, or PyTorch Tensor-Train / RSSM / GWT neural architectures.

Barrier 1 Guarantee:
  This module NEVER imports `torch` and is safe to import in all CPU and GPU processes.

License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ==============================================================================
# 1. Universal Embodied Constants Shared Across Profiles & Core
# ==============================================================================
PROPRIO_DIM = 16
MEM_COMPASS_DIM = 24
NUM_CANDIDATES = 8
CONT_POSE_DIM = 6
NUM_MOTOR_VERBS = 8

# Universal Embodied Motor Verbs (interpreted physically by the Unity Agent script)
VERB_IDLE_OBSERVE = 0       # Hold position / submerge-in-place / crouch regen & observe
VERB_APPROACH_HOVER = 1     # Standard locomotion / strafe-glide toward target_pose[0:3] while aiming [3:5]
VERB_COMMIT_INTERACT = 2    # Context interact / grab objective / enter cover-or-swim state
VERB_GRASP_PICK = 3         # Acquire pickup / claim resource / reload
VERB_RELEASE_PLACE = 4      # Primary ranged action: shoot / paint surface / throw sub-item toward aim [3:5]
VERB_EVADE_HAZARD = 5       # High-speed evasive dash / swim-retreat away from threat or into cover/turf
VERB_IMPULSE_PUSH = 6       # Burst / special / shield-break / close-range impulse countermeasure
VERB_COMMUNICATE_PING = 7   # Team tactical ping ("To Me!" / "Danger!" / coordinate push)

VERB_NAMES_EN = {
    VERB_IDLE_OBSERVE: "IdleObserve",
    VERB_APPROACH_HOVER: "ApproachHover",
    VERB_COMMIT_INTERACT: "CommitInteract",
    VERB_GRASP_PICK: "GraspPick",
    VERB_RELEASE_PLACE: "ReleasePlace",
    VERB_EVADE_HAZARD: "EvadeHazard",
    VERB_IMPULSE_PUSH: "ImpulsePush",
    VERB_COMMUNICATE_PING: "CommunicatePing",
}

# Top-Down Prefrontal / GWT Macro Tactical Modes (published from ConsciousGWT -> Spatial & Basal Ganglia)
GOAL_MODE_BALANCED_EXPLORE = 0   # Expand turf / clear epistemic fog
GOAL_MODE_PUSH_OBJECTIVE = 1     # Claim or escort primary objective (e.g., BlitzCore) toward goal
GOAL_MODE_ENGAGE_INTERCEPT = 2   # Intercept enemy carrier / contest active zone with lead-aim
GOAL_MODE_EVADE_REGROUP = 3      # Break line-of-fire, dive into friendly turf/cover, recover resource
GOAL_MODE_STEALTH_FLANK = 4      # Low-signature flank / submerged approach

GOAL_MODE_NAMES = {
    GOAL_MODE_BALANCED_EXPLORE: "EXPLORE-EXPAND",
    GOAL_MODE_PUSH_OBJECTIVE: "PUSH-OBJECTIVE",
    GOAL_MODE_ENGAGE_INTERCEPT: "ENGAGE-INTERCEPT",
    GOAL_MODE_EVADE_REGROUP: "EVADE-REGROUP",
    GOAL_MODE_STEALTH_FLANK: "STEALTH-FLANK",
}


# ==============================================================================
# 2. Universal Semantic Entity Ontology (`ENTITY_DIM` index 8: `affordance_tag`)
# ==============================================================================
# Negative tags (< 0.0) denote hostile / hazardous entities.
# Positive tags (> 0.0) denote tactical cover, resources, objectives, goals, or allies.
class SemanticTag:
    ENEMY_AREA_HAZARD = -3.0       # Hostile bomb / lethal AoE / hazard zone
    ENEMY_CARRIER_OR_SHOT = -2.0   # High-priority threat: opponent carrying objective OR incoming projectile
    OPPONENT_AGENT = -1.0          # Standard hostile opponent / patroller

    NEUTRAL_PROP = 0.0             # Static non-interactive obstacle / landmark
    COVER_OR_FRIENDLY_TURF = 1.0   # Tactical cover block, swim-refuge node, or safe shelter
    RESOURCE_OR_PICKUP = 2.0       # Sub-weapon, ink/ammo pack, or buff pickup
    PRIMARY_OBJECTIVE = 3.0        # Neutral or dropped central objective (e.g., BlitzCore / Core)
    ATTACK_GOAL_PEDESTAL = 4.0     # Enemy pedestal / goal checkpoint where objective scores
    TEAMMATE_ALLY = 5.0            # Friendly teammate (escort / rally target)
    DEFEND_BASE_PEDESTAL = 6.0     # Friendly base pedestal to defend against enemy push

    @staticmethod
    def is_hostile(tag: float) -> bool:
        return float(tag) < -0.25

    @staticmethod
    def is_high_priority_threat(tag: float) -> bool:
        return float(tag) <= -1.50

    @staticmethod
    def is_area_hazard(tag: float) -> bool:
        return float(tag) <= -2.50

    @staticmethod
    def is_enemy_carrier_or_shot(tag: float) -> bool:
        return -2.50 < float(tag) <= -1.50

    @staticmethod
    def is_cover_or_refuge(tag: float) -> bool:
        return 0.50 <= float(tag) < 1.50

    @staticmethod
    def is_resource_pickup(tag: float) -> bool:
        return 1.50 <= float(tag) < 2.50

    @staticmethod
    def is_primary_objective(tag: float) -> bool:
        return 2.50 <= float(tag) < 3.50

    @staticmethod
    def is_attack_goal(tag: float) -> bool:
        return 3.50 <= float(tag) < 4.50

    @staticmethod
    def is_teammate(tag: float) -> bool:
        return 4.50 <= float(tag) < 5.50

    @staticmethod
    def is_defend_pedestal(tag: float) -> bool:
        return 5.50 <= float(tag) < 6.50

    @staticmethod
    def is_static_landmark_for_slam(tag: float, speed_mps: float) -> bool:
        """
        Only genuinely stationary structural landmarks (pedestals, cover nodes, neutral props)
        may contribute EKF-SLAM self-pose loop-closure innovations. Moving opponents, teammates,
        and carried objectives NEVER corrupt self-pose estimation.
        """
        if speed_mps > 0.45:
            return False
        t = float(tag)
        if t < 0.0 or (4.50 <= t < 5.50):
            return False
        return (abs(t) < 0.40) or (0.50 <= t < 1.50) or (3.50 <= t < 4.50) or (5.50 <= t < 6.50)


# ==============================================================================
# 3. Universal Proprioception Channel Specification (`float32[16]`)
# ==============================================================================
# Index  Meaning
# -----  -----------------------------------------------------------------------
# 0..2   Normalized arena-local self position (nx, ny, nz) in [0, 1]
# 3..5   Normalized self velocity (nvx, nvy, nvz) in [-1, 1]
# 6      can_commit_now (1.0 when primary interact/fire cooldown is ready, else 0.0)
# 7      in_transit_or_respawn (0.0 normal, 0.5 action transit, 1.0 respawn jump/invuln)
# 8      hazard_level in [0, 1] (immediate damage / crossfire / proximity danger)
# 9      vis_ratio in [0, 1] (visible entity count / MAX_ENTITIES)
# 10     audio_dir_sin in [-1, 1] (ego-relative sin(delta_yaw) to strongest sound)
# 11     audio_dir_cos in [-1, 1] (ego-relative cos(delta_yaw) to strongest sound)
# 12     audio_loudness in [0, 1] (wall-attenuated acoustic intensity)
# 13     posture_surface_state in [0, 1]:
#          0.00..0.20 = stuck on enemy ink / slowed & exposed
#          0.21..0.55 = neutral floor standing
#          0.56..0.89 = friendly surface standing / mobile
#          0.90..1.00 = submerged in friendly turf OR inside cover refuge (stealth + regen)
# 14     resource_or_carrier_state in [0, 1]:
#          0.00..0.88 = normalized ammo / ink / stamina tank level
#          0.90..1.00 = actively carrying the Primary Objective (e.g., holding BlitzCore!)
# 15     objective_progress in [0, 1]:
#          0.50 = neutral match start; > 0.50 = team advancing toward 1.0 knockout;
#          < 0.50 = opponent advancing toward our base.
PROPRIO_IDX_POS_X = 0
PROPRIO_IDX_POS_Y = 1
PROPRIO_IDX_POS_Z = 2
PROPRIO_IDX_VEL_X = 3
PROPRIO_IDX_VEL_Y = 4
PROPRIO_IDX_VEL_Z = 5
PROPRIO_IDX_CAN_COMMIT = 6
PROPRIO_IDX_TRANSIT_RESPAWN = 7
PROPRIO_IDX_HAZARD = 8
PROPRIO_IDX_VIS_RATIO = 9
PROPRIO_IDX_AUDIO_SIN = 10
PROPRIO_IDX_AUDIO_COS = 11
PROPRIO_IDX_AUDIO_LOUD = 12
PROPRIO_IDX_SURFACE_POSTURE = 13
PROPRIO_IDX_RESOURCE_CARRIER = 14
PROPRIO_IDX_OBJ_PROGRESS = 15

# Authoritative 8-Candidate Tactical Slot Indices (`M0 .. M7`)
CAND_IDX_OBJECTIVE_ADVANCE = 0
CAND_IDX_FRONTIER_TURF = 1
CAND_IDX_STRAFE_RIGHT_ENGAGE = 2
CAND_IDX_FLANK_DETOUR = 3
CAND_IDX_EVADE_REFUGE = 4
CAND_IDX_SHIELD_OR_BURST = 5
CAND_IDX_RALLY_OR_PICKUP = 6
CAND_IDX_IDLE_OBSERVE = 7

CAND_TO_GOAL_MODE_TUPLE: Tuple[int, ...] = (
    GOAL_MODE_PUSH_OBJECTIVE,     # M0 -> 1
    GOAL_MODE_BALANCED_EXPLORE,   # M1 -> 0
    GOAL_MODE_ENGAGE_INTERCEPT,   # M2 -> 2
    GOAL_MODE_STEALTH_FLANK,      # M3 -> 4
    GOAL_MODE_EVADE_REGROUP,      # M4 -> 3
    GOAL_MODE_ENGAGE_INTERCEPT,   # M5 -> 2
    GOAL_MODE_EVADE_REGROUP,      # M6 -> 3
    GOAL_MODE_EVADE_REGROUP,      # M7 -> 3
)

# 24D Allocentric-to-Egocentric Memory Compass (`mem_compass[24]`) Indices
MC_OBJ_DX = 0
MC_OBJ_DZ = 1
MC_OBJ_DIST = 2
MC_OBJ_VALID = 3
MC_FRONT_DX = 4
MC_FRONT_DZ = 5
MC_FRONT_DIST = 6
MC_FRONT_VALID = 7
MC_COV_DX = 8
MC_COV_DZ = 9
MC_COV_DIST = 10
MC_COV_VALID = 11
MC_THR_DX = 12
MC_THR_DZ = 13
MC_THR_DIST = 14
MC_THR_VALID = 15
MC_CLR_N = 16
MC_CLR_E = 17
MC_CLR_S = 18
MC_CLR_W = 19
MC_SURF_N = 20
MC_SURF_E = 21
MC_SURF_S = 22
MC_SURF_W = 23

# 32D Candidate-to-Compass Feature Matrix (`cand_feats32[8, 32]`) Indices
CFEAT_DIM = 32
CFEAT_NX = 0
CFEAT_NY = 1
CFEAT_NZ = 2
CFEAT_AIM_YAW = 3
CFEAT_AIM_PITCH = 4
CFEAT_SPEED_SCALE = 5
CFEAT_VERB_ONEHOT_BASE = 6       # [6..13] motor verb one-hot (8)
CFEAT_REL_DX = 14
CFEAT_REL_DZ = 15
CFEAT_STEP_NORM = 16
CFEAT_STRAFE_COS = 17
CFEAT_DOT_GOAL = 18
CFEAT_DOT_FRONTIER = 19
CFEAT_DOT_REFUGE = 20
CFEAT_DOT_THREAT = 21
CFEAT_DIR_CLEARANCE = 22
CFEAT_DIR_FRIENDLY_SURF = 23
CFEAT_IN_REFUGE = 24
CFEAT_CARRY_OR_TANK = 25
CFEAT_SIN_AIM_YAW = 26
CFEAT_COS_AIM_YAW = 27
CFEAT_AIM_PITCH_DUP = 28
CFEAT_HAZARD = 29
CFEAT_AUDIO_LOUD = 30
CFEAT_OBJ_PROGRESS = 31


def is_carrying_primary_objective(proprio: np.ndarray) -> bool:
    return bool(proprio.shape[0] > PROPRIO_IDX_RESOURCE_CARRIER and float(proprio[PROPRIO_IDX_RESOURCE_CARRIER]) >= 0.90)


def is_in_stealth_or_cover_refuge(proprio: np.ndarray) -> bool:
    return bool(proprio.shape[0] > PROPRIO_IDX_SURFACE_POSTURE and float(proprio[PROPRIO_IDX_SURFACE_POSTURE]) >= 0.90)


def get_resource_tank_level(proprio: np.ndarray) -> float:
    if proprio.shape[0] <= PROPRIO_IDX_RESOURCE_CARRIER:
        return 1.0
    v = float(proprio[PROPRIO_IDX_RESOURCE_CARRIER])
    return 1.0 if v >= 0.90 else float(np.clip(v / 0.88, 0.0, 1.0))


# ==============================================================================
# 4. Pluggable `TaskProfile3D` Configuration & Registry
# ==============================================================================
@dataclass
class TaskProfile3D:
    """
    Declarative task profile controlling candidate scales, ballistic lead-aim speed,
    surface-control weighting, and optional early-training DAgger guidance.
    """
    name: str = "universal_tactical"
    description: str = "Domain-agnostic 3D navigation, cover/surface control, objective & combat baseline"
    step_move_m: float = 2.25
    evade_move_m: float = 2.65
    stealth_move_m: float = 1.65
    interact_reach_m: float = 2.50
    ranged_engage_max_m: float = 9.50
    ranged_optimal_m: float = 5.50
    projectile_speed_mps: float = 14.0
    projectile_gravity_mps2: float = 0.0
    min_resource_to_fire: float = 0.18
    cover_dive_radius_m: float = 4.00
    surface_staleness_decay_per_sec: float = 0.045
    opponent_memory_decay_per_sec: float = 0.42
    objective_memory_decay_per_sec: float = 0.08
    frontier_turf_weight: float = 1.65
    frontier_unknown_weight: float = 2.40
    enable_strafe_aim_decoupling: bool = True
    enable_ballistic_lead_aim: bool = True


TASK_PROFILES: Dict[str, TaskProfile3D] = {
    "universal_tactical": TaskProfile3D(
        name="universal_tactical",
        description="Balanced general-purpose 3D action-tactical & objective baseline (zero game hardcoding).",
    ),
    "turf_objective": TaskProfile3D(
        name="turf_objective",
        description="Template prepared for InkBlitz 3D Turf + BlitzCore objective, swim-stealth & ranged combat.",
        step_move_m=2.15,
        evade_move_m=2.65,
        stealth_move_m=1.95,
        interact_reach_m=2.40,
        ranged_engage_max_m=8.40,
        ranged_optimal_m=5.20,
        projectile_speed_mps=18.0,
        projectile_gravity_mps2=11.5,
        min_resource_to_fire=0.14,
        cover_dive_radius_m=4.20,
        surface_staleness_decay_per_sec=0.060,
        opponent_memory_decay_per_sec=0.55,
        objective_memory_decay_per_sec=0.10,
        frontier_turf_weight=2.20,
        frontier_unknown_weight=2.10,
        enable_strafe_aim_decoupling=True,
        enable_ballistic_lead_aim=True,
    ),
}


def get_task_profile(profile_name: str) -> TaskProfile3D:
    key = (profile_name or "universal_tactical").strip().lower()
    return TASK_PROFILES.get(key, TASK_PROFILES["universal_tactical"])


# ==============================================================================
# 5. Clean 512-Token Universal 3D Grounded Vocabulary (Zero Legacy Game Holdovers)
# ==============================================================================
ENDOGENOUS_VOCAB_SIZE = 512
INNER_SPEECH_MAX_LEN = 12


class GroundedVocabulary512:
    """
    Compact 512-token grounded vocabulary for Vygotskian Joint-Attention & Read-Only Inner-Speech Telemetry:
      - 0..3    : <PAD>, <BOS>, <EOS>, <UNK>
      - 4..35   : Emergent RVQ Proto-Symbol Tokens (`◆00` .. `◆31`)
      - 36..83  : Grounded 3D spatial, territorial, objective, combat, and metacognitive concepts
      - 84..511 : Hash-grounded compositional expansion slots for external Caregiver / Ollama phrases
    """
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    PROTO_OFFSET = 4

    CORE_WORDS: List[str] = [
        # 36..47: Spatial & Territorial Concepts
        "前進", "後退", "右展開", "左展開", "中央", "高台",
        "遮蔽", "味方領域", "敵領域", "未踏域", "塗拡", "経路確保",
        # 48..59: Objective & Team Coordination Concepts
        "目標発見", "目標確保", "前進護衛", "台座突入", "拠点防衛", "敵搬送阻止",
        "味方合流", "連携", "資源補給", "潜伏回復", "索敵", "視界クリア",
        # 60..71: Threat, Combat & Timing Concepts
        "敵接近", "射線注意", "偏差射撃", "制圧射撃", "緊急離脱", "側面回避",
        "奇襲警戒", "音響反応", "好機到来", "待機観察", "即時実行", "安全確認",
        # 72..83: Metacognitive & State Concepts
        "優勢", "劣勢", "復帰直後", "残弾僅少", "満タン", "予測一致",
        "予測誤差", "危険回避", "突破", "維持", "再構築", "集中",
    ]

    def __init__(self) -> None:
        self.id_to_tok: Dict[int, str] = {
            0: "<PAD>",
            1: "<BOS>",
            2: "<EOS>",
            3: "<UNK>",
        }
        for s in range(32):
            self.id_to_tok[self.PROTO_OFFSET + s] = f"◆{s:02d}"
        for i, w in enumerate(self.CORE_WORDS):
            self.id_to_tok[36 + i] = w
        self.tok_to_id: Dict[str, int] = {v: k for k, v in self.id_to_tok.items()}

    def encode_utterance(
        self,
        text: str,
        proto_sym_id: int = 0,
        max_len: int = INNER_SPEECH_MAX_LEN,
    ) -> np.ndarray:
        ids = [self.BOS_ID, self.PROTO_OFFSET + (int(proto_sym_id) % 32)]
        cleaned = (text or "").strip()
        if cleaned:
            matched_any = False
            for w_idx, w in enumerate(self.CORE_WORDS):
                if w in cleaned:
                    ids.append(36 + w_idx)
                    matched_any = True
                    if len(ids) >= max_len - 1:
                        break
            if not matched_any:
                for chunk in cleaned.replace("。", " ").replace("、", " ").split():
                    h_id = 84 + (abs(hash(chunk)) % (ENDOGENOUS_VOCAB_SIZE - 84))
                    ids.append(h_id)
                    if len(ids) >= max_len - 1:
                        break
        ids.append(self.EOS_ID)
        arr = np.full(max_len, self.PAD_ID, dtype=np.int32)
        n = min(max_len, len(ids))
        arr[:n] = ids[:n]
        return arr

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        parts: List[str] = []
        for tid in token_ids:
            t_int = int(tid)
            if t_int in (self.PAD_ID, self.BOS_ID):
                continue
            if t_int == self.EOS_ID:
                break
            parts.append(self.id_to_tok.get(t_int, f"語{t_int}"))
        return "·".join(parts[:6]) if parts else "◆00·索敵"

    def decode_grounded_tokens(
        self,
        token_ids: Sequence[int],
        *,
        in_refuge: bool = False,
        carrying_obj: bool = False,
        hazard_level: float = 0.0,
        audio_loud: float = 0.0,
    ) -> str:
        """
        Context-grounded inner-speech decoder (Plan A Telemetry Sanitizer):
        Prevents ungrounded 'wait/hold' tokens from appearing on HUD when the agent is actively
        moving in open space or escaping danger.
        """
        parts: List[str] = []
        for tid in token_ids:
            t_int = int(tid)
            if t_int in (self.PAD_ID, self.BOS_ID):
                continue
            if t_int == self.EOS_ID:
                break
            tok = self.id_to_tok.get(t_int, f"語{t_int}")
            if tok == "待機観察" and (not in_refuge):
                if hazard_level >= 0.30 or audio_loud >= 0.25:
                    tok = "緊急離脱"
                elif carrying_obj:
                    tok = "前進護衛"
                else:
                    tok = "前進"
            if len(parts) == 0 or parts[-1] != tok:
                parts.append(tok)
        return "·".join(parts[:6]) if parts else "◆00·索敵"


# ==============================================================================
# 6. Multi-Stage Interactive DAgger Schedule & Domain-Agnostic Teacher Scaffold
# ==============================================================================
def compute_beta_teacher(policy_mode: str, transitions_recorded: int, anneal_transitions: int) -> float:
    """
    Unified teacher-mixing probability `beta_teacher in [0.0, 1.0]`:
      - "scaffold"    : 1.0 (pure heuristic scaffold for environment sanity checks)
      - "anneal"      : linear 0.75 -> 0.0 over `anneal_transitions`
      - "dagger"      : 4-stage interactive DAgger (0.65 -> 0.40 -> 0.18 -> 0.0) so the neural learner
                        always acts on its own state distribution from Episode 1 without 100% rule takeover
      - "pure_neural" : 0.0 (100% autonomous neural control)
    """
    mode = (policy_mode or "pure_neural").strip().lower()
    if mode == "pure_neural":
        return 0.0
    if mode == "scaffold":
        return 1.0
    horizon = max(1, int(anneal_transitions))
    prog = float(transitions_recorded) / float(horizon)
    if mode == "dagger":
        if prog < 0.25:
            return 0.65
        if prog < 0.55:
            return 0.40
        if prog < 0.85:
            return 0.18
        return 0.0
    return float(np.clip(0.75 * (1.0 - prog), 0.0, 0.75))


def compute_task_scaffold_targets(
    spatial_rec: Any,
    world_rec: Any,
    cand_poses: np.ndarray,
    cand_verbs: np.ndarray,
    world_min: Tuple[float, float, float],
    world_max: Tuple[float, float, float],
    profile: Optional[TaskProfile3D] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Clean, domain-generalized auxiliary teacher targets `(teacher_q[8], teacher_surv[8], best_idx)`
    computed strictly from 1st-person believed state (`spatial_rec.mem_compass` + `proprio`).
    Used ONLY when `policy_mode` is `"scaffold"`, `"anneal"`, or early `"dagger"`.
    """
    prof = profile or TASK_PROFILES["universal_tactical"]
    proprio = np.asarray(world_rec["proprio"], dtype=np.float32)
    bel_haz = float(spatial_rec["believed_hazard"])
    haz = max(float(world_rec["hazard_level"]), bel_haz)
    audio_loud = float(proprio[PROPRIO_IDX_AUDIO_LOUD]) if proprio.shape[0] > PROPRIO_IDX_AUDIO_LOUD else 0.0
    in_refuge = is_in_stealth_or_cover_refuge(proprio)
    carrying = is_carrying_primary_objective(proprio)
    tank = get_resource_tank_level(proprio)
    can_commit = bool(proprio.shape[0] > PROPRIO_IDX_CAN_COMMIT and float(proprio[PROPRIO_IDX_CAN_COMMIT]) > 0.5)

    if isinstance(spatial_rec, dict) or ("mem_compass" in (spatial_rec.dtype.names or ())):
        mc = np.asarray(spatial_rec["mem_compass"], dtype=np.float32).reshape(MEM_COMPASS_DIM)
    else:
        mc = np.zeros(MEM_COMPASS_DIM, dtype=np.float32)

    # Check if believed primary objective has a large physical barrier/shield (`radius >= 1.05`)
    shielded_obj = False
    if (not carrying) and (isinstance(spatial_rec, dict) or ("believed_entities" in (spatial_rec.dtype.names or ()))):
        b_ents = spatial_rec.get("believed_entities") if isinstance(spatial_rec, dict) else spatial_rec["believed_entities"]
        b_cnt = int(spatial_rec.get("believed_count", 0) if isinstance(spatial_rec, dict) else spatial_rec["believed_count"])
        if b_ents is not None:
            for ei in range(min(len(b_ents), b_cnt)):
                if SemanticTag.is_primary_objective(float(b_ents[ei][8])) and float(b_ents[ei][9]) > 0.10:
                    if float(b_ents[ei][7]) >= 1.05:
                        shielded_obj = True
                    break

    if isinstance(spatial_rec, dict) or ("believed_self_pos" in (getattr(getattr(spatial_rec, "dtype", None), "names", None) or ())):
        b_pos = spatial_rec.get("believed_self_pos") if isinstance(spatial_rec, dict) else spatial_rec["believed_self_pos"]
    else:
        b_pos = None
    if b_pos is not None:
        wmin_a = np.asarray(world_min, dtype=np.float32)
        wmax_a = np.asarray(world_max, dtype=np.float32)
        span_xz = np.maximum(1.0, (wmax_a - wmin_a)[[0, 2]])
        norm_xz = np.clip((np.asarray(b_pos, dtype=np.float32)[[0, 2]] - wmin_a[[0, 2]]) / span_xz, 0.0, 1.0)
        ego_xz = 2.0 * norm_xz - 1.0
    else:
        ego_xz = 2.0 * proprio[[PROPRIO_IDX_POS_X, PROPRIO_IDX_POS_Z]] - 1.0
    teacher_q = np.zeros(NUM_CANDIDATES, dtype=np.float32)
    teacher_surv = np.full(NUM_CANDIDATES, 0.82, dtype=np.float32)

    under_threat = (haz >= 0.30) or (audio_loud >= 0.28)
    low_resource = (not carrying) and (tank < prof.min_resource_to_fire)
    has_fire = can_commit if carrying else (not low_resource)

    for m in range(NUM_CANDIDATES):
        verb = int(cand_verbs[m])
        rel_xz = cand_poses[m, [0, 2]] - ego_xz
        d_norm = float(math.hypot(float(rel_xz[0]), float(rel_xz[1])))
        if d_norm > 1e-4:
            ux, uz = float(rel_xz[0]) / d_norm, float(rel_xz[1]) / d_norm
        else:
            ux, uz = 0.0, 0.0

        dot_obj = (ux * float(mc[MC_OBJ_DX]) + uz * float(mc[MC_OBJ_DZ])) * float(mc[MC_OBJ_VALID])
        dot_front = (ux * float(mc[MC_FRONT_DX]) + uz * float(mc[MC_FRONT_DZ])) * float(mc[MC_FRONT_VALID])
        dot_cov = (ux * float(mc[MC_COV_DX]) + uz * float(mc[MC_COV_DZ])) * float(mc[MC_COV_VALID])
        dot_thr = (ux * float(mc[MC_THR_DX]) + uz * float(mc[MC_THR_DZ])) * float(mc[MC_THR_VALID])

        wn = max(0.0, uz)
        we = max(0.0, ux)
        ws = max(0.0, -uz)
        ww = max(0.0, -ux)
        wsum = max(1e-4, wn + we + ws + ww)
        dir_clr = (wn * float(mc[MC_CLR_N]) + we * float(mc[MC_CLR_E]) + ws * float(mc[MC_CLR_S]) + ww * float(mc[MC_CLR_W])) / wsum if d_norm > 1e-4 else 1.0
        surf_adv = (wn * float(mc[MC_SURF_N]) + we * float(mc[MC_SURF_E]) + ws * float(mc[MC_SURF_S]) + ww * float(mc[MC_SURF_W])) / wsum if d_norm > 1e-4 else 0.5

        q_m = 0.35 * dir_clr + 0.25 * surf_adv
        surv_m = float(np.clip(0.88 - 0.55 * haz - 0.30 * max(0.0, dot_thr), 0.08, 0.98))

        if in_refuge and (not carrying) and tank < 0.72:
            if verb == VERB_IDLE_OBSERVE:
                q_m += 1.95
                surv_m = 0.95
            elif m == 0:
                q_m -= 0.80
        elif under_threat and (haz >= 0.54 or low_resource):
            if verb == VERB_EVADE_HAZARD and d_norm > 0.04:
                q_m += 2.25 + 0.65 * max(0.0, -dot_thr) + 0.55 * max(0.0, dot_cov)
                surv_m = float(np.clip(surv_m + 0.32, 0.45, 0.96))
            elif verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH) and has_fire and haz < 0.72:
                q_m += 1.55
                surv_m = float(np.clip(surv_m + 0.15, 0.35, 0.90))
            elif verb == VERB_IDLE_OBSERVE:
                q_m -= 2.20
                surv_m = 0.10
            else:
                q_m += 0.45 * max(0.0, -dot_thr)
        else:
            if carrying:
                q_m += 2.55 * dot_obj + 0.40 * max(0.0, -dot_thr)
                if m == 0:
                    q_m += 0.75
                elif verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH) and has_fire and (under_threat or surf_adv < 0.45):
                    q_m += 1.25
            elif float(mc[3]) > 0.5:
                if shielded_obj and has_fire and float(mc[2]) <= 0.24:
                    # Breakable/shielded objective within ~8.4m firing range: favor standoff ranged fire (M5/M0 standoff/M2) over walking into blast
                    if m == 5 and verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH):
                        q_m += 2.45
                    elif m == 0 and verb == VERB_RELEASE_PLACE and float(mc[2]) >= 0.10:
                        q_m += 2.05
                    elif verb == VERB_RELEASE_PLACE:
                        q_m += 1.35 + 0.60 * dot_obj
                    elif m == 0 and float(mc[2]) < 0.10:
                        q_m -= 1.25
                        surv_m = min(surv_m, 0.28)
                    else:
                        q_m += 0.90 * dot_obj
                else:
                    # Collectable bare objective or distant objective/enemy carrier: rush directly onto it!
                    q_m += 2.25 * dot_obj
                    if m == 0:
                        q_m += 0.85
            else:
                q_m += 1.45 * dot_front

            if verb == VERB_RELEASE_PLACE and has_fire and (under_threat or float(mc[15]) > 0.5):
                q_m += 1.30
            if verb == VERB_IDLE_OBSERVE:
                q_m -= 1.15

        teacher_q[m] = float(q_m)
        teacher_surv[m] = float(np.clip(surv_m, 0.05, 0.98))

    best_idx = int(np.argmax(teacher_q))
    return teacher_q, teacher_surv, best_idx


def synthesize_joint_attention_phrase(
    world_rec: Any,
    spatial_rec: Any,
    chosen_verb: int,
    goal_mode: int = GOAL_MODE_BALANCED_EXPLORE,
) -> str:
    """Generate clean, grounded 3D Japanese telemetry / joint-attention phrases (zero legacy game words)."""
    proprio = np.asarray(world_rec["proprio"], dtype=np.float32)
    haz = max(float(world_rec["hazard_level"]), float(spatial_rec["believed_hazard"]))
    audio_loud = float(proprio[PROPRIO_IDX_AUDIO_LOUD]) if proprio.shape[0] > PROPRIO_IDX_AUDIO_LOUD else 0.0
    in_refuge = is_in_stealth_or_cover_refuge(proprio)
    carrying = is_carrying_primary_objective(proprio)
    tank = get_resource_tank_level(proprio)

    if in_refuge:
        if haz >= 0.30 or audio_loud >= 0.25:
            return "遮蔽 潜伏回復 射線注意"
        if tank < 0.70:
            return "味方領域 資源補給 潜伏回復"
        return "安全確認 経路確保 前進"
    if haz >= 0.55:
        if chosen_verb in (VERB_RELEASE_PLACE, VERB_IMPULSE_PUSH):
            return "敵接近 偏差射撃 迎撃"
        return "敵接近 緊急離脱 遮蔽"
    if carrying:
        return "目標確保 前進護衛 台座突入"
    if goal_mode == GOAL_MODE_ENGAGE_INTERCEPT or chosen_verb == VERB_RELEASE_PLACE:
        return "敵搬送阻止 偏差射撃 塗拡"
    if chosen_verb in (VERB_COMMIT_INTERACT, VERB_GRASP_PICK):
        return "目標発見 即時実行 目標確保"
    if audio_loud >= 0.24:
        return "音響反応 奇襲警戒 側面回避"
    return "索敵 塗拡 前進"
