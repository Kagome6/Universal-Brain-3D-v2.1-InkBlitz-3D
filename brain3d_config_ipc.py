# -*- coding: utf-8 -*-
"""
brain3d_config_ipc.py — Universal 3D Embodiment Config, POMDP/SafetyShield Rules,
Lock-Free SeqLock+CRC32 Shared-Memory Buses, Unity Wire Protocol & Retinotopic Helpers
======================================================================================
Barrier 1 Guarantee: Strictly Zero-PyTorch (`"torch" not in sys.modules`).
License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
import ctypes
from dataclasses import dataclass
import heapq
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import queue
import socket
import struct
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import zlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_PERIPH_MODULES_DIR = os.path.join(_CORE_DIR, "周辺モジュール")
if os.path.isdir(_PERIPH_MODULES_DIR) and _PERIPH_MODULES_DIR not in sys.path:
    sys.path.append(_PERIPH_MODULES_DIR)

from task_profiles_3d import (
    CAND_IDX_EVADE_REFUGE,
    CAND_IDX_FLANK_DETOUR,
    CAND_IDX_FRONTIER_TURF,
    CAND_IDX_IDLE_OBSERVE,
    CAND_IDX_OBJECTIVE_ADVANCE,
    CAND_IDX_RALLY_OR_PICKUP,
    CAND_IDX_SHIELD_OR_BURST,
    CAND_IDX_STRAFE_RIGHT_ENGAGE,
    CAND_TO_GOAL_MODE_TUPLE,
    CFEAT_AIM_PITCH,
    CFEAT_AIM_PITCH_DUP,
    CFEAT_AIM_YAW,
    CFEAT_AUDIO_LOUD,
    CFEAT_CARRY_OR_TANK,
    CFEAT_COS_AIM_YAW,
    CFEAT_DIM,
    CFEAT_DIR_CLEARANCE,
    CFEAT_DIR_FRIENDLY_SURF,
    CFEAT_DOT_FRONTIER,
    CFEAT_DOT_GOAL,
    CFEAT_DOT_REFUGE,
    CFEAT_DOT_THREAT,
    CFEAT_HAZARD,
    CFEAT_IN_REFUGE,
    CFEAT_NX,
    CFEAT_NY,
    CFEAT_NZ,
    CFEAT_OBJ_PROGRESS,
    CFEAT_REL_DX,
    CFEAT_REL_DZ,
    CFEAT_SIN_AIM_YAW,
    CFEAT_SPEED_SCALE,
    CFEAT_STEP_NORM,
    CFEAT_STRAFE_COS,
    CFEAT_VERB_ONEHOT_BASE,
    CONT_POSE_DIM,
    ENDOGENOUS_VOCAB_SIZE,
    GOAL_MODE_BALANCED_EXPLORE,
    GOAL_MODE_ENGAGE_INTERCEPT,
    GOAL_MODE_EVADE_REGROUP,
    GOAL_MODE_NAMES,
    GOAL_MODE_PUSH_OBJECTIVE,
    GOAL_MODE_STEALTH_FLANK,
    GroundedVocabulary512,
    INNER_SPEECH_MAX_LEN,
    MC_CLR_E,
    MC_CLR_N,
    MC_CLR_S,
    MC_CLR_W,
    MC_COV_DIST,
    MC_COV_DX,
    MC_COV_DZ,
    MC_COV_VALID,
    MC_FRONT_DIST,
    MC_FRONT_DX,
    MC_FRONT_DZ,
    MC_FRONT_VALID,
    MC_OBJ_DIST,
    MC_OBJ_DX,
    MC_OBJ_DZ,
    MC_OBJ_VALID,
    MC_SURF_E,
    MC_SURF_N,
    MC_SURF_S,
    MC_SURF_W,
    MC_THR_DIST,
    MC_THR_DX,
    MC_THR_DZ,
    MC_THR_VALID,
    MEM_COMPASS_DIM,
    NUM_CANDIDATES,
    NUM_MOTOR_VERBS,
    PROPRIO_DIM,
    PROPRIO_IDX_AUDIO_COS,
    PROPRIO_IDX_AUDIO_LOUD,
    PROPRIO_IDX_AUDIO_SIN,
    PROPRIO_IDX_CAN_COMMIT,
    PROPRIO_IDX_HAZARD,
    PROPRIO_IDX_OBJ_PROGRESS,
    PROPRIO_IDX_POS_X,
    PROPRIO_IDX_POS_Y,
    PROPRIO_IDX_POS_Z,
    PROPRIO_IDX_RESOURCE_CARRIER,
    PROPRIO_IDX_SURFACE_POSTURE,
    PROPRIO_IDX_TRANSIT_RESPAWN,
    PROPRIO_IDX_VEL_X,
    PROPRIO_IDX_VEL_Y,
    PROPRIO_IDX_VEL_Z,
    PROPRIO_IDX_VIS_RATIO,
    SemanticTag,
    TaskProfile3D,
    VERB_APPROACH_HOVER,
    VERB_COMMIT_INTERACT,
    VERB_COMMUNICATE_PING,
    VERB_EVADE_HAZARD,
    VERB_GRASP_PICK,
    VERB_IDLE_OBSERVE,
    VERB_IMPULSE_PUSH,
    VERB_NAMES_EN,
    VERB_RELEASE_PLACE,
    compute_beta_teacher,
    compute_task_scaffold_targets,
    get_resource_tank_level,
    get_task_profile,
    is_carrying_primary_objective,
    is_in_stealth_or_cover_refuge,
    synthesize_joint_attention_phrase,
)


# ==============================================================================
# 1. Universal 3D Embodiment Constants & Configuration (Zero Game Hardcoding)
# ==============================================================================
BRAIN3D_MAGIC = 0x42334432        # "B3D2" SharedMemory SeqLock Magic (v2.0)
BRAIN3D_VERSION = 2

UNITY_SENSORY_MAGIC = 0x33444255  # "UB3D" (Unity -> Python Sensory Frame)
UNITY_MOTOR_MAGIC = 0x434D4255    # "UBMC" (Python -> Unity Motor + Independent Aim/Gaze Command)
UNITY_PROTO_VERSION = 1
UNITY_PROTO_VERSION_V2 = 2

FOVEA_H = 16
FOVEA_W = 16
FOVEA_C = 4                       # [norm_depth, normal_x, normal_y, surface_or_salience_code]
VENTRAL_C = 4                     # [rg_opponent, by_opponent, local_tex_sat, ego_damped_temp_delta]
VIS8_C = FOVEA_C + VENTRAL_C      # 8-channel unified dorsal(4) + ventral(4) retinotopic patch
NUM_OBJECT_SLOTS = 4              # Slot 0 = wide background/turf, Slots 1..3 = salient foreground objects
OBJECT_SLOT_DIM = 32
MULTIMODAL_EMBED_DIM = 64

RETINA_H = 64
RETINA_W = 64
RETINA_C = 3
RETINA_EYE_BYTES = RETINA_H * RETINA_W * RETINA_C          # 12,288 bytes per eye
RETINA_STEREO_BYTES = 2 * RETINA_EYE_BYTES                 # 24,576 bytes (Left + Right RGB24)
CALIB_TEACHER_BYTES = FOVEA_H * FOVEA_W * FOVEA_C * 4      # 4,096 bytes (16x16x4 float32 teacher)

VISUAL_MODE_RAYCAST = "raycast"
VISUAL_MODE_NEURAL_STEREO = "neural_stereo"
VISUAL_MODE_CALIBRATE = "calibrate"
VISUAL_MODE_TO_ID = {
    VISUAL_MODE_RAYCAST: 0,
    VISUAL_MODE_NEURAL_STEREO: 1,
    VISUAL_MODE_CALIBRATE: 2,
}
ID_TO_VISUAL_MODE = {v: k for k, v in VISUAL_MODE_TO_ID.items()}

MAX_ENTITIES = 48
ENTITY_DIM = 10                   # [id_hash, x, y, z, vx, vy, vz, radius, affordance_tag, confidence]

MEC_GRID_DIM = 256                # 4 spatial wavelengths x (4x4x4 = 64) 3D FCC phase cells
HD_RING_DIM = 32                  # 32 Fibonacci-sphere 3D Head-Direction cells
FOG_GRID_X = 8
FOG_GRID_Y = 4
FOG_GRID_Z = 8
FOG_VOXELS = FOG_GRID_X * FOG_GRID_Y * FOG_GRID_Z  # 256
FIELD_MAP_MAX_N = 48              # 48x48 multi-layer cognitive field grid

LATENT_Z_DIM = 32
GWT_DIM = 128
SELF_MODEL_DIM = 8
RSSM_DETER_DIM = 256
RSSM_STOCH_CATS = 8
RSSM_STOCH_CLASSES = 8
RSSM_STOCH_FLAT = RSSM_STOCH_CATS * RSSM_STOCH_CLASSES  # 64

NUM_PROTO_SYMBOLS = 32
NUM_PROTO_MODS = 8
PROTO_EMBED_DIM = 32
INNER_SPEECH_EMBED_DIM = 32
S640_DIGEST_BYTES = 640

# Basal Ganglia STN-SNr Timing Modes & Generalized Hold Reasons
TIMING_MODE_BG_ADAPTIVE = 0
TIMING_MODE_INSTANT = 1
TIMING_MODE_FIXED_SETTLE = 2
TIMING_MODE_CAUTIOUS = 3
TIMING_MODE_NAMES = {
    TIMING_MODE_BG_ADAPTIVE: "BG-ADAPT",
    TIMING_MODE_INSTANT: "INSTANT",
    TIMING_MODE_FIXED_SETTLE: "FIX-500ms",
    TIMING_MODE_CAUTIOUS: "CAUTIOUS",
}

HOLD_GO_CLEAR = 0
HOLD_COOLDOWN_PREP = 1
HOLD_OPPORTUNITY_SYNC = 2         # Waiting for ballistic lead-aim / tactical window alignment
HOLD_KINETIC_CROSSFIRE = 3        # Holding risky exposure during peak incoming crossfire
HOLD_CONFLICT_ARBITRATION = 4     # Arbitrating System1 vs ConsciousGWT disagreement
HOLD_EPISTEMIC_FOG_SACCADE = 5    # Look-Before-Commit: clearing high epistemic fog before blind rush
HOLD_POST_RESPAWN_REANCHOR = 6    # Brief spatial/GWT re-anchoring right after respawn/event
HOLD_EXTERNAL_COACH_BRAKE = 7     # Explicit external coach / human hold command
HOLD_COVER_STEALTH_REGEN = 8      # Holding submerged/cover refuge while threat passes or tank regens
HOLD_REASON_NAMES = {
    HOLD_GO_CLEAR: "GO-CLEAR",
    HOLD_COOLDOWN_PREP: "COOLDOWN",
    HOLD_OPPORTUNITY_SYNC: "LEAD-SYNC",
    HOLD_KINETIC_CROSSFIRE: "CROSSFIRE-HOLD",
    HOLD_CONFLICT_ARBITRATION: "SYS1-GWT-ARB",
    HOLD_EPISTEMIC_FOG_SACCADE: "FOG-SACCADE",
    HOLD_POST_RESPAWN_REANCHOR: "REANCHOR",
    HOLD_EXTERNAL_COACH_BRAKE: "COACH-BRAKE",
    HOLD_COVER_STEALTH_REGEN: "COVER-REGEN-HOLD",
}

SPATIAL_MODE_FULL_MEC_3D = 0
SPATIAL_MODE_DRIFT_ONLY = 1
SPATIAL_MODE_TUNNEL_RAW = 2
SPATIAL_MODE_OMNI_DEBUG = 3

HIPPO_MEM_ADAPTIVE = "adaptive"                 # Inference: 100% same-stage structural carry; Training: stochastic carry + fresh mix
HIPPO_MEM_CARRY_SAME_STAGE = "carry_same_stage" # Always carry structural topology on same stage_id (>0), reset on stage_id change
HIPPO_MEM_RESET_ALWAYS = "reset_always"         # Legacy clean-slate reset every episode regardless of stage_id
HIPPO_MEM_MODES = (HIPPO_MEM_ADAPTIVE, HIPPO_MEM_CARRY_SAME_STAGE, HIPPO_MEM_RESET_ALWAYS)

CH_REFLEX_S1 = 0
CH_CONSCIOUS_GWT = 1
CH_GAZE_SACCADE = 2
CH_SYSTEM_CONTROL = 3
NUM_INTENT_CHANNELS = 4

SYS_CMD_NONE = 0
SYS_CMD_SLEEP_START = 1
SYS_CMD_SLEEP_DONE = 2
SYS_CMD_SHUTDOWN = 99


@dataclass
class Brain3DConfig:
    """Universal configuration for any 3D tactical, territorial, or action environment."""
    task_profile: str = "universal_tactical"  # "universal_tactical" | "turf_objective"
    world_min: Tuple[float, float, float] = (-12.0, 0.0, -16.0)
    world_max: Tuple[float, float, float] = (12.0, 6.0, 16.0)
    fovea_half_angle_deg: float = 38.0
    fovea_max_range_m: float = 16.0
    odom_noise_scale: float = 0.85
    survival_veto_threshold: float = 0.42
    timing_mode: int = TIMING_MODE_BG_ADAPTIVE
    spatial_mode: int = SPATIAL_MODE_FULL_MEC_3D
    default_stage_id: int = 1                 # Stage topology ID (>0 enables stage-keyed structural schema bank)
    hippo_memory_mode: str = HIPPO_MEM_ADAPTIVE
    hippo_train_carry_prob: float = 0.50      # In 'adaptive' training, 50% known-stage carry + 50% zero-shot fresh exploration
    enable_tactical_map_sync: bool = False    # Optional low-rate HUD turf/objective overview sync
    world_hz: float = 60.0
    spatial_hz: float = 60.0
    reflex_hz: float = 24.0
    conscious_hz: float = 12.0
    batch_window_ms: float = 2.0
    enable_endogenous_broca: bool = True      # Generates grounded inner-speech telemetry
    allow_broca_motor_bias: bool = False      # Plan A default: False (0.0 motor interference from inner speech)
    policy_mode: str = "pure_neural"          # "scaffold" | "anneal" | "dagger" | "pure_neural"
    anneal_transitions: int = 1600
    time_scale: float = 1.0
    enable_learning: bool = True
    autosave_per_episode: bool = False
    save_on_shutdown: bool = True
    enable_overfit_guard: bool = True
    continual_anchor_l2: float = 1.2e-4
    caregiver_curriculum: str = "developmental"
    sleep_epochs_per_episode: int = 6
    online_update_interval: int = 8
    load_weights_path: str = ""
    save_weights_path: str = ""
    metrics_out_path: str = ""
    visual_mode: str = VISUAL_MODE_RAYCAST
    depth_weights_path: str = ""
    save_depth_weights_path: str = ""
    depth_calib_metrics_path: str = ""
    visual_entity_dropout: float = 0.0
    stereo_baseline_m: float = 0.24
    stereo_fov_deg: float = 76.0
    llm_backend: str = "symbolic"             # "symbolic" | "ollama" | "auto"
    ollama_model: str = "gemma4:12b"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_compute_mode: str = "hybrid"       # "hybrid" (GPU+CPU split) | "cpu" (num_gpu=0) | "gpu"
    enable_strategy_kb: bool = False          # Enabled in play/train/spectate CLI entrypoints; False in synthetic tests
    strategy_db_path: str = ""
    enable_safety_shield: bool = True         # Unified SafetyShield toggle (True = tactical anti-freeze/escape shield; False = pure neural ablation)
    strict_pomdp_observation: bool = True     # Replaces ground-truth proprio[0:3] with EKF-SLAM believed_self_pos (eliminates true-pose leak)
    export_field_maps_to_shm: bool = False    # Optional 11.5KB 4-layer map SHM bus export (off by default to save 72% 60Hz SHM/CRC bandwidth)
    unity_socket_timeout_sec: float = 2.5     # Resilient TCP timeout preventing false disconnects during sleep or Unity frame spikes
    unity_host: str = "127.0.0.1"
    unity_port: int = 50055
    ux_telemetry_port: int = 0
    work_dir: str = "."
    seed: int = 2026
    # Evolved PBT Brain Genes
    gene_archetype: str = "Balanced"
    lr_scale: float = 1.0
    curiosity_fog_gain: float = 1.0
    stn_stealth_gain: float = 1.0
    broca_prior_gain: float = 1.0
    policy_temp: float = 0.34


# ==============================================================================
# 1.5 Unified POMDP Self-Pose Projection & Centralized SafetyShield Rules
# ==============================================================================
SHIELD_HAZARD_CF_MASK = 0.35
SHIELD_HAZARD_S1_OVERRIDE = 0.35
SHIELD_HAZARD_GWT_HIGH = 0.55
SHIELD_HAZARD_GWT_LOW_TANK = 0.32
SHIELD_HAZARD_BG_INSTANT = 0.42
SHIELD_KINETIC_BG_INSTANT = 0.48


def project_pomdp_proprio(
    proprio: np.ndarray,
    believed_self_pos: Optional[np.ndarray],
    world_min: Tuple[float, float, float],
    world_max: Tuple[float, float, float],
    cfg: Optional[Brain3DConfig] = None,
) -> np.ndarray:
    """
    Strict 1st-Person POMDP Observation Filter:
    Replaces ground-truth normalized coordinates `proprio[0:3]` with EKF-SLAM `believed_self_pos`
    normalized into `[0.0, 1.0]`, preserving exact `[16]` checkpoint input compatibility while
    eliminating omniscient position leakage into neural networks and policies.
    """
    out = np.asarray(proprio, dtype=np.float32).copy()
    if believed_self_pos is None:
        return out
    if cfg is not None and (not bool(getattr(cfg, "strict_pomdp_observation", True))):
        return out
    wmin = np.asarray(world_min, dtype=np.float32)
    wmax = np.asarray(world_max, dtype=np.float32)
    span = np.maximum(1.0, wmax - wmin)
    bpos = np.asarray(believed_self_pos, dtype=np.float32).reshape(3)
    out[PROPRIO_IDX_POS_X : PROPRIO_IDX_POS_Z + 1] = np.clip((bpos - wmin) / span, 0.0, 1.0)
    return out


def apply_s1_safety_shield(
    best_idx: int,
    cand_verbs: np.ndarray,
    proprio: np.ndarray,
    cfg: Optional[Brain3DConfig] = None,
) -> int:
    """Centralized SafetyShield Rule #1: Fast S1 Reflex open-space anti-freeze override."""
    idx = int(best_idx)
    if cfg is not None and (not bool(getattr(cfg, "enable_safety_shield", True))):
        return idx
    if (
        (not is_in_stealth_or_cover_refuge(proprio))
        and float(proprio[PROPRIO_IDX_HAZARD]) >= SHIELD_HAZARD_S1_OVERRIDE
        and int(cand_verbs[idx]) == VERB_IDLE_OBSERVE
    ):
        return CAND_IDX_EVADE_REFUGE
    return idx


def evaluate_gwt_safety_shield_state(
    proprio: np.ndarray,
    cfg: Optional[Brain3DConfig] = None,
) -> Tuple[bool, bool]:
    """Centralized SafetyShield Rule #2: ConsciousGWT exposed-threat detection & shield activation."""
    shield_on = True if cfg is None else bool(getattr(cfg, "enable_safety_shield", True))
    in_refuge = is_in_stealth_or_cover_refuge(proprio)
    haz = float(proprio[PROPRIO_IDX_HAZARD])
    tank = get_resource_tank_level(proprio)
    exposed_threat = (not in_refuge) and (
        haz >= SHIELD_HAZARD_GWT_HIGH or (haz >= SHIELD_HAZARD_GWT_LOW_TANK and tank < 0.16)
    )
    return shield_on, exposed_threat


def is_bg_emergency_escape_exempt(
    chosen_verb: int,
    in_refuge: bool,
    hazard_level: float,
    local_kinetic_risk: float,
    cfg: Optional[Brain3DConfig] = None,
) -> bool:
    """Centralized SafetyShield Rule #3: Basal Ganglia 0ms hyperdirect escape exemption."""
    if cfg is not None and (not bool(getattr(cfg, "enable_safety_shield", True))):
        return int(chosen_verb) == VERB_EVADE_HAZARD
    exposed_danger = (not in_refuge) and (
        float(hazard_level) >= SHIELD_HAZARD_BG_INSTANT
        or float(local_kinetic_risk) >= SHIELD_KINETIC_BG_INSTANT
    )
    return int(chosen_verb) == VERB_EVADE_HAZARD or exposed_danger


def create_runtime_strategy_controller(cfg: Brain3DConfig, slot_id: int = 0) -> Optional[Any]:
    """
    Barrier-1 safe factory for the 4-boundary Closed-Loop `RuntimeStrategyController`.
    In parallel 4x training (`slot_id > 0` or `time_scale > 1.8`), only `slot_id == 0` invokes
    background Ollama Gemma 4 12B (throttled to `>= 40s`) while all slots share fast SQLite retrieval.
    """
    if not bool(getattr(cfg, "enable_strategy_kb", False)):
        return None
    from strategy_knowledge_engine import RuntimeStrategyController

    db_path = str(getattr(cfg, "strategy_db_path", "") or "").strip()
    if not db_path:
        base_dir = os.path.abspath(str(getattr(cfg, "work_dir", ".") or "."))
        db_path = os.path.join(base_dir, "checkpoints", "user_runs", "strategy_knowledge.sqlite3")
    fast_train = float(getattr(cfg, "time_scale", 1.0)) > 2.2 or int(slot_id) > 0
    enable_async = int(slot_id) == 0
    enable_ollama = int(slot_id) == 0 and str(getattr(cfg, "llm_backend", "auto")) != "none"
    min_interval = 40.0 if fast_train else 6.0
    return RuntimeStrategyController(
        db_path=db_path,
        seed=int(getattr(cfg, "seed", 2026)) + int(slot_id) * 37,
        enable_async_analyst=enable_async,
        enable_ollama=enable_ollama,
        ollama_url=str(getattr(cfg, "ollama_url", "http://127.0.0.1:11434")),
        ollama_model=str(getattr(cfg, "ollama_model", "gemma4:12b")),
        ollama_compute_mode=str(getattr(cfg, "ollama_compute_mode", "hybrid")),
        min_ollama_interval_s=min_interval,
        emit_ui_stat=(int(slot_id) == 0),
    )


def extract_brain_genes_from_cfg(cfg: Brain3DConfig) -> Dict[str, Any]:
    return {
        "archetype": str(getattr(cfg, "gene_archetype", "Balanced")),
        "lr_scale": float(np.clip(getattr(cfg, "lr_scale", 1.0), 0.30, 2.50)),
        "curiosity_fog_gain": float(np.clip(getattr(cfg, "curiosity_fog_gain", 1.0), 0.25, 2.60)),
        "stn_stealth_gain": float(np.clip(getattr(cfg, "stn_stealth_gain", 1.0), 0.35, 2.60)),
        "broca_prior_gain": float(np.clip(getattr(cfg, "broca_prior_gain", 1.0), 0.20, 2.40)),
        "policy_temp": float(np.clip(getattr(cfg, "policy_temp", 0.34), 0.15, 0.85)),
        "survival_veto_threshold": float(np.clip(getattr(cfg, "survival_veto_threshold", 0.42), 0.25, 0.62)),
    }


def load_checkpoint_contract_meta(ckpt_path: str) -> Dict[str, Any]:
    """Barrier-1 safe (zero `torch` import) reader for `<ckpt>.meta.json`."""
    if not ckpt_path:
        return {}
    meta_path = ckpt_path + ".meta.json"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def apply_checkpoint_genes_to_cfg(cfg: Brain3DConfig, genes: Optional[Dict[str, Any]]) -> Brain3DConfig:
    if not isinstance(genes, dict):
        return cfg
    if "archetype" in genes:
        cfg.gene_archetype = str(genes["archetype"])
    if "lr_scale" in genes:
        cfg.lr_scale = float(np.clip(float(genes["lr_scale"]), 0.30, 2.50))
    if "curiosity_fog_gain" in genes:
        cfg.curiosity_fog_gain = float(np.clip(float(genes["curiosity_fog_gain"]), 0.25, 2.60))
    if "stn_stealth_gain" in genes:
        cfg.stn_stealth_gain = float(np.clip(float(genes["stn_stealth_gain"]), 0.35, 2.60))
    if "broca_prior_gain" in genes:
        cfg.broca_prior_gain = float(np.clip(float(genes["broca_prior_gain"]), 0.20, 2.40))
    if "policy_temp" in genes:
        cfg.policy_temp = float(np.clip(float(genes["policy_temp"]), 0.15, 0.85))
    if "survival_veto_threshold" in genes:
        cfg.survival_veto_threshold = float(np.clip(float(genes["survival_veto_threshold"]), 0.25, 0.62))
    return cfg


def _clamp_utf8_bytes(text: str, max_bytes: int) -> bytes:
    raw = (text or "").encode("utf-8", errors="ignore")
    if len(raw) <= max_bytes:
        return raw
    cut = raw[:max_bytes]
    while cut and (cut[-1] & 0xC0) == 0x80:
        cut = cut[:-1]
    if cut and (cut[-1] & 0x80) != 0:
        lead = cut[-1]
        seq_len = 4 if (lead & 0xF8) == 0xF0 else (3 if (lead & 0xF0) == 0xE0 else 2)
        if len(cut) < seq_len:
            cut = cut[:-1]
    try:
        cut.decode("utf-8")
        return cut
    except UnicodeDecodeError:
        return cut[:-1]


# ==============================================================================
# 2. OS Watchdog & Lock-Free SeqLock + CRC32 Shared Memory Buses
# ==============================================================================
_WIN_K32 = None


def _ensure_win_k32():
    global _WIN_K32
    if _WIN_K32 is None and os.name == "nt":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        k32.GetExitCodeProcess.restype = ctypes.c_int
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.CloseHandle.restype = ctypes.c_int
        k32.GetStdHandle.argtypes = [ctypes.c_uint32]
        k32.GetStdHandle.restype = ctypes.c_void_p
        k32.GetFileType.argtypes = [ctypes.c_void_p]
        k32.GetFileType.restype = ctypes.c_uint32
        k32.PeekNamedPipe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        k32.PeekNamedPipe.restype = ctypes.c_int
        k32.ReadFile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        k32.ReadFile.restype = ctypes.c_int
        _WIN_K32 = k32
    return _WIN_K32


def is_windows_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = _ensure_win_k32()
    h = k32.OpenProcess(0x1000, 0, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_uint32(0)
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        return bool(ok and code.value == 259)
    finally:
        k32.CloseHandle(h)


def start_parent_and_stdin_watchdog(
    parent_pid: int,
    on_parent_lost: Callable[[], None],
    watch_stdin: bool = False,
    on_emergency_abort: Optional[Callable[[], None]] = None,
) -> None:
    """Guarantees zero orphaned Python brain processes if Unity or parent process exits abruptly."""
    triggered = threading.Event()

    def _fire(is_emergency: bool = False) -> None:
        if not triggered.is_set():
            triggered.set()
            if is_emergency and on_emergency_abort is not None:
                try:
                    on_emergency_abort()
                except Exception:
                    pass
            try:
                on_parent_lost()
            except Exception:
                pass

    if parent_pid > 0:
        def _poll_pid() -> None:
            while not triggered.is_set():
                if not is_windows_pid_alive(parent_pid):
                    _fire(is_emergency=True)
                    return
                time.sleep(0.5)
        threading.Thread(target=_poll_pid, daemon=True, name="BrainParentWatchdog").start()

    if watch_stdin:
        if os.name == "nt":
            def _watch_stdin_win32() -> None:
                """
                Non-blocking Win32 `PeekNamedPipe` watchdog:
                  - Never blocks on synchronous `ReadFile` when empty (prevents Win32 `spawn` child CRT `GetFileType` deadlock).
                  - Never acquires CPython `sys.stdin.buffer` lock (prevents `_enter_buffered_busy` crash on normal exit).
                """
                try:
                    k32 = _ensure_win_k32()
                    h_stdin = k32.GetStdHandle(0xFFFFFFF6)  # STD_INPUT_HANDLE (-10)
                    invalid_handle = ctypes.c_void_p(-1).value
                    if not h_stdin or h_stdin == invalid_handle:
                        return
                    if k32.GetFileType(h_stdin) != 0x0003:  # FILE_TYPE_PIPE
                        return
                    avail = ctypes.c_uint32(0)
                    nread = ctypes.c_uint32(0)
                    buf = ctypes.create_string_buffer(64)
                    while not triggered.is_set():
                        avail.value = 0
                        ok = k32.PeekNamedPipe(h_stdin, None, 0, None, ctypes.byref(avail), None)
                        if not ok:
                            _fire(is_emergency=False)
                            return
                        if avail.value > 0:
                            to_read = min(int(avail.value), 64)
                            nread.value = 0
                            if not k32.ReadFile(h_stdin, buf, to_read, ctypes.byref(nread), None) or nread.value == 0:
                                _fire(is_emergency=False)
                                return
                            chunk = buf.raw[: int(nread.value)]
                            if b"EMERGENCY_STOP" in chunk or b"ABORT" in chunk:
                                _fire(is_emergency=True)
                                return
                            if b"SHUTDOWN" in chunk or b"STOP" in chunk:
                                _fire(is_emergency=False)
                                return
                        time.sleep(0.04)
                except Exception:
                    _fire(is_emergency=False)
            threading.Thread(target=_watch_stdin_win32, daemon=True, name="BrainStdinWatchdog").start()
        else:
            def _watch_eof_posix() -> None:
                import select
                try:
                    fd = sys.stdin.fileno()
                    while not triggered.is_set():
                        rdy, _, _ = select.select([fd], [], [], 0.05)
                        if rdy:
                            data = os.read(fd, 64)
                            if not data:
                                _fire(is_emergency=False)
                                return
                            if b"EMERGENCY_STOP" in data or b"ABORT" in data:
                                _fire(is_emergency=True)
                                return
                            if b"SHUTDOWN" in data or b"STOP" in data:
                                _fire(is_emergency=False)
                                return
                except Exception:
                    _fire(is_emergency=False)
            threading.Thread(target=_watch_eof_posix, daemon=True, name="BrainStdinWatchdog").start()


_SEQLOCK_HDR = struct.Struct("<IIQQQ")  # 32 bytes (0..31); bytes 40..63 reserved for header atoms 0..5
_SEQLOCK_FTR = struct.Struct("<II")       # 8 bytes CRC32 + pad footer


class StatelessSeqLockSlot:
    """64-byte cache-line aligned, double-buffered stateless SeqLock + CRC32 shared memory slot."""

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        base_offset: int,
        payload_dtype: np.dtype,
        initialize: bool = False,
    ):
        self._shm: Optional[shared_memory.SharedMemory] = shm
        self.base = int(base_offset)
        self.dtype = np.dtype(payload_dtype)
        self.payload_size = int(self.dtype.itemsize)
        self.slot_stride = ((self.payload_size + _SEQLOCK_FTR.size + 63) // 64) * 64
        self.total_size = 64 + 2 * self.slot_stride
        self._crc_mismatches = 0
        self._scratch = np.zeros((1,), dtype=self.dtype)

        if initialize and self._shm is not None:
            buf = self._shm.buf
            buf[self.base : self.base + self.total_size] = b"\x00" * self.total_size
            _SEQLOCK_HDR.pack_into(
                buf, self.base, BRAIN3D_MAGIC, BRAIN3D_VERSION, 0, 0, time.perf_counter_ns()
            )

    @property
    def crc_mismatches(self) -> int:
        return int(self._crc_mismatches)

    def detach(self) -> None:
        self._shm = None

    def _slot_offset(self, slot_idx: int) -> int:
        return self.base + 64 + (int(slot_idx) & 1) * self.slot_stride

    def publish(self, populate_fn: Callable[[np.ndarray], None]) -> int:
        if self._shm is None:
            return 0
        buf = self._shm.buf
        magic, ver, prev_seq, prev_slot, _ = _SEQLOCK_HDR.unpack_from(buf, self.base)
        if magic != BRAIN3D_MAGIC:
            magic, ver, prev_seq, prev_slot = BRAIN3D_MAGIC, BRAIN3D_VERSION, 0, 0

        even_base = prev_seq if (prev_seq & 1) == 0 else (prev_seq + 1)
        next_slot = 1 - (int(prev_slot) & 1)
        odd_seq = even_base + 1
        even_commit = even_base + 2

        populate_fn(self._scratch)
        raw_bytes = self._scratch.tobytes()
        crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF

        _SEQLOCK_HDR.pack_into(
            buf, self.base, magic, ver, odd_seq, prev_slot, time.perf_counter_ns()
        )
        soff = self._slot_offset(next_slot)
        buf[soff : soff + self.payload_size] = raw_bytes
        _SEQLOCK_FTR.pack_into(buf, soff + self.payload_size, crc, 0)
        _SEQLOCK_HDR.pack_into(
            buf, self.base, magic, ver, even_commit, next_slot, time.perf_counter_ns()
        )
        return int(even_commit // 2)

    def read_latest(self, max_retries: int = 6) -> Tuple[Optional[np.ndarray], int]:
        if self._shm is None:
            return None, 0
        buf = self._shm.buf
        for _ in range(max_retries):
            magic, _, seq1, slot1, _ = _SEQLOCK_HDR.unpack_from(buf, self.base)
            if magic != BRAIN3D_MAGIC or seq1 == 0 or (seq1 & 1) == 1:
                continue
            soff = self._slot_offset(slot1)
            raw = bytes(buf[soff : soff + self.payload_size])
            saved_crc, _ = _SEQLOCK_FTR.unpack_from(buf, soff + self.payload_size)
            _, _, seq2, slot2, _ = _SEQLOCK_HDR.unpack_from(buf, self.base)
            if seq1 != seq2 or slot1 != slot2 or (seq2 & 1) == 1:
                continue
            if (zlib.crc32(raw) & 0xFFFFFFFF) != saved_crc:
                self._crc_mismatches += 1
                continue
            return np.frombuffer(raw, dtype=self.dtype, count=1).copy(), int(seq2 // 2)
        return None, 0

    def set_header_atom(self, idx: int, val: int) -> None:
        if self._shm is None or not (0 <= idx <= 5):
            return
        struct.pack_into("<I", self._shm.buf, self.base + 40 + idx * 4, int(val) & 0xFFFFFFFF)

    def get_header_atom(self, idx: int) -> int:
        if self._shm is None or not (0 <= idx <= 5):
            return 0
        return int(struct.unpack_from("<I", self._shm.buf, self.base + 40 + idx * 4)[0])


def _encode_learning_ctrl_atom(enable_learning: bool, autosave_ep: bool, guard_tripped: bool = False) -> int:
    """
    Encode runtime learning/autosave flags for `cog_bus` `header_atom(2)`.
    Note: `overfit_guard_tripped` is published on dedicated Single-Writer `header_atom(3)` so
    `TelemetryConsoleProcess` (atom 2 writer) and `ConsciousGWTProcess` (atom 3 writer) NEVER race!
    """
    return 0x80 | (0x01 if enable_learning else 0) | (0x02 if autosave_ep else 0) | (0x04 if guard_tripped else 0)


def _read_runtime_learning_flags(cog_bus: Optional["SharedSingleRegionBus"], cfg: Brain3DConfig) -> Tuple[bool, bool]:
    if cog_bus is None or cog_bus.slot is None:
        return bool(cfg.enable_learning), bool(cfg.autosave_per_episode)
    atom2 = cog_bus.slot.get_header_atom(2)
    if (atom2 & 0x80) == 0:
        return bool(cfg.enable_learning), bool(cfg.autosave_per_episode)
    return bool(atom2 & 0x01), bool(atom2 & 0x02)


def _read_runtime_guard_tripped(cog_bus: Optional["SharedSingleRegionBus"]) -> bool:
    if cog_bus is None or cog_bus.slot is None:
        return False
    return bool(cog_bus.slot.get_header_atom(3) & 0x01) or bool(cog_bus.slot.get_header_atom(2) & 0x04)


def _encode_sleep_status_atom(sleep_active: bool, sleep_stage: int) -> int:
    return (0x01 if sleep_active else 0) | ((int(sleep_stage) & 0xFF) << 8)


def _read_sleep_status_atom(cog_bus: Optional["SharedSingleRegionBus"]) -> Tuple[bool, int]:
    if cog_bus is None or cog_bus.slot is None:
        return False, 0
    atom4 = cog_bus.slot.get_header_atom(4)
    active = bool(atom4 & 0x01)
    return active, (int((atom4 >> 8) & 0xFF) if active else 0)


# --- Upgraded Payload Dtypes for the Shared-Memory Neural Buses (v2.0) ---
WORLD3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("physics_tick", "<u8"),
    ("episode_id", "<u4"),
    ("action_epoch", "<u4"),
    ("stage_id", "<u4"),
    ("can_commit_now", "<u1"),
    ("episode_done", "<u1"),
    ("is_sleeping", "<u1"),
    ("visible_count", "<u1"),
    ("measured_fps", "<f4"),
    ("cumulative_score", "<f4"),
    ("last_step_reward", "<f4"),
    ("hazard_level", "<f4"),
    ("self_pos", "<f4", (3,)),
    ("self_vel", "<f4", (3,)),
    ("self_yaw_pitch", "<f4", (2,)),
    ("proprio", "<f4", (PROPRIO_DIM,)),
    ("visible_entities", "<f4", (MAX_ENTITIES, ENTITY_DIM)),
    ("fovea_patch", "<f4", (FOVEA_H, FOVEA_W, FOVEA_C)),
    ("caregiver_utf8", "S128"),
], align=False)

SPATIAL3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("physics_tick", "<u8"),
    ("episode_id", "<u4"),
    ("action_epoch", "<u4"),
    ("spatial_mode", "<u4"),
    ("believed_count", "<u4"),
    ("loop_closures_total", "<u4"),
    ("saccade_hold_active", "<u4"),
    ("respawn_event_flag", "<u1"),
    ("active_goal_mode", "<u1"),
    ("pad_sp", "<u2"),
    ("believed_self_pos", "<f4", (3,)),
    ("pose_drift_vec", "<f4", (3,)),
    ("active_gaze_ypd", "<f4", (3,)),         # [yaw, pitch, focus_dist]
    ("ego_aim_offset_yp", "<f4", (2,)),       # Parietal gaze/aim offset relative to body heading
    ("lead_aim_xyz", "<f4", (3,)),            # Cerebellar ballistic lead-aim point for active target
    ("believed_hazard", "<f4"),
    ("map_accuracy_iou", "<f4"),
    ("mean_fog", "<f4"),
    ("friendly_turf_ratio", "<f4"),           # Believed friendly surface share in explored map
    ("enemy_turf_ratio", "<f4"),              # Believed enemy surface share in explored map
    ("mec_grid_code", "<f4", (MEC_GRID_DIM,)),
    ("hd_ring_code", "<f4", (HD_RING_DIM,)),
    ("volumetric_fog", "<f4", (FOG_VOXELS,)),
    ("mem_compass", "<f4", (MEM_COMPASS_DIM,)),
    ("believed_entities", "<f4", (MAX_ENTITIES, ENTITY_DIM)),
    ("auth_cand_poses", "<f4", (NUM_CANDIDATES, CONT_POSE_DIM)),
    ("auth_cand_verbs", "<u1", (NUM_CANDIDATES,)),
], align=False)

FIELD_MAP3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("physics_tick", "<u8"),
    ("episode_id", "<u4"),
    ("field_grid_n", "<u4"),
    ("field_occ_u8", "<u1", (FIELD_MAP_MAX_N, FIELD_MAP_MAX_N)),
    ("field_surface_i8", "<i1", (FIELD_MAP_MAX_N, FIELD_MAP_MAX_N)),
    ("field_elev_i8", "<i1", (FIELD_MAP_MAX_N, FIELD_MAP_MAX_N)),
    ("field_visit_f16", "<f2", (FIELD_MAP_MAX_N, FIELD_MAP_MAX_N)),
], align=False)

TIMING3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("physics_tick", "<u8"),
    ("episode_id", "<u4"),
    ("action_epoch", "<u4"),
    ("timing_mode", "<u4"),
    ("commit_gate_open", "<u1"),
    ("hold_reason_id", "<u1"),
    ("coach_brake_active", "<u1"),
    ("target_cand_idx", "<u1"),
    ("holds_triggered", "<u4"),
    ("opportunities_caught", "<u4"),
    ("go_drive", "<f4"),
    ("stn_brake", "<f4"),
    ("urgency_val", "<f4"),
    ("net_accumulator", "<f4"),
    ("release_threshold", "<f4"),
    ("scene_opportunity_prob", "<f4"),
    ("local_kinetic_risk", "<f4"),
    ("believed_scene_speed", "<f4"),
    ("turn_wait_ms", "<f4"),
    ("mean_wait_ms", "<f4"),
    ("reward_gained_while_holding", "<f4"),
    ("stn_gains", "<f4", (4,)),
], align=False)

COGNITIVE3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("cog_basis_tick", "<u8"),
    ("episode_id", "<u4"),
    ("action_epoch", "<u4"),
    ("chosen_cand_idx", "<u4"),
    ("s1_fast_cand_idx", "<u4"),
    ("veto_active", "<u1"),
    ("virtual_alert_active", "<u1"),
    ("sleep_active", "<u1"),
    ("sleep_stage", "<u1"),
    ("topdown_goal_mode", "<u1"),
    ("strategic_lane_code", "<u1"),
    ("strategic_regime_id", "<u1"),
    ("autosave_count_u8", "<u1"),
    ("proto_sym_id", "<i4"),
    ("proto_mod_id", "<i4"),
    ("virtual_alert_count", "<u4"),
    ("sleep_cycle_count", "<u4"),
    ("brain_hz", "<f4"),
    ("inference_latency_ms", "<f4"),
    ("ignition", "<f4"),
    ("self_surprise", "<f4"),
    ("phi_geometric_bits", "<f4"),
    ("policy_entropy_norm", "<f4"),
    ("world_surprise", "<f4"),
    ("fisher_rao_step", "<f4"),
    ("survival_at_chosen", "<f4"),
    ("latest_td_reward", "<f4"),
    ("latest_hazard_delta", "<f4"),
    ("reward_action_epoch", "<u4"),
    ("topdown_focus_xyz", "<f4", (3,)),
    ("gwt_alpha", "<f4", (5,)),
    ("cand_poses", "<f4", (NUM_CANDIDATES, CONT_POSE_DIM)),
    ("cand_verbs", "<u1", (NUM_CANDIDATES,)),
    ("q_values", "<f4", (NUM_CANDIDATES,)),
    ("q_raw_values", "<f4", (NUM_CANDIDATES,)),
    ("survival_probs", "<f4", (NUM_CANDIDATES,)),
    ("pred_adv_probs", "<f4", (NUM_CANDIDATES,)),
    ("policy_probs", "<f4", (NUM_CANDIDATES,)),
    ("mip_q_star", "<f4", (NUM_CANDIDATES,)),
    ("proto_prior_bias", "<f4", (NUM_CANDIDATES,)),
    ("gwt_workspace", "<f4", (GWT_DIM,)),
    ("self_state", "<f4", (SELF_MODEL_DIM,)),
    ("latent_z", "<f4", (LATENT_Z_DIM,)),
    ("z_topdown_delta", "<f4", (LATENT_Z_DIM,)),
    ("rssm_h", "<f4", (RSSM_DETER_DIM,)),
    ("proto_embed", "<f4", (PROTO_EMBED_DIM,)),
    ("slot_ctx64", "<f4", (MULTIMODAL_EMBED_DIM,)),
    ("vis_slot_coords", "<f4", (NUM_OBJECT_SLOTS, 3)),
    ("vis_slot_valence", "<f4", (NUM_OBJECT_SLOTS, 3)),
    ("vis_pred_surprise", "<f4"),
    ("pred_rew", "<f4", (NUM_CANDIDATES,)),
    ("wm_has_prev_pred", "<u1"),
    ("wm_veto_mask_u8", "<u1"),
    ("mip_cut_mask_u8", "<u1"),
    ("wm_pad0", "<u1"),
    ("wm_vis_mae", "<f4"),
    ("wm_depth_mae_m", "<f4"),
    ("wm_cf_spread", "<f4"),
    ("wm_pred_depth_u8", "<u1", (FOVEA_H, FOVEA_W)),
    ("wm_pred_surf_u8", "<u1", (FOVEA_H, FOVEA_W)),
    ("wm_pred_err_u8", "<u1", (FOVEA_H, FOVEA_W)),
], align=False)

LANGUAGE3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("based_on_episode_id", "<u4"),
    ("based_on_action_epoch", "<u4"),
    ("valid_until_epoch", "<u4"),
    ("strategy_mode_id", "<u4"),
    ("llm_status", "<u4"),
    ("endogenous_active", "<u4"),
    ("bias_confidence", "<f4"),
    ("latency_ms", "<f4"),
    ("cand_prior_bias", "<f4", (NUM_CANDIDATES,)),
    ("inner_speech_tokens", "<i4", (INNER_SPEECH_MAX_LEN,)),
    ("inner_speech_embed", "<f4", (INNER_SPEECH_EMBED_DIM,)),
    ("monologue_utf8", "S256"),
    ("digest_s640", f"S{S640_DIGEST_BYTES}"),
], align=False)

MOTOR_INTENT3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("based_on_physics_tick", "<u8"),
    ("based_on_episode_id", "<u4"),
    ("based_on_action_epoch", "<u4"),
    ("command_type", "<u4"),
    ("chosen_cand_idx", "<u4"),
    ("motor_verb", "<u1"),
    ("commit_authorized", "<u1"),
    ("pad0", "<u2"),
    ("valid_horizon_ms", "<f4"),
    ("target_pose", "<f4", (CONT_POSE_DIM,)),
    ("gaze_ypd", "<f4", (3,)),
], align=False)

RETINAL3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("physics_tick", "<u8"),
    ("episode_id", "<u4"),
    ("action_epoch", "<u4"),
    ("visual_mode_id", "<u1"),
    ("teacher_valid", "<u1"),
    ("retina_valid", "<u1"),
    ("pad0", "<u1"),
    ("self_pos", "<f4", (3,)),
    ("self_yaw_pitch", "<f4", (2,)),
    ("gaze_ypd", "<f4", (3,)),
    ("stereo_rgb", "<u1", (2, RETINA_H, RETINA_W, RETINA_C)),
    ("teacher_fovea", "<f4", (FOVEA_H, FOVEA_W, FOVEA_C)),
], align=False)

VISION_CORTEX3D_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("physics_tick", "<u8"),
    ("episode_id", "<u4"),
    ("action_epoch", "<u4"),
    ("visual_mode_id", "<u1"),
    ("depth_ready", "<u1"),
    ("pad0", "<u2"),
    ("depth_infer_ms", "<f4"),
    ("mean_uncertainty", "<f4"),
    ("mean_depth_norm", "<f4"),
    ("calib_val_mae", "<f4"),
    ("calib_val_cos", "<f4"),
    ("calib_steps_done", "<u4"),
    ("pred_fovea_patch", "<f4", (FOVEA_H, FOVEA_W, FOVEA_C)),
    ("epistemic_uncert_16", "<f4", (FOVEA_H, FOVEA_W)),
    ("low_aperture_16", "<f4", (FOVEA_H, FOVEA_W)),
    ("floor_pit_16", "<f4", (FOVEA_H, FOVEA_W)),
    ("ventral_fovea_patch", "<f4", (FOVEA_H, FOVEA_W, VENTRAL_C)),
], align=False)


class SharedSingleRegionBus:
    """Single-region double-buffered SeqLock shared memory bus."""

    def __init__(self, name: str, dtype: np.dtype, create: bool = False):
        self.name = name
        self.dtype = np.dtype(dtype)
        stride = ((self.dtype.itemsize + _SEQLOCK_FTR.size + 63) // 64) * 64
        total_bytes = 64 + 2 * stride
        if create:
            try:
                old = shared_memory.SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except Exception:
                pass
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=total_bytes)
        else:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
        self.slot = StatelessSeqLockSlot(self.shm, 0, self.dtype, initialize=create)

    @property
    def crc_mismatches(self) -> int:
        return self.slot.crc_mismatches

    def publish_dict(self, values: Dict[str, Any]) -> int:
        def _fill(arr: np.ndarray) -> None:
            rec = arr[0]
            for k, v in values.items():
                rec[k] = v
        return self.slot.publish(_fill)

    def read_latest(self) -> Tuple[Optional[np.void], int]:
        arr, seq = self.slot.read_latest()
        return (arr[0], seq) if arr is not None else (None, 0)

    def close(self, unlink: bool = False) -> None:
        self.slot.detach()
        try:
            self.shm.close()
        except Exception:
            pass
        if unlink:
            try:
                self.shm.unlink()
            except Exception:
                pass


class SharedMultiChannelIntent3DBus:
    """4-channel priority-arbitrated lock-free motor + independent gaze/aim command bus."""

    def __init__(self, name: str, create: bool = False):
        self.name = name
        stride = ((MOTOR_INTENT3D_DTYPE.itemsize + _SEQLOCK_FTR.size + 63) // 64) * 64
        self._ch_size = 64 + 2 * stride
        total_bytes = NUM_INTENT_CHANNELS * self._ch_size
        if create:
            try:
                old = shared_memory.SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except Exception:
                pass
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=total_bytes)
        else:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
        self.channels = [
            StatelessSeqLockSlot(self.shm, ch * self._ch_size, MOTOR_INTENT3D_DTYPE, initialize=create)
            for ch in range(NUM_INTENT_CHANNELS)
        ]

    @property
    def crc_mismatches(self) -> int:
        return sum(c.crc_mismatches for c in self.channels)

    def publish_channel(self, ch: int, values: Dict[str, Any]) -> int:
        def _fill(arr: np.ndarray) -> None:
            rec = arr[0]
            for k, v in values.items():
                rec[k] = v
        return self.channels[ch].publish(_fill)

    def read_channel(self, ch: int) -> Tuple[Optional[np.void], int]:
        arr, seq = self.channels[ch].read_latest()
        return (arr[0], seq) if arr is not None else (None, 0)

    def close(self, unlink: bool = False) -> None:
        for c in self.channels:
            c.detach()
        try:
            self.shm.close()
        except Exception:
            pass
        if unlink:
            try:
                self.shm.unlink()
            except Exception:
                pass


# ==============================================================================
# 3. Unity Binary Wire Protocol (`UB3D` / `UBMC` / `UBNT` / `UBNC` / `UBND`)
# ==============================================================================
UB3D_HDR_STRUCT = struct.Struct("<IIQQIIII")
UB3D_FTR_STRUCT = struct.Struct("<II")
UB3D_FLOAT_COUNT = 10 + PROPRIO_DIM + (MAX_ENTITIES * ENTITY_DIM) + (FOVEA_H * FOVEA_W * FOVEA_C)  # 1530
UB3D_PAYLOAD_WITHOUT_FTR = UB3D_HDR_STRUCT.size + UB3D_FLOAT_COUNT * 4 + 128  # 6288
UB3D_TOTAL_BYTES = UB3D_PAYLOAD_WITHOUT_FTR + UB3D_FTR_STRUCT.size            # 6296
UB3D_V2_PAYLOAD_WITHOUT_FTR = UB3D_PAYLOAD_WITHOUT_FTR + RETINA_STEREO_BYTES + CALIB_TEACHER_BYTES  # 34960
UB3D_V2_TOTAL_BYTES = UB3D_V2_PAYLOAD_WITHOUT_FTR + UB3D_FTR_STRUCT.size                            # 34968

UBMC_HDR_STRUCT = struct.Struct("<IIQIIBBBB")
UBMC_FLOATS_STRUCT = struct.Struct("<15f")
UBMC_FTR_STRUCT = struct.Struct("<II")
UBMC_PAYLOAD_WITHOUT_FTR = UBMC_HDR_STRUCT.size + UBMC_FLOATS_STRUCT.size + 32  # 120
UBMC_TOTAL_BYTES = UBMC_PAYLOAD_WITHOUT_FTR + UBMC_FTR_STRUCT.size              # 128

UNITY_UX_MAGIC = 0x544E4255        # "UBNT"
UNITY_UX_CTRL_MAGIC = 0x434E4255   # "UBNC"
UNITY_UX_DEPTH_MAGIC = 0x444E4255  # "UBND"
UNITY_UX_WM_MAGIC = 0x4D574255     # "UBWM" (Live Predictive World-Model HUD Telemetry)
UBNT_HDR_STRUCT = struct.Struct("<IIQQIIQ16B8B4I")  # 80 bytes
UBNT_SCALAR_COUNT = 24
UBNT_FLOAT_COUNT = (
    UBNT_SCALAR_COUNT + 5 + GWT_DIM + RSSM_DETER_DIM + LATENT_Z_DIM * 2
    + MEC_GRID_DIM + HD_RING_DIM + SELF_MODEL_DIM + NUM_CANDIDATES * 5
    + NUM_CANDIDATES * CONT_POSE_DIM + 4 + 3 + 3
)  # 871 floats
UBNT_FLOATS_STRUCT = struct.Struct(f"<{UBNT_FLOAT_COUNT}f")
UBNT_SPEECH_BYTES = 64
UBNT_FTR_STRUCT = struct.Struct("<II")
UBNT_PAYLOAD_WITHOUT_FTR = UBNT_HDR_STRUCT.size + UBNT_FLOATS_STRUCT.size + UBNT_SPEECH_BYTES  # 3628
UBNT_TOTAL_BYTES = UBNT_PAYLOAD_WITHOUT_FTR + UBNT_FTR_STRUCT.size  # 3636

UBNC_STRUCT = struct.Struct("<IIBBBBI")  # 16 bytes
UBNC_TOTAL_BYTES = UBNC_STRUCT.size

UBND_HDR_STRUCT = struct.Struct("<IIIIBBH")
UBND_FTR_STRUCT = struct.Struct("<I")
UBND_GRID_BYTES = FOVEA_H * FOVEA_W
UBND_PAYLOAD_WITHOUT_FTR = UBND_HDR_STRUCT.size + UBND_GRID_BYTES * 2  # 532
UBND_TOTAL_BYTES = UBND_PAYLOAD_WITHOUT_FTR + UBND_FTR_STRUCT.size    # 536

UBWM_HDR_STRUCT = struct.Struct("<IIIIBBBB")  # 20 bytes
UBWM_FLOATS_COUNT = 4 + NUM_CANDIDATES * 3    # 28 floats (112 bytes)
UBWM_FLOATS_STRUCT = struct.Struct(f"<{UBWM_FLOATS_COUNT}f")
UBWM_FTR_STRUCT = struct.Struct("<I")         # 4 bytes CRC32
UBWM_GRID_BYTES = FOVEA_H * FOVEA_W           # 256 bytes
UBWM_PAYLOAD_WITHOUT_FTR = UBWM_HDR_STRUCT.size + UBWM_FLOATS_STRUCT.size + UBWM_GRID_BYTES * 3  # 900
UBWM_TOTAL_BYTES = UBWM_PAYLOAD_WITHOUT_FTR + UBWM_FTR_STRUCT.size                               # 904


def pack_ub3d_sensory_frame(
    *,
    frame_seq: int,
    timestamp_ns: int,
    episode_id: int,
    action_epoch: int,
    flags: int,
    self_pos: np.ndarray,
    self_vel: np.ndarray,
    self_yaw_pitch: np.ndarray,
    step_reward: float,
    hazard_level: float,
    proprio: np.ndarray,
    visible_entities: np.ndarray,
    visible_count: int,
    fovea_patch: np.ndarray,
    caregiver_text: str = "",
    proto_version: int = UNITY_PROTO_VERSION,
    stereo_rgb: Optional[np.ndarray] = None,
    teacher_fovea: Optional[np.ndarray] = None,
    visual_mode_id: int = 0,
    teacher_valid: bool = False,
    retina_valid: bool = False,
    stage_id: int = 0,
) -> bytes:
    eff_ver = UNITY_PROTO_VERSION_V2 if (int(proto_version) == UNITY_PROTO_VERSION_V2 or stereo_rgb is not None) else UNITY_PROTO_VERSION
    eff_flags = int(flags)
    if int(stage_id) > 0:
        eff_flags = (eff_flags & 0x0000FFFF) | ((int(stage_id) & 0xFFFF) << 16)
    if eff_ver == UNITY_PROTO_VERSION_V2:
        eff_flags = (
            (eff_flags & 0xFFFFF0FF)
            | ((int(visual_mode_id) & 0x03) << 8)
            | ((1 if teacher_valid else 0) << 10)
            | ((1 if (retina_valid or stereo_rgb is not None) else 0) << 11)
        )
    hdr = UB3D_HDR_STRUCT.pack(
        UNITY_SENSORY_MAGIC, eff_ver, int(frame_seq), int(timestamp_ns),
        int(episode_id), int(action_epoch), eff_flags, int(min(MAX_ENTITIES, max(0, visible_count))),
    )
    floats_buf = np.empty(UB3D_FLOAT_COUNT, dtype="<f4")
    floats_buf[0:3] = np.asarray(self_pos, dtype=np.float32)[:3]
    floats_buf[3:6] = np.asarray(self_vel, dtype=np.float32)[:3]
    floats_buf[6:8] = np.asarray(self_yaw_pitch, dtype=np.float32)[:2]
    floats_buf[8] = float(step_reward)
    floats_buf[9] = float(hazard_level)
    floats_buf[10 : 10 + PROPRIO_DIM] = np.asarray(proprio, dtype=np.float32)[:PROPRIO_DIM]
    ent_flat = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
    vc = min(MAX_ENTITIES, len(visible_entities))
    if vc > 0:
        ent_flat[:vc] = np.asarray(visible_entities[:vc], dtype=np.float32)
    idx_e = 10 + PROPRIO_DIM
    floats_buf[idx_e : idx_e + MAX_ENTITIES * ENTITY_DIM] = ent_flat.reshape(-1)
    idx_f = idx_e + MAX_ENTITIES * ENTITY_DIM
    floats_buf[idx_f:] = np.asarray(fovea_patch, dtype=np.float32).reshape(-1)[: FOVEA_H * FOVEA_W * FOVEA_C]

    cg_raw = _clamp_utf8_bytes(caregiver_text, 127)
    cg_padded = cg_raw + b"\x00" * (128 - len(cg_raw))
    body = hdr + floats_buf.tobytes() + cg_padded
    if eff_ver == UNITY_PROTO_VERSION_V2:
        st_bytes = (
            np.ascontiguousarray(np.asarray(stereo_rgb, dtype=np.uint8).reshape(2, RETINA_H, RETINA_W, RETINA_C)).tobytes()
            if stereo_rgb is not None
            else (b"\x00" * RETINA_STEREO_BYTES)
        )
        tf_bytes = (
            np.ascontiguousarray(np.asarray(teacher_fovea, dtype="<f4").reshape(FOVEA_H, FOVEA_W, FOVEA_C)).tobytes()
            if teacher_fovea is not None
            else (b"\x00" * CALIB_TEACHER_BYTES)
        )
        body = body + st_bytes + tf_bytes
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + UB3D_FTR_STRUCT.pack(crc, 0)


def unpack_ub3d_sensory_frame(packet: bytes) -> Optional[Dict[str, Any]]:
    plen = len(packet)
    if plen == UB3D_TOTAL_BYTES:
        pay_len = UB3D_PAYLOAD_WITHOUT_FTR
        expected_ver = UNITY_PROTO_VERSION
    elif plen == UB3D_V2_TOTAL_BYTES:
        pay_len = UB3D_V2_PAYLOAD_WITHOUT_FTR
        expected_ver = UNITY_PROTO_VERSION_V2
    else:
        return None
    body = packet[:pay_len]
    saved_crc, _ = UB3D_FTR_STRUCT.unpack_from(packet, pay_len)
    if (zlib.crc32(body) & 0xFFFFFFFF) != saved_crc:
        return None
    magic, ver, fseq, ts_ns, ep_id, act_ep, flags, vis_cnt = UB3D_HDR_STRUCT.unpack_from(body, 0)
    if magic != UNITY_SENSORY_MAGIC or ver != expected_ver:
        return None
    floats = np.frombuffer(body, dtype="<f4", count=UB3D_FLOAT_COUNT, offset=UB3D_HDR_STRUCT.size)
    idx_e = 10 + PROPRIO_DIM
    idx_f = idx_e + MAX_ENTITIES * ENTITY_DIM
    cg_bytes = body[UB3D_PAYLOAD_WITHOUT_FTR - 128 : UB3D_PAYLOAD_WITHOUT_FTR].split(b"\x00", 1)[0]
    out: Dict[str, Any] = {
        "proto_version": int(ver),
        "frame_seq": int(fseq),
        "timestamp_ns": int(ts_ns),
        "episode_id": int(ep_id),
        "action_epoch": int(act_ep),
        "flags": int(flags),
        "stage_id": int((flags >> 16) & 0xFFFF),
        "respawn_flag": bool((flags >> 4) & 1),
        "visual_mode_id": int((flags >> 8) & 0x03),
        "teacher_valid": bool((flags >> 10) & 1),
        "retina_valid": bool((flags >> 11) & 1),
        "visible_count": int(min(MAX_ENTITIES, vis_cnt)),
        "self_pos": floats[0:3].copy(),
        "self_vel": floats[3:6].copy(),
        "self_yaw_pitch": floats[6:8].copy(),
        "step_reward": float(floats[8]),
        "hazard_level": float(floats[9]),
        "proprio": floats[10:idx_e].copy(),
        "visible_entities": floats[idx_e:idx_f].reshape(MAX_ENTITIES, ENTITY_DIM).copy(),
        "fovea_patch": floats[idx_f:].reshape(FOVEA_H, FOVEA_W, FOVEA_C).copy(),
        "caregiver_utf8": cg_bytes,
    }
    if ver == UNITY_PROTO_VERSION_V2:
        st_off = UB3D_PAYLOAD_WITHOUT_FTR
        tf_off = st_off + RETINA_STEREO_BYTES
        out["stereo_rgb"] = np.frombuffer(
            body, dtype=np.uint8, count=RETINA_STEREO_BYTES, offset=st_off
        ).reshape(2, RETINA_H, RETINA_W, RETINA_C).copy()
        out["teacher_fovea"] = np.frombuffer(
            body, dtype="<f4", count=FOVEA_H * FOVEA_W * FOVEA_C, offset=tf_off
        ).reshape(FOVEA_H, FOVEA_W, FOVEA_C).copy()
    return out


def pack_ubmc_motor_command(
    *,
    cmd_seq: int,
    episode_id: int,
    action_epoch: int,
    motor_verb: int,
    commit_gate_open: bool,
    veto_active: bool,
    hold_reason_id: int,
    target_pose: np.ndarray,
    gaze_ypd: np.ndarray,
    telemetry6: np.ndarray,
    inner_speech_tag: str = "",
    time_scale: float = 1.0,
) -> bytes:
    hdr = UBMC_HDR_STRUCT.pack(
        UNITY_MOTOR_MAGIC, UNITY_PROTO_VERSION, int(cmd_seq), int(episode_id), int(action_epoch),
        int(motor_verb) & 0xFF, 1 if commit_gate_open else 0, 1 if veto_active else 0, int(hold_reason_id) & 0xFF,
    )
    tp = np.asarray(target_pose, dtype=np.float32).reshape(-1)[:6]
    gz = np.asarray(gaze_ypd, dtype=np.float32).reshape(-1)[:3]
    tm = np.asarray(telemetry6, dtype=np.float32).reshape(-1)[:6]
    fl_bytes = UBMC_FLOATS_STRUCT.pack(*tp.tolist(), *gz.tolist(), *tm.tolist())
    sp_raw = _clamp_utf8_bytes(inner_speech_tag, 31)
    sp_padded = sp_raw + b"\x00" * (32 - len(sp_raw))
    body = hdr + fl_bytes + sp_padded
    crc = zlib.crc32(body) & 0xFFFFFFFF
    ts_milli = 0 if abs(float(time_scale) - 1.0) < 1e-3 else int(np.clip(round(float(time_scale) * 1000.0), 250, 12000))
    return body + UBMC_FTR_STRUCT.pack(crc, ts_milli)


def unpack_ubmc_motor_command(packet: bytes) -> Optional[Dict[str, Any]]:
    if len(packet) != UBMC_TOTAL_BYTES:
        return None
    body = packet[:UBMC_PAYLOAD_WITHOUT_FTR]
    saved_crc, ts_milli = UBMC_FTR_STRUCT.unpack_from(packet, UBMC_PAYLOAD_WITHOUT_FTR)
    if (zlib.crc32(body) & 0xFFFFFFFF) != saved_crc:
        return None
    magic, ver, cseq, ep_id, act_ep, verb, commit_b, veto_b, h_reason = UBMC_HDR_STRUCT.unpack_from(body, 0)
    if magic != UNITY_MOTOR_MAGIC or ver != UNITY_PROTO_VERSION:
        return None
    fl = UBMC_FLOATS_STRUCT.unpack_from(body, UBMC_HDR_STRUCT.size)
    sp_bytes = body[UBMC_PAYLOAD_WITHOUT_FTR - 32 : UBMC_PAYLOAD_WITHOUT_FTR].split(b"\x00", 1)[0]
    return {
        "cmd_seq": int(cseq),
        "based_on_episode_id": int(ep_id),
        "based_on_action_epoch": int(act_ep),
        "motor_verb": int(verb),
        "commit_gate_open": bool(commit_b),
        "veto_active": bool(veto_b),
        "hold_reason_id": int(h_reason),
        "target_pose": np.array(fl[0:6], dtype=np.float32),
        "gaze_ypd": np.array(fl[6:9], dtype=np.float32),
        "telemetry": np.array(fl[9:15], dtype=np.float32),
        "inner_speech_tag": sp_bytes.decode("utf-8", errors="ignore"),
        "time_scale": (float(ts_milli) / 1000.0) if 250 <= ts_milli <= 12000 else 1.0,
    }


def pack_ubnc_ui_control(cmd_type: int, arg0: int = 0, arg1: int = 0) -> bytes:
    partial = struct.pack("<IIBBBB", UNITY_UX_CTRL_MAGIC, UNITY_PROTO_VERSION, cmd_type & 0xFF, arg0 & 0xFF, arg1 & 0xFF, 0)
    crc = zlib.crc32(partial) & 0xFFFFFFFF
    return UBNC_STRUCT.pack(UNITY_UX_CTRL_MAGIC, UNITY_PROTO_VERSION, cmd_type & 0xFF, arg0 & 0xFF, arg1 & 0xFF, 0, crc)


def unpack_ubnc_ui_control(packet: bytes) -> Optional[Dict[str, int]]:
    if len(packet) != UBNC_TOTAL_BYTES:
        return None
    magic, ver, ctype, a0, a1, _, saved_crc = UBNC_STRUCT.unpack(packet)
    if magic != UNITY_UX_CTRL_MAGIC or ver != UNITY_PROTO_VERSION:
        return None
    if (zlib.crc32(packet[:12]) & 0xFFFFFFFF) != saved_crc:
        return None
    return {"cmd_type": int(ctype), "arg0": int(a0), "arg1": int(a1)}


def _sendall_nonblocking_safe(sock: socket.socket, payload: bytes, timeout_s: float = 0.08) -> Tuple[bool, bool]:
    """
    Safely send a complete packet over a blocking or non-blocking TCP socket.
    Returns `(sent_all, fatal_disconnect)`:
      - If a non-blocking socket's OS send buffer is momentarily full (`BlockingIOError` before any byte
        of the frame is sent), drops ONLY this telemetry frame (`(False, False)`) instead of severing TCP!
      - Once partial bytes of a packet have entered the stream, waits up to `timeout_s` to finish framing
        atomically; only marks `fatal_disconnect = True` if stream alignment cannot be preserved or peer closed.
    """
    mv = memoryview(payload)
    total = len(mv)
    sent = 0
    deadline = time.perf_counter() + max(0.005, float(timeout_s))
    while sent < total:
        try:
            n = sock.send(mv[sent:])
            if n <= 0:
                return False, True
            sent += n
        except BlockingIOError:
            if sent == 0:
                return False, False
            if time.perf_counter() >= deadline:
                return False, True
            time.sleep(0.001)
        except InterruptedError:
            continue
        except OSError:
            return False, True
    return True, False


def _recv_exact(sock: socket.socket, nbytes: int) -> Optional[bytes]:
    chunks: List[bytes] = []
    rem = int(nbytes)
    while rem > 0:
        try:
            chunk = sock.recv(rem)
        except (BlockingIOError, InterruptedError):
            continue
        except socket.timeout:
            return b"" if not chunks else None
        except OSError:
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        rem -= len(chunk)
    return b"".join(chunks)


def _recv_ub3d_packet(sock: socket.socket) -> Optional[bytes]:
    hdr = _recv_exact(sock, UB3D_HDR_STRUCT.size)
    if hdr is None:
        return None
    if len(hdr) == 0:
        return b""
    magic, ver, *_ = UB3D_HDR_STRUCT.unpack(hdr)
    if magic != UNITY_SENSORY_MAGIC:
        return None
    total_len = UB3D_V2_TOTAL_BYTES if ver == UNITY_PROTO_VERSION_V2 else UB3D_TOTAL_BYTES
    rest = _recv_exact(sock, total_len - UB3D_HDR_STRUCT.size)
    if rest is None or len(rest) != (total_len - UB3D_HDR_STRUCT.size):
        return None
    return hdr + rest



def _ignore_sigint_in_worker() -> None:
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def pack_ubnt_neural_telemetry(
    *,
    seq: int,
    timestamp_ns: int,
    episode_id: int,
    action_epoch: int,
    physics_tick: int,
    chosen_cand_idx: int,
    s1_fast_cand_idx: int,
    commit_gate_open: bool,
    veto_active: bool,
    timing_mode: int,
    hold_reason_id: int,
    coach_brake_active: bool,
    sleep_active: bool,
    sleep_stage: int,
    virtual_alert_active: bool,
    endogenous_active: bool,
    proto_sym_id: int,
    proto_mod_id: int,
    believed_count: int,
    cand_verbs: np.ndarray,
    holds_triggered: int,
    opportunities_caught: int,
    loop_closures_total: int,
    virtual_alert_count: int,
    scalars24: Sequence[float],
    gwt_alpha: np.ndarray,
    gwt_workspace: np.ndarray,
    rssm_h: np.ndarray,
    latent_z: np.ndarray,
    z_topdown_delta: np.ndarray,
    mec_grid_code: np.ndarray,
    hd_ring_code: np.ndarray,
    self_state: np.ndarray,
    q_values: np.ndarray,
    q_raw_values: np.ndarray,
    survival_probs: np.ndarray,
    policy_probs: np.ndarray,
    mip_q_star: np.ndarray,
    cand_poses: np.ndarray,
    stn_gains: np.ndarray,
    believed_self_pos: np.ndarray,
    pose_drift_vec: np.ndarray,
    inner_speech_text: str = "",
    visual_mode_id: int = 0,
    depth_ready: bool = False,
    depth_mean_uncert: float = 0.0,
    depth_infer_ms: float = 0.0,
    depth_mean_norm: float = 0.0,
    learning_enabled: bool = False,
    autosave_enabled: bool = False,
    autosave_count: int = 0,
    overfit_guard_tripped: bool = False,
    mip_cut_mask: int = 3,
) -> bytes:
    """Serialize live internal neural/mathematical brain state into the `3,636`-byte `UBNT` TCP wire packet."""
    cv = np.zeros(NUM_CANDIDATES, dtype=np.uint8)
    cv_in = np.asarray(cand_verbs, dtype=np.int64).ravel()
    cv[: min(NUM_CANDIDATES, cv_in.size)] = np.clip(cv_in[: min(NUM_CANDIDATES, cv_in.size)], 0, 255).astype(np.uint8)

    byte50 = (1 if endogenous_active else 0) | ((min(127, max(0, int(autosave_count))) & 0x7F) << 1)
    byte52 = (int(max(0, proto_mod_id)) & 0x07) | ((int(mip_cut_mask) & 0x1F) << 3)
    byte54 = (
        (int(visual_mode_id) & 0x0F)
        | (0x10 if learning_enabled else 0)
        | (0x20 if (learning_enabled and autosave_enabled) else 0)
        | (0x40 if overfit_guard_tripped else 0)
        | (0x80 if depth_ready else 0)
    )
    byte55 = int(np.clip(round(float(depth_mean_uncert) * 255.0), 0, 255))

    hdr = UBNT_HDR_STRUCT.pack(
        UNITY_UX_MAGIC,
        UNITY_PROTO_VERSION,
        int(seq),
        int(timestamp_ns),
        int(episode_id),
        int(action_epoch),
        int(physics_tick),
        int(chosen_cand_idx) & 0xFF,
        int(s1_fast_cand_idx) & 0xFF,
        1 if commit_gate_open else 0,
        1 if veto_active else 0,
        int(timing_mode) & 0xFF,
        int(hold_reason_id) & 0xFF,
        1 if coach_brake_active else 0,
        1 if sleep_active else 0,
        int(sleep_stage) & 0xFF,
        1 if virtual_alert_active else 0,
        byte50,
        int(max(0, proto_sym_id)) & 0xFF,
        byte52,
        int(max(0, believed_count)) & 0xFF,
        byte54,
        byte55,
        *[int(x) for x in cv],
        int(holds_triggered),
        int(opportunities_caught),
        int(loop_closures_total),
        int(virtual_alert_count),
    )

    fvec = np.zeros(UBNT_FLOAT_COUNT, dtype=np.float32)
    sc = np.asarray(scalars24, dtype=np.float32).ravel()
    fvec[: min(UBNT_SCALAR_COUNT, sc.size)] = sc[: min(UBNT_SCALAR_COUNT, sc.size)]
    pos = UBNT_SCALAR_COUNT
    for arr, dim in (
        (gwt_alpha, 5),
        (gwt_workspace, GWT_DIM),
        (rssm_h, RSSM_DETER_DIM),
        (latent_z, LATENT_Z_DIM),
        (z_topdown_delta, LATENT_Z_DIM),
        (mec_grid_code, MEC_GRID_DIM),
        (hd_ring_code, HD_RING_DIM),
        (self_state, SELF_MODEL_DIM),
        (q_values, NUM_CANDIDATES),
        (q_raw_values, NUM_CANDIDATES),
        (survival_probs, NUM_CANDIDATES),
        (policy_probs, NUM_CANDIDATES),
        (mip_q_star, NUM_CANDIDATES),
        (cand_poses, NUM_CANDIDATES * CONT_POSE_DIM),
        (stn_gains, 4),
        (believed_self_pos, 3),
        (pose_drift_vec, 3),
    ):
        a = np.asarray(arr, dtype=np.float32).ravel()
        n = min(dim, a.size)
        if n > 0:
            fvec[pos : pos + n] = a[:n]
        pos += dim

    sp_raw = _clamp_utf8_bytes(inner_speech_text, UBNT_SPEECH_BYTES - 1)
    sp_padded = sp_raw + b"\x00" * (UBNT_SPEECH_BYTES - len(sp_raw))
    body = hdr + UBNT_FLOATS_STRUCT.pack(*[float(x) for x in fvec]) + sp_padded
    crc = zlib.crc32(body) & 0xFFFFFFFF
    infer_q16 = int(np.clip(round(float(depth_infer_ms) * 100.0), 0, 65535))
    dnorm_q16 = int(np.clip(round(float(depth_mean_norm) * 10000.0), 0, 10000))
    ftr_u32 = (dnorm_q16 << 16) | infer_q16
    return body + UBNT_FTR_STRUCT.pack(crc, ftr_u32)


def unpack_ubnt_neural_telemetry(packet: bytes) -> Optional[Dict[str, Any]]:
    """Verify CRC32 and deserialize a `3,636`-byte `UBNT` neural telemetry packet."""
    if len(packet) != UBNT_TOTAL_BYTES:
        return None
    body = packet[:UBNT_PAYLOAD_WITHOUT_FTR]
    saved_crc, ftr_u32 = UBNT_FTR_STRUCT.unpack_from(packet, UBNT_PAYLOAD_WITHOUT_FTR)
    if (zlib.crc32(body) & 0xFFFFFFFF) != saved_crc:
        return None
    hdr = UBNT_HDR_STRUCT.unpack_from(body, 0)
    if hdr[0] != UNITY_UX_MAGIC or hdr[1] != UNITY_PROTO_VERSION:
        return None
    fl = np.asarray(UBNT_FLOATS_STRUCT.unpack_from(body, UBNT_HDR_STRUCT.size), dtype=np.float32)
    sp_bytes = body[UBNT_PAYLOAD_WITHOUT_FTR - UBNT_SPEECH_BYTES : UBNT_PAYLOAD_WITHOUT_FTR].split(b"\x00", 1)[0]
    byte54 = int(hdr[21])
    byte55 = int(hdr[22])
    return {
        "seq": int(hdr[2]),
        "timestamp_ns": int(hdr[3]),
        "episode_id": int(hdr[4]),
        "action_epoch": int(hdr[5]),
        "physics_tick": int(hdr[6]),
        "chosen_cand_idx": int(hdr[7]),
        "s1_fast_cand_idx": int(hdr[8]),
        "commit_gate_open": bool(hdr[9]),
        "veto_active": bool(hdr[10]),
        "timing_mode": int(hdr[11]),
        "hold_reason_id": int(hdr[12]),
        "coach_brake_active": bool(hdr[13]),
        "sleep_active": bool(hdr[14]),
        "sleep_stage": int(hdr[15]),
        "endogenous_active": bool((int(hdr[17]) & 0x01) != 0),
        "autosave_count": int((int(hdr[17]) >> 1) & 0x7F),
        "proto_sym_id": int(hdr[18]),
        "proto_mod_id": int(hdr[19]) & 0x07,
        "mip_cut_mask": int((int(hdr[19]) >> 3) & 0x1F) or 3,
        "visual_mode_id": int(byte54 & 0x0F),
        "learning_enabled": bool((byte54 & 0x10) != 0),
        "autosave_enabled": bool((byte54 & 0x20) != 0),
        "overfit_guard_tripped": bool((byte54 & 0x40) != 0),
        "depth_ready": bool((byte54 & 0x80) != 0),
        "depth_mean_uncert": float(byte55) / 255.0,
        "depth_infer_ms": float(ftr_u32 & 0xFFFF) / 100.0,
        "depth_mean_norm": float((ftr_u32 >> 16) & 0xFFFF) / 10000.0,
        "holds_triggered": int(hdr[31]),
        "opportunities_caught": int(hdr[32]),
        "loop_closures_total": int(hdr[33]),
        "virtual_alert_count": int(hdr[34]),
        "scalars": fl[:UBNT_SCALAR_COUNT],
        "floats": fl,
        "inner_speech_text": sp_bytes.decode("utf-8", errors="ignore"),
    }


def pack_ubnd_neural_depth(
    *,
    episode_id: int,
    physics_tick: int,
    visual_mode_id: int,
    depth_source_id: int,
    depth_infer_ms: float,
    depth_norm_16x16: np.ndarray,
    uncert_16x16: np.ndarray,
) -> bytes:
    """Serialize a `536`-byte `UBND` 16x16 depth + uncertainty companion telemetry packet."""
    d_q = np.clip(
        np.rint(np.asarray(depth_norm_16x16, dtype=np.float32).reshape(UBND_GRID_BYTES) * 255.0),
        0.0,
        255.0,
    ).astype(np.uint8)
    u_q = np.clip(
        np.rint(np.asarray(uncert_16x16, dtype=np.float32).reshape(UBND_GRID_BYTES) * 255.0),
        0.0,
        255.0,
    ).astype(np.uint8)
    infer_q16 = int(np.clip(round(float(depth_infer_ms) * 100.0), 0, 65535))
    hdr = UBND_HDR_STRUCT.pack(
        UNITY_UX_DEPTH_MAGIC,
        UNITY_PROTO_VERSION,
        int(episode_id) & 0xFFFFFFFF,
        int(physics_tick) & 0xFFFFFFFF,
        int(visual_mode_id) & 0xFF,
        int(depth_source_id) & 0xFF,
        infer_q16,
    )
    body = hdr + d_q.tobytes() + u_q.tobytes()
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + UBND_FTR_STRUCT.pack(crc)


def unpack_ubnd_neural_depth(packet: bytes) -> Optional[Dict[str, Any]]:
    """Verify CRC32 and deserialize a `536`-byte `UBND` 16x16 depth packet."""
    if len(packet) != UBND_TOTAL_BYTES:
        return None
    body = packet[:UBND_PAYLOAD_WITHOUT_FTR]
    saved_crc = UBND_FTR_STRUCT.unpack_from(packet, UBND_PAYLOAD_WITHOUT_FTR)[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != saved_crc:
        return None
    magic, ver, ep_id, tick_lo, vis_mode_id, depth_src, infer_q16 = UBND_HDR_STRUCT.unpack_from(body, 0)
    if magic != UNITY_UX_DEPTH_MAGIC or ver != UNITY_PROTO_VERSION:
        return None
    d_u8 = np.frombuffer(body, dtype=np.uint8, count=UBND_GRID_BYTES, offset=UBND_HDR_STRUCT.size)
    u_u8 = np.frombuffer(body, dtype=np.uint8, count=UBND_GRID_BYTES, offset=UBND_HDR_STRUCT.size + UBND_GRID_BYTES)
    return {
        "episode_id": int(ep_id),
        "physics_tick": int(tick_lo),
        "visual_mode_id": int(vis_mode_id),
        "depth_source_id": int(depth_src),
        "depth_infer_ms": float(infer_q16) / 100.0,
        "depth_norm_16": (d_u8.astype(np.float32) / 255.0).reshape(FOVEA_H, FOVEA_W),
        "uncert_16": (u_u8.astype(np.float32) / 255.0).reshape(FOVEA_H, FOVEA_W),
    }


def pack_ubwm_world_model(
    *,
    episode_id: int,
    physics_tick: int,
    chosen_cand_idx: int,
    has_prev_pred: bool,
    veto_mask_u8: int,
    vis_pred_mae: float,
    depth_pred_mae_m: float,
    world_kl_surprise: float,
    action_cf_spread: float,
    pred_rew: np.ndarray,
    pred_surv: np.ndarray,
    pred_adv: np.ndarray,
    pred_depth_u8: np.ndarray,
    pred_surf_u8: np.ndarray,
    pred_err_u8: np.ndarray,
) -> bytes:
    """Serialize a `904`-byte `UBWM` World-Model imagined next-frame + 3-step rollout packet."""
    hdr = UBWM_HDR_STRUCT.pack(
        UNITY_UX_WM_MAGIC,
        UNITY_PROTO_VERSION,
        int(episode_id) & 0xFFFFFFFF,
        int(physics_tick) & 0xFFFFFFFF,
        int(chosen_cand_idx) & 0x07,
        1 if has_prev_pred else 0,
        int(veto_mask_u8) & 0xFF,
        0,
    )
    fvec = np.zeros(UBWM_FLOATS_COUNT, dtype=np.float32)
    fvec[0] = float(vis_pred_mae)
    fvec[1] = float(depth_pred_mae_m)
    fvec[2] = float(world_kl_surprise)
    fvec[3] = float(action_cf_spread)
    for off, arr in ((4, pred_rew), (12, pred_surv), (20, pred_adv)):
        a = np.asarray(arr, dtype=np.float32).ravel()
        n = min(NUM_CANDIDATES, a.size)
        if n > 0:
            fvec[off : off + n] = a[:n]

    d_bytes = np.ascontiguousarray(np.asarray(pred_depth_u8, dtype=np.uint8).reshape(UBWM_GRID_BYTES)).tobytes()
    s_bytes = np.ascontiguousarray(np.asarray(pred_surf_u8, dtype=np.uint8).reshape(UBWM_GRID_BYTES)).tobytes()
    e_bytes = np.ascontiguousarray(np.asarray(pred_err_u8, dtype=np.uint8).reshape(UBWM_GRID_BYTES)).tobytes()
    body = hdr + UBWM_FLOATS_STRUCT.pack(*[float(x) for x in fvec]) + d_bytes + s_bytes + e_bytes
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + UBWM_FTR_STRUCT.pack(crc)


def unpack_ubwm_world_model(packet: bytes) -> Optional[Dict[str, Any]]:
    """Verify CRC32 and deserialize a `904`-byte `UBWM` World-Model telemetry packet."""
    if len(packet) != UBWM_TOTAL_BYTES:
        return None
    body = packet[:UBWM_PAYLOAD_WITHOUT_FTR]
    saved_crc = UBWM_FTR_STRUCT.unpack_from(packet, UBWM_PAYLOAD_WITHOUT_FTR)[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != saved_crc:
        return None
    magic, ver, ep_id, tick_lo, chosen_idx, has_prev, veto_mask, _ = UBWM_HDR_STRUCT.unpack_from(body, 0)
    if magic != UNITY_UX_WM_MAGIC or ver != UNITY_PROTO_VERSION:
        return None
    fl = np.asarray(UBWM_FLOATS_STRUCT.unpack_from(body, UBWM_HDR_STRUCT.size), dtype=np.float32)
    g0 = UBWM_HDR_STRUCT.size + UBWM_FLOATS_STRUCT.size
    d_u8 = np.frombuffer(body, dtype=np.uint8, count=UBWM_GRID_BYTES, offset=g0).reshape(FOVEA_H, FOVEA_W).copy()
    s_u8 = np.frombuffer(body, dtype=np.uint8, count=UBWM_GRID_BYTES, offset=g0 + UBWM_GRID_BYTES).reshape(FOVEA_H, FOVEA_W).copy()
    e_u8 = np.frombuffer(body, dtype=np.uint8, count=UBWM_GRID_BYTES, offset=g0 + UBWM_GRID_BYTES * 2).reshape(FOVEA_H, FOVEA_W).copy()
    return {
        "episode_id": int(ep_id),
        "physics_tick": int(tick_lo),
        "chosen_cand_idx": int(chosen_idx),
        "has_prev_pred": bool(has_prev != 0),
        "veto_mask_u8": int(veto_mask),
        "vis_pred_mae": float(fl[0]),
        "depth_pred_mae_m": float(fl[1]),
        "world_kl_surprise": float(fl[2]),
        "action_cf_spread": float(fl[3]),
        "pred_rew": fl[4:12].copy(),
        "pred_surv": fl[12:20].copy(),
        "pred_adv": fl[20:28].copy(),
        "pred_depth_16": d_u8.astype(np.float32) / 255.0,
        "pred_surf_16": s_u8.astype(np.float32) / 255.0,
        "pred_err_16": e_u8.astype(np.float32) / 255.0,
        "pred_depth_u8": d_u8,
        "pred_surf_u8": s_u8,
        "pred_err_u8": e_u8,
    }


# ==============================================================================
# 4. Dorsal/Ventral Retinotopic Helpers & Standalone 3D Tactical Arena
# ==============================================================================
def compute_depth_geometry_cues_3d(
    fovea_patch: np.ndarray,
    max_range_m: float = 16.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive game-agnostic 3D geometric navigation cues from a `(16, 16, 4)` dorsal patch:
      - `low_aperture_16`: upper overhang + lower traversable aperture clearance.
      - `floor_pit_16`   : below-horizon positive distance drop / ledge discontinuity.
    """
    p = np.asarray(fovea_patch, dtype=np.float32).reshape(FOVEA_H, FOVEA_W, FOVEA_C)
    norm_d = np.clip(p[:, :, 0], 0.0, 1.0)
    ny = p[:, :, 2]
    dist_m = (1.0 - norm_d) * float(max_range_m)

    low_aperture = np.zeros((FOVEA_H, FOVEA_W), dtype=np.float32)
    upper_dist = np.mean(dist_m[2:6, :], axis=0)
    lower_dist = np.mean(dist_m[9:13, :], axis=0)
    upper_ny = np.mean(np.abs(ny[2:6, :]), axis=0)
    upper_blocked = (upper_dist < 4.5) & (upper_ny < 0.65)
    ap_strength = np.where(
        upper_blocked,
        np.clip((lower_dist - upper_dist - 0.75) / 2.5, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    low_aperture[9:13, :] = ap_strength[None, :]

    floor_pit = np.zeros((FOVEA_H, FOVEA_W), dtype=np.float32)
    depth_jump = np.maximum(0.0, dist_m[1:, :] - dist_m[:-1, :] - 0.35)
    steep_floor = np.clip((0.82 - ny[9:, :]) * (norm_d[9:, :] > 0.06), 0.0, 1.0)
    pit_jump = np.tanh(depth_jump[8:, :] * 1.4)
    floor_pit[9:, :] = np.clip(0.75 * pit_jump + 0.25 * (steep_floor * (dist_m[9:, :] > 1.2)), 0.0, 1.0).astype(np.float32)
    return low_aperture, floor_pit


def _resolve_active_fovea_patch(
    world_rec: Any,
    vc_rec: Any,
    cfg: Brain3DConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """Select active `16x16x4` foveal patch (`neural_stereo` reads GPU `VISION_CORTEX3D` strictly)."""
    if str(getattr(cfg, "visual_mode", VISUAL_MODE_RAYCAST)) == VISUAL_MODE_NEURAL_STEREO:
        max_tick_lag = max(12, int(round(12.0 * max(1.0, float(getattr(cfg, "time_scale", 1.0))))))
        if (
            vc_rec is not None
            and int(vc_rec["depth_ready"]) == 1
            and int(vc_rec["episode_id"]) == int(world_rec["episode_id"])
            and abs(int(world_rec["physics_tick"]) - int(vc_rec["physics_tick"])) <= max_tick_lag
        ):
            return (
                np.asarray(vc_rec["pred_fovea_patch"], dtype=np.float32).copy(),
                np.asarray(vc_rec["epistemic_uncert_16"], dtype=np.float32).copy(),
                True,
            )
        return (
            np.zeros((FOVEA_H, FOVEA_W, FOVEA_C), dtype=np.float32),
            np.ones((FOVEA_H, FOVEA_W), dtype=np.float32),
            False,
        )
    return np.asarray(world_rec["fovea_patch"], dtype=np.float32), None, True


def compute_ventral_appearance_patch_16(
    stereo_rgb_u8: np.ndarray,
    yaw_pitch: np.ndarray,
    prev_state: Optional[Dict[str, Any]] = None,
    episode_id: int = 0,
) -> np.ndarray:
    """
    Pure-NumPy Ventral Appearance Stream (`16x16x4`) aligned 1-to-1 with Left-eye `64x64` retina:
      ch0: R-G color-opponent in [-1, 1] (critical for team/surface color contrast!)
      ch1: B-(R+G)/2 color-opponent in [-1, 1]
      ch2: local 4x4 luminance-texture + chromatic saturation in [0, 1]
      ch3: ego-rotation-damped temporal motion delta in [0, 1]
    """
    st_arr = np.asarray(stereo_rgb_u8)
    rgb_raw = st_arr[0] if st_arr.ndim == 4 else st_arr
    rgb_f = rgb_raw.astype(np.float32)
    if st_arr.dtype == np.uint8 or float(np.max(rgb_f)) > 1.5:
        rgb_f = rgb_f / 255.0
    rgb_f = np.clip(rgb_f, 0.0, 1.0).reshape(RETINA_H, RETINA_W, RETINA_C)

    blocks = rgb_f.reshape(FOVEA_H, 4, FOVEA_W, 4, RETINA_C)
    mean_rgb = blocks.mean(axis=(1, 3))
    r16, g16, b16 = mean_rgb[:, :, 0], mean_rgb[:, :, 1], mean_rgb[:, :, 2]

    rg_16 = np.clip((r16 - g16) * 1.35, -1.0, 1.0).astype(np.float32)
    by_16 = np.clip((b16 - 0.5 * (r16 + g16)) * 1.35, -1.0, 1.0).astype(np.float32)

    luma_64 = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    luma_blk = luma_64.reshape(FOVEA_H, 4, FOVEA_W, 4)
    luma_16 = luma_blk.mean(axis=(1, 3)).astype(np.float32)
    std_16 = luma_blk.std(axis=(1, 3)).astype(np.float32)
    c_max = np.max(mean_rgb, axis=-1)
    c_min = np.min(mean_rgb, axis=-1)
    sat_16 = (c_max - c_min) / (c_max + 0.05)
    tex_sat_16 = np.clip(0.55 * (std_16 * 2.5) + 0.45 * sat_16, 0.0, 1.0).astype(np.float32)

    yp_arr = np.asarray(yaw_pitch, dtype=np.float32).reshape(2)
    temp_delta_16 = np.zeros((FOVEA_H, FOVEA_W), dtype=np.float32)
    if (
        prev_state is not None
        and int(prev_state.get("ep_id", -1)) == int(episode_id)
        and "luma_16" in prev_state
    ):
        raw_diff = 0.65 * np.abs(luma_16 - prev_state["luma_16"]) + 0.35 * np.abs(rg_16 - prev_state["rg_16"])
        prev_yp = np.asarray(prev_state["yaw_pitch"], dtype=np.float32)
        rot_mag = math.hypot(float(yp_arr[0] - prev_yp[0]), float(yp_arr[1] - prev_yp[1]))
        global_med = float(np.median(raw_diff))
        ego_sub = np.maximum(0.0, raw_diff - min(1.0, rot_mag * 5.5) * global_med)
        ego_gain = 1.0 / (1.0 + 6.0 * rot_mag)
        temp_delta_16 = np.clip(
            ego_sub * (1.8 * ego_gain + 0.35 * np.maximum(0.0, raw_diff - 1.6 * global_med)),
            0.0,
            1.0,
        ).astype(np.float32)

    if prev_state is not None:
        prev_state["ep_id"] = int(episode_id)
        prev_state["luma_16"] = luma_16.copy()
        prev_state["rg_16"] = rg_16.copy()
        prev_state["yaw_pitch"] = yp_arr.copy()

    return np.stack([rg_16, by_16, tex_sat_16, temp_delta_16], axis=-1).astype(np.float32)


def derive_ventral_fallback_from_dorsal_16(
    fovea_patch: np.ndarray,
    yaw_pitch: Optional[np.ndarray] = None,
    prev_state: Optional[Dict[str, Any]] = None,
    episode_id: int = 0,
) -> np.ndarray:
    """Deterministic geometric + temporal pseudo-ventral `16x16x4` fallback when RGB retina is absent."""
    p = np.asarray(fovea_patch, dtype=np.float32).reshape(FOVEA_H, FOVEA_W, FOVEA_C)
    norm_d = np.clip(p[:, :, 0], 0.0, 1.0)
    nx = np.clip(p[:, :, 1], -1.0, 1.0)
    ny = np.clip(p[:, :, 2], -1.0, 1.0)

    pad_d = np.pad(norm_d, 1, mode="edge")
    neigh_d = (
        pad_d[:-2, :-2] + pad_d[:-2, 1:-1] + pad_d[:-2, 2:]
        + pad_d[1:-1, :-2] + pad_d[1:-1, 1:-1] + pad_d[1:-1, 2:]
        + pad_d[2:, :-2] + pad_d[2:, 1:-1] + pad_d[2:, 2:]
    ) / 9.0
    fg_relief = norm_d - neigh_d
    ch0 = np.clip(fg_relief * 3.2 - 0.30 * np.maximum(0.0, ny - 0.35), -1.0, 1.0)
    ch1 = np.clip(nx * 0.65 - (norm_d - 0.5) * 0.5, -1.0, 1.0)

    gx = np.zeros_like(norm_d)
    gy = np.zeros_like(norm_d)
    gx[:, 1:] = np.abs(norm_d[:, 1:] - norm_d[:, :-1])
    gy[1:, :] = np.abs(norm_d[1:, :] - norm_d[:-1, :])
    gnorm = np.zeros_like(norm_d)
    gnorm[:, 1:] = 0.5 * (np.abs(nx[:, 1:] - nx[:, :-1]) + np.abs(ny[:, 1:] - ny[:, :-1]))
    gnorm[1:, :] += 0.5 * (np.abs(nx[1:, :] - nx[:-1, :]) + np.abs(ny[1:, :] - ny[:-1, :]))
    ch2 = np.clip(2.2 * (gx + gy) + 0.45 * gnorm, 0.0, 1.0)

    ch3 = np.zeros_like(norm_d)
    if prev_state is not None:
        yp_arr = (
            np.asarray(yaw_pitch, dtype=np.float32).reshape(2)
            if yaw_pitch is not None
            else np.zeros(2, dtype=np.float32)
        )
        if (
            prev_state.get("episode_id") == int(episode_id)
            and prev_state.get("norm_d") is not None
            and prev_state.get("yaw_pitch") is not None
        ):
            raw_diff = np.abs(norm_d - prev_state["norm_d"])
            rot_delta = float(np.linalg.norm(yp_arr - prev_state["yaw_pitch"]))
            rot_damp = math.exp(-14.0 * rot_delta)
            local_diff = np.maximum(0.0, raw_diff - 0.85 * float(np.median(raw_diff)))
            ch3 = np.clip(local_diff * 3.5 * rot_damp, 0.0, 1.0)
        prev_state["episode_id"] = int(episode_id)
        prev_state["norm_d"] = norm_d.copy()
        prev_state["yaw_pitch"] = yp_arr.copy()

    return np.stack([ch0, ch1, ch2, ch3], axis=-1).astype(np.float32)


def compose_vis8_patch(
    fovea_patch: np.ndarray,
    ventral_patch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fuse dorsal `(16,16,4)` geometry/depth and ventral `(16,16,4)` appearance into `(16,16,8)`."""
    d = np.asarray(fovea_patch, dtype=np.float32).reshape(FOVEA_H, FOVEA_W, FOVEA_C)
    v = (
        derive_ventral_fallback_from_dorsal_16(d)
        if ventral_patch is None
        else np.asarray(ventral_patch, dtype=np.float32).reshape(FOVEA_H, FOVEA_W, VENTRAL_C)
    )
    return np.concatenate([d, v], axis=-1).astype(np.float32)


def _resolve_active_ventral_patch(
    world_rec: Any,
    vc_rec: Any,
    cfg: Brain3DConfig,
    prev_state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Resolve live ventral `(16,16,4)` appearance stream from `VISION_CORTEX3D` or dorsal fallback."""
    max_tick_lag = max(12, int(round(12.0 * max(1.0, float(getattr(cfg, "time_scale", 1.0))))))
    vc_has_ventral = (
        ("ventral_fovea_patch" in vc_rec)
        if isinstance(vc_rec, dict)
        else ("ventral_fovea_patch" in (getattr(getattr(vc_rec, "dtype", None), "names", None) or ()))
    )
    if (
        vc_rec is not None
        and int(vc_rec["depth_ready"]) == 1
        and int(vc_rec["episode_id"]) == int(world_rec["episode_id"])
        and abs(int(world_rec["physics_tick"]) - int(vc_rec["physics_tick"])) <= max_tick_lag
        and vc_has_ventral
    ):
        return np.asarray(vc_rec["ventral_fovea_patch"], dtype=np.float32).copy()
    has_yp = (
        ("self_yaw_pitch" in world_rec)
        if isinstance(world_rec, dict)
        else ("self_yaw_pitch" in (getattr(getattr(world_rec, "dtype", None), "names", None) or ()))
    )
    yp = world_rec["self_yaw_pitch"] if has_yp else None
    return derive_ventral_fallback_from_dorsal_16(
        world_rec["fovea_patch"],
        yaw_pitch=yp,
        prev_state=prev_state,
        episode_id=int(world_rec["episode_id"]),
    )


