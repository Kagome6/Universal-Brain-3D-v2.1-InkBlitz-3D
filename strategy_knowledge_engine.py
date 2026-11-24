# -*- coding: utf-8 -*-
"""
strategy_knowledge_engine.py — Multi-Regime Strategic Knowledge DB, Hybrid Search & Ollama Gemma 4 Pipeline (v2.3.0)
====================================================================================================================
Strictly Zero-PyTorch (`Barrier 1` compliant) strategic memory, post-match Turning-Point differential
analyst, and 4-boundary closed-loop strategic controller for the 3D Functionalist Brain (`InkBlitz 3D`).

Architectural Separation of Concerns (Ontology & Timescale Discipline):
  1. STRATEGY (30s–90s Match & Macro-Regime Horizon — THIS MODULE):
     - Governs macro resource posture (`curiosity_mult`, `objective_mult`, `stealth_mult`, `caution_delta`),
       hippocampal corridor priors (`LEFT_WING / CENTER_AXIS / RIGHT_WING / DUAL_WINGS`), and state-conditional
       peace-time posture across 4 fixed board regimes:
         * `OPENING_CONTEST` (0): Neutral early/mid board contest
         * `ADVANTAGE_LOCK`  (1): Protecting a significant objective count lead without over-extending
         * `DEFENSE_CRISIS`  (2): Enemy BlitzCore carrier active or deep behind on count -> immediate rally
         * `COMEBACK_CLUTCH` (3): Late-game deficit, overtime, or deadlock -> high-initiative wing breakthrough
     - Re-anchors at FOUR discrete boundaries (NOT just match start `t=0`!):
         [B1] Match Start (`episode_id` boundary)
         [B2] Mid-Episode Respawn (`flags & 16` / `respawn_event_flag` -> penalizes repeated death lane!)
         [B3] Macro Regime Shift (`OPENING <-> LEAD_LOCK <-> DEFENSE_CRISIS <-> COMEBACK_CLUTCH`)
         [B4] Tactical Stagnation Break (deadlock `> 3s` -> flank lane reroute + curiosity boost)
  2. TACTICS (0.5s–8s Local Engagement — `ConsciousGWT` + `StrategicTransformer3D` + Hippocampus Dijkstra):
     - Immediate engagement (`ENGAGE_INTERCEPT`), cover dive (`EVADE_REGROUP`), local obstacle pathing.
     - Strategic modifiers yield 100% immediately whenever an enemy threat is present (`exposed_threat` or
       `nearest_hostile_within_range`), preventing pre-match plans from causing tactical rigidity!
  3. MOTOR / REFLEX (16ms / 60Hz — `D3QN` + `Cerebellum` + `BasalGanglia` + Plan-A Isolation):
     - Ballistic lead-aim, strafe decoupling, and `RSSM` Survival Veto are NEVER overridden by LLM outputs.

Fixed Measurement Yardstick vs. Dynamic Proposal Layer:
  - Fixed Schema: 8 canonical Turning-Point `CAUSE_TAGS`, 4 `StrategicRegime` IDs, 5 `StrategicLanePref` IDs.
  - Dynamic Proposal Separation: New doctrines from `Ollama gemma4:12b` (`think: false`, JSON Schema enforced)
    enter SQLite as `status = 'candidate'`, require `>= 3` match trials before promotion to `'validated'` or
    demotion to `'deprecated'`, and deprecated anti-patterns are retained to prevent repeating failed ideas.

Quick Self-Test:
  python -B strategy_knowledge_engine.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
import queue
import sqlite3
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PARENT_WS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isfile(os.path.join(_PARENT_WS_DIR, "universal_brain_3d_core.py")) and _PARENT_WS_DIR not in sys.path:
    sys.path.insert(0, _PARENT_WS_DIR)

from task_profiles_3d import (
    GOAL_MODE_BALANCED_EXPLORE,
    GOAL_MODE_ENGAGE_INTERCEPT,
    GOAL_MODE_EVADE_REGROUP,
    GOAL_MODE_PUSH_OBJECTIVE,
    GOAL_MODE_STEALTH_FLANK,
    PROPRIO_IDX_HAZARD,
    PROPRIO_IDX_OBJ_PROGRESS,
    PROPRIO_IDX_RESOURCE_CARRIER,
    PROPRIO_IDX_SURFACE_POSTURE,
    SemanticTag,
    get_resource_tank_level,
    is_carrying_primary_objective,
)


def assert_zero_torch_in_strategy_engine() -> None:
    if "torch" in sys.modules:
        raise RuntimeError("Barrier 1 violation: `torch` must NEVER be imported inside strategy_knowledge_engine!")


def _sleep_lock_path(db_path: str) -> str:
    return os.path.abspath(db_path) + ".sleep_active"


def coach_request_file_path(db_path: str) -> str:
    return os.path.abspath(db_path) + ".coach_req.json"


def submit_human_coach_request_file(db_path: str, user_text: str, stage_id: int = 1) -> bool:
    """
    Atomically write a human strategic suggestion (`user_text` from Whisper STT or keyboard form)
    for `AsyncStrategyFeedbackWorker` to translate via Gemma 4 12B and inject live into the brain!
    """
    cleaned = (user_text or "").strip()
    if not db_path or not cleaned:
        return False
    dst = coach_request_file_path(db_path)
    tmp = dst + ".tmp"
    try:
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {"ts": time.time(), "stage_id": int(stage_id), "user_text": cleaned[:280]}
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, dst)
        return True
    except OSError:
        return False


def set_sleep_consolidation_active(db_path: str, active: bool, max_hold_s: float = 20.0) -> None:
    """
    Cross-process Sleep-vs-Ollama Mutual Exclusion Gate:
    Signals that PyTorch Sleep S-1..S-4 consolidation is currently using the GPU so
    `AsyncStrategyFeedbackWorker` defers Ollama 12B HTTP inference until Sleep finishes!
    """
    if not db_path:
        return
    lp = _sleep_lock_path(db_path)
    try:
        if active:
            parent = os.path.dirname(lp)
            if parent:
                os.makedirs(parent, exist_ok=True)
            deadline = time.time() + float(max(1.0, max_hold_s))
            with open(lp, "w", encoding="utf-8") as fh:
                fh.write(f"{deadline:.3f}\n")
        else:
            if os.path.exists(lp):
                os.remove(lp)
    except OSError:
        pass


def is_sleep_consolidation_active(db_path: str) -> bool:
    if not db_path:
        return False
    lp = _sleep_lock_path(db_path)
    if not os.path.exists(lp):
        return False
    try:
        with open(lp, "r", encoding="utf-8") as fh:
            deadline = float(fh.read().strip() or "0")
        if time.time() < deadline:
            return True
        try:
            os.remove(lp)
        except OSError:
            pass
    except Exception:
        pass
    return False


def emit_gemma_broadcast_stat(payload: Dict[str, Any]) -> None:
    """
    Emit atomic single-line `[PBT_STAT]` JSON packet consumed by Unity `InkBlitzSessionDashboardUi`.
    Uses `ensure_ascii=True` (`\\uXXXX` escapes) so Windows pipe codepages (`CP932` vs `UTF-8`) can
    NEVER garble Japanese characters before `JsonUtility.FromJson` decodes them into UTF-16!
    """
    pkt = dict(payload)
    pkt["evt"] = "gemma_broadcast"
    try:
        line = "[PBT_STAT] " + json.dumps(pkt, ensure_ascii=True, separators=(",", ":")) + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass


# ==============================================================================
# 1. Fixed Measurement Taxonomy (Ontology Fixed Layer — Yardstick Never Drifts)
# ==============================================================================
SCHEMA_VERSION = "1.0"

REGIME_OPENING_CONTEST = 0
REGIME_ADVANTAGE_LOCK = 1
REGIME_DEFENSE_CRISIS = 2
REGIME_COMEBACK_CLUTCH = 3

REGIME_NAMES: Dict[int, str] = {
    REGIME_OPENING_CONTEST: "OPENING_CONTEST",
    REGIME_ADVANTAGE_LOCK: "ADVANTAGE_LOCK",
    REGIME_DEFENSE_CRISIS: "DEFENSE_CRISIS",
    REGIME_COMEBACK_CLUTCH: "COMEBACK_CLUTCH",
}
REGIME_JA_SHORT: Dict[int, str] = {
    REGIME_OPENING_CONTEST: "拮抗制圧",
    REGIME_ADVANTAGE_LOCK: "優勢固め",
    REGIME_DEFENSE_CRISIS: "危機防衛",
    REGIME_COMEBACK_CLUTCH: "劣勢打開",
}
NAME_TO_REGIME: Dict[str, int] = {v: k for k, v in REGIME_NAMES.items()}

LANE_BALANCED = 0
LANE_LEFT_WING = 1
LANE_CENTER_AXIS = 2
LANE_RIGHT_WING = 3
LANE_DUAL_WINGS = 4

LANE_NAMES: Dict[int, str] = {
    LANE_BALANCED: "BALANCED",
    LANE_LEFT_WING: "LEFT_WING",
    LANE_CENTER_AXIS: "CENTER_AXIS",
    LANE_RIGHT_WING: "RIGHT_WING",
    LANE_DUAL_WINGS: "DUAL_WINGS",
}
LANE_SPOKEN_JA: Dict[int, str] = {
    LANE_BALANCED: "ステージ全域",
    LANE_LEFT_WING: "左の高台ルート",
    LANE_CENTER_AXIS: "中央ルート",
    LANE_RIGHT_WING: "右の高台ルート",
    LANE_DUAL_WINGS: "左右の両翼ルート",
}
NAME_TO_LANE: Dict[str, int] = {v: k for k, v in LANE_NAMES.items()}

GENTLE_NAVIGATOR_SYSTEM_PROMPT = (
    "あなたは3Dブリッツコア対戦AIの大局戦略アナリスト兼・戦術ナビゲーターです。"
    "【重要】まず客観的な分析に基づいてスキーマの分類タグ・レーン・数値倍率を正確に決定してください（思考精度を最優先）。"
    "そのうえで、人間向けの解説フィールド `commentary_ja` のみ、"
    "『知的で落ち着いた優しい女の子のナビゲーター』として、丁寧で柔らかい語尾（〜ですね、〜しましょうね、〜してみますね、〜が良さそうです 等）で2文以内（85文字以内）にまとめてください。"
    "音声合成(AivisSpeech)でそのまま読み上げるため、`commentary_ja` の中には CENTER_AXIS 等の英字定数や +0.15 等の数値記号を書かず、"
    "『中央ルート』『左の高台』『塗りのリード』のように耳で聞いて自然な日本語表現だけを使ってください。"
)

# Canonical (Left -X, Center 0, Right +X) frontier multipliers in `[0.65, 1.45]`
LANE_WEIGHT_PRESETS: Dict[int, Tuple[float, float, float]] = {
    LANE_BALANCED: (1.00, 1.00, 1.00),
    LANE_LEFT_WING: (1.32, 0.86, 0.94),
    LANE_CENTER_AXIS: (0.92, 1.26, 0.92),
    LANE_RIGHT_WING: (0.94, 0.86, 1.32),
    LANE_DUAL_WINGS: (1.26, 0.76, 1.26),
}

# Fixed 8-category Turning-Point Cause Taxonomy (prevents moving-yardstick drift)
CAUSE_TAGS: Tuple[str, ...] = (
    "CENTRAL_OVEREXTEND_WIPE",     # Loss driver: repeated central corridor deaths without turf support
    "UNPAINTED_NO_REFUGE",         # Loss driver: starved of friendly ink / escape refuge during fights
    "ENEMY_CARRIER_UNCONTESTED",   # Loss driver: failed to intercept enemy BlitzCore carrier push
    "PASSIVE_TURF_STALL",          # Loss driver: painted edges passively while losing objective progress
    "LEAD_PROTECTION_THROW",       # Loss driver: held strong lead then over-pushed and got counter-wiped
    "WING_HIGHGROUND_CONTROL",     # Win driver: flanked via West/East Overlook or SkyBridge to break mid
    "CLEAN_PICK_TO_PUSH",          # Win driver: secured splat/turf advantage first, then pushed objective
    "CLUTCH_FLANK_BREAKTHROUGH",   # Win driver: broke deadlock / deficit via alternate corridor push
)

PEACE_HINT_TO_GOAL_MODE: Dict[str, int] = {
    "EXPLORE": GOAL_MODE_BALANCED_EXPLORE,
    "PUSH_OBJECTIVE": GOAL_MODE_PUSH_OBJECTIVE,
    "ENGAGE_INTERCEPT": GOAL_MODE_ENGAGE_INTERCEPT,
    "EVADE_REGROUP": GOAL_MODE_EVADE_REGROUP,
    "STEALTH_FLANK": GOAL_MODE_STEALTH_FLANK,
}
GOAL_MODE_TO_PEACE_HINT: Dict[int, str] = {v: k for k, v in PEACE_HINT_TO_GOAL_MODE.items()}

MIN_VALIDATION_TRIALS = 3

# Structured JSON Schema passed to Ollama `format` parameter (`gemma4:12b`, `think: false`)
OLLAMA_STRATEGY_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis_name": {"type": "string"},
        "cause_tag": {"type": "string", "enum": list(CAUSE_TAGS)},
        "commentary_ja": {"type": "string"},
        "opening_lane": {"type": "string", "enum": list(LANE_NAMES.values())},
        "opening_curiosity_mult": {"type": "number"},
        "opening_objective_mult": {"type": "number"},
        "lead_caution_delta": {"type": "number"},
        "lead_stealth_mult": {"type": "number"},
        "crisis_objective_mult": {"type": "number"},
        "comeback_lane": {"type": "string", "enum": list(LANE_NAMES.values())},
        "comeback_objective_mult": {"type": "number"},
    },
    "required": [
        "hypothesis_name",
        "cause_tag",
        "commentary_ja",
        "opening_lane",
        "opening_curiosity_mult",
        "opening_objective_mult",
        "lead_caution_delta",
        "comeback_lane",
        "comeback_objective_mult",
    ],
}


# ==============================================================================
# 2. Clamped Strategic Slice & Multi-Regime Doctrine Data Structures
# ==============================================================================
@dataclass
class RegimePolicySlice:
    lane_code: int = LANE_BALANCED
    curiosity_mult: float = 1.00     # Clamped to [0.75, 1.35] (peace-time frontier/paint drive)
    objective_mult: float = 1.00     # Clamped to [0.75, 1.40] (peace-time BlitzCore/pedestal drive)
    stealth_mult: float = 1.00       # Clamped to [0.80, 1.35] (swim/cover discipline)
    caution_delta: float = 0.00      # Clamped to [-0.04, +0.06] added to survival_veto_threshold
    peace_goal_mode: int = GOAL_MODE_BALANCED_EXPLORE

    def clamped(self) -> "RegimePolicySlice":
        lc = int(self.lane_code) if int(self.lane_code) in LANE_WEIGHT_PRESETS else LANE_BALANCED
        gm = int(self.peace_goal_mode) if int(self.peace_goal_mode) in GOAL_MODE_TO_PEACE_HINT else GOAL_MODE_BALANCED_EXPLORE
        return RegimePolicySlice(
            lane_code=lc,
            curiosity_mult=float(np.clip(self.curiosity_mult, 0.75, 1.35)),
            objective_mult=float(np.clip(self.objective_mult, 0.75, 1.40)),
            stealth_mult=float(np.clip(self.stealth_mult, 0.80, 1.35)),
            caution_delta=float(np.clip(self.caution_delta, -0.04, 0.06)),
            peace_goal_mode=gm,
        )

    def to_dict(self) -> Dict[str, Any]:
        c = self.clamped()
        return {
            "lane": LANE_NAMES.get(c.lane_code, "BALANCED"),
            "curiosity_mult": round(c.curiosity_mult, 3),
            "objective_mult": round(c.objective_mult, 3),
            "stealth_mult": round(c.stealth_mult, 3),
            "caution_delta": round(c.caution_delta, 3),
            "peace_mode": GOAL_MODE_TO_PEACE_HINT.get(c.peace_goal_mode, "EXPLORE"),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RegimePolicySlice":
        if not isinstance(d, dict):
            return RegimePolicySlice()
        lane_s = str(d.get("lane", "BALANCED")).upper().strip()
        mode_s = str(d.get("peace_mode", "EXPLORE")).upper().strip()
        return RegimePolicySlice(
            lane_code=NAME_TO_LANE.get(lane_s, LANE_BALANCED),
            curiosity_mult=float(d.get("curiosity_mult", 1.0)),
            objective_mult=float(d.get("objective_mult", 1.0)),
            stealth_mult=float(d.get("stealth_mult", 1.0)),
            caution_delta=float(d.get("caution_delta", 0.0)),
            peace_goal_mode=PEACE_HINT_TO_GOAL_MODE.get(mode_s, GOAL_MODE_BALANCED_EXPLORE),
        ).clamped()


@dataclass
class StrategicDoctrine:
    doctrine_id: int = 0
    stage_id: int = 1
    source: str = "seed"             # "seed" | "gemma4" | "stat_counter"
    status: str = "seed"             # "seed" | "candidate" | "validated" | "deprecated"
    target_cause_tag: str = "CLEAN_PICK_TO_PUSH"
    hypothesis_name: str = "Balanced_Turf_Then_Push"
    summary_ja: str = "開幕は自陣〜中央の塗りを確保し、人数・盤面有利を作ってからコアを前進させる標準戦略。"
    regimes: Dict[int, RegimePolicySlice] = field(default_factory=dict)
    context_vec8: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    trials_count: int = 0
    wins_count: int = 0
    mean_win_rate: float = 0.50
    mean_obj_delta: float = 0.00

    def slice_for_regime(self, regime_id: int) -> RegimePolicySlice:
        if int(regime_id) in self.regimes:
            return self.regimes[int(regime_id)].clamped()
        return RegimePolicySlice()

    def regimes_to_json(self) -> str:
        out: Dict[str, Any] = {}
        for r_id, r_name in REGIME_NAMES.items():
            out[r_name] = self.slice_for_regime(r_id).to_dict()
        return json.dumps(out, ensure_ascii=False)

    @staticmethod
    def regimes_from_json(raw_json: str) -> Dict[int, RegimePolicySlice]:
        out: Dict[int, RegimePolicySlice] = {r_id: RegimePolicySlice() for r_id in REGIME_NAMES}
        try:
            parsed = json.loads(raw_json or "{}")
            if isinstance(parsed, dict):
                for r_id, r_name in REGIME_NAMES.items():
                    if r_name in parsed and isinstance(parsed[r_name], dict):
                        out[r_id] = RegimePolicySlice.from_dict(parsed[r_name])
        except Exception:
            pass
        return out


def _cosine_sim8(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)[:8]
    vb = np.asarray(b, dtype=np.float32).reshape(-1)[:8]
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.clip(float(np.dot(va, vb)) / (na * nb), -1.0, 1.0))


def _lane_from_x(x_canon: float) -> int:
    if x_canon < -3.6:
        return LANE_LEFT_WING
    if x_canon > 3.6:
        return LANE_RIGHT_WING
    return LANE_CENTER_AXIS


# ==============================================================================
# 3. SQLite Strategic Knowledge DB + Dedicated Hybrid Search Engine
# ==============================================================================
class StrategyKnowledgeDB:
    """
    Single-file SQLite (`WAL` mode) Strategic Knowledge Base with a built-in Hybrid Search Engine:
      - Exact `stage_id` + `target_cause_tag` alignment
      - 8D Match-Situation Vector Cosine Similarity
      - `FTS5` Full-Text Search over Japanese/English strategy summaries & hypotheses (with LIKE fallback)
      - Strict Lifecycle Separation (`seed` / `candidate` / `validated` / `deprecated`)
    """

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._fts_enabled = False
        self._init_schema_and_seeds()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=3500;")
        return conn

    def _init_schema_and_seeds(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS match_episodes (
                        episode_uid INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_ts REAL NOT NULL,
                        stage_id INTEGER NOT NULL,
                        episode_id INTEGER NOT NULL,
                        outcome TEXT NOT NULL,
                        won INTEGER NOT NULL,
                        final_obj_progress REAL NOT NULL,
                        net_score REAL NOT NULL,
                        friendly_turf_peak REAL NOT NULL,
                        enemy_turf_peak REAL NOT NULL,
                        respawns INTEGER NOT NULL,
                        vetoes INTEGER NOT NULL,
                        death_lanes_json TEXT NOT NULL,
                        turning_point_json TEXT NOT NULL,
                        inner_speech_json TEXT NOT NULL,
                        feature_vec8_json TEXT NOT NULL,
                        applied_doctrine_id INTEGER NOT NULL,
                        cause_tag TEXT NOT NULL,
                        commentary_ja TEXT NOT NULL DEFAULT ''
                    );

                    CREATE INDEX IF NOT EXISTS idx_episodes_stage_cause
                        ON match_episodes(stage_id, cause_tag, created_ts DESC);

                    CREATE TABLE IF NOT EXISTS strategy_doctrines (
                        doctrine_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_ts REAL NOT NULL,
                        schema_version TEXT NOT NULL,
                        stage_id INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        status TEXT NOT NULL,
                        target_cause_tag TEXT NOT NULL,
                        hypothesis_name TEXT NOT NULL,
                        summary_ja TEXT NOT NULL,
                        regime_policy_json TEXT NOT NULL,
                        context_vec8_json TEXT NOT NULL,
                        trials_count INTEGER NOT NULL DEFAULT 0,
                        wins_count INTEGER NOT NULL DEFAULT 0,
                        sum_obj_delta REAL NOT NULL DEFAULT 0.0,
                        sum_score REAL NOT NULL DEFAULT 0.0,
                        mean_win_rate REAL NOT NULL DEFAULT 0.5,
                        mean_obj_delta REAL NOT NULL DEFAULT 0.0
                    );

                    CREATE INDEX IF NOT EXISTS idx_doctrines_stage_status
                        ON strategy_doctrines(stage_id, status, target_cause_tag);
                    """
                )
                try:
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS strategy_doctrines_fts USING fts5(
                            doctrine_id UNINDEXED,
                            hypothesis_name,
                            summary_ja,
                            target_cause_tag
                        );
                        """
                    )
                    self._fts_enabled = True
                except sqlite3.OperationalError:
                    self._fts_enabled = False

                cur = conn.execute("SELECT COUNT(*) AS cnt FROM strategy_doctrines;")
                cnt = int(cur.fetchone()["cnt"])
                if cnt == 0:
                    self._insert_seed_doctrines_locked(conn)
                conn.commit()
            finally:
                conn.close()

    def _insert_seed_doctrines_locked(self, conn: sqlite3.Connection) -> None:
        seeds = [
            StrategicDoctrine(
                stage_id=0,  # Universal fallback across any stage_id
                source="seed",
                status="seed",
                target_cause_tag="CLEAN_PICK_TO_PUSH",
                hypothesis_name="Seed_Balanced_Turf_Control",
                summary_ja="まずは自陣から中央の足場を丁寧に塗り広げて、リードしたら無理せず潜伏で固めていきましょうね。",
                regimes={
                    REGIME_OPENING_CONTEST: RegimePolicySlice(LANE_BALANCED, 1.10, 1.02, 1.00, 0.00, GOAL_MODE_BALANCED_EXPLORE),
                    REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_DUAL_WINGS, 1.08, 0.88, 1.18, +0.03, GOAL_MODE_STEALTH_FLANK),
                    REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.80, 1.32, 0.95, +0.02, GOAL_MODE_ENGAGE_INTERCEPT),
                    REGIME_COMEBACK_CLUTCH: RegimePolicySlice(LANE_LEFT_WING, 0.92, 1.30, 1.08, -0.02, GOAL_MODE_PUSH_OBJECTIVE),
                },
                context_vec8=np.array([0.0, 0.25, 0.20, 0.20, 0.20, 0.45, 0.20, 0.15], dtype=np.float32),
            ),
            StrategicDoctrine(
                stage_id=0,
                source="seed",
                status="seed",
                target_cause_tag="CENTRAL_OVEREXTEND_WIPE",
                hypothesis_name="Seed_Wing_Overlook_AntiWipe",
                summary_ja="中央正面での消耗を避けて、左右の高台とスカイブリッジから塗りと射線を広げてみましょうね。",
                regimes={
                    REGIME_OPENING_CONTEST: RegimePolicySlice(LANE_DUAL_WINGS, 1.18, 0.95, 1.15, +0.02, GOAL_MODE_STEALTH_FLANK),
                    REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_DUAL_WINGS, 1.12, 0.85, 1.22, +0.04, GOAL_MODE_STEALTH_FLANK),
                    REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_RIGHT_WING, 0.82, 1.28, 1.05, +0.02, GOAL_MODE_ENGAGE_INTERCEPT),
                    REGIME_COMEBACK_CLUTCH: RegimePolicySlice(LANE_RIGHT_WING, 0.95, 1.34, 1.12, -0.01, GOAL_MODE_PUSH_OBJECTIVE),
                },
                context_vec8=np.array([1.0, 0.75, 0.35, 0.25, 0.15, 0.35, 0.65, -0.25], dtype=np.float32),
            ),
            StrategicDoctrine(
                stage_id=0,
                source="seed",
                status="seed",
                target_cause_tag="PASSIVE_TURF_STALL",
                hypothesis_name="Seed_Decisive_Objective_Tempo",
                summary_ja="端塗りばかりで時間が切れないように、中央を押さえたら早めにコアを前へ進めていきましょうね。",
                regimes={
                    REGIME_OPENING_CONTEST: RegimePolicySlice(LANE_CENTER_AXIS, 0.94, 1.25, 1.00, 0.00, GOAL_MODE_PUSH_OBJECTIVE),
                    REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_BALANCED, 1.05, 0.95, 1.15, +0.03, GOAL_MODE_BALANCED_EXPLORE),
                    REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.78, 1.35, 0.95, +0.01, GOAL_MODE_ENGAGE_INTERCEPT),
                    REGIME_COMEBACK_CLUTCH: RegimePolicySlice(LANE_LEFT_WING, 0.85, 1.38, 1.05, -0.03, GOAL_MODE_PUSH_OBJECTIVE),
                },
                context_vec8=np.array([1.0, 0.20, 0.15, 0.20, 0.75, 0.55, 0.15, -0.10], dtype=np.float32),
            ),
        ]
        for doc in seeds:
            self._insert_doctrine_conn(conn, doc)

    def _insert_doctrine_conn(self, conn: sqlite3.Connection, doc: StrategicDoctrine) -> int:
        vec_list = [round(float(x), 4) for x in np.asarray(doc.context_vec8, dtype=np.float32).reshape(-1)[:8]]
        while len(vec_list) < 8:
            vec_list.append(0.0)
        cur = conn.execute(
            """
            INSERT INTO strategy_doctrines (
                created_ts, schema_version, stage_id, source, status,
                target_cause_tag, hypothesis_name, summary_ja,
                regime_policy_json, context_vec8_json,
                trials_count, wins_count, sum_obj_delta, sum_score,
                mean_win_rate, mean_obj_delta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                time.time(),
                SCHEMA_VERSION,
                int(doc.stage_id),
                str(doc.source),
                str(doc.status),
                str(doc.target_cause_tag),
                str(doc.hypothesis_name)[:96],
                str(doc.summary_ja)[:320],
                doc.regimes_to_json(),
                json.dumps(vec_list),
                int(doc.trials_count),
                int(doc.wins_count),
                0.0,
                0.0,
                float(doc.mean_win_rate),
                float(doc.mean_obj_delta),
            ),
        )
        doc_id = int(cur.lastrowid)
        if self._fts_enabled:
            conn.execute(
                "INSERT INTO strategy_doctrines_fts(doctrine_id, hypothesis_name, summary_ja, target_cause_tag) VALUES (?, ?, ?, ?);",
                (doc_id, str(doc.hypothesis_name)[:96], str(doc.summary_ja)[:320], str(doc.target_cause_tag)),
            )
        return doc_id

    def register_candidate_doctrine(self, doc: StrategicDoctrine) -> int:
        """Store a newly proposed strategy hypothesis as `status='candidate'` (separated from automatic validation!)."""
        doc.status = "candidate" if doc.source != "seed" else "seed"
        if doc.target_cause_tag not in CAUSE_TAGS:
            doc.target_cause_tag = "CLEAN_PICK_TO_PUSH"
        with self._lock:
            conn = self._connect()
            try:
                # Deduplicate if an active candidate with the exact same hypothesis_name on this stage already exists
                row = conn.execute(
                    """
                    SELECT doctrine_id FROM strategy_doctrines
                    WHERE stage_id = ? AND hypothesis_name = ? AND status IN ('candidate', 'validated')
                    LIMIT 1;
                    """,
                    (int(doc.stage_id), str(doc.hypothesis_name)[:96]),
                ).fetchone()
                if row is not None:
                    return int(row["doctrine_id"])
                did = self._insert_doctrine_conn(conn, doc)
                conn.commit()
                return did
            finally:
                conn.close()

    def record_episode_and_update_validation(
        self,
        ep_record: Dict[str, Any],
        applied_doctrine_id: int,
    ) -> Tuple[int, str]:
        """
        Persist immutable `match_episodes` fact log AND update empirical validation metrics on `applied_doctrine_id`.
        Promotes `candidate -> validated` or demotes `candidate -> deprecated` ONLY after `>= MIN_VALIDATION_TRIALS`
        matches so single-match luck never corrupts the validated knowledge base!
        """
        won_int = 1 if bool(ep_record.get("won", False)) else 0
        obj_prog = float(ep_record.get("final_obj_progress", 0.50))
        obj_delta = obj_prog - 0.50
        net_score = float(ep_record.get("net_score", 0.0))

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO match_episodes (
                        created_ts, stage_id, episode_id, outcome, won,
                        final_obj_progress, net_score, friendly_turf_peak, enemy_turf_peak,
                        respawns, vetoes, death_lanes_json, turning_point_json,
                        inner_speech_json, feature_vec8_json, applied_doctrine_id,
                        cause_tag, commentary_ja
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        time.time(),
                        int(ep_record.get("stage_id", 1)),
                        int(ep_record.get("episode_id", 0)),
                        str(ep_record.get("outcome", "DONE")),
                        won_int,
                        obj_prog,
                        net_score,
                        float(ep_record.get("friendly_turf_peak", 0.0)),
                        float(ep_record.get("enemy_turf_peak", 0.0)),
                        int(ep_record.get("respawns", 0)),
                        int(ep_record.get("vetoes", 0)),
                        json.dumps(ep_record.get("death_lanes", {}), ensure_ascii=False),
                        json.dumps(ep_record.get("turning_point", {}), ensure_ascii=False),
                        json.dumps(ep_record.get("inner_speech_events", []), ensure_ascii=False),
                        json.dumps([round(float(x), 4) for x in ep_record.get("feature_vec8", [0.0] * 8)]),
                        int(applied_doctrine_id),
                        str(ep_record.get("cause_tag", "CLEAN_PICK_TO_PUSH")),
                        str(ep_record.get("commentary_ja", ""))[:400],
                    ),
                )
                ep_uid = int(cur.lastrowid)

                new_status = "none"
                if applied_doctrine_id > 0:
                    d_row = conn.execute(
                        "SELECT status, source, trials_count, wins_count, sum_obj_delta, sum_score FROM strategy_doctrines WHERE doctrine_id = ?;",
                        (int(applied_doctrine_id),),
                    ).fetchone()
                    if d_row is not None:
                        old_status = str(d_row["status"])
                        src = str(d_row["source"])
                        trials = int(d_row["trials_count"]) + 1
                        wins = int(d_row["wins_count"]) + won_int
                        s_obj = float(d_row["sum_obj_delta"]) + obj_delta
                        s_scr = float(d_row["sum_score"]) + net_score
                        m_win = float(wins) / float(max(1, trials))
                        m_obj = s_obj / float(max(1, trials))

                        new_status = old_status
                        if src != "seed":
                            if old_status == "candidate" and trials >= MIN_VALIDATION_TRIALS:
                                if m_win >= 0.50 or (m_win >= 0.34 and m_obj >= 0.035):
                                    new_status = "validated"
                                elif m_win < 0.34 and m_obj < -0.015:
                                    new_status = "deprecated"
                            elif old_status == "validated" and trials >= (MIN_VALIDATION_TRIALS * 2):
                                if m_win < 0.28 and m_obj < -0.04:
                                    new_status = "deprecated"

                        conn.execute(
                            """
                            UPDATE strategy_doctrines
                            SET status = ?, trials_count = ?, wins_count = ?,
                                sum_obj_delta = ?, sum_score = ?,
                                mean_win_rate = ?, mean_obj_delta = ?
                            WHERE doctrine_id = ?;
                            """,
                            (new_status, trials, wins, s_obj, s_scr, m_win, m_obj, int(applied_doctrine_id)),
                        )
                conn.commit()
                return ep_uid, new_status
            finally:
                conn.close()

    def attach_episode_commentary(self, episode_uid: int, commentary_ja: str) -> None:
        if episode_uid <= 0 or not commentary_ja:
            return
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE match_episodes SET commentary_ja = ? WHERE episode_uid = ?;",
                    (str(commentary_ja)[:400], int(episode_uid)),
                )
                conn.commit()
            finally:
                conn.close()

    def _row_to_doctrine(self, row: sqlite3.Row) -> StrategicDoctrine:
        try:
            vec_raw = json.loads(row["context_vec8_json"] or "[]")
            v8 = np.asarray(vec_raw[:8] + [0.0] * max(0, 8 - len(vec_raw)), dtype=np.float32)
        except Exception:
            v8 = np.zeros(8, dtype=np.float32)
        return StrategicDoctrine(
            doctrine_id=int(row["doctrine_id"]),
            stage_id=int(row["stage_id"]),
            source=str(row["source"]),
            status=str(row["status"]),
            target_cause_tag=str(row["target_cause_tag"]),
            hypothesis_name=str(row["hypothesis_name"]),
            summary_ja=str(row["summary_ja"]),
            regimes=StrategicDoctrine.regimes_from_json(row["regime_policy_json"]),
            context_vec8=v8,
            trials_count=int(row["trials_count"]),
            wins_count=int(row["wins_count"]),
            mean_win_rate=float(row["mean_win_rate"]),
            mean_obj_delta=float(row["mean_obj_delta"]),
        )

    def search_doctrines(
        self,
        stage_id: int,
        context_vec8: Optional[np.ndarray] = None,
        target_cause_tag: str = "",
        text_query: str = "",
        top_k: int = 3,
        include_candidates: bool = True,
    ) -> List[Tuple[float, StrategicDoctrine]]:
        """
        Dedicated Hybrid Search Engine combining:
          1. Stage specificity (`stage_id` exact match OR universal `0`)
          2. Empirical reliability (`validated` > `seed` > `candidate`; `deprecated` strictly excluded!)
          3. Cause-tag alignment (`target_cause_tag`)
          4. 8D Match-Context Cosine Similarity (`context_vec8`)
          5. Optional `FTS5` lexical match boost (`text_query`)
        """
        statuses = ("validated", "seed", "candidate") if include_candidates else ("validated", "seed")
        fts_hits: Dict[int, float] = {}

        with self._lock:
            conn = self._connect()
            try:
                if text_query.strip():
                    q_clean = "".join(ch if (ch.isalnum() or ord(ch) > 127) else " " for ch in text_query).strip()
                    tokens = [t for t in q_clean.split() if len(t) >= 2][:6]
                    if tokens and self._fts_enabled:
                        fts_expr = " OR ".join(f'"{t}"' for t in tokens)
                        try:
                            for f_row in conn.execute(
                                "SELECT doctrine_id, bm25(strategy_doctrines_fts) AS rank_s FROM strategy_doctrines_fts WHERE strategy_doctrines_fts MATCH ? LIMIT 16;",
                                (fts_expr,),
                            ):
                                fts_hits[int(f_row["doctrine_id"])] = float(np.clip(1.0 / (1.0 + abs(float(f_row["rank_s"]))), 0.1, 1.0))
                        except sqlite3.OperationalError:
                            pass
                    if not fts_hits and tokens:
                        for tok in tokens[:3]:
                            for l_row in conn.execute(
                                "SELECT doctrine_id FROM strategy_doctrines WHERE summary_ja LIKE ? OR hypothesis_name LIKE ? LIMIT 12;",
                                (f"%{tok}%", f"%{tok}%"),
                            ):
                                did_l = int(l_row["doctrine_id"])
                                fts_hits[did_l] = max(fts_hits.get(did_l, 0.0), 0.55)

                placeholders = ",".join("?" for _ in statuses)
                rows = conn.execute(
                    f"""
                    SELECT * FROM strategy_doctrines
                    WHERE stage_id IN (?, 0) AND status IN ({placeholders})
                    ORDER BY trials_count DESC, doctrine_id DESC
                    LIMIT 64;
                    """,
                    (int(stage_id), *statuses),
                ).fetchall()
            finally:
                conn.close()

        q_vec = np.asarray(context_vec8, dtype=np.float32) if context_vec8 is not None else None
        scored: List[Tuple[float, StrategicDoctrine]] = []
        for r in rows:
            doc = self._row_to_doctrine(r)
            score = 0.0
            # Status reliability tier
            if doc.status == "validated":
                score += 1.35 + 0.65 * (doc.mean_win_rate - 0.50) + 0.50 * doc.mean_obj_delta
            elif doc.status == "seed":
                score += 1.00 + 0.25 * (doc.mean_win_rate - 0.50)
            elif doc.status == "candidate":
                score += 0.92 + 0.20 * (doc.mean_win_rate - 0.50)

            # Stage match bonus
            if doc.stage_id == int(stage_id) and int(stage_id) > 0:
                score += 0.22

            # Cause-tag counter alignment
            if target_cause_tag and doc.target_cause_tag == target_cause_tag:
                score += 0.58

            # 8D situation vector cosine similarity
            if q_vec is not None:
                score += 0.45 * _cosine_sim8(q_vec, doc.context_vec8)

            # FTS5 / keyword match boost
            if doc.doctrine_id in fts_hits:
                score += 0.55 * fts_hits[doc.doctrine_id]

            scored.append((float(score), doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[: max(1, int(top_k))]

    def select_active_doctrine_for_match(
        self,
        stage_id: int,
        context_vec8: Optional[np.ndarray],
        last_cause_tag: str,
        rng: np.random.Generator,
        candidate_explore_prob: float = 0.28,
    ) -> StrategicDoctrine:
        """
        Controlled Exploration vs. Exploitation selector (addresses Ontology Memo Section B.2/B.3):
          - With probability `candidate_explore_prob`, deliberately assigns an under-tested `candidate`
            (`trials_count < MIN_VALIDATION_TRIALS`) matching the stage/cause so new Gemma hypotheses
            actually accumulate fair validation trials!
          - Otherwise exploits the highest-scoring `validated` or `seed` doctrine.
        """
        ranked = self.search_doctrines(
            stage_id=stage_id,
            context_vec8=context_vec8,
            target_cause_tag=last_cause_tag,
            top_k=8,
            include_candidates=True,
        )
        if not ranked:
            return StrategicDoctrine(stage_id=stage_id)

        candidates = [d for _, d in ranked if d.status == "candidate" and d.trials_count < MIN_VALIDATION_TRIALS]
        reliable = [d for _, d in ranked if d.status in ("validated", "seed")]

        if candidates and float(rng.random()) < float(np.clip(candidate_explore_prob, 0.0, 0.75)):
            return candidates[0]
        if reliable:
            return reliable[0]
        return ranked[0][1]

    def get_deprecated_antipatterns(self, stage_id: int, limit: int = 3) -> List[StrategicDoctrine]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM strategy_doctrines
                    WHERE stage_id IN (?, 0) AND status = 'deprecated'
                    ORDER BY created_ts DESC
                    LIMIT ?;
                    """,
                    (int(stage_id), int(max(1, limit))),
                ).fetchall()
            finally:
                conn.close()
        return [self._row_to_doctrine(r) for r in rows]

    def get_stats_summary(self) -> Dict[str, int]:
        with self._lock:
            conn = self._connect()
            try:
                ep_cnt = int(conn.execute("SELECT COUNT(*) AS c FROM match_episodes;").fetchone()["c"])
                by_st: Dict[str, int] = {"episodes": ep_cnt, "seed": 0, "candidate": 0, "validated": 0, "deprecated": 0}
                for r in conn.execute("SELECT status, COUNT(*) AS c FROM strategy_doctrines GROUP BY status;"):
                    by_st[str(r["status"])] = int(r["c"])
                return by_st
            finally:
                conn.close()


# ==============================================================================
# 4. Match Flight Recorder & Turning-Point Winner-vs-Loser Differential Analyzer
# ==============================================================================
class MatchFlightRecorder:
    """
    Compresses a 60Hz 3D match into:
      1. Macro 0.5s trajectory snapshots (win-rate curve, turf balance, lane occupancy, carrier state)
      2. Exact death/respawn coordinates (`death_lanes`)
      3. Salient cognitive/inner-speech events (`GWT ignition`, `Survival Veto`, `Respawn`, `Regime Shift`)
      4. **Turning-Point Swing Differential**: isolates the 12-second window where match advantage swung
         most sharply so Gemma analyzes *what broke the equilibrium* rather than survivorship-biased averages!
    """

    def __init__(self, sample_stride_ticks: int = 30):
        self.sample_stride_ticks = max(10, int(sample_stride_ticks))
        self.reset(episode_id=0, stage_id=1)

    def reset(self, episode_id: int, stage_id: int = 1) -> None:
        self.episode_id = int(episode_id)
        self.stage_id = int(stage_id) if int(stage_id) > 0 else 1
        self.snapshots: List[Dict[str, Any]] = []
        self.inner_speech_events: List[Dict[str, Any]] = []
        self.death_lanes: Dict[str, int] = {"LEFT_WING": 0, "CENTER_AXIS": 0, "RIGHT_WING": 0}
        self.lane_ticks: Dict[str, int] = {"LEFT_WING": 0, "CENTER_AXIS": 0, "RIGHT_WING": 0}
        self.last_alive_pos_xz = np.zeros(2, dtype=np.float32)
        self.last_sample_tick = -9999
        self.last_speech_logged = ""
        self.peak_friendly_turf = 0.0
        self.peak_enemy_turf = 0.0
        self.peak_obj_progress = 0.50
        self.min_obj_progress = 0.50
        self.respawns_count = 0
        self.vetoes_count = 0
        self.enemy_carrier_ticks = 0
        self.friendly_carrier_ticks = 0
        self.total_samples = 0

    @staticmethod
    def estimate_instant_advantage01(
        obj_prog: float,
        friendly_turf: float,
        enemy_turf: float,
        carrying_self: bool,
        enemy_carrier_vis: bool,
        in_respawn: bool,
    ) -> float:
        logit = (
            4.2 * (float(obj_prog) - 0.50)
            + 1.6 * (float(friendly_turf) - float(enemy_turf))
            + (0.55 if carrying_self else 0.0)
            - (0.65 if enemy_carrier_vis else 0.0)
            - (0.45 if in_respawn else 0.0)
        )
        return float(1.0 / (1.0 + math.exp(-float(np.clip(logit, -8.0, 8.0)))))

    def record_step(
        self,
        tick: int,
        pos_xz: np.ndarray,
        proprio: np.ndarray,
        friendly_turf: float,
        enemy_turf: float,
        enemy_carrier_visible: bool,
        respawn_ev: bool,
        veto_active: bool,
        ignition: float = 0.0,
        active_regime: int = REGIME_OPENING_CONTEST,
        regime_switched: bool = False,
        inner_speech: str = "",
    ) -> None:
        px, pz = float(pos_xz[0]), float(pos_xz[1])
        obj_p = float(proprio[PROPRIO_IDX_OBJ_PROGRESS]) if proprio.shape[0] > PROPRIO_IDX_OBJ_PROGRESS else 0.50
        carrying = is_carrying_primary_objective(proprio)
        hp = float(np.clip(1.0 - (float(proprio[PROPRIO_IDX_HAZARD]) if proprio.shape[0] > PROPRIO_IDX_HAZARD else 0.0), 0.0, 1.0))

        self.peak_friendly_turf = max(self.peak_friendly_turf, float(friendly_turf))
        self.peak_enemy_turf = max(self.peak_enemy_turf, float(enemy_turf))
        self.peak_obj_progress = max(self.peak_obj_progress, obj_p)
        self.min_obj_progress = min(self.min_obj_progress, obj_p)

        if respawn_ev:
            self.respawns_count += 1
            d_lane = LANE_NAMES[_lane_from_x(float(self.last_alive_pos_xz[0]))]
            self.death_lanes[d_lane] = self.death_lanes.get(d_lane, 0) + 1
            self._push_cog_event(tick, "RESPAWN_DEATH", f"被撃破({d_lane})→復帰再編", obj_p, friendly_turf, inner_speech)
        else:
            self.last_alive_pos_xz[0] = px
            self.last_alive_pos_xz[1] = pz

        if veto_active:
            self.vetoes_count += 1

        if regime_switched:
            self._push_cog_event(
                tick,
                "REGIME_SHIFT",
                f"戦略局面→{REGIME_NAMES.get(active_regime, 'OPENING')}",
                obj_p,
                friendly_turf,
                inner_speech,
            )

        if (tick - self.last_sample_tick) >= self.sample_stride_ticks or respawn_ev:
            self.last_sample_tick = int(tick)
            self.total_samples += 1
            l_name = LANE_NAMES[_lane_from_x(px)]
            self.lane_ticks[l_name] = self.lane_ticks.get(l_name, 0) + 1
            if carrying:
                self.friendly_carrier_ticks += 1
            if enemy_carrier_visible:
                self.enemy_carrier_ticks += 1

            adv01 = self.estimate_instant_advantage01(
                obj_p, friendly_turf, enemy_turf, carrying, enemy_carrier_visible, respawn_ev
            )
            if len(self.snapshots) < 240:
                self.snapshots.append({
                    "tick": int(tick),
                    "x": round(px, 2),
                    "z": round(pz, 2),
                    "lane": l_name,
                    "obj": round(obj_p, 3),
                    "f_turf": round(float(friendly_turf), 3),
                    "e_turf": round(float(enemy_turf), 3),
                    "carry": 1 if carrying else (-1 if enemy_carrier_visible else 0),
                    "hp": round(hp, 2),
                    "adv": round(adv01, 3),
                    "regime": REGIME_NAMES.get(active_regime, "OPENING_CONTEST"),
                })

            # Salient inner-speech / GWT ignition capture (deduplicated)
            clean_sp = (inner_speech or "").strip()
            if (ignition >= 0.52 or veto_active or carrying) and clean_sp and clean_sp != self.last_speech_logged:
                self.last_speech_logged = clean_sp
                ev_t = "SURVIVAL_VETO" if veto_active else ("CARRIER_PUSH" if carrying else "GWT_IGNITION")
                self._push_cog_event(tick, ev_t, f"Pos({px:+.1f},{pz:+.1f})", obj_p, friendly_turf, clean_sp)

    def _push_cog_event(
        self,
        tick: int,
        ev_type: str,
        detail: str,
        obj_p: float,
        f_turf: float,
        speech: str,
    ) -> None:
        if len(self.inner_speech_events) >= 14:
            self.inner_speech_events.pop(0)
        self.inner_speech_events.append({
            "tick": int(tick),
            "type": str(ev_type),
            "detail": str(detail),
            "obj": round(float(obj_p), 2),
            "turf": round(float(f_turf), 2),
            "speech": str(speech)[:64],
        })

    def finalize_episode(self, final_obj_progress: float, net_score: float, won: bool, ko_flag: bool = False) -> Dict[str, Any]:
        outcome = ("WIN_KO" if ko_flag else "WIN_TIME") if won else (("LOSS_KO" if ko_flag else "LOSS_TIME") if net_score < -0.5 or final_obj_progress < 0.49 else "DRAW")

        # 1. Turning-Point Window Extraction (12s = ~24 snapshots)
        tp_info: Dict[str, Any] = {
            "start_tick": 0,
            "end_tick": 0,
            "adv_before": 0.50,
            "adv_after": round(float(final_obj_progress), 3),
            "swing_delta": 0.0,
            "dominant_lane": "CENTER_AXIS",
            "turf_gap_at_swing": 0.0,
            "winner_vs_loser_diff": "",
        }
        if len(self.snapshots) >= 4:
            win_len = min(24, max(2, len(self.snapshots) // 3))
            best_abs = -1.0
            best_i = 0
            for i in range(0, len(self.snapshots) - win_len + 1):
                j = i + win_len - 1
                delta = float(self.snapshots[j]["adv"]) - float(self.snapshots[i]["adv"])
                if abs(delta) > best_abs:
                    best_abs = abs(delta)
                    best_i = i
            j_end = min(len(self.snapshots) - 1, best_i + win_len - 1)
            seg = self.snapshots[best_i : j_end + 1]
            s0, s1 = seg[0], seg[-1]
            swing = float(s1["adv"]) - float(s0["adv"])
            mean_ft = float(np.mean([x["f_turf"] for x in seg]))
            mean_et = float(np.mean([x["e_turf"] for x in seg]))
            lanes_seg = [str(x["lane"]) for x in seg]
            dom_lane = max(("LEFT_WING", "CENTER_AXIS", "RIGHT_WING"), key=lambda k: lanes_seg.count(k))
            turf_gap = mean_ft - mean_et
            if swing >= 0.0:
                diff_txt = f"勝勢転換: {dom_lane}経由で味方塗優位({turf_gap:+.2f})を確保し進行度{s0['obj']:.2f}→{s1['obj']:.2f}へ前進"
            else:
                diff_txt = f"劣勢転換: {dom_lane}で塗劣勢({turf_gap:+.2f})/被弾が重なり進行度{s0['obj']:.2f}→{s1['obj']:.2f}へ後退"
            tp_info = {
                "start_tick": int(s0["tick"]),
                "end_tick": int(s1["tick"]),
                "adv_before": round(float(s0["adv"]), 3),
                "adv_after": round(float(s1["adv"]), 3),
                "swing_delta": round(swing, 3),
                "dominant_lane": dom_lane,
                "turf_gap_at_swing": round(turf_gap, 3),
                "winner_vs_loser_diff": diff_txt,
            }

        # 2. Deterministic Fixed-Yardstick Cause Classification (`CAUSE_TAGS`)
        total_deaths = max(1, sum(self.death_lanes.values()))
        center_death_share = float(self.death_lanes.get("CENTER_AXIS", 0)) / float(total_deaths)
        total_lane_t = max(1, sum(self.lane_ticks.values()))
        wing_share = float(self.lane_ticks.get("LEFT_WING", 0) + self.lane_ticks.get("RIGHT_WING", 0)) / float(total_lane_t)
        avg_ft = float(np.mean([s["f_turf"] for s in self.snapshots])) if self.snapshots else self.peak_friendly_turf
        stall_ratio = 1.0 if (self.peak_obj_progress < 0.55 and self.min_obj_progress > 0.45) else 0.0
        enemy_crisis_ratio = float(self.enemy_carrier_ticks) / float(max(1, self.total_samples))

        if won:
            if wing_share >= 0.42:
                cause_tag = "WING_HIGHGROUND_CONTROL"
            elif float(tp_info["adv_before"]) < 0.42:
                cause_tag = "CLUTCH_FLANK_BREAKTHROUGH"
            else:
                cause_tag = "CLEAN_PICK_TO_PUSH"
        else:
            if self.peak_obj_progress >= 0.66 and final_obj_progress < 0.50:
                cause_tag = "LEAD_PROTECTION_THROW"
            elif self.respawns_count >= 2 and center_death_share >= 0.55:
                cause_tag = "CENTRAL_OVEREXTEND_WIPE"
            elif avg_ft < 0.24 and self.respawns_count >= 1:
                cause_tag = "UNPAINTED_NO_REFUGE"
            elif enemy_crisis_ratio >= 0.22 or final_obj_progress <= 0.32:
                cause_tag = "ENEMY_CARRIER_UNCONTESTED"
            else:
                cause_tag = "PASSIVE_TURF_STALL"

        vec8 = [
            0.0 if won else 1.0,
            float(np.clip(center_death_share if self.respawns_count > 0 else 0.0, 0.0, 1.0)),
            1.0 if avg_ft < 0.28 else 0.0,
            float(np.clip(enemy_crisis_ratio * 2.0, 0.0, 1.0)),
            float(stall_ratio),
            float(np.clip(avg_ft, 0.0, 1.0)),
            float(np.clip(float(self.respawns_count) / 5.0, 0.0, 1.0)),
            float(np.clip(tp_info["swing_delta"], -1.0, 1.0)),
        ]

        return {
            "stage_id": self.stage_id,
            "episode_id": self.episode_id,
            "outcome": outcome,
            "won": bool(won),
            "final_obj_progress": round(float(final_obj_progress), 4),
            "net_score": round(float(net_score), 3),
            "friendly_turf_peak": round(self.peak_friendly_turf, 3),
            "enemy_turf_peak": round(self.peak_enemy_turf, 3),
            "avg_friendly_turf": round(avg_ft, 3),
            "respawns": int(self.respawns_count),
            "vetoes": int(self.vetoes_count),
            "death_lanes": dict(self.death_lanes),
            "lane_share": {k: round(float(v) / float(total_lane_t), 2) for k, v in self.lane_ticks.items()},
            "turning_point": tp_info,
            "inner_speech_events": list(self.inner_speech_events[-10:]),
            "feature_vec8": vec8,
            "cause_tag": cause_tag,
        }


# ==============================================================================
# 5. Deterministic Statistical Counter-Doctrine & Ollama Gemma 4 Synthesis
# ==============================================================================
def synthesize_statistical_counter_doctrine(ep_rec: Dict[str, Any]) -> Tuple[StrategicDoctrine, str]:
    """
    Fast, deterministic fallback synthesis from Turning-Point Winner-vs-Loser differentials when
    Ollama is offline or throttled during 4x parallel PBT training.
    """
    stage_id = int(ep_rec.get("stage_id", 1))
    cause = str(ep_rec.get("cause_tag", "CLEAN_PICK_TO_PUSH"))
    won = bool(ep_rec.get("won", False))
    tp = ep_rec.get("turning_point") or {}
    d_lanes = ep_rec.get("death_lanes") or {}
    left_d = int(d_lanes.get("LEFT_WING", 0))
    right_d = int(d_lanes.get("RIGHT_WING", 0))
    safer_wing = LANE_LEFT_WING if left_d <= right_d else LANE_RIGHT_WING
    safer_wing_ja = "左の高台ルート" if safer_wing == LANE_LEFT_WING else "右の高台ルート"

    if cause == "CENTRAL_OVEREXTEND_WIPE":
        name = f"Counter_CentralWipe_Via_{LANE_NAMES[safer_wing]}"
        comm = f"さっきの試合は中央正面で{ep_rec.get('respawns', 0)}回やられてしまいましたね。次は{safer_wing_ja}から塗りを広げて、安全に射線を分散させましょうね。"
        regimes = {
            REGIME_OPENING_CONTEST: RegimePolicySlice(safer_wing, 1.20, 0.94, 1.16, +0.02, GOAL_MODE_STEALTH_FLANK),
            REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_DUAL_WINGS, 1.12, 0.86, 1.22, +0.04, GOAL_MODE_STEALTH_FLANK),
            REGIME_DEFENSE_CRISIS: RegimePolicySlice(safer_wing, 0.82, 1.30, 1.05, +0.02, GOAL_MODE_ENGAGE_INTERCEPT),
            REGIME_COMEBACK_CLUTCH: RegimePolicySlice(LANE_DUAL_WINGS, 0.96, 1.32, 1.12, -0.01, GOAL_MODE_PUSH_OBJECTIVE),
        }
    elif cause == "UNPAINTED_NO_REFUGE":
        name = "Counter_TurfStarvation_ExpandFirst"
        comm = "さっきは味方のインクが少なくて逃げ場が足りなくなりましたね。開幕と復帰直後は足場の塗りをしっかり広げてから、コアへ向かいましょうね。"
        regimes = {
            REGIME_OPENING_CONTEST: RegimePolicySlice(LANE_BALANCED, 1.30, 0.90, 1.15, +0.02, GOAL_MODE_BALANCED_EXPLORE),
            REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_DUAL_WINGS, 1.20, 0.86, 1.20, +0.04, GOAL_MODE_BALANCED_EXPLORE),
            REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.88, 1.28, 1.00, +0.02, GOAL_MODE_ENGAGE_INTERCEPT),
            REGIME_COMEBACK_CLUTCH: RegimePolicySlice(safer_wing, 1.05, 1.28, 1.10, 0.00, GOAL_MODE_PUSH_OBJECTIVE),
        }
    elif cause == "ENEMY_CARRIER_UNCONTESTED":
        name = "Counter_CarrierCrisis_RapidIntercept"
        comm = "相手のコア持ちへの迎撃が少し遅れてしまいましたね。敵がコアを持った時はすぐ塗りを切り上げて、中央と高台から迎撃しましょうね。"
        regimes = {
            REGIME_OPENING_CONTEST: RegimePolicySlice(LANE_CENTER_AXIS, 1.02, 1.18, 1.00, +0.01, GOAL_MODE_ENGAGE_INTERCEPT),
            REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_BALANCED, 1.06, 0.94, 1.16, +0.03, GOAL_MODE_STEALTH_FLANK),
            REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.76, 1.38, 0.92, +0.02, GOAL_MODE_ENGAGE_INTERCEPT),
            REGIME_COMEBACK_CLUTCH: RegimePolicySlice(safer_wing, 0.88, 1.34, 1.06, -0.02, GOAL_MODE_PUSH_OBJECTIVE),
        }
    elif cause == "LEAD_PROTECTION_THROW":
        name = "Counter_LeadThrow_DisciplineLock"
        comm = "せっかくリードできたのに、深追いして逆転されてしまいましたね。リードした後は無理をせず、潜伏と塗り維持で丁寧に守り切りましょうね。"
        regimes = {
            REGIME_OPENING_CONTEST: RegimePolicySlice(LANE_BALANCED, 1.08, 1.06, 1.04, +0.01, GOAL_MODE_BALANCED_EXPLORE),
            REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_DUAL_WINGS, 1.15, 0.80, 1.30, +0.05, GOAL_MODE_STEALTH_FLANK),
            REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.80, 1.32, 1.00, +0.03, GOAL_MODE_ENGAGE_INTERCEPT),
            REGIME_COMEBACK_CLUTCH: RegimePolicySlice(safer_wing, 0.92, 1.30, 1.08, -0.01, GOAL_MODE_PUSH_OBJECTIVE),
        }
    elif cause == "PASSIVE_TURF_STALL":
        name = "Counter_PassiveStall_ObjectiveTempo"
        comm = "塗りは悪くなかったのですが、コアの前進が少し停滞しましたね。次はチャンスを見たら積極的にコアを持って、カウントを進めてみましょうね。"
        regimes = {
            REGIME_OPENING_CONTEST: RegimePolicySlice(LANE_CENTER_AXIS, 0.90, 1.28, 0.98, -0.01, GOAL_MODE_PUSH_OBJECTIVE),
            REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_BALANCED, 1.04, 0.96, 1.14, +0.03, GOAL_MODE_BALANCED_EXPLORE),
            REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.78, 1.35, 0.95, +0.01, GOAL_MODE_ENGAGE_INTERCEPT),
            REGIME_COMEBACK_CLUTCH: RegimePolicySlice(safer_wing, 0.82, 1.38, 1.04, -0.03, GOAL_MODE_PUSH_OBJECTIVE),
        }
    else:
        dom_lane_str = str(tp.get("dominant_lane", "BALANCED"))
        win_lane = NAME_TO_LANE.get(dom_lane_str, LANE_BALANCED)
        win_lane_ja = LANE_SPOKEN_JA.get(win_lane, "中央ルート")
        name = f"Reinforce_{cause}_{LANE_NAMES[win_lane]}"
        comm = (
            f"さっきの試合は{win_lane_ja}から綺麗に塗りの主導権を握って勝てましたね。"
            f"次の試合もこの良い流れと、リード後の丁寧なリスク管理を続けていきましょうね。"
        )
        regimes = {
            REGIME_OPENING_CONTEST: RegimePolicySlice(win_lane, 1.12, 1.10, 1.05, 0.00, GOAL_MODE_BALANCED_EXPLORE if win_lane == LANE_BALANCED else GOAL_MODE_STEALTH_FLANK),
            REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_DUAL_WINGS, 1.10, 0.88, 1.20, +0.03, GOAL_MODE_STEALTH_FLANK),
            REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.80, 1.32, 0.98, +0.02, GOAL_MODE_ENGAGE_INTERCEPT),
            REGIME_COMEBACK_CLUTCH: RegimePolicySlice(safer_wing, 0.90, 1.32, 1.08, -0.02, GOAL_MODE_PUSH_OBJECTIVE),
        }

    doc = StrategicDoctrine(
        stage_id=stage_id,
        source="stat_counter",
        status="candidate",
        target_cause_tag=cause if not won else cause,
        hypothesis_name=name,
        summary_ja=comm,
        regimes=regimes,
        context_vec8=np.asarray(ep_rec.get("feature_vec8", [0.0] * 8), dtype=np.float32),
    )
    return doc, comm


def build_gemma_strategy_prompt(
    ep_rec: Dict[str, Any],
    validated_refs: List[StrategicDoctrine],
    deprecated_refs: List[StrategicDoctrine],
) -> str:
    tp = ep_rec.get("turning_point") or {}
    lines = [
        f"【試合結果】Stage={ep_rec.get('stage_id', 1)} Outcome={ep_rec.get('outcome')} "
        f"最終コア進行={float(ep_rec.get('final_obj_progress', 0.5))*100:.1f}% "
        f"最大味方塗={float(ep_rec.get('friendly_turf_peak', 0.0))*100:.1f}% "
        f"被撃破={ep_rec.get('respawns', 0)} (死因レーン={ep_rec.get('death_lanes')}) "
        f"生存Veto={ep_rec.get('vetoes', 0)} 一次要因={ep_rec.get('cause_tag')}",
        f"【勝敗分岐点(Turning Point)】T{tp.get('start_tick', 0)}〜T{tp.get('end_tick', 0)}: "
        f"{tp.get('winner_vs_loser_diff', 'N/A')} (優位度変化={tp.get('swing_delta', 0.0):+.2f})",
    ]
    evs = ep_rec.get("inner_speech_events") or []
    if evs:
        ev_strs = [f"T{e['tick']}[{e['type']}]{e['speech']}" for e in evs[-6:] if e.get("speech")]
        if ev_strs:
            lines.append("【主要認知・内言推移】" + " / ".join(ev_strs))
    if validated_refs:
        lines.append("【DB内の検証済有効戦略】" + " ; ".join(
            f"{d.hypothesis_name}(勝率{d.mean_win_rate*100:.0f}%):{d.summary_ja[:55]}" for d in validated_refs[:2]
        ))
    if deprecated_refs:
        lines.append("【過去失効した悪手(再提案禁止)】" + " ; ".join(
            f"{d.hypothesis_name}:{d.summary_ja[:45]}" for d in deprecated_refs[:2]
        ))
    lines.append(
        "【指示】局所戦術(目の前の撃ち合い操作)ではなく、試合全体と局面別(開幕/優勢固め/危機防衛/劣勢打開)の『大局戦略』改善案と、"
        "知的で優しい女の子の戦術ナビゲーター口調（〜ですね、〜しましょうね、〜してみますね 等、英字定数や記号を含めない自然な話し言葉）での"
        "2文以内(85文字以内)の振り返り実況(commentary_ja)を指定JSONスキーマのみで出力せよ。"
        "倍率パラメータは [0.78〜1.35]、lead_caution_delta は [-0.03〜+0.05] の範囲とすること。"
    )
    return "\n".join(lines)


def synthesize_human_coaching_fallback(user_text: str, stage_id: int = 1) -> Tuple[StrategicDoctrine, str]:
    """Rule-grounded fallback translator when a human sends coaching input while Ollama is offline."""
    txt = (user_text or "").strip()
    if "左" in txt:
        open_lane = LANE_LEFT_WING
    elif "右" in txt:
        open_lane = LANE_RIGHT_WING
    elif any(k in txt for k in ("両翼", "側面", "高台", "裏", "回り込")):
        open_lane = LANE_DUAL_WINGS
    elif any(k in txt for k in ("中央", "正面", "真ん中")):
        open_lane = LANE_CENTER_AXIS
    else:
        open_lane = LANE_BALANCED

    want_paint = any(k in txt for k in ("塗", "足場", "広げ"))
    want_caut = any(k in txt for k in ("慎重", "守", "防衛", "生存", "無理せず", "潜伏"))
    want_push = any(k in txt for k in ("攻", "コア", "目標", "進", "打開", "積極", "前に"))

    cur_m = 1.26 if want_paint else 1.06
    obj_m = 1.30 if want_push else (0.92 if want_caut else 1.08)
    caut_d = +0.04 if want_caut else (-0.01 if want_push else 0.01)
    stl_m = 1.24 if want_caut else 1.08
    lane_ja = LANE_SPOKEN_JA.get(open_lane, "ステージ全域")
    posture_ja = "塗りと安全重視で丁寧に" if (want_paint or want_caut) else ("コア前進を積極意識して" if want_push else "バランス良く")
    reply = f"アドバイスありがとうございます。ここからは{lane_ja}を軸に、{posture_ja}立ち回ってみますね。"

    open_mode = GOAL_MODE_STEALTH_FLANK if open_lane in (LANE_LEFT_WING, LANE_RIGHT_WING, LANE_DUAL_WINGS) else (
        GOAL_MODE_PUSH_OBJECTIVE if want_push else GOAL_MODE_BALANCED_EXPLORE
    )
    regimes = {
        REGIME_OPENING_CONTEST: RegimePolicySlice(open_lane, cur_m, obj_m, stl_m, caut_d, open_mode).clamped(),
        REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_DUAL_WINGS, 1.12, 0.86, max(1.18, stl_m), max(0.03, caut_d), GOAL_MODE_STEALTH_FLANK).clamped(),
        REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.80, 1.34, 1.00, +0.02, GOAL_MODE_ENGAGE_INTERCEPT).clamped(),
        REGIME_COMEBACK_CLUTCH: RegimePolicySlice(open_lane if open_lane != LANE_BALANCED else LANE_LEFT_WING, 0.92, max(1.28, obj_m), 1.10, -0.01, GOAL_MODE_PUSH_OBJECTIVE).clamped(),
    }
    doc = StrategicDoctrine(
        stage_id=int(stage_id),
        source="human_coach",
        status="candidate",
        target_cause_tag="WING_HIGHGROUND_CONTROL" if open_lane != LANE_CENTER_AXIS else "CLEAN_PICK_TO_PUSH",
        hypothesis_name="HumanCoach_CustomPlan",
        summary_ja=reply,
        regimes=regimes,
    )
    return doc, reply


def parse_gemma_json_to_doctrine(
    parsed: Dict[str, Any],
    ep_rec: Dict[str, Any],
    fallback_doc: StrategicDoctrine,
) -> Tuple[StrategicDoctrine, str]:
    cause = str(parsed.get("cause_tag", ep_rec.get("cause_tag", "CLEAN_PICK_TO_PUSH"))).strip()
    if cause not in CAUSE_TAGS:
        cause = str(ep_rec.get("cause_tag", "CLEAN_PICK_TO_PUSH"))
    hyp_name = str(parsed.get("hypothesis_name", fallback_doc.hypothesis_name)).strip() or fallback_doc.hypothesis_name
    comm = str(parsed.get("commentary_ja", fallback_doc.summary_ja)).strip() or fallback_doc.summary_ja

    open_lane = NAME_TO_LANE.get(str(parsed.get("opening_lane", "BALANCED")).upper().strip(), LANE_BALANCED)
    come_lane = NAME_TO_LANE.get(str(parsed.get("comeback_lane", "LEFT_WING")).upper().strip(), LANE_LEFT_WING)
    open_cur = float(np.clip(float(parsed.get("opening_curiosity_mult", 1.08)), 0.75, 1.35))
    open_obj = float(np.clip(float(parsed.get("opening_objective_mult", 1.05)), 0.75, 1.40))
    lead_caut = float(np.clip(float(parsed.get("lead_caution_delta", 0.03)), -0.04, 0.06))
    lead_stl = float(np.clip(float(parsed.get("lead_stealth_mult", 1.18)), 0.80, 1.35))
    cris_obj = float(np.clip(float(parsed.get("crisis_objective_mult", 1.32)), 1.05, 1.40))
    come_obj = float(np.clip(float(parsed.get("comeback_objective_mult", 1.32)), 1.00, 1.40))

    open_mode = GOAL_MODE_STEALTH_FLANK if open_lane in (LANE_LEFT_WING, LANE_RIGHT_WING, LANE_DUAL_WINGS) else (
        GOAL_MODE_PUSH_OBJECTIVE if open_obj >= 1.18 else GOAL_MODE_BALANCED_EXPLORE
    )
    regimes = {
        REGIME_OPENING_CONTEST: RegimePolicySlice(open_lane, open_cur, open_obj, 1.05, 0.00, open_mode).clamped(),
        REGIME_ADVANTAGE_LOCK: RegimePolicySlice(LANE_DUAL_WINGS, 1.10, 0.86, lead_stl, lead_caut, GOAL_MODE_STEALTH_FLANK).clamped(),
        REGIME_DEFENSE_CRISIS: RegimePolicySlice(LANE_CENTER_AXIS, 0.80, cris_obj, 0.96, +0.02, GOAL_MODE_ENGAGE_INTERCEPT).clamped(),
        REGIME_COMEBACK_CLUTCH: RegimePolicySlice(come_lane, 0.90, come_obj, 1.10, -0.02, GOAL_MODE_PUSH_OBJECTIVE).clamped(),
    }
    doc = StrategicDoctrine(
        stage_id=int(ep_rec.get("stage_id", 1)),
        source="gemma4",
        status="candidate",
        target_cause_tag=cause,
        hypothesis_name=f"Gemma4_{hyp_name[:64]}",
        summary_ja=comm,
        regimes=regimes,
        context_vec8=np.asarray(ep_rec.get("feature_vec8", [0.0] * 8), dtype=np.float32),
    )
    return doc, comm


class AsyncStrategyFeedbackWorker:
    """
    Background Zero-PyTorch (`Barrier 1`) post-match analyst thread:
      - Never blocks the 60Hz Unity TCP loop!
      - Immediately writes numeric episode results & updates doctrine validation status in SQLite,
        then queries `Ollama gemma4:12b` (`think: false`, JSON Schema, `num_ctx=2048`) if online & unthrottled.
    """

    def __init__(
        self,
        db: StrategyKnowledgeDB,
        ollama_url: str = "http://127.0.0.1:11434",
        preferred_model: str = "gemma4:12b",
        ollama_compute_mode: str = "hybrid",
        enable_ollama: bool = True,
        min_ollama_interval_s: float = 6.0,
        emit_ui_stat: bool = False,
    ):
        assert_zero_torch_in_strategy_engine()
        from companion_voice_io import OllamaMatchAnalystClient

        self.db = db
        self.enable_ollama = bool(enable_ollama)
        self.min_ollama_interval_s = float(max(1.0, min_ollama_interval_s))
        self.emit_ui_stat = bool(emit_ui_stat)
        self.ollama = OllamaMatchAnalystClient(
            base_url=ollama_url,
            preferred_model=preferred_model,
            compute_mode=ollama_compute_mode,
        )
        self._q: "queue.Queue[Optional[Tuple[Dict[str, Any], int]]]" = queue.Queue(maxsize=16)
        self._stop_ev = threading.Event()
        self._last_ollama_call_ts = -1e9
        self._lock = threading.Lock()
        self.latest_commentary_ja: str = ""
        self.latest_hypothesis_name: str = ""
        self.latest_source_label: str = "stat_counter"
        self.ollama_deferred_by_sleep_count: int = 0
        self._pending_human_doctrine: Optional[StrategicDoctrine] = None
        self._thread = threading.Thread(target=self._worker_loop, name="StrategyGemmaAnalyst", daemon=True)
        self._thread.start()

    def pop_pending_human_doctrine(self) -> Optional[StrategicDoctrine]:
        with self._lock:
            doc = self._pending_human_doctrine
            self._pending_human_doctrine = None
            return doc

    def submit_finished_match(self, ep_rec: Dict[str, Any], applied_doctrine_id: int) -> None:
        try:
            self._q.put_nowait((ep_rec, int(applied_doctrine_id)))
        except queue.Full:
            pass

    def close(self, timeout_s: float = 1.0) -> None:
        self._stop_ev.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=timeout_s)

    def get_latest_commentary(self) -> Tuple[str, str]:
        with self._lock:
            return self.latest_hypothesis_name, self.latest_commentary_ja

    def _worker_loop(self) -> None:
        coach_file = coach_request_file_path(self.db.db_path)
        while not self._stop_ev.is_set():
            # 1. Check optional human voice/text coaching inbox (`<db_path>.coach_req.json`)
            if os.path.exists(coach_file):
                try:
                    with open(coach_file, "r", encoding="utf-8") as fh:
                        req_obj = json.load(fh)
                    try:
                        os.remove(coach_file)
                    except OSError:
                        pass
                    if isinstance(req_obj, dict) and str(req_obj.get("user_text", "")).strip():
                        self.process_human_coaching_sync(
                            user_text=str(req_obj["user_text"]).strip(),
                            stage_id=int(req_obj.get("stage_id", 1)),
                        )
                except Exception:
                    try:
                        if os.path.exists(coach_file):
                            os.remove(coach_file)
                    except OSError:
                        pass

            # 2. Check post-match episode analysis queue
            try:
                item = self._q.get(timeout=0.35)
            except queue.Empty:
                continue
            if item is None:
                break
            ep_rec, applied_id = item
            try:
                self.process_one_match_sync(ep_rec, applied_id, sys_prompt=GENTLE_NAVIGATOR_SYSTEM_PROMPT)
            except Exception:
                pass

    def process_human_coaching_sync(self, user_text: str, stage_id: int = 1) -> Tuple[int, str]:
        """
        Translate optional human voice/text strategic advice via `Ollama gemma4:12b` (`think: false`),
        register in SQLite, queue immediate live mid-match injection, and broadcast gentle voice reply!
        """
        fb_doc, fb_comm = synthesize_human_coaching_fallback(user_text, stage_id=stage_id)
        chosen_doc = fb_doc
        chosen_comm = fb_comm
        used_ollama_model = ""

        if self.enable_ollama:
            wait_t0 = time.perf_counter()
            while (
                not self._stop_ev.is_set()
                and is_sleep_consolidation_active(self.db.db_path)
                and (time.perf_counter() - wait_t0) < 20.0
            ):
                time.sleep(0.08)

            coach_prompt = (
                f"【人間コーチからのリアルタイム戦略提案】「{user_text}」\n"
                f"【対象ステージ】Stage={int(stage_id)}\n"
                "【指示】人間の提案意図を汲み取り、安全な大局戦略パラメータ(opening_lane, opening_curiosity_mult, "
                "opening_objective_mult, lead_caution_delta, comeback_lane, comeback_objective_mult)へ変換せよ。"
                "そして `commentary_ja` には、知的で落ち着いた優しい女の子のナビゲーターとして、"
                "提案への感謝と『ここからどう立ち回りを変えるか』を2文以内(80文字以内・英字記号なしの自然な話し言葉)で返答せよ。"
            )
            ok, parsed_json, _ = self.ollama.chat_structured_json(
                prompt_text=coach_prompt,
                system_prompt=GENTLE_NAVIGATOR_SYSTEM_PROMPT,
                json_schema=OLLAMA_STRATEGY_JSON_SCHEMA,
                max_tokens=260,
                temperature=0.28,
                num_ctx=2048,
                timeout_s=35.0,
            )
            if ok and isinstance(parsed_json, dict):
                chosen_doc, chosen_comm = parse_gemma_json_to_doctrine(
                    parsed_json,
                    {"stage_id": int(stage_id), "cause_tag": fb_doc.target_cause_tag, "feature_vec8": [0.0] * 8},
                    fb_doc,
                )
                chosen_doc.source = "human_gemma"
                chosen_doc.hypothesis_name = "Coach_" + chosen_doc.hypothesis_name[:56]
                used_ollama_model = self.ollama.active_model or "gemma4:12b"

        doc_id = self.db.register_candidate_doctrine(chosen_doc)
        chosen_doc.doctrine_id = doc_id
        src_lbl = used_ollama_model if used_ollama_model else "human_coach"

        with self._lock:
            self.latest_hypothesis_name = chosen_doc.hypothesis_name
            self.latest_commentary_ja = chosen_comm
            self.latest_source_label = src_lbl
            self._pending_human_doctrine = chosen_doc

        if self.emit_ui_stat:
            stats_now = self.db.get_stats_summary()
            emit_gemma_broadcast_stat({
                "evt_kind": "coach_reply",
                "source": src_lbl,
                "ollama_online": bool(used_ollama_model),
                "hypothesis": chosen_doc.hypothesis_name,
                "cause_tag": chosen_doc.target_cause_tag,
                "regime": "コーチ反映",
                "lane": LANE_NAMES.get(chosen_doc.slice_for_regime(REGIME_OPENING_CONTEST).lane_code, "BALANCED"),
                "commentary": chosen_comm,
                "tts_text": chosen_comm,
                "validated_cnt": int(stats_now.get("validated", 0)),
                "candidate_cnt": int(stats_now.get("candidate", 0)),
                "episodes_cnt": int(stats_now.get("episodes", 0)),
            })
        return doc_id, chosen_comm

    def process_one_match_sync(
        self,
        ep_rec: Dict[str, Any],
        applied_doctrine_id: int,
        sys_prompt: str = "あなたは3Dブリッツコア対戦AIの大局戦略アナリストです。",
    ) -> Tuple[int, int, str]:
        # 1. Immediately persist immutable episode facts + update validation stats
        stat_doc, stat_comm = synthesize_statistical_counter_doctrine(ep_rec)
        ep_rec_to_save = dict(ep_rec)
        ep_rec_to_save["commentary_ja"] = stat_comm
        ep_uid, _ = self.db.record_episode_and_update_validation(ep_rec_to_save, applied_doctrine_id)

        chosen_doc = stat_doc
        chosen_comm = stat_comm
        used_ollama_model = ""

        # 1.5 Sleep-vs-Ollama Mutual Exclusion Gate:
        # Give coordinator 60ms at episode-end to raise sleep lock, then wait until Sleep S-1..S-4 completes!
        if self.enable_ollama:
            if not self._stop_ev.is_set():
                time.sleep(0.06)
            waited_for_sleep = False
            wait_t0 = time.perf_counter()
            while (
                not self._stop_ev.is_set()
                and is_sleep_consolidation_active(self.db.db_path)
                and (time.perf_counter() - wait_t0) < 20.0
            ):
                waited_for_sleep = True
                time.sleep(0.08)
            if waited_for_sleep:
                self.ollama_deferred_by_sleep_count += 1
                if not self._stop_ev.is_set():
                    time.sleep(0.15)  # Allow CUDA empty_cache() settle before 12B LLM forward pass

        # 2. Query Ollama Gemma 4 12B (`think: false`) when online & past cooldown
        now = time.perf_counter()
        if (not self._stop_ev.is_set()) and self.enable_ollama and (now - self._last_ollama_call_ts) >= self.min_ollama_interval_s:
            self._last_ollama_call_ts = now
            stage_id = int(ep_rec.get("stage_id", 1))
            val_refs = [
                d for _, d in self.db.search_doctrines(
                    stage_id=stage_id,
                    context_vec8=np.asarray(ep_rec.get("feature_vec8", [0.0] * 8), dtype=np.float32),
                    target_cause_tag=str(ep_rec.get("cause_tag", "")),
                    top_k=2,
                    include_candidates=False,
                )
            ]
            dep_refs = self.db.get_deprecated_antipatterns(stage_id=stage_id, limit=2)
            prompt = build_gemma_strategy_prompt(ep_rec, val_refs, dep_refs)
            ok, parsed_json, _ = self.ollama.chat_structured_json(
                prompt_text=prompt,
                system_prompt=sys_prompt,
                json_schema=OLLAMA_STRATEGY_JSON_SCHEMA,
                max_tokens=260,
                temperature=0.25,
                num_ctx=2048,
                timeout_s=35.0,
            )
            if ok and isinstance(parsed_json, dict):
                chosen_doc, chosen_comm = parse_gemma_json_to_doctrine(parsed_json, ep_rec, stat_doc)
                used_ollama_model = self.ollama.active_model or "gemma4:12b"
                self.db.attach_episode_commentary(ep_uid, chosen_comm)

        # Register new candidate hypothesis when match was lost OR when Gemma synthesized a fresh insight
        new_doc_id = 0
        if (not bool(ep_rec.get("won", False))) or chosen_doc.source == "gemma4":
            new_doc_id = self.db.register_candidate_doctrine(chosen_doc)

        src_lbl = used_ollama_model if used_ollama_model else "stat_counter"
        with self._lock:
            self.latest_hypothesis_name = chosen_doc.hypothesis_name
            self.latest_commentary_ja = chosen_comm
            self.latest_source_label = src_lbl

        if self.emit_ui_stat:
            stats_now = self.db.get_stats_summary()
            emit_gemma_broadcast_stat({
                "evt_kind": "post_match",
                "source": src_lbl,
                "ollama_online": bool(used_ollama_model),
                "hypothesis": chosen_doc.hypothesis_name,
                "cause_tag": str(ep_rec.get("cause_tag", "")),
                "regime": "次戦準備",
                "lane": LANE_NAMES.get(chosen_doc.slice_for_regime(REGIME_OPENING_CONTEST).lane_code, "BALANCED"),
                "commentary": chosen_comm,
                "tts_text": chosen_comm,
                "validated_cnt": int(stats_now.get("validated", 0)),
                "candidate_cnt": int(stats_now.get("candidate", 0)),
                "episodes_cnt": int(stats_now.get("episodes", 0)),
            })

        return ep_uid, new_doc_id, chosen_comm


# ==============================================================================
# 6. Four-Boundary Closed-Loop Runtime Strategy Controller
# ==============================================================================
class RuntimeStrategyController:
    """
    Drives live closed-loop strategic modulation across FOUR match boundaries:
      [B1] Match Start (`episode_id` change) -> Load best `StrategicDoctrine` from SQLite
      [B2] Mid-Episode Respawn (`respawn_ev`) -> Penalize death lane & reroute to safer corridor!
      [B3] Macro Regime Shift (`OPENING / LEAD_LOCK / DEFENSE_CRISIS / COMEBACK_CLUTCH`) -> Dynamic slice switch!
      [B4] Stagnation Break (`stagnation_level >= 0.42` for >180 ticks) -> Flank & curiosity boost!

    Strictly respects Tactical Autonomy:
      - Returns peace-time strategic modifiers that `compute_gene_modulated_mip_and_fr` ONLY applies
        when `not exposed_threat`, leaving immediate combat, dodge, and `RSSM` Survival Veto 100% autonomous!
    """

    def __init__(
        self,
        db_path: str,
        seed: int = 2026,
        enable_async_analyst: bool = True,
        enable_ollama: bool = True,
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "gemma4:12b",
        ollama_compute_mode: str = "hybrid",
        min_ollama_interval_s: float = 6.0,
        emit_ui_stat: bool = False,
    ):
        assert_zero_torch_in_strategy_engine()
        self.db = StrategyKnowledgeDB(db_path)
        self.rng = np.random.default_rng(int(seed))
        self.emit_ui_stat = bool(emit_ui_stat)
        self._emitted_opening_brief = False
        self.recorder = MatchFlightRecorder(sample_stride_ticks=30)
        self.analyst: Optional[AsyncStrategyFeedbackWorker] = (
            AsyncStrategyFeedbackWorker(
                db=self.db,
                ollama_url=ollama_url,
                preferred_model=ollama_model,
                ollama_compute_mode=ollama_compute_mode,
                enable_ollama=enable_ollama,
                min_ollama_interval_s=min_ollama_interval_s,
                emit_ui_stat=emit_ui_stat,
            )
            if enable_async_analyst
            else None
        )

        self.current_episode_id: int = -1
        self.current_stage_id: int = 1
        self.active_doctrine: StrategicDoctrine = StrategicDoctrine(stage_id=1)
        self.active_regime: int = REGIME_OPENING_CONTEST
        self.active_slice: RegimePolicySlice = RegimePolicySlice()
        self.effective_lane_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.effective_lane_code: int = LANE_BALANCED

        # Per-episode mid-match adaptive counters
        self.ep_lane_deaths: Dict[int, int] = {LANE_LEFT_WING: 0, LANE_CENTER_AXIS: 0, LANE_RIGHT_WING: 0}
        self.last_regime_eval_tick: int = -999
        self.stagnation_high_ticks: int = 0
        self.stagnation_boost_until_tick: int = -1
        self.last_cause_tag: str = ""
        self.last_context_vec8: Optional[np.ndarray] = None
        self.last_cum_score: float = 0.0
        self.last_obj_prog: float = 0.50
        self.ep_started: bool = False
        self.reanchor_events_in_ep: int = 0
        self._last_announced_posture: Tuple[int, int] = (-1, -1)
        self._last_voice_milestone_tick: int = -9999
        self._human_override_active_in_ep: bool = False

    def close(self) -> None:
        if self.analyst is not None:
            self.analyst.close(timeout_s=0.8)

    @staticmethod
    def classify_board_regime(
        tick: int,
        obj_progress: float,
        enemy_carrier_visible: bool,
        carrying_self: bool,
    ) -> int:
        if enemy_carrier_visible or float(obj_progress) <= 0.38:
            return REGIME_DEFENSE_CRISIS
        if float(obj_progress) >= 0.64 and not carrying_self:
            return REGIME_ADVANTAGE_LOCK
        if (int(tick) >= 2200 and float(obj_progress) < 0.47) or float(obj_progress) <= 0.42:
            return REGIME_COMEBACK_CLUTCH
        return REGIME_OPENING_CONTEST

    def _recompute_effective_posture(self, tick: int) -> None:
        base_slice = self.active_doctrine.slice_for_regime(self.active_regime)
        lw, cw, rw = LANE_WEIGHT_PRESETS.get(base_slice.lane_code, (1.0, 1.0, 1.0))

        # [B2] Apply Mid-Match Respawn Death-Lane Adaptation!
        d_l = self.ep_lane_deaths.get(LANE_LEFT_WING, 0)
        d_c = self.ep_lane_deaths.get(LANE_CENTER_AXIS, 0)
        d_r = self.ep_lane_deaths.get(LANE_RIGHT_WING, 0)
        if d_c > 0:
            cw *= max(0.58, 1.0 - 0.18 * d_c)
            if d_l <= d_r:
                lw *= min(1.38, 1.0 + 0.14 * d_c)
            else:
                rw *= min(1.38, 1.0 + 0.14 * d_c)
        if d_l > 0:
            lw *= max(0.62, 1.0 - 0.16 * d_l)
            rw *= min(1.35, 1.0 + 0.10 * d_l)
        if d_r > 0:
            rw *= max(0.62, 1.0 - 0.16 * d_r)
            lw *= min(1.35, 1.0 + 0.10 * d_r)

        cur_m = base_slice.curiosity_mult
        obj_m = base_slice.objective_mult
        if tick < self.stagnation_boost_until_tick:
            # [B4] Anti-Stagnation Flank Boost
            cur_m = min(1.35, cur_m * 1.16)
            obj_m = min(1.40, obj_m * 1.12)
            if d_l <= d_r:
                lw = max(lw, 1.28)
            else:
                rw = max(rw, 1.28)
            cw = min(cw, 0.84)

        self.effective_lane_weights = (
            float(np.clip(lw, 0.60, 1.45)),
            float(np.clip(cw, 0.60, 1.45)),
            float(np.clip(rw, 0.60, 1.45)),
        )
        # Compact 1-byte lane code for shared-memory bus forwarding (`pad_cog[0]`)
        if self.effective_lane_weights[0] > 1.14 and self.effective_lane_weights[2] > 1.14 and self.effective_lane_weights[1] < 0.90:
            self.effective_lane_code = LANE_DUAL_WINGS
        elif self.effective_lane_weights[0] >= self.effective_lane_weights[1] + 0.14 and self.effective_lane_weights[0] >= self.effective_lane_weights[2]:
            self.effective_lane_code = LANE_LEFT_WING
        elif self.effective_lane_weights[2] >= self.effective_lane_weights[1] + 0.14 and self.effective_lane_weights[2] >= self.effective_lane_weights[0]:
            self.effective_lane_code = LANE_RIGHT_WING
        elif self.effective_lane_weights[1] >= 1.14:
            self.effective_lane_code = LANE_CENTER_AXIS
        else:
            self.effective_lane_code = base_slice.lane_code

        self.active_slice = RegimePolicySlice(
            lane_code=self.effective_lane_code,
            curiosity_mult=cur_m,
            objective_mult=obj_m,
            stealth_mult=base_slice.stealth_mult,
            caution_delta=base_slice.caution_delta,
            peace_goal_mode=base_slice.peace_goal_mode,
        ).clamped()

    def _finalize_previous_episode_if_needed(self, terminal_reward: float = 0.0, explicit_done: bool = False) -> None:
        if not self.ep_started or self.recorder.total_samples < 2:
            self.ep_started = False
            return
        won = bool(terminal_reward > 4.5 or (abs(terminal_reward) <= 4.5 and (self.last_obj_prog > 0.52 or self.last_cum_score > 3.5)))
        ko_flag = bool(abs(terminal_reward) >= 12.0 or self.last_obj_prog >= 0.96 or self.last_obj_prog <= 0.04)
        ep_rec = self.recorder.finalize_episode(
            final_obj_progress=self.last_obj_prog,
            net_score=self.last_cum_score,
            won=won,
            ko_flag=ko_flag,
        )
        self.last_cause_tag = str(ep_rec["cause_tag"])
        self.last_context_vec8 = np.asarray(ep_rec["feature_vec8"], dtype=np.float32)
        applied_id = int(self.active_doctrine.doctrine_id)
        if self.analyst is not None:
            self.analyst.submit_finished_match(ep_rec, applied_id)
        else:
            stat_doc, stat_comm = synthesize_statistical_counter_doctrine(ep_rec)
            ep_rec["commentary_ja"] = stat_comm
            self.db.record_episode_and_update_validation(ep_rec, applied_id)
            if not won:
                self.db.register_candidate_doctrine(stat_doc)
        self.ep_started = False

    def step(
        self,
        episode_id: int,
        stage_id: int,
        tick: int,
        pos_xz: np.ndarray,
        proprio: np.ndarray,
        believed_entities: Optional[np.ndarray],
        believed_count: int,
        friendly_turf: float,
        enemy_turf: float,
        cum_score: float,
        step_reward: float,
        respawn_ev: bool,
        ep_done: bool,
        veto_active: bool = False,
        ignition: float = 0.0,
        stagnation_level: float = 0.0,
        inner_speech: str = "",
        field_map: Optional[Any] = None,
    ) -> Dict[str, Any]:
        eff_stage = int(stage_id) if int(stage_id) > 0 else 1

        # [B1] Boundary 1: Episode Transition -> Finalize prior match & anchor initial strategy for new match!
        if int(episode_id) != self.current_episode_id:
            if self.ep_started:
                self._finalize_previous_episode_if_needed(terminal_reward=0.0, explicit_done=False)
            self.current_episode_id = int(episode_id)
            self.current_stage_id = eff_stage
            self.active_doctrine = self.db.select_active_doctrine_for_match(
                stage_id=eff_stage,
                context_vec8=self.last_context_vec8,
                last_cause_tag=self.last_cause_tag,
                rng=self.rng,
            )
            self.ep_lane_deaths = {LANE_LEFT_WING: 0, LANE_CENTER_AXIS: 0, LANE_RIGHT_WING: 0}
            self.active_regime = REGIME_OPENING_CONTEST
            self.last_regime_eval_tick = int(tick)
            self.stagnation_high_ticks = 0
            self.stagnation_boost_until_tick = -1
            self.reanchor_events_in_ep = 1
            self.recorder.reset(episode_id=int(episode_id), stage_id=eff_stage)
            self._recompute_effective_posture(int(tick))
            if field_map is not None and hasattr(field_map, "set_strategic_lane_weights"):
                field_map.set_strategic_lane_weights(*self.effective_lane_weights, clear_commitment=True)
            self.ep_started = True
            self._human_override_active_in_ep = False
            self._last_announced_posture = (int(self.active_regime), int(self.effective_lane_code))
            self._last_voice_milestone_tick = int(tick)
            if self.emit_ui_stat and not self._emitted_opening_brief:
                self._emitted_opening_brief = True
                o_on, _, o_chosen = (self.analyst.ollama.probe_available_models(timeout_s=0.45, force=True) if self.analyst is not None else (False, [], ""))
                stats0 = self.db.get_stats_summary()
                lane_open_ja = LANE_SPOKEN_JA.get(self.effective_lane_code, "ステージ全域")
                open_tts = f"観戦を開始しますね。まずは{lane_open_ja}から足場を塗り広げて、丁寧に主導権を握りましょうね。"
                emit_gemma_broadcast_stat({
                    "evt_kind": "opening",
                    "source": o_chosen if o_on else "stat_counter",
                    "ollama_online": bool(o_on),
                    "hypothesis": self.active_doctrine.hypothesis_name,
                    "cause_tag": self.active_doctrine.target_cause_tag,
                    "regime": REGIME_JA_SHORT.get(self.active_regime, "拮抗制圧"),
                    "lane": LANE_NAMES.get(self.effective_lane_code, "BALANCED"),
                    "commentary": self.active_doctrine.summary_ja,
                    "tts_text": open_tts,
                    "validated_cnt": int(stats0.get("validated", 0)),
                    "candidate_cnt": int(stats0.get("candidate", 0)),
                    "episodes_cnt": int(stats0.get("episodes", 0)),
                })

        # [B0] Live Human-to-Gemma Mid-Match Coaching Override (only triggers when user actually submitted input!)
        if self.analyst is not None:
            human_doc = self.analyst.pop_pending_human_doctrine()
            if human_doc is not None:
                self.active_doctrine = human_doc
                self._human_override_active_in_ep = True
                self._recompute_effective_posture(int(tick))
                self._last_announced_posture = (int(self.active_regime), int(self.effective_lane_code))
                self._last_voice_milestone_tick = int(tick)
                self.reanchor_events_in_ep += 1
                if field_map is not None and hasattr(field_map, "set_strategic_lane_weights"):
                    field_map.set_strategic_lane_weights(*self.effective_lane_weights, clear_commitment=True)

        # Detect Enemy Carrier or Immediate Hostile Threat from believed entities
        enemy_carrier_vis = False
        nearest_hostile_d = 999.0
        if believed_entities is not None and int(believed_count) > 0:
            ents = np.asarray(believed_entities, dtype=np.float32)
            for i in range(min(ents.shape[0], int(believed_count))):
                if float(ents[i, 9]) <= 0.08:
                    continue
                tag = float(ents[i, 8])
                if SemanticTag.is_hostile(tag):
                    dx = float(ents[i, 1]) - float(pos_xz[0])
                    dz = float(ents[i, 3]) - float(pos_xz[1])
                    nearest_hostile_d = min(nearest_hostile_d, math.hypot(dx, dz))
                    if abs(tag - SemanticTag.ENEMY_CARRIER_OR_SHOT) < 0.35:
                        enemy_carrier_vis = True

        obj_p = float(proprio[PROPRIO_IDX_OBJ_PROGRESS]) if proprio.shape[0] > PROPRIO_IDX_OBJ_PROGRESS else 0.50
        carrying_self = is_carrying_primary_objective(proprio)
        haz = float(proprio[PROPRIO_IDX_HAZARD]) if proprio.shape[0] > PROPRIO_IDX_HAZARD else 0.0

        regime_switched = False
        rerouted_now = False

        # [B2] Boundary 2: Mid-Episode Respawn (`flags & 16`) -> Penalize death lane & re-route immediately!
        if respawn_ev:
            dead_x = float(self.recorder.last_alive_pos_xz[0])
            dead_z = float(self.recorder.last_alive_pos_xz[1])
            dead_lane = _lane_from_x(dead_x)
            self.ep_lane_deaths[dead_lane] = self.ep_lane_deaths.get(dead_lane, 0) + 1
            self.active_regime = self.classify_board_regime(tick, obj_p, enemy_carrier_vis, carrying_self)
            self._recompute_effective_posture(int(tick))
            self.reanchor_events_in_ep += 1
            rerouted_now = True
            if field_map is not None:
                if hasattr(field_map, "register_strategic_hazard_scar"):
                    field_map.register_strategic_hazard_scar(dead_x, dead_z)
                if hasattr(field_map, "set_strategic_lane_weights"):
                    field_map.set_strategic_lane_weights(*self.effective_lane_weights, clear_commitment=True)

        # [B3] Boundary 3: Macro Regime Transition (checked every 30 ticks OR immediately on enemy carrier!)
        if enemy_carrier_vis or (int(tick) - self.last_regime_eval_tick) >= 30:
            self.last_regime_eval_tick = int(tick)
            new_reg = self.classify_board_regime(tick, obj_p, enemy_carrier_vis, carrying_self)
            if new_reg != self.active_regime:
                self.active_regime = new_reg
                self._recompute_effective_posture(int(tick))
                self.reanchor_events_in_ep += 1
                regime_switched = True
                rerouted_now = True

        # [B4] Boundary 4: Prolonged Stagnation Break
        if float(stagnation_level) >= 0.42:
            self.stagnation_high_ticks += 1
            if self.stagnation_high_ticks == 36:  # ~36 cognitive steps or ~3s of deadlock
                self.stagnation_boost_until_tick = int(tick) + 420
                self._recompute_effective_posture(int(tick))
                self.reanchor_events_in_ep += 1
                rerouted_now = True
        else:
            self.stagnation_high_ticks = max(0, self.stagnation_high_ticks - 1)

        if rerouted_now and field_map is not None and hasattr(field_map, "set_strategic_lane_weights"):
            field_map.set_strategic_lane_weights(*self.effective_lane_weights, clear_commitment=True)
        if rerouted_now and self.emit_ui_stat:
            new_posture = (int(self.active_regime), int(self.effective_lane_code))
            posture_changed = new_posture != self._last_announced_posture
            milestone_tts = ""
            # Only synthesize a short mid-match voice callout when strategic posture ACTUALLY changed & past 12s cooldown!
            if posture_changed and (int(tick) - self._last_voice_milestone_tick) >= 720:
                self._last_announced_posture = new_posture
                self._last_voice_milestone_tick = int(tick)
                lane_ja = LANE_SPOKEN_JA.get(self.effective_lane_code, "ステージ全域")
                if respawn_ev:
                    milestone_tts = f"正面は警戒されているので、復帰後は{lane_ja}から丁寧に立て直しますね。"
                elif self.active_regime == REGIME_DEFENSE_CRISIS:
                    milestone_tts = "相手にコアを持たれましたね。いったん塗りを抑えて、防衛と迎撃に集中しましょうね。"
                elif self.active_regime == REGIME_ADVANTAGE_LOCK:
                    milestone_tts = "しっかりリードできましたね。無理な深追いは控えて、潜伏と塗りで堅実に守りましょうね。"
                elif self.active_regime == REGIME_COMEBACK_CLUTCH:
                    milestone_tts = f"ここが勝負どころですね。{lane_ja}から積極的にコア前進を狙ってみますね。"
                else:
                    milestone_tts = f"盤面が落ち着きましたね。{lane_ja}からもう一度足場を広げましょう。"

            cur_hyp = self.active_doctrine.hypothesis_name
            cur_comm = milestone_tts if milestone_tts else self.active_doctrine.summary_ja
            cur_src = "stat_counter"
            if self.analyst is not None:
                h_lat, c_lat = self.analyst.get_latest_commentary()
                if not milestone_tts and c_lat:
                    cur_comm = c_lat
                    cur_hyp = h_lat or cur_hyp
                cur_src = self.analyst.latest_source_label
            stats_r = self.db.get_stats_summary()
            emit_gemma_broadcast_stat({
                "evt_kind": "milestone_shift" if milestone_tts else "silent_status",
                "source": cur_src,
                "ollama_online": bool(self.analyst is not None and self.analyst.ollama.active_model),
                "hypothesis": cur_hyp,
                "cause_tag": self.last_cause_tag or self.active_doctrine.target_cause_tag,
                "regime": REGIME_JA_SHORT.get(self.active_regime, "拮抗制圧"),
                "lane": LANE_NAMES.get(self.effective_lane_code, "BALANCED"),
                "commentary": cur_comm,
                "tts_text": milestone_tts,
                "validated_cnt": int(stats_r.get("validated", 0)),
                "candidate_cnt": int(stats_r.get("candidate", 0)),
                "episodes_cnt": int(stats_r.get("episodes", 0)),
            })

        self.last_cum_score = float(cum_score)
        self.last_obj_prog = obj_p

        self.recorder.record_step(
            tick=int(tick),
            pos_xz=pos_xz,
            proprio=proprio,
            friendly_turf=friendly_turf,
            enemy_turf=enemy_turf,
            enemy_carrier_visible=enemy_carrier_vis,
            respawn_ev=respawn_ev,
            veto_active=veto_active,
            ignition=ignition,
            active_regime=self.active_regime,
            regime_switched=regime_switched,
            inner_speech=inner_speech,
        )

        if ep_done:
            self._finalize_previous_episode_if_needed(terminal_reward=float(step_reward), explicit_done=True)

        # Tactical Autonomy Gate: ONLY hint peace-time goal_mode when NO immediate enemy combat crisis exists!
        in_immediate_combat = bool(nearest_hostile_d <= 12.5 or haz >= 0.35 or veto_active)
        peace_mode_override: Optional[int] = None if in_immediate_combat else int(self.active_slice.peace_goal_mode)

        return {
            "doctrine_id": int(self.active_doctrine.doctrine_id),
            "hypothesis_name": self.active_doctrine.hypothesis_name,
            "regime_id": int(self.active_regime),
            "regime_name": REGIME_NAMES.get(self.active_regime, "OPENING_CONTEST"),
            "regime_ja": REGIME_JA_SHORT.get(self.active_regime, "拮抗"),
            "lane_code": int(self.effective_lane_code),
            "lane_weights": self.effective_lane_weights,
            "curiosity_mult": float(self.active_slice.curiosity_mult),
            "objective_mult": float(self.active_slice.objective_mult),
            "stealth_mult": float(self.active_slice.stealth_mult),
            "caution_delta": float(self.active_slice.caution_delta),
            "human_override": bool(self._human_override_active_in_ep),
            "peace_goal_mode_override": peace_mode_override,
            "in_immediate_combat": in_immediate_combat,
            "reanchor_events_in_ep": int(self.reanchor_events_in_ep),
        }

    def format_hud_status_tag(self) -> str:
        reg_ja = REGIME_JA_SHORT.get(self.active_regime, "拮抗")
        lane_short = {
            LANE_BALANCED: "全域",
            LANE_LEFT_WING: "左翼",
            LANE_CENTER_AXIS: "中央",
            LANE_RIGHT_WING: "右翼",
            LANE_DUAL_WINGS: "両翼",
        }.get(self.effective_lane_code, "全域")
        if self.analyst is not None:
            _, comm = self.analyst.get_latest_commentary()
            if comm:
                return f"[戦略:{reg_ja}·{lane_short}] {comm[:54]}"
        return f"[戦略:{reg_ja}·{lane_short}] {self.active_doctrine.hypothesis_name[:28]}"


# ==============================================================================
# 7. Comprehensive Self-Test (`tempfile.TemporaryDirectory()` Isolated)
# ==============================================================================
def run_strategy_engine_self_test() -> bool:
    assert_zero_torch_in_strategy_engine()
    print("=" * 76)
    print("  Strategy Knowledge Engine & 4-Boundary Controller — Self-Test")
    print("=" * 76)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = os.path.join(tmp_dir, "test_strategy_knowledge.sqlite3")
        ctrl = RuntimeStrategyController(
            db_path=db_file,
            seed=42,
            enable_async_analyst=False,  # deterministic synchronous verification first
            enable_ollama=False,
        )

        # 1. Verify Seed Initialization & Hybrid Search Engine
        st0 = ctrl.db.get_stats_summary()
        assert st0["seed"] >= 3, f"Expected >=3 seed doctrines, got {st0}"
        hits = ctrl.db.search_doctrines(
            stage_id=1,
            context_vec8=np.array([1.0, 0.80, 0.30, 0.20, 0.10, 0.35, 0.70, -0.30], dtype=np.float32),
            target_cause_tag="CENTRAL_OVEREXTEND_WIPE",
            text_query="高台 側面",
            top_k=2,
        )
        assert len(hits) >= 1 and hits[0][1].target_cause_tag == "CENTRAL_OVEREXTEND_WIPE"
        print(f"[PASS] Check 1: SQLite Hybrid Search (FTS5={ctrl.db._fts_enabled}) matched '{hits[0][1].hypothesis_name}'.")

        # 2. Verify All 4 Mid-Match Strategic Injection Boundaries + Tactical Non-Interference!
        prop = np.zeros(16, dtype=np.float32)
        prop[PROPRIO_IDX_SURFACE_POSTURE] = 0.72
        prop[PROPRIO_IDX_RESOURCE_CARRIER] = 0.85
        prop[PROPRIO_IDX_OBJ_PROGRESS] = 0.50

        # [B1] Match Start
        out_b1 = ctrl.step(
            episode_id=1, stage_id=1, tick=0, pos_xz=np.array([0.0, -8.0], dtype=np.float32),
            proprio=prop, believed_entities=None, believed_count=0,
            friendly_turf=0.20, enemy_turf=0.20, cum_score=0.0, step_reward=0.0,
            respawn_ev=False, ep_done=False, inner_speech="◆01·開幕·塗拡",
        )
        assert out_b1["regime_id"] == REGIME_OPENING_CONTEST
        w_center_init = out_b1["lane_weights"][1]

        # Simulate agent advancing to center (x=0.0) and dying TWICE ([B2] Mid-Episode Respawn!)
        for k_die in range(2):
            ctrl.step(
                episode_id=1, stage_id=1, tick=60 + k_die * 90, pos_xz=np.array([0.4, 0.0], dtype=np.float32),
                proprio=prop, believed_entities=None, believed_count=0,
                friendly_turf=0.22, enemy_turf=0.45, cum_score=-2.0, step_reward=0.0,
                respawn_ev=False, ep_done=False,
            )
            out_b2 = ctrl.step(
                episode_id=1, stage_id=1, tick=90 + k_die * 90, pos_xz=np.array([0.0, -13.0], dtype=np.float32),
                proprio=prop, believed_entities=None, believed_count=0,
                friendly_turf=0.22, enemy_turf=0.48, cum_score=-5.0, step_reward=-3.0,
                respawn_ev=True, ep_done=False, inner_speech="◆19·復帰·側面再編",
            )
        assert out_b2["lane_weights"][1] < w_center_init - 0.20, "Center death lane must be penalized on respawn!"
        assert max(out_b2["lane_weights"][0], out_b2["lane_weights"][2]) > 1.15, "Wing lane must be boosted after central deaths!"

        # [B3] Macro Regime Shift -> Enemy BlitzCore Carrier Sighted (`DEFENSE_CRISIS`) + Tactical Combat Gate!
        ents_crisis = np.zeros((4, 10), dtype=np.float32)
        ents_crisis[0] = [201.0, 1.5, 1.2, -3.0, 0.0, 0.0, -4.0, 0.75, SemanticTag.ENEMY_CARRIER_OR_SHOT, 0.95]
        prop[PROPRIO_IDX_OBJ_PROGRESS] = 0.34
        out_b3 = ctrl.step(
            episode_id=1, stage_id=1, tick=300, pos_xz=np.array([-4.0, -6.0], dtype=np.float32),
            proprio=prop, believed_entities=ents_crisis, believed_count=1,
            friendly_turf=0.24, enemy_turf=0.52, cum_score=-8.0, step_reward=-1.0,
            respawn_ev=False, ep_done=False, ignition=0.68, inner_speech="◆15·敵コア迎撃",
        )
        assert out_b3["regime_id"] == REGIME_DEFENSE_CRISIS
        assert out_b3["in_immediate_combat"] is True
        assert out_b3["peace_goal_mode_override"] is None, "Strategy must NEVER override tactical mode during immediate combat!"

        # [B3b] Advantage Lock check when ahead (`obj_progress = 0.72`, no nearby enemy)
        prop[PROPRIO_IDX_OBJ_PROGRESS] = 0.72
        out_lead = ctrl.step(
            episode_id=1, stage_id=1, tick=360, pos_xz=np.array([-5.0, 4.0], dtype=np.float32),
            proprio=prop, believed_entities=None, believed_count=0,
            friendly_turf=0.52, enemy_turf=0.28, cum_score=8.0, step_reward=2.0,
            respawn_ev=False, ep_done=False,
        )
        assert out_lead["regime_id"] == REGIME_ADVANTAGE_LOCK
        assert out_lead["caution_delta"] > 0.0, "Advantage Lock regime must raise survival caution!"

        # Finish Episode 1 as a loss due to central wipes -> should generate a `candidate` counter-doctrine
        prop[PROPRIO_IDX_OBJ_PROGRESS] = 0.25
        ctrl.step(
            episode_id=1, stage_id=1, tick=540, pos_xz=np.array([0.2, -5.0], dtype=np.float32),
            proprio=prop, believed_entities=None, believed_count=0,
            friendly_turf=0.24, enemy_turf=0.56, cum_score=-14.0, step_reward=-10.0,
            respawn_ev=False, ep_done=True,
        )
        st1 = ctrl.db.get_stats_summary()
        assert st1["episodes"] == 1 and st1["candidate"] >= 1, f"Expected candidate generation after loss, got {st1}"
        print(f"[PASS] Check 2: 4-Boundary Re-Anchoring (`reanchors={out_lead['reanchor_events_in_ep']}`) & Turning-Point Candidate Creation verified.")

        # 3. Verify Multi-Trial Validation Lifecycle (`candidate -> validated` requires >= 3 trials, NOT 1!)
        cand_rows = ctrl.db.search_doctrines(stage_id=1, target_cause_tag="CENTRAL_OVEREXTEND_WIPE", top_k=5, include_candidates=True)
        cand_doc = [d for _, d in cand_rows if d.status == "candidate"][0]
        for t_idx in range(MIN_VALIDATION_TRIALS):
            _, new_st = ctrl.db.record_episode_and_update_validation(
                {
                    "stage_id": 1,
                    "episode_id": 10 + t_idx,
                    "outcome": "WIN_TIME",
                    "won": True,
                    "final_obj_progress": 0.68,
                    "net_score": 12.5,
                    "friendly_turf_peak": 0.54,
                    "enemy_turf_peak": 0.31,
                    "respawns": 1,
                    "vetoes": 2,
                    "feature_vec8": [0.0, 0.1, 0.0, 0.1, 0.0, 0.54, 0.2, 0.28],
                    "cause_tag": "WING_HIGHGROUND_CONTROL",
                },
                applied_doctrine_id=cand_doc.doctrine_id,
            )
            if t_idx < MIN_VALIDATION_TRIALS - 1:
                assert new_st == "candidate", "Must NOT prematurely validate before MIN_VALIDATION_TRIALS!"
            else:
                assert new_st == "validated", f"Expected promotion to 'validated' on trial {MIN_VALIDATION_TRIALS}, got {new_st}"

        # Also verify deprecation & anti-pattern retention on a bad candidate
        bad_id = ctrl.db.register_candidate_doctrine(StrategicDoctrine(
            stage_id=1, source="stat_counter", status="candidate",
            target_cause_tag="PASSIVE_TURF_STALL", hypothesis_name="Bad_Corner_Camp_Test",
            summary_ja="自陣隅に留まり続ける不適切仮説テスト",
        ))
        for t_idx in range(MIN_VALIDATION_TRIALS):
            _, bad_st = ctrl.db.record_episode_and_update_validation(
                {
                    "stage_id": 1, "episode_id": 30 + t_idx, "outcome": "LOSS_KO", "won": False,
                    "final_obj_progress": 0.10, "net_score": -18.0, "friendly_turf_peak": 0.18,
                    "enemy_turf_peak": 0.65, "respawns": 3, "vetoes": 0,
                    "feature_vec8": [1.0, 0.5, 1.0, 0.6, 0.0, 0.18, 0.6, -0.4],
                    "cause_tag": "PASSIVE_TURF_STALL",
                },
                applied_doctrine_id=bad_id,
            )
        assert bad_st == "deprecated"
        anti = ctrl.db.get_deprecated_antipatterns(stage_id=1)
        assert any(d.doctrine_id == bad_id for d in anti), "Deprecated doctrine must be retained as anti-pattern!"
        print("[PASS] Check 3: Multi-Trial Promotion (`candidate -> validated`) & Anti-Pattern Deprecation verified.")

        # 4. Verify Brain Integration: Hippocampal Lane/Scar Steering + MIP Tactical Combat Gate!
        from universal_brain_3d_core import (
            Brain3DConfig,
            MultiLayerCognitiveFieldMap3D,
            compute_gene_modulated_mip_and_fr,
        )

        fmap = MultiLayerCognitiveFieldMap3D((-12.0, 0.0, -16.0), (12.0, 4.0, 16.0), grid_n=48)
        fmap.occ_state[:, :] = fmap.FREE
        fmap.surface_map[:, :] = 0.0
        fmap.set_strategic_lane_weights(1.34, 0.72, 0.88, clear_commitment=True)
        fmap.register_strategic_hazard_scar(0.0, -4.0)
        dir_wing, _, ok_wing = fmap.find_best_frontier_or_turf_target(np.array([0.0, -10.0], dtype=np.float32))
        assert ok_wing and float(dir_wing[0]) < -0.15, f"Expected Left-Wing (-X) hippocampal steering, got dir={dir_wing}"

        cfg_b = Brain3DConfig(enable_strategy_kb=True, strategy_db_path=db_file)
        q_dummy = np.full(8, 0.5, dtype=np.float32)
        surv_dummy = np.full(8, 0.85, dtype=np.float32)
        adv_dummy = np.full(8, 0.5, dtype=np.float32)
        gwt_dummy = np.zeros(128, dtype=np.float32)
        poses_dummy = np.zeros((8, 6), dtype=np.float32)
        poses_dummy[0, [0, 2]] = [0.0, 0.8]  # aligns with objective +Z
        verbs_dummy = np.ones(8, dtype=np.uint8)
        mc_dummy = np.zeros(24, dtype=np.float32)
        mc_dummy[0:4] = [0.0, 1.0, 0.5, 1.0]
        mc_dummy[4:8] = [1.0, 0.0, 0.5, 1.0]

        # Under immediate exposed threat (`haz = 0.72, not in refuge`), strategic curiosity/objective multipliers MUST have 0.0 effect!
        prop_combat = np.zeros(16, dtype=np.float32)
        prop_combat[0], prop_combat[2] = 0.5, 0.5
        prop_combat[PROPRIO_IDX_HAZARD] = 0.72
        prop_combat[PROPRIO_IDX_SURFACE_POSTURE] = 0.40
        q_c1, _, _, _, _, _ = compute_gene_modulated_mip_and_fr(
            q_dummy, surv_dummy, adv_dummy, gwt_dummy, poses_dummy, verbs_dummy, prop_combat, mc_dummy, None, cfg_b,
            strategic_curiosity_mult=0.78, strategic_objective_mult=0.78,
        )
        q_c2, _, _, _, _, _ = compute_gene_modulated_mip_and_fr(
            q_dummy, surv_dummy, adv_dummy, gwt_dummy, poses_dummy, verbs_dummy, prop_combat, mc_dummy, None, cfg_b,
            strategic_curiosity_mult=1.35, strategic_objective_mult=1.40,
        )
        assert float(np.max(np.abs(q_c1 - q_c2))) == 0.0, "Peace-time strategy multipliers leaked into active combat tactics!"

        # In peace-time (`haz = 0.0`), strategic objective multiplier cleanly differentiates M0!
        prop_peace = prop_combat.copy()
        prop_peace[PROPRIO_IDX_HAZARD] = 0.0
        q_p1, _, _, _, _, _ = compute_gene_modulated_mip_and_fr(
            q_dummy, surv_dummy, adv_dummy, gwt_dummy, poses_dummy, verbs_dummy, prop_peace, mc_dummy, None, cfg_b,
            strategic_objective_mult=0.80,
        )
        q_p2, _, _, _, _, _ = compute_gene_modulated_mip_and_fr(
            q_dummy, surv_dummy, adv_dummy, gwt_dummy, poses_dummy, verbs_dummy, prop_peace, mc_dummy, None, cfg_b,
            strategic_objective_mult=1.35,
        )
        assert float(q_p2[0] - q_p1[0]) > 0.05, "Peace-time strategic objective gain failed to modulate M0!"
        print("[PASS] Check 4: Hippocampal Wing/Scar Steering & MIP Peace-vs-Combat Tactical Autonomy Gate (`max|ΔQ_combat| == 0.0`) verified.")

        # 5. Verify Sleep-vs-Ollama Mutual Exclusion Gate (`set_sleep_consolidation_active`)
        set_sleep_consolidation_active(db_file, True, max_hold_s=5.0)
        assert is_sleep_consolidation_active(db_file) is True
        gate_worker = AsyncStrategyFeedbackWorker(
            ctrl.db, preferred_model="gemma4:12b", enable_ollama=True, min_ollama_interval_s=1.0, emit_ui_stat=False
        )
        # Force fast offline probe state so HTTP doesn't add timeout noise after unlock
        gate_worker.ollama._last_probe_fail_ts = time.perf_counter()
        rel_thread = threading.Thread(target=lambda: (time.sleep(0.22), set_sleep_consolidation_active(db_file, False)), daemon=True)
        t_gate0 = time.perf_counter()
        rel_thread.start()
        gate_worker.process_one_match_sync(
            {
                "stage_id": 1, "episode_id": 77, "outcome": "WIN_TIME", "won": True,
                "final_obj_progress": 0.65, "net_score": 8.0, "friendly_turf_peak": 0.45,
                "enemy_turf_peak": 0.30, "respawns": 0, "vetoes": 1,
                "feature_vec8": [0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.0, 0.15],
                "cause_tag": "CLEAN_PICK_TO_PUSH",
            },
            applied_doctrine_id=0,
        )
        gate_elapsed = time.perf_counter() - t_gate0
        rel_thread.join(timeout=1.0)
        gate_worker.close()
        assert gate_worker.ollama_deferred_by_sleep_count >= 1 and gate_elapsed >= 0.25, (
            f"Expected Ollama worker to defer until Sleep completed! deferred={gate_worker.ollama_deferred_by_sleep_count}, dt={gate_elapsed:.3f}s"
        )
        assert is_sleep_consolidation_active(db_file) is False
        print(f"[PASS] Check 5: Sleep-vs-Ollama Mutual Exclusion Gate (`deferred={gate_worker.ollama_deferred_by_sleep_count}`, `wait={gate_elapsed:.2f}s`) verified.")

        # 6. Optional Live Ollama `gemma4:12b` (`think: false`) Non-Blocking Probe
        from companion_voice_io import OllamaMatchAnalystClient

        probe_client = OllamaMatchAnalystClient(preferred_model="gemma4:12b")
        o_ok, _, o_model = probe_client.probe_available_models(timeout_s=0.8, force=True)
        print(f"[INFO] Optional Ollama Gemma 4 Probe: online={o_ok}, selected_model='{o_model}'")
        if o_ok:
            worker = AsyncStrategyFeedbackWorker(ctrl.db, preferred_model=o_model, enable_ollama=True, min_ollama_interval_s=1.0)
            try:
                _, g_doc_id, g_comm = worker.process_one_match_sync(
                    {
                        "stage_id": 1, "episode_id": 99, "outcome": "LOSS_TIME", "won": False,
                        "final_obj_progress": 0.32, "net_score": -9.5, "friendly_turf_peak": 0.30,
                        "enemy_turf_peak": 0.55, "respawns": 3, "vetoes": 2,
                        "death_lanes": {"LEFT_WING": 0, "CENTER_AXIS": 3, "RIGHT_WING": 0},
                        "turning_point": {
                            "start_tick": 180, "end_tick": 360, "swing_delta": -0.34,
                            "winner_vs_loser_diff": "劣勢転換: CENTER_AXISで3連続被撃破されコア前進を許した",
                        },
                        "inner_speech_events": [{"tick": 240, "type": "RESPAWN_DEATH", "speech": "◆19·正面崩壊·側面要"}],
                        "feature_vec8": [1.0, 1.0, 0.5, 0.4, 0.0, 0.30, 0.6, -0.34],
                        "cause_tag": "CENTRAL_OVEREXTEND_WIPE",
                    },
                    applied_doctrine_id=0,
                )
                print(f"[PASS] Live Ollama `{o_model}` (`think: false` + JSON Schema) Doctrine #{g_doc_id}: {g_comm}")
            finally:
                worker.close()

        ctrl.close()
        assert_zero_torch_in_strategy_engine()

    print("=" * 76)
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Strategic Knowledge DB & Multi-Regime Controller")
    parser.add_argument("--self-test", action="store_true", help="Run isolated tempfile unit & integration checks")
    args = parser.parse_args()
    if args.self_test:
        return 0 if run_strategy_engine_self_test() else 1
    print("Run with `--self-test` to verify the Strategy Knowledge Engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
