# -*- coding: utf-8 -*-
"""
companion_voice_io.py — Ollama Post-Match Analyst, AivisSpeech/VOICEVOX TTS & faster-whisper STT Sidecar (v2.3.0)
=================================================================================================================
Reusable peripheral AI I/O & companion service module for 3D Pseudo-Brain Unity projects:
  1. `OllamaMatchAnalystClient`:
     - Auto-probes `http://127.0.0.1:11434/api/tags` to detect loaded local LLMs (prioritizing `gemma4:12b`,
       `gemma4`, `gemma3`, `qwen`).
     - Summarizes 3D match event histories (turf control swings, objective pushes, respawns, Basal Ganglia
       holds, survival vetoes, and inner-speech traces) for post-match debriefing or interactive Q&A.
     - Automatically passes `"think": false` (with fallback retry) so Gemma 4 / Qwen reasoning models emit
       concise Japanese answers directly into `message.content` without exhausting token budgets.
  2. `LocalSpeechTTSClient`:
     - Auto-probes AivisSpeech (`http://127.0.0.1:10101`) and VOICEVOX (`http://127.0.0.1:50021`),
       synthesizing 24kHz/44.1kHz WAV bytes from `/audio_query` + `/synthesis`.
  3. `run_stt_jsonl_sidecar`:
     - Push-to-talk USB microphone speech-to-text sidecar powered by `faster-whisper` + `sounddevice`,
       communicating with Unity C# via UTF-8 JSONL over `stdin` / `stdout` with zero-orphan PID watchdog.

Quick Start:
  python companion_voice_io.py --self-test
  python companion_voice_io.py --mode stt_sidecar --model small --language ja --parent-pid <UNITY_PID>

License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PARENT_WS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isfile(os.path.join(_PARENT_WS_DIR, "universal_brain_3d_core.py")) and _PARENT_WS_DIR not in sys.path:
    sys.path.insert(0, _PARENT_WS_DIR)

from universal_brain_3d_core import start_parent_and_stdin_watchdog


# ==============================================================================
# 1. Structured Match Timeline Digest Builder & Ollama LLM Companion Client
# ==============================================================================
@dataclass
class MatchTimelineRecorder:
    """
    Lightweight ring/episode recorder accumulating key 3D match milestones (turf lead changes,
    objective pickups/pushes, splats/respawns, survival vetoes) into a compact LLM-ready digest.
    """
    max_events: int = 48
    events: List[Dict[str, Any]] = field(default_factory=list)
    samples_count: int = 0
    peak_friendly_turf: float = 0.0
    peak_obj_progress: float = 0.50
    respawns_count: int = 0
    vetoes_count: int = 0

    def reset(self) -> None:
        self.events.clear()
        self.samples_count = 0
        self.peak_friendly_turf = 0.0
        self.peak_obj_progress = 0.50
        self.respawns_count = 0
        self.vetoes_count = 0

    def record_event(
        self,
        tick: int,
        event_type: str,
        detail: str,
        friendly_turf: float = 0.0,
        obj_progress: float = 0.50,
        inner_speech: str = "",
    ) -> None:
        self.samples_count += 1
        self.peak_friendly_turf = max(self.peak_friendly_turf, float(friendly_turf))
        self.peak_obj_progress = max(self.peak_obj_progress, float(obj_progress))
        if event_type == "RESPAWN":
            self.respawns_count += 1
        elif event_type == "SURVIVAL_VETO":
            self.vetoes_count += 1

        if len(self.events) >= self.max_events:
            self.events.pop(0)
        self.events.append({
            "tick": int(tick),
            "type": str(event_type),
            "detail": str(detail),
            "turf": round(float(friendly_turf), 2),
            "obj": round(float(obj_progress), 2),
            "speech": str(inner_speech),
        })

    def build_japanese_debrief_prompt(self, final_result: str = "終了", user_question: str = "") -> str:
        lines = [
            f"【試合サマリ】結果={final_result} / 最大味方塗率={self.peak_friendly_turf * 100.0:.1f}% "
            f"/ 最大目標進行度={self.peak_obj_progress * 100.0:.1f}% / 復帰回数={self.respawns_count} / 生存Veto発動={self.vetoes_count}",
            "【主要タイムライン】",
        ]
        for ev in self.events[-16:]:
            sp = f" (内言:{ev['speech']})" if ev["speech"] else ""
            lines.append(f"- T{ev['tick']:04d} [{ev['type']}] {ev['detail']} | 塗={ev['turf']:.2f} 進行={ev['obj']:.2f}{sp}")
        if user_question.strip():
            lines.append(f"【観察者からの質問】{user_question.strip()}")
        else:
            lines.append("【依頼】この疑似脳プレイヤーの良かった判断と次戦の改善点を、自然な日本語3文以内で振り返り実況してください。")
        return "\n".join(lines)


def resolve_ollama_compute_options(
    compute_mode: str = "hybrid",
    model_name: str = "gemma4:12b",
    num_ctx: int = 2048,
    num_predict: int = 260,
    temperature: float = 0.25,
    hybrid_gpu_layers: int = 18,
    cpu_threads: int = 0,
) -> Dict[str, Any]:
    """
    Build unified Ollama `/api/chat` `options` dict so all callers share identical `(num_gpu, num_ctx)`
    and NEVER trigger mid-session model reloads:
      - `"hybrid"` (Default): Partial GPU+CPU split (`num_gpu=18` of 48 layers on `gemma4:12b` -> ~3.2GB VRAM
        + ~5.1GB CPU RAM, `num_thread` capped to ~62.5% of logical cores so Unity & Brain stay silky smooth).
      - `"cpu"`             : 100% CPU execution (`num_gpu=0` -> 0MB VRAM).
      - `"gpu"`             : Full GPU offload (`num_gpu=-1`).
    """
    mode = (compute_mode or "hybrid").strip().lower()
    logical_cores = max(2, int(os.cpu_count() or 8))
    eff_threads = int(cpu_threads) if int(cpu_threads) > 0 else max(4, min(12, int(round(logical_cores * 0.625))))

    opts: Dict[str, Any] = {
        "num_ctx": int(max(512, num_ctx)),
        "num_predict": int(max(32, num_predict)),
        "temperature": float(np.clip(temperature, 0.0, 1.0)),
    }
    m_low = (model_name or "").lower()
    if mode == "cpu":
        opts["num_gpu"] = 0
        opts["num_thread"] = eff_threads
    elif mode == "gpu":
        opts["num_gpu"] = -1
    else:
        if any(tag in m_low for tag in ("26b", "27b", "32b")):
            layers = 12
        elif any(tag in m_low for tag in ("2b", "3b", "4b")):
            layers = 12
        else:
            layers = int(max(4, hybrid_gpu_layers))
        opts["num_gpu"] = layers
        opts["num_thread"] = eff_threads
    return opts


class OllamaMatchAnalystClient:
    """Auto-detecting local Ollama HTTP client (`gemma4:12b` ready with `think: false` & GPU+CPU hybrid split)."""

    PREFERRED_MODELS = ("gemma4:12b", "gemma4", "gemma3:12b", "gemma3", "qwen2.5", "llama3")

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        preferred_model: str = "",
        compute_mode: str = "hybrid",
        hybrid_gpu_layers: int = 18,
        cpu_threads: int = 0,
    ):
        self.base_url = base_url.rstrip("/")
        self.preferred_model = preferred_model.strip()
        mode_clean = (compute_mode or "hybrid").strip().lower()
        self.compute_mode = mode_clean if mode_clean in ("hybrid", "cpu", "gpu") else "hybrid"
        self.hybrid_gpu_layers = int(max(4, hybrid_gpu_layers))
        self.cpu_threads = int(max(0, cpu_threads))
        self.active_model = ""
        self._last_probe_fail_ts: float = -1e9
        self.offline_retry_cooldown_s: float = 20.0

    def probe_available_models(self, timeout_s: float = 1.5, force: bool = False) -> Tuple[bool, List[str], str]:
        now = time.perf_counter()
        if (not force) and (not self.active_model) and (now - self._last_probe_fail_ts) < self.offline_retry_cooldown_s:
            return False, [], ""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            names = [str(m.get("name", "")) for m in (data.get("models") or []) if m.get("name")]
            if not names:
                self._last_probe_fail_ts = now
                return False, [], ""
            chosen = ""
            if self.preferred_model:
                for n in names:
                    if n.lower() == self.preferred_model.lower() or n.lower().startswith(self.preferred_model.lower()):
                        chosen = n
                        break
            if not chosen:
                for pref in self.PREFERRED_MODELS:
                    for n in names:
                        if pref in n.lower():
                            chosen = n
                            break
                    if chosen:
                        break
            if not chosen:
                chosen = names[0]
            self.active_model = chosen
            self._last_probe_fail_ts = -1e9
            return True, names, chosen
        except Exception:
            self._last_probe_fail_ts = now
            return False, [], ""

    def chat_structured_json(
        self,
        prompt_text: str,
        system_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 280,
        temperature: float = 0.25,
        num_ctx: int = 2048,
        timeout_s: float = 15.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Query Ollama (`gemma4:12b` preferred) with `think: false` and structured JSON Schema output
        so strategy extraction is fast, VRAM-light (`num_ctx=2048`), and deterministically parseable.
        """
        if not self.active_model:
            ok, _, _ = self.probe_available_models(timeout_s=0.8)
            if not ok:
                return False, None, "Ollama offline or no model loaded"

        eff_timeout = float(timeout_s)
        if self.compute_mode == "cpu":
            eff_timeout = max(eff_timeout, 75.0)
        elif self.compute_mode == "hybrid":
            eff_timeout = max(eff_timeout, 55.0)

        opts = resolve_ollama_compute_options(
            compute_mode=self.compute_mode,
            model_name=self.active_model,
            num_ctx=num_ctx,
            num_predict=max_tokens,
            temperature=temperature,
            hybrid_gpu_layers=self.hybrid_gpu_layers,
            cpu_threads=self.cpu_threads,
        )

        fmt_variants: List[Any] = [json_schema, "json"] if isinstance(json_schema, dict) else ["json"]
        for fmt_spec in fmt_variants:
            for include_think_false in (True, False):
                payload: Dict[str, Any] = {
                    "model": self.active_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text},
                    ],
                    "stream": False,
                    "keep_alive": "20m",
                    "format": fmt_spec,
                    "options": opts,
                }
                if include_think_false:
                    payload["think"] = False
                req = urllib.request.Request(
                    f"{self.base_url}/api/chat",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                        res = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    msg = (res.get("message") or {}).get("content", "").strip()
                    if not msg:
                        continue
                    # Strip accidental markdown fences if older model returned ```json ... ```
                    if msg.startswith("```"):
                        lines = [ln for ln in msg.splitlines() if not ln.strip().startswith("```")]
                        msg = "\n".join(lines).strip()
                    parsed = json.loads(msg)
                    if isinstance(parsed, dict):
                        return True, parsed, msg
                except urllib.error.HTTPError as e:
                    if e.code == 400:
                        continue
                    return False, None, f"Ollama HTTP error ({e.code})"
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self._last_probe_fail_ts = time.perf_counter()
                    return False, None, f"Ollama request failed: {e}"
        return False, None, "Ollama returned empty or non-JSON content"

    def chat_debrief(
        self,
        prompt_text: str,
        system_prompt: str = "あなたは3D対戦アクションゲームをプレイした疑似脳AIの振り返りコメンテーターです。簡潔で分かりやすい日本語で応答してください。",
        max_tokens: int = 220,
        timeout_s: float = 20.0,
    ) -> Tuple[bool, str]:
        if not self.active_model:
            ok, _, _ = self.probe_available_models()
            if not ok:
                return False, "Ollamaサーバーが未起動か、モデルがロードされていません。"

        eff_timeout = max(float(timeout_s), 75.0 if self.compute_mode == "cpu" else (55.0 if self.compute_mode == "hybrid" else 20.0))
        opts = resolve_ollama_compute_options(
            compute_mode=self.compute_mode,
            model_name=self.active_model,
            num_ctx=2048,
            num_predict=max_tokens,
            temperature=0.45,
            hybrid_gpu_layers=self.hybrid_gpu_layers,
            cpu_threads=self.cpu_threads,
        )
        for include_think_false in (True, False):
            payload: Dict[str, Any] = {
                "model": self.active_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                "stream": False,
                "keep_alive": "20m",
                "options": opts,
            }
            if include_think_false:
                payload["think"] = False
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                    res = json.loads(resp.read().decode("utf-8", errors="ignore"))
                msg = (res.get("message") or {}).get("content", "").strip()
                if msg:
                    return True, msg
            except urllib.error.HTTPError as e:
                if include_think_false and e.code == 400:
                    continue
                return False, f"Ollama HTTPエラー ({e.code})"
            except Exception as e:
                return False, f"Ollama通信失敗: {e}"
        return False, "Ollamaから空の応答が返されました。"


# ==============================================================================
# 2. AivisSpeech / VOICEVOX Local Neural TTS Client
# ==============================================================================
class LocalSpeechTTSClient:
    """
    Auto-detects AivisSpeech (`127.0.0.1:10101`) or VOICEVOX (`127.0.0.1:50021`)
    and synthesizes spoken Japanese WAV bytes for Unity or desktop playback.
    """

    DEFAULT_ENDPOINTS = (
        ("AivisSpeech", "http://127.0.0.1:10101"),
        ("VOICEVOX", "http://127.0.0.1:50021"),
    )

    def __init__(self, custom_url: str = ""):
        self.custom_url = custom_url.rstrip("/") if custom_url else ""
        self.engine_name = ""
        self.active_url = ""
        self.speaker_id = 0

    def probe_tts_server(self, timeout_s: float = 1.2) -> Tuple[bool, str, int]:
        candidates = [("CustomTTS", self.custom_url)] if self.custom_url else list(self.DEFAULT_ENDPOINTS)
        for name, url in candidates:
            if not url:
                continue
            try:
                req = urllib.request.Request(f"{url}/speakers", method="GET")
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    speakers = json.loads(resp.read().decode("utf-8", errors="ignore"))
                if isinstance(speakers, list) and len(speakers) > 0:
                    styles = speakers[0].get("styles") or []
                    sid = int(styles[0].get("id", 0)) if styles else 0
                    self.engine_name = name
                    self.active_url = url
                    self.speaker_id = sid
                    return True, name, sid
            except Exception:
                continue
        return False, "", -1

    def synthesize_wav(self, text: str, speed_scale: float = 1.08, timeout_s: float = 10.0) -> Optional[bytes]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        if not self.active_url:
            ok, _, _ = self.probe_tts_server()
            if not ok:
                return None
        try:
            q_params = urllib.parse.urlencode({"text": cleaned, "speaker": self.speaker_id})
            q_req = urllib.request.Request(f"{self.active_url}/audio_query?{q_params}", data=b"", method="POST")
            with urllib.request.urlopen(q_req, timeout=timeout_s) as q_resp:
                query_json = json.loads(q_resp.read().decode("utf-8"))
            query_json["speedScale"] = float(speed_scale)

            s_params = urllib.parse.urlencode({"speaker": self.speaker_id})
            s_req = urllib.request.Request(
                f"{self.active_url}/synthesis?{s_params}",
                data=json.dumps(query_json).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(s_req, timeout=timeout_s) as s_resp:
                return s_resp.read()
        except Exception:
            return None


# ==============================================================================
# 3. faster-whisper Push-to-Talk USB Microphone JSONL Sidecar for Unity
# ==============================================================================
def _emit_jsonl(obj: Dict[str, Any]) -> None:
    # `ensure_ascii=True` emits `\uXXXX` escapes so Windows pipe codepages NEVER garble Japanese!
    sys.stdout.write(json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _select_best_input_device(sd_mod: Any) -> Tuple[Optional[int], str]:
    """
    Auto-select the best physical microphone (prioritizing USB microphones, then standard mics,
    while avoiding loopback Stereo Mix). Called fresh on each recording so hot-plugged USB mics work!
    """
    try:
        devs = sd_mod.query_devices()
    except Exception:
        return None, "default"

    usb_cands: List[Tuple[int, str]] = []
    mic_cands: List[Tuple[int, str]] = []
    other_cands: List[Tuple[int, str]] = []
    for idx, d in enumerate(devs):
        if int(d.get("max_input_channels", 0)) <= 0:
            continue
        nm = str(d.get("name", ""))
        low = nm.lower()
        if "ステレオ ミキサー" in nm or "stereo mix" in low or "loopback" in low:
            continue
        if "usb" in low:
            usb_cands.append((idx, nm))
        elif "マイク" in nm or "mic" in low or "headset" in low:
            mic_cands.append((idx, nm))
        else:
            other_cands.append((idx, nm))

    if usb_cands:
        return usb_cands[0]
    if mic_cands:
        return mic_cands[0]
    if other_cands:
        return other_cands[0]
    return None, "default"


def run_stt_jsonl_sidecar(model_size: str = "small", language: str = "ja", device_Preference: str = "cpu", parent_pid: int = 0) -> int:
    """
    Interactive stdin/stdout JSONL daemon for Unity Push-to-Talk (`faster-whisper`):
      Commands on stdin : `{"cmd": "start_record"}` | `{"cmd": "stop_record"}` | `{"cmd": "ping"}` | `{"cmd": "quit"}`
      Events on stdout  : `{"event": "ready", ...}` | `{"event": "transcript", "text": "..."}` | `{"event": "error", ...}`
      Defaults to `device="cpu"` (`int8`) so GPU VRAM usage is 0 MB alongside Unity + Brain + Ollama 12B!
    """
    stop_flag = threading.Event()
    if parent_pid > 0:
        start_parent_and_stdin_watchdog(parent_pid, lambda: os._exit(0), watch_stdin=False)

    try:
        import sounddevice as sd
        from faster_whisper import WhisperModel
    except ImportError as exc:
        _emit_jsonl({"event": "error", "detail": f"Missing STT dependency (`faster-whisper` / `sounddevice`): {exc}"})
        return 1

    dev = (device_Preference or "cpu").strip().lower()
    if dev == "auto":
        try:
            import ctranslate2
            dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            dev = "cpu"
    compute_type = "float16" if dev == "cuda" else "int8"

    try:
        whisper = WhisperModel(model_size, device=dev, compute_type=compute_type)
    except Exception:
        dev = "cpu"
        whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

    sample_rate = 16000
    frames_lock = threading.Lock()
    recorded_chunks: List[np.ndarray] = []
    is_recording = False
    stream = None
    domain_hint = "ブリッツコア、コア搬送、塗り広げ、中央ルート、左の高台、右の高台、スカイブリッジ、ダイブ潜伏、防衛迎撃、打開、コア前進、慎重に守る"

    def _audio_cb(indata, frames, time_info, status) -> None:
        if is_recording:
            with frames_lock:
                recorded_chunks.append(indata[:, 0].copy())

    def _close_stream() -> None:
        nonlocal stream
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            stream = None

    _, init_mic_name = _select_best_input_device(sd)
    _emit_jsonl({"event": "ready", "model": model_size, "device": dev, "language": language, "mic": init_mic_name})

    for raw_line in sys.stdin:
        if stop_flag.is_set():
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            cmd_obj = json.loads(line)
        except Exception:
            continue
        cmd = str(cmd_obj.get("cmd", "")).lower()

        if cmd == "ping":
            _emit_jsonl({"event": "pong", "recording": is_recording})
        elif cmd == "quit":
            break
        elif cmd == "start_record":
            _close_stream()
            with frames_lock:
                recorded_chunks.clear()
            dev_idx, mic_name = _select_best_input_device(sd)
            try:
                stream = sd.InputStream(
                    device=dev_idx,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=_audio_cb,
                )
                stream.start()
            except Exception as exc:
                stream = None
                _emit_jsonl({"event": "error", "detail": f"マイクを開けませんでした ({mic_name}): {exc}"})
                continue
            is_recording = True
            _emit_jsonl({"event": "recording_started", "mic": mic_name})
        elif cmd == "stop_record":
            is_recording = False
            _close_stream()
            with frames_lock:
                audio_np = np.concatenate(recorded_chunks, axis=0) if recorded_chunks else np.zeros((0,), dtype=np.float32)
                recorded_chunks.clear()
            if audio_np.size < sample_rate // 5:
                _emit_jsonl({"event": "transcript", "text": "", "duration_s": 0.0})
                continue
            t0 = time.perf_counter()
            segments, _ = whisper.transcribe(
                audio_np,
                language=language,
                beam_size=3,
                vad_filter=True,
                initial_prompt=domain_hint,
            )
            text = "".join(seg.text for seg in segments).strip()
            _emit_jsonl({
                "event": "transcript",
                "text": text,
                "duration_s": round(float(audio_np.size) / float(sample_rate), 2),
                "infer_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            })

    _close_stream()
    return 0


def run_companion_self_test() -> bool:
    print("=" * 72)
    print("  Companion Voice & Ollama Analyst Utility — Self-Test")
    print("=" * 72)
    rec = MatchTimelineRecorder()
    rec.record_event(60, "TURF_EXPAND", "中央エリアの味方塗りを拡大", friendly_turf=0.34, obj_progress=0.50, inner_speech="◆04·塗拡·前進")
    rec.record_event(180, "OBJ_PICKUP", "中央目標(コア)を確保し北台座へ前進", friendly_turf=0.48, obj_progress=0.68, inner_speech="◆11·目標確保·前進護衛")
    rec.record_event(240, "SURVIVAL_VETO", "敵射線を検知し遮蔽へ回避", friendly_turf=0.46, obj_progress=0.68, inner_speech="◆19·緊急離脱·遮蔽")
    prompt = rec.build_japanese_debrief_prompt(final_result="勝利 (KO)")
    assert "最大味方塗率=48.0%" in prompt and "T0180 [OBJ_PICKUP]" in prompt

    ollama = OllamaMatchAnalystClient()
    o_ok, o_models, o_chosen = ollama.probe_available_models(timeout_s=0.8)
    tts = LocalSpeechTTSClient()
    t_ok, t_name, t_sid = tts.probe_tts_server(timeout_s=0.8)

    print(f"[PASS] Timeline Digest Builder Verified ({len(prompt)} chars).")
    print(f"[INFO] Optional Local Ollama Probe : online={o_ok}, chosen='{o_chosen}', count={len(o_models)}")
    print(f"[INFO] Optional Local TTS Probe    : online={t_ok}, engine='{t_name}', speaker={t_sid}")
    print("=" * 72)
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Ollama Match Analyst, AivisSpeech/VOICEVOX TTS & faster-whisper Sidecar")
    ap.add_argument("--self-test", action="store_true", default=False)
    ap.add_argument("--mode", type=str, default="stt_sidecar", choices=["stt_sidecar", "probe"])
    ap.add_argument("--model", type=str, default="small")
    ap.add_argument("--language", type=str, default="ja")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--parent-pid", type=int, default=0)
    args = ap.parse_args()

    if args.self_test or args.mode == "probe":
        return 0 if run_companion_self_test() else 1
    return run_stt_jsonl_sidecar(args.model, args.language, args.device, args.parent_pid)


if __name__ == "__main__":
    sys.exit(main())
