# Universal Brain 3D (v2.3.1) & InkBlitz 3D

**Closed-Loop Asynchronous Functionalist Whole-Brain Cognitive Architecture for Partially Observable 3D Environments**

[![Version](https://img.shields.io/badge/version-2.3.1-blue.svg)](CITATION.cff)
[![License: GPL v3](https://img.shields.io/badge/Python_Brain-GPLv3-blue.svg)](LICENSE)
[![License: MIT](https://img.shields.io/badge/Unity_Arena-MIT-green.svg)](LICENSE_UNITY_MIT.txt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](requirements.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24470540.svg)](https://doi.org/10.5281/zenodo.24470540)

> **日本語要約（Japanese Summary）**
> 本リポジトリは、壁裏やインク潜伏中の敵が見えない「正直な一人称視点（部分観測3D空間）」でリアルタイム（60 Hz）に閉ループ動作する全脳型認知アーキテクチャ **`Universal Brain 3D (v2.3.1)`** と、その検証用にゼロから自作した180度点対称の3D領域塗り＆コア搬送対戦環境 **`InkBlitz 3D`** の研究実装です。
> 本プロジェクトは、HBPのような「神経細胞レベルの物理シミュレート」でも、日本の全脳アーキテクチャ（WBA）のような「生の脳解剖回路に忠実なボトムアップ再現」でも、Google DeepMindのような「個別認知モジュールや単一巨大方策の追求」でもなく、**WBA的な『多時間スケールの全脳機能分業』の枠組みの中に、DeepMind系を中心とする実績ある機械学習部品（`D3QN`・`RSSM`・`Slot Attention`・`PBT/PFSP` 等）を安全制約付きで組み込んだ「実用的な中間アプローチ」** を採っています。
> OSのように分離された7つの非同期プロセス（単一GPU専有1＋CPU6）の上で、**①海馬の空間地図記憶（EKF-SLAM）**、**②小脳の3D偏差射撃とSystem 1の高速反射（60/24 Hz）**、**③世界モデルによる3手先の反実仮想予測と大域ワークスペース（GNWT）の意識統合（10–12 Hz）**、**④自己モデルと因果統合情報量 $\Phi_G$ のリアルタイム計測**、**⑤運動系から安全隔離された内言（`Plan-A`）と、`v2.3.1` で拡張されたヴィゴツキー型二重ループ10-bit構成的プロト言語（2v2狭帯域味方無線＋1秒遅延自己メモ＋Live Rosetta Stone相互情報量解読）および世代間の睡眠学習（S-1〜S-4）** を同時に協調動作させています。**Unity無しのPython単体でも、全7段階の自動検証スイート（`--mode verify_all`）や2v2遮蔽無線実験がそのまま完結します。**

<img width="1920" height="1014" alt="ScreenshotGithub" src="https://github.com/user-attachments/assets/88930389-46e4-445b-b516-2bcb0c68c73c" />


---


## 1. Overview & Design Philosophy

**Universal Brain 3D (v2.3)** is a domain-agnostic cognitive engine built to study how distinct brain-inspired subsystems—spatial memory, fast motor reflexes, world-model imagination, conscious workspace broadcasting, metacognitive self-monitoring, grounded inner speech, and offline sleep—work together inside a **real-time 60 Hz sensorimotor closed loop** under strict 1st-person partial observability (POMDP).
To ground these mechanisms in a demanding dynamic task without relying on commercial game assets, the engine is paired with **InkBlitz 3D**—an original symmetric 3D tactical arena (with a pure-Python headless counterpart) where agents paint territory for high-speed stealth movement and escort a central `BlitzCore` objective.

Rather than claiming ion-channel biological realism or asserting phenomenal consciousness, this project takes a **testable functionalist systems-engineering approach** (Yamakawa, 2021; Butlin et al., 2023):
1. **Whole-System Integration First**: Run all fast and slow cognitive loops concurrently over a fault-tolerant, lock-free multi-process runtime ("a real-time cognitive microkernel") with explicit inter-region interfaces.
2. **Honest 1st-Person Partial Observability**: No omniscient map or wall-penetrating coordinates are ever exposed to the agent; unobserved territory gradually decays back toward uncertainty (`0.0`), and stealth-submerged opponents vanish from vision.
3. **Empirical Falsifiability**: Every cognitive claim—world-model future rollouts, global workspace handoffs, motor-isolated inner speech, and causal integration ($\Phi_G$)—is continuously audited by deterministic self-tests and live telemetry.

### Research Positioning: A Pragmatic Middle Ground

When building a "brain-inspired AI," existing research generally falls into three distinct paradigms—none of which alone satisfies real-time 60 Hz embodied control with modular cognitive transparency on a single workstation. **Universal Brain 3D (v2.3)** intentionally occupies an engineering middle ground:

| Paradigm | Representative Lineage | Core Focus | How Universal Brain 3D (v2.3) Differs |
| :--- | :--- | :--- | :--- |
| **Biophysical Brain Simulation** | Human Brain Project / Blue Brain (Markram et al., 2015) | Ion-channel, compartment, and spiking microcircuit fidelity. | **Not cellular simulation**: We abstract away biophysical spiking to achieve deterministic 60 Hz 3D sensorimotor control on a single consumer GPU. |
| **Anatomical Brain Reference Architecture** | Whole Brain Architecture / WBRA (Yamakawa, 2021) | Mapping software modules closely onto biological mesoscopic subcircuits. | **Looser, loop-first functionalism**: Rather than enforcing strict anatomical circuit isomorphism upfront, we prioritize closing the full 60 Hz sensorimotor–workspace–sleep loop first (treating selective WBRA motifs as a future refinement). |
| **Task-Driven RL & Cognitive Primitives** | Google DeepMind lineage (`DQN`, `Dreamer`, `Slot Attention`, `FTW/PBT`, `AlphaStar`) | High-performing standalone cognitive components or monolithic end-to-end policies. | **Beyond isolated primitives**: Rather than stopping at an individual world model or monolithic RL policy, we orchestrate these proven primitives as asynchronous organs inside a multi-rate whole-brain microkernel. |
| **Pragmatic Functionalist Middle Ground** | **Universal Brain 3D (v2.3)** *(This Project)* | **WBA-inspired macro-organization $\times$ DeepMind-lineage ML primitives** | Combines **multi-timescale whole-brain functional division** (60/24/10 Hz + offline sleep) with **battle-tested deep RL/ML building blocks** inside each region, bound together by explicit safety and epistemic invariants. |

---

## 2. System Architecture (7-Process Lock-Free Bus & `Barrier 1`)

To prevent Python GIL contention and PyTorch CUDA context duplication on single-GPU workstations, the runtime splits into **7 OS processes** communicating over 64-byte-aligned double-buffered `SeqLock + CRC32` shared-memory buses:

```mermaid
flowchart TB
    subgraph ENV["3D Environment (Unity 6 'InkBlitz 3D' via TCP_NODELAY OR Built-in 3D Arena)"]
        U3D["1st-Person UB3D Sensory Stream (60 Hz) / UBMC 6-DoF Motor Stream (60 Hz)"]
    end

    P1["1. WorldBridgeProcess (60 Hz, CPU-Only)<br/>POMDP Packet Parser, CRC32 & Safety Shield"]
    P3["3. SpatialTimingProcess (60 Hz, CPU-Only)<br/>3D MEC Grid, Landmark EKF-SLAM, 4-Layer Turf/Elev Map,<br/>Cerebellar Lead-Aim & Basal Ganglia STN-SNr Gate"]
    P4["4. System1ReflexProcess (24 Hz, CPU-Only)<br/>Fast TT-D3QN Candidate Valuation & Speed/Dive Reflex"]
    P5["5. ConsciousGWTProcess (10-12 Hz, CPU-Only)<br/>Metastable 5-Specialist ActiveGWT, Oizumi-2016 Causal Phi_G,<br/>3-Head RSSM Survival Shield & RVQ Proto-Symbols"]
    P6["6. LanguageCortexProcess (Event, CPU-Only)<br/>Plan-A Grounded Inner Speech + Optional Strategic Analyst"]
    P7["7. TelemetryConsoleProc (15 Hz, CPU-Only)<br/>Live UBNT / UBNC / UBND / UBWM Diagnostic Server"]

    subgraph GPU["Strict Barrier 1 Boundary (ONLY Process #2 Imports PyTorch / CUDA)"]
        P2["2. GPUInferenceProcess / PopGPUServer<br/>14 Neural Models + Batched RPC + NaN/Q-Spike Rollback + Sleep S-1..S-4"]
    end

    U3D <-->|"60 Hz Lockstep / Async"| P1
    P1 <-->|"SeqLock Bus"| P3
    P3 <-->|"Auth 8x6 Poses & 24D Compass"| P4
    P3 <-->|"Spatial & Vigilance Loop"| P5
    P4 <-->|"RPC Queue"| P2
    P5 <-->|"RPC Queue"| P2
    P5 -->|"Read-Only Workspace State"| P6
    P3 & P4 & P5 & P6 -->|"Lock-Free Read"| P7
```

### Architectural Invariants & Fault Tolerance
- **Strict `Barrier 1` Isolation**: Only `GPUInferenceProcess` (or `PopGPUServer` during parallel self-play) initializes `torch`. All CPU processes assert `"torch" not in sys.modules`, serializing IPC weight snapshots strictly as `numpy.ndarray` to eliminate Windows pagefile exhaustion.
- **100% Anchor Rollback on Numerical Faults**: If an online gradient step or extreme reward spike induces `NaN`/`Inf` or a Q-value spike exceeding safety bounds, `WholeBrainNeuralEngineWrapper` immediately restores the last verified anchor weights and clears Adam optimizer momentum (`optimizer.state.clear()`).

### Repository File Structure
* `universal_brain_3d_core.py`: Main CLI entry point, 7-process orchestrator, and 7-stage `verify_all` self-test suite.
* `brain3d_config_ipc.py`: Zero-PyTorch (`Barrier 1`) config, `SeqLock + CRC32` shared-memory buses, `UB3D`/`UBMC` wire protocol, and `10-bit` `@V1:r,s,i` radio packet helpers.
* `brain3d_torch_models.py`: PyTorch definitions for all 14 neural modules (Retinotopic/Stereo/Slot vision, `SpatialTopologicalGWTAdapter3D`, `SlotConceptLanguageGrounder3D`, 3-Head Categorical RSSM, `ActiveGWTWorkspace3D`, `CoarseSelfModel3D`, `StrategicTransformer3D`, `TT-D3QN`, `EndogenousBrocaWernickeCortex`, and `v2.3.1` `VygotskianProtolangCortex3D`).
* `brain3d_neural_engine.py`: Single-GPU batched RPC wrapper, dual-loop Wernicke epistemic integration (`Phi_comm`), `NaN`/Q-spike anchor rollback, and 4-phase offline sleep (`S-1..S-4`).
* `brain3d_spatial_timing.py`: 60 Hz 3D grid/HD cells, stationary-landmark EKF-SLAM, `48x48` 4-layer map, cerebellar 3D lead-aim, and `STN-SNr` timing gate.
* `brain3d_cognitive_procs.py`: CPU worker loops for `System1ReflexProcess`, `ConsciousGWTProcess`, `LanguageCortexProcess`, and live telemetry servers.
* `brain3d_arena_env.py` & `task_profiles_3d.py`: Pure-Python built-in 3D tactical arena and domain-agnostic embodiment profiles (`turf_objective`, etc.).
* `arena_pbt_trainer.py`: Single-GPU (`PopGPUServer`) 180-degree symmetric self-play league (DAgger/autonomous curriculum, PFSP matchmaking, Elo tracking, `--spectate-2v2` 4-brain radio spectate, and inter-generation sleep).
* `vygotsky_rosetta_2v2.py`: Zero-PyTorch (`Barrier 1`) 2v2 occluded twin-flank narrowband radio arena (`HeadToHead2v2Tactical3DArena`) and empirical mutual-information / PMI lexicon decoder (`LiveRosettaStoneProbe`).
* `native_accel/`: Optional Zero-PyTorch C++ (`/O2`) native spatial/foveal raycast acceleration kernels (`fast_brain3d_kernels.cpp`, `fast_kernels_bridge.py`) with automatic pure-NumPy fallback.
* `strategy_knowledge_engine.py` & `companion_voice_io.py`: Optional strategic knowledge distillation and read-only voice/commentary utilities.
* `generate_sfx_pack.py`: Procedural 48 kHz mathematical synthesizer generating all original game audio from scratch.
* `unity_csharp_scripts/` & `unity_bridge_template/`: MIT-licensed 60 Hz TCP sensorimotor bridge, `InkBlitz 3D` 1v1/2v2 arena mechanics, deterministic Editor scene builder (`InkBlitzPhase1Builder.cs`), and real-time neural & Rosetta Stone HUDs.

---

## 3. Whole-Brain Architecture: Step-by-Step Cognitive Flow

Standard deep reinforcement learning typically compresses perception and control into a single monolithic network (`observation -> network -> action`). When fast survival reflexes (16 ms) and deeper multi-step planning (100 ms) are forced into one synchronous loop, the agent either stalls while "thinking" or sacrifices foresight for speed.

**Universal Brain 3D (v2.3)** resolves this by organizing cognition into **multi-timescale functional brain regions** that run concurrently and close the loop from 1st-person sensation to motor execution and offline sleep:

```text
[Stage 1: 1st-Person Senses (60 Hz)]
       │  Line-of-sight entities, 16x16 foveal rays, directional audio, 16D body state
       ▼
[Stage 2: Hippocampal-Parietal Spatial Memory (60 Hz)]
       │  Landmark-only EKF-SLAM + 48x48 4-layer mental map (obstacles, paint decay, elevation, visits)
       ▼
[Stage 3: 6-DoF Body/Aim, Cerebellum & System-1 Reflex (60 Hz / 24 Hz)]
       │  8 tactical candidates + independent 3D lead-aim + TT-D3QN fast valuation + BG safety brake
       ▼
[Stage 4: System-2 World Model & Global Workspace Broadcast (10-12 Hz)]
       │  3-step counterfactual imagination (Survival Shield) + 5-specialist conscious ignition
       ├──▶ Top-Down Feedback (t-1 -> t): Biases visual gain, object slots & spatial scan vigilance
       ▼
[Stage 5 & 6: Self-Model, Causal Phi_G, Read-Only Inner Speech & Offline Sleep (S-1..S-4)]
```

### Stage 1: Honest 1st-Person Perception (Sight, Sound, & Somatic State — 60 Hz)
Unlike omniscient game AIs that read global coordinates, the brain only receives what its body can physically sense at 60 Hz (`UB3D`):
- **Foveal & Optional Stereo Vision**: A `16x16` raycast foveal patch directly ahead samples floor paint state, walls, and local elevation (expandable to `64x64x2` binocular RGB frames in `UB3D` v2).
- **Strict Line-of-Sight Entities**: Up to 32 semantic objects within the camera frustum that pass raycast occlusion checks. Opponents hiding behind cover or submerged in their own ink **completely vanish from visual input**, leaving only close-range splash cues on the directional audio channel (`audio_dir`).
- **16D Proprioception (`proprio`)**: Internal bodily state including health, ink reserve, surface/stealth state, objective progress, and mid-episode respawn status.

### Stage 2: Hippocampal-Parietal Spatial Memory & Self-Localization (60 Hz)
Because 1st-person vision covers only a narrow cone, `SpatialTimingProcess` continuously integrates fleeting glimpses into a persistent world map:
- **Stationary-Landmark-Only EKF-SLAM**: Coupling 3D grid/head-direction coding with Extended Kalman Filter SLAM, the brain corrects self-position drift using *only* static geometry (walls, blocks, goal pedestals)—never moving players or the mobile objective—preventing dynamic entities from corrupting localization.
- **4-Layer Allocentric Cognitive Field Map (`48x48`)**: Tracks **(1) obstacle occupancy**, **(2) territory paint valence** (`+1` friendly to `-1` enemy, which gradually decays toward `0.0` epistemic uncertainty when left unobserved so the brain knows its memory has grown stale), **(3) 3D elevation/cover**, and **(4) exploration freshness**.
- **Mid-Episode Respawn Continuity (`flags |= 16`)**: When eliminated and returned to spawn, the brain resets local pose drift and short-term threat traces while **retaining the stage layout and territory map** learned earlier in the match.

### Stage 3: Tactical Candidates, Decoupled 6-DoF Aim, & Fast "System 1" Reflexes (60 / 24 Hz)
Rather than emitting raw, jittery motor noise, the spatial-motor loop structures movement into interpretable tactical options:
- **8 Tactical Action Candidates**: Every cycle, the spatial cortex proposes 8 concrete maneuvers grounded in local geometry (e.g., *advance objective*, *expand unpainted frontier*, *flank via cover*, *strafe-engage*, *dive/evade to friendly ink*, *hold/ambush*) alongside a 24D spatial compass.
- **Decoupled 6-DoF Locomotion vs. Cerebellar Lead-Aim**: Body velocity (`move_xyz`) is completely decoupled from weapon orientation (`aim_yaw, aim_pitch`). The body can retreat into friendly ink while the **Cerebellar kinematic predictor** (`Cerebellar3DKinematicPredictorV2`) independently solves gravity-compensated 3D projectile interception against moving targets (*kiting*).
- **Fast System-1 Valuation (`24 Hz`) & Basal Ganglia Brake (`60 Hz`)**: `System1ReflexProcess` (`TT-D3QN`) rapidly scores all 8 candidates so the body stays responsive even before slow deliberation finishes. Simultaneously, a **Basal Ganglia `STN-SNr` timing gate** acts as a 60 Hz hyperdirect brake against impulsive commitment under hazard—while explicitly exempting active evasion maneuvers so emergency escapes are never blocked (Frank, 2006; Cisek et al., 2009).

### Stage 4: Slow "System 2" World-Model Foresight & Conscious Workspace (`10–12 Hz`)
In parallel with fast reflexes, `ConsciousGWTProcess` performs multi-step imagination and cross-module integration:
- **Counterfactual World Model & Survival Shield**: Using a `32x32` discrete latent state (`horizon=3` categorical RSSM; Hafner et al., 2023), the world model simulates what happens over the next three steps for *each* of the 8 candidate actions—predicting **reward**, **survival probability**, and **tactical advantage**. Counterfactual de-confounding ensures stationary actions in exposed areas do not inherit false survival optimism from successful dodges. If System 1's reflex choice violates the world model's survival threshold, System 2 overrides it and broadcasts live visual/depth predictions (`UBWM`) for verification.
- **Metastable Global Neuronal Workspace (`ActiveGWTWorkspace3D`)**: Five cortical specialists—**Perceptual (`P`)**, **Hippocampal (`I`)**, **World-Model (`W`)**, **Motor/BG (`M`)**, and **Self-Model (`S`)**—compete to "ignite" a shared 128D workspace broadcast (Baars, 1988; Dehaene et al., 2011, 2017). Short-term synaptic habituation prevents any single specialist from monopolizing consciousness, while sudden prediction errors trigger rapid attentional switching.
- **Closed Top-Down Loop (`t-1 -> t`)**: Workspace broadcasts feed back into the next cycle's thalamocortical visual gain, object-centric slot attention queries, and hippocampal scan vigilance—guiding *where to attend and explore next* without fabricating fake obstacles or landmarks into the physical map (Rao & Ballard, 1999; Friston, 2010; Locatello et al., 2020).

### Stage 5: Metacognitive Self-Model & Causal Integration Diagnostics ($\Phi_G$)
To prevent blind overconfidence under unfamiliar dynamics, the architecture monitors its own internal coherence:
- **Higher-Order Self-Model (`CoarseSelfModel3D`)**: Using motor efference copies, the brain predicts its own upcoming attentional focus and bodily state transition (Graziano, 2013; Lau & Rosenthal, 2011). High `self_surprise` (when internal or somatic reality deviates from expectation) dynamically adjusts workspace ignition thresholds, planning horizon, basal-ganglia caution, and exploration temperature.
- **Online Causal Integration ($\Phi_G$)**: Rather than treating "whole-brain integration" purely as a metaphor, the engine continuously measures lagged causal information flow (`t-1 -> t`) across all 15 bipartitions of the 5 cortical specialists (Oizumi, Tsuchiya, & Amari, 2016):

  $$\Phi_G = \min_{\pi} \min_{q \in \mathcal{M}_G(\pi)} D_{\mathrm{KL}}\!\left(p(X^{t-1}, X^t) \,\|\, q(X^{t-1}, X^t)\right)$$

  streaming both $\Phi_G$ and the weakest inter-module link (Minimum Information Partition) live to the diagnostic HUD (`UBNT`).

### Stage 6: Motor-Isolated Inner Speech (`Plan-A`), Vygotskian 10-bit Protolanguage (`v2.3.1`), & 4-Phase Offline Sleep (`S-1..S-4`)
Finally, slower symbolic and consolidation loops handle interpretability, multi-agent epistemic signaling, and long-term memory stabilization:
- **Grounded Inner Speech with Strict Motor Isolation (`Plan-A`)**: `LanguageCortexProcess` quantizes workspace states into discrete proto-symbols (`RVQ`) and grounds short tactical phrases via contrastive learning (`MIL-InfoNCE`; Vygotsky, 1934/1986). By default (`allow_broca_motor_bias = False`), this inner-speech stream has **strictly zero write access to motor logits (`max|Q_on - Q_off| == 0.0`)**—allowing researchers to read what context the brain is attending to without risking self-generated verbal tokens (*"hold position"*) freezing real-time combat control.
- **Vygotskian Dual-Loop 10-bit Protolanguage & 2v2 Narrowband Radio (`v2.3.1`)**: Without relying on any external LLM or hardcoded human dictionary, `VygotskianProtolangCortex3D` compresses 3 distinct cortical streams into a compositional `10-bit` packet `[◈R(8) · ◆S(16) · ▶I(8)]` (`1.0 Hz` rate limit). A **Dual-Loop Wernicke Receiver** decodes both **(A) Social Radio** (`@V1:r,s,i` packets sent by a same-team ally across occluded `Stage 2` twin flanks) and **(B) Temporal Self-Memos** (the agent's own packet delayed by `1.0 s` as a discrete commitment anchor), injecting epistemic context strictly into `StrategicTransformer3D` and `ActiveGWTWorkspace3D` via `apply_zero_gate` (`0.000000` step-0 perturbation on `v2.3.0` checkpoints; zero raw motor-Q interference). Live counterfactual listening ($\Phi_{\mathrm{comm}}$) and empirical symbol meanings are audited continuously by `LiveRosettaStoneProbe`.
- **4-Phase Offline Sleep Consolidation (`S-1` to `S-4`)**: Between competitive self-play generations (`arena_pbt_trainer.py`), the GPU server enters an offline sleep cycle: **`S-1` NREM prioritized replay** of high-surprise episodes, **`S-2` REM counterfactual dreaming** inside the world model, **`S-3` homeostatic synaptic downscaling** to prevent weight runaway, and **`S-4` cortical policy/schema distillation**.

### Dynamic Cognitive Arbitration During Live Play
Depending on real-time tactical risk and prediction error, control authority shifts fluidly across timescales without stalling the 60 Hz motor loop:

| Tactical Situation | Dominant Loop | Active Brain Regions | Behavioral Effect |
| :--- | :--- | :--- | :--- |
| **Clear / Low Risk** | **System 1 Reflex** (`24 Hz`) | `TT-D3QN` + Spatial Compass | Fast, fluid turf expansion and objective approach with zero planning lag. |
| **Line-of-Sight Engagement** | **Sensorimotor + Gate** (`60 Hz`) | Cerebellum + Basal Ganglia `STN-SNr` | Independent 3D ballistic lead-aim (*kiting*) while `STN-SNr` suppresses premature lunges. |
| **Predicted Lethal Trap** | **System 2 Override** (`10–12 Hz`) | 3-Head RSSM (`Survival Shield`) + `ActiveGWT` | World model foresees multi-step elimination risk, vetoes risky System-1 choices, and ignites global re-planning. |
| **Immediate Survival Crisis** | **Active Escape Exemption** (`60 Hz`) | `STN-SNr` Veto Bypass + Stealth Dive | Unlocks basal-ganglia braking exclusively for friendly-ink dive evasion so the agent never freezes under fire. |


---


## 4. Quick Start & Built-in Verification

### Prerequisites
- Python `3.10+` (`numpy>=1.26.0`, `torch>=2.2.0`; CUDA GPU recommended, CPU fallback supported).
- **Unity is NOT required** to run the cognitive engine, built-in 3D tactical arena, or verification suites.

```bash
pip install -r requirements.txt
```

### 1-Command Deterministic Verification Suites (No Unity Required)
```bash
# 7-Stage End-to-End Brain Verification (IPC/CRC32, Barrier 1, SLAM, 7-Proc Live Arena,
# Plan-A Isolation, Multi-Model Gradients, NaN Rollback, Sleep S-1..S-4, & Oizumi-2016 Phi_G)
python -B universal_brain_3d_core.py --mode verify_all

# World-Model Visual/Depth Prediction & Counterfactual Rollout Verification
python -B universal_brain_3d_core.py --mode verify_world_model

# Single-GPU Multi-Slot 180-deg Symmetric Self-Play PBT & Sleep League Verification
python -B arena_pbt_trainer.py --mode verify_pbt

# v2.3.1 2v2 Occluded Twin-Flank Emergent Protolanguage Bootstrap & Rosetta MI / Phi_comm Audit
python -B scripts/train_and_verify_v231_emergent_radio.py
```

### Running Headless Built-in 3D Arena or Launching with `InkBlitz 3D` (Unity)
| Use Case | Command / Procedure |
| :--- | :--- |
| **Interactive GUI Launcher (Recommended with Unity Build)** | Download the Windows 64-bit standalone archive (e.g., `inkblitz_3d_v2_3_0_win64_bundle.zip` or `Build/InkBlitz3D/`) from GitHub Releases or Zenodo, extract `Build/InkBlitz3D/` and `checkpoints/` into the repository root (with `.venv` or `venv` configured), and launch **`InkBlitz3D.exe`** to run **PBT Training**, **1v1 Spectate**, or **2v2 Vygotsky 10-bit Radio Spectate** from the GUI. |
| **Headless Built-in 3D Arena (Pure Python)** | `python -B universal_brain_3d_core.py --mode play --env builtin_3d --task-profile turf_objective` |
| **Manual CLI Unity TCP Session (1v1)** | Start `python -B universal_brain_3d_core.py --mode play --env unity_tcp --port 50055 --task-profile turf_objective`, then connect Unity on `127.0.0.1:50055`. |
| **2v2 4-Brain Narrowband Radio Spectate (`v2.3.1`)** | `python -B arena_pbt_trainer.py --mode spectate --spectate-2v2 --stage-id 2 --env unity_tcp --base-port 50055 --load-weights checkpoints/user_runs/v231_vygotsky_2v2_champion.pt` |
| **Parallel Competitive Self-Play PBT** | `python -B arena_pbt_trainer.py --mode train --env builtin_3d --pop-size 4 --generations 8` |

Key ablation flags on `universal_brain_3d_core.py`: `--timing-mode {0,1,2,3}` (`0=BG-ADAPT, 1=INSTANT, 2=FIX-500ms, 3=CAUTIOUS`), `--spatial-mode {0,1,2,3}`, `--hippo-memory-mode {adaptive,carry_same_stage,reset_always}`, `--visual-mode {raycast,neural_stereo,calibrate}`, `--scaffold-prior-scale <0.0..1.0>`, `--disable-safety-shield`, `--disable-strict-pomdp`, `--allow-broca-motor-bias` (opt-in `Plan-B` direct motor bias test), `--load-weights <path.pt>`.

### Automated Self-Play & Offline Sleep Lifecycle (`arena_pbt_trainer.py`)
Unlike multi-script manual pipelines, training from tabula rasa to a tactical champion runs as one closed evolutionary loop:

```text
Autonomous_PBT_Evolution_Loop/
├── Phase 0: Teacher-Guided Bootstrap (DAgger)
│            └─ Acquires initial 6-DoF locomotion, ink-dive recovery, and ballistic aiming priors
│
├── Phase 1: 180-Deg Rotationally Symmetric Self-Play League (PopGPUServer)
│            ├─ Prioritized Fictitious Self-Play (PFSP) across current & historical checkpoints
│            └─ Side-swapped symmetric evaluation & Elo rating updates
│
├── Phase 2: Inter-Generation Offline Sleep Consolidation (S-1 .. S-4)
│            ├─ S-1 (NREM Replay)     : Prioritized replay of high-surprise & high-regret transitions
│            ├─ S-2 (REM Dreaming)    : Counterfactual 3-step rollouts inside the Categorical RSSM
│            ├─ S-3 (Downscaling)     : Homeostatic synaptic scaling to prevent weight/Q-value runaway
│            └─ S-4 (Distillation)    : Consolidates conscious System-2 discoveries into fast System-1 reflexes
│
└── Phase 3: Champion Promotion & Live Spectate (`pbt_best_champion.pt`)
             └─ Automatic NaN/regression guard -> Promoted to live `InkBlitz 3D` arena & telemetry HUDs
```

---

## 5. The `InkBlitz 3D` Arena & Binary Wire Protocol

**Is Unity mandatory?** **No.** All core cognitive processing, built-in 3D arena physics, and the 7-stage verification suite run entirely in Python + NumPy + PyTorch.

**What is `InkBlitz 3D`, and why pair it with a whole-brain architecture?**
`InkBlitz 3D` is an original, 180-degree rotationally symmetric 3D tactical simulator (`24m x 32m`) combining **territory painting** and **central objective escort (`BlitzCore`)**—designed specifically so no single reflex or static map can solve the game:
1. **Dynamic Turf & True Stealth POMDP**: Shooting coats the floor in team ink. Diving into friendly ink boosts movement speed, regenerates health/ink, and renders the agent **visually invisible** to opponents—forcing the brain to rely on spatial memory, epistemic decay, and directional splash audio rather than raw reactive vision.
2. **3D Elevation, Cover, & Ballistic Physics**: Sloped high grounds, sightline-blocking cover, and gravity-affected projectiles directly test 6-DoF kiting (moving backward while aiming forward) and cerebellar 3D lead prediction, alongside optional `64x64x2` binocular stereo vision (`UB3D` v2).
3. **Tug-of-War Objective (`BlitzCore`) & Live Brain HUDs**: Agents must shatter a central barrier, escort the `BlitzCore` to the enemy pedestal (or intercept enemy carriers), switching fluidly between territory control, stealth flanking, push, and defense—while built-in Unity HUDs visualize the 4-layer mental map, world-model predictions, GWT ignition, and $\Phi_G$ causal cuts in real time.

| Packet Magic | Direction | Size (Bytes) | Role & Payload Summary |
| :--- | :--- | :--- | :--- |
| **`UB3D` (v1 / v2)** | Unity $\to$ Python (60 Hz) | `6,296` / `34,968` | Ego kinematics, 16D somatic `proprio`, `flags` (incl. bit 4 mid-episode respawn), up to 32 LOS-filtered semantic entities, `16x16x3` foveal ray patch (+ stereo `64x64x3x2` RGB in v2), CRC32. |
| **`UBMC`** | Python $\to$ Unity (60 Hz) | `128` | Decoupled 6-DoF target pose (`move_xyz`, `aim_yaw`, `aim_pitch`, `speed_scale`), discrete motor verb, `commit_gate_open`, `hold_reason_id`, grounded inner-speech UTF-8 preview, CRC32. |
| **`UBNT`** | Python $\to$ HUD (15 Hz) | `3,636` | Full 4-layer `48x48` cognitive map, `gwt_w[128]`, `gwt_alpha[5]`, `ignition`, Oizumi-2016 `phi_bits` + 5-bit `mip_cut_mask`, 8-candidate `Q`/policy/survival arrays, EKF-SLAM drift, CRC32. |
| **`UBWM`** | Python $\to$ HUD (15 Hz) | `904` | Live world-model visual verifier: `16x16` next-step prior RGB/salience prediction, absolute prediction-error map, depth MAE, KL surprise, and 8-candidate counterfactual spread, CRC32. |
| **`UBNC` / `UBND`** | Python $\to$ HUD (15 Hz) | `16` / `536` | Dynamic architecture graph topology & live pathway activation diagnostics, CRC32. |

---

## 6. Limitations & Future Research Roadmap (2027+)

### Current Limitations
- **Single-Node Execution**: Designed for single-workstation execution (`Barrier 1` single-GPU server + multi-core CPU processes); distributed multi-node IPC is not yet implemented.
- **Coarse Functionalist Granularity**: Neural modules use rate-coded PyTorch layers (`LayerNorm`, `GELU`, `GRU`, `Transformer`) coupled with analytic EKF-SLAM and ballistic kinematics for reliable 60 Hz control, rather than biophysical spiking dynamics.
- **Default Motor Isolation of Inner Speech (`Plan-A`) & Fixed-Length `10-bit` Protolanguage (`v2.3.1`)**: Direct verbal-to-motor Q-value coupling remains disabled by default to prevent self-referential action freezing; while `v2.3.1` introduces zero-gated epistemic reception of compositional `10-bit` `[◈R·◆S·▶I]` packets (`1 Hz` teammate radio + `1.0 s` self-memo), open-ended variable-length recursive syntax remains future work.

### Future Roadmap (2027+ — Selective WBRA & Biophysical Refinement)
- **Selective WBRA Circuit-Motif Integration**: Stepwise adoption of Whole Brain Reference Architecture (WBRA; Yamakawa, 2021) motifs—such as explicit hippocampal `CA3-CA1` episodic indexing, thalamic `Pulvinar/TRN` precision routing, and multi-neuromodulator gain—via non-breaking shadow evaluation followed by zero-initialized gated integration.
- **Controlled `Plan-B` Prefrontal Modulation**: Safe indirect modulation of high-level goal priors via grounded inner speech under survival-shield constraints.
- **Progressive Spiking / Local-Plasticity Exploration**: Investigating continuous-time circuit motifs and three-factor local plasticity where computationally justified.

---

## 7. Algorithmic Provenance, AI Co-Development & References

### Standing on Existing Deep RL, World-Model & Neuro-Cognitive Foundations
While the **7-process lock-free cognitive microkernel (`Barrier 1`), multi-timescale arbitration, and safety invariants (`Survival Shield`, `STN-SNr` escape exemption, `Plan-A` inner-speech isolation, zero-gated `10-bit` epistemic radio)** are original to **Universal Brain 3D (v2.3.1)**, its internal modules do not reinvent the wheel. Instead, each functional region builds directly upon established algorithms from **Google DeepMind / Google Research**, modern representation learning, classical robotics, and computational neuroscience:

| Brain Subsystem / Loop | Established Building Blocks & Lineage | Architectural Adaptation in Universal Brain 3D (v2.3.1) |
| :--- | :--- | :--- |
| **System-1 Fast Reflex (`24 Hz`) & NREM Replay (`S-1`)** | **DQN** [1], **Double DQN** [2], **Dueling DQN** [3], **Prioritized Experience Replay (PER)** [4] *(DeepMind)*; **Tensor-Train (`TTLinear`)** [5] | `TensorTrainD3QNCandidateNet3D` evaluates 8 spatially grounded 6-DoF tactical maneuvers + continuous dive/speed modulation under 60 Hz basal-ganglia gating. |
| **System-2 World Model (`10–12 Hz`) & REM Dreaming (`S-2`)** | **Categorical RSSM** with 1% unimix discrete latents (*DreamerV2 / DreamerV3* [6, 7], *Google Research / DeepMind*) | Extended into a **3-head (`reward`, pessimistic `survival`, `advantage`) 8-branch parallel simulator (`horizon=3`)** with counterfactual exposure de-confounding and live `UBWM` visual verification. |
| **Object-Centric Vision & Top-Down Attention** | **Slot Attention** [8] *(Google Research)*; **Predictive Coding & Free-Energy** [17, 18] | Asymmetric 4-slot binding (`Slot 0` floor/turf vs. `Slots 1–3` foreground entities) modulated top-down (`t-1 -> t`) by Global Workspace precision spotlights (`topdown_gain_16`). |
| **Prefrontal Episodic Strategy** | **Causal Transformer** [9] *(Google Brain / Research)* | `StrategicTransformer3D` attends over rolling workspace/self/compass history + metacognitive `self_surprise` residuals to steer macro tactical modes and spatial lookahead. |
| **Proto-Symbols, Inner Speech (`Plan-A`) & 2v2 Protolanguage (`v2.3.1`)** | **VQ-VAE / Residual VQ (`RVQ`)** [10, 11], **InfoNCE / MIL-NCE** [12, 13] *(DeepMind / Google)*; **Inner Speech** [28]; **Lewis Signaling & Emergent MARL Comm (`DIAL`)** [36, 37] | Quantizes workspace states into RVQ symbols (`max|Q_on - Q_off| == 0.0` direct motor isolation) plus `v2.3.1` 3-stream `10-bit` `[◈R·◆S·▶I]` dual-loop epistemic signaling (`1 Hz` 2v2 radio + `1.0 s` self-memo) audited via $\Phi_{\mathrm{comm}}$ and `LiveRosettaStoneProbe`. |
| **Competitive League & Teacher Bootstrap** | **Population-Based Training (PBT)** in 3D arenas (*FTW* [14, 15]) & **PFSP** (*AlphaStar* [16]) *(DeepMind)*; **DAgger** [19] | Single-GPU `PopGPUServer` (`Barrier 1`) running 180-deg rotationally symmetric matches, interleaved with **4-phase offline sleep (`S-1..S-4`)** [29, 30] and automatic regression rollback. |
| **Spatial Memory, Subcortical Gate, GNWT & $\Phi_G$** | **MEC Grid Cells & EKF-SLAM** [20, 21]; **BG `STN-SNr` Gate** [22, 23]; **GNWT & Synaptic Depression** [24–26]; **HOT/AST** [31, 32]; **Causal $\Phi_G$** [27] | Stationary-landmark-only SLAM with mid-episode respawn (`flags \|= 16`), active-escape-exempt `STN-SNr` brake, metastable 5-specialist workspace ignition, and live 15-bipartition causal $\Phi_G$. |

### AI Co-Development Disclosure
In accordance with contemporary open-science transparency standards, the author (**Alpha496**) conceived the overall system architecture, neuro-cognitive hypotheses, safety invariants, domain mechanics, and experimental evaluation criteria, while utilizing large language model coding assistants (Google Gemini) as an implementation, refactoring, and differential-verification partner. All architectural invariants and empirical test suites (`verify_all`, `verify_world_model`, `verify_pbt`, and the 23-check Unity mechanics suite) were deterministically executed and validated locally.

### References

#### A. Deep Reinforcement Learning, World Models, Self-Play & Representation Learning
1. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
2. van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with Double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1), 2094–2100.
3. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48, 1995–2003.
4. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *International Conference on Learning Representations (ICLR)*.
5. Novikov, A., Podoprikhin, D., Osokin, A., & Vetrov, D. P. (2015). Tensorizing neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 28.
6. Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). Mastering Atari with discrete world models. *International Conference on Learning Representations (ICLR)*.
7. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models. *arXiv preprint arXiv:2301.04104*.
8. Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., & Kipf, T. (2020). Object-centric learning with Slot Attention. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 11525–11538.
9. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
10. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
11. Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., & Tagliasacchi, M. (2021). SoundStream: An end-to-end neural audio codec. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 30, 495–507.
12. van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with Contrastive Predictive Coding. *arXiv preprint arXiv:1807.03748*.
13. Miech, A., Alayrac, J.-B., Smaira, L., Laptev, I., Sivic, J., & Zisserman, A. (2020). End-to-end learning of visual representations from uncurated instructional videos. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 9879–9889.
14. Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., et al. (2017). Population based training of neural networks. *arXiv preprint arXiv:1711.09846*.
15. Jaderberg, M., Czarnecki, W. M., Dunning, I., Marris, L., Lever, G., et al. (2019). Human-level performance in 3D multiplayer games with population-based reinforcement learning. *Science*, 364(6443), 859–865.
16. Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. *Nature*, 575(7782), 350–354.

#### B. Cognitive Neuroscience, Brain Architecture, Robotics & Consciousness Science
17. Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.
18. Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
19. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15, 627–635.
20. Hafting, T., Fyhn, M., Molden, S., Moser, M.-B., & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. *Nature*, 436(7052), 801–806.
21. Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous localization and mapping: Part I. *IEEE Robotics & Automation Magazine*, 13(2), 99–110.
22. Frank, M. J. (2006). Hold your horses: Impulsivity, deep brain stimulation, and medication in parkinsonism. *Neural Networks*, 19(8), 1120–1136.
23. Cisek, P., Puskas, G. A., & El-Murr, S. (2009). Decisions in changing conditions: The urgency-gating model. *Journal of Neuroscience*, 29(37), 11560–11571.
24. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
25. Dehaene, S., Changeux, J.-P., & Naccache, L. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2), 200–227; Dehaene, S., Lau, H., & Kouider, S. (2017). What is consciousness, and could machines have it? *Science*, 358(6362), 486–492.
26. Tsodyks, M. V., & Markram, H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *Proceedings of the National Academy of Sciences (PNAS)*, 94(2), 719–723.
27. Oizumi, M., Tsuchiya, N., & Amari, S. (2016). Unified framework for information integration based on information geometry. *Proceedings of the National Academy of Sciences (PNAS)*, 113(51), 14817–14822.
28. Vygotsky, L. S. (1934/1986). *Thought and Language*. MIT Press.
29. Diekelmann, S., & Born, J. (2010). The memory function of sleep. *Nature Reviews Neuroscience*, 11(2), 114–126.
30. Tononi, G., & Cirelli, C. (2014). Sleep and the price of plasticity: From synaptic and cellular homeostasis to memory consolidation and integration. *Neuron*, 81(1), 12–34.
31. Lau, H., & Rosenthal, D. (2011). Empirical support for higher-order theories of conscious awareness. *Trends in Cognitive Sciences*, 15(8), 365–373.
32. Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
33. Markram, H., Muller, E., Ramaswamy, S., Reimann, M. W., Abdellah, M., et al. (2015). Reconstruction and simulation of neocortical microcircuitry. *Cell*, 163(2), 456–492.
34. Yamakawa, H. (2021). The whole brain architecture approach: Accelerating the development of artificial general intelligence by referring to the brain. *Neural Networks*, 144, 478–495.
35. Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., et al. (2023). Consciousness in artificial intelligence: Insights from the science of consciousness. *arXiv preprint arXiv:2308.08708*.
36. Lewis, D. (1969). *Convention: A Philosophical Study*. Harvard University Press.
37. Foerster, J., Assael, I. A., de Freitas, N., & Whiteson, S. (2016). Learning to communicate with deep multi-agent reinforcement learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 2137–2145.

---

## 8. Copyright, Licenses, and Intellectual Property Rights

The 3D tactical arena (**InkBlitz 3D**) included in this project is an original research simulator adopting general "territory-painting and objective-escort" game mechanics. It is not an official application of any existing product and has no affiliation whatsoever with any commercial game titles or publishers.

All assets in this environment—including procedural 3D terrain meshes, primitive geometry, shaders, 48 kHz mathematically synthesized sound effects (`generate_sfx_pack.py`), UI code, and the underlying cognitive AI architecture—have been originally created from scratch, containing zero extracted data or third-party commercial game assets.

### Component-Split Open-Source Licensing
Because the Python cognitive engine and the Unity 3D simulator execute as separate OS processes communicating strictly over localhost TCP sockets (`UB3D` / `UBMC`), they are licensed cleanly per component for full compliance with Unity runtime redistribution terms:

* **Python Whole-Brain Cognitive Engine & Trainers (`*.py`)**: Released under the **[GNU General Public License v3.0 only (GPL-3.0-only)](LICENSE)**.
* **Unity 3D Arena C# Scripts & *InkBlitz 3D* Environment (`*.cs` & original scene/assets)**: Released under the permissive **[MIT License](LICENSE_UNITY_MIT.txt)**, permitting clean linkage and redistribution alongside the proprietary Unity Engine Runtime without copyleft propagation.

Runtime engines and external open-source libraries remain governed by their respective licenses and trademark owners:

* **Unity** (including bundled `UnityPlayer.dll` runtime components in standalone builds, © Unity Technologies) is a trademark or registered trademark of Unity Technologies or its affiliates.
* **PyTorch** is a trademark of The Linux Foundation (distributed under a BSD-style license).
* **NumPy** is distributed under the BSD 3-Clause License.

### Why GPL-3.0 for the Cognitive Core (and MIT for the 3D Arena)?
* **Open Evolution of Whole-Brain Architectures**: Licensing the Python cognitive engine under **GPL-3.0-only** ensures that improvements to its multi-process workspace, counterfactual world model, and sleep consolidation remain open to the research community rather than being locked into proprietary forks, while providing explicit GPLv3 patent protection.
* **Clean Simulator Interoperability**: Licensing the decoupled Unity C# simulator (`InkBlitz 3D`) under the **MIT License** allows anyone to build, inspect, and redistribute standalone 3D binaries legally alongside the Unity runtime.

### Scientific Stance on "Consciousness" Terminology
Terms such as *Global Neuronal Workspace (`GWT`)*, *Ignition*, *Self-Model*, *Inner Speech*, *Sleep*, and *Integrated Information ($\Phi_G$)* are used strictly in their **computational and functionalist senses**—denoting testable information-routing, predictive, and memory-consolidation mechanisms. The author makes **no claim** that this digital software system instantiates biological sentience or subjective phenomenal experience (*qualia*).
