# Universal Brain 3D (v2.1) & InkBlitz 3D: A Closed-Loop Asynchronous Functionalist Whole-Brain Architecture for Partially Observable 3D Environments

**Author:** Alpha496 (Independent Researcher)  
**Version:** 2.1.0 | **Date:** November 2026 | **License:** GNU General Public License v3.0 (GPL-3.0)

---

## Abstract

Designing embodied agents that integrate spatial mapping, predictive world modeling, action inhibition, global broadcasting, and offline memory consolidation in real time remains an open systems challenge. We present **Universal Brain 3D (v2.1)**, a domain-agnostic, 60 Hz closed-loop functionalist cognitive architecture operating under strict 1st-person partial observability (POMDP), together with **InkBlitz 3D**, an original 180-degree symmetric 3D paint-territory and core-escort research simulator. To achieve deterministic real-time throughput on a single workstation without GPU context contention, the cognitive engine decomposes into seven asynchronous OS processes connected by lock-free `SeqLock + CRC32` shared-memory buses under a strict single-GPU isolation invariant (`Barrier 1`). Key architectural mechanisms established in v2.1 include: (1) a 4-layer allocentric hippocampal field map (`occ`, epistemic-decaying `surface`, `elev`, `visit`) coupled with stationary-landmark EKF-SLAM and mid-episode respawn re-anchoring; (2) decoupled 6-DoF locomotion versus cerebellar 3D ballistic lead-aim gated by a basal ganglia `STN-SNr` hyperdirect brake with active-escape exemption; (3) a 3-head multi-step (`horizon=3`) categorical RSSM world model trained with counterfactual exposure de-confounding and audited via a live visual prediction stream (`UBWM`); (4) a metastable 5-specialist Global Neuronal Workspace (`ActiveGWTWorkspace3D`) stabilized by short-term synaptic adaptation and phasic novelty salience, closing top-down loops into perceptual object slots and hippocampal scan vigilance; (5) online lagged causal integrated information ($\Phi_G$) following Oizumi, Tsuchiya, & Amari (2016) across all 15 bipartitions; and (6) single-GPU Population-Based Training (PBT) self-play paired with 4-phase offline sleep consolidation (`S-1` to `S-4`).

---

## 1. Motivation & Functionalist Stance

Computational theories of cognition and consciousness—such as Global Neuronal Workspace Theory (GNWT; Baars, 1988; Dehaene et al., 2011, 2017), Predictive Coding and Active Inference (Rao & Ballard, 1999; Friston, 2010), Higher-Order / Attention-Schema Theories (Lau & Rosenthal, 2011; Graziano, 2013), and Information Geometry formulations of Integrated Information (Oizumi et al., 2016)—are most often studied in isolated toy benchmarks or static offline datasets. Conversely, high-performing deep reinforcement learning agents typically collapse perception and control into a monolithic feed-forward policy, lacking modular interpretability, explicit epistemic spatial memory, or mid-episode structural re-anchoring.

Following the Whole Brain Architecture engineering philosophy (Yamakawa, 2021) and computational indicator frameworks (Butlin et al., 2023), **Universal Brain 3D (v2.1)** constructs a *modular, falsifiable walking skeleton*: every cortical and subcortical module exchanges typed tensors at biologically inspired multi-rate timescales (60 Hz sensorimotor/subcortical, 24 Hz reflex valuation, 10–12 Hz conscious workspace broadcast, event-driven inner speech, and offline inter-episode sleep), while strictly abstaining from claims of phenomenal sentience.

---

## 2. Asynchronous 7-Process Lock-Free Architecture & `Barrier 1`

Running spatial SLAM, neural world-model rollouts, causal information geometry, and TCP sensorimotor I/O inside a single Python process causes severe frame jitter due to GIL contention, whereas naive multi-process PyTorch spawning duplicates multi-gigabyte CUDA contexts across workers. Universal Brain 3D resolves both bottlenecks through two systems invariants:

| Process ID & Name | Target Rate | PyTorch / GPU | Core Responsibility |
| :--- | :--- | :--- | :--- |
| **1. `WorldBridgeProcess`** | 60 Hz | Forbidden (`CPU`) | `UB3D`/`UBMC` `TCP_NODELAY` binary wire I/O or built-in 3D arena stepping; CRC32 & POMDP safety shield. |
| **2. `GPUInferenceProcess`** | Batched RPC | **Exclusive Owner** | Hosts all 11 PyTorch models (`WholeBrainNeuralEngineWrapper`), online chunk BPTT, NaN rollback, & Sleep `S-1..S-4`. |
| **3. `SpatialTimingProcess`** | 60 Hz | Forbidden (`CPU`) | 3D FCC MEC grid cells, landmark EKF-SLAM, 4-layer `48x48` field map, cerebellar 3D lead-aim, & `STN-SNr` gate. |
| **4. `System1ReflexProcess`** | 24 Hz | Forbidden (`CPU`) | Fast TT-D3QN candidate scoring and continuous speed / ink-dive posture modulation. |
| **5. `ConsciousGWTProcess`** | 10–12 Hz | Forbidden (`CPU`) | Metastable 5-specialist `ActiveGWT`, Oizumi-2016 causal $\Phi_G$, 3-head RSSM survival shield, & RVQ symbols. |
| **6. `LanguageCortexProcess`** | Event-Driven | Forbidden (`CPU`) | `Plan-A` motor-isolated grounded inner speech (`MIL-InfoNCE`) + optional asynchronous strategic analyst. |
| **7. `TelemetryConsoleProc`** | 15 Hz | Forbidden (`CPU`) | Streams `UBNT` (3,636 B), `UBWM` (904 B), `UBNC` (16 B), & `UBND` (536 B) diagnostic packets. |

- **Lock-Free `SeqLock + CRC32` Shared Memory**: Inter-process buses use 64-byte cache-line-aligned double slots with monotonic odd/even sequence counters and CRC32 payload checksums, guaranteeing wait-free 60 Hz reads without torn frames.
- **Numerical Rollback Guard**: Whenever an online optimization step encounters `NaN`/`Inf` parameters or Q-value divergence (`|Q| > Q_MAX_SAFE`), the engine atomically reverts to the last verified anchor checkpoint and purges Adam state buffers.

---

## 3. Allocentric 4-Layer Spatial Memory & Subcortical Control

### 3.1 Honest 1st-Person POMDP & 4-Layer Cognitive Field Map
Sensory packets (`UB3D`) enforce strict line-of-sight (LOS) frustum culling: opponents submerged at low speed in their own ink are excluded from both semantic entity lists and `16x16` foveal ray silhouettes (rays pass through hidden colliders to sample the underlying floor surface). `MultiLayerCognitiveFieldMap3D` maintains a `48x48` allocentric grid over the `24m x 32m` arena:
- **Layer 0 (`occ`)**: Obstacle and wall occupancy belief.
- **Layer 1 (`surface`)**: Territory paint state $\in [-1, +1]$ integrated from 1st-person foveal rays, decaying exponentially toward epistemic uncertainty `0.0` while unobserved.
- **Layer 2 (`elev`)**: Local 3D terrain elevation and cover height.
- **Layer 3 (`visit`)**: Observation recency driving epistemic frontier exploration.

### 3.2 Stationary-Landmark EKF-SLAM & Mid-Episode Respawn (`flags |= 16`)
Self-localization filters dynamic entities via `is_static_landmark_for_slam(tag, speed_mps)` so moving agents and carried objectives never corrupt loop closure. When an agent is eliminated and respawns at base mid-match (`flags |= 16`), EKF pose covariance and transient hazard traces reset cleanly to spawn while preserving the accumulated 4-layer stage and turf map.

### 3.3 Decoupled 6-DoF Strafe/Aim, Ballistic Cerebellum, & Basal Ganglia Gate
Action candidates specify `auth_cand_poses[8, 6] = [move_x, move_y, move_z, aim_yaw, aim_pitch, speed_scale]`. `Cerebellar3DKinematicPredictorV2` solves gravity-compensated 3D projectile interception on moving targets independently of locomotion direction. In parallel, a basal ganglia circuit (`Direct Go` vs. `Indirect NoGo` + hyperdirect `STN-SNr` braking driven by conflict entropy and world/self surprise; Frank, 2006; Cisek et al., 2009) gates shot commitment while granting an immediate escape exemption to survival evasion (`M4`).

---

## 4. Closed-Loop Functional Consciousness, World Model & Causal $\Phi_G$

### 4.1 3-Head Categorical RSSM & Counterfactual De-Confounding
The world model (` RSSMWorldModel3D`, `32x32` discrete stochastic latents + 256D deterministic GRU; Hafner et al., 2023) rolls out `horizon=3` steps for each of the 8 spatial-motor candidates, predicting reward, survival probability, and territorial/objective advantage. Because training solely onfactual trajectories confounds stationary observation (`M7`) with safe periods, a counterfactual open-space freeze loss (`cf_loss`) penalizes optimistic survival estimates when exposed to active threats. Live prior visual/depth predictions and 8-branch counterfactual spreads are exported at 15 Hz via `UBWM` (904 B).

### 4.2 Metastable GNWT Ignition & Bidirectional Cortical Feedback
Five cortical specialists `[P: Perceptual Slots, I: Hippocampal Map, W: World Model, M: Basal Ganglia/Motor, S: Self-Model]` project into `ActiveGWTWorkspace3D`. In naive soft-attention workspaces, a single low-variance module (e.g., `S`) can monopolize broadcast weights indefinitely with saturated ignition ($\approx 0.99$). Version 2.1 resolves this via two biologically grounded mechanisms:
1. **Short-Term Synaptic Adaptation & Phasic Salience**: Pre-softmax competition logits apply Tsodyks-Markram-style occupancy fatigue $-\lambda_{\mathrm{hab}}\max(0, \bar{\alpha}_k - 0.20)\exp(-\gamma\,\mathrm{nov}_k)$ alongside phasic prediction-error salience $+\beta\,\tanh(\mathrm{nov}_k)$, coupled with workspace state-shift ignition $1 - \cos(w_{\mathrm{pool}}^t, w_{\mathrm{GWT}}^{t-1})$. Empirically, this shifts ignition from saturated lock-in (`0.991`, `0` handoffs) to a metastable regime (`mean=0.406`, `57` specialist handoffs per 200 ticks).
2. **Closed $t-1 \to t$ Top-Down Thalamocortical, Slot, & Hippocampal Loops**: Broadcasted workspace states bias next-tick perceptual feature gain (`z_td`), object-centric Slot Attention queries (`td_slot_prior` gated from zero initialization for backward checkpoint compatibility; Locatello et al., 2020), and hippocampal epistemic fog/saccade vigilance around `topdown_focus_xyz`—strictly modulating *epistemic uncertainty and attention* without mutating physical occupancy or SLAM landmarks.

### 4.3 Lagged Causal Integrated Information ($\Phi_G$, Oizumi et al., 2016)
Rather than equal-time correlation mutual information (which suffers from common-input false positives), `ConsciousGWTProcess` estimates lagged causal transition disconnection across $X = [P, I, W, M, S] \in \mathbb{R}^5$ over rolling transition windows $(X^{t-1}, X^t)$:
$$\Phi_G = \min_{\pi = (M_1, M_2)} \min_{q \in \mathcal{M}_G(\pi)} D_{\mathrm{KL}}\!\left(p(X^{t-1}, X^t) \,\|\, q(X^{t-1}, X^t)\right)$$
Using row-wise off-diagonal ridge regularization (`0.14 * (I - e_i e_i^T) + 1e-3 * I`) to suppress common sensory drive leakage, Kitazono-style balanced MIP tie-breaking, and finite-sample Wilks debiasing, $\Phi_G$ scores `0.000 bits` on common-input noise, identifies exact structural splits (`mask=3`), and rises `15.9x` in mean (`0.0254b` vs. `0.0016b`, peak `0.4507b`) in trained vs. untrained live arena agents.

### 4.4 Plan-A Motor-Isolated Inner Speech & 1-GPU PBT + Sleep (`S-1..S-4`)
`EndogenousBrocaWernickeCortex` grounds discrete RVQ workspace tokens via foreground-vs-background MIL-InfoNCE while enforcing `max|Q_on - Q_off| == 0.0` motor isolation (`Plan-A`). Across episodes, `arena_pbt_trainer.py` multiplexes 4 symmetric arenas (8 brains) onto a single `PopGPUServer` with Prioritized Fictitious Self-Play (PFSP), Elo tracking, and 4-phase inter-generation sleep consolidation (`S-1` NREM replay, `S-2` REM counterfactual rollouts, `S-3` synaptic scaling, `S-4` schema distillation).

---

## 5. Deterministic Verification & Empirical Summary

All invariants are reproducible offline without Unity via three automated CLI suites:

| Suite Command | Verified Invariants & Empirical Outputs |
| :--- | :--- |
| `universal_brain_3d_core.py --mode verify_all` | **7/7 Stages PASS**: CRC32/SeqLock atomics; `Barrier 1` (`0` CPU `torch` imports); 4-layer map & respawn `flags \|= 16`; 504-tick 7-process arena; `Plan-A` (`max\|Q_on - Q_off\| == 0.000000`); 11-model gradient flow (`0` missing); NaN/Q-spike 100% rollback (`rb=2`); Oizumi-2016 $\Phi_G$ (`noise=0.000b`, `split=0.048b/mask=3`, `ring=0.39b`). |
| `universal_brain_3d_core.py --mode verify_world_model` | **PASS**: 904B `UBWM` CRC32 verified; trained `pbt_best_champion.pt` achieves `vis_pred_mae = 0.1146` (vs. `0.2026` untrained) and `depth_pred_mae = 1.98m` (vs. `5.23m` untrained); non-zero 8-candidate counterfactual rollout spread (`0.251`). |
| `arena_pbt_trainer.py --mode verify_pbt` | **2/2 Checks PASS**: Exact 180-degree team-symmetry observation/action involution + 2-generation 1-GPU self-play Elo & Sleep `S-1..S-4` convergence. |
| `InkBlitzMechanicsSelfTest` (Unity EditMode) | **23/23 Checks PASS**: Stage 180-deg symmetry, anti-hack first-claim turf reward, tug-of-war `BlitzCore` 280 HP barrier & 0.45s lethal fuse, 60s hold/overtime rules, 4.0s dive stamina & foveal stealth transparency, late-game 3D projectile parry, 48kHz studio SFX, & HUD diagnostics. |

---

## 6. Limitations & Future Roadmap (2027+)

1. **Single-Workstation Scope**: IPC buses use local OS shared memory; multi-node cluster training requires future RDMA/zero-copy transport extensions.
2. **Hybrid Analytic-Neural Granularity**: Subcortical SLAM and ballistic lead-aim intentionally retain analytic geometry alongside rate-coded PyTorch modules to guarantee 60 Hz stability on consumer GPUs.
3. **Selective WBRA & Biophysical Roadmap (2027+)**: Future iterations plan to selectively graft Whole Brain Reference Architecture (WBRA; Yamakawa, 2021) circuit motifs—specifically explicit hippocampal `CA3-CA1` episodic indexing, thalamic `Pulvinar/TRN` precision routing, and multi-neuromodulator gain—via Shadow (`Plan-A`) $\to$ Zero-Gate transitions, followed by controlled evaluation of `Plan-B` verbal goal priors and local-plasticity spiking motifs.

---

## References

1. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
2. Dehaene, S., Changeux, J.-P., & Naccache, L. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2), 200–227.
3. Dehaene, S., Lau, H., & Kouider, S. (2017). What is consciousness, and could machines have it? *Science*, 358(6362), 486–492.
4. Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., et al. (2023). Consciousness in Artificial Intelligence: Insights from the Science of Consciousness. *arXiv preprint arXiv:2308.08708*.
5. Oizumi, M., Tsuchiya, N., & Amari, S. (2016). Unified framework for information integration based on information geometry. *Proceedings of the National Academy of Sciences (PNAS)*, 113(51), 14817–14822.
6. Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.
7. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
8. Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
9. Lau, H., & Rosenthal, D. (2011). Empirical support for higher-order theories of conscious awareness. *Trends in Cognitive Sciences*, 15(8), 365–373.
10. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. *arXiv preprint arXiv:2301.04104*.
11. Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., & Kipf, T. (2020). Object-Centric Learning with Slot Attention. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 11525–11538.
12. Frank, M. J. (2006). Hold your horses: Impulsivity, deep brain stimulation, and medication in parkinsonism. *Neural Networks*, 19(8), 1120–1136.
13. Cisek, P., Puskas, G. A., & El-Murr, S. (2009). Decisions in changing conditions: The urgency-gating model. *Journal of Neuroscience*, 29(37), 11560–11571.
14. Yamakawa, H. (2021). The whole brain architecture approach: Accelerating the development of artificial general intelligence by referring to the brain. *Neural Networks*, 144, 478–495.
15. Vygotsky, L. S. (1934/1986). *Thought and Language*. MIT Press.
