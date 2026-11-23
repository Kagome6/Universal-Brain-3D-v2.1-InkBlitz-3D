# Universal Brain 3D (v2.3.0) & InkBlitz 3D: Asynchronous Multi-Rate Whole-Brain Cognitive Microkernel and Falsifiable Integration Audit

**Author:** Alpha496 (Independent Researcher)  
**Version:** 2.3.0 | **Date:** November 2026 | **Licenses:** GPL-3.0-only (Python Brain) & MIT (Unity 3D Arena)

---

## Abstract

Embodied agents in partially observable 3D environments must reconcile 60 Hz sensorimotor control (16.7 ms) with slower multi-step world-model planning, global workspace routing, and offline consolidation, yet naively wiring heterogeneous cognitive modules into a closed loop introduces severe timing, localization, gradient-interference, saddle-deadlock, and action-freezing pathologies. We present **Universal Brain 3D (v2.3.0)**, a domain-agnostic 60 Hz single-GPU cognitive microkernel governed by five formal cross-module coupling contracts, together with **InkBlitz 3D**, an original two-stage (`Stage 1: Classic Citadel` and `Stage 2: Occluded Twin-Flank`) 180-degree symmetric 3D territory-painting and core-escort POMDP simulator. Across seven OS processes connected by wait-free `SeqLock + CRC32` shared memory under a strict single-GPU isolation invariant (`Barrier 1`, 74.1 MB VRAM for 8 concurrent brains, 2.15 ms p99 spatial/gating latency with Zero-PyTorch C++ `/O2` kernels, 0 CRC32 errors), the architecture couples established primitives: stationary-landmark Kalman localization, a 4-layer allocentric spatial map with mid-episode respawn continuity, `TT-D3QN` fast candidate reflexes, a 3-head (`horizon=3`) categorical `RSSM` with Dreamer-style encoder ownership and counterfactual exposure de-confounding, a metastable 5-specialist `ActiveGWT` workspace with straight-through zero-gated (`apply_zero_gate`) top-down feedback, online lagged causal integrated information ($\Phi_G$), `Plan-A` write-isolated inner speech, and `PBT` self-play with progressive `Scaffold Annealing` ($\lambda_{\mathrm{scaf}}: 1.0 \to 0.0$) and 4-phase sleep (`S-1..S-4`). Across $K = 3$ independent self-play runs (`3,440–3,472` gradient updates/seed, 0 NaN rollbacks), 100% autonomous neural policies ($\lambda_{\mathrm{scaf}} = 0.0$) defeat untrained weights (`68.8%` on Stage 1; `86.1% +/- 4.3%` on Stage 2), match the full hand-crafted `L2` heuristic on open Stage 1 (`53.5%`), and significantly outperform the `L2` heuristic on centrally occluded Stage 2 (`61.1% +/- 12.9%`, $p = 0.0094$; primary seed `79.2%`, $p = 0.00012$), while our audit transparently documents remaining open-lane, calibration, and offline-sleep boundaries.

---

## 1. Research Positioning & Five Cross-Module Coupling Contracts

Rather than claiming novelty for any standalone neural layer or asserting biological sentience, **Universal Brain 3D (v2.3.0)** targets the functionalist middle ground between Whole Brain Architecture (Yamakawa, 2021) and established deep RL primitives (Hafner et al., 2025; Locatello et al., 2020; Goyal et al., 2022). Placing heterogeneous modules into a live 60 Hz 3D loop exposes five integration pathologies, resolved by five formal contracts:

1. **Contract P1 (Multi-Rate Microkernel, `Barrier 1`, & Shared 8-Candidate Manifold)**: Only Process 2 (`GPUInferenceProcess` / `PopGPUServer`, `1,054,099` trainable parameters across 12 modules) imports PyTorch/CUDA; all other processes enforce `"torch" not in sys.modules` and exchange 64-byte-aligned double-buffered `SeqLock + CRC32` frames over a shared `M = 8` obstacle-cleared 6-DoF candidate manifold (`SPATIAL3D`).
2. **Contract P2 (Stationary-Landmark Kalman Filter & Respawn Split `flags |= 16`)**: Self-localization admits only static geometry/pedestals so moving opponents and carried objectives cannot drag pose belief. On mid-episode respawn (`flags |= 16`), ego pose re-anchors to spawn while preserving `48x48` obstacle occupancy, 3D elevation, and decaying territory belief ($\lambda_{\mathrm{surf}} = 0.060\text{ s}^{-1}$).
3. **Contract P3 (World-Model Encoder Ownership, Detached Counterfactuals, Escape-Exempt BG, & `Scaffold Annealing`)**: Perceptual encoders (`foveal_enc`, `slot_cortex`) are owned exclusively by world-model/visual/slot losses (`sg[z_perc]` into GWT/D3QN), eliminating destructive TD-vs-world-model gradient conflict. Encoder-detached counterfactual exposure supervision (`cf_loss`), an escape-exempt `Survival Shield`, $0\text{ ms}$ basal ganglia emergency escape bypass, and progressive `Scaffold Annealing` ($\lambda_{\mathrm{scaf}}: 1.0 \to 0.0$) enable unassisted neural arbitration.
4. **Contract P4 (Metastable GNWT Habituation & Straight-Through Zero-Gates `apply_zero_gate`)**: Broadcast-occupancy fatigue and phasic novelty prevent trained specialist monopoly (`4.17/s` vs. `1.39/s` handoffs), while shadow-tangent straight-through zero-gates awaken all seven $t-1 \to t$ top-down residual pathways from exact zero initialization without mutating physical POMDP maps.
5. **Contract P5 (`Plan-A` Inner-Speech Write-Isolation, Anchor Rollback, & 4-Phase Sleep)**: Endogenous Broca utterances grounded via `MIL-InfoNCE` enforce exact `0.000000` additive motor bias (`Plan-A`), paired with `L2-SP` anchor rollback and inter-generation `S-1..S-4` sleep consolidation.

---

## 2. Seven-Process Real-Time Cognitive Microkernel & Two-Stage Arena

| Process ID & Name | Target Rate | PyTorch / GPU | Core Responsibility & Measured Latency (`RTX 3060`) |
| :--- | :--- | :--- | :--- |
| **1. `WorldBridgeProcess`** | 60 Hz | Forbidden (`CPU`) | `UB3D`/`UBMC` `TCP_NODELAY` wire framing or headless physics; `1.49 ms` p50 (`2.55 ms` p99), `0` CRC errors. |
| **2. `GPUInferenceProcess`** | Batched RPC | **Exclusive Owner** | Hosts 12 trainable PyTorch modules (`1,054,099` params), chunk BPTT, rollback guard, & Sleep `S-1..S-4` (`74.1 MB` for 8 brains). |
| **3. `SpatialTimingProcess`** | 60 Hz | Forbidden (`CPU`) | 3D FCC MEC grid, landmark Kalman SLAM, `48x48` 4-layer map, C++ `/O2` ray/ballistic lead-aim & BG gate (`1.81 ms` p50, `2.15 ms` p99). |
| **4. `System1ReflexProcess`** | 24 Hz | Forbidden (`CPU`) | Fast `TT-D3QN` candidate valuation & open-space anti-freeze reflex (`6.22 ms` p50 at B=1; `7.67 ms` p50 at B=8). |
| **5. `ConsciousGWTProcess`** | 12 Hz | Forbidden (`CPU`) | 5-specialist `ActiveGWT`, 3-head `RSSM` shield, Oizumi-2016 15-cut $\Phi_G$ (`34.26 ms` p50; exceeds 16.67 ms 60 Hz budget). |
| **6. `LanguageCortexProcess`** | Event-Driven | Forbidden (`CPU`) | `Plan-A` write-isolated grounded inner speech (`LANGUAGE3D`, 968 B) + asynchronous strategic retrieval. |
| **7. `TelemetryConsoleProc`** | 15 Hz | Forbidden (`CPU`) | Wait-free streamer for `UBNT` (3,636 B), `UBWM` (904 B), `UBND` (536 B), & `UBNC` (16 B) live diagnostics. |

**Two-Stage Clean-Room 3D POMDP (`InkBlitz 3D`)**: Implemented natively in both Unity 6 and deterministic pure-NumPy/C++ headless simulation, `InkBlitz 3D` pairs open-sightline **`Stage 1: Classic Citadel`** (`24m x 32m`, 4 symmetric pillars, 9s sprint) with sightline-occluded **`Stage 2: Occluded Twin-Flank`** (`28m x 36m`, staggered central baffles blocking spawn-to-core LOS, diagonal chokepoints, climbable twin lateral catwalks, and dropped-core progress decay).

---

## 3. Online Lagged Causal Integration ($\Phi_G$) & World-Model Verification

To avoid equal-time common-input false positives, `ConsciousGWTProcess` estimates lagged causal integrated information across the 5 specialist macro-variables $X = [P, I, W, M, S] \in \mathbb{R}^5$ via row-wise off-diagonal ridge VAR and 15-bipartition Majorization-Minimization $m$-projection (Oizumi, Tsuchiya, & Amari, 2016):

$$\Phi_G = \min_{\pi = (M_1, M_2)} \min_{q \in \mathcal{M}_G(\pi)} D_{\mathrm{KL}}\!\left(p(X^{t-1}, X^t) \,\|\, q(X^{t-1}, X^t)\right)$$

Across 50-trial Monte Carlo checks, debiased off-diagonal ridge scores `0.000 bits` on 50/50 white and AR(1) common-input nulls (where OLS scores `0.274 bits` and isotropic ridge scores `0.051–0.061 bits`) and recovers exact `2+3` block cuts in `70.0%` of 48-step windows. In live 1v1 matches (`N = 24` episodes), trained `v2.3.0` networks lift $\tilde{\Phi}_G$ from `0.04 +/- 0.11 x 10^-3 bits` (untrained) to **`9.38 +/- 6.63 x 10^-3 bits` ($p = 2.4 \times 10^{-7}$)**, while Dreamer-style encoder ownership cuts next-step `UBWM` visual MAE from `0.2138` to **`0.0934` ($p = 1.2 \times 10^{-7}$)** and 3D depth MAE fourfold from `5.88 m` to **`1.45 m` ($p = 1.2 \times 10^{-7}$)**.

---

## 4. Multi-Seed Autonomous Empirical Summary & Honest Limitations

| Benchmark / Audit Suite | Measured Result (`Universal Brain 3D v2.3.0`) |
| :--- | :--- |
| **`K=3` Seeds (`144`g/row) Autonomous (`scaf=0.0`) vs. Untrained** | **Stage 1**: `68.8% [60.4, 77.1]%` ($p = 5.7\times 10^{-5}$) \| **Stage 2**: **`86.1% +/- 4.3%`** (`54-2-16`, $p < 10^{-6}$) |
| **`K=3` Seeds (`144`g/row) Autonomous (`scaf=0.0`) vs. Full `L2` Heuristic** | **Stage 1 (Open)**: `53.5%` ($p = 0.384$, parity) \| **Stage 2 (Occluded)**: **`61.1% +/- 12.9%`** (`27-11-34`, **$p = 0.0094$**; Seed 2026: **`79.2%`**, $p = 0.00012$) |
| **Surgical Autonomous Ablations (`N=24` pairs = `48`g/row)** | `C1` vs. Untrained: **`95.8%`** ($p_{\mathrm{Holm}} = 5\times 10^{-6}$); `C5` vs. No BG Escape Bypass: **`81.2%`** ($p_{\mathrm{Holm}} = 0.0058$); `C7` vs. Zero GWT Broadcast: **`68.8%`** ($p = 0.049$); `C2` vs. No-Map shifts from `47.9%` (open S1) to **`67.7%`** (occluded S2) |
| **Three-Criteria Architectural Audit (`Stage 2`, `scaf=0.0`)** | `0` contract errors across all swaps `S1..S5` & ladder `L0..L5`; `9x9` shared-overlap gradient cosines in **`[-0.055, +0.278]`** (`0` conflicts); `S1` (no D3QN) drops `79.2% -> 64.6%`, `S4` (no loop closure) drops `79.2% -> 68.8%` |
| **Documented Negative & Null Boundaries** | Zero-shot straight-line open-lane `C8` loses to unmapped sprinting (`20.8%`, $p_{\mathrm{Holm}} = 0.0047$); counterfactual Eq. (7) restores `10/10`-seed escape-over-idle ranking (`+0.188` gap) but worsens Brier calibration (`0.151` vs. `0.117`); workspace pooling rules (`S2/S3`) and offline sleep (`S-1..S-4`: `45.0%–52.5%`) show no win-rate gain |
| **Deterministic CLI & Unity Self-Verification** | `verify_all` (**7/7 PASS**), `verify_pbt` (**PASS**), & Unity 6 `InkBlitzMechanicsSelfTest` (**24/24 PASS** across Stage 1 & Stage 2) |

---

## References

1. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
2. Dehaene, S., Lau, H., & Kouider, S. (2017). What is consciousness, and could machines have it? *Science*, 358(6362), 486–492.
3. Butlin, P., Long, R., Elmoznino, E., Bengio, Y., et al. (2023). Consciousness in Artificial Intelligence: Insights from the Science of Consciousness. *arXiv:2308.08708*.
4. Oizumi, M., Tsuchiya, N., & Amari, S. (2016). Unified framework for information integration based on information geometry. *PNAS*, 113(51), 14817–14822.
5. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). Mastering Diverse Control Tasks through World Models. *Nature*, 640, 647–653.
6. Locatello, F., et al. (2020). Object-Centric Learning with Slot Attention. *NeurIPS*, 33, 11525–11538.
7. Goyal, A., et al. (2022). Coordination Among Neural Modules Through a Shared Global Workspace. *ICLR*.
8. Frank, M. J. (2006). Hold your horses: Impulsivity, deep brain stimulation, and medication in parkinsonism. *Neural Networks*, 19(8), 1120–1136.
9. Yu, T., et al. (2020). Gradient Surgery for Multi-Task Learning. *NeurIPS*, 33, 5824–5836.
10. Yamakawa, H. (2021). The whole brain architecture approach: Accelerating the development of artificial general intelligence by referring to the brain. *Neural Networks*, 144, 478–495.
