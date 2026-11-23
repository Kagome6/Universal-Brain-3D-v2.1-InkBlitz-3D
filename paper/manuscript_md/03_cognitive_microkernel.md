## 3. Real-Time Cognitive Microkernel (`Barrier 1` & Multi-Rate IPC)

### 3.1 Addressing Pathology P1: Seven-Process Decomposition & `Barrier 1`

Running 60 Hz socket I/O, Dijkstra grid pathfinding, neural world-model rollouts, and causal information-geometry matrix inversions inside a single Python process produces severe latency spikes under the CPython Global Interpreter Lock (GIL). Conversely, importing `torch` across multiple `spawn` worker processes on Windows initializes separate CUDA runtime contexts (`~0.8–1.5 GB` virtual commit per process), rapidly exhausting OS pagefile memory during multi-agent training.

Universal Brain 3D resolves both constraints by organizing the runtime as a **7-process real-time cognitive microkernel** subject to a strict process-level invariant (**`Barrier 1`**):

> **Invariant 1 (`Barrier 1` Single-GPU Isolation):** Across the entire process tree, *only* Process 2 (`GPUInferenceProcess` in single-brain mode, or `PopGPUServer` in multi-arena PBT mode) is permitted to import `torch` or touch CUDA. Processes 1, 3, 4, 5, 6, and 7 execute in pure NumPy/Ctypes and enforce `assert "torch" not in sys.modules` at startup and shutdown. All cross-process weight transfers serialize exclusively as host `numpy.ndarray` dictionaries (`get_weights_state_dict(as_numpy=True)`).

| ID & Process Name | Target Rate | `torch` / GPU | Primary Computations & Published Buses |
| :--- | :--- | :--- | :--- |
| **P1. `WorldBridgeProcess`** | $60\text{ Hz}$ | Forbidden | `TCP_NODELAY` `UB3D`/`UBMC` framing or headless physics; publishes `WORLD3D` (`2,624` B); arbitrates `MOTOR_INTENT3D`. |
| **P2. `GPUInferenceProcess`** | Micro-batched RPC ($2\text{ ms}$ window) | **Exclusive Owner** | Hosts all 12 trainable neural modules (`1,053,454` parameters, plus target `TT-D3QN`), per-slot recurrent rings, online N-step/chunk BPTT, anchor rollback, and Sleep `S-1..S-4`. |
| **P3. `SpatialTimingProcess`** | $60\text{ Hz}$ | Forbidden | 3D FCC grid + Fibonacci HD coding, landmark Kalman filter, `48x48` 4-layer map, ballistic cerebellum, 8-candidate generator (`SPATIAL3D`), & `STN-SNr` gate (`TIMING3D`). |
| **P4. `System1ReflexProcess`** | $24\text{ Hz}$ | Forbidden | Queries `infer_s1` RPC; evaluates 8 candidates via `TT-D3QN` + open-space anti-freeze shield; writes `CH_REFLEX_S1` (`260 ms` horizon). |
| **P5. `ConsciousGWTProcess`** | $12\text{ Hz}$ + Event | Forbidden | Queries `infer_gwt` RPC; runs 5-specialist `ActiveGWT`, 3-head `RSSM` survival shield, Oizumi-2016 $\Phi_G$, action homeostasis; writes `COGNITIVE3D` & `CH_CONSCIOUS_GWT`. |
| **P6. `LanguageCortexProcess`** | Event-driven | Forbidden | Queries `infer_broca` RPC; generates `Plan-A` write-isolated grounded inner speech (`LANGUAGE3D`, `968` B) + optional asynchronous strategic retrieval. |
| **P7. `TelemetryConsoleProc`** | $15\text{ Hz}$ | Forbidden | Wait-free reader exporting `UBNT` (`3,636` B), `UBWM` (`904` B), `UBND` (`536` B), and `UBNC` (`16` B) diagnostic streams. |

To prevent orphaned background workers when an external simulator terminates abruptly, all processes run a dual Win32 `OpenProcess` parent-PID liveness poller and a non-blocking `PeekNamedPipe` stdin watchdog (`start_parent_and_stdin_watchdog`).

### 3.2 Wait-Free Double-Buffered `SeqLock + CRC32` Shared-Memory Buses

Inter-process state sharing avoids OS mutexes and `multiprocessing.Queue` pickling on the 60 Hz path. Each shared-memory bus (`StatelessSeqLockSlot`) allocates a 64-byte cache-line-aligned header (`magic = 0x42334432`, `version = 2`, monotonic 64-bit `seq`, active buffer index `slot` $\in \{0, 1\}$, `timestamp_ns`, plus six 32-bit single-writer control words at offsets `40..63`) followed by **two** 64-byte-aligned payload+CRC32 slots of stride $S = \lceil (\text{sizeof}(\text{dtype}) + 8)/64 \rceil \times 64$.

A writer publishing sequence $k$ never overwrites the buffer currently pointed to by `prev_slot` $b \in \{0, 1\}$. Instead, it marks the header with an odd sequence $2k - 1$, writes the new struct and its 32-bit IEEE `CRC32` into the opposite slot $1 - b$, and commits an even sequence $2k$ with updated slot index $1 - b$. Readers execute up to 6 lock-free retries (`read_latest`), accepting a frame if and only if:

$$\mathrm{seq}_1 = \mathrm{seq}_2 > 0,\quad (\mathrm{seq}_2 \bmod 2) = 0,\quad \mathrm{slot}_1 = \mathrm{slot}_2,\quad \mathrm{CRC32}(\mathrm{payload}) = \mathrm{crc}_{\mathrm{saved}}. \tag{3}$$

With respect to hardware memory models [@boehm2012seqlock], Python `ctypes`/`numpy` shared-memory stores do not issue explicit C11 acquire/release fences; on our target x86-64 architecture, Total Store Order (TSO) preserves program store-store and load-load ordering natively (porting to weakly ordered ARM64 would require explicit `atomic_thread_fence` primitives). Because double-buffering directs consecutive writes to alternating slots and the writer period ($\ge 16.7\text{ ms}$) is far longer than the time a reader needs to copy a frame, and because trailing IEEE `CRC32` detects a random torn read with probability $1 - 2^{-32}$ (`crc_mismatches = 0` across all live runs reported here), Eq. (3) provided wait-free reads without detected corruption in practice. Furthermore, single-writer header words partition control flags across processes without write contention (`atom 2` owned by `TelemetryConsoleProc` for live learning toggles; `atom 3` owned by `ConsciousGWTProcess` for overfit-guard status; `atom 4` owned by `GPUInferenceProcess` for sleep-stage telemetry).

### 3.3 The Shared Candidate Manifold as a Cross-Timescale Contract

A subtle failure in asynchronous layered control [@brooks1986robust] occurs when a slow planner ($12\text{ Hz}$) and a fast reflex ($24\text{ Hz}$) output raw velocity vectors computed at different timestamps: by the time the $12\text{ Hz}$ vector arrives, the agent has moved near an obstacle corner, turning a valid step into a wall collision.

Universal Brain 3D eliminates raw-vector staleness by treating spatially anchored tactical maneuvers—analogous to short-horizon semi-Markov options [@sutton1999options]—as a strict **Candidate Manifold Contract**. Every $16.7\text{ ms}$ ($60\text{ Hz}$), `SpatialTimingProcess` (`P3`) integrates the latest 1st-person pose/map state and publishes an authoritative matrix of $M = 8$ spatially smoothed, obstacle-repulsed 6-DoF candidate poses $\mathbf{P}_t \in \mathbb{R}^{8 \times 6}$, discrete verbs $\mathbf{v}_t \in \{0,\dots,7\}^8$, and a 24D egocentric-to-allocentric memory compass $\mathbf{c}_t \in \mathbb{R}^{24}$ onto `SPATIAL3D`. Both `System1ReflexProcess` (`P4`, $24\text{ Hz}$) and `ConsciousGWTProcess` (`P5`, $12\text{ Hz}$) evaluate the **same** semantic candidate indices (`M0`..`M7`) and publish indexed intents (`chosen_cand_idx` $m^\star \in \{0,\dots,7\}$ plus validity horizon $\Delta \tau_{\mathrm{valid}}$) onto separate channels of `SharedMultiChannelIntent3DBus`. At $60\text{ Hz}$, `WorldBridgeProcess` (`P1`) resolves active motor output by preferring fresh `CH_CONSCIOUS_GWT` intents within their validity window—unless `System 1` triggers an immediate exposed-hazard reflex override—while re-projecting the chosen semantic maneuver index onto the latest $60\text{ Hz}$ obstacle-cleared pose and gating discrete weapon/interaction commitment via `TIMING3D.commit_gate_open`.

### 3.4 External Wire Protocol & Live Verifiability

Communication between `WorldBridgeProcess` / `TelemetryConsoleProc` and external 3D engines (such as Unity 6 `InkBlitz 3D`) uses fixed-size little-endian `TCP_NODELAY` binary frames protected by trailing `CRC32` checksums:
- **Sensory Input (`UB3D`)**: `6,296` bytes in raycast foveal mode (`v1`), or `34,968` bytes in binocular `64x64x2` RGB24 stereo + calibration mode (`v2`). Bit 4 of `flags` (`flags |= 16`) signals mid-episode respawn without episode termination; bits `16..31` encode `stage_id` for structural schema indexing.
- **Motor Output (`UBMC`)**: `128` bytes carrying `target_pose[6]`, independent `gaze_ypd[3]`, `motor_verb`, `commit_gate_open`, `veto_active`, `hold_reason_id`, and a 32-byte UTF-8 grounded inner-speech tag.
- **Diagnostic Streams (`UBNT`, `UBWM`, `UBND`, `UBNC`)**: `UBNT` (`3,636` B) exports all internal neural vectors, candidate $Q$/survival/policy distributions, localization drift, $\Phi_G$, and the 5-bit MIP cut mask at $15\text{ Hz}$; `UBWM` (`904` B) exports live next-frame `16x16` world-model depth/surface predictions, pixel prediction error, depth MAE, and 8-candidate counterfactual spread so external observers can audit whether the world model genuinely predicts upcoming sensory geometry.
