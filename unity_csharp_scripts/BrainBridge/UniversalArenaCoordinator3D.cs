// UniversalArenaCoordinator3D.cs — Simultaneous Multi-Arena & Process Lifecycle Coordinator for Unity 6+
// Supports:
//   1. Single-Brain Play/Spectate Mode (`universal_brain_3d_core.py --mode play --env unity_tcp`)
//   2. Simultaneous N-Arena Parallel PBT & Self-Play (`arena_pbt_trainer.py --mode train --env unity_tcp`)
//      spawning spatially separated Arena prefabs (`arenaIndex * 80m` along X) inside ONE Unity instance!
//   3. Zero-Orphan Subprocess Management (`--parent-pid` + graceful kill on OnApplicationQuit/OnDestroy).
// License: MIT (Unity C# Bridge & InkBlitz 3D Arena; Python Brain Core over TCP is GPL-3.0)

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using UnityEngine;

namespace UniversalBrain3D
{
    public enum BrainSessionRunMode
    {
        SinglePlaySpectate = 0,
        ParallelPbtSolo = 1,
        ParallelCompetitiveSelfPlay = 2,
        SingleArenaSelfPlaySpectate = 3
    }

    public class UniversalArenaCoordinator3D : MonoBehaviour
    {
        [Header("Python Environment Paths")]
        public string pythonExePath = "";
        public string brainWorkspaceDir = "";
        public string taskProfile = "turf_objective";
        public string singlePlayPolicyMode = "pure_neural";
        public string loadCheckpointRelativePath = @"checkpoints\user_runs\pbt_latest_champion.pt";
        public string oppCheckpointRelativePath = "";

        [Header("Execution & Parallel Arena Mode")]
        public BrainSessionRunMode sessionMode = BrainSessionRunMode.SinglePlaySpectate;
        public bool autoLaunchOnStart = false;
        public int baseTcpPort = 50055;
        [Range(1, 8)] public int parallelArenaCount = 4;
        [Range(1f, 8f)] public float trainingTimeScale = 4.0f;
        public float arenaSpacingMeters = 100.0f;
        public GameObject arenaPrefab;

        [Header("Phase 4 PBT Configuration")]
        public int pbtGenerations = 12;
        public int pbtEpisodesPerGen = 2;
        public string pbtPolicyMode = "dagger";
        public bool resumeFromCheckpointInPbt = false;
        public string pbtCheckpointDir = @"checkpoints\user_runs";
        public bool enableSleepConsolidation = true;
        [Range(0, 16)] public int sleepConsolidationEpochs = 6;
        [Range(1, 2)] public int activeStageId = 1; // 1 = Classic Citadel, 2 = Occluded Twin-Flank
        public string hippoMemoryMode = "adaptive"; // "adaptive" | "carry_same_stage" | "reset_always"
        public bool spectateContinualLearning = false;
        public bool spectate2v2BrainRadio = false;
        public bool enableStrategyBroadcast = true;
        public string ollamaComputeMode = "hybrid"; // "hybrid" (GPU+CPU split) | "cpu" (num_gpu=0) | "gpu"
        [Range(1, 10)] public int maxTrainingStepsPerRenderFrame = 2;

        [Header("Live Backend Telemetry")]
        public string latestBackendLog = "";
        public string latestBackendError = "";
        public readonly ConcurrentQueue<string> PbtStatQueue = new ConcurrentQueue<string>();

        private Process _pythonProc;
        private Thread _shutdownThread;
        private readonly List<UniversalBrainBridge3D> _activeBridges = new List<UniversalBrainBridge3D>();
        private float _nextReconnectTime = 0f;
        private float _defaultMaxDeltaTime = 0.3333333f;

        private void Awake()
        {
            if (Time.maximumDeltaTime > 0.1f)
                _defaultMaxDeltaTime = Time.maximumDeltaTime;
            if (QualitySettings.vSyncCount == 0 && Application.targetFrameRate <= 0)
                Application.targetFrameRate = 60;
            EnsureResolvedPaths();
        }

        public bool IsTrainingSession =>
            sessionMode == BrainSessionRunMode.ParallelPbtSolo ||
            sessionMode == BrainSessionRunMode.ParallelCompetitiveSelfPlay;

        public int LastExitCode { get; private set; } = 0;
        public bool IsEmergencyHalted { get; private set; } = false;
        public int EmergencyStopsTriggeredCount { get; private set; } = 0;
        /// <summary>
        /// Tracks PC-friendly shutdown state: "IDLE", "RELEASING_MEMORY", "CLEAN_EXIT", or "FORCE_KILLED".
        /// </summary>
        public volatile string SafeShutdownPhase = "IDLE";

        public bool HasCompletedProcess => _pythonProc != null && SafeHasExited();

        private bool SafeHasExited()
        {
            try { return _pythonProc != null && _pythonProc.HasExited; }
            catch { return false; }
        }

        public int RegisteredBridgeCount => _activeBridges.Count;

        public int ConnectedBridgeCount
        {
            get
            {
                int n = 0;
                for (int i = 0; i < _activeBridges.Count; i++)
                {
                    if (_activeBridges[i] != null && _activeBridges[i].IsConnected)
                        n++;
                }
                return n;
            }
        }

        public bool IsBackendRunning
        {
            get
            {
                try
                {
                    if (_shutdownThread != null && _shutdownThread.IsAlive)
                        return true;
                    return _pythonProc != null && !_pythonProc.HasExited;
                }
                catch { return false; }
            }
        }

        private void Start()
        {
            if (autoLaunchOnStart)
                LaunchPythonBackendAndConnect();
        }

        private void Update()
        {
            if (IsEmergencyHalted)
                return;

            if (_pythonProc != null && SafeHasExited())
            {
                try { LastExitCode = _pythonProc.ExitCode; }
                catch { }
            }

            if (IsBackendRunning && Time.unscaledTime >= _nextReconnectTime)
            {
                _nextReconnectTime = Time.unscaledTime + 0.5f;
                for (int i = 0; i < _activeBridges.Count; i++)
                {
                    var bridge = _activeBridges[i];
                    if (bridge != null && !bridge.IsConnected)
                        bridge.ConnectToBrain(0.08f);
                }
            }
        }

        public static string ResolveDefaultWorkspaceDir()
        {
            try
            {
                var dir = new DirectoryInfo(Application.dataPath);
                for (int i = 0; i < 6 && dir != null; i++)
                {
                    if (File.Exists(Path.Combine(dir.FullName, "arena_pbt_trainer.py")))
                        return dir.FullName;
                    dir = dir.Parent;
                }
            }
            catch { }
            return Directory.GetCurrentDirectory();
        }

        public string GetResolvedWorkspaceDir()
        {
            EnsureResolvedPaths();
            return brainWorkspaceDir;
        }

        public void EnsureResolvedPaths()
        {
            bool workspaceValid = !string.IsNullOrWhiteSpace(brainWorkspaceDir)
                               && Directory.Exists(brainWorkspaceDir)
                               && File.Exists(Path.Combine(brainWorkspaceDir, "arena_pbt_trainer.py"));
            if (!workspaceValid)
            {
                try
                {
                    var dir = new DirectoryInfo(Application.dataPath);
                    for (int i = 0; i < 6 && dir != null; i++)
                    {
                        if (File.Exists(Path.Combine(dir.FullName, "arena_pbt_trainer.py")))
                        {
                            brainWorkspaceDir = dir.FullName;
                            workspaceValid = true;
                            break;
                        }
                        dir = dir.Parent;
                    }
                }
                catch { }
            }

            if (string.IsNullOrWhiteSpace(pythonExePath) || !File.Exists(pythonExePath))
            {
                if (!string.IsNullOrWhiteSpace(brainWorkspaceDir))
                {
                    string v1 = Path.Combine(brainWorkspaceDir, "venv", "Scripts", "python.exe");
                    string v2 = Path.Combine(brainWorkspaceDir, ".venv", "Scripts", "python.exe");
                    if (File.Exists(v1))
                        pythonExePath = v1;
                    else if (File.Exists(v2))
                        pythonExePath = v2;
                }
            }
        }

        private string ResolveCheckpointPath(string relOrAbs)
        {
            if (string.IsNullOrWhiteSpace(relOrAbs))
                return "";
            string trimmed = relOrAbs.Trim();
            string ws = GetResolvedWorkspaceDir();
            return Path.IsPathRooted(trimmed) ? trimmed : Path.Combine(ws, trimmed);
        }

        public void LaunchPythonBackendAndConnect()
        {
            IsEmergencyHalted = false;
            StopPythonBackend();
            SafeShutdownPhase = "IDLE";
            EnsureResolvedPaths();
            while (PbtStatQueue.TryDequeue(out _)) { }
            LastExitCode = 0;
            latestBackendLog = "";
            latestBackendError = "";

            int unityPid = Process.GetCurrentProcess().Id;
            int stageArgVal = Mathf.Clamp(activeStageId, 1, 2);
            string hippoArg = string.IsNullOrWhiteSpace(hippoMemoryMode) ? "adaptive" : hippoMemoryMode.Trim();
            string scriptPath;
            string args;

            string rawOllamaMode = string.IsNullOrWhiteSpace(ollamaComputeMode) ? "hybrid" : ollamaComputeMode.Trim().ToLowerInvariant();
            if (rawOllamaMode != "hybrid" && rawOllamaMode != "cpu" && rawOllamaMode != "gpu")
                rawOllamaMode = "hybrid";
            string stratArg = enableStrategyBroadcast ? $" --ollama-mode {rawOllamaMode}" : " --disable-strategy-kb";
            string contSaveFull = Path.Combine(brainWorkspaceDir, "checkpoints", "user_runs", "spectate_continual_latest.pt");

            if (sessionMode == BrainSessionRunMode.SinglePlaySpectate)
            {
                scriptPath = Path.Combine(brainWorkspaceDir, "universal_brain_3d_core.py");
                string ckptFull = ResolveCheckpointPath(loadCheckpointRelativePath);
                string loadArg = (!string.IsNullOrEmpty(ckptFull) && File.Exists(ckptFull)) ? $" --load-weights \"{ckptFull}\"" : "";
                string modeArg = string.IsNullOrEmpty(singlePlayPolicyMode) ? "pure_neural" : singlePlayPolicyMode;
                string learnArg = spectateContinualLearning
                    ? $" --enable-learning --autosave-per-episode --sleep-epochs 2 --save-weights \"{contSaveFull}\""
                    : " --disable-learning";
                args = $"-B \"{scriptPath}\" --mode play --env unity_tcp --task-profile {taskProfile} --stage-id {stageArgVal} "
                     + $"--policy-mode {modeArg} --hippo-memory-mode {hippoArg}{learnArg}{stratArg} --port {baseTcpPort} --ux-port {baseTcpPort + 100} --parent-pid {unityPid} --watch-stdin{loadArg}";
                Time.timeScale = 1.0f;
                Time.maximumDeltaTime = Time.fixedDeltaTime * 1.25f;
            }
            else if (sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate)
            {
                scriptPath = Path.Combine(brainWorkspaceDir, "arena_pbt_trainer.py");
                string ckptFull = ResolveCheckpointPath(loadCheckpointRelativePath);
                string oppFull = ResolveCheckpointPath(oppCheckpointRelativePath);
                string loadArg = (!string.IsNullOrEmpty(ckptFull) && File.Exists(ckptFull)) ? $" --load-weights \"{ckptFull}\"" : "";
                string oppArg = (!string.IsNullOrEmpty(oppFull) && File.Exists(oppFull)) ? $" --opp-weights \"{oppFull}\"" : "";
                string contArg = spectateContinualLearning
                    ? $" --continual-learn --sleep-consolidation 1 --sleep-epochs 2 --save-continual-weights \"{contSaveFull}\""
                    : "";
                string radio2v2Arg = spectate2v2BrainRadio ? " --spectate-2v2" : "";
                args = $"-B \"{scriptPath}\" --mode spectate --env unity_tcp --task-profile {taskProfile} --stage-id {stageArgVal} "
                     + $"--hippo-memory-mode {hippoArg}{contArg}{radio2v2Arg}{stratArg} --base-port {baseTcpPort} --ux-port {baseTcpPort + 100} --time-scale 1.0 --parent-pid {unityPid} --watch-stdin{loadArg}{oppArg}";
                Time.timeScale = 1.0f;
                Time.maximumDeltaTime = Time.fixedDeltaTime * 1.25f;
            }
            else
            {
                scriptPath = Path.Combine(brainWorkspaceDir, "arena_pbt_trainer.py");
                string mm = (sessionMode == BrainSessionRunMode.ParallelCompetitiveSelfPlay) ? "self_play" : "pbt_solo";
                string ckptDirArg = string.IsNullOrWhiteSpace(pbtCheckpointDir) ? "" : pbtCheckpointDir.Trim();
                string pbtPol = string.IsNullOrWhiteSpace(pbtPolicyMode) ? "dagger" : pbtPolicyMode.Trim();
                string ckptFull = ResolveCheckpointPath(loadCheckpointRelativePath);
                string initArg = (resumeFromCheckpointInPbt && !string.IsNullOrEmpty(ckptFull) && File.Exists(ckptFull))
                    ? $" --init-weights \"{ckptFull}\""
                    : "";
                int slpFlag = (enableSleepConsolidation && sleepConsolidationEpochs > 0) ? 1 : 0;
                int slpEp = Mathf.Max(0, sleepConsolidationEpochs);
                args = $"-B \"{scriptPath}\" --mode train --env unity_tcp --matchmaking {mm} "
                     + $"--task-profile {taskProfile} --stage-id {stageArgVal} --pop-size {parallelArenaCount} "
                     + $"--generations {Mathf.Max(1, pbtGenerations)} --episodes-per-gen {Mathf.Max(1, pbtEpisodesPerGen)} "
                     + $"--policy-mode {pbtPol} --hippo-memory-mode {hippoArg} --sleep-consolidation {slpFlag} --sleep-epochs {slpEp}{stratArg} "
                     + $"--checkpoint-dir \"{ckptDirArg}\"{initArg} "
                     + $"--base-port {baseTcpPort} --time-scale {trainingTimeScale:F1} --parent-pid {unityPid} --watch-stdin";
                Time.timeScale = trainingTimeScale;
                int stepCap = Mathf.Clamp(maxTrainingStepsPerRenderFrame, 1, 10);
                Time.maximumDeltaTime = Mathf.Max(Time.fixedDeltaTime, Time.fixedDeltaTime * stepCap);
            }

            var psi = new ProcessStartInfo
            {
                FileName = pythonExePath,
                Arguments = args,
                WorkingDirectory = brainWorkspaceDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = System.Text.Encoding.UTF8,
                StandardErrorEncoding = System.Text.Encoding.UTF8
            };
            psi.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";
            psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";

            try
            {
                _pythonProc = Process.Start(psi);
                if (_pythonProc != null)
                {
                    _pythonProc.OutputDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            if (e.Data.StartsWith("[PBT_STAT] "))
                            {
                                while (PbtStatQueue.Count > 128)
                                    PbtStatQueue.TryDequeue(out string dropped);
                                PbtStatQueue.Enqueue(e.Data.Substring(11).Trim());
                            }
                            else
                            {
                                latestBackendLog = e.Data;
                                if (e.Data.StartsWith("[PBT]") || e.Data.StartsWith("[Gen ") || e.Data.StartsWith("[Spectate]") || e.Data.StartsWith("{"))
                                    UnityEngine.Debug.Log($"[UniversalArenaCoordinator3D] {e.Data}");
                            }
                        }
                    };
                    _pythonProc.ErrorDataReceived += (_, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            if (e.Data.Contains("Error") || e.Data.Contains("Exception") || e.Data.Contains("Traceback"))
                            {
                                latestBackendError = e.Data;
                                UnityEngine.Debug.LogWarning($"[UniversalArenaCoordinator3D][PyErr] {e.Data}");
                            }
                        }
                    };
                    _pythonProc.BeginOutputReadLine();
                    _pythonProc.BeginErrorReadLine();
                }
                _nextReconnectTime = Time.unscaledTime + 0.5f;
                UnityEngine.Debug.Log($"[UniversalArenaCoordinator3D] Launched Python backend PID={_pythonProc?.Id}: {args}");
            }
            catch (Exception ex)
            {
                _pythonProc = null;
                latestBackendLog = $"ERROR: {ex.Message}";
                UnityEngine.Debug.LogError($"[UniversalArenaCoordinator3D] Failed to launch Python backend ({pythonExePath}): {ex.Message}");
            }
        }

        public void RegisterBridge(UniversalBrainBridge3D bridge)
        {
            if (bridge != null && !_activeBridges.Contains(bridge))
                _activeBridges.Add(bridge);
        }

        public void ClearRegisteredBridges()
        {
            for (int i = 0; i < _activeBridges.Count; i++)
            {
                if (_activeBridges[i] != null)
                    _activeBridges[i].Disconnect();
            }
            _activeBridges.Clear();
        }

        /// <summary>
        /// Two-Stage PC-Friendly Process Shutdown:
        ///   Stage 1 (Graceful): Sends "SHUTDOWN\n" + closes Stdin (`--watch-stdin`), giving Python `gracefulTimeoutMs`
        ///     to join workers, clear Replay RAM, release PyTorch CUDA VRAM (`empty_cache`), and `unlink` SharedMemory buses.
        ///   Stage 2 (Hard Fallback): Escalates to `taskkill /F /T /PID` ONLY if Python fails to exit within timeout.
        /// </summary>
        private void TerminateProcessGracefullyThenFallback(Process proc, int gracefulTimeoutMs, bool isEmergencyAbort = false)
        {
            if (proc == null) return;
            try
            {
                if (!proc.HasExited)
                {
                    int rootPid = proc.Id;
                    try
                    {
                        proc.StandardInput.WriteLine(isEmergencyAbort ? "EMERGENCY_STOP" : "SHUTDOWN");
                        proc.StandardInput.Flush();
                        proc.StandardInput.Close();
                    }
                    catch { }

                    bool exitedCleanly = proc.WaitForExit(Mathf.Max(200, gracefulTimeoutMs));
                    if (exitedCleanly)
                    {
                        SafeShutdownPhase = "CLEAN_EXIT";
                    }
                    else
                    {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
                        using (var killer = Process.Start(new ProcessStartInfo
                        {
                            FileName = "taskkill",
                            Arguments = $"/F /T /PID {rootPid}",
                            UseShellExecute = false,
                            CreateNoWindow = true
                        }))
                        {
                            killer?.WaitForExit(1500);
                        }
#endif
                        if (!proc.HasExited)
                            proc.Kill();
                        SafeShutdownPhase = "FORCE_KILLED";
                    }
                }
                else
                {
                    SafeShutdownPhase = "CLEAN_EXIT";
                }
            }
            catch
            {
                SafeShutdownPhase = "CLEAN_EXIT";
            }
            finally
            {
                try { proc.Dispose(); } catch { }
            }
        }

        public void StopPythonBackend()
        {
            foreach (var b in _activeBridges)
            {
                if (b != null) b.Disconnect();
            }

            Thread inflight = _shutdownThread;
            if (inflight != null && inflight.IsAlive)
            {
                try { inflight.Join(2800); } catch { }
            }
            _shutdownThread = null;

            Process proc = _pythonProc;
            _pythonProc = null;
            if (proc != null)
            {
                SafeShutdownPhase = "RELEASING_MEMORY";
                TerminateProcessGracefullyThenFallback(proc, 1800);
            }
            Time.timeScale = 1.0f;
            if (_defaultMaxDeltaTime > 0.01f)
                Time.maximumDeltaTime = _defaultMaxDeltaTime;
        }

        /// <summary>
        /// Two-Stage Safety Emergency Stop:
        ///   0ms Immediate Lock: severs every TCP sensory/motor bridge in the scene and freezes Unity physics (`Time.timeScale = 0`).
        ///   Stage 1->2 Async Teardown: signals Python via stdin to release RAM/VRAM/SharedMemory cleanly on a background thread
        ///   (escalating to `taskkill /F /T` only if unresponsive after 2.4s).
        /// </summary>
        public void EmergencyHaltAll()
        {
            IsEmergencyHalted = true;
            EmergencyStopsTriggeredCount++;

            // 1. Instant 0ms environment & socket isolation (zero further motor commands)
            ClearRegisteredBridges();
            var allBridges = UnityEngine.Object.FindObjectsByType<UniversalBrainBridge3D>(FindObjectsInactive.Include);
            for (int i = 0; i < allBridges.Length; i++)
            {
                if (allBridges[i] != null)
                    allBridges[i].Disconnect();
            }
            while (PbtStatQueue.TryDequeue(out _)) { }
            Time.timeScale = 0.0f;

            // 2. Non-blocking Two-Stage Graceful -> Fallback Shutdown for Python AI / GPU processes
            Process procToStop = _pythonProc;
            _pythonProc = null;
            if (procToStop != null)
            {
                SafeShutdownPhase = "RELEASING_MEMORY";
                latestBackendLog = "[EMERGENCY HALT] Skipping unverified weight save; releasing gradients, optimizer, replay, VRAM & SeqLock...";
                _shutdownThread = new Thread(() =>
                {
                    TerminateProcessGracefullyThenFallback(procToStop, 2400, isEmergencyAbort: true);
                    latestBackendLog = SafeShutdownPhase == "CLEAN_EXIT"
                        ? "[EMERGENCY HALT] Unverified save skipped; gradients/optimizer/replay/VRAM/SeqLock cleanly released."
                        : "[EMERGENCY HALT] AI process tree terminated via fallback timeout guard.";
                })
                {
                    IsBackground = true,
                    Name = "BrainSafeEmergencyShutdown"
                };
                _shutdownThread.Start();
            }
            else if (_shutdownThread == null || !_shutdownThread.IsAlive)
            {
                SafeShutdownPhase = "CLEAN_EXIT";
                latestBackendLog = "[EMERGENCY HALT] All TCP channels severed & physics frozen.";
            }

            UnityEngine.Debug.LogWarning("[UniversalArenaCoordinator3D] EMERGENCY STOP executed: TCP sockets severed, physics frozen, graceful RAM/VRAM teardown initiated.");
        }

        public void ClearEmergencyHalt()
        {
            IsEmergencyHalted = false;
            if (!IsBackendRunning)
                SafeShutdownPhase = "IDLE";
            Time.timeScale = 1.0f;
            if (_defaultMaxDeltaTime > 0.01f)
                Time.maximumDeltaTime = _defaultMaxDeltaTime;
        }

        private void OnApplicationQuit() => StopPythonBackend();
        private void OnDestroy() => StopPythonBackend();
    }
}
