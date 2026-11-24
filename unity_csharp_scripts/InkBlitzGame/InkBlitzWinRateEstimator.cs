// InkBlitzWinRateEstimator.cs — Self-Contained Hybrid Win-Probability Engine & Top-Center Tug-of-War HUD
// Architecture:
//   (A) Exact 180-deg Antisymmetric Analytic BlitzCore Log-Odds Baseline (calibrated from Match #0, exact 50.0% at symmetry)
//   (C) Pure-C# Online L2-Regularized AdaGrad Logistic Residual Learner (calibrates weights from actual match outcomes, Barrier-1 safe)
//   (B) Optional Brain Subjective Confidence Overlay (`UBNT` RSSM survival + integrated Q* value readout)
// License: GPL-3.0

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UniversalBrain3D;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace InkBlitz3D
{
    public class InkBlitzWinRateEstimator : MonoBehaviour
    {
        public const int FEATURE_DIM = 10;
        private const int MAX_SAMPLES_PER_MATCH = 128;
        private const int SPARK_W = 192;
        private const int SPARK_H = 14;

        [Serializable]
        private class CalibratorSaveData
        {
            public int schemaVersion = 1;
            public int calibratedMatches = 0;
            public int totalWinsA = 0;
            public int totalWinsB = 0;
            public float rollingWinRateA = 0.50f;
            public float[] residualWeights = new float[FEATURE_DIM];
            public float[] gradSqAccum = new float[FEATURE_DIM];
        }

        private sealed class ArenaMatchTracker
        {
            public int lastEpisodesCompleted = -1;
            public int lastWinsA = 0;
            public int lastWinsB = 0;
            public int lastSampledTick = -999;
            public int sampleCount = 0;
            public readonly float[][] featureRing = new float[MAX_SAMPLES_PER_MATCH][];

            public ArenaMatchTracker()
            {
                for (int i = 0; i < MAX_SAMPLES_PER_MATCH; i++)
                    featureRing[i] = new float[FEATURE_DIM];
            }
        }

        // (A) Analytic BlitzCore Domain Log-Odds Baseline Weights (antisymmetric, zero bias => exact 50.0% at symmetry)
        private static readonly float[] AnalyticBaseWeights =
        {
            2.35f, // [0] Time-urgency scaled best-count lead: tanh(1.35 * u(tau) * (bestA - bestB))
            0.85f, // [1] Raw best-count lead: (bestA - bestB)
            1.15f, // [2] Live BlitzCore carrier push momentum (boosted near/above best progress)
            2.60f, // [3] Knockout proximity surge within 8.5m of enemy Goal Pedestal
            0.55f, // [4] BlitzCore field position + Shield tug-of-war ratio
            0.72f, // [5] Main AI availability / HP / respawn advantage
            0.28f, // [6] Support Trooper wingman availability advantage
            0.68f, // [7] Turf ink map control advantage
            0.32f, // [8] Ink tank reserve & active Deflect Shield buff advantage
            2.45f  // [9] Endgame regulation clock lockout when leading & opponent lacks carrier
        };

        [Header("Scene References")]
        public UniversalArenaCoordinator3D coordinator;
        public InkBlitzMultiArenaManager multiManager;
        public InkBlitzSessionDashboardUi dashboardUi;
        public InkBlitzNeuralVisualizerHud neuralHud;

        [Header("HUD & Online Calibration Settings")]
        public bool showWinRateBar = true;
        public bool enableOnlineCalibration = true;
        public bool enableDiskPersistence = true;
        [Range(0.01f, 0.50f)] public float calibratorLearningRate = 0.08f;
        [Range(0.001f, 0.25f)] public float calibratorL2Reg = 0.035f;

        // Learned residual weights (C) & long-term stats
        private readonly float[] _residualWeights = new float[FEATURE_DIM];
        private readonly float[] _gradSqAccum = new float[FEATURE_DIM];
        private int _calibratedMatches = 0;
        private int _totalWinsA = 0;
        private int _totalWinsB = 0;
        private float _rollingWinRateA = 0.50f;

        // Live display & focused-arena in-match sparkline state
        private readonly float[] _liveFeatures = new float[FEATURE_DIM];
        private readonly Dictionary<int, ArenaMatchTracker> _trackers = new Dictionary<int, ArenaMatchTracker>(8);
        private readonly List<float> _focusedMatchCurve = new List<float>(160);
        private int _focusedArenaIdx = -1;
        private int _focusedLastEp = -1;
        private float _nextCurveSampleRealtime = 0f;
        private float _rawWinProbA = 0.50f;
        private float _smoothedWinProbA = 0.50f;
        private float _momentumLagProbA = 0.50f;
        private float _smoothedSubjProbA = 0.50f;
        private bool _hasSubjProbA = false;

        // Cached IMGUI textures & styles
        private bool _guiReady = false;
        private int _cachedPalIdx = -1;
        private Texture2D _texFrameBg;
        private Texture2D _texBorder;
        private Texture2D _texTeamA;
        private Texture2D _texTeamB;
        private Texture2D _texCenterTick;
        private Texture2D _texCursor;
        private Texture2D _texSubjTick;
        private Texture2D _texSparkline;
        private Color32[] _sparkPixels;
        private bool _sparkDirty = true;
        private GUIStyle _stTeamLeft;
        private GUIStyle _stTeamRight;
        private GUIStyle _stCenterMeta;

        public float CurrentWinProbabilityA01 => _smoothedWinProbA;
        public float RawWinProbabilityA01 => _rawWinProbA;
        public float RollingLongTermWinRateA01 => _rollingWinRateA;
        public int CalibratedMatchCount => _calibratedMatches;

        public bool IsBarVisibleNow =>
            showWinRateBar
            && (dashboardUi == null || !dashboardUi.IsLauncherOpen)
            && (multiManager == null || !multiManager.isPausedForLauncher);

        private void Awake()
        {
            AutoBindReferences();
            if (Application.isPlaying && enableDiskPersistence)
                TryLoadCalibratorFromDisk();
        }

        private void OnDestroy()
        {
            DestroyTex(_texFrameBg);
            DestroyTex(_texBorder);
            DestroyTex(_texTeamA);
            DestroyTex(_texTeamB);
            DestroyTex(_texCenterTick);
            DestroyTex(_texCursor);
            DestroyTex(_texSubjTick);
            DestroyTex(_texSparkline);
        }

        private static void DestroyTex(Texture2D t)
        {
            if (t == null) return;
            if (Application.isPlaying) Destroy(t);
            else DestroyImmediate(t);
        }

        public void AutoBindReferences()
        {
            if (coordinator == null) coordinator = FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (multiManager == null) multiManager = FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (dashboardUi == null) dashboardUi = FindAnyObjectByType<InkBlitzSessionDashboardUi>();
            if (neuralHud == null) neuralHud = FindAnyObjectByType<InkBlitzNeuralVisualizerHud>();
        }

        // =====================================================================================
        // 1. Exact 180-deg Antisymmetric Feature Extraction & Hybrid (A)+(C) Probability Model
        // =====================================================================================

        public static void ExtractAntisymmetricFeatures(InkBlitzArenaInstance arena, float[] featOut10)
        {
            if (featOut10 == null || featOut10.Length < FEATURE_DIM)
                return;
            Array.Clear(featOut10, 0, FEATURE_DIM);
            if (arena == null)
                return;

            InkBlitzArenaSettings cfg = arena.settings ?? new InkBlitzArenaSettings();
            Vector3 origin = arena.transform.position;

            // Effective remaining seconds (accounting for Overtime urgency)
            float matchDur = Mathf.Max(15f, cfg.matchDurationSec);
            float remSec = arena.isOvertime ? 8.0f : Mathf.Clamp(arena.matchTimeRemaining, 0f, matchDur);
            float urgency = Mathf.Sqrt(matchDur / (remSec + 6.0f)); // ~0.97 at 90s -> ~3.6 at 1s left

            // [0] & [1]: Best count advantage (in [0..1] progress units)
            float deltaBest = Mathf.Clamp(arena.bestProgressTeamA - arena.bestProgressTeamB, -1f, 1f);
            featOut10[0] = (float)Math.Tanh(1.35 * urgency * deltaBest);
            featOut10[1] = deltaBest;

            // [2], [3], [4]: BlitzCore Carrier Push, KO Proximity & Field Position / Shield Tug
            BlitzCoreObjective rm = arena.blitzCore;
            int carrierTeam = 0;
            if (rm != null)
            {
                Vector3 goalForA = cfg.teamBPedestalLocal; // (-8.0, 1.35, +12.8)
                Vector3 goalForB = cfg.teamAPedestalLocal; // (+8.0, 1.35, -12.8)
                Vector2 axisA = new Vector2(goalForA.x - goalForB.x, goalForA.z - goalForB.z);
                float axisLen = Mathf.Max(1f, axisA.magnitude);
                Vector2 axisUnit = axisA / axisLen;

                Vector3 rmLoc = rm.transform.position - origin;
                float proj = Vector2.Dot(new Vector2(rmLoc.x, rmLoc.z), axisUnit) / (axisLen * 0.5f);
                float shieldBias = (rm.State == BlitzCoreState.Shielded && rm.MaxShieldTug > 1f)
                    ? Mathf.Clamp(rm.ShieldTug / rm.MaxShieldTug, -1f, 1f) * 0.40f
                    : 0f;
                featOut10[4] = Mathf.Clamp(proj + shieldBias, -1f, 1f);

                if (rm.State == BlitzCoreState.Carried && rm.Carrier != null)
                {
                    carrierTeam = rm.Carrier.teamId > 0 ? 1 : -1;
                    Vector3 carrierLoc = rm.Carrier.transform.position - origin;
                    Vector3 targetPed = (carrierTeam > 0) ? goalForA : goalForB;
                    float distGoal = Vector2.Distance(new Vector2(carrierLoc.x, carrierLoc.z), new Vector2(targetPed.x, targetPed.z));
                    float centerDist = Mathf.Max(2f, new Vector2(targetPed.x, targetPed.z).magnitude);
                    float push01 = Mathf.Clamp01(1f - distGoal / centerDist);

                    float teamBest = (carrierTeam > 0) ? arena.bestProgressTeamA : arena.bestProgressTeamB;
                    float breakBonus = (push01 >= teamBest - 0.06f) ? 0.30f : 0f;
                    featOut10[2] = carrierTeam * Mathf.Clamp01(0.18f + 0.82f * push01 + breakBonus);

                    if (distGoal < 8.5f)
                    {
                        float k = (8.5f - distGoal) / 8.5f;
                        featOut10[3] = carrierTeam * (k * k);
                    }
                }
            }

            // [5], [6], [8]: Agent availability (alive vs respawning), HP, Ink & Deflect Buffs
            InkBlitzPlayerAgent mainA = null, mainB = null, supA = null, supB = null;
            for (int i = 0; i < arena.agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = arena.agents[i];
                if (ag == null) continue;
                if (ag.isSupportTrooperBot)
                {
                    if (!ag.gameObject.activeSelf) continue;
                    if (ag.teamId > 0 && supA == null) supA = ag;
                    else if (ag.teamId < 0 && supB == null) supB = ag;
                }
                else
                {
                    if (ag.teamId > 0 && mainA == null) mainA = ag;
                    else if (ag.teamId < 0 && mainB == null) mainB = ag;
                }
            }

            float availMainA = ComputeAgentAvailability01(mainA, Mathf.Max(0.5f, cfg.respawnDurationSec));
            float availMainB = ComputeAgentAvailability01(mainB, Mathf.Max(0.5f, cfg.respawnDurationSec));
            featOut10[5] = Mathf.Clamp(availMainA - availMainB, -1f, 1f);

            if (supA != null || supB != null)
            {
                float availSupA = ComputeAgentAvailability01(supA, Mathf.Max(0.5f, cfg.supportBotRespawnSec));
                float availSupB = ComputeAgentAvailability01(supB, Mathf.Max(0.5f, cfg.supportBotRespawnSec));
                featOut10[6] = Mathf.Clamp(availSupA - availSupB, -1f, 1f);
            }

            // [7]: Turf ink control advantage
            if (arena.turfGrid != null)
            {
                float turfDiff = arena.turfGrid.TeamARatio01 - arena.turfGrid.TeamBRatio01;
                featOut10[7] = Mathf.Clamp(turfDiff * 2.2f, -1f, 1f);
            }

            // [8]: Ink reserve + Deflect Shield Buff advantage
            float inkA = (mainA != null && !mainA.isRespawning) ? mainA.InkTank01 : 0f;
            float inkB = (mainB != null && !mainB.isRespawning) ? mainB.InkTank01 : 0f;
            float buffA = (mainA != null && mainA.HasDeflectBuff) ? 1f : 0f;
            float buffB = (mainB != null && mainB.HasDeflectBuff) ? 1f : 0f;
            featOut10[8] = Mathf.Clamp(0.42f * (inkA - inkB) + 0.58f * (buffA - buffB), -1f, 1f);

            // [9]: Endgame regulation clock lockout (trailing team cannot trigger Overtime without holding/popping BlitzCore)
            if (!arena.isOvertime && remSec < 8.0f && Mathf.Abs(deltaBest) > 0.01f)
            {
                int leader = deltaBest > 0f ? 1 : -1;
                bool trailerHasHope = (carrierTeam == -leader)
                                   || (rm != null && rm.State == BlitzCoreState.PickupReady && remSec > 1.5f);
                if (!trailerHasHope)
                {
                    float lock01 = Mathf.Clamp01((8.0f - remSec) / 7.2f);
                    featOut10[9] = leader * (lock01 * lock01);
                }
            }
        }

        private static float ComputeAgentAvailability01(InkBlitzPlayerAgent ag, float maxRespawnSec)
        {
            if (ag == null) return 0.5f;
            if (ag.isRespawning)
            {
                float waitFrac = Mathf.Clamp01(ag.RespawnTimeRemaining / Mathf.Max(0.2f, maxRespawnSec));
                return 0.18f * (1f - waitFrac);
            }
            float hp01 = Mathf.Clamp01(ag.hp / Mathf.Max(1f, ag.EffectiveMaxHp));
            return 0.35f + 0.65f * hp01;
        }

        public static float EvaluateAnalyticLogOdds(float[] features10)
        {
            float logit = 0f;
            int n = Mathf.Min(FEATURE_DIM, features10 != null ? features10.Length : 0);
            for (int i = 0; i < n; i++)
                logit += AnalyticBaseWeights[i] * features10[i];
            return logit;
        }

        public float EvaluateWinProbabilityFromFeatures(float[] features10)
        {
            float baseLogit = EvaluateAnalyticLogOdds(features10);
            float residual = 0f;
            float alpha = _calibratedMatches / (_calibratedMatches + 20f);
            if (alpha > 1e-4f && features10 != null)
            {
                int n = Mathf.Min(FEATURE_DIM, features10.Length);
                for (int i = 0; i < n; i++)
                    residual += _residualWeights[i] * features10[i];
            }
            float totalLogit = Mathf.Clamp(baseLogit + alpha * residual, -6.5f, 6.5f);
            return 1f / (1f + Mathf.Exp(-totalLogit));
        }

        public float EvaluateArenaWinProbability(InkBlitzArenaInstance arena, float[] outFeatures10 = null)
        {
            float[] buf = (outFeatures10 != null && outFeatures10.Length >= FEATURE_DIM) ? outFeatures10 : _liveFeatures;
            ExtractAntisymmetricFeatures(arena, buf);
            return EvaluateWinProbabilityFromFeatures(buf);
        }

        // =====================================================================================
        // 2. Online Logistic Residual Calibration from Completed Match Trajectories
        // =====================================================================================

        public void TrainCalibratorOnMatchOutcome(IReadOnlyList<float[]> trajectorySamples, int sampleCount, float teamAWon01)
        {
            if (!enableOnlineCalibration || trajectorySamples == null || sampleCount <= 0)
                return;

            float y = Mathf.Clamp01(teamAWon01);
            if (y > 0.75f) _totalWinsA++;
            else if (y < 0.25f) _totalWinsB++;

            float emaBeta = (_calibratedMatches == 0) ? 0.35f : 0.08f;
            _rollingWinRateA = Mathf.Lerp(_rollingWinRateA, y, emaBeta);
            _calibratedMatches++;

            float alpha = Mathf.Max(0.25f, _calibratedMatches / (_calibratedMatches + 20f));
            int count = Mathf.Min(sampleCount, trajectorySamples.Count);
            float invCount = 1f / Mathf.Max(1, count);

            // Accumulate match-averaged gradient on residual weights (time-weighted toward mid/late match)
            float[] grad = new float[FEATURE_DIM];
            for (int s = 0; s < count; s++)
            {
                float[] phi = trajectorySamples[s];
                if (phi == null || phi.Length < FEATURE_DIM) continue;
                float timeWeight = 0.65f + 0.70f * ((s + 1f) / count);
                float logit = EvaluateAnalyticLogOdds(phi);
                for (int k = 0; k < FEATURE_DIM; k++)
                    logit += alpha * _residualWeights[k] * phi[k];
                float p = 1f / (1f + Mathf.Exp(-Mathf.Clamp(logit, -6.5f, 6.5f)));
                float err = (p - y) * timeWeight * invCount;
                for (int k = 0; k < FEATURE_DIM; k++)
                    grad[k] += err * phi[k];
            }

            for (int k = 0; k < FEATURE_DIM; k++)
            {
                float g = grad[k] + calibratorL2Reg * _residualWeights[k];
                _gradSqAccum[k] += g * g;
                float step = (calibratorLearningRate / Mathf.Sqrt(_gradSqAccum[k] + 1.0f)) * g;
                _residualWeights[k] = Mathf.Clamp(_residualWeights[k] - step, -1.85f, 1.85f);
            }
        }

        private void StepAllArenaCalibrationTrackers()
        {
            if (multiManager == null || multiManager.arenas == null)
                return;

            bool savedNeeded = false;
            for (int i = 0; i < multiManager.arenas.Count; i++)
            {
                InkBlitzArenaInstance ar = multiManager.arenas[i];
                if (ar == null) continue;

                if (!_trackers.TryGetValue(ar.arenaIndex, out ArenaMatchTracker tr))
                {
                    tr = new ArenaMatchTracker
                    {
                        lastEpisodesCompleted = ar.episodesCompleted,
                        lastWinsA = ar.teamAWins,
                        lastWinsB = ar.teamBWins,
                        lastSampledTick = ar.TickCount
                    };
                    _trackers[ar.arenaIndex] = tr;
                }

                // Episode just concluded!
                if (ar.episodesCompleted > tr.lastEpisodesCompleted)
                {
                    bool aWon = ar.teamAWins > tr.lastWinsA;
                    bool bWon = ar.teamBWins > tr.lastWinsB;
                    if ((aWon || bWon) && tr.sampleCount >= 3)
                    {
                        TrainCalibratorOnMatchOutcome(tr.featureRing, tr.sampleCount, aWon ? 1f : 0f);
                        savedNeeded = true;
                    }
                    tr.lastEpisodesCompleted = ar.episodesCompleted;
                    tr.lastWinsA = ar.teamAWins;
                    tr.lastWinsB = ar.teamBWins;
                    tr.sampleCount = 0;
                    tr.lastSampledTick = ar.TickCount;
                }

                // Sample trajectory every 60 physics ticks (1.0 second of match time)
                if (ar.TickCount - tr.lastSampledTick >= 60 || (tr.sampleCount == 0 && ar.TickCount > 5))
                {
                    tr.lastSampledTick = ar.TickCount;
                    int slot = tr.sampleCount < MAX_SAMPLES_PER_MATCH
                        ? tr.sampleCount++
                        : UnityEngine.Random.Range(MAX_SAMPLES_PER_MATCH / 2, MAX_SAMPLES_PER_MATCH);
                    ExtractAntisymmetricFeatures(ar, tr.featureRing[slot]);
                }
            }

            if (savedNeeded && enableDiskPersistence && Application.isPlaying)
                TrySaveCalibratorToDisk();
        }

        private string GetCalibratorFilePath()
        {
            if (coordinator == null) return "";
            string ws = coordinator.GetResolvedWorkspaceDir();
            if (string.IsNullOrWhiteSpace(ws) || !Directory.Exists(ws)) return "";
            string dir = Path.Combine(ws, "checkpoints", "user_runs");
            return Path.Combine(dir, "win_rate_calibrator.json");
        }

        private void TryLoadCalibratorFromDisk()
        {
            try
            {
                string path = GetCalibratorFilePath();
                if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;
                CalibratorSaveData data = JsonUtility.FromJson<CalibratorSaveData>(File.ReadAllText(path));
                if (data == null || data.residualWeights == null || data.residualWeights.Length != FEATURE_DIM)
                    return;
                _calibratedMatches = Mathf.Max(0, data.calibratedMatches);
                _totalWinsA = Mathf.Max(0, data.totalWinsA);
                _totalWinsB = Mathf.Max(0, data.totalWinsB);
                _rollingWinRateA = Mathf.Clamp01(data.rollingWinRateA);
                Array.Copy(data.residualWeights, _residualWeights, FEATURE_DIM);
                if (data.gradSqAccum != null && data.gradSqAccum.Length == FEATURE_DIM)
                    Array.Copy(data.gradSqAccum, _gradSqAccum, FEATURE_DIM);
            }
            catch { }
        }

        private void TrySaveCalibratorToDisk()
        {
            try
            {
                string path = GetCalibratorFilePath();
                if (string.IsNullOrEmpty(path)) return;
                string dir = Path.GetDirectoryName(path);
                if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return;

                var data = new CalibratorSaveData
                {
                    schemaVersion = 1,
                    calibratedMatches = _calibratedMatches,
                    totalWinsA = _totalWinsA,
                    totalWinsB = _totalWinsB,
                    rollingWinRateA = _rollingWinRateA,
                    residualWeights = (float[])_residualWeights.Clone(),
                    gradSqAccum = (float[])_gradSqAccum.Clone()
                };
                File.WriteAllText(path, JsonUtility.ToJson(data, true));
            }
            catch { }
        }

        // =====================================================================================
        // 3. Live Update & Slim Top-Center Tug-of-War HUD + Micro-Sparkline Rendering
        // =====================================================================================

        private void Update()
        {
            AutoBindReferences();

#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null && GUIUtility.keyboardControl == 0 && (dashboardUi == null || !dashboardUi.IsLauncherOpen))
            {
                if (kb.wKey.wasPressedThisFrame)
                    showWinRateBar = !showWinRateBar;
            }
#endif

            if (!IsBarVisibleNow && !enableOnlineCalibration)
                return;

            StepAllArenaCalibrationTrackers();

            InkBlitzArenaInstance focused = GetFocusedArena();
            if (focused == null)
                return;

            float udt = Mathf.Clamp(Time.unscaledDeltaTime, 0.001f, 0.2f);
            _rawWinProbA = EvaluateArenaWinProbability(focused, _liveFeatures);

            if (focused.arenaIndex != _focusedArenaIdx || focused.episodesCompleted != _focusedLastEp)
            {
                _focusedArenaIdx = focused.arenaIndex;
                _focusedLastEp = focused.episodesCompleted;
                _focusedMatchCurve.Clear();
                _focusedMatchCurve.Add(_rawWinProbA);
                _smoothedWinProbA = _rawWinProbA;
                _momentumLagProbA = _rawWinProbA;
                _nextCurveSampleRealtime = Time.realtimeSinceStartup + 0.5f;
                _sparkDirty = true;
            }
            else
            {
                _smoothedWinProbA = Mathf.Lerp(_smoothedWinProbA, _rawWinProbA, Mathf.Clamp01(udt * 7.5f));
                _momentumLagProbA = Mathf.Lerp(_momentumLagProbA, _smoothedWinProbA, Mathf.Clamp01(udt * 0.85f));
                if (Time.realtimeSinceStartup >= _nextCurveSampleRealtime)
                {
                    _nextCurveSampleRealtime = Time.realtimeSinceStartup + 0.55f;
                    if (_focusedMatchCurve.Count >= 220)
                        _focusedMatchCurve.RemoveAt(0);
                    _focusedMatchCurve.Add(_smoothedWinProbA);
                    _sparkDirty = true;
                }
            }

            if (neuralHud != null && neuralHud.TryGetAgentSubjectiveConfidence01(out float rawSubj))
            {
                _hasSubjProbA = true;
                _smoothedSubjProbA = Mathf.Lerp(_smoothedSubjProbA, rawSubj, Mathf.Clamp01(udt * 6.0f));
            }
            else
            {
                _hasSubjProbA = false;
            }
        }

        private InkBlitzArenaInstance GetFocusedArena()
        {
            if (multiManager != null && multiManager.arenas != null && multiManager.arenas.Count > 0)
            {
                int idx = Mathf.Clamp(multiManager.focusedArenaIndex, 0, multiManager.arenas.Count - 1);
                if (multiManager.arenas[idx] != null)
                    return multiManager.arenas[idx];
                return multiManager.primaryArena;
            }
            return FindAnyObjectByType<InkBlitzArenaInstance>();
        }

        public Rect GetTopWinRateBarRect()
        {
            const float barY = 11f;
            const float barH = 54f; // Matches 54px height of adjacent 16:9 Emergency Stop switch!
            float leftLim = 486f;   // 10px right of Emergency Stop switch (380..476)

            bool isTraining = coordinator != null && coordinator.IsTrainingSession;
            float rightLim;
            if (isTraining)
            {
                bool collapsed = dashboardUi != null && dashboardUi.IsTrainingPanelCollapsed;
                rightLim = Screen.width - (collapsed ? 236f : 364f);
            }
            else
            {
                float rightInset = (neuralHud != null) ? neuralHud.ActiveRightPanelWidthPx : 0f;
                if (Screen.width >= 1660)
                {
                    float stripX = Screen.width - rightInset - 352f - 12f;
                    rightLim = stripX - 10f;
                }
                else
                {
                    rightLim = Screen.width - rightInset - 12f;
                }
            }

            float availW = Mathf.Max(250f, rightLim - leftLim);
            float barW = Mathf.Clamp(availW, 250f, 484f);
            float x0 = leftLim + Mathf.Max(0f, (availW - barW) * 0.5f);
            return new Rect(x0, barY, barW, barH);
        }

        private void EnsureGuiResources(int palIdx, Color colA, Color colB)
        {
            if (!_guiReady)
            {
                _texFrameBg = MakeSolidTex(new Color(0.055f, 0.068f, 0.088f, 0.94f)); // #0e1116 cyber matte
                _texBorder = MakeSolidTex(new Color(0.24f, 0.30f, 0.38f, 0.85f));
                _texCenterTick = MakeSolidTex(new Color(0.92f, 0.95f, 1.0f, 0.65f));
                _texCursor = MakeSolidTex(new Color(1.0f, 0.96f, 0.82f, 1.0f));
                _texSubjTick = MakeSolidTex(new Color(0.32f, 0.96f, 0.88f, 0.95f));

                _texSparkline = new Texture2D(SPARK_W, SPARK_H, TextureFormat.RGBA32, false)
                {
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
                _sparkPixels = new Color32[SPARK_W * SPARK_H];

                _stTeamLeft = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 12,
                    fontStyle = FontStyle.Bold,
                    richText = true,
                    alignment = TextAnchor.MiddleLeft
                };
                _stTeamRight = new GUIStyle(_stTeamLeft)
                {
                    alignment = TextAnchor.MiddleRight
                };
                _stCenterMeta = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 10,
                    richText = true,
                    alignment = TextAnchor.MiddleCenter
                };
                _stCenterMeta.normal.textColor = new Color(0.72f, 0.80f, 0.90f, 1f);
                _guiReady = true;
            }

            if (_cachedPalIdx != palIdx || _texTeamA == null || _texTeamB == null)
            {
                DestroyTex(_texTeamA);
                DestroyTex(_texTeamB);
                _texTeamA = MakeSolidTex(new Color(colA.r, colA.g, colA.b, 0.96f));
                _texTeamB = MakeSolidTex(new Color(colB.r, colB.g, colB.b, 0.96f));
                _cachedPalIdx = palIdx;
                _sparkDirty = true;
            }

            if (_sparkDirty)
            {
                RedrawSparklineTexture(colA, colB);
                _sparkDirty = false;
            }
        }

        private void RedrawSparklineTexture(Color colA, Color colB)
        {
            if (_texSparkline == null || _sparkPixels == null) return;

            Color32 bg = new Color32(10, 13, 18, 235);
            Color32 midLine = new Color32(58, 68, 84, 190);
            int midY = SPARK_H / 2;

            for (int y = 0; y < SPARK_H; y++)
            {
                int row = y * SPARK_W;
                bool isMid = (y == midY);
                for (int x = 0; x < SPARK_W; x++)
                    _sparkPixels[row + x] = (isMid && (x & 3) != 0) ? midLine : bg;
            }

            int n = _focusedMatchCurve.Count;
            if (n >= 1)
            {
                Color32 lineA = ToColor32(Color.Lerp(colA, Color.white, 0.22f), 255);
                Color32 lineB = ToColor32(Color.Lerp(colB, Color.white, 0.22f), 255);
                Color32 fillA = ToColor32(colA, 78);
                Color32 fillB = ToColor32(colB, 78);

                int activeSpanX = (n <= 1) ? 4 : Mathf.Clamp(Mathf.RoundToInt(SPARK_W * Mathf.Clamp01(n / 90f)), 18, SPARK_W - 2);
                int prevPy = Mathf.Clamp(Mathf.RoundToInt(_focusedMatchCurve[0] * (SPARK_H - 3)) + 1, 1, SPARK_H - 2);

                for (int px = 0; px <= activeSpanX; px++)
                {
                    float fIdx = (n <= 1) ? 0f : (px / (float)Mathf.Max(1, activeSpanX)) * (n - 1);
                    int i0 = Mathf.Clamp(Mathf.FloorToInt(fIdx), 0, n - 1);
                    int i1 = Mathf.Min(n - 1, i0 + 1);
                    float pVal = Mathf.Lerp(_focusedMatchCurve[i0], _focusedMatchCurve[i1], fIdx - i0);
                    int py = Mathf.Clamp(Mathf.RoundToInt(pVal * (SPARK_H - 3)) + 1, 1, SPARK_H - 2);

                    bool advA = pVal >= 0.5f;
                    Color32 fillC = advA ? fillA : fillB;
                    int yLo = Mathf.Min(midY, py);
                    int yHi = Mathf.Max(midY, py);
                    for (int fy = yLo; fy <= yHi; fy++)
                    {
                        if (fy != midY)
                            _sparkPixels[fy * SPARK_W + px] = fillC;
                    }

                    Color32 stroke = advA ? lineA : lineB;
                    int sLo = Mathf.Min(prevPy, py);
                    int sHi = Mathf.Max(prevPy, py);
                    for (int sy = sLo; sy <= sHi; sy++)
                        _sparkPixels[sy * SPARK_W + px] = stroke;
                    prevPy = py;
                }

                // Bright live head dot
                int headX = Mathf.Clamp(activeSpanX, 0, SPARK_W - 1);
                Color32 headCol = new Color32(255, 250, 224, 255);
                _sparkPixels[prevPy * SPARK_W + headX] = headCol;
                if (headX > 0) _sparkPixels[prevPy * SPARK_W + headX - 1] = headCol;
            }

            _texSparkline.SetPixels32(_sparkPixels);
            _texSparkline.Apply(false);
        }

        private void OnGUI()
        {
            if (!IsBarVisibleNow)
                return;

            int palIdx = multiManager != null ? multiManager.activePaletteIndex : 0;
            InkColorPalettePreset pal = InkBlitzInkPalettes.Get(palIdx);
            EnsureGuiResources(palIdx, pal.colorA, pal.colorB);

            Rect r = GetTopWinRateBarRect();

            // Outer cyber housing + 1px border
            GUI.DrawTexture(r, _texFrameBg);
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, 1f), _texBorder);
            GUI.DrawTexture(new Rect(r.x, r.y + r.height - 1f, r.width, 1f), _texBorder);
            GUI.DrawTexture(new Rect(r.x, r.y, 1f, r.height), _texBorder);
            GUI.DrawTexture(new Rect(r.x + r.width - 1f, r.y, 1f, r.height), _texBorder);

            float pA = Mathf.Clamp(_smoothedWinProbA, 0.005f, 0.995f);
            float pctA = pA * 100f;
            float pctB = 100f - pctA;
            float mom = _smoothedWinProbA - _momentumLagProbA;
            string momA = mom > 0.03f ? " ▲" : (mom < -0.03f ? " ▼" : "");
            string momB = mom < -0.03f ? "▲ " : (mom > 0.03f ? "▼ " : "");

            // Row 1: Team A % | Center Meta (Long-term + Brain Subjective) | Team B %
            float padX = 8f;
            float innerW = r.width - padX * 2f;
            GUI.Label(new Rect(r.x + padX, r.y + 2f, 118f, 18f),
                $"<color=#{pal.hexA}>A  {pctA:F1}%{momA}</color>", _stTeamLeft);
            GUI.Label(new Rect(r.x + r.width - padX - 118f, r.y + 2f, 118f, 18f),
                $"<color=#{pal.hexB}>{momB}{pctB:F1}%  B</color>", _stTeamRight);

            string subjNote = _hasSubjProbA ? $" | <color=#55F2E0>AI主観 {_smoothedSubjProbA * 100f:F0}%</color>" : "";
            string metaText = (_calibratedMatches > 0)
                ? $"勝率  <color=#99B4CC>長期A:{_rollingWinRateA * 100f:F0}% (校正{_calibratedMatches})</color>{subjNote}"
                : $"展開勝率 <color=#8FA8C0>(数理+学習校正)</color>{subjNote}";
            GUI.Label(new Rect(r.x + 114f, r.y + 2f, Mathf.Max(40f, r.width - 228f), 18f), metaText, _stCenterMeta);

            // Row 2: Matte Dual-Color Tug-of-War Split Bar
            float barX = r.x + padX;
            float barY = r.y + 21f;
            float splitX = Mathf.Round(innerW * pA);
            GUI.DrawTexture(new Rect(barX, barY, splitX, 11f), _texTeamA);
            GUI.DrawTexture(new Rect(barX + splitX, barY, Mathf.Max(0f, innerW - splitX), 11f), _texTeamB);

            // 50% centerline tick + live split cursor + optional AI subjective tick
            GUI.DrawTexture(new Rect(barX + Mathf.Round(innerW * 0.5f), barY - 1f, 1f, 13f), _texCenterTick);
            GUI.DrawTexture(new Rect(barX + Mathf.Clamp(splitX - 1f, 0f, innerW - 2f), barY - 1f, 2f, 13f), _texCursor);
            if (_hasSubjProbA)
            {
                float subjX = Mathf.Clamp(Mathf.Round(innerW * _smoothedSubjProbA) - 1f, 0f, innerW - 2f);
                GUI.DrawTexture(new Rect(barX + subjX, barY + 2f, 3f, 7f), _texSubjTick);
            }

            // Row 3: Compact In-Match Win-Probability Trajectory Sparkline (0s -> Now)
            GUI.DrawTexture(new Rect(barX, r.y + 35f, innerW, 14f), _texSparkline, ScaleMode.StretchToFill, false);
        }

        private static Texture2D MakeSolidTex(Color c)
        {
            var t = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            t.SetPixel(0, 0, c);
            t.Apply();
            return t;
        }

        private static Color32 ToColor32(Color c, byte alpha)
        {
            return new Color32(
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.r * 255f), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.g * 255f), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.b * 255f), 0, 255),
                alpha
            );
        }
    }
}
