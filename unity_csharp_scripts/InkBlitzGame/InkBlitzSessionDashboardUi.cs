// InkBlitzSessionDashboardUi.cs — Initial Mode Launcher (PBT Training vs Endless Pure-Inference Spectate)
// & Unobtrusive Top-Right Numerical Training UX Panel for InkBlitz 3D
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
    public class InkBlitzSessionDashboardUi : MonoBehaviour
    {
        [Serializable]
        private class CheckpointMetaJson
        {
            public int generation = 0;
            public float elo = 1200f;
            public float mean_score = 0f;
            public string archetype = "";
            public int train_steps_done = 0;
            public int sleep_cycles_done = 0;
        }

        [Serializable]
        private class PbtSlotJson
        {
            public int slot = 0;
            public string archetype = "";
            public float elo = 1200f;
            public float score = 0f;
            public int w = 0;
            public int l = 0;
            public int d = 0;
            public string opp = "";
        }

        [Serializable]
        private class PbtStatPacketJson
        {
            public string evt = "";
            public int gen = 0;
            public int total_gens = 0;
            public int eps_per_gen = 0;
            public int done_slots = 0;
            public int pop_size = 4;
            public string policy = "";
            public int champ_slot = 0;
            public string champ_arch = "";
            public float champ_elo = 1200f;
            public float champ_score = 0f;
            public float sleep_loss = 0f;
            public int sleep_epochs = 0;
            public int league_size = 0;
            public bool saved_ckpt = false;
            public string resumed_from = "";
            public int match = 0;
            public float score_a = 0f;
            public int wins_a = 0;
            public int wins_b = 0;
            public int draws = 0;
            public string source = "";
            public bool ollama_online = false;
            public string hypothesis = "";
            public string cause_tag = "";
            public string regime = "";
            public string lane = "";
            public string commentary = "";
            public string evt_kind = "";
            public string tts_text = "";
            public int validated_cnt = 0;
            public int candidate_cnt = 0;
            public int episodes_cnt = 0;
            public string glyph_a1 = "";
            public string glyph_a2 = "";
            public string glyph_b1 = "";
            public string glyph_b2 = "";
            public string gloss_a1 = "";
            public string gloss_a2 = "";
            public string gloss_b1 = "";
            public string gloss_b2 = "";
            public float joint_mi = 0f;
            public float slot_mi = 0f;
            public float phi_comm = 0f;
            public float phi_ally = 0f;
            public int unique_pkts = 0;
            public List<PbtSlotJson> slots = new List<PbtSlotJson>();
        }

        private class CheckpointEntry
        {
            public string fileName;
            public string relPath;
            public string label;
            public float elo;
            public int generation;
        }

        [Header("Scene References")]
        public UniversalArenaCoordinator3D coordinator;
        public InkBlitzMultiArenaManager multiManager;
        public InkBlitzNeuralVisualizerHud neuralHud;
        public InkBlitzWinRateEstimator winRateEstimator;
        public InkBlitzCompanionVoiceCoach voiceCoach;

        [Header("Launcher Behavior")]
        public bool showLauncherOnStart = true;

        [Header("Training Mode UI Defaults")]
        public int uiGenerations = 8;
        public int uiEpisodesPerGen = 2;
        public float uiTimeScale = 4.0f;
        public bool uiPureNeuralOnly = false;
        public int uiSleepMode = 1; // 0 = OFF, 1 = ON Standard (6ep S-1..S-4), 2 = DEEP Sleep (10ep)
        public bool uiResumeFromCheckpoint = true;
        public bool uiRepArenaOnly = true;
        public bool uiEnableSupportBots = true;

        [Header("Spectate Mode UI Defaults")]
        public bool uiSelfPlaySpectate = true;
        public bool uiSpectateContinualLearn = false;
        public bool uiSpectate2v2BrainRadio = false;
        public bool uiEnableGemmaBroadcast = true;
        public int uiOllamaComputeMode = 0; // 0 = hybrid (GPU+CPU split, default), 1 = cpu (num_gpu=0), 2 = gpu
        public bool showGemmaTickerInSpectate = true;
        public float spectateTimeScale = 1.0f;
        public int uiColorPaletteIndex = 0;
        public int uiStageId = 1; // 1 = Classic Citadel, 2 = Occluded Twin-Flank

        public bool IsLauncherOpen { get; private set; }
        public bool IsTrainingPanelCollapsed { get; private set; } = false;

        private readonly List<CheckpointEntry> _checkpoints = new List<CheckpointEntry>();
        private int _selectedCkptIdx = 0;
        private Vector2 _ckptScroll = Vector2.zero;
        private Vector2 _legalScroll = Vector2.zero;
        private Vector2 _leftCardScroll = Vector2.zero;
        private Vector2 _rightCardScroll = Vector2.zero;
        private bool _showLegalInfoPage = false;
        private bool _isSleepingNow = false;
        private int _activeSleepEpochs = 6;
        private string _genInputBuf = "8";
        private string _epsInputBuf = "2";
        private int _contGenerations = 4;
        private string _contGenInputBuf = "4";

        // Live Numerical PBT Stats
        private float _sessionStartRealtime = 0f;
        private float _avgSecPerGen = 0f;
        private int _gen = 0;
        private int _totalGens = 8;
        private int _epsPerGen = 2;
        private int _doneSlots = 0;
        private int _popSize = 4;
        private string _policy = "dagger";
        private int _champSlot = 0;
        private string _champArch = "-";
        private float _champElo = 1200f;
        private float _champScore = 0f;
        private float _sleepLoss = 0f;
        private int _leagueSize = 0;
        private bool _savedCkpt = false;
        private bool _trainingCompleted = false;
        private string _resumedFrom = "";
        private List<PbtSlotJson> _slotRows = new List<PbtSlotJson>();
        private readonly List<string> _genHistory = new List<string>();

        // Spectate Match Counter & Gemma Live Broadcast State
        private int _spectateMatchesReported = 0;
        private int _spectateWinsA = 0;
        private int _spectateWinsB = 0;
        private int _spectateDraws = 0;
        private bool _spectateSavedCkpt = false;
        private float _spectateSleepLoss = 0f;
        private string _gemmaSource = "";
        private bool _gemmaOllamaOnline = false;
        private string _gemmaHypothesis = "Seed_Balanced_Turf_Control";
        private string _gemmaRegime = "拮抗制圧";
        private string _gemmaLane = "BALANCED";
        private string _gemmaCommentary = "戦略知識DB待機中 — 試合展開・リスポーン・大局レジームに応じて4節目で戦略を適応注入し、試合終了後に振り返り実況を生成します。";
        private int _gemmaValidatedCnt = 0;
        private int _gemmaCandidateCnt = 0;
        private int _gemmaEpisodesCnt = 0;
        private string _rosettaGlyphA1 = "◈0·◆0·▶0";
        private string _rosettaGlyphA2 = "◈0·◆0·▶0";
        private string _rosettaGlyphB1 = "◈0·◆0·▶0";
        private string _rosettaGlyphB2 = "◈0·◆0·▶0";
        private string _rosettaGlossA1 = "待機中";
        private string _rosettaGlossA2 = "待機中";
        private float _rosettaJointMi = 0f;
        private float _rosettaPhiComm = 0f;
        private float _rosettaPhiAlly = 0f;
        private int _rosettaUniquePkts = 0;

        // Cached GUI textures & styles (Google Material 3 Dark / Luminous Dark 9-Slice System)
        private bool _stylesBuilt = false;
        private Texture2D _texOverlay;
        private Texture2D _texPanel;
        private Texture2D _texCard;
        private Texture2D _texInsetBox;
        private Texture2D _texChipIdle;
        private Texture2D _texChipHover;
        private Texture2D _texChipSelected;
        private Texture2D _texCtaTrainIdle;
        private Texture2D _texCtaTrainHover;
        private Texture2D _texCtaSpecIdle;
        private Texture2D _texCtaSpecHover;
        private Texture2D _texCtaDangerIdle;
        private Texture2D _texCtaDangerHover;
        private Texture2D _texSwitchOff;
        private Texture2D _texSwitchOn;
        private Texture2D _texAccentBlue;
        private Texture2D _texAccentOrange;
        private Texture2D _texAccentGreen;
        private Texture2D _texAccentCrimson;
        private Texture2D _texCrimsonModalBg;
        private Texture2D _texStopSwitchIdle;
        private Texture2D _texStopSwitchHover;
        private Texture2D _texStopSwitchHalted;
        private Texture2D _texBarBg;
        private GUIStyle _stPanel;
        private GUIStyle _stCrimsonModal;
        private GUIStyle _stCard;
        private GUIStyle _stInsetBox;
        private GUIStyle _stTitle;
        private GUIStyle _stHeader;
        private GUIStyle _stBody;
        private GUIStyle _stSmall;
        private GUIStyle _stBtnPrimary;
        private GUIStyle _stBtnSpectate;
        private GUIStyle _stBtnDanger;
        private GUIStyle _stBtnSub;
        private GUIStyle _stBtnChipSel;
        private GUIStyle _stListRowIdle;
        private GUIStyle _stListRowSel;
        private GUIStyle _stStopLabelIdle;
        private GUIStyle _stStopLabelHover;
        private GUIStyle _stTextField;
        private GUIStyle _stCoachTextField;
        private bool _coachInputFocused = false;

        private void Awake()
        {
            if (coordinator == null) coordinator = FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (multiManager == null) multiManager = FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (neuralHud == null) neuralHud = FindAnyObjectByType<InkBlitzNeuralVisualizerHud>();
            if (winRateEstimator == null) winRateEstimator = FindAnyObjectByType<InkBlitzWinRateEstimator>();
            if (voiceCoach == null) voiceCoach = FindAnyObjectByType<InkBlitzCompanionVoiceCoach>();
            if (voiceCoach == null)
            {
                voiceCoach = gameObject.AddComponent<InkBlitzCompanionVoiceCoach>();
                voiceCoach.coordinator = coordinator;
            }
            else if (voiceCoach.coordinator == null)
            {
                voiceCoach.coordinator = coordinator;
            }

            RefreshCheckpointList();

            bool autoStart = coordinator != null && coordinator.autoLaunchOnStart;
            IsLauncherOpen = showLauncherOnStart && !autoStart;
            if (multiManager != null)
                multiManager.isPausedForLauncher = IsLauncherOpen;

            if (coordinator != null)
            {
                _totalGens = Mathf.Max(1, coordinator.pbtGenerations);
                _epsPerGen = Mathf.Max(1, coordinator.pbtEpisodesPerGen);
                _popSize = Mathf.Max(1, coordinator.parallelArenaCount);
                _policy = string.IsNullOrEmpty(coordinator.pbtPolicyMode) ? "dagger" : coordinator.pbtPolicyMode;
                uiGenerations = _totalGens;
                uiEpisodesPerGen = _epsPerGen;
                uiTimeScale = Mathf.Clamp(coordinator.trainingTimeScale, 1f, 8f);
            }
            LoadUiPrefs();
            if (multiManager != null)
            {
                multiManager.SetActiveStage(uiStageId);
                multiManager.SetColorPalette(uiColorPaletteIndex);
            }
            _sessionStartRealtime = Time.realtimeSinceStartup;

            string resolvedWs = coordinator != null ? coordinator.GetResolvedWorkspaceDir() : "";
            Debug.Log($"[InkBlitzSessionDashboardUi] Game Ready! LauncherOpen={IsLauncherOpen}, Workspace='{resolvedWs}', CheckpointsFound={_checkpoints.Count}, Screen={Screen.width}x{Screen.height} ({Screen.fullScreenMode})");
        }

        private void LoadUiPrefs()
        {
            if (PlayerPrefs.HasKey("IB3D_Generations"))
                uiGenerations = Mathf.Clamp(PlayerPrefs.GetInt("IB3D_Generations", uiGenerations), 1, 500);
            if (PlayerPrefs.HasKey("IB3D_EpsPerGen"))
                uiEpisodesPerGen = Mathf.Clamp(PlayerPrefs.GetInt("IB3D_EpsPerGen", uiEpisodesPerGen), 1, 20);
            if (PlayerPrefs.HasKey("IB3D_TimeScale"))
                uiTimeScale = Mathf.Clamp(PlayerPrefs.GetFloat("IB3D_TimeScale", uiTimeScale), 1f, 8f);
            if (PlayerPrefs.HasKey("IB3D_PureNeural"))
                uiPureNeuralOnly = PlayerPrefs.GetInt("IB3D_PureNeural", uiPureNeuralOnly ? 1 : 0) != 0;
            if (PlayerPrefs.HasKey("IB3D_ResumeCkpt"))
                uiResumeFromCheckpoint = PlayerPrefs.GetInt("IB3D_ResumeCkpt", uiResumeFromCheckpoint ? 1 : 0) != 0;
            if (PlayerPrefs.HasKey("IB3D_RepOnly"))
                uiRepArenaOnly = PlayerPrefs.GetInt("IB3D_RepOnly", uiRepArenaOnly ? 1 : 0) != 0;
            if (PlayerPrefs.HasKey("IB3D_SelfPlaySpec"))
                uiSelfPlaySpectate = PlayerPrefs.GetInt("IB3D_SelfPlaySpec", uiSelfPlaySpectate ? 1 : 0) != 0;
            if (PlayerPrefs.HasKey("IB3D_SpecContLearn"))
                uiSpectateContinualLearn = PlayerPrefs.GetInt("IB3D_SpecContLearn", uiSpectateContinualLearn ? 1 : 0) != 0;
            if (PlayerPrefs.HasKey("IB3D_Spec2v2Radio"))
                uiSpectate2v2BrainRadio = PlayerPrefs.GetInt("IB3D_Spec2v2Radio", uiSpectate2v2BrainRadio ? 1 : 0) != 0;
            if (PlayerPrefs.HasKey("IB3D_GemmaBroadcast"))
                uiEnableGemmaBroadcast = PlayerPrefs.GetInt("IB3D_GemmaBroadcast", uiEnableGemmaBroadcast ? 1 : 0) != 0;
            if (PlayerPrefs.HasKey("IB3D_OllamaComputeMode"))
                uiOllamaComputeMode = Mathf.Clamp(PlayerPrefs.GetInt("IB3D_OllamaComputeMode", uiOllamaComputeMode), 0, 2);
            if (PlayerPrefs.HasKey("IB3D_SpecScale"))
                spectateTimeScale = Mathf.Clamp(PlayerPrefs.GetFloat("IB3D_SpecScale", spectateTimeScale), 0.5f, 4f);
            if (PlayerPrefs.HasKey("IB3D_SupportBots"))
                uiEnableSupportBots = PlayerPrefs.GetInt("IB3D_SupportBots", uiEnableSupportBots ? 1 : 0) != 0;
            if (PlayerPrefs.HasKey("IB3D_SleepMode"))
                uiSleepMode = Mathf.Clamp(PlayerPrefs.GetInt("IB3D_SleepMode", uiSleepMode), 0, 2);
            if (PlayerPrefs.HasKey("IB3D_ColorPalette"))
                uiColorPaletteIndex = Mathf.Clamp(PlayerPrefs.GetInt("IB3D_ColorPalette", uiColorPaletteIndex), 0, InkBlitzInkPalettes.Count - 1);
            if (PlayerPrefs.HasKey("IB3D_StageId"))
                uiStageId = (PlayerPrefs.GetInt("IB3D_StageId", uiStageId) == 2) ? 2 : 1;

            _genInputBuf = uiGenerations.ToString();
            _epsInputBuf = uiEpisodesPerGen.ToString();
            _contGenerations = Mathf.Clamp(uiGenerations, 1, 500);
            _contGenInputBuf = _contGenerations.ToString();
        }

        private void SaveUiPrefs()
        {
            PlayerPrefs.SetInt("IB3D_Generations", Mathf.Clamp(uiGenerations, 1, 500));
            PlayerPrefs.SetInt("IB3D_EpsPerGen", Mathf.Clamp(uiEpisodesPerGen, 1, 20));
            PlayerPrefs.SetFloat("IB3D_TimeScale", Mathf.Clamp(uiTimeScale, 1f, 8f));
            PlayerPrefs.SetInt("IB3D_PureNeural", uiPureNeuralOnly ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_ResumeCkpt", uiResumeFromCheckpoint ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_RepOnly", uiRepArenaOnly ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_SelfPlaySpec", uiSelfPlaySpectate ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_SpecContLearn", uiSpectateContinualLearn ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_Spec2v2Radio", uiSpectate2v2BrainRadio ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_GemmaBroadcast", uiEnableGemmaBroadcast ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_OllamaComputeMode", Mathf.Clamp(uiOllamaComputeMode, 0, 2));
            PlayerPrefs.SetFloat("IB3D_SpecScale", Mathf.Clamp(spectateTimeScale, 0.5f, 4f));
            PlayerPrefs.SetInt("IB3D_SupportBots", uiEnableSupportBots ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_SleepMode", Mathf.Clamp(uiSleepMode, 0, 2));
            PlayerPrefs.SetInt("IB3D_ColorPalette", Mathf.Clamp(uiColorPaletteIndex, 0, InkBlitzInkPalettes.Count - 1));
            PlayerPrefs.SetInt("IB3D_StageId", (uiStageId == 2) ? 2 : 1);
            PlayerPrefs.Save();
        }

        private void ApplyStageSelection(int stageId)
        {
            uiStageId = (stageId == 2) ? 2 : 1;
            if (coordinator != null)
                coordinator.activeStageId = uiStageId;
            if (multiManager != null)
                multiManager.SetActiveStage(uiStageId);
            PlayerPrefs.SetInt("IB3D_StageId", uiStageId);
        }

        private void ApplyPaletteSelection(int idx)
        {
            int n = InkBlitzInkPalettes.Count;
            uiColorPaletteIndex = ((idx % n) + n) % n;
            if (multiManager != null)
                multiManager.SetColorPalette(uiColorPaletteIndex);
            PlayerPrefs.SetInt("IB3D_ColorPalette", uiColorPaletteIndex);
        }

        private void SetGenerations(int v)
        {
            uiGenerations = Mathf.Clamp(v, 1, 500);
            _genInputBuf = uiGenerations.ToString();
        }

        private void SetEpisodesPerGen(int v)
        {
            uiEpisodesPerGen = Mathf.Clamp(v, 1, 20);
            _epsInputBuf = uiEpisodesPerGen.ToString();
        }

        private void SetContGenerations(int v)
        {
            _contGenerations = Mathf.Clamp(v, 1, 500);
            _contGenInputBuf = _contGenerations.ToString();
        }

        public void QuitGame()
        {
            SaveUiPrefs();
            if (voiceCoach != null)
            {
                voiceCoach.StopActiveSpeechImmediate();
                voiceCoach.StopSttSidecar();
            }
            if (coordinator != null)
                coordinator.StopPythonBackend();
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        private void OnDestroy()
        {
            DestroyTex(_texOverlay);
            DestroyTex(_texPanel);
            DestroyTex(_texCard);
            DestroyTex(_texInsetBox);
            DestroyTex(_texChipIdle);
            DestroyTex(_texChipHover);
            DestroyTex(_texChipSelected);
            DestroyTex(_texCtaTrainIdle);
            DestroyTex(_texCtaTrainHover);
            DestroyTex(_texCtaSpecIdle);
            DestroyTex(_texCtaSpecHover);
            DestroyTex(_texCtaDangerIdle);
            DestroyTex(_texCtaDangerHover);
            DestroyTex(_texSwitchOff);
            DestroyTex(_texSwitchOn);
            DestroyTex(_texAccentBlue);
            DestroyTex(_texAccentOrange);
            DestroyTex(_texAccentGreen);
            DestroyTex(_texAccentCrimson);
            DestroyTex(_texCrimsonModalBg);
            DestroyTex(_texStopSwitchIdle);
            DestroyTex(_texStopSwitchHover);
            DestroyTex(_texStopSwitchHalted);
            DestroyTex(_texBarBg);
        }

        private static void DestroyTex(Texture2D tex)
        {
            if (tex == null) return;
            if (Application.isPlaying) Destroy(tex);
            else DestroyImmediate(tex);
        }

        private void Update()
        {
            DrainPbtStatQueue();

            if (coordinator != null && coordinator.IsTrainingSession && !_trainingCompleted
                && coordinator.HasCompletedProcess && coordinator.LastExitCode == 0 && _gen > 0)
            {
                _trainingCompleted = true;
                RefreshCheckpointList();
            }

#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null)
            {
                bool editingText = GUIUtility.keyboardControl != 0;
                if (editingText)
                {
                    bool enterHit = kb.enterKey.wasPressedThisFrame || kb.numpadEnterKey.wasPressedThisFrame;
                    bool ctrlHeld = kb.leftCtrlKey.isPressed || kb.rightCtrlKey.isPressed;
                    if (kb.escapeKey.wasPressedThisFrame)
                    {
                        GUIUtility.keyboardControl = 0;
                        _coachInputFocused = false;
                    }
                    else if (_coachInputFocused)
                    {
                        // Keep focus on plain Enter so Japanese IME Kanji conversion works smoothly; Ctrl+Enter submits!
                        if (ctrlHeld && enterHit && voiceCoach != null)
                        {
                            voiceCoach.SubmitCoachingProposalToGemma();
                            GUIUtility.keyboardControl = 0;
                            _coachInputFocused = false;
                        }
                    }
                    else if (enterHit)
                    {
                        GUIUtility.keyboardControl = 0;
                    }
                }
                else if (IsLauncherOpen && !editingText)
                {
                    if (_showLegalInfoPage)
                    {
                        if (kb.escapeKey.wasPressedThisFrame)
                            _showLegalInfoPage = false;
                    }
                    else
                    {
                        if (kb.tKey.wasPressedThisFrame)
                            StartTrainingSession();
                        else if (kb.sKey.wasPressedThisFrame)
                            StartSpectateSession();
                        else if (kb.rKey.wasPressedThisFrame)
                            RefreshCheckpointList();
                        else if (kb.pKey.wasPressedThisFrame)
                            ApplyPaletteSelection(uiColorPaletteIndex + 1);
                        else if (kb.mKey.wasPressedThisFrame)
                            ApplyStageSelection(uiStageId == 1 ? 2 : 1);
                    }
                }
                else if (!IsLauncherOpen && !editingText)
                {
                    if (multiManager != null && multiManager.activePaletteIndex != uiColorPaletteIndex)
                    {
                        uiColorPaletteIndex = multiManager.activePaletteIndex;
                        PlayerPrefs.SetInt("IB3D_ColorPalette", uiColorPaletteIndex);
                    }
                    bool shiftHeld = kb.leftShiftKey.isPressed || kb.rightShiftKey.isPressed;
                    if (kb.f12Key.wasPressedThisFrame || (shiftHeld && kb.escapeKey.wasPressedThisFrame))
                    {
                        ExecuteEmergencyStop();
                    }
                    else if (kb.f2Key.wasPressedThisFrame)
                    {
                        IsTrainingPanelCollapsed = !IsTrainingPanelCollapsed;
                    }
                    else if (kb.gKey.wasPressedThisFrame && coordinator != null && !coordinator.IsTrainingSession)
                    {
                        showGemmaTickerInSpectate = !showGemmaTickerInSpectate;
                    }
                    else if (kb.nKey.wasPressedThisFrame && voiceCoach != null)
                    {
                        voiceCoach.SetVoiceNarrationEnabled(!voiceCoach.enableVoiceNarration);
                    }
                    else if (kb.escapeKey.wasPressedThisFrame)
                    {
                        ReturnToLauncherMenu();
                    }
                }
            }
#endif
        }

        private void DrainPbtStatQueue()
        {
            if (coordinator == null) return;
            while (coordinator.PbtStatQueue.TryDequeue(out string raw))
            {
                if (string.IsNullOrEmpty(raw)) continue;
                PbtStatPacketJson pkt;
                try { pkt = JsonUtility.FromJson<PbtStatPacketJson>(raw); }
                catch { continue; }
                if (pkt == null || string.IsNullOrEmpty(pkt.evt)) continue;

                if (pkt.evt == "gen_start")
                {
                    _isSleepingNow = false;
                    _gen = pkt.gen;
                    _totalGens = Mathf.Max(1, pkt.total_gens);
                    _epsPerGen = Mathf.Max(1, pkt.eps_per_gen);
                    _doneSlots = 0;
                    _popSize = Mathf.Max(1, pkt.pop_size);
                    _policy = pkt.policy;
                    _leagueSize = pkt.league_size;
                    _resumedFrom = pkt.resumed_from;
                    if (pkt.slots != null && pkt.slots.Count > 0)
                        _slotRows = pkt.slots;
                }
                else if (pkt.evt == "slot_done")
                {
                    _gen = pkt.gen;
                    _doneSlots = pkt.done_slots;
                }
                else if (pkt.evt == "sleep_start")
                {
                    _gen = pkt.gen;
                    _isSleepingNow = true;
                    _activeSleepEpochs = Mathf.Max(1, pkt.sleep_epochs);
                }
                else if (pkt.evt == "gen_done")
                {
                    _isSleepingNow = false;
                    _gen = pkt.gen;
                    _totalGens = Mathf.Max(1, pkt.total_gens);
                    _doneSlots = pkt.done_slots;
                    _policy = pkt.policy;
                    _champSlot = pkt.champ_slot;
                    _champArch = string.IsNullOrEmpty(pkt.champ_arch) ? "-" : pkt.champ_arch;
                    _champElo = pkt.champ_elo;
                    _champScore = pkt.champ_score;
                    _sleepLoss = pkt.sleep_loss;
                    _leagueSize = pkt.league_size;
                    _savedCkpt = pkt.saved_ckpt;
                    if (pkt.slots != null && pkt.slots.Count > 0)
                        _slotRows = pkt.slots;
                    float elapsed = Mathf.Max(0.1f, Time.realtimeSinceStartup - _sessionStartRealtime);
                    _avgSecPerGen = elapsed / Mathf.Max(1, _gen);
                    _genHistory.Add($"G{pkt.gen}:E{pkt.champ_elo:F0}");
                    if (_genHistory.Count > 6)
                        _genHistory.RemoveAt(0);
                    if (_savedCkpt)
                        RefreshCheckpointList();
                }
                else if (pkt.evt == "train_complete")
                {
                    _isSleepingNow = false;
                    _trainingCompleted = true;
                    _champSlot = pkt.champ_slot;
                    _champArch = string.IsNullOrEmpty(pkt.champ_arch) ? _champArch : pkt.champ_arch;
                    _champElo = pkt.champ_elo;
                    _champScore = pkt.champ_score;
                    _sleepLoss = pkt.sleep_loss;
                    _leagueSize = pkt.league_size;
                    _savedCkpt = pkt.saved_ckpt;
                    if (pkt.slots != null && pkt.slots.Count > 0)
                        _slotRows = pkt.slots;
                    RefreshCheckpointList();
                }
                else if (pkt.evt == "spectate_match")
                {
                    _spectateMatchesReported = pkt.match;
                    _spectateWinsA = pkt.wins_a;
                    _spectateWinsB = pkt.wins_b;
                    _spectateDraws = pkt.draws;
                    _spectateSavedCkpt = pkt.saved_ckpt;
                    _spectateSleepLoss = pkt.sleep_loss;
                }
                else if (pkt.evt == "gemma_broadcast")
                {
                    if (!string.IsNullOrEmpty(pkt.source)) _gemmaSource = pkt.source;
                    _gemmaOllamaOnline = pkt.ollama_online;
                    if (!string.IsNullOrEmpty(pkt.hypothesis)) _gemmaHypothesis = pkt.hypothesis;
                    if (!string.IsNullOrEmpty(pkt.regime)) _gemmaRegime = pkt.regime;
                    if (!string.IsNullOrEmpty(pkt.lane)) _gemmaLane = pkt.lane;
                    if (!string.IsNullOrEmpty(pkt.commentary)) _gemmaCommentary = pkt.commentary;
                    _gemmaValidatedCnt = pkt.validated_cnt;
                    _gemmaCandidateCnt = pkt.candidate_cnt;
                    _gemmaEpisodesCnt = pkt.episodes_cnt;
                    if (voiceCoach != null)
                        voiceCoach.NotifyBroadcastEvent(pkt.evt_kind, pkt.tts_text);
                }
                else if (pkt.evt == "rosetta_radio")
                {
                    if (!string.IsNullOrEmpty(pkt.glyph_a1)) _rosettaGlyphA1 = pkt.glyph_a1;
                    if (!string.IsNullOrEmpty(pkt.glyph_a2)) _rosettaGlyphA2 = pkt.glyph_a2;
                    if (!string.IsNullOrEmpty(pkt.glyph_b1)) _rosettaGlyphB1 = pkt.glyph_b1;
                    if (!string.IsNullOrEmpty(pkt.glyph_b2)) _rosettaGlyphB2 = pkt.glyph_b2;
                    if (!string.IsNullOrEmpty(pkt.gloss_a1)) _rosettaGlossA1 = pkt.gloss_a1;
                    if (!string.IsNullOrEmpty(pkt.gloss_a2)) _rosettaGlossA2 = pkt.gloss_a2;
                    _rosettaJointMi = pkt.joint_mi;
                    _rosettaPhiComm = pkt.phi_comm;
                    _rosettaPhiAlly = pkt.phi_ally;
                    _rosettaUniquePkts = pkt.unique_pkts;
                }
            }
        }

        public void RefreshCheckpointList()
        {
            string prevRel = (_selectedCkptIdx >= 0 && _selectedCkptIdx < _checkpoints.Count)
                ? _checkpoints[_selectedCkptIdx].relPath
                : "";
            _checkpoints.Clear();

            string ws = coordinator != null ? coordinator.GetResolvedWorkspaceDir() : UniversalArenaCoordinator3D.ResolveDefaultWorkspaceDir();
            string dir = Path.Combine(ws, "checkpoints", "user_runs");
            if (Directory.Exists(dir))
            {
                string[] files = Directory.GetFiles(dir, "*.pt", SearchOption.TopDirectoryOnly);
                Array.Sort(files, (a, b) => File.GetLastWriteTimeUtc(b).CompareTo(File.GetLastWriteTimeUtc(a)));
                for (int i = 0; i < files.Length; i++)
                {
                    string f = files[i];
                    string name = Path.GetFileName(f);
                    string rel = Path.Combine("checkpoints", "user_runs", name);
                    CheckpointMetaJson meta = ReadCheckpointMeta(f);
                    string tag = meta.generation > 0
                        ? $"Gen {meta.generation} | Elo {meta.elo:F0} | Steps {meta.train_steps_done}"
                        : (meta.train_steps_done > 0 ? $"Steps {meta.train_steps_done} | Elo {meta.elo:F0}" : "Ready");
                    _checkpoints.Add(new CheckpointEntry
                    {
                        fileName = name,
                        relPath = rel,
                        label = $"{name}  ({tag})",
                        elo = meta.elo,
                        generation = meta.generation
                    });
                }
            }

            if (_checkpoints.Count == 0)
            {
                _selectedCkptIdx = 0;
                return;
            }

            int pick = -1;
            if (!string.IsNullOrEmpty(prevRel))
                pick = _checkpoints.FindIndex(c => string.Equals(c.relPath, prevRel, StringComparison.OrdinalIgnoreCase));
            if (pick < 0)
                pick = _checkpoints.FindIndex(c => string.Equals(c.fileName, "pbt_latest_champion.pt", StringComparison.OrdinalIgnoreCase));
            if (pick < 0)
                pick = _checkpoints.FindIndex(c => string.Equals(c.fileName, "pbt_best_champion.pt", StringComparison.OrdinalIgnoreCase));
            _selectedCkptIdx = Mathf.Clamp(pick >= 0 ? pick : 0, 0, _checkpoints.Count - 1);
        }

        private static CheckpointMetaJson ReadCheckpointMeta(string ptPath)
        {
            string metaPath = ptPath + ".meta.json";
            if (!File.Exists(metaPath))
                return new CheckpointMetaJson();
            try
            {
                string raw = File.ReadAllText(metaPath);
                return JsonUtility.FromJson<CheckpointMetaJson>(raw) ?? new CheckpointMetaJson();
            }
            catch
            {
                return new CheckpointMetaJson();
            }
        }

        private string GetSelectedCheckpointRelativePath()
        {
            if (_checkpoints.Count == 0 || _selectedCkptIdx < 0 || _selectedCkptIdx >= _checkpoints.Count)
                return "";
            return _checkpoints[_selectedCkptIdx].relPath;
        }

        private static string ResolveOllamaComputeModeName(int modeIdx)
        {
            if (modeIdx == 1) return "cpu";
            if (modeIdx == 2) return "gpu";
            return "hybrid";
        }

        public void StartTrainingSession()
        {
            if (coordinator == null || multiManager == null) return;
            GUIUtility.keyboardControl = 0;
            SaveUiPrefs();

            _gen = 0;
            _totalGens = Mathf.Max(1, uiGenerations);
            _epsPerGen = Mathf.Max(1, uiEpisodesPerGen);
            _doneSlots = 0;
            _popSize = Mathf.Clamp(coordinator.parallelArenaCount, 2, 8);
            _policy = uiPureNeuralOnly ? "pure_neural" : "dagger";
            _champSlot = 0;
            _champArch = "-";
            _champElo = 1200f;
            _champScore = 0f;
            _sleepLoss = 0f;
            _leagueSize = 0;
            _savedCkpt = false;
            _trainingCompleted = false;
            _isSleepingNow = false;
            _resumedFrom = "";
            _slotRows.Clear();
            _genHistory.Clear();
            _sessionStartRealtime = Time.realtimeSinceStartup;
            _avgSecPerGen = 0f;

            coordinator.sessionMode = BrainSessionRunMode.ParallelCompetitiveSelfPlay;
            coordinator.pbtGenerations = _totalGens;
            coordinator.pbtEpisodesPerGen = _epsPerGen;
            coordinator.trainingTimeScale = Mathf.Clamp(uiTimeScale, 1f, 8f);
            coordinator.pbtPolicyMode = _policy;
            coordinator.enableSleepConsolidation = (uiSleepMode > 0);
            coordinator.sleepConsolidationEpochs = (uiSleepMode == 2) ? 10 : ((uiSleepMode == 1) ? 6 : 0);
            coordinator.activeStageId = (uiStageId == 2) ? 2 : 1;
            coordinator.enableStrategyBroadcast = uiEnableGemmaBroadcast;
            coordinator.ollamaComputeMode = ResolveOllamaComputeModeName(uiOllamaComputeMode);
            coordinator.pbtCheckpointDir = @"checkpoints\user_runs";
            string chosen = GetSelectedCheckpointRelativePath();
            coordinator.resumeFromCheckpointInPbt = uiResumeFromCheckpoint && !string.IsNullOrEmpty(chosen);
            if (!string.IsNullOrEmpty(chosen))
                coordinator.loadCheckpointRelativePath = chosen;

            multiManager.ClearEmergencyStopForReset();
            multiManager.activeStageId = coordinator.activeStageId;
            multiManager.enableSupportTrooperBots = uiEnableSupportBots;
            multiManager.activePaletteIndex = uiColorPaletteIndex;
            multiManager.renderRepresentativeArenaOnly = uiRepArenaOnly;
            multiManager.disableAll3DRendering = false;
            multiManager.isPausedForLauncher = false;
            IsLauncherOpen = false;
            IsTrainingPanelCollapsed = false;

            multiManager.ReconfigureArenasForCurrentSession();
            coordinator.LaunchPythonBackendAndConnect();
        }

        public void StartSpectateSession()
        {
            if (coordinator == null || multiManager == null) return;
            GUIUtility.keyboardControl = 0;
            SaveUiPrefs();

            _spectateMatchesReported = 0;
            _spectateWinsA = 0;
            _spectateWinsB = 0;
            _spectateDraws = 0;
            _spectateSavedCkpt = false;
            _spectateSleepLoss = 0f;
            _trainingCompleted = false;
            _sessionStartRealtime = Time.realtimeSinceStartup;

            if (uiSpectate2v2BrainRadio)
                uiSelfPlaySpectate = true;
            coordinator.sessionMode = uiSelfPlaySpectate
                ? BrainSessionRunMode.SingleArenaSelfPlaySpectate
                : BrainSessionRunMode.SinglePlaySpectate;
            coordinator.singlePlayPolicyMode = "pure_neural";
            coordinator.activeStageId = (uiStageId == 2) ? 2 : 1;
            coordinator.spectateContinualLearning = uiSpectateContinualLearn;
            coordinator.spectate2v2BrainRadio = uiSelfPlaySpectate && uiSpectate2v2BrainRadio;
            coordinator.enableStrategyBroadcast = uiEnableGemmaBroadcast;
            coordinator.ollamaComputeMode = ResolveOllamaComputeModeName(uiOllamaComputeMode);
            coordinator.loadCheckpointRelativePath = GetSelectedCheckpointRelativePath();
            coordinator.oppCheckpointRelativePath = "";

            multiManager.ClearEmergencyStopForReset();
            multiManager.activeStageId = coordinator.activeStageId;
            multiManager.enableSupportTrooperBots = uiEnableSupportBots;
            multiManager.activePaletteIndex = uiColorPaletteIndex;
            multiManager.disableAll3DRendering = false;
            multiManager.isPausedForLauncher = false;
            IsLauncherOpen = false;

            multiManager.ReconfigureArenasForCurrentSession();
            coordinator.LaunchPythonBackendAndConnect();
            ApplySpectateTimeScale(spectateTimeScale);
        }

        private void ApplySpectateTimeScale(float scale)
        {
            float clamped = Mathf.Clamp(scale, 0.5f, 4f);
            spectateTimeScale = clamped;
            Time.timeScale = clamped;
            // At <= 1.0x spectate, cap to 1 FixedUpdate per render frame so a minor frame hiccup
            // never triggers back-to-back blocking TCP lockstep round-trips (30Hz beat judder).
            float maxSteps = (clamped <= 1.05f) ? 1.25f : Mathf.Ceil(clamped * 1.5f);
            Time.maximumDeltaTime = Mathf.Max(Time.fixedDeltaTime, Time.fixedDeltaTime * maxSteps);
        }

        private void ApplyLauncherSupportBotPreview(bool enabled)
        {
            uiEnableSupportBots = enabled;
            if (multiManager != null)
            {
                multiManager.enableSupportTrooperBots = enabled;
                if (multiManager.primaryArena != null)
                    multiManager.primaryArena.SetSupportTrooperBotsEnabled(enabled);
            }
        }

        public void ReturnToLauncherMenu()
        {
            GUIUtility.keyboardControl = 0;
            _coachInputFocused = false;
            if (voiceCoach != null)
            {
                voiceCoach.StopActiveSpeechImmediate();
                voiceCoach.StopSttSidecar();
            }
            if (multiManager != null)
                multiManager.ClearEmergencyStopForReset();
            if (coordinator != null)
            {
                coordinator.StopPythonBackend();
                coordinator.ClearRegisteredBridges();
            }
            if (multiManager != null)
            {
                if (coordinator != null)
                    coordinator.sessionMode = BrainSessionRunMode.SinglePlaySpectate;
                multiManager.ReconfigureArenasForCurrentSession();
                multiManager.isPausedForLauncher = true;
            }
            Time.timeScale = 1.0f;
            RefreshCheckpointList();
            IsLauncherOpen = true;
        }

        public void ExecuteEmergencyStop()
        {
            GUIUtility.keyboardControl = 0;
            _coachInputFocused = false;
            if (voiceCoach != null)
            {
                voiceCoach.StopActiveSpeechImmediate();
                voiceCoach.StopSttSidecar();
            }
            if (multiManager != null)
                multiManager.TriggerEmergencyStop();
            else if (coordinator != null)
                coordinator.EmergencyHaltAll();
        }

        public Rect GetEmergencyStopButtonRect()
        {
            // Exact 16:9 horizontal landscape aspect ratio (96 x 54 px) docked cleanly at top-left of center viewport (380..476)
            const float btnW = 96f;
            const float btnH = 54f;
            float x0 = Mathf.Clamp(380f, 12f, Mathf.Max(12f, Screen.width - btnW - 12f));
            return new Rect(x0, 11f, btnW, btnH);
        }

        public Rect GetSpectateStatusStripRect()
        {
            float rightInset = (neuralHud != null) ? neuralHud.ActiveRightPanelWidthPx : 0f;
            float panelW = 352f;
            float panelH = 132f;
            float x0 = Mathf.Max(486f, Screen.width - rightInset - panelW - 12f);
            bool winBarNeedsTopRow = (winRateEstimator != null && winRateEstimator.showWinRateBar) && (Screen.width < 1660);
            float y0 = winBarNeedsTopRow ? 72f : 12f;
            return new Rect(x0, y0, panelW, panelH);
        }

        public Rect GetGemmaBroadcastRect()
        {
            if (neuralHud == null)
                neuralHud = GetComponent<InkBlitzNeuralVisualizerHud>() ?? UnityEngine.Object.FindAnyObjectByType<InkBlitzNeuralVisualizerHud>();

            float leftInset = 156f;
            if (neuralHud != null)
                leftInset = neuralHud.GetWorldModelWindowRect().xMax + 8f;
            float rightInset = (neuralHud != null) ? neuralHud.ActiveRightPanelWidthPx : 0f;
            float rightLim = Screen.width - rightInset - 12f;
            if (!showGemmaTickerInSpectate)
                return new Rect(leftInset, Screen.height - 26f, 168f, 22f);
            float avail = Mathf.Max(260f, rightLim - leftInset);
            float w = Mathf.Clamp(avail, Mathf.Min(308f, avail), 820f);
            return new Rect(leftInset, Screen.height - 118f, w, 114f);
        }

        public Rect GetEmergencyModalRect()
        {
            float mw = 480f;
            float mh = 228f;
            float rightInset = (coordinator != null && !coordinator.IsTrainingSession && neuralHud != null)
                ? neuralHud.ActiveRightPanelWidthPx
                : 0f;
            float availW = Mathf.Max(mw + 24f, Screen.width - rightInset);
            return new Rect((availW - mw) * 0.5f, (Screen.height - mh) * 0.5f, mw, mh);
        }

        private bool _texturesBuilt = false;

        public void EnsureMaterialStylesReady()
        {
            EnsureThemeTextures();
        }

        public bool HasValidMaterialThemeAssets =>
            _texturesBuilt && _texPanel != null && _texCard != null && _texChipIdle != null
            && _texChipSelected != null && _texSwitchOn != null && _texSwitchOff != null;

        private void EnsureThemeTextures()
        {
            if (_texturesBuilt && _texPanel != null) return;
            _texturesBuilt = true;

            // Google Material 3 Dark / Luminous Dark Surface & Tonal Palette
            _texOverlay = MakeSolidTex(new Color(0.031f, 0.039f, 0.055f, 0.78f));
            _texPanel = BuildRounded9SliceTex(48, 18f, new Color(0.075f, 0.084f, 0.104f, 0.98f), new Color(0.180f, 0.200f, 0.245f, 0.98f));
            _texCard = BuildRounded9SliceTex(40, 14f, new Color(0.106f, 0.118f, 0.145f, 0.98f), new Color(0.184f, 0.208f, 0.255f, 0.98f));
            _texInsetBox = BuildRounded9SliceTex(32, 9f, new Color(0.067f, 0.075f, 0.094f, 0.98f), new Color(0.215f, 0.243f, 0.300f, 0.98f));

            // M3 Tonal Pill Buttons (`border-radius: 999px` feel via 14px radius on 28-32px height)
            _texChipIdle = BuildRounded9SliceTex(32, 13f, new Color(0.145f, 0.162f, 0.200f, 0.98f), new Color(0.235f, 0.263f, 0.322f, 0.98f));
            _texChipHover = BuildRounded9SliceTex(32, 13f, new Color(0.196f, 0.220f, 0.272f, 0.99f), new Color(0.345f, 0.388f, 0.475f, 0.99f));
            _texChipSelected = BuildRounded9SliceTex(32, 13f, new Color(0.102f, 0.173f, 0.355f, 0.99f), new Color(0.560f, 0.760f, 1.000f, 0.99f));

            // M3 Primary Pill CTAs (42px height -> 18px radius)
            _texCtaTrainIdle = BuildRounded9SliceTex(44, 18f, new Color(0.74f, 0.35f, 0.14f, 0.99f), new Color(0.98f, 0.64f, 0.40f, 0.95f));
            _texCtaTrainHover = BuildRounded9SliceTex(44, 18f, new Color(0.85f, 0.42f, 0.18f, 1.00f), new Color(1.00f, 0.76f, 0.54f, 1.00f));
            _texCtaSpecIdle = BuildRounded9SliceTex(44, 18f, new Color(0.13f, 0.37f, 0.68f, 0.99f), new Color(0.58f, 0.80f, 1.00f, 0.96f));
            _texCtaSpecHover = BuildRounded9SliceTex(44, 18f, new Color(0.17f, 0.45f, 0.80f, 1.00f), new Color(0.74f, 0.88f, 1.00f, 1.00f));
            _texCtaDangerIdle = BuildRounded9SliceTex(36, 15f, new Color(0.58f, 0.15f, 0.18f, 0.99f), new Color(0.92f, 0.38f, 0.40f, 0.95f));
            _texCtaDangerHover = BuildRounded9SliceTex(36, 15f, new Color(0.72f, 0.20f, 0.23f, 1.00f), new Color(1.00f, 0.52f, 0.54f, 1.00f));

            // M3 Pill Switch Sprites (`38 x 20 px`)
            _texSwitchOff = BuildM3SwitchSprite(false);
            _texSwitchOn = BuildM3SwitchSprite(true);

            _texAccentBlue = MakeSolidTex(new Color(0.36f, 0.66f, 0.98f, 1.0f));
            _texAccentOrange = MakeSolidTex(new Color(0.94f, 0.54f, 0.24f, 1.0f));
            _texAccentGreen = MakeSolidTex(new Color(0.24f, 0.78f, 0.48f, 1.0f));
            _texAccentCrimson = MakeSolidTex(new Color(0.82f, 0.24f, 0.26f, 1.0f));
            _texCrimsonModalBg = BuildRounded9SliceTex(40, 14f, new Color(0.09f, 0.05f, 0.06f, 0.98f), new Color(0.75f, 0.22f, 0.24f, 0.98f));
            _texStopSwitchIdle = BuildCyberStopSwitchSprite(
                new Color32(158, 60, 61, 248),
                new Color32(202, 41, 42, 245),
                new Color32(10, 12, 15, 240));
            _texStopSwitchHover = BuildCyberStopSwitchSprite(
                new Color32(236, 86, 88, 255),
                new Color32(255, 68, 70, 255),
                new Color32(34, 13, 16, 250));
            _texStopSwitchHalted = BuildCyberStopSwitchSprite(
                new Color32(255, 72, 76, 255),
                new Color32(255, 72, 76, 255),
                new Color32(62, 16, 20, 252));
            _texBarBg = MakeSolidTex(new Color(0.06f, 0.08f, 0.11f, 1.0f));
        }

        private void EnsureStyles()
        {
            EnsureThemeTextures();
            if (_stylesBuilt && _stPanel != null) return;
            _stylesBuilt = true;

            _stPanel = new GUIStyle(GUI.skin.box)
            {
                border = new RectOffset(18, 18, 18, 18),
                padding = new RectOffset(14, 14, 12, 12)
            };
            _stPanel.normal.background = _texPanel;

            _stCrimsonModal = new GUIStyle(GUI.skin.box)
            {
                border = new RectOffset(14, 14, 14, 14),
                padding = new RectOffset(14, 14, 12, 12)
            };
            _stCrimsonModal.normal.background = _texCrimsonModalBg;

            _stCard = new GUIStyle(GUI.skin.box)
            {
                border = new RectOffset(14, 14, 14, 14),
                padding = new RectOffset(14, 14, 12, 12)
            };
            _stCard.normal.background = _texCard;

            _stInsetBox = new GUIStyle(GUI.skin.box)
            {
                border = new RectOffset(9, 9, 9, 9),
                padding = new RectOffset(6, 6, 4, 4)
            };
            _stInsetBox.normal.background = _texInsetBox;

            _stTitle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 18,
                fontStyle = FontStyle.Bold,
                richText = true
            };
            _stTitle.normal.textColor = new Color(0.93f, 0.95f, 0.98f, 1f);

            _stHeader = new GUIStyle(GUI.skin.label)
            {
                fontSize = 14,
                fontStyle = FontStyle.Bold,
                richText = true
            };
            _stHeader.normal.textColor = new Color(0.88f, 0.94f, 1f, 1f);

            _stBody = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                richText = true,
                wordWrap = true,
                alignment = TextAnchor.MiddleLeft
            };
            _stBody.normal.textColor = new Color(0.89f, 0.91f, 0.95f, 1f);

            _stSmall = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                richText = true,
                wordWrap = true
            };
            _stSmall.normal.textColor = new Color(0.68f, 0.74f, 0.83f, 1f);

            _stBtnPrimary = new GUIStyle(GUI.skin.button)
            {
                border = new RectOffset(18, 18, 18, 18),
                fontSize = 14,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                padding = new RectOffset(16, 16, 6, 6)
            };
            _stBtnPrimary.normal.background = _texCtaTrainIdle;
            _stBtnPrimary.hover.background = _texCtaTrainHover;
            _stBtnPrimary.active.background = _texCtaTrainIdle;
            _stBtnPrimary.normal.textColor = new Color(0.99f, 0.97f, 0.95f, 1f);
            _stBtnPrimary.hover.textColor = Color.white;

            _stBtnSpectate = new GUIStyle(GUI.skin.button)
            {
                border = new RectOffset(18, 18, 18, 18),
                fontSize = 14,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                padding = new RectOffset(16, 16, 6, 6)
            };
            _stBtnSpectate.normal.background = _texCtaSpecIdle;
            _stBtnSpectate.hover.background = _texCtaSpecHover;
            _stBtnSpectate.active.background = _texCtaSpecIdle;
            _stBtnSpectate.normal.textColor = new Color(0.96f, 0.98f, 1.0f, 1f);
            _stBtnSpectate.hover.textColor = Color.white;

            _stBtnDanger = new GUIStyle(GUI.skin.button)
            {
                border = new RectOffset(15, 15, 15, 15),
                fontSize = 13,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                padding = new RectOffset(12, 12, 4, 4)
            };
            _stBtnDanger.normal.background = _texCtaDangerIdle;
            _stBtnDanger.hover.background = _texCtaDangerHover;
            _stBtnDanger.active.background = _texCtaDangerIdle;
            _stBtnDanger.normal.textColor = Color.white;
            _stBtnDanger.hover.textColor = Color.white;

            _stBtnSub = new GUIStyle(GUI.skin.button)
            {
                border = new RectOffset(13, 13, 13, 13),
                fontSize = 12,
                fontStyle = FontStyle.Bold,
                richText = true,
                alignment = TextAnchor.MiddleCenter,
                margin = new RectOffset(2, 2, 1, 1),
                padding = new RectOffset(10, 10, 4, 4)
            };
            _stBtnSub.normal.background = _texChipIdle;
            _stBtnSub.hover.background = _texChipHover;
            _stBtnSub.active.background = _texChipSelected;
            _stBtnSub.normal.textColor = new Color(0.86f, 0.89f, 0.94f, 1f);
            _stBtnSub.hover.textColor = Color.white;

            _stBtnChipSel = new GUIStyle(_stBtnSub);
            _stBtnChipSel.normal.background = _texChipSelected;
            _stBtnChipSel.hover.background = _texChipSelected;
            _stBtnChipSel.normal.textColor = new Color(0.84f, 0.93f, 1.0f, 1f);
            _stBtnChipSel.hover.textColor = Color.white;

            _stListRowIdle = new GUIStyle(_stBtnSub)
            {
                alignment = TextAnchor.MiddleLeft,
                fontStyle = FontStyle.Normal,
                padding = new RectOffset(10, 10, 4, 4)
            };
            _stListRowSel = new GUIStyle(_stBtnChipSel)
            {
                alignment = TextAnchor.MiddleLeft,
                fontStyle = FontStyle.Bold,
                padding = new RectOffset(10, 10, 4, 4)
            };

            _stStopLabelIdle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 13,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter
            };
            _stStopLabelIdle.normal.textColor = new Color(180f / 255f, 68f / 255f, 70f / 255f, 1f);

            _stStopLabelHover = new GUIStyle(_stStopLabelIdle);
            _stStopLabelHover.normal.textColor = new Color(250f / 255f, 98f / 255f, 100f / 255f, 1f);

            _stTextField = new GUIStyle(GUI.skin.textField)
            {
                border = new RectOffset(9, 9, 9, 9),
                fontSize = 13,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                padding = new RectOffset(6, 6, 3, 3)
            };
            _stTextField.normal.background = _texInsetBox;
            _stTextField.focused.background = _texChipSelected;
            _stTextField.hover.background = _texChipIdle;
            _stTextField.normal.textColor = Color.white;
            _stTextField.focused.textColor = Color.white;
            _stTextField.hover.textColor = Color.white;

            _stCoachTextField = new GUIStyle(GUI.skin.textField)
            {
                border = new RectOffset(9, 9, 9, 9),
                fontSize = 12,
                alignment = TextAnchor.MiddleLeft,
                padding = new RectOffset(8, 8, 3, 3)
            };
            _stCoachTextField.normal.background = _texInsetBox;
            _stCoachTextField.focused.background = _texChipSelected;
            _stCoachTextField.normal.textColor = Color.white;
            _stCoachTextField.focused.textColor = Color.white;
        }

        private static Texture2D BuildRounded9SliceTex(int size, float radius, Color fill, Color border)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            float half = size * 0.5f;
            float r = Mathf.Min(radius, half - 1.2f);
            var cols = new Color32[size * size];
            for (int y = 0; y < size; y++)
            {
                float fy = y + 0.5f - half;
                for (int x = 0; x < size; x++)
                {
                    float fx = x + 0.5f - half;
                    float qx = Mathf.Abs(fx) - (half - 1f - r);
                    float qy = Mathf.Abs(fy) - (half - 1f - r);
                    float ox = qx > 0f ? qx : 0f;
                    float oy = qy > 0f ? qy : 0f;
                    float sdf = Mathf.Sqrt(ox * ox + oy * oy) + Mathf.Min(Mathf.Max(qx, qy), 0f) - r;
                    float cov = Mathf.Clamp01(0.65f - sdf);
                    float bMix = Mathf.Clamp01(1.25f - Mathf.Abs(sdf + 0.55f));
                    Color c = Color.Lerp(fill, border, bMix);
                    c.a *= cov;
                    cols[y * size + x] = c;
                }
            }
            tex.SetPixels32(cols);
            tex.Apply(false);
            return tex;
        }

        private static Texture2D BuildM3SwitchSprite(bool isOn)
        {
            const int W = 38;
            const int H = 20;
            var tex = new Texture2D(W, H, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            var cols = new Color32[W * H];
            Color trackFill = isOn ? new Color(0.32f, 0.62f, 0.98f, 1f) : new Color(0.16f, 0.18f, 0.22f, 1f);
            Color trackBrd = isOn ? new Color(0.62f, 0.82f, 1.00f, 1f) : new Color(0.42f, 0.46f, 0.54f, 1f);
            Color knobCol = isOn ? new Color(0.98f, 0.99f, 1.00f, 1f) : new Color(0.62f, 0.66f, 0.74f, 1f);
            float knobCx = isOn ? (W - 10.5f) : 10.5f;
            float knobR = isOn ? 6.8f : 5.4f;
            const float cy = H * 0.5f;
            const float trackR = 8.5f;

            for (int y = 0; y < H; y++)
            {
                float fy = y + 0.5f;
                for (int x = 0; x < W; x++)
                {
                    float fx = x + 0.5f;
                    float clampedX = Mathf.Clamp(fx, 10f, W - 10f);
                    float dTrack = Mathf.Sqrt((fx - clampedX) * (fx - clampedX) + (fy - cy) * (fy - cy)) - trackR;
                    float cov = Mathf.Clamp01(0.65f - dTrack);
                    float bMix = Mathf.Clamp01(1.25f - Mathf.Abs(dTrack + 0.5f));
                    Color c = Color.Lerp(trackFill, trackBrd, bMix);

                    float dKnob = Mathf.Sqrt((fx - knobCx) * (fx - knobCx) + (fy - cy) * (fy - cy)) - knobR;
                    float kAlpha = Mathf.Clamp01(0.65f - dKnob);
                    c = Color.Lerp(c, knobCol, kAlpha);
                    c.a = cov;
                    cols[y * W + x] = c;
                }
            }
            tex.SetPixels32(cols);
            tex.Apply(false);
            return tex;
        }

        private bool DrawM3SwitchRow(bool value, string label)
        {
            Rect row = GUILayoutUtility.GetRect(10f, 26f, GUILayout.ExpandWidth(true));
            if (GUI.Button(row, GUIContent.none, GUIStyle.none))
                value = !value;

            Rect swRect = new Rect(row.x + 2f, row.y + 3f, 38f, 20f);
            GUI.DrawTexture(swRect, value ? _texSwitchOn : _texSwitchOff, ScaleMode.StretchToFill, true);

            string txt = value ? $"<color=#E6F0FF>{label}</color>" : $"<color=#B8C2D2>{label}</color>";
            GUI.Label(new Rect(row.x + 48f, row.y + 1f, Mathf.Max(60f, row.width - 50f), 24f), txt, _stBody);
            return value;
        }

        private static Texture2D MakeSolidTex(Color c)
        {
            var t = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            t.SetPixel(0, 0, c);
            t.Apply();
            return t;
        }

        private void OnGUI()
        {
            EnsureStyles();

            if (IsLauncherOpen)
            {
                DrawLauncherScreen();
                return;
            }

            if (coordinator != null && coordinator.IsTrainingSession)
                DrawTrainingUxPanel();
            else if (coordinator != null)
            {
                DrawSpectateStatusStrip();
                if (coordinator.enableStrategyBroadcast)
                    DrawGemmaBroadcastCard();
            }

            DrawTopEmergencyStopButton();
            DrawBackendStartupBanner();
            if ((multiManager != null && multiManager.isEmergencyHalted) || (coordinator != null && coordinator.IsEmergencyHalted))
                DrawEmergencyHaltedModal();
        }

        private void DrawBackendStartupBanner()
        {
            if (coordinator == null || IsLauncherOpen)
                return;
            if ((multiManager != null && multiManager.isEmergencyHalted) || coordinator.IsEmergencyHalted)
                return;
            if (_trainingCompleted)
                return;

            int reg = coordinator.RegisteredBridgeCount;
            int conn = coordinator.ConnectedBridgeCount;
            if (reg > 0 && conn >= reg)
                return;

            float rightInset = (!coordinator.IsTrainingSession && neuralHud != null) ? neuralHud.ActiveRightPanelWidthPx : 0f;
            float availW = Mathf.Max(420f, Screen.width - rightInset);
            float bw = 468f;
            bool failed = coordinator.HasCompletedProcess && (coordinator.LastExitCode != 0 || !string.IsNullOrEmpty(coordinator.latestBackendError));
            float bh = failed ? 104f : 68f;
            float bx = Mathf.Clamp(380f + (availW - 380f - bw) * 0.5f, 380f, Mathf.Max(380f, availW - bw - 16f));
            float by = 78f;

            GUI.Box(new Rect(bx, by, bw, bh), GUIContent.none, failed ? _stCrimsonModal : _stCard);

            if (failed)
            {
                string errMsg = !string.IsNullOrEmpty(coordinator.latestBackendError)
                    ? coordinator.latestBackendError
                    : $"ExitCode={coordinator.LastExitCode} ({coordinator.latestBackendLog})";
                GUI.Label(new Rect(bx + 14f, by + 8f, bw - 130f, 22f),
                    "<color=#FF7B72><b>⚠ Python頭脳バックエンドの起動に失敗しました</b></color>", _stHeader);
                if (GUI.Button(new Rect(bx + bw - 112f, by + 8f, 98f, 26f), "メニューへ", _stBtnSub))
                {
                    ReturnToLauncherMenu();
                    return;
                }
                GUI.Label(new Rect(bx + 14f, by + 36f, bw - 28f, 60f),
                    $"<color=#FCA5A5>{errMsg}</color>", _stSmall);
                return;
            }

            float waitSec = Mathf.Max(0f, Time.realtimeSinceStartup - _sessionStartRealtime);
            bool loadingWeights = coordinator.IsTrainingSession
                ? coordinator.resumeFromCheckpointInPbt && !string.IsNullOrEmpty(coordinator.loadCheckpointRelativePath)
                : !string.IsNullOrEmpty(coordinator.loadCheckpointRelativePath);
            string ckptLabel = loadingWeights
                ? Path.GetFileName(coordinator.loadCheckpointRelativePath)
                : "新規初期重み (Random Init)";

            GUI.Label(new Rect(bx + 14f, by + 8f, bw - 28f, 22f),
                $"<color=#9DD2FF><b>⟳ AI頭脳・CUDA初期化 ＆ 重み読込中... ({waitSec:F1}s)</b></color>", _stHeader);
            GUI.Label(new Rect(bx + 14f, by + 29f, bw - 28f, 18f),
                $"対象重み: <b>{ckptLabel}</b>  |  TCP接続待機: <b>{conn}/{Mathf.Max(1, reg)}</b>", _stSmall);

            // Smooth indeterminate M3 progress bar
            Rect track = new Rect(bx + 14f, by + 51f, bw - 28f, 6f);
            GUI.DrawTexture(track, _texBarBg);
            float pulseW = track.width * 0.34f;
            float slide = Mathf.PingPong(Time.unscaledTime * 1.35f, 1f) * (track.width - pulseW);
            GUI.DrawTexture(new Rect(track.x + slide, track.y, pulseW, track.height), _texAccentBlue);
        }

        private static Texture2D BuildCyberStopSwitchSprite(Color32 borderCol, Color32 sideTickCol, Color32 plateBg)
        {
            // Exact 16:9 horizontal landscape module (96 x 54 px)
            const int W = 96;
            const int H = 54;
            var pixels = new Color32[W * H];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = new Color32(0, 0, 0, 0);

            void SetTopPx(int x, int yTop, Color32 c)
            {
                if ((uint)x >= W || (uint)yTop >= H) return;
                pixels[(H - 1 - yTop) * W + x] = c;
            }

            void BlendOverPlate(int x, int yTop, Color32 fg, float t)
            {
                if ((uint)x >= W || (uint)yTop >= H || t <= 0f) return;
                t = Mathf.Clamp01(t);
                Color32 dst = pixels[(H - 1 - yTop) * W + x];
                byte r = (byte)Mathf.RoundToInt(Mathf.Lerp(dst.r, fg.r, t));
                byte g = (byte)Mathf.RoundToInt(Mathf.Lerp(dst.g, fg.g, t));
                byte b = (byte)Mathf.RoundToInt(Mathf.Lerp(dst.b, fg.b, t));
                byte a = (byte)Mathf.Max(dst.a, Mathf.RoundToInt(Mathf.Lerp(dst.a, fg.a, t)));
                pixels[(H - 1 - yTop) * W + x] = new Color32(r, g, b, a);
            }

            // Rounded obsidian housing box [x = 7..88, yTop = 2..51], corner radius = 6
            const float xMin = 7f, xMax = 88f, yMin = 2f, yMax = 51f, rad = 6f;
            for (int y = 2; y <= 51; y++)
            {
                for (int x = 7; x <= 88; x++)
                {
                    float qx = Mathf.Max(0f, Mathf.Max((xMin + rad) - x, x - (xMax - rad)));
                    float qy = Mathf.Max(0f, Mathf.Max((yMin + rad) - y, y - (yMax - rad)));
                    float dCorner = Mathf.Sqrt(qx * qx + qy * qy);
                    if (dCorner <= rad + 0.65f)
                    {
                        float cov = Mathf.Clamp01((rad + 0.65f) - dCorner);
                        float edgeDist = (qx > 0f || qy > 0f)
                            ? (rad - dCorner)
                            : Mathf.Min(Mathf.Min(x - xMin, xMax - x), Mathf.Min(y - yMin, yMax - y));
                        float borderMix = Mathf.Clamp01(1.85f - edgeDist);
                        Color32 c = new Color32(
                            (byte)Mathf.RoundToInt(Mathf.Lerp(plateBg.r, borderCol.r, borderMix)),
                            (byte)Mathf.RoundToInt(Mathf.Lerp(plateBg.g, borderCol.g, borderMix)),
                            (byte)Mathf.RoundToInt(Mathf.Lerp(plateBg.b, borderCol.b, borderMix)),
                            (byte)Mathf.RoundToInt(Mathf.Lerp(plateBg.a, borderCol.a, borderMix) * cov));
                        SetTopPx(x, y, c);
                    }
                }
            }

            // Left & right horizontal crimson bus ticks (`—[ ⊘ STOP ]—`) centered at mid-height
            for (int y = 26; y <= 27; y++)
            {
                for (int x = 0; x <= 6; x++) SetTopPx(x, y, sideTickCol);
                for (int x = 89; x < W; x++) SetTopPx(x, y, sideTickCol);
            }

            // 4 subtle metallic corner rivets + 2 bay-divider pin dots inside the 16:9 plate
            Color32 pinDot = new Color32(142, 156, 172, 185);
            Color32 bayDot = new Color32(142, 156, 172, 135);
            BlendOverPlate(14, 9, pinDot, 0.72f);
            BlendOverPlate(81, 9, pinDot, 0.72f);
            BlendOverPlate(14, 44, pinDot, 0.72f);
            BlendOverPlate(81, 44, pinDot, 0.72f);
            BlendOverPlate(43, 12, bayDot, 0.52f);
            BlendOverPlate(43, 41, bayDot, 0.52f);

            // Left bay: Anti-aliased Prohibition circle + 45-deg diagonal slash (`⊘`) at (cx=25.5, cyTop=26.5, r=10.5)
            const float cx = 25.5f, cy = 26.5f, r0 = 10.5f;
            for (int y = 14; y <= 39; y++)
            {
                for (int x = 13; x <= 38; x++)
                {
                    float dx = x - cx;
                    float dy = y - cy;
                    float dist = Mathf.Sqrt(dx * dx + dy * dy);
                    float ringAlpha = Mathf.Clamp01(1.38f - Mathf.Abs(dist - r0));
                    float slashPerp = Mathf.Abs(dx - dy) * 0.70710678f;
                    float slashAlpha = dist <= (r0 + 0.15f) ? Mathf.Clamp01(1.32f - slashPerp) : 0f;
                    float a = Mathf.Max(ringAlpha, slashAlpha);
                    if (a > 0.02f)
                        BlendOverPlate(x, y, borderCol, a);
                }
            }

            var tex = new Texture2D(W, H, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            tex.SetPixels32(pixels);
            tex.Apply(false);
            return tex;
        }

        private void DrawTopEmergencyStopButton()
        {
            Rect r = GetEmergencyStopButtonRect();
            bool halted = (multiManager != null && multiManager.isEmergencyHalted)
                       || (coordinator != null && coordinator.IsEmergencyHalted);
            Vector2 mPos = Event.current != null ? Event.current.mousePosition : new Vector2(-999f, -999f);
            bool hovered = r.Contains(mPos);

            Texture2D switchTex = halted ? _texStopSwitchHalted : (hovered ? _texStopSwitchHover : _texStopSwitchIdle);
            GUI.DrawTexture(r, switchTex, ScaleMode.StretchToFill, true);

            // Right bay inside 16:9 landscape housing: crisp bold crimson STOP / HALT label
            GUI.Label(new Rect(r.x + 41f, r.y + 15f, 44f, 24f),
                halted ? "HALT" : "STOP",
                (hovered || halted) ? _stStopLabelHover : _stStopLabelIdle);

            if (GUI.Button(r, GUIContent.none, GUIStyle.none))
            {
                ExecuteEmergencyStop();
            }
        }

        private void DrawEmergencyHaltedModal()
        {
            Rect mr = GetEmergencyModalRect();
            GUI.Box(mr, GUIContent.none, _stCrimsonModal);

            // Crisp crimson alert border
            GUI.DrawTexture(new Rect(mr.x, mr.y, mr.width, 2f), _texAccentCrimson);
            GUI.DrawTexture(new Rect(mr.x, mr.y + mr.height - 2f, mr.width, 2f), _texAccentCrimson);
            GUI.DrawTexture(new Rect(mr.x, mr.y, 2f, mr.height), _texAccentCrimson);
            GUI.DrawTexture(new Rect(mr.x + mr.width - 2f, mr.y, 2f, mr.height), _texAccentCrimson);

            GUI.Label(new Rect(mr.x + 16f, mr.y + 12f, mr.width - 32f, 26f),
                "<color=#FF6B6E><b>⊘ EMERGENCY STOP — 緊急安全停止が作動しました</b></color>", _stHeader);

            string phase = coordinator != null ? coordinator.SafeShutdownPhase : "CLEAN_EXIT";
            bool releasing = phase == "RELEASING_MEMORY" || (coordinator != null && coordinator.IsBackendRunning);
            string memStatusLine = releasing
                ? "<color=#FFD27F>協調解放中 (未検証重み保存スキップ → 勾配・Opt・Replay・VRAM・SeqLock解放中...)</color>"
                : (phase == "FORCE_KILLED"
                    ? "<color=#FFAA66>✓ 未検証重み保存スキップ ＆ タイムアウト保護により強制終了済</color>"
                    : "<color=#66FFAA>✓ 未検証重み保存スキップ ＆ 勾配・Opt・Replay・VRAM・SeqLock 解放完了</color>");

            int openSockets = coordinator != null ? coordinator.ConnectedBridgeCount : 0;
            string statusText =
                "TCP運動出力ソケットと3D物理を 0ms で即時凍結し、PCに優しい段階的メモリ解放を実行しました。\n\n" +
                $"・Python AI / GPU メモリ解放 : <b>{memStatusLine}</b>\n" +
                $"・TCP 運動・知覚ソケット接続 : <b><color=#66FFAA>全切断済 ({openSockets} Active)</color></b>\n" +
                $"・3D アリーナ物理・アクチュエータ : <b><color=#66FFAA>凍結停止中 (TimeScale = {Time.timeScale:F1})</color></b>";
            GUI.Label(new Rect(mr.x + 16f, mr.y + 42f, mr.width - 32f, 118f), statusText, _stBody);

            float btnY = mr.y + mr.height - 48f;
            float halfW = (mr.width - 42f) * 0.5f;
            if (GUI.Button(new Rect(mr.x + 16f, btnY, halfW, 34f), "■ アプリを完全終了 (Quit)", _stBtnDanger))
            {
                QuitGame();
                return;
            }
            if (GUI.Button(new Rect(mr.x + 26f + halfW, btnY, halfW, 34f), "↺ 安全確認済・メニューへ復帰", _stBtnSub))
            {
                ReturnToLauncherMenu();
            }
        }

        private void DrawLauncherScreen()
        {
            GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height), _texOverlay);

            float panelW = Mathf.Min(968f, Screen.width - 28f);
            float panelH = Mathf.Min(688f, Screen.height - 16f);
            float x0 = (Screen.width - panelW) * 0.5f;
            float y0 = (Screen.height - panelH) * 0.5f;

            GUI.Box(new Rect(x0, y0, panelW, panelH), GUIContent.none, _stPanel);

            // Top App Bar (Material 3 Dark Header + Rounded Tonal Action Pills)
            GUI.Label(new Rect(x0 + 22f, y0 + 13f, panelW - 350f, 28f),
                "<b>InkBlitz 3D — 汎用脳型アーキテクチャ v2.3.1 実験環境</b>", _stTitle);
            string infoBtnLabel = _showLegalInfoPage ? "← メニューへ戻る" : "規約・ライセンス・開発情報";
            if (GUI.Button(new Rect(x0 + panelW - 322f, y0 + 12f, 196f, 30f), infoBtnLabel, _showLegalInfoPage ? _stBtnChipSel : _stBtnSub))
            {
                _showLegalInfoPage = !_showLegalInfoPage;
                return;
            }
            if (GUI.Button(new Rect(x0 + panelW - 118f, y0 + 12f, 98f, 30f), "終了 (Quit)", _stBtnSub))
            {
                QuitGame();
                return;
            }

            if (_showLegalInfoPage)
            {
                DrawLegalAndCreditsPage(x0, y0, panelW, panelH);
                return;
            }
            GUI.Label(new Rect(x0 + 22f, y0 + 44f, panelW - 44f, 20f),
                "「並列自己対戦 PBT 学習」または「学習済み重みによる自動継続・純粋推論観戦」を選択してください。", _stSmall);

            float cardW = (panelW - 56f) * 0.5f;
            float cardH = panelH - 102f;
            float cardY = y0 + 68f;
            const float ctaBlockH = 68f;

            // =========================================================================
            // Left Card: Training Mode (Material 3 Dark Card + Comfortable Pill Controls)
            // =========================================================================
            float leftX = x0 + 20f;
            GUI.Box(new Rect(leftX, cardY, cardW, cardH), GUIContent.none, _stCard);

            GUILayout.BeginArea(new Rect(leftX + 14f, cardY + 12f, cardW - 28f, cardH - ctaBlockH - 16f));
            _leftCardScroll = GUILayout.BeginScrollView(_leftCardScroll, GUIStyle.none, GUI.skin.verticalScrollbar);

            GUILayout.Label("<color=#FFB685>学習モード (Phase 4 自己対戦 PBT)</color>", _stHeader);
            GUILayout.Label("4アリーナ×8脳を並列実行し、DAgger→自律自己対戦で重みとEloを育成して自動保存します。", _stSmall);
            GUILayout.Space(6f);

            GUILayout.BeginHorizontal();
            GUILayout.Label("ステージ(M):", _stBody, GUILayout.Width(92f), GUILayout.Height(29f));
            if (GUILayout.Button(uiStageId == 1 ? "● Stage 1: 標準シタデル" : "Stage 1: 標準シタデル", uiStageId == 1 ? _stBtnChipSel : _stBtnSub, GUILayout.Height(29f)))
                ApplyStageSelection(1);
            if (GUILayout.Button(uiStageId == 2 ? "● Stage 2: 遮蔽フランク" : "Stage 2: 遮蔽フランク", uiStageId == 2 ? _stBtnChipSel : _stBtnSub, GUILayout.Height(29f)))
                ApplyStageSelection(2);
            GUILayout.EndHorizontal();
            GUILayout.Space(5f);

            GUILayout.BeginHorizontal();
            GUILayout.Label("世代数 (Generations):", _stBody, GUILayout.Width(142f), GUILayout.Height(28f));
            string nextGenBuf = GUILayout.TextField(_genInputBuf, 4, _stTextField, GUILayout.Width(54f), GUILayout.Height(28f));
            if (nextGenBuf != _genInputBuf)
            {
                _genInputBuf = nextGenBuf;
                if (int.TryParse(nextGenBuf, out int parsedGen))
                    uiGenerations = Mathf.Clamp(parsedGen, 1, 500);
            }
            if (GUILayout.Button("-5", _stBtnSub, GUILayout.Width(40f), GUILayout.Height(28f))) SetGenerations(uiGenerations - 5);
            if (GUILayout.Button("-1", _stBtnSub, GUILayout.Width(38f), GUILayout.Height(28f))) SetGenerations(uiGenerations - 1);
            if (GUILayout.Button("+1", _stBtnSub, GUILayout.Width(38f), GUILayout.Height(28f))) SetGenerations(uiGenerations + 1);
            if (GUILayout.Button("+5", _stBtnSub, GUILayout.Width(40f), GUILayout.Height(28f))) SetGenerations(uiGenerations + 5);
            GUILayout.EndHorizontal();

            GUILayout.Space(5f);
            GUILayout.BeginHorizontal();
            int sliderCur = Mathf.Clamp(uiGenerations, 1, 100);
            GUILayout.BeginVertical(GUILayout.ExpandWidth(true));
            GUILayout.Space(7f);
            int sliderNext = Mathf.RoundToInt(GUILayout.HorizontalSlider(sliderCur, 1f, 100f, GUILayout.ExpandWidth(true)));
            GUILayout.EndVertical();
            if (sliderNext != sliderCur)
                SetGenerations(sliderNext);
            int[] genPresets = { 1, 4, 8, 16, 32, 50 };
            for (int gp = 0; gp < genPresets.Length; gp++)
            {
                int gv = genPresets[gp];
                GUIStyle gs = (uiGenerations == gv) ? _stBtnChipSel : _stBtnSub;
                if (GUILayout.Button(gv.ToString(), gs, GUILayout.Width(gv >= 10 ? 36f : 32f), GUILayout.Height(27f)))
                    SetGenerations(gv);
            }
            GUILayout.EndHorizontal();

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Ep/世代 (試合数/世代):", _stBody, GUILayout.Width(142f), GUILayout.Height(28f));
            string nextEpsBuf = GUILayout.TextField(_epsInputBuf, 2, _stTextField, GUILayout.Width(46f), GUILayout.Height(28f));
            if (nextEpsBuf != _epsInputBuf)
            {
                _epsInputBuf = nextEpsBuf;
                if (int.TryParse(nextEpsBuf, out int parsedEps))
                    uiEpisodesPerGen = Mathf.Clamp(parsedEps, 1, 20);
            }
            if (GUILayout.Button("−", _stBtnSub, GUILayout.Width(34f), GUILayout.Height(28f))) SetEpisodesPerGen(uiEpisodesPerGen - 1);
            if (GUILayout.Button("＋", _stBtnSub, GUILayout.Width(34f), GUILayout.Height(28f))) SetEpisodesPerGen(uiEpisodesPerGen + 1);
            int[] epPresets = { 1, 2, 3, 5 };
            for (int ep = 0; ep < epPresets.Length; ep++)
            {
                int ev = epPresets[ep];
                GUIStyle es = (uiEpisodesPerGen == ev) ? _stBtnChipSel : _stBtnSub;
                if (GUILayout.Button(ev.ToString(), es, GUILayout.Width(34f), GUILayout.Height(28f)))
                    SetEpisodesPerGen(ev);
            }
            GUILayout.EndHorizontal();

            int slots = coordinator != null ? Mathf.Clamp(coordinator.parallelArenaCount, 2, 8) : 4;
            int epPerSlot = uiGenerations * uiEpisodesPerGen;
            int totalMatches = epPerSlot * slots;
            GUILayout.Space(5f);
            GUILayout.Label($"<color=#FFD27F><b>総実行数: {epPerSlot} Ep/枠</b> (全{slots}アリーナ合計 <b>{totalMatches} 試合</b> | 毎世代自動保存)</color>", _stSmall);

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUILayout.Label($"物理倍速 (TimeScale): <b>{uiTimeScale:F1}x</b>", _stBody, GUILayout.Width(164f), GUILayout.Height(30f));
            float[] tsPresets = { 2.0f, 4.0f, 6.0f };
            for (int ti = 0; ti < tsPresets.Length; ti++)
            {
                float tv = tsPresets[ti];
                bool selTs = Mathf.Abs(uiTimeScale - tv) < 0.05f;
                if (GUILayout.Button($"{tv:F1}x", selTs ? _stBtnChipSel : _stBtnSub, GUILayout.Width(58f), GUILayout.Height(30f)))
                    uiTimeScale = tv;
            }
            GUILayout.EndHorizontal();

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUILayout.Label("方策カリキュラム:", _stBody, GUILayout.Width(116f), GUILayout.Height(30f));
            if (GUILayout.Button(!uiPureNeuralOnly ? "● DAgger→自律" : "DAgger→自律", !uiPureNeuralOnly ? _stBtnChipSel : _stBtnSub, GUILayout.Height(30f)))
                uiPureNeuralOnly = false;
            if (GUILayout.Button(uiPureNeuralOnly ? "● 完全自律(Pure)" : "完全自律(Pure)", uiPureNeuralOnly ? _stBtnChipSel : _stBtnSub, GUILayout.Height(30f)))
                uiPureNeuralOnly = true;
            GUILayout.EndHorizontal();

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUILayout.Label("睡眠学習(S1-S4):", _stBody, GUILayout.Width(116f), GUILayout.Height(30f));
            if (GUILayout.Button(uiSleepMode == 1 ? "● ON(標準6ep)" : "ON(標準6ep)", uiSleepMode == 1 ? _stBtnChipSel : _stBtnSub, GUILayout.Height(30f)))
                uiSleepMode = 1;
            if (GUILayout.Button(uiSleepMode == 2 ? "● 深睡眠(10ep)" : "深睡眠(10ep)", uiSleepMode == 2 ? _stBtnChipSel : _stBtnSub, GUILayout.Height(30f)))
                uiSleepMode = 2;
            if (GUILayout.Button(uiSleepMode == 0 ? "● OFF" : "OFF", uiSleepMode == 0 ? _stBtnChipSel : _stBtnSub, GUILayout.Width(64f), GUILayout.Height(30f)))
                uiSleepMode = 0;
            GUILayout.EndHorizontal();

            GUILayout.Space(8f);
            string selName = _checkpoints.Count > 0 ? _checkpoints[_selectedCkptIdx].fileName : "(既存重みなし・新規初期化)";
            uiResumeFromCheckpoint = DrawM3SwitchRow(uiResumeFromCheckpoint, $"選択重みから継続育成: {selName}");
            uiRepArenaOnly = DrawM3SwitchRow(uiRepArenaOnly, "学習中は代表アリーナ0のみ描画 (RTX3060最適化)");
            bool nextSupTrain = DrawM3SwitchRow(uiEnableSupportBots, "ウイングマン型サポートBot追加 (2v2戦術 / OFFで純粋1v1)");
            if (nextSupTrain != uiEnableSupportBots)
                ApplyLauncherSupportBotPreview(nextSupTrain);

            GUILayout.EndScrollView();
            GUILayout.EndArea();

            // Pinned Bottom CTA Area (Left Card)
            GUILayout.BeginArea(new Rect(leftX + 14f, cardY + cardH - ctaBlockH, cardW - 28f, ctaBlockH - 8f));
            GUILayout.Label("<color=#94B8DC>保存先: checkpoints/user_runs/ (pbt_latest / best / session)</color>", _stSmall);
            GUILayout.Space(4f);
            if (GUILayout.Button("▶ 学習モードを開始 (T)", _stBtnPrimary, GUILayout.Height(40f)))
                StartTrainingSession();
            GUILayout.EndArea();

            // =========================================================================
            // Right Card: Spectate Mode + Gemma Coach (Material 3 Dark Card)
            // =========================================================================
            float rightX = leftX + cardW + 16f;
            GUI.Box(new Rect(rightX, cardY, cardW, cardH), GUIContent.none, _stCard);

            GUILayout.BeginArea(new Rect(rightX + 14f, cardY + 12f, cardW - 28f, cardH - ctaBlockH - 16f));
            _rightCardScroll = GUILayout.BeginScrollView(_rightCardScroll, GUIStyle.none, GUI.skin.verticalScrollbar);

            string specCardHdr = uiSpectateContinualLearn
                ? "<color=#80E896>推論・観戦モード (自動継続 / 実戦継続学習ON)</color>"
                : "<color=#9DD2FF>推論・観戦モード (自動継続 / 純粋推論・学習OFF)</color>";
            string specCardSub = uiSpectateContinualLearn
                ? "観戦しながらオンライン学習＋試合間ミニ睡眠(S1-S4)＋重み自動保存を行います。"
                : "試合終了後も次の試合へ自動で連続進行します（重み更新なし・等倍速既定）。";
            GUILayout.Label(specCardHdr, _stHeader);
            GUILayout.Label(specCardSub, _stSmall);
            GUILayout.Space(7f);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(uiSelfPlaySpectate ? "● AI vs AI 自己対戦" : "AI vs AI 自己対戦", uiSelfPlaySpectate ? _stBtnChipSel : _stBtnSub, GUILayout.Height(30f)))
                uiSelfPlaySpectate = true;
            if (GUILayout.Button(!uiSelfPlaySpectate ? "● AI vs 教師Bot" : "AI vs 教師Bot", !uiSelfPlaySpectate ? _stBtnChipSel : _stBtnSub, GUILayout.Height(30f)))
            {
                uiSelfPlaySpectate = false;
                uiSpectate2v2BrainRadio = false;
            }
            GUILayout.EndHorizontal();

            GUILayout.Space(4f);
            bool next2v2Radio = DrawM3SwitchRow(uiSpectate2v2BrainRadio, "📡 2v2 双脳狭帯域無線・創発言語 (Vygotsky 10-bit Radio)");
            if (next2v2Radio != uiSpectate2v2BrainRadio)
            {
                uiSpectate2v2BrainRadio = next2v2Radio;
                if (uiSpectate2v2BrainRadio)
                    uiSelfPlaySpectate = true;
            }

            GUILayout.Space(5f);
            GUILayout.BeginHorizontal();
            GUILayout.Label("ステージ(M):", _stBody, GUILayout.Width(92f), GUILayout.Height(29f));
            if (GUILayout.Button(uiStageId == 1 ? "● Stage 1: 標準シタデル" : "Stage 1: 標準シタデル", uiStageId == 1 ? _stBtnChipSel : _stBtnSub, GUILayout.Height(29f)))
                ApplyStageSelection(1);
            if (GUILayout.Button(uiStageId == 2 ? "● Stage 2: 遮蔽フランク" : "Stage 2: 遮蔽フランク", uiStageId == 2 ? _stBtnChipSel : _stBtnSub, GUILayout.Height(29f)))
                ApplyStageSelection(2);
            GUILayout.EndHorizontal();

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUILayout.Label($"使用チェックポイント ({_checkpoints.Count}件):", _stBody, GUILayout.Height(28f));
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("更新 (R)", _stBtnSub, GUILayout.Width(78f), GUILayout.Height(28f)))
                RefreshCheckpointList();
            GUILayout.EndHorizontal();

            _ckptScroll = GUILayout.BeginScrollView(_ckptScroll, _stInsetBox, GUILayout.Height(72f));
            if (_checkpoints.Count == 0)
            {
                GUILayout.Label("checkpoints/user_runs に .pt がありません。\n（未選択のまま初期重みで推論観戦できます）", _stSmall);
            }
            else
            {
                for (int i = 0; i < _checkpoints.Count; i++)
                {
                    bool sel = (i == _selectedCkptIdx);
                    string line = (sel ? "● " : "   ") + _checkpoints[i].label;
                    if (GUILayout.Button(line, sel ? _stListRowSel : _stListRowIdle, GUILayout.Height(26f)))
                        _selectedCkptIdx = i;
                }
            }
            GUILayout.EndScrollView();

            GUILayout.Space(5f);
            GUILayout.BeginHorizontal();
            GUILayout.Label($"観戦速度: <b>{spectateTimeScale:F1}x</b>", _stBody, GUILayout.Width(104f), GUILayout.Height(29f));
            bool sp1 = Mathf.Abs(spectateTimeScale - 1.0f) < 0.05f;
            bool sp2 = Mathf.Abs(spectateTimeScale - 2.0f) < 0.05f;
            if (GUILayout.Button("1.0x", sp1 ? _stBtnChipSel : _stBtnSub, GUILayout.Width(58f), GUILayout.Height(29f))) spectateTimeScale = 1.0f;
            if (GUILayout.Button("2.0x", sp2 ? _stBtnChipSel : _stBtnSub, GUILayout.Width(58f), GUILayout.Height(29f))) spectateTimeScale = 2.0f;
            GUILayout.EndHorizontal();

            GUILayout.Space(4f);
            GUILayout.BeginHorizontal();
            GUILayout.Label("観戦中の継続学習:", _stBody, GUILayout.Width(120f), GUILayout.Height(29f));
            if (GUILayout.Button(!uiSpectateContinualLearn ? "● OFF (純粋推論)" : "OFF (純粋推論)", !uiSpectateContinualLearn ? _stBtnChipSel : _stBtnSub, GUILayout.Height(29f)))
                uiSpectateContinualLearn = false;
            if (GUILayout.Button(uiSpectateContinualLearn ? "● ON (実戦育成)" : "ON (実戦育成)", uiSpectateContinualLearn ? _stBtnChipSel : _stBtnSub, GUILayout.Height(29f)))
                uiSpectateContinualLearn = true;
            GUILayout.EndHorizontal();

            GUILayout.Space(4f);
            uiEnableGemmaBroadcast = DrawM3SwitchRow(uiEnableGemmaBroadcast, "Gemma 4 実況・4節目戦略DB・双方向コーチング連携");
            if (uiEnableGemmaBroadcast)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("Ollama演算先:", _stSmall, GUILayout.Width(96f), GUILayout.Height(28f));
                if (GUILayout.Button(uiOllamaComputeMode == 0 ? "● 分割(GPU+CPU)" : "分割(GPU+CPU)", uiOllamaComputeMode == 0 ? _stBtnChipSel : _stBtnSub, GUILayout.Height(28f)))
                    uiOllamaComputeMode = 0;
                if (GUILayout.Button(uiOllamaComputeMode == 1 ? "● 完全CPU" : "完全CPU", uiOllamaComputeMode == 1 ? _stBtnChipSel : _stBtnSub, GUILayout.Width(84f), GUILayout.Height(28f)))
                    uiOllamaComputeMode = 1;
                if (GUILayout.Button(uiOllamaComputeMode == 2 ? "● GPU" : "GPU", uiOllamaComputeMode == 2 ? _stBtnChipSel : _stBtnSub, GUILayout.Width(62f), GUILayout.Height(28f)))
                    uiOllamaComputeMode = 2;
                GUILayout.EndHorizontal();
            }
            if (voiceCoach != null)
            {
                GUILayout.Space(2f);
                GUILayout.BeginHorizontal();
                string ttsStateTag = voiceCoach.IsTtsServerOnline
                    ? $"<color=#7EE787>{voiceCoach.GetSelectedVoiceShortLabel()}</color>"
                    : "<color=#99AABB>Aivis未検出(再検索)</color>";
                GUILayout.Label("音声実況(Aivis):", _stSmall, GUILayout.Width(96f), GUILayout.Height(28f));
                bool vOn = voiceCoach.enableVoiceNarration;
                if (GUILayout.Button(vOn ? "● ON" : "OFF", vOn ? _stBtnChipSel : _stBtnSub, GUILayout.Width(58f), GUILayout.Height(28f)))
                    voiceCoach.SetVoiceNarrationEnabled(!vOn);
                if (GUILayout.Button("◀", _stBtnSub, GUILayout.Width(32f), GUILayout.Height(28f)))
                    voiceCoach.CycleVoiceStyle(-1);
                if (GUILayout.Button(ttsStateTag, _stBtnSub, GUILayout.ExpandWidth(true), GUILayout.Height(28f)))
                    voiceCoach.CycleVoiceStyle(1);
                if (GUILayout.Button("▶", _stBtnSub, GUILayout.Width(32f), GUILayout.Height(28f)))
                    voiceCoach.CycleVoiceStyle(1);
                GUILayout.EndHorizontal();
            }
            bool nextSupSpec = DrawM3SwitchRow(uiEnableSupportBots, "ウイングマン型サポートBot追加 (2v2 / OFFで1v1)");
            if (nextSupSpec != uiEnableSupportBots)
                ApplyLauncherSupportBotPreview(nextSupSpec);

            // 7-Preset Matte Ink Color Palette Selector
            GUILayout.Space(4f);
            InkColorPalettePreset curPal = InkBlitzInkPalettes.Get(uiColorPaletteIndex);
            GUILayout.BeginHorizontal();
            GUILayout.Label(
                $"インク配色(P): <color=#{curPal.hexA}>■</color><color=#{curPal.hexB}>■</color> <b>{uiColorPaletteIndex + 1}.{curPal.label}</b> <color=#99AABB>(#{curPal.hexA}/#{curPal.hexB})</color>",
                _stSmall, GUILayout.Height(26f));
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("◀", _stBtnSub, GUILayout.Width(34f), GUILayout.Height(26f)))
                ApplyPaletteSelection(uiColorPaletteIndex - 1);
            if (GUILayout.Button("▶", _stBtnSub, GUILayout.Width(34f), GUILayout.Height(26f)))
                ApplyPaletteSelection(uiColorPaletteIndex + 1);
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            for (int p = 0; p < InkBlitzInkPalettes.Count; p++)
            {
                InkColorPalettePreset pr = InkBlitzInkPalettes.Get(p);
                bool isSelP = (p == uiColorPaletteIndex);
                string prefix = isSelP ? $"<b>{p + 1}</b>" : $"{p + 1}";
                if (GUILayout.Button($"{prefix}<color=#{pr.hexA}>●</color><color=#{pr.hexB}>●</color>", isSelP ? _stBtnChipSel : _stBtnSub, GUILayout.Height(28f)))
                    ApplyPaletteSelection(p);
            }
            GUILayout.EndHorizontal();

            GUILayout.EndScrollView();
            GUILayout.EndArea();

            // Pinned Bottom CTA Area (Right Card)
            GUILayout.BeginArea(new Rect(rightX + 14f, cardY + cardH - ctaBlockH, cardW - 28f, ctaBlockH - 8f));
            string learnSuffix = uiSpectateContinualLearn ? "継続学習ON (spectate_continual_latest.pt 保存)" : "純粋推論 (重み固定)";
            string modeNote = uiSelfPlaySpectate
                ? $"AI vs AI 自己対戦 — {learnSuffix}"
                : $"AI vs 教師Bot — {learnSuffix}";
            GUILayout.Label($"<color=#94B8DC>{modeNote}</color>", _stSmall);
            GUILayout.Space(4f);
            if (GUILayout.Button("▶ 推論・観戦モードを開始 (S)", _stBtnSpectate, GUILayout.Height(40f)))
                StartSpectateSession();
            GUILayout.EndArea();

            GUI.Label(new Rect(x0 + 22f, y0 + panelH - 26f, panelW - 44f, 20f),
                "ショートカット: [T]学習 [S]観戦 [M]ステージ [P]配色 [G]実況 [F1]戦況 [F3]思考UX [F4]主観 [F5]深度 [F6]世界モデル [F7]脳回路図 [F12]緊急停止", _stSmall);
        }

        private void DrawLegalAndCreditsPage(float x0, float y0, float panelW, float panelH)
        {
            GUI.Label(new Rect(x0 + 18f, y0 + 42f, panelW - 36f, 22f),
                "利用規約・オープンソースライセンス・独自実装クレジット・開発元情報（試験用表示）", _stSmall);

            Rect cardRect = new Rect(x0 + 16f, y0 + 70f, panelW - 32f, panelH - 132f);
            GUI.Box(cardRect, GUIContent.none, _stCard);

            GUILayout.BeginArea(new Rect(cardRect.x + 16f, cardRect.y + 12f, cardRect.width - 32f, cardRect.height - 24f));
            _legalScroll = GUILayout.BeginScrollView(_legalScroll);

            GUILayout.Label("<color=#FFD27F><b>1. 頭脳部・システム開発元 (Developer & Brain Architecture)</b></color>", _stHeader);
            GUILayout.Space(3f);
            GUILayout.Label(
                "・3D汎用疑似脳アーキテクチャ（Universal Brain 3D v2.3.0）および本実験環境の開発元：<b>Alpha496</b>\n" +
                "・本ソフトウェアは、海馬空間認知・小脳3D弾道予測・大脳基底核ゲート制御・GWT意識統合・情報幾何学（Φ_G）および睡眠記憶固定化（S-1〜S-4）の統合動作を検証するための非公開技術試験ビルドです。",
                _stBody);

            GUILayout.Space(12f);
            GUILayout.Label("<color=#66CCFF><b>2. 独自3Dシミュレーション環境・アセット・音響実装について (Original Implementation)</b></color>", _stHeader);
            GUILayout.Space(3f);
            GUILayout.Label(
                "・本環境「InkBlitz 3D」は、動的3D空間における汎用AIの戦術創発・領域制圧・ブリッツコア搬送・運動学習を評価する目的でゼロから独自設計された技術研究シミュレータです。\n" +
                "・本環境内の3Dステージ地形・プリミティブ造形・マットインク描画処理・3D弾道およびダイブ潜伏物理演算・48kHz手続き型効果音合成・UIおよび制御プログラムはすべて独自に実装されたオリジナル資産であり、第三者の市販ゲームソフトの内部データ・モデル・音源・テクスチャ・固有商標等は一切含まれていません。",
                _stBody);

            GUILayout.Space(12f);
            GUILayout.Label("<color=#88EEAA><b>3. ライセンス (License)</b></color>", _stHeader);
            GUILayout.Space(3f);
            GUILayout.Label(
                "・本プロジェクトの疑似脳コア、C#通信ブリッジ、およびゲーム制御ソースコードは <b>GNU General Public License v3.0（GPL-3.0）</b> に準拠します。\n" +
                "・実行環境として利用している外部エンジンおよびOSSライブラリ（Unity Engine, Python, PyTorch, NumPy 等）については、各提供元の公式ライセンス条項が適用されます。",
                _stBody);

            GUILayout.Space(12f);
            GUILayout.Label("<color=#FFA888><b>4. 利用規約・免責事項 (Terms of Use & Disclaimer)</b></color>", _stHeader);
            GUILayout.Space(3f);
            GUILayout.Label(
                "・本ソフトウェアは開発途上の実験・検証用ビルドとして「現状有姿（AS-IS）」で提供され、動作の完全性・安全性・特定目的への適合性について開発元（Alpha496）はいかなる保証も負いません。\n" +
                "・第三者の権利を侵害する行為、ならびにライセンス条項（GPL-3.0）に違反する無断商用利用・不正転用を固く禁じます。",
                _stBody);

            GUILayout.EndScrollView();
            GUILayout.EndArea();

            if (GUI.Button(new Rect(x0 + panelW * 0.5f - 150f, y0 + panelH - 50f, 300f, 36f), "← セッション選択メニューへ戻る (Esc)", _stBtnSpectate))
            {
                _showLegalInfoPage = false;
            }
        }

        private void DrawTrainingUxPanel()
        {
            if (IsTrainingPanelCollapsed)
            {
                float cw = 216f;
                float cx = Screen.width - cw - 12f;
                GUI.Box(new Rect(cx, 12f, cw, 32f), GUIContent.none, _stPanel);
                string mini = _trainingCompleted
                    ? $"<color=#66FFAA><b>学習完了 G{_gen}/{_totalGens}</b></color>"
                    : $"<b>Gen {_gen}/{_totalGens}</b> Elo:{_champElo:F0}";
                GUI.Label(new Rect(cx + 8f, 17f, 134f, 22f), mini, _stSmall);
                if (GUI.Button(new Rect(cx + 144f, 16f, 64f, 22f), "展開 F2", _stBtnSub))
                    IsTrainingPanelCollapsed = false;
                return;
            }

            float panelW = 344f;
            float panelH = _trainingCompleted ? 418f : 352f;
            float x0 = Screen.width - panelW - 12f;
            float y0 = 12f;

            GUI.Box(new Rect(x0, y0, panelW, panelH), GUIContent.none, _stPanel);

            // Top bar with title and collapse / menu buttons
            string hdr = _trainingCompleted
                ? "<color=#66FFAA><b>✓ Phase 4 学習完了</b></color>"
                : (_isSleepingNow
                    ? $"<color=#D4A8FF><b>🌙 睡眠記憶固定化 S1-S4 ({_activeSleepEpochs}ep)</b></color>"
                    : "<color=#FFA855><b>Phase 4 並列学習モニター</b></color>");
            GUI.Label(new Rect(x0 + 10f, y0 + 8f, 185f, 22f), hdr, _stHeader);
            if (GUI.Button(new Rect(x0 + panelW - 142f, y0 + 7f, 62f, 21f), "折畳 F2", _stBtnSub))
                IsTrainingPanelCollapsed = true;
            if (GUI.Button(new Rect(x0 + panelW - 74f, y0 + 7f, 64f, 21f), "Menu Esc", _stBtnSub))
            {
                ReturnToLauncherMenu();
                return;
            }

            float elapsed = Mathf.Max(0f, Time.realtimeSinceStartup - _sessionStartRealtime);
            int pct = Mathf.Clamp(Mathf.RoundToInt(100f * (_gen > 0 ? (_trainingCompleted ? _totalGens : (_gen - 1f) + (_doneSlots / (float)Mathf.Max(1, _popSize))) / Mathf.Max(1, _totalGens) : 0f)), 0, 100);
            float progress01 = pct / 100f;

            // Progress bar
            float barX = x0 + 10f;
            float barY = y0 + 34f;
            float barW = panelW - 20f;
            GUI.DrawTexture(new Rect(barX, barY, barW, 10f), _texBarBg);
            GUI.DrawTexture(new Rect(barX, barY, Mathf.Max(2f, barW * progress01), 10f),
                _trainingCompleted ? _texAccentGreen : _texAccentOrange);

            int totalEp = 0;
            int totalKo = 0;
            int totalTo = 0;
            int totalWinsA = 0;
            int totalWinsB = 0;
            if (multiManager != null)
            {
                for (int i = 0; i < multiManager.arenas.Count; i++)
                {
                    var ar = multiManager.arenas[i];
                    if (ar == null) continue;
                    totalEp += ar.episodesCompleted;
                    totalKo += ar.knockoutCount;
                    totalTo += ar.timeoutCount;
                    totalWinsA += ar.teamAWins;
                    totalWinsB += ar.teamBWins;
                }
            }
            float tps = multiManager != null ? multiManager.SmoothedTicksPerSec : 0f;
            int brCnt = coordinator != null ? Mathf.Max(1, coordinator.ConnectedBridgeCount) : 1;
            string resumedNote = string.IsNullOrEmpty(_resumedFrom) ? "" : $" | 継続元:{_resumedFrom}";

            var sb = new System.Text.StringBuilder(512);
            sb.AppendLine($"世代進捗: <b>Gen {_gen:D2}/{_totalGens:D2} ({pct}%)</b>  現Gen完了: <b>{_doneSlots}/{_popSize}</b>  方策:<b>{_policy}</b>");
            sb.AppendLine($"目標規模: <b>{_totalGens}世代×{_epsPerGen}Ep</b> (計<b>{_totalGens * _epsPerGen}</b>Ep/枠){resumedNote}");
            sb.AppendLine($"時間: <b>{elapsed:F0}s</b>  平均:<b>{(_avgSecPerGen > 0.1f ? $"{_avgSecPerGen:F1}s/Gen" : "-")}</b>  速度:<b>{tps:F1}t/s</b> ({tps * brCnt:F0} br/s)");
            sb.AppendLine($"全4アリーナ累計: <b>{totalEp}試合</b> (KO:{totalKo} / T:{totalTo})  勝敗 A {totalWinsA}-{totalWinsB} B");
            float eloDelta = _champElo - 1200f;
            string deltaStr = eloDelta >= 0f ? $"+{eloDelta:F1}" : $"{eloDelta:F1}";
            sb.AppendLine($"Champion: <b>Slot#{_champSlot}</b> ({_champArch})  Elo:<b>{_champElo:F1}</b> ({deltaStr})");
            string slpInfo = (coordinator != null && coordinator.enableSleepConsolidation && coordinator.sleepConsolidationEpochs > 0)
                ? $"ON({coordinator.sleepConsolidationEpochs}ep) Loss:<b>{_sleepLoss:F4}</b>"
                : "<color=#99AABB>OFF</color>";
            sb.AppendLine($"Champ Score: <b>{_champScore:+0.0;-0.0;0.0}</b>  Sleep: {slpInfo}  League:<b>{_leagueSize}</b>");
            sb.AppendLine("<b>--- 4スロット数値 (Slot | Type | Elo | Score | W-L-D) ---</b>");

            for (int i = 0; i < _slotRows.Count; i++)
            {
                var r = _slotRows[i];
                string mark = (i == _champSlot && _gen > 0) ? "<color=#FFDD55>*</color>" : " ";
                string arch = ShortArchetype(r.archetype);
                sb.AppendLine($"{mark}S#{r.slot} {arch,-11} Elo:{r.elo,6:F1} Sc:{r.score,+6:F1} ({r.w}-{r.l}-{r.d}) {r.opp}");
            }
            if (_slotRows.Count == 0)
                sb.AppendLine("  Python GPUサーバー/ワーカー起動・初回接続待機中...");

            string histStr = _genHistory.Count > 0 ? string.Join(" → ", _genHistory) : "Gen1完了後に推移表示";
            sb.AppendLine($"Elo推移: <color=#AADDFF>{histStr}</color>");
            string saveTag = _savedCkpt ? "<color=#66FFAA>pbt_latest / best 更新済</color>" : (string.IsNullOrWhiteSpace(coordinator.pbtCheckpointDir) ? "テスト用未保存モード" : "初回Gen完了待ち");
            sb.Append($"保存: {saveTag}");

            GUI.Label(new Rect(x0 + 10f, y0 + 48f, panelW - 20f, panelH - (_trainingCompleted ? 134f : 56f)), sb.ToString(), _stSmall);

            if (_trainingCompleted)
            {
                float rowY = y0 + panelH - 82f;
                GUI.Label(new Rect(x0 + 10f, rowY + 3f, 92f, 22f), "追加育成(世代):", _stSmall);
                if (GUI.Button(new Rect(x0 + 104f, rowY, 26f, 26f), "-", _stBtnSub))
                    SetContGenerations(_contGenerations - 1);
                string nextCont = GUI.TextField(new Rect(x0 + 132f, rowY + 2f, 42f, 22f), _contGenInputBuf, 4, _stTextField);
                if (nextCont != _contGenInputBuf)
                {
                    _contGenInputBuf = nextCont;
                    if (int.TryParse(nextCont, out int parsedCont))
                        _contGenerations = Mathf.Clamp(parsedCont, 1, 500);
                }
                if (GUI.Button(new Rect(x0 + 176f, rowY, 26f, 26f), "+", _stBtnSub))
                    SetContGenerations(_contGenerations + 1);
                if (GUI.Button(new Rect(x0 + 206f, rowY, panelW - 216f, 26f), "▶ 続けて学習", _stBtnPrimary))
                {
                    SetGenerations(_contGenerations);
                    uiResumeFromCheckpoint = true;
                    RefreshCheckpointList();
                    int latestIdx = _checkpoints.FindIndex(c => string.Equals(c.fileName, "pbt_latest_champion.pt", StringComparison.OrdinalIgnoreCase));
                    if (latestIdx >= 0) _selectedCkptIdx = latestIdx;
                    StartTrainingSession();
                    return;
                }

                if (GUI.Button(new Rect(x0 + 10f, y0 + panelH - 44f, panelW - 20f, 32f), "▶ 育てた最新重みで観戦モードへ切替", _stBtnSpectate))
                {
                    RefreshCheckpointList();
                    uiSelfPlaySpectate = true;
                    StartSpectateSession();
                }
            }
        }

        private void DrawSpectateStatusStrip()
        {
            Rect stripRect = GetSpectateStatusStripRect();
            float panelW = stripRect.width;
            float panelH = stripRect.height;
            float x0 = stripRect.x;
            float y0 = stripRect.y;

            GUI.Box(new Rect(x0, y0, panelW, panelH), GUIContent.none, _stPanel);
            bool contLearnActive = coordinator != null && coordinator.spectateContinualLearning;
            string topHdr = contLearnActive
                ? "<color=#7EE787><b>観戦モード (自動継続・継続学習ON)</b></color>"
                : "<color=#66CCFF><b>推論・観戦モード (自動継続・学習OFF)</b></color>";
            GUI.Label(new Rect(x0 + 10f, y0 + 7f, panelW - 96f, 22f), topHdr, _stHeader);
            if (GUI.Button(new Rect(x0 + panelW - 82f, y0 + 6f, 72f, 22f), "Menu Esc", _stBtnSub))
            {
                ReturnToLauncherMenu();
                return;
            }

            var ar = multiManager != null && multiManager.arenas.Count > 0 ? multiManager.arenas[0] : null;
            int doneEp = Mathf.Max(_spectateMatchesReported, ar != null ? ar.episodesCompleted : 0);
            int wA = ar != null ? ar.teamAWins : _spectateWinsA;
            int wB = ar != null ? ar.teamBWins : _spectateWinsB;
            int ko = ar != null ? ar.knockoutCount : 0;

            string learnStateTag = contLearnActive
                ? (_spectateSavedCkpt ? $"ContinualSave済/Slp:{_spectateSleepLoss:F3}" : "ContinualLearning")
                : "Frozen";
            string modeName = coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate
                ? $"AI vs AI ({learnStateTag})"
                : $"AI vs 教師Bot ({learnStateTag})";
            string ckptShort = string.IsNullOrEmpty(coordinator.loadCheckpointRelativePath)
                ? "初期重み"
                : Path.GetFileName(coordinator.loadCheckpointRelativePath);
            string stShort = InkBlitzArenaSettings.GetStageShortName(coordinator != null ? coordinator.activeStageId : uiStageId);

            GUI.Label(new Rect(x0 + 10f, y0 + 30f, panelW - 20f, 44f),
                $"対戦: <b>{modeName}</b> [{stShort}]\n使用重み: <color=#AADDFF>{ckptShort}</color> | 第<b>{doneEp + 1}</b>試合進行中 (A {wA}-{wB} B, KO:{ko})", _stSmall);

            InkColorPalettePreset actPal = InkBlitzInkPalettes.Get(multiManager != null ? multiManager.activePaletteIndex : uiColorPaletteIndex);
            GUI.Label(new Rect(x0 + 10f, y0 + 78f, 74f, 22f), $"速度:<b>{Time.timeScale:F1}x</b>", _stSmall);
            if (GUI.Button(new Rect(x0 + 84f, y0 + 76f, 40f, 22f), "1.0x", _stBtnSub))
            {
                ApplySpectateTimeScale(1.0f);
            }
            if (GUI.Button(new Rect(x0 + 128f, y0 + 76f, 40f, 22f), "2.0x", _stBtnSub))
            {
                ApplySpectateTimeScale(2.0f);
            }
            if (GUI.Button(new Rect(x0 + 172f, y0 + 76f, 80f, 22f), $"配色<color=#{actPal.hexA}>●</color><color=#{actPal.hexB}>●</color>(P)", _stBtnSub))
            {
                ApplyPaletteSelection((multiManager != null ? multiManager.activePaletteIndex : uiColorPaletteIndex) + 1);
            }
            if (GUI.Button(new Rect(x0 + 256f, y0 + 76f, 88f, 22f), "視点戻す(C)", _stBtnSub))
            {
                if (multiManager != null)
                    multiManager.ResetCameraView(instant: false);
            }

            float pitchDeg = multiManager != null ? multiManager.CurrentCameraPitchDeg : 43f;
            GUI.Label(new Rect(x0 + 10f, y0 + 104f, panelW - 16f, 20f),
                $"<color=#99BBDD>カメラ({pitchDeg:F0}°): ドラッグ回転・水平化 / Shiftドラッグ移動 / ホイール拡大</color>", _stSmall);

            if (coordinator != null
                && coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate
                && coordinator.spectate2v2BrainRadio)
            {
                Rect rr = new Rect(x0, y0 + panelH + 6f, panelW, 74f);
                GUI.Box(rr, GUIContent.none, _stPanel);
                GUI.DrawTexture(new Rect(rr.x, rr.y, rr.width, 2f), _texAccentGreen);
                GUI.Label(new Rect(rr.x + 10f, rr.y + 5f, rr.width - 20f, 20f),
                    $"<color=#86EFAC><b>📡 Vygotsky 10-bit Radio (1Hz)</b></color> <color=#CBD5E1>MI:<b>{_rosettaJointMi:F2}b</b> Φc:<b>{_rosettaPhiComm:F3}b</b> 語彙:<b>{_rosettaUniquePkts}</b></color>",
                    _stSmall);
                GUI.Label(new Rect(rr.x + 10f, rr.y + 26f, rr.width - 20f, 44f),
                    $"A1 <color=#FDE68A>{_rosettaGlyphA1}</color>[{_rosettaGlossA1}] ⇄ A2 <color=#FDE68A>{_rosettaGlyphA2}</color>[{_rosettaGlossA2}]\n"
                    + $"B1 <color=#93C5FD>{_rosettaGlyphB1}</color> ⇄ B2 <color=#93C5FD>{_rosettaGlyphB2}</color>  (Ally聴取KL: {_rosettaPhiAlly:F3}b)",
                    _stSmall);
            }
        }

        private void DrawGemmaBroadcastCard()
        {
            Rect r = GetGemmaBroadcastRect();
            if (!showGemmaTickerInSpectate)
            {
                _coachInputFocused = false;
                if (GUI.Button(r, "▲ Gemma実況・提案 (G)", _stBtnSub))
                    showGemmaTickerInSpectate = true;
                return;
            }

            GUI.Box(r, GUIContent.none, _stPanel);
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, 2f), _gemmaOllamaOnline ? _texAccentGreen : _texAccentBlue);

            bool wideCard = r.width >= 620f;
            float rightControlsW = wideCard ? 266f : 148f;

            string srcBadge = _gemmaOllamaOnline
                ? $"<color=#7EE787><b>● Ollama ({_gemmaSource})</b></color>"
                : "<color=#88CCFF><b>◆ 戦略DBアナリスト</b></color>";
            string dbMeta = wideCard
                ? $"局面:<b>{_gemmaRegime}</b>/<b>{_gemmaLane}</b> | 検証:<b>{_gemmaValidatedCnt}</b> 仮説:<b>{_gemmaCandidateCnt}</b> 蓄積:<b>{_gemmaEpisodesCnt}</b>戦"
                : $"<b>{_gemmaRegime}</b>/<b>{_gemmaLane}</b> | 蓄積:<b>{_gemmaEpisodesCnt}</b>戦";

            GUI.Label(new Rect(r.x + 10f, r.y + 4f, Mathf.Max(120f, r.width - rightControlsW - 14f), 20f),
                $"{srcBadge}  <color=#CBD5E1>{dbMeta}</color>", _stSmall);

            // Top-right AivisSpeech voice narrator controls + hide button
            float rx = r.x + r.width - 46f;
            if (GUI.Button(new Rect(rx, r.y + 4f, 38f, 19f), "隠す", _stBtnSub))
            {
                _coachInputFocused = false;
                GUIUtility.keyboardControl = 0;
                showGemmaTickerInSpectate = false;
                return;
            }
            if (voiceCoach != null)
            {
                bool ttsOn = voiceCoach.enableVoiceNarration;
                bool spkOnline = voiceCoach.IsTtsServerOnline;
                string spkIndicator = voiceCoach.IsSpeakingNow ? "♪" : (ttsOn ? "●" : "○");
                if (wideCard)
                {
                    rx -= 24f;
                    if (GUI.Button(new Rect(rx, r.y + 4f, 22f, 19f), "▶", _stBtnSub))
                        voiceCoach.CycleVoiceStyle(1);
                    rx -= 120f;
                    string vLbl = spkOnline
                        ? $"<color=#AADDFF>{voiceCoach.GetSelectedVoiceShortLabel()}</color>"
                        : "<color=#94A3B8>Aivis再接続</color>";
                    if (GUI.Button(new Rect(rx, r.y + 4f, 118f, 19f), vLbl, _stBtnSub))
                        voiceCoach.CycleVoiceStyle(1);
                    rx -= 24f;
                    if (GUI.Button(new Rect(rx, r.y + 4f, 22f, 19f), "◀", _stBtnSub))
                        voiceCoach.CycleVoiceStyle(-1);
                    rx -= 66f;
                    string toggleTxt = ttsOn
                        ? (spkOnline ? $"<color=#7EE787>{spkIndicator}音ON(N)</color>" : "<color=#FFD27F>音ON(未)</color>")
                        : "<color=#94A3B8>音OFF(N)</color>";
                    if (GUI.Button(new Rect(rx, r.y + 4f, 64f, 19f), toggleTxt, _stBtnSub))
                        voiceCoach.SetVoiceNarrationEnabled(!ttsOn);
                }
                else
                {
                    rx -= 52f;
                    if (GUI.Button(new Rect(rx, r.y + 4f, 50f, 19f), "声切替", _stBtnSub))
                        voiceCoach.CycleVoiceStyle(1);
                    rx -= 56f;
                    string toggleCompact = ttsOn ? $"<color=#7EE787>{spkIndicator}音ON</color>" : "音OFF";
                    if (GUI.Button(new Rect(rx, r.y + 4f, 54f, 19f), toggleCompact, _stBtnSub))
                        voiceCoach.SetVoiceNarrationEnabled(!ttsOn);
                }
            }

            // Middle: 2-line gentle girl navigator commentary
            GUI.Label(new Rect(r.x + 10f, r.y + 25f, r.width - 20f, 42f),
                $"<color=#FDE68A>[{_gemmaHypothesis}]</color> {_gemmaCommentary}", _stBody);

            // Bottom: Optional Interactive Coaching Bar (USB Mic faster-whisper STT -> Editable Input -> Gemma 4)
            if (voiceCoach != null)
            {
                float rowY = r.y + 69f;
                float micW = 94f;
                float sendW = 78f;
                float autoW = 58f;
                float inputX = r.x + 10f + micW + 6f;
                float inputW = Mathf.Max(110f, r.width - 20f - micW - sendW - autoW - 18f);

                string micLabel = voiceCoach.IsRecordingMic
                    ? "■ 停止・認識"
                    : (voiceCoach.IsTranscribingMic || voiceCoach.IsSttLoading ? "認識準備中..." : "● 音声入力");
                GUIStyle micStyle = voiceCoach.IsRecordingMic ? _stBtnDanger : _stBtnSub;
                if (GUI.Button(new Rect(r.x + 10f, rowY, micW, 22f), micLabel, micStyle))
                {
                    voiceCoach.ToggleMicRecording();
                }

                Rect tfRect = new Rect(inputX, rowY, inputW, 22f);
                GUI.SetNextControlName("CoachInputField");
                voiceCoach.coachInputText = GUI.TextField(tfRect, voiceCoach.coachInputText ?? "", 260, _stCoachTextField);
                _coachInputFocused = (GUI.GetNameOfFocusedControl() == "CoachInputField");

                if (!_coachInputFocused && string.IsNullOrEmpty(voiceCoach.coachInputText))
                {
                    GUI.Label(
                        new Rect(tfRect.x + 6f, tfRect.y + 2f, tfRect.width - 10f, 19f),
                        wideCard
                            ? "<color=#7F93A8>任意: 戦略提案を入力 / USBマイク認識 (未入力時はAIがそのまま自律継続)</color>"
                            : "<color=#7F93A8>任意: 戦略提案入力 / マイク認識</color>",
                        _stSmall);
                }

                float sendX = inputX + inputW + 6f;
                bool hasProposal = !string.IsNullOrWhiteSpace(voiceCoach.coachInputText);
                Color prevSendBg = GUI.backgroundColor;
                if (hasProposal) GUI.backgroundColor = new Color(0.55f, 0.92f, 0.72f, 1f);
                if (GUI.Button(new Rect(sendX, rowY, sendW, 22f), "Gemma提案", _stBtnSub))
                {
                    if (voiceCoach.SubmitCoachingProposalToGemma())
                    {
                        GUIUtility.keyboardControl = 0;
                        _coachInputFocused = false;
                    }
                }
                GUI.backgroundColor = prevSendBg;

                float autoX = sendX + sendW + 6f;
                string autoLbl = voiceCoach.autoSendAfterWhisper ? "<color=#7EE787>●自動送</color>" : "○自動送";
                if (GUI.Button(new Rect(autoX, rowY, autoW, 22f), autoLbl, _stBtnSub))
                {
                    voiceCoach.autoSendAfterWhisper = !voiceCoach.autoSendAfterWhisper;
                    voiceCoach.SaveVoicePrefs();
                }

                string hintLine;
                if (!string.IsNullOrEmpty(voiceCoach.coachStatusMessage))
                {
                    hintLine = $"<color=#A5F3FC>{voiceCoach.coachStatusMessage}</color>";
                }
                else if (voiceCoach.IsTtsServerOnline)
                {
                    hintLine = "<color=#94A3B8>AivisSpeech連携中 (重要節目・試合後のみ自動発声) | 提案送信は [Gemma提案] または Ctrl+Enter</color>";
                }
                else
                {
                    hintLine = "<color=#94A3B8>重要節目で方針変更がない時は無言で自律継続します | AivisSpeech起動時は [音ON] で自動接続</color>";
                }
                GUI.Label(new Rect(r.x + 10f, r.y + 93f, r.width - 20f, 18f), hintLine, _stSmall);
            }
        }

        private static string ShortArchetype(string arch)
        {
            if (string.IsNullOrEmpty(arch)) return "-";
            switch (arch)
            {
                case "ObjectiveStriker": return "ObjStriker";
                case "TurfController": return "TurfCtrl";
                case "StealthInterceptor": return "StealthInt";
                case "BalancedTactician": return "Balanced";
                default: return arch.Length > 10 ? arch.Substring(0, 10) : arch;
            }
        }
    }
}
