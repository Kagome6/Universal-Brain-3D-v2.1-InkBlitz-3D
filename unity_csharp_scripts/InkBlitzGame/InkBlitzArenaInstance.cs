// InkBlitzArenaInstance.cs — Arena-local 1v1 BlitzCore orchestrator wiring Turf, Projectiles, Audio, BlitzCore & BrainBridge
using System.Collections.Generic;
using UnityEngine;
using UniversalBrain3D;

namespace InkBlitz3D
{
    public class InkBlitzArenaInstance : MonoBehaviour
    {
        public const byte HOLD_INTER_GEN_WAIT = 250;

        [Header("Arena Configuration")]
        public int arenaIndex = 0;
        [HideInInspector] public bool drivenByMultiManager = false;
        public InkBlitzArenaSettings settings = new InkBlitzArenaSettings();

        [Header("Core Subsystems")]
        public TurfGridSystem turfGrid;
        public ArenaStageBuilder stageBuilder;
        public ArenaAudioSystem audioSystem;
        public InkProjectileManager projectileManager;
        [UnityEngine.Serialization.FormerlySerializedAs("rainmaker")]
        public BlitzCoreObjective blitzCore;
        public List<DeflectPowerPickup> deflectPickups = new List<DeflectPowerPickup>();
        public List<InkBlitzPlayerAgent> agents = new List<InkBlitzPlayerAgent>();

        [Header("Match & Episode State (Read-Only)")]
        public float matchTimeRemaining = 90f;
        public bool isOvertime = false;
        public float bestProgressTeamA = 0f;
        public float bestProgressTeamB = 0f;
        public int episodesCompleted = 0;
        public int teamAWins = 0;
        public int teamBWins = 0;
        public int knockoutCount = 0;
        public int timeoutCount = 0;
        public string lastMatchSummary = "In Progress";

        [Header("Debug & Verification")]
        public bool logPeriodicTelemetry = true;
        public int telemetryIntervalTicks = 120;

        private readonly List<UniversalSemanticEntityTag> _arenaEntities = new List<UniversalSemanticEntityTag>();
        private readonly List<UniversalBrainBridge3D> _pendingRecvBridges = new List<UniversalBrainBridge3D>(4);
        private readonly Dictionary<InkBlitzPlayerAgent, HeuristicSparringBot> _bots = new Dictionary<InkBlitzPlayerAgent, HeuristicSparringBot>();
        private readonly Dictionary<InkBlitzPlayerAgent, SupportTrooperBot> _trooperBots = new Dictionary<InkBlitzPlayerAgent, SupportTrooperBot>();
        private readonly Dictionary<InkBlitzPlayerAgent, string> _radioLastPacket = new Dictionary<InkBlitzPlayerAgent, string>();
        private readonly Dictionary<InkBlitzPlayerAgent, int> _radioLastEmitTick = new Dictionary<InkBlitzPlayerAgent, int>();
        private UniversalArenaCoordinator3D _coordinator;
        private float _prevPushTeamA = 0f;
        private float _prevPushTeamB = 0f;
        private InkBlitzPlayerAgent _activeCarryAgent = null;
        private float _activeCarryBestPot = 0f;
        private float _overtimeElapsed = 0f;
        private float _overtimePickupGraceRemaining = 0f;
        private int _overtimeTrailingTeam = 0;
        private bool _episodeEndedThisStep = false;
        private bool _initialized = false;
        private bool _floorUvsApplied = false;
        private int _tickCount = 0;
        private Renderer[] _cachedRenderers;
        private Material _runtimeMatTeamA;
        private Material _runtimeMatTeamB;
        private Material _runtimeMatTrooperA;
        private Material _runtimeMatTrooperB;

        public int TickCount => _tickCount;
        public bool IsRenderingEnabled { get; private set; } = true;

        public bool IsHoldingForNextGen
        {
            get
            {
                for (int i = 0; i < agents.Count; i++)
                {
                    var br = agents[i] != null ? agents[i].brainBridge : null;
                    if (br != null && br.IsConnected && br.latestHoldReasonId == HOLD_INTER_GEN_WAIT)
                        return true;
                }
                return false;
            }
        }

        private void Awake()
        {
            InitializeArena(registerWithCoordinator: true);
        }

        public void InitializeArena(bool registerWithCoordinator = true)
        {
            if (_initialized) return;

            if (settings == null) settings = new InkBlitzArenaSettings();
            if (turfGrid == null) turfGrid = GetComponent<TurfGridSystem>() ?? gameObject.AddComponent<TurfGridSystem>();
            if (stageBuilder == null) stageBuilder = GetComponent<ArenaStageBuilder>() ?? gameObject.AddComponent<ArenaStageBuilder>();
            if (audioSystem == null) audioSystem = GetComponent<ArenaAudioSystem>() ?? gameObject.AddComponent<ArenaAudioSystem>();
            audioSystem.BindArena(this);
            if (projectileManager == null) projectileManager = GetComponent<InkProjectileManager>() ?? gameObject.AddComponent<InkProjectileManager>();
            stageBuilder?.EnsureTerrainManifoldReady(settings);
            if (blitzCore == null) blitzCore = GetComponentInChildren<BlitzCoreObjective>(true);

            if (turfGrid.templateFloorMaterial == null && stageBuilder != null)
                turfGrid.templateFloorMaterial = stageBuilder.defaultFloorMaterial;
            turfGrid.Initialize(transform.position, settings);
            turfGrid.BakeWallObstacleMask(transform.Find("WallsAndCovers"));
            projectileManager.Initialize(this);

            if (stageBuilder != null && !_floorUvsApplied)
            {
                for (int i = 0; i < stageBuilder.paintableFloorRenderers.Count; i++)
                {
                    MeshRenderer mr = stageBuilder.paintableFloorRenderers[i];
                    if (mr == null) continue;
                    mr.sharedMaterial = turfGrid.TurfFloorMaterial;
                    TurfGridSystem.ApplyArenaPlanarUVs(
                        mr.GetComponent<MeshFilter>(),
                        transform.position,
                        settings.arenaWidthX,
                        settings.arenaLengthZ
                    );
                }
                _floorUvsApplied = true;
            }

            if (agents.Count == 0)
                GetComponentsInChildren(true, agents);
            SortAgentsPrimaryFirst();

            _bots.Clear();
            _trooperBots.Clear();
            for (int i = 0; i < agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = agents[i];
                if (ag == null) continue;
                ag.arenaIndex = arenaIndex;
                if (ag.isSupportTrooperBot)
                    ag.gameObject.SetActive(settings.enableSupportTrooperBots);
                ag.Initialize(transform.position, settings, turfGrid, this);
                var bot = ag.GetComponent<HeuristicSparringBot>();
                if (bot != null)
                    _bots[ag] = bot;
                var tr = ag.GetComponent<SupportTrooperBot>();
                if (tr != null)
                {
                    tr.ResetTrooperState();
                    _trooperBots[ag] = tr;
                }
            }

            if (blitzCore != null)
                blitzCore.Initialize(transform.position, settings);

            if (deflectPickups.Count == 0)
                GetComponentsInChildren(true, deflectPickups);
            for (int i = 0; i < deflectPickups.Count; i++)
            {
                if (deflectPickups[i] != null)
                    deflectPickups[i].Initialize(settings);
            }

            RefreshArenaEntities();
            ResetMatch();

            if (registerWithCoordinator)
            {
                if (_coordinator == null)
                    _coordinator = Object.FindAnyObjectByType<UniversalArenaCoordinator3D>();
                if (_coordinator != null)
                {
                    bool dualBrains = _coordinator.sessionMode == BrainSessionRunMode.ParallelCompetitiveSelfPlay
                                   || _coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate;
                    for (int i = 0; i < agents.Count; i++)
                    {
                        if (agents[i] == null || agents[i].isSupportTrooperBot || agents[i].brainBridge == null) continue;
                        if (!dualBrains && agents[i].teamId < 0)
                            continue;
                        _coordinator.RegisterBridge(agents[i].brainBridge);
                    }
                }
            }

            _initialized = true;
        }

        public void ResetSessionStats()
        {
            episodesCompleted = 0;
            teamAWins = 0;
            teamBWins = 0;
            knockoutCount = 0;
            timeoutCount = 0;
            _tickCount = 0;
            _episodeEndedThisStep = false;
            lastMatchSummary = "Waiting / In Progress";
            if (_initialized)
                ResetMatch();
        }

        public void ConfigureForSlot(int slotIndex, int portTeamA, int portTeamB, UniversalArenaCoordinator3D coordinator)
        {
            arenaIndex = slotIndex;
            _coordinator = coordinator;
            Physics.SyncTransforms();
            InitializeArena(registerWithCoordinator: false);
            turfGrid?.BakeWallObstacleMask(transform.Find("WallsAndCovers"));

            // Re-sync arena material onto all paintable floors (covers repeated calls safely)
            if (stageBuilder != null && turfGrid != null)
            {
                for (int i = 0; i < stageBuilder.paintableFloorRenderers.Count; i++)
                {
                    MeshRenderer mr = stageBuilder.paintableFloorRenderers[i];
                    if (mr != null)
                        mr.sharedMaterial = turfGrid.TurfFloorMaterial;
                }
            }

            if (agents.Count == 0)
                GetComponentsInChildren(true, agents);
            SortAgentsPrimaryFirst();

            _bots.Clear();
            _trooperBots.Clear();
            bool is2v2PeerRadio = coordinator != null
                && coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate
                && coordinator.spectate2v2BrainRadio;
            LayerMask defaultRayMask = (agents.Count > 0 && agents[0] != null && agents[0].brainBridge != null)
                ? agents[0].brainBridge.raycastMask
                : Physics.DefaultRaycastLayers;

            for (int i = 0; i < agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = agents[i];
                if (ag == null) continue;
                ag.arenaIndex = slotIndex;
                ag.isPromotedPeerBrain = ag.isSupportTrooperBot && is2v2PeerRadio;
                if (ag.isSupportTrooperBot)
                    ag.gameObject.SetActive(settings.enableSupportTrooperBots || is2v2PeerRadio);
                if (ag.isPromotedPeerBrain && ag.brainBridge == null)
                {
                    ag.brainBridge = ag.gameObject.GetComponent<UniversalBrainBridge3D>() ?? ag.gameObject.AddComponent<UniversalBrainBridge3D>();
                    ag.brainBridge.raycastMask = defaultRayMask;
                }
                if ((!ag.isSupportTrooperBot || ag.isPromotedPeerBrain) && ag.brainBridge != null)
                {
                    ag.brainBridge.arenaIndex = slotIndex;
                    ag.brainBridge.arenaOrigin = transform.position;
                    ag.brainBridge.stageId = settings != null ? settings.stageId : 1;
                    if (!ag.isSupportTrooperBot)
                        ag.brainBridge.port = (ag.teamId > 0) ? portTeamA : portTeamB;
                    else
                        ag.brainBridge.port = (ag.teamId > 0) ? (portTeamA + 2) : (portTeamA + 3);
                }
                ag.Initialize(transform.position, settings, turfGrid, this);
                var bot = ag.GetComponent<HeuristicSparringBot>();
                if (bot != null)
                    _bots[ag] = bot;
                var tr = ag.GetComponent<SupportTrooperBot>();
                if (tr != null)
                {
                    tr.enableTrooperAi = !ag.isPromotedPeerBrain;
                    tr.ResetTrooperState();
                    _trooperBots[ag] = tr;
                }
            }

            if (blitzCore != null)
                blitzCore.Initialize(transform.position, settings);

            if (deflectPickups.Count == 0)
                GetComponentsInChildren(true, deflectPickups);
            for (int i = 0; i < deflectPickups.Count; i++)
            {
                if (deflectPickups[i] != null)
                    deflectPickups[i].Initialize(settings);
            }

            RefreshArenaEntities();
            ResetMatch();

            if (coordinator != null)
            {
                bool dualBrains = coordinator.sessionMode == BrainSessionRunMode.ParallelCompetitiveSelfPlay
                               || coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate;
                for (int i = 0; i < agents.Count; i++)
                {
                    InkBlitzPlayerAgent ag = agents[i];
                    if (ag == null || ag.brainBridge == null) continue;
                    if (ag.isSupportTrooperBot && !ag.isPromotedPeerBrain)
                    {
                        ag.brainBridge.Disconnect();
                        continue;
                    }
                    if (ag.teamId > 0 || dualBrains)
                        coordinator.RegisterBridge(ag.brainBridge);
                    else
                        ag.brainBridge.Disconnect();
                }
            }
        }

        private void SortAgentsPrimaryFirst()
        {
            agents.Sort((a, b) =>
            {
                int supA = (a != null && a.isSupportTrooperBot) ? 1 : 0;
                int supB = (b != null && b.isSupportTrooperBot) ? 1 : 0;
                if (supA != supB)
                    return supA.CompareTo(supB); // Primary brains (0) before Support bots (1)
                int ta = a != null ? a.teamId : 0;
                int tb = b != null ? b.teamId : 0;
                return tb.CompareTo(ta); // Team A (+1) before Team B (-1)
            });
        }

        public void RebuildArenaForStage(int newStageId)
        {
            if (settings == null) settings = new InkBlitzArenaSettings();
            bool supEnabled = settings.enableSupportTrooperBots;
            Color prevColA = settings.teamAColor;
            Color prevColB = settings.teamBColor;
            settings.ApplyStagePreset(newStageId);
            settings.enableSupportTrooperBots = supEnabled;
            settings.teamAColor = prevColA;
            settings.teamBColor = prevColB;

            if (stageBuilder == null) stageBuilder = GetComponent<ArenaStageBuilder>() ?? gameObject.AddComponent<ArenaStageBuilder>();
            if (turfGrid == null) turfGrid = GetComponent<TurfGridSystem>() ?? gameObject.AddComponent<TurfGridSystem>();

            stageBuilder.BuildStage(arenaIndex, settings);
            Physics.SyncTransforms();

            if (turfGrid.templateFloorMaterial == null && stageBuilder != null)
                turfGrid.templateFloorMaterial = stageBuilder.defaultFloorMaterial;
            turfGrid.Initialize(transform.position, settings);
            turfGrid.BakeWallObstacleMask(stageBuilder.transform.Find("WallsAndCovers"));

            for (int i = 0; i < stageBuilder.paintableFloorRenderers.Count; i++)
            {
                MeshRenderer mr = stageBuilder.paintableFloorRenderers[i];
                if (mr == null) continue;
                mr.sharedMaterial = turfGrid.TurfFloorMaterial;
                TurfGridSystem.ApplyArenaPlanarUVs(
                    mr.GetComponent<MeshFilter>(),
                    transform.position,
                    settings.arenaWidthX,
                    settings.arenaLengthZ
                );
            }
            _floorUvsApplied = true;

            Transform lm = stageBuilder.transform.Find("Landmarks");
            blitzCore = lm != null ? lm.GetComponentInChildren<BlitzCoreObjective>(true) : GetComponentInChildren<BlitzCoreObjective>(true);
            if (blitzCore != null)
                blitzCore.Initialize(transform.position, settings);

            deflectPickups.Clear();
            if (lm != null)
                lm.GetComponentsInChildren(true, deflectPickups);
            for (int i = 0; i < deflectPickups.Count; i++)
            {
                if (deflectPickups[i] != null)
                    deflectPickups[i].Initialize(settings);
            }

            for (int i = 0; i < agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = agents[i];
                if (ag == null) continue;
                if (!ag.isSupportTrooperBot && ag.brainBridge != null)
                    ag.brainBridge.stageId = settings.stageId;
                ag.Initialize(transform.position, settings, turfGrid, this);
                if (_trooperBots.TryGetValue(ag, out var tr) && tr != null)
                    tr.ResetTrooperState();
            }

            _cachedRenderers = null;
            RefreshArenaEntities();
            if (_initialized)
                ResetMatch();
            if (!IsRenderingEnabled)
                SetArenaRenderingEnabled(false);
        }

        public void SetSupportTrooperBotsEnabled(bool enabled)
        {
            if (settings == null) settings = new InkBlitzArenaSettings();
            settings.enableSupportTrooperBots = enabled;
            for (int i = 0; i < agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = agents[i];
                if (ag == null || !ag.isSupportTrooperBot) continue;
                if (enabled || ag.isPromotedPeerBrain)
                {
                    ag.gameObject.SetActive(true);
                    ag.ResetForNewMatch();
                    if (_trooperBots.TryGetValue(ag, out var tr) && tr != null)
                    {
                        tr.enableTrooperAi = !ag.isPromotedPeerBrain;
                        tr.ResetTrooperState();
                    }
                }
                else
                {
                    ag.gameObject.SetActive(false);
                }
            }
            _cachedRenderers = null;
            RefreshArenaEntities();
        }

        public void SetArenaRenderingEnabled(bool enabled)
        {
            IsRenderingEnabled = enabled;
            if (_cachedRenderers == null || _cachedRenderers.Length == 0)
                _cachedRenderers = GetComponentsInChildren<Renderer>(true);
            for (int i = 0; i < _cachedRenderers.Length; i++)
            {
                if (_cachedRenderers[i] != null)
                    _cachedRenderers[i].enabled = enabled;
            }
            turfGrid?.SetRenderingEnabled(enabled);
        }

        public void OnBlitzCoreHoldTimeoutExplode(InkBlitzPlayerAgent carrier)
        {
            if (carrier == null)
                return;

            Vector3 explodePos = carrier.transform.position;
            int team = carrier.teamId;
            blitzCore?.ResetToCenter();
            carrier.EnterRespawnState(settings.respawnDurationSec);
            carrier.brainBridge?.AddStepReward(settings.deathPenalty);
            audioSystem?.EmitSound(explodePos, -team, 1.0f, 0.45f);
            audioSystem?.PlaySplatSfx(explodePos, false);
        }

        public void RefreshArenaEntities()
        {
            _arenaEntities.Clear();
            UniversalSemanticEntityTag[] tags = GetComponentsInChildren<UniversalSemanticEntityTag>(true);
            for (int i = 0; i < tags.Length; i++)
            {
                if (tags[i] != null)
                {
                    tags[i].arenaIndex = arenaIndex;
                    _arenaEntities.Add(tags[i]);
                }
            }
        }

        public void TurfPaint(Vector3 worldPos, float radiusM, int teamId, InkBlitzPlayerAgent creditee = null)
        {
            if (turfGrid == null) return;
            var res = turfGrid.PaintCircle(worldPos, radiusM, teamId);
            if (creditee != null && res.newlyExploredCells > 0)
                creditee.AddExploreCells(res.newlyExploredCells);
        }

        public void OnProjectileHitAgent(InkBlitzPlayerAgent shooter, InkBlitzPlayerAgent victim, float damage, Vector3 hitPos, bool isSplash)
        {
            if (victim == null || victim.isRespawning) return;

            bool splatted = victim.TakeDamage(damage);
            if (!splatted) return;

            bool wasCarrier = victim.isCarryingObjective;
            Vector3 dropWorldPos = victim.transform.position;
            float respawnDur = victim.IsScriptedSupportBot ? settings.supportBotRespawnSec : settings.respawnDurationSec;
            victim.EnterRespawnState(respawnDur);
            if (wasCarrier && blitzCore != null)
                blitzCore.DropAt(dropWorldPos);

            if (shooter != null && shooter.teamId != victim.teamId)
            {
                float reward = victim.IsScriptedSupportBot
                    ? settings.splatSupportBotReward
                    : (wasCarrier ? settings.splatEnemyCarrierReward : settings.splatStandardReward);
                shooter.brainBridge?.AddStepReward(reward);
            }
            victim.brainBridge?.AddStepReward(settings.deathPenalty);
            audioSystem?.EmitSound(hitPos, victim.teamId, 0.90f, 0.35f);
            audioSystem?.PlaySplatSfx(hitPos, victim.IsScriptedSupportBot);
        }

        public float GetObjectiveProgressForTeam(int teamId)
        {
            float livePushA = 0f;
            float livePushB = 0f;
            if (blitzCore != null && blitzCore.State == BlitzCoreState.Carried && blitzCore.Carrier != null)
            {
                float p = ComputeCarrierPush01(blitzCore.Carrier);
                if (blitzCore.Carrier.teamId > 0) livePushA = p;
                else livePushB = p;
            }

            float ourBest = (teamId > 0) ? bestProgressTeamA : bestProgressTeamB;
            float oppBest = (teamId > 0) ? bestProgressTeamB : bestProgressTeamA;
            float ourCurr = (teamId > 0) ? livePushA : livePushB;
            float oppCurr = (teamId > 0) ? livePushB : livePushA;

            float ourScore = Mathf.Max(ourBest, ourCurr * 0.9f);
            float oppScore = Mathf.Max(oppBest, oppCurr * 0.9f);
            return Mathf.Clamp01(0.50f + 0.50f * (ourScore - oppScore));
        }

        private void FixedUpdate()
        {
            if (drivenByMultiManager) return;
            if (_coordinator != null && _coordinator.IsEmergencyHalted) return;
            SimulateStep(Time.fixedDeltaTime);
        }

        public bool BeginParallelArenaStep(float dt, out bool holdOnly)
        {
            holdOnly = false;
            if (_coordinator != null && _coordinator.IsEmergencyHalted)
                return false;
            if (!_initialized) InitializeArena(registerWithCoordinator: false);
            if (dt <= 0f) return false;

            if (_coordinator != null)
            {
                bool dualBrains = _coordinator.sessionMode == BrainSessionRunMode.ParallelCompetitiveSelfPlay
                               || _coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate;
                bool quadBrains = _coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate
                               && _coordinator.spectate2v2BrainRadio;
                if (quadBrains)
                {
                    bool allFourConnected = agents.Count >= 4;
                    for (int k = 0; k < 4 && allFourConnected; k++)
                    {
                        allFourConnected = agents[k] != null && agents[k].brainBridge != null && agents[k].brainBridge.IsConnected;
                    }
                    if (!allFourConnected)
                    {
                        SyncAllAgentsIdleInterpolation();
                        return false;
                    }
                }
                else if (dualBrains)
                {
                    bool bothConnected = agents.Count >= 2
                        && agents[0] != null && agents[0].brainBridge != null && agents[0].brainBridge.IsConnected
                        && agents[1] != null && agents[1].brainBridge != null && agents[1].brainBridge.IsConnected;
                    if (!bothConnected)
                    {
                        SyncAllAgentsIdleInterpolation();
                        return false;
                    }
                }
                else
                {
                    bool teamAConnected = agents.Count >= 1
                        && agents[0] != null && agents[0].brainBridge != null && agents[0].brainBridge.IsConnected;
                    if (!teamAConnected)
                    {
                        SyncAllAgentsIdleInterpolation();
                        return false;
                    }
                }
            }

            for (int i = 0; i < agents.Count; i++)
            {
                var br = agents[i] != null ? agents[i].brainBridge : null;
                if (br != null && br.IsConnected && br.latestHoldReasonId == HOLD_INTER_GEN_WAIT)
                {
                    holdOnly = true;
                    break;
                }
            }
            if (holdOnly)
            {
                SyncAllAgentsIdleInterpolation();
                BeginExchangeBrainsForStep(creditExplore: false, epDone: false);
                return true;
            }

            _tickCount++;
            _episodeEndedThisStep = false;

            audioSystem?.StepAudio(dt);

            for (int i = 0; i < agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = agents[i];
                if (ag == null || !ag.gameObject.activeSelf) continue;
                if (_bots.TryGetValue(ag, out var bot) && bot != null)
                    bot.StepBot(dt, this);
                if (_trooperBots.TryGetValue(ag, out var tr) && tr != null)
                    tr.StepTrooper(dt, this);
                ag.StepAgent(dt);
            }

            projectileManager?.StepProjectiles(dt);
            for (int i = 0; i < deflectPickups.Count; i++)
            {
                if (deflectPickups[i] != null)
                    deflectPickups[i].StepPickup(dt, this);
            }
            blitzCore?.StepBlitzCore(dt, this);

            UpdateObjectiveAndMatchState(dt);
            BeginExchangeBrainsForStep(creditExplore: true, epDone: _episodeEndedThisStep);
            return true;
        }

        private void SyncAllAgentsIdleInterpolation()
        {
            for (int i = 0; i < agents.Count; i++)
            {
                if (agents[i] != null)
                    agents[i].SyncFixedInterpolationPose();
            }
        }

        public void CompleteParallelArenaStep(bool holdOnly)
        {
            CompleteExchangeBrainsForStep();
            if (holdOnly)
                return;

            if (_episodeEndedThisStep)
            {
                _episodeEndedThisStep = false;
                ResetMatch();
            }

            if (logPeriodicTelemetry && telemetryIntervalTicks > 0 && (_tickCount % telemetryIntervalTicks) == 0)
                LogHeartbeat();
        }

        public void SimulateStep(float dt)
        {
            if (BeginParallelArenaStep(dt, out bool holdOnly))
                CompleteParallelArenaStep(holdOnly);
        }

        private void BeginExchangeBrainsForStep(bool creditExplore, bool epDone)
        {
            _pendingRecvBridges.Clear();
            bool relay2v2Radio = _coordinator != null
                && _coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate
                && _coordinator.spectate2v2BrainRadio;

            // 1. Harvest outgoing 1.0 Hz (`60`-tick) narrowband `@V1:r,s,i` packets from each brain's UBMC tag
            if (relay2v2Radio)
            {
                for (int i = 0; i < agents.Count; i++)
                {
                    InkBlitzPlayerAgent ag = agents[i];
                    if (ag == null || ag.brainBridge == null || !ag.brainBridge.IsConnected) continue;
                    if (ag.IsScriptedSupportBot) continue;
                    string sp = ag.brainBridge.latestInnerSpeech;
                    if (!string.IsNullOrEmpty(sp) && sp.StartsWith("@V1:", System.StringComparison.Ordinal))
                    {
                        int prevTick = _radioLastEmitTick.TryGetValue(ag, out int tPrev) ? tPrev : -999;
                        if (_tickCount - prevTick >= 60)
                        {
                            _radioLastPacket[ag] = sp;
                            _radioLastEmitTick[ag] = _tickCount;
                        }
                    }
                }
            }

            for (int i = 0; i < agents.Count; i++)
            {
                InkBlitzPlayerAgent agent = agents[i];
                if (agent == null || agent.IsScriptedSupportBot) continue;

                if (creditExplore)
                {
                    int newCells = agent.ConsumeNewlyExploredCells();
                    if (newCells > 0 && agent.brainBridge != null)
                        agent.brainBridge.AddStepReward(newCells * settings.turfExploreCellReward);
                }

                if (agent.brainBridge != null && agent.brainBridge.IsConnected)
                {
                    Vector2 egoSinCos = Vector2.zero;
                    float loudness = 0f;
                    if (audioSystem != null)
                    {
                        Transform eye = agent.eyePivot != null ? agent.eyePivot : agent.transform;
                        audioSystem.ComputeObserverAudio(eye, agent.teamId, agent.brainBridge.raycastMask, out egoSinCos, out loudness);
                    }

                    string allyUtterance = "";
                    if (relay2v2Radio)
                    {
                        for (int j = 0; j < agents.Count; j++)
                        {
                            InkBlitzPlayerAgent ally = agents[j];
                            if (ally == null || ally == agent || ally.teamId != agent.teamId || ally.IsScriptedSupportBot)
                                continue;
                            if (_radioLastPacket.TryGetValue(ally, out string pktStr)
                                && _radioLastEmitTick.TryGetValue(ally, out int eTick)
                                && (_tickCount - eTick) <= 180)
                            {
                                allyUtterance = pktStr;
                                break;
                            }
                        }
                    }

                    if (agent.brainBridge.BeginBrainExchangeSend(
                        agent.Velocity,
                        agent.CanCommitNow,
                        epDone,
                        agent.HazardLevel01,
                        agent.SurfacePosture01,
                        agent.ResourceOrCarrier01,
                        GetObjectiveProgressForTeam(agent.teamId),
                        egoSinCos,
                        loudness,
                        _arenaEntities,
                        allyUtterance
                    ))
                    {
                        _pendingRecvBridges.Add(agent.brainBridge);
                    }
                }
            }
        }

        private void CompleteExchangeBrainsForStep()
        {
            for (int i = 0; i < _pendingRecvBridges.Count; i++)
            {
                var bridge = _pendingRecvBridges[i];
                if (bridge != null)
                    bridge.CompleteBrainExchangeRecv();
            }
            _pendingRecvBridges.Clear();
        }

        private void ExchangeBrainsForStep(bool creditExplore, bool epDone)
        {
            BeginExchangeBrainsForStep(creditExplore, epDone);
            CompleteExchangeBrainsForStep();
        }

        private float ComputeCarrierPush01(InkBlitzPlayerAgent carrier)
        {
            Vector3 loc = carrier.transform.position - transform.position;
            Vector3 goalLocal = (carrier.teamId > 0) ? settings.teamBPedestalLocal : settings.teamAPedestalLocal;
            float distToGoal = Vector2.Distance(new Vector2(loc.x, loc.z), new Vector2(goalLocal.x, goalLocal.z));
            float centerDist = Mathf.Max(2.0f, new Vector2(goalLocal.x, goalLocal.z).magnitude);
            return Mathf.Clamp01(1f - distToGoal / centerDist);
        }

        private float ComputeFullArenaCarrierPotential01(InkBlitzPlayerAgent carrier, out float distToGoal, out float vertDelta)
        {
            Vector3 loc = carrier.transform.position - transform.position;
            Vector3 goalLocal = (carrier.teamId > 0) ? settings.teamBPedestalLocal : settings.teamAPedestalLocal;
            distToGoal = Vector2.Distance(new Vector2(loc.x, loc.z), new Vector2(goalLocal.x, goalLocal.z));
            vertDelta = Mathf.Abs(loc.y - goalLocal.y);
            float fullSpan = Mathf.Max(18.0f, Mathf.Sqrt(settings.arenaWidthX * settings.arenaWidthX + settings.arenaLengthZ * settings.arenaLengthZ) * 0.85f);
            float distTerm = Mathf.Clamp01(1f - distToGoal / fullSpan);
            float forwardZ = (carrier.teamId > 0) ? loc.z : -loc.z;
            float goalForwardZ = Mathf.Max(4.0f, Mathf.Abs(goalLocal.z));
            float zTerm = Mathf.Clamp01((forwardZ + goalForwardZ) / (2.0f * goalForwardZ));
            return Mathf.Clamp01(0.82f * distTerm + 0.18f * zTerm);
        }

        private void UpdateObjectiveAndMatchState(float dt)
        {
            if (blitzCore != null && blitzCore.State == BlitzCoreState.Carried && blitzCore.Carrier != null)
            {
                InkBlitzPlayerAgent carrier = blitzCore.Carrier;
                float pot01 = ComputeFullArenaCarrierPotential01(carrier, out float distToGoal, out float vertDelta);
                float push01 = ComputeCarrierPush01(carrier);

                // Initialize per-carry monotonic high-water potential at exact pickup spot (zero oscillation farming, works across full arena!)
                if (_activeCarryAgent != carrier)
                {
                    _activeCarryAgent = carrier;
                    _activeCarryBestPot = pot01;
                }
                else if (pot01 > _activeCarryBestPot)
                {
                    float deltaPot = pot01 - _activeCarryBestPot;
                    _activeCarryBestPot = pot01;
                    AddTeamReward(carrier.teamId, settings.advancePotentialScale * deltaPot);
                }

                if (carrier.teamId > 0)
                {
                    _prevPushTeamA = push01;
                    if (push01 > bestProgressTeamA)
                    {
                        float deltaBest = push01 - bestProgressTeamA;
                        bestProgressTeamA = push01;
                        AddTeamReward(1, settings.bestProgressBonusScale * deltaBest);
                        AddTeamReward(-1, -settings.enemyBestProgressPenaltyScale * deltaBest);
                    }

                    if (distToGoal <= settings.pedestalTouchdownRadiusM && vertDelta <= 1.25f)
                    {
                        bestProgressTeamA = 1.0f;
                        ConcludeMatch(1, isKnockout: true);
                        return;
                    }
                }
                else
                {
                    _prevPushTeamB = push01;
                    if (push01 > bestProgressTeamB)
                    {
                        float deltaBest = push01 - bestProgressTeamB;
                        bestProgressTeamB = push01;
                        AddTeamReward(-1, settings.bestProgressBonusScale * deltaBest);
                        AddTeamReward(1, -settings.enemyBestProgressPenaltyScale * deltaBest);
                    }

                    if (distToGoal <= settings.pedestalTouchdownRadiusM && vertDelta <= 1.25f)
                    {
                        bestProgressTeamB = 1.0f;
                        ConcludeMatch(-1, isKnockout: true);
                        return;
                    }
                }
            }
            else
            {
                _activeCarryAgent = null;
                _activeCarryBestPot = 0f;
                _prevPushTeamA = 0f;
                _prevPushTeamB = 0f;
            }

            if (isOvertime)
            {
                if (_overtimeTrailingTeam > 0 && bestProgressTeamA > bestProgressTeamB + 0.005f)
                {
                    ConcludeMatch(1, isKnockout: false);
                    return;
                }
                if (_overtimeTrailingTeam < 0 && bestProgressTeamB > bestProgressTeamA + 0.005f)
                {
                    ConcludeMatch(-1, isKnockout: false);
                    return;
                }

                _overtimeElapsed += dt;
                bool trailingCarrying = blitzCore != null && blitzCore.State == BlitzCoreState.Carried
                    && blitzCore.Carrier != null && blitzCore.Carrier.teamId == _overtimeTrailingTeam;
                bool pickupGrace = false;
                if (!trailingCarrying && blitzCore != null && blitzCore.State == BlitzCoreState.PickupReady && _overtimePickupGraceRemaining > 0f)
                {
                    _overtimePickupGraceRemaining -= dt;
                    pickupGrace = _overtimePickupGraceRemaining > 0f;
                }

                if ((!trailingCarrying && !pickupGrace) || _overtimeElapsed >= settings.maxOvertimeSec)
                    ConcludeMatch(-_overtimeTrailingTeam, isKnockout: false);
                return;
            }

            matchTimeRemaining = Mathf.Max(0f, matchTimeRemaining - dt);
            if (matchTimeRemaining <= 0f)
            {
                int leader = (bestProgressTeamA > bestProgressTeamB + 0.005f)
                    ? 1
                    : ((bestProgressTeamB > bestProgressTeamA + 0.005f) ? -1 : 0);

                if (settings.enableOvertime && blitzCore != null)
                {
                    if (leader != 0)
                    {
                        int trailing = -leader;
                        bool enterOt = (blitzCore.State == BlitzCoreState.Carried && blitzCore.Carrier != null && blitzCore.Carrier.teamId == trailing)
                                    || (blitzCore.State == BlitzCoreState.PickupReady);
                        if (enterOt)
                        {
                            isOvertime = true;
                            _overtimeTrailingTeam = trailing;
                            _overtimeElapsed = 0f;
                            _overtimePickupGraceRemaining = settings.overtimePickupGraceSec;
                            return;
                        }
                    }
                    else if (blitzCore.State == BlitzCoreState.Carried && blitzCore.Carrier != null)
                    {
                        isOvertime = true;
                        _overtimeTrailingTeam = blitzCore.Carrier.teamId;
                        _overtimeElapsed = 0f;
                        _overtimePickupGraceRemaining = settings.overtimePickupGraceSec;
                        return;
                    }
                }

                int winner = (leader != 0) ? leader : (turfGrid.TeamACells >= turfGrid.TeamBCells ? 1 : -1);
                ConcludeMatch(winner, isKnockout: false);
            }
        }

        private void ConcludeMatch(int winnerTeamId, bool isKnockout)
        {
            episodesCompleted++;
            if (isKnockout)
            {
                knockoutCount++;
                Vector3 goalWorld = transform.position + (winnerTeamId > 0 ? settings.teamBPedestalLocal : settings.teamAPedestalLocal);
                audioSystem?.PlayGoalTouchdownSfx(goalWorld);
            }
            else timeoutCount++;
            if (winnerTeamId > 0) teamAWins++;
            else teamBWins++;

            float winRew = isKnockout ? settings.knockoutWinReward : settings.timeoutWinReward;
            float lossPen = isKnockout ? settings.knockoutLossPenalty : settings.matchLossPenalty;
            AddTeamReward(winnerTeamId, winRew);
            AddTeamReward(-winnerTeamId, lossPen);

            int cntA = Mathf.RoundToInt((1f - bestProgressTeamA) * 100f);
            int cntB = Mathf.RoundToInt((1f - bestProgressTeamB) * 100f);
            string endTag = isKnockout ? "KO" : (isOvertime ? "OT" : "TIME");
            lastMatchSummary = $"Ep#{episodesCompleted} {endTag} Win={(winnerTeamId > 0 ? "TeamA" : "TeamB")} ({cntA}:{cntB})";
            Debug.Log($"[InkBlitzArena] Match Ended! {lastMatchSummary}");
            _episodeEndedThisStep = true;
        }

        private void ResetMatch()
        {
            matchTimeRemaining = settings.matchDurationSec;
            isOvertime = false;
            _overtimeElapsed = 0f;
            _overtimePickupGraceRemaining = 0f;
            _overtimeTrailingTeam = 0;
            bestProgressTeamA = 0f;
            bestProgressTeamB = 0f;
            _prevPushTeamA = 0f;
            _prevPushTeamB = 0f;
            _activeCarryAgent = null;
            _activeCarryBestPot = 0f;
            _radioLastPacket.Clear();
            _radioLastEmitTick.Clear();

            turfGrid?.ResetTurf();
            projectileManager?.ClearAll();
            audioSystem?.Clear();
            blitzCore?.ResetToCenter();
            for (int i = 0; i < deflectPickups.Count; i++)
            {
                if (deflectPickups[i] != null)
                    deflectPickups[i].ResetPickup();
            }

            for (int i = 0; i < agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = agents[i];
                if (ag == null) continue;
                if (ag.isSupportTrooperBot)
                    ag.gameObject.SetActive(settings.enableSupportTrooperBots || ag.isPromotedPeerBrain);
                if (!ag.gameObject.activeSelf)
                    continue;
                ag.ResetForNewMatch();
                if (_trooperBots.TryGetValue(ag, out var tr) && tr != null)
                {
                    tr.enableTrooperAi = !ag.isPromotedPeerBrain;
                    tr.ResetTrooperState();
                }
                ag.brainBridge?.NotifyNewEpisodeReset();
            }
        }

        public void ApplyTeamStepReward(int teamId, float reward) => AddTeamReward(teamId, reward);

        private void AddTeamReward(int teamId, float reward)
        {
            if (Mathf.Abs(reward) < 1e-6f) return;
            for (int i = 0; i < agents.Count; i++)
            {
                if (agents[i] != null && agents[i].teamId == teamId)
                    agents[i].brainBridge?.AddStepReward(reward);
            }
        }

        private void LogHeartbeat()
        {
            InkBlitzPlayerAgent primary = (agents.Count > 0) ? agents[0] : null;
            if (primary == null) return;

            bool conn = primary.brainBridge != null && primary.brainBridge.IsConnected;
            int verb = primary.brainBridge != null ? primary.brainBridge.latestMotorVerb : primary.currentVerb;
            string speech = primary.brainBridge != null ? primary.brainBridge.latestInnerSpeech : "";
            Vector3 loc = primary.transform.position - transform.position;
            string form = primary.isRespawning ? "RESPAWN" : (primary.isSubmergedInFriendlyInk ? "DIVE_SUB" : (primary.isDiveForm ? "DIVE" : "HUMAN"));
            string rmState = blitzCore != null ? $"{blitzCore.State}(sh:{blitzCore.ShieldHp:F0})" : "-";

            Debug.Log($"[InkBlitzArena] tick={_tickCount} connected={conn} pos=({loc.x:F2},{loc.y:F2},{loc.z:F2}) " +
                      $"state={form} hp={primary.hp:F0} ink={primary.InkTank01 * 100f:F0}% " +
                      $"turfA={turfGrid.TeamARatio01 * 100f:F2}% turfB={turfGrid.TeamBRatio01 * 100f:F2}% " +
                      $"rm={rmState} verb={verb} speech=\"{speech}\"");
        }

        public void ApplyColorPalette(int paletteIndex)
        {
            if (settings == null) settings = new InkBlitzArenaSettings();
            InkColorPalettePreset preset = InkBlitzInkPalettes.Get(paletteIndex);
            settings.teamAColor = preset.colorA;
            settings.teamBColor = preset.colorB;

            Material seedMat = (stageBuilder != null && stageBuilder.teamAPedestalMaterial != null)
                ? stageBuilder.teamAPedestalMaterial
                : (turfGrid != null ? turfGrid.templateFloorMaterial : null);

            EnsureRuntimeMatteMat(ref _runtimeMatTeamA, $"RuntimeTeamA_{arenaIndex}", seedMat, preset.colorA);
            EnsureRuntimeMatteMat(ref _runtimeMatTeamB, $"RuntimeTeamB_{arenaIndex}", seedMat, preset.colorB);
            Color trColA = Color.Lerp(preset.colorA, new Color(0.10f, 0.12f, 0.16f, 1f), 0.28f);
            Color trColB = Color.Lerp(preset.colorB, new Color(0.12f, 0.10f, 0.12f, 1f), 0.28f);
            EnsureRuntimeMatteMat(ref _runtimeMatTrooperA, $"RuntimeTrooperA_{arenaIndex}", seedMat, trColA);
            EnsureRuntimeMatteMat(ref _runtimeMatTrooperB, $"RuntimeTrooperB_{arenaIndex}", seedMat, trColB);

            // 1. Goal Pedestals
            Transform landmarks = transform.Find("Landmarks");
            if (landmarks != null)
            {
                AssignPedestalMaterial(landmarks.Find("Pedestal_TeamA"), _runtimeMatTeamA);
                AssignPedestalMaterial(landmarks.Find("Pedestal_TeamB"), _runtimeMatTeamB);
            }

            // 2. Main Agents & Support Troopers (Body / TrooperBody + DiveDisc)
            for (int i = 0; i < agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = agents[i];
                if (ag == null) continue;
                Material targetMat = ag.isSupportTrooperBot
                    ? (ag.teamId > 0 ? _runtimeMatTrooperA : _runtimeMatTrooperB)
                    : (ag.teamId > 0 ? _runtimeMatTeamA : _runtimeMatTeamB);
                AssignAgentBodyMaterial(ag, targetMat);
            }

            // 3. Projectiles & Turf Floor Texture (instant full-res 576x768 recolor!)
            if (projectileManager != null)
                projectileManager.RefreshPaletteColors(preset.colorA, preset.colorB);
            if (turfGrid != null)
                turfGrid.RefreshPaletteColors();
        }

        private static void EnsureRuntimeMatteMat(ref Material slot, string matName, Material templateMat, Color col)
        {
            if (slot == null)
            {
                if (templateMat != null)
                {
                    slot = new Material(templateMat) { name = matName };
                    if (slot.HasProperty("_BaseMap")) slot.SetTexture("_BaseMap", null);
                    if (slot.HasProperty("_MainTex")) slot.SetTexture("_MainTex", null);
                }
                else
                {
                    Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                    slot = new Material(sh) { name = matName };
                }
            }
            InkBlitzInkPalettes.ApplyMatteColor(slot, col);
        }

        private static void AssignPedestalMaterial(Transform pedRoot, Material mat)
        {
            if (pedRoot == null || mat == null) return;
            MeshRenderer[] mrs = pedRoot.GetComponentsInChildren<MeshRenderer>(true);
            for (int i = 0; i < mrs.Length; i++)
            {
                if (mrs[i] != null)
                    mrs[i].sharedMaterial = mat;
            }
        }

        private static void AssignAgentBodyMaterial(InkBlitzPlayerAgent ag, Material mat)
        {
            if (ag == null || mat == null) return;
            if (ag.humanVisualRoot != null)
            {
                Transform body = ag.humanVisualRoot.Find("Body") ?? ag.humanVisualRoot.Find("TrooperBody");
                if (body != null)
                {
                    var mr = body.GetComponent<MeshRenderer>();
                    if (mr != null) mr.sharedMaterial = mat;
                }
            }
            if (ag.diveVisualRoot != null)
            {
                Transform sq = ag.diveVisualRoot.Find("DiveDisc");
                if (sq != null)
                {
                    var mr = sq.GetComponent<MeshRenderer>();
                    if (mr != null) mr.sharedMaterial = mat;
                }
            }
        }

        private void OnDestroy()
        {
            DestroyRuntimeMat(ref _runtimeMatTeamA);
            DestroyRuntimeMat(ref _runtimeMatTeamB);
            DestroyRuntimeMat(ref _runtimeMatTrooperA);
            DestroyRuntimeMat(ref _runtimeMatTrooperB);
        }

        private static void DestroyRuntimeMat(ref Material m)
        {
            if (m == null) return;
            if (Application.isPlaying) Destroy(m);
            else DestroyImmediate(m);
            m = null;
        }
    }
}
