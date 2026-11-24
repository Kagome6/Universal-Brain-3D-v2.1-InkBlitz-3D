// InkBlitzPhase1Builder.cs — Deterministic Unity 6 Editor builder for 1v1 BlitzCore Arena Scene & Prefab
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UniversalBrain3D;

namespace InkBlitz3D.Editor
{
    public static class InkBlitzPhase1Builder
    {
        private const string MaterialsDir = "Assets/Materials";
        private const string PrefabsDir = "Assets/Prefabs";
        private const string ScenesDir = "Assets/Scenes";

        [MenuItem("InkBlitzLab/Setup Phase 1 Arena Scene")]
        public static void BuildPhase1Scene()
        {
            Build1v1BlitzCoreScene();
        }

        [MenuItem("InkBlitzLab/Setup 1v1 BlitzCore Arena Scene")]
        public static void Build1v1BlitzCoreScene()
        {
            EnsureDirectory(MaterialsDir);
            EnsureDirectory(PrefabsDir);
            EnsureDirectory(ScenesDir);

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // 1. Directional Light, Trilight Ambient & Overview Camera
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.42f, 0.48f, 0.58f);
            RenderSettings.ambientEquatorColor = new Color(0.32f, 0.35f, 0.41f);
            RenderSettings.ambientGroundColor = new Color(0.20f, 0.22f, 0.26f);

            var lightGo = new GameObject("Directional Light");
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.48f;
            light.color = new Color(1.0f, 0.97f, 0.91f);
            light.shadows = LightShadows.Soft;
            lightGo.transform.rotation = Quaternion.Euler(50f, -34f, 0f);

            var camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.14f, 0.17f, 0.22f, 1f);
            cam.fieldOfView = 55f;
            camGo.AddComponent<AudioListener>();
            camGo.transform.position = new Vector3(0f, 22.0f, -22.8f);
            camGo.transform.rotation = Quaternion.Euler(48f, 0f, 0f);

            // 2. Persistent Material Assets so Prefabs & Standalone Builds never strip shader variants
            var settings = new InkBlitzArenaSettings();
            Texture2D whiteBaseTex = EnsureWhiteTextureAsset("WhiteBaseMap.asset");
            Material floorPreviewMat = EnsureLitMaterial("ArenaFloorPreviewMat.mat", settings.neutralFloorColor, 0.0f);
            if (floorPreviewMat.HasProperty("_BaseMap")) floorPreviewMat.SetTexture("_BaseMap", whiteBaseTex);
            if (floorPreviewMat.HasProperty("_MainTex")) floorPreviewMat.SetTexture("_MainTex", whiteBaseTex);
            EditorUtility.SetDirty(floorPreviewMat);

            Material wallMat = EnsureLitMaterial("ArenaWallMat.mat", new Color(0.15f, 0.16f, 0.19f), 0.0f);
            Material coverMat = EnsureLitMaterial("ArenaCoverMat.mat", new Color(0.50f, 0.53f, 0.58f), 0.0f);
            Material pedAMat = EnsureLitMaterial("PedestalTeamAMat.mat", settings.teamAColor, 0.0f);
            Material pedBMat = EnsureLitMaterial("PedestalTeamBMat.mat", settings.teamBColor, 0.0f);
            Material objMat = EnsureLitMaterial("PrimaryObjectiveMat.mat", new Color(0.95f, 0.80f, 0.24f), 0.15f);
            Material shieldMat = EnsureTransparentLitMaterial("BlitzCoreShieldMat.mat", new Color(0.96f, 0.84f, 0.34f, 0.28f), 0.15f);
            Material deflectOrbCoreMat = EnsureLitMaterial("DeflectOrbCoreMat.mat", new Color(0.38f, 0.86f, 0.78f), 0.20f);
            Material deflectOrbHaloMat = EnsureTransparentLitMaterial("DeflectOrbHaloMat.mat", new Color(0.42f, 0.88f, 0.80f, 0.28f), 0.20f);
            Material deflectShieldMat = EnsureTransparentLitMaterial("DeflectShieldMat.mat", new Color(0.42f, 0.88f, 0.80f, 0.36f), 0.20f);
            Material teamABodyMat = EnsureLitMaterial("AgentTeamAMat.mat", settings.teamAColor, 0.0f);
            Material teamBBodyMat = EnsureLitMaterial("AgentTeamBMat.mat", settings.teamBColor, 0.0f);
            Material trooperAMat = EnsureLitMaterial("TrooperTeamAMat.mat", new Color(0.36f, 0.46f, 0.72f), 0.0f);
            Material trooperBMat = EnsureLitMaterial("TrooperTeamBMat.mat", new Color(0.67f, 0.59f, 0.45f), 0.0f);
            Material trooperGogglesMat = EnsureLitMaterial("TrooperGogglesMat.mat", new Color(0.92f, 0.78f, 0.32f), 0.10f);
            Material visorMat = EnsureLitMaterial("AgentVisorMat.mat", new Color(0.92f, 0.94f, 0.97f), 0.10f);
            Material nozzleMat = EnsureLitMaterial("AgentWeaponMat.mat", new Color(0.14f, 0.15f, 0.18f), 0.0f);

            // 3. Build Arena Root, Subsystems & Stage
            var arenaGo = new GameObject("InkBlitzArena_0");
            var turfGrid = arenaGo.AddComponent<TurfGridSystem>();
            turfGrid.templateFloorMaterial = floorPreviewMat;
            var audioSys = arenaGo.AddComponent<ArenaAudioSystem>();
            var projMgr = arenaGo.AddComponent<InkProjectileManager>();
            projMgr.templateOpaqueMaterial = pedAMat;

            var stageBuilder = arenaGo.AddComponent<ArenaStageBuilder>();
            stageBuilder.defaultFloorMaterial = floorPreviewMat;
            stageBuilder.wallMaterial = wallMat;
            stageBuilder.coverMaterial = coverMat;
            stageBuilder.teamAPedestalMaterial = pedAMat;
            stageBuilder.teamBPedestalMaterial = pedBMat;
            stageBuilder.objectiveMaterial = objMat;
            stageBuilder.objectiveShieldMaterial = shieldMat;
            stageBuilder.deflectOrbCoreMaterial = deflectOrbCoreMat;
            stageBuilder.deflectOrbHaloMaterial = deflectOrbHaloMat;
            stageBuilder.sharedTerrainMeshAsset = EnsureTerrainManifoldMeshAsset("ArenaTerrainManifold_ArenaPlanarUV.asset", settings);
            stageBuilder.sharedPedestalBevelMeshAsset = EnsurePedestalBevelMeshAsset("GoalPedestalBevelMesh.asset");
            stageBuilder.BuildStage(0, settings);

            var arenaInstance = arenaGo.AddComponent<InkBlitzArenaInstance>();
            arenaInstance.arenaIndex = 0;
            arenaInstance.settings = settings;
            arenaInstance.turfGrid = turfGrid;
            arenaInstance.stageBuilder = stageBuilder;
            arenaInstance.audioSystem = audioSys;
            arenaInstance.projectileManager = projMgr;
            arenaInstance.blitzCore = arenaGo.GetComponentInChildren<BlitzCoreObjective>(true);
            arenaGo.GetComponentsInChildren(true, arenaInstance.deflectPickups);
            arenaInstance.logPeriodicTelemetry = true;
            arenaInstance.telemetryIntervalTicks = 120;

            // 4. Spawn 180-deg Symmetric 1v1 Agents (Team A = Brain, Team B = Bot / Phase 4 Self-Play)
            InkBlitzPlayerAgent teamAAgent = CreateAgent(
                arenaGo.transform,
                "Diver_TeamA_0",
                arenaIndex: 0,
                teamId: 1,
                entityId: 1,
                port: 50055,
                localPos: settings.teamASpawnLocal,
                turfGrid: turfGrid,
                bodyMat: teamABodyMat,
                visorMat: visorMat,
                nozzleMat: nozzleMat,
                deflectShieldMat: deflectShieldMat,
                enableBot: true,
                yieldToBrain: true
            );
            InkBlitzPlayerAgent teamBAgent = CreateAgent(
                arenaGo.transform,
                "Diver_TeamB_0",
                arenaIndex: 0,
                teamId: -1,
                entityId: 2,
                port: 50056,
                localPos: settings.teamBSpawnLocal,
                turfGrid: turfGrid,
                bodyMat: teamBBodyMat,
                visorMat: visorMat,
                nozzleMat: nozzleMat,
                deflectShieldMat: deflectShieldMat,
                enableBot: true,
                yieldToBrain: true
            );
            InkBlitzPlayerAgent trooperAAgent = CreateSupportTrooper(
                arenaGo.transform,
                "Trooper_TeamA_0",
                arenaIndex: 0,
                teamId: 1,
                entityId: 3,
                localPos: settings.teamASupportSpawnLocal,
                bodyMat: trooperAMat,
                gogglesMat: trooperGogglesMat,
                nozzleMat: nozzleMat
            );
            InkBlitzPlayerAgent trooperBAgent = CreateSupportTrooper(
                arenaGo.transform,
                "Trooper_TeamB_0",
                arenaIndex: 0,
                teamId: -1,
                entityId: 4,
                localPos: settings.teamBSupportSpawnLocal,
                bodyMat: trooperBMat,
                gogglesMat: trooperGogglesMat,
                nozzleMat: nozzleMat
            );
            arenaInstance.agents.Add(teamAAgent);
            arenaInstance.agents.Add(teamBAgent);
            arenaInstance.agents.Add(trooperAAgent);
            arenaInstance.agents.Add(trooperBAgent);

            // 5. Save Reusable Arena Prefab
            string prefabPath = $"{PrefabsDir}/InkBlitzArenaPrefab.prefab";
            GameObject prefabAsset = PrefabUtility.SaveAsPrefabAssetAndConnect(arenaGo, prefabPath, InteractionMode.AutomatedAction);

            // 6. BrainSessionCoordinator & Telemetry HUD
            var coordGo = new GameObject("BrainSessionCoordinator");
            var coordinator = coordGo.AddComponent<UniversalArenaCoordinator3D>();
            coordinator.sessionMode = BrainSessionRunMode.SinglePlaySpectate;
            coordinator.autoLaunchOnStart = true;
            coordinator.singlePlayPolicyMode = "dagger";
            coordinator.taskProfile = "turf_objective";
            coordinator.baseTcpPort = 50055;
            coordinator.arenaSpacingMeters = 100.0f;
            coordinator.arenaPrefab = prefabAsset;
            coordinator.RegisterBridge(teamAAgent.brainBridge);

            var hud = coordGo.AddComponent<InkBlitzTelemetryHud>();
            hud.arena = arenaInstance;
            hud.coordinator = coordinator;

            var multiMgr = coordGo.AddComponent<InkBlitzMultiArenaManager>();
            multiMgr.coordinator = coordinator;
            multiMgr.primaryArena = arenaInstance;
            multiMgr.hud = hud;
            multiMgr.mainCamera = cam;
            hud.multiManager = multiMgr;

            EnsureDashboardUi(coordinator, multiMgr, showLauncherOnStart: true);

            // 7. Save Scene, Register in Build Settings & Default to Interactive Launcher Mode
            string scenePath = $"{ScenesDir}/InkBlitzArenaScene.unity";
            EditorSceneManager.SaveScene(scene, scenePath);
            RegisterSceneInBuildSettings(scenePath);
            ConfigureInteractiveLauncherMode();

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"[InkBlitzPhase1Builder] Built 1v1 BlitzCore {scenePath} and {prefabPath} successfully.");
        }

        private static InkBlitzSessionDashboardUi EnsureDashboardUi(
            UniversalArenaCoordinator3D coordinator,
            InkBlitzMultiArenaManager multiMgr,
            bool showLauncherOnStart)
        {
            if (coordinator == null) return null;
            var dash = coordinator.GetComponent<InkBlitzSessionDashboardUi>();
            if (dash == null)
                dash = coordinator.gameObject.AddComponent<InkBlitzSessionDashboardUi>();
            var neuralHud = coordinator.GetComponent<InkBlitzNeuralVisualizerHud>();
            if (neuralHud == null)
                neuralHud = coordinator.gameObject.AddComponent<InkBlitzNeuralVisualizerHud>();
            var winEst = coordinator.GetComponent<InkBlitzWinRateEstimator>();
            if (winEst == null)
                winEst = coordinator.gameObject.AddComponent<InkBlitzWinRateEstimator>();
            var voiceCoach = coordinator.GetComponent<InkBlitzCompanionVoiceCoach>();
            if (voiceCoach == null)
                voiceCoach = coordinator.gameObject.AddComponent<InkBlitzCompanionVoiceCoach>();
            var circuitHud = coordinator.GetComponent<InkBlitzBrainCircuitHud>();
            if (circuitHud == null)
                circuitHud = coordinator.gameObject.AddComponent<InkBlitzBrainCircuitHud>();

            dash.coordinator = coordinator;
            dash.multiManager = multiMgr;
            dash.neuralHud = neuralHud;
            dash.winRateEstimator = winEst;
            dash.voiceCoach = voiceCoach;
            dash.showLauncherOnStart = showLauncherOnStart;

            neuralHud.coordinator = coordinator;
            neuralHud.multiManager = multiMgr;
            neuralHud.dashboardUi = dash;
            neuralHud.circuitHud = circuitHud;

            circuitHud.coordinator = coordinator;
            circuitHud.dashboardUi = dash;
            circuitHud.neuralHud = neuralHud;

            winEst.coordinator = coordinator;
            winEst.multiManager = multiMgr;
            winEst.dashboardUi = dash;
            winEst.neuralHud = neuralHud;

            voiceCoach.coordinator = coordinator;

            EditorUtility.SetDirty(dash);
            EditorUtility.SetDirty(neuralHud);
            EditorUtility.SetDirty(circuitHud);
            EditorUtility.SetDirty(winEst);
            EditorUtility.SetDirty(voiceCoach);
            return dash;
        }

        [MenuItem("InkBlitzLab/Configure Scene: Interactive Launcher (Default)")]
        public static void ConfigureInteractiveLauncherMode()
        {
            var coordinator = Object.FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (coordinator == null)
            {
                Debug.LogWarning("[InkBlitzPhase1Builder] UniversalArenaCoordinator3D not found in active scene.");
                return;
            }
            coordinator.sessionMode = BrainSessionRunMode.SinglePlaySpectate;
            coordinator.autoLaunchOnStart = false;
            coordinator.parallelArenaCount = 4;
            coordinator.trainingTimeScale = 4.0f;
            coordinator.pbtGenerations = 8;
            coordinator.pbtEpisodesPerGen = 2;
            coordinator.pbtPolicyMode = "dagger";
            coordinator.resumeFromCheckpointInPbt = true;
            coordinator.pbtCheckpointDir = @"checkpoints\user_runs";
            coordinator.singlePlayPolicyMode = "pure_neural";
            coordinator.loadCheckpointRelativePath = @"checkpoints\user_runs\pbt_latest_champion.pt";
            coordinator.oppCheckpointRelativePath = "";
            coordinator.maxTrainingStepsPerRenderFrame = 2;
            EditorUtility.SetDirty(coordinator);

            var multiMgr = Object.FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (multiMgr != null)
            {
                multiMgr.overviewMode = false;
                multiMgr.renderRepresentativeArenaOnly = true;
                multiMgr.disableAll3DRendering = false;
                multiMgr.disableShadowsInTraining = true;
                EditorUtility.SetDirty(multiMgr);
            }

            EnsureDashboardUi(coordinator, multiMgr, showLauncherOnStart: true);
            EditorSceneManager.SaveOpenScenes();
            Debug.Log("[InkBlitzPhase1Builder] Configured scene for Interactive Launcher Mode (Menu on Start).");
        }

        [MenuItem("InkBlitzLab/Configure Scene: 1v1 Spectate Mode")]
        public static void ConfigureSpectateMode()
        {
            var coordinator = Object.FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (coordinator == null)
            {
                Debug.LogWarning("[InkBlitzPhase1Builder] UniversalArenaCoordinator3D not found in active scene.");
                return;
            }
            coordinator.sessionMode = BrainSessionRunMode.SinglePlaySpectate;
            coordinator.autoLaunchOnStart = true;
            coordinator.singlePlayPolicyMode = "pure_neural";
            coordinator.loadCheckpointRelativePath = @"checkpoints\user_runs\pbt_latest_champion.pt";
            coordinator.trainingTimeScale = 1.0f;
            coordinator.pbtGenerations = 12;
            coordinator.pbtEpisodesPerGen = 2;
            coordinator.pbtCheckpointDir = @"checkpoints\user_runs";
            EditorUtility.SetDirty(coordinator);

            var multiMgr = Object.FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (multiMgr != null)
            {
                multiMgr.overviewMode = false;
                multiMgr.disableAll3DRendering = false;
                EditorUtility.SetDirty(multiMgr);
            }

            EnsureDashboardUi(coordinator, multiMgr, showLauncherOnStart: false);
            EditorSceneManager.SaveOpenScenes();
            Debug.Log("[InkBlitzPhase1Builder] Configured scene for 1v1 Spectate Mode (SinglePlaySpectate, PureNeural).");
        }

        [MenuItem("InkBlitzLab/Configure Scene: Continuous Spectate (AI vs AI)")]
        public static void ConfigureContinuousSelfPlaySpectateMode()
        {
            var coordinator = Object.FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (coordinator == null)
            {
                Debug.LogWarning("[InkBlitzPhase1Builder] UniversalArenaCoordinator3D not found in active scene.");
                return;
            }
            coordinator.sessionMode = BrainSessionRunMode.SingleArenaSelfPlaySpectate;
            coordinator.autoLaunchOnStart = true;
            coordinator.singlePlayPolicyMode = "pure_neural";
            coordinator.loadCheckpointRelativePath = @"checkpoints\user_runs\pbt_latest_champion.pt";
            coordinator.oppCheckpointRelativePath = "";
            coordinator.trainingTimeScale = 1.0f;
            EditorUtility.SetDirty(coordinator);

            var multiMgr = Object.FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (multiMgr != null)
            {
                multiMgr.overviewMode = false;
                multiMgr.disableAll3DRendering = false;
                EditorUtility.SetDirty(multiMgr);
            }

            EnsureDashboardUi(coordinator, multiMgr, showLauncherOnStart: false);
            EditorSceneManager.SaveOpenScenes();
            Debug.Log("[InkBlitzPhase1Builder] Configured scene for Continuous Spectate (AI vs AI Endless PureInference).");
        }

        [MenuItem("InkBlitzLab/Configure Scene: Phase 4 Self-Play PBT (4 Arenas)")]
        public static void ConfigureSelfPlayPbtMode()
        {
            var coordinator = Object.FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (coordinator == null)
            {
                Debug.LogWarning("[InkBlitzPhase1Builder] UniversalArenaCoordinator3D not found in active scene.");
                return;
            }
            coordinator.sessionMode = BrainSessionRunMode.ParallelCompetitiveSelfPlay;
            coordinator.autoLaunchOnStart = true;
            coordinator.parallelArenaCount = 4;
            coordinator.trainingTimeScale = 4.0f;
            coordinator.pbtGenerations = 4;
            coordinator.pbtEpisodesPerGen = 1;
            coordinator.pbtPolicyMode = "dagger";
            coordinator.resumeFromCheckpointInPbt = false;
            coordinator.pbtCheckpointDir = @"checkpoints\user_runs";
            coordinator.maxTrainingStepsPerRenderFrame = 2;
            EditorUtility.SetDirty(coordinator);

            var multiMgr = Object.FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (multiMgr != null)
            {
                multiMgr.overviewMode = false;
                multiMgr.renderRepresentativeArenaOnly = true;
                multiMgr.disableAll3DRendering = false;
                multiMgr.disableShadowsInTraining = true;
                EditorUtility.SetDirty(multiMgr);
            }

            EnsureDashboardUi(coordinator, multiMgr, showLauncherOnStart: false);
            EditorSceneManager.SaveOpenScenes();
            Debug.Log("[InkBlitzPhase1Builder] Configured scene for Phase 4 Self-Play PBT (4 Arenas, 4 Gen x 1 Ep, 4.0x Speed, Rep Arena Only).");
        }

        [MenuItem("InkBlitzLab/Configure Scene: Phase 4 Smoke Test (Clean Checkpoints)")]
        public static void ConfigureSelfPlayPbtSmokeTest()
        {
            var coordinator = Object.FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (coordinator == null)
            {
                Debug.LogWarning("[InkBlitzPhase1Builder] UniversalArenaCoordinator3D not found in active scene.");
                return;
            }
            coordinator.sessionMode = BrainSessionRunMode.ParallelCompetitiveSelfPlay;
            coordinator.autoLaunchOnStart = true;
            coordinator.parallelArenaCount = 4;
            coordinator.trainingTimeScale = 4.0f;
            coordinator.pbtGenerations = 4;
            coordinator.pbtEpisodesPerGen = 1;
            coordinator.pbtPolicyMode = "dagger";
            coordinator.resumeFromCheckpointInPbt = false;
            coordinator.pbtCheckpointDir = "";
            coordinator.maxTrainingStepsPerRenderFrame = 2;
            EditorUtility.SetDirty(coordinator);

            var multiMgr = Object.FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (multiMgr != null)
            {
                multiMgr.overviewMode = false;
                multiMgr.renderRepresentativeArenaOnly = true;
                multiMgr.disableAll3DRendering = false;
                multiMgr.disableShadowsInTraining = true;
                EditorUtility.SetDirty(multiMgr);
            }

            EnsureDashboardUi(coordinator, multiMgr, showLauncherOnStart: false);
            EditorSceneManager.SaveOpenScenes();
            Debug.Log("[InkBlitzPhase1Builder] Configured scene for Phase 4 Smoke Test (4 Arenas, 4.0x, Clean Checkpoints, Rep Arena Only).");
        }

        private static InkBlitzPlayerAgent CreateAgent(
            Transform arenaRoot,
            string agentName,
            int arenaIndex,
            int teamId,
            int entityId,
            int port,
            Vector3 localPos,
            TurfGridSystem turfGrid,
            Material bodyMat,
            Material visorMat,
            Material nozzleMat,
            Material deflectShieldMat,
            bool enableBot,
            bool yieldToBrain)
        {
            var agentGo = new GameObject(agentName);
            agentGo.transform.SetParent(arenaRoot, false);
            agentGo.transform.localPosition = localPos;
            float faceCenterYawDeg = Mathf.Atan2(-localPos.x, -localPos.z) * Mathf.Rad2Deg;
            agentGo.transform.localRotation = Quaternion.Euler(0f, faceCenterYawDeg, 0f);

            var cc = agentGo.AddComponent<CharacterController>();
            cc.height = 1.5f;
            cc.radius = 0.42f;
            cc.center = new Vector3(0f, 0.75f, 0f);
            cc.slopeLimit = 45f;
            cc.stepOffset = 0.35f;

            var tag = agentGo.AddComponent<UniversalSemanticEntityTag>();
            tag.entityId = entityId;
            tag.arenaIndex = arenaIndex;
            tag.teamId = teamId;
            tag.baseKind = UniversalEntityKind.OpponentAgent;
            tag.boundingRadius = 0.65f;

            var eyePivot = new GameObject("EyePivot");
            eyePivot.transform.SetParent(agentGo.transform, false);
            eyePivot.transform.localPosition = new Vector3(0f, 1.18f, 0f);

            var bridge = agentGo.AddComponent<UniversalBrainBridge3D>();
            bridge.arenaIndex = arenaIndex;
            bridge.teamId = teamId;
            bridge.port = port;
            bridge.eyeTransform = eyePivot.transform;
            bridge.floorSurfaceSamplerComponent = turfGrid;

            var humanRoot = new GameObject("HumanVisualRoot");
            humanRoot.transform.SetParent(agentGo.transform, false);

            var body = CreateVisualPrimitive(PrimitiveType.Capsule, "Body", humanRoot.transform,
                new Vector3(0f, 0.78f, 0f), new Vector3(0.68f, 0.72f, 0.68f), bodyMat);
            CreateVisualPrimitive(PrimitiveType.Cube, "Visor", body.transform,
                new Vector3(0f, 0.42f, 0.36f), new Vector3(0.66f, 0.24f, 0.35f), visorMat);

            var weaponPivot = new GameObject("WeaponNozzlePivot");
            weaponPivot.transform.SetParent(humanRoot.transform, false);
            weaponPivot.transform.localPosition = new Vector3(0.28f, 0.92f, 0.15f);
            CreateVisualPrimitive(PrimitiveType.Cube, "WeaponBarrel", weaponPivot.transform,
                new Vector3(0f, 0f, 0.38f), new Vector3(0.18f, 0.18f, 0.85f), nozzleMat);

            var diveRoot = new GameObject("DiveVisualRoot");
            diveRoot.transform.SetParent(agentGo.transform, false);
            CreateVisualPrimitive(PrimitiveType.Cylinder, "DiveDisc", diveRoot.transform,
                new Vector3(0f, 0.12f, 0f), new Vector3(0.75f, 0.08f, 1.05f), bodyMat);
            diveRoot.SetActive(false);

            var agent = agentGo.AddComponent<InkBlitzPlayerAgent>();
            agent.arenaIndex = arenaIndex;
            agent.teamId = teamId;
            agent.characterController = cc;
            agent.entityTag = tag;
            agent.brainBridge = bridge;
            agent.eyePivot = eyePivot.transform;
            agent.humanVisualRoot = humanRoot.transform;
            agent.diveVisualRoot = diveRoot.transform;
            agent.weaponNozzlePivot = weaponPivot.transform;
            agent.templateDeflectShieldMaterial = deflectShieldMat;
            agent.EnsureDeflectShieldVisual();

            var bot = agentGo.AddComponent<HeuristicSparringBot>();
            bot.agent = agent;
            bot.enableHeuristicBot = enableBot;
            bot.yieldWhenBrainConnected = yieldToBrain;

            return agent;
        }

        private static InkBlitzPlayerAgent CreateSupportTrooper(
            Transform arenaRoot,
            string agentName,
            int arenaIndex,
            int teamId,
            int entityId,
            Vector3 localPos,
            Material bodyMat,
            Material gogglesMat,
            Material nozzleMat)
        {
            var agentGo = new GameObject(agentName);
            agentGo.transform.SetParent(arenaRoot, false);
            agentGo.transform.localPosition = localPos;
            float faceCenterYawDeg = Mathf.Atan2(-localPos.x, -localPos.z) * Mathf.Rad2Deg;
            agentGo.transform.localRotation = Quaternion.Euler(0f, faceCenterYawDeg, 0f);

            var cc = agentGo.AddComponent<CharacterController>();
            cc.height = 1.45f;
            cc.radius = 0.40f;
            cc.center = new Vector3(0f, 0.72f, 0f);
            cc.slopeLimit = 45f;
            cc.stepOffset = 0.35f;

            var tag = agentGo.AddComponent<UniversalSemanticEntityTag>();
            tag.entityId = entityId;
            tag.arenaIndex = arenaIndex;
            tag.teamId = teamId;
            tag.baseKind = UniversalEntityKind.OpponentAgent;
            tag.boundingRadius = 0.62f;

            var eyePivot = new GameObject("EyePivot");
            eyePivot.transform.SetParent(agentGo.transform, false);
            eyePivot.transform.localPosition = new Vector3(0f, 1.12f, 0f);

            var humanRoot = new GameObject("HumanVisualRoot");
            humanRoot.transform.SetParent(agentGo.transform, false);

            var body = CreateVisualPrimitive(PrimitiveType.Capsule, "TrooperBody", humanRoot.transform,
                new Vector3(0f, 0.72f, 0f), new Vector3(0.66f, 0.66f, 0.66f), bodyMat);
            CreateVisualPrimitive(PrimitiveType.Cube, "TacticalGoggles", body.transform,
                new Vector3(0f, 0.40f, 0.38f), new Vector3(0.76f, 0.22f, 0.38f), gogglesMat);
            CreateVisualPrimitive(PrimitiveType.Cylinder, "TrooperHelmetCrest", body.transform,
                new Vector3(0f, 0.88f, -0.04f), new Vector3(0.32f, 0.18f, 0.48f), gogglesMat);

            var weaponPivot = new GameObject("WeaponNozzlePivot");
            weaponPivot.transform.SetParent(humanRoot.transform, false);
            weaponPivot.transform.localPosition = new Vector3(0.26f, 0.86f, 0.14f);
            CreateVisualPrimitive(PrimitiveType.Cube, "TrooperWeaponBarrel", weaponPivot.transform,
                new Vector3(0f, 0f, 0.34f), new Vector3(0.16f, 0.16f, 0.76f), nozzleMat);

            var diveRoot = new GameObject("DiveVisualRoot");
            diveRoot.transform.SetParent(agentGo.transform, false);
            CreateVisualPrimitive(PrimitiveType.Cylinder, "DiveDisc", diveRoot.transform,
                new Vector3(0f, 0.12f, 0f), new Vector3(0.70f, 0.08f, 0.98f), bodyMat);
            diveRoot.SetActive(false);

            var agent = agentGo.AddComponent<InkBlitzPlayerAgent>();
            agent.arenaIndex = arenaIndex;
            agent.teamId = teamId;
            agent.isSupportTrooperBot = true;
            agent.characterController = cc;
            agent.entityTag = tag;
            agent.brainBridge = null; // Support bots never consume TCP brain slots!
            agent.eyePivot = eyePivot.transform;
            agent.humanVisualRoot = humanRoot.transform;
            agent.diveVisualRoot = diveRoot.transform;
            agent.weaponNozzlePivot = weaponPivot.transform;

            var trooperAi = agentGo.AddComponent<SupportTrooperBot>();
            trooperAi.agent = agent;
            trooperAi.enableTrooperAi = true;

            return agent;
        }

        private static GameObject CreateVisualPrimitive(PrimitiveType type, string name, Transform parent,
            Vector3 localPos, Vector3 localScale, Material mat)
        {
            GameObject go = GameObject.CreatePrimitive(type);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = localScale;

            Collider col = go.GetComponent<Collider>();
            if (col != null)
                Object.DestroyImmediate(col);

            if (mat != null)
                go.GetComponent<MeshRenderer>().sharedMaterial = mat;
            return go;
        }

        [MenuItem("InkBlitzLab/Prepare Build Settings (Windows 64-bit)")]
        public static void PrepareStandaloneBuildSettings()
        {
            RegisterSceneInBuildSettings($"{ScenesDir}/InkBlitzArenaScene.unity");
            PlayerSettings.productName = "InkBlitz 3D";
            PlayerSettings.companyName = "Alpha496";
            PlayerSettings.bundleVersion = "2.3.1";
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.allowFullscreenSwitch = true;
            PlayerSettings.runInBackground = true;
            AssetDatabase.SaveAssets();
            Debug.Log("[InkBlitzPhase1Builder] Prepared Build Settings: Scene[0]=Assets/Scenes/InkBlitzArenaScene.unity (1280x720 Windowed).");
        }

        [MenuItem("InkBlitzLab/Build Standalone Game (Windows 64-bit)")]
        public static void BuildStandaloneWindows64()
        {
            ConfigureInteractiveLauncherMode();
            PrepareStandaloneBuildSettings();

            string exePath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "Build", "InkBlitz3D", "InkBlitz3D.exe"));
            string buildDir = Path.GetDirectoryName(exePath);
            if (!string.IsNullOrEmpty(buildDir))
                Directory.CreateDirectory(buildDir);

            var options = new BuildPlayerOptions
            {
                scenes = new[] { $"{ScenesDir}/InkBlitzArenaScene.unity" },
                locationPathName = exePath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result == BuildResult.Succeeded)
                Debug.Log($"[InkBlitzPhase1Builder] Standalone build SUCCEEDED: {exePath} ({report.summary.totalSize / (1024f * 1024f):F1} MB)");
            else
                Debug.LogError($"[InkBlitzPhase1Builder] Standalone build FAILED ({report.summary.result}): {report.summary.totalErrors} errors");
        }

        public static void VerifySceneAndSelfTestOnlyBatch()
        {
            Build1v1BlitzCoreScene();
            ConfigureInteractiveLauncherMode();
            if (!InkBlitzMechanicsSelfTest.RunMechanicsVerification())
            {
                EditorApplication.Exit(1);
                return;
            }
            Debug.Log("[InkBlitzPhase1Builder] Scene sync + All SelfTests SUCCEEDED (Standalone build skipped per request).");
        }

        [MenuItem("InkBlitzLab/Verify Scene, SelfTest & Build Win64 %#&b")]
        public static void VerifyAndBuildStandaloneBatch()
        {
            Build1v1BlitzCoreScene();
            if (!InkBlitzMechanicsSelfTest.RunMechanicsVerification())
            {
                EditorApplication.Exit(1);
                return;
            }
            BuildStandaloneWindows64();
        }

        private static Mesh EnsureTerrainManifoldMeshAsset(string fileName, InkBlitzArenaSettings settings)
        {
            string assetPath = $"{MaterialsDir}/{fileName}";
            Mesh fresh = ArenaStageBuilder.BuildSmoothTerrainMesh(settings);
            Mesh existing = AssetDatabase.LoadAssetAtPath<Mesh>(assetPath);
            if (existing == null)
            {
                AssetDatabase.CreateAsset(fresh, assetPath);
                return fresh;
            }
            existing.Clear();
            existing.name = fresh.name;
            existing.vertices = fresh.vertices;
            existing.normals = fresh.normals;
            existing.uv = fresh.uv;
            existing.triangles = fresh.triangles;
            existing.RecalculateBounds();
            Physics.BakeMesh(existing.GetEntityId(), false);
            Object.DestroyImmediate(fresh);
            EditorUtility.SetDirty(existing);
            return existing;
        }

        private static Mesh EnsurePedestalBevelMeshAsset(string fileName)
        {
            string assetPath = $"{MaterialsDir}/{fileName}";
            Mesh fresh = ArenaStageBuilder.BuildBeveledPedestalMesh();
            Mesh existing = AssetDatabase.LoadAssetAtPath<Mesh>(assetPath);
            if (existing == null)
            {
                AssetDatabase.CreateAsset(fresh, assetPath);
                return fresh;
            }
            existing.Clear();
            existing.name = fresh.name;
            existing.vertices = fresh.vertices;
            existing.normals = fresh.normals;
            existing.triangles = fresh.triangles;
            existing.RecalculateBounds();
            Physics.BakeMesh(existing.GetEntityId(), false);
            Object.DestroyImmediate(fresh);
            EditorUtility.SetDirty(existing);
            return existing;
        }

        private static Texture2D EnsureWhiteTextureAsset(string fileName)
        {
            string assetPath = $"{MaterialsDir}/{fileName}";
            Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(assetPath);
            if (tex == null)
            {
                tex = new Texture2D(4, 4, TextureFormat.RGBA32, false);
                var cols = new Color32[16];
                for (int i = 0; i < cols.Length; i++)
                    cols[i] = new Color32(255, 255, 255, 255);
                tex.SetPixels32(cols);
                tex.Apply();
                AssetDatabase.CreateAsset(tex, assetPath);
            }
            return tex;
        }

        private static Material EnsureTransparentLitMaterial(string fileName, Color color, float smoothness)
        {
            Material mat = EnsureLitMaterial(fileName, color, smoothness);
            Shader urpLit = Shader.Find("Universal Render Pipeline/Lit");
            if (urpLit != null && mat.shader != urpLit)
                mat.shader = urpLit;
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);
            if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", smoothness);
            if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);
            if (mat.HasProperty("_Blend")) mat.SetFloat("_Blend", 0f);
            if (mat.HasProperty("_SrcBlend")) mat.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
            if (mat.HasProperty("_DstBlend")) mat.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
            if (mat.HasProperty("_ZWrite")) mat.SetFloat("_ZWrite", 0f);
            mat.SetOverrideTag("RenderType", "Transparent");
            mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            mat.renderQueue = (int)RenderQueue.Transparent;
            EditorUtility.SetDirty(mat);
            return mat;
        }

        private static Material EnsureLitMaterial(string fileName, Color color, float smoothness)
        {
            string assetPath = $"{MaterialsDir}/{fileName}";
            Shader urpLit = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            Material mat = AssetDatabase.LoadAssetAtPath<Material>(assetPath);
            if (mat == null)
            {
                mat = new Material(urpLit);
                AssetDatabase.CreateAsset(mat, assetPath);
            }
            else if (urpLit != null && mat.shader != urpLit)
            {
                mat.shader = urpLit;
            }
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);
            if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", smoothness);
            if (mat.HasProperty("_Glossiness")) mat.SetFloat("_Glossiness", smoothness);
            bool noGloss = smoothness <= 0.01f;
            if (mat.HasProperty("_SpecularHighlights")) mat.SetFloat("_SpecularHighlights", noGloss ? 0f : 1f);
            if (mat.HasProperty("_EnvironmentReflections")) mat.SetFloat("_EnvironmentReflections", noGloss ? 0f : 1f);
            if (noGloss)
            {
                mat.EnableKeyword("_SPECULARHIGHLIGHTS_OFF");
                mat.EnableKeyword("_ENVIRONMENTREFLECTIONS_OFF");
            }
            else
            {
                mat.DisableKeyword("_SPECULARHIGHLIGHTS_OFF");
                mat.DisableKeyword("_ENVIRONMENTREFLECTIONS_OFF");
            }
            EditorUtility.SetDirty(mat);
            return mat;
        }

        private static void EnsureDirectory(string assetRelativePath)
        {
            if (!AssetDatabase.IsValidFolder(assetRelativePath))
            {
                string parent = Path.GetDirectoryName(assetRelativePath)?.Replace('\\', '/');
                string folder = Path.GetFileName(assetRelativePath);
                if (!string.IsNullOrEmpty(parent) && !string.IsNullOrEmpty(folder))
                    AssetDatabase.CreateFolder(parent, folder);
            }
        }

        private static void RegisterSceneInBuildSettings(string scenePath)
        {
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(scenePath, true) };
        }
    }
}
