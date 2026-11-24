// InkBlitzMultiArenaManager.cs — Simultaneous N-Arena Spawner, Camera Focus & Port Coordinator
using System.Collections.Generic;
using UnityEngine;
using UniversalBrain3D;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace InkBlitz3D
{
    public class InkBlitzMultiArenaManager : MonoBehaviour
    {
        [Header("Coordinator & Scene References")]
        public UniversalArenaCoordinator3D coordinator;
        public InkBlitzArenaInstance primaryArena;
        public InkBlitzTelemetryHud hud;
        public Camera mainCamera;

        [Header("Spawned Parallel Arenas")]
        public List<InkBlitzArenaInstance> arenas = new List<InkBlitzArenaInstance>();
        public int focusedArenaIndex = 0;
        public bool overviewMode = false;
        public bool enableSupportTrooperBots = true;
        public int activePaletteIndex = 0;
        public int activeStageId = 1;

        [Header("Training Render Optimization (RTX 3060 Friendly)")]
        public bool renderRepresentativeArenaOnly = true;
        public bool disableAll3DRendering = false;
        public bool disableShadowsInTraining = true;
        [HideInInspector] public bool isPausedForLauncher = false;
        [HideInInspector] public bool isEmergencyHalted = false;

        [Header("Interactive Spectator Camera (Mouse Orbit / Zoom / Near-Horizontal Tilt)")]
        public bool enableInteractiveCamera = true;
        public float mouseOrbitSensitivity = 0.28f;
        public float mousePitchSensitivity = 0.22f;
        public float mousePanSensitivity = 0.032f;
        public float minPitchDeg = 2.5f;
        public float maxPitchDeg = 86.0f;
        public float minZoomDistance = 3.8f;
        public float maxZoomDistance = 64.0f;
        public float cameraSmoothTime = 0.055f;

        public float SmoothedTicksPerSec { get; private set; } = 0f;
        public float CurrentCameraPitchDeg => _curCamPitch;
        public float CurrentCameraYawDeg => _curCamYaw;
        public float CurrentCameraDistance => _lastEffectiveDist;

        private InkBlitzSessionDashboardUi _dashboardUi;
        private InkBlitzNeuralVisualizerHud _neuralHud;
        private InkBlitzWinRateEstimator _winRateEstimator;
        private float _targetCamYaw = 0f;
        private float _targetCamPitch = 43f;
        private float _targetCamDist = 28.8f;
        private Vector2 _targetPanXZ = Vector2.zero;
        private float _curCamYaw = 0f;
        private float _curCamPitch = 43f;
        private float _curCamDist = 28.8f;
        private Vector2 _curPanXZ = Vector2.zero;
        private float _lastEffectiveDist = 28.8f;
        private bool _orbitDragging = false;
        private bool _panDragging = false;
        private Vector2 _prevMouseScreenPos;

        private bool[] _stepActive;
        private bool[] _stepHoldOnly;
        private int _savedCullingMask = -1;
        private CameraClearFlags _savedClearFlags = CameraClearFlags.Skybox;
        private Color _savedBgColor = new Color(0.10f, 0.12f, 0.15f);
        private bool _cameraStateSaved = false;
        private Light _mainLight;
        private LightShadows _savedLightShadows = LightShadows.Soft;
        private bool _lightStateSaved = false;
        private int _lastSampledTick = 0;
        private float _lastSampledRealtime = 0f;

        private void Awake()
        {
            if (coordinator == null)
                coordinator = Object.FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (hud == null)
                hud = Object.FindAnyObjectByType<InkBlitzTelemetryHud>();
            if (mainCamera == null)
                mainCamera = Camera.main;
            if (primaryArena == null)
                primaryArena = Object.FindAnyObjectByType<InkBlitzArenaInstance>();
            if (_dashboardUi == null)
                _dashboardUi = Object.FindAnyObjectByType<InkBlitzSessionDashboardUi>();
            if (_neuralHud == null)
                _neuralHud = Object.FindAnyObjectByType<InkBlitzNeuralVisualizerHud>();
            if (_winRateEstimator == null)
                _winRateEstimator = Object.FindAnyObjectByType<InkBlitzWinRateEstimator>();

            if (mainCamera != null)
            {
                _savedCullingMask = mainCamera.cullingMask;
                _savedClearFlags = mainCamera.clearFlags;
                _savedBgColor = mainCamera.backgroundColor;
                if (mainCamera.nearClipPlane > 0.20f)
                    mainCamera.nearClipPlane = 0.20f;
                _cameraStateSaved = true;
            }
            _mainLight = Object.FindAnyObjectByType<Light>();
            if (_mainLight != null)
            {
                _savedLightShadows = _mainLight.shadows;
                _lightStateSaved = true;
            }

            if (coordinator == null || primaryArena == null)
                return;

            var dash = _dashboardUi != null ? _dashboardUi : Object.FindAnyObjectByType<InkBlitzSessionDashboardUi>();
            if (dash != null && dash.showLauncherOnStart && !coordinator.autoLaunchOnStart)
                isPausedForLauncher = true;

            if (isPausedForLauncher)
            {
                primaryArena.drivenByMultiManager = true;
                primaryArena.transform.position = Vector3.zero;
                arenas.Clear();
                arenas.Add(primaryArena);
                primaryArena.ApplyColorPalette(activePaletteIndex);
                FocusArena(0, overview: false);
                return;
            }

            ReconfigureArenasForCurrentSession();
        }

        public void ReconfigureArenasForCurrentSession()
        {
            if (coordinator == null || primaryArena == null)
                return;

            coordinator.EnsureBindablePortBlock();

            for (int i = 0; i < arenas.Count; i++)
            {
                if (arenas[i] != null && arenas[i] != primaryArena)
                    Destroy(arenas[i].gameObject);
            }
            arenas.Clear();

            bool isTraining = coordinator.IsTrainingSession;
            bool dualBrains = coordinator.sessionMode == BrainSessionRunMode.ParallelCompetitiveSelfPlay
                           || coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate;

            if (!isTraining)
                disableAll3DRendering = false;

            if (_lightStateSaved && _mainLight != null)
                _mainLight.shadows = (isTraining && disableShadowsInTraining) ? LightShadows.None : _savedLightShadows;

            primaryArena.transform.position = Vector3.zero;
            primaryArena.logPeriodicTelemetry = true;
            primaryArena.drivenByMultiManager = true;
            primaryArena.ResetSessionStats();
            arenas.Add(primaryArena);

            int targetCount = isTraining ? Mathf.Clamp(coordinator.parallelArenaCount, 1, 8) : 1;
            if (targetCount > 1 && coordinator.arenaPrefab != null)
            {
                for (int i = 1; i < targetCount; i++)
                {
                    Vector3 arenaPos = new Vector3(i * coordinator.arenaSpacingMeters, 0f, 0f);
                    GameObject clone = Instantiate(coordinator.arenaPrefab, arenaPos, Quaternion.identity);
                    clone.name = $"InkBlitzArena_{i}";
                    var inst = clone.GetComponent<InkBlitzArenaInstance>();
                    if (inst != null)
                    {
                        inst.logPeriodicTelemetry = false;
                        inst.drivenByMultiManager = true;
                        arenas.Add(inst);
                    }
                }
            }

            coordinator.ClearRegisteredBridges();

            for (int i = 0; i < arenas.Count; i++)
            {
                int portA;
                int portB;
                if (dualBrains)
                {
                    portA = coordinator.baseTcpPort + 2 * i;
                    portB = coordinator.baseTcpPort + 2 * i + 1;
                }
                else
                {
                    portA = coordinator.baseTcpPort + i;
                    portB = coordinator.baseTcpPort + 1000 + i;
                }

                bool is2v2Radio = coordinator.sessionMode == BrainSessionRunMode.SingleArenaSelfPlaySpectate
                               && coordinator.spectate2v2BrainRadio;
                bool effSupportActive = enableSupportTrooperBots || is2v2Radio;
                arenas[i].drivenByMultiManager = true;
                int targetStage = Mathf.Clamp(activeStageId, 1, 2);
                if (arenas[i].settings != null && arenas[i].settings.stageId != targetStage)
                    arenas[i].RebuildArenaForStage(targetStage);
                if (arenas[i].settings != null)
                    arenas[i].settings.enableSupportTrooperBots = effSupportActive;
                arenas[i].ConfigureForSlot(i, portA, portB, coordinator);
                arenas[i].SetSupportTrooperBotsEnabled(effSupportActive);
                arenas[i].ApplyColorPalette(activePaletteIndex);
            }

            _stepActive = new bool[arenas.Count];
            _stepHoldOnly = new bool[arenas.Count];

            _lastSampledRealtime = Time.realtimeSinceStartup;
            _lastSampledTick = primaryArena != null ? primaryArena.TickCount : 0;
            SmoothedTicksPerSec = 0f;

            overviewMode = false;
            FocusArena(0, overview: false);
            ApplyRenderingPolicy();
        }

        private void FixedUpdate()
        {
            if (isPausedForLauncher || isEmergencyHalted || (coordinator != null && coordinator.IsEmergencyHalted))
                return;
            int n = arenas.Count;
            if (n <= 0)
                return;
            if (_stepActive == null || _stepActive.Length != n)
            {
                _stepActive = new bool[n];
                _stepHoldOnly = new bool[n];
            }

            float dt = Time.fixedDeltaTime;
            for (int i = 0; i < n; i++)
            {
                var ar = arenas[i];
                _stepActive[i] = (ar != null) && ar.BeginParallelArenaStep(dt, out _stepHoldOnly[i]);
            }
            for (int i = 0; i < n; i++)
            {
                if (_stepActive[i] && arenas[i] != null)
                    arenas[i].CompleteParallelArenaStep(_stepHoldOnly[i]);
            }
        }

        private void Update()
        {
            if (isPausedForLauncher)
                return;

            float now = Time.realtimeSinceStartup;
            float elapsed = now - _lastSampledRealtime;
            if (elapsed >= 0.25f && primaryArena != null)
            {
                int curTick = primaryArena.TickCount;
                float instRate = Mathf.Max(0f, (curTick - _lastSampledTick) / elapsed);
                SmoothedTicksPerSec = (SmoothedTicksPerSec <= 0.01f) ? instRate : Mathf.Lerp(SmoothedTicksPerSec, instRate, 0.35f);
                _lastSampledTick = curTick;
                _lastSampledRealtime = now;
            }

#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null && GUIUtility.keyboardControl == 0)
            {
                if (kb.vKey.wasPressedThisFrame)
                {
                    disableAll3DRendering = !disableAll3DRendering;
                    ApplyRenderingPolicy();
                }
                else if (kb.cKey.wasPressedThisFrame)
                {
                    ResetCameraView(instant: false);
                }
                else if (kb.pKey.wasPressedThisFrame)
                {
                    CycleColorPalette(1);
                }
                else if (kb.mKey.wasPressedThisFrame)
                {
                    ArenaAudioSystem.IsHumanSfxMuted = !ArenaAudioSystem.IsHumanSfxMuted;
                }
                else if (arenas.Count > 1)
                {
                    if (kb.tabKey.wasPressedThisFrame)
                    {
                        disableAll3DRendering = false;
                        FocusArena((focusedArenaIndex + 1) % arenas.Count, overview: false);
                    }
                    else if (kb.digit0Key.wasPressedThisFrame || kb.numpad0Key.wasPressedThisFrame)
                    {
                        disableAll3DRendering = false;
                        FocusArena(focusedArenaIndex, overview: true);
                    }
                    else
                    {
                        for (int k = 0; k < Mathf.Min(8, arenas.Count); k++)
                        {
                            Key key = Key.Digit1 + k;
                            if (kb[key].wasPressedThisFrame)
                            {
                                disableAll3DRendering = false;
                                FocusArena(k, overview: false);
                                break;
                            }
                        }
                    }
                }
            }
#endif

            if (enableInteractiveCamera && !disableAll3DRendering)
                HandleInteractiveCameraInput();
        }

        private float _smoothedFloorClearanceY = 0.62f;
        private readonly RaycastHit[] _camFloorHits = new RaycastHit[8];

        private void LateUpdate()
        {
            if (isPausedForLauncher || disableAll3DRendering || mainCamera == null || arenas.Count == 0)
                return;

            StepCameraSmoothing(Time.unscaledDeltaTime);
        }

        public void StepCameraSmoothing(float dt)
        {
            if (mainCamera == null || arenas.Count == 0)
                return;

            float udt = Mathf.Clamp(dt, 0.0005f, 0.10f);
            float alpha = cameraSmoothTime <= 0.001f ? 1f : (1f - Mathf.Exp(-udt / cameraSmoothTime));

            _curCamYaw = Mathf.LerpAngle(_curCamYaw, _targetCamYaw, alpha);
            _curCamPitch = Mathf.Lerp(_curCamPitch, _targetCamPitch, alpha);
            _curCamDist = Mathf.Lerp(_curCamDist, _targetCamDist, alpha);
            _curPanXZ = Vector2.Lerp(_curPanXZ, _targetPanXZ, alpha);

            ApplyCameraTransform(udt);
        }

        public void FocusArena(int index, bool overview = false)
        {
            if (arenas.Count == 0)
                return;

            focusedArenaIndex = Mathf.Clamp(index, 0, arenas.Count - 1);
            overviewMode = overview && arenas.Count > 1;

            if (hud != null)
                hud.arena = arenas[focusedArenaIndex];

            ResetCameraView(instant: true);
            ApplyRenderingPolicy();
        }

        public void SetColorPalette(int paletteIndex)
        {
            int n = InkBlitzInkPalettes.Count;
            activePaletteIndex = ((paletteIndex % n) + n) % n;
            for (int i = 0; i < arenas.Count; i++)
            {
                if (arenas[i] != null)
                    arenas[i].ApplyColorPalette(activePaletteIndex);
            }
            if (arenas.Count == 0 && primaryArena != null)
                primaryArena.ApplyColorPalette(activePaletteIndex);
        }

        public void CycleColorPalette(int delta = 1)
        {
            SetColorPalette(activePaletteIndex + delta);
        }

        public void SetActiveStage(int stageId)
        {
            activeStageId = (stageId == 2) ? 2 : 1;
            if (coordinator != null)
                coordinator.activeStageId = activeStageId;

            if (arenas.Count > 0)
            {
                for (int i = 0; i < arenas.Count; i++)
                {
                    if (arenas[i] != null && (arenas[i].settings == null || arenas[i].settings.stageId != activeStageId))
                    {
                        arenas[i].RebuildArenaForStage(activeStageId);
                        arenas[i].SetSupportTrooperBotsEnabled(enableSupportTrooperBots);
                        arenas[i].ApplyColorPalette(activePaletteIndex);
                    }
                }
            }
            else if (primaryArena != null && (primaryArena.settings == null || primaryArena.settings.stageId != activeStageId))
            {
                primaryArena.RebuildArenaForStage(activeStageId);
                primaryArena.SetSupportTrooperBotsEnabled(enableSupportTrooperBots);
                primaryArena.ApplyColorPalette(activePaletteIndex);
            }
        }

        public void TriggerEmergencyStop()
        {
            isEmergencyHalted = true;
            _orbitDragging = false;
            _panDragging = false;

            for (int i = 0; i < arenas.Count; i++)
            {
                var ar = arenas[i];
                if (ar == null) continue;
                ar.projectileManager?.ClearAll();
                ar.audioSystem?.StopAllHumanSfx();
                ar.audioSystem?.Clear();
            }
            if (primaryArena != null)
            {
                primaryArena.projectileManager?.ClearAll();
                primaryArena.audioSystem?.StopAllHumanSfx();
                primaryArena.audioSystem?.Clear();
            }

            if (coordinator != null)
                coordinator.EmergencyHaltAll();
            else
                Time.timeScale = 0.0f;
        }

        public void ClearEmergencyStopForReset()
        {
            isEmergencyHalted = false;
            if (coordinator != null)
                coordinator.ClearEmergencyHalt();
            else
                Time.timeScale = 1.0f;
        }

        public void ResetCameraView(bool instant = false)
        {
            _orbitDragging = false;
            _panDragging = false;
            _targetPanXZ = Vector2.zero;
            _targetCamYaw = 0f;

            if (overviewMode && coordinator != null)
            {
                float distScale = Mathf.Max(1f, arenas.Count / 2.2f);
                _targetCamPitch = 42f;
                _targetCamDist = 112f * distScale;
            }
            else
            {
                _targetCamPitch = 43f;
                _targetCamDist = 28.8f;
            }

            if (instant)
            {
                _curCamYaw = _targetCamYaw;
                _curCamPitch = _targetCamPitch;
                _curCamDist = _targetCamDist;
                _curPanXZ = _targetPanXZ;
                _lastEffectiveDist = _targetCamDist;
                _smoothedFloorClearanceY = 0.62f;
                ApplyCameraTransform(0.016f);
            }
        }

        public void SetCameraOrbitState(float yawDeg, float pitchDeg, float distance, Vector2 panLocalXZ, bool instant = true)
        {
            _orbitDragging = false;
            _panDragging = false;
            _targetCamYaw = Mathf.Repeat(yawDeg, 360f);
            float minP = overviewMode ? 7.5f : minPitchDeg;
            _targetCamPitch = Mathf.Clamp(pitchDeg, minP, maxPitchDeg);
            float minD = overviewMode ? 18f : minZoomDistance;
            float maxD = overviewMode ? 260f : maxZoomDistance;
            _targetCamDist = Mathf.Clamp(distance, minD, maxD);
            _targetPanXZ = new Vector2(
                Mathf.Clamp(panLocalXZ.x, -10.0f, 10.0f),
                Mathf.Clamp(panLocalXZ.y, -14.0f, 14.0f));

            if (instant)
            {
                _curCamYaw = _targetCamYaw;
                _curCamPitch = _targetCamPitch;
                _curCamDist = _targetCamDist;
                _curPanXZ = _targetPanXZ;
                _lastEffectiveDist = _targetCamDist;
                ApplyCameraTransform(1.0f);
            }
        }

        private void HandleInteractiveCameraInput()
        {
            if (_dashboardUi == null)
                _dashboardUi = Object.FindAnyObjectByType<InkBlitzSessionDashboardUi>();
            if (_neuralHud == null)
                _neuralHud = Object.FindAnyObjectByType<InkBlitzNeuralVisualizerHud>();
            if (_winRateEstimator == null)
                _winRateEstimator = Object.FindAnyObjectByType<InkBlitzWinRateEstimator>();

            if (_dashboardUi != null && _dashboardUi.IsLauncherOpen)
            {
                _orbitDragging = false;
                _panDragging = false;
                return;
            }

#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            bool editingText = GUIUtility.keyboardControl != 0;
            bool shiftHeld = kb != null && (kb.leftShiftKey.isPressed || kb.rightShiftKey.isPressed);

            // Optional gentle arrow-key target pan
            if (kb != null && !editingText)
            {
                Vector2 keyPan = Vector2.zero;
                if (kb.leftArrowKey.isPressed) keyPan.x -= 1f;
                if (kb.rightArrowKey.isPressed) keyPan.x += 1f;
                if (kb.downArrowKey.isPressed) keyPan.y -= 1f;
                if (kb.upArrowKey.isPressed) keyPan.y += 1f;
                if (keyPan.sqrMagnitude > 0.01f)
                {
                    float panSpd = (overviewMode ? 36f : 11f) * Mathf.Clamp(Time.unscaledDeltaTime, 0f, 0.05f);
                    ApplyYawAlignedPanDelta(keyPan.normalized * panSpd);
                }
            }

            var mouse = Mouse.current;
            if (mouse == null)
                return;

            Vector2 mp = mouse.position.ReadValue();
            Vector2 guiPos = new Vector2(mp.x, Screen.height - mp.y);
            bool inWindow = mp.x >= 0f && mp.x <= Screen.width && mp.y >= 0f && mp.y <= Screen.height;
            bool overHud = IsPointerOverAnyHud(guiPos);

            bool lmbDown = mouse.leftButton.wasPressedThisFrame;
            bool rmbDown = mouse.rightButton.wasPressedThisFrame;
            bool mmbDown = mouse.middleButton.wasPressedThisFrame;

            if (inWindow && !overHud)
            {
                if (mmbDown || (shiftHeld && (lmbDown || rmbDown)))
                {
                    _panDragging = true;
                    _orbitDragging = false;
                    _prevMouseScreenPos = mp;
                }
                else if (lmbDown || rmbDown)
                {
                    _orbitDragging = true;
                    _panDragging = false;
                    _prevMouseScreenPos = mp;
                }
            }

            bool keepPan = mouse.middleButton.isPressed || (shiftHeld && (mouse.leftButton.isPressed || mouse.rightButton.isPressed));
            bool keepOrbit = !shiftHeld && (mouse.leftButton.isPressed || mouse.rightButton.isPressed);

            if (_panDragging && !keepPan)
                _panDragging = false;
            if (_orbitDragging && !keepOrbit)
                _orbitDragging = false;

            if (_orbitDragging || _panDragging)
            {
                Vector2 dpx = Vector2.ClampMagnitude(mp - _prevMouseScreenPos, 160f);
                _prevMouseScreenPos = mp;

                if (_orbitDragging)
                {
                    _targetCamYaw = Mathf.Repeat(_targetCamYaw + dpx.x * mouseOrbitSensitivity, 360f);
                    float minP = overviewMode ? 7.5f : minPitchDeg;
                    _targetCamPitch = Mathf.Clamp(_targetCamPitch - dpx.y * mousePitchSensitivity, minP, maxPitchDeg);
                }
                else if (_panDragging)
                {
                    float distScale = Mathf.Clamp(_lastEffectiveDist / 28.8f, 0.22f, 4.0f);
                    Vector2 panLocal = new Vector2(-dpx.x, -dpx.y) * (mousePanSensitivity * distScale);
                    ApplyYawAlignedPanDelta(panLocal);
                }
            }

            if (inWindow && !overHud)
            {
                float rawScroll = mouse.scroll.ReadValue().y;
                if (Mathf.Abs(rawScroll) > 0.01f)
                    ApplyMouseWheelScroll(rawScroll);
            }
#endif
        }

        public void ApplyMouseWheelScroll(float rawScroll)
        {
            if (Mathf.Abs(rawScroll) <= 0.01f)
                return;

            // Unity 6 InputSystem reports ~1.0 per wheel click (so a fast flick is 4..12), whereas legacy Win32 raw delta is 120.
            float notches = Mathf.Abs(rawScroll) >= 50f ? (rawScroll / 120f) : rawScroll;
            notches = Mathf.Clamp(notches, -4.5f, 4.5f);

            float minD = overviewMode ? 18f : minZoomDistance;
            float maxD = overviewMode ? 260f : maxZoomDistance;

            if (notches > 0f)
            {
                // Zooming IN: sync to actual effective distance only when rim-glide is actively clamping `_curCamDist`
                if (!overviewMode && _lastEffectiveDist < _curCamDist - 0.25f && _lastEffectiveDist < _targetCamDist - 0.25f)
                    _targetCamDist = _lastEffectiveDist;
                _targetCamDist = Mathf.Clamp(_targetCamDist * Mathf.Pow(0.85f, notches), minD, maxD);
            }
            else
            {
                // Zooming OUT (`notches < 0`):
                // 1) Never overwrite `_targetCamDist` downward with `_lastEffectiveDist` (which caused multi-frame smooth zoom-out to reset!),
                //    and guarantee crisp pullback from `minZoomDistance = 3.8m` via +1.65m/notch linear floor.
                float outNotches = -notches;
                float baseDist = !overviewMode ? Mathf.Max(_targetCamDist, _lastEffectiveDist) : _targetCamDist;
                float multNext = baseDist * Mathf.Pow(0.84f, notches);
                float linFloorNext = baseDist + outNotches * (overviewMode ? 4.5f : 1.65f);
                _targetCamDist = Mathf.Clamp(Mathf.Max(multNext, linFloorNext), minD, maxD);

                // 2) Prevent stadium wall `rimGlide` (`yAtWall < 5.50m`) from trapping zoom-out at low pitch or near walls!
                if (!overviewMode)
                {
                    float tWallEst = EstimateStadiumWallRayDistance(_targetCamPitch, _targetCamYaw, _targetPanXZ);
                    if (_targetCamDist > tWallEst - 0.5f)
                    {
                        float horizWallDist = Mathf.Max(1.5f, tWallEst * Mathf.Cos(_targetCamPitch * Mathf.Deg2Rad));
                        if (horizWallDist < 9.5f && _targetPanXZ.sqrMagnitude > 0.05f)
                        {
                            _targetPanXZ = Vector2.MoveTowards(_targetPanXZ, Vector2.zero, outNotches * 2.8f);
                            tWallEst = EstimateStadiumWallRayDistance(_targetCamPitch, _targetCamYaw, _targetPanXZ);
                            horizWallDist = Mathf.Max(1.5f, tWallEst * Mathf.Cos(_targetCamPitch * Mathf.Deg2Rad));
                        }

                        float clearPitch = Mathf.Clamp(
                            Mathf.Atan2(5.65f - 0.95f, Mathf.Max(2.5f, horizWallDist)) * Mathf.Rad2Deg,
                            minPitchDeg,
                            58.0f);
                        if (_targetCamPitch < clearPitch)
                            _targetCamPitch = Mathf.MoveTowards(_targetCamPitch, clearPitch, outNotches * 8.5f);
                    }
                }
            }
        }

        private static float EstimateStadiumWallRayDistance(float pitchDeg, float yawDeg, Vector2 panXZ)
        {
            const float innerHalfW = 11.45f;
            const float innerHalfL = 15.45f;
            Vector3 rayDir = Quaternion.Euler(pitchDeg, yawDeg, 0f) * Vector3.back;
            float tx = Mathf.Abs(rayDir.x) > 1e-5f
                ? ((rayDir.x > 0f ? innerHalfW : -innerHalfW) - panXZ.x) / rayDir.x
                : float.PositiveInfinity;
            float tz = Mathf.Abs(rayDir.z) > 1e-5f
                ? ((rayDir.z > 0f ? innerHalfL : -innerHalfL) - panXZ.y) / rayDir.z
                : float.PositiveInfinity;
            return Mathf.Max(1.0f, Mathf.Min(tx, tz));
        }

        private void ApplyYawAlignedPanDelta(Vector2 screenRightForward)
        {
            float rad = _curCamYaw * Mathf.Deg2Rad;
            float cos = Mathf.Cos(rad);
            float sin = Mathf.Sin(rad);
            // At yaw=0: camera looks +Z, right is +X, forward is +Z
            float dx = screenRightForward.x * cos + screenRightForward.y * sin;
            float dz = -screenRightForward.x * sin + screenRightForward.y * cos;
            _targetPanXZ.x += dx;
            _targetPanXZ.y += dz;

            if (overviewMode && coordinator != null)
            {
                float halfSpan = Mathf.Max(25f, (arenas.Count - 1) * coordinator.arenaSpacingMeters * 0.6f);
                _targetPanXZ.x = Mathf.Clamp(_targetPanXZ.x, -halfSpan, halfSpan);
                _targetPanXZ.y = Mathf.Clamp(_targetPanXZ.y, -65f, 65f);
            }
            else
            {
                _targetPanXZ.x = Mathf.Clamp(_targetPanXZ.x, -10.0f, 10.0f);
                _targetPanXZ.y = Mathf.Clamp(_targetPanXZ.y, -14.0f, 14.0f);
            }
        }

        private bool IsPointerOverAnyHud(Vector2 guiPos)
        {
            bool isTraining = coordinator != null && coordinator.IsTrainingSession;

            // 1. Right-Edge Mathematical Thought Visualizer (Spectate Mode)
            float rightPanelW = (_neuralHud != null) ? _neuralHud.ActiveRightPanelWidthPx : 0f;
            if (rightPanelW > 0f && guiPos.x >= Screen.width - rightPanelW - 2f)
                return true;
            if (!isTraining && _neuralHud != null && !_neuralHud.showRightNeuralPanel)
            {
                if (new Rect(Screen.width - 122f, 8f, 114f, 28f).Contains(guiPos))
                    return true;
            }

            // 2. Top-Left Arena Telemetry HUD
            if (hud != null && hud.showHud)
            {
                Rect tlRect = hud.isCollapsed
                    ? new Rect(8f, 8f, 160f, 30f)
                    : new Rect(8f, 8f, 366f, isTraining ? 282f : 240f);
                if (tlRect.Contains(guiPos))
                    return true;
            }

            // 3. Bottom-Left Agent A Subjective POV, 02 DEPTH, 03 WORLD MODEL & 04 BRAIN DIAGNOSTIC HUDs
            if (!isTraining && _neuralHud != null)
            {
                if (_neuralHud.GetSubjectivePovWindowRect().Contains(guiPos))
                    return true;
                if (_neuralHud.GetDepthCortexWindowRect().Contains(guiPos))
                    return true;
                if (_neuralHud.GetWorldModelWindowRect().Contains(guiPos))
                    return true;
                if (_neuralHud.circuitHud != null && _neuralHud.circuitHud.GetPanelRect().Contains(guiPos))
                    return true;
            }

            // 4. Top-Right Training Monitor / Spectate Status Strip + Top Emergency Stop Switch
            if (_dashboardUi != null)
            {
                if (_dashboardUi.GetEmergencyStopButtonRect().Contains(guiPos))
                    return true;
                if (isEmergencyHalted && _dashboardUi.GetEmergencyModalRect().Contains(guiPos))
                    return true;

                if (isTraining)
                {
                    Rect trRect = _dashboardUi.IsTrainingPanelCollapsed
                        ? new Rect(Screen.width - 232f, 8f, 224f, 40f)
                        : new Rect(Screen.width - 360f, 8f, 352f, 426f);
                    if (trRect.Contains(guiPos))
                        return true;
                }
                else
                {
                    Rect sr = _dashboardUi.GetSpectateStatusStripRect();
                    if (new Rect(sr.x - 2f, sr.y - 4f, sr.width + 4f, sr.height + 8f).Contains(guiPos))
                        return true;
                    if (coordinator != null && coordinator.enableStrategyBroadcast && _dashboardUi.GetGemmaBroadcastRect().Contains(guiPos))
                        return true;
                }
            }

            // 5. Top-Center Slim Win-Rate Tug-of-War Bar
            if (_winRateEstimator == null)
                _winRateEstimator = Object.FindAnyObjectByType<InkBlitzWinRateEstimator>();
            if (_winRateEstimator != null && _winRateEstimator.IsBarVisibleNow)
            {
                if (_winRateEstimator.GetTopWinRateBarRect().Contains(guiPos))
                    return true;
            }

            return false;
        }

        private void ApplyCameraTransform(float udt)
        {
            if (mainCamera == null || arenas.Count == 0)
                return;

            Vector3 arenaBase;
            if (overviewMode && coordinator != null)
            {
                float spanX = (arenas.Count - 1) * coordinator.arenaSpacingMeters;
                arenaBase = new Vector3(spanX * 0.5f, 0f, 0f);
            }
            else
            {
                int idx = Mathf.Clamp(focusedArenaIndex, 0, arenas.Count - 1);
                arenaBase = arenas[idx] != null ? arenas[idx].transform.position : Vector3.zero;
            }

            Vector3 focusLocal = new Vector3(_curPanXZ.x, overviewMode ? 1.5f : 0.95f, _curPanXZ.y);
            Quaternion orbitRot = Quaternion.Euler(_curCamPitch, _curCamYaw, 0f);
            Vector3 rayDir = orbitRot * Vector3.back;

            float effectiveDist = _curCamDist;
            if (!overviewMode)
            {
                // Smoothly glide over & inside the 3.4m stadium perimeter walls when tilting near-horizontal
                // so the outer walls never block the sightline even at 2.5-deg ground-level pitch!
                const float innerHalfW = 11.45f;
                const float innerHalfL = 15.45f;
                float tx = Mathf.Abs(rayDir.x) > 1e-5f
                    ? ((rayDir.x > 0f ? innerHalfW : -innerHalfW) - focusLocal.x) / rayDir.x
                    : float.PositiveInfinity;
                float tz = Mathf.Abs(rayDir.z) > 1e-5f
                    ? ((rayDir.z > 0f ? innerHalfL : -innerHalfL) - focusLocal.z) / rayDir.z
                    : float.PositiveInfinity;
                float tWall = Mathf.Max(1.0f, Mathf.Min(tx, tz));

                if (tWall < 1000f && _curCamDist > tWall)
                {
                    float yAtWall = focusLocal.y + rayDir.y * tWall;
                    float rimGlide01 = Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(5.50f, 3.50f, yAtWall));
                    effectiveDist = Mathf.Max(minZoomDistance, Mathf.Lerp(_curCamDist, tWall, rimGlide01));
                }
            }

            _lastEffectiveDist = effectiveDist;
            Vector3 focusWorld = arenaBase + focusLocal;
            Vector3 camPos = focusWorld + rayDir * effectiveDist;

            if (!overviewMode)
            {
                float targetMinLocalY = 0.62f;
                Vector3 probeOrigin = new Vector3(camPos.x, arenaBase.y + 4.4f, camPos.z);
                int hitCount = Physics.RaycastNonAlloc(probeOrigin, Vector3.down, _camFloorHits, 4.8f, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore);
                for (int i = 0; i < hitCount; i++)
                {
                    Collider col = _camFloorHits[i].collider;
                    if (col == null) continue;
                    if (col.GetComponentInParent<InkBlitzPlayerAgent>() != null) continue;
                    if (col.GetComponentInParent<BlitzCoreObjective>() != null) continue;
                    if (col.GetComponentInParent<DeflectPowerPickup>() != null) continue;

                    float localHitY = _camFloorHits[i].point.y - arenaBase.y;
                    if (localHitY <= 2.65f)
                        targetMinLocalY = Mathf.Max(targetMinLocalY, localHitY + 0.58f);
                }

                _smoothedFloorClearanceY = Mathf.Lerp(_smoothedFloorClearanceY, targetMinLocalY, 1f - Mathf.Exp(-udt * 16f));
                float minWorldY = arenaBase.y + _smoothedFloorClearanceY;
                if (camPos.y < minWorldY)
                    camPos.y = minWorldY;
            }
            else
            {
                camPos.y = Mathf.Max(arenaBase.y + 2.5f, camPos.y);
            }

            mainCamera.transform.position = camPos;
            Vector3 toFocus = focusWorld - camPos;
            mainCamera.transform.rotation = toFocus.sqrMagnitude > 1e-4f
                ? Quaternion.LookRotation(toFocus.normalized, Vector3.up)
                : orbitRot;
        }

        public void ApplyRenderingPolicy()
        {
            bool isTraining = coordinator != null && coordinator.IsTrainingSession;
            if (isTraining && coordinator != null)
            {
                int stepCap = disableAll3DRendering
                    ? Mathf.Max(4, coordinator.maxTrainingStepsPerRenderFrame * 2)
                    : Mathf.Clamp(coordinator.maxTrainingStepsPerRenderFrame, 1, 10);
                Time.maximumDeltaTime = Mathf.Max(Time.fixedDeltaTime, Time.fixedDeltaTime * stepCap);
            }

            if (disableAll3DRendering)
            {
                for (int i = 0; i < arenas.Count; i++)
                {
                    if (arenas[i] != null)
                        arenas[i].SetArenaRenderingEnabled(false);
                }
                if (mainCamera != null && _cameraStateSaved)
                {
                    mainCamera.cullingMask = 0;
                    mainCamera.clearFlags = CameraClearFlags.SolidColor;
                    mainCamera.backgroundColor = new Color(0.08f, 0.10f, 0.13f, 1f);
                }
                return;
            }

            if (mainCamera != null && _cameraStateSaved)
            {
                mainCamera.cullingMask = _savedCullingMask;
                mainCamera.clearFlags = _savedClearFlags;
                mainCamera.backgroundColor = _savedBgColor;
            }

            bool repOnly = isTraining && renderRepresentativeArenaOnly && !overviewMode;
            for (int i = 0; i < arenas.Count; i++)
            {
                if (arenas[i] != null)
                    arenas[i].SetArenaRenderingEnabled(!repOnly || i == focusedArenaIndex);
            }
        }

        private void OnDestroy()
        {
            if (_lightStateSaved && _mainLight != null)
                _mainLight.shadows = _savedLightShadows;
            if (_cameraStateSaved && mainCamera != null)
            {
                mainCamera.cullingMask = _savedCullingMask;
                mainCamera.clearFlags = _savedClearFlags;
                mainCamera.backgroundColor = _savedBgColor;
            }
        }
    }
}
