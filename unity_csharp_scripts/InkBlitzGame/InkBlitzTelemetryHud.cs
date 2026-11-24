// InkBlitzTelemetryHud.cs — Compact OnGUI telemetry HUD for BlitzCore 1v1 & Brain observation
using UnityEngine;
using UniversalBrain3D;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace InkBlitz3D
{
    public class InkBlitzTelemetryHud : MonoBehaviour
    {
        public InkBlitzArenaInstance arena;
        public UniversalArenaCoordinator3D coordinator;
        public InkBlitzMultiArenaManager multiManager;
        public bool showHud = true;
        public bool isCollapsed = false;

        private float _fpsSmoothed = 60f;
        private GUIStyle _boxStyle;
        private GUIStyle _labelStyle;
        private GUIStyle _collapseBtnStyle;

        private void Awake()
        {
            if (arena == null) arena = Object.FindAnyObjectByType<InkBlitzArenaInstance>();
            if (coordinator == null) coordinator = Object.FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (multiManager == null) multiManager = Object.FindAnyObjectByType<InkBlitzMultiArenaManager>();
        }

        private void Update()
        {
            if (Time.unscaledDeltaTime > 1e-4f)
                _fpsSmoothed = Mathf.Lerp(_fpsSmoothed, 1f / Time.unscaledDeltaTime, 0.1f);
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null && GUIUtility.keyboardControl == 0 && kb.f1Key.wasPressedThisFrame)
                isCollapsed = !isCollapsed;
#endif
        }

        private void OnGUI()
        {
            if (multiManager != null && multiManager.isPausedForLauncher)
                return;
            if (!showHud || arena == null || arena.turfGrid == null)
                return;

            if (_boxStyle == null)
            {
                _boxStyle = new GUIStyle(GUI.skin.box);
                _labelStyle = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 12,
                    richText = true,
                    wordWrap = true
                };
                _labelStyle.normal.textColor = Color.white;
                _collapseBtnStyle = new GUIStyle(GUI.skin.button)
                {
                    fontSize = 10,
                    fontStyle = FontStyle.Bold,
                    alignment = TextAnchor.MiddleCenter
                };
            }

            if (isCollapsed)
            {
                if (GUI.Button(new Rect(12f, 12f, 152f, 24f), "▼ 戦況テレメトリ (F1)", _collapseBtnStyle))
                    isCollapsed = false;
                return;
            }

            InkBlitzPlayerAgent agA = arena.agents.Count > 0 ? arena.agents[0] : null;
            InkBlitzPlayerAgent agB = arena.agents.Count > 1 ? arena.agents[1] : null;
            InkBlitzPlayerAgent supA = (arena.agents.Count > 2 && arena.agents[2] != null && arena.agents[2].isSupportTrooperBot && arena.agents[2].gameObject.activeSelf) ? arena.agents[2] : null;
            InkBlitzPlayerAgent supB = (arena.agents.Count > 3 && arena.agents[3] != null && arena.agents[3].isSupportTrooperBot && arena.agents[3].gameObject.activeSelf) ? arena.agents[3] : null;
            UniversalBrainBridge3D bridge = agA != null ? agA.brainBridge : null;

            bool isTraining = coordinator != null && coordinator.IsTrainingSession;
            float boxH = isTraining ? 276f : 234f;
            GUI.Box(new Rect(12f, 12f, 358f, boxH), GUIContent.none, _boxStyle);
            if (GUI.Button(new Rect(314f, 16f, 50f, 18f), "畳むF1", _collapseBtnStyle))
            {
                isCollapsed = true;
                return;
            }

            string connStr = (bridge != null && bridge.IsConnected)
                ? "<color=#4DFF88>CONN(60Hz)</color>"
                : "<color=#FFCC44>CONNECTING</color>";

            int cntA = Mathf.Clamp(Mathf.RoundToInt((1f - arena.bestProgressTeamA) * 100f), 0, 100);
            int cntB = Mathf.Clamp(Mathf.RoundToInt((1f - arena.bestProgressTeamB) * 100f), 0, 100);
            string hexA = arena.settings != null ? ColorUtility.ToHtmlStringRGB(arena.settings.teamAColor) : "7F99DC";
            string hexB = arena.settings != null ? ColorUtility.ToHtmlStringRGB(arena.settings.teamBColor) : "D1BF9B";

            string rmInfo = "-";
            if (arena.blitzCore != null)
            {
                if (arena.blitzCore.State == BlitzCoreState.Shielded)
                {
                    float tug = arena.blitzCore.ShieldTug;
                    string tugStr = tug >= 0f ? $"+{tug:F0}" : $"{tug:F0}";
                    rmInfo = $"<color=#FFDD55>SHIELD ({arena.blitzCore.ShieldHp:F0} HP | {tugStr})</color>";
                }
                else if (arena.blitzCore.State == BlitzCoreState.PickupReady)
                {
                    rmInfo = "<color=#66FFAA>PICKUP READY!</color>";
                }
                else if (arena.blitzCore.Carrier != null)
                {
                    string timerStr = $"({arena.blitzCore.CarrierHoldTimer:F0}s)";
                    rmInfo = arena.blitzCore.Carrier.teamId > 0
                        ? $"<color=#{hexA}>TEAM A {timerStr}</color>"
                        : $"<color=#{hexB}>TEAM B {timerStr}</color>";
                }
            }

            string supInfoA = supA != null
                ? (supA.isRespawning ? $" <color=#FF7777>[+Bot:{supA.RespawnTimeRemaining:F1}s]</color>" : $" <color=#88EEAA>[+Bot:{supA.hp:F0}]</color>")
                : "";
            string supInfoB = supB != null
                ? (supB.isRespawning ? $" <color=#FF7777>[+Bot:{supB.RespawnTimeRemaining:F1}s]</color>" : $" <color=#88EEAA>[+Bot:{supB.hp:F0}]</color>")
                : "";
            string buffA = (agA != null && agA.HasDeflectBuff) ? $" <color=#33FFCC>[RFL {agA.deflectTimeRemaining:F1}s]</color>" : "";
            string buffB = (agB != null && agB.HasDeflectBuff) ? $" <color=#33FFCC>[RFL {agB.deflectTimeRemaining:F1}s]</color>" : "";
            string formA = agA != null
                ? (agA.isRespawning ? $"RESPAWN({agA.RespawnTimeRemaining:F1}s)" : (agA.isSubmergedInFriendlyInk ? "DIVE_SUB" : (agA.isDiveForm ? "DIVE" : "HUMAN"))) + buffA
                : "-";
            string formB = agB != null
                ? (agB.isRespawning ? $"RESPAWN({agB.RespawnTimeRemaining:F1}s)" : (agB.isSubmergedInFriendlyInk ? "DIVE_SUB" : (agB.isDiveForm ? "DIVE" : "HUMAN"))) + buffB
                : "-";

            int verbId = bridge != null ? bridge.latestMotorVerb : (agA != null ? agA.currentVerb : 0);
            float aimYaw = bridge != null ? bridge.latestAimYawRad * Mathf.Rad2Deg : 0f;
            float aimPitch = bridge != null ? bridge.latestAimPitchRad * Mathf.Rad2Deg : 0f;
            string speech = bridge != null ? bridge.latestInnerSpeech : "";
            string otBanner = arena.isOvertime ? " <color=#FF5533><b>OT!</b></color>" : "";
            string sessionModeStr = coordinator != null ? coordinator.sessionMode.ToString() : "-";
            string bridgesStr = coordinator != null ? $"{coordinator.ConnectedBridgeCount}/{coordinator.RegisteredBridgeCount}" : "-";
            string trainingLines = "";
            if (isTraining && multiManager != null)
            {
                float tps = multiManager.SmoothedTicksPerSec;
                int brainCount = Mathf.Max(1, coordinator.ConnectedBridgeCount);
                string viewStr = multiManager.disableAll3DRendering
                    ? "<color=#FFCC44>3D OFF (Turbo)</color>"
                    : (multiManager.overviewMode
                        ? "ALL 4"
                        : $"<color=#66FFAA>REP A{multiManager.focusedArenaIndex}</color>");
                var sb = new System.Text.StringBuilder(120);
                for (int i = 0; i < multiManager.arenas.Count; i++)
                {
                    var ar = multiManager.arenas[i];
                    if (ar == null) continue;
                    if (i > 0) sb.Append(' ');
                    string st = ar.IsHoldingForNextGen
                        ? "Hold"
                        : (ar.blitzCore != null && ar.blitzCore.State == BlitzCoreState.Carried
                            ? (ar.blitzCore.Carrier != null && ar.blitzCore.Carrier.teamId > 0 ? $"{ar.matchTimeRemaining:F0}s:A" : $"{ar.matchTimeRemaining:F0}s:B")
                            : $"{ar.matchTimeRemaining:F0}s:s{(ar.blitzCore != null ? ar.blitzCore.ShieldHp : 0f):F0}");
                    sb.Append($"[A{i}:{st}]");
                }
                trainingLines =
                    $"\nRate: <b>{tps:F1}t/s</b> ({tps * brainCount:F0} br/s) Tick:{arena.TickCount} [{viewStr}]" +
                    $"\n<color=#D0E8FF>{sb}</color> <color=#999999>[1-4/V/0/C/P]</color>";
            }

            string text =
                $"<b>InkBlitz 3D | Arena {arena.arenaIndex}</b>  FPS:{_fpsSmoothed:F1} {Time.timeScale:F1}x  <b>{arena.matchTimeRemaining:F1}s</b>{otBanner}\n" +
                $"TCP:{connStr}  Br:{bridgesStr}  <color=#8FB8DE>{sessionModeStr}</color>\n" +
                $"Wins: <color=#{hexA}>A {arena.teamAWins}</color>-<color=#{hexB}>{arena.teamBWins} B</color>  " +
                $"Count: <color=#{hexA}>{cntA}</color> vs <color=#{hexB}>{cntB}</color>\n" +
                $"Core: {rmInfo}\n" +
                $"Turf: <color=#{hexA}>A {arena.turfGrid.TeamARatio01 * 100f:F1}%</color> / " +
                $"<color=#{hexB}>B {arena.turfGrid.TeamBRatio01 * 100f :F1}%</color>\n" +
                $"A: <b>{formA}</b> HP:{(agA != null ? agA.hp : 0f):F0} Ink:{(agA != null ? agA.InkTank01 * 100f : 0f):F0}%{supInfoA}\n" +
                $"B: <b>{formB}</b> HP:{(agB != null ? agB.hp : 0f):F0} Ink:{(agB != null ? agB.InkTank01 * 100f : 0f):F0}%{supInfoB}\n" +
                $"Verb: {GetVerbName(verbId)}({verbId}) Aim:({aimYaw:F0},{aimPitch:F0})\n" +
                $"Speech: <color=#FFEE88>{speech}</color>\n" +
                $"Last: {arena.lastMatchSummary}" +
                trainingLines;

            GUI.Label(new Rect(20f, 18f, 340f, boxH - 12f), text, _labelStyle);
        }

        private static string GetVerbName(int verbId)
        {
            switch (verbId)
            {
                case UniversalBrainBridge3D.VERB_IDLE_OBSERVE: return "IdleObserve";
                case UniversalBrainBridge3D.VERB_APPROACH_HOVER: return "ApproachHover";
                case UniversalBrainBridge3D.VERB_COMMIT_INTERACT: return "CommitInteract";
                case UniversalBrainBridge3D.VERB_GRASP_PICK: return "GraspPick";
                case UniversalBrainBridge3D.VERB_RELEASE_PLACE: return "ReleasePlace(Shoot)";
                case UniversalBrainBridge3D.VERB_EVADE_HAZARD: return "EvadeHazard(Swim)";
                case UniversalBrainBridge3D.VERB_IMPULSE_PUSH: return "ImpulsePush";
                case UniversalBrainBridge3D.VERB_COMMUNICATE_PING: return "CommunicatePing";
                default: return "Unknown";
            }
        }
    }
}
