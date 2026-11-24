// InkBlitzBrainCircuitHud.cs — Collapsible Square Functional-Diagnostic Brain Architecture HUD
// Unifies Luminous / Material Dark Architecture Wiring with Real-Time Theoretical Competence Verification:
//   - Node Luminescence = Functional Competence F_i in [0, 1] x Live Activity (NOT naive vector RMS!):
//       * HEALTHY (OK)  : Bright theme glow + outer halo + emerald/cyan dot when theoretical role invariants hold
//       * WEAK (INIT)   : Dimmed low-contrast slate + hollow dot when untrained / flat Q / high pred error / zero CF spread
//       * WARN (FAULT)  : Pulsing amber-coral ring + `!` when frozen representation, prolonged BG gate lock, or Q-spike
//   - Wire Luminescence = Exact Mathematical Pathway Coupling (`gwtAlpha[0..4]`, `||z_td||`, `|Q - Q_raw|`, `S1!=GWT`, `GateOpen`)
//       with Basal Ganglia physical pulse-blocking bar when `commitGateOpen == false`!
//   - Strictly ZERO rasterization cost (`0.000 ms`, `CircuitRasterCount` unchanged) when collapsed (`showCircuitHud == false`).
// License: GPL-3.0

using System;
using UnityEngine;
using UniversalBrain3D;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace InkBlitz3D
{
    public class InkBlitzBrainCircuitHud : MonoBehaviour
    {
        public enum ModuleDiagState : byte
        {
            WeakOrUntrained = 0,
            Healthy = 1,
            FaultOrFrozen = 2
        }

        public enum PathwayKind : byte
        {
            Generic = 0,
            GwtSpec0Vis = 1,
            GwtSpec1Spatial = 2,
            GwtSpec2Rssm = 3,
            GwtSpec34Strat = 4,
            TopDownMod = 5,
            RssmToValue = 6,
            MetacogSelf = 7,
            S1ConflictToBg = 8,
            BgGateToMotor = 9,
            GwtToMotor = 10,
            GwtToBroca = 11
        }

        public const int PANEL_W = 352;
        public const int HEADER_H = 44;
        public const int FOOTER_H = 24;
        public const int CANVAS_W = 344;
        public const int CANVAS_H = 276;
        public const int PANEL_H = HEADER_H + CANVAS_H + FOOTER_H + 8; // 352x352 exact square!

        [Header("References")]
        public UniversalArenaCoordinator3D coordinator;
        public InkBlitzSessionDashboardUi dashboardUi;
        public InkBlitzNeuralVisualizerHud neuralHud;

        [Header("Circuit HUD Settings")]
        public bool showCircuitHud = true;
        [Range(0, 2)] public int activeViewIndex = 0; // 0: Whole-Brain, 1: GPU 9-Model, 2: Motor Loop

        public int CircuitRasterCount { get; private set; } = 0;
        public Texture2D CircuitCanvasTexture => _canvasTex;

        // Live whole-brain functional summary (View 0: 9 core regions)
        public int HealthyNodeCount { get; private set; } = 0;
        public int WeakNodeCount { get; private set; } = 0;
        public int WarnNodeCount { get; private set; } = 0;
        public float OverallHealthScore01 { get; private set; } = 0f;

        private struct NodeDef
        {
            public string id;
            public string title;
            public string sub;
            public float x, y, w, h, rx;
            public Color32 accent;
            public byte role; // 0: step, 1: focal, 2: store/gate
        }

        private struct EdgeDef
        {
            public int src;
            public int dst;
            public sbyte laneOffset;
            public Color32 tint;
            public PathwayKind kind;
        }

        private sealed class CircuitViewDef
        {
            public string tabLabel;
            public NodeDef[] nodes;
            public EdgeDef[] edges;
            public int selectedIdx;
        }

        private CircuitViewDef[] _views;
        private readonly float[][] _nodeSmoothLum = new float[3][];
        private readonly float[][] _nodeHealth01 = new float[3][];
        private readonly ModuleDiagState[][] _nodeDiag = new ModuleDiagState[3][];
        private readonly string[][] _nodeLiveMetric = new string[3][];
        private readonly string[][] _nodeInspectorLine = new string[3][];

        // Rolling Temporal Verification Trackers (O(1) zero-GC EMA trackers across incoming UBNT frames)
        private ulong _lastIngestedSeq = 0;
        private uint _lastEpisodeId = 0;
        private int _framesInEp = 0;
        private readonly float[] _prevZ = new float[32];
        private readonly float[] _prevRssm = new float[256];
        private readonly float[] _prevMec = new float[256];
        private readonly float[] _prevSelf = new float[8];
        private Vector3 _prevBelievedPos;
        private float _prevIgn = 0.5f;
        private float _prevAlpha4 = 0.2f;
        private bool _hasPrevVecs = false;

        private float _emaDeltaZ = 0.08f;
        private float _emaDeltaRssm = 0.08f;
        private float _emaDeltaMec = 0.08f;
        private float _emaDeltaPos = 0.05f;
        private float _emaDeltaSelf = 0.04f;
        private float _emaIgnChange = 0.08f;
        private float _emaStratChange = 0.04f;
        private float _emaQSpread = 0.35f;
        private float _emaCfSpread = 0.12f;
        private float _emaDepthMaeM = 2.2f;
        private float _emaVisMae = 0.12f;
        private int _consecutiveGateHoldFrames = 0;
        private int _consecutiveGateOpenFrames = 0;
        private uint _protoSymbolBitmask = 0; // Tracks distinct RVQ symbols used (`0..31`)
        private byte _candUsedBitmask = 0;    // Tracks distinct motor candidates used (`0..7`)

        private Texture2D _canvasTex;
        private Color32[] _pixels;
        private float _nextRasterTime = 0f;
        private uint _lastLoopClosures = 0;
        private float _loopClosureFlash = 0f;
        private int _lastProtoSym = -1;
        private float _brocaSymFlash = 0f;

        // Cached IMGUI styles & 9-slice textures
        private bool _stylesReady = false;
        private Texture2D _texOuterCard;
        private Texture2D _texPillIdle;
        private Texture2D _texPillActive;
        private GUIStyle _stOuterCard;
        private GUIStyle _stHdrTitle;
        private GUIStyle _stHdrBadge;
        private GUIStyle _stNodeTitle;
        private GUIStyle _stNodeSub;
        private GUIStyle _stFooter;
        private GUIStyle _stPillIdle;
        private GUIStyle _stPillActive;

        public bool IsVisibleInSessionNow =>
            coordinator != null
            && !coordinator.IsTrainingSession
            && (dashboardUi == null || !dashboardUi.IsLauncherOpen);

        private void Awake()
        {
            AutoBind();
            BuildCircuitDefinitions();
            EnsureCanvasBuffer();
        }

        private void OnDestroy()
        {
            DestroySafe(_canvasTex);
            DestroySafe(_texOuterCard);
            DestroySafe(_texPillIdle);
            DestroySafe(_texPillActive);
        }

        private static void DestroySafe(UnityEngine.Object o)
        {
            if (o == null) return;
            if (Application.isPlaying) Destroy(o);
            else DestroyImmediate(o);
        }

        public void AutoBind()
        {
            if (coordinator == null) coordinator = FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (dashboardUi == null) dashboardUi = FindAnyObjectByType<InkBlitzSessionDashboardUi>();
            if (neuralHud == null) neuralHud = GetComponent<InkBlitzNeuralVisualizerHud>() ?? FindAnyObjectByType<InkBlitzNeuralVisualizerHud>();
        }

        public void ResetDiagnosticTrackers()
        {
            _lastIngestedSeq = 0;
            _lastEpisodeId = 0;
            _framesInEp = 0;
            _hasPrevVecs = false;
            _emaDeltaZ = 0.08f;
            _emaDeltaRssm = 0.08f;
            _emaDeltaMec = 0.08f;
            _emaDeltaPos = 0.05f;
            _emaDeltaSelf = 0.04f;
            _emaIgnChange = 0.08f;
            _emaStratChange = 0.04f;
            _emaQSpread = 0.35f;
            _emaCfSpread = 0.12f;
            _emaDepthMaeM = 2.2f;
            _emaVisMae = 0.12f;
            _consecutiveGateHoldFrames = 0;
            _consecutiveGateOpenFrames = 0;
            _protoSymbolBitmask = 0;
            _candUsedBitmask = 0;
        }

        private void EnsureCanvasBuffer()
        {
            if (_pixels == null || _pixels.Length != CANVAS_W * CANVAS_H)
                _pixels = new Color32[CANVAS_W * CANVAS_H];
            if (_canvasTex == null)
            {
                _canvasTex = new Texture2D(CANVAS_W, CANVAS_H, TextureFormat.RGBA32, false)
                {
                    name = "UniversalBrain3D_CircuitHudCanvas",
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
                ClearCanvas(new Color32(14, 17, 21, 255));
                _canvasTex.SetPixels32(_pixels);
                _canvasTex.Apply(false);
            }
        }

        private void BuildCircuitDefinitions()
        {
            if (_views != null) return;

            Color32 cCyan = new Color32(157, 210, 255, 255);    // #9dd2ff (--primary)
            Color32 cEmer = new Color32(80, 227, 137, 255);     // #50e389 (--chart-3)
            Color32 cGate = new Color32(14, 188, 95, 255);      // #0ebc5f (--positive)
            Color32 cViol = new Color32(186, 164, 255, 255);    // #baa4ff (RSSM world model)
            Color32 cAmbr = new Color32(245, 215, 127, 255);    // #f5d77f (--chart-8)
            Color32 cSky = new Color32(130, 199, 255, 255);     // #82c7ff (--chart-7)
            Color32 cMint = new Color32(96, 214, 115, 255);     // #60d673 (--code-green)
            Color32 cPink = new Color32(255, 166, 215, 255);    // #ffa6d7 (RVQ/Broca)

            // -----------------------------------------------------------------
            // View 0: 9-Node Whole-Brain Functional Diagnostic Circuit
            // -----------------------------------------------------------------
            var v0Nodes = new[]
            {
                new NodeDef { id = "fov_slot", title = "Fovea·4Slot", sub = "知覚・視覚予測", x = 8, y = 14, w = 98, h = 66, rx = 16, accent = cCyan, role = 0 },
                new NodeDef { id = "hippo_mec", title = "海馬 MEC·地図", sub = "SLAM・4層塗", x = 8, y = 104, w = 98, h = 66, rx = 16, accent = cEmer, role = 0 },
                new NodeDef { id = "amyg_strat", title = "扁桃体·StratTF", sub = "脅威・前頭戦略", x = 8, y = 194, w = 98, h = 66, rx = 16, accent = cAmbr, role = 0 },

                new NodeDef { id = "cat_rssm", title = "3-Head RSSM", sub = "世界予測 H=3", x = 123, y = 14, w = 98, h = 66, rx = 16, accent = cViol, role = 1 },
                new NodeDef { id = "active_gwt", title = "ActiveGWT", sub = "5領域意識点火", x = 123, y = 102, w = 98, h = 70, rx = 18, accent = cCyan, role = 1 },
                new NodeDef { id = "bg_stnsnr", title = "基底核 STN門", sub = "60Hz Go/Brake", x = 123, y = 194, w = 98, h = 66, rx = 16, accent = cGate, role = 2 },

                new NodeDef { id = "self_hot", title = "SelfModel", sub = "自己誤差 e_s", x = 238, y = 14, w = 98, h = 66, rx = 16, accent = cSky, role = 2 },
                new NodeDef { id = "s1_d3qn", title = "S1·TT-D3QN", sub = "24Hz反射・価値", x = 238, y = 104, w = 98, h = 66, rx = 16, accent = cMint, role = 1 },
                new NodeDef { id = "motor_broca", title = "Motor·内言", sub = "6DoF・象徴接地", x = 238, y = 194, w = 98, h = 66, rx = 16, accent = cPink, role = 0 },
            };
            var v0Edges = new[]
            {
                new EdgeDef { src = 0, dst = 3, laneOffset = 0, tint = cCyan, kind = PathwayKind.Generic },
                new EdgeDef { src = 0, dst = 4, laneOffset = -1, tint = cCyan, kind = PathwayKind.GwtSpec0Vis },
                new EdgeDef { src = 4, dst = 0, laneOffset = 1, tint = cSky, kind = PathwayKind.TopDownMod },
                new EdgeDef { src = 1, dst = 4, laneOffset = 0, tint = cEmer, kind = PathwayKind.GwtSpec1Spatial },
                new EdgeDef { src = 1, dst = 7, laneOffset = 1, tint = cEmer, kind = PathwayKind.Generic },
                new EdgeDef { src = 2, dst = 4, laneOffset = 1, tint = cAmbr, kind = PathwayKind.GwtSpec34Strat },
                new EdgeDef { src = 3, dst = 4, laneOffset = 0, tint = cViol, kind = PathwayKind.GwtSpec2Rssm },
                new EdgeDef { src = 3, dst = 7, laneOffset = -1, tint = cViol, kind = PathwayKind.RssmToValue },
                new EdgeDef { src = 4, dst = 6, laneOffset = 0, tint = cSky, kind = PathwayKind.Generic },
                new EdgeDef { src = 6, dst = 4, laneOffset = 1, tint = cAmbr, kind = PathwayKind.MetacogSelf },
                new EdgeDef { src = 4, dst = 7, laneOffset = 0, tint = cCyan, kind = PathwayKind.Generic },
                new EdgeDef { src = 4, dst = 5, laneOffset = 0, tint = cCyan, kind = PathwayKind.Generic },
                new EdgeDef { src = 7, dst = 5, laneOffset = -1, tint = cAmbr, kind = PathwayKind.S1ConflictToBg },
                new EdgeDef { src = 7, dst = 8, laneOffset = 0, tint = cMint, kind = PathwayKind.GwtToMotor },
                new EdgeDef { src = 5, dst = 8, laneOffset = 0, tint = cGate, kind = PathwayKind.BgGateToMotor },
                new EdgeDef { src = 4, dst = 8, laneOffset = 1, tint = cPink, kind = PathwayKind.GwtToBroca },
            };

            // -----------------------------------------------------------------
            // View 1: 9-Node Exclusive GPU Neural Suite (Matches HTML `VIEWS.gpu`)
            // -----------------------------------------------------------------
            var v1Nodes = new[]
            {
                new NodeDef { id = "m_fov", title = "TT-FovealEnc", sub = "16x16x4 → z32", x = 8, y = 14, w = 98, h = 66, rx = 16, accent = cCyan, role = 0 },
                new NodeDef { id = "m_slot", title = "ObjectSlot", sub = "BG0 + FG1..3", x = 8, y = 104, w = 98, h = 66, rx = 16, accent = cSky, role = 0 },
                new NodeDef { id = "m_topo", title = "TopoAdapter", sub = "ctx64·Spot16", x = 8, y = 194, w = 98, h = 66, rx = 16, accent = cEmer, role = 2 },

                new NodeDef { id = "m_rssm", title = "3-Head RSSM", sub = "GRU256·8x8", x = 123, y = 14, w = 98, h = 66, rx = 16, accent = cViol, role = 0 },
                new NodeDef { id = "m_strat", title = "Hazard·Strat", sub = "8履歴·Mode5", x = 123, y = 104, w = 98, h = 66, rx = 16, accent = cAmbr, role = 0 },
                new NodeDef { id = "m_gwt", title = "ActiveGWT", sub = "Ignition→w128", x = 123, y = 192, w = 98, h = 70, rx = 18, accent = cCyan, role = 1 },

                new NodeDef { id = "m_self", title = "CoarseSelf", sub = "HOT·meta_err9", x = 238, y = 14, w = 98, h = 66, rx = 16, accent = cSky, role = 2 },
                new NodeDef { id = "m_d3qn", title = "TT-D3QN", sub = "768D×8候補Q", x = 238, y = 104, w = 98, h = 66, rx = 16, accent = cMint, role = 1 },
                new NodeDef { id = "m_lang", title = "RVQ·Broca", sub = "32x8象徴·Bias0", x = 238, y = 194, w = 98, h = 66, rx = 16, accent = cPink, role = 0 },
            };
            var v1Edges = new[]
            {
                new EdgeDef { src = 0, dst = 1, laneOffset = 0, tint = cCyan, kind = PathwayKind.Generic },
                new EdgeDef { src = 1, dst = 2, laneOffset = 0, tint = cSky, kind = PathwayKind.Generic },
                new EdgeDef { src = 2, dst = 3, laneOffset = -1, tint = cEmer, kind = PathwayKind.Generic },
                new EdgeDef { src = 0, dst = 3, laneOffset = 0, tint = cCyan, kind = PathwayKind.Generic },
                new EdgeDef { src = 0, dst = 4, laneOffset = 1, tint = cAmbr, kind = PathwayKind.Generic },
                new EdgeDef { src = 2, dst = 5, laneOffset = 0, tint = cEmer, kind = PathwayKind.GwtSpec0Vis },
                new EdgeDef { src = 5, dst = 2, laneOffset = 1, tint = cSky, kind = PathwayKind.TopDownMod },
                new EdgeDef { src = 3, dst = 5, laneOffset = 0, tint = cViol, kind = PathwayKind.GwtSpec2Rssm },
                new EdgeDef { src = 4, dst = 5, laneOffset = 0, tint = cAmbr, kind = PathwayKind.GwtSpec34Strat },
                new EdgeDef { src = 5, dst = 6, laneOffset = 0, tint = cSky, kind = PathwayKind.Generic },
                new EdgeDef { src = 6, dst = 4, laneOffset = 1, tint = cAmbr, kind = PathwayKind.MetacogSelf },
                new EdgeDef { src = 5, dst = 7, laneOffset = 0, tint = cCyan, kind = PathwayKind.Generic },
                new EdgeDef { src = 3, dst = 7, laneOffset = -1, tint = cViol, kind = PathwayKind.RssmToValue },
                new EdgeDef { src = 5, dst = 8, laneOffset = 0, tint = cPink, kind = PathwayKind.GwtToBroca },
            };

            // -----------------------------------------------------------------
            // View 2: 6-Node Closed-Loop Motor Arbitration (Matches HTML `VIEWS.loop`)
            // -----------------------------------------------------------------
            var v2Nodes = new[]
            {
                new NodeDef { id = "sp_cand", title = "#3 海馬·小脳", sub = "8候補生成 60Hz", x = 8, y = 32, w = 100, h = 84, rx = 18, accent = cEmer, role = 0 },
                new NodeDef { id = "td_fb", title = "Top-Down帰還", sub = "視線·SNc可塑性", x = 8, y = 160, w = 100, h = 84, rx = 18, accent = cAmbr, role = 2 },

                new NodeDef { id = "s1_fast", title = "#4 S1 反射", sub = "ch0 24Hz (260ms)", x = 124, y = 14, w = 100, h = 68, rx = 16, accent = cMint, role = 0 },
                new NodeDef { id = "s2_gwt", title = "#5 GWT 熟慮", sub = "ch1 12Hz + Veto", x = 124, y = 102, w = 100, h = 70, rx = 18, accent = cCyan, role = 1 },
                new NodeDef { id = "bg_gate", title = "#3 STN-SNr門", sub = "Go vs Brake", x = 124, y = 192, w = 100, h = 68, rx = 16, accent = cGate, role = 2 },

                new NodeDef { id = "arb_wb", title = "#1 WorldBridge", sub = "ch1>ch0 調停\nUBMC 6DoF出力", x = 240, y = 82, w = 96, h = 110, rx = 20, accent = cCyan, role = 1 },
            };
            var v2Edges = new[]
            {
                new EdgeDef { src = 0, dst = 2, laneOffset = 0, tint = cEmer, kind = PathwayKind.Generic },
                new EdgeDef { src = 0, dst = 3, laneOffset = 0, tint = cEmer, kind = PathwayKind.GwtSpec1Spatial },
                new EdgeDef { src = 2, dst = 5, laneOffset = -1, tint = cMint, kind = PathwayKind.Generic },
                new EdgeDef { src = 2, dst = 4, laneOffset = 0, tint = cAmbr, kind = PathwayKind.S1ConflictToBg },
                new EdgeDef { src = 3, dst = 5, laneOffset = 0, tint = cCyan, kind = PathwayKind.GwtToMotor },
                new EdgeDef { src = 3, dst = 4, laneOffset = 0, tint = cCyan, kind = PathwayKind.Generic },
                new EdgeDef { src = 3, dst = 1, laneOffset = 1, tint = cAmbr, kind = PathwayKind.TopDownMod },
                new EdgeDef { src = 1, dst = 0, laneOffset = 0, tint = cAmbr, kind = PathwayKind.TopDownMod },
                new EdgeDef { src = 4, dst = 5, laneOffset = 1, tint = cGate, kind = PathwayKind.BgGateToMotor },
            };

            _views = new[]
            {
                new CircuitViewDef { tabLabel = "1:統合診断", nodes = v0Nodes, edges = v0Edges, selectedIdx = 4 },
                new CircuitViewDef { tabLabel = "2:GPU 9層", nodes = v1Nodes, edges = v1Edges, selectedIdx = 5 },
                new CircuitViewDef { tabLabel = "3:運動調停", nodes = v2Nodes, edges = v2Edges, selectedIdx = 3 },
            };

            for (int v = 0; v < _views.Length; v++)
            {
                int n = _views[v].nodes.Length;
                _nodeSmoothLum[v] = new float[n];
                _nodeHealth01[v] = new float[n];
                _nodeDiag[v] = new ModuleDiagState[n];
                _nodeLiveMetric[v] = new string[n];
                _nodeInspectorLine[v] = new string[n];
                for (int i = 0; i < n; i++)
                {
                    _nodeSmoothLum[v][i] = 0.14f;
                    _nodeHealth01[v][i] = 0.50f;
                    _nodeDiag[v][i] = ModuleDiagState.WeakOrUntrained;
                    _nodeLiveMetric[v][i] = "--";
                    _nodeInspectorLine[v][i] = $"{_views[v].nodes[i].title}: テレメトリ接続待ち...";
                }
            }
        }

        public Rect GetPanelRect()
        {
            float rightPanelPx = neuralHud != null ? neuralHud.ActiveRightPanelWidthPx : 0f;
            float stripBottom = 148f;
            if (dashboardUi != null)
            {
                Rect sr = dashboardUi.GetSpectateStatusStripRect();
                stripBottom = sr.yMax + 6f;
            }
            if (!showCircuitHud)
            {
                float cw = 182f;
                float cx = Mathf.Max(380f, Screen.width - rightPanelPx - cw - 10f);
                return new Rect(cx, stripBottom, cw, 24f);
            }

            float x0 = Mathf.Max(372f, Screen.width - rightPanelPx - PANEL_W - 8f);
            float maxH = Screen.height - stripBottom - 8f;
            float y0 = stripBottom;
            if (maxH < PANEL_H)
                y0 = Mathf.Max(8f, Screen.height - PANEL_H - 8f);
            return new Rect(x0, y0, PANEL_W, PANEL_H);
        }

        private void Update()
        {
            if (coordinator == null || neuralHud == null)
                AutoBind();
            if (!IsVisibleInSessionNow)
                return;

#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null && GUIUtility.keyboardControl == 0 && kb.f7Key.wasPressedThisFrame)
            {
                showCircuitHud = !showCircuitHud;
                if (showCircuitHud) _nextRasterTime = 0f;
            }
#endif

            // Strictly ZERO CPU/GPU rasterization work when collapsed!
            if (!showCircuitHud)
                return;

            if (Time.unscaledTime >= _nextRasterTime)
            {
                _nextRasterTime = Time.unscaledTime + 0.045f; // ~22 Hz smooth diagnostic + pulse animation
                var snap = neuralHud != null ? neuralHud.LatestUbntSnapshot : null;
                var wmSnap = neuralHud != null ? neuralHud.LatestUbwmSnapshot : null;
                RefreshCircuitCanvasNow(snap, wmSnap, Time.unscaledTime);
            }
        }

        public bool RefreshCircuitCanvasNow(
            InkBlitzNeuralVisualizerHud.UbntSnapshot snap,
            InkBlitzNeuralVisualizerHud.UbwmSnapshot wmSnap,
            float animTimeSec)
        {
            if (!showCircuitHud)
                return false;

            BuildCircuitDefinitions();
            EnsureCanvasBuffer();

            int vIdx = Mathf.Clamp(activeViewIndex, 0, _views.Length - 1);
            EvaluateTheoreticalCompetenceAndActivity(snap, wmSnap);
            RasterizeCircuitView(vIdx, snap, animTimeSec);
            CircuitRasterCount++;
            return true;
        }

        // =====================================================================
        // 1. Theoretical Competence & Fault Evaluator (`OK` / `WEAK` / `WARN`)
        // =====================================================================
        private void EvaluateTheoreticalCompetenceAndActivity(
            InkBlitzNeuralVisualizerHud.UbntSnapshot s,
            InkBlitzNeuralVisualizerHud.UbwmSnapshot wm)
        {
            _loopClosureFlash = Mathf.Max(0f, _loopClosureFlash - 0.08f);
            _brocaSymFlash = Mathf.Max(0f, _brocaSymFlash - 0.07f);

            if (s == null)
                return;

            // Update rolling temporal statistics only once per new `UBNT` packet (`s.seq`)
            if (s.seq != _lastIngestedSeq || !_hasPrevVecs)
            {
                _lastIngestedSeq = s.seq;
                if (s.episodeId != _lastEpisodeId)
                {
                    _lastEpisodeId = s.episodeId;
                    _framesInEp = 0;
                    _consecutiveGateHoldFrames = 0;
                    _consecutiveGateOpenFrames = 0;
                }
                _framesInEp++;

                if (_hasPrevVecs)
                {
                    float dz = MeanAbsDiff(s.latentZ, _prevZ);
                    float dh = MeanAbsDiff(s.rssmH, _prevRssm);
                    float dm = MeanAbsDiff(s.mecGrid, _prevMec);
                    float ds = MeanAbsDiff(s.selfState, _prevSelf);
                    float dp = Vector3.Distance(s.believedSelfPos, _prevBelievedPos);
                    float dIgn = Mathf.Abs(s.ignition - _prevIgn);
                    float dSt = Mathf.Abs(s.gwtAlpha[4] - _prevAlpha4);

                    _emaDeltaZ = Mathf.Lerp(_emaDeltaZ, dz, 0.22f);
                    _emaDeltaRssm = Mathf.Lerp(_emaDeltaRssm, dh, 0.22f);
                    _emaDeltaMec = Mathf.Lerp(_emaDeltaMec, dm, 0.22f);
                    _emaDeltaPos = Mathf.Lerp(_emaDeltaPos, dp, 0.22f);
                    _emaDeltaSelf = Mathf.Lerp(_emaDeltaSelf, ds, 0.22f);
                    _emaIgnChange = Mathf.Lerp(_emaIgnChange, dIgn, 0.18f);
                    _emaStratChange = Mathf.Lerp(_emaStratChange, dSt, 0.18f);
                }
                Array.Copy(s.latentZ, _prevZ, 32);
                Array.Copy(s.rssmH, _prevRssm, 256);
                Array.Copy(s.mecGrid, _prevMec, 256);
                Array.Copy(s.selfState, _prevSelf, 8);
                _prevBelievedPos = s.believedSelfPos;
                _prevIgn = s.ignition;
                _prevAlpha4 = s.gwtAlpha[4];
                _hasPrevVecs = true;

                if (s.commitGateOpen)
                {
                    _consecutiveGateOpenFrames++;
                    _consecutiveGateHoldFrames = 0;
                }
                else
                {
                    _consecutiveGateHoldFrames++;
                    _consecutiveGateOpenFrames = 0;
                }

                _protoSymbolBitmask |= (1u << (s.protoSymId & 31));
                _candUsedBitmask |= (byte)(1 << (s.chosenCand & 7));

                if (s.loopClosuresTotal > _lastLoopClosures)
                {
                    if (_lastLoopClosures > 0) _loopClosureFlash = 1f;
                    _lastLoopClosures = s.loopClosuresTotal;
                }
                if (s.protoSymId != _lastProtoSym)
                {
                    if (_lastProtoSym >= 0) _brocaSymFlash = 0.90f;
                    _lastProtoSym = s.protoSymId;
                }
            }

            // Instantaneous spread & error metrics
            float qMin = s.qValues[0], qMax = s.qValues[0], qAbsMax = Mathf.Abs(s.qValues[0]);
            float maxPi = s.policyProbs[0];
            for (int c = 1; c < 8; c++)
            {
                float q = s.qValues[c];
                if (q < qMin) qMin = q;
                if (q > qMax) qMax = q;
                if (Mathf.Abs(q) > qAbsMax) qAbsMax = Mathf.Abs(q);
                if (s.policyProbs[c] > maxPi) maxPi = s.policyProbs[c];
            }
            float qSpread = qMax - qMin;
            _emaQSpread = Mathf.Lerp(_emaQSpread, qSpread, 0.28f);

            float visMae = wm != null ? wm.visPredMae : 0.16f;
            float depthMaeM = wm != null ? wm.depthPredMaeM : 3.2f;
            float cfSpread = wm != null ? wm.actionCfSpread : 0.04f;
            if (wm != null)
            {
                _emaVisMae = Mathf.Lerp(_emaVisMae, visMae, 0.25f);
                _emaDepthMaeM = Mathf.Lerp(_emaDepthMaeM, depthMaeM, 0.25f);
                _emaCfSpread = Mathf.Lerp(_emaCfSpread, cfSpread, 0.25f);
            }

            bool warmedUp = _framesInEp >= 10;
            float zStd = StdDev(s.latentZ);
            float zTdNorm = Norm(s.zDelta);
            float driftM = s.poseDriftVec.magnitude;
            int bestC = Mathf.Clamp(s.chosenCand, 0, 7);
            int fastC = Mathf.Clamp(s.s1FastCand, 0, 7);
            int symDiversity = PopCount32(_protoSymbolBitmask);

            // --- Evaluate Each Core Brain Module's Functional Competence (`health01`, `diagState`) ---

            // [0] Fovea + 4-Slot Visual Predictive Cortex:
            // Healthy when `z` has non-collapsed dimensional spread (`zStd > 0.06`), temporal responsiveness, AND `visMae < 0.165`
            bool fovFrozen = warmedUp && _emaDeltaPos > 0.03f && _emaDeltaZ < 0.0005f;
            float fovPredQual = wm != null ? Mathf.Clamp01((0.21f - visMae) / 0.12f) : 0.55f;
            float fovSpreadQual = Mathf.Clamp01(zStd / 0.16f);
            float hFov = fovFrozen ? 0.15f : Mathf.Clamp01(0.45f * fovSpreadQual + 0.55f * fovPredQual);
            ModuleDiagState stFov = fovFrozen
                ? ModuleDiagState.FaultOrFrozen
                : (hFov >= 0.48f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained);

            // [1] Hippocampus MEC + 4-Layer Map + EKF-SLAM:
            // Healthy when MEC grid phase shifts when moving (`_emaDeltaMec > 0.002` when moving) and SLAM drift < 3.0m
            bool slamFault = driftM > 3.2f || (warmedUp && _emaDeltaPos > 0.08f && _emaDeltaMec < 0.0004f);
            float hMec = slamFault
                ? 0.18f
                : Mathf.Clamp01(0.55f * Mathf.Clamp01(Rms(s.mecGrid) * 2.8f) + 0.45f * Mathf.Clamp01(1f - driftM / 2.8f));
            ModuleDiagState stMec = slamFault
                ? ModuleDiagState.FaultOrFrozen
                : (hMec >= 0.42f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained);

            // [2] Amygdala Hazard + Prefrontal Strategic Transformer:
            // Healthy when strategic attention `gwtAlpha[3..4]` + `zDelta` actively participate without NaN/saturation
            float stratShare = s.gwtAlpha[3] + s.gwtAlpha[4];
            float hStrat = Mathf.Clamp01(0.45f * Mathf.Clamp01(stratShare * 2.4f) + 0.35f * Mathf.Clamp01(zTdNorm * 1.8f) + 0.20f * Mathf.Clamp01(_emaStratChange * 12f + 0.35f));
            ModuleDiagState stStrat = hStrat >= 0.40f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained;

            // [3] 3-Head Causal Categorical RSSM (World Model):
            // Healthy when 8-candidate counterfactual futures genuinely branch (`cfSpread >= 0.045`) AND depth forecast error is calibrated (`depthMaeM < 3.5m`)
            bool rssmFrozen = warmedUp && _emaDeltaRssm < 0.0003f && _emaDeltaZ > 0.005f;
            float cfQual = Mathf.Clamp01((cfSpread - 0.015f) / 0.14f);
            float dMaeQual = wm != null ? Mathf.Clamp01((4.4f - depthMaeM) / 2.8f) : 0.50f;
            float hRssm = rssmFrozen ? 0.12f : Mathf.Clamp01(0.55f * cfQual + 0.45f * dMaeQual);
            ModuleDiagState stRssm = rssmFrozen
                ? ModuleDiagState.FaultOrFrozen
                : (hRssm >= 0.45f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained);

            // [4] ActiveGWT (5-Specialist Conscious Workspace & Ignition):
            // Healthy when Φ_G > 0.05 bits (genuine integration) AND ignition is neither dead (<0.04 always) nor stuck saturated (>0.98 with 0 specialist competition)
            float alphaMax = 0f;
            for (int k = 0; k < 5; k++) if (s.gwtAlpha[k] > alphaMax) alphaMax = s.gwtAlpha[k];
            bool gwtMonopolized = warmedUp && alphaMax > 0.96f && s.phiBits < 0.02f;
            float phiQual = Mathf.Clamp01(s.phiBits / 0.25f);
            float ignQual = (s.ignition > 0.06f && s.ignition < 0.98f) ? 0.90f : 0.35f;
            float hGwt = gwtMonopolized ? 0.20f : Mathf.Clamp01(0.55f * phiQual + 0.45f * ignQual);
            ModuleDiagState stGwt = gwtMonopolized
                ? ModuleDiagState.FaultOrFrozen
                : (hGwt >= 0.44f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained);

            // [5] Basal Ganglia STN-SNr 60Hz Timing Gate:
            // Fault (`WARN`) if gate stays locked (`commitGateOpen == false`) for >= 38 telemetry frames (~2.5s freeze)!
            bool bgFreezeFault = _consecutiveGateHoldFrames >= 38;
            float hBg = bgFreezeFault
                ? 0.15f
                : Mathf.Clamp01(s.commitGateOpen ? (0.65f + 0.35f * Mathf.Clamp01(s.goDrive)) : (0.55f + 0.30f * Mathf.Clamp01(s.stnBrake)));
            ModuleDiagState stBg = bgFreezeFault ? ModuleDiagState.FaultOrFrozen : ModuleDiagState.Healthy;

            // [6] CoarseSelfModel (HOT Metacognition & Efference Copy):
            // Healthy when `selfSurprise` stays bounded (< 1.2) and `selfState` tracks bodily posture
            bool selfRunaway = s.selfSurprise > 1.35f;
            float hSelf = selfRunaway
                ? 0.20f
                : Mathf.Clamp01(0.55f * Mathf.Clamp01(1.1f - s.selfSurprise) + 0.45f * Mathf.Clamp01(Rms(s.selfState) * 2.2f));
            ModuleDiagState stSelf = selfRunaway
                ? ModuleDiagState.FaultOrFrozen
                : (hSelf >= 0.42f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained);

            // [7] S1 Reflex + TT-D3QN Value Evaluator:
            // Untrained (`WEAK`) when `qSpread < 0.10` & `entropyNorm > 0.92` (flat coin-flip); `WARN` if Q-spike `|Q| > 35` or `overfitGuardTripped`!
            bool qSpikeFault = qAbsMax > 35f || s.overfitGuardTripped;
            float qDiscQual = Mathf.Clamp01((qSpread - 0.04f) / 0.45f);
            float entQual = Mathf.Clamp01((0.97f - s.entropyNorm) / 0.35f);
            float hD3qn = qSpikeFault ? 0.10f : Mathf.Clamp01(0.65f * qDiscQual + 0.35f * entQual);
            ModuleDiagState stD3qn = qSpikeFault
                ? ModuleDiagState.FaultOrFrozen
                : (hD3qn >= 0.42f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained);

            // [8] Motor Output + RVQ Proto-Symbols + Plan-A Broca:
            float symQual = (_framesInEp < 8) ? 0.70f : Mathf.Clamp01(symDiversity / 2.0f);
            float hMotor = Mathf.Clamp01(0.55f * (s.commitGateOpen ? 0.90f : 0.50f) + 0.45f * symQual);
            ModuleDiagState stMotor = hMotor >= 0.45f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained;

            string gateLbl = s.commitGateOpen ? "GO" : HoldShortName(s.holdReasonId);
            string verbLbl = $"M{bestC}:{ShortVerb(s.candVerbs[bestC])}";
            float rssmValShift = MeanAbsDiff(s.qValues, s.qRawValues);

            // --- View 0: 9-Node Whole-Brain Diagnostic Summary ---
            ModuleDiagState[] d0 = { stFov, stMec, stStrat, stRssm, stGwt, stBg, stSelf, stD3qn, stMotor };
            float[] h0 = { hFov, hMec, hStrat, hRssm, hGwt, hBg, hSelf, hD3qn, hMotor };
            float[] act0 = {
                ComposeNodeLum(stFov, hFov, Mathf.Clamp01(zStd * 3.0f)),
                ComposeNodeLum(stMec, hMec, Mathf.Clamp01(Rms(s.mecGrid) * 2.5f + _loopClosureFlash * 0.4f)),
                ComposeNodeLum(stStrat, hStrat, Mathf.Clamp01(stratShare * 1.9f + s.believedHazard * 0.4f)),
                ComposeNodeLum(stRssm, hRssm, Mathf.Clamp01(cfSpread * 2.4f + s.worldSurprise * 0.4f)),
                ComposeNodeLum(stGwt, hGwt, Mathf.Clamp01(0.25f + 0.75f * s.ignition)),
                ComposeNodeLum(stBg, hBg, s.commitGateOpen ? 0.82f : 0.65f),
                ComposeNodeLum(stSelf, hSelf, Mathf.Clamp01(0.35f + Rms(s.selfState) * 1.4f)),
                ComposeNodeLum(stD3qn, hD3qn, Mathf.Clamp01(0.25f + maxPi * 0.85f)),
                ComposeNodeLum(stMotor, hMotor, Mathf.Clamp01((s.commitGateOpen ? 0.75f : 0.30f) + _brocaSymFlash * 0.3f))
            };
            string[] m0 = {
                $"{DiagTag(stFov)} MAE:{visMae:F2}",
                $"{DiagTag(stMec)} SL:{s.loopClosuresTotal}",
                $"{DiagTag(stStrat)} α:{stratShare:F2}",
                s.vetoActive ? "! VETO発動" : $"{DiagTag(stRssm)} ΔCF:{cfSpread:F2}",
                $"{DiagTag(stGwt)} Φ:{s.phiBits:F2}b",
                $"{DiagTag(stBg)} {gateLbl}",
                $"{DiagTag(stSelf)} e_s:{s.selfSurprise:F2}",
                $"{DiagTag(stD3qn)} ΔQ:{qSpread:F2}",
                $"{DiagTag(stMotor)} {verbLbl}"
            };
            string[] info0 = {
                $"Fovea·4Slot [{DiagWord(stFov)}] | 1step誤差 MAE={visMae:F3} z分散={zStd:F2} Δz_t={_emaDeltaZ:F3}",
                $"海馬 MEC·地図 [{DiagWord(stMec)}] | 座標移動追従 ΔMEC={_emaDeltaMec:F3} Drift={driftM:F2}m Loop={s.loopClosuresTotal}",
                $"扁桃体·StratTF [{DiagWord(stStrat)}] | 戦略注意 α3+4={stratShare:F2} TopDown||Δz||={zTdNorm:F2} Haz={s.believedHazard:F2}",
                $"3-Head RSSM [{DiagWord(stRssm)}] | 行動分岐 ΔCF={cfSpread:F3} 深度予測誤差={depthMaeM:F2}m KL={s.worldSurprise:F2}",
                $"ActiveGWT [{DiagWord(stGwt)}] | 統合情報量 Φ_G={s.phiBits:F2}b 点火={s.ignition:F2} 主導α_max={alphaMax:F2}",
                $"基底核 STN門 [{DiagWord(stBg)}] | {gateLbl} Go={s.goDrive:F2} Brake={s.stnBrake:F2} 連続Hold={_consecutiveGateHoldFrames}f",
                $"SelfModel [{DiagWord(stSelf)}] | 自己予測誤差 e_s={s.selfSurprise:F3} 状態変化={_emaDeltaSelf:F3}",
                $"S1·TT-D3QN [{DiagWord(stD3qn)}] | Q識別幅 ΔQ={qSpread:F2} H={s.entropyNorm:F2} RSSM寄与={rssmValShift:F2}",
                $"Motor·RVQ [{DiagWord(stMotor)}] | {verbLbl} 象徴多様性={symDiversity}種 PlanA干渉=0.0"
            };
            ApplyViewDiagnostics(0, d0, h0, act0, m0, info0);

            int okCnt = 0, weakCnt = 0, warnCnt = 0;
            float hSum = 0f;
            for (int i = 0; i < d0.Length; i++)
            {
                hSum += h0[i];
                if (d0[i] == ModuleDiagState.Healthy) okCnt++;
                else if (d0[i] == ModuleDiagState.FaultOrFrozen) warnCnt++;
                else weakCnt++;
            }
            HealthyNodeCount = okCnt;
            WeakNodeCount = weakCnt;
            WarnNodeCount = warnCnt;
            OverallHealthScore01 = hSum / d0.Length;

            // --- View 1: GPU 9-Model ---
            float hEnc = fovFrozen ? 0.15f : Mathf.Clamp01(fovSpreadQual * 0.8f + 0.2f);
            ModuleDiagState stEnc = fovFrozen ? ModuleDiagState.FaultOrFrozen : (hEnc >= 0.45f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained);
            ModuleDiagState stSlot = fovPredQual >= 0.42f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained;
            float hTopo = Mathf.Clamp01(0.5f * hMec + 0.5f * Mathf.Clamp01(s.gwtAlpha[0] * 3f + zTdNorm));
            ModuleDiagState stTopo = hTopo >= 0.42f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained;

            ModuleDiagState[] d1 = { stEnc, stSlot, stTopo, stRssm, stStrat, stGwt, stSelf, stD3qn, stMotor };
            float[] h1 = { hEnc, fovPredQual, hTopo, hRssm, hStrat, hGwt, hSelf, hD3qn, hMotor };
            float[] act1 = {
                ComposeNodeLum(stEnc, hEnc, zStd * 2.8f),
                ComposeNodeLum(stSlot, fovPredQual, 1f - visMae * 2f),
                ComposeNodeLum(stTopo, hTopo, s.gwtAlpha[0] * 2.5f + zTdNorm),
                act0[3], act0[2], act0[4], act0[6], act0[7],
                ComposeNodeLum(stMotor, symQual, (s.endogenousActive ? 0.55f : 0.15f) + _brocaSymFlash * 0.45f)
            };
            string[] m1 = {
                $"{DiagTag(stEnc)} σz:{zStd:F2}",
                $"{DiagTag(stSlot)} MAE:{visMae:F2}",
                $"{DiagTag(stTopo)} α0:{s.gwtAlpha[0]:F2}",
                m0[3], m0[2], m0[4], m0[6], m0[7],
                $"{DiagTag(stMotor)} S#{s.protoSymId}({symDiversity})"
            };
            ApplyViewDiagnostics(1, d1, h1, act1, m1, info0);

            // --- View 2: 6-Node Closed-Loop Motor Arbitration ---
            float hTd = Mathf.Clamp01(zTdNorm * 1.9f + stratShare);
            ModuleDiagState stTd = hTd >= 0.35f ? ModuleDiagState.Healthy : ModuleDiagState.WeakOrUntrained;
            ModuleDiagState[] d2 = { stMec, stTd, stD3qn, stGwt, stBg, stMotor };
            float[] h2 = { hMec, hTd, hD3qn, hGwt, hBg, hMotor };
            float[] act2 = {
                act0[1],
                ComposeNodeLum(stTd, hTd, zTdNorm * 2.2f),
                act0[7], act0[4], act0[5], act0[8]
            };
            string[] m2 = {
                $"{DiagTag(stMec)} 8候補",
                $"{DiagTag(stTd)} ||Δz||:{zTdNorm:F2}",
                $"{DiagTag(stD3qn)} S1:M{fastC}",
                $"{DiagTag(stGwt)} GWT:M{bestC}",
                $"{DiagTag(stBg)} {gateLbl}",
                $"{DiagTag(stMotor)} {verbLbl}"
            };
            string[] info2 = {
                info0[1],
                $"Top-Down帰還 [{DiagWord(stTd)}] | 知覚変調 ||z_td||={zTdNorm:F3} 戦略比率={stratShare:F2}",
                $"#4 S1反射 [{DiagWord(stD3qn)}] | Fast=M{fastC} GWT=M{bestC} ({(fastC == bestC ? "一致" : "競合→STN")}) ΔQ={qSpread:F2}",
                info0[4], info0[5], info0[8]
            };
            ApplyViewDiagnostics(2, d2, h2, act2, m2, info2);
        }

        private static float ComposeNodeLum(ModuleDiagState st, float health01, float rawAct01)
        {
            if (st == ModuleDiagState.WeakOrUntrained)
                return Mathf.Clamp(0.05f + 0.14f * health01, 0.04f, 0.19f); // Honestly dim when untrained/flat!
            if (st == ModuleDiagState.FaultOrFrozen)
                return 0.78f; // Strong warning highlight!
            return Mathf.Clamp(0.34f + 0.36f * health01 + 0.30f * Mathf.Clamp01(rawAct01), 0.34f, 1.0f);
        }

        private static string DiagTag(ModuleDiagState st)
        {
            switch (st)
            {
                case ModuleDiagState.Healthy: return "●";
                case ModuleDiagState.FaultOrFrozen: return "!";
                default: return "◌";
            }
        }

        private static string DiagWord(ModuleDiagState st)
        {
            switch (st)
            {
                case ModuleDiagState.Healthy: return "正常機能 OK";
                case ModuleDiagState.FaultOrFrozen: return "固着/異常 WARN";
                default: return "未学習/弱 WEAK";
            }
        }

        private void ApplyViewDiagnostics(
            int v,
            ModuleDiagState[] diag,
            float[] health,
            float[] targetLum,
            string[] metrics,
            string[] infos)
        {
            float[] sm = _nodeSmoothLum[v];
            for (int i = 0; i < sm.Length && i < targetLum.Length; i++)
            {
                float rate = targetLum[i] > sm[i] ? 0.42f : 0.22f;
                sm[i] = Mathf.Lerp(sm[i], Mathf.Clamp01(targetLum[i]), rate);
                _nodeDiag[v][i] = diag[i];
                _nodeHealth01[v][i] = Mathf.Clamp01(health[i]);
                _nodeLiveMetric[v][i] = metrics[i];
                if (i < infos.Length) _nodeInspectorLine[v][i] = infos[i];
            }
        }

        // =====================================================================
        // 2. Exact Mathematical Pathway Wire Coupling & Software SDF Rasterizer
        // =====================================================================
        private float ComputeEdgePathwayCoupling(
            EdgeDef ed,
            float[] nodeLums,
            InkBlitzNeuralVisualizerHud.UbntSnapshot s,
            out bool blockedAtMidGate)
        {
            blockedAtMidGate = false;
            if (s == null)
                return 0.10f;

            switch (ed.kind)
            {
                case PathwayKind.GwtSpec0Vis:
                    return Mathf.Clamp01(s.gwtAlpha[0] * 2.6f) * Mathf.Clamp01(nodeLums[ed.src] * 1.4f);
                case PathwayKind.GwtSpec1Spatial:
                    return Mathf.Clamp01(s.gwtAlpha[1] * 2.6f) * Mathf.Clamp01(nodeLums[ed.src] * 1.4f);
                case PathwayKind.GwtSpec2Rssm:
                    return Mathf.Clamp01(s.gwtAlpha[2] * 2.6f) * Mathf.Clamp01(nodeLums[ed.src] * 1.4f);
                case PathwayKind.GwtSpec34Strat:
                    return Mathf.Clamp01((s.gwtAlpha[3] + s.gwtAlpha[4]) * 2.0f) * Mathf.Clamp01(nodeLums[ed.src] * 1.4f);
                case PathwayKind.TopDownMod:
                    return Mathf.Clamp01(Norm(s.zDelta) * 1.8f);
                case PathwayKind.RssmToValue:
                    return Mathf.Clamp01(MeanAbsDiff(s.qValues, s.qRawValues) * 2.6f + (s.vetoActive ? 0.65f : 0f));
                case PathwayKind.MetacogSelf:
                    return Mathf.Clamp01(s.selfSurprise * 2.0f + 0.12f);
                case PathwayKind.S1ConflictToBg:
                    return (s.s1FastCand != s.chosenCand) ? 0.92f : 0.16f;
                case PathwayKind.BgGateToMotor:
                    if (!s.commitGateOpen)
                    {
                        blockedAtMidGate = true;
                        return Mathf.Clamp01(0.45f + s.stnBrake * 0.45f);
                    }
                    return Mathf.Clamp01(0.65f + s.goDrive * 0.35f);
                case PathwayKind.GwtToMotor:
                    return s.commitGateOpen ? Mathf.Clamp01(nodeLums[ed.src]) : 0.22f;
                case PathwayKind.GwtToBroca:
                    return s.endogenousActive ? Mathf.Clamp01(0.28f + _brocaSymFlash * 0.72f) : 0.05f;
                default:
                    return Mathf.Clamp01(Mathf.Sqrt(Mathf.Max(0.01f, nodeLums[ed.src] * nodeLums[ed.dst])));
            }
        }

        private void RasterizeCircuitView(int vIdx, InkBlitzNeuralVisualizerHud.UbntSnapshot snap, float tSec)
        {
            ClearCanvas(new Color32(14, 17, 21, 255));
            Color32 gridDot = new Color32(25, 30, 39, 255);
            for (int gy = 10; gy < CANVAS_H - 6; gy += 16)
            {
                int row = (CANVAS_H - 1 - gy) * CANVAS_W;
                for (int gx = 10; gx < CANVAS_W - 6; gx += 16)
                    _pixels[row + gx] = gridDot;
            }

            CircuitViewDef view = _views[vIdx];
            float[] lums = _nodeSmoothLum[vIdx];
            ModuleDiagState[] diags = _nodeDiag[vIdx];
            float[] healths = _nodeHealth01[vIdx];
            bool vetoNow = snap != null && snap.vetoActive;
            float warnPulse = 0.55f + 0.45f * Mathf.Sin(tSec * 6.5f);

            // 1. Orthogonal Rounded Wires Driven by True Pathway Coupling
            for (int e = 0; e < view.edges.Length; e++)
            {
                EdgeDef ed = view.edges[e];
                NodeDef n1 = view.nodes[ed.src];
                NodeDef n2 = view.nodes[ed.dst];
                float flux = ComputeEdgePathwayCoupling(ed, lums, snap, out bool blockedGate);
                bool isSelPeer = (ed.src == view.selectedIdx || ed.dst == view.selectedIdx);

                Color32 wireCol = ed.tint;
                if (vetoNow && ed.kind == PathwayKind.RssmToValue)
                    wireCol = new Color32(255, 86, 86, 255);
                else if (blockedGate)
                    wireCol = new Color32(255, 148, 68, 255);

                DrawOrthogonalGlowingEdge(n1, n2, ed.laneOffset, wireCol, flux, isSelPeer, blockedGate, tSec + e * 0.19f);
            }

            // 2. Nodes Driven by Functional Competence (`OK` / `WEAK` / `WARN`)
            for (int i = 0; i < view.nodes.Length; i++)
            {
                NodeDef nd = view.nodes[i];
                ModuleDiagState st = diags[i];
                float lum = lums[i];
                float h01 = healths[i];
                bool isSel = (i == view.selectedIdx);

                Color32 acc = nd.accent;
                if (st == ModuleDiagState.FaultOrFrozen)
                {
                    acc = new Color32(255, 108, 68, 255);
                    lum = Mathf.Clamp01(0.55f + 0.45f * warnPulse);
                }
                else if (st == ModuleDiagState.WeakOrUntrained)
                {
                    // Desaturated cool slate when untrained / flat!
                    acc = new Color32(92, 104, 124, 255);
                }
                else if (vetoNow && nd.id.Contains("rssm"))
                {
                    acc = new Color32(255, 92, 92, 255);
                }
                else if (snap != null && !snap.commitGateOpen && nd.id.Contains("bg_"))
                {
                    acc = new Color32(255, 158, 76, 255);
                }

                DrawDiagnosticRoundedNode(nd, acc, st, lum, h01, isSel);
            }

            _canvasTex.SetPixels32(_pixels);
            _canvasTex.Apply(false);
        }

        private void DrawOrthogonalGlowingEdge(
            NodeDef n1,
            NodeDef n2,
            int laneOff,
            Color32 accent,
            float flux01,
            bool emphasize,
            bool blockedAtMidGate,
            float phaseT)
        {
            float c1x = n1.x + n1.w * 0.5f;
            float c1y = n1.y + n1.h * 0.5f;
            float c2x = n2.x + n2.w * 0.5f;
            float c2y = n2.y + n2.h * 0.5f;
            bool sameCol = Mathf.Abs(c1x - c2x) < 24f;

            Vector2 p0, p1, p2, p3;
            if (sameCol)
            {
                bool leftSide = c1x < CANVAS_W * 0.5f;
                float sx = leftSide ? n1.x + 14f : n1.x + n1.w - 14f;
                float sy = c2y > c1y ? n1.y + n1.h + 1f : n1.y - 1f;
                float ey = c2y > c1y ? n2.y - 1f : n2.y + n2.h + 1f;
                float mx = sx + laneOff * 6f;
                p0 = new Vector2(mx, sy);
                p1 = new Vector2(mx, Mathf.Lerp(sy, ey, 0.33f));
                p2 = new Vector2(mx, Mathf.Lerp(sy, ey, 0.67f));
                p3 = new Vector2(mx, ey);
            }
            else
            {
                bool right = c2x > c1x;
                float sx = right ? n1.x + n1.w + 1f : n1.x - 1f;
                float ex = right ? n2.x - 2f : n2.x + n2.w + 2f;
                float sy = c1y + laneOff * 6f;
                float ey = c2y - laneOff * 5f;
                float midX = 0.5f * (sx + ex) + laneOff * 4f;
                p0 = new Vector2(sx, sy);
                p1 = new Vector2(midX, sy);
                p2 = new Vector2(midX, ey);
                p3 = new Vector2(ex, ey);
            }

            byte baseA = (byte)(emphasize ? 165 : 95);
            DrawOrthoRoundedPolyline(p0, p1, p2, p3, new Color32(44, 50, 62, baseA), glowRadius: 0f, maxU: 1f);

            float effGlow = Mathf.Clamp01(flux01 * (emphasize ? 1.15f : 0.92f));
            if (effGlow > 0.12f)
            {
                float maxReachU = blockedAtMidGate ? 0.52f : 1f;
                byte haloA = (byte)Mathf.Clamp(Mathf.RoundToInt(effGlow * 76f), 0, 98);
                byte coreA = (byte)Mathf.Clamp(Mathf.RoundToInt(60f + effGlow * 190f), 0, 252);
                DrawOrthoRoundedPolyline(p0, p1, p2, p3, new Color32(accent.r, accent.g, accent.b, haloA), glowRadius: 2.2f, maxU: maxReachU);
                DrawOrthoRoundedPolyline(p0, p1, p2, p3, new Color32(accent.r, accent.g, accent.b, coreA), glowRadius: 0.65f, maxU: maxReachU);

                if (blockedAtMidGate)
                {
                    // Draw glowing amber gate-block bar `┤` right at u=0.52 where STN-SNr holds the pulse!
                    Vector2 bp = SampleOrthoPolyline(p0, p1, p2, p3, 0.52f);
                    DrawGlowSegment(bp + new Vector2(0f, -4f), bp + new Vector2(0f, 4f), 1.4f, new Color32(255, 165, 76, 235));
                    AddGlowDisc(bp.x - 1.5f, bp.y, 2.5f, new Color32(255, 200, 120, 200));
                }
                else
                {
                    float speed = 0.50f + effGlow * 1.25f;
                    float u = Mathf.Repeat(phaseT * speed, 1f);
                    Vector2 pt = SampleOrthoPolyline(p0, p1, p2, p3, u);
                    byte pulseA = (byte)Mathf.Clamp(Mathf.RoundToInt(125f + effGlow * 130f), 0, 255);
                    AddGlowDisc(pt.x, pt.y, 4.1f, new Color32(accent.r, accent.g, accent.b, (byte)(pulseA / 2)));
                    AddGlowDisc(pt.x, pt.y, 1.8f, new Color32(238, 249, 255, pulseA));
                }
            }
        }

        private void DrawOrthoRoundedPolyline(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, Color32 col, float glowRadius, float maxU)
        {
            const int STEPS = 36;
            int maxStep = Mathf.Clamp(Mathf.RoundToInt(STEPS * Mathf.Clamp01(maxU)), 1, STEPS);
            Vector2 prev = SampleOrthoPolyline(p0, p1, p2, p3, 0f);
            for (int k = 1; k <= maxStep; k++)
            {
                Vector2 cur = SampleOrthoPolyline(p0, p1, p2, p3, k / (float)STEPS);
                if (glowRadius <= 0.1f)
                    DrawThinSegment(prev, cur, col);
                else
                    DrawGlowSegment(prev, cur, glowRadius, col);
                prev = cur;
            }
        }

        private static Vector2 SampleOrthoPolyline(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float u)
        {
            u = Mathf.Clamp01(u);
            float d0 = Vector2.Distance(p0, p1);
            float d1 = Vector2.Distance(p1, p2);
            float d2 = Vector2.Distance(p2, p3);
            float tot = Mathf.Max(1e-3f, d0 + d1 + d2);
            float s = u * tot;

            float cornerR = Mathf.Min(7f, 0.45f * Mathf.Min(Mathf.Max(1f, d0), Mathf.Max(1f, d1)));
            if (d1 < 2f)
                return Vector2.Lerp(p0, p3, u);

            if (s < d0 - cornerR)
                return Vector2.Lerp(p0, p1, s / Mathf.Max(1e-3f, d0));
            if (s < d0 + cornerR)
            {
                float t = (s - (d0 - cornerR)) / Mathf.Max(1e-3f, 2f * cornerR);
                Vector2 a = Vector2.MoveTowards(p1, p0, cornerR);
                Vector2 b = Vector2.MoveTowards(p1, p2, cornerR);
                return (1f - t) * (1f - t) * a + 2f * (1f - t) * t * p1 + t * t * b;
            }
            if (s < d0 + d1 - cornerR)
                return Vector2.Lerp(p1, p2, (s - d0) / Mathf.Max(1e-3f, d1));
            if (s < d0 + d1 + cornerR)
            {
                float t = (s - (d0 + d1 - cornerR)) / Mathf.Max(1e-3f, 2f * cornerR);
                Vector2 a = Vector2.MoveTowards(p2, p1, cornerR);
                Vector2 b = Vector2.MoveTowards(p2, p3, cornerR);
                return (1f - t) * (1f - t) * a + 2f * (1f - t) * t * p2 + t * t * b;
            }
            return Vector2.Lerp(p2, p3, (s - d0 - d1) / Mathf.Max(1e-3f, d2));
        }

        private void DrawDiagnosticRoundedNode(
            NodeDef nd,
            Color32 accent,
            ModuleDiagState state,
            float lum01,
            float health01,
            bool isSelected)
        {
            float haloPad = 7f;
            int x0 = Mathf.Max(0, Mathf.FloorToInt(nd.x - haloPad));
            int x1 = Mathf.Min(CANVAS_W - 1, Mathf.CeilToInt(nd.x + nd.w + haloPad));
            int y0 = Mathf.Max(0, Mathf.FloorToInt(nd.y - haloPad));
            int y1 = Mathf.Min(CANVAS_H - 1, Mathf.CeilToInt(nd.y + nd.h + haloPad));

            float cx = nd.x + nd.w * 0.5f;
            float cy = nd.y + nd.h * 0.5f;
            float hx = nd.w * 0.5f;
            float hy = nd.h * 0.5f;
            float r = Mathf.Min(nd.rx, Mathf.Min(hx, hy));

            // Weak/Untrained stays visibly muted (#14171d); Healthy lights up richly; Fault glows warm coral!
            float fillTint = (state == ModuleDiagState.WeakOrUntrained)
                ? (0.02f + (isSelected ? 0.05f : 0f))
                : Mathf.Clamp01(0.08f + lum01 * (nd.role == 1 ? 0.36f : 0.26f) + (isSelected ? 0.08f : 0f));
            byte fillR = (byte)Mathf.RoundToInt(Mathf.Lerp(20f, accent.r * 0.44f + 16f, fillTint));
            byte fillG = (byte)Mathf.RoundToInt(Mathf.Lerp(23f, accent.g * 0.44f + 20f, fillTint));
            byte fillB = (byte)Mathf.RoundToInt(Mathf.Lerp(29f, accent.b * 0.48f + 28f, fillTint));

            float borderLum = (state == ModuleDiagState.WeakOrUntrained)
                ? (isSelected ? 0.45f : 0.18f)
                : Mathf.Clamp01(0.32f + lum01 * 0.68f + (isSelected ? 0.20f : 0f));
            byte brdR = (byte)Mathf.RoundToInt(Mathf.Lerp(44f, accent.r, borderLum));
            byte brdG = (byte)Mathf.RoundToInt(Mathf.Lerp(50f, accent.g, borderLum));
            byte brdB = (byte)Mathf.RoundToInt(Mathf.Lerp(62f, accent.b, borderLum));

            // Strictly ZERO outer neon halo when WeakOrUntrained!
            float glowStrength = (state == ModuleDiagState.WeakOrUntrained)
                ? 0f
                : Mathf.Clamp01((lum01 - 0.20f) * (nd.role == 1 ? 1.22f : 0.92f) + (isSelected ? 0.16f : 0f));

            for (int py = y0; py <= y1; py++)
            {
                float fy = py + 0.5f;
                for (int px = x0; px <= x1; px++)
                {
                    float fx = px + 0.5f;
                    float qx = Mathf.Abs(fx - cx) - (hx - r);
                    float qy = Mathf.Abs(fy - cy) - (hy - r);
                    float outX = qx > 0f ? qx : 0f;
                    float outY = qy > 0f ? qy : 0f;
                    float sdf = Mathf.Sqrt(outX * outX + outY * outY) + Mathf.Min(Mathf.Max(qx, qy), 0f) - r;

                    if (sdf <= 0.65f)
                    {
                        float edgeAlpha = Mathf.Clamp01(0.65f - sdf);
                        float borderBand = Mathf.Clamp01(1.45f - Mathf.Abs(sdf + 0.55f));
                        byte rr = (byte)Mathf.RoundToInt(Mathf.Lerp(fillR, brdR, borderBand));
                        byte gg = (byte)Mathf.RoundToInt(Mathf.Lerp(fillG, brdG, borderBand));
                        byte bb = (byte)Mathf.RoundToInt(Mathf.Lerp(fillB, brdB, borderBand));
                        BlendAlpha(px, py, new Color32(rr, gg, bb, (byte)Mathf.RoundToInt(edgeAlpha * 255f)));
                    }
                    else if (glowStrength > 0.02f && sdf < haloPad)
                    {
                        float falloff = Mathf.Exp(-0.50f * sdf) * glowStrength;
                        byte ga = (byte)Mathf.Clamp(Mathf.RoundToInt(falloff * 120f), 0, 148);
                        if (ga > 1)
                            AddPixel(px, py, new Color32(accent.r, accent.g, accent.b, ga));
                    }
                }
            }

            // Corner Diagnostic LED Dot (`OK` = Emerald, `WEAK` = Dim Slate, `WARN` = Bright Coral)
            float ledX = nd.x + nd.w - 10f;
            float ledY = nd.y + 10f;
            if (state == ModuleDiagState.Healthy)
            {
                AddGlowDisc(ledX, ledY, 4.0f, new Color32(40, 225, 130, 120));
                AddGlowDisc(ledX, ledY, 2.0f, new Color32(120, 255, 180, 245));
            }
            else if (state == ModuleDiagState.FaultOrFrozen)
            {
                AddGlowDisc(ledX, ledY, 4.8f, new Color32(255, 92, 64, 180));
                AddGlowDisc(ledX, ledY, 2.2f, new Color32(255, 210, 160, 255));
            }
            else
            {
                BlendAlpha(Mathf.RoundToInt(ledX), Mathf.RoundToInt(ledY), new Color32(95, 108, 128, 180));
            }

            // Bottom Competence Gauge Bar (`health01`) inside node
            int fullW = Mathf.RoundToInt(nd.w - 22f);
            int barW = Mathf.RoundToInt(fullW * Mathf.Clamp01(health01));
            int bx0 = Mathf.RoundToInt(nd.x + 11f);
            int by0 = Mathf.RoundToInt(nd.y + nd.h - 7f);
            Color32 barActiveCol = state == ModuleDiagState.Healthy
                ? new Color32(accent.r, accent.g, accent.b, 225)
                : (state == ModuleDiagState.FaultOrFrozen
                    ? new Color32(255, 108, 68, 235)
                    : new Color32(82, 94, 114, 155));
            for (int x = 0; x < fullW; x++)
            {
                Color32 bc = x < barW ? barActiveCol : new Color32(36, 42, 52, 155);
                BlendAlpha(bx0 + x, by0, bc);
                BlendAlpha(bx0 + x, by0 + 1, bc);
            }
        }

        private void DrawThinSegment(Vector2 a, Vector2 b, Color32 c)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) * 1.5f));
            for (int i = 0; i <= steps; i++)
            {
                Vector2 p = Vector2.Lerp(a, b, i / (float)steps);
                BlendAlpha(Mathf.RoundToInt(p.x), Mathf.RoundToInt(p.y), c);
            }
        }

        private void DrawGlowSegment(Vector2 a, Vector2 b, float radius, Color32 c)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b)));
            for (int i = 0; i <= steps; i++)
            {
                Vector2 p = Vector2.Lerp(a, b, i / (float)steps);
                AddGlowDisc(p.x, p.y, radius, c);
            }
        }

        private void AddGlowDisc(float cx, float cy, float radius, Color32 c)
        {
            int rCeil = Mathf.CeilToInt(radius + 0.5f);
            int ix = Mathf.RoundToInt(cx);
            int iy = Mathf.RoundToInt(cy);
            float invR = 1f / Mathf.Max(0.35f, radius);
            for (int dy = -rCeil; dy <= rCeil; dy++)
            {
                int py = iy + dy;
                if ((uint)py >= CANVAS_H) continue;
                for (int dx = -rCeil; dx <= rCeil; dx++)
                {
                    int px = ix + dx;
                    if ((uint)px >= CANVAS_W) continue;
                    float dist = Mathf.Sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
                    if (dist <= radius + 0.5f)
                    {
                        float w = Mathf.Clamp01(1f - dist * invR);
                        byte a = (byte)Mathf.RoundToInt(c.a * w);
                        if (a > 0)
                            AddPixel(px, py, new Color32(c.r, c.g, c.b, a));
                    }
                }
            }
        }

        private void ClearCanvas(Color32 c)
        {
            for (int i = 0; i < _pixels.Length; i++)
                _pixels[i] = c;
        }

        private void BlendAlpha(int x, int yTop, Color32 src)
        {
            if ((uint)x >= CANVAS_W || (uint)yTop >= CANVAS_H || src.a == 0) return;
            int idx = (CANVAS_H - 1 - yTop) * CANVAS_W + x;
            if (src.a == 255) { _pixels[idx] = src; return; }
            Color32 dst = _pixels[idx];
            int a = src.a, inv = 255 - a;
            _pixels[idx] = new Color32(
                (byte)((src.r * a + dst.r * inv) >> 8),
                (byte)((src.g * a + dst.g * inv) >> 8),
                (byte)((src.b * a + dst.b * inv) >> 8),
                255);
        }

        private void AddPixel(int x, int yTop, Color32 src)
        {
            if ((uint)x >= CANVAS_W || (uint)yTop >= CANVAS_H || src.a == 0) return;
            int idx = (CANVAS_H - 1 - yTop) * CANVAS_W + x;
            Color32 dst = _pixels[idx];
            int a = src.a;
            _pixels[idx] = new Color32(
                (byte)Mathf.Min(255, dst.r + ((src.r * a) >> 8)),
                (byte)Mathf.Min(255, dst.g + ((src.g * a) >> 8)),
                (byte)Mathf.Min(255, dst.b + ((src.b * a) >> 8)),
                255);
        }

        private void EnsureStyles()
        {
            if (_stylesReady && _texOuterCard != null)
                return;

            _texOuterCard = MakeRounded9SliceTex(32, 12f, new Color(0.055f, 0.067f, 0.084f, 0.96f), new Color(0.18f, 0.21f, 0.27f, 0.95f));
            _texPillIdle = MakeRounded9SliceTex(24, 9f, new Color(0.11f, 0.13f, 0.17f, 0.96f), new Color(0.22f, 0.25f, 0.32f, 0.95f));
            _texPillActive = MakeRounded9SliceTex(24, 9f, new Color(0.10f, 0.16f, 0.38f, 0.98f), new Color(0.58f, 0.80f, 1.00f, 0.98f));

            _stOuterCard = new GUIStyle(GUI.skin.box)
            {
                border = new RectOffset(12, 12, 12, 12),
                padding = new RectOffset(0, 0, 0, 0)
            };
            _stOuterCard.normal.background = _texOuterCard;

            _stPillIdle = new GUIStyle(GUI.skin.button)
            {
                border = new RectOffset(9, 9, 9, 9),
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                padding = new RectOffset(4, 4, 1, 1)
            };
            _stPillIdle.normal.background = _texPillIdle;
            _stPillIdle.hover.background = _texPillActive;
            _stPillIdle.active.background = _texPillActive;
            _stPillIdle.normal.textColor = new Color(0.76f, 0.82f, 0.90f, 1f);
            _stPillIdle.hover.textColor = Color.white;

            _stPillActive = new GUIStyle(_stPillIdle);
            _stPillActive.normal.background = _texPillActive;
            _stPillActive.normal.textColor = new Color(0.86f, 0.94f, 1.0f, 1f);

            _stHdrTitle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                fontStyle = FontStyle.Bold,
                richText = true,
                alignment = TextAnchor.MiddleLeft
            };
            _stHdrTitle.normal.textColor = new Color(0.88f, 0.93f, 0.99f, 1f);

            _stHdrBadge = new GUIStyle(GUI.skin.label)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                richText = true,
                alignment = TextAnchor.MiddleRight
            };
            _stHdrBadge.normal.textColor = Color.white;

            _stNodeTitle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                richText = true,
                alignment = TextAnchor.MiddleCenter,
                clipping = TextClipping.Clip
            };
            _stNodeTitle.normal.textColor = new Color(0.92f, 0.95f, 0.99f, 1f);

            _stNodeSub = new GUIStyle(GUI.skin.label)
            {
                fontSize = 9,
                alignment = TextAnchor.MiddleCenter,
                richText = true,
                clipping = TextClipping.Clip
            };
            _stNodeSub.normal.textColor = new Color(0.66f, 0.75f, 0.86f, 1f);

            _stFooter = new GUIStyle(GUI.skin.label)
            {
                fontSize = 9,
                richText = true,
                alignment = TextAnchor.MiddleLeft,
                clipping = TextClipping.Clip
            };
            _stFooter.normal.textColor = new Color(0.74f, 0.83f, 0.94f, 1f);
            _stylesReady = true;
        }

        private static Texture2D MakeRounded9SliceTex(int size, float radius, Color fill, Color border)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            float half = size * 0.5f;
            float r = Mathf.Min(radius, half - 1f);
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
                    float cov = Mathf.Clamp01(0.6f - sdf);
                    float bBand = Mathf.Clamp01(1.3f - Mathf.Abs(sdf + 0.5f));
                    Color c = Color.Lerp(fill, border, bBand);
                    c.a *= cov;
                    cols[y * size + x] = c;
                }
            }
            tex.SetPixels32(cols);
            tex.Apply(false);
            return tex;
        }

        private void OnGUI()
        {
            if (!IsVisibleInSessionNow)
                return;

            BuildCircuitDefinitions();
            EnsureStyles();

            Rect r = GetPanelRect();
            string badgePlain = WarnNodeCount > 0
                ? $"! WARN:{WarnNodeCount}"
                : $"健全 {HealthyNodeCount}/9";
            if (!showCircuitHud)
            {
                if (GUI.Button(r, $"▼ 脳回路診断 [{badgePlain}] (F7)", _stPillActive))
                {
                    showCircuitHud = true;
                    _nextRasterTime = 0f;
                }
                return;
            }

            GUI.Box(r, GUIContent.none, _stOuterCard);

            // Row 1: Title + Whole-Brain Functional Competence Summary Badge + Collapse Button
            GUI.Label(new Rect(r.x + 10f, r.y + 4f, 168f, 18f),
                "<color=#9DD2FF>04 BRAIN DIAGNOSTIC</color>", _stHdrTitle);

            string summaryRich = WarnNodeCount > 0
                ? $"<color=#FF8566>! 異常/固着:{WarnNodeCount} ({HealthyNodeCount}/9 OK)</color>"
                : (HealthyNodeCount >= 7
                    ? $"<color=#6EE7A0>● 健全 {HealthyNodeCount}/9 OK ({OverallHealthScore01 * 100f:F0}%)</color>"
                    : $"<color=#94A3B8>◌ 未学習/弱 {HealthyNodeCount}/9 OK ({OverallHealthScore01 * 100f:F0}%)</color>");
            GUI.Label(new Rect(r.x + 150f, r.y + 4f, r.width - 208f, 18f), summaryRich, _stHdrBadge);

            if (GUI.Button(new Rect(r.x + r.width - 52f, r.y + 4f, 44f, 18f), "畳む", _stPillIdle))
            {
                showCircuitHud = false;
                return;
            }

            // Row 2: 3 Luminous Segmented View Pills
            float tabW = (r.width - 24f - 8f) / 3f;
            for (int v = 0; v < _views.Length; v++)
            {
                Rect tr = new Rect(r.x + 12f + v * (tabW + 4f), r.y + 24f, tabW, 18f);
                if (GUI.Button(tr, _views[v].tabLabel, v == activeViewIndex ? _stPillActive : _stPillIdle))
                {
                    activeViewIndex = v;
                    _nextRasterTime = 0f;
                }
            }

            // Center Square Glowing Circuit Canvas
            Rect cRect = new Rect(r.x + 4f, r.y + HEADER_H + 2f, CANVAS_W, CANVAS_H);
            if (_canvasTex != null)
                GUI.DrawTexture(cRect, _canvasTex, ScaleMode.StretchToFill, false);

            // Crisp Native Font Overlay + Interactive Node Selection on Click
            int vIdx = Mathf.Clamp(activeViewIndex, 0, _views.Length - 1);
            CircuitViewDef view = _views[vIdx];
            Event ev = Event.current;
            for (int i = 0; i < view.nodes.Length; i++)
            {
                NodeDef nd = view.nodes[i];
                ModuleDiagState st = _nodeDiag[vIdx][i];
                Rect nr = new Rect(cRect.x + nd.x, cRect.y + nd.y, nd.w, nd.h);
                if (ev.type == EventType.MouseDown && ev.button == 0 && nr.Contains(ev.mousePosition))
                {
                    view.selectedIdx = i;
                    _nextRasterTime = 0f;
                    ev.Use();
                }

                string titleRich = st == ModuleDiagState.WeakOrUntrained
                    ? $"<color=#7E8B9E>{nd.title}</color>"
                    : (st == ModuleDiagState.FaultOrFrozen
                        ? $"<color=#FFB49E>{nd.title}</color>"
                        : nd.title);
                string metricColor = st == ModuleDiagState.Healthy
                    ? "#9CF2C4"
                    : (st == ModuleDiagState.FaultOrFrozen ? "#FF9B82" : "#7B899E");

                float midY = nr.y + nr.height * 0.5f;
                GUI.Label(new Rect(nr.x + 3f, midY - 23f, nr.width - 10f, 16f), titleRich, _stNodeTitle);
                GUI.Label(new Rect(nr.x + 2f, midY - 7f, nr.width - 4f, 14f), nd.sub, _stNodeSub);
                GUI.Label(new Rect(nr.x + 2f, midY + 7f, nr.width - 4f, 14f),
                    $"<color={metricColor}>{_nodeLiveMetric[vIdx][i]}</color>", _stNodeSub);
            }

            // Bottom Selected-Node Live Theoretical Diagnostic Readout
            int sel = Mathf.Clamp(view.selectedIdx, 0, view.nodes.Length - 1);
            ModuleDiagState selSt = _nodeDiag[vIdx][sel];
            string dotHex = selSt == ModuleDiagState.Healthy ? "#50E389" : (selSt == ModuleDiagState.FaultOrFrozen ? "#FF7B5C" : "#8A99AD");
            GUI.Label(
                new Rect(r.x + 8f, r.y + r.height - FOOTER_H + 2f, r.width - 14f, 18f),
                $"<color={dotHex}>●</color> {_nodeInspectorLine[vIdx][sel]}",
                _stFooter);
        }

        private static float Rms(float[] v)
        {
            if (v == null || v.Length == 0) return 0f;
            float s = 0f;
            for (int i = 0; i < v.Length; i++) s += v[i] * v[i];
            return Mathf.Sqrt(s / v.Length);
        }

        private static float Norm(float[] v)
        {
            if (v == null || v.Length == 0) return 0f;
            float s = 0f;
            for (int i = 0; i < v.Length; i++) s += v[i] * v[i];
            return Mathf.Sqrt(s);
        }

        private static float StdDev(float[] v)
        {
            if (v == null || v.Length == 0) return 0f;
            float mean = 0f;
            for (int i = 0; i < v.Length; i++) mean += v[i];
            mean /= v.Length;
            float vr = 0f;
            for (int i = 0; i < v.Length; i++) { float d = v[i] - mean; vr += d * d; }
            return Mathf.Sqrt(vr / v.Length);
        }

        private static float MeanAbsDiff(float[] a, float[] b)
        {
            if (a == null || b == null) return 0f;
            int n = Mathf.Min(a.Length, b.Length);
            if (n == 0) return 0f;
            float s = 0f;
            for (int i = 0; i < n; i++) s += Mathf.Abs(a[i] - b[i]);
            return s / n;
        }

        private static int PopCount32(uint x)
        {
            int c = 0;
            while (x != 0) { x &= (x - 1); c++; }
            return c;
        }

        private static string HoldShortName(int holdId)
        {
            switch (holdId)
            {
                case 1: return "WAIT:CD";
                case 2: return "LEAD-SYNC";
                case 3: return "XFIRE-HOLD";
                case 4: return "S1/GWT競合";
                case 5: return "FOG視線";
                case 6: return "REANCHOR";
                case 7: return "COACH";
                case 8: return "REGEN潜伏";
                default: return "HOLD";
            }
        }

        private static string ShortVerb(byte verb)
        {
            switch (verb)
            {
                case UniversalBrainBridge3D.VERB_IDLE_OBSERVE: return "Idle";
                case UniversalBrainBridge3D.VERB_APPROACH_HOVER: return "Move";
                case UniversalBrainBridge3D.VERB_COMMIT_INTERACT: return "Commit";
                case UniversalBrainBridge3D.VERB_GRASP_PICK: return "Grasp";
                case UniversalBrainBridge3D.VERB_RELEASE_PLACE: return "Shoot";
                case UniversalBrainBridge3D.VERB_EVADE_HAZARD: return "Swim";
                case UniversalBrainBridge3D.VERB_IMPULSE_PUSH: return "Burst";
                case UniversalBrainBridge3D.VERB_COMMUNICATE_PING: return "Rally";
                default: return "Act";
            }
        }
    }
}
