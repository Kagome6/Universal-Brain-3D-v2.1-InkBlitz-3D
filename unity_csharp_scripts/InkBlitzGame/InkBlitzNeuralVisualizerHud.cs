// InkBlitzNeuralVisualizerHud.cs — Spectate/Inference-Only Mathematical Brain HUD & 1st-Person POV for Agent A
// Ports the 2D-prototype `AIVisualizer` (#0e1115 cyber mathematical aesthetic) to Unity 6 C# with:
//   1. Right-Edge Panel (Spectate Mode Only):
//      - Top: 4-Tab 3D Oblique Cortical Neural Sheet (1:GWT 128, 2:WM 256, 3:SDR, 4:MEC 256) + Real Catmull-Rom Phase Trace
//      - Mid: GWT Ignition, Self-Prediction Error `e_s`, & 5-Module Arbitration [P, I, W, M, S]
//      - Bot: Switchable 3D Info-Geometric Thought Manifold (A: Variational Free-Energy Silk Wave / B: Amari-Oizumi Phi_G MIP)
//      - Foot: Match Score / Return History Graph (Inner-speech logical panel intentionally omitted for clean focus)
//   2. Bottom-Left PIP Window (Spectate Mode Only):
//      - Live 1st-Person Subjective Retina POV (`eyePivot`, 76-deg foveal frustum) of Representative Agent A!
// License: GPL-3.0

using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using UniversalBrain3D;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace InkBlitz3D
{
    public class InkBlitzNeuralVisualizerHud : MonoBehaviour
    {
        public const uint UBNT_MAGIC = 0x544E4255; // "UBNT"
        public const uint UBND_MAGIC = 0x444E4255; // "UBND"
        public const uint UBWM_MAGIC = 0x4D574255; // "UBWM"
        public const int UBNT_TOTAL_BYTES = 3636;
        public const int UBND_TOTAL_BYTES = 536;
        public const int UBWM_TOTAL_BYTES = 904;
        public const int UBNT_HDR_BYTES = 80;
        public const int UBNT_FLOAT_COUNT = 871;
        public const int NUM_CANDS = 8;

        public const int PANEL_W = 356;
        private const int CANVAS_H = 700;

        [Header("References")]
        public UniversalArenaCoordinator3D coordinator;
        public InkBlitzMultiArenaManager multiManager;
        public InkBlitzSessionDashboardUi dashboardUi;
        public InkBlitzBrainCircuitHud circuitHud;
        public InkBlitzTelemetryHud telemetryHud;

        [Header("Spectate HUD Toggles")]
        public bool showRightNeuralPanel = true;
        public bool showSubjectivePovPip = true;
        public bool showDepthCortexPip = true;
        public bool showWorldModelPip = true;
        [Range(0, 1)] public int depthViewMode = 0; // 0 = Plasma, 1 = Blend

        public int DepthRasterCount { get; private set; } = 0;
        public int WorldModelRasterCount { get; private set; } = 0;
        public Texture2D DepthCortexTexture => _depthTex;
        public Texture2D WorldModelPredTexture => _wmPredTex;
        public Texture2D WorldModelErrTexture => _wmErrTex;

        public bool IsRightPanelVisibleNow =>
            showRightNeuralPanel
            && coordinator != null
            && !coordinator.IsTrainingSession
            && (dashboardUi == null || !dashboardUi.IsLauncherOpen);

        public float ActiveRightPanelWidthPx => IsRightPanelVisibleNow ? PANEL_W : 0f;

        public bool TryGetAgentSubjectiveConfidence01(out float subjWin01)
        {
            subjWin01 = 0.50f;
            if (!_uxConnected)
                return false;
            lock (_ubntLock)
            {
                if (_latestUbnt == null || _latestUbnt.seq == 0)
                    return false;
                int idx = Mathf.Clamp(_latestUbnt.chosenCand, 0, NUM_CANDS - 1);
                float surv = Mathf.Clamp01(_latestUbnt.survivalProbs[idx]);
                float qStar = _latestUbnt.mipQStar[idx];
                float logit = Mathf.Clamp(0.38f * qStar + 1.85f * (surv - 0.65f), -4.5f, 4.5f);
                subjWin01 = 1f / (1f + Mathf.Exp(-logit));
                return true;
            }
        }

        // --- Live Unpacked UBNT & UBWM Snapshots (Double-Buffered from Background Reader Thread) ---
        public sealed class UbntSnapshot
        {
            public ulong seq;
            public uint episodeId;
            public int chosenCand;
            public int s1FastCand;
            public bool commitGateOpen;
            public bool vetoActive;
            public int holdReasonId;
            public bool endogenousActive;
            public bool overfitGuardTripped;
            public int protoSymId;
            public byte mipCutMask = 0x03;
            public uint loopClosuresTotal;
            public byte[] candVerbs = new byte[NUM_CANDS];
            public float cumulativeScore;
            public float believedHazard;
            public float mapAccuracyIou = 1f;
            public float meanFog;
            public float ignition;
            public float phiBits;
            public float entropyNorm;
            public float worldSurprise;
            public float selfSurprise;
            public float fisherRaoStep;
            public float goDrive;
            public float stnBrake;
            public float netAccumulator;
            public float releaseThreshold;
            public float friendlyTurfRatio;
            public float enemyTurfRatio;
            public float[] gwtAlpha = new float[5];
            public float[] gwtW = new float[128];
            public float[] rssmH = new float[256];
            public float[] latentZ = new float[32];
            public float[] zDelta = new float[32];
            public float[] mecGrid = new float[256];
            public float[] selfState = new float[8];
            public float[] qValues = new float[NUM_CANDS];
            public float[] qRawValues = new float[NUM_CANDS];
            public float[] survivalProbs = new float[NUM_CANDS];
            public float[] policyProbs = new float[NUM_CANDS];
            public float[] mipQStar = new float[NUM_CANDS];
            public Vector3 believedSelfPos;
            public Vector3 poseDriftVec;
            public string innerSpeechText = "";
        }

        public sealed class UbwmSnapshot
        {
            public ulong seq;
            public uint episodeId;
            public uint physicsTick;
            public int chosenCand;
            public bool hasPrevPred;
            public byte vetoMask;
            public float visPredMae;
            public float depthPredMaeM;
            public float worldKlSurprise;
            public float actionCfSpread;
            public float[] predRew = new float[NUM_CANDS];
            public float[] predSurv = new float[NUM_CANDS];
            public float[] predAdv = new float[NUM_CANDS];
            public float[] predDepth16 = new float[256];
            public float[] predSurf16 = new float[256];
            public float[] predErr16 = new float[256];
        }

        private readonly object _ubntLock = new object();
        private UbntSnapshot _latestUbnt;
        private UbwmSnapshot _latestUbwm;
        private ulong _ubwmSeqCounter = 0;
        private ulong _consumedUbwmSeq = 0;

        public UbntSnapshot LatestUbntSnapshot
        {
            get { lock (_ubntLock) { return _latestUbnt; } }
        }

        public UbwmSnapshot LatestUbwmSnapshot
        {
            get { lock (_ubntLock) { return _latestUbwm; } }
        }
        private ulong _consumedUbntSeq = 0;
        private Thread _rxThread;
        private volatile bool _rxRun = false;
        private volatile bool _wantUxStream = false;
        private volatile int _targetUxPort = 38155;
        private volatile bool _uxConnected = false;

        // --- Cortical Sheet & Info-Geometry State (Cyber-Dark `AIVisualizer` Layout) ---
        private int _activeTab = 0; // 0:GWT, 1:WM, 2:SDR, 3:MEC
        private int _geoMode = 0;   // 0:GEO-LANDSCAPE, 1:PHI-MIP [IIT/IG]
        private readonly string[] _tabLabels = { "1:GWT", "2:WM", "3:SDR", "4:MEC" };
        private readonly string[] _geoLabels = { "A: GEO-LANDSCAPE", "B: PHI-MIP [IIT/IG]" };

        private readonly int[] _tabSizes = { 128, 256, 256, 256 };
        private float[][] _tabPrevVec;
        private float[][] _tabSmoothSheet;
        private float[][] _tabSpikeSheet;
        private List<Vector2>[] _tabTraj;
        private float[][] _tabLastPushUnit;
        private float[] _tabLastDeltaNorm;
        private readonly Dictionary<int, float[]> _projAxes = new Dictionary<int, float[]>();
        private float[] _gridU256, _gridV256, _gridU128, _gridV128;

        public float[] CurrentCorticalSmoothSheet => (_tabSmoothSheet != null && _activeTab >= 0 && _activeTab < 4) ? _tabSmoothSheet[_activeTab] : null;
        public float[] CurrentCorticalSpikeSheet => (_tabSpikeSheet != null && _activeTab >= 0 && _activeTab < 4) ? _tabSpikeSheet[_activeTab] : null;
        public byte CurrentMipCutMask => _dispMipMask;

        private struct GeoSlice
        {
            public float[] wave;
            public float[] surv;
            public float phiBits;
            public float frStep;
            public float entNorm;
            public float focusX;
            public int bestCol;
            public bool veto;
        }

        private const int GEO_HIST_LEN = 22;
        private readonly List<GeoSlice> _geoHistory = new List<GeoSlice>(GEO_HIST_LEN);
        private float[] _smoothGeoWave;
        private float _smoothFocusX = 0.5f;
        private bool _geoSeeded = false;
        private float[] _lastGeoSig;
        private float _lastGeoPushTime = 0f;
        private float _displayFrEma = 0f;
        private readonly List<float> _phiTrace = new List<float>(44);
        private readonly List<float> _scoreHistory = new List<float>(50);
        private uint _lastRecordedEp = 0;
        private float _lastEpScore = 0f;

        // Latest HUD readout strings
        private string _regionSubText = "GWT W_t [16x8=128] | WAITING TELEMETRY";
        private string _controlModeText = "PURE_NEURAL | M0 ApproachHover";
        private float _dispIgnition = 0f;
        private float _dispSelfErr = 0f;
        private bool _dispVeto = false;
        private readonly float[] _dispAlpha = { 0.2f, 0.2f, 0.2f, 0.2f, 0.2f };
        private float _dispPhiBits = 0f;
        private float _dispFrStep = 0f;
        private float _dispEntNorm = 1f;
        private int _dispFastCol = 0;
        private int _dispBestCol = 0;
        private byte _dispMipMask = 0x03;
        private readonly Vector2[] _lastMipNode2d =
        {
            new Vector2(266f, 450f), new Vector2(266f, 484f),
            new Vector2(320f, 442f), new Vector2(320f, 466f), new Vector2(320f, 490f)
        };
        private Vector2 _lastQStarCanvasPt = new Vector2(128f, 564f);
        private Vector2 _lastQS1CanvasPt = new Vector2(158f, 568f);
        private Vector2 _lastPIntCanvasPt = new Vector2(128f, 564f);
        private float _lastLiftPx = 0f;

        // --- Software Rasterizer Buffer (`PANEL_W x CANVAS_H` -> Texture2D) ---
        private Texture2D _canvasTex;
        private Color32[] _pixels;
        private float _nextRasterTime = 0f;
        private float _t0 = 0f;

        // --- Agent A 1st-Person Subjective POV Camera & RenderTexture ---
        private Camera _povCamera;
        private RenderTexture _povRt;
        private Transform _boundEyePivot;

        // --- Lightweight V1 Depth Cortex Viewer (`02 DEPTH`: 96x54 Plasma LUT, 0ms when hidden!) ---
        private const int DEPTH_TEX_W = 96;
        private const int DEPTH_TEX_H = 54;
        private static Color32[] s_plasmaLut;
        private Texture2D _depthTex;
        private Color32[] _depthPixels;
        private readonly float[] _depthRowTmp = new float[16 * DEPTH_TEX_W];
        private readonly float[] _salRowTmp = new float[16 * DEPTH_TEX_W];
        private readonly float[] _ubndDepth16 = new float[256];
        private ulong _ubndSeq = 0;
        private ulong _consumedDepthSeq = 0;
        private int _consumedDepthMode = -1;
        private float _nextDepthRasterTime = 0f;
        private float _depthMeanMeters = 6.0f;

        // --- Lightweight Predictive World-Model Viewer (`03 WORLD MODEL`: Dual 88x56 Imagined + Surprise, 0ms when hidden!) ---
        private const int WM_TEX_W = 88;
        private const int WM_TEX_H = 56;
        private static Color32[] s_surpriseLut;
        private Texture2D _wmPredTex;
        private Texture2D _wmErrTex;
        private Color32[] _wmPredPixels;
        private Color32[] _wmErrPixels;
        private readonly float[] _wmRowTmpD = new float[16 * WM_TEX_W];
        private readonly float[] _wmRowTmpS = new float[16 * WM_TEX_W];
        private readonly float[] _wmRowTmpE = new float[16 * WM_TEX_W];
        private float _nextWmRasterTime = 0f;
        private Texture2D _texWmAccent;
        private Texture2D _texBarBg;
        private Texture2D _texBarSurv;
        private Texture2D _texBarVeto;
        private Texture2D _texBarAdv;

        // --- Cached IMGUI Styles ---
        private bool _stylesReady = false;
        private Texture2D _texDarkHeader;
        private Texture2D _texTabSel;
        private Texture2D _texTabIdle;
        private Texture2D _texDepthAccent;
        private Texture2D _texPlasmaSel;
        private GUIStyle _stGoldHdr;
        private GUIStyle _stSmallSub;
        private GUIStyle _stModeLine;
        private GUIStyle _stTiny;
        private GUIStyle _stTabSel;
        private GUIStyle _stTabIdle;
        private GUIStyle _stPlasmaSel;
        private GUIStyle _stPipBox;

        public InkBlitzBrainCircuitHud EnsureCircuitHudBound()
        {
            if (coordinator == null) coordinator = FindAnyObjectByType<UniversalArenaCoordinator3D>();
            if (multiManager == null) multiManager = FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (dashboardUi == null) dashboardUi = FindAnyObjectByType<InkBlitzSessionDashboardUi>();
            if (telemetryHud == null) telemetryHud = FindAnyObjectByType<InkBlitzTelemetryHud>();
            if (circuitHud == null)
                circuitHud = GetComponent<InkBlitzBrainCircuitHud>() ?? gameObject.AddComponent<InkBlitzBrainCircuitHud>();
            circuitHud.coordinator = coordinator;
            circuitHud.dashboardUi = dashboardUi;
            circuitHud.neuralHud = this;
            return circuitHud;
        }

        private void Awake()
        {
            EnsureCircuitHudBound();

            _t0 = Time.realtimeSinceStartup;
            InitMathBuffers();
            StartUbntBackgroundThread();
        }

        private void OnDestroy()
        {
            StopUbntBackgroundThread();
            ReleasePovResources();
            if (_canvasTex != null) DestroyObjSafe(_canvasTex);
            if (_depthTex != null) DestroyObjSafe(_depthTex);
            if (_wmPredTex != null) DestroyObjSafe(_wmPredTex);
            if (_wmErrTex != null) DestroyObjSafe(_wmErrTex);
            if (_texDarkHeader != null) DestroyObjSafe(_texDarkHeader);
            if (_texTabSel != null) DestroyObjSafe(_texTabSel);
            if (_texTabIdle != null) DestroyObjSafe(_texTabIdle);
            if (_texDepthAccent != null) DestroyObjSafe(_texDepthAccent);
            if (_texPlasmaSel != null) DestroyObjSafe(_texPlasmaSel);
            if (_texWmAccent != null) DestroyObjSafe(_texWmAccent);
            if (_texBarBg != null) DestroyObjSafe(_texBarBg);
            if (_texBarSurv != null) DestroyObjSafe(_texBarSurv);
            if (_texBarVeto != null) DestroyObjSafe(_texBarVeto);
            if (_texBarAdv != null) DestroyObjSafe(_texBarAdv);
        }

        private static void DestroyObjSafe(UnityEngine.Object obj)
        {
            if (obj == null) return;
            if (Application.isPlaying) Destroy(obj);
            else DestroyImmediate(obj);
        }

        private void InitMathBuffers()
        {
            _tabPrevVec = new float[4][];
            _tabSmoothSheet = new float[4][];
            _tabSpikeSheet = new float[4][];
            _tabTraj = new List<Vector2>[4];
            _tabLastPushUnit = new float[4][];
            _tabLastDeltaNorm = new float[4];
            for (int i = 0; i < 4; i++)
            {
                _tabSmoothSheet[i] = new float[_tabSizes[i]];
                _tabSpikeSheet[i] = new float[_tabSizes[i]];
                _tabTraj[i] = new List<Vector2>(18);
            }

            _gridU256 = new float[256];
            _gridV256 = new float[256];
            for (int r = 0; r < 16; r++)
            {
                float v = Mathf.Lerp(-0.88f, 0.88f, r / 15f);
                for (int c = 0; c < 16; c++)
                {
                    int k = r * 16 + c;
                    _gridU256[k] = Mathf.Lerp(-0.88f, 0.88f, c / 15f);
                    _gridV256[k] = v;
                }
            }

            _gridU128 = new float[128];
            _gridV128 = new float[128];
            for (int r = 0; r < 8; r++)
            {
                float v = Mathf.Lerp(-0.84f, 0.84f, r / 7f);
                for (int c = 0; c < 16; c++)
                {
                    int k = r * 16 + c;
                    _gridU128[k] = Mathf.Lerp(-0.88f, 0.88f, c / 15f);
                    _gridV128[k] = v;
                }
            }

            for (int i = 0; i < GEO_HIST_LEN; i++)
            {
                _geoHistory.Add(new GeoSlice
                {
                    wave = new float[NUM_CANDS],
                    surv = new float[] { 1, 1, 1, 1, 1, 1, 1, 1 },
                    phiBits = 0f,
                    frStep = 0f,
                    entNorm = 1f,
                    focusX = 0.5f,
                    bestCol = 0,
                    veto = false
                });
            }
            for (int i = 0; i < 44; i++)
                _phiTrace.Add(0f);

            _pixels = new Color32[PANEL_W * CANVAS_H];
            _canvasTex = new Texture2D(PANEL_W, CANVAS_H, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            ClearCanvas(new Color32(14, 17, 21, 255));
            _canvasTex.SetPixels32(_pixels);
            _canvasTex.Apply(false);
        }

        private static float NextGaussian(System.Random r)
        {
            double u1 = Math.Max(1e-7, r.NextDouble());
            double u2 = r.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        // =====================================================================
        // 1. Background Non-Blocking UBNT Telemetry Stream Reader
        // =====================================================================
        private void StartUbntBackgroundThread()
        {
            if (_rxThread != null) return;
            _rxRun = true;
            _rxThread = new Thread(UbntReadLoop)
            {
                IsBackground = true,
                Name = "UBNT_SpectateReader"
            };
            _rxThread.Start();
        }

        private void StopUbntBackgroundThread()
        {
            _rxRun = false;
            if (_rxThread != null)
            {
                try { _rxThread.Join(350); } catch { }
                _rxThread = null;
            }
        }

        private static bool ReadExact(NetworkStream s, byte[] buf, int offset, int count)
        {
            int got = 0;
            while (got < count)
            {
                int n = s.Read(buf, offset + got, count - got);
                if (n <= 0) return false;
                got += n;
            }
            return true;
        }

        private void UbntReadLoop()
        {
            byte[] pktBuf = new byte[UBNT_TOTAL_BYTES];
            byte[] skipBuf = new byte[UBND_TOTAL_BYTES];
            byte[] wmBuf = new byte[UBWM_TOTAL_BYTES];

            while (_rxRun)
            {
                if (!IsSpectateBackendActive())
                {
                    _uxConnected = false;
                    Thread.Sleep(200);
                    continue;
                }

                TcpClient cli = null;
                try
                {
                    cli = new TcpClient();
                    cli.NoDelay = true;
                    var ar = cli.BeginConnect("127.0.0.1", _targetUxPort, null, null);
                    if (!ar.AsyncWaitHandle.WaitOne(250))
                    {
                        cli.Close();
                        Thread.Sleep(250);
                        continue;
                    }
                    cli.EndConnect(ar);
                    cli.ReceiveTimeout = 1500;
                    NetworkStream ns = cli.GetStream();
                    _uxConnected = true;

                    while (_rxRun && IsSpectateBackendActive())
                    {
                        if (!ReadExact(ns, pktBuf, 0, 4))
                            break;
                        uint magic = BitConverter.ToUInt32(pktBuf, 0);
                        if (magic == UBNT_MAGIC)
                        {
                            if (!ReadExact(ns, pktBuf, 4, UBNT_TOTAL_BYTES - 4))
                                break;
                            ParseUbntPacket(pktBuf);
                        }
                        else if (magic == UBND_MAGIC)
                        {
                            Buffer.BlockCopy(pktBuf, 0, skipBuf, 0, 4);
                            if (!ReadExact(ns, skipBuf, 4, UBND_TOTAL_BYTES - 4))
                                break;
                            ParseUbndPacket(skipBuf);
                        }
                        else if (magic == UBWM_MAGIC)
                        {
                            Buffer.BlockCopy(pktBuf, 0, wmBuf, 0, 4);
                            if (!ReadExact(ns, wmBuf, 4, UBWM_TOTAL_BYTES - 4))
                                break;
                            ParseUbwmPacket(wmBuf);
                        }
                        else
                        {
                            // Resync on unexpected stream byte
                            break;
                        }
                    }
                }
                catch
                {
                    // Normal during session switches / reconnects
                }
                finally
                {
                    _uxConnected = false;
                    try { cli?.Close(); } catch { }
                }
                Thread.Sleep(200);
            }
        }

        private bool IsSpectateBackendActive() => _wantUxStream;

        public bool IngestRawUbntPacketForTest(byte[] pkt)
        {
            if (pkt == null || pkt.Length != UBNT_TOTAL_BYTES) return false;
            ParseUbntPacket(pkt);
            return _latestUbnt != null;
        }

        public bool IngestRawUbwmPacketForTest(byte[] pkt)
        {
            if (pkt == null || pkt.Length != UBWM_TOTAL_BYTES) return false;
            ulong prev = _ubwmSeqCounter;
            ParseUbwmPacket(pkt);
            return _ubwmSeqCounter > prev;
        }

        private void ParseUbntPacket(byte[] pkt)
        {
            uint savedCrc = BitConverter.ToUInt32(pkt, UBNT_TOTAL_BYTES - 8);
            if (UniversalBrainBridge3D.ComputeCrc32(pkt, 0, UBNT_TOTAL_BYTES - 8) != savedCrc)
                return;

            byte rawMipMask = (byte)((pkt[52] >> 3) & 0x1F);
            var snap = new UbntSnapshot
            {
                seq = BitConverter.ToUInt64(pkt, 8),
                episodeId = BitConverter.ToUInt32(pkt, 24),
                chosenCand = pkt[40] & 0x07,
                s1FastCand = pkt[41] & 0x07,
                commitGateOpen = pkt[42] != 0,
                vetoActive = pkt[43] != 0,
                holdReasonId = pkt[45],
                endogenousActive = (pkt[50] & 0x01) != 0,
                protoSymId = pkt[51],
                mipCutMask = (rawMipMask >= 1 && rawMipMask <= 30) ? rawMipMask : (byte)0x03,
                overfitGuardTripped = (pkt[54] & 0x40) != 0,
                loopClosuresTotal = BitConverter.ToUInt32(pkt, 72)
            };
            Buffer.BlockCopy(pkt, 56, snap.candVerbs, 0, NUM_CANDS);

            int fOff = UBNT_HDR_BYTES;
            // scalars24 (0..24)
            snap.cumulativeScore = BitConverter.ToSingle(pkt, fOff + 3 * 4);
            snap.believedHazard = BitConverter.ToSingle(pkt, fOff + 6 * 4);
            snap.mapAccuracyIou = BitConverter.ToSingle(pkt, fOff + 7 * 4);
            snap.meanFog = BitConverter.ToSingle(pkt, fOff + 8 * 4);
            snap.ignition = BitConverter.ToSingle(pkt, fOff + 9 * 4);
            snap.phiBits = BitConverter.ToSingle(pkt, fOff + 10 * 4);
            snap.entropyNorm = BitConverter.ToSingle(pkt, fOff + 11 * 4);
            snap.worldSurprise = BitConverter.ToSingle(pkt, fOff + 12 * 4);
            snap.selfSurprise = BitConverter.ToSingle(pkt, fOff + 13 * 4);
            snap.fisherRaoStep = BitConverter.ToSingle(pkt, fOff + 14 * 4);
            snap.goDrive = BitConverter.ToSingle(pkt, fOff + 16 * 4);
            snap.stnBrake = BitConverter.ToSingle(pkt, fOff + 17 * 4);
            snap.netAccumulator = BitConverter.ToSingle(pkt, fOff + 19 * 4);
            snap.releaseThreshold = BitConverter.ToSingle(pkt, fOff + 20 * 4);
            snap.friendlyTurfRatio = BitConverter.ToSingle(pkt, fOff + 22 * 4);
            snap.enemyTurfRatio = BitConverter.ToSingle(pkt, fOff + 23 * 4);

            int pos = 24;
            CopyFloats(pkt, fOff, ref pos, snap.gwtAlpha, 5);
            CopyFloats(pkt, fOff, ref pos, snap.gwtW, 128);
            CopyFloats(pkt, fOff, ref pos, snap.rssmH, 256);
            CopyFloats(pkt, fOff, ref pos, snap.latentZ, 32);
            CopyFloats(pkt, fOff, ref pos, snap.zDelta, 32);
            CopyFloats(pkt, fOff, ref pos, snap.mecGrid, 256);
            pos += 32; // skip hd_ring_code (32)
            CopyFloats(pkt, fOff, ref pos, snap.selfState, 8);
            CopyFloats(pkt, fOff, ref pos, snap.qValues, NUM_CANDS);
            CopyFloats(pkt, fOff, ref pos, snap.qRawValues, NUM_CANDS);
            CopyFloats(pkt, fOff, ref pos, snap.survivalProbs, NUM_CANDS);
            CopyFloats(pkt, fOff, ref pos, snap.policyProbs, NUM_CANDS);
            CopyFloats(pkt, fOff, ref pos, snap.mipQStar, NUM_CANDS);
            pos += NUM_CANDS * 6 + 4; // skip cand_poses (48) + stn_gains (4) -> pos = 865
            snap.believedSelfPos = new Vector3(
                BitConverter.ToSingle(pkt, fOff + (pos + 0) * 4),
                BitConverter.ToSingle(pkt, fOff + (pos + 1) * 4),
                BitConverter.ToSingle(pkt, fOff + (pos + 2) * 4));
            snap.poseDriftVec = new Vector3(
                BitConverter.ToSingle(pkt, fOff + (pos + 3) * 4),
                BitConverter.ToSingle(pkt, fOff + (pos + 4) * 4),
                BitConverter.ToSingle(pkt, fOff + (pos + 5) * 4));

            int spOff = UBNT_TOTAL_BYTES - 8 - 64;
            int spLen = 0;
            while (spLen < 64 && pkt[spOff + spLen] != 0) spLen++;
            if (spLen > 0)
                snap.innerSpeechText = System.Text.Encoding.UTF8.GetString(pkt, spOff, spLen);

            lock (_ubntLock)
            {
                _latestUbnt = snap;
            }
        }

        private void ParseUbwmPacket(byte[] pkt)
        {
            uint savedCrc = BitConverter.ToUInt32(pkt, UBWM_TOTAL_BYTES - 4);
            if (UniversalBrainBridge3D.ComputeCrc32(pkt, 0, UBWM_TOTAL_BYTES - 4) != savedCrc)
                return;
            uint ver = BitConverter.ToUInt32(pkt, 4);
            if (ver != 1)
                return;

            var wm = new UbwmSnapshot
            {
                episodeId = BitConverter.ToUInt32(pkt, 8),
                physicsTick = BitConverter.ToUInt32(pkt, 12),
                chosenCand = pkt[16] & 0x07,
                hasPrevPred = pkt[17] != 0,
                vetoMask = pkt[18]
            };

            const int fOff = 20;
            wm.visPredMae = BitConverter.ToSingle(pkt, fOff + 0);
            wm.depthPredMaeM = BitConverter.ToSingle(pkt, fOff + 4);
            wm.worldKlSurprise = BitConverter.ToSingle(pkt, fOff + 8);
            wm.actionCfSpread = BitConverter.ToSingle(pkt, fOff + 12);
            int fIdx = 4;
            CopyFloats(pkt, fOff, ref fIdx, wm.predRew, NUM_CANDS);
            CopyFloats(pkt, fOff, ref fIdx, wm.predSurv, NUM_CANDS);
            CopyFloats(pkt, fOff, ref fIdx, wm.predAdv, NUM_CANDS);

            const int gOff = 20 + 28 * 4; // 132
            for (int i = 0; i < 256; i++)
            {
                wm.predDepth16[i] = pkt[gOff + i] / 255f;
                wm.predSurf16[i] = pkt[gOff + 256 + i] / 255f;
                wm.predErr16[i] = pkt[gOff + 512 + i] / 255f;
            }

            lock (_ubntLock)
            {
                _ubwmSeqCounter++;
                wm.seq = _ubwmSeqCounter;
                _latestUbwm = wm;
            }
        }

        private void ParseUbndPacket(byte[] pkt)
        {
            uint savedCrc = BitConverter.ToUInt32(pkt, UBND_TOTAL_BYTES - 4);
            if (UniversalBrainBridge3D.ComputeCrc32(pkt, 0, UBND_TOTAL_BYTES - 4) != savedCrc)
                return;
            lock (_ubntLock)
            {
                for (int i = 0; i < 256; i++)
                    _ubndDepth16[i] = pkt[20 + i] / 255f;
                _ubndSeq++;
            }
        }

        private static void CopyFloats(byte[] pkt, int baseByteOff, ref int floatIdx, float[] dst, int count)
        {
            Buffer.BlockCopy(pkt, baseByteOff + floatIdx * 4, dst, 0, count * 4);
            floatIdx += count;
        }

        // =====================================================================
        // 2. Unity Update: Hotkeys, POV Camera Management & Canvas Redraw
        // =====================================================================
        private void Update()
        {
            if (coordinator != null)
                _targetUxPort = coordinator.baseTcpPort + 100;

            bool spectateActive = coordinator != null
                               && !coordinator.IsTrainingSession
                               && (dashboardUi == null || !dashboardUi.IsLauncherOpen);
            _wantUxStream = spectateActive && coordinator.IsBackendRunning;

            UpdatePovCameraState(spectateActive && showSubjectivePovPip);
            UpdateMainCameraViewport();

            if (!spectateActive)
                return;

#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null && GUIUtility.keyboardControl == 0)
            {
                if (kb.f3Key.wasPressedThisFrame)
                {
                    showRightNeuralPanel = !showRightNeuralPanel;
                    UpdateMainCameraViewport();
                }
                if (kb.f4Key.wasPressedThisFrame)
                {
                    showSubjectivePovPip = !showSubjectivePovPip;
                    UpdatePovCameraState(spectateActive && showSubjectivePovPip);
                }
                if (kb.f5Key.wasPressedThisFrame)
                {
                    showDepthCortexPip = !showDepthCortexPip;
                }
                if (kb.f6Key.wasPressedThisFrame)
                {
                    showWorldModelPip = !showWorldModelPip;
                }
                if (showRightNeuralPanel)
                {
                    if (kb.digit1Key.wasPressedThisFrame || kb.numpad1Key.wasPressedThisFrame) _activeTab = 0;
                    if (kb.digit2Key.wasPressedThisFrame || kb.numpad2Key.wasPressedThisFrame) _activeTab = 1;
                    if (kb.digit3Key.wasPressedThisFrame || kb.numpad3Key.wasPressedThisFrame) _activeTab = 2;
                    if (kb.digit4Key.wasPressedThisFrame || kb.numpad4Key.wasPressedThisFrame) _activeTab = 3;
                    if (kb.digit5Key.wasPressedThisFrame || kb.numpad5Key.wasPressedThisFrame) _geoMode = 0;
                    if (kb.digit6Key.wasPressedThisFrame || kb.numpad6Key.wasPressedThisFrame) _geoMode = 1;
                }
            }
#endif

            UbntSnapshot snap = null;
            lock (_ubntLock)
            {
                snap = _latestUbnt;
            }
            if (snap != null && snap.seq != _consumedUbntSeq)
            {
                _consumedUbntSeq = snap.seq;
                IngestUbntSnapshot(snap);
            }

            if (showRightNeuralPanel && Time.unscaledTime >= _nextRasterTime)
            {
                _nextRasterTime = Time.unscaledTime + 0.045f; // ~22 Hz crisp smooth redraw
                RenderMathematicalCanvas(snap);
            }

            // Strictly zero CPU/GPU depth rasterization work when `02 DEPTH` is closed!
            if (showDepthCortexPip && Time.unscaledTime >= _nextDepthRasterTime)
            {
                _nextDepthRasterTime = Time.unscaledTime + 0.050f; // ~20 Hz lightweight LUT update
                InkBlitzArenaInstance arena = (multiManager != null && multiManager.primaryArena != null)
                    ? multiManager.primaryArena
                    : FindAnyObjectByType<InkBlitzArenaInstance>();
                InkBlitzPlayerAgent agA = (arena != null && arena.agents.Count > 0) ? arena.agents[0] : null;
                RefreshDepthCortexNow(agA, forceSampleIfOffline: false);
            }

            // Strictly zero CPU/GPU rasterization work when `03 WORLD MODEL` is closed!
            if (showWorldModelPip && Time.unscaledTime >= _nextWmRasterTime)
            {
                _nextWmRasterTime = Time.unscaledTime + 0.055f;
                RefreshWorldModelNow();
            }
        }

        private void UpdateMainCameraViewport()
        {
            if (multiManager == null || multiManager.mainCamera == null)
                return;
            if (multiManager.disableAll3DRendering)
                return;

            float rightPx = ActiveRightPanelWidthPx;
            float normW = (rightPx > 1f && Screen.width > rightPx + 200f)
                ? Mathf.Clamp01((Screen.width - rightPx) / Screen.width)
                : 1f;
            Rect cur = multiManager.mainCamera.rect;
            if (Mathf.Abs(cur.width - normW) > 1e-4f || Mathf.Abs(cur.x) > 1e-4f)
                multiManager.mainCamera.rect = new Rect(0f, 0f, normW, 1f);
        }

        private void UpdatePovCameraState(bool shouldRenderPov)
        {
            if (!shouldRenderPov)
            {
                if (_povCamera != null && _povCamera.enabled)
                    _povCamera.enabled = false;
                return;
            }

            InkBlitzArenaInstance arena = (multiManager != null && multiManager.primaryArena != null)
                ? multiManager.primaryArena
                : FindAnyObjectByType<InkBlitzArenaInstance>();
            InkBlitzPlayerAgent agA = (arena != null && arena.agents.Count > 0) ? arena.agents[0] : null;
            if (agA == null || agA.eyePivot == null)
            {
                if (_povCamera != null) _povCamera.enabled = false;
                return;
            }

            if (_povRt == null)
            {
                _povRt = new RenderTexture(320, 180, 24, RenderTextureFormat.ARGB32)
                {
                    name = "AgentA_SubjectivePOV_RT",
                    antiAliasing = 2
                };
                _povRt.Create();
            }

            if (_povCamera == null || _boundEyePivot != agA.eyePivot)
            {
                if (_povCamera != null)
                    Destroy(_povCamera.gameObject);

                var camGo = new GameObject("AgentA_SubjectivePovCamera");
                camGo.transform.SetParent(agA.eyePivot, false);
                // Placed +0.28m forward along eye gaze so Agent A's own capsule/visor never occlude, while weapon tip/racket stay in peripheral FPS view!
                camGo.transform.localPosition = new Vector3(0f, 0.05f, 0.28f);
                camGo.transform.localRotation = Quaternion.identity;

                _povCamera = camGo.AddComponent<Camera>();
                _povCamera.fieldOfView = 76f; // Matches `foveaHalfAngleDeg = 38` x 2!
                _povCamera.nearClipPlane = 0.12f;
                _povCamera.farClipPlane = 48f;
                _povCamera.clearFlags = CameraClearFlags.SolidColor;
                _povCamera.backgroundColor = new Color(0.12f, 0.15f, 0.20f, 1f);
                _povCamera.targetTexture = _povRt;
                _povCamera.depth = -5f;
                _boundEyePivot = agA.eyePivot;
            }

            _povCamera.enabled = !agA.isRespawning;
        }

        private void ReleasePovResources()
        {
            if (_povCamera != null)
            {
                Destroy(_povCamera.gameObject);
                _povCamera = null;
            }
            if (_povRt != null)
            {
                _povRt.Release();
                Destroy(_povRt);
                _povRt = null;
            }
        }

        // =====================================================================
        // 3. Mathematical Signal Extraction & Info-Geometry History Ingestion
        // =====================================================================
        private void IngestUbntSnapshot(UbntSnapshot s)
        {
            _dispIgnition = Mathf.Clamp01(s.ignition);
            _dispSelfErr = Mathf.Max(0f, s.selfSurprise);
            _dispVeto = s.vetoActive;
            for (int i = 0; i < 5; i++)
                _dispAlpha[i] = Mathf.Clamp01(s.gwtAlpha[i]);

            _dispPhiBits = Mathf.Max(0f, s.phiBits);
            _dispMipMask = (s.mipCutMask >= 1 && s.mipCutMask <= 30) ? s.mipCutMask : (byte)0x03;
            _dispEntNorm = Mathf.Clamp01(s.entropyNorm);
            _dispFastCol = Mathf.Clamp(s.s1FastCand, 0, NUM_CANDS - 1);
            _dispBestCol = Mathf.Clamp(s.chosenCand, 0, NUM_CANDS - 1);

            byte verb = s.candVerbs[_dispBestCol];
            string gateStr = s.commitGateOpen ? "GO" : "HOLD";
            _controlModeText = $"M{_dispBestCol} {ShortVerb(verb)} | Gate:{gateStr}" + (s.vetoActive ? " [VETO]" : "");

            // Track episode score history
            if (_lastRecordedEp == 0)
            {
                _lastRecordedEp = s.episodeId;
                _lastEpScore = s.cumulativeScore;
            }
            else if (s.episodeId != _lastRecordedEp)
            {
                _scoreHistory.Add(_lastEpScore);
                if (_scoreHistory.Count > 50)
                    _scoreHistory.RemoveAt(0);
                _lastRecordedEp = s.episodeId;
                _lastEpScore = s.cumulativeScore;
            }
            else
            {
                _lastEpScore = s.cumulativeScore;
            }

            // Update Info-Geometric Wave Slice (`theta_a = ln(0.86 p_int + 0.14/8) - 0.65*(1 - S_a)`)
            float pSum = 0f;
            for (int i = 0; i < NUM_CANDS; i++)
                pSum += Mathf.Max(1e-6f, s.policyProbs[i]);
            float[] pInt = new float[NUM_CANDS];
            for (int i = 0; i < NUM_CANDS; i++)
                pInt[i] = Mathf.Max(1e-6f, s.policyProbs[i]) / Mathf.Max(1e-6f, pSum);

            float qMin = s.qValues[0], qMax = s.qValues[0];
            for (int i = 1; i < NUM_CANDS; i++)
            {
                if (s.qValues[i] < qMin) qMin = s.qValues[i];
                if (s.qValues[i] > qMax) qMax = s.qValues[i];
            }
            float qSpan = qMax - qMin;
            float[] qNorm = new float[NUM_CANDS];
            for (int i = 0; i < NUM_CANDS; i++)
                qNorm[i] = (qSpan > 1e-4f) ? ((s.qValues[i] - qMin) / qSpan) : 0.30f;

            float[] sig = new float[NUM_CANDS * 3];
            for (int i = 0; i < NUM_CANDS; i++)
            {
                sig[i] = pInt[i];
                sig[NUM_CANDS + i] = s.survivalProbs[i];
                sig[NUM_CANDS * 2 + i] = qNorm[i] * 0.25f;
            }
            float sigDiff = 1f;
            if (_lastGeoSig != null)
            {
                sigDiff = 0f;
                for (int i = 0; i < sig.Length; i++)
                    sigDiff += Mathf.Abs(sig[i] - _lastGeoSig[i]);
            }

            float now = Time.unscaledTime;
            if (!_geoSeeded || (sigDiff > 0.003f && (now - _lastGeoPushTime) >= 0.03f))
            {
                _displayFrEma = Mathf.Max(s.fisherRaoStep, _displayFrEma * 0.84f);
                _dispFrStep = _displayFrEma;
                _lastGeoSig = sig;
                _lastGeoPushTime = now;

                float[] theta = new float[NUM_CANDS];
                float thMin = float.MaxValue, thMax = float.MinValue;
                const float unif = 1f / NUM_CANDS;
                for (int i = 0; i < NUM_CANDS; i++)
                {
                    float pTemp = 0.86f * Mathf.Clamp(pInt[i], 1e-5f, 1f) + 0.14f * unif;
                    float surv = Mathf.Clamp(s.survivalProbs[i], 0.02f, 1f);
                    float th = Mathf.Log(pTemp) - 0.65f * (1f - surv);
                    theta[i] = th;
                    if (th < thMin) thMin = th;
                    if (th > thMax) thMax = th;
                }

                float[] rawWave = new float[NUM_CANDS];
                for (int i = 0; i < NUM_CANDS; i++)
                {
                    float thN = (thMax - thMin > 1e-5f) ? ((theta[i] - thMin) / (thMax - thMin)) : 0.30f;
                    rawWave[i] = 0.78f * thN + 0.22f * qNorm[i];
                }

                if (_smoothGeoWave == null || !_geoSeeded)
                {
                    _smoothGeoWave = (float[])rawWave.Clone();
                }
                else
                {
                    for (int i = 0; i < NUM_CANDS; i++)
                        _smoothGeoWave[i] = 0.42f * _smoothGeoWave[i] + 0.58f * rawWave[i];
                }

                float swSum = 0f, fxAcc = 0f;
                for (int i = 0; i < NUM_CANDS; i++)
                {
                    float w = Mathf.Pow(pInt[i], 2.4f);
                    swSum += w;
                    fxAcc += (i / (float)(NUM_CANDS - 1)) * w;
                }
                float targetFx = fxAcc / Mathf.Max(1e-9f, swSum);
                _smoothFocusX = !_geoSeeded ? targetFx : (0.42f * _smoothFocusX + 0.58f * targetFx);

                var entry = new GeoSlice
                {
                    wave = (float[])_smoothGeoWave.Clone(),
                    surv = (float[])s.survivalProbs.Clone(),
                    phiBits = _dispPhiBits,
                    frStep = _dispFrStep,
                    entNorm = _dispEntNorm,
                    focusX = _smoothFocusX,
                    bestCol = _dispBestCol,
                    veto = _dispVeto
                };

                if (!_geoSeeded)
                {
                    _geoHistory.Clear();
                    for (int k = 0; k < GEO_HIST_LEN; k++)
                    {
                        _geoHistory.Add(new GeoSlice
                        {
                            wave = (float[])entry.wave.Clone(),
                            surv = (float[])entry.surv.Clone(),
                            phiBits = entry.phiBits,
                            frStep = 0f,
                            entNorm = entry.entNorm,
                            focusX = entry.focusX,
                            bestCol = entry.bestCol,
                            veto = entry.veto
                        });
                    }
                    _geoSeeded = true;
                }
                else
                {
                    if (_geoHistory.Count >= GEO_HIST_LEN)
                        _geoHistory.RemoveAt(0);
                    _geoHistory.Add(entry);
                }

                if (_phiTrace.Count >= 44)
                    _phiTrace.RemoveAt(0);
                _phiTrace.Add(_dispPhiBits);
            }
        }

        // =====================================================================
        // 4. Software Canvas Rendering (`#0e1115` Cyber-Dark Mathematical Aesthetic)
        // =====================================================================
        private void RenderMathematicalCanvas(UbntSnapshot snap)
        {
            ClearCanvas(new Color32(14, 17, 21, 255));
            // Left vertical border divider
            DrawLine(0, 0, 0, CANVAS_H - 1, new Color32(42, 50, 64, 255));

            DrawCorticalSheetAndAttractor(snap);
            DrawMiddleGwtCard();
            if (_geoMode == 0)
                DrawInfoGeoLandscape();
            else
                DrawPhiMipIntegration(snap);
            DrawScoreHistoryGraph();

            _canvasTex.SetPixels32(_pixels);
            _canvasTex.Apply(false);
        }

        public void RenderSnapshotForTest(UbntSnapshot snap, int tabIdx = 0, int geoMode = 1)
        {
            if (_pixels == null)
                InitMathBuffers();
            _activeTab = Mathf.Clamp(tabIdx, 0, 3);
            _geoMode = Mathf.Clamp(geoMode, 0, 1);
            if (snap != null)
                IngestUbntSnapshot(snap);
            RenderMathematicalCanvas(snap);
        }

        private void DrawCorticalSheetAndAttractor(UbntSnapshot snap)
        {
            int ti = _activeTab;
            int nUnits = _tabSizes[ti];
            float[] rawSheet = new float[nUnits];
            float[] normSheet = new float[nUnits];
            float[] rawVec;
            float deltaScale = 0.25f;

            // Strictly honest physical activation normalization per brain region (zero rank-percentile auto-stretching!)
            if (snap != null)
            {
                if (ti == 0)
                {
                    // 1:GWT (`gwt_workspace[128]`, LayerNorm output)
                    rawVec = snap.gwtW;
                    Array.Copy(snap.gwtW, rawSheet, 128);
                    deltaScale = 0.55f;
                    int activeCnt = 0;
                    for (int i = 0; i < 128; i++)
                    {
                        float av = Mathf.Abs(rawSheet[i]);
                        if (av > 0.45f) activeCnt++;
                        normSheet[i] = Mathf.Clamp01((av - 0.30f) / 1.55f);
                    }
                    int actPct = Mathf.RoundToInt(100f * activeCnt / 128f);
                    _regionSubText = $"GWT W_t [16x8=128] | ACT:{actPct,2}% | IGN:{_dispIgnition:F2}";
                }
                else if (ti == 1)
                {
                    // 2:WM (`rssm_h[256]`, GRUCell deterministic hidden state in [-1, 1])
                    rawVec = snap.rssmH;
                    Array.Copy(snap.rssmH, rawSheet, 256);
                    deltaScale = 0.22f;
                    int activeCnt = 0;
                    for (int i = 0; i < 256; i++)
                    {
                        float av = Mathf.Abs(rawSheet[i]);
                        if (av > 0.18f) activeCnt++;
                        normSheet[i] = Mathf.Clamp01((av - 0.08f) / 0.58f);
                    }
                    int actPct = Mathf.RoundToInt(100f * activeCnt / 256f);
                    _regionSubText = $"RSSM h_t [16x16=256] | ACT:{actPct,2}% | SURP:{snap.worldSurprise:F2}";
                }
                else if (ti == 2)
                {
                    // 3:SDR (`latent_z[32]` top 8x16 + `z_topdown_delta[32]` bottom 8x16, true unstretched magnitudes)
                    rawVec = new float[64];
                    for (int i = 0; i < 32; i++)
                    {
                        rawVec[i] = snap.latentZ[i];
                        rawVec[32 + i] = snap.zDelta[i];
                    }
                    deltaScale = 0.20f;
                    int activeCnt = 0;
                    for (int r = 0; r < 4; r++)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            float vz = snap.latentZ[r * 8 + c];
                            float vd = snap.zDelta[r * 8 + c];
                            float nz = Mathf.Clamp01((Mathf.Abs(vz) - 0.10f) / 0.68f);
                            float nd = Mathf.Clamp01((Mathf.Abs(vd) - 0.015f) / 0.22f);
                            if (nz > 0.18f) activeCnt += 4;
                            if (nd > 0.18f) activeCnt += 4;
                            for (int dr = 0; dr < 2; dr++)
                            {
                                for (int dc = 0; dc < 2; dc++)
                                {
                                    int idxTop = (r * 2 + dr) * 16 + (c * 2 + dc);
                                    int idxBot = (8 + r * 2 + dr) * 16 + (c * 2 + dc);
                                    rawSheet[idxTop] = vz;
                                    rawSheet[idxBot] = vd * 3.0f;
                                    normSheet[idxTop] = nz;
                                    normSheet[idxBot] = nd;
                                }
                            }
                        }
                    }
                    int actPct = Mathf.RoundToInt(100f * activeCnt / 256f);
                    float dzNorm = Norm(snap.zDelta);
                    _regionSubText = $"SENSORY z+z_td | ACT:{actPct,2}% | |z_td|:{dzNorm:F2}";
                }
                else
                {
                    // 4:MEC (`mec_grid_code[256]`, 3D FCC grid-cell firing rates in [0, 1])
                    rawVec = snap.mecGrid;
                    Array.Copy(snap.mecGrid, rawSheet, 256);
                    deltaScale = 0.20f;
                    int activeCnt = 0;
                    for (int i = 0; i < 256; i++)
                    {
                        float av = Mathf.Abs(rawSheet[i]);
                        if (av > 0.22f) activeCnt++;
                        normSheet[i] = Mathf.Clamp01((av - 0.16f) / 0.66f);
                    }
                    int actPct = Mathf.RoundToInt(100f * activeCnt / 256f);
                    _regionSubText = $"3D-MEC GRID [16x16=256] | ACT:{actPct,2}% | e_s:{_dispSelfErr:F2}";
                }
            }
            else
            {
                rawVec = rawSheet;
                _regionSubText = _uxConnected ? "SYNCING NEURAL STREAM..." : "CONNECTING UBNT STREAM...";
            }

            // Strictly real phasic temporal derivative `|x_i(t) - x_i(t-1)|` (zero random Poisson fake sparks!)
            float[] prev = _tabPrevVec[ti];
            float[] burst = new float[nUnits];
            float meanAbsDelta = 0f;
            if (prev != null && prev.Length == nUnits)
            {
                float dFloor = deltaScale * 0.35f;
                float dSpan = Mathf.Max(1e-4f, deltaScale * 1.15f);
                for (int i = 0; i < nUnits; i++)
                {
                    float diff = Mathf.Abs(rawSheet[i] - prev[i]);
                    meanAbsDelta += diff;
                    if (diff > dFloor)
                        burst[i] = Mathf.Clamp01((diff - dFloor) / dSpan);
                }
                meanAbsDelta /= nUnits;
            }
            _tabPrevVec[ti] = (float[])rawSheet.Clone();
            _tabLastDeltaNorm[ti] = Mathf.Clamp01(meanAbsDelta / Mathf.Max(1e-4f, deltaScale));

            float[] smooth = _tabSmoothSheet[ti];
            float[] spikes = _tabSpikeSheet[ti];
            for (int i = 0; i < nUnits; i++)
            {
                smooth[i] = 0.78f * smooth[i] + 0.22f * normSheet[i];
                float decayed = spikes[i] * 0.76f;
                if (decayed < 0.015f) decayed = 0f;
                spikes[i] = Mathf.Max(decayed, burst[i]);
            }

            // 1. Real 2D Orthogonal Projection Phase Trajectory (Catmull-Rom Spline, Top-Right)
            if (snap != null)
            {
                Vector2 pt2d = ProjectOrthogonal2D(ti, rawVec, out float[] vUnit);
                float[] lastU = _tabLastPushUnit[ti];
                float dUnit = 1f;
                if (lastU != null && lastU.Length == vUnit.Length)
                {
                    float s2 = 0f;
                    for (int i = 0; i < vUnit.Length; i++) { float d = vUnit[i] - lastU[i]; s2 += d * d; }
                    dUnit = Mathf.Sqrt(s2);
                }
                if (lastU == null || dUnit > 0.025f || _tabTraj[ti].Count < 4)
                {
                    if (_tabTraj[ti].Count >= 16) _tabTraj[ti].RemoveAt(0);
                    _tabTraj[ti].Add(pt2d);
                    _tabLastPushUnit[ti] = vUnit;
                }
            }
            DrawAttractorTrace(ti);

            // 2. Anatomically Fixed 3D Oblique Cortical Sheet (zero artificial turntable spin)
            const float theta = 0.38f;
            float cosT = Mathf.Cos(theta);
            float sinT = Mathf.Sin(theta);
            const float cx = 158f;
            const float cy = 215f;
            const float rx = 96f;
            const float ry = 24f;

            Vector2 ProjectSheet(float u, float v, float liftPx, out float yrOut)
            {
                float xr = u * cosT - v * sinT;
                float yr = u * sinT + v * cosT;
                float persp = 1f / (1f - 0.13f * yr);
                yrOut = yr;
                return new Vector2(cx + xr * rx * persp, cy + yr * ry * persp - liftPx);
            }

            Vector2 c0 = ProjectSheet(-1f, -1f, 0f, out _);
            Vector2 c1 = ProjectSheet(1f, -1f, 0f, out _);
            Vector2 c2 = ProjectSheet(1f, 1f, 0f, out _);
            Vector2 c3 = ProjectSheet(-1f, 1f, 0f, out _);

            FillQuad(
                new Vector2(c0.x, c0.y + 3.5f), new Vector2(c1.x, c1.y + 3.5f),
                new Vector2(c2.x, c2.y + 3.5f), new Vector2(c3.x, c3.y + 3.5f),
                new Color32(36, 34, 44, 140));
            FillQuad(c0, c1, c2, c3, new Color32(20, 22, 29, 230));

            float[] gTicks = { -0.5f, 0f, 0.5f };
            foreach (float gt in gTicks)
            {
                Vector2 pa = ProjectSheet(gt, -1f, 0f, out _);
                Vector2 pb = ProjectSheet(gt, 1f, 0f, out _);
                DrawLine((int)pa.x, (int)pa.y, (int)pb.x, (int)pb.y, new Color32(110, 105, 124, 45));
                Vector2 qa = ProjectSheet(-1f, gt, 0f, out _);
                Vector2 qb = ProjectSheet(1f, gt, 0f, out _);
                DrawLine((int)qa.x, (int)qa.y, (int)qb.x, (int)qb.y, new Color32(110, 105, 124, 45));
            }
            DrawQuadOutline(c0, c1, c2, c3, new Color32(158, 144, 165, 210));

            float[] gu = (nUnits == 128) ? _gridU128 : _gridU256;
            float[] gv = (nUnits == 128) ? _gridV128 : _gridV256;
            int[] order = new int[nUnits];
            float[] yrArr = new float[nUnits];
            Vector2[] scrPts = new Vector2[nUnits];
            for (int i = 0; i < nUnits; i++)
            {
                order[i] = i;
                float lift = smooth[i] * 4.2f + spikes[i] * 3.0f;
                scrPts[i] = ProjectSheet(gu[i], gv[i], lift, out yrArr[i]);
            }
            Array.Sort(yrArr, order);

            for (int k = 0; k < nUnits; k++)
            {
                int idx = order[k];
                float av = smooth[idx];
                float sv = spikes[idx];
                int px = Mathf.RoundToInt(scrPts[idx].x);
                int py = Mathf.RoundToInt(scrPts[idx].y);

                if (sv > 0.34f && av > 0.04f)
                {
                    byte glowA = (byte)Mathf.Clamp(45f + sv * 110f, 0f, 140f);
                    FillCircle(px, py, 3, new Color32(125, 175, 232, glowA));
                    byte cr = (byte)Mathf.Clamp(185f + 70f * Mathf.Min(1f, sv), 0f, 255f);
                    byte cg = (byte)Mathf.Clamp(215f + 40f * Mathf.Min(1f, sv), 0f, 255f);
                    byte ca = (byte)Mathf.Clamp(185f + sv * 70f, 0f, 255f);
                    FillCircle(px, py, sv > 0.62f ? 2 : 1, new Color32(cr, cg, 255, ca));
                }
                else if (av > 0.12f)
                {
                    float mix = Mathf.Clamp01((av - 0.12f) / 0.76f);
                    byte cr = (byte)(95f * (1f - mix) + 168f * mix);
                    byte cg = (byte)(132f * (1f - mix) + 142f * mix);
                    byte cb = (byte)(176f * (1f - mix) + 178f * mix);
                    byte ca = (byte)(68f + 150f * mix);
                    FillCircle(px, py, 1, new Color32(cr, cg, cb, ca));
                }
                else if (nUnits == 128 || (idx & 1) == 0)
                {
                    BlendPixel(px, py, new Color32(72, 80, 96, 55));
                }
            }
        }

        private void DrawAttractorTrace(int tabIdx)
        {
            List<Vector2> traj = _tabTraj[tabIdx];
            var wps = new List<Vector2>(8);
            if (traj.Count >= 2)
            {
                int mLen = Mathf.Min(8, traj.Count);
                for (int i = 0; i < mLen; i++)
                {
                    int srcIdx = Mathf.RoundToInt(i * (traj.Count - 1) / (float)Mathf.Max(1, mLen - 1));
                    Vector2 v = traj[srcIdx];
                    float sProg = i / (float)Mathf.Max(1, mLen - 1);
                    float zx = Mathf.Clamp(v.x, -1.25f, 1.25f);
                    float zy = Mathf.Clamp(v.y, -1.25f, 1.25f);
                    float wx = Mathf.Clamp(218f + 92f * sProg + zx * 11f, 198f, PANEL_W - 14f);
                    float wy = Mathf.Clamp(132f - 18f * sProg + zy * 13.5f, 88f, 156f);
                    wps.Add(new Vector2(wx, wy));
                }
            }
            else
            {
                wps.Add(new Vector2(220f, 132f));
                wps.Add(new Vector2(310f, 114f));
            }

            List<Vector2> spline = CatmullRomChain(wps, 5);
            for (int i = 0; i < spline.Count - 1; i++)
            {
                float prog = i / (float)Mathf.Max(1, spline.Count - 1);
                byte la = (byte)Mathf.Clamp(68f + 145f * prog, 0f, 220f);
                DrawLine(Mathf.RoundToInt(spline[i].x), Mathf.RoundToInt(spline[i].y),
                         Mathf.RoundToInt(spline[i + 1].x), Mathf.RoundToInt(spline[i + 1].y),
                         new Color32(124, 148, 182, la));
            }

            // Plot ONLY genuine historical state waypoints along the 2D Gram-Schmidt phase projection (zero fake Gaussian clouds!)
            for (int i = 0; i < wps.Count - 1; i++)
            {
                float age01 = (i + 1f) / wps.Count;
                byte wa = (byte)Mathf.Clamp(60f + 125f * age01, 0f, 200f);
                FillCircle(Mathf.RoundToInt(wps[i].x), Mathf.RoundToInt(wps[i].y), 1, new Color32(118, 154, 196, wa));
            }

            Vector2 pEnd = wps[wps.Count - 1];
            float delta01 = (_tabLastDeltaNorm != null && tabIdx < _tabLastDeltaNorm.Length) ? _tabLastDeltaNorm[tabIdx] : 0f;
            int haloR = 2 + Mathf.RoundToInt(3f * delta01);
            byte haloA = (byte)Mathf.Clamp(40f + 95f * delta01, 0f, 145f);
            FillCircle(Mathf.RoundToInt(pEnd.x), Mathf.RoundToInt(pEnd.y), haloR, new Color32(125, 175, 232, haloA));
            FillCircle(Mathf.RoundToInt(pEnd.x), Mathf.RoundToInt(pEnd.y), 2, new Color32(226, 240, 255, 240));
        }

        private void DrawMiddleGwtCard()
        {
            const int px = 16, py = 274, pw = PANEL_W - 32, ph = 76;
            FillRect(px, py, pw, ph, new Color32(22, 27, 34, 220));
            DrawRectOutline(px, py, pw, ph, new Color32(100, 125, 150, 88));

            // Ignition bar
            const int bx = px + 8, bw = 142, bh = 8;
            FillRect(bx, py + 23, bw, bh, new Color32(40, 48, 58, 255));
            Color32 ignCol = _dispVeto
                ? new Color32(253, 164, 175, 255)
                : (_dispIgnition >= 0.45f ? new Color32(255, 230, 190, 255) : new Color32(181, 225, 252, 255));
            FillRect(bx, py + 23, Mathf.RoundToInt(bw * _dispIgnition), bh, ignCol);

            // Self-Error bar
            FillRect(bx, py + 54, bw, bh, new Color32(40, 48, 58, 255));
            float eNorm = Mathf.Clamp01(_dispSelfErr / 2.0f);
            Color32 eCol = _dispSelfErr > 0.85f ? new Color32(253, 164, 175, 255) : new Color32(208, 175, 203, 255);
            FillRect(bx, py + 54, Mathf.RoundToInt(bw * eNorm), bh, eCol);

            // 5-Module Salience Arbitration [P, I, W, M, S]
            int sx0 = px + 172;
            const int maxBarH = 36;
            const int byBot = py + 59;
            float maxA = 0f;
            for (int i = 0; i < 5; i++) if (_dispAlpha[i] > maxA) maxA = _dispAlpha[i];

            for (int i = 0; i < 5; i++)
            {
                int colX = sx0 + i * 28;
                FillRect(colX, byBot - maxBarH, 18, maxBarH, new Color32(35, 42, 52, 255));
                int h = Mathf.Max(2, Mathf.RoundToInt(maxBarH * Mathf.Clamp01(_dispAlpha[i])));
                Color32 barC = (_dispAlpha[i] >= maxA - 1e-4f)
                    ? new Color32(255, 230, 190, 255)
                    : new Color32(125, 185, 225, 255);
                FillRect(colX, byBot - h, 18, h, barC);
            }
        }

        private void DrawInfoGeoLandscape()
        {
            int nHist = _geoHistory.Count;
            if (nHist < 2) return;

            const float waveCy = 580f;
            const float ampPx = 68f;
            const int nPts = 52;
            float[][] smoothRows = new float[nHist][];
            float[][] riskRows = new float[nHist][];
            float[] xGrid = new float[nPts];
            for (int i = 0; i < nPts; i++) xGrid[i] = i / (float)(nPts - 1);

            for (int j = 0; j < nHist; j++)
            {
                GeoSlice sl = _geoHistory[j];
                float[] invSurv = new float[NUM_CANDS];
                for (int c = 0; c < NUM_CANDS; c++) invSurv[c] = 1f - Mathf.Clamp01(sl.surv[c]);
                smoothRows[j] = UpsampleCatmull1D(sl.wave, xGrid);
                riskRows[j] = UpsampleCatmull1D(invSurv, xGrid);
            }

            void SliceProj(int jIdx, out float dVal, out float xsVal, out float wVal, out float ybVal)
            {
                dVal = jIdx / (float)Mathf.Max(1, nHist - 1);
                wVal = PANEL_W * (0.40f + 0.54f * dVal);
                xsVal = (PANEL_W - wVal) * 0.5f;
                ybVal = waveCy - (nHist - 1 - jIdx) * 4.6f;
            }

            var geoWps = new List<Vector2>(nHist);
            for (int j = 0; j < nHist; j++)
            {
                SliceProj(j, out float dj, out float xsj, out float wj, out float ybj);
                float fx = Mathf.Clamp01(_geoHistory[j].focusX);
                int hi = Mathf.Clamp(Mathf.RoundToInt(fx * (nPts - 1)), 0, nPts - 1);
                float hFx = smoothRows[j][hi];
                geoWps.Add(new Vector2(xsj + fx * wj, ybj - hFx * ampPx * dj - 1.2f));
            }
            const int subdivGeo = 3;
            List<Vector2> splinePts = CatmullRomChain(geoWps, subdivGeo);

            for (int j = 0; j < nHist - 1; j++)
            {
                SliceProj(j, out float db, out float xsb, out float wb, out float ybb);
                SliceProj(j + 1, out float df, out float xsf, out float wf, out float ybf);
                float[] hb = smoothRows[j];
                float[] hf = smoothRows[j + 1];
                float[] rk = riskRows[j + 1];

                for (int i = 0; i < nPts - 1; i++)
                {
                    Vector2 pBl = new Vector2(xsb + xGrid[i] * wb, ybb - hb[i] * ampPx * db);
                    Vector2 pBr = new Vector2(xsb + xGrid[i + 1] * wb, ybb - hb[i + 1] * ampPx * db);
                    Vector2 pFr = new Vector2(xsf + xGrid[i + 1] * wf, ybf - hf[i + 1] * ampPx * df);
                    Vector2 pFl = new Vector2(xsf + xGrid[i] * wf, ybf - hf[i] * ampPx * df);

                    float val = Mathf.Max(hf[i], hf[i + 1]);
                    float rkV = Mathf.Max(rk[i], rk[i + 1]);
                    float ratio = Mathf.Clamp01((val - 0.10f) / 0.70f);

                    float rBase = 125f * (1f - ratio) + 165f * ratio;
                    float gBase = 158f * (1f - ratio) + 138f * ratio;
                    float bBase = 198f * (1f - ratio) + 164f * ratio;
                    float rkMix = Mathf.Min(0.52f, rkV * 0.55f);

                    byte cr = (byte)(rBase * (1f - rkMix) + 198f * rkMix);
                    byte cg = (byte)(gBase * (1f - rkMix) + 124f * rkMix);
                    byte cb = (byte)(bBase * (1f - rkMix) + 142f * rkMix);
                    byte ca = (byte)Mathf.Clamp(22f + 196f * df, 0f, 255f);
                    FillQuad(pBl, pBr, pFr, pFl, new Color32(cr, cg, cb, ca));
                }

                int spBase = j * subdivGeo;
                if (spBase + subdivGeo < splinePts.Count)
                {
                    float leap = Mathf.Min(1f, _geoHistory[j + 1].frStep / 1.4f);
                    if (_geoHistory[j + 1].veto) leap = Mathf.Max(leap, 0.75f);
                    byte lr = (byte)(190f * (1f - leap) + 255f * leap);
                    byte lg = (byte)(226f * (1f - leap) + 232f * leap);
                    byte lb = (byte)(252f * (1f - leap) + 182f * leap);
                    byte la = (byte)Mathf.Clamp(36f + 195f * Mathf.Pow(df, 1.2f), 0f, 255f);
                    for (int sk = 0; sk < subdivGeo; sk++)
                    {
                        Vector2 a = splinePts[spBase + sk];
                        Vector2 b = splinePts[spBase + sk + 1];
                        DrawLine(Mathf.RoundToInt(a.x), Mathf.RoundToInt(a.y), Mathf.RoundToInt(b.x), Mathf.RoundToInt(b.y),
                                 new Color32(lr, lg, lb, la));
                    }
                }

                // Front silhouette cutout (#0e1115) for crisp Q-Wave silk edge
                if (j == nHist - 2)
                {
                    float pyBot = Mathf.Min(612f, ybf + 32f);
                    for (int i = 0; i < nPts - 1; i++)
                    {
                        float pxl = xsf + xGrid[i] * wf;
                        float pxr = xsf + xGrid[i + 1] * wf;
                        float pyl = ybf - hf[i] * ampPx * df;
                        float pyr = ybf - hf[i + 1] * ampPx * df;
                        FillQuad(new Vector2(pxl, pyl), new Vector2(pxr, pyr),
                                 new Vector2(pxr, pyBot), new Vector2(pxl, pyBot),
                                 new Color32(14, 17, 21, 255));
                    }
                }
            }

            if (geoWps.Count > 0)
            {
                Vector2 gNow = geoWps[geoWps.Count - 1];
                int rxA = Mathf.RoundToInt(6f + 26f * _dispEntNorm);
                int ryA = Mathf.RoundToInt(3f + 7f * _dispEntNorm);
                Color32 auraC = _dispVeto ? new Color32(255, 218, 178, 42) : new Color32(195, 232, 255, 42);
                FillEllipse(Mathf.RoundToInt(gNow.x), Mathf.RoundToInt(gNow.y), rxA, ryA, auraC);
                FillCircle(Mathf.RoundToInt(gNow.x), Mathf.RoundToInt(gNow.y), 2, new Color32(248, 252, 255, 240));
            }
        }

        private void DrawPhiMipIntegration(UbntSnapshot snap)
        {
            float phiRatio = Mathf.Clamp01(_dispPhiBits / 2.2f);
            float liftPx = phiRatio * 92f;
            const float mcx = 128f, mcy = 564f;
            float cyw = Mathf.Cos(0.22f), syw = Mathf.Sin(0.22f);

            Vector2 ProjMi(float uc, float vc, float hUp)
            {
                float xr = uc * cyw - vc * syw;
                float yr = uc * syw + vc * cyw;
                float persp = 1f / (1f - 0.12f * yr);
                return new Vector2(mcx + xr * 86f * persp, mcy + yr * 25f * persp - hUp);
            }

            Vector2 s0 = ProjMi(-1f, -0.95f, 0f);
            Vector2 s1 = ProjMi(1f, -0.95f, 0f);
            Vector2 s2 = ProjMi(1f, 0.95f, 0f);
            Vector2 s3 = ProjMi(-1f, 0.95f, 0f);
            FillQuad(new Vector2(s0.x, s0.y + 5f), new Vector2(s1.x, s1.y + 5f),
                     new Vector2(s2.x, s2.y + 5f), new Vector2(s3.x, s3.y + 5f), new Color32(42, 38, 36, 225));
            FillQuad(s0, s1, s2, s3, new Color32(62, 56, 52, 215));

            Vector2 mipBack = ProjMi(0f, -0.95f, 0f);
            Vector2 mipFront = ProjMi(0f, 0.95f, 0f);
            DrawLine((int)mipBack.x, (int)mipBack.y, (int)mipFront.x, (int)mipFront.y, new Color32(214, 92, 100, 185));
            DrawQuadOutline(s0, s1, s2, s3, new Color32(176, 158, 140, 205));

            float uStar = Mathf.Clamp((_dispBestCol - 3.5f) / 3.5f, -0.68f, 0.68f) * 0.65f;
            float vStar = Mathf.Clamp((0.55f - _dispEntNorm) * 0.80f, -0.58f, 0.58f);
            float uS1 = Mathf.Clamp((_dispFastCol - 3.5f) / 3.5f, -0.72f, 0.72f) * 0.65f;
            float vS1 = vStar + 0.18f;
            if (Mathf.Abs(uS1 - uStar) < 0.22f)
                uS1 = Mathf.Clamp(uStar + (uStar <= 0.2f ? 0.34f : -0.34f), -0.72f, 0.72f);

            Vector2 qStarPt = ProjMi(uStar, vStar, 0f);
            Vector2 qS1Pt = ProjMi(uS1, vS1, 0f);
            Vector2 pIntPt = ProjMi(uStar, vStar, liftPx);
            _lastQStarCanvasPt = qStarPt;
            _lastQS1CanvasPt = qS1Pt;
            _lastPIntCanvasPt = pIntPt;
            _lastLiftPx = liftPx;

            // Dynamically partition the 5 modules `{P, I, W, M, S}` according to live Oizumi-2016 MIP cut `_dispMipMask`!
            var partA = new List<int>(2);
            var partB = new List<int>(4);
            byte effMask = (_dispMipMask >= 1 && _dispMipMask <= 30) ? _dispMipMask : (byte)0x03;
            for (int i = 0; i < 5; i++)
            {
                if (((effMask >> i) & 1) != 0) partA.Add(i);
                else partB.Add(i);
            }
            if (partA.Count > partB.Count)
            {
                var tmp = partA; partA = partB; partB = tmp;
            }

            Vector2[] modUv = new Vector2[5];
            for (int k = 0; k < partA.Count; k++)
            {
                float v = (partA.Count == 1) ? 0.02f : Mathf.Lerp(-0.36f, 0.38f, k / (float)(partA.Count - 1));
                modUv[partA[k]] = new Vector2(-0.52f - 0.04f * (k & 1), v);
                float y2d = (partA.Count == 1) ? 466f : Mathf.Lerp(450f, 484f, k / (float)(partA.Count - 1));
                _lastMipNode2d[partA[k]] = new Vector2(266f, y2d);
            }
            for (int k = 0; k < partB.Count; k++)
            {
                float v = (partB.Count == 1) ? 0.02f : Mathf.Lerp(-0.46f, 0.48f, k / (float)(partB.Count - 1));
                modUv[partB[k]] = new Vector2(0.50f + 0.08f * ((k == 1 || k == 2) ? 1f : 0f), v);
                float y2d = (partB.Count == 1) ? 466f : Mathf.Lerp(440f, 492f, k / (float)(partB.Count - 1));
                _lastMipNode2d[partB[k]] = new Vector2(320f, y2d);
            }

            Vector2[] modPts = new Vector2[5];
            for (int i = 0; i < 5; i++)
            {
                float mh = liftPx * (0.24f + 0.50f * Mathf.Min(1f, _dispAlpha[i] * 2.3f));
                modPts[i] = ProjMi(modUv[i].x, modUv[i].y, mh);
            }

            // Intra-partition links within M_1 and within M_2
            for (int k = 0; k < partA.Count - 1; k++)
            {
                Vector2 a = modPts[partA[k]], b = modPts[partA[k + 1]];
                DrawLine((int)a.x, (int)a.y, (int)b.x, (int)b.y, new Color32(168, 112, 116, 155));
            }
            for (int k = 0; k < partB.Count - 1; k++)
            {
                Vector2 a = modPts[partB[k]], b = modPts[partB[k + 1]];
                DrawLine((int)a.x, (int)a.y, (int)b.x, (int)b.y, new Color32(168, 112, 116, 155));
            }

            if (phiRatio > 0.03f)
            {
                byte ca = (byte)Mathf.Clamp(55f + 165f * phiRatio, 0f, 255f);
                for (int ka = 0; ka < partA.Count; ka++)
                {
                    for (int kb = 0; kb < partB.Count; kb++)
                    {
                        if ((ka + kb) % 2 == 0 || partA.Count == 1)
                        {
                            Vector2 a = modPts[partA[ka]], b = modPts[partB[kb]];
                            DrawLine((int)a.x, (int)a.y, (int)b.x, (int)b.y, new Color32(206, 98, 104, ca));
                        }
                    }
                }
                for (int i = 0; i < 5; i++)
                    DrawLine((int)modPts[i].x, (int)modPts[i].y, (int)pIntPt.x, (int)pIntPt.y, new Color32(222, 138, 132, ca));

                // Pythagoras base & dashed hypotenuse
                DrawLine((int)qStarPt.x, (int)qStarPt.y, (int)qS1Pt.x, (int)qS1Pt.y, new Color32(148, 140, 132, 150));
                for (int kd = 0; kd < 9; kd += 2)
                {
                    float t0 = kd / 9f, t1 = Mathf.Min(1f, (kd + 1) / 9f);
                    Vector2 d0 = Vector2.Lerp(pIntPt, qS1Pt, t0);
                    Vector2 d1 = Vector2.Lerp(pIntPt, qS1Pt, t1);
                    DrawLine((int)d0.x, (int)d0.y, (int)d1.x, (int)d1.y, new Color32(182, 168, 160, 170));
                }
            }

            for (int i = 0; i < 5; i++)
                FillCircle((int)modPts[i].x, (int)modPts[i].y, 2, new Color32(242, 214, 138, 235));

            int qx = Mathf.RoundToInt(qStarPt.x), qy = Mathf.RoundToInt(qStarPt.y);
            int px = Mathf.RoundToInt(pIntPt.x), py = Mathf.RoundToInt(pIntPt.y);
            if (liftPx >= 6f)
            {
                int rDir = (qS1Pt.x >= qStarPt.x) ? 7 : -7;
                DrawLine(qx, qy - 7, qx + rDir, qy - 7, new Color32(235, 224, 205, 220));
                DrawLine(qx + rDir, qy - 7, qx + rDir, qy, new Color32(235, 224, 205, 220));
                Color32 beamCol = (_dispVeto || _dispPhiBits >= 0.5f)
                    ? new Color32(255, 222, 162, 245)
                    : new Color32(175, 224, 255, 245);
                DrawLine(qx, qy, px, py, beamCol);
                DrawLine(qx + 1, qy, px + 1, py, beamCol);
            }
            FillCircle((int)qS1Pt.x, (int)qS1Pt.y, 3, new Color32(166, 176, 192, 215));
            FillCircle(qx, qy, 3, new Color32(232, 205, 168, 245));
            FillCircle(px, py, 3, new Color32(248, 252, 255, 255));

            // Right sub-boxes: Dynamic Oizumi MIP Partition & Phi_G(t) Trace
            FillRect(246, 418, 94, 88, new Color32(20, 24, 31, 215));
            DrawRectOutline(246, 418, 94, 88, new Color32(88, 102, 122, 95));
            for (int dy = 436; dy < 500; dy += 6)
                DrawLine(293, dy, 293, Mathf.Min(500, dy + 3), new Color32(214, 92, 100, 185));

            if (partA.Count >= 2)
                DrawLine((int)_lastMipNode2d[partA[0]].x, (int)_lastMipNode2d[partA[0]].y, (int)_lastMipNode2d[partA[partA.Count - 1]].x, (int)_lastMipNode2d[partA[partA.Count - 1]].y, new Color32(130, 150, 175, 150));
            if (partB.Count >= 2)
                DrawLine((int)_lastMipNode2d[partB[0]].x, (int)_lastMipNode2d[partB[0]].y, (int)_lastMipNode2d[partB[partB.Count - 1]].x, (int)_lastMipNode2d[partB[partB.Count - 1]].y, new Color32(130, 150, 175, 150));
            if (phiRatio > 0.04f)
            {
                byte ca2 = (byte)Mathf.Clamp(60f + 185f * phiRatio, 0f, 255f);
                for (int ka = 0; ka < partA.Count; ka++)
                {
                    for (int kb = 0; kb < partB.Count; kb++)
                    {
                        if ((ka + kb) % 2 == 0 || partA.Count == 1)
                        {
                            Vector2 pa2 = _lastMipNode2d[partA[ka]], pb2 = _lastMipNode2d[partB[kb]];
                            DrawLine((int)pa2.x + 6, (int)pa2.y, (int)pb2.x - 6, (int)pb2.y, new Color32(238, 168, 132, ca2));
                        }
                    }
                }
            }
            for (int i = 0; i < 5; i++)
            {
                bool act = _dispAlpha[i] >= 0.22f;
                FillCircle((int)_lastMipNode2d[i].x, (int)_lastMipNode2d[i].y, 6, act ? new Color32(238, 206, 132, 235) : new Color32(58, 68, 84, 225));
            }

            FillRect(246, 514, 94, 92, new Color32(20, 24, 31, 215));
            DrawRectOutline(246, 514, 94, 92, new Color32(88, 102, 122, 95));
            if (_phiTrace.Count >= 2)
            {
                float maxP = 1.5f;
                for (int i = 0; i < _phiTrace.Count; i++) if (_phiTrace[i] * 1.15f > maxP) maxP = _phiTrace[i] * 1.15f;
                const int gx0 = 252, gy0 = 534, gw = 82, gh = 62;
                DrawLine(gx0, gy0 + gh, gx0 + gw, gy0 + gh, new Color32(70, 82, 98, 110));
                for (int i = 0; i < _phiTrace.Count - 1; i++)
                {
                    int x0 = Mathf.RoundToInt(gx0 + (i / (float)(_phiTrace.Count - 1)) * gw);
                    int y0 = Mathf.RoundToInt(gy0 + gh - Mathf.Clamp01(_phiTrace[i] / maxP) * gh);
                    int x1 = Mathf.RoundToInt(gx0 + ((i + 1) / (float)(_phiTrace.Count - 1)) * gw);
                    int y1 = Mathf.RoundToInt(gy0 + gh - Mathf.Clamp01(_phiTrace[i + 1] / maxP) * gh);
                    DrawLine(x0, y0, x1, y1, new Color32(242, 208, 148, 235));
                }
            }
        }

        private void DrawScoreHistoryGraph()
        {
            const int gx = 16, gy = 636, gw = PANEL_W - 32, gh = 48;
            FillRect(gx, gy, gw, gh, new Color32(18, 22, 28, 200));
            DrawRectOutline(gx, gy, gw, gh, new Color32(72, 86, 104, 80));
            int midY = gy + gh / 2;
            DrawLine(gx + 2, midY, gx + gw - 2, midY, new Color32(70, 82, 98, 90));

            int n = _scoreHistory.Count;
            if (n == 0)
                return;
            float maxAbs = 15f;
            for (int i = 0; i < n; i++) maxAbs = Mathf.Max(maxAbs, Mathf.Abs(_scoreHistory[i]) * 1.1f);
            for (int i = 0; i < n - 1; i++)
            {
                int x0 = Mathf.RoundToInt(gx + 4 + (i / (float)Mathf.Max(1, n - 1)) * (gw - 8));
                int y0 = Mathf.RoundToInt(midY - (_scoreHistory[i] / maxAbs) * (gh * 0.44f));
                int x1 = Mathf.RoundToInt(gx + 4 + ((i + 1) / (float)Mathf.Max(1, n - 1)) * (gw - 8));
                int y1 = Mathf.RoundToInt(midY - (_scoreHistory[i + 1] / maxAbs) * (gh * 0.44f));
                Color32 lc = (_scoreHistory[i + 1] >= 0f) ? new Color32(255, 230, 190, 235) : new Color32(208, 155, 175, 235);
                DrawLine(x0, y0, x1, y1, lc);
            }
        }

        // =====================================================================
        // 5. OnGUI Overlay: Right Mathematical Panel + Bottom-Left Subjective POV
        // =====================================================================
        private void EnsureStyles()
        {
            if (_stylesReady) return;
            _stylesReady = true;

            _texDarkHeader = MakeTex(new Color(0.055f, 0.067f, 0.086f, 0.96f));
            _texTabSel = MakeTex(new Color(0.16f, 0.19f, 0.25f, 0.95f));
            _texTabIdle = MakeTex(new Color(0.08f, 0.10f, 0.13f, 0.85f));
            _texDepthAccent = MakeTex(new Color(0.78f, 0.42f, 0.88f, 0.95f));
            _texPlasmaSel = MakeTex(new Color(0.28f, 0.15f, 0.36f, 0.95f));

            _stGoldHdr = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                fontStyle = FontStyle.Bold,
                richText = true
            };
            _stGoldHdr.normal.textColor = new Color(1.0f, 0.90f, 0.74f);

            _stSmallSub = new GUIStyle(GUI.skin.label)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                richText = true
            };
            _stSmallSub.normal.textColor = new Color(0.54f, 0.61f, 0.70f);

            _stModeLine = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                fontStyle = FontStyle.Bold,
                richText = true
            };
            _stModeLine.normal.textColor = new Color(0.71f, 0.88f, 0.99f);

            _stTiny = new GUIStyle(GUI.skin.label)
            {
                fontSize = 9,
                fontStyle = FontStyle.Bold,
                richText = true
            };
            _stTiny.normal.textColor = new Color(0.75f, 0.80f, 0.88f);

            _stTabSel = new GUIStyle(GUI.skin.button)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter
            };
            _stTabSel.normal.background = _texTabSel;
            _stTabSel.hover.background = _texTabSel;
            _stTabSel.normal.textColor = new Color(1.0f, 0.90f, 0.74f);
            _stTabSel.hover.textColor = Color.white;

            _stTabIdle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter
            };
            _stTabIdle.normal.background = _texTabIdle;
            _stTabIdle.hover.background = _texTabSel;
            _stTabIdle.normal.textColor = new Color(0.56f, 0.62f, 0.70f);
            _stTabIdle.hover.textColor = new Color(0.85f, 0.92f, 1.0f);

            _stPlasmaSel = new GUIStyle(GUI.skin.button)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter
            };
            _stPlasmaSel.normal.background = _texPlasmaSel;
            _stPlasmaSel.hover.background = _texPlasmaSel;
            _stPlasmaSel.normal.textColor = new Color(0.95f, 0.72f, 1.0f);
            _stPlasmaSel.hover.textColor = Color.white;

            _stPipBox = new GUIStyle(GUI.skin.box);
            _stPipBox.normal.background = _texDarkHeader;

            if (_texWmAccent == null) _texWmAccent = MakeTex(new Color(0.38f, 0.78f, 1.0f, 0.92f));
            if (_texBarBg == null) _texBarBg = MakeTex(new Color(0.11f, 0.14f, 0.19f, 0.96f));
            if (_texBarSurv == null) _texBarSurv = MakeTex(new Color(0.28f, 0.88f, 0.54f, 0.95f));
            if (_texBarVeto == null) _texBarVeto = MakeTex(new Color(0.98f, 0.34f, 0.34f, 0.95f));
            if (_texBarAdv == null) _texBarAdv = MakeTex(new Color(0.55f, 0.80f, 1.0f, 0.95f));
        }

        private static Texture2D MakeTex(Color c)
        {
            var t = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            t.SetPixel(0, 0, c);
            t.Apply();
            return t;
        }

        private void OnGUI()
        {
            if (coordinator == null || coordinator.IsTrainingSession)
                return;
            if (dashboardUi != null && dashboardUi.IsLauncherOpen)
                return;

            EnsureStyles();

            // 1. Bottom-Left Agent A Subjective 1st-Person POV + 02 DEPTH + 03 WORLD MODEL Mini-HUD
            DrawSubjectivePovWindow();
            DrawDepthCortexWindow();
            DrawWorldModelWindow();

            // 2. Right-Edge Agent A Mathematical Thought Visualizer
            if (!showRightNeuralPanel)
            {
                if (GUI.Button(new Rect(Screen.width - 118f, 12f, 108f, 24f), "◀ 思考UX (F3)", _stTabSel))
                {
                    showRightNeuralPanel = true;
                    UpdateMainCameraViewport();
                }
                return;
            }

            float x0 = Screen.width - PANEL_W;
            float sy = Screen.height / (float)CANVAS_H;
            GUI.DrawTexture(new Rect(x0, 0f, PANEL_W, Screen.height), _canvasTex, ScaleMode.StretchToFill);

            // Top Cortical Header & Clickable Region Tabs
            GUI.Label(new Rect(x0 + 16f, 10f * sy, 268f, 18f), "CORTICAL NEURAL SHEET [AGENT A]", _stGoldHdr);
            if (GUI.Button(new Rect(x0 + PANEL_W - 58f, 8f * sy, 44f, 18f), "隠す", _stTabIdle))
            {
                showRightNeuralPanel = false;
                UpdateMainCameraViewport();
                return;
            }

            float tabW = (PANEL_W - 32f - 18f) / 4f;
            for (int i = 0; i < 4; i++)
            {
                Rect tr = new Rect(x0 + 16f + i * (tabW + 6f), 30f * sy, tabW, 18f);
                if (GUI.Button(tr, _tabLabels[i], i == _activeTab ? _stTabSel : _stTabIdle))
                    _activeTab = i;
            }

            GUI.Label(new Rect(x0 + 16f, 51f * sy, PANEL_W - 28f, 16f), $"CONTROL | {_regionSubText}", _stSmallSub);
            GUI.Label(new Rect(x0 + 16f, 66f * sy, PANEL_W - 28f, 18f), _controlModeText, _stModeLine);

            // Middle GWT Card Labels
            string vetoBadge = _dispVeto ? " <color=#FDA4AF>[VETO]</color>" : "";
            GUI.Label(new Rect(x0 + 24f, 278f * sy, 165f, 16f), $"GWT IGNITION: {_dispIgnition:F2}{vetoBadge}", _stGoldHdr);
            GUI.Label(new Rect(x0 + 24f, 309f * sy, 165f, 16f), $"SELF ERR e_s: {_dispSelfErr:F2}", _stSmallSub);
            GUI.Label(new Rect(x0 + 188f, 278f * sy, 148f, 16f), "GWT ARBITRATION", _stSmallSub);
            string[] modLetters = { "P", "I", "W", "M", "S" };
            for (int i = 0; i < 5; i++)
                GUI.Label(new Rect(x0 + 193f + i * 28f, 334f * sy, 18f, 14f), modLetters[i], _stTiny);

            // Lower Info-Geometric Thought Manifold Header & Mode Tabs
            GUI.Label(new Rect(x0 + 16f, 356f * sy, PANEL_W - 32f, 18f), "INFO-GEOMETRIC THOUGHT MANIFOLD", _stGoldHdr);
            float gTabW = (PANEL_W - 32f - 8f) * 0.5f;
            for (int m = 0; m < 2; m++)
            {
                Rect mr = new Rect(x0 + 16f + m * (gTabW + 8f), 374f * sy, gTabW, 18f);
                if (GUI.Button(mr, _geoLabels[m], m == _geoMode ? _stTabSel : _stTabIdle))
                    _geoMode = m;
            }

            if (_geoMode == 0)
            {
                string gSub = $"d_FR:{_dispFrStep:F2}rad | H:{_dispEntNorm:F2} | Phi_G:{_dispPhiBits:F2}b | M{_dispBestCol}";
                GUI.Label(new Rect(x0 + 16f, 395f * sy, PANEL_W - 28f, 16f), gSub, _stSmallSub);
            }
            else
            {
                string cutStr = FormatMipCutLabel(_dispMipMask);
                string pSub = $"Phi_G=min D(p||q): {_dispPhiBits:F2}b [{cutStr}] | M{_dispFastCol}->M{_dispBestCol}";
                GUI.Label(new Rect(x0 + 16f, 395f * sy, PANEL_W - 28f, 16f), pSub, _dispPhiBits >= 0.35f ? _stGoldHdr : _stSmallSub);
                GUI.Label(new Rect(x0 + 251f, 420f * sy, 90f, 14f), "MIP PARTITION", _stTiny);
                GUI.Label(new Rect(x0 + 251f, 516f * sy, 90f, 14f), "Phi_G TRACE [b]", _stTiny);
                GUI.Label(new Rect(x0 + 44f, 566f * sy, 70f, 14f), "M_G (MIP)", _stTiny);

                // Module letters inside the 5 dynamic Oizumi MIP Partition nodes
                for (int i = 0; i < 5; i++)
                    GUI.Label(new Rect(x0 + _lastMipNode2d[i].x - 4f, (_lastMipNode2d[i].y - 6f) * sy, 14f, 14f), modLetters[i], _stTiny);

                GUI.Label(new Rect(x0 + _lastQS1CanvasPt.x + 5f, (_lastQS1CanvasPt.y - 6f) * sy, 42f, 14f), "q(S1)", _stTiny);
                if (_lastLiftPx >= 10f)
                {
                    GUI.Label(new Rect(x0 + _lastQStarCanvasPt.x - 16f, (_lastQStarCanvasPt.y - 2f) * sy, 22f, 14f), "q*", _stTiny);
                    GUI.Label(new Rect(x0 + _lastPIntCanvasPt.x + 6f, Mathf.Max(414f, _lastPIntCanvasPt.y - 8f) * sy, 52f, 14f), "p(X_t)", _stTiny);
                    float midY = 0.5f * (_lastQStarCanvasPt.y + _lastPIntCanvasPt.y) - 7f;
                    float lblX = Mathf.Clamp(_lastPIntCanvasPt.x - 62f, 18f, 182f);
                    GUI.Label(new Rect(x0 + lblX, midY * sy, 68f, 15f), $"Phi={_dispPhiBits:F2}b", _stGoldHdr);
                }
            }

            // Bottom Score History Label
            string lastSc = _scoreHistory.Count > 0 ? $" (Last: {_scoreHistory[_scoreHistory.Count - 1]:+0.0;-0.0;0.0})" : "";
            GUI.Label(new Rect(x0 + 16f, 618f * sy, PANEL_W - 32f, 16f), $"EPISODE RETURN HISTORY (RECENT 50){lastSc}", _stSmallSub);
        }

        public Rect GetSubjectivePovWindowRect()
        {
            if (!showSubjectivePovPip)
                return new Rect(12f, Screen.height - 32f, 132f, 22f);

            const float pipW = 288f;
            const float imgH = 162f;
            const float hdrH = 22f;
            float boxH = hdrH + imgH + 6f;
            return new Rect(12f, Screen.height - boxH - 10f, pipW + 6f, boxH);
        }

        public Rect GetDepthCortexWindowRect()
        {
            Rect povR = GetSubjectivePovWindowRect();
            if (!showDepthCortexPip)
                return new Rect(12f, povR.y - 26f, 132f, 22f);

            const float pipW = 288f;
            const float hdrH = 22f;
            float topGuardY = (telemetryHud != null && telemetryHud.isCollapsed) ? 44f : 256f;
            float imgH = Mathf.Clamp(povR.y - topGuardY - (hdrH + 6f), 64f, 152f);
            float boxH = hdrH + imgH + 6f;
            float py = povR.y - boxH - 6f;
            return new Rect(12f, py, pipW + 6f, boxH);
        }

        public Rect GetWorldModelWindowRect()
        {
            float xLeft = showSubjectivePovPip ? 312f : 150f;
            const float boxW = 282f;
            if (!showWorldModelPip)
                return new Rect(xLeft, Screen.height - 26f, 166f, 22f);

            const float boxH = 172f;
            return new Rect(xLeft, Mathf.Max(68f, Screen.height - boxH - 4f), boxW, boxH);
        }

        private static void EnsureSurpriseLut()
        {
            if (s_surpriseLut != null && s_surpriseLut.Length == 256)
                return;
            s_surpriseLut = new Color32[256];
            // Quiet dark slate (#0B0E14) -> deep cyan (#123B60) -> luminous cyan (#38BDF8) -> amber (#F59E0B) -> flash gold (#FEF08A)
            Color[] stops =
            {
                new Color(0.043f, 0.055f, 0.078f, 1f),
                new Color(0.071f, 0.231f, 0.376f, 1f),
                new Color(0.220f, 0.741f, 0.973f, 1f),
                new Color(0.961f, 0.620f, 0.043f, 1f),
                new Color(0.996f, 0.941f, 0.541f, 1f),
            };
            int nSeg = stops.Length - 1;
            for (int i = 0; i < 256; i++)
            {
                float t = (i / 255f) * nSeg;
                int idx = Mathf.Clamp(Mathf.FloorToInt(t), 0, nSeg - 1);
                Color c = Color.Lerp(stops[idx], stops[idx + 1], t - idx);
                s_surpriseLut[i] = new Color32(
                    (byte)Mathf.RoundToInt(c.r * 255f),
                    (byte)Mathf.RoundToInt(c.g * 255f),
                    (byte)Mathf.RoundToInt(c.b * 255f),
                    255);
            }
        }

        /// <summary>
        /// Rasterize the Brain's 16x16 Imagined `t+1` Visual Field + Pixel Prediction-Error Heatmap.
        /// Strictly ZERO work (`0.000 ms`, `WorldModelRasterCount` unchanged) when `showWorldModelPip == false`!
        /// </summary>
        public bool RefreshWorldModelNow()
        {
            if (!showWorldModelPip)
                return false;

            UbwmSnapshot wm;
            lock (_ubntLock)
            {
                wm = _latestUbwm;
            }
            if (wm == null || wm.seq == 0 || (_wmPredTex != null && wm.seq == _consumedUbwmSeq))
                return false;

            EnsurePlasmaLut();
            EnsureSurpriseLut();

            int nPix = WM_TEX_W * WM_TEX_H;
            if (_wmPredPixels == null || _wmPredPixels.Length != nPix) _wmPredPixels = new Color32[nPix];
            if (_wmErrPixels == null || _wmErrPixels.Length != nPix) _wmErrPixels = new Color32[nPix];
            if (_wmPredTex == null)
            {
                _wmPredTex = new Texture2D(WM_TEX_W, WM_TEX_H, TextureFormat.RGBA32, false)
                {
                    name = "AgentA_WorldModel_Pred_88x56",
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
            }
            if (_wmErrTex == null)
            {
                _wmErrTex = new Texture2D(WM_TEX_W, WM_TEX_H, TextureFormat.RGBA32, false)
                {
                    name = "AgentA_WorldModel_Err_88x56",
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
            }

            float[] d16 = wm.predDepth16;
            float[] s16 = wm.predSurf16;
            float[] e16 = wm.predErr16;

            for (int r = 0; r < 16; r++)
            {
                int rOff = r * 16;
                int dstRow = r * WM_TEX_W;
                for (int x = 0; x < WM_TEX_W; x++)
                {
                    float u = (x / (float)(WM_TEX_W - 1)) * 15f;
                    int c1 = Mathf.Clamp(Mathf.FloorToInt(u), 0, 14);
                    float tx = u - c1;
                    int c0 = c1 > 0 ? c1 - 1 : 0;
                    int c2 = c1 + 1;
                    int c3 = c2 < 15 ? c2 + 1 : 15;
                    _wmRowTmpD[dstRow + x] = EvalMonotoneCatmull4(d16[rOff + c0], d16[rOff + c1], d16[rOff + c2], d16[rOff + c3], tx);
                    _wmRowTmpS[dstRow + x] = Mathf.Lerp(s16[rOff + c1], s16[rOff + c2], tx);
                    _wmRowTmpE[dstRow + x] = EvalMonotoneCatmull4(e16[rOff + c0], e16[rOff + c1], e16[rOff + c2], e16[rOff + c3], tx);
                }
            }

            for (int yTex = 0; yTex < WM_TEX_H; yTex++)
            {
                float vTop = (1f - (yTex / (float)(WM_TEX_H - 1))) * 15f;
                int r1 = Mathf.Clamp(Mathf.FloorToInt(vTop), 0, 14);
                float ty = vTop - r1;
                int r0 = r1 > 0 ? r1 - 1 : 0;
                int r2 = r1 + 1;
                int r3 = r2 < 15 ? r2 + 1 : 15;

                int r0Off = r0 * WM_TEX_W, r1Off = r1 * WM_TEX_W, r2Off = r2 * WM_TEX_W, r3Off = r3 * WM_TEX_W;
                int outRow = yTex * WM_TEX_W;

                for (int x = 0; x < WM_TEX_W; x++)
                {
                    float d01 = EvalMonotoneCatmull4(_wmRowTmpD[r0Off + x], _wmRowTmpD[r1Off + x], _wmRowTmpD[r2Off + x], _wmRowTmpD[r3Off + x], ty);
                    float mappedD = Mathf.Clamp01(Mathf.Pow(Mathf.Clamp01(d01), 0.86f));
                    Color32 colP = s_plasmaLut[Mathf.Clamp(Mathf.RoundToInt(mappedD * 255f), 0, 255)];

                    // Subtle turf/salience tint so imagined friendly ink vs enemy ink is directly visible!
                    float sal = Mathf.Lerp(_wmRowTmpS[r1Off + x], _wmRowTmpS[r2Off + x], ty);
                    if (sal > 0.68f)
                    {
                        colP.r = (byte)((colP.r * 185 + 95 * 70) >> 8);
                        colP.g = (byte)((colP.g * 185 + 215 * 70) >> 8);
                        colP.b = (byte)((colP.b * 185 + 250 * 70) >> 8);
                    }
                    else if (sal > 0.15f && sal < 0.35f)
                    {
                        colP.r = (byte)((colP.r * 185 + 235 * 70) >> 8);
                        colP.g = (byte)((colP.g * 185 + 150 * 70) >> 8);
                        colP.b = (byte)((colP.b * 185 + 95 * 70) >> 8);
                    }
                    _wmPredPixels[outRow + x] = colP;

                    float err01 = EvalMonotoneCatmull4(_wmRowTmpE[r0Off + x], _wmRowTmpE[r1Off + x], _wmRowTmpE[r2Off + x], _wmRowTmpE[r3Off + x], ty);
                    _wmErrPixels[outRow + x] = s_surpriseLut[Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(err01) * 255f), 0, 255)];
                }
            }

            _wmPredTex.SetPixels32(_wmPredPixels);
            _wmPredTex.Apply(false);
            _wmErrTex.SetPixels32(_wmErrPixels);
            _wmErrTex.Apply(false);
            _consumedUbwmSeq = wm.seq;
            WorldModelRasterCount++;
            return true;
        }

        private void DrawWorldModelWindow()
        {
            Rect r = GetWorldModelWindowRect();
            if (!showWorldModelPip)
            {
                if (GUI.Button(r, "▲ 03 WORLD MODEL (F6)", _stTabSel))
                {
                    showWorldModelPip = true;
                    _nextWmRasterTime = 0f;
                }
                return;
            }

            GUI.Box(r, GUIContent.none, _stPipBox);
            if (_texWmAccent != null)
                GUI.DrawTexture(new Rect(r.x, r.y, r.width, 2f), _texWmAccent);

            UbwmSnapshot wm;
            UbntSnapshot ubnt;
            lock (_ubntLock)
            {
                wm = _latestUbwm;
                ubnt = _latestUbnt;
            }

            string hdrStats = wm != null
                ? $"<color=#9DD2FF>03 WORLD MODEL</color> <color=#94A3B8>Δd:{wm.depthPredMaeM:F2}m KL:{wm.worldKlSurprise:F2}</color>"
                : "<color=#9DD2FF>03 WORLD MODEL</color> <color=#94A3B8>WAITING...</color>";
            GUI.Label(new Rect(r.x + 8f, r.y + 3f, r.width - 52f, 18f), hdrStats, _stGoldHdr);
            if (GUI.Button(new Rect(r.x + r.width - 42f, r.y + 2f, 38f, 18f), "隠す", _stTabIdle))
            {
                showWorldModelPip = false;
                return;
            }

            float tileW = (r.width - 18f) * 0.5f;
            const float tileH = 68f;
            Rect leftTile = new Rect(r.x + 6f, r.y + 22f, tileW, tileH);
            Rect rightTile = new Rect(r.x + 12f + tileW, r.y + 22f, tileW, tileH);

            if (wm != null && _wmPredTex != null && _wmErrTex != null)
            {
                GUI.DrawTexture(leftTile, _wmPredTex, ScaleMode.StretchToFill, false);
                GUI.DrawTexture(rightTile, _wmErrTex, ScaleMode.StretchToFill, false);

                int cIdx = Mathf.Clamp(wm.chosenCand, 0, NUM_CANDS - 1);
                string verbTag = ubnt != null ? ShortVerb(ubnt.candVerbs[cIdx]) : "";
                GUI.Label(new Rect(leftTile.x, leftTile.yMax + 1f, tileW + 4f, 15f),
                    $"<color=#BAE6FD>想像視野(t+1)</color> M{cIdx} {verbTag}", _stTiny);
                GUI.Label(new Rect(rightTile.x, rightTile.yMax + 1f, tileW + 4f, 15f),
                    $"<color=#FDE68A>予測誤差|Δ|</color> MAE:{wm.visPredMae:F3}", _stTiny);

                // Bottom 3-Step Causal RSSM Rollout Bars (8 Candidates: Surv & Adv + Action-CF Spread)
                float secY = leftTile.yMax + 17f;
                GUI.Label(new Rect(r.x + 6f, secY, r.width - 12f, 14f),
                    $"3step想像分岐 M0..M7 (<color=#50E389>生存</color>/<color=#9DD2FF>前進</color>)  行動感度ΔCF:<b>{wm.actionCfSpread:F2}</b>",
                    _stTiny);

                float barTop = secY + 15f;
                const float maxBarH = 28f;
                float slotW = (r.width - 16f) / NUM_CANDS;
                for (int c = 0; c < NUM_CANDS; c++)
                {
                    float sx = r.x + 8f + c * slotW;
                    float colW = slotW - 4f;
                    GUI.DrawTexture(new Rect(sx, barTop, colW, maxBarH), _texBarBg);

                    float surv01 = Mathf.Clamp01(wm.predSurv[c]);
                    float adv01 = Mathf.Clamp01(wm.predAdv[c]);
                    bool isVetoed = ((wm.vetoMask >> c) & 1) != 0;

                    float halfCol = Mathf.Max(2f, (colW - 2f) * 0.5f);
                    float sh = Mathf.Max(1f, maxBarH * surv01);
                    float ah = Mathf.Max(1f, maxBarH * adv01);
                    GUI.DrawTexture(new Rect(sx + 1f, barTop + maxBarH - sh, halfCol, sh), isVetoed ? _texBarVeto : _texBarSurv);
                    GUI.DrawTexture(new Rect(sx + 1f + halfCol, barTop + maxBarH - ah, halfCol, ah), _texBarAdv);

                    bool isChosen = (c == wm.chosenCand);
                    GUI.Label(new Rect(sx + 1f, barTop + maxBarH - 1f, colW + 4f, 13f),
                        isChosen ? $"<color=#FDE047><b>M{c}</b></color>" : $"M{c}", _stTiny);
                }
            }
            else
            {
                GUI.Box(leftTile, GUIContent.none, _stPipBox);
                GUI.Box(rightTile, GUIContent.none, _stPipBox);
                GUI.Label(new Rect(r.x + 38f, r.y + 78f, 230f, 22f),
                    "<color=#94A3B8>世界モデル予測ストリーム接続待ち...</color>", _stSmallSub);
            }
        }

        private static void EnsurePlasmaLut()
        {
            if (s_plasmaLut != null && s_plasmaLut.Length == 256)
                return;
            s_plasmaLut = new Color32[256];
            // 6-stop perceptual Matplotlib Plasma colormap: deep indigo (#0D0887) -> violet -> magenta -> coral -> orange -> bright yellow (#F0F921)
            Color[] stops =
            {
                new Color(0.051f, 0.031f, 0.529f, 1f), // 0.00
                new Color(0.416f, 0.000f, 0.659f, 1f), // 0.20
                new Color(0.694f, 0.165f, 0.565f, 1f), // 0.40
                new Color(0.882f, 0.392f, 0.384f, 1f), // 0.60
                new Color(0.988f, 0.651f, 0.212f, 1f), // 0.80
                new Color(0.941f, 0.976f, 0.129f, 1f), // 1.00
            };
            int nSeg = stops.Length - 1;
            for (int i = 0; i < 256; i++)
            {
                float t = (i / 255f) * nSeg;
                int idx = Mathf.Clamp(Mathf.FloorToInt(t), 0, nSeg - 1);
                float frac = t - idx;
                Color c = Color.Lerp(stops[idx], stops[idx + 1], frac);
                s_plasmaLut[i] = new Color32(
                    (byte)Mathf.Clamp(Mathf.RoundToInt(c.r * 255f), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(c.g * 255f), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(c.b * 255f), 0, 255),
                    255);
            }
        }

        private static float EvalMonotoneCatmull4(float p0, float p1, float p2, float p3, float t)
        {
            float t2 = t * t;
            float t3 = t2 * t;
            float v = 0.5f * ((2f * p1) + (-p0 + p2) * t
                + (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2
                + (-p0 + 3f * p1 - 3f * p2 + p3) * t3);
            float lo = p1 < p2 ? p1 : p2;
            float hi = p1 > p2 ? p1 : p2;
            return Mathf.Clamp(v, lo, hi);
        }

        /// <summary>
        /// Rasterize Agent A's live 16x16 V1 foveal depth field into the 96x54 Plasma Texture2D.
        /// Strictly ZERO work (`0.000 ms`, `DepthRasterCount` unchanged) whenever `showDepthCortexPip == false`!
        /// </summary>
        public bool RefreshDepthCortexNow(InkBlitzPlayerAgent agA, bool forceSampleIfOffline = false)
        {
            if (!showDepthCortexPip || agA == null || agA.isRespawning || agA.brainBridge == null)
                return false;

            UniversalBrainBridge3D br = agA.brainBridge;
            if (!br.IsConnected && forceSampleIfOffline)
                br.SampleLiveFovealPatchForHud();

            ulong ubndSeqLocal = 0;
            lock (_ubntLock)
            {
                ubndSeqLocal = _ubndSeq;
            }
            bool useUbnd = ubndSeqLocal > 0 && br.IsConnected;
            ulong srcSeq = useUbnd ? (0x8000000000000000UL | ubndSeqLocal) : br.LatestFoveaSeq;
            if (srcSeq == 0)
                return false;
            if (_depthTex != null && srcSeq == _consumedDepthSeq && depthViewMode == _consumedDepthMode)
                return false;

            EnsurePlasmaLut();
            if (_depthPixels == null || _depthPixels.Length != DEPTH_TEX_W * DEPTH_TEX_H)
                _depthPixels = new Color32[DEPTH_TEX_W * DEPTH_TEX_H];
            if (_depthTex == null)
            {
                _depthTex = new Texture2D(DEPTH_TEX_W, DEPTH_TEX_H, TextureFormat.RGBA32, false)
                {
                    name = "AgentA_DepthCortex_96x54",
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
            }

            float[] srcDepth16 = br.latestFoveaDepth16;
            float[] srcSal16 = br.latestFoveaSalience16;
            if (useUbnd)
            {
                lock (_ubntLock)
                {
                    srcDepth16 = _ubndDepth16;
                }
            }

            float maxRangeM = Mathf.Max(1f, br.foveaMaxRangeM);
            float distAcc = 0f;
            for (int i = 0; i < 256; i++)
                distAcc += (1f - Mathf.Clamp01(srcDepth16[i])) * maxRangeM;
            _depthMeanMeters = distAcc / 256f;

            // Offline Blend fallback tint when POV camera is closed (keeps `_povCamera.enabled == false` for 0 GPU cost!)
            bool blendSurfaceRelief = (depthViewMode == 1) && (!showSubjectivePovPip || _povRt == null);

            // Pass 1: Horizontal 16 -> 96 monotone Catmull-Rom interpolation
            for (int r = 0; r < 16; r++)
            {
                int rOff = r * 16;
                int dstRow = r * DEPTH_TEX_W;
                for (int x = 0; x < DEPTH_TEX_W; x++)
                {
                    float u = (x / (float)(DEPTH_TEX_W - 1)) * 15f;
                    int c1 = Mathf.Clamp(Mathf.FloorToInt(u), 0, 14);
                    float tx = u - c1;
                    int c0 = c1 > 0 ? c1 - 1 : 0;
                    int c2 = c1 + 1;
                    int c3 = c2 < 15 ? c2 + 1 : 15;
                    _depthRowTmp[dstRow + x] = EvalMonotoneCatmull4(
                        srcDepth16[rOff + c0], srcDepth16[rOff + c1], srcDepth16[rOff + c2], srcDepth16[rOff + c3], tx);
                    if (blendSurfaceRelief)
                    {
                        _salRowTmp[dstRow + x] = Mathf.Lerp(srcSal16[rOff + c1], srcSal16[rOff + c2], tx);
                    }
                }
            }

            // Pass 2: Vertical 16 -> 54 interpolation + Plasma LUT mapping
            // Note: Texture2D y=0 is BOTTOM (`r = 15`), y=DEPTH_TEX_H-1 is TOP (`r = 0`)!
            for (int yTex = 0; yTex < DEPTH_TEX_H; yTex++)
            {
                float vTop = (1f - (yTex / (float)(DEPTH_TEX_H - 1))) * 15f;
                int r1 = Mathf.Clamp(Mathf.FloorToInt(vTop), 0, 14);
                float ty = vTop - r1;
                int r0 = r1 > 0 ? r1 - 1 : 0;
                int r2 = r1 + 1;
                int r3 = r2 < 15 ? r2 + 1 : 15;

                int r0Off = r0 * DEPTH_TEX_W;
                int r1Off = r1 * DEPTH_TEX_W;
                int r2Off = r2 * DEPTH_TEX_W;
                int r3Off = r3 * DEPTH_TEX_W;
                int outRow = yTex * DEPTH_TEX_W;

                for (int x = 0; x < DEPTH_TEX_W; x++)
                {
                    float d01 = EvalMonotoneCatmull4(
                        _depthRowTmp[r0Off + x], _depthRowTmp[r1Off + x], _depthRowTmp[r2Off + x], _depthRowTmp[r3Off + x], ty);
                    // Gentle perceptual contrast curve matching reference Plasma depth cortex aesthetic
                    float mapped = Mathf.Clamp01(Mathf.Pow(Mathf.Clamp01(d01), 0.86f));
                    int lutIdx = Mathf.Clamp(Mathf.RoundToInt(mapped * 255f), 0, 255);
                    Color32 col = s_plasmaLut[lutIdx];

                    if (blendSurfaceRelief)
                    {
                        float sal = Mathf.Lerp(_salRowTmp[r1Off + x], _salRowTmp[r2Off + x], ty);
                        if (sal > 0.68f) // Friendly turf / hostile highlight
                        {
                            col.r = (byte)((col.r * 185 + 95 * 70) >> 8);
                            col.g = (byte)((col.g * 185 + 215 * 70) >> 8);
                            col.b = (byte)((col.b * 185 + 250 * 70) >> 8);
                        }
                        else if (sal > 0.15f && sal < 0.35f) // Enemy ink tint
                        {
                            col.r = (byte)((col.r * 185 + 235 * 70) >> 8);
                            col.g = (byte)((col.g * 185 + 150 * 70) >> 8);
                            col.b = (byte)((col.b * 185 + 95 * 70) >> 8);
                        }
                    }

                    _depthPixels[outRow + x] = col;
                }
            }

            _depthTex.SetPixels32(_depthPixels);
            _depthTex.Apply(false);
            _consumedDepthSeq = srcSeq;
            _consumedDepthMode = depthViewMode;
            DepthRasterCount++;
            return true;
        }

        private void DrawDepthCortexWindow()
        {
            Rect boxRect = GetDepthCortexWindowRect();
            if (!showDepthCortexPip)
            {
                if (GUI.Button(boxRect, "▲ 02 DEPTH (F5)", _stTabSel))
                {
                    showDepthCortexPip = true;
                    _nextDepthRasterTime = 0f;
                }
                return;
            }

            GUI.Box(boxRect, GUIContent.none, _stPipBox);
            if (_texDepthAccent != null)
                GUI.DrawTexture(new Rect(boxRect.x, boxRect.y, boxRect.width, 2f), _texDepthAccent);

            InkBlitzArenaInstance arena = (multiManager != null && multiManager.primaryArena != null)
                ? multiManager.primaryArena
                : null;
            InkBlitzPlayerAgent agA = (arena != null && arena.agents.Count > 0) ? arena.agents[0] : null;

            const float hdrH = 22f;
            float px = boxRect.x;
            float py = boxRect.y;
            float pipW = boxRect.width - 6f;

            GUI.Label(
                new Rect(px + 8f, py + 3f, 142f, 18f),
                $"<color=#5CE6A0>02 DEPTH</color> <color=#8EA3B8>{_depthMeanMeters:F1}m</color>",
                _stGoldHdr);

            if (GUI.Button(new Rect(px + pipW - 144f, py + 2f, 54f, 18f), "Plasma", depthViewMode == 0 ? _stPlasmaSel : _stTabIdle))
            {
                depthViewMode = 0;
                _nextDepthRasterTime = 0f;
            }
            if (GUI.Button(new Rect(px + pipW - 87f, py + 2f, 48f, 18f), "Blend", depthViewMode == 1 ? _stPlasmaSel : _stTabIdle))
            {
                depthViewMode = 1;
                _nextDepthRasterTime = 0f;
            }
            if (GUI.Button(new Rect(px + pipW - 36f, py + 2f, 38f, 18f), "隠す", _stTabIdle))
            {
                showDepthCortexPip = false;
                return;
            }

            Rect imgRect = new Rect(px + 3f, py + hdrH, pipW, boxRect.height - hdrH - 6f);
            if (agA != null && agA.isRespawning)
            {
                GUI.Box(imgRect, GUIContent.none, _stPipBox);
                GUI.Label(new Rect(imgRect.x + 56f, imgRect.y + imgRect.height * 0.5f - 12f, 200f, 24f),
                    $"<color=#C66CE0>DEPTH STANDBY ({agA.RespawnTimeRemaining:F1}s)</color>", _stModeLine);
            }
            else if (_depthTex != null)
            {
                if (depthViewMode == 1 && showSubjectivePovPip && _povRt != null)
                {
                    GUI.DrawTexture(imgRect, _povRt, ScaleMode.StretchToFill, false);
                    Color prevC = GUI.color;
                    GUI.color = new Color(1f, 1f, 1f, 0.62f);
                    GUI.DrawTexture(imgRect, _depthTex, ScaleMode.StretchToFill, true);
                    GUI.color = prevC;
                }
                else
                {
                    GUI.DrawTexture(imgRect, _depthTex, ScaleMode.StretchToFill, false);
                }
            }
            else
            {
                GUI.Box(imgRect, GUIContent.none, _stPipBox);
                GUI.Label(new Rect(imgRect.x + 64f, imgRect.y + imgRect.height * 0.5f - 12f, 190f, 24f),
                    "<color=#94A3B8>WAITING V1 RAYCAST...</color>", _stSmallSub);
            }
        }

        private void DrawSubjectivePovWindow()
        {
            Rect boxRect = GetSubjectivePovWindowRect();
            if (!showSubjectivePovPip)
            {
                if (GUI.Button(boxRect, "▲ A主観視点 (F4)", _stTabSel))
                {
                    showSubjectivePovPip = true;
                    UpdatePovCameraState(true);
                }
                return;
            }

            const float hdrH = 22f;
            float px = boxRect.x;
            float py = boxRect.y;
            float pipW = boxRect.width - 6f;
            float imgH = boxRect.height - hdrH - 6f;

            GUI.Box(boxRect, GUIContent.none, _stPipBox);

            InkBlitzArenaInstance arena = (multiManager != null && multiManager.primaryArena != null)
                ? multiManager.primaryArena
                : null;
            InkBlitzPlayerAgent agA = (arena != null && arena.agents.Count > 0) ? arena.agents[0] : null;
            string formBadge = (agA != null && agA.isSubmergedInFriendlyInk) ? "<color=#4DFFAA>[DIVE]</color>" : "<color=#66CCFF>[EYE 76°]</color>";

            GUI.Label(new Rect(px + 8f, py + 3f, pipW - 44f, 18f), $"AGENT A 主観視点 {formBadge}", _stGoldHdr);
            if (GUI.Button(new Rect(px + pipW - 34f, py + 2f, 36f, 18f), "隠す", _stTabIdle))
            {
                showSubjectivePovPip = false;
                UpdatePovCameraState(false);
                return;
            }

            Rect imgRect = new Rect(px + 3f, py + hdrH, pipW, imgH);
            if (agA != null && agA.isRespawning)
            {
                GUI.Box(imgRect, GUIContent.none, _stPipBox);
                GUI.Label(new Rect(imgRect.x + 56f, imgRect.y + 70f, 200f, 24f),
                    $"<color=#FF8877>RESPAWNING... ({agA.RespawnTimeRemaining:F1}s)</color>", _stModeLine);
            }
            else if (_povRt != null)
            {
                GUI.DrawTexture(imgRect, _povRt, ScaleMode.StretchToFill, false);
                // Crisp minimal foveal center crosshair
                float cx = imgRect.x + imgRect.width * 0.5f;
                float cy = imgRect.y + imgRect.height * 0.5f;
                GUI.Label(new Rect(cx - 5f, cy - 9f, 16f, 18f), "<color=#AAFFEE>+</color>", _stModeLine);
            }
        }

        // =====================================================================
        // 6. Lightweight Software Drawing & Math Primitives (Zero External Deps)
        // =====================================================================
        private Vector2 ProjectOrthogonal2D(int tabIdx, float[] vec, out float[] unitVec)
        {
            int dim = Mathf.Max(2, vec.Length);
            unitVec = new float[dim];
            float norm = 1e-6f;
            for (int i = 0; i < vec.Length; i++) norm += vec[i] * vec[i];
            norm = Mathf.Sqrt(norm);
            for (int i = 0; i < vec.Length; i++) unitVec[i] = vec[i] / norm;

            int key = tabIdx * 1009 + dim;
            if (!_projAxes.TryGetValue(key, out float[] axes))
            {
                var r = new System.Random(1000 + tabIdx * 97 + dim);
                axes = new float[dim * 2];
                float n0 = 1e-8f;
                for (int i = 0; i < dim; i++) { float g = NextGaussian(r); axes[i] = g; n0 += g * g; }
                n0 = Mathf.Sqrt(n0);
                for (int i = 0; i < dim; i++) axes[i] /= n0;

                float dot = 0f;
                for (int i = 0; i < dim; i++) { float g = NextGaussian(r); axes[dim + i] = g; dot += g * axes[i]; }
                float n1 = 1e-8f;
                for (int i = 0; i < dim; i++) { axes[dim + i] -= dot * axes[i]; n1 += axes[dim + i] * axes[dim + i]; }
                n1 = Mathf.Sqrt(n1);
                for (int i = 0; i < dim; i++) axes[dim + i] /= n1;
                _projAxes[key] = axes;
            }

            float px = 0f, py = 0f;
            for (int i = 0; i < dim; i++)
            {
                px += axes[i] * unitVec[i];
                py += axes[dim + i] * unitVec[i];
            }
            return new Vector2(px, py);
        }

        private static List<Vector2> CatmullRomChain(List<Vector2> wps, int subdiv)
        {
            var res = new List<Vector2>(wps.Count * subdiv + 2);
            if (wps.Count <= 1) return new List<Vector2>(wps);
            if (wps.Count == 2)
            {
                for (int k = 0; k <= subdiv; k++)
                    res.Add(Vector2.Lerp(wps[0], wps[1], k / (float)subdiv));
                return res;
            }
            for (int i = 0; i < wps.Count - 1; i++)
            {
                Vector2 p0 = wps[Mathf.Max(0, i - 1)];
                Vector2 p1 = wps[i];
                Vector2 p2 = wps[i + 1];
                Vector2 p3 = wps[Mathf.Min(wps.Count - 1, i + 2)];
                int steps = (i < wps.Count - 2) ? subdiv : (subdiv + 1);
                for (int k = 0; k < steps; k++)
                {
                    float t = k / (float)subdiv;
                    float t2 = t * t, t3 = t2 * t;
                    Vector2 q = 0.5f * ((2f * p1) + (-p0 + p2) * t
                        + (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2
                        + (-p0 + 3f * p1 - 3f * p2 + p3) * t3);
                    res.Add(q);
                }
            }
            return res;
        }

        private static float[] UpsampleCatmull1D(float[] vals, float[] xNew)
        {
            int m = vals.Length;
            float[] raw = new float[xNew.Length];
            for (int i = 0; i < xNew.Length; i++)
            {
                float pos = Mathf.Clamp01(xNew[i]) * (m - 1);
                int idx = Mathf.Clamp(Mathf.FloorToInt(pos), 0, m - 2);
                float t = pos - idx;
                float t2 = t * t, t3 = t2 * t;
                float p0 = vals[Mathf.Max(0, idx - 1)];
                float p1 = vals[idx];
                float p2 = vals[idx + 1];
                float p3 = vals[Mathf.Min(m - 1, idx + 2)];
                float v = 0.5f * ((2f * p1) + (-p0 + p2) * t
                    + (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2
                    + (-p0 + 3f * p1 - 3f * p2 + p3) * t3);
                raw[i] = Mathf.Clamp01(v);
            }
            // 5-tap spatial smoothing kernel `[0.10, 0.24, 0.32, 0.24, 0.10]`
            float[] sm = new float[raw.Length];
            float[] kw = { 0.10f, 0.24f, 0.32f, 0.24f, 0.10f };
            for (int i = 0; i < raw.Length; i++)
            {
                if (i == 0 || i == raw.Length - 1) { sm[i] = raw[i]; continue; }
                float acc = 0f;
                for (int k = -2; k <= 2; k++)
                    acc += raw[Mathf.Clamp(i + k, 0, raw.Length - 1)] * kw[k + 2];
                sm[i] = acc;
            }
            return sm;
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

        private static int ActivePercentAboveStd(float[] v, float mult)
        {
            float mean = 0f;
            for (int i = 0; i < v.Length; i++) mean += v[i];
            mean /= v.Length;
            float vr = 0f;
            for (int i = 0; i < v.Length; i++) { float d = v[i] - mean; vr += d * d; }
            float thr = Mathf.Sqrt(vr / v.Length) * mult + 1e-4f;
            return ActivePercentAboveAbs(v, thr);
        }

        private static int ActivePercentAboveAbs(float[] v, float thr)
        {
            int c = 0;
            for (int i = 0; i < v.Length; i++) if (Mathf.Abs(v[i]) > thr) c++;
            return Mathf.RoundToInt(100f * c / Mathf.Max(1, v.Length));
        }

        private static float PercentileAbs(float[] v, float q01)
        {
            float[] tmp = new float[v.Length];
            for (int i = 0; i < v.Length; i++) tmp[i] = Mathf.Abs(v[i]);
            Array.Sort(tmp);
            int idx = Mathf.Clamp(Mathf.RoundToInt(q01 * (tmp.Length - 1)), 0, tmp.Length - 1);
            return tmp[idx];
        }

        private static string FormatMipCutLabel(byte mask)
        {
            string[] names = { "P", "I", "W", "M", "S" };
            byte eff = (mask >= 1 && mask <= 30) ? mask : (byte)0x03;
            var a = new List<string>(3);
            var b = new List<string>(4);
            for (int i = 0; i < 5; i++)
            {
                if (((eff >> i) & 1) != 0) a.Add(names[i]);
                else b.Add(names[i]);
            }
            if (a.Count > b.Count)
                return string.Join("", b) + "|" + string.Join("", a);
            return string.Join("", a) + "|" + string.Join("", b);
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

        // Note: Unity Texture2D row 0 is BOTTOM, so screen `y` (`0` = top) maps to `(CANVAS_H - 1 - y)`!
        private void ClearCanvas(Color32 c)
        {
            for (int i = 0; i < _pixels.Length; i++)
                _pixels[i] = c;
        }

        private void BlendPixel(int x, int yTop, Color32 src)
        {
            if ((uint)x >= PANEL_W || (uint)yTop >= CANVAS_H || src.a == 0) return;
            int idx = (CANVAS_H - 1 - yTop) * PANEL_W + x;
            if (src.a == 255)
            {
                _pixels[idx] = src;
                return;
            }
            Color32 dst = _pixels[idx];
            int a = src.a;
            int inv = 255 - a;
            _pixels[idx] = new Color32(
                (byte)((src.r * a + dst.r * inv) >> 8),
                (byte)((src.g * a + dst.g * inv) >> 8),
                (byte)((src.b * a + dst.b * inv) >> 8),
                255);
        }

        private void FillRect(int x, int yTop, int w, int h, Color32 c)
        {
            int x1 = Mathf.Min(PANEL_W, x + w);
            int y1 = Mathf.Min(CANVAS_H, yTop + h);
            int x0 = Mathf.Max(0, x);
            int y0 = Mathf.Max(0, yTop);
            for (int y = y0; y < y1; y++)
                for (int px = x0; px < x1; px++)
                    BlendPixel(px, y, c);
        }

        private void DrawRectOutline(int x, int yTop, int w, int h, Color32 c)
        {
            DrawLine(x, yTop, x + w - 1, yTop, c);
            DrawLine(x + w - 1, yTop, x + w - 1, yTop + h - 1, c);
            DrawLine(x + w - 1, yTop + h - 1, x, yTop + h - 1, c);
            DrawLine(x, yTop + h - 1, x, yTop, c);
        }

        private void DrawQuadOutline(Vector2 a, Vector2 b, Vector2 c, Vector2 d, Color32 col)
        {
            DrawLine(Mathf.RoundToInt(a.x), Mathf.RoundToInt(a.y), Mathf.RoundToInt(b.x), Mathf.RoundToInt(b.y), col);
            DrawLine(Mathf.RoundToInt(b.x), Mathf.RoundToInt(b.y), Mathf.RoundToInt(c.x), Mathf.RoundToInt(c.y), col);
            DrawLine(Mathf.RoundToInt(c.x), Mathf.RoundToInt(c.y), Mathf.RoundToInt(d.x), Mathf.RoundToInt(d.y), col);
            DrawLine(Mathf.RoundToInt(d.x), Mathf.RoundToInt(d.y), Mathf.RoundToInt(a.x), Mathf.RoundToInt(a.y), col);
        }

        private void DrawLine(int x0, int y0, int x1, int y1, Color32 c)
        {
            int dx = Mathf.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
            int dy = -Mathf.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
            int err = dx + dy;
            for (int guard = 0; guard < 1024; guard++)
            {
                BlendPixel(x0, y0, c);
                if (x0 == x1 && y0 == y1) break;
                int e2 = 2 * err;
                if (e2 >= dy) { err += dy; x0 += sx; }
                if (e2 <= dx) { err += dx; y0 += sy; }
            }
        }

        private void FillCircle(int cx, int cy, int r, Color32 c)
        {
            int r2 = r * r;
            for (int dy = -r; dy <= r; dy++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    if (dx * dx + dy * dy <= r2)
                        BlendPixel(cx + dx, cy + dy, c);
                }
            }
        }

        private void FillEllipse(int cx, int cy, int rx, int ry, Color32 c)
        {
            if (rx <= 0 || ry <= 0) return;
            float invRx2 = 1f / (rx * rx);
            float invRy2 = 1f / (ry * ry);
            for (int dy = -ry; dy <= ry; dy++)
            {
                for (int dx = -rx; dx <= rx; dx++)
                {
                    if (dx * dx * invRx2 + dy * dy * invRy2 <= 1f)
                        BlendPixel(cx + dx, cy + dy, c);
                }
            }
        }

        private void FillQuad(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, Color32 c)
        {
            FillTriangle(p0, p1, p2, c);
            FillTriangle(p0, p2, p3, c);
        }

        private void FillTriangle(Vector2 v0, Vector2 v1, Vector2 v2, Color32 c)
        {
            int minX = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(v0.x, Mathf.Min(v1.x, v2.x))));
            int maxX = Mathf.Min(PANEL_W - 1, Mathf.CeilToInt(Mathf.Max(v0.x, Mathf.Max(v1.x, v2.x))));
            int minY = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(v0.y, Mathf.Min(v1.y, v2.y))));
            int maxY = Mathf.Min(CANVAS_H - 1, Mathf.CeilToInt(Mathf.Max(v0.y, Mathf.Max(v1.y, v2.y))));
            if (minX > maxX || minY > maxY) return;

            float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
            if (Mathf.Abs(area) < 0.01f) return;
            bool ccw = area > 0f;

            for (int y = minY; y <= maxY; y++)
            {
                float py = y + 0.5f;
                for (int x = minX; x <= maxX; x++)
                {
                    float px = x + 0.5f;
                    float e0 = (v1.x - v0.x) * (py - v0.y) - (v1.y - v0.y) * (px - v0.x);
                    float e1 = (v2.x - v1.x) * (py - v1.y) - (v2.y - v1.y) * (px - v1.x);
                    float e2 = (v0.x - v2.x) * (py - v2.y) - (v0.y - v2.y) * (px - v2.x);
                    if (ccw ? (e0 >= 0f && e1 >= 0f && e2 >= 0f) : (e0 <= 0f && e1 <= 0f && e2 <= 0f))
                        BlendPixel(x, y, c);
                }
            }
        }
    }
}
