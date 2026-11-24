// InkBlitzCompanionVoiceCoach.cs — AivisSpeech/VOICEVOX Auto-Narrator + USB Mic faster-whisper STT + Human->Gemma Coach
// Completely modular companion component:
//   1. Speaks Gemma's gentle tactical navigator commentary via local AivisSpeech (`127.0.0.1:10101`) or VOICEVOX (`50021`)
//      on a dedicated 2D AudioSource (100% isolated from AI hearing `PROPRIO[10..12]`), restricted to key match milestones
//      & post-match debriefs with anti-backlog skipping so 60Hz gameplay never lags.
//   2. Optional Push-to-Talk USB Microphone speech recognition (`companion_voice_io.py --mode stt_sidecar --device cpu`)
//      that auto-fills the editable Coaching Input Box (`coachInputText`) and sends natural Japanese strategy proposals
//      through Ollama Gemma 4 12B (`<strategy_db>.coach_req.json`) for live mid-match strategic injection!
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UniversalBrain3D;

namespace InkBlitz3D
{
    public class InkBlitzCompanionVoiceCoach : MonoBehaviour
    {
        [Serializable]
        public class SpeakerStyleOption
        {
            public int styleId;
            public string displayLabel;
        }

        [Serializable]
        private class TtsStyleDto
        {
            public string name;
            public int id;
        }

        [Serializable]
        private class TtsSpeakerDto
        {
            public string name;
            public List<TtsStyleDto> styles;
        }

        [Serializable]
        private class TtsSpeakerListWrapper
        {
            public List<TtsSpeakerDto> items;
        }

        [Serializable]
        private class SttSidecarEventDto
        {
            public string @event = "";
            public string text = "";
            public string detail = "";
            public string mic = "";
            public string device = "";
            public float duration_s = 0f;
            public float infer_ms = 0f;
        }

        [Serializable]
        private class CoachReqPayloadDto
        {
            public double ts;
            public int stage_id;
            public string user_text;
        }

        [Header("Scene References")]
        public UniversalArenaCoordinator3D coordinator;

        [Header("AivisSpeech / VOICEVOX Narrator Settings")]
        public bool enableVoiceNarration = true;
        [Range(0.1f, 1.2f)] public float narrationVolume = 0.90f;
        [Range(0.85f, 1.30f)] public float speechSpeedScale = 1.04f;

        [Header("Optional Human Coaching Settings")]
        public bool autoSendAfterWhisper = false;
        public string whisperModelSize = "small";

        // Runtime TTS State
        public bool IsTtsServerOnline { get; private set; } = false;
        public string ActiveTtsEngineName { get; private set; } = "未検出";
        public string ActiveTtsBaseUrl { get; private set; } = "";
        public readonly List<SpeakerStyleOption> AvailableVoices = new List<SpeakerStyleOption>();
        public int SelectedVoiceIndex { get; private set; } = 0;
        public bool IsSpeakingNow => (_narratorSource != null && _narratorSource.isPlaying) || _ttsBusy;

        // Runtime Whisper STT & Coaching Form State
        public string coachInputText = "";
        public string coachStatusMessage = "";
        public bool IsSttReady { get; private set; } = false;
        public bool IsSttLoading { get; private set; } = false;
        public bool IsRecordingMic { get; private set; } = false;
        public bool IsTranscribingMic { get; private set; } = false;
        public string ActiveMicName { get; private set; } = "";

        private AudioSource _narratorSource;
        private AudioClip _currentDynamicClip;
        private bool _ttsBusy = false;
        private bool _probeInFlight = false;
        private string _pendingHighPriorityTts = "";
        private float _lastSpokenMilestoneTime = -99f;
        private string _lastSpokenText = "";

        // Lazy faster-whisper CPU-int8 sidecar process
        private Process _sttProc;
        private readonly ConcurrentQueue<string> _sttEventQueue = new ConcurrentQueue<string>();
        private bool _autoStartRecordWhenReady = false;

        private static readonly (string Name, string Url)[] TtsEndpoints =
        {
            ("AivisSpeech", "http://127.0.0.1:10101"),
            ("VOICEVOX", "http://127.0.0.1:50021"),
        };

        private void Awake()
        {
            if (coordinator == null)
                coordinator = FindAnyObjectByType<UniversalArenaCoordinator3D>();

            LoadVoicePrefs();
            EnsureNarratorAudioSource();
        }

        private void Start()
        {
            ProbeTtsServerAsync();
        }

        private void OnDestroy()
        {
            StopSttSidecar();
            if (_currentDynamicClip != null)
            {
                Destroy(_currentDynamicClip);
                _currentDynamicClip = null;
            }
        }

        private void OnApplicationQuit()
        {
            StopSttSidecar();
        }

        private void LoadVoicePrefs()
        {
            if (PlayerPrefs.HasKey("IB3D_VoiceNarration"))
                enableVoiceNarration = PlayerPrefs.GetInt("IB3D_VoiceNarration", 1) != 0;
            if (PlayerPrefs.HasKey("IB3D_AutoSendWhisper"))
                autoSendAfterWhisper = PlayerPrefs.GetInt("IB3D_AutoSendWhisper", 0) != 0;
        }

        public void SaveVoicePrefs()
        {
            PlayerPrefs.SetInt("IB3D_VoiceNarration", enableVoiceNarration ? 1 : 0);
            PlayerPrefs.SetInt("IB3D_AutoSendWhisper", autoSendAfterWhisper ? 1 : 0);
            if (AvailableVoices.Count > 0 && SelectedVoiceIndex >= 0 && SelectedVoiceIndex < AvailableVoices.Count)
                PlayerPrefs.SetInt("IB3D_TtsSpeakerId", AvailableVoices[SelectedVoiceIndex].styleId);
            PlayerPrefs.Save();
        }

        private void EnsureNarratorAudioSource()
        {
            if (_narratorSource != null) return;
            var go = new GameObject("Narrator2D_AivisSpeech");
            go.transform.SetParent(transform, false);
            _narratorSource = go.AddComponent<AudioSource>();
            _narratorSource.playOnAwake = false;
            _narratorSource.loop = false;
            _narratorSource.spatialBlend = 0f; // Strictly 2D UI voice (zero leakage into 3D AI hearing!)
            _narratorSource.priority = 32;
            _narratorSource.volume = narrationVolume;
        }

        private void Update()
        {
            if (_narratorSource != null)
                _narratorSource.volume = ArenaAudioSystem.IsHumanSfxMuted ? 0f : narrationVolume;

            DrainSttSidecarEvents();

            // Dispatch queued high-priority post-match / coaching reply once previous clip finishes
            if (!_ttsBusy && (_narratorSource == null || !_narratorSource.isPlaying) && !string.IsNullOrEmpty(_pendingHighPriorityTts))
            {
                string nextTxt = _pendingHighPriorityTts;
                _pendingHighPriorityTts = "";
                StartCoroutine(SynthesizeAndPlayCoroutine(nextTxt));
            }
        }

        public string GetSelectedVoiceShortLabel()
        {
            if (!IsTtsServerOnline || AvailableVoices.Count == 0)
                return "Aivis未検出";
            int idx = Mathf.Clamp(SelectedVoiceIndex, 0, AvailableVoices.Count - 1);
            return AvailableVoices[idx].displayLabel;
        }

        public void CycleVoiceStyle(int delta)
        {
            if (AvailableVoices.Count == 0)
            {
                ProbeTtsServerAsync();
                return;
            }
            int n = AvailableVoices.Count;
            SelectedVoiceIndex = ((SelectedVoiceIndex + delta) % n + n) % n;
            SaveVoicePrefs();
        }

        public void SetVoiceNarrationEnabled(bool enabled)
        {
            enableVoiceNarration = enabled;
            if (!enabled)
            {
                _pendingHighPriorityTts = "";
                if (_narratorSource != null && _narratorSource.isPlaying)
                    _narratorSource.Stop();
            }
            else if (!IsTtsServerOnline)
            {
                ProbeTtsServerAsync();
            }
            SaveVoicePrefs();
        }

        public void StopActiveSpeechImmediate()
        {
            _pendingHighPriorityTts = "";
            if (_narratorSource != null && _narratorSource.isPlaying)
                _narratorSource.Stop();
        }

        public void ProbeTtsServerAsync()
        {
            if (!_probeInFlight && gameObject.activeInHierarchy)
                StartCoroutine(ProbeTtsEndpointsCoroutine());
        }

        private IEnumerator ProbeTtsEndpointsCoroutine()
        {
            _probeInFlight = true;
            int savedStyleId = PlayerPrefs.GetInt("IB3D_TtsSpeakerId", -1);

            for (int e = 0; e < TtsEndpoints.Length; e++)
            {
                string engName = TtsEndpoints[e].Name;
                string baseUrl = TtsEndpoints[e].Url;
                using (UnityWebRequest req = UnityWebRequest.Get(baseUrl + "/speakers"))
                {
                    req.timeout = 2;
                    yield return req.SendWebRequest();
                    if (req.result != UnityWebRequest.Result.Success)
                        continue;

                    string rawJson = req.downloadHandler.text;
                    if (string.IsNullOrWhiteSpace(rawJson))
                        continue;

                    TtsSpeakerListWrapper wrapped = null;
                    try
                    {
                        wrapped = JsonUtility.FromJson<TtsSpeakerListWrapper>("{\"items\":" + rawJson + "}");
                    }
                    catch { }

                    if (wrapped?.items == null || wrapped.items.Count == 0)
                        continue;

                    AvailableVoices.Clear();
                    int preferredIdx = -1;
                    int savedMatchIdx = -1;

                    for (int i = 0; i < wrapped.items.Count; i++)
                    {
                        var spk = wrapped.items[i];
                        if (spk?.styles == null) continue;
                        for (int s = 0; s < spk.styles.Count; s++)
                        {
                            var st = spk.styles[s];
                            int idxNow = AvailableVoices.Count;
                            string label = $"{spk.name}({st.name})";
                            AvailableVoices.Add(new SpeakerStyleOption
                            {
                                styleId = st.id,
                                displayLabel = label
                            });
                            if (st.id == savedStyleId)
                                savedMatchIdx = idxNow;
                            // Prefer calm/gentle style ("おちつき" or "ふつー" or "ノーマル") by default
                            if (preferredIdx < 0 && (st.name.Contains("おちつき") || st.name.Contains("ふつー")))
                                preferredIdx = idxNow;
                        }
                    }

                    if (AvailableVoices.Count > 0)
                    {
                        IsTtsServerOnline = true;
                        ActiveTtsEngineName = engName;
                        ActiveTtsBaseUrl = baseUrl;
                        SelectedVoiceIndex = savedMatchIdx >= 0 ? savedMatchIdx : (preferredIdx >= 0 ? preferredIdx : 0);
                        _probeInFlight = false;
                        yield break;
                    }
                }
            }

            IsTtsServerOnline = false;
            ActiveTtsEngineName = "未接続";
            _probeInFlight = false;
        }

        /// <summary>
        /// Called when a `[PBT_STAT] {"evt":"gemma_broadcast", ...}` packet arrives from Python.
        /// Speaks ONLY on meaningful events (`opening`, `milestone_shift`, `post_match`, `coach_reply`)
        /// and never queues stale mid-match callouts!
        /// </summary>
        public void NotifyBroadcastEvent(string evtKind, string ttsText)
        {
            if (!enableVoiceNarration || ArenaAudioSystem.IsHumanSfxMuted)
                return;
            string cleaned = (ttsText ?? "").Trim();
            if (string.IsNullOrEmpty(cleaned) || evtKind == "silent_status")
                return;
            if (cleaned == _lastSpokenText && evtKind != "coach_reply")
                return;

            if (!IsTtsServerOnline)
            {
                ProbeTtsServerAsync();
                return;
            }

            bool isHighPriority = (evtKind == "post_match" || evtKind == "coach_reply");
            if (isHighPriority)
            {
                _lastSpokenText = cleaned;
                if (_ttsBusy || (_narratorSource != null && _narratorSource.isPlaying))
                {
                    // Queue latest post-match / coaching reply (overwriting any older pending line)
                    _pendingHighPriorityTts = cleaned;
                }
                else
                {
                    StartCoroutine(SynthesizeAndPlayCoroutine(cleaned));
                }
            }
            else
            {
                // Mid-match milestone (`opening` / `milestone_shift`): skip in high-speed >2.2x training, if already speaking, or within 11s cooldown
                if (Time.timeScale > 2.2f)
                    return;
                if (_ttsBusy || (_narratorSource != null && _narratorSource.isPlaying))
                    return;
                if (evtKind != "opening" && (Time.unscaledTime - _lastSpokenMilestoneTime) < 11.0f)
                    return;
                _lastSpokenMilestoneTime = Time.unscaledTime;
                _lastSpokenText = cleaned;
                StartCoroutine(SynthesizeAndPlayCoroutine(cleaned));
            }
        }

        private IEnumerator SynthesizeAndPlayCoroutine(string speakText)
        {
            if (AvailableVoices.Count == 0 || string.IsNullOrEmpty(ActiveTtsBaseUrl))
                yield break;

            _ttsBusy = true;
            int speakerId = AvailableVoices[Mathf.Clamp(SelectedVoiceIndex, 0, AvailableVoices.Count - 1)].styleId;
            string qUrl = $"{ActiveTtsBaseUrl}/audio_query?text={UnityWebRequest.EscapeURL(speakText)}&speaker={speakerId}";

            string queryJson = null;
            using (UnityWebRequest qReq = new UnityWebRequest(qUrl, "POST"))
            {
                qReq.uploadHandler = new UploadHandlerRaw(Array.Empty<byte>());
                qReq.downloadHandler = new DownloadHandlerBuffer();
                qReq.timeout = 6;
                yield return qReq.SendWebRequest();
                if (qReq.result == UnityWebRequest.Result.Success)
                    queryJson = qReq.downloadHandler.text;
                else
                    IsTtsServerOnline = false;
            }

            if (string.IsNullOrEmpty(queryJson))
            {
                _ttsBusy = false;
                yield break;
            }

            // Gently apply natural speech speedScale
            string speedStr = speechSpeedScale.ToString("F2", CultureInfo.InvariantCulture);
            int spIdx = queryJson.IndexOf("\"speedScale\":", StringComparison.Ordinal);
            if (spIdx >= 0)
            {
                int colon = spIdx + "\"speedScale\":".Length;
                int comma = queryJson.IndexOfAny(new[] { ',', '}' }, colon);
                if (comma > colon)
                    queryJson = queryJson.Substring(0, colon) + speedStr + queryJson.Substring(comma);
            }

            byte[] wavBytes = null;
            string sUrl = $"{ActiveTtsBaseUrl}/synthesis?speaker={speakerId}";
            using (UnityWebRequest sReq = new UnityWebRequest(sUrl, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(queryJson);
                sReq.uploadHandler = new UploadHandlerRaw(bodyRaw);
                sReq.downloadHandler = new DownloadHandlerBuffer();
                sReq.SetRequestHeader("Content-Type", "application/json");
                sReq.timeout = 10;
                yield return sReq.SendWebRequest();
                if (sReq.result == UnityWebRequest.Result.Success)
                    wavBytes = sReq.downloadHandler.data;
            }

            if (wavBytes != null && wavBytes.Length > 64)
            {
                float[] parsedSamples = null;
                int parsedFrames = 0;
                int parsedChannels = 1;
                int parsedRate = 24000;
                bool parseOk = false;

                // Decode RIFF WAVE PCM -> float[] on a background CPU worker thread so Unity's 60Hz main thread has 0ms stall!
                var decodeTask = System.Threading.Tasks.Task.Run(() =>
                {
                    parseOk = TryParseWavSamples(wavBytes, out parsedSamples, out parsedFrames, out parsedChannels, out parsedRate);
                });
                while (!decodeTask.IsCompleted)
                    yield return null;

                if (parseOk && parsedSamples != null && parsedFrames > 0)
                {
                    AudioClip clip = AudioClip.Create("AivisNarratorLine", parsedFrames, parsedChannels, parsedRate, false);
                    clip.SetData(parsedSamples, 0);
                    EnsureNarratorAudioSource();
                    if (_currentDynamicClip != null)
                        Destroy(_currentDynamicClip);
                    _currentDynamicClip = clip;
                    _narratorSource.clip = clip;
                    _narratorSource.volume = ArenaAudioSystem.IsHumanSfxMuted ? 0f : narrationVolume;
                    _narratorSource.Play();
                }
            }

            _ttsBusy = false;
        }

        // =====================================================================
        // Pure-C# Chunk-Scanning RIFF WAVE Decoder (16/24/32-bit PCM & Float32)
        // =====================================================================
        public static bool TryDecodeWavToAudioClip(byte[] wav, string clipName, out AudioClip clip)
        {
            clip = null;
            if (!TryParseWavSamples(wav, out float[] samples, out int frameCount, out int channels, out int sampleRate))
                return false;
            clip = AudioClip.Create(clipName, frameCount, channels, sampleRate, false);
            clip.SetData(samples, 0);
            return true;
        }

        public static bool TryParseWavSamples(
            byte[] wav,
            out float[] samples,
            out int frameCount,
            out int channels,
            out int sampleRate)
        {
            samples = null;
            frameCount = 0;
            channels = 1;
            sampleRate = 24000;
            if (wav == null || wav.Length < 44) return false;
            if (wav[0] != 'R' || wav[1] != 'I' || wav[2] != 'F' || wav[3] != 'F') return false;
            if (wav[8] != 'W' || wav[9] != 'A' || wav[10] != 'V' || wav[11] != 'E') return false;

            int audioFormat = 1;
            int bitsPerSample = 16;
            int dataOffset = -1;
            int dataBytes = 0;

            int pos = 12;
            while (pos + 8 <= wav.Length)
            {
                string chunkId = Encoding.ASCII.GetString(wav, pos, 4);
                int chunkSize = BitConverter.ToInt32(wav, pos + 4);
                pos += 8;
                if (chunkSize < 0 || pos > wav.Length) break;

                if (chunkId == "fmt " && pos + 16 <= wav.Length)
                {
                    audioFormat = BitConverter.ToUInt16(wav, pos);
                    channels = Math.Max(1, (int)BitConverter.ToUInt16(wav, pos + 2));
                    sampleRate = Math.Max(8000, BitConverter.ToInt32(wav, pos + 4));
                    bitsPerSample = BitConverter.ToUInt16(wav, pos + 14);
                }
                else if (chunkId == "data")
                {
                    dataOffset = pos;
                    dataBytes = Math.Min(chunkSize, wav.Length - pos);
                    break;
                }
                pos += chunkSize + (chunkSize & 1);
            }

            if (dataOffset < 0 || dataBytes <= 0) return false;
            int bytesPerSample = Math.Max(2, bitsPerSample / 8);
            int totalSamples = dataBytes / bytesPerSample;
            frameCount = totalSamples / channels;
            if (frameCount <= 0) return false;

            samples = new float[frameCount * channels];
            int idx = dataOffset;
            for (int i = 0; i < samples.Length && idx + bytesPerSample <= wav.Length; i++)
            {
                if (audioFormat == 3 && bitsPerSample == 32)
                {
                    float v = BitConverter.ToSingle(wav, idx);
                    samples[i] = v < -1f ? -1f : (v > 1f ? 1f : v);
                }
                else if (bitsPerSample == 16)
                {
                    short s16 = BitConverter.ToInt16(wav, idx);
                    samples[i] = s16 / 32768f;
                }
                else if (bitsPerSample == 24)
                {
                    int s24 = (wav[idx] | (wav[idx + 1] << 8) | ((sbyte)wav[idx + 2] << 16));
                    samples[i] = s24 / 8388608f;
                }
                else if (bitsPerSample == 32)
                {
                    int s32 = BitConverter.ToInt32(wav, idx);
                    samples[i] = s32 / 2147483648f;
                }
                idx += bytesPerSample;
            }
            return true;
        }

        // =====================================================================
        // USB Microphone + faster-whisper (CPU-int8, 0MB GPU VRAM) Sidecar
        // =====================================================================
        public void ToggleMicRecording()
        {
            if (IsTranscribingMic)
                return;

            if (_sttProc == null || SafeProcExited(_sttProc))
            {
                _autoStartRecordWhenReady = true;
                StartSttSidecar();
                return;
            }

            if (!IsSttReady)
            {
                _autoStartRecordWhenReady = true;
                coachStatusMessage = "Whisper音声認識モデル準備中(CPU)...";
                return;
            }

            if (!IsRecordingMic)
            {
                SendSttCommand("{\"cmd\":\"start_record\"}");
                coachStatusMessage = "録音開始中...";
            }
            else
            {
                IsRecordingMic = false;
                IsTranscribingMic = true;
                SendSttCommand("{\"cmd\":\"stop_record\"}");
                coachStatusMessage = "Whisperで日本語文字起こし中...";
            }
        }

        private void StartSttSidecar()
        {
            if (coordinator == null) return;
            StopSttSidecar();

            string ws = coordinator.GetResolvedWorkspaceDir();
            string pyExe = coordinator.pythonExePath;
            string scriptPath = Path.Combine(ws, "companion_voice_io.py");
            if (string.IsNullOrEmpty(pyExe) || !File.Exists(pyExe) || !File.Exists(scriptPath))
            {
                coachStatusMessage = "Python環境または companion_voice_io.py が見つかりません。";
                return;
            }

            IsSttLoading = true;
            IsSttReady = false;
            coachStatusMessage = "Whisper(CPU int8) 起動中... 初回のみ数秒かかります";

            int unityPid = Process.GetCurrentProcess().Id;
            string args = $"-B \"{scriptPath}\" --mode stt_sidecar --model {whisperModelSize} --device cpu --language ja --parent-pid {unityPid}";
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = pyExe,
                    Arguments = args,
                    WorkingDirectory = ws,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };
                psi.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
                psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                psi.EnvironmentVariables["PYTHONUTF8"] = "1";

                _sttProc = new Process { StartInfo = psi, EnableRaisingEvents = true };
                _sttProc.OutputDataReceived += (_, e) =>
                {
                    if (!string.IsNullOrWhiteSpace(e.Data))
                        _sttEventQueue.Enqueue(e.Data.Trim());
                };
                if (_sttProc.Start())
                {
                    _sttProc.BeginOutputReadLine();
                    _sttProc.BeginErrorReadLine();
                }
            }
            catch (Exception ex)
            {
                _sttProc = null;
                IsSttLoading = false;
                coachStatusMessage = $"Whisper起動エラー: {ex.Message}";
            }
        }

        private void SendSttCommand(string jsonLine)
        {
            try
            {
                if (_sttProc != null && !SafeProcExited(_sttProc))
                {
                    _sttProc.StandardInput.WriteLine(jsonLine);
                    _sttProc.StandardInput.Flush();
                }
            }
            catch (Exception ex)
            {
                coachStatusMessage = $"STT通信失敗: {ex.Message}";
            }
        }

        public void StopSttSidecar()
        {
            IsRecordingMic = false;
            IsTranscribingMic = false;
            IsSttReady = false;
            IsSttLoading = false;
            if (_sttProc == null) return;
            try
            {
                if (!SafeProcExited(_sttProc))
                {
                    try
                    {
                        _sttProc.StandardInput.WriteLine("{\"cmd\":\"quit\"}");
                        _sttProc.StandardInput.Flush();
                        _sttProc.StandardInput.Close();
                    }
                    catch { }
                    if (!_sttProc.WaitForExit(800))
                        _sttProc.Kill();
                }
            }
            catch { }
            finally
            {
                try { _sttProc.Dispose(); } catch { }
                _sttProc = null;
            }
        }

        private static bool SafeProcExited(Process p)
        {
            try { return p == null || p.HasExited; }
            catch { return true; }
        }

        private void DrainSttSidecarEvents()
        {
            while (_sttEventQueue.TryDequeue(out string raw))
            {
                SttSidecarEventDto ev = null;
                try { ev = JsonUtility.FromJson<SttSidecarEventDto>(raw); }
                catch { continue; }
                if (ev == null || string.IsNullOrEmpty(ev.@event)) continue;

                if (ev.@event == "ready")
                {
                    IsSttLoading = false;
                    IsSttReady = true;
                    ActiveMicName = ev.mic ?? "";
                    if (_autoStartRecordWhenReady)
                    {
                        _autoStartRecordWhenReady = false;
                        SendSttCommand("{\"cmd\":\"start_record\"}");
                    }
                    else
                    {
                        coachStatusMessage = $"マイク準備完了 ({ActiveMicName})";
                    }
                }
                else if (ev.@event == "recording_started")
                {
                    IsRecordingMic = true;
                    IsTranscribingMic = false;
                    if (!string.IsNullOrEmpty(ev.mic))
                        ActiveMicName = ev.mic;
                    coachStatusMessage = $"録音中 ({ActiveMicName}) — もう一度押すと文字起こしします";
                }
                else if (ev.@event == "transcript")
                {
                    IsRecordingMic = false;
                    IsTranscribingMic = false;
                    string recognized = (ev.text ?? "").Trim();
                    if (string.IsNullOrEmpty(recognized))
                    {
                        coachStatusMessage = "音声が検出されませんでした。もう一度お試しください。";
                    }
                    else
                    {
                        coachInputText = recognized;
                        if (autoSendAfterWhisper)
                        {
                            SubmitCoachingProposalToGemma();
                        }
                        else
                        {
                            coachStatusMessage = $"文字起こし完了 ({ev.infer_ms:F0}ms) — 内容確認後に[提案送信]を押してください";
                        }
                    }
                }
                else if (ev.@event == "error")
                {
                    IsRecordingMic = false;
                    IsTranscribingMic = false;
                    IsSttLoading = false;
                    coachStatusMessage = $"音声認識エラー: {ev.detail}";
                }
            }
        }

        // =====================================================================
        // Submit Human Strategic Proposal -> Gemma 4 12B -> Brain Live Override
        // =====================================================================
        public bool SubmitCoachingProposalToGemma()
        {
            string cleaned = (coachInputText ?? "").Trim();
            if (string.IsNullOrEmpty(cleaned) || coordinator == null)
                return false;

            string ws = coordinator.GetResolvedWorkspaceDir();
            string reqPath = Path.Combine(ws, "checkpoints", "user_runs", "strategy_knowledge.sqlite3.coach_req.json");
            string tmpPath = reqPath + ".tmp";
            try
            {
                string dir = Path.GetDirectoryName(reqPath);
                if (!string.IsNullOrEmpty(dir))
                    Directory.CreateDirectory(dir);

                var dto = new CoachReqPayloadDto
                {
                    ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
                    stage_id = 1,
                    user_text = cleaned.Length > 260 ? cleaned.Substring(0, 260) : cleaned
                };
                File.WriteAllText(tmpPath, JsonUtility.ToJson(dto), new UTF8Encoding(false));
                if (File.Exists(reqPath))
                    File.Delete(reqPath);
                File.Move(tmpPath, reqPath);

                coachInputText = "";
                coachStatusMessage = $"Gemmaへ提案送信済:「{dto.user_text}」→ 戦略変換・注入中...";
                return true;
            }
            catch (Exception ex)
            {
                coachStatusMessage = $"提案送信エラー: {ex.Message}";
                return false;
            }
        }
    }
}
