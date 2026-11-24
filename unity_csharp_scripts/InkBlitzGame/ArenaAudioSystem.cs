// ArenaAudioSystem.cs — Arena-local acoustic impulse ring buffer (AI hearing) + Pooled Studio SFX (Human recording)
using UnityEngine;

namespace InkBlitz3D
{
    public class ArenaAudioSystem : MonoBehaviour
    {
        private const int MaxEvents = 32;
        private const int SfxVoicePoolSize = 10;

        public static bool IsHumanSfxMuted = false;

        [Header("Human Recording SFX Settings (Zero Effect on AI Hearing)")]
        public bool enableHumanSfx = true;
        [Range(0f, 1.5f)] public float masterSfxVolume = 0.88f;

        private struct AcousticImpulse
        {
            public bool active;
            public Vector3 worldPos;
            public int teamId;
            public float intensity;
            public float timeRemaining;
        }

        private readonly AcousticImpulse[] _events = new AcousticImpulse[MaxEvents];
        private InkBlitzArenaInstance _arena;
        private InkBlitzMultiArenaManager _multiManager;
        private Transform _voicePoolRoot;
        private AudioSource[] _voices;
        private int _voiceCursor = 0;
        private int _shootVariantCursor = 0;
        private float _lastShootSfxRealtime = -10f;
        private float _lastDiveSfxRealtime = -10f;

        private static bool _clipsLoaded = false;
        private static AudioClip[] _clipsShoot;
        private static AudioClip _clipShootBlitzCore;
        private static AudioClip _clipInkDive;
        private static AudioClip _clipSplatKo;
        private static AudioClip _clipShieldPop;
        private static AudioClip _clipGoalTouchdown;

        public int ShootSfxPlayedCount { get; private set; } = 0;
        public int DiveSfxPlayedCount { get; private set; } = 0;
        public int SplatSfxPlayedCount { get; private set; } = 0;
        public int ShieldPopSfxPlayedCount { get; private set; } = 0;
        public int GoalSfxPlayedCount { get; private set; } = 0;

        public static bool EnsureClipsLoaded()
        {
            if (_clipsLoaded && _clipGoalTouchdown != null)
                return true;

            _clipsShoot = new[]
            {
                Resources.Load<AudioClip>("Audio/sfx_shoot_1"),
                Resources.Load<AudioClip>("Audio/sfx_shoot_2"),
                Resources.Load<AudioClip>("Audio/sfx_shoot_3"),
            };
            _clipShootBlitzCore = Resources.Load<AudioClip>("Audio/sfx_shoot_blitzcore");
            _clipInkDive = Resources.Load<AudioClip>("Audio/sfx_ink_dive");
            _clipSplatKo = Resources.Load<AudioClip>("Audio/sfx_splat_ko");
            _clipShieldPop = Resources.Load<AudioClip>("Audio/sfx_shield_pop");
            _clipGoalTouchdown = Resources.Load<AudioClip>("Audio/sfx_goal_touchdown");

            _clipsLoaded = _clipsShoot[0] != null
                        && _clipShootBlitzCore != null
                        && _clipInkDive != null
                        && _clipSplatKo != null
                        && _clipShieldPop != null
                        && _clipGoalTouchdown != null;
            return _clipsLoaded;
        }

        public void BindArena(InkBlitzArenaInstance arena)
        {
            _arena = arena;
            if (_multiManager == null)
                _multiManager = Object.FindAnyObjectByType<InkBlitzMultiArenaManager>();
            if (Application.isPlaying)
            {
                EnsureClipsLoaded();
                EnsureVoicePool();
            }
        }

        private void EnsureVoicePool()
        {
            if (_voices != null && _voices.Length == SfxVoicePoolSize && _voices[0] != null)
                return;

            if (_voicePoolRoot == null)
            {
                var go = new GameObject("SfxVoicePool");
                go.transform.SetParent(transform, false);
                _voicePoolRoot = go.transform;
            }

            _voices = new AudioSource[SfxVoicePoolSize];
            for (int i = 0; i < SfxVoicePoolSize; i++)
            {
                var vGo = new GameObject($"SfxVoice_{i}");
                vGo.transform.SetParent(_voicePoolRoot, false);
                var src = vGo.AddComponent<AudioSource>();
                src.playOnAwake = false;
                src.loop = false;
                src.rolloffMode = AudioRolloffMode.Linear;
                src.minDistance = 12f;
                src.maxDistance = 62f;
                src.dopplerLevel = 0f;
                _voices[i] = src;
            }
        }

        private bool ShouldOutputHumanAudio()
        {
            if (!enableHumanSfx || IsHumanSfxMuted || !Application.isPlaying)
                return false;
            if (_arena == null)
                _arena = GetComponent<InkBlitzArenaInstance>();
            if (_multiManager == null)
                _multiManager = Object.FindAnyObjectByType<InkBlitzMultiArenaManager>();

            if (_multiManager != null)
            {
                if (_multiManager.isPausedForLauncher || _multiManager.disableAll3DRendering)
                    return false;
                if (_arena != null)
                {
                    int targetArena = _multiManager.overviewMode ? 0 : _multiManager.focusedArenaIndex;
                    if (_arena.arenaIndex != targetArena)
                        return false;
                }
            }
            return true;
        }

        private void PlayClipAt(AudioClip clip, Vector3 worldPos, float volume, float pitch, float spatialBlend)
        {
            if (clip == null || !ShouldOutputHumanAudio())
                return;

            EnsureVoicePool();
            if (_voices == null || _voices.Length == 0)
                return;

            AudioSource chosen = null;
            for (int k = 0; k < SfxVoicePoolSize; k++)
            {
                int idx = (_voiceCursor + k) % SfxVoicePoolSize;
                if (_voices[idx] != null && !_voices[idx].isPlaying)
                {
                    chosen = _voices[idx];
                    _voiceCursor = (idx + 1) % SfxVoicePoolSize;
                    break;
                }
            }
            if (chosen == null)
            {
                chosen = _voices[_voiceCursor];
                _voiceCursor = (_voiceCursor + 1) % SfxVoicePoolSize;
            }
            if (chosen == null)
                return;

            float fastScaleAtten = Time.timeScale > 2.3f ? 0.38f : 1.0f;
            chosen.transform.position = worldPos;
            chosen.spatialBlend = Mathf.Clamp01(spatialBlend);
            chosen.pitch = Mathf.Clamp(pitch, 0.5f, 1.8f);
            chosen.volume = Mathf.Clamp01(volume * masterSfxVolume * fastScaleAtten);
            chosen.clip = clip;
            chosen.Play();
        }

        /// <summary>
        /// Shoot SE: strictly restricted to Main AI Agents (`!isSupportBot`). Support Trooper Bots are silent!
        /// </summary>
        public void PlayShootSfx(Vector3 worldPos, bool isBlitzCoreBlast, bool isSupportBot)
        {
            if (isSupportBot)
                return;

            ShootSfxPlayedCount++;
            if (!EnsureClipsLoaded())
                return;

            float now = Time.realtimeSinceStartup;
            if (!isBlitzCoreBlast && (now - _lastShootSfxRealtime) < 0.042f)
                return;
            _lastShootSfxRealtime = now;

            if (isBlitzCoreBlast)
            {
                PlayClipAt(_clipShootBlitzCore, worldPos, 0.72f, Random.Range(0.96f, 1.04f), 0.30f);
            }
            else
            {
                AudioClip c = _clipsShoot[_shootVariantCursor % _clipsShoot.Length];
                _shootVariantCursor++;
                PlayClipAt(c, worldPos, 0.44f, Random.Range(0.95f, 1.05f), 0.35f);
            }
        }

        public void PlayInkDiveSfx(Vector3 worldPos, bool isSupportBot)
        {
            DiveSfxPlayedCount++;
            if (!EnsureClipsLoaded())
                return;

            float now = Time.realtimeSinceStartup;
            if ((now - _lastDiveSfxRealtime) < 0.06f)
                return;
            _lastDiveSfxRealtime = now;

            float vol = isSupportBot ? 0.22f : 0.60f;
            PlayClipAt(_clipInkDive, worldPos, vol, Random.Range(0.94f, 1.06f), 0.35f);
        }

        public void PlaySplatSfx(Vector3 worldPos, bool victimIsSupportBot)
        {
            SplatSfxPlayedCount++;
            if (!EnsureClipsLoaded())
                return;

            float vol = victimIsSupportBot ? 0.62f : 0.88f;
            PlayClipAt(_clipSplatKo, worldPos, vol, Random.Range(0.95f, 1.04f), 0.25f);
        }

        public void PlayShieldPopSfx(Vector3 worldPos)
        {
            ShieldPopSfxPlayedCount++;
            if (!EnsureClipsLoaded())
                return;

            PlayClipAt(_clipShieldPop, worldPos, 0.96f, 1.0f, 0.15f);
        }

        public void PlayGoalTouchdownSfx(Vector3 worldPos)
        {
            GoalSfxPlayedCount++;
            if (!EnsureClipsLoaded())
                return;

            PlayClipAt(_clipGoalTouchdown, worldPos, 1.0f, 1.0f, 0.05f);
        }

        public void StopAllHumanSfx()
        {
            if (_voices == null) return;
            for (int i = 0; i < _voices.Length; i++)
            {
                if (_voices[i] != null && _voices[i].isPlaying)
                    _voices[i].Stop();
            }
        }

        public void Clear()
        {
            for (int i = 0; i < MaxEvents; i++)
                _events[i].active = false;
        }

        public void EmitSound(Vector3 worldPos, int sourceTeamId, float intensity, float durationSec = 0.30f)
        {
            int targetIdx = -1;
            float minRem = float.MaxValue;

            for (int i = 0; i < MaxEvents; i++)
            {
                if (!_events[i].active)
                {
                    targetIdx = i;
                    break;
                }
                if (_events[i].timeRemaining < minRem)
                {
                    minRem = _events[i].timeRemaining;
                    targetIdx = i;
                }
            }

            if (targetIdx < 0) targetIdx = 0;
            _events[targetIdx] = new AcousticImpulse
            {
                active = true,
                worldPos = worldPos,
                teamId = sourceTeamId,
                intensity = Mathf.Clamp(intensity, 0f, 2f),
                timeRemaining = Mathf.Max(0.02f, durationSec)
            };
        }

        public void StepAudio(float dt)
        {
            if (dt <= 0f) return;
            for (int i = 0; i < MaxEvents; i++)
            {
                if (!_events[i].active) continue;
                _events[i].timeRemaining -= dt;
                if (_events[i].timeRemaining <= 0f)
                    _events[i].active = false;
            }
        }

        public void ComputeObserverAudio(
            Transform observerEye,
            int observerTeamId,
            LayerMask wallMask,
            out Vector2 egoSinCos,
            out float loudness01)
        {
            egoSinCos = Vector2.zero;
            loudness01 = 0f;
            if (observerEye == null) return;

            Vector3 eyePos = observerEye.position;
            Quaternion invYaw = Quaternion.Inverse(Quaternion.Euler(0f, observerEye.eulerAngles.y, 0f));

            Vector3 weightedLocalDir = Vector3.zero;
            float maxAtt = 0f;

            for (int i = 0; i < MaxEvents; i++)
            {
                if (!_events[i].active) continue;
                if (_events[i].teamId == observerTeamId) continue;

                Vector3 delta = _events[i].worldPos - eyePos;
                float d = delta.magnitude;
                if (d < 0.05f || d > 24f) continue;

                float att = _events[i].intensity / (1f + 0.045f * d * d);
                if (d > 0.4f && Physics.Raycast(eyePos, delta / d, out RaycastHit hit, d - 0.3f, wallMask, QueryTriggerInteraction.Ignore))
                {
                    if (hit.collider.GetComponentInParent<InkBlitzPlayerAgent>() == null)
                        att *= 0.55f;
                }

                Vector3 local = invYaw * delta;
                local.y = 0f;
                float hLen = local.magnitude;
                if (hLen > 1e-4f)
                {
                    weightedLocalDir += (local / hLen) * att;
                    if (att > maxAtt) maxAtt = att;
                }
            }

            if (weightedLocalDir.sqrMagnitude > 1e-6f)
            {
                weightedLocalDir.Normalize();
                egoSinCos = new Vector2(weightedLocalDir.x, weightedLocalDir.z);
                loudness01 = Mathf.Clamp01(maxAtt);
            }
        }
    }
}
