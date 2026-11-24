// UniversalBrainBridge3D.cs — Byte-Exact UB3D(v1/v2) & UBMC TCP_NODELAY Bridge for Unity 6+
// Supports:
//   - 16x16x4 Foveal Raycast + Floor Ink/Turf Surface Encoding (`IFloorSurfaceSampler`)
//   - Optional 2x64x64x3 Binocular Stereo RGB24 (`UB3D` v2 `34,968` bytes)
//   - Decoupled Body Locomotion (`target_pose[0:3]`) vs. Independent Weapon/Camera Aim (`target_pose[3:5]`)
//   - Mid-Episode Respawn Discontinuity Signaling (`flags |= 16`, preserving Python's 4-layer map!)
// License: GPL-3.0

using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

namespace UniversalBrain3D
{
    public class UniversalBrainBridge3D : MonoBehaviour
    {
        public const uint UNITY_SENSORY_MAGIC = 0x33444255; // "UB3D"
        public const uint UNITY_MOTOR_MAGIC = 0x434D4255;   // "UBMC"
        public const int UB3D_V1_BYTES = 6296;
        public const int UB3D_V2_BYTES = 34968;
        public const int UBMC_BYTES = 128;

        public const int PROPRIO_DIM = 16;
        public const int MAX_ENTITIES = 48;
        public const int ENTITY_DIM = 10;
        public const int FOVEA_H = 16;
        public const int FOVEA_W = 16;
        public const int FOVEA_C = 4;

        // Universal Motor Verbs (matches `task_profiles_3d.py`)
        public const int VERB_IDLE_OBSERVE = 0;
        public const int VERB_APPROACH_HOVER = 1;
        public const int VERB_COMMIT_INTERACT = 2;
        public const int VERB_GRASP_PICK = 3;
        public const int VERB_RELEASE_PLACE = 4;   // Primary ranged shoot / paint!
        public const int VERB_EVADE_HAZARD = 5;    // Fast swim / dodge retreat!
        public const int VERB_IMPULSE_PUSH = 6;    // Sub/special burst!
        public const int VERB_COMMUNICATE_PING = 7;

        [Header("Network & Arena Settings")]
        public string host = "127.0.0.1";
        public int port = 38055;
        public int arenaIndex = 0;
        public int stageId = 1; // >0 keys Python's Hippocampal Structural Schema Bank (auto-resets when stageId changes!)
        public int teamId = 1; // +1 = Team A (South->North), -1 = Team B (North->South)
        public Vector3 arenaOrigin = Vector3.zero;
        public Vector3 localWorldMin = new Vector3(-12f, 0f, -16f);
        public Vector3 localWorldMax = new Vector3(12f, 6f, 16f);

        [Header("1st-Person Sensor Configuration")]
        public Transform eyeTransform;
        public LayerMask raycastMask = ~0;
        public float foveaHalfAngleDeg = 38f;
        public float foveaMaxRangeM = 16f;
        [Range(1, 4)] public int foveaCaptureStride = 1; // 1 = 60Hz full rate (HUD agent), 2 = staggered 30Hz reuse on peer bridges
        public MonoBehaviour floorSurfaceSamplerComponent; // Optional component implementing IFloorSurfaceSampler
        public bool isCurrentlyRespawning = false;
        [Range(0f, 0.78f)] public float activeBuffOrTransit01 = 0f; // Somatic buff/special timer in [0..0.78] (< 0.90 respawn threshold)

        [Header("Live Decoded Brain Output (Read-Only Telemetry)")]
        public int latestMotorVerb = VERB_IDLE_OBSERVE;
        public bool latestCommitGateOpen = false;
        public bool latestVetoActive = false;
        public int latestHoldReasonId = 0;
        public Vector3 latestWorldMoveTarget;
        public float latestAimYawRad = 0f;
        public float latestAimPitchRad = -0.12f;
        public float latestSpeedScale = 0.85f;
        public Vector3 latestGazeYpd = new Vector3(0f, -0.15f, 6f);
        public float requestedTimeScale = 1.0f;
        public string latestInnerSpeech = "";

        // Action hooks for your InkBlitz 3D / character controller to subscribe to:
        public event Action<Vector3, float, int, bool> OnLocomotionCommand; // (worldMoveTarget, speedScale, motorVerb, commitOpen)
        public event Action<float, float, Vector3> OnAimAndGazeCommand;     // (aimYawRad, aimPitchRad, gazeYpd)

        private TcpClient _client;
        private NetworkStream _stream;
        private ulong _frameSeq = 0;
        private uint _episodeId = 1;
        private uint _actionEpoch = 1;
        private bool _pendingRespawnFlag = false;
        private float _accumStepReward = 0f;

        private readonly byte[] _txBufV1 = new byte[UB3D_V1_BYTES];
        private readonly byte[] _rxBuf = new byte[UBMC_BYTES];
        private readonly float[] _floatsBuf = new float[1530];
        private readonly float[] _proprio = new float[PROPRIO_DIM];
        private readonly float[] _cachedFoveaFloats = new float[FOVEA_H * FOVEA_W * FOVEA_C];
        private bool _hasCachedFovea = false;
        private readonly Dictionary<Collider, UniversalSemanticEntityTag> _colliderDirectTagCache = new Dictionary<Collider, UniversalSemanticEntityTag>(64);
        private readonly Dictionary<Collider, UniversalSemanticEntityTag> _colliderParentTagCache = new Dictionary<Collider, UniversalSemanticEntityTag>(128);
        private IFloorSurfaceSampler _surfaceSampler;
        private ulong _foveaSampleSeq = 0;

        // Zero-allocation read-only 16x16 V1 foveal depth/surface buffers for lightweight HUD inspection (`02 DEPTH`)
        public readonly float[] latestFoveaDepth16 = new float[FOVEA_H * FOVEA_W];
        public readonly float[] latestFoveaNormalY16 = new float[FOVEA_H * FOVEA_W];
        public readonly float[] latestFoveaSalience16 = new float[FOVEA_H * FOVEA_W];
        public ulong LatestFoveaSeq => _foveaSampleSeq;
        public float PendingStepReward => _accumStepReward;

        private static readonly uint[] Crc32Table = BuildCrc32Table();

        private static uint[] BuildCrc32Table()
        {
            uint[] table = new uint[256];
            for (uint i = 0; i < 256; i++)
            {
                uint c = i;
                for (int k = 0; k < 8; k++)
                    c = ((c & 1) != 0) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                table[i] = c;
            }
            return table;
        }

        public static uint ComputeCrc32(byte[] data, int offset, int length)
        {
            uint crc = 0xFFFFFFFFu;
            int end = offset + length;
            for (int i = offset; i < end; i++)
                crc = Crc32Table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
            return crc ^ 0xFFFFFFFFu;
        }

        public void SetFloorSurfaceSampler(IFloorSurfaceSampler sampler)
        {
            _surfaceSampler = sampler;
            if (sampler is MonoBehaviour mb)
                floorSurfaceSamplerComponent = mb;
        }

        private void Awake()
        {
            if (eyeTransform == null)
                eyeTransform = transform;
            if (floorSurfaceSamplerComponent != null)
                _surfaceSampler = floorSurfaceSamplerComponent as IFloorSurfaceSampler;
        }

        public bool ConnectToBrain(float timeoutSec = 2.0f)
        {
            Disconnect();
            try
            {
                _client = new TcpClient();
                _client.NoDelay = true;
                var ar = _client.BeginConnect(host, port, null, null);
                if (!ar.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(timeoutSec)))
                {
                    _client.Close();
                    _client = null;
                    return false;
                }
                _client.EndConnect(ar);
                _client.SendTimeout = 5000;
                _client.ReceiveTimeout = 5000;
                _stream = _client.GetStream();
                return true;
            }
            catch
            {
                Disconnect();
                return false;
            }
        }

        public void Disconnect()
        {
            try { _stream?.Close(); } catch { }
            try { _client?.Close(); } catch { }
            _stream = null;
            _client = null;
            _hasCachedFovea = false;
            _colliderDirectTagCache.Clear();
            _colliderParentTagCache.Clear();
        }

        public bool IsConnected => _client != null && _client.Connected && _stream != null;

        public void AddStepReward(float r) => _accumStepReward += r;

        public void NotifyActionCommitted() => _actionEpoch++;

        /// <summary>
        /// Call when the player is splatted and respawns at base within the SAME match (`episodeId` unchanged).
        /// Python's `EntorhinalHippocampal3DEngineV2` will keep the explored 4-layer map while resetting EKF drift!
        /// </summary>
        public void NotifyMidEpisodeRespawn()
        {
            _pendingRespawnFlag = true;
            _actionEpoch++;
        }

        public void NotifyNewEpisodeReset()
        {
            _episodeId++;
            _actionEpoch = 1;
            _pendingRespawnFlag = false;
            _accumStepReward = 0f;
            _hasCachedFovea = false;
            _colliderDirectTagCache.Clear();
            _colliderParentTagCache.Clear();
        }

        /// <summary>
        /// Execute one synchronous 60Hz Sensory -> Brain -> Motor lockstep exchange.
        /// </summary>
        public bool StepBrainExchange(
            Vector3 worldVel,
            bool canCommitNow,
            bool episodeDone,
            float hazardLevel,
            float surfacePostureState,
            float resourceOrCarrierState,
            float objectiveProgress01,
            Vector2 egoAudioSinCos,
            float audioLoudness,
            IReadOnlyList<UniversalSemanticEntityTag> arenaEntities,
            string caregiverUtterance = ""
        )
        {
            if (!BeginBrainExchangeSend(
                worldVel,
                canCommitNow,
                episodeDone,
                hazardLevel,
                surfacePostureState,
                resourceOrCarrierState,
                objectiveProgress01,
                egoAudioSinCos,
                audioLoudness,
                arenaEntities,
                caregiverUtterance))
            {
                return false;
            }
            return CompleteBrainExchangeRecv();
        }

        public bool BeginBrainExchangeSend(
            Vector3 worldVel,
            bool canCommitNow,
            bool episodeDone,
            float hazardLevel,
            float surfacePostureState,
            float resourceOrCarrierState,
            float objectiveProgress01,
            Vector2 egoAudioSinCos,
            float audioLoudness,
            IReadOnlyList<UniversalSemanticEntityTag> arenaEntities,
            string caregiverUtterance = ""
        )
        {
            if (!IsConnected)
                return false;
            if (_surfaceSampler == null && floorSurfaceSamplerComponent != null)
                _surfaceSampler = floorSurfaceSamplerComponent as IFloorSurfaceSampler;

            _frameSeq++;
            Vector3 localPos = transform.position - arenaOrigin;
            Vector3 span = Vector3.Max(Vector3.one, localWorldMax - localWorldMin);

            // 1. Populate 16D Universal Proprioception Contract
            _proprio[0] = Mathf.Clamp01((localPos.x - localWorldMin.x) / span.x);
            _proprio[1] = Mathf.Clamp01((localPos.y - localWorldMin.y) / span.y);
            _proprio[2] = Mathf.Clamp01((localPos.z - localWorldMin.z) / span.z);
            _proprio[3] = Mathf.Clamp(worldVel.x / 8f, -1f, 1f);
            _proprio[4] = Mathf.Clamp(worldVel.y / 8f, -1f, 1f);
            _proprio[5] = Mathf.Clamp(worldVel.z / 8f, -1f, 1f);
            _proprio[6] = canCommitNow ? 1f : 0f;
            _proprio[7] = (_pendingRespawnFlag || isCurrentlyRespawning) ? 1f : Mathf.Clamp(activeBuffOrTransit01, 0f, 0.78f);
            _proprio[8] = Mathf.Clamp01(hazardLevel);
            _proprio[10] = Mathf.Clamp(egoAudioSinCos.x, -1f, 1f);
            _proprio[11] = Mathf.Clamp(egoAudioSinCos.y, -1f, 1f);
            _proprio[12] = Mathf.Clamp01(audioLoudness);
            _proprio[13] = Mathf.Clamp01(surfacePostureState);
            _proprio[14] = Mathf.Clamp01(resourceOrCarrierState);
            _proprio[15] = Mathf.Clamp01(objectiveProgress01);

            // 2. Frustum-cull & line-of-sight filter 1st-person visible entities (strictly zero wallhacks!)
            Array.Clear(_floatsBuf, 0, _floatsBuf.Length);
            int idxEntBase = 10 + PROPRIO_DIM;
            int visCount = CollectVisibleEntities(arenaEntities, idxEntBase);
            _proprio[9] = (float)visCount / MAX_ENTITIES;

            // 3. Cast 16x16 Foveal Rays + Encode Floor Ink/Turf Surface into Channel 3
            int idxFoveaBase = idxEntBase + MAX_ENTITIES * ENTITY_DIM;
            int effStride = Mathf.Clamp(foveaCaptureStride, 1, 4);
            bool forceFreshFovea = !_hasCachedFovea || episodeDone || _pendingRespawnFlag || effStride <= 1
                                || (((_frameSeq + (ulong)Mathf.Max(0, port)) % (ulong)effStride) == 0UL);
            if (forceFreshFovea)
            {
                CaptureFovealRaycastPatch(idxFoveaBase);
                Array.Copy(_floatsBuf, idxFoveaBase, _cachedFoveaFloats, 0, _cachedFoveaFloats.Length);
                _hasCachedFovea = true;
            }
            else
            {
                Array.Copy(_cachedFoveaFloats, 0, _floatsBuf, idxFoveaBase, _cachedFoveaFloats.Length);
            }

            float eyeYawRad = eyeTransform.eulerAngles.y * Mathf.Deg2Rad;
            float pitchDeg = eyeTransform.eulerAngles.x;
            if (pitchDeg > 180f) pitchDeg -= 360f;
            float eyePitchRad = -pitchDeg * Mathf.Deg2Rad;

            _floatsBuf[0] = localPos.x;
            _floatsBuf[1] = localPos.y;
            _floatsBuf[2] = localPos.z;
            _floatsBuf[3] = worldVel.x;
            _floatsBuf[4] = worldVel.y;
            _floatsBuf[5] = worldVel.z;
            _floatsBuf[6] = eyeYawRad;
            _floatsBuf[7] = eyePitchRad;
            _floatsBuf[8] = _accumStepReward;
            _floatsBuf[9] = Mathf.Clamp01(hazardLevel);
            Array.Copy(_proprio, 0, _floatsBuf, 10, PROPRIO_DIM);

            uint stageBits = ((uint)Mathf.Clamp(stageId, 0, 65535)) << 16;
            uint flags = (canCommitNow ? 1u : 0u)
                       | (episodeDone ? 2u : 0u)
                       | (_pendingRespawnFlag ? 16u : 0u)
                       | stageBits;
            _pendingRespawnFlag = false;
            _accumStepReward = 0f;

            // 4. Pack 6,296-byte UB3D v1 packet
            ulong tsNs = (ulong)(Time.realtimeSinceStartupAsDouble * 1e9);
            Buffer.BlockCopy(BitConverter.GetBytes(UNITY_SENSORY_MAGIC), 0, _txBufV1, 0, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(1u), 0, _txBufV1, 4, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(_frameSeq), 0, _txBufV1, 8, 8);
            Buffer.BlockCopy(BitConverter.GetBytes(tsNs), 0, _txBufV1, 16, 8);
            Buffer.BlockCopy(BitConverter.GetBytes(_episodeId), 0, _txBufV1, 24, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(_actionEpoch), 0, _txBufV1, 28, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(flags), 0, _txBufV1, 32, 4);
            Buffer.BlockCopy(BitConverter.GetBytes((uint)visCount), 0, _txBufV1, 36, 4);
            Buffer.BlockCopy(_floatsBuf, 0, _txBufV1, 40, _floatsBuf.Length * 4);

            int cgOffset = 40 + _floatsBuf.Length * 4;
            Array.Clear(_txBufV1, cgOffset, 128);
            if (!string.IsNullOrEmpty(caregiverUtterance))
            {
                int charCap = Mathf.Min(caregiverUtterance.Length, 42);
                Encoding.UTF8.GetBytes(caregiverUtterance, 0, charCap, _txBufV1, cgOffset);
            }
            uint crc = ComputeCrc32(_txBufV1, 0, UB3D_V1_BYTES - 8);
            Buffer.BlockCopy(BitConverter.GetBytes(crc), 0, _txBufV1, UB3D_V1_BYTES - 8, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(0u), 0, _txBufV1, UB3D_V1_BYTES - 4, 4);

            try
            {
                _stream.Write(_txBufV1, 0, UB3D_V1_BYTES);
                return true;
            }
            catch
            {
                Disconnect();
                return false;
            }
        }

        public bool CompleteBrainExchangeRecv()
        {
            if (!IsConnected)
                return false;
            try
            {
                int readTotal = 0;
                while (readTotal < UBMC_BYTES)
                {
                    int n = _stream.Read(_rxBuf, readTotal, UBMC_BYTES - readTotal);
                    if (n <= 0) { Disconnect(); return false; }
                    readTotal += n;
                }
            }
            catch
            {
                Disconnect();
                return false;
            }

            return UnpackAndDispatchMotorCommand(_rxBuf);
        }

        private readonly RaycastHit[] _rayHitsBuf = new RaycastHit[16];

        public bool IsEntityHiddenByStealth(UniversalSemanticEntityTag e, Vector3 observerEyePos)
        {
            if (e == null || !e.isStealthSubmerged || e.teamId == teamId || e.isCarryingPrimaryObjective)
                return false;
            float dist = Vector3.Distance(observerEyePos, e.transform.position);
            return e.EstimatedVelocity.magnitude < 2.5f || dist > 2.5f;
        }

        private UniversalSemanticEntityTag GetCachedColliderTag(Collider col, bool includeParents)
        {
            if (object.ReferenceEquals(col, null))
                return null;
            var dict = includeParents ? _colliderParentTagCache : _colliderDirectTagCache;
            if (dict.TryGetValue(col, out UniversalSemanticEntityTag cached))
                return cached;
            UniversalSemanticEntityTag resolved = includeParents
                ? col.GetComponentInParent<UniversalSemanticEntityTag>()
                : col.GetComponent<UniversalSemanticEntityTag>();
            if (dict.Count > 512)
                dict.Clear();
            dict[col] = resolved;
            return resolved;
        }

        private bool RaycastIgnoringSelf(Vector3 origin, Vector3 dir, float maxDist, out RaycastHit validHit)
        {
            int hitCount = Physics.RaycastNonAlloc(origin, dir, _rayHitsBuf, maxDist, raycastMask, QueryTriggerInteraction.Ignore);
            float bestDist = float.MaxValue;
            int bestIdx = -1;
            for (int i = 0; i < hitCount; i++)
            {
                ref RaycastHit h = ref _rayHitsBuf[i];
                Transform ht = h.transform;
                if (ht == null || ht == transform || ht.IsChildOf(transform))
                    continue;
                if (h.collider is CharacterController)
                {
                    var hitTag = GetCachedColliderTag(h.collider, includeParents: false);
                    if (hitTag != null && hitTag.isStealthSubmerged)
                    {
                        // Pass through stealth-hidden opponents AND empty air above flat submerged divers!
                        if (IsEntityHiddenByStealth(hitTag, origin) || h.point.y > hitTag.transform.position.y + 0.38f)
                            continue;
                    }
                }
                if (h.distance < bestDist)
                {
                    bestDist = h.distance;
                    bestIdx = i;
                }
            }
            if (bestIdx >= 0)
            {
                validHit = _rayHitsBuf[bestIdx];
                return true;
            }
            validHit = default;
            return false;
        }

        private int CollectVisibleEntities(IReadOnlyList<UniversalSemanticEntityTag> entities, int baseIdx)
        {
            if (entities == null || entities.Count == 0)
                return 0;
            Vector3 eyePos = eyeTransform.position;
            Vector3 fwd = eyeTransform.forward;
            float cosThresh = Mathf.Cos(foveaHalfAngleDeg * Mathf.Deg2Rad);
            int count = 0;

            for (int i = 0; i < entities.Count && count < MAX_ENTITIES; i++)
            {
                var e = entities[i];
                if (e == null || !e.isActiveAndEnabled || e.arenaIndex != arenaIndex || e.transform == transform)
                    continue;

                Vector3 rel = e.transform.position - eyePos;
                float dist = rel.magnitude;
                if (dist < 0.05f)
                    continue;

                bool isAlwaysKnownLandmark =
                    e.baseKind == UniversalEntityKind.AttackGoalPedestal ||
                    e.baseKind == UniversalEntityKind.DefendBasePedestal ||
                    e.isCarryingPrimaryObjective ||
                    (e.baseKind == UniversalEntityKind.PrimaryObjective && !e.isStealthSubmerged);

                Vector3 dir = rel / dist;
                float cosAng = Vector3.Dot(dir, fwd);

                if (isAlwaysKnownLandmark)
                {
                    if (dist > 45f)
                        continue;
                }
                else
                {
                    if (dist > foveaMaxRangeM)
                        continue;
                    // Stationary/slow stealth-submerged opponents on enemy ink are visually hidden!
                    if (IsEntityHiddenByStealth(e, eyePos))
                        continue;
                    if (cosAng < cosThresh && dist > 2.2f)
                        continue;

                    if (RaycastIgnoringSelf(eyePos, dir, dist - e.boundingRadius * 0.6f, out RaycastHit hit))
                    {
                        if (hit.transform != e.transform && !hit.transform.IsChildOf(e.transform))
                            continue;
                    }
                }

                Vector3 locEntPos = e.transform.position - arenaOrigin;
                Vector3 vel = e.EstimatedVelocity;
                int o = baseIdx + count * ENTITY_DIM;
                _floatsBuf[o + 0] = e.entityId;
                _floatsBuf[o + 1] = locEntPos.x;
                _floatsBuf[o + 2] = locEntPos.y;
                _floatsBuf[o + 3] = locEntPos.z;
                _floatsBuf[o + 4] = vel.x;
                _floatsBuf[o + 5] = vel.y;
                _floatsBuf[o + 6] = vel.z;
                _floatsBuf[o + 7] = e.boundingRadius;
                _floatsBuf[o + 8] = e.GetObserverRelativeTag(teamId);
                _floatsBuf[o + 9] = Mathf.Clamp(cosAng, 0.2f, 1.0f);
                count++;
            }
            return count;
        }

        /// <summary>
        /// Sample 16x16 foveal depth/surface into `latestFoveaDepth16` when offline (during live 60Hz TCP,
        /// `BeginBrainExchangeSend` already populates these buffers automatically with zero extra raycasts).
        /// </summary>
        public void SampleLiveFovealPatchForHud()
        {
            if (eyeTransform == null)
                eyeTransform = transform;
            if (_surfaceSampler == null && floorSurfaceSamplerComponent != null)
                _surfaceSampler = floorSurfaceSamplerComponent as IFloorSurfaceSampler;
            int idxFoveaBase = 10 + PROPRIO_DIM + MAX_ENTITIES * ENTITY_DIM;
            CaptureFovealRaycastPatch(idxFoveaBase);
        }

        private void CaptureFovealRaycastPatch(int baseOffset)
        {
            float tanHalf = Mathf.Tan(foveaHalfAngleDeg * Mathf.Deg2Rad);
            Vector3 origin = eyeTransform.position;
            Vector3 fwd = eyeTransform.forward;
            Vector3 right = eyeTransform.right;
            Vector3 up = eyeTransform.up;

            int ptr = baseOffset;
            int px = 0;
            for (int r = 0; r < FOVEA_H; r++)
            {
                float v = Mathf.Lerp(tanHalf, -tanHalf, r / (float)(FOVEA_H - 1));
                for (int c = 0; c < FOVEA_W; c++)
                {
                    float u = Mathf.Lerp(-tanHalf, tanHalf, c / (float)(FOVEA_W - 1));
                    Vector3 dir = (fwd + u * right + v * up).normalized;
                    if (RaycastIgnoringSelf(origin, dir, foveaMaxRangeM, out RaycastHit hit))
                    {
                        float normD = 1f - Mathf.Clamp01(hit.distance / foveaMaxRangeM);
                        Vector3 nrm = hit.normal;
                        float salCode = 0.50f;

                        var tagComp = GetCachedColliderTag(hit.collider, includeParents: true);
                        if (tagComp != null)
                        {
                            float t = tagComp.GetObserverRelativeTag(teamId);
                            salCode = (t < 0f) ? 0.95f : Mathf.Clamp(0.40f + 0.12f * t, 0.30f, 0.92f);
                        }
                        else if (nrm.y > 0.58f && _surfaceSampler != null)
                        {
                            // Encode Floor Ink/Turf state directly into Channel 3!
                            float surfVal = _surfaceSampler.SampleSurfaceValence(hit.point, teamId);
                            salCode = (surfVal > 0.25f) ? 0.78f : ((surfVal < -0.25f) ? 0.25f : 0.50f);
                        }

                        _floatsBuf[ptr + 0] = normD;
                        _floatsBuf[ptr + 1] = nrm.x;
                        _floatsBuf[ptr + 2] = nrm.y;
                        _floatsBuf[ptr + 3] = salCode;
                        latestFoveaDepth16[px] = normD;
                        latestFoveaNormalY16[px] = nrm.y;
                        latestFoveaSalience16[px] = salCode;
                    }
                    else
                    {
                        _floatsBuf[ptr + 0] = 0f;
                        _floatsBuf[ptr + 1] = 0f;
                        _floatsBuf[ptr + 2] = 0f;
                        _floatsBuf[ptr + 3] = 0.05f;
                        latestFoveaDepth16[px] = 0f;
                        latestFoveaNormalY16[px] = 0f;
                        latestFoveaSalience16[px] = 0.05f;
                    }
                    ptr += FOVEA_C;
                    px++;
                }
            }
            _foveaSampleSeq++;
        }

        private bool UnpackAndDispatchMotorCommand(byte[] pkt)
        {
            uint magic = BitConverter.ToUInt32(pkt, 0);
            if (magic != UNITY_MOTOR_MAGIC)
                return false;
            uint savedCrc = BitConverter.ToUInt32(pkt, 120);
            if (ComputeCrc32(pkt, 0, 120) != savedCrc)
                return false;

            latestMotorVerb = pkt[24];
            latestCommitGateOpen = pkt[25] != 0;
            latestVetoActive = pkt[26] != 0;
            latestHoldReasonId = pkt[27];

            float nx = BitConverter.ToSingle(pkt, 28);
            float ny = BitConverter.ToSingle(pkt, 32);
            float nz = BitConverter.ToSingle(pkt, 36);
            latestAimYawRad = BitConverter.ToSingle(pkt, 40);
            latestAimPitchRad = BitConverter.ToSingle(pkt, 44);
            latestSpeedScale = Mathf.Clamp(BitConverter.ToSingle(pkt, 48), 0.2f, 1.0f);

            latestGazeYpd = new Vector3(
                BitConverter.ToSingle(pkt, 52),
                BitConverter.ToSingle(pkt, 56),
                BitConverter.ToSingle(pkt, 60)
            );

            uint tsMilli = BitConverter.ToUInt32(pkt, 124);
            if (tsMilli >= 250 && tsMilli <= 12000)
                requestedTimeScale = tsMilli / 1000f;

            int strLen = 0;
            while (strLen < 32 && pkt[88 + strLen] != 0) strLen++;
            latestInnerSpeech = strLen > 0 ? Encoding.UTF8.GetString(pkt, 88, strLen) : "";

            // Denormalize [-1..1] move target into world space (decoupled from weapon/camera aim!)
            Vector3 mid = 0.5f * (localWorldMin + localWorldMax);
            Vector3 half = 0.5f * (localWorldMax - localWorldMin);
            Vector3 localTarget = new Vector3(mid.x + nx * half.x, mid.y + ny * half.y, mid.z + nz * half.z);
            latestWorldMoveTarget = arenaOrigin + localTarget;

            // Orient active eye/camera frustum smoothly toward independent gaze/aim
            float yawDeg = latestGazeYpd.x * Mathf.Rad2Deg;
            float pitchDeg = -latestGazeYpd.y * Mathf.Rad2Deg;
            eyeTransform.rotation = Quaternion.Euler(Mathf.Clamp(pitchDeg, -55f, 55f), yawDeg, 0f);

            OnLocomotionCommand?.Invoke(latestWorldMoveTarget, latestSpeedScale, latestMotorVerb, latestCommitGateOpen);
            OnAimAndGazeCommand?.Invoke(latestAimYawRad, latestAimPitchRad, latestGazeYpd);
            return true;
        }

        private void OnDestroy() => Disconnect();
    }
}
