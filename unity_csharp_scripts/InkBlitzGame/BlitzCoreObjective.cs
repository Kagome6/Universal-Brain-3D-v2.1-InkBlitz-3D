// BlitzCoreObjective.cs — Interactive BlitzCore (ブリッツコア) Shield, Pickup, Carrier & Drop mechanics
using UnityEngine;
using UnityEngine.Rendering;
using UniversalBrain3D;

namespace InkBlitz3D
{
    public enum BlitzCoreState
    {
        Shielded = 0,
        PickupReady = 1,
        Carried = 2
    }

    [RequireComponent(typeof(UniversalSemanticEntityTag))]
    public class BlitzCoreObjective : MonoBehaviour
    {
        [Header("References & State")]
        public UniversalSemanticEntityTag entityTag;
        public Transform coreVisual;
        public Transform shieldVisual;
        public float shieldRadiusM = 1.25f;

        public BlitzCoreState State { get; private set; } = BlitzCoreState.Shielded;
        public float MaxShieldTug { get; private set; } = 180f;
        public float ShieldTug { get; private set; } = 0f;
        public float ShieldHp => Mathf.Max(0f, MaxShieldTug - Mathf.Abs(ShieldTug));
        public InkBlitzPlayerAgent Carrier { get; private set; }
        public float CarrierHoldTimer { get; private set; } = 0f;

        private Vector3 _arenaOrigin;
        private InkBlitzArenaSettings _settings;
        private Vector3 _dropGroundPos;
        private Vector3 _lastCarrierGroundPos;
        private Collider[] _arenaWallAndCoverColliders;
        private float _idleTimer;
        private float _bobPhase;
        private float _sinceLastShieldHit;
        private float _bestTugRatioA;
        private float _bestTugRatioB;
        private bool _pickupRewardGrantedThisPop;
        private bool _enemyPickupPenaltyGrantedThisPop;
        private bool _isDroppedFromCarrier;
        private bool _fuseActive;
        private float _fuseTimer;
        private int _fusePopTeam;
        private InkBlitzPlayerAgent _fuseAttacker;
        private Material _runtimeShieldMat;

        public bool IsShieldActive => State == BlitzCoreState.Shielded;
        public bool IsDetonationFuseActive => State == BlitzCoreState.Shielded && _fuseActive;
        public float FuseTimeRemaining => _fuseActive ? Mathf.Max(0f, _fuseTimer) : 0f;
        public int FusePopTeam => _fuseActive ? _fusePopTeam : 0;
        public float EffectiveShieldHitRadiusM
        {
            get
            {
                if (_fuseActive)
                {
                    float maxFuse = (_settings != null && _settings.blitzCoreShieldPopFuseSec > 0.01f) ? _settings.blitzCoreShieldPopFuseSec : 0.45f;
                    float fuseProg = 1f - Mathf.Clamp01(_fuseTimer / maxFuse);
                    return Mathf.Lerp(shieldRadiusM * 1.18f, shieldRadiusM * 1.46f, fuseProg);
                }
                float maxTug = Mathf.Max(1f, MaxShieldTug);
                float tugRatio = Mathf.Clamp01(Mathf.Abs(ShieldTug) / maxTug);
                float visualRadius = 0.5f * Mathf.Lerp(1.55f, shieldRadiusM * 2.35f, tugRatio);
                return Mathf.Max(shieldRadiusM, visualRadius);
            }
        }

        public void Initialize(Vector3 arenaOrigin, InkBlitzArenaSettings settings)
        {
            _arenaOrigin = arenaOrigin;
            _settings = settings ?? new InkBlitzArenaSettings();
            if (entityTag == null)
                entityTag = GetComponent<UniversalSemanticEntityTag>();
            CacheArenaWallAndCoverColliders();
            EnsureVisualChildren();
            ResetToCenter();
        }

        private void CacheArenaWallAndCoverColliders()
        {
            Transform arenaRoot = transform.parent != null ? transform.parent.parent : null;
            Transform wallsRoot = arenaRoot != null ? arenaRoot.Find("WallsAndCovers") : null;
            _arenaWallAndCoverColliders = wallsRoot != null
                ? wallsRoot.GetComponentsInChildren<Collider>(true)
                : null;
        }

        private Vector3 ResolveSafeWalkableGroundPos(float localX, float localZ, InkBlitzArenaSettings cfg)
        {
            float maxX = cfg.arenaWidthX * 0.5f - 1.15f;
            float maxZ = cfg.arenaLengthZ * 0.5f - 1.15f;
            float lx = Mathf.Clamp(localX, -maxX, maxX);
            float lz = Mathf.Clamp(localZ, -maxZ, maxZ);

            if (_arenaWallAndCoverColliders == null || _arenaWallAndCoverColliders.Length == 0)
                CacheArenaWallAndCoverColliders();

            if (_arenaWallAndCoverColliders != null && _arenaWallAndCoverColliders.Length > 0)
            {
                const float minClearanceM = 0.82f;
                for (int pass = 0; pass < 2; pass++)
                {
                    for (int i = 0; i < _arenaWallAndCoverColliders.Length; i++)
                    {
                        Collider col = _arenaWallAndCoverColliders[i];
                        if (col == null || !col.enabled || !col.gameObject.activeInHierarchy)
                            continue;

                        Bounds b = col.bounds;
                        float wx = _arenaOrigin.x + lx;
                        float wz = _arenaOrigin.z + lz;
                        if (Mathf.Abs(wx - b.center.x) > b.extents.x + minClearanceM + 0.15f ||
                            Mathf.Abs(wz - b.center.z) > b.extents.z + minClearanceM + 0.15f)
                            continue;

                        Vector3 testPt = new Vector3(wx, b.center.y, wz);
                        Vector3 cp = col.ClosestPoint(testPt);
                        Vector2 diffXZ = new Vector2(wx - cp.x, wz - cp.z);
                        float d = diffXZ.magnitude;
                        if (d > 1e-4f)
                        {
                            if (d < minClearanceM)
                            {
                                Vector2 push = (diffXZ / d) * (minClearanceM - d);
                                lx += push.x;
                                lz += push.y;
                            }
                        }
                        else if (col is BoxCollider box)
                        {
                            Vector3 localPt = col.transform.InverseTransformPoint(testPt) - box.center;
                            Vector3 halfSize = box.size * 0.5f;
                            Vector3 lossy = col.transform.lossyScale;
                            float sx = Mathf.Max(0.05f, Mathf.Abs(lossy.x));
                            float sz = Mathf.Max(0.05f, Mathf.Abs(lossy.z));
                            float remX = (halfSize.x - Mathf.Abs(localPt.x)) * sx;
                            float remZ = (halfSize.z - Mathf.Abs(localPt.z)) * sz;
                            Vector3 pushLocal = Vector3.zero;
                            if (remX <= remZ)
                            {
                                float signX = localPt.x >= 0f ? 1f : -1f;
                                pushLocal.x = signX * (halfSize.x + minClearanceM / sx) - localPt.x;
                            }
                            else
                            {
                                float signZ = localPt.z >= 0f ? 1f : -1f;
                                pushLocal.z = signZ * (halfSize.z + minClearanceM / sz) - localPt.z;
                            }
                            Vector3 pushedWorld = col.transform.TransformPoint(box.center + localPt + pushLocal);
                            lx = pushedWorld.x - _arenaOrigin.x;
                            lz = pushedWorld.z - _arenaOrigin.z;
                        }
                    }
                    lx = Mathf.Clamp(lx, -maxX, maxX);
                    lz = Mathf.Clamp(lz, -maxZ, maxZ);
                }
            }

            float groundY = Mathf.Clamp(ArenaStageBuilder.SampleTerrainHeightLocal(lx, lz, cfg), 0f, 2.5f);
            return _arenaOrigin + new Vector3(lx, groundY + 0.65f, lz);
        }

        [Header("Standalone Material Template")]
        public Material templateShieldMaterial;

        public void EnsureVisualChildren()
        {
            if (coreVisual == null)
            {
                Transform existingCore = transform.Find("CoreVisual");
                if (existingCore != null)
                {
                    coreVisual = existingCore;
                }
                else
                {
                    GameObject coreGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    coreGo.name = "CoreVisual";
                    coreGo.transform.SetParent(transform, false);
                    coreGo.transform.localScale = new Vector3(0.85f, 0.85f, 0.85f);
                    RemoveCollider(coreGo);
                    var mr = coreGo.GetComponent<MeshRenderer>();
                    if (Application.isPlaying)
                        mr.sharedMaterial = CreateSimpleColorMat(new Color(1.0f, 0.82f, 0.12f, 1f), transparent: false);
                    coreVisual = coreGo.transform;
                }
            }

            if (shieldVisual == null)
            {
                Transform existingShield = transform.Find("ShieldVisual");
                if (existingShield != null)
                {
                    shieldVisual = existingShield;
                }
                else
                {
                    GameObject shieldGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    shieldGo.name = "ShieldVisual";
                    shieldGo.transform.SetParent(transform, false);
                    shieldGo.transform.localScale = Vector3.one * (shieldRadiusM * 2f);
                    RemoveCollider(shieldGo);
                    var mr = shieldGo.GetComponent<MeshRenderer>();
                    if (Application.isPlaying)
                    {
                        _runtimeShieldMat = CreateSimpleColorMat(new Color(1.0f, 0.86f, 0.25f, 0.28f), transparent: true, templateShieldMaterial);
                        mr.sharedMaterial = _runtimeShieldMat;
                    }
                    shieldVisual = shieldGo.transform;
                }
            }

            if (Application.isPlaying && shieldVisual != null && _runtimeShieldMat == null)
            {
                var smr = shieldVisual.GetComponent<MeshRenderer>();
                if (smr != null)
                {
                    _runtimeShieldMat = CreateSimpleColorMat(new Color(1.0f, 0.86f, 0.25f, 0.28f), transparent: true, templateShieldMaterial);
                    smr.sharedMaterial = _runtimeShieldMat;
                }
            }
        }

        public void ResetToCenter()
        {
            ClearCarrierState();
            var cfg = _settings ?? new InkBlitzArenaSettings();
            _dropGroundPos = ResolveSafeWalkableGroundPos(0f, 0f, cfg);
            _lastCarrierGroundPos = _dropGroundPos;
            transform.position = _dropGroundPos;
            State = BlitzCoreState.Shielded;
            MaxShieldTug = cfg.blitzCoreShieldMaxHp;
            ShieldTug = 0f;
            _sinceLastShieldHit = 0f;
            _bestTugRatioA = 0f;
            _bestTugRatioB = 0f;
            _pickupRewardGrantedThisPop = false;
            _enemyPickupPenaltyGrantedThisPop = false;
            _isDroppedFromCarrier = false;
            _fuseActive = false;
            _fuseTimer = 0f;
            _fusePopTeam = 0;
            _fuseAttacker = null;
            _idleTimer = 0f;
            CarrierHoldTimer = 0f;
            if (entityTag != null)
            {
                entityTag.enabled = true;
                entityTag.teamId = 0;
                entityTag.isImminentAreaHazard = false;
                entityTag.isStealthSubmerged = false;
                entityTag.boundingRadius = shieldRadiusM;
                entityTag.NotifyTeleported();
            }
            SyncVisuals();
        }

        public void ApplyShieldDamage(InkBlitzPlayerAgent attacker, float damage, InkBlitzArenaInstance arena, bool immediatePop = false)
        {
            if (State != BlitzCoreState.Shielded || damage <= 0f)
                return;
            if (_fuseActive)
            {
                if (immediatePop)
                    ExecuteShieldDetonation(_fusePopTeam, _fuseAttacker ?? attacker, arena);
                return;
            }

            int atkSign = (attacker != null && attacker.teamId < 0) ? -1 : 1;
            ShieldTug = Mathf.Clamp(ShieldTug + atkSign * damage, -MaxShieldTug, MaxShieldTug);
            _sinceLastShieldHit = 0f;

            // Unhackable high-water-mark shield damage shaping reward (bounded to +0.8 per pop cycle)
            if (attacker != null && attacker.brainBridge != null && MaxShieldTug > 1f)
            {
                if (atkSign > 0)
                {
                    float rA = Mathf.Clamp01(ShieldTug / MaxShieldTug);
                    if (rA > _bestTugRatioA)
                    {
                        attacker.brainBridge.AddStepReward(0.80f * (rA - _bestTugRatioA));
                        _bestTugRatioA = rA;
                    }
                }
                else
                {
                    float rB = Mathf.Clamp01(-ShieldTug / MaxShieldTug);
                    if (rB > _bestTugRatioB)
                    {
                        attacker.brainBridge.AddStepReward(0.80f * (rB - _bestTugRatioB));
                        _bestTugRatioB = rB;
                    }
                }
            }

            if (Mathf.Abs(ShieldTug) < MaxShieldTug - 1e-4f)
            {
                SyncVisuals();
                return;
            }

            int popTeam = (ShieldTug >= 0f) ? 1 : -1;
            var cfg = (arena != null && arena.settings != null) ? arena.settings : (_settings ?? new InkBlitzArenaSettings());
            if (immediatePop || cfg.blitzCoreShieldPopFuseSec <= 0.005f)
            {
                ExecuteShieldDetonation(popTeam, attacker, arena);
                return;
            }

            // Start swelling detonation fuse (0.45s warning window so losing side can evade via M4!)
            _fuseActive = true;
            _fuseTimer = cfg.blitzCoreShieldPopFuseSec;
            _fusePopTeam = popTeam;
            _fuseAttacker = attacker;
            if (entityTag != null)
            {
                entityTag.teamId = popTeam;
                entityTag.isImminentAreaHazard = true;
                entityTag.boundingRadius = cfg.blitzCoreShieldPopLethalRadiusM;
            }
            arena?.audioSystem?.EmitSound(transform.position, popTeam, 0.88f, 0.45f);
            SyncVisuals();
        }

        private void ExecuteShieldDetonation(int popTeam, InkBlitzPlayerAgent attacker, InkBlitzArenaInstance arena)
        {
            _fuseActive = false;
            _fuseTimer = 0f;
            _fusePopTeam = 0;
            _fuseAttacker = null;
            ShieldTug = 0f;
            State = BlitzCoreState.PickupReady;
            _idleTimer = 0f;
            if (entityTag != null)
            {
                entityTag.teamId = 0;
                entityTag.isImminentAreaHazard = false;
                entityTag.boundingRadius = 0.48f;
            }

            if (arena != null)
            {
                var cfg = arena.settings ?? new InkBlitzArenaSettings();
                bool attackerOnPopTeam = attacker != null && ((attacker.teamId > 0) == (popTeam > 0));
                arena.TurfPaint(transform.position, cfg.blitzCoreShieldPopRadiusM, popTeam, attackerOnPopTeam ? attacker : null);
                if (attackerOnPopTeam && attacker.brainBridge != null)
                    attacker.brainBridge.AddStepReward(cfg.shieldPopReward);

                for (int i = 0; i < arena.agents.Count; i++)
                {
                    InkBlitzPlayerAgent ag = arena.agents[i];
                    if (ag == null || !ag.gameObject.activeSelf || ag.teamId == popTeam || ag.isRespawning) continue;
                    Vector3 targetCenter = ag.transform.position + Vector3.up * 0.70f;
                    float d = Vector3.Distance(transform.position, targetCenter);
                    if (d > cfg.blitzCoreShieldPopRadiusM) continue;
                    if (arena.projectileManager != null && arena.projectileManager.IsLineBlockedByEnvironment(transform.position + Vector3.up * 0.25f, targetCenter))
                        continue;
                    if (d <= cfg.blitzCoreShieldPopLethalRadiusM)
                        arena.OnProjectileHitAgent(attackerOnPopTeam ? attacker : null, ag, cfg.blitzCoreShieldPopDamage, transform.position, true);
                    else
                        arena.OnProjectileHitAgent(attackerOnPopTeam ? attacker : null, ag, cfg.blitzCoreShieldPopOuterDamage, transform.position, true);
                }

                arena.audioSystem?.EmitSound(transform.position, popTeam, 1.0f, 0.45f);
                arena.audioSystem?.PlayShieldPopSfx(transform.position);
            }

            SyncVisuals();
        }

        public void DropAt(Vector3 worldPos)
        {
            ClearCarrierState();
            var cfg = _settings ?? new InkBlitzArenaSettings();
            Vector3 loc = worldPos - _arenaOrigin;
            _dropGroundPos = ResolveSafeWalkableGroundPos(loc.x, loc.z, cfg);
            _lastCarrierGroundPos = _dropGroundPos;
            transform.position = _dropGroundPos;
            State = BlitzCoreState.Shielded;
            MaxShieldTug = cfg.blitzCoreShieldMaxHp * 0.65f;
            ShieldTug = 0f;
            _sinceLastShieldHit = 0f;
            _bestTugRatioA = 0f;
            _bestTugRatioB = 0f;
            _pickupRewardGrantedThisPop = false;
            _enemyPickupPenaltyGrantedThisPop = false;
            _fuseActive = false;
            _fuseTimer = 0f;
            _fusePopTeam = 0;
            _fuseAttacker = null;
            _isDroppedFromCarrier = (new Vector2(_dropGroundPos.x - _arenaOrigin.x, _dropGroundPos.z - _arenaOrigin.z).sqrMagnitude > 2.25f);
            _idleTimer = 0f;
            CarrierHoldTimer = 0f;
            if (entityTag != null)
            {
                entityTag.enabled = true;
                entityTag.teamId = 0;
                entityTag.isImminentAreaHazard = false;
                entityTag.isStealthSubmerged = false;
                entityTag.boundingRadius = shieldRadiusM;
                entityTag.NotifyTeleported();
            }
            SyncVisuals();
        }

        public void StepBlitzCore(float dt, InkBlitzArenaInstance arena)
        {
            if (arena == null) return;
            var cfg = arena.settings ?? new InkBlitzArenaSettings();
            _bobPhase += dt * 3.5f;

            if (State == BlitzCoreState.Shielded)
            {
                if (_fuseActive)
                {
                    _fuseTimer -= dt;
                    if (_fuseTimer <= 0f)
                        ExecuteShieldDetonation(_fusePopTeam, _fuseAttacker, arena);
                    else
                        SyncVisuals();
                    return;
                }

                _sinceLastShieldHit += dt;
                if (_isDroppedFromCarrier && _sinceLastShieldHit >= Mathf.Max(20f, cfg.blitzCoreIdleResetSec + 4f))
                {
                    ResetToCenter();
                    return;
                }
                if (_sinceLastShieldHit >= cfg.blitzCoreShieldDecayDelaySec && Mathf.Abs(ShieldTug) > 0f)
                    ShieldTug = Mathf.MoveTowards(ShieldTug, 0f, cfg.blitzCoreShieldDecayPerSec * dt);

                // Soft-lock prevention: allow close-range contact/interact to chip shield when out of ink (Brain players only)
                for (int i = 0; i < arena.agents.Count; i++)
                {
                    InkBlitzPlayerAgent ag = arena.agents[i];
                    if (ag == null || !ag.gameObject.activeSelf || ag.isRespawning || ag.IsScriptedSupportBot) continue;
                    bool closeInteract = ag.currentVerb == UniversalBrainBridge3D.VERB_COMMIT_INTERACT ||
                                         ag.currentVerb == UniversalBrainBridge3D.VERB_GRASP_PICK ||
                                         ag.currentVerb == UniversalBrainBridge3D.VERB_IMPULSE_PUSH;
                    if (closeInteract && Vector3.Distance(ag.transform.position, transform.position) <= 1.85f)
                    {
                        ApplyShieldDamage(ag, 25f * dt, arena);
                        if (State != BlitzCoreState.Shielded) break;
                    }
                }
            }
            else if (State == BlitzCoreState.PickupReady)
            {
                _idleTimer += dt;
                if (_idleTimer >= cfg.blitzCoreIdleResetSec)
                {
                    ResetToCenter();
                    return;
                }

                Vector3 p = _dropGroundPos;
                p.y += Mathf.Sin(_bobPhase) * 0.12f;
                transform.position = p;

                InkBlitzPlayerAgent claimer = null;
                float bestDist = cfg.blitzCorePickupRadiusM;
                for (int i = 0; i < arena.agents.Count; i++)
                {
                    InkBlitzPlayerAgent ag = arena.agents[i];
                    if (ag == null || !ag.gameObject.activeSelf || ag.isRespawning || ag.IsScriptedSupportBot) continue;
                    float d = Vector2.Distance(
                        new Vector2(ag.transform.position.x, ag.transform.position.z),
                        new Vector2(transform.position.x, transform.position.z)
                    );
                    if (d <= bestDist)
                    {
                        bestDist = d;
                        claimer = ag;
                    }
                }

                if (claimer != null)
                {
                    State = BlitzCoreState.Carried;
                    Carrier = claimer;
                    _lastCarrierGroundPos = claimer.transform.position;
                    CarrierHoldTimer = cfg.blitzCoreHoldLimitSec;
                    claimer.isCarryingObjective = true;
                    claimer.carrierHoldTimeRemaining = CarrierHoldTimer;
                    if (!_pickupRewardGrantedThisPop && claimer.brainBridge != null)
                    {
                        claimer.brainBridge.AddStepReward(cfg.claimObjectiveReward);
                        _pickupRewardGrantedThisPop = true;
                    }
                    if (!_enemyPickupPenaltyGrantedThisPop && arena != null)
                    {
                        arena.ApplyTeamStepReward(-claimer.teamId, cfg.enemyClaimObjectivePenalty);
                        _enemyPickupPenaltyGrantedThisPop = true;
                    }
                    if (claimer.entityTag != null)
                        claimer.entityTag.isCarryingPrimaryObjective = true;
                    if (entityTag != null)
                        entityTag.enabled = false;
                }
            }
            else if (State == BlitzCoreState.Carried)
            {
                if (Carrier == null || Carrier.isRespawning)
                {
                    DropAt(_lastCarrierGroundPos);
                    return;
                }

                _lastCarrierGroundPos = Carrier.transform.position;

                float forwardZ = (Carrier.teamId > 0)
                    ? (Carrier.transform.position.z - _arenaOrigin.z)
                    : (_arenaOrigin.z - Carrier.transform.position.z);
                float drainMult = (forwardZ < -2.0f) ? cfg.blitzCoreRetreatTimerMultiplier : 1.0f;
                CarrierHoldTimer -= dt * drainMult;
                Carrier.carrierHoldTimeRemaining = Mathf.Max(0f, CarrierHoldTimer);

                if (CarrierHoldTimer <= 0f)
                {
                    arena.OnBlitzCoreHoldTimeoutExplode(Carrier);
                    return;
                }

                transform.position = Carrier.transform.position + Vector3.up * (Carrier.isSubmergedInFriendlyInk ? 0.95f : 1.75f);
            }

            SyncVisuals();
        }

        private void ClearCarrierState()
        {
            if (Carrier != null)
            {
                Carrier.isCarryingObjective = false;
                Carrier.carrierHoldTimeRemaining = 0f;
                if (Carrier.entityTag != null)
                    Carrier.entityTag.isCarryingPrimaryObjective = false;
                Carrier = null;
            }
            CarrierHoldTimer = 0f;
        }

        private void SyncVisuals()
        {
            if (shieldVisual != null)
            {
                bool showShield = (State == BlitzCoreState.Shielded);
                shieldVisual.gameObject.SetActive(showShield);
                if (showShield)
                {
                    float maxTug = Mathf.Max(1f, MaxShieldTug);
                    float tugRatio = _fuseActive ? 1.0f : Mathf.Clamp01(Mathf.Abs(ShieldTug) / maxTug);
                    float maxFuse = (_settings != null && _settings.blitzCoreShieldPopFuseSec > 0.01f) ? _settings.blitzCoreShieldPopFuseSec : 0.45f;
                    float fuseProg = _fuseActive ? (1f - Mathf.Clamp01(_fuseTimer / maxFuse)) : 0f;
                    float dia = _fuseActive
                        ? Mathf.Lerp(shieldRadiusM * 2.35f, shieldRadiusM * 2.95f, fuseProg)
                        : Mathf.Lerp(1.55f, shieldRadiusM * 2.35f, tugRatio);
                    shieldVisual.localScale = new Vector3(dia, dia, dia);

                    if (Application.isPlaying && _runtimeShieldMat != null)
                    {
                        Color gold = new Color(1.0f, 0.86f, 0.25f, 0.28f);
                        Color leadCol = gold;
                        int domTeam = _fuseActive ? _fusePopTeam : (Mathf.Abs(ShieldTug) > 1e-3f ? (ShieldTug > 0f ? 1 : -1) : 0);
                        if (domTeam != 0 && _settings != null)
                            leadCol = (domTeam > 0) ? _settings.teamAColor : _settings.teamBColor;
                        Color tint = Color.Lerp(gold, leadCol, tugRatio * 0.85f);
                        if (_fuseActive)
                        {
                            float flash = 0.5f + 0.5f * Mathf.Sin(fuseProg * Mathf.PI * 6.0f);
                            tint = Color.Lerp(leadCol, Color.white, 0.35f + 0.35f * flash);
                            tint.a = Mathf.Lerp(0.46f, 0.64f, fuseProg);
                        }
                        else
                        {
                            tint.a = Mathf.Lerp(0.28f, 0.42f, tugRatio);
                        }
                        if (_runtimeShieldMat.HasProperty("_BaseColor")) _runtimeShieldMat.SetColor("_BaseColor", tint);
                        if (_runtimeShieldMat.HasProperty("_Color")) _runtimeShieldMat.SetColor("_Color", tint);
                    }
                }
            }

            if (coreVisual != null)
            {
                float s = (State == BlitzCoreState.Carried) ? 0.58f : 0.85f;
                coreVisual.localScale = new Vector3(s, s, s);
                if (State != BlitzCoreState.Carried)
                    coreVisual.localPosition = Vector3.zero;
            }
        }

        private void LateUpdate()
        {
            if (State == BlitzCoreState.Carried && Carrier != null && !Carrier.isRespawning && coreVisual != null)
            {
                coreVisual.position = Carrier.VisualWorldPosition + Vector3.up * (Carrier.isSubmergedInFriendlyInk ? 0.95f : 1.75f);
            }
        }

        private void OnDestroy()
        {
            if (_runtimeShieldMat != null)
            {
                if (Application.isPlaying) Destroy(_runtimeShieldMat);
                else DestroyImmediate(_runtimeShieldMat);
                _runtimeShieldMat = null;
            }
        }

        private static void RemoveCollider(GameObject go)
        {
            Collider col = go.GetComponent<Collider>();
            if (col != null)
            {
                if (Application.isPlaying) Destroy(col);
                else DestroyImmediate(col);
            }
        }

        private static Material CreateSimpleColorMat(Color col, bool transparent, Material templateMat = null)
        {
            Material mat;
            if (transparent && templateMat != null)
            {
                mat = new Material(templateMat);
                if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", col);
                if (mat.HasProperty("_Color")) mat.SetColor("_Color", col);
                return mat;
            }
            Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            mat = new Material(sh);
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", col);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", col);
            if (transparent)
            {
                if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);
                if (mat.HasProperty("_Blend")) mat.SetFloat("_Blend", 0f);
                if (mat.HasProperty("_SrcBlend")) mat.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
                if (mat.HasProperty("_DstBlend")) mat.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
                if (mat.HasProperty("_ZWrite")) mat.SetFloat("_ZWrite", 0f);
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.renderQueue = 3000;
            }
            return mat;
        }
    }
}
