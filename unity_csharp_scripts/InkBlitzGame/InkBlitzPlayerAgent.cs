// InkBlitzPlayerAgent.cs — 6-DoF decoupled Strafe/Aim + Human/Dive physics, projectiles & mid-episode respawn
using UnityEngine;
using UnityEngine.Rendering;
using UniversalBrain3D;

namespace InkBlitz3D
{
    [RequireComponent(typeof(CharacterController))]
    [RequireComponent(typeof(UniversalSemanticEntityTag))]
    public class InkBlitzPlayerAgent : MonoBehaviour
    {
        [Header("Identity & Team")]
        public int arenaIndex = 0;
        public int teamId = 1; // +1 = Team A, -1 = Team B
        public bool isSupportTrooperBot = false;
        [HideInInspector] public bool isPromotedPeerBrain = false;
        public bool IsScriptedSupportBot => isSupportTrooperBot && !isPromotedPeerBrain;

        [Header("References")]
        public UniversalBrainBridge3D brainBridge;
        public UniversalSemanticEntityTag entityTag;
        public CharacterController characterController;
        public Transform eyePivot;
        public Transform humanVisualRoot;
        [UnityEngine.Serialization.FormerlySerializedAs("squidVisualRoot")]
        public Transform diveVisualRoot;
        public Transform weaponNozzlePivot;
        public Transform deflectShieldVisual;
        public Material templateDeflectShieldMaterial;

        [Header("Live Gameplay State (Read-Only)")]
        public float hp = 100f;
        public float inkTank01 = 1.0f;
        [UnityEngine.Serialization.FormerlySerializedAs("isSquidForm")]
        public bool isDiveForm = false;
        public bool isSubmergedInFriendlyInk = false;
        public float submergeStaminaSec = 4.0f;
        public float submergeExhaustTimer = 0f;
        public bool isRespawning = false;
        public bool isCarryingObjective = false;
        public float carrierHoldTimeRemaining = 0f;
        public float deflectTimeRemaining = 0f;
        public int currentVerb = UniversalBrainBridge3D.VERB_IDLE_OBSERVE;
        public float lastFloorValence = 0f;

        private InkBlitzArenaInstance _arena;
        private InkBlitzArenaSettings _settings;
        private TurfGridSystem _turfGrid;
        private Vector3 _arenaOrigin;
        private Vector3 _spawnWorldPos;

        private Vector3 _commandedMoveTarget;
        private float _commandedSpeedScale = 0.85f;
        private float _commandedAimYawRad = 0f;
        private float _commandedAimPitchRad = -0.05f;
        private bool _commandedCommitOpen = true;
        private bool _hasCommand = false;

        private Vector3 _prevFixedPos;
        private Vector3 _currFixedPos;

        private float _vertVel = 0f;
        private float _fireCooldown = 0f;
        private float _hpRegenDelayTimer = 0f;
        private float _respawnTimer = 0f;
        private float _swimAudioTimer = 0f;
        private float _diveSfxCooldown = 0f;
        private float _parryFlashTimer = 0f;
        private int _parryRewardsRemainingThisBuff = 0;
        private float _eyeLocalHeight = 1.18f;
        private float _smoothedEyeOffsetY = 1.18f;
        private Vector3 _velocity = Vector3.zero;
        private int _pendingNewlyExploredCells = 0;
        private uint _shotSpreadCounter = 0;

        public Vector3 Velocity => _velocity;
        public Vector3 VisualWorldPosition { get; private set; }
        public float InkTank01 => inkTank01;
        public float SubmergeStamina01 => (_settings != null && _settings.maxSubmergeDurationSec > 0.05f)
            ? Mathf.Clamp01(submergeStaminaSec / _settings.maxSubmergeDurationSec)
            : 1f;
        public float RespawnTimeRemaining => _respawnTimer;
        public float HpRegenDelayRemaining => _hpRegenDelayTimer;
        public bool HasDeflectBuff => deflectTimeRemaining > 0f && !isRespawning && !isCarryingObjective;
        public bool IsDeflectActive => deflectTimeRemaining > 0f && !isRespawning && !isCarryingObjective && !isDiveForm;
        public Vector3 CurrentAimDirection
        {
            get
            {
                float aimYawDeg = _commandedAimYawRad * Mathf.Rad2Deg;
                float aimPitchDeg = Mathf.Clamp(-_commandedAimPitchRad * Mathf.Rad2Deg, -50f, 50f);
                return Quaternion.Euler(aimPitchDeg, aimYawDeg, 0f) * Vector3.forward;
            }
        }
        public bool CanCommitNow
        {
            get
            {
                if (_settings == null || isRespawning || _fireCooldown > 0f) return false;
                float cost = isCarryingObjective ? _settings.blitzCoreInkCost : _settings.inkCostPerShot;
                return inkTank01 >= cost;
            }
        }
        public float EffectiveMaxHp => (_settings == null)
            ? 100f
            : Mathf.Max(1f, IsScriptedSupportBot ? _settings.supportBotMaxHp : _settings.maxHp);

        public float HazardLevel01
        {
            get
            {
                float maxHp = EffectiveMaxHp;
                float hpTerm = (1f - Mathf.Clamp01(hp / maxHp)) * 0.80f;
                float enemyInkTerm = (lastFloorValence < -0.25f) ? 0.32f : 0f;
                float holdTerm = (isCarryingObjective && carrierHoldTimeRemaining > 0f && carrierHoldTimeRemaining < 15f)
                    ? Mathf.Clamp01((15f - carrierHoldTimeRemaining) / 15f) * 0.70f
                    : 0f;
                return Mathf.Clamp01(hpTerm + enemyInkTerm + holdTerm);
            }
        }
        public float SurfacePosture01
        {
            get
            {
                if (isSubmergedInFriendlyInk)
                {
                    float stam = SubmergeStamina01;
                    return (stam >= 0.20f) ? (0.90f + 0.08f * stam) : (0.66f + 1.15f * stam);
                }
                if (lastFloorValence > 0.25f)
                    return (submergeExhaustTimer > 0f) ? 0.58f : 0.72f;
                if (lastFloorValence < -0.25f) return 0.10f;
                return 0.40f;
            }
        }
        public float ResourceOrCarrier01 => isCarryingObjective ? 0.95f : Mathf.Clamp(inkTank01 * 0.88f, 0f, 0.88f);

        public void Initialize(Vector3 arenaOrigin, InkBlitzArenaSettings settings, TurfGridSystem turfGrid, InkBlitzArenaInstance arena = null)
        {
            _arenaOrigin = arenaOrigin;
            _settings = settings ?? new InkBlitzArenaSettings();
            _turfGrid = turfGrid;
            _arena = arena;

            Vector3 spawnLocal = isSupportTrooperBot
                ? ((teamId > 0) ? _settings.teamASupportSpawnLocal : _settings.teamBSupportSpawnLocal)
                : ((teamId > 0) ? _settings.teamASpawnLocal : _settings.teamBSpawnLocal);
            float spawnGroundY = ArenaStageBuilder.SampleTerrainHeightLocal(spawnLocal.x, spawnLocal.z, _settings);
            _spawnWorldPos = _arenaOrigin + new Vector3(spawnLocal.x, spawnGroundY, spawnLocal.z);

            if (characterController == null) characterController = GetComponent<CharacterController>();
            if (entityTag == null) entityTag = GetComponent<UniversalSemanticEntityTag>();
            if (brainBridge == null) brainBridge = GetComponent<UniversalBrainBridge3D>();
            EnsureDeflectShieldVisual();

            ResetForNewMatch();

            if (brainBridge != null)
            {
                brainBridge.arenaIndex = arenaIndex;
                brainBridge.stageId = Mathf.Max(0, _settings.stageId);
                brainBridge.teamId = teamId;
                brainBridge.arenaOrigin = arenaOrigin;
                brainBridge.localWorldMin = new Vector3(-_settings.arenaWidthX * 0.5f, 0f, -_settings.arenaLengthZ * 0.5f);
                brainBridge.localWorldMax = new Vector3(_settings.arenaWidthX * 0.5f, _settings.arenaHeightY, _settings.arenaLengthZ * 0.5f);
                if (eyePivot != null)
                    brainBridge.eyeTransform = eyePivot;
                brainBridge.SetFloorSurfaceSampler(_turfGrid);

                brainBridge.OnLocomotionCommand -= HandleLocomotionCommand;
                brainBridge.OnLocomotionCommand += HandleLocomotionCommand;
                brainBridge.OnAimAndGazeCommand -= HandleAimAndGazeCommand;
                brainBridge.OnAimAndGazeCommand += HandleAimAndGazeCommand;
            }
        }

        private void OnDestroy()
        {
            if (brainBridge != null)
            {
                brainBridge.OnLocomotionCommand -= HandleLocomotionCommand;
                brainBridge.OnAimAndGazeCommand -= HandleAimAndGazeCommand;
            }
        }

        public void ResetForNewMatch()
        {
            if (_settings == null) _settings = new InkBlitzArenaSettings();
            Vector3 spawnLocal = isSupportTrooperBot
                ? ((teamId > 0) ? _settings.teamASupportSpawnLocal : _settings.teamBSupportSpawnLocal)
                : ((teamId > 0) ? _settings.teamASpawnLocal : _settings.teamBSpawnLocal);
            float spawnGroundY = ArenaStageBuilder.SampleTerrainHeightLocal(spawnLocal.x, spawnLocal.z, _settings);
            _spawnWorldPos = _arenaOrigin + new Vector3(spawnLocal.x, spawnGroundY, spawnLocal.z);

            if (characterController != null)
            {
                characterController.enabled = false;
                transform.position = _spawnWorldPos;
                characterController.enabled = true;
            }
            else
            {
                transform.position = _spawnWorldPos;
            }

            hp = EffectiveMaxHp;
            inkTank01 = 1.0f;
            isDiveForm = false;
            isSubmergedInFriendlyInk = false;
            submergeStaminaSec = (_settings.maxSubmergeDurationSec > 0.05f) ? _settings.maxSubmergeDurationSec : 4.0f;
            submergeExhaustTimer = 0f;
            isRespawning = false;
            isCarryingObjective = false;
            carrierHoldTimeRemaining = 0f;
            deflectTimeRemaining = 0f;
            _parryFlashTimer = 0f;
            _parryRewardsRemainingThisBuff = 0;
            currentVerb = UniversalBrainBridge3D.VERB_IDLE_OBSERVE;
            _fireCooldown = 0f;
            _hpRegenDelayTimer = 0f;
            _respawnTimer = 0f;
            _swimAudioTimer = 0f;
            _diveSfxCooldown = 0f;
            _vertVel = 0f;
            _eyeLocalHeight = 1.18f;
            _smoothedEyeOffsetY = 1.18f;
            _velocity = Vector3.zero;
            _pendingNewlyExploredCells = 0;
            _commandedMoveTarget = _spawnWorldPos;
            _commandedSpeedScale = 0.85f;
            _commandedAimYawRad = Mathf.Atan2(-spawnLocal.x, -spawnLocal.z);
            _commandedAimPitchRad = -0.05f;
            _commandedCommitOpen = true;
            _hasCommand = false;
            SnapVisualPositionToTransform();

            if (entityTag != null)
            {
                entityTag.enabled = true;
                entityTag.isStealthSubmerged = false;
                entityTag.isCarryingPrimaryObjective = false;
                entityTag.NotifyTeleported();
            }
            if (brainBridge != null)
            {
                brainBridge.isCurrentlyRespawning = false;
                brainBridge.activeBuffOrTransit01 = 0f;
            }
            if (deflectShieldVisual != null)
                deflectShieldVisual.gameObject.SetActive(false);
            if (humanVisualRoot != null)
                humanVisualRoot.gameObject.SetActive(true);
            if (diveVisualRoot != null)
                diveVisualRoot.gameObject.SetActive(false);
        }

        public void SyncFixedInterpolationPose()
        {
            _prevFixedPos = transform.position;
            _currFixedPos = transform.position;
            VisualWorldPosition = transform.position;
        }

        private void SnapVisualPositionToTransform()
        {
            _prevFixedPos = transform.position;
            _currFixedPos = transform.position;
            VisualWorldPosition = transform.position;
            _smoothedEyeOffsetY = transform.position.y + _eyeLocalHeight;
            if (humanVisualRoot != null)
                humanVisualRoot.localPosition = Vector3.zero;
            if (diveVisualRoot != null)
                diveVisualRoot.localPosition = Vector3.zero;
            if (eyePivot != null)
                eyePivot.position = new Vector3(transform.position.x, _smoothedEyeOffsetY, transform.position.z);
        }

        private void HandleLocomotionCommand(Vector3 worldMoveTarget, float speedScale, int motorVerb, bool commitOpen)
        {
            _commandedMoveTarget = worldMoveTarget;
            _commandedSpeedScale = Mathf.Clamp(speedScale, 0.2f, 1.0f);
            currentVerb = motorVerb;
            _commandedCommitOpen = commitOpen;
            _hasCommand = true;
        }

        private void HandleAimAndGazeCommand(float aimYawRad, float aimPitchRad, Vector3 gazeYpd)
        {
            _commandedAimYawRad = aimYawRad;
            _commandedAimPitchRad = aimPitchRad;
            _hasCommand = true;
        }

        public void ApplyExternalCommand(Vector3 worldMoveTarget, float speedScale, int motorVerb, float aimYawRad, float aimPitchRad, bool commitOpen = true)
        {
            _commandedMoveTarget = worldMoveTarget;
            _commandedSpeedScale = Mathf.Clamp(speedScale, 0.2f, 1.0f);
            currentVerb = motorVerb;
            _commandedAimYawRad = aimYawRad;
            _commandedAimPitchRad = aimPitchRad;
            _commandedCommitOpen = commitOpen;
            _hasCommand = true;
        }

        public void AddExploreCells(int count)
        {
            if (count > 0)
                _pendingNewlyExploredCells += count;
        }

        public int ConsumeNewlyExploredCells()
        {
            int c = _pendingNewlyExploredCells;
            _pendingNewlyExploredCells = 0;
            return c;
        }

        public bool TakeDamage(float amount)
        {
            if (isRespawning || amount <= 0f)
                return false;
            hp = Mathf.Max(0f, hp - amount);
            float delaySec = (_settings != null) ? Mathf.Max(0f, _settings.damageHpRegenDelaySec) : 1.0f;
            _hpRegenDelayTimer = Mathf.Max(_hpRegenDelayTimer, delaySec);
            return hp <= 0f;
        }

        public void ActivateDeflectBuff(float durationSec)
        {
            if (isRespawning || isCarryingObjective)
                return;
            EnsureDeflectShieldVisual();
            deflectTimeRemaining = Mathf.Max(deflectTimeRemaining, Mathf.Max(0.5f, durationSec));
            _parryRewardsRemainingThisBuff = 4;
            _parryFlashTimer = 0.18f;
            if (brainBridge != null)
            {
                float maxDur = (_settings != null && _settings.deflectBuffDurationSec > 0.1f) ? _settings.deflectBuffDurationSec : durationSec;
                brainBridge.activeBuffOrTransit01 = 0.75f * Mathf.Clamp01(deflectTimeRemaining / Mathf.Max(0.1f, maxDur));
                brainBridge.NotifyActionCommitted();
            }
        }

        public void OnSuccessfulProjectileParry(float reward)
        {
            _parryFlashTimer = 0.22f;
            if (_parryRewardsRemainingThisBuff > 0)
            {
                _parryRewardsRemainingThisBuff--;
                if (reward > 0f)
                    brainBridge?.AddStepReward(reward);
            }
            brainBridge?.NotifyActionCommitted();
        }

        public void EnsureDeflectShieldVisual()
        {
            if (weaponNozzlePivot == null)
                return;
            if (deflectShieldVisual == null)
            {
                Transform existing = weaponNozzlePivot.Find("DeflectShieldVisual");
                if (existing != null)
                {
                    deflectShieldVisual = existing;
                }
                else
                {
                    GameObject shieldGo = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    shieldGo.name = "DeflectShieldVisual";
                    shieldGo.transform.SetParent(weaponNozzlePivot, false);
                    shieldGo.transform.localPosition = new Vector3(-0.28f, 0.0f, 0.62f);
                    shieldGo.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
                    shieldGo.transform.localScale = new Vector3(1.42f, 0.035f, 1.18f);
                    Collider col = shieldGo.GetComponent<Collider>();
                    if (col != null)
                    {
                        if (Application.isPlaying) Destroy(col);
                        else DestroyImmediate(col);
                    }
                    var mr = shieldGo.GetComponent<MeshRenderer>();
                    if (templateDeflectShieldMaterial != null)
                    {
                        mr.sharedMaterial = templateDeflectShieldMaterial;
                    }
                    else if (Application.isPlaying)
                    {
                        Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                        var mat = new Material(sh);
                        Color colShield = new Color(0.18f, 1.0f, 0.84f, 0.36f);
                        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", colShield);
                        if (mat.HasProperty("_Color")) mat.SetColor("_Color", colShield);
                        if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);
                        if (mat.HasProperty("_SrcBlend")) mat.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
                        if (mat.HasProperty("_DstBlend")) mat.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
                        if (mat.HasProperty("_ZWrite")) mat.SetFloat("_ZWrite", 0f);
                        mat.SetOverrideTag("RenderType", "Transparent");
                        mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                        mat.renderQueue = (int)RenderQueue.Transparent;
                        mr.sharedMaterial = mat;
                    }
                    shieldGo.SetActive(false);
                    deflectShieldVisual = shieldGo.transform;
                }
            }
        }

        public void EnterRespawnState(float durationSec)
        {
            isRespawning = true;
            hp = 0f;
            _respawnTimer = Mathf.Max(0.2f, durationSec);
            _velocity = Vector3.zero;
            isDiveForm = false;
            isSubmergedInFriendlyInk = false;
            isCarryingObjective = false;
            carrierHoldTimeRemaining = 0f;
            deflectTimeRemaining = 0f;
            _parryFlashTimer = 0f;
            _parryRewardsRemainingThisBuff = 0;

            if (entityTag != null)
            {
                entityTag.isStealthSubmerged = false;
                entityTag.isCarryingPrimaryObjective = false;
                entityTag.enabled = false;
            }
            if (brainBridge != null)
            {
                brainBridge.isCurrentlyRespawning = true;
                brainBridge.activeBuffOrTransit01 = 0f;
            }
            if (deflectShieldVisual != null)
                deflectShieldVisual.gameObject.SetActive(false);
            if (humanVisualRoot != null)
                humanVisualRoot.gameObject.SetActive(false);
            if (diveVisualRoot != null)
                diveVisualRoot.gameObject.SetActive(false);

            if (characterController != null)
                characterController.enabled = false;
            transform.position = _spawnWorldPos + Vector3.up * 2.5f;
            entityTag?.NotifyTeleported();
            SnapVisualPositionToTransform();
        }

        private void CompleteRespawn()
        {
            if (characterController != null)
                characterController.enabled = false;
            transform.position = _spawnWorldPos;
            if (characterController != null)
                characterController.enabled = true;
            SnapVisualPositionToTransform();

            hp = EffectiveMaxHp;
            inkTank01 = 1.0f;
            submergeStaminaSec = (_settings != null && _settings.maxSubmergeDurationSec > 0.05f) ? _settings.maxSubmergeDurationSec : 4.0f;
            submergeExhaustTimer = 0f;
            _vertVel = 0f;
            _fireCooldown = 0f;
            _hpRegenDelayTimer = 0f;
            _respawnTimer = 0f;
            isRespawning = false;
            _velocity = Vector3.zero;

            if (entityTag != null)
            {
                entityTag.isStealthSubmerged = false;
                entityTag.isCarryingPrimaryObjective = false;
                entityTag.enabled = true;
                entityTag.NotifyTeleported();
            }

            if (_turfGrid != null)
                _turfGrid.PaintCircle(_spawnWorldPos, _settings.spawnPadPaintRadiusM, teamId, grantExploreCredit: false);

            if (humanVisualRoot != null)
                humanVisualRoot.gameObject.SetActive(true);
            if (diveVisualRoot != null)
                diveVisualRoot.gameObject.SetActive(false);

            if (brainBridge != null)
            {
                brainBridge.isCurrentlyRespawning = false;
                brainBridge.NotifyMidEpisodeRespawn();
            }
        }

        public void StepAgent(float dt)
        {
            if (_settings == null || _turfGrid == null || dt <= 0f)
                return;

            if (isRespawning)
            {
                _velocity = Vector3.zero;
                _respawnTimer -= dt;
                if (_respawnTimer <= 0f)
                    CompleteRespawn();
                return;
            }

            if (_fireCooldown > 0f)
                _fireCooldown = Mathf.Max(0f, _fireCooldown - dt);
            if (_hpRegenDelayTimer > 0f)
                _hpRegenDelayTimer = Mathf.Max(0f, _hpRegenDelayTimer - dt);
            if (_parryFlashTimer > 0f)
                _parryFlashTimer = Mathf.Max(0f, _parryFlashTimer - dt);

            if (isCarryingObjective && deflectTimeRemaining > 0f)
            {
                deflectTimeRemaining = 0f;
                _parryRewardsRemainingThisBuff = 0;
            }
            else if (deflectTimeRemaining > 0f)
            {
                deflectTimeRemaining = Mathf.Max(0f, deflectTimeRemaining - dt);
            }

            if (brainBridge != null)
            {
                float maxDur = Mathf.Max(0.1f, _settings.deflectBuffDurationSec);
                brainBridge.activeBuffOrTransit01 = HasDeflectBuff
                    ? (0.75f * Mathf.Clamp01(deflectTimeRemaining / maxDur))
                    : 0f;
            }

            lastFloorValence = _turfGrid.SampleSurfaceValence(transform.position, teamId);

            if (submergeExhaustTimer > 0f)
                submergeExhaustTimer = Mathf.Max(0f, submergeExhaustTimer - dt);

            bool wantFire = currentVerb == UniversalBrainBridge3D.VERB_RELEASE_PLACE ||
                            currentVerb == UniversalBrainBridge3D.VERB_IMPULSE_PUSH;

            bool wasSubmerged = isSubmergedInFriendlyInk;
            float minStamToDive = wasSubmerged ? 1e-5f : 0.25f;
            bool canSubmergeNow = (submergeExhaustTimer <= 0f) &&
                                  (_settings.maxSubmergeDurationSec <= 0.05f || submergeStaminaSec > minStamToDive);

            bool wantSwim = canSubmergeNow && !wantFire &&
                            (currentVerb == UniversalBrainBridge3D.VERB_EVADE_HAZARD ||
                             ((currentVerb == UniversalBrainBridge3D.VERB_IDLE_OBSERVE ||
                               currentVerb == UniversalBrainBridge3D.VERB_COMMIT_INTERACT) && lastFloorValence > 0.25f));

            if (_diveSfxCooldown > 0f)
                _diveSfxCooldown = Mathf.Max(0f, _diveSfxCooldown - dt);

            isDiveForm = wantSwim;
            isSubmergedInFriendlyInk = wantSwim && (lastFloorValence > 0.25f);
            if (isSubmergedInFriendlyInk && _settings.maxSubmergeDurationSec > 0.05f)
            {
                submergeStaminaSec = Mathf.Max(0f, submergeStaminaSec - dt);
                if (submergeStaminaSec <= 1e-4f)
                {
                    submergeStaminaSec = 0f;
                    isSubmergedInFriendlyInk = false;
                    isDiveForm = false;
                    submergeExhaustTimer = Mathf.Max(0.2f, _settings.submergeExhaustCooldownSec);
                }
            }
            else if (_settings.maxSubmergeDurationSec > 0.05f)
            {
                submergeStaminaSec = Mathf.Min(_settings.maxSubmergeDurationSec, submergeStaminaSec + _settings.submergeRegenPerSec * dt);
            }
            if (!wasSubmerged && isSubmergedInFriendlyInk && _diveSfxCooldown <= 0f)
            {
                _diveSfxCooldown = 0.26f;
                _arena?.audioSystem?.PlayInkDiveSfx(transform.position, isSupportTrooperBot);
            }

            if (entityTag != null)
            {
                entityTag.isStealthSubmerged = isSubmergedInFriendlyInk && !isCarryingObjective;
                entityTag.isCarryingPrimaryObjective = isCarryingObjective;
            }

            // Decoupled locomotion toward commandedWorldMoveTarget
            Vector3 horizVel = Vector3.zero;
            if (_hasCommand && currentVerb != UniversalBrainBridge3D.VERB_IDLE_OBSERVE)
            {
                Vector3 toTarget = _commandedMoveTarget - transform.position;
                toTarget.y = 0f;
                float dist = toTarget.magnitude;
                if (dist > 0.18f)
                {
                    float surfMult = isSubmergedInFriendlyInk
                        ? _settings.swimSpeedMultiplier
                        : (lastFloorValence < -0.25f ? _settings.enemyInkSpeedMultiplier : 1.0f);
                    float carrierMult = isCarryingObjective ? _settings.carrierMoveSpeedMultiplier : 1.0f;
                    float botMult = isSupportTrooperBot ? _settings.supportBotMoveSpeedMultiplier : 1.0f;
                    float speed = _settings.walkSpeedMps * _commandedSpeedScale * surfMult * carrierMult * botMult;
                    horizVel = (toTarget / dist) * speed;
                }
            }

            _prevFixedPos = transform.position;
            if (characterController != null && characterController.enabled)
            {
                Vector3 prevPos = transform.position;
                Vector3 prevLoc = prevPos - _arenaOrigin;
                float prevTerrainY = ArenaStageBuilder.SampleTerrainHeightLocal(prevLoc.x, prevLoc.z, _settings);
                bool wasGrounded = characterController.isGrounded || (prevLoc.y <= prevTerrainY + 0.08f && _vertVel <= 0.01f);
                float horizSpeed = new Vector2(horizVel.x, horizVel.z).magnitude;

                if (wasGrounded && _vertVel <= 0f)
                    _vertVel = -Mathf.Max(2.5f, horizSpeed * 0.95f);
                else
                    _vertVel -= _settings.gravityMps2 * dt;

                Vector3 moveVec = new Vector3(horizVel.x, _vertVel, horizVel.z);
                characterController.Move(moveVec * dt);

                Vector3 loc = transform.position - _arenaOrigin;
                float maxX = _settings.arenaWidthX * 0.5f - 0.6f;
                float maxZ = _settings.arenaLengthZ * 0.5f - 0.6f;
                float clampedX = Mathf.Clamp(loc.x, -maxX, maxX);
                float clampedZ = Mathf.Clamp(loc.z, -maxZ, maxZ);

                // Physical BlitzCore Barrier Pushback: prevent agents from clipping inside a shielded/swelling BlitzCore!
                if (_arena != null && _arena.blitzCore != null && _arena.blitzCore.IsShieldActive)
                {
                    Vector3 rmLoc = _arena.blitzCore.transform.position - _arenaOrigin;
                    if (Mathf.Abs(loc.y - rmLoc.y) < 1.65f)
                    {
                        float minShieldR = _arena.blitzCore.EffectiveShieldHitRadiusM + 0.22f;
                        Vector2 diffRm = new Vector2(clampedX - rmLoc.x, clampedZ - rmLoc.z);
                        float dRm = diffRm.magnitude;
                        if (dRm < minShieldR)
                        {
                            Vector2 outDir = dRm > 1e-4f
                                ? (diffRm / dRm)
                                : new Vector2(0f, (teamId > 0) ? -1f : 1f);
                            clampedX = Mathf.Clamp(rmLoc.x + outDir.x * minShieldR, -maxX, maxX);
                            clampedZ = Mathf.Clamp(rmLoc.z + outDir.y * minShieldR, -maxZ, maxZ);
                        }
                    }
                }

                float curTerrainY = ArenaStageBuilder.SampleTerrainHeightLocal(clampedX, clampedZ, _settings);

                float targetY = loc.y;
                if (wasGrounded && _vertVel <= 0f && loc.y <= curTerrainY + 0.28f)
                    targetY = curTerrainY;
                else
                    targetY = Mathf.Max(curTerrainY, loc.y);

                Vector3 clampedLoc = new Vector3(clampedX, targetY, clampedZ);
                if ((clampedLoc - loc).sqrMagnitude > 1e-7f)
                {
                    characterController.enabled = false;
                    transform.position = _arenaOrigin + clampedLoc;
                    characterController.enabled = true;
                }
                _velocity = (transform.position - prevPos) / dt;
                _velocity.y = Mathf.Clamp(_velocity.y, -8f, 8f);
                if (wasGrounded && Mathf.Abs(transform.position.y - (_arenaOrigin.y + curTerrainY)) <= 0.03f)
                    _vertVel = -1.5f;
            }
            else
            {
                _velocity = horizVel;
            }
            _currFixedPos = transform.position;

            // Fast swim splash acoustic cue (silent when stationary or slow < 2.6 m/s)
            if (isSubmergedInFriendlyInk && _velocity.magnitude >= 2.6f)
            {
                _swimAudioTimer -= dt;
                if (_swimAudioTimer <= 0f)
                {
                    _swimAudioTimer = 0.20f;
                    _arena?.audioSystem?.EmitSound(transform.position, teamId, 0.52f, 0.25f);
                }
            }
            else
            {
                _swimAudioTimer = 0f;
            }

            // Independent aim orientation (body/weapon yaw & pitch decoupled from velocity)
            float aimYawDeg = _commandedAimYawRad * Mathf.Rad2Deg;
            float aimPitchDeg = Mathf.Clamp(-_commandedAimPitchRad * Mathf.Rad2Deg, -50f, 50f);
            Quaternion aimRot = Quaternion.Euler(aimPitchDeg, aimYawDeg, 0f);
            Quaternion bodyRot = Quaternion.Euler(0f, aimYawDeg, 0f);
            if (humanVisualRoot != null)
                humanVisualRoot.rotation = bodyRot;
            if (diveVisualRoot != null)
                diveVisualRoot.rotation = bodyRot;
            if (weaponNozzlePivot != null)
                weaponNozzlePivot.rotation = aimRot;

            if (eyePivot != null)
            {
                float targetEyeY = isSubmergedInFriendlyInk ? 0.42f : 1.18f;
                _eyeLocalHeight = Mathf.Lerp(_eyeLocalHeight, targetEyeY, Mathf.Clamp01(dt * 14f));
                float rawEyeWorldY = transform.position.y + _eyeLocalHeight;
                if (Mathf.Abs(rawEyeWorldY - _smoothedEyeOffsetY) > 1.2f)
                    _smoothedEyeOffsetY = rawEyeWorldY;
                else
                    _smoothedEyeOffsetY = Mathf.Clamp(
                        Mathf.Lerp(_smoothedEyeOffsetY, rawEyeWorldY, Mathf.Clamp01(dt * 24f)),
                        rawEyeWorldY - 0.14f,
                        rawEyeWorldY + 0.14f);
                eyePivot.position = new Vector3(transform.position.x, _smoothedEyeOffsetY, transform.position.z);
            }
            if (humanVisualRoot != null)
                humanVisualRoot.gameObject.SetActive(!isDiveForm);
            if (diveVisualRoot != null)
                diveVisualRoot.gameObject.SetActive(isDiveForm);
            if (deflectShieldVisual != null)
            {
                bool showShield = IsDeflectActive;
                deflectShieldVisual.gameObject.SetActive(showShield);
                if (showShield)
                {
                    float flashScale = 1f + (_parryFlashTimer > 0f ? 0.22f * (_parryFlashTimer / 0.22f) : 0f);
                    deflectShieldVisual.localScale = new Vector3(1.42f * flashScale, 0.035f, 1.18f * flashScale);
                }
            }

            // Pooled 3D ballistic ink shooting (gated by Basal Ganglia 60Hz commit_gate_open!)
            float fireInterval = isSupportTrooperBot
                ? _settings.supportBotFireIntervalSec
                : (isCarryingObjective ? _settings.blitzCoreFireIntervalSec : _settings.fireIntervalSec);
            float inkCost = isCarryingObjective ? _settings.blitzCoreInkCost : _settings.inkCostPerShot;
            bool actuallyFiring = wantFire && _commandedCommitOpen;
            if (actuallyFiring && _fireCooldown <= 0f && inkTank01 >= inkCost)
                FireInkProjectile(aimRot * Vector3.forward, fireInterval, inkCost, isCarryingObjective);

            // Ink & HP recovery / enemy-ink slip damage
            if (isSubmergedInFriendlyInk)
                inkTank01 = Mathf.Min(1f, inkTank01 + _settings.swimInkRegenPerSec * dt);
            else if (!actuallyFiring)
                inkTank01 = Mathf.Min(1f, inkTank01 + _settings.humanInkRegenPerSec * dt);

            float maxHpCap = EffectiveMaxHp;
            if (lastFloorValence < -0.25f)
            {
                float prevHp = hp;
                hp = Mathf.Max(Mathf.Min(maxHpCap, _settings.enemyInkMinHpFloor), hp - _settings.enemyInkSlipDps * dt);
                if (hp < prevHp - 1e-5f)
                    _hpRegenDelayTimer = Mathf.Max(_hpRegenDelayTimer, Mathf.Min(0.35f, _settings.damageHpRegenDelaySec));
            }
            else if (_hpRegenDelayTimer <= 0f)
            {
                if (isSubmergedInFriendlyInk)
                    hp = Mathf.Min(maxHpCap, hp + _settings.swimHpRegenPerSec * dt);
                else
                    hp = Mathf.Min(maxHpCap, hp + _settings.humanHpRegenPerSec * dt);
            }
        }

        private void LateUpdate()
        {
            float fdt = Mathf.Max(1e-4f, Time.fixedDeltaTime);
            float alpha = Mathf.Clamp01((Time.time - Time.fixedTime) / fdt);
            RefreshVisualInterpolation(alpha);
        }

        public void RefreshVisualInterpolation(float alpha)
        {
            if (isRespawning || (_arena != null && !_arena.IsRenderingEnabled))
            {
                VisualWorldPosition = transform.position;
                return;
            }

            alpha = Mathf.Clamp01(alpha);
            if ((_currFixedPos - _prevFixedPos).sqrMagnitude > 9.0f)
                VisualWorldPosition = _currFixedPos;
            else
                VisualWorldPosition = Vector3.Lerp(_prevFixedPos, _currFixedPos, alpha);

            if (humanVisualRoot != null)
                humanVisualRoot.position = VisualWorldPosition;
            if (diveVisualRoot != null)
                diveVisualRoot.position = VisualWorldPosition;
            if (eyePivot != null)
                eyePivot.position = new Vector3(VisualWorldPosition.x, _smoothedEyeOffsetY, VisualWorldPosition.z);
        }

        private void FireInkProjectile(Vector3 aimDir, float interval, float inkCost, bool isBlitzCoreBlast)
        {
            inkTank01 = Mathf.Max(0f, inkTank01 - inkCost);
            _fireCooldown = interval;
            brainBridge?.NotifyActionCommitted();

            var footRes = _turfGrid.PaintCircle(transform.position, _settings.footPaintRadiusM, teamId);
            _pendingNewlyExploredCells += footRes.newlyExploredCells;

            Vector3 launchDir = aimDir;
            if (!isBlitzCoreBlast && !isSupportTrooperBot && _settings != null && _settings.shotSpreadDeg > 0.01f)
            {
                _shotSpreadCounter++;
                float goldenAng = _shotSpreadCounter * 2.3999632f;
                float radial = (0.35f + 0.65f * ((_shotSpreadCounter * 0.6180339f) % 1.0f)) * _settings.shotSpreadDeg;
                float yawJit = Mathf.Cos(goldenAng) * radial;
                float pitchJit = Mathf.Sin(goldenAng) * radial * 0.55f;
                launchDir = (Quaternion.LookRotation(aimDir.sqrMagnitude > 1e-6f ? aimDir.normalized : Vector3.forward)
                    * Quaternion.Euler(pitchJit, yawJit, 0f)) * Vector3.forward;
            }

            Vector3 eyeOrigin = eyePivot != null ? eyePivot.position : transform.position + Vector3.up * 1.1f;
            if (_arena != null && _arena.projectileManager != null)
                _arena.projectileManager.SpawnProjectile(this, eyeOrigin, launchDir, isBlitzCoreBlast);
        }
    }
}
