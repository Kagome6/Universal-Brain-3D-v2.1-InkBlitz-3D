// SupportTrooperBot.cs — Wingman-style Non-Carrier Support Wingman (3-Mode State Machine)
// Guardrails:
//   1. Never picks up BlitzCore or Sky-Bridge Deflect Orbs (100% Brain AI agency).
//   2. Strictly 1st-person LOS + blind to ink-submerged stealth opponents.
//   3. 0.30s visual aim telegraph before 3-shot burst + pause (fair counterplay & 3D parry fodder).
//   4. Zero-Oscillation Stabilization:
//      - Slew-rate-limited smooth yaw/pitch tracking (eliminates 1-frame snap vibration),
//      - 0.65s short-term target lock persistence + FOV hysteresis,
//      - Sticky obstacle-bypass side memory (`_avoidSideSign`) + stuck-progress watchdog,
//      - Clean station arrival deadzone facing the enemy approach.
using UnityEngine;
using UniversalBrain3D;

namespace InkBlitz3D
{
    [RequireComponent(typeof(InkBlitzPlayerAgent))]
    public class SupportTrooperBot : MonoBehaviour
    {
        public InkBlitzPlayerAgent agent;
        public bool enableTrooperAi = true;
        public float telegraphLockSec = 0.30f;
        public float sightRangeM = 13.5f;
        public float maxYawTurnDegPerSec = 300f;

        private int _patrolIdx = 0;
        private int _patrolStepDir = 1;
        private float _weavePhase = 0f;
        private float _targetLockTimer = 0f;
        private int _burstShotsFired = 0;
        private float _burstPauseTimer = 0f;
        private float _paintShotTimer = 0f;
        private bool _refillingTank = false;
        private bool _prevCanCommit = true;

        // Stabilization & Anti-Oscillation State
        private float _smoothedYawDeg = 0f;
        private float _yawVelDegPerSec = 0f;
        private float _smoothedPitchRad = -0.10f;
        private bool _yawInitialized = false;
        private InkBlitzPlayerAgent _rememberedEnemy = null;
        private Vector3 _rememberedAimPoint = Vector3.zero;
        private float _enemyMemoryTimer = 0f;
        private int _avoidSideSign = 0;
        private float _avoidSideTimer = 0f;
        private Vector3 _committedAvoidDir = Vector3.zero;
        private float _avoidCommitHoldTimer = 0f;
        private Vector3 _stuckCheckAnchor = Vector3.zero;
        private float _stuckCheckTimer = 0f;

        public float CurrentSmoothedYawDeg => _smoothedYawDeg;

        private void Awake()
        {
            if (agent == null)
                agent = GetComponent<InkBlitzPlayerAgent>();
        }

        public void ResetTrooperState()
        {
            _patrolIdx = 0;
            _patrolStepDir = 1;
            _weavePhase = (agent != null && agent.teamId < 0) ? 1.57f : 0f;
            _targetLockTimer = 0f;
            _burstShotsFired = 0;
            _burstPauseTimer = 0f;
            _paintShotTimer = 0.35f;
            _refillingTank = false;
            _prevCanCommit = true;

            _smoothedYawDeg = (agent != null && agent.teamId < 0) ? 180f : 0f;
            _yawVelDegPerSec = 0f;
            _smoothedPitchRad = -0.10f;
            _yawInitialized = true;
            _rememberedEnemy = null;
            _enemyMemoryTimer = 0f;
            _avoidSideSign = 0;
            _avoidSideTimer = 0f;
            _committedAvoidDir = Vector3.zero;
            _avoidCommitHoldTimer = 0f;
            _stuckCheckAnchor = agent != null ? agent.transform.position : Vector3.zero;
            _stuckCheckTimer = 0f;
        }

        public void StepTrooper(float dt, InkBlitzArenaInstance arena)
        {
            if (!enableTrooperAi || agent == null || !agent.gameObject.activeSelf || arena == null || dt <= 0f)
                return;

            if (!_yawInitialized)
                ResetTrooperState();

            if (agent.isRespawning)
            {
                _targetLockTimer = 0f;
                _burstShotsFired = 0;
                _burstPauseTimer = 0f;
                _rememberedEnemy = null;
                _enemyMemoryTimer = 0f;
                _avoidSideSign = 0;
                _avoidSideTimer = 0f;
                _committedAvoidDir = Vector3.zero;
                _avoidCommitHoldTimer = 0f;
                _yawVelDegPerSec = 0f;
                _smoothedYawDeg = (agent.teamId < 0) ? 180f : 0f;
                return;
            }

            var cfg = arena.settings ?? new InkBlitzArenaSettings();
            _weavePhase += dt * 1.6f;
            if (_burstPauseTimer > 0f)
                _burstPauseTimer = Mathf.Max(0f, _burstPauseTimer - dt);
            if (_paintShotTimer > 0f)
                _paintShotTimer = Mathf.Max(0f, _paintShotTimer - dt);
            if (_avoidCommitHoldTimer > 0f)
                _avoidCommitHoldTimer = Mathf.Max(0f, _avoidCommitHoldTimer - dt);
            if (_avoidSideTimer > 0f)
            {
                _avoidSideTimer -= dt;
                if (_avoidSideTimer <= 0f)
                {
                    _avoidSideTimer = 0f;
                    _avoidSideSign = 0;
                }
            }

            // 1. Honest Ink Tank Refill in Friendly Ink (strictly non-fallthrough so yaw never chatters!)
            if (agent.InkTank01 < 0.18f)
                _refillingTank = true;
            else if (agent.InkTank01 >= 0.72f)
                _refillingTank = false;

            if (_refillingTank)
            {
                _targetLockTimer = 0f;
                float holdYawRad = _smoothedYawDeg * Mathf.Deg2Rad;
                if (agent.lastFloorValence > 0.25f)
                {
                    SmoothUpdateAim(dt, _smoothedYawDeg, -0.08f, out float outYawR, out float outPitchR);
                    agent.ApplyExternalCommand(agent.transform.position, 0.35f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, outYawR, outPitchR, commitOpen: false);
                    return;
                }
                if (agent.CanCommitNow)
                {
                    SmoothUpdateAim(dt, _smoothedYawDeg, -0.68f, out float outYawR, out float outPitchR);
                    agent.ApplyExternalCommand(agent.transform.position, 0.35f, UniversalBrainBridge3D.VERB_RELEASE_PLACE, outYawR, outPitchR, commitOpen: true);
                    return;
                }
                agent.ApplyExternalCommand(agent.transform.position, 0.35f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, holdYawRad, -0.35f, commitOpen: false);
                return;
            }

            // 2. Locate Ally Brain Carrier / Enemy Carrier & Honest 1st-Person Visible Opponent (with 0.65s short-term persistence!)
            InkBlitzPlayerAgent allyCarrier = null;
            InkBlitzPlayerAgent enemyCarrier = null;
            if (arena.blitzCore != null && arena.blitzCore.State == BlitzCoreState.Carried && arena.blitzCore.Carrier != null)
            {
                if (arena.blitzCore.Carrier.teamId == agent.teamId)
                    allyCarrier = arena.blitzCore.Carrier;
                else
                    enemyCarrier = arena.blitzCore.Carrier;
            }

            InkBlitzPlayerAgent liveVisibleEnemy = FindVisibleOpponent(arena);
            if (liveVisibleEnemy != null)
            {
                _rememberedEnemy = liveVisibleEnemy;
                _rememberedAimPoint = ComputeLeadAimPoint(agent, liveVisibleEnemy, cfg.supportBotProjectileSpeedMps, cfg.projectileGravityMps2);
                _enemyMemoryTimer = 0.65f;
            }
            else if (_enemyMemoryTimer > 0f)
            {
                _enemyMemoryTimer = Mathf.Max(0f, _enemyMemoryTimer - dt);
                if (_rememberedEnemy == null || _rememberedEnemy.isRespawning || !_rememberedEnemy.gameObject.activeSelf)
                {
                    _rememberedEnemy = null;
                    _enemyMemoryTimer = 0f;
                }
            }

            // 3. Determine Tactical Station from 3-Mode State Machine (180-deg Symmetric via `s`)
            float s = (agent.teamId > 0) ? 1f : -1f;
            Vector3 arenaOrigin = arena.transform.position;
            Vector3 rawMoveTarget;
            float moveSpeedScale = 0.88f;
            bool isPatrolling = false;

            if (allyCarrier != null)
            {
                // Mode A: ESCORT VANGUARD — lead 3.4m ahead-and-aside of friendly BlitzCore carrier toward enemy goal!
                Vector3 atkGoalWorld = arenaOrigin + ((agent.teamId > 0) ? cfg.teamBPedestalLocal : cfg.teamAPedestalLocal);
                Vector3 toGoal = atkGoalWorld - allyCarrier.transform.position;
                toGoal.y = 0f;
                Vector3 fwd = toGoal.sqrMagnitude > 1e-4f ? toGoal.normalized : new Vector3(0f, 0f, s);
                Vector3 side = new Vector3(fwd.z, 0f, -fwd.x);
                rawMoveTarget = allyCarrier.transform.position + fwd * 3.4f + side * (1.15f * s);
                moveSpeedScale = 0.96f;
            }
            else if (enemyCarrier != null)
            {
                // Mode B: CITADEL GATEKEEPER — hold open plaza before home Citadel ramp (`(4.6s, 0, -6.4s)`)
                Vector3 gateSpot = arenaOrigin + new Vector3(4.6f * s, 0f, -6.4f * s);
                rawMoveTarget = gateSpot;
                moveSpeedScale = 0.92f;
            }
            else
            {
                // Mode C: FRONTLINE PATROL & RETREAT-TURF PAINTER — 3 wide-open symmetric waypoints
                isPatrolling = true;
                Vector3 wp = GetPatrolWaypointWorld(arenaOrigin, s, _patrolIdx);
                Vector2 flatDelta = new Vector2(wp.x - agent.transform.position.x, wp.z - agent.transform.position.z);
                if (flatDelta.magnitude <= 1.55f)
                {
                    AdvancePatrolWaypoint();
                    wp = GetPatrolWaypointWorld(arenaOrigin, s, _patrolIdx);
                }
                rawMoveTarget = wp;
                moveSpeedScale = 0.85f;
            }

            Vector3 clampedTarget = ClampInsideArena(arenaOrigin, rawMoveTarget, cfg);
            Vector2 stationDelta = new Vector2(clampedTarget.x - agent.transform.position.x, clampedTarget.z - agent.transform.position.z);
            float distToStation = stationDelta.magnitude;

            // Stuck-progress watchdog while moving (> 0.9m from target): unstick cleanly if blocked for > 1.0s
            if (distToStation > 0.90f)
            {
                _stuckCheckTimer += dt;
                if (_stuckCheckTimer >= 1.05f)
                {
                    float moved = Vector2.Distance(
                        new Vector2(agent.transform.position.x, agent.transform.position.z),
                        new Vector2(_stuckCheckAnchor.x, _stuckCheckAnchor.z));
                    _stuckCheckAnchor = agent.transform.position;
                    _stuckCheckTimer = 0f;
                    if (moved < 0.35f)
                    {
                        _avoidSideSign = (_avoidSideSign == 0) ? 1 : -_avoidSideSign;
                        _avoidSideTimer = 1.10f;
                        if (isPatrolling)
                            AdvancePatrolWaypoint();
                    }
                }
            }
            else
            {
                _stuckCheckAnchor = agent.transform.position;
                _stuckCheckTimer = 0f;
            }

            Vector3 moveTarget = SteerAroundLocalObstacleSticky(agent.transform.position, clampedTarget);

            // 4. Combat Aim (Smooth Slew + 0.30s Telegraph -> 3-Shot Burst -> Pause) OR Forward Turf Maintenance
            bool canCommit = agent.CanCommitNow;
            if (liveVisibleEnemy != null || _enemyMemoryTimer > 0f)
            {
                Vector3 aimPoint = (liveVisibleEnemy != null) ? _rememberedAimPoint : _rememberedAimPoint;
                Vector3 enemyPos = (liveVisibleEnemy != null) ? liveVisibleEnemy.transform.position : aimPoint;
                Vector3 toVis = enemyPos - agent.transform.position;
                toVis.y = 0f;
                float dVis = Mathf.Max(0.1f, toVis.magnitude);
                Vector3 dirVis = toVis / dVis;
                Vector3 perp = new Vector3(-dirVis.z, 0f, dirVis.x) * (Mathf.Sin(_weavePhase * 1.2f) * 1.25f);

                if (dVis < 4.5f)
                    moveTarget = ClampInsideArena(arenaOrigin, agent.transform.position - dirVis * 1.8f + perp, cfg);
                else
                    moveTarget = ClampInsideArena(arenaOrigin, Vector3.Lerp(moveTarget, agent.transform.position + perp, 0.55f), cfg);

                ComputeDesiredAimDegAndPitch(agent, aimPoint, out float desYawDeg, out float desPitchRad);
                float yawErrDeg = SmoothUpdateAim(dt, desYawDeg, desPitchRad, out float outYawRad, out float outPitchRad);

                if (liveVisibleEnemy != null)
                    _targetLockTimer += dt;
                else
                    _targetLockTimer = Mathf.Max(0f, _targetLockTimer - dt * 1.5f);

                // Only fire when enemy is genuinely visible right now, telegraph elapsed, AND gun has rotated within 22 deg!
                bool lockReady = (liveVisibleEnemy != null)
                              && _targetLockTimer >= telegraphLockSec
                              && _burstPauseTimer <= 0f
                              && dVis <= sightRangeM
                              && yawErrDeg <= 22f;

                if (_prevCanCommit && !canCommit && lockReady)
                {
                    _burstShotsFired++;
                    if (_burstShotsFired >= 3)
                    {
                        _burstShotsFired = 0;
                        _burstPauseTimer = 0.85f;
                    }
                }
                _prevCanCommit = canCommit;

                bool fireNow = lockReady && canCommit;
                int verb = fireNow ? UniversalBrainBridge3D.VERB_RELEASE_PLACE : UniversalBrainBridge3D.VERB_APPROACH_HOVER;
                agent.ApplyExternalCommand(moveTarget, moveSpeedScale, verb, outYawRad, outPitchRad, commitOpen: fireNow);
                return;
            }

            // No visible or remembered enemy: decay telegraph lock & patrol/hold smoothly
            _targetLockTimer = Mathf.Max(0f, _targetLockTimer - dt * 2.5f);
            _burstShotsFired = 0;
            _prevCanCommit = canCommit;

            Vector3 moveDir = moveTarget - agent.transform.position;
            moveDir.y = 0f;
            float moveDist = moveDir.magnitude;

            float desiredTravelYawDeg;
            if (moveDist > 0.65f)
            {
                desiredTravelYawDeg = Mathf.Atan2(moveDir.x, moveDir.z) * Mathf.Rad2Deg;
            }
            else
            {
                // Station arrival deadzone: hold position cleanly & face forward toward enemy approach!
                moveTarget = agent.transform.position;
                moveSpeedScale = 0.25f;
                if (enemyCarrier != null)
                {
                    Vector3 toCar = enemyCarrier.transform.position - agent.transform.position;
                    desiredTravelYawDeg = (toCar.sqrMagnitude > 0.05f)
                        ? (Mathf.Atan2(toCar.x, toCar.z) * Mathf.Rad2Deg)
                        : ((agent.teamId > 0) ? 0f : 180f);
                }
                else
                {
                    desiredTravelYawDeg = (agent.teamId > 0) ? 0f : 180f;
                }
            }

            Vector3 fwdDir = Quaternion.Euler(0f, _smoothedYawDeg, 0f) * Vector3.forward;
            Vector3 probeAhead = agent.transform.position + fwdDir * 2.4f;
            float aheadValence = arena.turfGrid != null ? arena.turfGrid.SampleSurfaceValence(probeAhead, agent.teamId) : 0f;
            bool shouldPaintLane = moveDist > 0.65f
                                && (aheadValence < 0.25f || agent.lastFloorValence < 0.25f)
                                && _paintShotTimer <= 0f
                                && canCommit;

            float targetPitch = shouldPaintLane ? -0.24f : -0.10f;
            SmoothUpdateAim(dt, desiredTravelYawDeg, targetPitch, out float smoothYawR, out float smoothPitchR);

            if (shouldPaintLane)
            {
                _paintShotTimer = 0.75f;
                agent.ApplyExternalCommand(moveTarget, moveSpeedScale, UniversalBrainBridge3D.VERB_RELEASE_PLACE, smoothYawR, smoothPitchR, commitOpen: true);
            }
            else
            {
                agent.ApplyExternalCommand(moveTarget, moveSpeedScale, UniversalBrainBridge3D.VERB_APPROACH_HOVER, smoothYawR, smoothPitchR, commitOpen: false);
            }
        }

        private void AdvancePatrolWaypoint()
        {
            _patrolIdx += _patrolStepDir;
            if (_patrolIdx >= 2) { _patrolIdx = 2; _patrolStepDir = -1; }
            else if (_patrolIdx <= 0) { _patrolIdx = 0; _patrolStepDir = 1; }
        }

        private float SmoothUpdateAim(float dt, float desiredYawDeg, float desiredPitchRad, out float outYawRad, out float outPitchRad)
        {
            // Critically-damped 2nd-order angular servo: velocity passes smoothly through 0 before any reversal (zero high-freq vibration!)
            _smoothedYawDeg = Mathf.SmoothDampAngle(
                _smoothedYawDeg,
                desiredYawDeg,
                ref _yawVelDegPerSec,
                0.14f,
                Mathf.Max(60f, maxYawTurnDegPerSec),
                dt);
            _smoothedPitchRad = Mathf.MoveTowards(_smoothedPitchRad, desiredPitchRad, 4.5f * dt);
            outYawRad = _smoothedYawDeg * Mathf.Deg2Rad;
            outPitchRad = _smoothedPitchRad;
            return Mathf.Abs(Mathf.DeltaAngle(_smoothedYawDeg, desiredYawDeg));
        }

        private static Vector3 GetPatrolWaypointWorld(Vector3 arenaOrigin, float s, int idx)
        {
            // Wide-open symmetric plaza waypoints >= 2.0m clear of all cover blocks and ramp side lips!
            switch (idx)
            {
                case 0: return arenaOrigin + new Vector3(-6.8f * s, 0f, -6.2f * s);
                case 1: return arenaOrigin + new Vector3(-0.4f * s, 0f, -6.2f * s);
                default: return arenaOrigin + new Vector3(4.8f * s, 0f, -5.2f * s);
            }
        }

        private static Vector3 ClampInsideArena(Vector3 arenaOrigin, Vector3 worldPt, InkBlitzArenaSettings cfg)
        {
            float maxX = cfg.arenaWidthX * 0.5f - 1.2f;
            float maxZ = cfg.arenaLengthZ * 0.5f - 1.2f;
            Vector3 loc = worldPt - arenaOrigin;
            loc.x = Mathf.Clamp(loc.x, -maxX, maxX);
            loc.z = Mathf.Clamp(loc.z, -maxZ, maxZ);
            return arenaOrigin + loc;
        }

        private InkBlitzPlayerAgent FindVisibleOpponent(InkBlitzArenaInstance arena)
        {
            Vector3 eyePos = agent.eyePivot != null ? agent.eyePivot.position : agent.transform.position + Vector3.up * 1.1f;
            Vector3 fwd = Quaternion.Euler(0f, _smoothedYawDeg, 0f) * Vector3.forward;

            InkBlitzPlayerAgent best = null;
            float bestScore = -1f;

            for (int i = 0; i < arena.agents.Count; i++)
            {
                InkBlitzPlayerAgent other = arena.agents[i];
                if (other == null || !other.gameObject.activeSelf || other.teamId == agent.teamId || other.isRespawning)
                    continue;
                // Strictly blind to ink-submerged stealth opponents!
                if (other.isSubmergedInFriendlyInk && !other.isCarryingObjective)
                    continue;

                Vector3 targetPt = other.transform.position + Vector3.up * 0.88f;
                Vector3 delta = targetPt - eyePos;
                float d = delta.magnitude;
                if (d > sightRangeM || d < 0.05f)
                    continue;

                Vector3 dir = delta / d;
                // Hysteresis FOV: wider retention (-0.55) for currently tracked target so turning never drops lock!
                float cosLimit = (other == _rememberedEnemy && _enemyMemoryTimer > 0f) ? -0.55f : -0.20f;
                if (!other.isCarryingObjective && Vector3.Dot(fwd, dir) < cosLimit && d > 3.0f)
                    continue;

                if (Physics.Raycast(eyePos, dir, out RaycastHit hit, d - 0.48f, ~0, QueryTriggerInteraction.Ignore))
                {
                    if (hit.collider.GetComponentInParent<InkBlitzPlayerAgent>() != other)
                        continue;
                }

                float roleWeight = other.isCarryingObjective ? 1.65f : (other.isSupportTrooperBot ? 0.80f : 1.20f);
                // Slight stickiness bonus (+12%) for current target prevents tie-breaking chatter between two equidistant enemies
                if (other == _rememberedEnemy && _enemyMemoryTimer > 0f)
                    roleWeight *= 1.12f;
                float score = roleWeight / (1.0f + 0.12f * d);
                if (score > bestScore)
                {
                    bestScore = score;
                    best = other;
                }
            }
            return best;
        }

        private Vector3 SteerAroundLocalObstacleSticky(Vector3 selfPos, Vector3 targetPos)
        {
            Vector3 flat = targetPos - selfPos;
            flat.y = 0f;
            float dist = flat.magnitude;
            if (dist < 0.75f)
                return targetPos;

            Vector3 dir = flat / dist;
            Vector3 origin = selfPos + Vector3.up * 0.85f;
            if (Physics.SphereCast(origin, 0.40f, dir, out RaycastHit hit, Mathf.Min(2.4f, dist), ~0, QueryTriggerInteraction.Ignore))
            {
                if (Mathf.Abs(hit.normal.y) < 0.35f
                    && hit.collider.GetComponentInParent<InkBlitzPlayerAgent>() == null
                    && hit.collider.GetComponentInParent<BlitzCoreObjective>() == null)
                {
                    Vector3 n = new Vector3(hit.normal.x, 0f, hit.normal.z).normalized;
                    Vector3 tangL = new Vector3(-n.z, 0f, n.x);

                    // Sticky bypass handedness: never flip +1 <-> -1 frame-to-frame when approaching head-on!
                    if (_avoidSideSign == 0 || _avoidSideTimer <= 0f)
                    {
                        float dotL = Vector3.Dot(tangL, dir);
                        _avoidSideSign = (dotL >= -0.05f) ? 1 : -1;
                        _avoidSideTimer = 0.85f;
                    }
                    else
                    {
                        _avoidSideTimer = Mathf.Max(_avoidSideTimer, 0.35f);
                    }

                    Vector3 tang = tangL * _avoidSideSign;
                    _committedAvoidDir = (tang * 0.85f + n * 0.35f).normalized;
                    _avoidCommitHoldTimer = 0.32f;
                    return selfPos + _committedAvoidDir * Mathf.Min(3.0f, dist);
                }
            }

            // Corner-clearance commitment: continue walking past the obstacle corner smoothly instead of snapping back on the grazing pixel!
            if (_avoidCommitHoldTimer > 0f && _committedAvoidDir.sqrMagnitude > 0.25f)
            {
                float blend = Mathf.Clamp01(_avoidCommitHoldTimer / 0.32f);
                Vector3 smoothDir = Vector3.Slerp(dir, _committedAvoidDir, blend).normalized;
                return selfPos + smoothDir * Mathf.Min(3.0f, dist);
            }
            return targetPos;
        }

        private static Vector3 ComputeLeadAimPoint(InkBlitzPlayerAgent shooter, InkBlitzPlayerAgent target, float projSpeed, float gravity)
        {
            Vector3 origin = shooter.eyePivot != null ? shooter.eyePivot.position : shooter.transform.position + Vector3.up * 1.1f;
            Vector3 targetCenter = target.transform.position + Vector3.up * 0.85f;
            float dist = Vector3.Distance(origin, targetCenter);
            float t = dist / Mathf.Max(4f, projSpeed);
            Vector3 predicted = targetCenter + target.Velocity * t * 0.85f;
            predicted.y += Mathf.Clamp(0.5f * gravity * t * t, 0f, 1.4f);
            return predicted;
        }

        private static void ComputeDesiredAimDegAndPitch(InkBlitzPlayerAgent shooter, Vector3 targetWorldPoint, out float yawDeg, out float pitchRad)
        {
            Vector3 origin = shooter.eyePivot != null ? shooter.eyePivot.position : shooter.transform.position + Vector3.up * 1.1f;
            Vector3 delta = targetWorldPoint - origin;
            yawDeg = Mathf.Atan2(delta.x, delta.z) * Mathf.Rad2Deg;
            float horiz = Mathf.Max(1e-4f, Mathf.Sqrt(delta.x * delta.x + delta.z * delta.z));
            pitchRad = Mathf.Clamp(Mathf.Atan2(delta.y, horiz), -0.8f, 0.8f);
        }
    }
}
