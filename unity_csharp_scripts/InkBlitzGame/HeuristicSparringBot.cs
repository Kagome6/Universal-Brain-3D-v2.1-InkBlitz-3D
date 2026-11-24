// HeuristicSparringBot.cs — Honest 1st-person sparring bot for 1v1 BlitzCore playtesting
using UnityEngine;
using UniversalBrain3D;

namespace InkBlitz3D
{
    [RequireComponent(typeof(InkBlitzPlayerAgent))]
    public class HeuristicSparringBot : MonoBehaviour
    {
        public InkBlitzPlayerAgent agent;
        public bool enableHeuristicBot = true;
        public bool yieldWhenBrainConnected = true;

        private float _weavePhase;
        private bool _refillingTank;
        private int _footShotBurstCount;

        private void Awake()
        {
            if (agent == null) agent = GetComponent<InkBlitzPlayerAgent>();
        }

        public void StepBot(float dt, InkBlitzArenaInstance arena)
        {
            if (!enableHeuristicBot || agent == null || arena == null || agent.isRespawning)
                return;
            if (yieldWhenBrainConnected && agent.brainBridge != null && agent.brainBridge.IsConnected)
                return;

            _weavePhase += dt * 2.2f;
            var cfg = arena.settings ?? new InkBlitzArenaSettings();

            // 1. Tank management
            if (!agent.isCarryingObjective)
            {
                if (agent.InkTank01 < 0.20f)
                    _refillingTank = true;
                else if (agent.InkTank01 >= 0.75f)
                    _refillingTank = false;

                if (_refillingTank)
                {
                    if (agent.lastFloorValence > 0.25f)
                    {
                        agent.ApplyExternalCommand(agent.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE,
                            (agent.teamId > 0) ? 0f : Mathf.PI, -0.05f);
                        return;
                    }
                    else if (_footShotBurstCount < 3 && agent.CanCommitNow)
                    {
                        _footShotBurstCount++;
                        agent.ApplyExternalCommand(agent.transform.position, 0.5f, UniversalBrainBridge3D.VERB_RELEASE_PLACE,
                            (agent.teamId > 0) ? 0f : Mathf.PI, -0.75f);
                        return;
                    }
                }
                else
                {
                    _footShotBurstCount = 0;
                }
            }

            // 2. Locate visible opponent (no wallhacks!)
            InkBlitzPlayerAgent visibleEnemy = FindVisibleOpponent(arena, cfg);

            // 3. Carrier push toward Enemy Pedestal
            if (agent.isCarryingObjective)
            {
                Vector3 goalLocal = (agent.teamId > 0) ? cfg.teamBPedestalLocal : cfg.teamAPedestalLocal;
                Vector3 goalWorld = arena.transform.position + goalLocal;
                Vector3 toGoal = goalWorld - agent.transform.position;
                float distGoal = new Vector2(toGoal.x, toGoal.z).magnitude;

                Vector3 moveTarget = goalWorld;
                if (distGoal > 3.5f)
                    moveTarget += Vector3.right * (Mathf.Sin(_weavePhase) * 1.6f);
                moveTarget = SteerAroundLocalObstacle(agent.transform.position, moveTarget);

                Vector3 aimPos = (visibleEnemy != null && Vector3.Distance(agent.transform.position, visibleEnemy.transform.position) <= 9.5f)
                    ? ComputeLeadAimPoint(agent, visibleEnemy, cfg.blitzCoreProjectileSpeedMps, cfg.projectileGravityMps2)
                    : goalWorld + Vector3.up * 0.5f;

                ComputeAimAngles(agent, aimPos, out float yaw, out float pitch);
                bool shouldShoot = agent.CanCommitNow &&
                                   ((visibleEnemy != null && Vector3.Distance(agent.transform.position, visibleEnemy.transform.position) <= 9.5f) ||
                                    (agent.lastFloorValence <= 0.25f && distGoal > 2.2f));
                int verb = shouldShoot
                    ? UniversalBrainBridge3D.VERB_RELEASE_PLACE
                    : (agent.lastFloorValence > 0.25f ? UniversalBrainBridge3D.VERB_EVADE_HAZARD : UniversalBrainBridge3D.VERB_APPROACH_HOVER);
                agent.ApplyExternalCommand(moveTarget, 1.0f, verb, yaw, pitch);
                return;
            }

            // 4. Intercept Enemy BlitzCore Carrier
            BlitzCoreObjective rm = arena.blitzCore;
            if (rm != null && rm.State == BlitzCoreState.Carried && rm.Carrier != null && rm.Carrier.teamId != agent.teamId)
            {
                InkBlitzPlayerAgent carrier = rm.Carrier;
                Vector3 toCarrier = carrier.transform.position - agent.transform.position;
                float d = toCarrier.magnitude;
                Vector3 dir = d > 1e-3f ? toCarrier / d : Vector3.forward;
                Vector3 optimalSpot = carrier.transform.position - dir * 5.8f;

                Vector3 leadPt = ComputeLeadAimPoint(agent, carrier, cfg.projectileSpeedMps, cfg.projectileGravityMps2);
                ComputeAimAngles(agent, leadPt, out float yaw, out float pitch);
                int verb = (d <= cfg.shotRangeM && agent.CanCommitNow)
                    ? UniversalBrainBridge3D.VERB_RELEASE_PLACE
                    : (agent.lastFloorValence > 0.25f ? UniversalBrainBridge3D.VERB_EVADE_HAZARD : UniversalBrainBridge3D.VERB_APPROACH_HOVER);
                agent.ApplyExternalCommand(optimalSpot, 1.0f, verb, yaw, pitch);
                return;
            }

            // 5. Contest Central / Dropped BlitzCore (or engage nearby enemy)
            float maxEngageR = Mathf.Max(5.5f, cfg.shotRangeM);
            if (visibleEnemy != null && Vector3.Distance(agent.transform.position, visibleEnemy.transform.position) <= maxEngageR)
            {
                float d = Vector3.Distance(agent.transform.position, visibleEnemy.transform.position);
                Vector3 dir = (visibleEnemy.transform.position - agent.transform.position).normalized;
                Vector3 strafe = Vector3.Cross(Vector3.up, dir) * Mathf.Sin(_weavePhase * 1.4f) * 2.2f;
                Vector3 moveTarget = visibleEnemy.transform.position - dir * 4.8f + strafe;

                Vector3 leadPt = ComputeLeadAimPoint(agent, visibleEnemy, cfg.projectileSpeedMps, cfg.projectileGravityMps2);
                ComputeAimAngles(agent, leadPt, out float yaw, out float pitch);
                int verb = agent.CanCommitNow ? UniversalBrainBridge3D.VERB_RELEASE_PLACE : UniversalBrainBridge3D.VERB_APPROACH_HOVER;
                agent.ApplyExternalCommand(moveTarget, 0.95f, verb, yaw, pitch);
                return;
            }

            if (rm != null)
            {
                Vector3 rmPos = rm.transform.position;
                float dRm = Vector3.Distance(agent.transform.position, rmPos);
                if (rm.State == BlitzCoreState.PickupReady)
                {
                    ComputeAimAngles(agent, rmPos, out float yaw, out float pitch);
                    int verb = (agent.lastFloorValence > 0.25f)
                        ? UniversalBrainBridge3D.VERB_EVADE_HAZARD
                        : UniversalBrainBridge3D.VERB_APPROACH_HOVER;
                    agent.ApplyExternalCommand(SteerAroundLocalObstacle(agent.transform.position, rmPos), 1.0f, verb, yaw, pitch);
                    return;
                }
                else
                {
                    Vector3 dir = (rmPos - agent.transform.position).normalized;
                    float safeStandoff = (rm.IsDetonationFuseActive && rm.FusePopTeam != agent.teamId) ? 4.8f : 4.6f;
                    Vector3 standOff = rmPos - dir * safeStandoff + Vector3.Cross(Vector3.up, dir) * Mathf.Sin(_weavePhase) * 1.2f;
                    ComputeAimAngles(agent, rmPos + Vector3.up * 0.15f, out float yaw, out float pitch);
                    int verb = (!rm.IsDetonationFuseActive && dRm <= maxEngageR && agent.CanCommitNow)
                        ? UniversalBrainBridge3D.VERB_RELEASE_PLACE
                        : (agent.lastFloorValence > 0.25f ? UniversalBrainBridge3D.VERB_EVADE_HAZARD : UniversalBrainBridge3D.VERB_APPROACH_HOVER);
                    Vector3 rawTgt = dRm <= 5.6f ? standOff : rmPos;
                    agent.ApplyExternalCommand(SteerAroundLocalObstacle(agent.transform.position, rawTgt), 0.95f, verb, yaw, pitch);
                    return;
                }
            }
        }

        private static Vector3 SteerAroundLocalObstacle(Vector3 selfPos, Vector3 targetPos)
        {
            Vector3 flat = targetPos - selfPos;
            flat.y = 0f;
            float dist = flat.magnitude;
            if (dist < 0.8f) return targetPos;
            Vector3 dir = flat / dist;
            Vector3 origin = selfPos + Vector3.up * 0.85f;
            if (Physics.SphereCast(origin, 0.38f, dir, out RaycastHit hit, Mathf.Min(2.4f, dist), ~0, QueryTriggerInteraction.Ignore))
            {
                if (Mathf.Abs(hit.normal.y) < 0.35f
                    && hit.collider.GetComponentInParent<InkBlitzPlayerAgent>() == null
                    && hit.collider.GetComponentInParent<BlitzCoreObjective>() == null)
                {
                    Vector3 n = new Vector3(hit.normal.x, 0f, hit.normal.z).normalized;
                    Vector3 tangL = new Vector3(-n.z, 0f, n.x);
                    Vector3 tang = (Vector3.Dot(tangL, dir) >= 0f) ? tangL : -tangL;
                    return selfPos + (tang * 0.82f + n * 0.35f).normalized * Mathf.Min(3.2f, dist);
                }
            }
            return targetPos;
        }

        private InkBlitzPlayerAgent FindVisibleOpponent(InkBlitzArenaInstance arena, InkBlitzArenaSettings cfg)
        {
            Vector3 eyePos = agent.eyePivot != null ? agent.eyePivot.position : agent.transform.position + Vector3.up * 1.1f;
            InkBlitzPlayerAgent best = null;
            float bestDist = 16.0f;

            for (int i = 0; i < arena.agents.Count; i++)
            {
                InkBlitzPlayerAgent other = arena.agents[i];
                if (other == null || !other.gameObject.activeSelf || other.teamId == agent.teamId || other.isRespawning)
                    continue;
                if (other.isSubmergedInFriendlyInk && !other.isCarryingObjective)
                    continue;

                Vector3 targetPt = other.transform.position + Vector3.up * 0.9f;
                Vector3 delta = targetPt - eyePos;
                float d = delta.magnitude;
                if (d > bestDist || d < 0.05f)
                    continue;

                if (!other.isCarryingObjective && Physics.Raycast(eyePos, delta / d, out RaycastHit hit, d - 0.5f, ~0, QueryTriggerInteraction.Ignore))
                {
                    if (hit.collider.GetComponentInParent<InkBlitzPlayerAgent>() != other)
                        continue;
                }

                bestDist = d;
                best = other;
            }
            return best;
        }

        private static Vector3 ComputeLeadAimPoint(InkBlitzPlayerAgent shooter, InkBlitzPlayerAgent target, float projSpeed, float gravity)
        {
            Vector3 origin = shooter.eyePivot != null ? shooter.eyePivot.position : shooter.transform.position + Vector3.up * 1.1f;
            Vector3 targetCenter = target.transform.position + Vector3.up * 0.85f;
            float dist = Vector3.Distance(origin, targetCenter);
            float t = dist / Mathf.Max(4f, projSpeed);
            Vector3 predicted = targetCenter + target.Velocity * t;
            predicted.y += Mathf.Clamp(0.5f * gravity * t * t, 0f, 1.5f);
            return predicted;
        }

        private static void ComputeAimAngles(InkBlitzPlayerAgent shooter, Vector3 targetWorldPoint, out float yawRad, out float pitchRad)
        {
            Vector3 origin = shooter.eyePivot != null ? shooter.eyePivot.position : shooter.transform.position + Vector3.up * 1.1f;
            Vector3 delta = targetWorldPoint - origin;
            yawRad = Mathf.Atan2(delta.x, delta.z);
            float horiz = Mathf.Max(1e-4f, Mathf.Sqrt(delta.x * delta.x + delta.z * delta.z));
            pitchRad = Mathf.Clamp(Mathf.Atan2(delta.y, horiz), -0.8f, 0.8f);
        }
    }
}
