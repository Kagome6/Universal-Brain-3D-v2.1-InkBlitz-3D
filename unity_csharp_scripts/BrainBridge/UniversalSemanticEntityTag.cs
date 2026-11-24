// UniversalSemanticEntityTag.cs — Domain-Agnostic 3D Semantic Affordance Tag for Unity 6+
// Matches `task_profiles_3d.py` (`SemanticTag` ontology) 1-to-1 with zero game-specific hardcoding.
// License: GPL-3.0

using UnityEngine;

namespace UniversalBrain3D
{
    public enum UniversalEntityKind
    {
        EnemyAreaHazard = -3,       // Lethal bomb / hazard zone (-3.0f)
        EnemyCarrierOrShot = -2,    // Opponent carrying objective OR incoming projectile (-2.0f)
        OpponentAgent = -1,         // Standard hostile opponent (-1.0f)
        NeutralProp = 0,            // Static landmark / wall prop (0.0f)
        CoverOrFriendlyRefuge = 1,  // Tactical cover block / safe swim node (+1.0f)
        ResourceOrPickup = 2,       // Ammo / ink pack / sub-item pickup (+2.0f)
        PrimaryObjective = 3,       // Central objective, e.g., BlitzCore (+3.0f)
        AttackGoalPedestal = 4,     // Enemy goal pedestal where objective scores (+4.0f)
        TeammateAlly = 5,           // Friendly teammate (+5.0f)
        DefendBasePedestal = 6      // Friendly base pedestal to defend (+6.0f)
    }

    /// <summary>
    /// Interface implemented by the stage's Ink / Turf manager so 1st-person foveal floor rays
    /// encode Friendly (+1) vs Neutral (0) vs Enemy (-1) surface state into `fovea_patch[h,w,3]`.
    /// </summary>
    public interface IFloorSurfaceSampler
    {
        /// <returns>Value in [-1..+1] relative to `observerTeamId`: > +0.25 = Friendly Turf, < -0.25 = Enemy Turf.</returns>
        float SampleSurfaceValence(Vector3 worldPoint, int observerTeamId);
    }

    [DisallowMultipleComponent]
    public class UniversalSemanticEntityTag : MonoBehaviour
    {
        [Header("Universal Brain 3D Affordance Identity")]
        public int entityId = 1;
        public int arenaIndex = 0;
        public int teamId = 0; // 0 = neutral, +1 = Team A (Blue), -1 = Team B (Orange)
        public UniversalEntityKind baseKind = UniversalEntityKind.NeutralProp;
        public float boundingRadius = 0.75f;
        public bool isCarryingPrimaryObjective = false;
        public bool isStealthSubmerged = false;
        public bool isImminentAreaHazard = false;

        private Vector3 _prevPos;
        private Vector3 _estimatedVel;
        private bool _initialized;

        public Vector3 EstimatedVelocity => _estimatedVel;

        private void Awake()
        {
            _prevPos = transform.position;
            _estimatedVel = Vector3.zero;
            _initialized = true;
        }

        private void OnEnable()
        {
            _prevPos = transform.position;
            _estimatedVel = Vector3.zero;
            _initialized = true;
        }

        public void NotifyTeleported()
        {
            _prevPos = transform.position;
            _estimatedVel = Vector3.zero;
            _initialized = true;
        }

        private void FixedUpdate()
        {
            if (!_initialized)
            {
                _prevPos = transform.position;
                _estimatedVel = Vector3.zero;
                _initialized = true;
                return;
            }
            Vector3 delta = transform.position - _prevPos;
            if (delta.sqrMagnitude > 16.0f)
            {
                _prevPos = transform.position;
                _estimatedVel = Vector3.zero;
                return;
            }
            float dt = Mathf.Max(1e-4f, Time.fixedDeltaTime);
            Vector3 rawVel = delta / dt;
            _estimatedVel = Vector3.Lerp(_estimatedVel, rawVel, 0.5f);
            _prevPos = transform.position;
        }

        /// <summary>
        /// Resolve the observer-relative `affordance_tag` float sent in `UB3D` (`ENTITY_DIM` index 8).
        /// Automatically flips teammate/opponent & attack/defend pedestal polarity based on `observerTeamId`!
        /// </summary>
        public float GetObserverRelativeTag(int observerTeamId)
        {
            if (baseKind == UniversalEntityKind.PrimaryObjective)
            {
                if (isImminentAreaHazard && teamId != 0 && observerTeamId != 0 && teamId != observerTeamId)
                    return -3.0f; // Enemy-popped barrier during detonation fuse -> EnemyAreaHazard (-3.0f)
                return 3.0f;
            }
            if (baseKind == UniversalEntityKind.CoverOrFriendlyRefuge)
                return 1.0f;
            if (baseKind == UniversalEntityKind.ResourceOrPickup)
                return 2.0f;
            if (baseKind == UniversalEntityKind.NeutralProp)
                return 0.0f;

            if (baseKind == UniversalEntityKind.AttackGoalPedestal || baseKind == UniversalEntityKind.DefendBasePedestal)
            {
                if (teamId != 0 && observerTeamId != 0)
                {
                    // Enemy team's pedestal is OUR Attack Goal (+4.0); our team's pedestal is Defend Base (+6.0)
                    return (teamId != observerTeamId) ? 4.0f : 6.0f;
                }
                return (float)((int)baseKind);
            }

            // Agent / projectile team-relative hostility
            if (teamId != 0 && observerTeamId != 0)
            {
                if (teamId == observerTeamId)
                    return 5.0f; // Friendly Teammate
                if (baseKind == UniversalEntityKind.EnemyAreaHazard)
                    return -3.0f;
                return isCarryingPrimaryObjective ? -2.0f : -1.0f;
            }
            return (float)((int)baseKind);
        }
    }
}
