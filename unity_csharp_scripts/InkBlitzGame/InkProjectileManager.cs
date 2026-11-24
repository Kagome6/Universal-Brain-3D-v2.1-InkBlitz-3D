// InkProjectileManager.cs — Pooled 3D ballistic ink projectiles & BlitzCore blasts
using UnityEngine;

namespace InkBlitz3D
{
    public class InkProjectileManager : MonoBehaviour
    {
        private const int PoolSize = 80;

        private struct ProjectileSlot
        {
            public bool active;
            public bool isBlitzCoreBlast;
            public int teamId;
            public int bounceCount;
            public float damageScale;
            public InkBlitzPlayerAgent shooter;
            public Vector3 pos;
            public Vector3 vel;
            public float age;
            public float traveledHorizM;
            public float nextDropletAge;
            public GameObject visualGo;
            public MeshRenderer renderer;
        }

        private readonly ProjectileSlot[] _pool = new ProjectileSlot[PoolSize];
        private InkBlitzArenaInstance _arena;
        private Transform _poolRoot;
        public Material templateOpaqueMaterial;
        private Material _matTeamA;
        private Material _matTeamB;
        private bool _initialized;

        public void Initialize(InkBlitzArenaInstance arena)
        {
            _arena = arena;
            if (_initialized) return;

            var settings = (_arena != null && _arena.settings != null) ? _arena.settings : new InkBlitzArenaSettings();
            Material baseMat = templateOpaqueMaterial != null
                ? templateOpaqueMaterial
                : (_arena != null && _arena.stageBuilder != null
                    ? (_arena.stageBuilder.teamAPedestalMaterial ?? _arena.stageBuilder.defaultFloorMaterial)
                    : null);

            if (baseMat != null)
            {
                _matTeamA = new Material(baseMat) { name = "ProjMat_TeamA" };
                _matTeamB = new Material(baseMat) { name = "ProjMat_TeamB" };
                if (_matTeamA.HasProperty("_BaseMap")) _matTeamA.SetTexture("_BaseMap", null);
                if (_matTeamA.HasProperty("_MainTex")) _matTeamA.SetTexture("_MainTex", null);
                if (_matTeamB.HasProperty("_BaseMap")) _matTeamB.SetTexture("_BaseMap", null);
                if (_matTeamB.HasProperty("_MainTex")) _matTeamB.SetTexture("_MainTex", null);
            }
            else
            {
                Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                _matTeamA = new Material(sh) { name = "ProjMat_TeamA" };
                _matTeamB = new Material(sh) { name = "ProjMat_TeamB" };
            }
            SetMatColor(_matTeamA, settings.teamAColor);
            SetMatColor(_matTeamB, settings.teamBColor);

            var rootGo = new GameObject("ProjectilePool");
            rootGo.transform.SetParent(transform, false);
            _poolRoot = rootGo.transform;

            for (int i = 0; i < PoolSize; i++)
            {
                GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.name = $"InkProj_{i}";
                sphere.transform.SetParent(_poolRoot, false);
                Collider col = sphere.GetComponent<Collider>();
                if (col != null)
                {
                    if (Application.isPlaying) Destroy(col);
                    else DestroyImmediate(col);
                }
                sphere.SetActive(false);

                _pool[i] = new ProjectileSlot
                {
                    active = false,
                    visualGo = sphere,
                    renderer = sphere.GetComponent<MeshRenderer>()
                };
            }

            _initialized = true;
        }

        public void ClearAll()
        {
            for (int i = 0; i < PoolSize; i++)
                DeactivateSlot(i);
        }

        public void RefreshPaletteColors(Color teamA, Color teamB)
        {
            if (_matTeamA != null) SetMatColor(_matTeamA, teamA);
            if (_matTeamB != null) SetMatColor(_matTeamB, teamB);
        }

        public void SpawnProjectile(InkBlitzPlayerAgent shooter, Vector3 origin, Vector3 aimDir, bool isBlitzCoreBlast)
        {
            if (_arena == null) return;
            if (!_initialized) Initialize(_arena);

            var cfg = _arena.settings ?? new InkBlitzArenaSettings();
            int freeIdx = -1;
            float oldestAge = -1f;
            for (int i = 0; i < PoolSize; i++)
            {
                if (!_pool[i].active)
                {
                    freeIdx = i;
                    break;
                }
                if (_pool[i].age > oldestAge)
                {
                    oldestAge = _pool[i].age;
                    freeIdx = i;
                }
            }
            if (freeIdx < 0) freeIdx = 0;

            bool isSupportShot = !isBlitzCoreBlast && shooter != null && shooter.isSupportTrooperBot;
            float speed = isBlitzCoreBlast
                ? cfg.blitzCoreProjectileSpeedMps
                : (isSupportShot ? cfg.supportBotProjectileSpeedMps : cfg.projectileSpeedMps);
            Vector3 dir = aimDir.sqrMagnitude > 1e-6f ? aimDir.normalized : Vector3.forward;
            int team = shooter != null ? shooter.teamId : 1;
            float sphereRad = isBlitzCoreBlast ? 0.18f : 0.09f;

            // Muzzle sweep from eyeOrigin -> +0.48m: prevent spawning inside or behind adjacent walls/covers!
            const float muzzleOffset = 0.48f;
            bool muzzleHitEnv = SweepEnvironmentIgnoringAgents(origin, dir, muzzleOffset + 0.04f, sphereRad, out RaycastHit muzzleHit);
            Vector3 muzzleEnd = muzzleHitEnv ? (origin + dir * muzzleHit.distance) : (origin + dir * muzzleOffset);

            // Point-blank BlitzCore Shield hit check before returning on close floor/wall muzzle contact!
            if (_arena.blitzCore != null && _arena.blitzCore.IsShieldActive)
            {
                Vector3 rmPos = _arena.blitzCore.transform.position;
                float hitRad = _arena.blitzCore.EffectiveShieldHitRadiusM + sphereRad * 0.5f;
                Vector3 closestMuzzle = ClosestPointOnSegment(rmPos, origin, muzzleEnd);
                if (Vector3.Distance(rmPos, closestMuzzle) <= hitRad
                    && !IsLineBlockedByEnvironment(Vector3.Lerp(closestMuzzle, origin, 0.15f), rmPos + Vector3.up * 0.22f))
                {
                    float dmgScale = isSupportShot ? (cfg.supportBotDamage / Mathf.Max(1f, cfg.shooterDamage)) : 1f;
                    float shieldDmg = (isBlitzCoreBlast ? 90f : 40f) * dmgScale;
                    _arena.blitzCore.ApplyShieldDamage(shooter, shieldDmg, _arena);
                    _arena.TurfPaint(closestMuzzle, 0.90f, team, shooter);
                    if (isBlitzCoreBlast)
                        ApplyBlastAoE(closestMuzzle, team, shooter, null, cfg, dmgScale);
                    _arena.audioSystem?.EmitSound(closestMuzzle, team, isBlitzCoreBlast ? 0.85f : 0.55f, 0.25f);
                    _arena.audioSystem?.PlayShootSfx(closestMuzzle, isBlitzCoreBlast, isSupportShot);
                    return;
                }
            }

            if (muzzleHitEnv)
            {
                float splashRadius = isBlitzCoreBlast ? cfg.blitzCoreBlastRadiusM : cfg.shotSplashRadiusM;
                PaintAtImpactOrDownward(muzzleHit.point, muzzleHit.normal, team, shooter, splashRadius);
                if (isBlitzCoreBlast)
                    ApplyBlastAoE(muzzleHit.point, team, shooter, null, cfg);
                _arena.audioSystem?.EmitSound(muzzleHit.point, team, isBlitzCoreBlast ? 0.85f : 0.55f, 0.25f);
                _arena.audioSystem?.PlayShootSfx(muzzleHit.point, isBlitzCoreBlast, isSupportShot);
                return;
            }

            Vector3 spawnPos = origin + dir * muzzleOffset;
            ref ProjectileSlot slot = ref _pool[freeIdx];
            slot.active = true;
            slot.isBlitzCoreBlast = isBlitzCoreBlast;
            slot.teamId = team;
            slot.bounceCount = 0;
            slot.damageScale = isSupportShot ? (cfg.supportBotDamage / Mathf.Max(1f, cfg.shooterDamage)) : 1f;
            slot.shooter = shooter;
            slot.pos = spawnPos;
            slot.vel = dir * speed;
            slot.age = 0f;
            slot.traveledHorizM = muzzleOffset;
            slot.nextDropletAge = 0.08f;

            if (slot.visualGo != null)
            {
                float dia = isBlitzCoreBlast ? 0.50f : 0.24f;
                slot.visualGo.transform.position = spawnPos;
                slot.visualGo.transform.localScale = new Vector3(dia, dia, dia);
                if (slot.renderer != null)
                    slot.renderer.sharedMaterial = (slot.teamId > 0) ? _matTeamA : _matTeamB;
                slot.visualGo.SetActive(true);
            }

            _arena.audioSystem?.EmitSound(spawnPos, slot.teamId, isBlitzCoreBlast ? 0.95f : 0.78f, 0.30f);
            _arena.audioSystem?.PlayShootSfx(spawnPos, isBlitzCoreBlast, isSupportShot);
        }

        public void StepProjectiles(float dt)
        {
            if (!_initialized || _arena == null || dt <= 0f) return;

            var cfg = _arena.settings ?? new InkBlitzArenaSettings();

            for (int i = 0; i < PoolSize; i++)
            {
                if (!_pool[i].active) continue;

                ref ProjectileSlot slot = ref _pool[i];
                bool isSupportShot = !slot.isBlitzCoreBlast && slot.shooter != null && slot.shooter.isSupportTrooperBot;
                float maxLife = slot.isBlitzCoreBlast ? cfg.blitzCoreMaxLifetimeSec : cfg.projectileMaxLifetimeSec;
                float maxRange = slot.isBlitzCoreBlast
                    ? cfg.blitzCoreShotRangeM
                    : (isSupportShot ? Mathf.Max(cfg.shotRangeM, 9.2f) : cfg.shotRangeM);

                slot.age += dt;
                if (slot.age >= maxLife || slot.traveledHorizM >= maxRange)
                {
                    PaintDownwardFromAir(slot.pos, slot.teamId, slot.shooter,
                        slot.isBlitzCoreBlast ? cfg.blitzCoreBlastRadiusM : cfg.shotSplashRadiusM);
                    if (slot.isBlitzCoreBlast)
                        ApplyBlastAoE(slot.pos, slot.teamId, slot.shooter, null, cfg, slot.damageScale);
                    DeactivateSlot(i);
                    continue;
                }

                // Ballistic shooter terminal air-drag & ink-drop in the final 22% of range!
                if (!slot.isBlitzCoreBlast && slot.traveledHorizM > 0.78f * maxRange)
                {
                    float drag = Mathf.Exp(-9.0f * dt);
                    slot.vel.x *= drag;
                    slot.vel.z *= drag;
                    slot.vel.y -= cfg.projectileGravityMps2 * 2.15f * dt;
                }
                else
                {
                    slot.vel.y -= cfg.projectileGravityMps2 * dt;
                }

                Vector3 nextPos = slot.pos + slot.vel * dt;
                Vector2 horizStepVec = new Vector2(nextPos.x - slot.pos.x, nextPos.z - slot.pos.z);
                float horizStep = horizStepVec.magnitude;
                float remRange = Mathf.Max(0f, maxRange - slot.traveledHorizM);
                if (horizStep > remRange && horizStep > 1e-5f)
                {
                    float frac = remRange / horizStep;
                    nextPos.x = slot.pos.x + horizStepVec.x * frac;
                    nextPos.z = slot.pos.z + horizStepVec.y * frac;
                    horizStep = remRange;
                }
                slot.traveledHorizM += horizStep;

                Vector3 delta = nextPos - slot.pos;
                float stepDist = delta.magnitude;
                float sphereRad = slot.isBlitzCoreBlast ? 0.18f : 0.09f;

                // 1. Static environment sweep FIRST so walls/covers strictly clamp projectile flight!
                RaycastHit envHit = default;
                bool hitEnv = stepDist > 1e-4f && SweepEnvironmentIgnoringAgents(slot.pos, delta / stepDist, stepDist, sphereRad, out envHit);
                Vector3 effectiveEnd = hitEnv ? (slot.pos + (delta / stepDist) * envHit.distance) : nextPos;

                // 1b. 3D Counter-Tennis / Genji-Airblast Frontal Aim Deflection Check!
                if (TryDeflectProjectile(ref slot, i, effectiveEnd, sphereRad, cfg))
                    continue;

                // 2. Direct opposing agent hit check along unblocked segment [slot.pos, effectiveEnd]
                InkBlitzPlayerAgent directVictim = FindHitOpponentAgent(slot.pos, effectiveEnd, slot.teamId,
                    cfg.projectileHitRadiusM + (slot.isBlitzCoreBlast ? 0.18f : 0f));
                if (directVictim != null)
                {
                    float dmg = (slot.isBlitzCoreBlast ? cfg.blitzCoreDirectDamage : cfg.shooterDamage) * slot.damageScale;
                    _arena.TurfPaint(directVictim.transform.position, 0.95f, slot.teamId, slot.shooter);
                    _arena.OnProjectileHitAgent(slot.shooter, directVictim, dmg, effectiveEnd, slot.isBlitzCoreBlast);
                    if (slot.isBlitzCoreBlast)
                        ApplyBlastAoE(effectiveEnd, slot.teamId, slot.shooter, directVictim, cfg, slot.damageScale);
                    _arena.audioSystem?.EmitSound(effectiveEnd, slot.teamId, 0.45f, 0.22f);
                    DeactivateSlot(i);
                    continue;
                }

                // 3. BlitzCore Shield hit check along unblocked segment [slot.pos, effectiveEnd]
                if (_arena.blitzCore != null && _arena.blitzCore.IsShieldActive)
                {
                    Vector3 rmPos = _arena.blitzCore.transform.position;
                    float hitRad = _arena.blitzCore.EffectiveShieldHitRadiusM + sphereRad * 0.5f;
                    Vector3 closestPt = ClosestPointOnSegment(rmPos, slot.pos, effectiveEnd);
                    float shieldDist = Vector3.Distance(rmPos, closestPt);
                    Vector3 losFrom = hitEnv ? Vector3.MoveTowards(closestPt, slot.pos, 0.04f) : closestPt;
                    if (shieldDist <= hitRad
                        && !IsLineBlockedByEnvironment(losFrom, rmPos + Vector3.up * 0.22f))
                    {
                        float shieldDmg = (slot.isBlitzCoreBlast ? 90f : 40f) * slot.damageScale;
                        _arena.blitzCore.ApplyShieldDamage(slot.shooter, shieldDmg, _arena);
                        _arena.TurfPaint(closestPt, 0.90f, slot.teamId, slot.shooter);
                        _arena.audioSystem?.EmitSound(closestPt, slot.teamId, 0.55f, 0.25f);
                        DeactivateSlot(i);
                        continue;
                    }
                }

                // 4. Environment wall / floor impact
                if (hitEnv)
                {
                    float splashRadius = slot.isBlitzCoreBlast ? cfg.blitzCoreBlastRadiusM : cfg.shotSplashRadiusM;
                    PaintAtImpactOrDownward(envHit.point, envHit.normal, slot.teamId, slot.shooter, splashRadius);
                    if (slot.isBlitzCoreBlast)
                        ApplyBlastAoE(envHit.point + envHit.normal * 0.15f, slot.teamId, slot.shooter, null, cfg, slot.damageScale);
                    _arena.audioSystem?.EmitSound(envHit.point, slot.teamId, 0.45f, 0.22f);
                    DeactivateSlot(i);
                    continue;
                }

                // 4. Advance & paint intermediate in-flight droplet trail
                slot.pos = nextPos;
                if (slot.visualGo != null)
                    slot.visualGo.transform.position = nextPos;

                if (slot.age >= slot.nextDropletAge)
                {
                    slot.nextDropletAge += 0.11f;
                    PaintDownwardFromAir(nextPos, slot.teamId, slot.shooter, cfg.shotDropletRadiusM);
                }
            }
        }

        private bool TryDeflectProjectile(ref ProjectileSlot slot, int slotIdx, Vector3 effectiveEnd,
            float sphereRad, InkBlitzArenaSettings cfg)
        {
            if (slot.bounceCount >= Mathf.Max(1, cfg.deflectMaxBounces))
                return false;

            Vector3 incomingDir = slot.vel.sqrMagnitude > 1e-6f ? (-slot.vel.normalized) : Vector3.zero;
            float interceptR = Mathf.Max(0.85f, cfg.deflectInterceptRadiusM) + (slot.isBlitzCoreBlast ? 0.22f : 0f);

            for (int a = 0; a < _arena.agents.Count; a++)
            {
                InkBlitzPlayerAgent defender = _arena.agents[a];
                if (defender == null || !defender.gameObject.activeSelf || defender.teamId == slot.teamId || defender.isRespawning || !defender.IsDeflectActive)
                    continue;

                Vector3 defenderCenter = defender.transform.position + Vector3.up * 0.88f;
                Vector3 closestPt = ClosestPointOnSegment(defenderCenter, slot.pos, effectiveEnd);
                float d = Vector3.Distance(defenderCenter, closestPt);
                if (d > interceptR)
                    continue;

                // Must be within frontal 3D aim cone (Genji / Airblast racket orientation; flanking from behind bypasses!)
                Vector3 aimDir = defender.CurrentAimDirection;
                if (aimDir.sqrMagnitude < 1e-6f)
                    aimDir = Vector3.forward;
                else
                    aimDir.Normalize();

                float velAlign = Vector3.Dot(aimDir, incomingDir);
                Vector3 relToProj = closestPt - defenderCenter;
                float posAlign = relToProj.sqrMagnitude > 1e-5f ? Vector3.Dot(aimDir, relToProj.normalized) : velAlign;
                if (velAlign < cfg.deflectMinCosAngle || posAlign < -0.25f)
                    continue;

                if (IsLineBlockedByEnvironment(closestPt, defenderCenter))
                    continue;

                // Successful 3D Counter-Tennis Deflection! Flip team, boost speed & damage, launch along defender's 3D aim!
                int newTeam = defender.teamId;
                slot.teamId = newTeam;
                slot.shooter = defender;
                slot.bounceCount++;
                slot.damageScale = Mathf.Max(1.0f, slot.damageScale) * Mathf.Max(1.0f, cfg.deflectDamageMultiplier);

                float baseSpeed = slot.isBlitzCoreBlast ? cfg.blitzCoreProjectileSpeedMps : cfg.projectileSpeedMps;
                float curSpeed = Mathf.Max(baseSpeed, slot.vel.magnitude);
                float newSpeed = Mathf.Min(34.0f, curSpeed * Mathf.Max(1.0f, cfg.deflectSpeedMultiplier));

                Vector3 eyeOrigin = defender.eyePivot != null ? defender.eyePivot.position : defenderCenter;
                const float racketLaunchOffset = 0.62f;
                if (SweepEnvironmentIgnoringAgents(eyeOrigin, aimDir, racketLaunchOffset + 0.04f, sphereRad, out RaycastHit wallHit))
                {
                    float splashR = slot.isBlitzCoreBlast ? cfg.blitzCoreBlastRadiusM : cfg.shotSplashRadiusM;
                    PaintAtImpactOrDownward(wallHit.point, wallHit.normal, newTeam, defender, splashR);
                    if (slot.isBlitzCoreBlast)
                        ApplyBlastAoE(wallHit.point + wallHit.normal * 0.15f, newTeam, defender, null, cfg, slot.damageScale);
                    defender.OnSuccessfulProjectileParry(cfg.deflectParryReward);
                    _arena.audioSystem?.EmitSound(wallHit.point, newTeam, 0.90f, 0.30f);
                    DeactivateSlot(slotIdx);
                    return true;
                }

                Vector3 launchPos = eyeOrigin + aimDir * racketLaunchOffset;
                slot.pos = launchPos;
                slot.vel = aimDir * newSpeed;
                slot.age = 0f;
                slot.traveledHorizM = racketLaunchOffset;
                slot.nextDropletAge = 0.07f;

                if (slot.visualGo != null)
                {
                    float dia = (slot.isBlitzCoreBlast ? 0.56f : 0.30f);
                    slot.visualGo.transform.position = launchPos;
                    slot.visualGo.transform.localScale = new Vector3(dia, dia, dia);
                    if (slot.renderer != null)
                        slot.renderer.sharedMaterial = (newTeam > 0) ? _matTeamA : _matTeamB;
                }

                _arena.TurfPaint(defender.transform.position, 0.72f, newTeam, defender);
                defender.OnSuccessfulProjectileParry(cfg.deflectParryReward);
                _arena.audioSystem?.EmitSound(launchPos, newTeam, 0.92f, 0.32f);
                return true;
            }
            return false;
        }

        private void ApplyBlastAoE(Vector3 center, int attackerTeamId, InkBlitzPlayerAgent shooter,
            InkBlitzPlayerAgent excludeVictim, InkBlitzArenaSettings cfg, float damageScale = 1f)
        {
            float r = cfg.blitzCoreBlastRadiusM;
            for (int a = 0; a < _arena.agents.Count; a++)
            {
                InkBlitzPlayerAgent ag = _arena.agents[a];
                if (ag == null || !ag.gameObject.activeSelf || ag == excludeVictim || ag.teamId == attackerTeamId || ag.isRespawning)
                    continue;
                Vector3 victimCenter = ag.transform.position + Vector3.up * (ag.isDiveForm ? 0.24f : 0.70f);
                float d = Vector3.Distance(center, victimCenter);
                if (d <= r && !IsLineBlockedByEnvironment(center + Vector3.up * 0.20f, victimCenter))
                {
                    float falloff = Mathf.Clamp(1f - 0.5f * (d / r), 0.4f, 1f);
                    _arena.OnProjectileHitAgent(shooter, ag, cfg.blitzCoreSplashDamage * falloff * damageScale, center, true);
                }
            }

            if (_arena.blitzCore != null && _arena.blitzCore.IsShieldActive)
            {
                float ds = Vector3.Distance(center, _arena.blitzCore.transform.position);
                if (ds <= r + _arena.blitzCore.EffectiveShieldHitRadiusM)
                    _arena.blitzCore.ApplyShieldDamage(shooter, cfg.blitzCoreSplashDamage * 0.8f * damageScale, _arena);
            }
        }

        private InkBlitzPlayerAgent FindHitOpponentAgent(Vector3 segStart, Vector3 segEnd, int projectileTeamId, float projRadius)
        {
            float humanHitThreshold = 0.55f + projRadius;
            InkBlitzPlayerAgent closest = null;
            float bestDist = humanHitThreshold;

            for (int i = 0; i < _arena.agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = _arena.agents[i];
                if (ag == null || !ag.gameObject.activeSelf || ag.teamId == projectileTeamId || ag.isRespawning)
                    continue;

                if (ag.isSubmergedInFriendlyInk || ag.isDiveForm)
                {
                    // Low-profile flat dive hitbox: high horizontal shots pass overhead, while
                    // downward floor-aimed shots, gravity-drop arcs, and blasts hit & force surface!
                    float maxDiveHeight = (ag.isSubmergedInFriendlyInk ? 0.36f : 0.50f) + projRadius;
                    float diveHorizThreshold = (ag.isSubmergedInFriendlyInk ? 0.45f : 0.50f) + projRadius;
                    Vector3 diveCenter = ag.transform.position + Vector3.up * 0.16f;
                    Vector3 closestPt = ClosestPointOnSegment(diveCenter, segStart, segEnd);
                    float relY = closestPt.y - ag.transform.position.y;
                    if (relY > maxDiveHeight || relY < -0.25f)
                        continue;

                    float dxz = Vector2.Distance(
                        new Vector2(closestPt.x, closestPt.z),
                        new Vector2(ag.transform.position.x, ag.transform.position.z));
                    float d3 = Vector3.Distance(diveCenter, closestPt);
                    if (dxz <= diveHorizThreshold && d3 <= bestDist
                        && !IsLineBlockedByEnvironment(0.5f * (segStart + segEnd), ag.transform.position + Vector3.up * 0.24f))
                    {
                        bestDist = d3;
                        closest = ag;
                    }
                }
                else
                {
                    Vector3 lowPt = ag.transform.position + Vector3.up * 0.45f;
                    Vector3 highPt = ag.transform.position + Vector3.up * 1.05f;
                    float d = Mathf.Min(
                        DistancePointToSegment(lowPt, segStart, segEnd),
                        DistancePointToSegment(highPt, segStart, segEnd)
                    );
                    if (d <= bestDist && !IsLineBlockedByEnvironment(0.5f * (segStart + segEnd), ag.transform.position + Vector3.up * 0.75f))
                    {
                        bestDist = d;
                        closest = ag;
                    }
                }
            }
            return closest;
        }

        public bool IsLineBlockedByEnvironment(Vector3 from, Vector3 to)
        {
            Vector3 delta = to - from;
            float dist = delta.magnitude;
            if (dist <= 0.08f) return false;
            return RaycastEnvironmentIgnoringAgents(from, delta / dist, dist - 0.04f, out _);
        }

        private bool SweepEnvironmentIgnoringAgents(Vector3 origin, Vector3 dir, float maxDist, float radius, out RaycastHit validHit)
        {
            if (RaycastEnvironmentIgnoringAgents(origin, dir, maxDist, out validHit))
                return true;

            Vector3 cur = origin;
            float rem = maxDist;
            for (int hop = 0; hop < 8 && rem > 1e-4f; hop++)
            {
                if (!Physics.SphereCast(cur, radius, dir, out validHit, rem, ~0, QueryTriggerInteraction.Ignore))
                    return false;
                if (validHit.collider.GetComponentInParent<InkBlitzPlayerAgent>() == null
                    && validHit.collider.GetComponentInParent<BlitzCoreObjective>() == null)
                    return true;
                float adv = validHit.distance + 0.05f;
                cur += dir * adv;
                rem -= adv;
            }
            validHit = default;
            return false;
        }

        private bool RaycastEnvironmentIgnoringAgents(Vector3 origin, Vector3 dir, float maxDist, out RaycastHit validHit)
        {
            Vector3 cur = origin;
            float rem = maxDist;
            for (int hop = 0; hop < 8 && rem > 1e-4f; hop++)
            {
                if (!Physics.Raycast(cur, dir, out validHit, rem, ~0, QueryTriggerInteraction.Ignore))
                    return false;
                if (validHit.collider.GetComponentInParent<InkBlitzPlayerAgent>() == null
                    && validHit.collider.GetComponentInParent<BlitzCoreObjective>() == null)
                    return true;
                float adv = validHit.distance + 0.05f;
                cur += dir * adv;
                rem -= adv;
            }
            validHit = default;
            return false;
        }

        private void PaintAtImpactOrDownward(Vector3 point, Vector3 normal, int teamId, InkBlitzPlayerAgent shooter, float radius)
        {
            if (normal.y > 0.58f)
            {
                _arena.TurfPaint(point, radius, teamId, shooter);
                return;
            }
            PaintDownwardFromAir(point + normal * 0.15f, teamId, shooter, radius);
        }

        private void PaintDownwardFromAir(Vector3 airPoint, int teamId, InkBlitzPlayerAgent shooter, float radius)
        {
            Vector3 cur = airPoint;
            float rem = 6f;
            for (int hop = 0; hop < 3 && rem > 0.05f; hop++)
            {
                if (!Physics.Raycast(cur, Vector3.down, out RaycastHit hit, rem, ~0, QueryTriggerInteraction.Ignore))
                    return;
                if (hit.collider.GetComponentInParent<InkBlitzPlayerAgent>() != null)
                {
                    float adv = hit.distance + 0.05f;
                    cur += Vector3.down * adv;
                    rem -= adv;
                    continue;
                }
                if (hit.normal.y > 0.58f)
                    _arena.TurfPaint(hit.point, radius, teamId, shooter);
                return;
            }
        }

        private static Vector3 ClosestPointOnSegment(Vector3 p, Vector3 a, Vector3 b)
        {
            Vector3 ab = b - a;
            float lenSq = ab.sqrMagnitude;
            if (lenSq < 1e-8f)
                return a;
            float t = Mathf.Clamp01(Vector3.Dot(p - a, ab) / lenSq);
            return a + ab * t;
        }

        private static float DistancePointToSegment(Vector3 p, Vector3 a, Vector3 b)
        {
            Vector3 ab = b - a;
            float lenSq = ab.sqrMagnitude;
            if (lenSq < 1e-8f)
                return Vector3.Distance(p, a);
            float t = Mathf.Clamp01(Vector3.Dot(p - a, ab) / lenSq);
            return Vector3.Distance(p, a + ab * t);
        }

        private void DeactivateSlot(int idx)
        {
            _pool[idx].active = false;
            _pool[idx].shooter = null;
            if (_pool[idx].visualGo != null)
                _pool[idx].visualGo.SetActive(false);
        }

        private static void SetMatColor(Material mat, Color color)
        {
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);
            if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", 0.0f);
            if (mat.HasProperty("_Glossiness")) mat.SetFloat("_Glossiness", 0.0f);
            if (mat.HasProperty("_SpecularHighlights")) mat.SetFloat("_SpecularHighlights", 0.0f);
            if (mat.HasProperty("_EnvironmentReflections")) mat.SetFloat("_EnvironmentReflections", 0.0f);
            mat.EnableKeyword("_SPECULARHIGHLIGHTS_OFF");
            mat.EnableKeyword("_ENVIRONMENTREFLECTIONS_OFF");
        }

        private void OnDestroy()
        {
            if (_matTeamA != null)
            {
                if (Application.isPlaying) Destroy(_matTeamA);
                else DestroyImmediate(_matTeamA);
            }
            if (_matTeamB != null)
            {
                if (Application.isPlaying) Destroy(_matTeamB);
                else DestroyImmediate(_matTeamB);
            }
        }
    }
}
