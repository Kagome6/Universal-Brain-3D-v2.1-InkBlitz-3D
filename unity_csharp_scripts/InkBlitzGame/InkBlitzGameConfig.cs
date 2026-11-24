// InkBlitzGameConfig.cs — Core configuration & constants for InkBlitz 3D
using System;
using UnityEngine;

namespace InkBlitz3D
{
    public struct InkColorPalettePreset
    {
        public string label;
        public string hexA;
        public string hexB;
        public Color colorA;
        public Color colorB;

        public InkColorPalettePreset(string label, string hexA, byte rA, byte gA, byte bA, string hexB, byte rB, byte gB, byte bB)
        {
            this.label = label;
            this.hexA = hexA;
            this.hexB = hexB;
            this.colorA = new Color(rA / 255f, gA / 255f, bA / 255f, 1f);
            this.colorB = new Color(rB / 255f, gB / 255f, bB / 255f, 1f);
        }
    }

    public static class InkBlitzInkPalettes
    {
        public static readonly InkColorPalettePreset[] Presets =
        {
            new InkColorPalettePreset("Slate × Sand",     "7F99DC", 0x7F, 0x99, 0xDC, "D1BF9B", 0xD1, 0xBF, 0x9B),
            new InkColorPalettePreset("Ice × Rose",       "B6CEF4", 0xB6, 0xCE, 0xF4, "CF78A4", 0xCF, 0x78, 0xA4),
            new InkColorPalettePreset("Leaf × Violet",    "52C44A", 0x52, 0xC4, 0x4A, "7164C3", 0x71, 0x64, 0xC3),
            new InkColorPalettePreset("Aqua × Gold",      "97E0EE", 0x97, 0xE0, 0xEE, "FFD95A", 0xFF, 0xD9, 0x5A),
            new InkColorPalettePreset("Matcha × Apricot", "CAEE97", 0xCA, 0xEE, 0x97, "EECA97", 0xEE, 0xCA, 0x97),
            new InkColorPalettePreset("Mint × Coral",     "A1D8CA", 0xA1, 0xD8, 0xCA, "D7928F", 0xD7, 0x92, 0x8F),
            new InkColorPalettePreset("Green × Pink",     "52C44A", 0x52, 0xC4, 0x4A, "E564BA", 0xE5, 0x64, 0xBA),
        };

        public static int Count => Presets.Length;

        public static InkColorPalettePreset Get(int index)
        {
            int n = Presets.Length;
            int idx = ((index % n) + n) % n;
            return Presets[idx];
        }

        public static void ApplyMatteColor(Material mat, Color color)
        {
            if (mat == null) return;
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);
            if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", 0f);
            if (mat.HasProperty("_Glossiness")) mat.SetFloat("_Glossiness", 0f);
            if (mat.HasProperty("_SpecularHighlights")) mat.SetFloat("_SpecularHighlights", 0f);
            if (mat.HasProperty("_EnvironmentReflections")) mat.SetFloat("_EnvironmentReflections", 0f);
            mat.EnableKeyword("_SPECULARHIGHLIGHTS_OFF");
            mat.EnableKeyword("_ENVIRONMENTREFLECTIONS_OFF");
        }
    }

    [Serializable]
    public class InkBlitzArenaSettings
    {
        [Header("Arena Dimensions & Stage Identity (180-deg Symmetric)")]
        public int stageId = 1;             // Unique topology ID (>0 keys Python Hippocampus structural schema bank!)
        public float arenaWidthX = 24f;     // X in [-12, +12]
        public float arenaLengthZ = 32f;    // Z in [-16, +16]
        public float arenaHeightY = 6f;

        [Header("Logical Turf Grid & High-Res Visual Ink")]
        public int gridWidth = 72;          // ~0.333m logical gameplay cell
        public int gridHeight = 96;
        [Range(1, 10)] public int visualTexSubdivisions = 8; // 576x768 (~4.1cm crisp pure flat matte ink texels!)
        public float spawnPadPaintRadiusM = 2.2f;

        [Header("Team & Floor Colors")]
        public Color teamAColor = new Color(127f / 255f, 153f / 255f, 220f / 255f, 1f); // #7f99dc (Muted Periwinkle Blue)
        public Color teamBColor = new Color(209f / 255f, 191f / 255f, 155f / 255f, 1f); // #d1bf9b (Warm Champagne Sand)
        public Color neutralFloorColor = new Color(0.24f, 0.25f, 0.28f, 1f);

        [Header("Locomotion, Submerge Stamina & Ink Physics")]
        public float walkSpeedMps = 4.3f;
        public float swimSpeedMultiplier = 1.48f;          // ~6.36 m/s tactical swim (balanced for 24x32m arena)
        public float enemyInkSpeedMultiplier = 0.35f;
        public float carrierMoveSpeedMultiplier = 0.72f;   // Heavy BlitzCore carry (~3.10 m/s walk, ~4.58 m/s swim)
        public float maxSubmergeDurationSec = 4.0f;        // Max continuous dive dive before breath surface
        public float submergeRegenPerSec = 2.0f;           // Full 4.0s dive stamina recovered in 2.0s in human form
        public float submergeExhaustCooldownSec = 1.2f;    // Forced surface catch-breath lock when stamina hits 0
        public float gravityMps2 = 22.0f;

        [Header("Health & Ink Tank")]
        public float maxHp = 100f;
        public float swimHpRegenPerSec = 35f;
        public float humanHpRegenPerSec = 6f;
        public float damageHpRegenDelaySec = 1.0f;          // Delay after taking damage before natural HP regen resumes
        public float enemyInkSlipDps = 15f;
        public float enemyInkMinHpFloor = 30f;
        public float swimInkRegenPerSec = 0.48f;
        public float humanInkRegenPerSec = 0.09f;

        [Header("Main Shooter & Ballistic Projectiles")]
        public float fireIntervalSec = 0.13f;
        public float inkCostPerShot = 0.026f;              // ~38 shots per full tank
        public float shotRangeM = 8.5f;                    // Mid-short Blaster-class effective horizontal range
        public float shotSpreadDeg = 1.2f;                 // Subtle organic shooter dispersion
        public float projectileSpeedMps = 18.0f;
        public float projectileGravityMps2 = 11.5f;
        public float projectileMaxLifetimeSec = 0.50f;
        public float projectileHitRadiusM = 0.20f;
        public float shooterDamage = 28f;                  // 4-shot splat (TTK ~0.39s, enables mid-fight dive evade)
        public float shotSplashRadiusM = 1.15f;
        public float shotDropletRadiusM = 0.50f;
        public float footPaintRadiusM = 0.50f;

        [Header("BlitzCore Carrier Cannon")]
        public float blitzCoreFireIntervalSec = 0.65f;
        public float blitzCoreInkCost = 0.08f;
        public float blitzCoreShotRangeM = 11.8f;
        public float blitzCoreProjectileSpeedMps = 16.0f;
        public float blitzCoreMaxLifetimeSec = 0.72f;
        public float blitzCoreBlastRadiusM = 2.25f;
        public float blitzCoreDirectDamage = 90f;
        public float blitzCoreSplashDamage = 50f;

        [Header("Match, Respawn & BlitzCore Rules")]
        public float matchDurationSec = 90f;
        public float respawnDurationSec = 2.0f;            // 1v1 balanced respawn (allows goal-line last stand)
        public float blitzCoreShieldMaxHp = 280f;          // 7 shooter hits solo (prevents 0.5s coin-flip insta-pop)
        public float blitzCoreShieldPopFuseSec = 0.45f;    // Swelling detonation warning window before burst
        public float blitzCoreShieldPopRadiusM = 3.0f;
        public float blitzCoreShieldPopLethalRadiusM = 1.95f;
        public float blitzCoreShieldPopDamage = 140f;
        public float blitzCoreShieldPopOuterDamage = 55f;
        public float blitzCoreShieldDecayDelaySec = 1.5f;
        public float blitzCoreShieldDecayPerSec = 18.0f;
        public float blitzCoreHoldLimitSec = 60.0f;
        public float blitzCoreRetreatTimerMultiplier = 2.5f;
        public float blitzCorePickupRadiusM = 1.45f;
        public float blitzCoreIdleResetSec = 18f;
        public float pedestalTouchdownRadiusM = 1.35f;
        public bool enableOvertime = true;
        public float maxOvertimeSec = 30.0f;
        public float overtimePickupGraceSec = 5.0f;

        [Header("Anti-Hack Reward Shaping")]
        public float turfExploreCellReward = 0.03f;
        public float shieldPopReward = 1.5f;
        public float claimObjectiveReward = 1.20f;
        public float enemyClaimObjectivePenalty = -0.80f;
        public float advancePotentialScale = 3.2f;
        public float bestProgressBonusScale = 8.5f;
        public float enemyBestProgressPenaltyScale = 4.0f;
        public float splatEnemyCarrierReward = 6.0f;
        public float splatStandardReward = 2.0f;
        public float deathPenalty = -3.0f;
        public float knockoutWinReward = 40.0f;
        public float timeoutWinReward = 20.0f;
        public float matchLossPenalty = -20.0f;
        public float knockoutLossPenalty = -32.0f;

        [Header("3D Counter-Tennis / Deflect Power-Up (Late-Game / Stalemate Only)")]
        public float deflectAppearRemainingSec = 22.0f; // Spawns ONLY when match drags to <= 22s left or Overtime!
        public float deflectBuffDurationSec = 6.5f;
        public float deflectPickupRespawnSec = 28.0f;
        public float deflectPickupRadiusM = 1.35f;
        public float deflectInterceptRadiusM = 1.55f;
        public float deflectMinCosAngle = 0.28f;     // ~73-deg frontal 3D aim cone
        public float deflectSpeedMultiplier = 1.32f;
        public float deflectDamageMultiplier = 1.45f;
        public int deflectMaxBounces = 3;
        public float deflectParryReward = 0.25f;

        [Header("Support Trooper Bot (Wingman-Style Non-Carrier Wingman)")]
        public bool enableSupportTrooperBots = true;
        public float supportBotMaxHp = 75f;
        public float supportBotMoveSpeedMultiplier = 0.84f; // ~3.61 m/s (consistent SupportWingman wingman pace)
        public float supportBotFireIntervalSec = 0.48f;
        public float supportBotProjectileSpeedMps = 14.8f;
        public float supportBotDamage = 18.0f;
        public float supportBotRespawnSec = 7.5f;
        public float splatSupportBotReward = 0.35f;

        [Header("Spawn, Pedestal & Sky-Bridge Pickup Positions (Arena-Local, 180-deg Symmetric)")]
        public Vector3 teamASpawnLocal = new Vector3(-8.8f, 0.65f, -13.6f);
        public Vector3 teamBSpawnLocal = new Vector3(8.8f, 0.65f, 13.6f);
        public Vector3 teamASupportSpawnLocal = new Vector3(-6.6f, 0.65f, -12.4f);
        public Vector3 teamBSupportSpawnLocal = new Vector3(6.6f, 0.65f, 12.4f);
        public Vector3 teamAPedestalLocal = new Vector3(8.0f, 1.35f, -12.8f);
        public Vector3 teamBPedestalLocal = new Vector3(-8.0f, 1.35f, 12.8f);
        public Vector3 deflectOrbSELocal = new Vector3(10.45f, 1.325f, -5.8f);
        public Vector3 deflectOrbNWLocal = new Vector3(-10.45f, 1.325f, 5.8f);

        public static string GetStageDisplayName(int id)
        {
            return id == 2 ? "Stage 2: 遮蔽ツインフランク" : "Stage 1: 標準シタデル";
        }

        public static string GetStageShortName(int id)
        {
            return id == 2 ? "S2:遮蔽ツインフランク" : "S1:標準シタデル";
        }

        public void ApplyStagePreset(int targetStageId)
        {
            stageId = (targetStageId == 2) ? 2 : 1;
            if (stageId == 2)
            {
                teamASpawnLocal = new Vector3(-7.2f, 0.65f, -13.6f);
                teamBSpawnLocal = new Vector3(7.2f, 0.65f, 13.6f);
                teamASupportSpawnLocal = new Vector3(-4.8f, 0.65f, -12.6f);
                teamBSupportSpawnLocal = new Vector3(4.8f, 0.65f, 12.6f);
                teamAPedestalLocal = new Vector3(6.8f, 1.28f, -12.8f);
                teamBPedestalLocal = new Vector3(-6.8f, 1.28f, 12.8f);
                deflectOrbSELocal = new Vector3(10.35f, 1.22f, -5.2f);
                deflectOrbNWLocal = new Vector3(-10.35f, 1.22f, 5.2f);
                blitzCoreIdleResetSec = 8.5f;
            }
            else
            {
                teamASpawnLocal = new Vector3(-8.8f, 0.65f, -13.6f);
                teamBSpawnLocal = new Vector3(8.8f, 0.65f, 13.6f);
                teamASupportSpawnLocal = new Vector3(-6.6f, 0.65f, -12.4f);
                teamBSupportSpawnLocal = new Vector3(6.6f, 0.65f, 12.4f);
                teamAPedestalLocal = new Vector3(8.0f, 1.35f, -12.8f);
                teamBPedestalLocal = new Vector3(-8.0f, 1.35f, 12.8f);
                deflectOrbSELocal = new Vector3(10.45f, 1.325f, -5.8f);
                deflectOrbNWLocal = new Vector3(-10.45f, 1.325f, 5.8f);
                blitzCoreIdleResetSec = 18.0f;
            }
        }
    }
}
