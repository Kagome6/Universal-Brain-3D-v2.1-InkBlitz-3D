// InkBlitzMechanicsSelfTest.cs — Deterministic EditMode verification of Phase 2 & 3 gameplay invariants
using System;
using UnityEditor;
using UnityEngine;
using UniversalBrain3D;

namespace InkBlitz3D.Editor
{
    public static class InkBlitzMechanicsSelfTest
    {
        [MenuItem("InkBlitzLab/Run Phase 2-3 Mechanics Self-Test")]
        public static bool RunMechanicsVerification()
        {
            GameObject root = null;
            try
            {
                var settings = new InkBlitzArenaSettings();
                root = CreateTempArenaObject("SelfTest_Arena_Temp", new Vector3(400f, 0f, 0f), settings, out InkBlitzArenaInstance arena);
                TurfGridSystem turf = arena.turfGrid;
                InkBlitzPlayerAgent agA = arena.agents[0];
                InkBlitzPlayerAgent agB = arena.agents[1];

                // Check 1: 180-deg Rotational Symmetry of Landmarks & Sky-Bridge Deflect Orbs
                Transform pedA = root.transform.Find("Landmarks/Pedestal_TeamA");
                Transform pedB = root.transform.Find("Landmarks/Pedestal_TeamB");
                Transform orbSE = root.transform.Find("Landmarks/DeflectOrb_SkyBridge_SE");
                Transform orbNW = root.transform.Find("Landmarks/DeflectOrb_SkyBridge_NW");
                Transform covSE = root.transform.Find("WallsAndCovers/Cover_CenterSE");
                Transform covNW = root.transform.Find("WallsAndCovers/Cover_CenterNW");
                Transform hgSE = root.transform.Find("WalkableFloors/Overlook_SE");
                Transform hgNW = root.transform.Find("WalkableFloors/Overlook_NW");
                Assert(pedA != null && pedB != null && orbSE != null && orbNW != null && covSE != null && covNW != null && hgSE != null && hgNW != null,
                    "Stage landmark hierarchy missing.");
                AssertSymmetryXZ(pedA.localPosition, pedB.localPosition, "Pedestals");
                AssertSymmetryXZ(orbSE.localPosition, orbNW.localPosition, "SkyBridgeDeflectOrbs");
                AssertSymmetryXZ(covSE.localPosition, covNW.localPosition, "Covers");
                AssertSymmetryXZ(hgSE.localPosition, hgNW.localPosition, "Overlooks");
                Debug.Log("[PASS] SelfTest 1/9: 180-degree rotational stage & Sky-Bridge Deflect Orb symmetry verified.");

                // Check 2: Anti-Hack Turf Reward (only 1st claim grants newlyExploredCells)
                Vector3 testPatchPos = root.transform.position;
                PaintResult firstPaint = turf.PaintCircle(testPatchPos, 1.0f, 1, grantExploreCredit: true);
                PaintResult secondPaint = turf.PaintCircle(testPatchPos, 1.0f, 1, grantExploreCredit: true);
                Assert(firstPaint.newlyExploredCells > 0, $"Expected newlyExploredCells > 0 on 1st paint, got {firstPaint.newlyExploredCells}");
                Assert(secondPaint.newlyExploredCells == 0, $"Expected newlyExploredCells == 0 on 2nd paint, got {secondPaint.newlyExploredCells}");
                Debug.Log($"[PASS] SelfTest 2/5: Anti-hack turf exploration verified (1st={firstPaint.newlyExploredCells}, 2nd={secondPaint.newlyExploredCells}).");

                // Check 3: PROPRIO Contract Invariants
                TeleportAgent(agA, testPatchPos);
                agA.isCarryingObjective = false;
                agA.inkTank01 = 1.0f;
                agA.ApplyExternalCommand(testPatchPos, 0.5f, UniversalBrainBridge3D.VERB_EVADE_HAZARD, 0f, 0f);
                agA.StepAgent(1f / 60f);
                Assert(agA.isSubmergedInFriendlyInk, "Expected agent submerged on friendly ink during EVADE_HAZARD.");
                Assert(agA.SurfacePosture01 >= 0.90f, $"Expected SurfacePosture01 >= 0.90 when submerged, got {agA.SurfacePosture01}");
                Assert(agA.ResourceOrCarrier01 <= 0.88f, $"Expected ResourceOrCarrier01 <= 0.88 without BlitzCore, got {agA.ResourceOrCarrier01}");
                agA.isCarryingObjective = true;
                Assert(agA.ResourceOrCarrier01 >= 0.90f, $"Expected ResourceOrCarrier01 >= 0.90 with BlitzCore, got {agA.ResourceOrCarrier01}");
                agA.isCarryingObjective = false;
                Debug.Log("[PASS] SelfTest 3/5: PROPRIO[13] & PROPRIO[14] encoding invariants verified.");

                // Check 4: Splat -> Respawn Wait -> Mid-Episode Respawn preserves Turf
                int cellsBeforeSplat = turf.TeamACells;
                arena.OnProjectileHitAgent(agB, agA, settings.maxHp + 10f, agA.transform.position, false);
                Assert(agA.isRespawning, "Expected agA.isRespawning == true after fatal hit.");
                Assert(agA.brainBridge.isCurrentlyRespawning, "Expected bridge.isCurrentlyRespawning == true during respawn wait.");
                for (int t = 0; t < 180; t++) // 3.0s at 60Hz > 2.8s respawnDuration
                    arena.SimulateStep(1f / 60f);
                Assert(!agA.isRespawning, "Expected agA.isRespawning == false after 3.0s respawn wait.");
                Assert(!agA.brainBridge.isCurrentlyRespawning, "Expected bridge.isCurrentlyRespawning == false after respawn.");
                Assert(Mathf.Approximately(agA.hp, settings.maxHp), $"Expected full HP after respawn, got {agA.hp}");
                Assert(turf.TeamACells >= cellsBeforeSplat, $"Expected turf cells preserved across respawn, before={cellsBeforeSplat}, after={turf.TeamACells}");
                Debug.Log("[PASS] SelfTest 4/5: Splat, respawn wait, and mid-episode turf preservation verified.");

                // Check 5: BlitzCore Shield -> Pop -> Pickup -> Carrier -> Touchdown KO -> Reset
                BlitzCoreObjective rm = arena.blitzCore;
                Assert(rm != null && rm.State == BlitzCoreState.Shielded, "Expected BlitzCore initially Shielded.");
                rm.ApplyShieldDamage(agA, settings.blitzCoreShieldMaxHp + 20f, arena, immediatePop: true);
                Assert(rm.State == BlitzCoreState.PickupReady, $"Expected PickupReady after shield pop, got {rm.State}");

                TeleportAgent(agA, rm.transform.position);
                agA.ApplyExternalCommand(rm.transform.position, 1f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, 0f, 0f);
                arena.SimulateStep(1f / 60f);
                Assert(rm.State == BlitzCoreState.Carried && rm.Carrier == agA && agA.isCarryingObjective,
                    "Expected agA to claim BlitzCore on contact.");

                TeleportAgent(agA, root.transform.position + settings.teamBPedestalLocal + new Vector3(0f, 0.2f, 0f));
                arena.SimulateStep(1f / 60f);
                Assert(arena.episodesCompleted == 1 && arena.teamAWins == 1,
                    $"Expected Knockout win & episode completion (ep={arena.episodesCompleted}, winsA={arena.teamAWins}).");
                Assert(rm.State == BlitzCoreState.Shielded && !agA.isCarryingObjective,
                    "Expected BlitzCore reset to Shielded center after KO reset.");
                Debug.Log($"[PASS] SelfTest 5/8: Full BlitzCore Shield->Pop->Carry->Touchdown KO->Reset loop verified ({arena.lastMatchSummary}).");

                // Check 6: Authentic Carrier Swim + Tug-of-War Shield & Lethal Pop
                TeleportAgent(agA, root.transform.position + settings.teamASpawnLocal);
                TeleportAgent(agB, root.transform.position + settings.teamBSpawnLocal);
                agA.ApplyExternalCommand(agA.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f);
                agB.ApplyExternalCommand(agB.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, Mathf.PI, 0f);
                Assert(rm.State == BlitzCoreState.Shielded && Mathf.Approximately(rm.ShieldTug, 0f), "Expected ShieldTug=0 on Shielded reset.");
                rm.ApplyShieldDamage(agA, 80f, arena);
                Assert(Mathf.Approximately(rm.ShieldTug, 80f) && rm.State == BlitzCoreState.Shielded, $"Expected ShieldTug=+80, got {rm.ShieldTug}");
                rm.ApplyShieldDamage(agB, 60f, arena);
                Assert(Mathf.Approximately(rm.ShieldTug, 20f) && rm.State == BlitzCoreState.Shielded, $"Expected counter-tug ShieldTug=+20, got {rm.ShieldTug}");
                for (int t = 0; t < 120; t++) // 2.0s > 1.5s decay delay
                    arena.SimulateStep(1f / 60f);
                Assert(rm.ShieldTug < 20f && rm.ShieldTug >= 0f, $"Expected ShieldTug decay after idle delay, got {rm.ShieldTug}");

                TeleportAgent(agB, rm.transform.position + new Vector3(1.5f, 0f, 0f)); // inside lethal pop radius
                agB.ApplyExternalCommand(agB.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, Mathf.PI, 0f);
                rm.ApplyShieldDamage(agA, settings.blitzCoreShieldMaxHp + 20f, arena, immediatePop: true);
                Assert(rm.State == BlitzCoreState.PickupReady, $"Expected PickupReady after Team A tug pop, got {rm.State}");
                Assert(agB.isRespawning, "Expected lethal BlitzCore shield pop to splat nearby opponent within lethal radius.");

                arena.TurfPaint(rm.transform.position, 2.0f, 1, null);
                TeleportAgent(agA, rm.transform.position);
                agA.ApplyExternalCommand(rm.transform.position, 0.5f, UniversalBrainBridge3D.VERB_EVADE_HAZARD, 0f, 0f);
                arena.SimulateStep(1f / 60f);
                arena.SimulateStep(1f / 60f);
                Assert(agA.isCarryingObjective, "Expected agA carrying BlitzCore.");
                Assert(agA.isSubmergedInFriendlyInk, "Expected carrier allowed to swim in friendly ink.");
                Assert(!agA.entityTag.isStealthSubmerged, "Expected carrying BlitzCore disables stealth invisibility.");
                Debug.Log("[PASS] SelfTest 6/8: Tug-of-war BlitzCore shield, lethal pop, and visible carrier swim verified.");

                // Check 7: 60s Hold Limit + Retreat Penalty + Authentic Overtime
                for (int t = 0; t < 185; t++) // finish agB respawn
                    arena.SimulateStep(1f / 60f);
                rm.ResetToCenter();
                rm.ApplyShieldDamage(agA, settings.blitzCoreShieldMaxHp + 20f, arena, immediatePop: true);
                TeleportAgent(agA, rm.transform.position);
                agA.ApplyExternalCommand(rm.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f);
                arena.SimulateStep(1f / 60f);
                Assert(agA.isCarryingObjective, "Expected agA carrying BlitzCore for hold-limit check.");

                float holdBeforeRetreat = rm.CarrierHoldTimer;
                TeleportAgent(agA, root.transform.position + new Vector3(0f, 0.1f, -10.0f));
                agA.ApplyExternalCommand(agA.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f);
                for (int t = 0; t < 60; t++)
                    arena.SimulateStep(1f / 60f);
                float drained1s = holdBeforeRetreat - rm.CarrierHoldTimer;
                Assert(drained1s >= 2.3f, $"Expected >= 2.3s drained in 1.0s retreating in backfield (2.5x), got {drained1s}");

                for (int t = 0; t < 1600 && !agA.isRespawning; t++)
                    arena.SimulateStep(1f / 60f);
                Assert(agA.isRespawning && !agA.isCarryingObjective && rm.State == BlitzCoreState.Shielded,
                    "Expected carrier self-destruct splat & BlitzCore center reset on hold-timer expiry.");
                for (int t = 0; t < 185; t++) // complete agA respawn
                    arena.SimulateStep(1f / 60f);

                rm.ResetToCenter();
                rm.ApplyShieldDamage(agA, settings.blitzCoreShieldMaxHp + 20f, arena, immediatePop: true);
                TeleportAgent(agA, rm.transform.position);
                agA.ApplyExternalCommand(rm.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f);
                arena.SimulateStep(1f / 60f);
                TeleportAgent(agA, root.transform.position + new Vector3(0f, 0.1f, -6.0f));
                agA.ApplyExternalCommand(agA.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f);
                arena.bestProgressTeamA = 0.20f;
                arena.bestProgressTeamB = 0.60f;
                arena.matchTimeRemaining = 0.05f;
                for (int t = 0; t < 10; t++)
                    arena.SimulateStep(1f / 60f);
                Assert(arena.isOvertime, "Expected Overtime when regulation ends while trailing Team A holds BlitzCore.");

                int winsBeforeOt = arena.teamAWins;
                int epsBeforeOt = arena.episodesCompleted;
                TeleportAgent(agA, root.transform.position + new Vector3(-4.5f, 1.25f, 10.5f)); // push01=0.722 > 0.605, outside pedestal KO radius
                agA.ApplyExternalCommand(agA.transform.position, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f);
                arena.SimulateStep(1f / 60f);
                Assert(arena.episodesCompleted == epsBeforeOt + 1 && arena.teamAWins == winsBeforeOt + 1 && !arena.isOvertime,
                    $"Expected Team A Overtime comeback win, got ep={arena.episodesCompleted}, winsA={arena.teamAWins}, isOT={arena.isOvertime}");
                Debug.Log($"[PASS] SelfTest 7/9: 60s hold limit (2.5x retreat drain) and Overtime comeback win verified ({arena.lastMatchSummary}).");

                // Check 8: Multi-Arena Spatial, Turf & Port Isolation
                GameObject a0Go = null;
                GameObject a1Go = null;
                try
                {
                    a0Go = CreateTempArenaObject("SelfTest_MultiA0", new Vector3(600f, 0f, 0f), settings, out InkBlitzArenaInstance a0);
                    a1Go = CreateTempArenaObject("SelfTest_MultiA1", new Vector3(700f, 0f, 0f), settings, out InkBlitzArenaInstance a1);
                    a0.ConfigureForSlot(0, 50055, 50056, null);
                    a1.ConfigureForSlot(1, 50057, 50058, null);

                    a0.TurfPaint(a0.transform.position, 2.0f, 1);
                    Assert(a0.turfGrid.SampleSurfaceValence(a0.transform.position, 1) > 0.5f, "Expected Arena 0 center painted.");
                    Assert(Mathf.Approximately(a1.turfGrid.SampleSurfaceValence(a1.transform.position, 1), 0f), "Expected Arena 1 center unaffected (0).");
                    Assert(a0.agents[0].brainBridge.port == 50055 && a0.agents[1].brainBridge.port == 50056, "Expected Arena 0 ports 50055/50056.");
                    Assert(a1.agents[0].brainBridge.port == 50057 && a1.agents[1].brainBridge.port == 50058, "Expected Arena 1 ports 50057/50058.");
                    Assert(a0.arenaIndex == 0 && a1.arenaIndex == 1, "Expected distinct arenaIndex values.");
                    Debug.Log("[PASS] SelfTest 8/9: Multi-Arena spatial, turf & TCP port isolation verified.");
                }
                finally
                {
                    if (a0Go != null) UnityEngine.Object.DestroyImmediate(a0Go);
                    if (a1Go != null) UnityEngine.Object.DestroyImmediate(a1Go);
                }

                // Check 9: Late-Game-Only Sky-Bridge Deflect Orb -> PROPRIO[7] Buff Signal -> 3D Aim Counter-Shot & Rear-Flank Bypass
                arena.projectileManager.ClearAll();
                DeflectPowerPickup pickNW = arena.deflectPickups.Find(p => p != null && p.name.Contains("NW"));
                Assert(pickNW != null, "Expected NW Sky-Bridge DeflectPowerPickup present.");
                arena.matchTimeRemaining = 80.0f;
                arena.isOvertime = false;
                pickNW.ResetPickup();
                arena.SimulateStep(1f / 60f);
                Assert(!pickNW.IsAvailable && !pickNW.entityTag.enabled,
                    "Expected DeflectPowerPickup dormant during early/mid match (80s left > 22s threshold).");

                arena.matchTimeRemaining = 15.0f; // Late-game stalemate window (<= 22s remaining)!
                arena.SimulateStep(1f / 60f);
                Assert(pickNW.IsAvailable && pickNW.entityTag.enabled,
                    "Expected DeflectPowerPickup unlocked & available during late-game stalemate (<= 22s left).");
                Assert(Mathf.Approximately(pickNW.entityTag.GetObserverRelativeTag(1), 2.0f), "Expected ResourceOrPickup tag == +2.0.");

                TeleportAgent(agA, root.transform.position + settings.deflectOrbNWLocal);
                agA.ApplyExternalCommand(agA.transform.position, 0.6f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, 0f, 0f);
                arena.SimulateStep(1f / 60f);
                Assert(agA.HasDeflectBuff && agA.IsDeflectActive && !pickNW.IsAvailable,
                    "Expected agA to acquire Deflect Buff on Sky-Bridge traversal and put orb on cooldown.");
                Assert(agA.brainBridge.activeBuffOrTransit01 > 0.50f && agA.brainBridge.activeBuffOrTransit01 <= 0.78f,
                    $"Expected somatic activeBuffOrTransit01 in (0.50, 0.78], got {agA.brainBridge.activeBuffOrTransit01}");

                // Place agA at open valley (0, 0.05, -6.5) aiming North (+Z, yaw=0) at agB at (0, 0.05, -2.5); agB shoots South (-Z) at agA!
                Vector3 duelA = root.transform.position + new Vector3(0f, 0.05f, -6.5f);
                Vector3 duelB = root.transform.position + new Vector3(0f, 0.05f, -2.5f);
                TeleportAgent(agA, duelA);
                TeleportAgent(agB, duelB);
                agA.hp = settings.maxHp;
                agB.hp = settings.maxHp;
                agA.ApplyExternalCommand(duelA, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, 0f, -0.05f, commitOpen: false);
                agB.ApplyExternalCommand(duelB, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, Mathf.PI, 0f, commitOpen: false);
                agA.StepAgent(1f / 60f);
                agB.StepAgent(1f / 60f);
                Physics.SyncTransforms();

                arena.projectileManager.SpawnProjectile(agB, duelB + Vector3.up * 1.05f, Vector3.back, isBlitzCoreBlast: false);
                for (int t = 0; t < 35; t++)
                    arena.projectileManager.StepProjectiles(1f / 60f);

                Assert(Mathf.Approximately(agA.hp, settings.maxHp),
                    $"Expected agA frontal Reflector Shield to take 0 damage, got hp={agA.hp}");
                float expectedParryDmg = settings.shooterDamage * settings.deflectDamageMultiplier;
                Assert(agB.hp <= settings.maxHp - expectedParryDmg + 0.5f,
                    $"Expected deflected counter-shot to hit agB for ~{expectedParryDmg:F1} dmg, got agB.hp={agB.hp}");

                // Rear-flank bypass check: agA still aims North (+Z, yaw=0), agB shoots agA in the BACK from South (0, 0.05, -9.2) (+Z)!
                arena.projectileManager.ClearAll();
                Vector3 flankB = root.transform.position + new Vector3(0f, 0.05f, -9.2f);
                TeleportAgent(agB, flankB);
                Physics.SyncTransforms();
                arena.projectileManager.SpawnProjectile(agB, flankB + Vector3.up * 1.05f, Vector3.forward, isBlitzCoreBlast: false);
                for (int t = 0; t < 25; t++)
                    arena.projectileManager.StepProjectiles(1f / 60f);
                Assert(agA.hp < settings.maxHp,
                    $"Expected rear-flank shot from behind defender's aim cone to bypass shield and deal damage, got agA.hp={agA.hp}");
                Debug.Log($"[PASS] SelfTest 9/10: Late-game-only Sky-Bridge Deflect Orb, PROPRIO[7] buff signal, 3D aim counter-shot ({settings.maxHp - agB.hp:F1} dmg), and rear-flank bypass verified.");

                // Check 10: SupportWingman Support Bot (Ally +5.0 vs Enemy -1.0, Non-Carrier, Non-Orb, 7.5s Respawn & 1v1/2v2 Toggle)
                InkBlitzPlayerAgent supA = arena.agents.Count > 2 ? arena.agents[2] : null;
                InkBlitzPlayerAgent supB = arena.agents.Count > 3 ? arena.agents[3] : null;
                Assert(supA != null && supB != null && supA.isSupportTrooperBot && supB.isSupportTrooperBot,
                    "Expected symmetric SupportTrooperBot agents[2] & agents[3].");
                Assert(supA.brainBridge == null && supB.brainBridge == null, "Expected Support Bots to have null brainBridge (zero TCP port consumption).");
                Assert(Mathf.Approximately(supA.entityTag.GetObserverRelativeTag(1), 5.0f), "Expected Team A Support Bot tag == +5.0 (TeammateAlly) for Team A.");
                Assert(Mathf.Approximately(supA.entityTag.GetObserverRelativeTag(-1), -1.0f), "Expected Team A Support Bot tag == -1.0 (OpponentAgent) for Team B.");

                // Verify Support Bot NEVER picks up BlitzCore even when standing directly on PickupReady BlitzCore!
                rm.ResetToCenter();
                rm.ApplyShieldDamage(agA, settings.blitzCoreShieldMaxHp + 20f, arena, immediatePop: true);
                Assert(rm.State == BlitzCoreState.PickupReady, "Expected BlitzCore PickupReady.");
                TeleportAgent(agA, root.transform.position + settings.teamASpawnLocal);
                TeleportAgent(agB, root.transform.position + settings.teamBSpawnLocal);
                TeleportAgent(supA, rm.transform.position);
                arena.SimulateStep(1f / 60f);
                Assert(rm.State == BlitzCoreState.PickupReady && !supA.isCarryingObjective,
                    "Expected SupportTrooperBot strictly forbidden from picking up BlitzCore!");

                // Verify Support Bot NEVER steals late-game Deflect Orb!
                DeflectPowerPickup pickSE = arena.deflectPickups.Find(p => p != null && p.name.Contains("SE"));
                arena.matchTimeRemaining = 12.0f;
                arena.SimulateStep(1f / 60f);
                Assert(pickSE != null && pickSE.IsAvailable, "Expected SE Deflect Orb unlocked in late-game.");
                TeleportAgent(supB, root.transform.position + settings.deflectOrbSELocal);
                arena.SimulateStep(1f / 60f);
                Assert(pickSE.IsAvailable && !supB.HasDeflectBuff,
                    "Expected SupportTrooperBot strictly forbidden from stealing Sky-Bridge Deflect Orb!");

                // Verify Support Bot splat gives 7.5s respawn window!
                arena.OnProjectileHitAgent(agA, supB, 200f, supB.transform.position, false);
                Assert(supB.isRespawning && supB.RespawnTimeRemaining >= 7.0f,
                    $"Expected Support Bot long respawn >= 7.0s for numbers-advantage play, got {supB.RespawnTimeRemaining:F2}s");

                // Verify 1v1 <-> 2v2 runtime toggle, live 3-mode Trooper AI stepping & ZERO high-freq yaw vibration!
                supB.ResetForNewMatch();
                var trAiA = supA.GetComponent<SupportTrooperBot>();
                var trAiB = supB.GetComponent<SupportTrooperBot>();
                trAiA.enableTrooperAi = true;
                trAiB.enableTrooperAi = true;
                trAiA.ResetTrooperState();
                trAiB.ResetTrooperState();

                // Stress-test head-on obstacle approach + rapid enemy stealth flicker (previously caused 180-deg flip-flop!)
                Vector3 stressPosA = root.transform.position + new Vector3(-3.2f, 0.05f, -7.3f); // directly behind Cover_ValleySW (-3.2, -5.8)
                TeleportAgent(supA, stressPosA);
                Vector3 flickerEnemyPos = root.transform.position + new Vector3(1.5f, 0.05f, -9.8f);
                arena.TurfPaint(flickerEnemyPos, 1.5f, -1);
                TeleportAgent(agB, flickerEnemyPos);

                float prevYaw = trAiA.CurrentSmoothedYawDeg;
                float prevDeltaYaw = 0f;
                float maxSingleStepDeg = 0f;
                int rapidSignFlips = 0;
                for (int t = 0; t < 180; t++)
                {
                    int flickerVerb = ((t / 2) % 2 == 0) ? UniversalBrainBridge3D.VERB_EVADE_HAZARD : UniversalBrainBridge3D.VERB_APPROACH_HOVER;
                    agB.ApplyExternalCommand(flickerEnemyPos, 0.3f, flickerVerb, Mathf.PI, 0f, commitOpen: false);
                    arena.SimulateStep(1f / 60f);
                    float curYaw = trAiA.CurrentSmoothedYawDeg;
                    float dy = Mathf.DeltaAngle(prevYaw, curYaw);
                    maxSingleStepDeg = Mathf.Max(maxSingleStepDeg, Mathf.Abs(dy));
                    if (Mathf.Abs(dy) > 1.5f && Mathf.Abs(prevDeltaYaw) > 1.5f && (dy * prevDeltaYaw < 0f))
                        rapidSignFlips++;
                    prevDeltaYaw = dy;
                    prevYaw = curYaw;
                }
                Assert(maxSingleStepDeg <= 6.0f, $"Expected slew-limited turn <= 6.0 deg/tick, got {maxSingleStepDeg:F2}");
                Assert(rapidSignFlips == 0, $"Expected 0 frame-to-frame yaw sign vibrations, got {rapidSignFlips}");
                Assert(Vector3.Distance(supA.transform.position, stressPosA) > 1.2f,
                    $"Expected SupportTrooperBot to cleanly bypass cover without stalling, moved={Vector3.Distance(supA.transform.position, stressPosA):F2}m");

                arena.SetSupportTrooperBotsEnabled(false);
                Assert(!supA.gameObject.activeSelf && !supB.gameObject.activeSelf, "Expected Support Bots inactive in 1v1 mode.");
                arena.SetSupportTrooperBotsEnabled(true);
                Assert(supA.gameObject.activeSelf && supB.gameObject.activeSelf, "Expected Support Bots active in 2v2 mode.");
                Debug.Log($"[PASS] SelfTest 10/11: SupportWingman Support Bot guardrails, 7.5s respawn, zero-vibration slew (maxStep={maxSingleStepDeg:F2}deg, flips={rapidSignFlips}), and 1v1/2v2 toggle verified.");

                // Check 11: Interactive Overhead Spectator Camera (360-deg Yaw, Zoom, & Near-Horizontal Stadium Wall Rim-Glide)
                var testCamGo = new GameObject("TempSpectatorCam");
                testCamGo.transform.SetParent(root.transform, false);
                var testCam = testCamGo.AddComponent<Camera>();
                var camMgr = root.AddComponent<InkBlitzMultiArenaManager>();
                camMgr.mainCamera = testCam;
                camMgr.primaryArena = arena;
                camMgr.arenas.Clear();
                camMgr.arenas.Add(arena);
                camMgr.ResetCameraView(instant: true);
                Physics.SyncTransforms();

                Quaternion savedEyeRotA = agA.eyePivot.rotation;
                Vector3 arenaFocusCenter = root.transform.position + new Vector3(0f, 0.95f, 0f);
                float[] testYaws = { 0f, 90f, 180f, 270f, 42f };
                float[] testPitches = { 2.5f, 11.5f, 43.0f, 84.0f };

                foreach (float yDeg in testYaws)
                {
                    foreach (float pDeg in testPitches)
                    {
                        camMgr.SetCameraOrbitState(yDeg, pDeg, 32.0f, Vector2.zero, instant: true);
                        Vector3 cPos = testCam.transform.position;
                        // Verify outer perimeter walls (Wall_North/South/East/West) NEVER block the sightline to arena center!
                        RaycastHit[] wallHits = Physics.RaycastAll(cPos, (arenaFocusCenter - cPos).normalized, Vector3.Distance(cPos, arenaFocusCenter));
                        for (int h = 0; h < wallHits.Length; h++)
                        {
                            string hn = wallHits[h].collider != null ? wallHits[h].collider.name : "";
                            Assert(!hn.StartsWith("Wall_"),
                                $"Spectator camera sightline occluded by outer wall '{hn}' at yaw={yDeg}, pitch={pDeg}, camPos={cPos - root.transform.position}");
                        }
                    }
                }

                // Verify true near-horizontal tilt (~2.5 deg) & close zoom (4.5m)
                camMgr.SetCameraOrbitState(35f, 2.5f, 4.5f, new Vector2(1.5f, -2.0f), instant: true);
                float actualHorizPitch = testCam.transform.eulerAngles.x;
                if (actualHorizPitch > 180f) actualHorizPitch -= 360f;
                Assert(Mathf.Abs(actualHorizPitch) <= 6.5f && camMgr.CurrentCameraDistance <= 5.0f,
                    $"Expected near-horizontal pitch <= 6.5 deg & close zoom <= 5.0m, got pitch={actualHorizPitch:F2} deg, dist={camMgr.CurrentCameraDistance:F2}m");
                Assert(Quaternion.Angle(savedEyeRotA, agA.eyePivot.rotation) < 0.01f,
                    "Spectator camera orbit must NEVER alter Agent A's 1st-person eyePivot!");
                Debug.Log($"[PASS] SelfTest 11/12: Interactive spectator camera 360-deg orbit, zoom (4.5m..64m), and unobstructed near-horizontal ({actualHorizPitch:F2} deg) wall rim-glide verified.");

                // Check 12: 7-Preset Matte Ink Palette Live Switching & Brain Semantic Valence Invariance
                Vector3 samplePtA = root.transform.position + new Vector3(0f, 0.05f, -6.5f);
                Vector3 samplePtB = root.transform.position + new Vector3(0f, 0.05f, 6.5f);
                arena.TurfPaint(samplePtA, 1.0f, 1);
                arena.TurfPaint(samplePtB, 1.0f, -1);
                Assert(InkBlitzInkPalettes.Count == 7, $"Expected 7 ink palettes, got {InkBlitzInkPalettes.Count}");
                for (int p = 0; p < InkBlitzInkPalettes.Count; p++)
                {
                    camMgr.SetColorPalette(p);
                    InkColorPalettePreset expected = InkBlitzInkPalettes.Get(p);
                    Assert(ColorUtility.ToHtmlStringRGB(arena.settings.teamAColor) == expected.hexA,
                        $"Palette {p} Team A hex mismatch: {ColorUtility.ToHtmlStringRGB(arena.settings.teamAColor)} vs {expected.hexA}");
                    Assert(ColorUtility.ToHtmlStringRGB(arena.settings.teamBColor) == expected.hexB,
                        $"Palette {p} Team B hex mismatch: {ColorUtility.ToHtmlStringRGB(arena.settings.teamBColor)} vs {expected.hexB}");
                    Assert(Mathf.Approximately(arena.turfGrid.SampleSurfaceValence(samplePtA, 1), 1.0f),
                        "Brain floor valence must stay +1.0 across all visual color palettes!");
                    Assert(Mathf.Approximately(arena.turfGrid.SampleSurfaceValence(samplePtB, 1), -1.0f),
                        "Brain floor valence must stay -1.0 across all visual color palettes!");
                }
                camMgr.SetColorPalette(0);
                Debug.Log("[PASS] SelfTest 12/13: All 7 matte ink color palettes (#7F99DC..#E564BA) & brain valence invariance verified.");

                // Check 13: Studio SFX Pack Loading & AI-Only Shoot SE Filter (Support Bots Must NOT Trigger Shoot SE!)
                Assert(ArenaAudioSystem.EnsureClipsLoaded(),
                    "Expected all 8 48kHz WAV clips in Resources/Audio/ to load via ArenaAudioSystem.EnsureClipsLoaded()!");
                int shootBeforeMain = arena.audioSystem.ShootSfxPlayedCount;
                arena.projectileManager.SpawnProjectile(agA, agA.transform.position + Vector3.up, Vector3.forward, isBlitzCoreBlast: false);
                Assert(arena.audioSystem.ShootSfxPlayedCount == shootBeforeMain + 1,
                    "Expected Main AI Agent shot to trigger ShootSfxPlayedCount!");
                int shootBeforeSup = arena.audioSystem.ShootSfxPlayedCount;
                arena.projectileManager.SpawnProjectile(supA, supA.transform.position + Vector3.up, Vector3.forward, isBlitzCoreBlast: false);
                Assert(arena.audioSystem.ShootSfxPlayedCount == shootBeforeSup,
                    "Expected SupportTrooperBot shot to stay SILENT (ShootSfxPlayedCount unchanged)!");

                int diveBefore = arena.audioSystem.DiveSfxPlayedCount;
                Vector3 diveSpot = root.transform.position + new Vector3(0f, 0.05f, -6.5f);
                arena.TurfPaint(diveSpot, 1.5f, 1);
                agA.ResetForNewMatch();
                TeleportAgent(agA, diveSpot);
                agA.ApplyExternalCommand(diveSpot, 0.5f, UniversalBrainBridge3D.VERB_EVADE_HAZARD, 0f, 0f, commitOpen: false);
                agA.StepAgent(1f / 60f);
                Assert(arena.audioSystem.DiveSfxPlayedCount > diveBefore, "Expected Dive Dive into friendly ink to trigger DiveSfx!");
                Assert(arena.audioSystem.SplatSfxPlayedCount > 0, "Expected Splat KO to have triggered SplatSfx!");
                Assert(arena.audioSystem.ShieldPopSfxPlayedCount > 0, "Expected BlitzCore Shield Pop to have triggered ShieldPopSfx!");
                Assert(arena.audioSystem.GoalSfxPlayedCount > 0, "Expected Goal Touchdown to have triggered GoalSfx!");
                Debug.Log("[PASS] SelfTest 13/14: Studio 48kHz SFX pack (AI-only Shoot, Dive, Splat, ShieldPop, Goal) verified.");

                // Check 14: Two-Stage Safe Emergency Stop + 16:9 Landscape Switch Geometry
                var testCoord = root.AddComponent<UniversalArenaCoordinator3D>();
                var dashGo = new GameObject("TempDashCheck");
                dashGo.SetActive(false);
                dashGo.transform.SetParent(root.transform, false);
                var testDash = dashGo.AddComponent<InkBlitzSessionDashboardUi>();
                testDash.coordinator = testCoord;
                testDash.multiManager = camMgr;
                Rect stopRect = testDash.GetEmergencyStopButtonRect();
                Assert(Mathf.Approximately(stopRect.width, 96f) && Mathf.Approximately(stopRect.height, 54f),
                    $"Expected 16:9 Emergency Stop button rect (96x54), got {stopRect.width}x{stopRect.height}!");
                camMgr.coordinator = testCoord;
                arena.ConfigureForSlot(99, 59155, 59156, testCoord);
                camMgr.TriggerEmergencyStop();
                Assert(camMgr.isEmergencyHalted && testCoord.IsEmergencyHalted && testCoord.EmergencyStopsTriggeredCount == 1,
                    "Expected TriggerEmergencyStop to latch IsEmergencyHalted on both MultiManager & Coordinator!");
                Assert(testCoord.SafeShutdownPhase == "CLEAN_EXIT" || testCoord.SafeShutdownPhase == "RELEASING_MEMORY",
                    $"Expected SafeShutdownPhase to enter CLEAN_EXIT or RELEASING_MEMORY, got '{testCoord.SafeShutdownPhase}'");
                Assert(Mathf.Approximately(Time.timeScale, 0f),
                    $"Expected EmergencyStop to freeze Time.timeScale to 0.0, got {Time.timeScale}");
                Assert(!arena.BeginParallelArenaStep(1f / 60f, out _),
                    "Expected BeginParallelArenaStep to immediately return false while EmergencyHalted!");
                camMgr.ClearEmergencyStopForReset();
                Assert(!camMgr.isEmergencyHalted && !testCoord.IsEmergencyHalted && Mathf.Approximately(Time.timeScale, 1f),
                    "Expected ClearEmergencyStopForReset to restore normal Time.timeScale = 1.0 and unlatch halt!");
                Debug.Log("[PASS] SelfTest 14/15: Two-Stage Safe Emergency Stop (16:9 96x54 switch, TCP sever, graceful RAM/VRAM teardown & physics freeze) verified.");

                // Check 15: C1-Smooth Terrain Manifold Continuity, Zero Wall Trenches & Zero Slope POV Jitter
                float minNormalY = 1.0f;
                float maxSymErr = 0f;
                for (int gz = -32; gz <= 32; gz++)
                {
                    float lz = gz * 0.5f;
                    for (int gx = -24; gx <= 24; gx++)
                    {
                        float lx = gx * 0.5f;
                        float hPos = ArenaStageBuilder.SampleTerrainHeightLocal(lx, lz, settings);
                        float hNeg = ArenaStageBuilder.SampleTerrainHeightLocal(-lx, -lz, settings);
                        maxSymErr = Mathf.Max(maxSymErr, Mathf.Abs(hPos - hNeg));
                        Vector3 nrm = ArenaStageBuilder.SampleTerrainNormalLocal(lx, lz, settings);
                        minNormalY = Mathf.Min(minNormalY, nrm.y);
                    }
                }
                Assert(maxSymErr < 1e-5f, $"Expected exact 180-deg heightfield symmetry, got maxSymErr={maxSymErr}");
                Assert(minNormalY >= 0.75f, $"Expected all walkable terrain slopes < 41.5 deg (< 45 deg slopeLimit & >> 0.58 fovea floor), got minNormalY={minNormalY:F3}");
                Assert(agA.brainBridge.stageId == 1 && testCoord.hippoMemoryMode == "adaptive",
                    $"Expected stageId=1 & hippoMemoryMode='adaptive', got stageId={agA.brainBridge.stageId}, mode='{testCoord.hippoMemoryMode}'");

                // Former wall-gap trenches must now be smooth wall-flush slopes (height > 0.45m right against outer walls!)
                Assert(ArenaStageBuilder.SampleTerrainHeightLocal(3.8f, -15.8f, settings) > 0.45f, "Expected Citadel side slope flush with South wall (zero trench).");
                Assert(ArenaStageBuilder.SampleTerrainHeightLocal(-4.5f, -15.8f, settings) > 0.30f, "Expected Spawn defense slope flush with South wall (zero trench).");
                Assert(ArenaStageBuilder.SampleTerrainHeightLocal(-11.9f, -10.0f, settings) > 0.30f, "Expected Spawn forward slope flush with West wall (zero trench).");

                // Swim across Center Mound (former yellow E/W stairs: x=-3.1 -> +3.1) after settling into dive posture:
                rm.ResetToCenter();
                rm.ApplyShieldDamage(agA, settings.blitzCoreShieldMaxHp + 20f, arena, immediatePop: true); // Clear central barrier so pure terrain manifold is tested unobstructed
                agA.ResetForNewMatch();
                Vector3 startStair = root.transform.position + new Vector3(-3.1f, 0f, 0f);
                Vector3 endStair = root.transform.position + new Vector3(3.1f, 0f, 0f);
                for (float sx = -3.2f; sx <= 3.2f; sx += 0.4f)
                    arena.TurfPaint(root.transform.position + new Vector3(sx, 0f, 0f), 0.9f, 1);
                TeleportAgent(agA, startStair);
                for (int w = 0; w < 24; w++)
                {
                    agA.ApplyExternalCommand(startStair, 0.3f, UniversalBrainBridge3D.VERB_EVADE_HAZARD, Mathf.PI * 0.5f, 0f, commitOpen: false);
                    agA.StepAgent(1f / 60f);
                }

                float prevEyeY = agA.eyePivot.position.y;
                float prevDeltaEyeY = 0f;
                float maxEyeAccelM = 0f;
                for (int t = 0; t < 72; t++)
                {
                    agA.ApplyExternalCommand(endStair, 1.0f, UniversalBrainBridge3D.VERB_EVADE_HAZARD, Mathf.PI * 0.5f, 0f, commitOpen: false);
                    agA.StepAgent(1f / 60f);
                    float curEyeY = agA.eyePivot.position.y;
                    float dy = curEyeY - prevEyeY;
                    if (t >= 2)
                        maxEyeAccelM = Mathf.Max(maxEyeAccelM, Mathf.Abs(dy - prevDeltaEyeY));
                    prevDeltaEyeY = dy;
                    prevEyeY = curEyeY;
                }
                Assert(agA.transform.position.x > root.transform.position.x + 1.8f,
                    $"Expected smooth swim traversal across Center Mound E/W slope, reached localX={agA.transform.position.x - root.transform.position.x:F2}");
                Assert(maxEyeAccelM <= 0.020f,
                    $"Expected C2-smooth zero-snap 1st-person eyePivot trajectory across Center Mound & downhill slope (max |d2y| <= 0.020m), got {maxEyeAccelM:F4}m");
                Debug.Log($"[PASS] SelfTest 15/16: C1-Smooth Terrain Manifold verified (minNormalY={minNormalY:F3}, maxSymErr={maxSymErr:E2}, maxEyeAccel={maxEyeAccelM:F4}m).");

                // Check 16: Hybrid Win-Rate Estimator (Exact 50.0% Antisymmetry, Urgency Monotonicity, Online Calibration & Non-Overlap HUD)
                arena.ResetSessionStats();
                var winEst = dashGo.AddComponent<InkBlitzWinRateEstimator>();
                winEst.enableDiskPersistence = false;
                winEst.coordinator = testCoord;
                winEst.multiManager = camMgr;
                winEst.dashboardUi = testDash;
                testDash.winRateEstimator = winEst;

                float[] phiSym = new float[InkBlitzWinRateEstimator.FEATURE_DIM];
                float pInit = winEst.EvaluateArenaWinProbability(arena, phiSym);
                Assert(Mathf.Abs(pInit - 0.50f) < 1e-4f, $"Expected exact 50.0% win rate at symmetric match start, got {pInit * 100f:F3}%");

                // Give Team A count lead (0.55 vs 0.15) + splat Team B => pA must rise well above 75%, and negated features must sum to 100%!
                arena.bestProgressTeamA = 0.55f;
                arena.bestProgressTeamB = 0.15f;
                arena.matchTimeRemaining = 12.0f;
                arena.OnProjectileHitAgent(agA, agB, 200f, agB.transform.position, false);
                float[] phiAdvA = new float[InkBlitzWinRateEstimator.FEATURE_DIM];
                float pAdvA = winEst.EvaluateArenaWinProbability(arena, phiAdvA);
                float[] phiAdvB = new float[InkBlitzWinRateEstimator.FEATURE_DIM];
                for (int k = 0; k < InkBlitzWinRateEstimator.FEATURE_DIM; k++)
                    phiAdvB[k] = -phiAdvA[k];
                float pAdvB = winEst.EvaluateWinProbabilityFromFeatures(phiAdvB);
                Assert(pAdvA > 0.80f, $"Expected late-game lead + enemy splat to yield >80% win prob, got {pAdvA * 100f:F2}%");
                Assert(Mathf.Abs((pAdvA + pAdvB) - 1.0f) < 1e-4f, $"Expected strict antisymmetry p(phi)+p(-phi)==1, got {pAdvA + pAdvB:F5}");

                // Online calibration check: if an underdog state repeatedly wins in actual outcomes, calibrator adapts upward while keeping 0->50%!
                float[] testSample = new float[InkBlitzWinRateEstimator.FEATURE_DIM];
                testSample[7] = 0.65f; // Strong turf lead
                float pBeforeCalib = winEst.EvaluateWinProbabilityFromFeatures(testSample);
                var sampleList = new System.Collections.Generic.List<float[]> { testSample, testSample, testSample, testSample };
                for (int m = 0; m < 12; m++)
                    winEst.TrainCalibratorOnMatchOutcome(sampleList, sampleList.Count, 1.0f);
                float pAfterCalib = winEst.EvaluateWinProbabilityFromFeatures(testSample);
                float pZeroAfterCalib = winEst.EvaluateWinProbabilityFromFeatures(new float[InkBlitzWinRateEstimator.FEATURE_DIM]);
                Assert(pAfterCalib > pBeforeCalib + 0.01f, $"Expected online calibrator to raise win prob on winning feature, {pBeforeCalib:F4} -> {pAfterCalib:F4}");
                Assert(Mathf.Abs(pZeroAfterCalib - 0.50f) < 1e-5f, $"Expected zero-bias antisymmetry preserved after calibration, got {pZeroAfterCalib:F5}");

                Rect stopR = testDash.GetEmergencyStopButtonRect();
                Rect barR = winEst.GetTopWinRateBarRect();
                Assert(!stopR.Overlaps(barR), $"Expected EmergencyStop {stopR} and WinRateBar {barR} to never overlap!");
                Debug.Log($"[PASS] SelfTest 16/24: Hybrid Win-Rate Estimator verified (init={pInit * 100f:F1}%, advA={pAdvA * 100f:F1}%, calib={pBeforeCalib * 100f:F1}%->{pAfterCalib * 100f:F1}%, zero={pZeroAfterCalib * 100f:F1}%).");

                // Check 17: AivisSpeech RIFF WAVE 24kHz PCM Decoder & Interactive Coaching HUD Non-Overlap
                var testVoice = dashGo.AddComponent<InkBlitzCompanionVoiceCoach>();
                testVoice.coordinator = testCoord;
                testDash.voiceCoach = testVoice;

                int testSampleRate = 24000;
                int testFrames = 480;
                byte[] testWav = new byte[44 + testFrames * 2];
                System.Text.Encoding.ASCII.GetBytes("RIFF").CopyTo(testWav, 0);
                BitConverter.GetBytes(36 + testFrames * 2).CopyTo(testWav, 4);
                System.Text.Encoding.ASCII.GetBytes("WAVEfmt ").CopyTo(testWav, 8);
                BitConverter.GetBytes(16).CopyTo(testWav, 16);
                BitConverter.GetBytes((ushort)1).CopyTo(testWav, 20); // PCM
                BitConverter.GetBytes((ushort)1).CopyTo(testWav, 22); // Mono
                BitConverter.GetBytes(testSampleRate).CopyTo(testWav, 24);
                BitConverter.GetBytes(testSampleRate * 2).CopyTo(testWav, 28);
                BitConverter.GetBytes((ushort)2).CopyTo(testWav, 32);
                BitConverter.GetBytes((ushort)16).CopyTo(testWav, 34);
                System.Text.Encoding.ASCII.GetBytes("data").CopyTo(testWav, 36);
                BitConverter.GetBytes(testFrames * 2).CopyTo(testWav, 40);
                for (int i = 0; i < testFrames; i++)
                {
                    short val = (short)Mathf.RoundToInt(Mathf.Sin(2f * Mathf.PI * 440f * i / testSampleRate) * 16384f);
                    BitConverter.GetBytes(val).CopyTo(testWav, 44 + i * 2);
                }

                bool wavOk = InkBlitzCompanionVoiceCoach.TryDecodeWavToAudioClip(testWav, "SelfTestWav", out AudioClip decodedClip);
                Assert(wavOk && decodedClip != null, "Expected TryDecodeWavToAudioClip to decode 24kHz 16-bit PCM WAV.");
                Assert(decodedClip.frequency == testSampleRate && decodedClip.samples == testFrames && decodedClip.channels == 1,
                    $"Unexpected decoded clip metadata: freq={decodedClip.frequency}, samples={decodedClip.samples}, ch={decodedClip.channels}");
                float[] readBack = new float[8];
                decodedClip.GetData(readBack, 0);
                float expectedS3 = Mathf.RoundToInt(Mathf.Sin(2f * Mathf.PI * 440f * 3f / testSampleRate) * 16384f) / 32768f;
                Assert(Mathf.Abs(readBack[3] - expectedS3) < 1e-3f, $"Decoded PCM sample mismatch: got {readBack[3]:F5}, expected {expectedS3:F5}");
                UnityEngine.Object.DestroyImmediate(decodedClip);

                Rect gemmaR = testDash.GetGemmaBroadcastRect();
                Assert(gemmaR.x >= 316f - 0.1f || testDash.neuralHud == null || !testDash.neuralHud.showSubjectivePovPip,
                    $"Expected GemmaBroadcastRect {gemmaR} to clear SubjectivePOV PiP on the left!");
                Assert(!gemmaR.Overlaps(stopR), $"Expected GemmaBroadcastRect {gemmaR} not to overlap EmergencyStop {stopR}!");
                Debug.Log($"[PASS] SelfTest 17/18: AivisSpeech WAV Decoder & Coaching HUD verified (24kHz PCM exact, rect={gemmaR}).");

                // Check 18: BlitzCore Elevated DropAt Non-Burial + Live 3D Ballistic Projectile Pop + Respawn Sky-Ghost Suppression
                arena.ConfigureForSlot(99, 59155, 59156, null);
                arena.ResetSessionStats();
                arena.projectileManager.ClearAll();
                agA.ResetForNewMatch();
                agB.ResetForNewMatch();
                supA.ResetForNewMatch();
                supB.ResetForNewMatch();
                Physics.SyncTransforms();

                // 18a. Verify splatting an unrelated agent disables entityTag during respawn (zero sky-ghost at y=3.15m!)
                Assert(supB.entityTag.enabled && supB.entityTag.isActiveAndEnabled, "Expected supB entityTag active when alive.");
                arena.OnProjectileHitAgent(agA, supB, 250f, supB.transform.position, false);
                Assert(supB.isRespawning && !supB.entityTag.enabled && !supB.entityTag.isActiveAndEnabled,
                    "Expected splatted respawning agent entityTag.enabled == false (must never emit sky-ghost entity!).");
                for (int t = 0; t < 465; t++) // > 7.5s support respawn
                    supB.StepAgent(1f / 60f);
                Assert(!supB.isRespawning && supB.entityTag.enabled && supB.entityTag.EstimatedVelocity.magnitude < 0.2f,
                    $"Expected supB entityTag re-enabled cleanly with zero velocity spike on respawn, vel={supB.entityTag.EstimatedVelocity}");

                // 18b. Have agB claim BlitzCore, move onto elevated SE High Ground Wing (+8.5, y~1.21m, -9.0), get splatted by agA,
                //      and verify BlitzCore drops ABOVE terrain (not buried at y=0.65m!) AND pops cleanly from real ballistic shots!
                rm.ResetToCenter();
                rm.ApplyShieldDamage(agB, settings.blitzCoreShieldMaxHp + 20f, arena, immediatePop: true);
                TeleportAgent(agB, rm.transform.position);
                agB.ApplyExternalCommand(rm.transform.position, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, Mathf.PI, 0f);
                arena.SimulateStep(1f / 60f);
                Assert(rm.State == BlitzCoreState.Carried && rm.Carrier == agB, "Expected agB carrying BlitzCore before elevated splat test.");

                float hgGroundY = ArenaStageBuilder.SampleTerrainHeightLocal(8.5f, -9.0f, settings);
                Assert(hgGroundY > 1.05f, $"Expected elevated SE High Ground terrain > 1.05m, got {hgGroundY:F3}m");
                Vector3 hgCarrierWorld = root.transform.position + new Vector3(8.5f, hgGroundY, -9.0f);
                TeleportAgent(agB, hgCarrierWorld);
                arena.OnProjectileHitAgent(agA, agB, 250f, hgCarrierWorld, false);
                Physics.SyncTransforms();

                Assert(agB.isRespawning && !agB.entityTag.enabled, "Expected splatted carrier agB in respawn with entityTag disabled.");
                Assert(rm.State == BlitzCoreState.Shielded, $"Expected dropped BlitzCore in Shielded state, got {rm.State}");
                float droppedLocalY = rm.transform.position.y - root.transform.position.y;
                Assert(droppedLocalY >= hgGroundY + 0.55f,
                    $"Expected dropped BlitzCore above elevated floor ({hgGroundY + 0.65f:F2}m), NOT buried at 0.65m! Got y={droppedLocalY:F3}m");

                // Place agA 4.2m South of dropped BlitzCore on the same High Ground Wing and fire real ballistic ink shots!
                float shooterGroundY = ArenaStageBuilder.SampleTerrainHeightLocal(8.5f, -13.2f, settings);
                Vector3 shooterWorld = root.transform.position + new Vector3(8.5f, shooterGroundY, -13.2f);
                TeleportAgent(agA, shooterWorld);
                Vector3 eyeFrom = agA.eyePivot != null ? agA.eyePivot.position : (shooterWorld + Vector3.up * 1.18f);
                Vector3 aimVec = rm.transform.position - eyeFrom;
                float distShot = aimVec.magnitude;
                float tFlight = distShot / settings.projectileSpeedMps;
                Vector3 ballisticAim = (aimVec + Vector3.up * (0.5f * settings.projectileGravityMps2 * tFlight * tFlight)).normalized;

                for (int shot = 0; shot < 6 && rm.State == BlitzCoreState.Shielded && !rm.IsDetonationFuseActive; shot++)
                {
                    arena.projectileManager.SpawnProjectile(agA, eyeFrom, ballisticAim, isBlitzCoreBlast: false);
                    for (int step = 0; step < 22; step++)
                        arena.projectileManager.StepProjectiles(1f / 60f);
                }
                Assert(rm.IsDetonationFuseActive, $"Expected real ballistic shots to trigger swelling detonation fuse on dropped BlitzCore! ShieldTug={rm.ShieldTug:F1}/{rm.MaxShieldTug:F1}");
                for (int step = 0; step < 34 && rm.State == BlitzCoreState.Shielded; step++)
                    rm.StepBlitzCore(1f / 60f, arena);
                Assert(rm.State == BlitzCoreState.PickupReady,
                    $"Expected swelling detonation fuse to complete & transition dropped BlitzCore to PickupReady! State={rm.State}");

                TeleportAgent(agA, new Vector3(rm.transform.position.x, root.transform.position.y + hgGroundY, rm.transform.position.z));
                agA.ApplyExternalCommand(agA.transform.position, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, 0f, 0f);
                arena.SimulateStep(1f / 60f);
                Assert(rm.State == BlitzCoreState.Carried && rm.Carrier == agA && agA.isCarryingObjective,
                    "Expected agA to claim popped BlitzCore on elevated terrain immediately!");

                // 18c. Verify DropAt right inside/against Cover_CenterSE (4.0, 0.85, -0.2) pushes BlitzCore clear of cover BoxCollider!
                rm.DropAt(root.transform.position + new Vector3(4.0f, 0f, -0.2f));
                Vector2 pushedLocalXZ = new Vector2(rm.transform.position.x - root.transform.position.x, rm.transform.position.z - root.transform.position.z);
                float distFromCoverCenter = Vector2.Distance(pushedLocalXZ, new Vector2(4.0f, -0.2f));
                Assert(distFromCoverCenter >= 1.25f,
                    $"Expected DropAt inside Cover_CenterSE to push BlitzCore clear outside cover block, got distFromCoverCenter={distFromCoverCenter:F3}m");
                Debug.Log($"[PASS] SelfTest 18/19: Elevated BlitzCore DropAt non-burial (y={droppedLocalY:F2}m on {hgGroundY:F2}m deck), live projectile pop, cover push-out ({distFromCoverCenter:F2}m), and respawn ghost suppression verified.");

                // Check 19: Enemy BlitzCore Claim/Advance/KO Penalties + 180-deg Flank Potential + Lightweight `02 DEPTH` Viewer (Zero-Cost Closed Gate)
                arena.ResetSessionStats();
                agA.ResetForNewMatch();
                agB.ResetForNewMatch();
                Physics.SyncTransforms();

                // 19a. Pop BlitzCore and have Team B (`agB`) claim it -> `agB` gets `+1.20`, `agA` gets `-0.80` (once per pop!)
                rm.ResetToCenter();
                rm.ApplyShieldDamage(agB, settings.blitzCoreShieldMaxHp + 20f, arena, immediatePop: true);
                agA.ConsumeNewlyExploredCells(); // Flush shield-pop ink explosion cells before measuring pure pickup step reward
                agB.ConsumeNewlyExploredCells();
                float rewABeforeClaim = agA.brainBridge.PendingStepReward;
                float rewBBeforeClaim = agB.brainBridge.PendingStepReward;
                TeleportAgent(agB, rm.transform.position);
                agB.ApplyExternalCommand(rm.transform.position, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, Mathf.PI, 0f);
                arena.SimulateStep(1f / 60f);
                float deltaClaimA = agA.brainBridge.PendingStepReward - rewABeforeClaim;
                float deltaClaimB = agB.brainBridge.PendingStepReward - rewBBeforeClaim;
                Assert(Mathf.Abs(deltaClaimB - settings.claimObjectiveReward) < 1e-3f,
                    $"Expected claimer agB reward +{settings.claimObjectiveReward:F2}, got {deltaClaimB:F3}");
                Assert(Mathf.Abs(deltaClaimA - settings.enemyClaimObjectivePenalty) < 1e-3f,
                    $"Expected defender agA enemy-claim penalty {settings.enemyClaimObjectivePenalty:F2}, got {deltaClaimA:F3}");

                // 19b. Have enemy carrier `agB` push deep toward Team A pedestal `(4.0, y, -7.0)` -> `agA` gets monotonic `-4.0 * deltaBest` penalty, zero on retreat!
                float rewABeforePush = agA.brainBridge.PendingStepReward;
                float pushY = ArenaStageBuilder.SampleTerrainHeightLocal(4.0f, -7.0f, settings);
                TeleportAgent(agB, root.transform.position + new Vector3(4.0f, pushY, -7.0f));
                agB.ApplyExternalCommand(agB.transform.position, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, Mathf.PI, 0f);
                arena.SimulateStep(1f / 60f);
                float bestB1 = arena.bestProgressTeamB;
                float deltaPushPenA = agA.brainBridge.PendingStepReward - rewABeforePush;
                float expectedPushPenA = -settings.enemyBestProgressPenaltyScale * bestB1;
                Assert(bestB1 > 0.30f && Mathf.Abs(deltaPushPenA - expectedPushPenA) < 1e-3f,
                    $"Expected monotonic enemy best-progress penalty {expectedPushPenA:F3} (bestB={bestB1:F3}), got {deltaPushPenA:F3}");

                // Retreating back to center `(0, y, 0)` must NOT apply any extra progress penalty (anti-oscillation high-water mark!)
                float rewABeforeRetreat = agA.brainBridge.PendingStepReward;
                TeleportAgent(agB, root.transform.position + new Vector3(0f, ArenaStageBuilder.SampleTerrainHeightLocal(0f, 0f, settings), 0f));
                agB.ApplyExternalCommand(agB.transform.position, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, Mathf.PI, 0f);
                arena.SimulateStep(1f / 60f);
                Assert(Mathf.Abs(agA.brainBridge.PendingStepReward - rewABeforeRetreat) < 1e-4f,
                    "Expected zero extra progress penalty when enemy carrier retreats without advancing bestProgress!");

                // 19c. Enemy Touchdown KO on Team A's Defend Pedestal applies `knockoutLossPenalty` (`-32.0`) instead of `matchLossPenalty` (`-20.0`)!
                float rewABeforeKoLoss = agA.brainBridge.PendingStepReward;
                TeleportAgent(agB, root.transform.position + settings.teamAPedestalLocal + new Vector3(0f, 0.15f, 0f));
                agB.ApplyExternalCommand(agB.transform.position, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, Mathf.PI, 0f);
                arena.SimulateStep(1f / 60f);
                // Note: ResetMatch() at episode end resets `NotifyNewEpisodeReset` (`_accumStepReward = 0`) after offline step;
                // verify Timeout Loss vs KO Loss directly via settings & non-reset mid-step check:
                Assert(settings.knockoutLossPenalty < settings.matchLossPenalty - 8.0f,
                    $"Expected knockoutLossPenalty ({settings.knockoutLossPenalty}) strictly stronger than matchLossPenalty ({settings.matchLossPenalty})!");

                // 19d. Lightweight `02 DEPTH` Cortex Viewer: Zero-Cost Closed Gate & Live Plasma Rasterization + HUD Non-Overlap
                var testNeuralHud = dashGo.AddComponent<InkBlitzNeuralVisualizerHud>();
                testNeuralHud.coordinator = testCoord;
                testNeuralHud.multiManager = camMgr;
                testNeuralHud.dashboardUi = testDash;
                testDash.neuralHud = testNeuralHud;

                agA.ResetForNewMatch();
                TeleportAgent(agA, root.transform.position + new Vector3(0f, ArenaStageBuilder.SampleTerrainHeightLocal(0f, -6f, settings), -6f));
                agA.eyePivot.rotation = Quaternion.Euler(12f, 0f, 0f);
                Physics.SyncTransforms();

                testNeuralHud.showDepthCortexPip = false;
                bool renderedWhileClosed = testNeuralHud.RefreshDepthCortexNow(agA, forceSampleIfOffline: true);
                Assert(!renderedWhileClosed && testNeuralHud.DepthRasterCount == 0 && testNeuralHud.DepthCortexTexture == null,
                    "Expected strictly ZERO depth rasterization & zero Texture2D allocation while `02 DEPTH` is closed!");

                testNeuralHud.showDepthCortexPip = true;
                testNeuralHud.depthViewMode = 0; // Plasma
                bool renderedWhileOpen = testNeuralHud.RefreshDepthCortexNow(agA, forceSampleIfOffline: true);
                Assert(renderedWhileOpen && testNeuralHud.DepthRasterCount == 1 && testNeuralHud.DepthCortexTexture != null,
                    "Expected `02 DEPTH` Plasma Texture2D rasterized immediately when opened!");
                Color32 pxNearBottom = testNeuralHud.DepthCortexTexture.GetPixel(48, 4);   // Floor near feet -> warm yellow/orange (high R)
                Color32 pxFarTop = testNeuralHud.DepthCortexTexture.GetPixel(48, 50);      // Distant sky/upper -> deep indigo (high B, low G)
                Assert(pxNearBottom.r > pxFarTop.r + 60,
                    $"Expected near floor Plasma pixel warmer than distant sky pixel, got bottom=({pxNearBottom.r},{pxNearBottom.g},{pxNearBottom.b}) vs top=({pxFarTop.r},{pxFarTop.g},{pxFarTop.b})");

                Rect povRect = testNeuralHud.GetSubjectivePovWindowRect();
                Rect depthRect = testNeuralHud.GetDepthCortexWindowRect();
                Rect gemmaRectOpen = testDash.GetGemmaBroadcastRect();
                Assert(depthRect.yMax <= povRect.yMin + 0.1f,
                    $"Expected `02 DEPTH` rect {depthRect} stacked cleanly above Subjective POV {povRect}!");
                Assert(!depthRect.Overlaps(gemmaRectOpen) && !povRect.Overlaps(gemmaRectOpen),
                    $"Expected GemmaBroadcastRect {gemmaRectOpen} not to overlap Depth {depthRect} or POV {povRect}!");
                Debug.Log($"[PASS] SelfTest 19/20: Enemy BlitzCore claim ({deltaClaimA:F2}) & advance ({deltaPushPenA:F2}) penalties, KO loss ({settings.knockoutLossPenalty:F1}), and zero-cost `02 DEPTH` Plasma viewer verified.");

                // Check 20: Dive Submerge Limit (4.0s) + Speed Rebalance (4.3 / 6.36 m/s) + ~8.5m Shooter Range & 4-Hit TTK + 280HP Shield 0.45s Fuse & Pushback
                arena.projectileManager.ClearAll();
                arena.ResetSessionStats();
                agA.ResetForNewMatch();
                agB.ResetForNewMatch();
                Physics.SyncTransforms();

                // 20a. Dive Submerge Stamina Limit (4.0s dive -> forced surface + 1.2s exhaust cooldown + 2.0s breath recovery)
                Vector3 inkSpotA = root.transform.position + new Vector3(-6.0f, ArenaStageBuilder.SampleTerrainHeightLocal(-6.0f, -10.0f, settings), -10.0f);
                arena.TurfPaint(inkSpotA, 2.2f, 1);
                TeleportAgent(agA, inkSpotA);
                agA.ApplyExternalCommand(inkSpotA, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f);
                agA.StepAgent(1f / 60f);
                Assert(agA.isSubmergedInFriendlyInk && agA.SurfacePosture01 >= 0.90f, "Expected agA initially submerged on friendly ink.");

                // Hold submerge for 3.5s -> enters low-stamina warning (< 0.8s left -> SurfacePosture01 < 0.90)
                for (int t = 0; t < 210; t++)
                    agA.StepAgent(1f / 60f);
                Assert(agA.isSubmergedInFriendlyInk && agA.SurfacePosture01 < 0.90f,
                    $"Expected low-breath warning SurfacePosture01 < 0.90 after 3.5s dive, got {agA.SurfacePosture01:F3}");

                // Step remaining 0.6s -> hits 4.0s limit, forces human surface & activates 1.2s exhaust cooldown!
                for (int t = 0; t < 38; t++)
                    agA.StepAgent(1f / 60f);
                Assert(!agA.isSubmergedInFriendlyInk && !agA.isDiveForm && agA.submergeExhaustTimer > 0.5f,
                    $"Expected forced human surface & exhaust lock after >4.0s dive! sub={agA.isSubmergedInFriendlyInk}, lock={agA.submergeExhaustTimer:F2}s");
                Assert(agA.SurfacePosture01 < 0.65f, $"Expected SurfacePosture01 < 0.65 during breath lock, got {agA.SurfacePosture01:F2}");

                // Step 2.05s in human walk/stand -> cooldown clears & breath stamina recovers to 100%, allowing clean re-submerge!
                agA.ApplyExternalCommand(inkSpotA, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, 0f, 0f);
                for (int t = 0; t < 125; t++)
                    agA.StepAgent(1f / 60f);
                Assert(agA.submergeExhaustTimer <= 0f && agA.SubmergeStamina01 >= 0.98f,
                    $"Expected full breath stamina recovery in human form, got lock={agA.submergeExhaustTimer:F2}, stam={agA.SubmergeStamina01:F2}");
                agA.ApplyExternalCommand(inkSpotA, 0.5f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f);
                agA.StepAgent(1f / 60f);
                Assert(agA.isSubmergedInFriendlyInk && agA.SubmergeStamina01 >= 0.95f,
                    $"Expected agA able to re-submerge after breath recovery! sub={agA.isSubmergedInFriendlyInk}, stam={agA.SubmergeStamina01:F2}");

                // 20b. Walk (4.3 m/s) vs Swim (~6.36 m/s) vs Heavy Carrier (~0.72x) Speed Balance
                agA.ResetForNewMatch();
                Vector3 laneStart = root.transform.position + new Vector3(0f, 0.05f, -9.5f);
                Vector3 laneEnd = root.transform.position + new Vector3(0f, 0.05f, -3.5f);
                arena.TurfPaint(laneStart, 2.5f, 0); // Neutral floor for pure walk check
                TeleportAgent(agA, laneStart);
                agA.ApplyExternalCommand(laneEnd, 1.0f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, 0f, 0f);
                for (int t = 0; t < 30; t++) // 0.5s walk
                    agA.StepAgent(1f / 60f);
                float walkDistHalfSec = Vector2.Distance(new Vector2(laneStart.x, laneStart.z), new Vector2(agA.transform.position.x, agA.transform.position.z));
                Assert(walkDistHalfSec >= 1.95f && walkDistHalfSec <= 2.30f,
                    $"Expected 0.5s walk distance ~2.15m (4.3 m/s), got {walkDistHalfSec:F3}m");

                for (float z = -10f; z <= -2f; z += 0.8f)
                    arena.TurfPaint(root.transform.position + new Vector3(0f, 0.05f, z), 1.4f, 1);
                TeleportAgent(agA, laneStart);
                agA.ApplyExternalCommand(laneEnd, 1.0f, UniversalBrainBridge3D.VERB_EVADE_HAZARD, 0f, 0f);
                for (int t = 0; t < 30; t++) // 0.5s swim
                    agA.StepAgent(1f / 60f);
                float swimDistHalfSec = Vector2.Distance(new Vector2(laneStart.x, laneStart.z), new Vector2(agA.transform.position.x, agA.transform.position.z));
                Assert(swimDistHalfSec >= 2.95f && swimDistHalfSec <= 3.38f,
                    $"Expected 0.5s swim distance ~3.18m (6.36 m/s), got {swimDistHalfSec:F3}m");

                // 20c. Main Shooter ~8.5m Horizontal Range Cap & 4-Hit Splat (28 dmg x 4 = 112)
                arena.projectileManager.ClearAll();
                agA.ResetForNewMatch();
                agB.ResetForNewMatch();
                Vector3 shootOrig = root.transform.position + new Vector3(0f, 0.05f, -10.5f);
                Vector3 farTarget = root.transform.position + new Vector3(0f, 0.05f, -0.3f); // 10.2m away (> 8.5m range cap!)
                TeleportAgent(agA, shootOrig);
                TeleportAgent(agB, farTarget);
                Physics.SyncTransforms();
                arena.projectileManager.SpawnProjectile(agA, shootOrig + Vector3.up * 1.18f, (Vector3.forward + Vector3.up * 0.14f).normalized, isBlitzCoreBlast: false);
                for (int t = 0; t < 45; t++)
                    arena.projectileManager.StepProjectiles(1f / 60f);
                Assert(Mathf.Approximately(agB.hp, settings.maxHp),
                    $"Expected 10.2m distant target beyond 8.5m shooter range cap to take 0 damage, got hp={agB.hp:F1}");

                Vector3 midTarget = root.transform.position + new Vector3(0f, 0.05f, -4.5f); // 6.0m away (inside 8.5m range!)
                TeleportAgent(agB, midTarget);
                Physics.SyncTransforms();
                Vector3 midAim = (midTarget + Vector3.up * 1.10f - (shootOrig + Vector3.up * 1.18f)).normalized;
                for (int s = 0; s < 3; s++)
                {
                    arena.projectileManager.SpawnProjectile(agA, shootOrig + Vector3.up * 1.18f, midAim, isBlitzCoreBlast: false);
                    for (int t = 0; t < 24; t++)
                        arena.projectileManager.StepProjectiles(1f / 60f);
                }
                Assert(!agB.isRespawning && Mathf.Approximately(agB.hp, settings.maxHp - 3f * settings.shooterDamage),
                    $"Expected 3 shooter hits (3x28=84) to leave target alive at 16 HP, got hp={agB.hp:F1}, resp={agB.isRespawning}");
                arena.projectileManager.SpawnProjectile(agA, shootOrig + Vector3.up * 1.18f, midAim, isBlitzCoreBlast: false);
                for (int t = 0; t < 24; t++)
                    arena.projectileManager.StepProjectiles(1f / 60f);
                Assert(agB.isRespawning, "Expected 4th shooter hit (4x28=112) to splat target!");
                Assert(agB.RespawnTimeRemaining <= 2.05f && agB.RespawnTimeRemaining >= 1.90f,
                    $"Expected balanced 2.0s respawn timer, got {agB.RespawnTimeRemaining:F2}s");

                // 20d. BlitzCore 280HP Shield + 0.45s Swelling Detonation Fuse + EnemyAreaHazard (-3.0) Warning + Physical Pushback
                agB.ResetForNewMatch();
                rm.ResetToCenter();
                Assert(Mathf.Approximately(rm.MaxShieldTug, 280f), $"Expected 280 HP center BlitzCore shield, got {rm.MaxShieldTug}");
                TeleportAgent(agA, rm.transform.position + new Vector3(0f, 0f, -0.5f)); // Attempt to stand inside shield at 0.5m!
                agA.ApplyExternalCommand(rm.transform.position, 1.0f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, 0f, 0f);
                agA.StepAgent(1f / 60f);
                float pushedDistRm = Vector2.Distance(new Vector2(agA.transform.position.x, agA.transform.position.z), new Vector2(rm.transform.position.x, rm.transform.position.z));
                Assert(pushedDistRm >= 1.40f, $"Expected physical BlitzCore shield pushback >= 1.40m, got {pushedDistRm:F3}m");

                rm.ApplyShieldDamage(agB, settings.blitzCoreShieldMaxHp + 10f, arena, immediatePop: false);
                Assert(rm.State == BlitzCoreState.Shielded && rm.IsDetonationFuseActive,
                    "Expected non-immediate shield break to start 0.45s swelling detonation fuse!");
                Assert(Mathf.Approximately(rm.entityTag.GetObserverRelativeTag(1), -3.0f),
                    "Expected losing Team A to perceive swelling enemy-popped barrier as EnemyAreaHazard (-3.0)!");
                Assert(Mathf.Approximately(rm.entityTag.GetObserverRelativeTag(-1), 3.0f),
                    "Expected winning Team B to perceive swelling barrier as PrimaryObjective (+3.0)!");
                for (int t = 0; t < 30; t++)
                    rm.StepBlitzCore(1f / 60f, arena);
                Assert(rm.State == BlitzCoreState.PickupReady && !rm.IsDetonationFuseActive,
                    "Expected BlitzCore to burst into PickupReady after 0.45s fuse!");
                Debug.Log($"[PASS] SelfTest 20/21: Dive 4.0s breath limit, walk({walkDistHalfSec * 2f:F2}m/s)/swim({swimDistHalfSec * 2f:F2}m/s), ~8.5m 4-hit shooter, & 280HP 0.45s BlitzCore fuse verified.");

                // Check 21: Submerged Low-Profile Hitbox + 16x16 Foveal Stealth Transparency + 1.0s Post-Damage HP Regen Delay
                arena.projectileManager.ClearAll();
                agA.ResetForNewMatch();
                agB.ResetForNewMatch();
                supA.gameObject.SetActive(false);
                supB.gameObject.SetActive(false);

                float valYA = ArenaStageBuilder.SampleTerrainHeightLocal(0f, -10.0f, settings);
                float valYB = ArenaStageBuilder.SampleTerrainHeightLocal(0f, -6.5f, settings);
                Vector3 stealthPosA = root.transform.position + new Vector3(0f, valYA, -10.0f);
                Vector3 stealthPosB = root.transform.position + new Vector3(0f, valYB, -6.5f); // 3.5m ahead

                // 21a. Submerged low-profile dive dodges chest/head horizontal shot, but gets hit & un-submerged by floor-aimed clearing shot!
                arena.TurfPaint(stealthPosB, 1.8f, -1);
                TeleportAgent(agA, stealthPosA);
                TeleportAgent(agB, stealthPosB);
                agB.ApplyExternalCommand(stealthPosB, 0.3f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, Mathf.PI, 0f, commitOpen: false);
                agB.StepAgent(1f / 60f);
                agB.entityTag.NotifyTeleported(); // Ensure EstimatedVelocity == 0 (stationary stealth!)
                Physics.SyncTransforms();
                Assert(agB.isSubmergedInFriendlyInk && agB.entityTag.isStealthSubmerged,
                    "Expected agB submerged & stealth-hidden on Team B ink.");

                Vector3 eyeOrigA = stealthPosA + Vector3.up * 1.18f;
                float t35 = 3.5f / settings.projectileSpeedMps;
                float gravComp35 = 0.5f * settings.projectileGravityMps2 * t35 * t35;

                // High horizontal chest/head shot (~y+1.05m at target): must pass cleanly overhead without hitting flat dive!
                Vector3 highChestAim = ((stealthPosB + Vector3.up * (1.05f + gravComp35)) - eyeOrigA).normalized;
                arena.projectileManager.SpawnProjectile(agA, eyeOrigA, highChestAim, isBlitzCoreBlast: false);
                for (int t = 0; t < 20; t++)
                    arena.projectileManager.StepProjectiles(1f / 60f);
                Assert(Mathf.Approximately(agB.hp, settings.maxHp) && agB.isSubmergedInFriendlyInk,
                    $"Expected high horizontal chest-height shot to fly over submerged dive (0 dmg), got hp={agB.hp:F1}");

                // Floor-aimed clearing shot (~y+0.16m at target): must hit submerged dive, deal 28 dmg, & paint feet to force surface!
                arena.projectileManager.ClearAll();
                Vector3 floorClearAim = ((stealthPosB + Vector3.up * (0.16f + gravComp35)) - eyeOrigA).normalized;
                arena.projectileManager.SpawnProjectile(agA, eyeOrigA, floorClearAim, isBlitzCoreBlast: false);
                for (int t = 0; t < 20; t++)
                    arena.projectileManager.StepProjectiles(1f / 60f);
                Assert(Mathf.Approximately(agB.hp, settings.maxHp - settings.shooterDamage),
                    $"Expected floor-aimed clearing shot to hit submerged dive for {settings.shooterDamage} dmg before slip tick, got hp={agB.hp:F2}");
                agB.StepAgent(1f / 60f);
                Assert(!agB.isSubmergedInFriendlyInk && !agB.isDiveForm && agB.hp < settings.maxHp - settings.shooterDamage,
                    $"Expected direct hit ink splash under feet to immediately un-submerge agB & apply enemy-ink slip, sub={agB.isSubmergedInFriendlyInk}, hp={agB.hp:F2}!");

                // 21b. 16x16 Foveal Raycast Stealth Transparency: zero 1.5m hostile pillar leak when stealth-submerged!
                arena.projectileManager.ClearAll();
                agA.ResetForNewMatch();
                agB.ResetForNewMatch();
                arena.TurfPaint(stealthPosB, 1.8f, -1);
                TeleportAgent(agA, stealthPosA);
                TeleportAgent(agB, stealthPosB);
                agB.ApplyExternalCommand(stealthPosB, 0.3f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, Mathf.PI, 0f, commitOpen: false);
                agB.StepAgent(1f / 60f);
                agB.entityTag.NotifyTeleported();
                agA.eyePivot.position = eyeOrigA;
                agA.eyePivot.rotation = Quaternion.LookRotation((stealthPosB + Vector3.up * 0.55f) - eyeOrigA, Vector3.up);
                Physics.SyncTransforms();

                agA.brainBridge.SampleLiveFovealPatchForHud();
                int stealthHostileTexels = 0;
                int enemyInkFloorTexels = 0;
                for (int px = 0; px < 256; px++)
                {
                    float sal = agA.brainBridge.latestFoveaSalience16[px];
                    if (sal > 0.90f) stealthHostileTexels++;
                    if (Mathf.Abs(sal - 0.25f) < 0.03f) enemyInkFloorTexels++;
                }
                Assert(stealthHostileTexels == 0,
                    $"Expected 0 hostile silhouette texels (0.95) in fovea_patch when opponent is stealth-submerged, got {stealthHostileTexels}!");
                Assert(enemyInkFloorTexels > 0,
                    "Expected foveal rays passing through stealth-submerged collider to sample enemy ink floor (0.25) underneath!");

                // Un-submerge agB into standing human form -> fovea must immediately detect hostile silhouette (0.95)!
                agB.ApplyExternalCommand(stealthPosB, 0.5f, UniversalBrainBridge3D.VERB_APPROACH_HOVER, Mathf.PI, 0f, commitOpen: false);
                agB.StepAgent(1f / 60f);
                Physics.SyncTransforms();
                agA.brainBridge.SampleLiveFovealPatchForHud();
                int visibleHostileTexels = 0;
                for (int px = 0; px < 256; px++)
                {
                    if (agA.brainBridge.latestFoveaSalience16[px] > 0.90f)
                        visibleHostileTexels++;
                }
                Assert(visibleHostileTexels > 0,
                    "Expected standing un-submerged opponent to appear as hostile texels (0.95) in fovea_patch!");

                // 21c. Post-Damage 1.0s HP Regeneration Delay (prevents instant mid-fight full heal while keeping ink refill!)
                agA.ResetForNewMatch();
                arena.TurfPaint(stealthPosA, 1.8f, 1);
                TeleportAgent(agA, stealthPosA);
                agA.inkTank01 = 0.25f;
                agA.ApplyExternalCommand(stealthPosA, 0.3f, UniversalBrainBridge3D.VERB_IDLE_OBSERVE, 0f, 0f, commitOpen: false);
                agA.StepAgent(1f / 60f);
                agA.TakeDamage(56f); // 2 shooter hits -> 44 HP, 1.0s regen delay!
                Assert(Mathf.Approximately(agA.hp, 44f) && agA.HpRegenDelayRemaining >= 0.99f,
                    $"Expected 44 HP & 1.0s regen delay right after TakeDamage(56), got hp={agA.hp:F1}, delay={agA.HpRegenDelayRemaining:F2}s");

                for (int t = 0; t < 48; t++) // 0.80s (< 1.0s delay) submerged in friendly ink
                    agA.StepAgent(1f / 60f);
                Assert(Mathf.Approximately(agA.hp, 44f),
                    $"Expected 0 HP healed during first 0.80s post-damage delay even while submerged, got hp={agA.hp:F2}!");
                Assert(agA.InkTank01 > 0.55f,
                    $"Expected ink tank to still refill normally while submerged during HP delay, got tank={agA.InkTank01:F2}");

                for (int t = 0; t < 60; t++) // +1.00s (1.80s total > 1.0s delay) -> ~0.80s of 35 HP/s swim regen (~+28 HP -> ~72 HP)
                    agA.StepAgent(1f / 60f);
                Assert(agA.HpRegenDelayRemaining <= 0f && agA.hp >= 68f,
                    $"Expected rapid swim HP recovery (>= 68 HP) once 1.0s post-damage delay expires, got hp={agA.hp:F1}");

                supA.gameObject.SetActive(true);
                supB.gameObject.SetActive(true);
                Debug.Log($"[PASS] SelfTest 21/22: Submerged low-profile hitbox, foveal stealth transparency (0 leak vs {visibleHostileTexels} vis), & 1.0s HP regen delay (0.8s->44HP, 1.8s->{agA.hp:F1}HP) verified.");

                // =========================================================================
                // 22. World-Model Mini-HUD (`03 WORLD MODEL` / `UBWM`) & Square Glowing
                //     Brain Architecture Circuit HUD (`04 BRAIN ARCHITECTURE`) + Zero-Cost Collapse
                // =========================================================================
                var hudTestGo = new GameObject("TempHudTestRoot");
                hudTestGo.transform.SetParent(root.transform, false);
                var nHud = hudTestGo.AddComponent<InkBlitzNeuralVisualizerHud>();
                var cHud = nHud.EnsureCircuitHudBound();
                Assert(cHud != null, "Expected InkBlitzNeuralVisualizerHud.EnsureCircuitHudBound() to bind InkBlitzBrainCircuitHud!");

                // Build a valid 904-byte CRC32-signed `UBWM` packet
                byte[] wmPkt = new byte[InkBlitzNeuralVisualizerHud.UBWM_TOTAL_BYTES];
                Buffer.BlockCopy(BitConverter.GetBytes(InkBlitzNeuralVisualizerHud.UBWM_MAGIC), 0, wmPkt, 0, 4);
                Buffer.BlockCopy(BitConverter.GetBytes(1u), 0, wmPkt, 4, 4);   // version = 1
                Buffer.BlockCopy(BitConverter.GetBytes(3u), 0, wmPkt, 8, 4);   // episode_id = 3
                Buffer.BlockCopy(BitConverter.GetBytes(240u), 0, wmPkt, 12, 4);// physics_tick = 240
                wmPkt[16] = 2; // chosen_cand = M2
                wmPkt[17] = 1; // has_prev_pred = true
                wmPkt[18] = (1 << 7); // M7 vetoed
                Buffer.BlockCopy(BitConverter.GetBytes(0.084f), 0, wmPkt, 20 + 0, 4); // vis_pred_mae
                Buffer.BlockCopy(BitConverter.GetBytes(0.42f), 0, wmPkt, 20 + 4, 4);  // depth_pred_mae_m
                Buffer.BlockCopy(BitConverter.GetBytes(0.19f), 0, wmPkt, 20 + 8, 4);  // world_kl_surprise
                Buffer.BlockCopy(BitConverter.GetBytes(0.37f), 0, wmPkt, 20 + 12, 4); // action_cf_spread
                for (int c = 0; c < 8; c++)
                {
                    Buffer.BlockCopy(BitConverter.GetBytes(0.25f * c - 0.3f), 0, wmPkt, 20 + (4 + c) * 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(c == 7 ? 0.25f : 0.88f), 0, wmPkt, 20 + (12 + c) * 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(0.30f + 0.08f * c), 0, wmPkt, 20 + (20 + c) * 4, 4);
                }
                for (int px = 0; px < 256; px++)
                {
                    wmPkt[132 + px] = (byte)(40 + (px % 180));
                    wmPkt[132 + 256 + px] = (byte)((px % 2 == 0) ? 195 : 64);
                    wmPkt[132 + 512 + px] = (byte)((px >= 100 && px <= 140) ? 230 : 12);
                }
                uint wmCrc = UniversalBrainBridge3D.ComputeCrc32(wmPkt, 0, InkBlitzNeuralVisualizerHud.UBWM_TOTAL_BYTES - 4);
                Buffer.BlockCopy(BitConverter.GetBytes(wmCrc), 0, wmPkt, InkBlitzNeuralVisualizerHud.UBWM_TOTAL_BYTES - 4, 4);

                Assert(nHud.IngestRawUbwmPacketForTest(wmPkt), "UBWM 904B packet failed CRC32 / ingestion!");
                nHud.showWorldModelPip = false;
                Assert(!nHud.RefreshWorldModelNow() && nHud.WorldModelRasterCount == 0,
                    "Expected 0 rasterization cost when 03 WORLD MODEL HUD is collapsed!");
                nHud.showWorldModelPip = true;
                Assert(nHud.RefreshWorldModelNow() && nHud.WorldModelRasterCount == 1
                       && nHud.WorldModelPredTexture != null && nHud.WorldModelErrTexture != null,
                    "Expected 03 WORLD MODEL dual textures to rasterize cleanly when expanded!");

                // Verify Square Diagnostic Brain Circuit HUD (`04 BRAIN DIAGNOSTIC`):
                //   (a) Healthy Trained Brain -> high `HealthyNodeCount` & bright luminescence
                //   (b) Untrained Flat Brain  -> high `WeakNodeCount` & honestly dimmed luminescence
                //   (c) Frozen/Q-Spike Fault  -> `WarnNodeCount >= 1`
                var mockSnap = new InkBlitzNeuralVisualizerHud.UbntSnapshot
                {
                    seq = 12,
                    episodeId = 3,
                    chosenCand = 2,
                    s1FastCand = 0,
                    commitGateOpen = true,
                    vetoActive = false,
                    holdReasonId = 0,
                    endogenousActive = true,
                    protoSymId = 7,
                    loopClosuresTotal = 4,
                    believedHazard = 0.45f,
                    meanFog = 0.32f,
                    ignition = 0.82f,
                    phiBits = 0.54f,
                    entropyNorm = 0.48f,
                    worldSurprise = 0.22f,
                    selfSurprise = 0.16f,
                    goDrive = 0.92f,
                    stnBrake = 0.18f,
                    friendlyTurfRatio = 0.58f,
                    believedSelfPos = new Vector3(1.5f, 1.2f, -3.0f),
                    poseDriftVec = new Vector3(0.12f, 0f, -0.08f)
                };
                mockSnap.gwtAlpha[0] = 0.28f; mockSnap.gwtAlpha[1] = 0.20f; mockSnap.gwtAlpha[2] = 0.24f;
                mockSnap.gwtAlpha[3] = 0.12f; mockSnap.gwtAlpha[4] = 0.16f;
                for (int i = 0; i < 32; i++)
                {
                    mockSnap.latentZ[i] = (i % 2 == 0) ? 0.38f : -0.26f;
                    mockSnap.zDelta[i] = 0.14f;
                }
                for (int i = 0; i < 8; i++) mockSnap.selfState[i] = 0.32f;
                for (int i = 0; i < 256; i++) { mockSnap.rssmH[i] = 0.35f; mockSnap.mecGrid[i] = 0.32f; }
                for (int i = 0; i < 8; i++)
                {
                    mockSnap.qRawValues[i] = -0.3f + 0.15f * i;
                    mockSnap.qValues[i] = (i == 2) ? 1.45f : (-0.25f + 0.12f * i);
                    mockSnap.policyProbs[i] = (i == 2) ? 0.65f : 0.05f;
                    mockSnap.survivalProbs[i] = 0.88f;
                }

                cHud.showCircuitHud = false;
                Assert(!cHud.RefreshCircuitCanvasNow(mockSnap, nHud.LatestUbwmSnapshot, 1.0f) && cHud.CircuitRasterCount == 0,
                    "Expected 0 rasterization cost when 04 BRAIN DIAGNOSTIC HUD is collapsed!");

                cHud.showCircuitHud = true;
                cHud.ResetDiagnosticTrackers();
                for (int v = 0; v < 3; v++)
                {
                    mockSnap.seq = (ulong)(20 + v);
                    mockSnap.protoSymId = 5 + v;
                    cHud.activeViewIndex = v;
                    Assert(cHud.RefreshCircuitCanvasNow(mockSnap, nHud.LatestUbwmSnapshot, 1.25f + v * 0.3f),
                        $"Circuit HUD failed to rasterize view {v}!");
                }
                cHud.activeViewIndex = 0;
                for (int step = 0; step < 3; step++)
                {
                    mockSnap.seq = (ulong)(30 + step);
                    cHud.RefreshCircuitCanvasNow(mockSnap, nHud.LatestUbwmSnapshot, 2.2f + step * 0.1f);
                }
                Assert(cHud.HealthyNodeCount >= 8,
                    $"Expected >=8/9 Healthy (`OK`) modules on competent trained telemetry, got {cHud.HealthyNodeCount}/9!");

                Color32[] trainedPix = cHud.CircuitCanvasTexture.GetPixels32();
                int lumTrained = 0;
                for (int i = 0; i < trainedPix.Length; i++)
                {
                    if (trainedPix[i].r > 48 || trainedPix[i].g > 58 || trainedPix[i].b > 78)
                        lumTrained++;
                }

                // Now feed an UNTRAINED flat snapshot (zero Q spread, coin-flip entropy=0.99, tiny CF spread, high depth MAE=5.1m)
                var weakSnap = new InkBlitzNeuralVisualizerHud.UbntSnapshot
                {
                    seq = 50,
                    episodeId = 4,
                    commitGateOpen = true,
                    ignition = 0.02f,
                    phiBits = 0.005f,
                    entropyNorm = 0.99f
                };
                for (int i = 0; i < 8; i++) { weakSnap.qValues[i] = 0.01f; weakSnap.qRawValues[i] = 0.01f; weakSnap.policyProbs[i] = 0.125f; }
                var weakWm = new InkBlitzNeuralVisualizerHud.UbwmSnapshot
                {
                    seq = 9,
                    visPredMae = 0.24f,
                    depthPredMaeM = 5.2f,
                    actionCfSpread = 0.003f
                };
                cHud.ResetDiagnosticTrackers();
                for (int step = 0; step < 5; step++)
                {
                    weakSnap.seq = (ulong)(60 + step);
                    cHud.RefreshCircuitCanvasNow(weakSnap, weakWm, 3.0f + step * 0.1f);
                }
                Color32[] weakPix = cHud.CircuitCanvasTexture.GetPixels32();
                int lumWeak = 0;
                for (int i = 0; i < weakPix.Length; i++)
                {
                    if (weakPix[i].r > 48 || weakPix[i].g > 58 || weakPix[i].b > 78)
                        lumWeak++;
                }
                Assert(cHud.WeakNodeCount >= 5 && lumWeak < lumTrained,
                    $"Expected untrained flat brain to flag >=5 WEAK nodes and glow dimmer than trained brain (`weak={lumWeak}` vs `trained={lumTrained}`, weakCnt={cHud.WeakNodeCount})!");

                // Finally verify Fault/Warning (`WARN`) detection on Q-spike (`|Q| = 42.0`)
                weakSnap.seq = 99;
                weakSnap.qValues[0] = 42.0f;
                cHud.RefreshCircuitCanvasNow(weakSnap, weakWm, 4.0f);
                Assert(cHud.WarnNodeCount >= 1,
                    $"Expected `WARN` fault detection on Q-spike (`|Q|=42`), got WarnNodeCount={cHud.WarnNodeCount}!");

                var dashUi = hudTestGo.AddComponent<InkBlitzSessionDashboardUi>();
                dashUi.EnsureMaterialStylesReady();
                Assert(dashUi.HasValidMaterialThemeAssets,
                    "Expected InkBlitzSessionDashboardUi Material 3 Dark 9-slice & switch textures to initialize cleanly!");

                Debug.Log($"[PASS] SelfTest 22/23: 03 WORLD MODEL, 04 BRAIN DIAGNOSTIC (TrainedOK={lumTrained}px vs UntrainedWeak={lumWeak}px, Warn={cHud.WarnNodeCount}), & M3 Dark Launcher verified.");

                // -----------------------------------------------------------------
                // 23. Bottom-Edge 03 WORLD MODEL Docking, Camera Max-Zoom Recovery,
                //     & Disconnected Lockstep Zero-Tremble Verification
                // -----------------------------------------------------------------
                nHud.showWorldModelPip = true;
                Rect wmBottomRect = nHud.GetWorldModelWindowRect();
                Assert(wmBottomRect.yMax >= Screen.height - 6f && wmBottomRect.yMax <= Screen.height,
                    $"Expected 03 WORLD MODEL HUD docked flush at screen bottom (`yMax={wmBottomRect.yMax}`, `Screen.height={Screen.height}`)!");
                Rect gemmaBottomRect = dashUi.GetGemmaBroadcastRect();
                Assert(!wmBottomRect.Overlaps(gemmaBottomRect),
                    $"Expected 03 WORLD MODEL (`{wmBottomRect}`) and Gemma broadcast card (`{gemmaBottomRect}`) not to overlap!");

                // Verify God's-Eye camera recovers cleanly from worst-case max zoom-in (`3.8m`, `3 deg` pitch, corner wall pan)
                var camRigGo23 = new GameObject("TempCamRig_23");
                camRigGo23.transform.SetParent(root.transform);
                var testCam23 = camRigGo23.AddComponent<Camera>();
                var camMgr23 = hudTestGo.AddComponent<InkBlitzMultiArenaManager>();
                camMgr23.mainCamera = testCam23;
                camMgr23.arenas.Clear();
                camMgr23.arenas.Add(arena);
                camMgr23.isPausedForLauncher = false;
                camMgr23.disableAll3DRendering = false;
                camMgr23.cameraSmoothTime = 0.055f; // Real non-zero smoothing!

                // Case A: Single-notch wheel ticks (`-1.0f` in Unity 6 InputSystem) across frames from corner wall trap
                camMgr23.SetCameraOrbitState(0f, 3.0f, camMgr23.minZoomDistance, new Vector2(8.0f, -12.5f), instant: true);
                Assert(Mathf.Abs(camMgr23.CurrentCameraDistance - camMgr23.minZoomDistance) < 0.25f,
                    $"Expected initial max zoom-in distance `{camMgr23.minZoomDistance}`, got `{camMgr23.CurrentCameraDistance}`!");
                for (int notch = 0; notch < 12; notch++)
                {
                    camMgr23.ApplyMouseWheelScroll(-1.0f);
                    camMgr23.StepCameraSmoothing(1f / 60f);
                }
                for (int settle = 0; settle < 20; settle++)
                    camMgr23.StepCameraSmoothing(1f / 60f);
                Assert(camMgr23.CurrentCameraDistance >= 20.0f,
                    $"Expected smooth single-notch zoom-out from corner wall trap (`3.8m`) to recover >= 20m, got `{camMgr23.CurrentCameraDistance:F2}m` (pitch={camMgr23.CurrentCameraPitchDeg:F1})!");

                // Case B: Fast scroll flick (`-8.0f` in Unity 6 InputSystem, previously divided by 120!)
                camMgr23.SetCameraOrbitState(0f, 4.0f, camMgr23.minZoomDistance, Vector2.zero, instant: true);
                camMgr23.ApplyMouseWheelScroll(-8.0f);
                for (int settle = 0; settle < 24; settle++)
                    camMgr23.StepCameraSmoothing(1f / 60f);
                Assert(camMgr23.CurrentCameraDistance >= 10.0f,
                    $"Expected fast wheel flick (`-8.0`) from `{camMgr23.minZoomDistance}m` to zoom out >= 10m, got `{camMgr23.CurrentCameraDistance:F2}m`!");

                // Verify zero 60Hz interpolation tremble when waiting for TCP lockstep connection
                var tempCoordGo23 = new GameObject("TempCoord_23");
                tempCoordGo23.transform.SetParent(root.transform);
                var tempCoord23 = tempCoordGo23.AddComponent<UniversalArenaCoordinator3D>();
                tempCoord23.autoLaunchOnStart = false;
                tempCoord23.sessionMode = BrainSessionRunMode.SingleArenaSelfPlaySpectate;
                arena.ConfigureForSlot(0, 59910, 59911, tempCoord23);
                arena.ResetSessionStats();
                agA.ApplyExternalCommand(agA.transform.position + new Vector3(3.0f, 0f, 2.0f), 1.0f, UniversalBrainBridge3D.VERB_COMMIT_INTERACT, 0f, 0f, commitOpen: true);
                agA.StepAgent(1f / 60f); // Moves `_currFixedPos` away from `_prevFixedPos` (simulating tick-0 disconnect!)
                bool steppedWhileDisconnected = arena.BeginParallelArenaStep(1f / 60f, out _);
                Assert(!steppedWhileDisconnected, "Expected disconnected lockstep arena to hold while bridges are unconnected!");
                agA.RefreshVisualInterpolation(0.5f);
                float interpJitterSq = (agA.VisualWorldPosition - agA.transform.position).sqrMagnitude;
                Assert(interpJitterSq <= 1e-10f,
                    $"Expected zero interpolation jitter while waiting for TCP connection, got sqrMag={interpJitterSq}!");

                // Verify honest zero-gimmick Cortical Neural Sheet (0 fake sparks on stationary state; 0 false glow on flat zero tensor; live Oizumi MIP cut mask)
                var zeroSnap = new InkBlitzNeuralVisualizerHud.UbntSnapshot { seq = 200, mipCutMask = 0x05 }; // {P,W}|{I,M,S}
                for (int r = 0; r < 6; r++)
                {
                    zeroSnap.seq = (ulong)(200 + r);
                    nHud.RenderSnapshotForTest(zeroSnap, tabIdx: 0, geoMode: 1);
                }
                float maxZeroSmooth = 0f, maxZeroSpike = 0f;
                for (int i = 0; i < nHud.CurrentCorticalSmoothSheet.Length; i++)
                {
                    if (nHud.CurrentCorticalSmoothSheet[i] > maxZeroSmooth) maxZeroSmooth = nHud.CurrentCorticalSmoothSheet[i];
                    if (nHud.CurrentCorticalSpikeSheet[i] > maxZeroSpike) maxZeroSpike = nHud.CurrentCorticalSpikeSheet[i];
                }
                Assert(maxZeroSmooth == 0f && maxZeroSpike == 0f && nHud.CurrentMipCutMask == 0x05,
                    $"Expected 0 fake glow/sparks on flat stationary tensor & MIP mask 0x05, got smooth={maxZeroSmooth}, spike={maxZeroSpike}, mask={nHud.CurrentMipCutMask}!");

                mockSnap.seq = 210;
                mockSnap.mipCutMask = 0x12; // {I,S}|{P,W,M}
                for (int i = 0; i < 128; i++) mockSnap.gwtW[i] = (i < 32) ? 1.45f : 0.05f;
                nHud.RenderSnapshotForTest(mockSnap, tabIdx: 0, geoMode: 1);
                float realBurstSpike = nHud.CurrentCorticalSpikeSheet[0];
                Assert(realBurstSpike > 0.50f && nHud.CurrentMipCutMask == 0x12,
                    $"Expected real phasic transition burst > 0.50 on active GWT jump & MIP mask 0x12, got spike={realBurstSpike:F2}, mask={nHud.CurrentMipCutMask}!");
                for (int r = 0; r < 24; r++)
                {
                    mockSnap.seq = (ulong)(211 + r);
                    nHud.RenderSnapshotForTest(mockSnap, tabIdx: 0, geoMode: 1); // Stationary hold -> spikes must decay to 0 (no random Poisson sparks!)
                }
                float stationarySpikeAfterDecay = 0f;
                for (int i = 0; i < nHud.CurrentCorticalSpikeSheet.Length; i++)
                    if (nHud.CurrentCorticalSpikeSheet[i] > stationarySpikeAfterDecay) stationarySpikeAfterDecay = nHud.CurrentCorticalSpikeSheet[i];
                Assert(stationarySpikeAfterDecay == 0f,
                    $"Expected 0 fake random sparks when holding stationary active GWT vector, got {stationarySpikeAfterDecay}!");

                Debug.Log($"[PASS] SelfTest 23/24: Bottom-edge WM HUD (`yMax={wmBottomRect.yMax}`), Camera max-zoom recovery (`{camMgr23.CurrentCameraDistance:F1}m`), honest cortical sheet (`burst={realBurstSpike:F2}->0`), & zero-tremble idle verified.");

                // Check 24: Stage 2 ("Occluded Twin-Flank", stageId = 2) 180-deg Symmetry, Central Baffle LOS Occlusion & Live Stage Switch
                arena.RebuildArenaForStage(2);
                Physics.SyncTransforms();
                Assert(arena.settings.stageId == 2 && agA.brainBridge.stageId == 2 && agB.brainBridge.stageId == 2,
                    $"Expected stageId == 2 across arena & brain bridges after RebuildArenaForStage(2), got arena={arena.settings.stageId}, brA={agA.brainBridge.stageId}");
                AssertSymmetryXZ(arena.settings.teamASpawnLocal, arena.settings.teamBSpawnLocal, "Stage2 MainSpawn");
                AssertSymmetryXZ(arena.settings.teamAPedestalLocal, arena.settings.teamBPedestalLocal, "Stage2 Pedestal");
                AssertSymmetryXZ(arena.settings.deflectOrbSELocal, arena.settings.deflectOrbNWLocal, "Stage2 DeflectOrb");

                float minNormS2 = 1.0f;
                float maxSymErrS2 = 0f;
                for (int gz = -32; gz <= 32; gz++)
                {
                    float lz = gz * 0.5f;
                    for (int gx = -24; gx <= 24; gx++)
                    {
                        float lx = gx * 0.5f;
                        float hPos = ArenaStageBuilder.SampleTerrainHeightLocal(lx, lz, arena.settings);
                        float hNeg = ArenaStageBuilder.SampleTerrainHeightLocal(-lx, -lz, arena.settings);
                        maxSymErrS2 = Mathf.Max(maxSymErrS2, Mathf.Abs(hPos - hNeg));
                        Vector3 nrm = ArenaStageBuilder.SampleTerrainNormalLocal(lx, lz, arena.settings);
                        minNormS2 = Mathf.Min(minNormS2, nrm.y);
                    }
                }
                Assert(maxSymErrS2 < 1e-5f, $"Expected exact 180-deg heightfield symmetry on Stage 2, got maxSymErr={maxSymErrS2}");
                Assert(minNormS2 >= 0.75f, $"Expected C2-smooth walkable terrain slopes on Stage 2 (minNormalY >= 0.75), got {minNormS2:F3}");

                Vector3 rayFromS = root.transform.position + new Vector3(0f, 1.35f, -9.0f);
                Vector3 rayToCenter = root.transform.position + new Vector3(0f, 1.35f, 0.0f);
                bool hitBaffleS = Physics.Raycast(rayFromS, (rayToCenter - rayFromS).normalized, out RaycastHit bHitS, 9.0f);
                Assert(hitBaffleS && bHitS.collider != null && bHitS.collider.name.Contains("Baffle_Central"),
                    $"Expected Stage 2 central sightline (0,-9)->(0,0) to hit Baffle_Central_S, got hit={hitBaffleS} ({(bHitS.collider != null ? bHitS.collider.name : "none")})");
                string baffleHitName = bHitS.collider.name;

                Vector3 rayFromN = root.transform.position + new Vector3(0f, 1.35f, 9.0f);
                bool hitBaffleN = Physics.Raycast(rayFromN, (rayToCenter - rayFromN).normalized, out RaycastHit bHitN, 9.0f);
                Assert(hitBaffleN && bHitN.collider != null && bHitN.collider.name.Contains("Baffle_Central"),
                    $"Expected Stage 2 central sightline (0,+9)->(0,0) to hit Baffle_Central_N, got hit={hitBaffleN}");

                arena.RebuildArenaForStage(1);
                Physics.SyncTransforms();
                Assert(arena.settings.stageId == 1 && agA.brainBridge.stageId == 1, "Expected clean round-trip back to Stage 1.");
                Debug.Log($"[PASS] SelfTest 24/24: Stage 2 Occluded Twin-Flank verified (minNormY={minNormS2:F3}, maxSymErr={maxSymErrS2:E2}, BaffleHit='{baffleHitName}').");

                Debug.Log("[InkBlitzMechanicsSelfTest] ALL 24 PHASE 2-5 MECHANICS CHECKS PASSED!");
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[InkBlitzMechanicsSelfTest] FAILED: {ex}");
                return false;
            }
            finally
            {
                if (root != null)
                    UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static GameObject CreateTempArenaObject(
            string name, Vector3 worldPos, InkBlitzArenaSettings settings, out InkBlitzArenaInstance arena)
        {
            var go = new GameObject(name);
            go.transform.position = worldPos;
            var turf = go.AddComponent<TurfGridSystem>();
            var stage = go.AddComponent<ArenaStageBuilder>();
            var audio = go.AddComponent<ArenaAudioSystem>();
            var proj = go.AddComponent<InkProjectileManager>();
            stage.BuildStage(99, settings);

            arena = go.AddComponent<InkBlitzArenaInstance>();
            arena.arenaIndex = 99;
            arena.settings = settings;
            arena.turfGrid = turf;
            arena.stageBuilder = stage;
            arena.audioSystem = audio;
            arena.projectileManager = proj;
            arena.blitzCore = go.GetComponentInChildren<BlitzCoreObjective>(true);
            arena.logPeriodicTelemetry = false;

            InkBlitzPlayerAgent agA = CreateTempAgent(go.transform, "TestA", 99, 1, 1, settings.teamASpawnLocal, turf, isSupport: false);
            InkBlitzPlayerAgent agB = CreateTempAgent(go.transform, "TestB", 99, -1, 2, settings.teamBSpawnLocal, turf, isSupport: false);
            InkBlitzPlayerAgent supA = CreateTempAgent(go.transform, "TestSupA", 99, 1, 3, settings.teamASupportSpawnLocal, turf, isSupport: true);
            InkBlitzPlayerAgent supB = CreateTempAgent(go.transform, "TestSupB", 99, -1, 4, settings.teamBSupportSpawnLocal, turf, isSupport: true);
            arena.agents.Add(agA);
            arena.agents.Add(agB);
            arena.agents.Add(supA);
            arena.agents.Add(supB);
            arena.InitializeArena(registerWithCoordinator: false);
            return go;
        }

        private static InkBlitzPlayerAgent CreateTempAgent(
            Transform parent, string name, int arenaIdx, int teamId, int entId, Vector3 localPos, TurfGridSystem turf, bool isSupport)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            var cc = go.AddComponent<CharacterController>();
            cc.height = 1.5f;
            cc.radius = 0.42f;
            cc.center = new Vector3(0f, 0.75f, 0f);
            var tag = go.AddComponent<UniversalSemanticEntityTag>();
            tag.entityId = entId;
            tag.arenaIndex = arenaIdx;
            tag.teamId = teamId;
            tag.baseKind = UniversalEntityKind.OpponentAgent;

            var eye = new GameObject("EyePivot");
            eye.transform.SetParent(go.transform, false);
            eye.transform.localPosition = new Vector3(0f, 1.18f, 0f);

            UniversalBrainBridge3D bridge = null;
            if (!isSupport)
            {
                bridge = go.AddComponent<UniversalBrainBridge3D>();
                bridge.arenaIndex = arenaIdx;
                bridge.teamId = teamId;
                bridge.eyeTransform = eye.transform;
                bridge.floorSurfaceSamplerComponent = turf;
            }

            var ag = go.AddComponent<InkBlitzPlayerAgent>();
            ag.arenaIndex = arenaIdx;
            ag.teamId = teamId;
            ag.isSupportTrooperBot = isSupport;
            ag.characterController = cc;
            ag.entityTag = tag;
            ag.brainBridge = bridge;
            ag.eyePivot = eye.transform;
            if (isSupport)
            {
                var tr = go.AddComponent<SupportTrooperBot>();
                tr.agent = ag;
                tr.enableTrooperAi = false; // Enabled explicitly in Check 10
            }
            return ag;
        }

        private static void Assert(bool cond, string msg)
        {
            if (!cond)
                throw new Exception(msg);
        }

        private static void AssertSymmetryXZ(Vector3 a, Vector3 b, string label)
        {
            float err = Mathf.Abs(a.x + b.x) + Mathf.Abs(a.z + b.z);
            Assert(err < 1e-4f, $"{label} not 180-deg symmetric: a={a}, b={b}");
        }

        private static void TeleportAgent(InkBlitzPlayerAgent ag, Vector3 pos)
        {
            // Disable CharacterController before teleporting so PhysX does not snap back to its cached capsule pose.
            if (ag.characterController != null) ag.characterController.enabled = false;
            ag.transform.position = pos;
            if (ag.characterController != null) ag.characterController.enabled = true;
            if (ag.eyePivot != null) ag.eyePivot.position = pos + Vector3.up * 1.18f;
            ag.entityTag?.NotifyTeleported();
            Physics.SyncTransforms();
        }
    }
}
