// ArenaStageBuilder.cs — 180-degree rotationally symmetric 3D BlitzCore/Turf smooth manifold arena builder
using System.Collections.Generic;
using UnityEngine;
using UniversalBrain3D;

namespace InkBlitz3D
{
    public class ArenaStageBuilder : MonoBehaviour
    {
        [Header("Stage Materials & Persistent Meshes (Optional Assets)")]
        public Material defaultFloorMaterial;
        public Material wallMaterial;
        public Material coverMaterial;
        public Material teamAPedestalMaterial;
        public Material teamBPedestalMaterial;
        public Material objectiveMaterial;
        public Material objectiveShieldMaterial;
        public Material deflectOrbCoreMaterial;
        public Material deflectOrbHaloMaterial;
        public Mesh sharedTerrainMeshAsset;
        public Mesh sharedPedestalBevelMeshAsset;

        [Header("Runtime Paintable Floor Renderers")]
        public List<MeshRenderer> paintableFloorRenderers = new List<MeshRenderer>();

        private static Mesh s_cachedRuntimeTerrainMesh;
        private static Mesh s_cachedRuntimeTerrainMeshStage2;
        private static Mesh s_cachedRuntimePedestalMesh;

        // =========================================================================================
        // 180-Degree Symmetric C1-Smooth 2.5D Terrain Heightfield (Zero Cliffs, Trenches, or Stairs)
        // =========================================================================================

        private static float SmoothStep01(float t)
        {
            t = Mathf.Clamp01(t);
            // C2 curvature-continuous quintic smootherstep (zero derivative & zero curvature jump at toe/crest!)
            return t * t * t * (t * (t * 6f - 15f) + 10f);
        }

        private static float SmoothMax(float a, float b, float k)
        {
            if (k <= 1e-5f) return Mathf.Max(a, b);
            float h = Mathf.Max(k - Mathf.Abs(a - b), 0f) / k;
            return Mathf.Max(a, b) + h * h * k * 0.25f;
        }

        private static float EvalSmoothPlateau(
            float x, float z,
            float minX, float maxX, float minZ, float maxZ,
            float wNegX, float wPosX, float wNegZ, float wPosZ,
            float topH)
        {
            float ux = 0f;
            if (x < minX) ux = (minX - x) / Mathf.Max(0.05f, wNegX);
            else if (x > maxX) ux = (x - maxX) / Mathf.Max(0.05f, wPosX);

            float uz = 0f;
            if (z < minZ) uz = (minZ - z) / Mathf.Max(0.05f, wNegZ);
            else if (z > maxZ) uz = (z - maxZ) / Mathf.Max(0.05f, wPosZ);

            float rSq = ux * ux + uz * uz;
            if (rSq >= 1f) return 0f;
            if (rSq <= 0f) return topH;
            float r = Mathf.Sqrt(rSq);
            float s = r * r * r * (r * (r * 6f - 15f) + 10f);
            return topH * (1f - s);
        }

        private static float EvalCenterMoundHeight(float x, float z)
        {
            // Smooth 360-deg accessible central BlitzCore causeway (replaces E/W stairs, stepped corners & boxy plinth)
            float baseMound = EvalSmoothPlateau(
                x, z,
                minX: -0.95f, maxX: 0.95f, minZ: -1.55f, maxZ: 1.55f,
                wNegX: 1.95f, wPosX: 1.95f, wNegZ: 2.65f, wPosZ: 2.65f,
                topH: 0.64f);

            float crown = EvalSmoothPlateau(
                x, z,
                minX: -0.45f, maxX: 0.45f, minZ: -0.45f, maxZ: 0.45f,
                wNegX: 1.35f, wPosX: 1.35f, wNegZ: 1.35f, wPosZ: 1.35f,
                topH: 0.08f);

            return baseMound + crown;
        }

        private static float EvalSpawnTerraceSW(float x, float z)
        {
            // Flush into SW outer walls (x = -12, z = -16) with continuous radial slope fan (zero wall trenches!)
            return EvalSmoothPlateau(
                x, z,
                minX: -13.2f, maxX: -6.0f, minZ: -17.2f, maxZ: -11.6f,
                wNegX: 1.0f, wPosX: 3.3f, wNegZ: 1.0f, wPosZ: 3.3f,
                topH: 0.65f);
        }

        private static float EvalHighGroundWingSE(float x, float z, Vector3 pedSE)
        {
            // Single unified C2-smooth S-curve wing uniting Overlook (1.35m) -> Sky-Bridge -> Goal Citadel (1.18m) with zero seam ripples!
            float tGrade = SmoothStep01((-4.2f - z) / 6.0f);
            float tFlare = SmoothStep01((-2.8f - z) / 7.8f);

            float deckTopH = Mathf.Lerp(1.35f, 1.18f, tGrade);
            float innerMinX = Mathf.Lerp(8.25f, 5.65f, tFlare);
            float innerRampW = Mathf.Lerp(3.25f, 3.75f, tFlare);

            float wingBase = EvalSmoothPlateau(
                x, z,
                minX: innerMinX, maxX: 13.2f, minZ: -17.2f, maxZ: 0.5f,
                wNegX: innerRampW, wPosX: 1.0f, wNegZ: 1.0f, wPosZ: 4.0f,
                topH: deckTopH);

            // Gentle C2 Summit Crown (1.18m -> 1.35m) centered at SE Pedestal (replaces sharp 0.20m UpperMound step!)
            float hSummit = EvalSmoothPlateau(
                x, z,
                minX: pedSE.x - 1.10f, maxX: pedSE.x + 1.10f,
                minZ: pedSE.z - 1.10f, maxZ: pedSE.z + 1.10f,
                wNegX: 1.50f, wPosX: 1.50f, wNegZ: 1.50f, wPosZ: 1.50f,
                topH: 0.17f);

            return wingBase + hSummit;
        }

        private static float SampleStage2TerrainHeightLocal(float lx, float lz, Vector3 pedSE)
        {
            // Stage 2: Occluded Twin-Flank — C2-smooth 180-deg symmetric dual-lane basin + high outer catwalks
            float hSanctum = EvalSmoothPlateau(
                lx, lz,
                minX: -1.35f, maxX: 1.35f, minZ: -1.65f, maxZ: 1.65f,
                wNegX: 2.55f, wPosX: 2.55f, wNegZ: 2.35f, wPosZ: 2.35f,
                topH: 0.44f);

            // Wall-flush SW / NE Spawn Terraces (zero wall trenches!)
            float EvalSpawnS2(float x, float z) => EvalSmoothPlateau(
                x, z,
                minX: -13.2f, maxX: -4.2f, minZ: -17.2f, maxZ: -11.4f,
                wNegX: 1.0f, wPosX: 3.6f, wNegZ: 1.0f, wPosZ: 3.5f,
                topH: 0.65f);

            // Outer Catwalk + SE/NW Citadel Summit (smoothly bridges flank overlook z=+2.8 into SE goal citadel)
            float EvalCatwalkS2(float x, float z)
            {
                float tFlare = SmoothStep01((-3.2f - z) / 7.5f);
                float innerMinX = Mathf.Lerp(8.15f, 4.65f, tFlare);
                float deckTopH = Mathf.Lerp(1.20f, 1.14f, tFlare);
                float catwalk = EvalSmoothPlateau(
                    x, z,
                    minX: innerMinX, maxX: 13.2f, minZ: -17.2f, maxZ: 2.6f,
                    wNegX: 3.65f, wPosX: 1.0f, wNegZ: 1.0f, wPosZ: 3.8f,
                    topH: deckTopH);

                float summit = EvalSmoothPlateau(
                    x, z,
                    minX: pedSE.x - 1.15f, maxX: pedSE.x + 1.15f,
                    minZ: pedSE.z - 1.15f, maxZ: pedSE.z + 1.15f,
                    wNegX: 1.65f, wPosX: 1.65f, wNegZ: 1.65f, wPosZ: 1.65f,
                    topH: 0.14f);
                return catwalk + summit;
            }

            float hSW = EvalSpawnS2(lx, lz);
            float hNE = EvalSpawnS2(-lx, -lz);
            float hCatSE = EvalCatwalkS2(lx, lz);
            float hCatNW = EvalCatwalkS2(-lx, -lz);
            return Mathf.Max(hSanctum, Mathf.Max(Mathf.Max(hSW, hNE), Mathf.Max(hCatSE, hCatNW)));
        }

        public static float SampleTerrainHeightLocal(float lx, float lz, InkBlitzArenaSettings settings = null)
        {
            if (settings != null && settings.stageId == 2)
            {
                Vector3 pedS2 = new Vector3(Mathf.Abs(settings.teamAPedestalLocal.x), settings.teamAPedestalLocal.y, -Mathf.Abs(settings.teamAPedestalLocal.z));
                return SampleStage2TerrainHeightLocal(lx, lz, pedS2);
            }

            Vector3 pedSE = (settings != null) ? settings.teamAPedestalLocal : new Vector3(8.0f, 1.35f, -12.8f);
            Vector3 p = new Vector3(Mathf.Abs(pedSE.x), pedSE.y, -Mathf.Abs(pedSE.z));

            float hCenter = EvalCenterMoundHeight(lx, lz);
            float hSW = EvalSpawnTerraceSW(lx, lz);
            float hNE = EvalSpawnTerraceSW(-lx, -lz);
            float hWingSE = EvalHighGroundWingSE(lx, lz, p);
            float hWingNW = EvalHighGroundWingSE(-lx, -lz, p);

            return Mathf.Max(hCenter, Mathf.Max(Mathf.Max(hSW, hNE), Mathf.Max(hWingSE, hWingNW)));
        }

        public static Vector3 SampleTerrainNormalLocal(float lx, float lz, InkBlitzArenaSettings settings = null)
        {
            const float eps = 0.08f;
            float hx1 = SampleTerrainHeightLocal(lx + eps, lz, settings);
            float hx0 = SampleTerrainHeightLocal(lx - eps, lz, settings);
            float hz1 = SampleTerrainHeightLocal(lx, lz + eps, settings);
            float hz0 = SampleTerrainHeightLocal(lx, lz - eps, settings);
            float dhdx = (hx1 - hx0) / (2f * eps);
            float dhdz = (hz1 - hz0) / (2f * eps);
            return new Vector3(-dhdx, 1f, -dhdz).normalized;
        }

        public static Mesh BuildSmoothTerrainMesh(InkBlitzArenaSettings settings = null)
        {
            InkBlitzArenaSettings cfg = settings ?? new InkBlitzArenaSettings();
            const int nx = 96;  // 0.25m X cells across 24m
            const int nz = 128; // 0.25m Z cells across 32m
            const float baseBottomY = -0.60f;

            float widthX = cfg.arenaWidthX;
            float lengthZ = cfg.arenaLengthZ;
            float halfW = widthX * 0.5f;
            float halfL = lengthZ * 0.5f;
            float invW = 1f / Mathf.Max(0.01f, widthX);
            float invL = 1f / Mathf.Max(0.01f, lengthZ);

            int topVertCount = (nx + 1) * (nz + 1);
            int skirtVertCount = 2 * ((nx + 1) + (nz + 1)) * 2 + 4;
            var verts = new List<Vector3>(topVertCount + skirtVertCount);
            var norms = new List<Vector3>(topVertCount + skirtVertCount);
            var uvs = new List<Vector2>(topVertCount + skirtVertCount);
            var tris = new List<int>(nx * nz * 6 + 2 * (nx + nz) * 6 + 6);

            for (int iz = 0; iz <= nz; iz++)
            {
                float lz = -halfL + (iz / (float)nz) * lengthZ;
                float v = (lz + halfL) * invL;
                for (int ix = 0; ix <= nx; ix++)
                {
                    float lx = -halfW + (ix / (float)nx) * widthX;
                    float ly = SampleTerrainHeightLocal(lx, lz, cfg);
                    verts.Add(new Vector3(lx, ly, lz));
                    norms.Add(SampleTerrainNormalLocal(lx, lz, cfg));
                    uvs.Add(new Vector2((lx + halfW) * invW, v));
                }
            }

            // 180-deg rotationally invariant triangulation (v00 -> v11 diagonal maps to itself under (x,z)->(-x,-z))
            int stride = nx + 1;
            for (int iz = 0; iz < nz; iz++)
            {
                int row0 = iz * stride;
                int row1 = (iz + 1) * stride;
                for (int ix = 0; ix < nx; ix++)
                {
                    int v00 = row0 + ix;
                    int v10 = row0 + ix + 1;
                    int v01 = row1 + ix;
                    int v11 = row1 + ix + 1;

                    tris.Add(v00);
                    tris.Add(v01);
                    tris.Add(v11);

                    tris.Add(v00);
                    tris.Add(v11);
                    tris.Add(v10);
                }
            }

            // Watertight outer skirt & solid bottom cap
            void AppendSkirtStrip(int count, System.Func<int, Vector3> sampleEdgeTop, Vector3 outwardNormal)
            {
                int baseIdx = verts.Count;
                for (int i = 0; i <= count; i++)
                {
                    Vector3 topPt = sampleEdgeTop(i);
                    Vector2 uv = new Vector2((topPt.x + halfW) * invW, (topPt.z + halfL) * invL);
                    verts.Add(topPt);
                    norms.Add(outwardNormal);
                    uvs.Add(uv);

                    verts.Add(new Vector3(topPt.x, baseBottomY, topPt.z));
                    norms.Add(outwardNormal);
                    uvs.Add(uv);
                }
                for (int i = 0; i < count; i++)
                {
                    int t0 = baseIdx + i * 2;
                    int b0 = t0 + 1;
                    int t1 = baseIdx + (i + 1) * 2;
                    int b1 = t1 + 1;
                    tris.Add(t0); tris.Add(t1); tris.Add(b1);
                    tris.Add(t0); tris.Add(b1); tris.Add(b0);
                }
            }

            AppendSkirtStrip(nx, i => verts[0 * stride + i], Vector3.back);
            AppendSkirtStrip(nz, i => verts[i * stride + nx], Vector3.right);
            AppendSkirtStrip(nx, i => verts[nz * stride + (nx - i)], Vector3.forward);
            AppendSkirtStrip(nz, i => verts[(nz - i) * stride + 0], Vector3.left);

            int botBase = verts.Count;
            verts.Add(new Vector3(-halfW, baseBottomY, -halfL)); norms.Add(Vector3.down); uvs.Add(new Vector2(0f, 0f));
            verts.Add(new Vector3(halfW, baseBottomY, -halfL));  norms.Add(Vector3.down); uvs.Add(new Vector2(1f, 0f));
            verts.Add(new Vector3(halfW, baseBottomY, halfL));   norms.Add(Vector3.down); uvs.Add(new Vector2(1f, 1f));
            verts.Add(new Vector3(-halfW, baseBottomY, halfL));  norms.Add(Vector3.down); uvs.Add(new Vector2(0f, 1f));
            tris.Add(botBase + 0); tris.Add(botBase + 1); tris.Add(botBase + 2);
            tris.Add(botBase + 0); tris.Add(botBase + 2); tris.Add(botBase + 3);

            var mesh = new Mesh
            {
                name = "ArenaTerrainManifold_ArenaPlanarUV"
            };
            mesh.SetVertices(verts);
            mesh.SetNormals(norms);
            mesh.SetUVs(0, uvs);
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateBounds();
            return mesh;
        }

        public static Mesh BuildBeveledPedestalMesh()
        {
            // Flush chamfered low-profile Goal Pad (outer 1.88m -> inner 1.32m over 0.055m rise => ~11-deg bevel, zero step!)
            const float outerHalf = 0.94f;
            const float innerHalf = 0.66f;
            const float botY = -0.04f;
            const float topY = 0.055f;

            Vector3[] v =
            {
                // Top quad (0..3)
                new Vector3(-innerHalf, topY, -innerHalf),
                new Vector3(-innerHalf, topY,  innerHalf),
                new Vector3( innerHalf, topY,  innerHalf),
                new Vector3( innerHalf, topY, -innerHalf),
                // South bevel (4..7)
                new Vector3(-outerHalf, botY, -outerHalf),
                new Vector3(-innerHalf, topY, -innerHalf),
                new Vector3( innerHalf, topY, -innerHalf),
                new Vector3( outerHalf, botY, -outerHalf),
                // East bevel (8..11)
                new Vector3( outerHalf, botY, -outerHalf),
                new Vector3( innerHalf, topY, -innerHalf),
                new Vector3( innerHalf, topY,  innerHalf),
                new Vector3( outerHalf, botY,  outerHalf),
                // North bevel (12..15)
                new Vector3( outerHalf, botY,  outerHalf),
                new Vector3( innerHalf, topY,  innerHalf),
                new Vector3(-innerHalf, topY,  innerHalf),
                new Vector3(-outerHalf, botY,  outerHalf),
                // West bevel (16..19)
                new Vector3(-outerHalf, botY,  outerHalf),
                new Vector3(-innerHalf, topY,  innerHalf),
                new Vector3(-innerHalf, topY, -innerHalf),
                new Vector3(-outerHalf, botY, -outerHalf),
            };

            int[] tris = new int[5 * 6];
            for (int f = 0; f < 5; f++)
            {
                int b = f * 4;
                tris[f * 6 + 0] = b + 0;
                tris[f * 6 + 1] = b + 1;
                tris[f * 6 + 2] = b + 2;
                tris[f * 6 + 3] = b + 0;
                tris[f * 6 + 4] = b + 2;
                tris[f * 6 + 5] = b + 3;
            }

            var mesh = new Mesh { name = "GoalPedestalBevelMesh" };
            mesh.vertices = v;
            mesh.triangles = tris;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private Mesh GetOrCreateTerrainMesh(InkBlitzArenaSettings cfg)
        {
            int stId = (cfg != null) ? cfg.stageId : 1;
            if (stId == 2)
            {
                if (s_cachedRuntimeTerrainMeshStage2 == null)
                {
                    s_cachedRuntimeTerrainMeshStage2 = BuildSmoothTerrainMesh(cfg);
                    s_cachedRuntimeTerrainMeshStage2.name = "ArenaTerrainManifold_Stage2_TwinFlank";
                }
                return s_cachedRuntimeTerrainMeshStage2;
            }
            if (sharedTerrainMeshAsset != null)
                return sharedTerrainMeshAsset;
            if (s_cachedRuntimeTerrainMesh == null)
                s_cachedRuntimeTerrainMesh = BuildSmoothTerrainMesh(cfg);
            return s_cachedRuntimeTerrainMesh;
        }

        private Mesh GetOrCreatePedestalMesh()
        {
            if (sharedPedestalBevelMeshAsset != null)
                return sharedPedestalBevelMeshAsset;
            if (s_cachedRuntimePedestalMesh == null)
                s_cachedRuntimePedestalMesh = BuildBeveledPedestalMesh();
            return s_cachedRuntimePedestalMesh;
        }

        public void EnsureTerrainManifoldReady(InkBlitzArenaSettings settings = null)
        {
            Transform floorsRoot = transform.Find("WalkableFloors");
            if (floorsRoot == null) return;

            Transform manifold = floorsRoot.Find("ArenaTerrainManifold");
            if (manifold == null) return;

            Mesh terrainMesh = GetOrCreateTerrainMesh(settings);
            var mf = manifold.GetComponent<MeshFilter>();
            if (mf != null && mf.sharedMesh == null)
                mf.sharedMesh = terrainMesh;
            var mc = manifold.GetComponent<MeshCollider>();
            if (mc != null && mc.sharedMesh == null)
                mc.sharedMesh = terrainMesh;

            Transform landmarks = transform.Find("Landmarks");
            if (landmarks != null)
            {
                Mesh pedMesh = GetOrCreatePedestalMesh();
                string[] peds = { "Pedestal_TeamA/BaseStep", "Pedestal_TeamB/BaseStep" };
                for (int i = 0; i < peds.Length; i++)
                {
                    Transform bs = landmarks.Find(peds[i]);
                    if (bs == null) continue;
                    var pmf = bs.GetComponent<MeshFilter>();
                    if (pmf != null && pmf.sharedMesh == null)
                        pmf.sharedMesh = pedMesh;
                }
            }
        }

        public void BuildStage(int arenaIndex, InkBlitzArenaSettings settings)
        {
            ClearExistingChildren();
            paintableFloorRenderers.Clear();

            InkBlitzArenaSettings cfg = settings ?? new InkBlitzArenaSettings();
            Vector3 pedA = cfg.teamAPedestalLocal; // (+8.0, 1.35, -12.8) SE Goal Citadel Summit
            Vector3 pedB = cfg.teamBPedestalLocal; // (-8.0, 1.35, +12.8) NW Goal Citadel Summit

            // 1. Unified C1-Smooth 180-deg Symmetric Walkable Terrain Manifold (Zero cliffs, stairs, lips, or wall trenches!)
            GameObject floorsRoot = CreateChildFolder("WalkableFloors");
            CreatePaintableTerrainManifold(floorsRoot.transform, "ArenaTerrainManifold", cfg);

            // Preserve symmetric high-ground spatial anchors for stage inspection & self-tests
            CreateSpatialAnchor(floorsRoot.transform, "Overlook_SE", new Vector3(9.6f, 0.675f, -2.0f));
            CreateSpatialAnchor(floorsRoot.transform, "Overlook_NW", new Vector3(-9.6f, 0.675f, 2.0f));

            // 2. Perimeter Walls & Ground-Seated Tactical Covers (Unpaintable)
            GameObject wallsRoot = CreateChildFolder("WallsAndCovers");
            float wallH = 3.4f;
            float wallT = 0.8f;
            float halfW = cfg.arenaWidthX * 0.5f;
            float halfL = cfg.arenaLengthZ * 0.5f;

            CreateStaticWall(wallsRoot.transform, "Wall_North",
                new Vector3(0f, wallH * 0.5f, halfL + wallT * 0.5f), new Vector3(cfg.arenaWidthX + wallT * 2f, wallH, wallT));
            CreateStaticWall(wallsRoot.transform, "Wall_South",
                new Vector3(0f, wallH * 0.5f, -halfL - wallT * 0.5f), new Vector3(cfg.arenaWidthX + wallT * 2f, wallH, wallT));
            CreateStaticWall(wallsRoot.transform, "Wall_East",
                new Vector3(halfW + wallT * 0.5f, wallH * 0.5f, 0f), new Vector3(wallT, wallH, cfg.arenaLengthZ));
            CreateStaticWall(wallsRoot.transform, "Wall_West",
                new Vector3(-halfW - wallT * 0.5f, wallH * 0.5f, 0f), new Vector3(wallT, wallH, cfg.arenaLengthZ));

            // Overlook sniper screens & Mid-SkyBridge cover parapets (seated cleanly inside flat decks down to y = 0)
            CreateStaticWall(wallsRoot.transform, "Pillar_Overlook_SE",
                new Vector3(8.55f, 1.125f, -1.2f), new Vector3(0.52f, 2.25f, 1.15f));
            CreateStaticWall(wallsRoot.transform, "Pillar_Overlook_NW",
                new Vector3(-8.55f, 1.125f, 1.2f), new Vector3(0.52f, 2.25f, 1.15f));

            CreateStaticWall(wallsRoot.transform, "Parapet_SkyBridge_SE",
                new Vector3(9.40f, 0.95f, -7.2f), new Vector3(0.46f, 1.90f, 1.25f));
            CreateStaticWall(wallsRoot.transform, "Parapet_SkyBridge_NW",
                new Vector3(-9.40f, 0.95f, 7.2f), new Vector3(0.46f, 1.90f, 1.25f));

            if (cfg.stageId == 2)
            {
                // Stage 2: Central Sightline-Blocking S-Baffle Chicane + Twin-Flank Choke Covers (180-deg Symmetric)
                CreateCoverBlock(wallsRoot.transform, "Baffle_Central_S", arenaIndex, 20,
                    new Vector3(1.15f, 1.25f, -3.55f), new Vector3(5.60f, 2.50f, 1.10f), 14f);
                CreateCoverBlock(wallsRoot.transform, "Baffle_Central_N", arenaIndex, 21,
                    new Vector3(-1.15f, 1.25f, 3.55f), new Vector3(5.60f, 2.50f, 1.10f), 194f);

                CreateCoverBlock(wallsRoot.transform, "Cover_CenterSE", arenaIndex, 10,
                    new Vector3(4.65f, 0.90f, -0.4f), new Vector3(1.55f, 1.80f, 1.35f), 24f);
                CreateCoverBlock(wallsRoot.transform, "Cover_CenterNW", arenaIndex, 11,
                    new Vector3(-4.65f, 0.90f, 0.4f), new Vector3(1.55f, 1.80f, 1.35f), 204f);

                CreateCoverBlock(wallsRoot.transform, "Cover_FlankGate_SW", arenaIndex, 12,
                    new Vector3(-6.15f, 0.90f, -1.85f), new Vector3(1.85f, 1.80f, 1.25f), -20f);
                CreateCoverBlock(wallsRoot.transform, "Cover_FlankGate_NE", arenaIndex, 13,
                    new Vector3(6.15f, 0.90f, 1.85f), new Vector3(1.85f, 1.80f, 1.25f), 160f);

                CreateCoverBlock(wallsRoot.transform, "Cover_FlankMid_W", arenaIndex, 14,
                    new Vector3(-6.80f, 0.85f, 4.40f), new Vector3(1.75f, 1.70f, 1.25f), 16f);
                CreateCoverBlock(wallsRoot.transform, "Cover_FlankMid_E", arenaIndex, 15,
                    new Vector3(6.80f, 0.85f, -4.40f), new Vector3(1.75f, 1.70f, 1.25f), 196f);

                CreateCoverBlock(wallsRoot.transform, "Cover_CitadelScreen_SE", arenaIndex, 16,
                    new Vector3(3.00f, 0.90f, -8.80f), new Vector3(2.35f, 1.80f, 1.15f), 20f);
                CreateCoverBlock(wallsRoot.transform, "Cover_CitadelScreen_NW", arenaIndex, 17,
                    new Vector3(-3.00f, 0.90f, 8.80f), new Vector3(2.35f, 1.80f, 1.15f), 200f);
            }
            else
            {
                // Stage 1: 8 Symmetric Tactical Valley Cover Blocks (static EKF-SLAM landmarks + ambush/refuge cover)
                CreateCoverBlock(wallsRoot.transform, "Cover_CenterSE", arenaIndex, 10,
                    new Vector3(4.0f, 0.85f, -0.2f), new Vector3(1.7f, 1.70f, 1.3f), 20f);
                CreateCoverBlock(wallsRoot.transform, "Cover_CenterNW", arenaIndex, 11,
                    new Vector3(-4.0f, 0.85f, 0.2f), new Vector3(1.7f, 1.70f, 1.3f), 200f);

                CreateCoverBlock(wallsRoot.transform, "Cover_ValleySW", arenaIndex, 12,
                    new Vector3(-3.2f, 0.85f, -5.8f), new Vector3(2.0f, 1.70f, 1.25f), -18f);
                CreateCoverBlock(wallsRoot.transform, "Cover_ValleyNE", arenaIndex, 13,
                    new Vector3(3.2f, 0.85f, 5.8f), new Vector3(2.0f, 1.70f, 1.25f), 162f);

                CreateCoverBlock(wallsRoot.transform, "Cover_CitadelGate_SE", arenaIndex, 14,
                    new Vector3(2.6f, 0.85f, -7.8f), new Vector3(2.1f, 1.70f, 1.15f), 22f);
                CreateCoverBlock(wallsRoot.transform, "Cover_CitadelGate_NW", arenaIndex, 15,
                    new Vector3(-2.6f, 0.85f, 7.8f), new Vector3(2.1f, 1.70f, 1.15f), 202f);

                CreateCoverBlock(wallsRoot.transform, "Cover_BackMid_S", arenaIndex, 16,
                    new Vector3(-1.2f, 0.80f, -9.6f), new Vector3(1.8f, 1.60f, 1.2f), -12f);
                CreateCoverBlock(wallsRoot.transform, "Cover_BackMid_N", arenaIndex, 17,
                    new Vector3(1.2f, 0.80f, 9.6f), new Vector3(1.8f, 1.60f, 1.2f), 168f);
            }

            // 3. Flush Beveled Goal Pedestal Crowns & Central BlitzCore
            GameObject landmarksRoot = CreateChildFolder("Landmarks");
            float pedAY = SampleTerrainHeightLocal(pedA.x, pedA.z, cfg);
            float pedBY = SampleTerrainHeightLocal(pedB.x, pedB.z, cfg);
            CreatePedestal(landmarksRoot.transform, "Pedestal_TeamA", arenaIndex, 101, 1,
                new Vector3(pedA.x, pedAY, pedA.z), teamAPedestalMaterial, cfg.teamAColor);
            CreatePedestal(landmarksRoot.transform, "Pedestal_TeamB", arenaIndex, 102, -1,
                new Vector3(pedB.x, pedBY, pedB.z), teamBPedestalMaterial, cfg.teamBColor);

            float centerGroundY = SampleTerrainHeightLocal(0f, 0f, cfg);
            CreateBlitzCoreObjective(landmarksRoot.transform, "BlitzCoreObjective", arenaIndex, 100,
                new Vector3(0f, centerGroundY + 0.65f, 0f));

            // 4. 180-deg Symmetric Wall-Side Sky-Bridge 3D Counter-Tennis / Deflect Orbs (`ResourceOrPickup`, tag = +2.0)
            CreateDeflectPowerOrb(landmarksRoot.transform, "DeflectOrb_SkyBridge_SE", arenaIndex, 110, cfg.deflectOrbSELocal, cfg);
            CreateDeflectPowerOrb(landmarksRoot.transform, "DeflectOrb_SkyBridge_NW", arenaIndex, 111, cfg.deflectOrbNWLocal, cfg);
        }

        private GameObject CreateChildFolder(string folderName)
        {
            var go = new GameObject(folderName);
            go.transform.SetParent(transform, false);
            return go;
        }

        private static void CreateSpatialAnchor(Transform parent, string name, Vector3 localPos)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
        }

        private void ClearExistingChildren()
        {
            string[] folders = { "WalkableFloors", "WallsAndCovers", "Landmarks" };
            foreach (string f in folders)
            {
                Transform existing = transform.Find(f);
                if (existing != null)
                {
                    if (Application.isPlaying)
                    {
                        existing.SetParent(null, false);
                        existing.gameObject.SetActive(false);
                        Destroy(existing.gameObject);
                    }
                    else
                    {
                        DestroyImmediate(existing.gameObject);
                    }
                }
            }
        }

        private void CreatePaintableTerrainManifold(Transform parent, string objName, InkBlitzArenaSettings cfg)
        {
            var go = new GameObject(objName);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;

            Mesh terrainMesh = GetOrCreateTerrainMesh(cfg);
            var mf = go.AddComponent<MeshFilter>();
            mf.sharedMesh = terrainMesh;

            var mr = go.AddComponent<MeshRenderer>();
            if (defaultFloorMaterial != null)
                mr.sharedMaterial = defaultFloorMaterial;

            var mc = go.AddComponent<MeshCollider>();
            mc.convex = false;
            mc.sharedMesh = terrainMesh;

            paintableFloorRenderers.Add(mr);
        }

        private void CreateStaticWall(Transform parent, string objName, Vector3 localPos, Vector3 scale)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = objName;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            if (wallMaterial != null)
                go.GetComponent<MeshRenderer>().sharedMaterial = wallMaterial;
        }

        private void CreateCoverBlock(Transform parent, string objName, int arenaIndex, int entityId, Vector3 localPos, Vector3 scale, float yawDeg = 0f)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = objName;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localRotation = Quaternion.Euler(0f, yawDeg, 0f);
            go.transform.localScale = scale;
            if (coverMaterial != null)
                go.GetComponent<MeshRenderer>().sharedMaterial = coverMaterial;

            var tag = go.AddComponent<UniversalSemanticEntityTag>();
            tag.entityId = entityId;
            tag.arenaIndex = arenaIndex;
            tag.teamId = 0;
            tag.baseKind = UniversalEntityKind.CoverOrFriendlyRefuge;
            tag.boundingRadius = Mathf.Max(0.85f, 0.45f * Mathf.Max(scale.x, scale.z));
        }

        private void CreatePedestal(Transform parent, string objName, int arenaIndex, int entityId, int teamId,
            Vector3 localGroundPos, Material mat, Color fallbackColor)
        {
            // Flush low-bevel Goal Pad seated smoothly into the 1.35m Citadel Summit (zero vertical step snap!)
            var root = new GameObject(objName);
            root.transform.SetParent(parent, false);
            root.transform.localPosition = new Vector3(localGroundPos.x, localGroundPos.y, localGroundPos.z);

            var tag = root.AddComponent<UniversalSemanticEntityTag>();
            tag.entityId = entityId;
            tag.arenaIndex = arenaIndex;
            tag.teamId = teamId;
            tag.baseKind = UniversalEntityKind.DefendBasePedestal;
            tag.boundingRadius = 1.2f;

            Material chosenMat = mat;
            if (chosenMat == null && Application.isPlaying)
            {
                Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                chosenMat = new Material(sh);
                chosenMat.color = fallbackColor;
            }

            var baseStep = new GameObject("BaseStep");
            baseStep.transform.SetParent(root.transform, false);
            baseStep.transform.localPosition = Vector3.zero;
            var bmf = baseStep.AddComponent<MeshFilter>();
            bmf.sharedMesh = GetOrCreatePedestalMesh();
            var bmr = baseStep.AddComponent<MeshRenderer>();
            if (chosenMat != null)
                bmr.sharedMaterial = chosenMat;

            GameObject plate = GameObject.CreatePrimitive(PrimitiveType.Cube);
            plate.name = "GoalPlate";
            plate.transform.SetParent(root.transform, false);
            plate.transform.localPosition = new Vector3(0f, 0.058f, 0f);
            plate.transform.localScale = new Vector3(1.18f, 0.024f, 1.18f);
            Collider plateCol = plate.GetComponent<Collider>();
            if (plateCol != null)
            {
                if (Application.isPlaying) Destroy(plateCol);
                else DestroyImmediate(plateCol);
            }
            if (chosenMat != null)
                plate.GetComponent<MeshRenderer>().sharedMaterial = chosenMat;
        }

        private void CreateBlitzCoreObjective(Transform parent, string objName, int arenaIndex, int entityId, Vector3 localPos)
        {
            var go = new GameObject(objName);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;

            var tag = go.AddComponent<UniversalSemanticEntityTag>();
            tag.entityId = entityId;
            tag.arenaIndex = arenaIndex;
            tag.teamId = 0;
            tag.baseKind = UniversalEntityKind.PrimaryObjective;
            tag.boundingRadius = 0.85f;

            var rm = go.AddComponent<BlitzCoreObjective>();
            rm.entityTag = tag;
            rm.templateShieldMaterial = objectiveShieldMaterial;
            rm.EnsureVisualChildren();

            if (objectiveMaterial != null && rm.coreVisual != null)
            {
                var mr = rm.coreVisual.GetComponent<MeshRenderer>();
                if (mr != null) mr.sharedMaterial = objectiveMaterial;
            }
            if (objectiveShieldMaterial != null && rm.shieldVisual != null)
            {
                var smr = rm.shieldVisual.GetComponent<MeshRenderer>();
                if (smr != null) smr.sharedMaterial = objectiveShieldMaterial;
            }
        }

        private void CreateDeflectPowerOrb(Transform parent, string objName, int arenaIndex, int entityId, Vector3 deckSurfaceLocal, InkBlitzArenaSettings cfg)
        {
            float surfY = SampleTerrainHeightLocal(deckSurfaceLocal.x, deckSurfaceLocal.z, cfg);
            var go = new GameObject(objName);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = new Vector3(deckSurfaceLocal.x, surfY + 0.68f, deckSurfaceLocal.z);

            var tag = go.AddComponent<UniversalSemanticEntityTag>();
            tag.entityId = entityId;
            tag.arenaIndex = arenaIndex;
            tag.teamId = 0;
            tag.baseKind = UniversalEntityKind.ResourceOrPickup;
            tag.boundingRadius = 0.65f;

            var pickup = go.AddComponent<DeflectPowerPickup>();
            pickup.entityTag = tag;
            pickup.templateCoreMaterial = deflectOrbCoreMaterial;
            pickup.templateHaloMaterial = deflectOrbHaloMaterial;
            pickup.EnsureVisualChildren();

            if (deflectOrbCoreMaterial != null)
            {
                if (pickup.coreVisual != null)
                {
                    var mr = pickup.coreVisual.GetComponent<MeshRenderer>();
                    if (mr != null) mr.sharedMaterial = deflectOrbCoreMaterial;
                }
                if (pickup.basePadVisual != null)
                {
                    var pmr = pickup.basePadVisual.GetComponent<MeshRenderer>();
                    if (pmr != null) pmr.sharedMaterial = deflectOrbCoreMaterial;
                }
            }
            if (deflectOrbHaloMaterial != null && pickup.haloVisual != null)
            {
                var hmr = pickup.haloVisual.GetComponent<MeshRenderer>();
                if (hmr != null) hmr.sharedMaterial = deflectOrbHaloMaterial;
            }
        }
    }
}
