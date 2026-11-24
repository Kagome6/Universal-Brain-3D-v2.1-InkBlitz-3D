// ObstacleWallFluidSystem.cs — Real-time Bingham-plastic viscous thin-film wall ink & splash physics
// Strictly visual-only (Spectate / Human Play) and restricted to INTERIOR box obstacles (never outer frame walls).
// Zero AI raycast, collider, wall-climb, or training-mode throughput side effects.
using System;
using System.Collections.Generic;
using UnityEngine;

namespace InkBlitz3D
{
    public class ObstacleWallFluidSystem : MonoBehaviour
    {
        [Header("Simulation & Quality (Visual-Only, Auto-Bypassed in Fast Training)")]
        public bool enableVisualWallFluid = true;
        public float pixelsPerMeter = 24.0f;
        public float binghamYieldThickness = 0.26f;
        public float gravityMobilityGain = 2.85f;
        public float maxDripSpeedMps = 0.92f;

        private const float SimStepDt = 1f / 30f;
        private const float VWallTop = 0.78f;
        private const float VRoofMin = 0.82f;
        private const float VRoofMax = 0.98f;
        private const int DropletPoolSize = 56;
        private const int MaxSlopeRivulets = 96;

        private sealed class ObstacleFluidSurface
        {
            public BoxCollider collider;
            public Transform transform;
            public MeshRenderer renderer;
            public MeshFilter meshFilter;
            public Material originalSharedMaterial;
            public Mesh originalSharedMesh;

            public Mesh unwrappedMesh;
            public Texture2D texture;
            public Material runtimeMaterial;

            public float sizeX;
            public float sizeY;
            public float sizeZ;
            public float perimeterM;
            public float uFace1; // South -> East seam
            public float uFace2; // East  -> North seam
            public float uFace3; // North -> West seam

            public int texW;
            public int texH;
            public int wallRows;
            public int roofRowMin;
            public int roofRowMax;
            public float cellWidthM;
            public float cellHeightM;

            public float[] h;          // Physical ink film thickness (>= 0)
            public float[] hNext;      // Scratch advection buffer
            public float[] vy;         // Downward lubrication film velocity (m/s)
            public sbyte[] team;       // +1 Team A, -1 Team B, 0 Dry
            public sbyte[] teamNext;
            public float[] yieldMap;   // Spatially correlated micro-roughness yield stress threshold
            public int[] groundRow;    // Visible floor-intersection row per perimeter column
            public Vector3[] footWorldPos; // World floor contact position per perimeter column

            public Color32[] baseDryPixels;
            public Color32[] pixels;

            public bool hasAnyInk;
            public bool hasMobileInk;
            public bool textureDirty;
            public int activeColMin;
            public int activeColMax;
        }

        private struct SplashDroplet3D
        {
            public bool active;
            public int teamId;
            public Vector3 pos;
            public Vector3 vel;
            public float age;
            public float maxLife;
            public float baseScale;
            public GameObject go;
            public MeshRenderer renderer;
        }

        private struct SlopeRivulet
        {
            public bool active;
            public int teamId;
            public float localX;
            public float localZ;
            public float vx;
            public float vz;
            public float mass;
            public float age;
            public float nextStampAge;
        }

        private InkBlitzArenaInstance _arena;
        private readonly List<ObstacleFluidSurface> _surfaces = new List<ObstacleFluidSurface>(16);
        private readonly Dictionary<Collider, ObstacleFluidSurface> _surfaceByCollider = new Dictionary<Collider, ObstacleFluidSurface>(16);
        private readonly SplashDroplet3D[] _droplets = new SplashDroplet3D[DropletPoolSize];
        private readonly SlopeRivulet[] _slopeRivulets = new SlopeRivulet[MaxSlopeRivulets];

        private Transform _dropletRoot;
        private Material _dropletMatA;
        private Material _dropletMatB;
        private float _simAccumSec = 0f;
        private uint _rngState = 0x9E3779B9u;
        private bool _initialized = false;
        private bool _materialsBoundToRenderers = false;

        public int RegisteredObstacleCount => _surfaces.Count;

        public int ActiveWetObstacleCount
        {
            get
            {
                int c = 0;
                for (int i = 0; i < _surfaces.Count; i++)
                {
                    if (_surfaces[i].hasAnyInk) c++;
                }
                return c;
            }
        }

        public bool IsRegisteredObstacle(Collider col) => col != null && _surfaceByCollider.ContainsKey(col);

        public bool ShouldSimulateVisualFluid()
        {
            if (!enableVisualWallFluid || _arena == null || !_arena.IsRenderingEnabled)
                return false;
            var coord = _arena.Coordinator;
            if (coord != null && coord.IsTrainingSession)
                return false;
            if (Application.isPlaying && Time.timeScale > 1.85f)
                return false;
            return true;
        }

        public void Initialize(InkBlitzArenaInstance arena)
        {
            _arena = arena;
            RebuildForCurrentStage(arena);
            EnsureDropletPool();
            _initialized = true;
        }

        public void RebuildForCurrentStage(InkBlitzArenaInstance arena)
        {
            _arena = arena;
            DisposeSurfaces(restoreOriginalRenderers: true);
            ClearTransientParticlesAndRivulets();

            if (_arena == null) return;
            Transform wallsRoot = _arena.transform.Find("WallsAndCovers");
            if (wallsRoot == null && _arena.stageBuilder != null)
                wallsRoot = _arena.stageBuilder.transform.Find("WallsAndCovers");
            if (wallsRoot == null) return;

            InkBlitzArenaSettings cfg = _arena.settings ?? new InkBlitzArenaSettings();
            float halfW = cfg.arenaWidthX * 0.5f;
            float halfL = cfg.arenaLengthZ * 0.5f;

            Material templateTexMat = (_arena.turfGrid != null && _arena.turfGrid.templateFloorMaterial != null)
                ? _arena.turfGrid.templateFloorMaterial
                : (_arena.stageBuilder != null ? _arena.stageBuilder.defaultFloorMaterial : null);

            Color dryCoverCol = new Color(0.50f, 0.53f, 0.58f, 1f);
            if (_arena.stageBuilder != null && _arena.stageBuilder.coverMaterial != null)
            {
                Material cm = _arena.stageBuilder.coverMaterial;
                if (cm.HasProperty("_BaseColor")) dryCoverCol = cm.GetColor("_BaseColor");
                else if (cm.HasProperty("_Color")) dryCoverCol = cm.GetColor("_Color");
            }

            BoxCollider[] boxes = wallsRoot.GetComponentsInChildren<BoxCollider>(true);
            for (int i = 0; i < boxes.Length; i++)
            {
                BoxCollider bc = boxes[i];
                if (!IsPaintableInteriorObstacle(bc, halfW, halfL))
                    continue;

                MeshRenderer mr = bc.GetComponent<MeshRenderer>();
                MeshFilter mf = bc.GetComponent<MeshFilter>();
                if (mr == null || mf == null)
                    continue;

                Color baseCol = dryCoverCol;
                if (mr.sharedMaterial != null)
                {
                    if (mr.sharedMaterial.HasProperty("_BaseColor")) baseCol = mr.sharedMaterial.GetColor("_BaseColor");
                    else if (mr.sharedMaterial.HasProperty("_Color")) baseCol = mr.sharedMaterial.GetColor("_Color");
                }

                ObstacleFluidSurface surf = CreateObstacleSurface(bc, mf, mr, templateTexMat, baseCol, cfg, i);
                _surfaces.Add(surf);
                _surfaceByCollider[bc] = surf;
            }

            _materialsBoundToRenderers = false;
            SyncRendererMaterialBinding(ShouldSimulateVisualFluid());
            RefreshPaletteColors(cfg.teamAColor, cfg.teamBColor);
        }

        private static bool IsPaintableInteriorObstacle(BoxCollider bc, float halfW, float halfL)
        {
            if (bc == null || !bc.enabled || !bc.gameObject.activeSelf)
                return false;

            string objName = bc.gameObject.name ?? string.Empty;
            // Strictly exclude outer stage frame walls (`Wall_North`, `Wall_South`, `Wall_East`, `Wall_West`)!
            if (objName.StartsWith("Wall_", StringComparison.OrdinalIgnoreCase))
                return false;

            Vector3 loc = bc.transform.localPosition;
            if (Mathf.Abs(loc.x) >= halfW - 0.15f || Mathf.Abs(loc.z) >= halfL - 0.15f)
                return false;

            Vector3 ls = bc.transform.localScale;
            return ls.x >= 0.25f && ls.y >= 0.45f && ls.z >= 0.25f;
        }

        private ObstacleFluidSurface CreateObstacleSurface(
            BoxCollider bc,
            MeshFilter mf,
            MeshRenderer mr,
            Material templateTexMat,
            Color baseStoneColor,
            InkBlitzArenaSettings cfg,
            int seedOffset)
        {
            Transform tr = bc.transform;
            Vector3 scale = tr.localScale;
            float sx = Mathf.Max(0.3f, Mathf.Abs(scale.x));
            float sy = Mathf.Max(0.4f, Mathf.Abs(scale.y));
            float sz = Mathf.Max(0.3f, Mathf.Abs(scale.z));
            float perim = 2f * (sx + sz);

            float ppm = Mathf.Clamp(pixelsPerMeter, 14f, 36f);
            int texW = Mathf.Clamp(Mathf.RoundToInt(perim * ppm), 88, 320);
            int wallRows = Mathf.Clamp(Mathf.RoundToInt(sy * ppm), 28, 72);
            int texH = Mathf.Max(wallRows + 10, Mathf.CeilToInt(wallRows / VWallTop));
            int roofMin = Mathf.Clamp(Mathf.FloorToInt(texH * VRoofMin), wallRows + 2, texH - 4);
            int roofMax = Mathf.Clamp(Mathf.CeilToInt(texH * VRoofMax), roofMin + 2, texH - 1);

            float u1 = sx / perim;
            float u2 = (sx + sz) / perim;
            float u3 = (2f * sx + sz) / perim;

            int totalPx = texW * texH;
            var surf = new ObstacleFluidSurface
            {
                collider = bc,
                transform = tr,
                renderer = mr,
                meshFilter = mf,
                originalSharedMaterial = mr.sharedMaterial,
                originalSharedMesh = mf.sharedMesh,
                unwrappedMesh = BuildUnwrappedBoxMesh(bc.gameObject.name, u1, u2, u3),
                sizeX = sx,
                sizeY = sy,
                sizeZ = sz,
                perimeterM = perim,
                uFace1 = u1,
                uFace2 = u2,
                uFace3 = u3,
                texW = texW,
                texH = texH,
                wallRows = wallRows,
                roofRowMin = roofMin,
                roofRowMax = roofMax,
                cellWidthM = perim / texW,
                cellHeightM = sy / wallRows,
                h = new float[totalPx],
                hNext = new float[totalPx],
                vy = new float[totalPx],
                team = new sbyte[totalPx],
                teamNext = new sbyte[totalPx],
                yieldMap = new float[totalPx],
                groundRow = new int[texW],
                footWorldPos = new Vector3[texW],
                baseDryPixels = new Color32[totalPx],
                pixels = new Color32[totalPx],
                activeColMin = 0,
                activeColMax = texW - 1
            };

            // Precompute spatially smooth horizontal roughness (creates authentic 6-14cm wide paint rivulet fingers!)
            uint sSeed = (uint)(2166136261u ^ (uint)((seedOffset + 1) * 16777619));
            float[] colRough = new float[texW];
            for (int x = 0; x < texW; x++)
            {
                float ang1 = (x / (float)texW) * Mathf.PI * 2f * Mathf.Max(3f, Mathf.Round(perim * 1.6f));
                float ang2 = (x / (float)texW) * Mathf.PI * 2f * Mathf.Max(7f, Mathf.Round(perim * 3.7f));
                float ph1 = Hash01(sSeed + 11u) * Mathf.PI * 2f;
                float ph2 = Hash01(sSeed + 29u) * Mathf.PI * 2f;
                colRough[x] = 0.5f + 0.28f * Mathf.Sin(ang1 + ph1) + 0.22f * Mathf.Cos(ang2 + ph2);
            }

            Color32 cBase = ToColor32(baseStoneColor, 1.00f);
            Color32 cSeam = ToColor32(baseStoneColor, 0.91f);
            Color32 cTopCap = ToColor32(baseStoneColor, 1.05f);

            int seam1 = Mathf.Clamp(Mathf.RoundToInt(u1 * texW), 0, texW - 1);
            int seam2 = Mathf.Clamp(Mathf.RoundToInt(u2 * texW), 0, texW - 1);
            int seam3 = Mathf.Clamp(Mathf.RoundToInt(u3 * texW), 0, texW - 1);
            int midBandRow = Mathf.Clamp(Mathf.RoundToInt(wallRows * 0.52f), 2, wallRows - 2);

            Vector3 arenaOrigin = _arena != null ? _arena.transform.position : Vector3.zero;
            for (int x = 0; x < texW; x++)
            {
                float uNorm = (x + 0.5f) / texW;
                Vector3 localEdge = PerimeterUToLocalPoint(uNorm, -0.5f, u1, u2, u3, out Vector3 localOutN);
                Vector3 worldBase = tr.TransformPoint(localEdge + localOutN * 0.06f);
                Vector3 arenaLoc = worldBase - arenaOrigin;
                float terrainY = ArenaStageBuilder.SampleTerrainHeightLocal(arenaLoc.x, arenaLoc.z, cfg);
                float worldBottomY = tr.TransformPoint(new Vector3(localEdge.x, -0.5f, localEdge.z)).y;
                float buriedM = Mathf.Max(0f, terrainY - worldBottomY);
                surf.groundRow[x] = Mathf.Clamp(Mathf.FloorToInt(buriedM / Mathf.Max(0.01f, surf.cellHeightM)), 0, wallRows - 3);
                surf.footWorldPos[x] = new Vector3(worldBase.x, terrainY + 0.02f, worldBase.z);
            }

            for (int y = 0; y < texH; y++)
            {
                int row = y * texW;
                bool isHorizGroove = (y == midBandRow);
                bool isRoof = (y >= roofMin);
                for (int x = 0; x < texW; x++)
                {
                    float fine = Hash01(sSeed ^ (uint)(x * 73856093 ^ y * 19349663));
                    float pin = Mathf.Clamp(0.70f + 0.44f * colRough[x] + 0.16f * fine, 0.62f, 1.38f);
                    surf.yieldMap[row + x] = binghamYieldThickness * pin;

                    bool isCornerBevel = (x == 0 || x == texW - 1 || x == seam1 || x == seam2 || x == seam3);
                    Color32 dryPx = isRoof ? cTopCap : ((isCornerBevel || isHorizGroove) ? cSeam : cBase);
                    surf.baseDryPixels[row + x] = dryPx;
                    surf.pixels[row + x] = dryPx;
                }
            }

            surf.texture = new Texture2D(texW, texH, TextureFormat.RGBA32, false)
            {
                name = $"ObstacleFluidTex_{bc.gameObject.name}_{texW}x{texH}",
                wrapModeU = TextureWrapMode.Repeat,
                wrapModeV = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
                anisoLevel = 2
            };
            surf.texture.SetPixels32(surf.pixels);
            surf.texture.Apply(false);

            if (templateTexMat != null)
            {
                surf.runtimeMaterial = new Material(templateTexMat)
                {
                    name = $"ObstacleFluidMat_{bc.gameObject.name}"
                };
            }
            else
            {
                Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                surf.runtimeMaterial = new Material(sh)
                {
                    name = $"ObstacleFluidMat_{bc.gameObject.name}"
                };
            }

            if (surf.runtimeMaterial.HasProperty("_BaseColor")) surf.runtimeMaterial.SetColor("_BaseColor", Color.white);
            if (surf.runtimeMaterial.HasProperty("_Color")) surf.runtimeMaterial.SetColor("_Color", Color.white);
            if (surf.runtimeMaterial.HasProperty("_Smoothness")) surf.runtimeMaterial.SetFloat("_Smoothness", 0.18f);
            if (surf.runtimeMaterial.HasProperty("_Glossiness")) surf.runtimeMaterial.SetFloat("_Glossiness", 0.18f);
            if (surf.runtimeMaterial.HasProperty("_BaseMap")) surf.runtimeMaterial.SetTexture("_BaseMap", surf.texture);
            if (surf.runtimeMaterial.HasProperty("_MainTex")) surf.runtimeMaterial.SetTexture("_MainTex", surf.texture);

            return surf;
        }

        private static Mesh BuildUnwrappedBoxMesh(string baseName, float u1, float u2, float u3)
        {
            // 24-vertex unit cube [-0.5..+0.5]^3 with non-overlapping gravity-aligned side walls (+V = +Y up)
            Vector3[] verts =
            {
                // 0..3: South (-Z), outward normal (0, 0, -1)
                new Vector3(-0.5f, -0.5f, -0.5f),
                new Vector3(-0.5f,  0.5f, -0.5f),
                new Vector3( 0.5f,  0.5f, -0.5f),
                new Vector3( 0.5f, -0.5f, -0.5f),
                // 4..7: East (+X), outward normal (+1, 0, 0)
                new Vector3( 0.5f, -0.5f, -0.5f),
                new Vector3( 0.5f,  0.5f, -0.5f),
                new Vector3( 0.5f,  0.5f,  0.5f),
                new Vector3( 0.5f, -0.5f,  0.5f),
                // 8..11: North (+Z), outward normal (0, 0, +1)
                new Vector3( 0.5f, -0.5f,  0.5f),
                new Vector3( 0.5f,  0.5f,  0.5f),
                new Vector3(-0.5f,  0.5f,  0.5f),
                new Vector3(-0.5f, -0.5f,  0.5f),
                // 12..15: West (-X), outward normal (-1, 0, 0)
                new Vector3(-0.5f, -0.5f,  0.5f),
                new Vector3(-0.5f,  0.5f,  0.5f),
                new Vector3(-0.5f,  0.5f, -0.5f),
                new Vector3(-0.5f, -0.5f, -0.5f),
                // 16..19: Top (+Y), outward normal (0, +1, 0)
                new Vector3(-0.5f,  0.5f, -0.5f),
                new Vector3(-0.5f,  0.5f,  0.5f),
                new Vector3( 0.5f,  0.5f,  0.5f),
                new Vector3( 0.5f,  0.5f, -0.5f),
                // 20..23: Bottom (-Y), outward normal (0, -1, 0)
                new Vector3(-0.5f, -0.5f,  0.5f),
                new Vector3(-0.5f, -0.5f, -0.5f),
                new Vector3( 0.5f, -0.5f, -0.5f),
                new Vector3( 0.5f, -0.5f,  0.5f),
            };

            const float vMidDry = 0.80f;
            Vector2[] uvs =
            {
                // South
                new Vector2(0f, 0f), new Vector2(0f, VWallTop), new Vector2(u1, VWallTop), new Vector2(u1, 0f),
                // East
                new Vector2(u1, 0f), new Vector2(u1, VWallTop), new Vector2(u2, VWallTop), new Vector2(u2, 0f),
                // North
                new Vector2(u2, 0f), new Vector2(u2, VWallTop), new Vector2(u3, VWallTop), new Vector2(u3, 0f),
                // West
                new Vector2(u3, 0f), new Vector2(u3, VWallTop), new Vector2(1f, VWallTop), new Vector2(1f, 0f),
                // Top Roof (isolated band above VRoofMin)
                new Vector2(0.02f, VRoofMin), new Vector2(0.02f, VRoofMax), new Vector2(0.98f, VRoofMax), new Vector2(0.98f, VRoofMin),
                // Bottom (unseen dry gutter)
                new Vector2(0.5f, vMidDry), new Vector2(0.5f, vMidDry), new Vector2(0.5f, vMidDry), new Vector2(0.5f, vMidDry),
            };

            int[] tris = new int[6 * 6];
            for (int f = 0; f < 6; f++)
            {
                int b = f * 4;
                tris[f * 6 + 0] = b + 0;
                tris[f * 6 + 1] = b + 1;
                tris[f * 6 + 2] = b + 2;
                tris[f * 6 + 3] = b + 0;
                tris[f * 6 + 4] = b + 2;
                tris[f * 6 + 5] = b + 3;
            }

            var mesh = new Mesh { name = $"UnwrappedObstacleCube_{baseName}" };
            mesh.vertices = verts;
            mesh.uv = uvs;
            mesh.triangles = tris;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static Vector3 PerimeterUToLocalPoint(float u, float localY, float u1, float u2, float u3, out Vector3 outNormal)
        {
            u = u - Mathf.Floor(u);
            if (u < u1)
            {
                float t = u / Mathf.Max(1e-4f, u1);
                outNormal = Vector3.back;
                return new Vector3(Mathf.Lerp(-0.5f, 0.5f, t), localY, -0.5f);
            }
            if (u < u2)
            {
                float t = (u - u1) / Mathf.Max(1e-4f, u2 - u1);
                outNormal = Vector3.right;
                return new Vector3(0.5f, localY, Mathf.Lerp(-0.5f, 0.5f, t));
            }
            if (u < u3)
            {
                float t = (u - u2) / Mathf.Max(1e-4f, u3 - u2);
                outNormal = Vector3.forward;
                return new Vector3(Mathf.Lerp(0.5f, -0.5f, t), localY, 0.5f);
            }
            else
            {
                float t = (u - u3) / Mathf.Max(1e-4f, 1f - u3);
                outNormal = Vector3.left;
                return new Vector3(-0.5f, localY, Mathf.Lerp(0.5f, -0.5f, t));
            }
        }

        private void SyncRendererMaterialBinding(bool useFluidMaterials)
        {
            if (_materialsBoundToRenderers == useFluidMaterials)
                return;
            for (int i = 0; i < _surfaces.Count; i++)
            {
                var s = _surfaces[i];
                if (s.renderer == null || s.meshFilter == null) continue;
                if (useFluidMaterials)
                {
                    if (s.unwrappedMesh != null) s.meshFilter.sharedMesh = s.unwrappedMesh;
                    if (s.runtimeMaterial != null) s.renderer.sharedMaterial = s.runtimeMaterial;
                }
                else
                {
                    if (s.originalSharedMesh != null) s.meshFilter.sharedMesh = s.originalSharedMesh;
                    if (s.originalSharedMaterial != null) s.renderer.sharedMaterial = s.originalSharedMaterial;
                }
            }
            _materialsBoundToRenderers = useFluidMaterials;
        }

        public void ResetAllFluid()
        {
            _simAccumSec = 0f;
            ClearTransientParticlesAndRivulets();
            bool shouldRender = ShouldSimulateVisualFluid();
            SyncRendererMaterialBinding(shouldRender);

            for (int i = 0; i < _surfaces.Count; i++)
            {
                ObstacleFluidSurface s = _surfaces[i];
                if (!s.hasAnyInk) continue;
                Array.Clear(s.h, 0, s.h.Length);
                Array.Clear(s.vy, 0, s.vy.Length);
                Array.Clear(s.team, 0, s.team.Length);
                Array.Copy(s.baseDryPixels, s.pixels, s.pixels.Length);
                s.hasAnyInk = false;
                s.hasMobileInk = false;
                s.textureDirty = false;
                if (shouldRender && s.texture != null)
                {
                    s.texture.SetPixels32(s.pixels);
                    s.texture.Apply(false);
                }
            }
        }

        public void RefreshPaletteColors(Color teamA, Color teamB)
        {
            if (_dropletMatA != null) ApplyDropletMatColor(_dropletMatA, teamA);
            if (_dropletMatB != null) ApplyDropletMatColor(_dropletMatB, teamB);

            bool shouldRender = ShouldSimulateVisualFluid();
            for (int i = 0; i < _surfaces.Count; i++)
            {
                ObstacleFluidSurface s = _surfaces[i];
                if (!s.hasAnyInk) continue;
                RenderSurfacePixels(s, 0, s.texW - 1, includeRoof: true);
                if (shouldRender && s.texture != null)
                {
                    s.texture.SetPixels32(s.pixels);
                    s.texture.Apply(false);
                    s.textureDirty = false;
                }
            }
        }

        public bool OnEnvironmentImpact(
            Collider hitCollider,
            Vector3 worldPoint,
            Vector3 worldNormal,
            Vector3 projectileVelocity,
            int teamId,
            float splashRadiusM,
            bool isBlast = false,
            bool spawn3DSplash = true)
        {
            if (!ShouldSimulateVisualFluid() || hitCollider == null || teamId == 0)
                return false;

            SyncRendererMaterialBinding(true);

            // Floor / Ramp impact: trigger downhill viscous rivulets on sloped hills only!
            if (worldNormal.y > 0.58f && !_surfaceByCollider.ContainsKey(hitCollider))
            {
                TrySpawnHillSlopeRivulets(worldPoint, worldNormal, teamId, splashRadiusM);
                return false;
            }

            if (!_surfaceByCollider.TryGetValue(hitCollider, out ObstacleFluidSurface surf))
                return false;

            sbyte tSign = (teamId > 0) ? (sbyte)1 : (sbyte)-1;
            Vector3 localPt = surf.transform.InverseTransformPoint(worldPoint);
            Vector3 localN = surf.transform.InverseTransformDirection(worldNormal);
            Vector3 localVel = surf.transform.InverseTransformDirection(projectileVelocity);

            if (localN.y > 0.62f)
            {
                StampObstacleRoofAndRimSpill(surf, localPt, tSign, splashRadiusM, isBlast);
            }
            else
            {
                StampObstacleVerticalWall(surf, localPt, localN, localVel, tSign, splashRadiusM, isBlast);
            }

            if (spawn3DSplash)
            {
                int count = isBlast ? 12 : 6;
                SpawnSplashDroplets3D(worldPoint, worldNormal, projectileVelocity, teamId, count, isBlast);
            }

            return true;
        }

        private void StampObstacleRoofAndRimSpill(
            ObstacleFluidSurface surf,
            Vector3 localPt,
            sbyte tSign,
            float splashRadiusM,
            bool isBlast)
        {
            int roofH = Mathf.Max(2, surf.roofRowMax - surf.roofRowMin + 1);
            float uCenter = Mathf.Clamp01(localPt.x + 0.5f);
            float vCenter = Mathf.Clamp01(localPt.z + 0.5f);
            int cx = Mathf.Clamp(Mathf.RoundToInt(uCenter * (surf.texW - 1)), 2, surf.texW - 3);
            int cy = Mathf.Clamp(surf.roofRowMin + Mathf.RoundToInt(vCenter * (roofH - 1)), surf.roofRowMin, surf.roofRowMax);

            int rx = Mathf.Max(2, Mathf.RoundToInt((splashRadiusM / Mathf.Max(0.25f, surf.sizeX)) * surf.texW * 0.55f));
            int ry = Mathf.Max(1, Mathf.RoundToInt((splashRadiusM / Mathf.Max(0.25f, surf.sizeZ)) * roofH * 0.55f));
            float massPeak = isBlast ? 0.62f : 0.42f;

            for (int dy = -ry; dy <= ry; dy++)
            {
                int py = cy + dy;
                if (py < surf.roofRowMin || py > surf.roofRowMax) continue;
                float ny = dy / (float)Mathf.Max(1, ry);
                int row = py * surf.texW;
                for (int dx = -rx; dx <= rx; dx++)
                {
                    int px = cx + dx;
                    if (px < 1 || px >= surf.texW - 1) continue;
                    float nx = dx / (float)Mathf.Max(1, rx);
                    float r2 = nx * nx + ny * ny;
                    if (r2 > 1.0f) continue;
                    int idx = row + px;
                    surf.h[idx] = Mathf.Min(1.25f, surf.h[idx] + massPeak * (1f - 0.65f * r2));
                    surf.team[idx] = tSign;
                }
            }

            // Rim spillover: if ink lands near any roof edge, feed excess mass over the top rim onto that vertical face!
            float normRx = splashRadiusM / Mathf.Max(0.3f, surf.sizeX);
            float normRz = splashRadiusM / Mathf.Max(0.3f, surf.sizeZ);
            if (localPt.z - normRz < -0.44f) // South rim
                InjectTopRimSpill(surf, Mathf.Lerp(0f, surf.uFace1, uCenter), splashRadiusM * 0.65f, tSign, isBlast);
            if (localPt.z + normRz > 0.44f)  // North rim
                InjectTopRimSpill(surf, Mathf.Lerp(surf.uFace2, surf.uFace3, 1f - uCenter), splashRadiusM * 0.65f, tSign, isBlast);
            if (localPt.x + normRx > 0.44f)  // East rim
                InjectTopRimSpill(surf, Mathf.Lerp(surf.uFace1, surf.uFace2, vCenter), splashRadiusM * 0.65f, tSign, isBlast);
            if (localPt.x - normRx < -0.44f) // West rim
                InjectTopRimSpill(surf, Mathf.Lerp(surf.uFace3, 1f, 1f - vCenter), splashRadiusM * 0.65f, tSign, isBlast);

            surf.hasAnyInk = true;
            RenderSurfacePixels(surf, 0, surf.texW - 1, includeRoof: true);
            surf.textureDirty = true;
        }

        private void InjectTopRimSpill(ObstacleFluidSurface surf, float uCenter, float halfSpanM, sbyte tSign, bool isBlast)
        {
            int cx = WrapX(Mathf.RoundToInt(uCenter * surf.texW), surf.texW);
            int halfCols = Mathf.Clamp(Mathf.RoundToInt(halfSpanM / Mathf.Max(0.01f, surf.cellWidthM)), 2, surf.texW / 4);
            float rimLoad = isBlast ? 0.88f : 0.60f;

            for (int dx = -halfCols; dx <= halfCols; dx++)
            {
                float frac = 1f - Mathf.Abs(dx) / (float)(halfCols + 1);
                int px = WrapX(cx + dx, surf.texW);
                for (int r = 1; r <= 2; r++)
                {
                    int py = surf.wallRows - r;
                    if (py <= surf.groundRow[px]) continue;
                    int idx = py * surf.texW + px;
                    DepositInkCell(surf, idx, tSign, rimLoad * frac * (r == 1 ? 1.0f : 0.65f), initialVy: 0.25f);
                }
            }
            ExpandActiveColumnWindow(surf, cx - halfCols - 2, cx + halfCols + 2);
            surf.hasMobileInk = true;
        }

        private void StampObstacleVerticalWall(
            ObstacleFluidSurface surf,
            Vector3 localPt,
            Vector3 localN,
            Vector3 localVel,
            sbyte tSign,
            float splashRadiusM,
            bool isBlast)
        {
            // Determine dominant hit vertical face & tangent right axis around perimeter
            float uHit;
            Vector3 tangentRight;
            if (Mathf.Abs(localN.z) >= Mathf.Abs(localN.x))
            {
                if (localN.z <= 0f) // South (-Z)
                {
                    uHit = Mathf.Lerp(0f, surf.uFace1, Mathf.Clamp01(localPt.x + 0.5f));
                    tangentRight = Vector3.right;
                }
                else // North (+Z)
                {
                    uHit = Mathf.Lerp(surf.uFace2, surf.uFace3, Mathf.Clamp01(0.5f - localPt.x));
                    tangentRight = Vector3.left;
                }
            }
            else
            {
                if (localN.x >= 0f) // East (+X)
                {
                    uHit = Mathf.Lerp(surf.uFace1, surf.uFace2, Mathf.Clamp01(localPt.z + 0.5f));
                    tangentRight = Vector3.forward;
                }
                else // West (-X)
                {
                    uHit = Mathf.Lerp(surf.uFace3, 1f, Mathf.Clamp01(0.5f - localPt.z));
                    tangentRight = Vector3.back;
                }
            }

            float vHit01 = Mathf.Clamp01(localPt.y + 0.5f);
            int cx = WrapX(Mathf.RoundToInt(uHit * surf.texW), surf.texW);
            int cy = Mathf.Clamp(Mathf.RoundToInt(vHit01 * (surf.wallRows - 1)), 0, surf.wallRows - 1);

            // Anisotropic tangential shear elongation from oblique grazing shots!
            float vHoriz = Vector3.Dot(localVel, tangentRight) * surf.sizeX;
            float vVert = localVel.y * surf.sizeY;
            Vector2 tang2D = new Vector2(vHoriz, vVert);
            float tangMag = tang2D.magnitude;
            Vector2 shearDir = tangMag > 0.8f ? (tang2D / tangMag) : Vector2.down;
            float shearStretch = Mathf.Clamp(1f + tangMag * 0.035f, 1.0f, 1.65f);

            float effRadiusM = Mathf.Clamp(splashRadiusM * (isBlast ? 0.78f : 0.62f), 0.28f, 1.65f);
            int radX = Mathf.Clamp(Mathf.CeilToInt((effRadiusM * shearStretch) / Mathf.Max(0.01f, surf.cellWidthM)), 3, surf.texW / 2 - 1);
            int radY = Mathf.Clamp(Mathf.CeilToInt((effRadiusM * shearStretch) / Mathf.Max(0.01f, surf.cellHeightM)), 3, surf.wallRows - 1);

            float corePeakH = isBlast ? 1.08f : 0.74f;
            float impactDownVy = Mathf.Clamp(-vVert * 0.04f + 0.20f, 0.05f, maxDripSpeedMps);
            float phase = NextRand01() * Mathf.PI * 2f;
            float phase2 = NextRand01() * Mathf.PI * 2f;

            for (int dy = -radY; dy <= radY; dy++)
            {
                int py = cy + dy;
                if (py < 0 || py >= surf.wallRows) continue;
                float my = dy * surf.cellHeightM;
                int row = py * surf.texW;

                for (int dx = -radX; dx <= radX; dx++)
                {
                    int px = WrapX(cx + dx, surf.texW);
                    if (py < surf.groundRow[px]) continue;

                    float mx = dx * surf.cellWidthM;
                    // Decompose into shear-aligned & perpendicular axes
                    float along = mx * shearDir.x + my * shearDir.y;
                    float perp = -mx * shearDir.y + my * shearDir.x;
                    float anisoDist = Mathf.Sqrt((along / shearStretch) * (along / shearStretch) + perp * perp);
                    float rNorm = anisoDist / effRadiusM;
                    if (rNorm > 1.28f) continue;

                    float theta = Mathf.Atan2(my, mx);
                    float lobe = 1f + 0.20f * Mathf.Sin(5f * theta + phase) + 0.14f * Mathf.Cos(9f * theta + phase2);
                    float effR = rNorm / Mathf.Max(0.65f, lobe);
                    if (effR > 1.0f) continue;

                    // Thick center (> yield stress -> drips!) + thin pinned splatter rim (<= yield stress -> stays crisp!)
                    float addH;
                    if (effR < 0.62f)
                    {
                        float tCore = effR / 0.62f;
                        addH = corePeakH * (1f - 0.52f * tCore * tCore);
                    }
                    else
                    {
                        float tRim = (effR - 0.62f) / 0.38f;
                        addH = binghamYieldThickness * 0.86f * (1f - tRim);
                    }

                    DepositInkCell(surf, row + px, tSign, addH, impactDownVy * (1f - 0.5f * effR));
                }
            }

            // Radial ballistic splash rays & satellite beads (pinned micro-splatter droplets around main impact)
            int numRays = isBlast ? 10 : 6;
            for (int k = 0; k < numRays; k++)
            {
                float baseAng = (k / (float)numRays) * Mathf.PI * 2f + (NextRand01() - 0.5f) * 0.45f;
                Vector2 dir = new Vector2(Mathf.Cos(baseAng), Mathf.Sin(baseAng));
                dir = (dir + shearDir * 0.32f).normalized;
                float rayLenM = effRadiusM * Mathf.Lerp(1.05f, 1.62f, NextRand01());
                int steps = Mathf.Clamp(Mathf.RoundToInt(rayLenM / Mathf.Max(0.015f, surf.cellWidthM)), 3, 16);
                for (int s = 2; s <= steps; s++)
                {
                    float distM = (s / (float)steps) * rayLenM;
                    int px = WrapX(cx + Mathf.RoundToInt((dir.x * distM) / surf.cellWidthM), surf.texW);
                    int py = cy + Mathf.RoundToInt((dir.y * distM) / surf.cellHeightM);
                    if (py < 0 || py >= surf.wallRows || py < surf.groundRow[px]) continue;
                    // Beaded satellite droplet at tip, thin streak along ray
                    bool isTipBead = (s >= steps - 1);
                    float beadH = isTipBead ? (binghamYieldThickness * 0.92f) : (binghamYieldThickness * 0.48f);
                    DepositInkCell(surf, py * surf.texW + px, tSign, beadH, 0.05f);
                }
            }

            ExpandActiveColumnWindow(surf, cx - radX - 8, cx + radX + 8);
            surf.hasAnyInk = true;
            surf.hasMobileInk = true;
            RenderSurfacePixels(surf, surf.activeColMin, surf.activeColMax, includeRoof: false);
            surf.textureDirty = true;
        }

        private static void DepositInkCell(ObstacleFluidSurface surf, int idx, sbyte tSign, float addH, float initialVy)
        {
            if (addH <= 1e-4f) return;
            sbyte prevTeam = surf.team[idx];
            if (prevTeam == 0 || prevTeam == tSign)
            {
                surf.h[idx] = Mathf.Min(1.55f, surf.h[idx] + addH);
                surf.team[idx] = tSign;
            }
            else
            {
                // Opposing ink washes over / displaces prior coat when thicker, or neutralizes down before recoloring
                if (addH >= surf.h[idx] * 0.55f)
                {
                    surf.h[idx] = Mathf.Clamp(addH + 0.25f * surf.h[idx], 0.12f, 1.55f);
                    surf.team[idx] = tSign;
                }
                else
                {
                    surf.h[idx] = Mathf.Max(0.10f, surf.h[idx] - 0.5f * addH);
                }
            }
            if (initialVy > surf.vy[idx])
                surf.vy[idx] = initialVy;
        }

        private static void ExpandActiveColumnWindow(ObstacleFluidSurface surf, int minCol, int maxCol)
        {
            if (!surf.hasMobileInk)
            {
                if (maxCol - minCol >= surf.texW - 2)
                {
                    surf.activeColMin = 0;
                    surf.activeColMax = surf.texW - 1;
                }
                else
                {
                    surf.activeColMin = WrapX(minCol, surf.texW);
                    surf.activeColMax = WrapX(maxCol, surf.texW);
                }
                return;
            }
            // Conservative full-width promotion when wrap-around or multiple faces are active (texW <= 320 is tiny!)
            surf.activeColMin = 0;
            surf.activeColMax = surf.texW - 1;
        }

        public void StepVisualFluid(float dt)
        {
            if (!_initialized || dt <= 0f)
                return;

            bool shouldRun = ShouldSimulateVisualFluid();
            SyncRendererMaterialBinding(shouldRun);
            if (!shouldRun)
                return;

            StepDroplets3D(dt);
            StepHillSlopeRivulets(dt);

            _simAccumSec = Mathf.Min(_simAccumSec + dt, SimStepDt * 3f);
            while (_simAccumSec >= SimStepDt)
            {
                _simAccumSec -= SimStepDt;
                for (int i = 0; i < _surfaces.Count; i++)
                {
                    ObstacleFluidSurface s = _surfaces[i];
                    if (s.hasMobileInk)
                        StepBinghamThinFilmPDE(s, SimStepDt);
                }
            }
        }

        private void LateUpdate()
        {
            if (!ShouldSimulateVisualFluid())
                return;

            for (int i = 0; i < _surfaces.Count; i++)
            {
                ObstacleFluidSurface s = _surfaces[i];
                if (s.textureDirty && s.texture != null)
                {
                    s.texture.SetPixels32(s.pixels);
                    s.texture.Apply(false);
                    s.textureDirty = false;
                }
            }
        }

        private void StepBinghamThinFilmPDE(ObstacleFluidSurface s, float dt)
        {
            int w = s.texW;
            int rows = s.wallRows;
            float invDy = 1f / Mathf.Max(0.012f, s.cellHeightM);

            Array.Copy(s.h, 0, s.hNext, 0, w * rows);
            Array.Copy(s.team, 0, s.teamNext, 0, w * rows);

            bool anyStillMobile = false;

            for (int x = 0; x < w; x++)
            {
                int gRow = s.groundRow[x];
                int xl = (x == 0) ? (w - 1) : (x - 1);
                int xr = (x == w - 1) ? 0 : (x + 1);

                for (int y = rows - 1; y >= gRow; y--)
                {
                    int idx = y * w + x;
                    float hCur = s.h[idx];
                    if (hCur <= 1e-4f)
                    {
                        s.vy[idx] = 0f;
                        continue;
                    }

                    sbyte tCur = s.team[idx];
                    float hYield = s.yieldMap[idx];
                    float hMob = hCur - hYield;
                    if (hMob <= 0.004f)
                    {
                        s.vy[idx] *= 0.45f;
                        continue;
                    }

                    // At the visible floor line: pool into a base fillet and drip excess onto the floor!
                    if (y == gRow)
                    {
                        float baseCap = hYield * 1.48f;
                        if (hCur > baseCap)
                        {
                            float runoff = Mathf.Min(hCur - baseCap, 0.35f * hMob);
                            s.hNext[idx] -= runoff;
                            if (runoff > 0.018f && _arena != null && _arena.turfGrid != null)
                            {
                                Vector3 footPos = s.footWorldPos[x];
                                _arena.turfGrid.PaintVisualDripOnly(footPos, 0.22f + runoff * 0.45f, tCur);
                                Vector3 arenaLoc = footPos - _arena.transform.position;
                                Vector3 floorN = ArenaStageBuilder.SampleTerrainNormalLocal(arenaLoc.x, arenaLoc.z, _arena.settings);
                                if (floorN.y < 0.988f && NextRand01() < 0.45f)
                                    SpawnSingleSlopeRivulet(arenaLoc.x, arenaLoc.z, tCur, 0.28f + runoff);
                            }
                            if (s.hNext[idx] > baseCap + 0.01f)
                                anyStillMobile = true;
                        }
                        continue;
                    }

                    int idxBelow = (y - 1) * w + x;
                    float hBelow = s.h[idxBelow];

                    // Contact-angle meniscus hysteresis: dry wall below resists wetting until a plump teardrop head forms!
                    if (hBelow < 0.025f && hCur < hYield * 1.24f)
                    {
                        anyStillMobile = (s.vy[idx] > 0.04f);
                        s.vy[idx] *= 0.60f;
                        continue;
                    }

                    // Lubrication theory quadratic gravity velocity: v_eq ~ (rho*g / 3*mu) * h_mob^2
                    float vEq = maxDripSpeedMps * (float)Math.Tanh(gravityMobilityGain * hMob * hMob * 4.2f);
                    float vyNew = Mathf.Lerp(s.vy[idx], vEq, 0.55f);
                    s.vy[idx] = vyNew;

                    // CFL-bounded upwind downward flux (strictly <= 0.46 * hMob guarantees positivity & conservation!)
                    float cflFrac = Mathf.Min(0.46f, vyNew * dt * invDy);
                    float fluxDown = hMob * cflFrac;
                    if (fluxDown <= 0.0015f)
                        continue;

                    anyStillMobile = true;
                    s.hNext[idx] -= fluxDown;

                    if (s.teamNext[idxBelow] == 0 || s.teamNext[idxBelow] == tCur || fluxDown >= s.hNext[idxBelow] * 0.45f)
                        s.teamNext[idxBelow] = tCur;
                    s.hNext[idxBelow] = Mathf.Min(1.60f, s.hNext[idxBelow] + fluxDown);
                    if (vyNew * 0.82f > s.vy[idxBelow])
                        s.vy[idxBelow] = vyNew * 0.82f;

                    // Subtle capillary lateral rounding at swollen droplet tips (prevents 1-px needle artifacts)
                    if (hCur > hYield * 1.55f)
                    {
                        float latFlux = 0.035f * (hCur - hYield * 1.55f);
                        int idxL = y * w + xl;
                        int idxR = y * w + xr;
                        if (s.h[idxL] + 0.08f < hCur && s.hNext[idx] > latFlux * 2f)
                        {
                            s.hNext[idx] -= latFlux;
                            s.hNext[idxL] += latFlux;
                            if (s.teamNext[idxL] == 0) s.teamNext[idxL] = tCur;
                        }
                        if (s.h[idxR] + 0.08f < hCur && s.hNext[idx] > latFlux * 2f)
                        {
                            s.hNext[idx] -= latFlux;
                            s.hNext[idxR] += latFlux;
                            if (s.teamNext[idxR] == 0) s.teamNext[idxR] = tCur;
                        }
                    }
                }
            }

            Array.Copy(s.hNext, 0, s.h, 0, w * rows);
            Array.Copy(s.teamNext, 0, s.team, 0, w * rows);
            s.hasMobileInk = anyStillMobile;

            RenderSurfacePixels(s, 0, w - 1, includeRoof: false);
            s.textureDirty = true;
        }

        private void RenderSurfacePixels(ObstacleFluidSurface s, int colMin, int colMax, bool includeRoof)
        {
            InkBlitzArenaSettings cfg = (_arena != null && _arena.settings != null) ? _arena.settings : new InkBlitzArenaSettings();
            Color colA = cfg.teamAColor;
            Color colB = cfg.teamBColor;

            int w = s.texW;
            int maxY = includeRoof ? (s.texH - 1) : (s.wallRows - 1);
            int startX = Mathf.Clamp(colMin, 0, w - 1);
            int endX = Mathf.Clamp(colMax, 0, w - 1);

            for (int y = 0; y <= maxY; y++)
            {
                int row = y * w;
                int rowUp = Mathf.Min(maxY, y + 1) * w;
                int rowDn = Mathf.Max(0, y - 1) * w;

                for (int x = startX; x <= endX; x++)
                {
                    int idx = row + x;
                    float hVal = s.h[idx];
                    sbyte t = s.team[idx];
                    Color32 basePx = s.baseDryPixels[idx];

                    if (t == 0 || hVal <= 0.012f)
                    {
                        s.pixels[idx] = basePx;
                        continue;
                    }

                    Color inkC = (t > 0) ? colA : colB;

                    // 3D film-thickness normal relief: top meniscus catches light, swollen bottom drip bead casts subtle lip shadow
                    int xl = (x == 0) ? (w - 1) : (x - 1);
                    int xr = (x == w - 1) ? 0 : (x + 1);
                    float dhx = s.h[row + xr] - s.h[row + xl];
                    float dhy = (y < s.wallRows) ? (s.h[rowUp + x] - s.h[rowDn + x]) : 0f;

                    float relief = Mathf.Clamp((-0.35f * dhx - 0.85f * dhy) * 0.55f, -0.14f, 0.20f);
                    float thickRichness = Mathf.Clamp(hVal * 0.10f, 0f, 0.08f);
                    float shade = 1.0f + relief - thickRichness * 0.35f;

                    float alpha = Mathf.Clamp01(hVal / 0.085f);
                    // Damp darkened concrete halo right at ultra-thin contact edge
                    float dryDarken = (alpha < 0.92f) ? 0.88f : 1.0f;

                    float br = (basePx.r / 255f) * dryDarken;
                    float bg = (basePx.g / 255f) * dryDarken;
                    float bb = (basePx.b / 255f) * dryDarken;

                    float ir = Mathf.Clamp01(inkC.r * shade + Mathf.Max(0f, relief) * 0.28f);
                    float ig = Mathf.Clamp01(inkC.g * shade + Mathf.Max(0f, relief) * 0.28f);
                    float ib = Mathf.Clamp01(inkC.b * shade + Mathf.Max(0f, relief) * 0.28f);

                    s.pixels[idx] = new Color32(
                        (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(br, ir, alpha) * 255f), 0, 255),
                        (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(bg, ig, alpha) * 255f), 0, 255),
                        (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(bb, ib, alpha) * 255f), 0, 255),
                        255
                    );
                }
            }
        }

        // =========================================================================================
        // 3D Ballistic Rebound Splash Droplets (Collider-Free so AI Raycasts Never Intersect!)
        // =========================================================================================

        private void EnsureDropletPool()
        {
            if (_dropletRoot != null) return;

            InkBlitzArenaSettings cfg = (_arena != null && _arena.settings != null) ? _arena.settings : new InkBlitzArenaSettings();
            Material seedMat = (_arena != null && _arena.stageBuilder != null && _arena.stageBuilder.teamAPedestalMaterial != null)
                ? _arena.stageBuilder.teamAPedestalMaterial
                : null;

            if (seedMat != null)
            {
                _dropletMatA = new Material(seedMat) { name = "WallSplashDrop_TeamA" };
                _dropletMatB = new Material(seedMat) { name = "WallSplashDrop_TeamB" };
            }
            else
            {
                Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                _dropletMatA = new Material(sh) { name = "WallSplashDrop_TeamA" };
                _dropletMatB = new Material(sh) { name = "WallSplashDrop_TeamB" };
            }
            ApplyDropletMatColor(_dropletMatA, cfg.teamAColor);
            ApplyDropletMatColor(_dropletMatB, cfg.teamBColor);

            var rootGo = new GameObject("WallSplashDropletPool_VisualOnly");
            rootGo.transform.SetParent(transform, false);
            _dropletRoot = rootGo.transform;

            for (int i = 0; i < DropletPoolSize; i++)
            {
                GameObject sp = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sp.name = $"SplashDrop_{i}";
                sp.transform.SetParent(_dropletRoot, false);
                Collider col = sp.GetComponent<Collider>();
                if (col != null) DestroySafe(col); // Strictly zero physics collider!
                var mr = sp.GetComponent<MeshRenderer>();
                if (mr != null)
                {
                    mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    mr.receiveShadows = false;
                }
                sp.SetActive(false);

                _droplets[i] = new SplashDroplet3D
                {
                    active = false,
                    go = sp,
                    renderer = mr
                };
            }
        }

        private void SpawnSplashDroplets3D(Vector3 hitPos, Vector3 hitNormal, Vector3 inVel, int teamId, int count, bool isBlast)
        {
            EnsureDropletPool();
            float inSpeed = inVel.magnitude;
            Vector3 refl = inSpeed > 0.5f ? Vector3.Reflect(inVel / inSpeed, hitNormal) : hitNormal;
            Vector3 outBasis = (refl * 0.55f + hitNormal * 0.75f + Vector3.up * 0.30f).normalized;

            for (int n = 0; n < count; n++)
            {
                int slotIdx = -1;
                float oldest = -1f;
                for (int i = 0; i < DropletPoolSize; i++)
                {
                    if (!_droplets[i].active) { slotIdx = i; break; }
                    if (_droplets[i].age > oldest) { oldest = _droplets[i].age; slotIdx = i; }
                }
                if (slotIdx < 0) slotIdx = 0;

                Vector3 jitter = new Vector3(NextRand01() - 0.5f, NextRand01() * 0.75f - 0.15f, NextRand01() - 0.5f) * 1.15f;
                Vector3 dir = (outBasis + jitter).normalized;
                if (Vector3.Dot(dir, hitNormal) < 0.12f)
                    dir = (dir + hitNormal * 0.45f).normalized;

                float spd = Mathf.Lerp(2.4f, isBlast ? 6.8f : 4.9f, NextRand01());
                float sc = Mathf.Lerp(0.065f, isBlast ? 0.145f : 0.115f, NextRand01());

                ref SplashDroplet3D d = ref _droplets[slotIdx];
                d.active = true;
                d.teamId = teamId;
                d.pos = hitPos + hitNormal * 0.06f;
                d.vel = dir * spd;
                d.age = 0f;
                d.maxLife = Mathf.Lerp(0.24f, 0.52f, NextRand01());
                d.baseScale = sc;

                if (d.go != null)
                {
                    d.go.transform.position = d.pos;
                    d.go.transform.localScale = new Vector3(sc, sc, sc);
                    if (d.renderer != null)
                        d.renderer.sharedMaterial = (teamId > 0) ? _dropletMatA : _dropletMatB;
                    d.go.SetActive(true);
                }
            }
        }

        private void StepDroplets3D(float dt)
        {
            if (_arena == null) return;
            Vector3 arenaOrigin = _arena.transform.position;
            InkBlitzArenaSettings cfg = _arena.settings;

            for (int i = 0; i < DropletPoolSize; i++)
            {
                if (!_droplets[i].active) continue;
                ref SplashDroplet3D d = ref _droplets[i];
                d.age += dt;
                if (d.age >= d.maxLife)
                {
                    d.active = false;
                    if (d.go != null) d.go.SetActive(false);
                    continue;
                }

                d.vel.y -= 17.5f * dt;
                Vector3 next = d.pos + d.vel * dt;

                Vector3 loc = next - arenaOrigin;
                float floorY = arenaOrigin.y + ArenaStageBuilder.SampleTerrainHeightLocal(loc.x, loc.z, cfg);
                if (next.y <= floorY + 0.03f)
                {
                    Vector3 landWorld = new Vector3(next.x, floorY, next.z);
                    _arena.turfGrid?.PaintVisualDripOnly(landWorld, d.baseScale * 2.1f, d.teamId);
                    d.active = false;
                    if (d.go != null) d.go.SetActive(false);
                    continue;
                }

                d.pos = next;
                if (d.go != null)
                {
                    float shrink = 1f - 0.45f * (d.age / d.maxLife);
                    float s = d.baseScale * shrink;
                    d.go.transform.position = next;
                    d.go.transform.localScale = new Vector3(s * 0.9f, s * 1.25f, s * 0.9f);
                }
            }
        }

        // =========================================================================================
        // Hill-Slope Viscous Runoff Rivulets (Visual-Only Floor Layer — Zero Logical Grid Impact!)
        // =========================================================================================

        private void TrySpawnHillSlopeRivulets(Vector3 worldPoint, Vector3 worldNormal, int teamId, float radiusM)
        {
            if (_arena == null || _arena.turfGrid == null || worldNormal.y >= 0.988f)
                return;

            Vector3 loc = worldPoint - _arena.transform.position;
            Vector2 downhill = new Vector2(worldNormal.x, worldNormal.z);
            float slopeMag = downhill.magnitude;
            if (slopeMag < 0.14f) return;
            downhill /= slopeMag;
            Vector2 perp = new Vector2(-downhill.y, downhill.x);

            int count = (slopeMag > 0.26f) ? 3 : 2;
            for (int k = 0; k < count; k++)
            {
                float lat = (NextRand01() - 0.5f) * radiusM * 0.95f;
                float sx = loc.x + downhill.x * (radiusM * 0.55f) + perp.x * lat;
                float sz = loc.z + downhill.y * (radiusM * 0.55f) + perp.y * lat;
                SpawnSingleSlopeRivulet(sx, sz, teamId, Mathf.Lerp(0.32f, 0.54f, NextRand01()));
            }
        }

        private void SpawnSingleSlopeRivulet(float localX, float localZ, int teamId, float mass)
        {
            int idx = -1;
            float oldest = -1f;
            for (int i = 0; i < MaxSlopeRivulets; i++)
            {
                if (!_slopeRivulets[i].active) { idx = i; break; }
                if (_slopeRivulets[i].age > oldest) { oldest = _slopeRivulets[i].age; idx = i; }
            }
            if (idx < 0) idx = 0;

            _slopeRivulets[idx] = new SlopeRivulet
            {
                active = true,
                teamId = teamId,
                localX = localX,
                localZ = localZ,
                vx = 0f,
                vz = 0f,
                mass = mass,
                age = 0f,
                nextStampAge = 0.03f
            };
        }

        private void StepHillSlopeRivulets(float dt)
        {
            if (_arena == null || _arena.turfGrid == null) return;
            InkBlitzArenaSettings cfg = _arena.settings ?? new InkBlitzArenaSettings();
            float halfW = cfg.arenaWidthX * 0.5f - 0.25f;
            float halfL = cfg.arenaLengthZ * 0.5f - 0.25f;
            Vector3 origin = _arena.transform.position;

            for (int i = 0; i < MaxSlopeRivulets; i++)
            {
                if (!_slopeRivulets[i].active) continue;
                ref SlopeRivulet r = ref _slopeRivulets[i];
                r.age += dt;
                if (r.age >= 1.65f || r.mass <= 0.14f)
                {
                    r.active = false;
                    continue;
                }

                Vector3 nrm = ArenaStageBuilder.SampleTerrainNormalLocal(r.localX, r.localZ, cfg);
                float slopeMag = Mathf.Sqrt(nrm.x * nrm.x + nrm.z * nrm.z);
                // Bingham yield stop when reaching flat floor or depleted film thickness
                if (slopeMag < 0.11f || (r.mass - 0.14f) * slopeMag < 0.035f)
                {
                    r.active = false;
                    continue;
                }

                float mob = Mathf.Max(0f, r.mass - 0.14f);
                float targetSpd = Mathf.Min(1.35f, 3.6f * slopeMag * mob);
                r.vx = Mathf.Lerp(r.vx, (nrm.x / slopeMag) * targetSpd, 8f * dt);
                r.vz = Mathf.Lerp(r.vz, (nrm.z / slopeMag) * targetSpd, 8f * dt);

                float stepX = r.vx * dt;
                float stepZ = r.vz * dt;
                float dist = Mathf.Sqrt(stepX * stepX + stepZ * stepZ);
                r.localX = Mathf.Clamp(r.localX + stepX, -halfW, halfW);
                r.localZ = Mathf.Clamp(r.localZ + stepZ, -halfL, halfL);
                r.mass -= dist * 0.26f + 0.04f * dt;

                if (r.age >= r.nextStampAge)
                {
                    r.nextStampAge += 0.045f;
                    float rad = Mathf.Clamp(0.09f + r.mass * 0.18f, 0.09f, 0.22f);
                    _arena.turfGrid.PaintVisualDripOnly(origin + new Vector3(r.localX, 0f, r.localZ), rad, r.teamId);
                }
            }
        }

        private void ClearTransientParticlesAndRivulets()
        {
            for (int i = 0; i < DropletPoolSize; i++)
            {
                _droplets[i].active = false;
                if (_droplets[i].go != null) _droplets[i].go.SetActive(false);
            }
            for (int i = 0; i < MaxSlopeRivulets; i++)
                _slopeRivulets[i].active = false;
        }

        private void DisposeSurfaces(bool restoreOriginalRenderers)
        {
            for (int i = 0; i < _surfaces.Count; i++)
            {
                ObstacleFluidSurface s = _surfaces[i];
                if (restoreOriginalRenderers)
                {
                    if (s.meshFilter != null && s.originalSharedMesh != null)
                        s.meshFilter.sharedMesh = s.originalSharedMesh;
                    if (s.renderer != null && s.originalSharedMaterial != null)
                        s.renderer.sharedMaterial = s.originalSharedMaterial;
                }
                if (s.unwrappedMesh != null) DestroySafe(s.unwrappedMesh);
                if (s.texture != null) DestroySafe(s.texture);
                if (s.runtimeMaterial != null) DestroySafe(s.runtimeMaterial);
            }
            _surfaces.Clear();
            _surfaceByCollider.Clear();
            _materialsBoundToRenderers = false;
        }

        private void OnDestroy()
        {
            DisposeSurfaces(restoreOriginalRenderers: false);
            if (_dropletMatA != null) DestroySafe(_dropletMatA);
            if (_dropletMatB != null) DestroySafe(_dropletMatB);
        }

        private static void ApplyDropletMatColor(Material mat, Color col)
        {
            if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", null);
            if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", null);
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", col);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", col);
            if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", 0.22f);
            if (mat.HasProperty("_Glossiness")) mat.SetFloat("_Glossiness", 0.22f);
        }

        private static int WrapX(int x, int w)
        {
            int m = x % w;
            return (m < 0) ? (m + w) : m;
        }

        private float NextRand01()
        {
            _rngState ^= _rngState << 13;
            _rngState ^= _rngState >> 17;
            _rngState ^= _rngState << 5;
            return (_rngState & 0x00FFFFFFu) / 16777216f;
        }

        private static float Hash01(uint x)
        {
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            x *= 0x846ca68bu;
            x ^= x >> 16;
            return (x & 0x00FFFFFFu) / 16777216f;
        }

        private static Color32 ToColor32(Color c, float mult)
        {
            return new Color32(
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.r * mult * 255f), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.g * mult * 255f), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.b * mult * 255f), 0, 255),
                255
            );
        }

        private static void DestroySafe(UnityEngine.Object obj)
        {
            if (obj == null) return;
            if (Application.isPlaying) Destroy(obj);
            else DestroyImmediate(obj);
        }
    }
}
