// TurfGridSystem.cs — CPU-authoritative 2D logical ink grid + GPU texture sync (zero ReadPixels)
using UnityEngine;
using UniversalBrain3D;

namespace InkBlitz3D
{
    public struct PaintResult
    {
        public int newlyExploredCells;
        public int overwrittenEnemyCells;
        public int changedCells;
    }

    public class TurfGridSystem : MonoBehaviour, IFloorSurfaceSampler
    {
        [Header("Rendering")]
        public bool enableTextureUpload = true;
        public Material templateFloorMaterial;

        private Vector3 _arenaOrigin;
        private InkBlitzArenaSettings _settings;
        private sbyte[] _ownerGrid;
        private byte[] _exploredBitmask;
        private bool[] _wallBlockedGrid;

        // High-resolution visual splatter layer (decoupled from logical 72x96 CPU grid!)
        private int _texW;
        private int _texH;
        private bool[] _texWallBlocked;
        private sbyte[] _texOwnerPixels;
        private Color32[] _neutralBasePixels;
        private Color32[] _pixels;
        private Texture2D _turfTex;
        private Material _turfMaterial;
        private bool _dirty;
        private bool _visualNeedsFullRebuild;

        public int TeamACells { get; private set; }
        public int TeamBCells { get; private set; }
        public int TotalCells => (_settings != null) ? Mathf.Max(1, _settings.gridWidth * _settings.gridHeight) : 1;
        public float TeamARatio01 => (float)TeamACells / TotalCells;
        public float TeamBRatio01 => (float)TeamBCells / TotalCells;
        public Material TurfFloorMaterial => _turfMaterial;

        public void Initialize(Vector3 arenaOrigin, InkBlitzArenaSettings settings)
        {
            _arenaOrigin = arenaOrigin;
            _settings = settings ?? new InkBlitzArenaSettings();

            int gw = Mathf.Max(8, _settings.gridWidth);
            int gh = Mathf.Max(8, _settings.gridHeight);
            int total = gw * gh;
            int sub = Mathf.Clamp(_settings.visualTexSubdivisions, 1, 10);
            _texW = gw * sub;
            _texH = gh * sub;
            int texTotal = _texW * _texH;

            _ownerGrid = new sbyte[total];
            _exploredBitmask = new byte[total];
            _wallBlockedGrid = new bool[total];
            _texWallBlocked = new bool[texTotal];
            _texOwnerPixels = new sbyte[texTotal];
            _neutralBasePixels = new Color32[texTotal];
            _pixels = new Color32[texTotal];
            BakeNeutralArchitecturalTilePixels();

            if (_turfTex != null)
            {
                if (Application.isPlaying) Destroy(_turfTex);
                else DestroyImmediate(_turfTex);
            }
            // No mipmaps + Point filter: strictly preserves pure flat #7f99dc & #d1bf9b without blurry intermediate color-gradation bands!
            _turfTex = new Texture2D(_texW, _texH, TextureFormat.RGBA32, false)
            {
                name = $"TurfTex_{gameObject.name}_{_texW}x{_texH}",
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Point,
                anisoLevel = 1
            };

            if (_turfMaterial == null)
            {
                if (templateFloorMaterial != null)
                {
                    _turfMaterial = new Material(templateFloorMaterial)
                    {
                        name = $"TurfMat_{gameObject.name}"
                    };
                }
                else
                {
                    Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                    _turfMaterial = new Material(shader)
                    {
                        name = $"TurfMat_{gameObject.name}"
                    };
                }
                // Pure matte zero-gloss surface
                if (_turfMaterial.HasProperty("_Smoothness"))
                    _turfMaterial.SetFloat("_Smoothness", 0.0f);
                if (_turfMaterial.HasProperty("_Glossiness"))
                    _turfMaterial.SetFloat("_Glossiness", 0.0f);
                if (_turfMaterial.HasProperty("_SpecularHighlights"))
                    _turfMaterial.SetFloat("_SpecularHighlights", 0.0f);
                if (_turfMaterial.HasProperty("_EnvironmentReflections"))
                    _turfMaterial.SetFloat("_EnvironmentReflections", 0.0f);
                _turfMaterial.EnableKeyword("_SPECULARHIGHLIGHTS_OFF");
                _turfMaterial.EnableKeyword("_ENVIRONMENTREFLECTIONS_OFF");
                if (_turfMaterial.HasProperty("_BaseColor"))
                    _turfMaterial.SetColor("_BaseColor", Color.white);
                if (_turfMaterial.HasProperty("_Color"))
                    _turfMaterial.SetColor("_Color", Color.white);
            }

            if (_turfMaterial.HasProperty("_BaseMap"))
                _turfMaterial.SetTexture("_BaseMap", _turfTex);
            if (_turfMaterial.HasProperty("_MainTex"))
                _turfMaterial.SetTexture("_MainTex", _turfTex);

            ResetTurf();
        }

        private void BakeNeutralArchitecturalTilePixels()
        {
            if (_settings == null || _neutralBasePixels == null) return;
            Color32 flatNeutral = ToColor32Scaled(_settings.neutralFloorColor, 1.0f);
            Color32 subtleGridLine = ToColor32Scaled(_settings.neutralFloorColor, 0.92f);
            float pxW = _settings.arenaWidthX / Mathf.Max(1, _texW);
            float pxH = _settings.arenaLengthZ / Mathf.Max(1, _texH);
            float halfW = _settings.arenaWidthX * 0.5f;
            float halfL = _settings.arenaLengthZ * 0.5f;

            for (int py = 0; py < _texH; py++)
            {
                float wz = -halfL + (py + 0.5f) * pxH;
                float modZ2 = Mathf.Abs((wz + 32f) % 2.0f);
                bool majorSeamZ = modZ2 < pxH * 0.65f || modZ2 > 2.0f - pxH * 0.65f;
                int row = py * _texW;

                for (int px = 0; px < _texW; px++)
                {
                    float wx = -halfW + (px + 0.5f) * pxW;
                    float modX2 = Mathf.Abs((wx + 32f) % 2.0f);
                    bool majorSeam = majorSeamZ || modX2 < pxW * 0.65f || modX2 > 2.0f - pxW * 0.65f;
                    _neutralBasePixels[row + px] = majorSeam ? subtleGridLine : flatNeutral;
                }
            }
        }

        public void ResetTurf()
        {
            if (_settings == null || _ownerGrid == null)
                return;

            System.Array.Clear(_ownerGrid, 0, _ownerGrid.Length);
            System.Array.Clear(_exploredBitmask, 0, _exploredBitmask.Length);
            if (_texOwnerPixels != null)
                System.Array.Clear(_texOwnerPixels, 0, _texOwnerPixels.Length);
            TeamACells = 0;
            TeamBCells = 0;
            _visualNeedsFullRebuild = false;

            if (_neutralBasePixels != null && _pixels != null && _neutralBasePixels.Length == _pixels.Length)
                System.Array.Copy(_neutralBasePixels, _pixels, _pixels.Length);

            // Pre-paint spawn pads (already marked explored so agents do not farm spawn)
            PaintCircle(_arenaOrigin + _settings.teamASpawnLocal, _settings.spawnPadPaintRadiusM, 1, grantExploreCredit: false);
            PaintCircle(_arenaOrigin + _settings.teamBSpawnLocal, _settings.spawnPadPaintRadiusM, -1, grantExploreCredit: false);

            UploadTextureImmediate();
        }

        public float SampleSurfaceValence(Vector3 worldPoint, int observerTeamId)
        {
            if (_settings == null || _ownerGrid == null || observerTeamId == 0)
                return 0f;

            Vector3 local = worldPoint - _arenaOrigin;
            float u = (local.x + _settings.arenaWidthX * 0.5f) / _settings.arenaWidthX;
            float v = (local.z + _settings.arenaLengthZ * 0.5f) / _settings.arenaLengthZ;
            if (u < 0f || u >= 1f || v < 0f || v >= 1f)
                return 0f;

            int gx = Mathf.Clamp(Mathf.FloorToInt(u * _settings.gridWidth), 0, _settings.gridWidth - 1);
            int gz = Mathf.Clamp(Mathf.FloorToInt(v * _settings.gridHeight), 0, _settings.gridHeight - 1);
            sbyte owner = _ownerGrid[gz * _settings.gridWidth + gx];
            if (owner == 0)
                return 0f;

            sbyte obsSign = (observerTeamId > 0) ? (sbyte)1 : (sbyte)-1;
            return (owner == obsSign) ? 1.0f : -1.0f;
        }

        public void BakeWallObstacleMask(Transform wallsAndCoversRoot)
        {
            if (_settings == null || _wallBlockedGrid == null)
                return;

            System.Array.Clear(_wallBlockedGrid, 0, _wallBlockedGrid.Length);
            if (_texWallBlocked != null)
                System.Array.Clear(_texWallBlocked, 0, _texWallBlocked.Length);
            if (wallsAndCoversRoot == null)
                return;

            BoxCollider[] boxes = wallsAndCoversRoot.GetComponentsInChildren<BoxCollider>(true);
            if (boxes == null || boxes.Length == 0)
                return;

            int gw = _settings.gridWidth;
            int gh = _settings.gridHeight;
            float halfW = _settings.arenaWidthX * 0.5f;
            float halfL = _settings.arenaLengthZ * 0.5f;
            float cellW = _settings.arenaWidthX / gw;
            float cellH = _settings.arenaLengthZ / gh;
            float pxW = _settings.arenaWidthX / Mathf.Max(1, _texW);
            float pxH = _settings.arenaLengthZ / Mathf.Max(1, _texH);

            for (int b = 0; b < boxes.Length; b++)
            {
                BoxCollider bc = boxes[b];
                if (bc == null || !bc.enabled) continue;
                Transform bTr = bc.transform;
                Vector3 bCenter = bc.center;
                Vector3 halfSize = bc.size * 0.5f;

                float minLx = float.MaxValue, maxLx = float.MinValue;
                float minLz = float.MaxValue, maxLz = float.MinValue;
                Vector3 worldCenter = bTr.TransformPoint(bCenter);
                for (int sx = -1; sx <= 1; sx += 2)
                {
                    for (int sy = -1; sy <= 1; sy += 2)
                    {
                        for (int sz = -1; sz <= 1; sz += 2)
                        {
                            Vector3 wp = bTr.TransformPoint(bCenter + Vector3.Scale(halfSize, new Vector3(sx, sy, sz))) - _arenaOrigin;
                            if (wp.x < minLx) minLx = wp.x;
                            if (wp.x > maxLx) maxLx = wp.x;
                            if (wp.z < minLz) minLz = wp.z;
                            if (wp.z > maxLz) maxLz = wp.z;
                        }
                    }
                }

                int minGx = Mathf.Clamp(Mathf.FloorToInt((minLx + halfW) / cellW), 0, gw - 1);
                int maxGx = Mathf.Clamp(Mathf.FloorToInt((maxLx + halfW) / cellW), 0, gw - 1);
                int minGz = Mathf.Clamp(Mathf.FloorToInt((minLz + halfL) / cellH), 0, gh - 1);
                int maxGz = Mathf.Clamp(Mathf.FloorToInt((maxLz + halfL) / cellH), 0, gh - 1);

                for (int gz = minGz; gz <= maxGz; gz++)
                {
                    float wz = _arenaOrigin.z - halfL + (gz + 0.5f) * cellH;
                    int rowOff = gz * gw;
                    for (int gx = minGx; gx <= maxGx; gx++)
                    {
                        float wx = _arenaOrigin.x - halfW + (gx + 0.5f) * cellW;
                        Vector3 localPt = bTr.InverseTransformPoint(new Vector3(wx, worldCenter.y, wz)) - bCenter;
                        if (Mathf.Abs(localPt.x) <= halfSize.x && Mathf.Abs(localPt.z) <= halfSize.z)
                            _wallBlockedGrid[rowOff + gx] = true;
                    }
                }

                if (_texWallBlocked != null && _texW > 0 && _texH > 0)
                {
                    int minPx = Mathf.Clamp(Mathf.FloorToInt((minLx + halfW) / pxW), 0, _texW - 1);
                    int maxPx = Mathf.Clamp(Mathf.FloorToInt((maxLx + halfW) / pxW), 0, _texW - 1);
                    int minPz = Mathf.Clamp(Mathf.FloorToInt((minLz + halfL) / pxH), 0, _texH - 1);
                    int maxPz = Mathf.Clamp(Mathf.FloorToInt((maxLz + halfL) / pxH), 0, _texH - 1);
                    for (int pz = minPz; pz <= maxPz; pz++)
                    {
                        float wz = _arenaOrigin.z - halfL + (pz + 0.5f) * pxH;
                        int pRow = pz * _texW;
                        for (int px = minPx; px <= maxPx; px++)
                        {
                            float wx = _arenaOrigin.x - halfW + (px + 0.5f) * pxW;
                            Vector3 localPt = bTr.InverseTransformPoint(new Vector3(wx, worldCenter.y, wz)) - bCenter;
                            if (Mathf.Abs(localPt.x) <= halfSize.x && Mathf.Abs(localPt.z) <= halfSize.z)
                                _texWallBlocked[pRow + px] = true;
                        }
                    }
                }
            }

            ResetTurf();
        }

        private static bool IsMaskLineBlocked(bool[] mask, int x0, int z0, int x1, int z1, int w, int h)
        {
            if (mask == null) return false;
            int dx = x1 - x0;
            int dz = z1 - z0;
            int steps = Mathf.Max(Mathf.Abs(dx), Mathf.Abs(dz));
            if (steps <= 0) return mask[z1 * w + x1];

            // Sample up to 16 evenly spaced points along ray for sub-cell shadow precision
            int samples = Mathf.Min(steps, 16);
            float inv = 1f / samples;
            for (int s = 1; s <= samples; s++)
            {
                float t = s * inv;
                int cx = Mathf.Clamp(Mathf.RoundToInt(x0 + dx * t), 0, w - 1);
                int cz = Mathf.Clamp(Mathf.RoundToInt(z0 + dz * t), 0, h - 1);
                if (mask[cz * w + cx])
                    return true;
            }
            return false;
        }

        public PaintResult PaintCircle(Vector3 worldPoint, float radiusM, int teamId, bool grantExploreCredit = true)
        {
            PaintResult result = default;
            if (_settings == null || _ownerGrid == null || teamId == 0 || radiusM <= 0f)
                return result;

            sbyte tVal = (teamId > 0) ? (sbyte)1 : (sbyte)-1;
            byte exploreBit = (tVal > 0) ? (byte)1 : (byte)2;
            Color inkColor = (tVal > 0) ? _settings.teamAColor : _settings.teamBColor;

            int gw = _settings.gridWidth;
            int gh = _settings.gridHeight;
            float halfW = _settings.arenaWidthX * 0.5f;
            float halfL = _settings.arenaLengthZ * 0.5f;
            float cellW = _settings.arenaWidthX / gw;
            float cellH = _settings.arenaLengthZ / gh;

            Vector3 local = worldPoint - _arenaOrigin;
            int srcGx = Mathf.Clamp(Mathf.FloorToInt((local.x + halfW) / cellW), 0, gw - 1);
            int srcGz = Mathf.Clamp(Mathf.FloorToInt((local.z + halfL) / cellH), 0, gh - 1);
            if (_wallBlockedGrid != null && _wallBlockedGrid[srcGz * gw + srcGx])
            {
                bool foundOpen = false;
                for (int r = 1; r <= 2 && !foundOpen; r++)
                {
                    for (int oz = -r; oz <= r && !foundOpen; oz++)
                    {
                        for (int ox = -r; ox <= r; ox++)
                        {
                            int nx = srcGx + ox;
                            int nz = srcGz + oz;
                            if (nx >= 0 && nx < gw && nz >= 0 && nz < gh && !_wallBlockedGrid[nz * gw + nx])
                            {
                                srcGx = nx;
                                srcGz = nz;
                                foundOpen = true;
                                break;
                            }
                        }
                    }
                }
                if (!foundOpen)
                    return result;
            }

            int minGx = Mathf.Clamp(Mathf.FloorToInt((local.x - radiusM + halfW) / cellW), 0, gw - 1);
            int maxGx = Mathf.Clamp(Mathf.FloorToInt((local.x + radiusM + halfW) / cellW), 0, gw - 1);
            int minGz = Mathf.Clamp(Mathf.FloorToInt((local.z - radiusM + halfL) / cellH), 0, gh - 1);
            int maxGz = Mathf.Clamp(Mathf.FloorToInt((local.z + radiusM + halfL) / cellH), 0, gh - 1);
            float rSq = radiusM * radiusM;

            for (int gz = minGz; gz <= maxGz; gz++)
            {
                float cz = -halfL + (gz + 0.5f) * cellH;
                float dz = cz - local.z;
                float dzSq = dz * dz;
                int rowOffset = gz * gw;

                for (int gx = minGx; gx <= maxGx; gx++)
                {
                    float cx = -halfW + (gx + 0.5f) * cellW;
                    float dx = cx - local.x;
                    if (dx * dx + dzSq > rSq)
                        continue;

                    int idx = rowOffset + gx;
                    if (_wallBlockedGrid != null && (_wallBlockedGrid[idx] || IsMaskLineBlocked(_wallBlockedGrid, srcGx, srcGz, gx, gz, gw, gh)))
                        continue;
                    sbyte prev = _ownerGrid[idx];
                    if (prev != tVal)
                    {
                        if (prev == 1) TeamACells--;
                        else if (prev == -1) TeamBCells--;

                        if (tVal == 1) TeamACells++;
                        else TeamBCells++;

                        if (prev == -tVal)
                            result.overwrittenEnemyCells++;

                        _ownerGrid[idx] = tVal;
                        result.changedCells++;
                    }

                    if ((_exploredBitmask[idx] & exploreBit) == 0)
                    {
                        _exploredBitmask[idx] |= exploreBit;
                        if (grantExploreCredit)
                            result.newlyExploredCells++;
                    }
                }
            }

            if (enableTextureUpload)
            {
                StampOrganicHighResSplat(local.x, local.z, radiusM, inkColor, tVal);
            }
            else if (result.changedCells > 0)
            {
                _visualNeedsFullRebuild = true;
            }

            return result;
        }

        private void StampOrganicHighResSplat(float localX, float localZ, float radiusM, Color inkColor, sbyte tVal)
        {
            if (_pixels == null || _texW <= 0 || _texH <= 0) return;

            float halfW = _settings.arenaWidthX * 0.5f;
            float halfL = _settings.arenaLengthZ * 0.5f;
            float pxW = _settings.arenaWidthX / _texW;
            float pxH = _settings.arenaLengthZ / _texH;

            int srcPx = Mathf.Clamp(Mathf.FloorToInt((localX + halfW) / pxW), 0, _texW - 1);
            int srcPz = Mathf.Clamp(Mathf.FloorToInt((localZ + halfL) / pxH), 0, _texH - 1);
            if (_texWallBlocked != null && _texWallBlocked[srcPz * _texW + srcPx])
            {
                bool found = false;
                for (int r = 1; r <= 6 && !found; r++)
                {
                    for (int oz = -r; oz <= r && !found; oz++)
                    {
                        for (int ox = -r; ox <= r; ox++)
                        {
                            int nx = srcPx + ox;
                            int nz = srcPz + oz;
                            if (nx >= 0 && nx < _texW && nz >= 0 && nz < _texH && !_texWallBlocked[nz * _texW + nx])
                            {
                                srcPx = nx;
                                srcPz = nz;
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if (!found) return;
            }

            // Pure uniform flat matte disc stamp: exact #HEX with ZERO mottling, ZERO sheen spots, and ZERO dark rim bands!
            int minPx = Mathf.Clamp(Mathf.FloorToInt((localX - radiusM + halfW) / pxW), 0, _texW - 1);
            int maxPx = Mathf.Clamp(Mathf.FloorToInt((localX + radiusM + halfW) / pxW), 0, _texW - 1);
            int minPz = Mathf.Clamp(Mathf.FloorToInt((localZ - radiusM + halfL) / pxH), 0, _texH - 1);
            int maxPz = Mathf.Clamp(Mathf.FloorToInt((localZ + radiusM + halfL) / pxH), 0, _texH - 1);

            Color32 flatInk = ToColor32Scaled(inkColor, 1.0f);
            float rSq = radiusM * radiusM;

            for (int pz = minPz; pz <= maxPz; pz++)
            {
                float dz = (-halfL + (pz + 0.5f) * pxH) - localZ;
                float dzSq = dz * dz;
                int pRow = pz * _texW;
                for (int px = minPx; px <= maxPx; px++)
                {
                    float dx = (-halfW + (px + 0.5f) * pxW) - localX;
                    if (dx * dx + dzSq > rSq)
                        continue;

                    int pIdx = pRow + px;
                    if (_texWallBlocked != null && (_texWallBlocked[pIdx] || IsMaskLineBlocked(_texWallBlocked, srcPx, srcPz, px, pz, _texW, _texH)))
                        continue;

                    _pixels[pIdx] = flatInk;
                    if (_texOwnerPixels != null)
                        _texOwnerPixels[pIdx] = tVal;
                    _dirty = true;
                }
            }
        }

        private void RebuildVisualFromLogicalGrid()
        {
            if (_pixels == null || _neutralBasePixels == null || _ownerGrid == null) return;
            System.Array.Copy(_neutralBasePixels, _pixels, _pixels.Length);
            if (_texOwnerPixels != null)
                System.Array.Clear(_texOwnerPixels, 0, _texOwnerPixels.Length);
            int gw = _settings.gridWidth;
            int gh = _settings.gridHeight;
            int sub = Mathf.Max(1, _texW / Mathf.Max(1, gw));
            Color32 colA = ToColor32Scaled(_settings.teamAColor, 1.0f);
            Color32 colB = ToColor32Scaled(_settings.teamBColor, 1.0f);

            for (int gz = 0; gz < gh; gz++)
            {
                for (int gx = 0; gx < gw; gx++)
                {
                    sbyte own = _ownerGrid[gz * gw + gx];
                    if (own == 0) continue;
                    Color32 c = (own > 0) ? colA : colB;
                    sbyte tv = (own > 0) ? (sbyte)1 : (sbyte)-1;
                    for (int sz = 0; sz < sub; sz++)
                    {
                        int pz = gz * sub + sz;
                        int pRow = pz * _texW;
                        for (int sx = 0; sx < sub; sx++)
                        {
                            int px = gx * sub + sx;
                            int pIdx = pRow + px;
                            if (_texWallBlocked == null || !_texWallBlocked[pIdx])
                            {
                                _pixels[pIdx] = c;
                                if (_texOwnerPixels != null)
                                    _texOwnerPixels[pIdx] = tv;
                            }
                        }
                    }
                }
            }
            _visualNeedsFullRebuild = false;
            _dirty = true;
        }

        public void RefreshPaletteColors()
        {
            if (_settings == null || _pixels == null) return;
            if (_visualNeedsFullRebuild || _texOwnerPixels == null || _texOwnerPixels.Length != _pixels.Length)
            {
                RebuildVisualFromLogicalGrid();
            }
            else
            {
                Color32 colA = ToColor32Scaled(_settings.teamAColor, 1.0f);
                Color32 colB = ToColor32Scaled(_settings.teamBColor, 1.0f);
                for (int i = 0; i < _pixels.Length; i++)
                {
                    sbyte own = _texOwnerPixels[i];
                    if (own > 0) _pixels[i] = colA;
                    else if (own < 0) _pixels[i] = colB;
                }
                _dirty = true;
            }

            if (enableTextureUpload)
                UploadTextureImmediate();
        }

        public void SetRenderingEnabled(bool enabled)
        {
            bool wasEnabled = enableTextureUpload;
            enableTextureUpload = enabled;
            if (enabled && _turfTex != null && _pixels != null)
            {
                if (_visualNeedsFullRebuild)
                    RebuildVisualFromLogicalGrid();
                if (!wasEnabled || _dirty)
                {
                    _dirty = true;
                    UploadTextureImmediate();
                }
            }
        }

        private void LateUpdate()
        {
            if (_dirty && enableTextureUpload && _turfTex != null)
                UploadTextureImmediate();
        }

        private void UploadTextureImmediate()
        {
            if (!enableTextureUpload || _turfTex == null || _pixels == null)
                return;
            _turfTex.SetPixels32(_pixels);
            _turfTex.Apply(false);
            _dirty = false;
        }

        private static Color32 ToColor32Scaled(Color c, float scale)
        {
            return new Color32(
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.r * scale * 255f), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.g * scale * 255f), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(c.b * scale * 255f), 0, 255),
                255
            );
        }

        public static void ApplyArenaPlanarUVs(MeshFilter mf, Vector3 arenaOrigin, float widthX, float lengthZ)
        {
            if (mf == null || mf.sharedMesh == null)
                return;
            if (mf.sharedMesh.name != null && mf.sharedMesh.name.EndsWith("_ArenaPlanarUV"))
                return;

            Mesh mesh = Instantiate(mf.sharedMesh);
            mesh.name = $"{mf.sharedMesh.name}_ArenaPlanarUV";
            Vector3[] verts = mesh.vertices;
            Vector2[] uvs = new Vector2[verts.Length];
            Transform tr = mf.transform;
            float invW = 1f / Mathf.Max(0.01f, widthX);
            float invL = 1f / Mathf.Max(0.01f, lengthZ);

            for (int i = 0; i < verts.Length; i++)
            {
                Vector3 loc = tr.TransformPoint(verts[i]) - arenaOrigin;
                uvs[i] = new Vector2((loc.x + widthX * 0.5f) * invW, (loc.z + lengthZ * 0.5f) * invL);
            }

            mesh.uv = uvs;
            mf.sharedMesh = mesh;
        }

        private void OnDestroy()
        {
            if (_turfTex != null)
            {
                if (Application.isPlaying) Destroy(_turfTex);
                else DestroyImmediate(_turfTex);
            }
            if (_turfMaterial != null)
            {
                if (Application.isPlaying) Destroy(_turfMaterial);
                else DestroyImmediate(_turfMaterial);
            }
        }
    }
}
