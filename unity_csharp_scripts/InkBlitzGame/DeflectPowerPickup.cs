// DeflectPowerPickup.cs — 180-deg symmetric Sky-Bridge 3D Deflect / Counter-Tennis Pickup Orb
// Uses generic `UniversalEntityKind.ResourceOrPickup` (tag = +2.0) with strict 1st-person LOS perception.
using UnityEngine;
using UnityEngine.Rendering;
using UniversalBrain3D;

namespace InkBlitz3D
{
    [RequireComponent(typeof(UniversalSemanticEntityTag))]
    public class DeflectPowerPickup : MonoBehaviour
    {
        [Header("References & Materials")]
        public UniversalSemanticEntityTag entityTag;
        public Transform coreVisual;
        public Transform haloVisual;
        public Transform basePadVisual;
        public Material templateCoreMaterial;
        public Material templateHaloMaterial;

        private InkBlitzArenaSettings _settings;
        private Vector3 _hoverBaseLocalPos;
        private bool _unlockedThisMatch = false;
        private float _respawnTimer = 0f;
        private float _spinPhase = 0f;

        public bool IsAvailable => _unlockedThisMatch && _respawnTimer <= 0f;
        public float CooldownRemaining => Mathf.Max(0f, _respawnTimer);

        public void Initialize(InkBlitzArenaSettings settings)
        {
            _settings = settings ?? new InkBlitzArenaSettings();
            if (entityTag == null)
                entityTag = GetComponent<UniversalSemanticEntityTag>();
            _hoverBaseLocalPos = transform.localPosition;
            EnsureVisualChildren();
            ResetPickup();
        }

        public void EnsureVisualChildren()
        {
            if (basePadVisual == null)
            {
                Transform existingPad = transform.Find("BasePadVisual");
                if (existingPad != null)
                {
                    basePadVisual = existingPad;
                }
                else
                {
                    GameObject padGo = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    padGo.name = "BasePadVisual";
                    padGo.transform.SetParent(transform, false);
                    padGo.transform.localPosition = new Vector3(0f, -0.64f, 0f);
                    padGo.transform.localScale = new Vector3(1.10f, 0.04f, 1.10f);
                    RemoveCollider(padGo);
                    var mr = padGo.GetComponent<MeshRenderer>();
                    if (templateCoreMaterial != null)
                        mr.sharedMaterial = templateCoreMaterial;
                    else if (Application.isPlaying)
                        mr.sharedMaterial = CreateFallbackMat(new Color(0.10f, 0.95f, 0.78f, 1f), transparent: false);
                    basePadVisual = padGo.transform;
                }
            }

            if (coreVisual == null)
            {
                Transform existingCore = transform.Find("CoreDiamondVisual");
                if (existingCore != null)
                {
                    coreVisual = existingCore;
                }
                else
                {
                    GameObject coreGo = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    coreGo.name = "CoreDiamondVisual";
                    coreGo.transform.SetParent(transform, false);
                    coreGo.transform.localPosition = Vector3.zero;
                    coreGo.transform.localRotation = Quaternion.Euler(35.26f, 45f, 45f);
                    coreGo.transform.localScale = new Vector3(0.46f, 0.46f, 0.46f);
                    RemoveCollider(coreGo);
                    var mr = coreGo.GetComponent<MeshRenderer>();
                    if (templateCoreMaterial != null)
                        mr.sharedMaterial = templateCoreMaterial;
                    else if (Application.isPlaying)
                        mr.sharedMaterial = CreateFallbackMat(new Color(0.15f, 1.0f, 0.82f, 1f), transparent: false);
                    coreVisual = coreGo.transform;
                }
            }

            if (haloVisual == null)
            {
                Transform existingHalo = transform.Find("OuterHaloVisual");
                if (existingHalo != null)
                {
                    haloVisual = existingHalo;
                }
                else
                {
                    GameObject haloGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    haloGo.name = "OuterHaloVisual";
                    haloGo.transform.SetParent(transform, false);
                    haloGo.transform.localPosition = Vector3.zero;
                    haloGo.transform.localScale = new Vector3(0.92f, 0.92f, 0.92f);
                    RemoveCollider(haloGo);
                    var mr = haloGo.GetComponent<MeshRenderer>();
                    if (templateHaloMaterial != null)
                        mr.sharedMaterial = templateHaloMaterial;
                    else if (Application.isPlaying)
                        mr.sharedMaterial = CreateFallbackMat(new Color(0.20f, 1.0f, 0.85f, 0.30f), transparent: true);
                    haloVisual = haloGo.transform;
                }
            }
        }

        public void ResetPickup()
        {
            _respawnTimer = 0f;
            _spinPhase = 0f;
            if (_hoverBaseLocalPos.sqrMagnitude > 1e-4f)
                transform.localPosition = _hoverBaseLocalPos;

            bool immediateUnlock = _settings != null && _settings.deflectAppearRemainingSec >= _settings.matchDurationSec;
            _unlockedThisMatch = immediateUnlock;
            SyncAvailabilityState(immediateUnlock);
        }

        public void StepPickup(float dt, InkBlitzArenaInstance arena)
        {
            if (dt <= 0f || arena == null)
                return;

            InkBlitzArenaSettings cfg = _settings ?? arena.settings ?? new InkBlitzArenaSettings();

            if (!_unlockedThisMatch)
            {
                bool lateGame = arena.isOvertime || (arena.matchTimeRemaining <= cfg.deflectAppearRemainingSec);
                if (!lateGame)
                    return;

                _unlockedThisMatch = true;
                _respawnTimer = 0f;
                SyncAvailabilityState(true);
                arena.audioSystem?.EmitSound(transform.position, 0, 0.55f, 0.25f);
            }

            if (_respawnTimer > 0f)
            {
                _respawnTimer -= dt;
                if (_respawnTimer <= 0f)
                {
                    _respawnTimer = 0f;
                    SyncAvailabilityState(true);
                }
                return;
            }

            _spinPhase += dt * 2.6f;
            if (coreVisual != null)
            {
                float bobY = Mathf.Sin(_spinPhase * 1.4f) * 0.08f;
                coreVisual.localPosition = new Vector3(0f, bobY, 0f);
                coreVisual.localRotation = Quaternion.Euler(35.26f, _spinPhase * 68f, 45f);
            }
            if (haloVisual != null)
            {
                float pulse = 0.90f + 0.08f * Mathf.Sin(_spinPhase * 2.2f);
                haloVisual.localScale = new Vector3(pulse, pulse, pulse);
            }

            Vector3 orbPos = transform.position;
            float pickR = cfg.deflectPickupRadiusM;
            float pickRSq = pickR * pickR;

            for (int i = 0; i < arena.agents.Count; i++)
            {
                InkBlitzPlayerAgent ag = arena.agents[i];
                if (ag == null || !ag.gameObject.activeSelf || ag.isRespawning || ag.isSupportTrooperBot || ag.isCarryingObjective || ag.HasDeflectBuff)
                    continue;

                Vector3 agPos = ag.transform.position;
                float dx = agPos.x - orbPos.x;
                float dz = agPos.z - orbPos.z;
                float dy = Mathf.Abs((agPos.y + 0.68f) - orbPos.y);
                if ((dx * dx + dz * dz) <= pickRSq && dy <= 1.25f)
                {
                    ag.ActivateDeflectBuff(cfg.deflectBuffDurationSec);
                    _respawnTimer = Mathf.Max(2.0f, cfg.deflectPickupRespawnSec);
                    SyncAvailabilityState(false);
                    arena.audioSystem?.EmitSound(orbPos, ag.teamId, 0.72f, 0.28f);
                    break;
                }
            }
        }

        private void SyncAvailabilityState(bool available)
        {
            if (entityTag != null)
                entityTag.enabled = available;
            if (coreVisual != null)
                coreVisual.gameObject.SetActive(available);
            if (haloVisual != null)
                haloVisual.gameObject.SetActive(available);
        }

        private static void RemoveCollider(GameObject go)
        {
            Collider col = go.GetComponent<Collider>();
            if (col != null)
            {
                if (Application.isPlaying) Destroy(col);
                else DestroyImmediate(col);
            }
        }

        private static Material CreateFallbackMat(Color col, bool transparent)
        {
            Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var mat = new Material(sh);
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", col);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", col);
            if (transparent)
            {
                if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);
                if (mat.HasProperty("_SrcBlend")) mat.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
                if (mat.HasProperty("_DstBlend")) mat.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
                if (mat.HasProperty("_ZWrite")) mat.SetFloat("_ZWrite", 0f);
                mat.SetOverrideTag("RenderType", "Transparent");
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.renderQueue = (int)RenderQueue.Transparent;
            }
            return mat;
        }
    }
}
