// native_accel/fast_brain3d_kernels.cpp
// Zero-PyTorch C++17 /O2 Native Spatial, A*, and 3D Foveal Raycast Kernels for Universal Brain 3D v2.3
// Preserves 100% mathematical parity with pure-Python reference implementations while accelerating
// 60Hz POMDP arena raycasting, 48x48 4-layer field map integration, and 8-connected tactical A*.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#define UB3D_API extern "C" __declspec(dllexport)
#else
#define UB3D_API extern "C"
#endif

namespace {

struct HeapNode {
    float f;
    float g;
    int flat;
};

inline bool heap_less(const HeapNode& a, const HeapNode& b) {
    if (a.f != b.f) return a.f < b.f;
    if (a.g != b.g) return a.g < b.g;
    return a.flat < b.flat;
}

inline void heap_push(HeapNode* heap, int& sz, float f, float g, int flat) {
    HeapNode node = {f, g, flat};
    int i = sz++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!heap_less(node, heap[p])) break;
        heap[i] = heap[p];
        i = p;
    }
    heap[i] = node;
}

inline HeapNode heap_pop(HeapNode* heap, int& sz) {
    HeapNode top = heap[0];
    HeapNode last = heap[--sz];
    int i = 0;
    while (true) {
        int l = (i << 1) + 1;
        if (l >= sz) break;
        int r = l + 1;
        int c = (r < sz && heap_less(heap[r], heap[l])) ? r : l;
        if (!heap_less(heap[c], last)) break;
        heap[i] = heap[c];
        i = c;
    }
    heap[i] = last;
    return top;
}

inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace

UB3D_API int ub3d_astar_tactical_step(
    const uint8_t* occ_state,
    const float* elev_map,
    const float* surface_map,
    const float* scar_map,
    int n,
    int sz, int sx,
    int gz, int gx,
    int tz, int tx,
    float cell_w, float cell_d,
    float wmin_x, float wmin_z,
    float start_x, float start_z,
    float direct_x, float direct_z, float direct_d, float max_diag,
    float* out_dir_xz,
    float* out_norm_route,
    float* out_wp_xz,
    int* out_has_wp
) {
    const int nn = n * n;
    if (nn <= 0 || nn > 4096) return 0;

    float dist_l[4096];
    int parent_l[4096];
    HeapNode pq[16384];
    int pq_sz = 0;

    for (int i = 0; i < nn; ++i) {
        dist_l[i] = 1e9f;
        parent_l[i] = -1;
    }

    const int s_flat = sz * n + sx;
    const int g_flat = gz * n + gx;
    dist_l[s_flat] = 0.0f;

    int ad0 = std::abs(sz - gz);
    int bd0 = std::abs(sx - gx);
    int mn0 = ad0 < bd0 ? ad0 : bd0;
    float h0 = 0.749f * (float(std::abs(ad0 - bd0)) + 1.414f * float(mn0));
    heap_push(pq, pq_sz, h0, 0.0f, s_flat);

    static const int dz_tab[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static const int dx_tab[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    static const float step_tab[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.414f, 1.414f, 1.414f, 1.414f};

    float lof_ux = 0.0f, lof_uz = 0.0f;
    bool lof_active = false;
    if (tz >= 0) {
        float dz_ts = float(sz - tz);
        float dx_ts = float(sx - tx);
        float d_ts = std::hypot(dx_ts, dz_ts);
        if (d_ts >= 1.5f && d_ts <= 22.0f) {
            lof_ux = dx_ts / d_ts;
            lof_uz = dz_ts / d_ts;
            lof_active = true;
        }
    }

    bool reached = false;
    while (pq_sz > 0) {
        HeapNode cur = heap_pop(pq, pq_sz);
        if (cur.g > dist_l[cur.flat] + 1e-5f) continue;
        if (cur.flat == g_flat) {
            reached = true;
            break;
        }
        int cz = cur.flat / n;
        int cx = cur.flat - cz * n;
        float elev_c = elev_map[cur.flat];

        for (int k = 0; k < 8; ++k) {
            int nz = cz + dz_tab[k];
            int nx_i = cx + dx_tab[k];
            if (nz < 0 || nz >= n || nx_i < 0 || nx_i >= n) continue;
            int n_flat = nz * n + nx_i;
            uint8_t occ = occ_state[n_flat];
            if (occ == 2 && n_flat != g_flat) continue; // WALL == 2
            float dh = elev_map[n_flat] - elev_c;
            if (dh > 0.62f && n_flat != g_flat) continue;

            float base_step = step_tab[k];
            float surf = surface_map[n_flat];
            float s_pos = surf > 0.0f ? surf : 0.0f;
            float s_neg = surf < 0.0f ? -surf : 0.0f;
            float dh_pos = dh > 0.0f ? dh : 0.0f;
            float step_cost = base_step * (1.0f - 0.25f * s_pos + 1.15f * s_neg + 0.65f * dh_pos + 1.35f * scar_map[n_flat]);
            if (occ == 0) step_cost += 0.22f * base_step; // UNKNOWN == 0

            if (tz >= 0) {
                float rdz = float(nz - tz);
                float rdx = float(nx_i - tx);
                float d2_thr = rdz * rdz + rdx * rdx;
                if (d2_thr < 49.0f) {
                    step_cost += 2.6f / (1.0f + 0.28f * d2_thr);
                }
                if (lof_active && d2_thr > 2.0f && d2_thr < 196.0f) {
                    float proj_lof = rdx * lof_ux + rdz * lof_uz;
                    if (proj_lof > 0.5f) {
                        float perp2 = d2_thr - proj_lof * proj_lof;
                        if (perp2 < 0.0f) perp2 = 0.0f;
                        if (perp2 < 6.25f) {
                            step_cost += 0.85f * base_step / (1.0f + 0.45f * perp2);
                        }
                    }
                }
            }

            float nc = cur.g + step_cost;
            if (nc + 1e-5f < dist_l[n_flat] && pq_sz < 16380) {
                dist_l[n_flat] = nc;
                parent_l[n_flat] = cur.flat;
                int ad = std::abs(nz - gz);
                int bd = std::abs(nx_i - gx);
                int mn = ad < bd ? ad : bd;
                float h_val = 0.749f * (float(std::abs(ad - bd)) + 1.414f * float(mn));
                heap_push(pq, pq_sz, nc + h_val, nc, n_flat);
            }
        }
    }

    if (!reached) {
        float inv = 1.0f / std::max(1e-4f, direct_d);
        out_dir_xz[0] = direct_x * inv;
        out_dir_xz[1] = direct_z * inv;
        *out_norm_route = clamp_f(direct_d / max_diag, 0.0f, 1.0f);
        *out_has_wp = 0;
        return 1;
    }

    int path_flats[256];
    int plen = 0;
    int cur_f = g_flat;
    const int max_plen = std::min(255, n * 2);
    while (cur_f != s_flat && plen < max_plen) {
        path_flats[plen++] = cur_f;
        int pf = parent_l[cur_f];
        if (pf < 0) break;
        cur_f = pf;
    }
    // Reverse index for look_idx = min(plen - 1, 2) from start
    int rev_idx = plen - 1 - std::min(plen - 1, 2);
    int wp_flat = path_flats[rev_idx];
    int wp_iz = wp_flat / n;
    int wp_ix = wp_flat - wp_iz * n;
    float wp_x = wmin_x + (float(wp_ix) + 0.5f) * cell_w;
    float wp_z = wmin_z + (float(wp_iz) + 0.5f) * cell_d;
    out_wp_xz[0] = wp_x;
    out_wp_xz[1] = wp_z;
    *out_has_wp = 1;

    float sx_v = wp_x - start_x;
    float sz_v = wp_z - start_z;
    float sd = std::hypot(sx_v, sz_v);
    if (sd < 1e-4f) {
        sx_v = direct_x;
        sz_v = direct_z;
        sd = direct_d;
    }
    float inv_sd = 1.0f / std::max(1e-4f, sd);
    out_dir_xz[0] = sx_v * inv_sd;
    out_dir_xz[1] = sz_v * inv_sd;
    float route_len_m = float(plen) * 0.5f * (cell_w + cell_d);
    *out_norm_route = clamp_f(route_len_m / max_diag, 0.0f, 1.0f);
    return 1;
}

UB3D_API void ub3d_render_fovea_16x16(
    float me_x, float me_y, float me_z,
    float yaw, float pitch,
    float tan_half_fov, float max_r,
    float wmin_x, float wmin_y, float wmin_z,
    float span_x, float span_z,
    const float* surface_grid, int grid_n, float team_sign,
    const float* vis_ents, int n_ents,
    float* out_fovea_16x16x4
) {
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float fx = sy * cp, fy = sp, fz = cy * cp;
    const float rx = cy, ry = 0.0f, rz = -sy;
    // up = cross(fwd, right)
    const float ux = fy * rz - fz * ry;
    const float uy = fz * rx - fx * rz;
    const float uz = fx * ry - fy * rx;

    const int H = 16, W = 16;
    const int ent_cap = n_ents < 12 ? n_ents : 12;

    for (int iy = 0; iy < H; ++iy) {
        float v = tan_half_fov * (1.0f - 2.0f * float(iy) / float(H - 1));
        for (int ix = 0; ix < W; ++ix) {
            float u = tan_half_fov * (-1.0f + 2.0f * float(ix) / float(W - 1));
            float dx = fx + u * rx + v * ux;
            float dy = fy + u * ry + v * uy;
            float dz = fz + u * rz + v * uz;
            float dn = std::sqrt(dx * dx + dy * dy + dz * dz);
            float inv_dn = 1.0f / std::max(1e-6f, dn);
            dx *= inv_dn; dy *= inv_dn; dz *= inv_dn;

            float best_t = max_r;
            float nrm_x = 0.0f;
            float nrm_y = 0.0f;
            float sal = 0.10f;

            if (dy < -1e-4f) {
                float t_fl = (wmin_y - me_y) / dy;
                if (t_fl > 0.05f && t_fl < best_t) {
                    best_t = t_fl;
                    nrm_y = 1.0f;
                    float hx = me_x + best_t * dx;
                    float hz = me_z + best_t * dz;
                    int c_ix = clamp_int(int(std::floor((hx - wmin_x) / span_x * float(grid_n))), 0, grid_n - 1);
                    int c_iz = clamp_int(int(std::floor((hz - wmin_z) / span_z * float(grid_n))), 0, grid_n - 1);
                    float sv = surface_grid[c_iz * grid_n + c_ix] * team_sign;
                    sal = (sv > 0.25f) ? 0.78f : ((sv < -0.25f) ? 0.25f : 0.50f);
                }
            }

            for (int e = 0; e < ent_cap; ++e) {
                const float* ent = vis_ents + e * 10;
                float oc_x = ent[1] - me_x;
                float oc_y = ent[2] - me_y;
                float oc_z = ent[3] - me_z;
                float rad = ent[7];
                float bp = dx * oc_x + dy * oc_y + dz * oc_z;
                float oc2 = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z;
                float disc = bp * bp - (oc2 - rad * rad);
                if (disc > 0.0f) {
                    float ts = bp - std::sqrt(disc);
                    if (ts > 0.05f && ts < best_t) {
                        best_t = ts;
                        float inv_r = 1.0f / std::max(1e-3f, rad);
                        nrm_x = (me_x + best_t * dx - ent[1]) * inv_r;
                        nrm_y = (me_y + best_t * dy - ent[2]) * inv_r;
                        sal = (ent[8] < 0.0f) ? 0.95f : 0.60f;
                    }
                }
            }

            float* dst = out_fovea_16x16x4 + (iy * W + ix) * 4;
            dst[0] = 1.0f - clamp_f(best_t / max_r, 0.0f, 1.0f);
            dst[1] = nrm_x;
            dst[2] = nrm_y;
            dst[3] = sal;
        }
    }
}

UB3D_API void ub3d_directional_clearance_and_surface(
    const uint8_t* occ_state,
    const float* surface_map,
    int n, int cz, int cx,
    float* out_clr4,
    float* out_surf4
) {
    static const int dz_4[4] = {1, 0, -1, 0};
    static const int dx_4[4] = {0, 1, 0, -1};
    const int near_steps = 6;
    const int corridor_steps = 11;

    for (int di = 0; di < 4; ++di) {
        int dz = dz_4[di], dx = dx_4[di];
        int pz = -dx, px = dz;
        int free_steps = 0;
        float near_sum = 0.0f; int near_cnt = 0;
        float far_sum = 0.0f;  int far_cnt = 0;
        bool blocked_center = false;

        for (int k = 1; k <= corridor_steps; ++k) {
            int nz = cz + dz * k;
            int nx_i = cx + dx * k;
            if (nz < 0 || nz >= n || nx_i < 0 || nx_i >= n) break;
            if (occ_state[nz * n + nx_i] == 2) {
                blocked_center = true;
                break;
            }
            if (k <= near_steps) {
                ++free_steps;
                near_sum += 0.5f * (surface_map[nz * n + nx_i] + 1.0f);
                ++near_cnt;
            } else {
                float s_sum = 0.5f * (surface_map[nz * n + nx_i] + 1.0f);
                int s_cnt = 1;
                for (int lat = -1; lat <= 1; lat += 2) {
                    int lz_i = nz + pz * lat;
                    int lx_i = nx_i + px * lat;
                    if (lz_i >= 0 && lz_i < n && lx_i >= 0 && lx_i < n && occ_state[lz_i * n + lx_i] != 2) {
                        s_sum += 0.5f * (surface_map[lz_i * n + lx_i] + 1.0f);
                        ++s_cnt;
                    }
                }
                far_sum += s_sum / float(s_cnt);
                ++far_cnt;
            }
        }
        out_clr4[di] = float(free_steps) / float(near_steps);
        if (near_cnt > 0 && far_cnt > 0) {
            out_surf4[di] = clamp_f(0.56f * (near_sum / float(near_cnt)) + 0.44f * (far_sum / float(far_cnt)), 0.0f, 1.0f);
        } else if (near_cnt > 0) {
            out_surf4[di] = clamp_f((near_sum / float(near_cnt)) * (blocked_center ? 0.92f : 1.0f), 0.0f, 1.0f);
        } else {
            out_surf4[di] = 0.5f;
        }
    }
}

UB3D_API void ub3d_smooth_local_step(
    const uint8_t* occ_state,
    int n,
    float cell_w, float cell_d,
    float wmin_x, float wmin_z,
    float wmax_x, float wmax_z,
    float ego_x, float ego_z,
    float des_x, float des_z,
    float* out_xz
) {
    float ox = clamp_f(des_x, wmin_x + 0.65f, wmax_x - 0.65f);
    float oz = clamp_f(des_z, wmin_z + 0.65f, wmax_z - 0.65f);
    int ix = clamp_int(int(std::floor((ox - wmin_x) / cell_w)), 0, n - 1);
    int iz = clamp_int(int(std::floor((oz - wmin_z) / cell_d)), 0, n - 1);
    float rep_x = 0.0f, rep_z = 0.0f;
    const float min_clear = 0.75f * std::max(cell_w, cell_d) + 0.40f;

    for (int dz = -1; dz <= 1; ++dz) {
        int jz = iz + dz;
        if (jz < 0 || jz >= n) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int jx = ix + dx;
            if (jx < 0 || jx >= n) continue;
            if (occ_state[jz * n + jx] == 2) {
                float wcx = wmin_x + (float(jx) + 0.5f) * cell_w;
                float wcz = wmin_z + (float(jz) + 0.5f) * cell_d;
                float dfx = ox - wcx;
                float dfz = oz - wcz;
                float d = std::hypot(dfx, dfz);
                if (d < min_clear) {
                    float px, pz;
                    if (d > 1e-4f) {
                        px = dfx / d;
                        pz = dfz / d;
                    } else {
                        float ex = ego_x - wcx;
                        float ez = ego_z - wcz;
                        float ed = std::max(1e-4f, std::hypot(ex, ez));
                        px = ex / ed;
                        pz = ez / ed;
                    }
                    float scale = min_clear - d;
                    rep_x += px * scale;
                    rep_z += pz * scale;
                }
            }
        }
    }
    out_xz[0] = clamp_f(ox + rep_x, wmin_x + 0.65f, wmax_x - 0.65f);
    out_xz[1] = clamp_f(oz + rep_z, wmin_z + 0.65f, wmax_z - 0.65f);
}

UB3D_API void ub3d_integrate_first_person_frame(
    uint8_t* occ_state,
    float* wall_score,
    float* surface_map,
    float* surface_obs_age,
    float* elev_map,
    float* visit_freshness,
    float* dwell_saturation,
    float* strategic_hazard_scar,
    int n,
    float wmin_x, float wmin_y, float wmin_z,
    float cell_w, float cell_d,
    float surface_decay_per_sec,
    float bel_x, float bel_y, float bel_z,
    float yaw, float pitch,
    float surf_posture,
    float tan_half_fov, float max_range_m, float dt,
    const float* fovea_16x16x4,
    const float* ventral_16x16x4,
    const float* uncert_16x16,
    float* out_ratios_2
) {
    const int nn = n * n;
    const float decay_surf = (surface_decay_per_sec > 0.0f) ? std::exp(-surface_decay_per_sec * dt) : 1.0f;
    const float decay_vis = std::exp(-0.025f * dt);
    const float decay_dwell = std::exp(-0.085f * dt);
    const float decay_scar = std::exp(-0.045f * dt);

    for (int i = 0; i < nn; ++i) {
        float age = std::min(99.0f, surface_obs_age[i] + dt);
        surface_obs_age[i] = age;
        if (surface_decay_per_sec > 0.0f && age > 1.0f) {
            surface_map[i] *= decay_surf;
        }
        visit_freshness[i] *= decay_vis;
        dwell_saturation[i] *= decay_dwell;
        strategic_hazard_scar[i] *= decay_scar;
    }

    int ego_ix = clamp_int(int(std::floor((bel_x - wmin_x) / cell_w)), 0, n - 1);
    int ego_iz = clamp_int(int(std::floor((bel_z - wmin_z) / cell_d)), 0, n - 1);
    int ego_f = ego_iz * n + ego_ix;
    occ_state[ego_f] = 1;
    wall_score[ego_f] = std::max(0.0f, wall_score[ego_f] - 0.5f);
    visit_freshness[ego_f] = 1.0f;
    dwell_saturation[ego_f] = std::min(2.5f, dwell_saturation[ego_f] + 0.42f * dt);

    static const int ndz4[4] = {-1, 1, 0, 0};
    static const int ndx4[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
        int jz = ego_iz + ndz4[k];
        int jx = ego_ix + ndx4[k];
        if (jz >= 0 && jz < n && jx >= 0 && jx < n) {
            int jf = jz * n + jx;
            if (occ_state[jf] == 0) occ_state[jf] = 1;
            if (visit_freshness[jf] < 0.88f) visit_freshness[jf] = 0.88f;
            dwell_saturation[jf] = std::min(2.5f, dwell_saturation[jf] + 0.24f * dt);
        }
    }

    if (surf_posture >= 0.62f) {
        surface_map[ego_f] = clamp_f(0.35f * surface_map[ego_f] + 0.65f * 0.90f, -1.0f, 1.0f);
        surface_obs_age[ego_f] = 0.0f;
    } else if (surf_posture <= 0.20f) {
        surface_map[ego_f] = clamp_f(0.35f * surface_map[ego_f] - 0.65f * 0.90f, -1.0f, 1.0f);
        surface_obs_age[ego_f] = 0.0f;
    }

    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float fx = sy * cp, fy = sp, fz = cy * cp;
    const float rx = cy, ry = 0.0f, rz = -sy;
    const float ux = fy * rz - fz * ry;
    const float uy = fz * rx - fx * rz;
    const float uz = fx * ry - fy * rx;

    const int H = 16, W = 16;
    float dirs[256 * 3];
    float dist_m[256];
    float conf_w[256];

    for (int iy = 0; iy < H; ++iy) {
        float v = tan_half_fov * (1.0f - 2.0f * float(iy) / float(H - 1));
        for (int ix = 0; ix < W; ++ix) {
            int idx = iy * W + ix;
            float u = tan_half_fov * (-1.0f + 2.0f * float(ix) / float(W - 1));
            float dx = fx + u * rx + v * ux;
            float dy = fy + u * ry + v * uy;
            float dz = fz + u * rz + v * uz;
            float inv_d = 1.0f / std::max(1e-6f, std::sqrt(dx * dx + dy * dy + dz * dz));
            dirs[idx * 3 + 0] = dx * inv_d;
            dirs[idx * 3 + 1] = dy * inv_d;
            dirs[idx * 3 + 2] = dz * inv_d;

            float nd = clamp_f(fovea_16x16x4[idx * 4 + 0], 0.0f, 1.0f);
            dist_m[idx] = (1.0f - nd) * max_range_m;
            conf_w[idx] = (uncert_16x16 != nullptr) ? clamp_f(1.0f - uncert_16x16[idx], 0.15f, 1.0f) : 1.0f;
        }
    }

    // Pass 1..3 with exact NumPy slice-assignment RHS snapshot semantics (`arr[idx] = f(old_arr[idx])`)
    uint8_t old_occ[4096];
    float old_wsc[4096];
    float old_elv[4096];
    float old_vis[4096];
    float old_srf[4096];

    std::memcpy(old_occ, occ_state, nn * sizeof(uint8_t));
    std::memcpy(old_wsc, wall_score, nn * sizeof(float));
    std::memcpy(old_vis, visit_freshness, nn * sizeof(float));
    for (int idx = 0; idx < 256; ++idx) {
        float dm = dist_m[idx];
        float dy = dirs[idx * 3 + 1];
        float cw = conf_w[idx];
        if (dm > 1.6f && std::abs(dy) < 0.45f && cw > 0.35f) {
            float mx = bel_x + 0.55f * dm * dirs[idx * 3 + 0];
            float mz = bel_z + 0.55f * dm * dirs[idx * 3 + 2];
            int m_ix = clamp_int(int(std::floor((mx - wmin_x) / cell_w)), 0, n - 1);
            int m_iz = clamp_int(int(std::floor((mz - wmin_z) / cell_d)), 0, n - 1);
            int mf = m_iz * n + m_ix;
            wall_score[mf] = std::max(0.0f, old_wsc[mf] - 0.18f);
        }
    }
    for (int idx = 0; idx < 256; ++idx) {
        float dm = dist_m[idx];
        float dy = dirs[idx * 3 + 1];
        float cw = conf_w[idx];
        if (dm > 1.6f && std::abs(dy) < 0.45f && cw > 0.35f) {
            float mx = bel_x + 0.55f * dm * dirs[idx * 3 + 0];
            float mz = bel_z + 0.55f * dm * dirs[idx * 3 + 2];
            int m_ix = clamp_int(int(std::floor((mx - wmin_x) / cell_w)), 0, n - 1);
            int m_iz = clamp_int(int(std::floor((mz - wmin_z) / cell_d)), 0, n - 1);
            int mf = m_iz * n + m_ix;
            occ_state[mf] = (wall_score[mf] < 0.45f) ? uint8_t(1) : old_occ[mf];
            visit_freshness[mf] = std::max(old_vis[mf], 0.55f);
        }
    }

    std::memcpy(old_occ, occ_state, nn * sizeof(uint8_t));
    std::memcpy(old_wsc, wall_score, nn * sizeof(float));
    std::memcpy(old_elv, elev_map, nn * sizeof(float));
    std::memcpy(old_vis, visit_freshness, nn * sizeof(float));
    std::memcpy(old_srf, surface_map, nn * sizeof(float));
    for (int idx = 0; idx < 256; ++idx) {
        float nd = clamp_f(fovea_16x16x4[idx * 4 + 0], 0.0f, 1.0f);
        float ny = fovea_16x16x4[idx * 4 + 2];
        float dm = dist_m[idx];
        float cw = conf_w[idx];
        if (nd > 0.03f && dm > 0.25f && cw > 0.30f && ny > 0.58f) {
            float hx = bel_x + dm * dirs[idx * 3 + 0];
            float hz = bel_z + dm * dirs[idx * 3 + 2];
            int f_ix = clamp_int(int(std::floor((hx - wmin_x) / cell_w)), 0, n - 1);
            int f_iz = clamp_int(int(std::floor((hz - wmin_z) / cell_d)), 0, n - 1);
            int ff = f_iz * n + f_ix;
            wall_score[ff] = std::max(0.0f, old_wsc[ff] - 0.25f * cw);
        }
    }
    for (int idx = 0; idx < 256; ++idx) {
        float nd = clamp_f(fovea_16x16x4[idx * 4 + 0], 0.0f, 1.0f);
        float ny = fovea_16x16x4[idx * 4 + 2];
        float sal = fovea_16x16x4[idx * 4 + 3];
        float dm = dist_m[idx];
        float cw = conf_w[idx];
        if (nd > 0.03f && dm > 0.25f && cw > 0.30f && ny > 0.58f) {
            float hx = bel_x + dm * dirs[idx * 3 + 0];
            float hy = bel_y + dm * dirs[idx * 3 + 1];
            float hz = bel_z + dm * dirs[idx * 3 + 2];
            int f_ix = clamp_int(int(std::floor((hx - wmin_x) / cell_w)), 0, n - 1);
            int f_iz = clamp_int(int(std::floor((hz - wmin_z) / cell_d)), 0, n - 1);
            int ff = f_iz * n + f_ix;

            occ_state[ff] = (wall_score[ff] < 0.50f) ? uint8_t(1) : old_occ[ff];
            float rel_elev = clamp_f(hy - wmin_y, -4.0f, 4.0f);
            elev_map[ff] = 0.70f * old_elv[ff] + 0.30f * rel_elev;
            visit_freshness[ff] = std::max(old_vis[ff], 0.85f * cw);

            float surf_obs = (sal > 0.66f) ? 1.0f : ((sal < 0.36f) ? -1.0f : 0.0f);
            if (ventral_16x16x4 != nullptr) {
                float rg = ventral_16x16x4[idx * 4 + 0];
                float by = ventral_16x16x4[idx * 4 + 1];
                if (sal >= 0.36f && sal <= 0.66f && std::abs(by - rg) > 0.45f) {
                    surf_obs = clamp_f(by - rg, -0.75f, 0.75f);
                }
            }
            float alpha_s = clamp_f(0.45f * cw, 0.10f, 0.55f);
            surface_map[ff] = clamp_f((1.0f - alpha_s) * old_srf[ff] + alpha_s * surf_obs, -1.0f, 1.0f);
            surface_obs_age[ff] = 0.0f;
        }
    }

    std::memcpy(old_occ, occ_state, nn * sizeof(uint8_t));
    std::memcpy(old_wsc, wall_score, nn * sizeof(float));
    std::memcpy(old_elv, elev_map, nn * sizeof(float));
    std::memcpy(old_vis, visit_freshness, nn * sizeof(float));
    for (int idx = 0; idx < 256; ++idx) {
        float nd = clamp_f(fovea_16x16x4[idx * 4 + 0], 0.0f, 1.0f);
        float ny = fovea_16x16x4[idx * 4 + 2];
        float sal = fovea_16x16x4[idx * 4 + 3];
        float dm = dist_m[idx];
        float cw = conf_w[idx];
        float hy = bel_y + dm * dirs[idx * 3 + 1];
        if (nd > 0.03f && dm > 0.25f && cw > 0.30f && std::abs(ny) < 0.42f && sal < 0.70f && hy < bel_y + 1.6f) {
            float hx = bel_x + dm * dirs[idx * 3 + 0];
            float hz = bel_z + dm * dirs[idx * 3 + 2];
            int w_ix = clamp_int(int(std::floor((hx - wmin_x) / cell_w)), 0, n - 1);
            int w_iz = clamp_int(int(std::floor((hz - wmin_z) / cell_d)), 0, n - 1);
            int wf = w_iz * n + w_ix;
            wall_score[wf] = std::min(3.0f, old_wsc[wf] + 0.38f * cw);
        }
    }
    for (int idx = 0; idx < 256; ++idx) {
        float nd = clamp_f(fovea_16x16x4[idx * 4 + 0], 0.0f, 1.0f);
        float ny = fovea_16x16x4[idx * 4 + 2];
        float sal = fovea_16x16x4[idx * 4 + 3];
        float dm = dist_m[idx];
        float cw = conf_w[idx];
        float hy = bel_y + dm * dirs[idx * 3 + 1];
        if (nd > 0.03f && dm > 0.25f && cw > 0.30f && std::abs(ny) < 0.42f && sal < 0.70f && hy < bel_y + 1.6f) {
            float hx = bel_x + dm * dirs[idx * 3 + 0];
            float hz = bel_z + dm * dirs[idx * 3 + 2];
            int w_ix = clamp_int(int(std::floor((hx - wmin_x) / cell_w)), 0, n - 1);
            int w_iz = clamp_int(int(std::floor((hz - wmin_z) / cell_d)), 0, n - 1);
            int wf = w_iz * n + w_ix;
            occ_state[wf] = (wall_score[wf] >= 0.75f) ? uint8_t(2) : old_occ[wf];
            elev_map[wf] = std::max(old_elv[wf], clamp_f(hy - wmin_y, 0.0f, 4.0f));
            visit_freshness[wf] = std::max(old_vis[wf], 0.70f * cw);
        }
    }

    int n_free = 0, n_fr = 0, n_en = 0;
    for (int i = 0; i < nn; ++i) {
        if (occ_state[i] == 1) {
            ++n_free;
            if (surface_map[i] > 0.25f) ++n_fr;
            else if (surface_map[i] < -0.25f) ++n_en;
        }
    }
    if (n_free <= 0) {
        out_ratios_2[0] = 0.0f;
        out_ratios_2[1] = 0.0f;
    } else {
        out_ratios_2[0] = float(n_fr) / float(n_free);
        out_ratios_2[1] = float(n_en) / float(n_free);
    }
}

UB3D_API int ub3d_select_frontier_cell(
    const uint8_t* occ_state,
    const float* surface_map,
    const float* visit_freshness,
    const float* lane_mult_grid,
    const float* dwell_saturation,
    const float* strategic_hazard_scar,
    int n,
    int ego_iz, int ego_ix,
    float unknown_weight, float turf_weight,
    int committed_cz, int committed_cx, int committed_ticks,
    int* out_bz, int* out_bx, int* out_new_ticks
) {
    bool any_free = false;
    bool any_valid = false;
    float best_score = -1e9f;
    int best_flat = -1;
    float comm_score = -1e9f;

    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            int f = iz * n + ix;
            if (occ_state[f] != 1) continue;
            any_free = true;

            int unk_adj = 0;
            if (iz > 0 && occ_state[(iz - 1) * n + ix] == 0) ++unk_adj;
            if (iz + 1 < n && occ_state[(iz + 1) * n + ix] == 0) ++unk_adj;
            if (ix > 0 && occ_state[iz * n + (ix - 1)] == 0) ++unk_adj;
            if (ix + 1 < n && occ_state[iz * n + (ix + 1)] == 0) ++unk_adj;

            float unpainted = clamp_f(0.65f - surface_map[f], 0.0f, 1.65f);
            float stale = clamp_f(1.0f - visit_freshness[f], 0.0f, 1.0f);
            float dist_c = std::hypot(float(ix - ego_ix), float(iz - ego_iz));

            float sc = (unknown_weight * (0.45f * float(unk_adj) + 0.55f * stale) + turf_weight * unpainted) * lane_mult_grid[f]
                     - 1.35f * dwell_saturation[f]
                     - 1.25f * strategic_hazard_scar[f]
                     - 0.14f * dist_c;

            if (iz == committed_cz && ix == committed_cx) {
                comm_score = sc;
            }
            if (dist_c >= 1.8f) {
                any_valid = true;
                if (sc > best_score) {
                    best_score = sc;
                    best_flat = f;
                }
            }
        }
    }

    if (!any_free) return 0; // code 0 -> no free cells
    if (!any_valid) return 1; // code 1 -> no valid >= 1.8 cells

    if (committed_cz >= 0 && committed_cz < n && committed_cx >= 0 && committed_cx < n) {
        if (occ_state[committed_cz * n + committed_cx] == 1) {
            float c_dist = std::hypot(float(committed_cx - ego_ix), float(committed_cz - ego_iz));
            if (c_dist >= 1.15f && committed_ticks < 42 && comm_score > 0.15f) {
                *out_bz = committed_cz;
                *out_bx = committed_cx;
                *out_new_ticks = committed_ticks + 1;
                return 2;
            }
        }
    }

    *out_bz = best_flat / n;
    *out_bx = best_flat % n;
    *out_new_ticks = 1;
    return 2;
}
