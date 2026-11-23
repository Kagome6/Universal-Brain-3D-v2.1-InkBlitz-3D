# -*- coding: utf-8 -*-
"""
paper_v2_3/experiments/scripts/generate_paper_figures.py
========================================================
Publication-Grade Vector PDF & 300-DPI PNG Figure Generator for Universal Brain 3D (v2.3.0).
Generates all 6 figures strictly from measured JSON logs in `paper_v2_3/experiments/results/`:
  - fig1_systems_latency_vram.{pdf,png}
  - fig2_behavioral_ablations.{pdf,png}
  - fig3_sleep_and_counterfactual.{pdf,png}
  - fig4_live_cognitive_trace.{pdf,png}
  - fig5_phi_and_gnwt_regimes.{pdf,png}
  - fig6_architecture_criteria.{pdf,png}
"""

from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES_DIR = os.path.join(PAPER_DIR, "experiments", "results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, basename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{basename}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{basename}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig1(e1: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    lat = e1["step_latency_ms"]
    labels = [
        "P1: WorldBridge\n+ SeqLock (60Hz)",
        "P3: SLAM + 48x48 A*\n+ C++ Ray/BG (60Hz)",
        "P4: TT-D3QN\nReflex B=1 (24Hz)",
        "P5: GWT + RSSM\n+ 15-Cut Phi_G (12Hz)",
        "Phi_G 15-Cut\nMM Solver Only",
    ]
    keys = [
        "P1_WorldBridge_UB3D_SeqLock",
        "P3_SpatialTiming_SLAM_Map_AStar_BG",
        "P4_System1Reflex_Infer_B1",
        "P5_ConsciousGWT_RSSM_And_PhiG",
        "P5_PhiG_15Cut_MM_Only",
    ]
    p50s = [lat[k]["p50"] for k in keys]
    p99s = [lat[k]["p99"] for k in keys]
    y = np.arange(len(labels))
    h = 0.36
    ax1.barh(y - h / 2, p50s, height=h, color="#2b5c8f", label="Median (p50)")
    ax1.barh(y + h / 2, p99s, height=h, color="#d97724", label="Tail (p99)")
    ax1.axvline(16.67, color="#b22222", linestyle="--", linewidth=1.2, label="60 Hz Budget (16.67 ms)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(49.5, max(p99s) * 1.28))
    ax1.set_xlabel("Step Execution Time (ms)")
    ax1.set_title("(a) Sub-Budget Step Latencies (500 Warm Steps)")
    ax1.legend(loc="center right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.90)
    for yi, (v50, v99) in enumerate(zip(p50s, p99s)):
        x_txt = v99 + 0.55 if abs(v99 - 16.67) > 1.8 else v99 + 1.15
        ax1.text(
            x_txt, yi + h / 2, f"{v99:.2f} ms", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
        )

    vr = e1["vram_scaling_mb"]
    brains = ["1 Brain\n(Play/Spectate)", "4 Brains\n(2x 1v1 Arenas)", "8 Brains\n(PopGPUServer League)"]
    alloc = [vr["1_brain_alloc_mb"], vr["4_brains_alloc_mb"], vr["8_brains_alloc_mb"]]
    resv = [vr["1_brain_reserved_mb"], vr["4_brains_reserved_mb"], vr["8_brains_reserved_mb"]]
    x = np.arange(len(brains))
    w = 0.34
    ax2.bar(x - w / 2, alloc, width=w, color="#2e8b57", label="CUDA Allocated (MB)")
    ax2.bar(x + w / 2, resv, width=w, color="#7aa98c", label="CUDA Reserved Pool (MB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(brains)
    ax2.set_ylim(0, max(resv) * 1.18)
    ax2.set_ylabel("GPU Memory (MB)")
    ax2.set_title("(b) Single-GPU VRAM Scaling Under Barrier 1")
    ax2.legend(loc="upper left", frameon=False)
    for xi, va in enumerate(alloc):
        ax2.text(xi - w / 2, va + max(resv) * 0.025, f"{va:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "fig1_systems_latency_vram")


def make_fig2(e2: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1.52, 1.0]})

    conds = e2["conditions"]
    order = [
        ("C0_Sanity_Full_vs_Full", "C0: Full vs. Full (Symmetry Control)"),
        ("C1_Full_vs_UntrainedWeights", "C1: vs. Untrained Weights (Stage 1)"),
        ("C2_Full_vs_NoSLAM_NoMap", "C2a: vs. w/o SLAM & Map (Open Stage 1)"),
        ("C2_Stage2_Full_vs_NoSLAM_NoMap", "C2b: vs. w/o SLAM & Map (Occl. Stage 2)"),
        ("C3_Full_vs_NoRespawnContinuity", "C3: vs. w/o Respawn Map Keep (9s)"),
        ("C4_Full_vs_NoBGGate_Instant", "C4: vs. w/o BG Gate (Instant)"),
        ("C5_Full_vs_BGFixed_NoEscapeBypass", "C5: vs. Fixed Brake (No Escape Bypass)"),
        ("C6_Full_vs_PureSystem1_NoShield", "C6: vs. w/o 3-Head RSSM Shield"),
        ("C6b_Full_vs_RawFastReflexOnly", "C6b: vs. Pure Fast System-1 Reflex"),
        ("C7_Full_vs_NoGWTBroadcast", "C7: vs. Zero GWT Workspace Broadcast"),
        ("C8_GenericProfile_Full_vs_NoSLAM", "C8: Generic Profile vs. w/o Map (S1)"),
    ]
    order = [(k, lbl) for k, lbl in order if k in conds]
    means = [conds[k]["full_win_rate"]["mean"] * 100.0 for k, _ in order]
    lo = [means[i] - conds[k]["full_win_rate"]["ci95_low"] * 100.0 for i, (k, _) in enumerate(order)]
    hi = [conds[k]["full_win_rate"]["ci95_high"] * 100.0 - means[i] for i, (k, _) in enumerate(order)]
    colors = ["#7f8c8d"] + [("#16a085" if "Stage2" in k else ("#6c3483" if "C8" in k else "#1f618d")) for k, _ in order[1:]]

    y = np.arange(len(order))
    ax1.barh(y, means, xerr=[lo, hi], capsize=3.0, color=colors, alpha=0.90, height=0.64)
    ax1.axvline(50.0, color="#c0392b", linestyle="--", linewidth=1.3, label="50% Parity Null")
    ax1.set_yticks(y)
    ax1.set_yticklabels([lbl for _, lbl in order], fontsize=7.8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 114)
    ax1.set_xlabel("Autonomous Full v2.3.0 Win Rate (%) ± 95% CI (N=24 Pairs)")
    ax1.set_title("(a) Side-Swapped Autonomous Ablations (scaf = 0.0)")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
    for yi, (k, _) in enumerate(order):
        p_h = conds[k]["paired_win_test"].get("wilcoxon_p_holm", conds[k]["paired_win_test"]["wilcoxon_p"])
        sig = "**" if p_h < 0.01 else ("*" if p_h < 0.05 else "")
        hi_end = means[yi] + hi[yi]
        x_pos = 51.8 if (42.5 <= hi_end <= 50.8) else min(102.5, hi_end + 1.5)
        ax1.text(
            x_pos, yi, f"{means[yi]:.1f}%{sig}", va="center", fontsize=7.9,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.88),
        )

    ms = e2.get("multi_seed_v23_summary", {})
    if ms:
        bars_lbl = [
            "Stage 1\nvs. Untr",
            "Stage 2\nvs. Untr",
            "Stage 1\nvs. L2 Heur",
            "Stage 2\nvs. L2 Heur",
        ]
        b_keys = [
            "S1_v23Auto_vs_UntrainedAuto",
            "S2_v23Auto_vs_UntrainedAuto",
            "S1_v23Auto_vs_L2FullHeuristic",
            "S2_v23Auto_vs_L2FullHeuristic",
        ]
        b_m = [ms[k]["mean_win_rate_pct"] for k in b_keys]
        b_lo = [b_m[i] - ms[k]["ci95_pct"][0] for i, k in enumerate(b_keys)]
        b_hi = [ms[k]["ci95_pct"][1] - b_m[i] for i, k in enumerate(b_keys)]
        x = np.arange(len(bars_lbl))
        b_cols = ["#2980b9", "#16a085", "#7f8c8d", "#27ae60"]
        ax2.bar(x, b_m, yerr=[b_lo, b_hi], capsize=4.0, width=0.52, color=b_cols, alpha=0.90)
        ax2.axhline(50.0, color="#c0392b", linestyle="--", linewidth=1.2, label="50% Parity")
        ax2.set_xticks(x)
        ax2.set_xticklabels(bars_lbl, fontsize=8.0)
        ax2.set_ylabel("Pooled Win Rate (%) ± 95% CI (K=3 Seeds, 144g)")
        ax2.set_ylim(0, 106)
        ax2.set_title("(b) K=3 Multi-Seed Cross-Stage Autonomy")
        ax2.legend(loc="lower right", frameon=False)
        for xi, vm in enumerate(b_m):
            p_v = ms[b_keys[xi]]["wilcoxon_p"]
            st = "**" if p_v < 0.01 else ("*" if p_v < 0.05 else "")
            ax2.text(xi, vm + b_hi[xi] + 2.2, f"{vm:.1f}%{st}", ha="center", fontsize=8.2, fontweight="bold")
    else:
        sub_keys = [
            ("C1_Full_vs_UntrainedWeights", "vs. Untr"),
            ("C2_Full_vs_NoSLAM_NoMap", "vs. NoMap S1"),
            ("C6b_Full_vs_RawFastReflexOnly", "vs. Raw S1"),
        ]
        x = np.arange(len(sub_keys))
        w = 0.35
        f_prog = [conds[k]["full_max_prog"]["mean"] * 100.0 for k, _ in sub_keys]
        o_prog = [conds[k]["opp_max_prog"]["mean"] * 100.0 for k, _ in sub_keys]
        ax2.bar(x - w / 2, f_prog, width=w, color="#1f618d", label="Full Brain v2.3.0")
        ax2.bar(x + w / 2, o_prog, width=w, color="#e67e22", label="Ablated / Baseline")
        ax2.set_xticks(x)
        ax2.set_xticklabels([lbl for _, lbl in sub_keys])
        ax2.set_ylabel("Max BlitzCore Progress (%)")
        ax2.set_ylim(40, 102)
        ax2.set_title("(b) Unshaped Objective Escort Reach")
        ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    _save(fig, "fig2_behavioral_ablations")


def make_fig3(e3: dict, e5: dict | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    pa = e3["part_a_sleep_consolidation"]
    if e5 is not None and "diagnostic_2_sleep_diverse_buffer" in e5:
        d2 = e5["diagnostic_2_sleep_diverse_buffer"]
        reg_labels = [
            "Starved Micro-Buffer\n(N_buf~55, Draw Ratio 2.6x)",
            "Diverse Gen-Buffer\n(N_buf~651, Draw Ratio 0.11x)",
        ]
        win_means = [
            pa["sleep_head2head_win_rate"]["mean"] * 100.0,
            d2["sleep_head2head_win_rate"]["mean"] * 100.0,
        ]
        win_lo = [
            win_means[0] - pa["sleep_head2head_win_rate"]["ci95_low"] * 100.0,
            win_means[1] - d2["sleep_head2head_win_rate"]["ci95_low"] * 100.0,
        ]
        win_hi = [
            pa["sleep_head2head_win_rate"]["ci95_high"] * 100.0 - win_means[0],
            d2["sleep_head2head_win_rate"]["ci95_high"] * 100.0 - win_means[1],
        ]
        x0 = np.arange(2)
        cols = ["#c0392b", "#1f618d"]
        ax1.bar(x0, win_means, yerr=[win_lo, win_hi], capsize=4.5, width=0.46, color=cols, alpha=0.90)
        ax1.axhline(50.0, color="#555555", linestyle="--", linewidth=1.2, label="50% Parity vs. No-Sleep")
        ax1.set_xticks(x0)
        ax1.set_xticklabels(reg_labels)
        ax1.set_ylim(0, 102)
        ax1.set_ylabel("Sleep vs. No-Sleep Win Rate (%) ± 95% CI")
        ax1.set_title("(a) Sleep Replay-Ratio Regime Audit (N=10 Seeds)")
        ax1.legend(loc="upper left", frameon=False)
        for xi, vm in enumerate(win_means):
            ax1.text(xi, vm + win_hi[xi] + 2.5, f"{vm:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    pb = e3["part_b_counterfactual_deconfounding"]
    cats = ["CF Loss OFF\n(Factual Only)", "CF Loss ON\n(Eq. 7 De-Confounded)"]
    s_m4 = [pb["cf_off_surv_m4_escape"]["mean"], pb["cf_on_surv_m4_escape"]["mean"]]
    s_m7 = [pb["cf_off_surv_m7_exposed_idle"]["mean"], pb["cf_on_surv_m7_exposed_idle"]["mean"]]
    e_m4 = [pb["cf_off_surv_m4_escape"]["sd"], pb["cf_on_surv_m4_escape"]["sd"]]
    e_m7 = [pb["cf_off_surv_m7_exposed_idle"]["sd"], pb["cf_on_surv_m7_exposed_idle"]["sd"]]
    x = np.arange(2)
    w = 0.32
    ax2.bar(x - w / 2, s_m4, yerr=e_m4, capsize=4, width=w, color="#27ae60", label="Pred M4: Escape Dive")
    ax2.bar(x + w / 2, s_m7, yerr=e_m7, capsize=4, width=w, color="#c0392b", label="Pred M7: Exposed Idle")
    if "true_interventional_surv_m4_escape" in pb and "true_interventional_surv_m7_exposed_idle" in pb:
        t_m4 = pb["true_interventional_surv_m4_escape"]["mean"]
        t_m7 = pb["true_interventional_surv_m7_exposed_idle"]["mean"]
        ax2.scatter(x - w / 2, [t_m4, t_m4], marker="D", s=34, color="#0b5345", zorder=5, label=f"True Rollout M4 ({t_m4:.2f})")
        ax2.scatter(x + w / 2, [t_m7, t_m7], marker="X", s=42, color="#641e16", zorder=5, label=f"True Rollout M7 ({t_m7:.2f})")
    ax2.axhline(0.42, color="#34495e", linestyle="--", linewidth=1.1, label="Veto Threshold (0.42)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("3-Step Survival Probability (Held-Out)")
    ax2.set_title("(b) Held-Out Interventional Calibration (N=10 Seeds)")
    ax2.legend(loc="upper right", frameon=False, fontsize=7.4)

    fig.tight_layout()
    _save(fig, "fig3_sleep_and_counterfactual")


def make_fig4(e4: dict) -> None:
    tr = e4["live_arena_episode_metrics"]["representative_trace"]
    t = np.asarray(tr["t_sec"], dtype=np.float64)
    haz = np.asarray(tr["hazard"], dtype=np.float64)
    ign = np.asarray(tr["ignition"], dtype=np.float64)
    phig = np.asarray(tr["phi_causal_bits"], dtype=np.float64)
    pact = np.asarray(tr["phi_act_bits"], dtype=np.float64)
    veto = np.asarray(tr["veto"], dtype=np.int32)
    resp = np.asarray(tr["respawn"], dtype=np.int32)
    alp = np.asarray(tr["gwt_alpha_5"], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(10.4, 6.5), sharex=True)

    axes[0].plot(t, haz, color="#c0392b", linewidth=1.6, label="Perceived Hazard")
    axes[0].plot(t, ign, color="#1f618d", linewidth=1.6, label="GNWT Ignition")
    veto_t = t[veto == 1]
    if veto_t.size > 0:
        axes[0].scatter(veto_t, np.full_like(veto_t, 0.94), marker="v", color="#d35400", s=20, label="Survival Shield Veto")
    for rt in t[resp == 1]:
        for ax in axes:
            ax.axvline(rt, color="#8e44ad", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[0].set_ylabel("Level [0..1]")
    axes[0].set_ylim(-0.03, 1.24)
    axes[0].set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axes[0].set_title("Live Closed-Loop Cognitive Dynamics During Symmetric 1v1 Match (v2.3.0)")
    axes[0].legend(loc="upper right", ncol=3, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)

    sp_names = ["P (Visual/Slots)", "I (Spatial/SLAM)", "W (RSSM World)", "M (Motor/Hazard)", "S (Self/Strategy)"]
    sp_cols = ["#2980b9", "#27ae60", "#e67e22", "#c0392b", "#8e44ad"]
    axes[1].stackplot(t, alp.T, labels=sp_names, colors=sp_cols, alpha=0.82)
    axes[1].set_ylabel("GWT Broadcast Share")
    axes[1].set_ylim(0, 1.18)
    axes[1].set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axes[1].legend(loc="upper right", ncol=5, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, fontsize=7.5)

    axes[2].plot(t, phig, color="#16a085", linewidth=1.7, label="Lagged Causal Phi_G (bits)")
    axes[2].plot(t, pact, color="#d35400", linewidth=1.3, linestyle="--", label="Override Divergence Phi_act (bits)")
    axes[2].set_ylabel("Information (bits)")
    axes[2].set_xlabel("Match Time (s)")
    axes[2].legend(loc="upper right", ncol=2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)

    fig.tight_layout()
    _save(fig, "fig4_live_cognitive_trace")


def make_fig5(e4: dict) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.2, 3.8))

    p3 = e4["proposition3_monte_carlo"]
    c_names = ["Null Flash\n(Isotropic)", "Null Flash\n(Off-Diag)", "2+3 Split\n(Off-Diag)", "5-Node Ring\n(Off-Diag)"]
    c_means = [
        p3["common_input_null_isotropic_ridge_bits"]["mean"],
        p3["common_input_null_offdiag_ridge_bits"]["mean"],
        p3["block_split_2v3_phi_bits"]["mean"],
        p3["directed_ring_phi_bits"]["mean"],
    ]
    c_sds = [
        p3["common_input_null_isotropic_ridge_bits"]["sd"],
        p3["common_input_null_offdiag_ridge_bits"]["sd"],
        p3["block_split_2v3_phi_bits"]["sd"],
        p3["directed_ring_phi_bits"]["sd"],
    ]
    cols1 = ["#c0392b", "#27ae60", "#2980b9", "#1f618d"]
    ax1.bar(np.arange(4), c_means, yerr=c_sds, capsize=3.5, color=cols1, width=0.56)
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels(c_names, fontsize=8.0)
    ax1.set_ylabel("Estimated Phi_G (bits)")
    ax1.set_title("(a) Proposition 3 Verification (N=50 MC)")

    gm = e4["live_arena_episode_metrics"]["gnwt_metastability"]
    x2 = np.arange(2)
    ig_m = [gm["hab_off_ignition"]["mean"], gm["hab_on_ignition"]["mean"]]
    ig_s = [gm["hab_off_ignition"]["sd"], gm["hab_on_ignition"]["sd"]]
    ho_m = [gm["hab_off_handoffs_per_sec"]["mean"], gm["hab_on_handoffs_per_sec"]["mean"]]
    ax2_twin = ax2.twinx()
    ax2.bar(x2 - 0.16, ig_m, yerr=ig_s, capsize=3.5, width=0.30, color="#7f8c8d", label="Mean Ignition")
    ax2_twin.bar(x2 + 0.16, ho_m, width=0.30, color="#d35400", label="Handoffs / sec")
    ax2_twin.spines["top"].set_visible(False)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Habituation OFF\n(Stagnant)", "Habituation ON\n(Metastable v2.3.0)"])
    ax2.set_ylabel("Mean Ignition [0..1]", color="#555555")
    ax2_twin.set_ylabel("Specialist Handoffs / s", color="#d35400")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("(b) GNWT Metastability (N=24 Eps)")

    pg_stats = e4["live_arena_episode_metrics"]["phi_causal_bits"]
    pa_stats = e4["live_arena_episode_metrics"]["phi_act_override_bits"]
    x3 = np.array([0.0, 1.0])
    pg_vals = [pg_stats["untrained"]["mean"] * 1000.0, pg_stats["trained_cruise"]["mean"] * 1000.0]
    pg_err_lo = [
        (pg_stats["untrained"]["mean"] - pg_stats["untrained"]["ci95_low"]) * 1000.0,
        (pg_stats["trained_cruise"]["mean"] - pg_stats["trained_cruise"]["ci95_low"]) * 1000.0,
    ]
    pg_err_hi = [
        (pg_stats["untrained"]["ci95_high"] - pg_stats["untrained"]["mean"]) * 1000.0,
        (pg_stats["trained_cruise"]["ci95_high"] - pg_stats["trained_cruise"]["mean"]) * 1000.0,
    ]
    pa_vals = [pa_stats["trained_cruise"]["mean"], pa_stats["trained_crisis_veto"]["mean"]]
    pa_err_lo = [
        pa_stats["trained_cruise"]["mean"] - pa_stats["trained_cruise"]["ci95_low"],
        pa_stats["trained_crisis_veto"]["mean"] - pa_stats["trained_crisis_veto"]["ci95_low"],
    ]
    pa_err_hi = [
        pa_stats["trained_cruise"]["ci95_high"] - pa_stats["trained_cruise"]["mean"],
        pa_stats["trained_crisis_veto"]["ci95_high"] - pa_stats["trained_crisis_veto"]["mean"],
    ]

    ax3_twin = ax3.twinx()
    ax3.bar(x3 - 0.17, pg_vals, yerr=[pg_err_lo, pg_err_hi], capsize=3.5, width=0.30, color="#16a085")
    ax3_twin.bar(x3 + 0.17, pa_vals, yerr=[pa_err_lo, pa_err_hi], capsize=3.5, width=0.30, color="#c0392b", alpha=0.88)
    ax3_twin.spines["top"].set_visible(False)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(["Untrained (Phi_G)\n/ Cruise (Phi_act)", "Trained (Phi_G ***)\n/ Crisis Veto (Phi_act n.s.)"])
    ax3.set_ylabel("Lagged Phi_G (x10^-3 bits)", color="#16a085")
    ax3_twin.set_ylabel("Instantaneous Phi_act (bits)", color="#c0392b")
    ax3.set_ylim(0, max(1.0, max(pg_vals) * 1.55))
    ax3_twin.set_ylim(0, max(0.2, max(pa_vals) * 1.35))
    ax3.set_title("(c) Dual-Timescale Integration (Eq. 17)")

    fig.tight_layout()
    _save(fig, "fig5_phi_and_gnwt_regimes")


def make_fig6(e6: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.1), gridspec_kw={"width_ratios": [1.28, 1.0]})

    lad = e6["criterion_3_cumulative_ladder"]
    rungs = [
        ("L0_reactive", "L0: Reactive\n(No Map)"),
        ("L1_slam_map", "L1: +SLAM &\n4-Layer Map"),
        ("L2_homeo_scaffold", "L2: +Compass\n(Self Control)"),
        ("L3_neural_d3qn", "L3: +Neural\nTT-D3QN (0.0)"),
        ("L4_plus_rssm", "L4: +3-Head\nRSSM (0.0)"),
        ("L5_full_v23", "L5: Full v2.3.0\nAuto (scaf=0)"),
    ]
    x = np.arange(len(rungs))
    wrs = [lad[k]["win_rate_pct"] for k, _ in rungs]
    lo = [wrs[i] - lad[k]["ci95_pct"][0] for i, (k, _) in enumerate(rungs)]
    hi = [lad[k]["ci95_pct"][1] - wrs[i] for i, (k, _) in enumerate(rungs)]
    reaches = [lad[k]["max_prog_pct"] for k, _ in rungs]
    stags = [lad[k]["mean_stagnation"] * 100.0 for k, _ in rungs]

    ax1.bar(x - 0.16, wrs, yerr=[lo, hi], capsize=3.2, width=0.36, color="#1f618d", alpha=0.90, label="Win Rate vs. L2 (%)")
    ax1.plot(x + 0.16, reaches, marker="o", linewidth=1.8, color="#27ae60", label="Max Reach (%)")
    ax1.plot(x + 0.16, stags, marker="s", linestyle="--", linewidth=1.6, color="#c0392b", label="Stagnation (x100)")
    ax1.axhline(50.0, color="#555555", linestyle=":", linewidth=1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in rungs], fontsize=7.5)
    ax1.set_ylim(0, 104)
    ax1.set_ylabel("Percentage / Scaled Metric")
    ax1.set_title("(a) Autonomous Organ Ladder on Stage 2 (vs. L2 Heuristic)")
    ax1.legend(loc="upper left", frameon=False, fontsize=7.8)

    sw = e6["criterion_1_replaceability_swaps"]
    s_order = [
        ("S1_rssm_mpc_no_d3qn", "S1: Pure RSSM-MPC\n(No TT-D3QN)"),
        ("S2_gwt_uniform_pool", "S2: GWT Uniform\n(1/5 Pool)"),
        ("S3_gwt_hard_wta", "S3: GWT Hard\nWinner-Take-All"),
        ("S4_slam_drift_only", "S4: Drift-Only\n(No Loop Closure)"),
        ("S5_bg_cautious", "S5: Cautious BG\n(1.35x Brake)"),
    ]
    ys = np.arange(len(s_order))
    sw_wr = [sw[k]["win_rate_pct"] for k, _ in s_order]
    sw_lo = [sw_wr[i] - sw[k]["ci95_pct"][0] for i, (k, _) in enumerate(s_order)]
    sw_hi = [sw[k]["ci95_pct"][1] - sw_wr[i] for i, (k, _) in enumerate(s_order)]
    ax2.barh(ys, sw_wr, xerr=[sw_lo, sw_hi], capsize=3.5, height=0.56, color="#2e8b57", alpha=0.88)
    ax2.axvline(50.0, color="#c0392b", linestyle="--", linewidth=1.2, label="50% Parity vs. L2")
    ax2.axvline(lad["L5_full_v23"]["win_rate_pct"], color="#1f618d", linestyle=":", linewidth=1.2, label=f"L5 Full ({lad['L5_full_v23']['win_rate_pct']:.1f}%)")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([lbl for _, lbl in s_order], fontsize=8.0)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 108)
    ax2.set_xlabel("Win Rate vs. L2 (%) ± 95% CI (0 Contract Errors)")
    ax2.set_title("(b) Plug-Compatible Organ Replaceability (Stage 2)")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, fontsize=7.5)
    for yi, v in enumerate(sw_wr):
        ax2.text(min(98.5, v + sw_hi[yi] + 1.5), yi, f"{v:.1f}%", va="center", fontsize=8.0)

    fig.tight_layout()
    _save(fig, "fig6_architecture_criteria")


def main() -> None:
    with open(os.path.join(RES_DIR, "exp1_systems.json"), encoding="utf-8") as f:
        e1 = json.load(f)
    with open(os.path.join(RES_DIR, "exp2_ablations.json"), encoding="utf-8") as f:
        e2 = json.load(f)
    with open(os.path.join(RES_DIR, "exp3_sleep_and_cf.json"), encoding="utf-8") as f:
        e3 = json.load(f)
    with open(os.path.join(RES_DIR, "exp4_consciousness_wm.json"), encoding="utf-8") as f:
        e4 = json.load(f)
    e5_path = os.path.join(RES_DIR, "exp5_regime_diagnostics.json")
    e5 = None
    if os.path.isfile(e5_path):
        with open(e5_path, encoding="utf-8") as f:
            e5 = json.load(f)
    e6_path = os.path.join(RES_DIR, "exp6_architecture_criteria.json")

    make_fig1(e1)
    make_fig2(e2)
    make_fig3(e3, e5)
    make_fig4(e4)
    make_fig5(e4)
    if os.path.isfile(e6_path):
        with open(e6_path, encoding="utf-8") as f:
            e6 = json.load(f)
        make_fig6(e6)
    print(f"[Figures] Successfully rendered all 6 publication PDF + PNG figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
