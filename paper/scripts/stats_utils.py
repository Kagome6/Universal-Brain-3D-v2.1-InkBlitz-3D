# -*- coding: utf-8 -*-
"""
stats_utils.py — Trick-Free Non-Parametric & Paired Statistical Utilities
=========================================================================
Enforces:
  1. Strict independence of sampling units (seed-matched pairs or episode means, never raw correlated ticks).
  2. Two-sided non-parametric tests (Wilcoxon signed-rank, exact Binomial sign test, Mann-Whitney U).
  3. Holm-Bonferroni family-wise error rate (FWER) adjustment across multiple hypotheses.
  4. Non-parametric bootstrap 95% confidence intervals (10,000 resamples) and standardized effect sizes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def summary_stats(vals: Sequence[float], seed: int = 2026, n_boot: int = 10000) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "sd": 0.0, "median": 0.0, "iqr": 0.0,
            "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
        }
    mean_v = float(np.mean(arr))
    sd_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    p05, p95, p99 = np.percentile(arr, [5.0, 95.0, 99.0])
    if n > 1 and sd_v > 1e-15:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(int(n_boot), n))
        boot_means = np.mean(arr[idx], axis=1)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    else:
        ci_lo, ci_hi = mean_v, mean_v
    return {
        "n": n,
        "mean": mean_v,
        "sd": sd_v,
        "median": float(q50),
        "iqr": float(q75 - q25),
        "p05": float(p05),
        "p50": float(q50),
        "p95": float(p95),
        "p99": float(p99),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
    }


def _exact_wilcoxon_signed_rank(nz: np.ndarray) -> Tuple[float, float]:
    """
    Exact conditional two-sided sign-flip permutation test for Wilcoxon signed-rank statistic
    (handles arbitrary ties cleanly without switching to asymptotic normal approximation for n <= 24).
    """
    m = int(nz.size)
    if m == 0:
        return 0.0, 1.0
    ranks = stats.rankdata(np.abs(nz), method="average").astype(np.float64)
    w_plus = float(np.sum(ranks[nz > 0]))
    w_minus = float(np.sum(ranks[nz < 0]))
    w_stat = min(w_plus, w_minus)
    mean_w = 0.5 * float(np.sum(ranks))
    obs_dev = abs(w_plus - mean_w)

    if m <= 24:
        total = 1 << m
        chunk = 1 << min(m, 16)
        bits_lo = np.arange(chunk, dtype=np.uint32)[:, None]
        shifts_lo = np.arange(min(m, 16), dtype=np.uint32)[None, :]
        mat_lo = ((bits_lo >> shifts_lo) & 1).astype(np.float64)
        sums_lo = mat_lo @ ranks[: min(m, 16)]
        if m <= 16:
            ge_cnt = int(np.sum(np.abs(sums_lo - mean_w) >= obs_dev - 1e-12))
        else:
            m_hi = m - 16
            n_hi = 1 << m_hi
            bits_hi = np.arange(n_hi, dtype=np.uint32)[:, None]
            shifts_hi = np.arange(m_hi, dtype=np.uint32)[None, :]
            sums_hi = ((bits_hi >> shifts_hi) & 1).astype(np.float64) @ ranks[16:]
            ge_cnt = 0
            for sh in sums_hi:
                ge_cnt += int(np.sum(np.abs((sums_lo + float(sh)) - mean_w) >= obs_dev - 1e-12))
        return w_stat, float(min(1.0, ge_cnt / float(total)))

    w_res = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
    return float(w_res.statistic), float(w_res.pvalue)


def paired_comparison(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    seed: int = 2026,
    n_boot: int = 10000,
) -> Dict[str, Any]:
    """Paired two-sided statistical comparison on independent seed-matched units `d_i = x_i - y_i`."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    assert x.shape == y.shape and x.ndim == 1, "Paired arrays must be 1D of identical length"
    diffs = x - y
    n = int(diffs.size)
    d_sum = summary_stats(diffs, seed=seed, n_boot=n_boot)

    nz = diffs[np.abs(diffs) > 1e-12]
    n_pos = int(np.sum(diffs > 1e-12))
    n_neg = int(np.sum(diffs < -1e-12))
    n_tie = n - n_pos - n_neg

    if nz.size >= 1:
        w_stat, wilcoxon_p = _exact_wilcoxon_signed_rank(nz)
        ranks = stats.rankdata(np.abs(nz))
        w_plus = float(np.sum(ranks[nz > 0]))
        w_minus = float(np.sum(ranks[nz < 0]))
        denom = float(nz.size * (nz.size + 1)) / 2.0
        rank_biserial = (w_plus - w_minus) / max(1.0, denom)
    else:
        wilcoxon_p = 1.0
        w_stat = 0.0
        rank_biserial = 0.0

    if (n_pos + n_neg) > 0:
        binom_p = float(stats.binomtest(n_pos, n_pos + n_neg, p=0.5, alternative="two-sided").pvalue)
    else:
        binom_p = 1.0

    sd_d = d_sum["sd"]
    cohens_dz = float(d_sum["mean"] / sd_d) if sd_d > 1e-12 else 0.0

    return {
        "n_pairs": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_tie": n_tie,
        "diff_summary": d_sum,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": wilcoxon_p,
        "binom_sign_p": binom_p,
        "cohens_dz": cohens_dz,
        "rank_biserial": float(rank_biserial),
    }


def unpaired_comparison(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    seed: int = 2026,
    n_boot: int = 10000,
) -> Dict[str, Any]:
    """Two-sided Mann-Whitney U test + Cliff's delta + bootstrap 95% CI of difference in means."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = int(x.size), int(y.size)
    if nx == 0 or ny == 0:
        return {"mannwhitney_u": 0.0, "mannwhitney_p": 1.0, "cliffs_delta": 0.0, "diff_mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

    u_res = stats.mannwhitneyu(x, y, alternative="two-sided")
    diff_mat = np.sign(x[:, None] - y[None, :])
    cliffs_d = float(np.mean(diff_mat))
    diff_mean = float(np.mean(x) - np.mean(y))

    rng = np.random.default_rng(seed)
    bx = np.mean(x[rng.integers(0, nx, size=(int(n_boot), nx))], axis=1)
    by = np.mean(y[rng.integers(0, ny, size=(int(n_boot), ny))], axis=1)
    ci_lo, ci_hi = np.percentile(bx - by, [2.5, 97.5])

    return {
        "n_x": nx,
        "n_y": ny,
        "mannwhitney_u": float(u_res.statistic),
        "mannwhitney_p": float(u_res.pvalue),
        "cliffs_delta": cliffs_d,
        "diff_mean": diff_mean,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
    }


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Exact step-down Holm-Bonferroni FWER adjustment across a family of tests."""
    items = sorted(p_values.items(), key=lambda kv: float(kv[1]))
    m = len(items)
    adj: Dict[str, float] = {}
    running_max = 0.0
    for rank_idx, (k, p_raw) in enumerate(items):
        mult = m - rank_idx
        p_corr = min(1.0, float(p_raw) * float(mult))
        running_max = max(running_max, p_corr)
        adj[k] = running_max
    return adj
